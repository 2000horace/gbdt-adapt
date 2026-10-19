"""Phase 1 / Gate G1: synthetic regime-switching benchmark.

Example
-------
    python run_phase1.py --variants disjoint shared --seeds 0 1 2 --out results
    python run_phase1.py --variants disjoint --seeds 0 --ablations --tune

Caching: each run stores its predictions in <out>/<variant>_seed<k>/cache.npz. A rerun
with the same settings reuses them (only scoring and plots are redone). Use --replot to
redraw from caches without ever recomputing, and --no-cache to force a recomputation.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase1 import baselines as B
from phase1 import cache
from phase1 import metrics as M
from phase1.adaptive import AdaptiveConfig, AdaptiveLeafFilter
from phase1.dgp import DGPConfig, generate
from phase1.ensemble import EnsembleConfig, LeafMap, train_ensemble

G1_BASELINES = ["frozen+robust", "sgd+robust", "refresh+robust"]
EARLY = 1_000   # events at the start of a segment used for "early" error

ABLATIONS = {
    "abl:no_robust": dict(use_robust=False),
    "abl:no_discount": dict(use_discount=False),
    "abl:no_scale": dict(use_scale=False),
    "abl:no_calibration": dict(use_calibration=False),
    "abl:no_overlap": dict(overlap_factor=1.0),
    "abl:no_alarm_action": dict(use_alarm_action=False),
    "abl:raw_z2_detector": dict(detector="raw", zeta=0.5, lam=30.0),
}


# ---------------------------------------------------------------- setup
def prepare(dgp_cfg, ens_cfg):
    data = generate(dgp_cfg)
    booster = train_ensemble(data["X_pre"], data["y_pre"], ens_cfg)
    leafmap = LeafMap(booster)
    leafmap.calibrate_bias(data["X_pre"])
    idx = leafmap.leaf_index(data["X"])
    return data, booster, leafmap, idx


def run_adaptive(leafmap, idx, data, cfg):
    filt = AdaptiveLeafFilter(leafmap.w0, leafmap.bias_index, cfg, data["h"])
    (mu, s, df), sec = B.timed(filt.run, idx, data["y_obs"])
    return (mu, s, df), sec, filt


def run_all(data, booster, leafmap, idx, ens_cfg, hp, ablations=False):
    """Return {name: dict(mu, s, df, seconds)} and the fitted adaptive filter."""
    h, X, y = data["h"], data["X"], data["y_obs"]
    out = {}

    def add(name, mu, s, df, sec):
        out[name] = dict(mu=mu, s=s, df=df, seconds=sec)

    (mu, s, df), sec, filt = run_adaptive(leafmap, idx, data, hp["adaptive"])
    add("adaptive", mu, s, df, sec)

    mu_f, sec = B.timed(B.frozen, leafmap, idx)
    add("frozen+ewma", mu_f, *B.ewma_gaussian_scale(mu_f, y, h), sec)
    add("frozen+robust", mu_f, *B.robust_t_scale(mu_f, y, h), sec)

    mu_g, sec = B.timed(B.online_sgd, leafmap, idx, y, h, eta0=hp["sgd_eta0"])
    add("sgd+robust", mu_g, *B.robust_t_scale(mu_g, y, h), sec)

    mu_r, sec = B.timed(B.xgb_refresh, booster, ens_cfg, X, y, h, **hp["refresh"])
    add("refresh+robust", mu_r, *B.robust_t_scale(mu_r, y, h), sec)

    mu_o, sec = B.timed(B.xgb_rolling, ens_cfg, X, y, h, X_pre=data["X_pre"], y_pre=data["y_pre"],
                        **hp["rolling"])
    add("rolling+robust", mu_o, *B.robust_t_scale(mu_o, y, h), sec)

    if ablations:
        for name, change in ABLATIONS.items():
            (mu, s, df), sec, _ = run_adaptive(leafmap, idx, data, replace(hp["adaptive"], **change))
            add(name, mu, s, df, sec)
    return out, filt


# ---------------------------------------------------------------- scoring
def score(data, results):
    """Per-method and per-segment scores.

    Forecast errors (rmse, mae) are against the realized clean label y_clean, which
    is what a forecaster is judged on. Signal errors compare the forecast mean with
    the true conditional mean (only available on synthetic data).
    """
    y, sig = data["y_clean"], data["signal"]
    rows, seg_rows, per_event = [], [], {}
    starts = list(data["seg_starts"]) + [len(y)]
    regimes = [data["regime"][a] for a in starts[:-1]]
    visits = [regimes[:k + 1].count(regimes[k]) for k in range(len(regimes))]
    for name, r in results.items():
        mu = r["mu"]
        ls = M.log_score(y, mu, r["s"], r["df"])
        per_event[name] = dict(log_score=ls, sq_err=(y - mu) ** 2, abs_err=np.abs(y - mu))
        rows.append(dict(
            method=name,
            log_score=ls.mean(),
            crps=M.crps(y, mu, r["s"], r["df"]).mean(),
            coverage80=M.coverage(y, mu, r["s"], r["df"], alpha=0.2),
            rmse=np.sqrt(np.mean((y - mu) ** 2)),
            mae=np.mean(np.abs(y - mu)),
            signal_rmse=np.sqrt(np.mean((mu - sig) ** 2)),
            signal_mae=np.mean(np.abs(mu - sig)),
            seconds=r["seconds"],
        ))
        for k in range(len(starts) - 1):
            a, b = starts[k], starts[k + 1]
            e = slice(a, min(a + EARLY, b))
            seg_rows.append(dict(
                method=name, segment=k, regime=regimes[k], visit=visits[k],
                log_score=ls[a:b].mean(),
                rmse=np.sqrt(np.mean((y[a:b] - mu[a:b]) ** 2)),
                mae=np.mean(np.abs(y[a:b] - mu[a:b])),
                signal_rmse=np.sqrt(np.mean((mu[a:b] - sig[a:b]) ** 2)),
                signal_mae=np.mean(np.abs(mu[a:b] - sig[a:b])),
                signal_mae_early=np.mean(np.abs(mu[e] - sig[e])),
            ))
    return pd.DataFrame(rows), pd.DataFrame(seg_rows), per_event


COMPARE = G1_BASELINES + ["rolling+robust"]


def compare(per_event, h, seed, reference="adaptive", baselines=COMPARE):
    """Paired differences with block-bootstrap CIs; positive always means reference is better.

    log score: reference - baseline. MSE and MAE: baseline - reference.
    The RMSE gain is reported as sqrt(MSE_base) - sqrt(MSE_ref) with the CI mapped
    through the same transformation.
    """
    block = max(500, 5 * h)
    ref = per_event[reference]
    rows = []
    for base in baselines:
        if base not in per_event:
            continue
        b = per_event[base]
        row = dict(reference=reference, baseline=base)
        m, lo, hi = M.block_bootstrap_ci(ref["log_score"] - b["log_score"], block=block, seed=seed)
        row.update(d_log_score=m, ls_lo=lo, ls_hi=hi)
        mse_b = b["sq_err"].mean()
        m, lo, hi = M.block_bootstrap_ci(b["sq_err"] - ref["sq_err"], block=block, seed=seed)
        row.update(d_rmse=np.sqrt(mse_b) - np.sqrt(mse_b - m),
                   rmse_lo=np.sqrt(mse_b) - np.sqrt(mse_b - lo),
                   rmse_hi=np.sqrt(mse_b) - np.sqrt(mse_b - hi))
        m, lo, hi = M.block_bootstrap_ci(b["abs_err"] - ref["abs_err"], block=block, seed=seed)
        row.update(d_mae=m, mae_lo=lo, mae_hi=hi)
        row["wins_all"] = (row["ls_lo"] > 0) and (row["rmse_lo"] > 0) and (row["mae_lo"] > 0)
        row["g1_pass"] = row["ls_lo"] > 0 if base in G1_BASELINES else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def memory_table(segs, methods=("adaptive", "abl:no_discount", "sgd+robust", "refresh+robust", "rolling+robust")):
    """Early-segment signal MAE on first visits vs revisits of a regime."""
    s = segs[segs.method.isin(methods)].copy()
    s["visit_type"] = np.where(s["visit"] == 1, "first visit", "revisit")
    return s.pivot_table(index="method", columns="visit_type", values="signal_mae_early", aggfunc="mean")


# ---------------------------------------------------------------- plots
def plot_rolling_error(data, results, path, metric="mse", window=1_000,
                       names=("adaptive", "frozen+robust", "sgd+robust", "refresh+robust", "rolling+robust")):
    """Rolling signal error of the forecast mean; metric is "mse" or "mae"."""
    sig = data["signal"]
    kernel = np.ones(window) / window
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in names:
        if name in results:
            gap = results[name]["mu"] - sig
            err = gap ** 2 if metric == "mse" else np.abs(gap)
            ax.plot(np.convolve(err, kernel, mode="valid"), label=name, lw=1)
    for k, a in enumerate(data["seg_starts"]):
        ax.axvline(a, color="grey", ls=":", lw=0.8)
        ax.text(a + 100, ax.get_ylim()[1] * 0.97, data["regime"][a], va="top", fontsize=8)
    ax.set_xlabel("stream event")
    ax.set_ylabel(f"rolling signal {metric.upper()} (window {window})")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_influence(log_r, log_dpred, path):
    r = np.abs(np.asarray(log_r))
    d = np.abs(np.asarray(log_dpred))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(r, d, s=2, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|residual|")
    ax.set_ylabel("|change of prediction at x|")
    ax.set_title("Empirical influence (Prop. 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pit(data, results, path, names=("adaptive", "frozen+ewma")):
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        r = results[name]
        u = M.pit(data["y_clean"], r["mu"], r["s"], r["df"])
        ax.hist(u, bins=20, density=True)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_title(f"PIT: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- tuning
def tune(dgp_cfg, ens_cfg, hp):
    """Pick hyperparameters on a separate seed, by mean log score on clean labels."""
    data, booster, leafmap, idx = prepare(replace(dgp_cfg, seed=dgp_cfg.seed + 1_000), ens_cfg)
    y, h = data["y_clean"], data["h"]

    def ls(mu, s, df):
        return M.log_score(y, mu, s, df).mean()

    best = {}
    scores = {eta: ls(m, *B.robust_t_scale(m, data["y_obs"], h))
              for eta in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
              for m in [B.online_sgd(leafmap, idx, data["y_obs"], h, eta0=eta)]}
    best["sgd_eta0"] = max(scores, key=scores.get)

    scores = {}
    for P0 in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3):
        for delta in (0.998, 0.999, 0.9995, 0.9999, 1.0):
            cfg = replace(hp["adaptive"], P0=P0, P0_bias=10 * P0, delta=delta)
            (mu, s, df), _, _ = run_adaptive(leafmap, idx, data, cfg)
            scores[(P0, delta)] = ls(mu, s, df)
    P0, delta = max(scores, key=scores.get)
    best["adaptive"] = replace(hp["adaptive"], P0=P0, delta=delta)

    scores = {}
    for window in (2_500, 5_000, 10_000, 20_000, 40_000):
        mu = B.xgb_refresh(booster, ens_cfg, data["X"], data["y_obs"], h, every=2_000, window=window)
        scores[window] = ls(mu, *B.robust_t_scale(mu, data["y_obs"], h))
    best["refresh"] = dict(every=2_000, window=max(scores, key=scores.get))

    scores = {}
    for window in (5_000, 10_000, 20_000, 40_000):
        mu = B.xgb_rolling(ens_cfg, data["X"], data["y_obs"], h, every=2_000, window=window,
                           X_pre=data["X_pre"], y_pre=data["y_pre"])
        scores[window] = ls(mu, *B.robust_t_scale(mu, data["y_obs"], h))
    best["rolling"] = dict(every=2_000, window=max(scores, key=scores.get))
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["disjoint", "shared"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--bad-tick-rate", type=float, default=0.001)
    ap.add_argument("--bad-tick-scale", type=float, default=10.0)
    ap.add_argument("--innov-df", type=float, default=None, help="Student-t df of noise innovations")
    ap.add_argument("--seg-len", type=int, default=20_000)
    ap.add_argument("--regimes", default="ABAC", help="regime sequence, e.g. ABACBCABACBCABAC")
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-cache", action="store_true", help="recompute even if a valid cache exists")
    ap.add_argument("--replot", action="store_true",
                    help="only rebuild tables and plots from existing caches; never recompute")
    args = ap.parse_args()
    if args.replot and args.no_cache:
        ap.error("--replot and --no-cache are mutually exclusive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ens_cfg = EnsembleConfig(n_trees=args.n_trees)
    g1_all, summary_all, seg_all = [], [], []

    for variant in args.variants:
        for seed in args.seeds:
            dgp_cfg = DGPConfig(variant=variant, seed=seed, h=args.h,
                                bad_tick_rate=args.bad_tick_rate, bad_tick_scale=args.bad_tick_scale,
                                innov_df=args.innov_df, seg_len=args.seg_len,
                                regimes=tuple(args.regimes))
            hp = dict(adaptive=AdaptiveConfig(), sgd_eta0=0.01,
                      refresh=dict(every=2_000, window=5_000),
                      rolling=dict(every=2_000, window=10_000))
            run_dir = out / f"{variant}_seed{seed}"
            spec = dict(dgp=asdict(dgp_cfg), ensemble=asdict(ens_cfg), tune=args.tune,
                        ablations=args.ablations, defaults=dict(hp, adaptive=asdict(hp["adaptive"])))
            key = cache.cache_key(spec, functions=(prepare, run_all, run_adaptive, tune))
            cached = None if args.no_cache else cache.load(run_dir)

            if args.replot:
                if cached is None:
                    print(f"\n=== {variant}, seed {seed}: no cache in {run_dir}, skipped")
                    continue
                if cached[0] != key:
                    print(f"\n[warning] {run_dir}: cache was made with different settings or code;"
                          " replotting it anyway (--replot never recomputes)")
                _, data, results, diag, meta = cached
            elif cached is not None and cached[0] == key:
                print(f"\n[cache] {run_dir}: reusing predictions from {cached[4]['created']}")
                _, data, results, diag, meta = cached
            else:
                if cached is not None:
                    print(f"\n[cache] {run_dir}: settings or code changed, recomputing")
                if args.tune:
                    hp = tune(dgp_cfg, ens_cfg, hp)
                data, booster, leafmap, idx = prepare(dgp_cfg, ens_cfg)
                results, filt = run_all(data, booster, leafmap, idx, ens_cfg, hp, args.ablations)
                diag = dict(log_r=filt.log_r, log_dpred=filt.log_dpred)
                meta = dict(dgp=asdict(dgp_cfg), ensemble=asdict(ens_cfg),
                            adaptive=asdict(hp["adaptive"]), sgd_eta0=hp["sgd_eta0"],
                            refresh=hp["refresh"], rolling=hp["rolling"],
                            alarms=filt.alarms, n_leaves=leafmap.size)
                cache.save(run_dir, key, data, results, diag, meta)
            alarms = meta["alarms"]

            summary, segs, per_event = score(data, results)
            g1 = compare(per_event, data["h"], seed)
            for df_ in (summary, segs, g1):
                df_.insert(0, "seed", seed)
                df_.insert(0, "variant", variant)
            summary_all.append(summary)
            seg_all.append(segs)
            g1_all.append(g1)

            run_dir.mkdir(parents=True, exist_ok=True)
            plot_rolling_error(data, results, run_dir / "rolling_signal_mse.png", metric="mse")
            plot_rolling_error(data, results, run_dir / "rolling_signal_mae.png", metric="mae")
            plot_influence(diag["log_r"], diag["log_dpred"], run_dir / "influence.png")
            plot_pit(data, results, run_dir / "pit.png")
            config = {k: meta[k] for k in ("dgp", "ensemble", "adaptive", "sgd_eta0", "refresh",
                                            "rolling", "alarms", "n_leaves")}
            (run_dir / "config.json").write_text(json.dumps(config, indent=4, default=str))

            print(f"\n=== {variant}, seed {seed}: alarms at {alarms[:10]}"
                  f"{' ...' if len(alarms) > 10 else ''} ({len(alarms)} total)")
            print(summary.drop(columns=["variant", "seed"]).to_string(index=False, float_format="%.4f"))
            print(g1.drop(columns=["variant", "seed", "reference"]).to_string(index=False, float_format="%.4f"))
            print("early-segment signal MAE, first visit vs revisit:")
            print(memory_table(segs).to_string(float_format="%.4f"))

    if not summary_all:
        print("nothing to report")
        return
    pd.concat(summary_all).to_csv(out / "summary.csv", index=False)
    pd.concat(seg_all).to_csv(out / "segments.csv", index=False)
    g1 = pd.concat(g1_all)
    g1.to_csv(out / "comparisons.csv", index=False)
    g1_rows = g1[g1["baseline"].isin(G1_BASELINES)]
    print("\nGate G1 passed on all runs:", bool(g1_rows["g1_pass"].astype(bool).all()))


if __name__ == "__main__":
    main()
