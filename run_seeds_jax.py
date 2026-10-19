"""Phase 1 over many seeds, with the online filters in JAX (GPU).

Pipeline
--------
1. Tuning on separate seeds: the adaptive filter over a (P0, delta) grid and SGD over
   a step-size grid, all vmapped in one call each; optionally the XGBoost windows.
2. Evaluation in batches of seeds: worker processes generate data, pre-train XGBoost
   and run the XGBoost baselines; JAX runs the adaptive filter (+ ablations), SGD and
   the predictive-scale wrappers for all seeds of the batch at once.
3. Aggregation: per-seed metrics, and the gain of "adaptive" over each method with a
   95% CI across seeds (seeds are independent, so a plain t-interval is valid).

Example (GPU)
-------------
    XLA_PYTHON_CLIENT_PREALLOCATE=false python run_seeds_jax.py \\
        --variant disjoint_drift --n-seeds 100 --batch 25 --workers 16 --xgb --ablations

Use --x64 for float64 (slower on most GPUs; float32 is the default).
"""

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from scipy import stats

from phase1 import metrics as M
from phase1.adaptive import AdaptiveConfig
from phase1.dgp import DGPConfig
from phase1.ensemble import EnsembleConfig
from phase1.prepare import prepare_seed

ABLATIONS = {
    "abl:no_robust": dict(use_robust=False),
    "abl:no_discount": dict(use_discount=False),
    "abl:no_scale": dict(use_scale=False),
    "abl:no_calibration": dict(use_calibration=False),
    "abl:no_overlap": dict(overlap_factor=1.0),
    "abl:no_alarm_action": dict(use_alarm_action=False),
    "abl:raw_z2_detector": dict(detector="raw", zeta=0.5, lam=30.0),
}
P0_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
DELTA_GRID = (0.998, 0.999, 0.9995, 0.9999, 1.0)
SGD_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
EARLY = 1_000


# ---------------------------------------------------------------- helpers
def make_jobs(args, seeds, refresh_windows, rolling_windows):
    ens = EnsembleConfig(n_trees=args.n_trees, max_depth=args.max_depth, device=args.xgb_device)
    jobs = []
    for sd in seeds:
        dgp = DGPConfig(variant=args.variant, seed=sd, h=args.h, seg_len=args.seg_len,
                        regimes=tuple(args.regimes), bad_tick_rate=args.bad_tick_rate,
                        bad_tick_scale=args.bad_tick_scale, innov_df=args.innov_df)
        jobs.append(dict(dgp=dgp, ens=replace(ens, seed=sd), every=args.every,
                         refresh_windows=refresh_windows, rolling_windows=rolling_windows))
    return jobs


def stack(items, D_pad, dtype):
    from phase1.jax_filters import pad_batch
    w0, is_bias, idx = pad_batch(items, dtype=dtype)
    if w0.shape[1] < D_pad:   # pad to a fixed size so jit compiles once
        extra = D_pad - w0.shape[1]
        w0 = np.pad(w0, ((0, 0), (0, extra)))
        is_bias = np.pad(is_bias, ((0, 0), (0, extra)))
    y_obs = np.stack([it["y_obs"] for it in items]).astype(dtype)
    y_clean = np.stack([it["y_clean"] for it in items]).astype(dtype)
    return w0, is_bias, idx, y_obs, y_clean


def seed_metrics(item, mu, s, df):
    y, sig = item["y_clean"], item["signal"]
    mu = np.asarray(mu, float)
    s = np.asarray(s, float)
    out = dict(
        log_score=M.log_score(y, mu, s, df).mean(),
        crps=M.crps(y, mu, s, df).mean(),
        coverage80=M.coverage(y, mu, s, df, alpha=0.2),
        rmse=np.sqrt(np.mean((y - mu) ** 2)),
        mae=np.mean(np.abs(y - mu)),
        signal_rmse=np.sqrt(np.mean((mu - sig) ** 2)),
        signal_mae=np.mean(np.abs(mu - sig)),
    )
    # early-segment signal MAE on first visits vs revisits of a regime
    starts = list(item["seg_starts"]) + [len(y)]
    regimes = [item["regime"][a] for a in starts[:-1]]
    first, again = [], []
    for k in range(len(regimes)):
        a = starts[k]
        e = np.mean(np.abs(mu[a:a + EARLY] - sig[a:a + EARLY]))
        (first if regimes[:k].count(regimes[k]) == 0 else again).append(e)
    out["early_mae_first"] = np.mean(first)
    out["early_mae_revisit"] = np.mean(again) if again else np.nan
    return out


def summarize_gains(df, reference="adaptive"):
    """Gain of reference over every other method, per seed, then mean and 95% CI over seeds."""
    rows = []
    ref = df[df.method == reference].set_index("seed")
    for name in df.method.unique():
        if name == reference:
            continue
        other = df[df.method == name].set_index("seed").loc[ref.index]
        gains = {
            "log_score": ref["log_score"] - other["log_score"],
            "crps": other["crps"] - ref["crps"],
            "rmse": other["rmse"] - ref["rmse"],
            "mae": other["mae"] - ref["mae"],
            "signal_rmse": other["signal_rmse"] - ref["signal_rmse"],
            "signal_mae": other["signal_mae"] - ref["signal_mae"],
        }
        row = dict(vs=name, n_seeds=len(ref))
        for metric, g in gains.items():
            g = g.to_numpy()
            n = len(g)
            se = g.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            half = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
            row[f"{metric}_gain"] = g.mean()
            row[f"{metric}_lo"] = g.mean() - half
            row[f"{metric}_hi"] = g.mean() + half
            row[f"{metric}_win"] = np.mean(g > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def print_gains(g):
    def cell(r, m):
        star = "*" if r[f"{m}_lo"] > 0 else ("-" if r[f"{m}_hi"] < 0 else " ")
        return f"{r[f'{m}_gain']:+.4f}{star} ({100 * r[f'{m}_win']:.0f}%)"
    lines = [f"{'vs':22s} {'log score':>18s} {'RMSE':>18s} {'MAE':>18s} {'signal MAE':>18s}"]
    for _, r in g.iterrows():
        lines.append(f"{r['vs']:22s} {cell(r, 'log_score'):>18s} {cell(r, 'rmse'):>18s} "
                     f"{cell(r, 'mae'):>18s} {cell(r, 'signal_mae'):>18s}")
    print("\n".join(lines))
    print("gain = adaptive better if positive; * / - : 95% CI over seeds excludes 0; (%) = share of seeds won")


# ---------------------------------------------------------------- stages
def tune(args, pool, dtype, D_pad):
    from phase1 import jax_filters as J
    seeds = [args.seed_start + 100_000 + k for k in range(args.tune_seeds)]
    rw = list(args.refresh_grid) if (args.xgb and args.tune_xgb) else []
    lw = list(args.rolling_grid) if (args.xgb and args.tune_xgb) else []
    t0 = time.perf_counter()
    items = pool.map(prepare_seed, make_jobs(args, seeds, rw, lw))
    print(f"tuning data prepared in {time.perf_counter() - t0:.1f}s")
    w0, is_bias, idx, y_obs, y_clean = stack(items, D_pad, dtype)
    h = args.h

    cfgs = [AdaptiveConfig(P0=p0, P0_bias=10 * p0, delta=d) for p0 in P0_GRID for d in DELTA_GRID]
    params = J.stack_params([J.adaptive_params(c, h) for c in cfgs])
    t0 = time.perf_counter()
    sc = np.asarray(J.adaptive_tune_scores(params, w0, is_bias, idx, y_obs, y_clean, h)).mean(axis=0)
    best_cfg = cfgs[int(np.nanargmax(sc))]
    print(f"adaptive grid ({len(cfgs)} configs x {len(seeds)} seeds) in {time.perf_counter() - t0:.1f}s:"
          f" P0={best_cfg.P0:g}, delta={best_cfg.delta}")

    etas = np.asarray(SGD_GRID, dtype)
    sc = np.asarray(J.sgd_tune_scores(etas, w0, idx, y_obs, y_clean, h)).mean(axis=0)
    best_eta = float(SGD_GRID[int(np.nanargmax(sc))])
    print(f"sgd eta0={best_eta}")

    best_refresh, best_rolling = args.refresh_window, args.rolling_window
    if rw or lw:
        from phase1 import baselines as B

        def mean_ls(key, w):
            vals = []
            for it in items:
                mu = it[key][w]
                s, df = B.robust_t_scale(mu, it["y_obs"], h)
                vals.append(M.log_score(it["y_clean"], mu, s, df).mean())
            return np.mean(vals)
        if rw:
            best_refresh = max(rw, key=lambda w: mean_ls("refresh", w))
        if lw:
            best_rolling = max(lw, key=lambda w: mean_ls("rolling", w))
        print(f"refresh window={best_refresh}, rolling window={best_rolling}")
    return best_cfg, best_eta, best_refresh, best_rolling


def evaluate(args, pool, dtype, D_pad, best_cfg, best_eta, best_refresh, best_rolling):
    from phase1 import jax_filters as J
    h = args.h
    names = ["adaptive"] + (list(ABLATIONS) if args.ablations else [])
    cfgs = [best_cfg] + ([replace(best_cfg, **ch) for ch in ABLATIONS.values()] if args.ablations else [])
    params = J.stack_params([J.adaptive_params(c, h) for c in cfgs])
    nus = [c.nu for c in cfgs]

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    rows, timing = [], []
    rw = [best_refresh] if args.xgb else []
    lw = [best_rolling] if args.xgb else []
    for b0 in range(0, len(seeds), args.batch):
        batch = seeds[b0:b0 + args.batch]
        t0 = time.perf_counter()
        items = pool.map(prepare_seed, make_jobs(args, batch, rw, lw))
        t_prep = time.perf_counter() - t0
        w0, is_bias, idx, y_obs, y_clean = stack(items, D_pad, dtype)

        t0 = time.perf_counter()
        mu_a, s_a, fire = J.adaptive_batch(params, w0, is_bias, idx, y_obs, h)
        mu_a.block_until_ready()
        t_adapt = time.perf_counter() - t0
        t0 = time.perf_counter()
        mu_g = J.sgd_batch(np.asarray([best_eta], dtype), w0, idx, y_obs, h)[:, 0]
        mu_g.block_until_ready()
        t_sgd = time.perf_counter() - t0

        point = {"frozen+robust": np.stack([it["mu_frozen"] for it in items]).astype(dtype),
                 "sgd+robust": np.asarray(mu_g)}
        if args.xgb:
            point["refresh+robust"] = np.stack([it["refresh"][best_refresh] for it in items]).astype(dtype)
            point["rolling+robust"] = np.stack([it["rolling"][best_rolling] for it in items]).astype(dtype)
        scales = {k: np.asarray(J.robust_scale_batch(v, y_obs, h)) for k, v in point.items()}
        s_ewma = np.asarray(J.ewma_scale_batch(point["frozen+robust"], y_obs, h))

        mu_a, s_a, fire = np.asarray(mu_a), np.asarray(s_a), np.asarray(fire)
        for k, it in enumerate(items):
            for g, name in enumerate(names):
                rows.append(dict(seed=it["seed"], method=name, n_alarms=int(fire[k, g].sum()),
                                 **seed_metrics(it, mu_a[k, g], s_a[k, g], nus[g])))
            for name, mu in point.items():
                rows.append(dict(seed=it["seed"], method=name,
                                 **seed_metrics(it, mu[k], scales[name][k], 4.0)))
            rows.append(dict(seed=it["seed"], method="frozen+ewma",
                             **seed_metrics(it, point["frozen+robust"][k], s_ewma[k], np.inf)))
        timing.append(dict(batch=b0 // args.batch, n=len(batch), prep_s=t_prep, adaptive_s=t_adapt,
                           sgd_s=t_sgd, configs=len(cfgs),
                           xgb_refresh_s=np.mean([it["seconds"].get(f"refresh_{best_refresh}", np.nan) for it in items]),
                           xgb_rolling_s=np.mean([it["seconds"].get(f"rolling_{best_rolling}", np.nan) for it in items])))
        print(f"batch {b0 // args.batch}: {len(batch)} seeds, prep {t_prep:.1f}s, "
              f"adaptive x{len(cfgs)} {t_adapt:.1f}s, sgd {t_sgd:.1f}s")
    return pd.DataFrame(rows), pd.DataFrame(timing)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="disjoint_drift", choices=["disjoint", "shared", "disjoint_drift"])
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tune-seeds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--seg-len", type=int, default=5_000)
    ap.add_argument("--regimes", default="ABACBCABACBCABAC")
    ap.add_argument("--bad-tick-rate", type=float, default=0.01)
    ap.add_argument("--bad-tick-scale", type=float, default=10.0)
    ap.add_argument("--innov-df", type=float, default=3.0)
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--xgb", action="store_true", help="run the refresh and rolling baselines")
    ap.add_argument("--tune-xgb", action="store_true", help="tune their windows on the tuning seeds")
    ap.add_argument("--xgb-device", default="cpu")
    ap.add_argument("--every", type=int, default=2_000)
    ap.add_argument("--refresh-window", type=int, default=40_000)
    ap.add_argument("--rolling-window", type=int, default=40_000)
    ap.add_argument("--refresh-grid", type=int, nargs="+", default=[5_000, 10_000, 20_000, 40_000])
    ap.add_argument("--rolling-grid", type=int, nargs="+", default=[10_000, 20_000, 40_000])
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--out", default="results_jax")
    args = ap.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    dtype = np.float64 if args.x64 else np.float32
    D_pad = args.n_trees * 2 ** args.max_depth + 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        best_cfg, best_eta, best_refresh, best_rolling = tune(args, pool, dtype, D_pad)
        df, timing = evaluate(args, pool, dtype, D_pad, best_cfg, best_eta, best_refresh, best_rolling)

    df.to_csv(out / "per_seed.csv", index=False)
    timing.to_csv(out / "timing.csv", index=False)
    gains = summarize_gains(df)
    gains.to_csv(out / "gains.csv", index=False)
    means = df.groupby("method").mean(numeric_only=True).drop(columns="seed")
    means.to_csv(out / "means.csv")
    (out / "config.json").write_text(json.dumps(dict(
        args=vars(args), adaptive=asdict(best_cfg), sgd_eta0=best_eta,
        refresh_window=best_refresh, rolling_window=best_rolling), indent=4, default=str))

    print("\nMean over seeds:")
    print(means[["log_score", "crps", "coverage80", "rmse", "mae", "signal_rmse", "signal_mae",
                 "early_mae_first", "early_mae_revisit"]].to_string(float_format="%.4f"))
    print(f"\nGain of adaptive over each method ({df.seed.nunique()} seeds):")
    print_gains(gains)


if __name__ == "__main__":
    main()
