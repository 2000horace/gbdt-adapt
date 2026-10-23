"""Event-level experiment on preprocessed raw LOBSTER data (see phase1/events.py).

Example
-------
    # quick look: cheap methods only, two units
    python run_events.py --tickers AMZN NVDA --months 2019-10 --skip-batch --out runs/events_quick
    # full run
    python run_events.py --workers 6 --out runs/events

Each finished unit is cached in <out>/units/<TICKER>-<MONTH>.json (with its settings); a
rerun with the same settings skips it, so interrupted runs resume. Memory: the largest
units (~40 million events) need ~10 GB each, so choose --workers accordingly.
"""

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from phase1 import events as EV


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit_job(cfg_dict):
    cfg = EV.EventRunConfig(**cfg_dict)
    try:
        rows, seconds, meta = EV.run_unit(cfg)
        return dict(status="ok", rows=rows, seconds=seconds, meta=meta, settings=cfg_dict)
    except Exception as err:                      # noqa: BLE001
        import traceback
        return dict(status="error", unit=f"{cfg.ticker}-{cfg.month}", error=f"{type(err).__name__}: {err}",
                    trace=traceback.format_exc(), settings=cfg_dict)


def summarize(per_unit, reference="adaptive"):
    """Gain of `reference` over each method per month, with a t-interval across stocks."""
    out = []
    for month, d in per_unit.groupby("month"):
        ref = d[d.method == reference].set_index("ticker")
        for name in d.method.unique():
            if name == reference:
                continue
            oth = d[d.method == name].set_index("ticker").loc[ref.index]
            row = dict(month=month, vs=name, n=len(ref))
            for m, sign in (("log_score", 1), ("crps", -1), ("mae", -1), ("rmse", -1), ("corr", 1), ("dir_acc", 1)):
                if m not in d.columns:
                    continue
                g = sign * (ref[m] - oth[m]).to_numpy()
                n = len(g)
                half = stats.t.ppf(0.975, n - 1) * g.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                row.update({f"{m}_gain": g.mean(), f"{m}_lo": g.mean() - half, f"{m}_hi": g.mean() + half,
                            f"{m}_win": float(np.mean(g > 0))})
            out.append(row)
    return pd.DataFrame(out)


def print_gains(g):
    def cell(r, m):
        mark = "*" if r[f"{m}_lo"] > 0 else ("-" if r[f"{m}_hi"] < 0 else " ")
        return f"{r[f'{m}_gain']:+.4f}{mark} ({100 * r[f'{m}_win']:.0f}%)"
    for month, d in g.groupby("month"):
        print(f"\n{month}: gain of adaptive (n = {int(d.n.iloc[0])} stocks)")
        ms = [m for m in ("log_score", "crps", "mae", "rmse", "corr", "dir_acc") if f"{m}_gain" in d.columns]
        print(f"  {'vs':26s} " + " ".join(f"{m:>16s}" for m in ms))
        for _, r in d.iterrows():
            print(f"  {r.vs:26s} " + " ".join(f"{cell(r, m):>16s}" for m in ms))
    print("gain = adaptive better if positive; * / - : 95% CI across stocks excludes 0; (%) = stocks won")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/lobster_events")
    ap.add_argument("--months", nargs="+", default=["2019-10", "2020-03"])
    ap.add_argument("--tickers", nargs="+", default=None, help="default: every ticker folder in --root")
    ap.add_argument("--h", type=int, default=10_000)
    ap.add_argument("--target", default="logrv", choices=["logrv", "return"],
                    help="logrv: log realized variance over the horizon; return: mid-price return over the horizon")
    ap.add_argument("--pre-days", type=int, default=3)
    ap.add_argument("--tune-days", type=int, default=2)
    ap.add_argument("--batch", nargs="+", default=["1d/1h", "1d/1d", "5d/1d"],
                    help="refresh / rolling retrain settings as window/refit-every, in trading days (d) or hours (h)")
    ap.add_argument("--skip-batch", action="store_true", help="skip refresh and rolling retraining (quick look)")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--no-ens-tune", action="store_true", help="fixed ensemble settings instead of Bayesian optimisation")
    ap.add_argument("--ens-trials", type=int, default=30)
    ap.add_argument("--tune-threads", type=int, default=4, help="XGBoost threads for ensemble training (one-off)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="runs/events")
    ap.add_argument("--batch-cache", default="runs/event_batch_cache",
                    help="folder for cached refresh / rolling predictions, reused across runs ('' to disable)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    tickers = args.tickers or sorted(p.name for p in Path(args.root).iterdir() if p.is_dir())
    base = dict(root=args.root, h=args.h, target=args.target, ens_tune=not args.no_ens_tune,
                ens_trials=args.ens_trials, tune_threads=args.tune_threads, pre_days=args.pre_days, tune_days=args.tune_days, chunk=args.chunk,
                skip_batch=args.skip_batch, batch_cache=args.batch_cache, batch=tuple(tuple(b.split("/")) for b in args.batch))
    jobs, cached = [], []
    for m in args.months:
        for t in tickers:
            n_events = sum(json.loads(f.read_text())["n_kept"] for f in Path(args.root, t).glob(f"{m}-*.json"))
            if n_events == 0:
                log(f"skipping {t} {m}: no processed days")
                continue
            cfg = asdict(EV.EventRunConfig(ticker=t, month=m, **base))
            cfg = {k: (list(map(list, v)) if k == "batch" else list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.items()}
            cache = out / "units" / f"{t}-{m}.json"
            if cache.exists() and not args.no_cache and json.loads(cache.read_text()).get("settings") == cfg:
                cached.append(cache)
            else:
                jobs.append((n_events, cfg))
    jobs.sort(key=lambda j: -j[0])                    # largest first, for load balance
    log(f"{len(jobs) + len(cached)} units: {len(cached)} cached, {len(jobs)} to run on {args.workers} workers")

    t0 = time.perf_counter()
    if jobs:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, res in enumerate(pool.imap_unordered(unit_job, [c for _, c in jobs]), 1):
                el = (time.perf_counter() - t0) / 60
                if res["status"] == "ok":
                    u = res["meta"]["unit"]
                    (out / "units" / f"{u}.json").write_text(json.dumps(res, indent=1, default=float))
                    log(f"{k}/{len(jobs)} {u} done in {res['meta']['total_seconds'] / 60:.1f} min "
                        f"({res['meta']['n_eval_events']:,} evaluation events) | elapsed {el:.1f} min")
                else:
                    log(f"{k}/{len(jobs)} {res['unit']}: ERROR {res['error']}")
                    (out / "units" / f"{res['unit']}.error.txt").write_text(res["trace"])

    rows, secs, metas = [], [], []
    for f in sorted((out / "units").glob("*.json")):
        res = json.loads(f.read_text())
        meta = res["meta"]
        for r in res["rows"]:
            rows.append(dict(ticker=meta["ticker"], month=meta["month"], **r))
        secs.append(dict(ticker=meta["ticker"], month=meta["month"], n_eval_events=meta["n_eval_events"],
                         **res["seconds"]))
        metas.append({**{k: v for k, v in meta.items() if not isinstance(v, dict)},
                      "P0": meta["adaptive"]["P0"], "delta": meta["adaptive"]["delta"],
                      "overlap": meta["adaptive"]["overlap_factor"]})
    if not rows:
        log("no finished units")
        return
    per_unit = pd.DataFrame(rows)
    per_unit.to_csv(out / "per_unit.csv", index=False)
    pd.DataFrame(secs).to_csv(out / "timing.csv", index=False)
    pd.DataFrame(metas).to_csv(out / "units.csv", index=False)
    means = per_unit.groupby(["month", "method"]).mean(numeric_only=True)
    means.to_csv(out / "means.csv")
    gains = summarize(per_unit)
    gains.to_csv(out / "gains.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nUnits:")
    um = pd.DataFrame(metas)
    ucols = [c for c in ("unit", "n_eval_events", "events_per_day", "horizon_median_seconds", "floor_share",
                         "scale_memory", "P0", "delta", "overlap", "sgd_eta0", "grid_edges", "ens_learning_rate",
                         "ens_n_trees", "ens_mcw_per_million", "ens_edges") if c in um.columns]
    print(um[ucols].to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    print("\nMean scores per month:")
    cols = [c for c in ("log_score", "crps", "coverage80", "rmse", "mae", "corr", "dir_acc", "r2_oos") if c in means]
    print(means[cols].to_string(float_format="%.4f"))
    print_gains(gains)
    t = pd.DataFrame(secs)
    per_event = (t.drop(columns=["ticker", "month", "n_eval_events"]).div(t["n_eval_events"], axis=0) * 1e6).mean()
    print("\nMean CPU microseconds per evaluation event, by component:")
    print(per_event.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
