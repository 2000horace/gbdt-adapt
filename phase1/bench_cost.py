"""Matched-compute benchmark: every method on ONE CPU core.

What is timed (median over repeats, after warm-up and compilation):

    tree inference      leaf indices for one event: batched (amortised) and one row at a time
    adaptive update     the online filter's per-event step (JAX on CPU, one config, one stream);
                        also the plain-Python reference implementation, as an upper bound
    SGD update          online SGD on leaves, per event (JAX on CPU)
    refresh refit       one XGBoost `refresh` of all leaves on a window of W rows
    rolling refit       one full XGBoost retrain (same trees and depth) on W rows

From these, the cost of one trading day for one instrument, at an event rate E per day:

    adaptive:  E * (inference + update)
    rolling:   E * inference + (E / I) * refit(W)      I = events between refits

with W either fixed (40,000 rows) or covering a fixed time span (rows grow with E).

Example:
    python bench_cost.py --source lobster --year 2019 --tickers AAPL MSFT NVDA --out runs/bench
    python bench_cost.py --source synthetic --out runs/bench_synth
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()

import argparse                                    # noqa: E402
import json                                        # noqa: E402
import time                                        # noqa: E402
from pathlib import Path                           # noqa: E402

import jax                                         # noqa: E402
import jax.numpy as jnp                            # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
import xgboost as xgb                              # noqa: E402

from phase1 import fast_cpu as F                   # noqa: E402
from phase1 import jax_filters as J                # noqa: E402
from phase1 import lobster as L                    # noqa: E402
from phase1.adaptive import AdaptiveConfig, AdaptiveLeafFilter   # noqa: E402
from phase1.dgp import DGPConfig                   # noqa: E402
from phase1.dgp_drift import generate              # noqa: E402
from phase1.ensemble import EnsembleConfig, LeafMap, train_ensemble   # noqa: E402

EVENTS_PER_DAY = {  # instrument-level event rates for the cost table
    "1-minute bars": 390,
    "1-second bars": 23_400,
    "order-book events (100k/day)": 100_000,
    "order-book events (1M/day)": 1_000_000,
}


def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def load(args, ticker=None, seed=0):
    if args.source == "lobster":
        cfg = L.LobsterConfig(root=args.lobster_root, ticker=ticker, year=args.year, pre_days=args.pre_days)
        return L.load_unit(cfg), ticker
    cfg = DGPConfig(variant="gradual", seed=seed, regimes=tuple("ABACBCABACBCABAC"),
                    bad_tick_rate=0.01, innov_df=3.0)
    return generate(cfg), f"synthetic-{seed}"


def bench_unit(args, data, name, cfg):
    ens = EnsembleConfig(n_trees=args.n_trees, max_depth=args.max_depth)   # nthread=1 in its params
    booster = train_ensemble(data["X_pre"], data["y_pre"], ens)
    lm = LeafMap(booster)
    lm.calibrate_bias(data["X_pre"])
    X, y = data["X"], data["y_obs"]
    T = len(y)
    idx = lm.leaf_index(X)
    rows = []

    # --- tree inference
    dm = xgb.DMatrix(X)
    t = median_time(lambda: booster.predict(dm, pred_leaf=True), args.repeats)
    rows.append(dict(unit=name, item="tree inference, batched", seconds=t / T, per="event"))
    n1 = min(2_000, T)
    single = [xgb.DMatrix(X[i:i + 1]) for i in range(n1)]
    t = median_time(lambda: [booster.predict(d, pred_leaf=True) for d in single], max(1, args.repeats // 2))
    rows.append(dict(unit=name, item="tree inference, one row at a time", seconds=t / n1, per="event"))

    # --- online updates (JAX on one CPU core); first call compiles
    is_bias = np.zeros(lm.size, bool)
    is_bias[lm.bias_index] = True
    p = {k: jnp.asarray(v, jnp.float32) for k, v in J.adaptive_params(cfg, data["h"]).items()}
    w0, ib = jnp.asarray(lm.w0, jnp.float32), jnp.asarray(is_bias)
    ij, yj = jnp.asarray(idx.astype(np.int32)), jnp.asarray(y, jnp.float32)
    f = jax.jit(J.adaptive_filter, static_argnames=("h",))
    f(p, w0, ib, ij, yj, h=data["h"])[0].block_until_ready()
    t = median_time(lambda: f(p, w0, ib, ij, yj, h=data["h"])[0].block_until_ready(), args.repeats)
    rows.append(dict(unit=name, item="adaptive update (JAX, CPU)", seconds=t / T, per="event"))
    g = jax.jit(J.online_sgd, static_argnames=("h",))
    g(jnp.float32(0.01), w0, ij, yj, h=data["h"]).block_until_ready()
    t = median_time(lambda: g(jnp.float32(0.01), w0, ij, yj, h=data["h"]).block_until_ready(), args.repeats)
    rows.append(dict(unit=name, item="SGD update (JAX, CPU)", seconds=t / T, per="event"))
    # lean compiled CPU implementation (Numba): whole stream, and one call per event from Python
    F.OnlineFilter(lm.w0, lm.bias_index, cfg, data["h"]).run(idx, y)          # compile
    t = median_time(lambda: F.OnlineFilter(lm.w0, lm.bias_index, cfg, data["h"]).run(idx, y), args.repeats)
    rows.append(dict(unit=name, item="adaptive update (Numba, CPU)", seconds=t / T, per="event"))
    F.run_sgd(lm.w0, idx, y, data["h"], 0.01)
    t = median_time(lambda: F.run_sgd(lm.w0, idx, y, data["h"], 0.01), args.repeats)
    rows.append(dict(unit=name, item="SGD update (Numba, CPU)", seconds=t / T, per="event"))
    filt, i64, h_ = F.OnlineFilter(lm.w0, lm.bias_index, cfg, data["h"]), idx.astype(np.int64), data["h"]
    n_call = min(20_000, T)

    def per_call():
        for i in range(n_call):
            j = i - h_
            filt.step(i, i64[i], i64[j] if j >= 0 else None, y[j] if j >= 0 else 0.0)
    per_call()
    t = median_time(per_call, max(1, args.repeats // 2))
    rows.append(dict(unit=name, item="adaptive update (Numba, one call per event from Python)",
                     seconds=t / n_call, per="event"))
    n_py = min(args.python_events, T)
    ref = AdaptiveLeafFilter(lm.w0, lm.bias_index, cfg, data["h"])
    t0 = time.perf_counter()
    ref.run(idx[:n_py], y[:n_py])
    rows.append(dict(unit=name, item="adaptive update (plain Python reference)",
                     seconds=(time.perf_counter() - t0) / n_py, per="event"))

    # --- refits on windows of W rows (rows beyond the data are tiled, for timing only)
    Xa = np.vstack([data["X_pre"], X])
    ya = np.concatenate([data["y_pre"], y])
    params = dict(ens.params(), process_type="update", updater="refresh", refresh_leaf=True)
    for W in args.windows:
        reps = int(np.ceil(W / len(ya)))
        Xw, yw = np.tile(Xa, (reps, 1))[:W], np.tile(ya, reps)[:W]
        dw = xgb.DMatrix(Xw, label=yw)
        t = median_time(lambda: xgb.train(ens.params(), dw, num_boost_round=ens.n_trees), args.refit_repeats)
        rows.append(dict(unit=name, item=f"rolling refit, W={W:,}", seconds=t, per="refit", window=W))
        t = median_time(lambda: xgb.train(params, dw, num_boost_round=booster.num_boosted_rounds(),
                                          xgb_model=booster.copy()), args.refit_repeats)
        rows.append(dict(unit=name, item=f"refresh refit, W={W:,}", seconds=t, per="refit", window=W))
    return rows


def cost_table(t, windows, refits_per_day):
    """CPU seconds per instrument and trading day for ADAPTING the model (adaptive filter: the
    compiled Numba implementation).

    Tree inference is the same for every method (each forecast needs the leaf indices), so it
    is reported separately and left out here. Refit windows are either fixed at 40,000 rows or
    cover the same time span as 40,000 minutes (rows grow with the event rate; refit cost is
    extrapolated linearly from the measured windows, so large-window rows are estimates).
    """
    upd = t["adaptive update (Numba, CPU)"]
    w_fixed = 40_000 if 40_000 in windows else sorted(windows)[len(windows) // 2]
    ws = np.array(sorted(windows), float)
    cs = np.array([t[f"rolling refit, W={int(w):,}"] for w in ws])
    cr = np.array([t[f"refresh refit, W={int(w):,}"] for w in ws])
    slope_roll, icpt_roll = np.polyfit(ws, cs, 1)
    slope_ref, icpt_ref = np.polyfit(ws, cr, 1)
    rows = []
    for rate, E in EVENTS_PER_DAY.items():
        span_rows = max(w_fixed, E * 40_000 / 390)
        rows.append(dict(event_rate=rate, method="adaptive: per-event update", window="-",
                         seconds_per_day=E * upd))
        for per_day in refits_per_day:
            for kind, fixed_cost, slope, icpt in (("rolling retrain", t[f"rolling refit, W={w_fixed:,}"], slope_roll, icpt_roll),
                                                  ("refresh", t[f"refresh refit, W={w_fixed:,}"], slope_ref, icpt_ref)):
                rows.append(dict(event_rate=rate, method=f"{kind}: {per_day:g} refits/day", window="fixed 40k rows",
                                 seconds_per_day=per_day * fixed_cost))
                rows.append(dict(event_rate=rate, method=f"{kind}: {per_day:g} refits/day", window="same time span",
                                 seconds_per_day=per_day * max(fixed_cost, icpt + slope * span_rows),
                                 window_rows=span_rows))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lobster", choices=["lobster", "synthetic"])
    ap.add_argument("--lobster-root", default="/nfs/data/lobster_data/lobster_minutely/3")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    ap.add_argument("--pre-days", type=int, default=50)
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--windows", type=int, nargs="+", default=[10_000, 40_000, 160_000])
    ap.add_argument("--refits-per-day", type=float, nargs="+", default=[0.2, 1, 10, 100])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--refit-repeats", type=int, default=3)
    ap.add_argument("--python-events", type=int, default=5_000)
    ap.add_argument("--out", default="runs/bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"JAX devices: {jax.devices()}  (XLA_FLAGS={os.environ['XLA_FLAGS']})", flush=True)

    cfg = AdaptiveConfig(P0=1e-3, P0_bias=1e-2, delta=1.0, overlap_factor=10.0)
    rows = []
    units = args.tickers if args.source == "lobster" else [0, 1, 2][:len(args.tickers)]
    for u in units:
        data, name = load(args, ticker=u, seed=u if args.source == "synthetic" else 0)
        print(f"{name}: {len(data['y_obs']):,} events", flush=True)
        rows += bench_unit(args, data, name, cfg)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "bench_raw.csv", index=False)
    t = raw.groupby("item")["seconds"].median().to_dict()
    summary = pd.Series(t).rename("median seconds").to_frame()
    summary["per"] = raw.groupby("item")["per"].first()
    summary.to_csv(out / "bench_summary.csv")
    costs = cost_table(t, args.windows, args.refits_per_day)
    windows_used = args.windows
    costs.to_csv(out / "cost_per_day.csv", index=False)
    (out / "bench_config.json").write_text(json.dumps(vars(args), indent=4))

    pd.set_option("display.width", 200)
    print("\nMedian cost on one CPU core:")
    for item, r in summary.iterrows():
        v = r["median seconds"]
        print(f"  {item:45s} {v * 1e6:12.1f} us per {r['per']}" if r["per"] == "event"
              else f"  {item:45s} {v:12.2f} s  per {r['per']}")
    infer = t["tree inference, one row at a time"]
    print(f"\nShared by all methods: tree inference ~{infer * 1e6:.0f} us per event one row at a time "
          f"({t['tree inference, batched'] * 1e6:.1f} us batched), i.e. "
          + ", ".join(f"{E * infer:,.0f} s/day at {r}" for r, E in EVENTS_PER_DAY.items()))
    for window in ("fixed 40k rows", "same time span"):
        sub = costs[(costs.window == window) | (costs.method.str.startswith("adaptive"))]
        piv = sub.pivot_table(index="method", columns="event_rate", values="seconds_per_day", sort=False)
        print(f"\nCPU seconds per instrument and trading day to ADAPT the model (refit window: {window}):")
        print(piv[list(EVENTS_PER_DAY)].to_string(float_format=lambda v: f"{v:,.2f}"))
    upd = t["adaptive update (Numba, CPU)"]
    print(f"\nBreak-even refits per day (retraining costs more above this), adaptive update {upd * 1e6:.2f} us/event:")
    w_fixed = 40_000 if 40_000 in windows_used else sorted(windows_used)[len(windows_used) // 2]
    for E_name, E in EVENTS_PER_DAY.items():
        fixed = E * upd / t[f"rolling refit, W={w_fixed:,}"]
        span = costs[(costs.event_rate == E_name) & (costs.window == "same time span")
                     & (costs.method == "rolling retrain: 1 refits/day")]["seconds_per_day"].iloc[0]
        print(f"  {E_name:32s} fixed {w_fixed:,}-row window: {fixed:8.2f} /day   "
              f"same-time-span window: {E * upd / span:8.4f} /day (once every {span / (E * upd):,.0f} days)")
    print("\n'same time span' windows hold 40,000 minutes of events; their refit costs are linear "
          "extrapolations from the measured windows.")


if __name__ == "__main__":
    main()
