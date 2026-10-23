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

Caching and resuming
--------------------
Tuning results and every finished batch of seeds are saved in <out>/cache/, tagged with
a key that hashes the settings and the source code that produces the numbers. A rerun
with the same settings reuses them: an interrupted run resumes, a finished run only
rebuilds the summary tables, and a larger --n-seeds only computes the new seeds.
Cache files carry the key in their names, so results for different settings can share an
--out folder without overwriting each other. --batch and --workers do not affect the key.
--report-only never computes anything; --no-cache deletes the cache and recomputes.
"""

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import shutil
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
from phase1.dgp_drift import DRIFT_VARIANTS
from phase1 import lobster as L
from phase1.ensemble import EnsembleConfig
from phase1.prepare import prepare_seed

ABLATIONS = {
    "abl:no_robust": dict(use_robust=False),
    "abl:no_discount": dict(use_discount=False),
    "abl:no_scale": dict(use_scale=False),
    "abl:no_calibration": dict(use_calibration=False),
    "abl:no_overlap": dict(overlap_factor=1.0),
    "abl:with_alarm_action": dict(use_alarm_action=True),
}
P0_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
DELTA_GRID = (0.998, 0.999, 0.9995, 0.9999, 1.0)
P_MAX_GRID = (4.0, 16.0, 64.0)
OVERLAP_GRID = ("1", "sqrt_h", "h")      # noise inflation for overlapping labels: 1, sqrt(h) or h


def overlap_value(name, h):
    return {"1": 1.0, "sqrt_h": float(np.sqrt(h)), "h": float(h)}[name]
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


PAD_MULTIPLE = 4_096      # padded lengths are rounded up, so jit recompiles rarely


def pad_time(a, T):
    """Pad a (T_k, ...) array to length T by repeating its last row."""
    a = np.asarray(a)
    if len(a) == T:
        return a
    return np.concatenate([a, np.repeat(a[-1:], T - len(a), axis=0)], axis=0)


def stack_padded(items, D_pad, dtype):
    """Like stack(), for streams of different lengths: pad each stream at the end.

    Exact for the online filters, which are causal: padding after a stream's end never
    changes its real predictions. Also returns the mask of real events and the lengths.
    """
    lengths = np.array([len(it["y_obs"]) for it in items])
    T = int(-(-lengths.max() // PAD_MULTIPLE) * PAD_MULTIPLE)
    D = max(D_pad, max(len(it["w0"]) for it in items))
    w0 = np.zeros((len(items), D), dtype)
    is_bias = np.zeros((len(items), D), bool)
    for k, it in enumerate(items):
        w0[k, :len(it["w0"])] = it["w0"]
        is_bias[k, it["bias_index"]] = True
    idx = np.stack([pad_time(it["idx"], T) for it in items])
    idx = idx.astype(np.int16 if D < 2 ** 15 else np.int32)
    y_obs = np.stack([pad_time(it["y_obs"], T) for it in items]).astype(dtype)
    y_clean = np.stack([pad_time(it["y_clean"], T) for it in items]).astype(dtype)
    mask = np.arange(T)[None, :] < lengths[:, None]
    return w0, is_bias, idx, y_obs, y_clean, mask, lengths


def stack_any(items, D_pad, dtype):
    """stack() when all streams have the same length (synthetic), else stack_padded()."""
    if len({len(it["y_obs"]) for it in items}) == 1:
        w0, is_bias, idx, y_obs, y_clean = stack(items, D_pad, dtype)
        return w0, is_bias, idx, y_obs, y_clean, None, np.full(len(items), y_obs.shape[1])
    return stack_padded(items, D_pad, dtype)


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
    metrics = [("log_score", "log score"), ("crps", "CRPS"), ("rmse", "RMSE"), ("mae", "MAE"),
               ("signal_mae", "signal MAE")]
    metrics = [(m, n) for m, n in metrics if g[f"{m}_gain"].notna().any()]
    lines = [f"{'vs':22s} " + " ".join(f"{n:>18s}" for _, n in metrics)]
    for _, r in g.iterrows():
        lines.append(f"{r['vs']:22s} " + " ".join(f"{cell(r, m):>18s}" for m, _ in metrics))
    print("\n".join(lines))
    print("gain = adaptive better if positive; * / - : 95% CI over seeds excludes 0; (%) = share of seeds won")


# ---------------------------------------------------------------- logging
def log(msg=""):
    """Timestamped, immediately flushed output (stdout is block-buffered under nohup)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else "", flush=True)


def _fmt(sec):
    sec = int(round(sec))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def prepare_many(pool, jobs, label, verbose=False):
    """Run prepare_seed over jobs in the worker pool, logging every finished seed with an ETA.

    Returns the items sorted by seed.
    """
    jobs = [dict(j, verbose=verbose) for j in jobs]
    n = len(jobs)
    any_bl = any(j.get("refresh_windows") or j.get("rolling_windows") for j in jobs)
    log(f"{label}: preparing {n} seeds on {pool._processes} workers "
        f"(data, XGBoost{', refresh/rolling baselines' if any_bl else ''})")
    t0 = time.perf_counter()
    items = []
    for it in pool.imap_unordered(prepare_seed, jobs):
        items.append(it)
        k = len(items)
        el = time.perf_counter() - t0
        if "failed" in it:
            log(f"{label}: {k}/{n} [skipped] {it['failed']} | elapsed {_fmt(el)}, ETA {_fmt(el / k * (n - k))}")
            continue
        sec = it["seconds"]
        parts = [f"data {sec.get('data', 0):.1f}s", f"train {sec.get('train', 0):.1f}s"]
        parts += [f"{name} {v:.1f}s" for name, v in sec.items() if name.startswith(("refresh", "rolling", "har"))]
        log(f"{label}: {k}/{n} seeds ready ({it.get('unit', it['seed'])}: {', '.join(parts)}) | "
            f"elapsed {_fmt(el)}, ETA {_fmt(el / k * (n - k))}")
    return sorted(items, key=lambda it: it["seed"])


# ---------------------------------------------------------------- cache
CACHE_VERSION = 1
PKG = Path(__file__).resolve().parent / "phase1"
# Only code that determines the numbers enters the key; orchestration and logging
# (prepare.py, tune, evaluate, printing) can change without invalidating the cache.
CACHED_MODULES = ("dgp.py", "ensemble.py", "adaptive.py", "baselines.py", "jax_filters.py", "metrics.py")
LOBSTER_ARGS = ("source", "lobster_root", "year", "tickers", "pre_days", "har_window")
KEY_EXCLUDE = {"n_seeds", "seed_start", "batch", "workers", "out", "no_cache", "report_only", "gpu_chunk",
               "reuse_tuning", "verbose", "baseline_cache", "min_coverage"}


def run_key(args):
    """Hash of everything that determines the per-seed numbers (not batch size or workers)."""
    settings = {k: v for k, v in vars(args).items() if k not in KEY_EXCLUDE}
    if not settings.get("sweep_every"):                          # unused sweep: keys as before
        settings.pop("sweep_every", None)
    if settings.get("source", "synthetic") == "synthetic":      # keep synthetic keys as before
        for k in LOBSTER_ARGS:
            settings.pop(k, None)
    grids = dict(P0=P0_GRID, delta=DELTA_GRID, p_max=P_MAX_GRID, overlap=OVERLAP_GRID, sgd=SGD_GRID,
                 ablations=ABLATIONS, early=EARLY)
    h = hashlib.sha256()
    h.update(json.dumps(dict(version=CACHE_VERSION, settings=settings, grids=grids),
                        sort_keys=True, default=str).encode())
    extra = ("lobster.py",) if getattr(args, "source", "synthetic") == "lobster" else \
        (("dgp_drift.py",) if args.variant in DRIFT_VARIANTS else ())
    for name in CACHED_MODULES + extra:
        h.update((PKG / name).read_bytes())
    if getattr(args, "source", "synthetic") == "lobster":
        h.update(json.dumps(UNITS, sort_keys=True).encode())
        h.update((PKG / "jax_masked.py").read_bytes())
        for fn in (pad_time, stack_padded):
            h.update(inspect.getsource(fn).encode())
    for fn in (seed_metrics, make_jobs, stack):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def load_tuning(cache_dir, key):
    path = cache_dir / f"tuning_{key}.json"
    if not path.exists():
        return None
    t = json.loads(path.read_text())
    return AdaptiveConfig(**t["adaptive"]), t["sgd_eta0"], t["refresh_window"], t["rolling_window"]


def save_tuning(cache_dir, key, best_cfg, best_eta, best_refresh, best_rolling, source=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"tuning_{key}.json").write_text(json.dumps(dict(
        key=key, adaptive=asdict(best_cfg), sgd_eta0=best_eta,
        refresh_window=best_refresh, rolling_window=best_rolling, reused_from=source),
        indent=4, default=str))


def adopt_latest_cache(cache_dir, key):
    """Adopt the newest cache made under another key: its tuning and its finished seeds.

    For when only code that does not change the numbers was edited (the key hashes code).
    The output folder is per variant, so everything adopted belongs to the same variant.
    Files are copied under the current key; the originals are kept.
    """
    files = sorted(cache_dir.glob("tuning_*.json"), key=lambda f: f.stat().st_mtime)
    files = [f for f in files if f.name != f"tuning_{key}.json"]
    if not files:
        return None
    old_key = files[-1].stem[len("tuning_"):]
    t = json.loads(files[-1].read_text())
    tuned = (AdaptiveConfig(**t["adaptive"]), t["sgd_eta0"], t["refresh_window"], t["rolling_window"])
    save_tuning(cache_dir, key, *tuned, source=files[-1].name)
    n_files = 0
    for f in cache_dir.glob(f"*_{old_key}_*.csv"):
        if f.name.endswith(".tmp.csv"):
            continue
        new = f.with_name(f.name.replace(f"_{old_key}_", f"_{key}_"))
        if not new.exists():
            d = pd.read_csv(f)
            if "cache_key" in d.columns:
                d["cache_key"] = key
            d.to_csv(new, index=False)
            n_files += 1
    log(f"[reuse-tuning] adopted tuning and {n_files} cached files from key {old_key} for key {key}")
    return tuned


def load_cached_seeds(cache_dir, key, seeds):
    """Per-seed rows and timing rows from cached batches with this key, restricted to `seeds`."""
    rows, timing = [], []
    wanted = set(seeds)
    for f in sorted(cache_dir.glob(f"seeds_{key}_*.csv")):
        if f.name.endswith(".tmp.csv"):
            continue
        d = pd.read_csv(f)
        rows.append(d[d.seed.isin(wanted)])
        t = f.with_name(f.name.replace("seeds_", "timing_"))
        if t.exists():
            timing.append(pd.read_csv(t))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["seed", "method"])
    df = df.drop_duplicates(subset=["seed", "method"], keep="last")
    tm = pd.concat(timing, ignore_index=True) if timing else pd.DataFrame()
    return df, tm


def save_failed(cache_dir, key):
    if FAILED:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"failed_{key}.json").write_text(json.dumps({str(k): v for k, v in FAILED.items()}, indent=4))


def load_failed(cache_dir, key):
    path = cache_dir / f"failed_{key}.json"
    return {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}


def save_batch(cache_dir, key, rows, timing_row, batch_seeds):
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{key}_{min(batch_seeds):06d}-{max(batch_seeds):06d}"
    d = pd.DataFrame(rows)
    d.insert(0, "cache_key", key)
    tmp = cache_dir / f"seeds_{tag}.tmp.csv"
    d.to_csv(tmp, index=False)
    pd.DataFrame([timing_row]).to_csv(cache_dir / f"timing_{tag}.csv", index=False)
    tmp.replace(cache_dir / f"seeds_{tag}.csv")      # atomic: a crash never leaves half a batch


# ---------------------------------------------------------------- GPU memory
class Chunker:
    """Runs a batched JAX function over slices of the seed axis.

    Starts with `size` seeds per call; on a GPU out-of-memory error it halves the size and
    retries, and keeps the smaller size for later calls. Results come back as NumPy arrays.
    """

    def __init__(self, size):
        self.size = max(1, int(size))

    @staticmethod
    def _is_oom(err):
        msg = str(err)
        return "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()

    def __call__(self, fn, arrays):
        while True:
            try:
                n = arrays[0].shape[0]
                parts = []
                for a in range(0, n, self.size):
                    res = fn(*[x[a:a + self.size] for x in arrays])
                    res = res if isinstance(res, tuple) else (res,)
                    parts.append([np.asarray(r) for r in res])
                return tuple(np.concatenate([q[k] for q in parts]) for k in range(len(parts[0])))
            except Exception as err:          # JaxRuntimeError / XlaRuntimeError depending on version
                if not self._is_oom(err) or self.size == 1:
                    raise
                self.size = max(1, self.size // 2)
                log(f"[gpu] out of memory; retrying with {self.size} seeds per GPU call")


# ---------------------------------------------------------------- real data (LOBSTER)
FAILED = {}                          # seed -> reason, for units that could not be loaded
UNITS = {"eval": [], "tune": []}       # tickers, set in main() for --source lobster


def unit_ticker(args, seed):
    """Eval seeds 0..n-1 -> eval tickers; tuning seeds 100000 + k -> tuning tickers."""
    if seed >= 100_000:
        return UNITS["tune"][seed - 100_000]
    return UNITS["eval"][seed - args.seed_start]


def make_lobster_jobs(args, seeds, refresh_windows, rolling_windows):
    ens = EnsembleConfig(n_trees=args.n_trees, max_depth=args.max_depth, device=args.xgb_device)
    jobs = []
    for sd in seeds:
        cfg = L.LobsterConfig(root=args.lobster_root, ticker=unit_ticker(args, sd), year=args.year,
                              h=args.h, pre_days=args.pre_days)
        jobs.append(dict(lobster=cfg, seed=sd, ens=replace(ens, seed=sd % 100_000), every=args.every,
                         har_window=args.har_window,
                         refresh_windows=refresh_windows, rolling_windows=rolling_windows))
    return jobs


def build_jobs(args, seeds, rw, lw):
    return make_lobster_jobs(args, seeds, rw, lw) if args.source == "lobster" else make_jobs(args, seeds, rw, lw)


def job_unit(job):
    """Identifier used for baseline-cache files: the seed, or the ticker for real data."""
    return job["lobster"].ticker if "lobster" in job else job["dgp"].seed


# ---------------------------------------------------------------- baseline cache
BASELINE_MODULES = ("dgp.py", "ensemble.py", "baselines.py")
BCACHE = None          # set in main()


class BaselineCache:
    """Per-seed predictions of the XGBoost baselines (refresh, rolling), one file per
    (kind, window, seed). Keyed only by what they depend on (data and ensemble settings,
    baseline schedule, and the data/ensemble/baseline code), so changing anything about the
    adaptive filter reuses them. Shared by all variants under the same folder."""

    VERSION = 1

    def __init__(self, root, args):
        if args.source == "lobster":
            settings = dict(version=self.VERSION, source="lobster", root=str(args.lobster_root), year=args.year,
                            h=args.h, pre_days=args.pre_days, n_trees=args.n_trees, max_depth=args.max_depth,
                            xgb_device=args.xgb_device, every=args.every)
            modules = ("ensemble.py", "baselines.py", "lobster.py")
        else:
            settings = dict(version=self.VERSION, variant=args.variant, h=args.h, seg_len=args.seg_len,
                            regimes=args.regimes, bad_tick_rate=args.bad_tick_rate,
                            bad_tick_scale=args.bad_tick_scale, innov_df=args.innov_df, n_trees=args.n_trees,
                            max_depth=args.max_depth, xgb_device=args.xgb_device, every=args.every)
            modules = BASELINE_MODULES + (("dgp_drift.py",) if args.variant in DRIFT_VARIANTS else ())
        hsh = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
        for name in modules:
            hsh.update((PKG / name).read_bytes())
        self.key = hsh.hexdigest()[:16]
        self.dir = Path(root) / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "settings.json").write_text(json.dumps(settings, indent=4))

    def _path(self, kind, window, unit):
        tag = f"seed{unit:06d}" if isinstance(unit, (int, np.integer)) else str(unit)
        return self.dir / f"{kind}_w{window}_{tag}.npz"

    def has(self, kind, window, seed):
        return self._path(kind, window, seed).exists()

    def get(self, kind, window, seed):
        with np.load(self._path(kind, window, seed)) as z:
            return z["mu"], float(z["seconds"])

    def put(self, kind, window, seed, mu, seconds):
        tmp = self._path(kind, window, seed).with_suffix(".tmp.npz")
        np.savez(tmp, mu=np.asarray(mu, np.float32), seconds=seconds)
        tmp.replace(self._path(kind, window, seed))


def prepare_cached(pool, args, seeds, rw, lw, label, sweep=()):
    """prepare_many, but XGBoost baselines already in the baseline cache are loaded, not rerun.

    sweep: (kind, refit interval, window) triples for the retraining-frequency sweep; cached
    under the kind name '<kind>_e<interval>'."""
    jobs = build_jobs(args, seeds, rw, lw)
    n_cached = 0
    for job in jobs:
        sd = job_unit(job)
        job["sweep"] = [(k, e, w) for k, e, w in sweep if not BCACHE.has(f"{k}_e{e}", w, sd)]
        n_cached += len(sweep) - len(job["sweep"])
        job["refresh_windows"] = [w for w in rw if not BCACHE.has("refresh", w, sd)]
        job["rolling_windows"] = [w for w in lw if not BCACHE.has("rolling", w, sd)]
        n_cached += (len(rw) - len(job["refresh_windows"])) + (len(lw) - len(job["rolling_windows"]))
    if rw or lw or sweep:
        total = len(seeds) * (len(rw) + len(lw) + len(sweep))
        log(f"{label}: XGBoost baselines {n_cached}/{total} cached, {total - n_cached} to compute")
    unit_of = {job["seed"] if "lobster" in job else job["dgp"].seed: job_unit(job) for job in jobs}
    items = prepare_many(pool, jobs, label, args.verbose)
    failed = [it for it in items if "failed" in it]
    if failed:
        FAILED.update({int(it["seed"]): it["failed"] for it in failed})
        log(f"{label}: {len(failed)} unit(s) skipped: {', '.join(it['unit'] for it in failed)}")
    items = [it for it in items if "failed" not in it]
    for it in items:
        sd = unit_of[it["seed"]]
        for kind, windows in (("refresh", rw), ("rolling", lw)):
            for w in windows:
                if w in it[kind]:
                    BCACHE.put(kind, w, sd, it[kind][w], it["seconds"][f"{kind}_{w}"])
                else:
                    mu, sec = BCACHE.get(kind, w, sd)
                    it[kind][w] = mu
                    it["seconds"][f"{kind}_{w}"] = sec
        for kind, e, w in sweep:
            if (kind, e) in it.get("sweep", {}):
                BCACHE.put(f"{kind}_e{e}", w, sd, it["sweep"][(kind, e)], it["seconds"][f"{kind}@{e}"])
            else:
                mu, sec = BCACHE.get(f"{kind}_e{e}", w, sd)
                it.setdefault("sweep", {})[(kind, e)] = mu
                it["seconds"][f"{kind}@{e}"] = sec
    return items


# ---------------------------------------------------------------- stages
def tune(args, pool, dtype, D_pad):
    from phase1 import jax_filters as J
    seeds = [args.seed_start + 100_000 + k for k in range(args.tune_seeds)]
    rw = list(args.refresh_grid) if (args.xgb and args.tune_xgb) else []
    lw = list(args.rolling_grid) if (args.xgb and args.tune_xgb) else []
    t0 = time.perf_counter()
    items = prepare_cached(pool, args, seeds, rw, lw, "tuning")
    log(f"tuning: data prepared in {_fmt(time.perf_counter() - t0)}")
    w0, is_bias, idx, y_obs, y_clean, mask, _ = stack_any(items, D_pad, dtype)
    h = args.h

    cfgs = [AdaptiveConfig(P0=p0, P0_bias=10 * p0, delta=d, P_max_factor=pm, overlap_factor=overlap_value(o, h))
            for p0 in P0_GRID for d in DELTA_GRID for pm in P_MAX_GRID for o in OVERLAP_GRID]
    params = J.stack_params([J.adaptive_params(c, h) for c in cfgs])
    log(f"tuning: adaptive filter, {len(cfgs)} configs x {len(seeds)} seeds on GPU "
        "(first call includes compilation)...")
    t0 = time.perf_counter()
    chunk = Chunker(args.gpu_chunk or len(seeds))
    if mask is None:
        sc = chunk(lambda w, b, i, yo, yc: J.adaptive_tune_scores(params, w, b, i, yo, yc, h),
                   (w0, is_bias, idx, y_obs, y_clean))[0].mean(axis=0)
    else:
        from phase1 import jax_masked as JM
        sc = chunk(lambda w, b, i, yo, yc, m: JM.adaptive_tune_scores_masked(params, w, b, i, yo, yc, m, h),
                   (w0, is_bias, idx, y_obs, y_clean, mask))[0].mean(axis=0)
    best_cfg = cfgs[int(np.nanargmax(sc))]
    log(f"tuning: adaptive grid done in {_fmt(time.perf_counter() - t0)}: P0={best_cfg.P0:g}, "
        f"delta={best_cfg.delta}, P_max_factor={best_cfg.P_max_factor:g}, "
        f"overlap_factor={best_cfg.overlap_factor:g}")
    edges = [name for name, v, grid in (("P0", best_cfg.P0, P0_GRID), ("delta", best_cfg.delta, DELTA_GRID),
                                        ("P_max_factor", best_cfg.P_max_factor, P_MAX_GRID))
             if v in (grid[0], grid[-1]) and not (name == "delta" and v == 1.0)]
    if edges:
        log(f"tuning: note, {', '.join(edges)} at the edge of its grid; consider extending it")
    log(f"tuning: SGD, {len(SGD_GRID)} step sizes x {len(seeds)} seeds on GPU...")

    etas = np.asarray(SGD_GRID, dtype)
    if mask is None:
        sc = chunk(lambda w, i, yo, yc: J.sgd_tune_scores(etas, w, i, yo, yc, h),
                   (w0, idx, y_obs, y_clean))[0].mean(axis=0)
    else:
        sc = chunk(lambda w, i, yo, yc, m: JM.sgd_tune_scores_masked(etas, w, i, yo, yc, m, h),
                   (w0, idx, y_obs, y_clean, mask))[0].mean(axis=0)
    best_eta = float(SGD_GRID[int(np.nanargmax(sc))])
    log(f"tuning: SGD eta0={best_eta}")

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
        log("tuning: scoring XGBoost baseline windows...")
        if rw:
            best_refresh = max(rw, key=lambda w: mean_ls("refresh", w))
        if lw:
            best_rolling = max(lw, key=lambda w: mean_ls("rolling", w))
        log(f"tuning: refresh window={best_refresh}, rolling window={best_rolling}")
    return best_cfg, best_eta, best_refresh, best_rolling


def evaluate(args, pool, dtype, D_pad, best_cfg, best_eta, best_refresh, best_rolling,
             seeds, cache_dir, key):
    """Compute the given seeds in batches; each finished batch is written to the cache."""
    from phase1 import jax_filters as J
    h = args.h
    names = ["adaptive"] + (list(ABLATIONS) if args.ablations else [])
    cfgs = [best_cfg] + ([replace(best_cfg, **ch) for ch in ABLATIONS.values()] if args.ablations else [])
    params = J.stack_params([J.adaptive_params(c, h) for c in cfgs])
    nus = [c.nu for c in cfgs]

    rw = [best_refresh] if args.xgb else []
    lw = [best_rolling] if args.xgb else []
    chunk = Chunker(args.gpu_chunk or args.batch)
    etas = np.asarray([best_eta], dtype)
    sweep = [(k, e, best_refresh if k == "refresh" else best_rolling)
             for k in ("refresh", "rolling") for e in (args.sweep_every or []) if e != args.every] if args.xgb else []
    n_batches = -(-len(seeds) // args.batch)
    t_start = time.perf_counter()
    for b0 in range(0, len(seeds), args.batch):
        batch = seeds[b0:b0 + args.batch]
        bno = b0 // args.batch + 1
        label = f"batch {bno}/{n_batches}"
        t0 = time.perf_counter()
        items = prepare_cached(pool, args, batch, rw, lw, label, sweep)
        t_prep = time.perf_counter() - t0
        save_failed(cache_dir, key)
        if not items:
            log(f"{label}: no usable units in this batch")
            continue
        batch = [it["seed"] for it in items]
        w0, is_bias, idx, y_obs, y_clean, _, lengths = stack_any(items, D_pad, dtype)
        T_pad = y_obs.shape[1]

        log(f"{label}: adaptive filter on GPU, {len(batch)} seeds x {len(cfgs)} configs"
            f"{' (first call includes compilation)' if bno == 1 else ''}...")
        t0 = time.perf_counter()
        mu_a, s_a, fire = chunk(lambda w, b, i, y: J.adaptive_batch(params, w, b, i, y, h),
                                (w0, is_bias, idx, y_obs))
        t_adapt = time.perf_counter() - t0
        log(f"{label}: adaptive done in {_fmt(t_adapt)}; SGD and scale wrappers on GPU...")
        t0 = time.perf_counter()
        mu_g = chunk(lambda w, i, y: J.sgd_batch(etas, w, i, y, h), (w0, idx, y_obs))[0][:, 0]
        t_sgd = time.perf_counter() - t0

        point = {"frozen+robust": np.stack([pad_time(it["mu_frozen"], T_pad) for it in items]).astype(dtype),
                 "sgd+robust": mu_g}
        if all("mu_har" in it for it in items):
            point["har+robust"] = np.stack([pad_time(it["mu_har"], T_pad) for it in items]).astype(dtype)
        if args.xgb:
            point["refresh+robust"] = np.stack([pad_time(it["refresh"][best_refresh], T_pad)
                                                for it in items]).astype(dtype)
            point["rolling+robust"] = np.stack([pad_time(it["rolling"][best_rolling], T_pad)
                                                for it in items]).astype(dtype)
            for k, e, _ in sweep:
                point[f"{k}@{e}+robust"] = np.stack([pad_time(it["sweep"][(k, e)], T_pad)
                                                     for it in items]).astype(dtype)
        scales = {k: chunk(lambda m, y: J.robust_scale_batch(m, y, h), (v, y_obs))[0]
                  for k, v in point.items()}
        s_ewma = chunk(lambda m, y: J.ewma_scale_batch(m, y, h), (point["frozen+robust"], y_obs))[0]
        log(f"{label}: scoring {len(items)} seeds on CPU...")
        rows = []
        for k, it in enumerate(items):
            for g, name in enumerate(names):
                rows.append(dict(seed=it["seed"], unit=it.get("unit", ""), method=name, n_alarms=int(fire[k, g, :lengths[k]].sum()),
                                 **seed_metrics(it, mu_a[k, g, :lengths[k]], s_a[k, g, :lengths[k]], nus[g])))
            for name, mu in point.items():
                rows.append(dict(seed=it["seed"], unit=it.get("unit", ""), method=name,
                                 **seed_metrics(it, mu[k, :lengths[k]], scales[name][k, :lengths[k]], 4.0)))
            rows.append(dict(seed=it["seed"], unit=it.get("unit", ""), method="frozen+ewma",
                             **seed_metrics(it, point["frozen+robust"][k, :lengths[k]], s_ewma[k, :lengths[k]],
                                            np.inf)))
        timing_row = dict(first_seed=min(batch), last_seed=max(batch), n=len(batch), prep_s=t_prep,
                          adaptive_s=t_adapt, sgd_s=t_sgd, configs=len(cfgs),
                          xgb_refresh_s=np.mean([it["seconds"].get(f"refresh_{best_refresh}", np.nan) for it in items]),
                          xgb_rolling_s=np.mean([it["seconds"].get(f"rolling_{best_rolling}", np.nan) for it in items]),
                          **{f"xgb_{k}@{e}_s": np.mean([it["seconds"].get(f"{k}@{e}", np.nan) for it in items])
                             for k, e, _ in sweep})
        save_batch(cache_dir, key, rows, timing_row, batch)
        done_el = time.perf_counter() - t_start
        log(f"{label}: seeds {min(batch)}-{max(batch)} saved | prep {_fmt(t_prep)}, adaptive {_fmt(t_adapt)}, "
            f"sgd {_fmt(t_sgd)} | elapsed {_fmt(done_el)}, ETA {_fmt(done_el / bno * (n_batches - bno))}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "lobster"],
                    help="synthetic benchmark, or real LOBSTER one-minute data (realized-volatility task)")
    ap.add_argument("--lobster-root", default="/nfs/data/lobster_data/lobster_minutely/3")
    ap.add_argument("--year", type=int, default=2020, help="lobster: calendar year of each stock-year unit")
    ap.add_argument("--tickers", nargs="+", default=["auto"],
                    help="lobster: 'auto' (default) ranks all tickers present on enough days by liquidity; "
                         "or give candidate tickers in order. The first n-seeds are evaluated, the next "
                         "tune-seeds are used for tuning")
    ap.add_argument("--pre-days", type=int, default=50, help="lobster: pre-training days per unit")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="lobster: a ticker needs a file on this share of the year's trading days")
    ap.add_argument("--har-window", type=int, default=40_000, help="lobster: rolling window of the HAR baseline")
    ap.add_argument("--variant", default="disjoint_drift",
                    choices=["disjoint", "shared", "disjoint_drift", *DRIFT_VARIANTS])
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tune-seeds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--gpu-chunk", type=int, default=0,
                    help="seeds per GPU call (0 = whole batch); halves automatically on out-of-memory")
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
    ap.add_argument("--sweep-every", type=int, nargs="+", default=None,
                    help="also run refresh and rolling retraining with these refit intervals (events), "
                         "e.g. 500 8000 32000; reported as refresh@N / rolling@N")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="also log every step inside the workers (data, training, each baseline)")
    ap.add_argument("--out", default="results_jax")
    ap.add_argument("--baseline-cache", default=None,
                    help="folder for cached XGBoost baseline predictions (default: <out>/../baseline_cache, "
                         "shared by the variants of one run)")
    ap.add_argument("--no-cache", action="store_true", help="delete the cache and recompute everything")
    ap.add_argument("--reuse-tuning", action="store_true",
                    help="if nothing is cached for the current key, adopt the newest cache in <out>/cache "
                         "(tuning and finished seeds); use after edits that do not change the numbers")
    ap.add_argument("--report-only", action="store_true",
                    help="only rebuild summaries from cached seeds; never compute")
    args = ap.parse_args()
    if args.no_cache and args.report_only:
        ap.error("--no-cache and --report-only are mutually exclusive")

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    dtype = np.float64 if args.x64 else np.float32
    D_pad = args.n_trees * 2 ** args.max_depth + 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    what = f"lobster {args.year}, {args.n_seeds} stock-years" if args.source == "lobster" else \
        f"variant={args.variant}, seeds {args.seed_start}-{args.seed_start + args.n_seeds - 1}"
    log(f"{what}, "
        f"tuning seeds={args.tune_seeds}, batch={args.batch}, workers={args.workers}, "
        f"xgb={'on' if args.xgb else 'off'}{' (tuned)' if args.tune_xgb else ''}, "
        f"ablations={'on' if args.ablations else 'off'}, dtype={'float64' if args.x64 else 'float32'}")
    log(f"JAX devices: {jax.devices()}")

    if args.source == "lobster":
        if args.seed_start != 0:
            ap.error("--seed-start must be 0 with --source lobster")
        dates = L.list_dates(args.lobster_root, args.year)
        need = args.n_seeds + args.tune_seeds
        if args.tickers == ["auto"]:
            log(f"lobster: ranking tickers by liquidity over {len(dates)} trading days of {args.year}...")
            avail, liq = L.rank_tickers(args.lobster_root, args.year, dates, args.min_coverage)
            log(f"lobster: {len(avail)} tickers on >= {args.min_coverage:.0%} of days; most liquid: "
                + ", ".join(f"{t} ({liq[t]:,.0f} trades/day)" for t in avail[:5]))
        else:
            avail = L.available_tickers(args.lobster_root, args.year, args.tickers, args.min_coverage, dates)
            log(f"lobster: {len(dates)} trading days in {args.year}; {len(avail)}/{len(args.tickers)} candidate "
                f"tickers available")
        if len(avail) < need:
            ap.error(f"only {len(avail)} tickers available, need n-seeds + tune-seeds = {need}")
        UNITS["eval"], UNITS["tune"] = avail[:args.n_seeds], avail[args.n_seeds:need]
        log(f"lobster: evaluating {', '.join(UNITS['eval'])}")
        log(f"lobster: tuning on {', '.join(UNITS['tune'])}")
    global BCACHE
    BCACHE = BaselineCache(args.baseline_cache or (out.resolve().parent / "baseline_cache"), args)
    log(f"baseline cache: {BCACHE.dir}")
    cache_dir = out / "cache"
    if args.no_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        log("cache deleted")
    key = run_key(args)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    tuned = load_tuning(cache_dir, key)
    if tuned is None and args.reuse_tuning and cache_dir.exists():
        tuned = adopt_latest_cache(cache_dir, key)
    done, _ = load_cached_seeds(cache_dir, key, seeds)
    FAILED.update(load_failed(cache_dir, key))
    todo = [sd for sd in seeds if sd not in set(done["seed"]) and sd not in FAILED]
    log(f"cache key {key}: tuning {'cached' if tuned else 'needed'}, "
          f"{done['seed'].nunique()}/{len(seeds)} seeds cached, "
          f"{sum(sd in FAILED for sd in seeds)} skipped, {len(todo)} to compute")

    if FAILED:
        log(f"skipping {len(FAILED)} unit(s) that could not be loaded: "
            + "; ".join(f"{unit_ticker(args, k) if args.source == 'lobster' else k}: {v}"
                        for k, v in sorted(FAILED.items())))
    if args.report_only:
        if tuned is None or len(todo) == len(seeds):
            raise SystemExit("nothing cached for these settings; run without --report-only first")
        if todo:
            log(f"[report-only] {len(todo)} seeds missing; reporting on the {len(seeds) - len(todo)} cached")
    elif tuned is None or todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            if tuned is None:
                tuned = tune(args, pool, dtype, D_pad)
                save_tuning(cache_dir, key, *tuned)
            else:
                log("using cached tuning")
            if todo:
                evaluate(args, pool, dtype, D_pad, *tuned, seeds=todo, cache_dir=cache_dir, key=key)
    best_cfg, best_eta, best_refresh, best_rolling = tuned

    df, timing = load_cached_seeds(cache_dir, key, seeds)
    df = df.drop(columns="cache_key").sort_values(["seed", "method"]).reset_index(drop=True)
    df.to_csv(out / "per_seed.csv", index=False)
    timing.to_csv(out / "timing.csv", index=False)
    gains = summarize_gains(df)
    gains.to_csv(out / "gains.csv", index=False)
    means = df.groupby("method").mean(numeric_only=True).drop(columns="seed")
    means.to_csv(out / "means.csv")
    (out / "config.json").write_text(json.dumps(dict(
        args=vars(args), cache_key=key, seeds_reported=int(df.seed.nunique()),
        adaptive=asdict(best_cfg), sgd_eta0=best_eta,
        refresh_window=best_refresh, rolling_window=best_rolling), indent=4, default=str))

    log("writing summaries")
    print("\nMean over seeds:", flush=True)
    shown = means[["log_score", "crps", "coverage80", "rmse", "mae", "signal_rmse", "signal_mae",
                   "early_mae_first", "early_mae_revisit"]].dropna(axis=1, how="all")
    print(shown.to_string(float_format="%.4f"))
    print(f"\nGain of adaptive over each method ({df.seed.nunique()} seeds):")
    print_gains(gains)


if __name__ == "__main__":
    main()
