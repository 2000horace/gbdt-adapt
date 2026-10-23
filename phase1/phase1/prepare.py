"""Per-seed work that stays outside JAX: data, XGBoost pre-training, leaf indices,
and the XGBoost-based baselines (refresh, rolling retrain).

Kept free of JAX imports so it can run in worker processes.
"""

import os
import time

import numpy as np

from . import baselines as B
from .dgp import DGPConfig
from .dgp_drift import generate          # dispatches to dgp.generate for the original variants
from .ensemble import EnsembleConfig, LeafMap, train_ensemble


def _unit(job):
    """Seed number for synthetic jobs, 'TICKER-YEAR' for real-data jobs."""
    if "lobster" in job:
        return f"{job['lobster'].ticker}-{job['lobster'].year}"
    return job["dgp"].seed


def _say(job, msg):
    """Per-step progress from a worker (only with job["verbose"])."""
    if job.get("verbose"):
        print(f"[{time.strftime('%H:%M:%S')}]   worker {os.getpid()} {_unit(job)}: {msg}", flush=True)


def prepare_seed(job):
    """job: dict(dgp=DGPConfig or lobster=LobsterConfig, seed=int, ens=EnsembleConfig,
    refresh_windows=[...], rolling_windows=[...], every=int, verbose=bool).
    Returns a dict of NumPy arrays (no XGBoost objects)."""
    ens_cfg: EnsembleConfig = job["ens"]
    seconds = {}

    t0 = time.perf_counter()
    if "lobster" in job:
        from .lobster import load_unit
        seed = job["seed"]
        try:
            data = load_unit(job["lobster"])
        except ValueError as err:       # unusable stock-year: report it, don't crash the pool
            _say(job, f"skipped: {err}")
            return dict(seed=seed, unit=str(_unit(job)), failed=str(err), seconds={})
    else:
        data = generate(job["dgp"])
        seed = job["dgp"].seed
    seconds["data"] = time.perf_counter() - t0
    _say(job, f"data generated ({seconds['data']:.1f}s)")

    t0 = time.perf_counter()
    booster = train_ensemble(data["X_pre"], data["y_pre"], ens_cfg)
    lm = LeafMap(booster)
    lm.calibrate_bias(data["X_pre"])
    idx = lm.leaf_index(data["X"])
    seconds["train"] = time.perf_counter() - t0
    _say(job, f"ensemble trained, {lm.size} leaves ({seconds['train']:.1f}s)")
    h = data["h"]

    out = dict(
        seed=seed, unit=str(_unit(job)), h=h, w0=lm.w0, bias_index=lm.bias_index, idx=idx.astype(np.int32),
        y_obs=data["y_obs"], y_clean=data["y_clean"], signal=data["signal"],
        seg_starts=data["seg_starts"], regime=data["regime"],
        mu_frozen=lm.predict(idx), refresh={}, rolling={}, seconds=seconds,
    )
    every = job.get("every", 2_000)
    if "har_cols" in data:
        from .lobster import har_rolling
        window = max(job.get("refresh_windows") or job.get("rolling_windows") or [job.get("har_window", 40_000)])
        t0 = time.perf_counter()
        out["mu_har"] = har_rolling(data["X_pre"], data["y_pre"], data["X"], data["y_obs"], data["har_cols"],
                                    h, every=every, window=job.get("har_window", window))
        seconds["har"] = time.perf_counter() - t0
        out["dates"] = data["dates"]
    for w in job.get("refresh_windows", []):
        t0 = time.perf_counter()
        out["refresh"][w] = B.xgb_refresh(booster, ens_cfg, data["X"], data["y_obs"], h, every=every, window=w)
        seconds[f"refresh_{w}"] = time.perf_counter() - t0
        _say(job, f"refresh baseline, window {w} ({seconds[f'refresh_{w}']:.1f}s)")
    for w in job.get("rolling_windows", []):
        t0 = time.perf_counter()
        out["rolling"][w] = B.xgb_rolling(ens_cfg, data["X"], data["y_obs"], h, every=every, window=w,
                                          X_pre=data["X_pre"], y_pre=data["y_pre"])
        seconds[f"rolling_{w}"] = time.perf_counter() - t0
        _say(job, f"rolling baseline, window {w} ({seconds[f'rolling_{w}']:.1f}s)")
    # retraining-frequency sweep: (kind, refit interval, window)
    out["sweep"] = {}
    for kind, e, w in job.get("sweep", []):
        t0 = time.perf_counter()
        if kind == "refresh":
            out["sweep"][(kind, e)] = B.xgb_refresh(booster, ens_cfg, data["X"], data["y_obs"], h, every=e, window=w)
        else:
            out["sweep"][(kind, e)] = B.xgb_rolling(ens_cfg, data["X"], data["y_obs"], h, every=e, window=w,
                                                    X_pre=data["X_pre"], y_pre=data["y_pre"])
        seconds[f"{kind}@{e}"] = time.perf_counter() - t0
        _say(job, f"{kind} every {e} ({seconds[f'{kind}@{e}']:.1f}s)")
    return out
