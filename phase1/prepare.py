"""Per-seed work that stays outside JAX: data, XGBoost pre-training, leaf indices,
and the XGBoost-based baselines (refresh, rolling retrain).

Kept free of JAX imports so it can run in worker processes.
"""

import time

import numpy as np

from . import baselines as B
from .dgp import DGPConfig, generate
from .ensemble import EnsembleConfig, LeafMap, train_ensemble


def prepare_seed(job):
    """job: dict(dgp=DGPConfig, ens=EnsembleConfig, refresh_windows=[...], rolling_windows=[...],
    every=int). Returns a dict of NumPy arrays (no XGBoost objects)."""
    dgp_cfg: DGPConfig = job["dgp"]
    ens_cfg: EnsembleConfig = job["ens"]
    data = generate(dgp_cfg)
    booster = train_ensemble(data["X_pre"], data["y_pre"], ens_cfg)
    lm = LeafMap(booster)
    lm.calibrate_bias(data["X_pre"])
    idx = lm.leaf_index(data["X"])
    h = data["h"]

    out = dict(
        seed=dgp_cfg.seed, h=h, w0=lm.w0, bias_index=lm.bias_index, idx=idx.astype(np.int32),
        y_obs=data["y_obs"], y_clean=data["y_clean"], signal=data["signal"],
        seg_starts=data["seg_starts"], regime=data["regime"],
        mu_frozen=lm.predict(idx), refresh={}, rolling={}, seconds={},
    )
    every = job.get("every", 2_000)
    for w in job.get("refresh_windows", []):
        t0 = time.perf_counter()
        out["refresh"][w] = B.xgb_refresh(booster, ens_cfg, data["X"], data["y_obs"], h, every=every, window=w)
        out["seconds"][f"refresh_{w}"] = time.perf_counter() - t0
    for w in job.get("rolling_windows", []):
        t0 = time.perf_counter()
        out["rolling"][w] = B.xgb_rolling(ens_cfg, data["X"], data["y_obs"], h, every=every, window=w,
                                          X_pre=data["X_pre"], y_pre=data["y_pre"])
        out["seconds"][f"rolling_{w}"] = time.perf_counter() - t0
    return out
