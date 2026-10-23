"""Baselines for Gate G1, plus wrappers that give point forecasters a predictive scale.

Every baseline respects delayed labels: a label for event j is used only from
event j + h onward.
"""

import time
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from scipy import stats

from .ensemble import EnsembleConfig


# ---------------------------------------------------------------- point forecasters
def frozen(leafmap, idx_stream):
    return leafmap.predict(idx_stream)


def online_sgd(leafmap, idx_stream, y_obs, h, eta0=0.01):
    """SGD on leaf values with a scalar step, split over the K + 1 active leaves."""
    m = leafmap.w0.copy()
    T, K1 = idx_stream.shape
    step = eta0 / K1
    mu = np.empty(T)
    for i in range(T):
        j = i - h
        if j >= 0:
            idx = idx_stream[j]
            r = y_obs[j] - m[idx].sum()
            m[idx] += step * r
        mu[i] = m[idx_stream[i]].sum()
    return mu


def _windowed_blocks(T, h, every, window):
    """Yield (start, stop, lo, hi): predict events [start, stop) with labels [lo, hi)."""
    for start in range(0, T, every):
        stop = min(start + every, T)
        hi = start - h + 1          # labels j <= start - h are observable at start
        lo = max(0, hi - window)
        yield start, stop, lo, hi


def xgb_refresh(booster, ens_cfg: EnsembleConfig, X, y_obs, h, every=2_000, window=5_000):
    """Batch refresh of leaf values on a recent window, structure fixed."""
    T = len(X)
    mu = booster.predict(xgb.DMatrix(X))
    params = dict(ens_cfg.params(), process_type="update", updater="refresh", refresh_leaf=True)
    n_rounds = booster.num_boosted_rounds()
    for start, stop, lo, hi in _windowed_blocks(T, h, every, window):
        if hi - lo < 500:
            continue
        dtrain = xgb.DMatrix(X[lo:hi], label=y_obs[lo:hi])
        refreshed = xgb.train(params, dtrain, num_boost_round=n_rounds, xgb_model=booster.copy())
        mu[start:stop] = refreshed.predict(xgb.DMatrix(X[start:stop]))
    return mu


def xgb_rolling(ens_cfg: EnsembleConfig, X, y_obs, h, every=2_000, window=10_000,
                X_pre=None, y_pre=None):
    """Retrain the whole ensemble on a rolling window of observable labels."""
    T = len(X)
    if X_pre is not None:
        Xa = np.vstack([X_pre, X])
        ya = np.concatenate([y_pre, y_obs])
        off = len(X_pre)
    else:
        Xa, ya, off = X, y_obs, 0
    mu = np.empty(T)
    for start, stop, lo, hi in _windowed_blocks(T, h, every, window):
        lo_a, hi_a = max(0, off + hi - window), off + hi
        dtrain = xgb.DMatrix(Xa[lo_a:hi_a], label=ya[lo_a:hi_a])
        model = xgb.train(ens_cfg.params(), dtrain, num_boost_round=ens_cfg.n_trees)
        mu[start:stop] = model.predict(xgb.DMatrix(X[start:stop]))
    return mu


# ---------------------------------------------------------------- predictive scales
def ewma_gaussian_scale(mu, y_obs, h, var_init=1.0, lam=0.995):
    """Gaussian predictive with an EWMA of past squared residuals (practitioner default).

    Returns variance s (used as Gaussian variance) and df = inf.
    """
    T = len(mu)
    s = np.empty(T)
    v = var_init
    for i in range(T):
        j = i - h
        if j >= 0:
            r = y_obs[j] - mu[j]
            v = lam * v + (1.0 - lam) * r * r
        s[i] = v
    return s, np.full(T, np.inf)


@dataclass
class ScaleConfig:
    nu: float = 4.0
    delta_v: float = 0.995
    sigma2_init: float = 1.0
    sigma2_min: float = 1e-4
    alpha: float = 0.2
    eta_c: float = 0.005


def robust_t_scale(mu, y_obs, h, cfg: ScaleConfig = ScaleConfig()):
    """Same robust scale tracker and calibration factor as the method, without leaf updates.

    Used to give baselines a fair predictive distribution, so that comparisons
    isolate the effect of adapting the leaves.
    """
    T = len(mu)
    s = np.empty(T)
    q = stats.t.ppf(1.0 - cfg.alpha / 2.0, cfg.nu)
    sigma2, n_eff, d_eff, c = cfg.sigma2_init, 1.0, cfg.sigma2_init, 1.0
    for i in range(T):
        j = i - h
        if j >= 0:
            sj = c * sigma2
            r = y_obs[j] - mu[j]
            omega = (cfg.nu + 1.0) / (cfg.nu + r * r / sj)
            n_eff = cfg.delta_v * n_eff + 1.0
            d_eff = cfg.delta_v * d_eff + omega * r * r
            sigma2 = max(d_eff / n_eff, cfg.sigma2_min)
            c *= np.exp(cfg.eta_c * (float(abs(r) > q * np.sqrt(sj)) - cfg.alpha))
        s[i] = c * sigma2
    return s, np.full(T, cfg.nu)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
