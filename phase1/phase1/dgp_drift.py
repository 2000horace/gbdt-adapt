"""Drift variants where regimes never return (kept separate from dgp.py so that the
caches of the original variants stay valid).

    nonrecurring  every segment gets a fresh random change to the signal (abrupt switches,
                  no regime ever comes back)
    gradual       the change drifts continuously: within segment k its coefficients move
                  linearly from those of state k to those of state k + 1

Features, noise, contamination and pre-training are as in dgp.py. Features use the
"shared" distribution, so the drift happens in place: the same leaves see the change.
The change is a random combination of tree-friendly pieces:

    delta_t(x) = sum_i a_i(t) * piece_i(x),   a(t) drawn per state, a_i ~ N(0, COEF_SD^2)
"""

import numpy as np

from . import dgp
from .dgp import DGPConfig

DRIFT_VARIANTS = ("nonrecurring", "gradual")
COEF_SD = 0.45


def _pieces(x):
    x1, x2, x3, x4, x5 = (x[:, i] for i in range(1, 6))
    return np.stack([
        np.sign(x2) * (x3 > 0),
        np.tanh(2.0 * x3),
        (x4 > 0.5).astype(float),
        np.sign(x5) * (x1 < 0),
        np.sign(x2) * (x1 > 0),
        np.ones(len(x)),
    ], axis=1)


def coefficients(cfg: DGPConfig):
    """Random coefficient states, one per segment plus one (the end point for "gradual")."""
    rng = np.random.default_rng([cfg.seed, 7_777])
    n_seg = len(cfg.regimes)
    a = rng.normal(0.0, COEF_SD, size=(n_seg + 1, 6))
    a[:, -1] *= 0.5            # smaller shifts of the level
    return a


def generate(cfg: DGPConfig):
    """Same output dict as dgp.generate. Dispatches to dgp.generate for the original variants."""
    if cfg.variant not in DRIFT_VARIANTS:
        return dgp.generate(cfg)
    if cfg.d < 6:
        raise ValueError("d must be at least 6")
    rng = np.random.default_rng(cfg.seed)
    n_seg = len(cfg.regimes)
    n_stream = cfg.seg_len * n_seg
    p = cfg.n_pretrain
    n = p + n_stream

    X = dgp._ar1(n, cfg.d, cfg.feature_rho, rng)
    X[:, 0] = dgp.SHARED_X0_SD * X[:, 0]

    signal = dgp.f0(X)
    a = coefficients(cfg)
    pieces = _pieces(X[p:])
    tau = (np.arange(n_stream) % cfg.seg_len) / cfg.seg_len
    seg = np.arange(n_stream) // cfg.seg_len
    if cfg.variant == "nonrecurring":
        coef = a[seg]
    else:
        coef = a[seg] + (a[seg + 1] - a[seg]) * tau[:, None]
    signal[p:] += (coef * pieces).sum(axis=1)
    signal *= cfg.signal_scale

    mult = dgp._intraday_multiplier(n + cfg.h, cfg.session_len, cfg.u_shape_amp)
    e, var_e = dgp._garch_innovations(n + cfg.h, cfg.garch_alpha, cfg.garch_beta, mult, rng, cfg.innov_df)
    csum = np.concatenate([[0.0], np.cumsum(e)])
    csum_v = np.concatenate([[0.0], np.cumsum(var_e)])
    idx = np.arange(n)
    noise = (csum[idx + cfg.h + 1] - csum[idx + 1]) / np.sqrt(cfg.h)
    noise_var = (csum_v[idx + cfg.h + 1] - csum_v[idx + 1]) / cfg.h
    y_clean = signal + noise

    is_bad = rng.random(n) < cfg.bad_tick_rate
    bad = cfg.bad_tick_scale * rng.standard_t(cfg.bad_tick_df, size=n)
    y_obs = y_clean + is_bad * bad

    regime = np.array([f"R{k}" for k in range(n_seg) for _ in range(cfg.seg_len)])
    return {
        "X_pre": X[:p], "y_pre": y_obs[:p], "X": X[p:], "y_obs": y_obs[p:], "y_clean": y_clean[p:],
        "signal": signal[p:], "noise_var": noise_var[p:], "regime": regime,
        "seg_starts": np.arange(n_seg) * cfg.seg_len, "is_bad": is_bad[p:], "h": cfg.h,
    }
