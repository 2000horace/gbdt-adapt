"""Synthetic regime-switching data for the Phase 1 benchmark (N1).

Design
------
* Features: d AR(1) processes (persistent, like HF features). Feature x0 carries
  a regime-dependent offset. In the "disjoint" variant each regime lives in its
  own band of x0, so trees that split on x0 give each regime its own leaves.
  In the "shared" variant all regimes use the same x0 distribution (negative
  control: leaf memory cannot help there).
* Signal: g_r(x) = scale * (f0(x) + delta_r(x)), with f0 the "base" function
  used for pre-training and delta_r a regime-specific change. All pieces are
  step-like or tanh-like, so a tree ensemble can represent them.
* Noise: per-event innovations with GARCH(1,1) dynamics and an intraday
  U-shaped multiplier. The label for event t sums the next h innovations, so
  consecutive labels overlap and residuals are autocorrelated.
* Contamination: with probability p a label gets a heavy-tailed "bad tick".
  Models are trained on the contaminated labels; metrics use the clean ones.
"""

from dataclasses import dataclass, field

import numpy as np


# Offset of x0 per regime. "base" is used only for pre-training, where x0 cycles
# through all bands so the pre-trained trees cover every region.
DISJOINT_OFFSETS = {"A": -2.0, "B": 2.0, "C": 0.0}
DISJOINT_X0_SD = 0.5
SHARED_X0_SD = 1.5


@dataclass
class DGPConfig:
    d: int = 6
    n_pretrain: int = 20_000
    seg_len: int = 20_000
    regimes: tuple = ("A", "B", "A", "C")
    variant: str = "disjoint"           # "disjoint" or "shared"
    h: int = 10                         # label horizon (events)
    session_len: int = 2_000            # events per "trading session"
    u_shape_amp: float = 1.5            # strength of intraday U-shape
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    signal_scale: float = 0.5           # controls signal-to-noise ratio
    feature_rho: float = 0.95           # AR(1) persistence of features
    bad_tick_rate: float = 0.001
    bad_tick_scale: float = 10.0
    bad_tick_df: float = 2.0
    seed: int = 0
    extra: dict = field(default_factory=dict)


def f0(x):
    """Base signal, learned by the pre-trained ensemble."""
    x0, x1, x2, x3, x4, x5 = (x[:, i] for i in range(6))
    band = (x0 < -1.0).astype(float) - (x0 > 1.0).astype(float)
    return (
        0.6 * band * np.sign(x2)
        + 0.6 * (x1 > 0) * np.sign(x2)
        + 0.4 * np.tanh(2.0 * x3)
        - 0.5 * (x4 > 0.5)
        + 0.3 * x5 * (x1 < 0)
    )


def delta(regime, x):
    """Regime-specific change to the base signal."""
    x1, x2, x3, x4, x5 = (x[:, i] for i in range(1, 6))
    if regime == "base":
        return np.zeros(len(x))
    if regime == "A":
        return 0.8 * np.sign(x2) * (x3 > 0) - 0.3
    if regime == "B":
        return -0.8 * np.tanh(2.0 * x3) + 0.5 * (x4 > 0.5)
    if regime == "C":
        return 0.7 * np.sign(x5) * (x1 < 0) + 0.3
    raise ValueError(f"unknown regime {regime}")


def _ar1(n, d, rho, rng):
    """Unit-variance AR(1) processes, shape (n, d)."""
    out = np.empty((n, d))
    out[0] = rng.standard_normal(d)
    scale = np.sqrt(1.0 - rho ** 2)
    eps = rng.standard_normal((n, d)) * scale
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def _intraday_multiplier(n, session_len, amp):
    """U-shaped variance multiplier with mean 1 over a session."""
    tau = (np.arange(n) % session_len) / session_len
    raw = 1.0 + amp * (2.0 * tau - 1.0) ** 2
    return raw / (1.0 + amp / 3.0)


def _garch_innovations(n, alpha, beta, mult, rng):
    """Per-event innovations e_s = sqrt(mult_s * v_s) * eps_s, unconditional var ~ 1."""
    omega = 1.0 - alpha - beta
    v = np.empty(n)
    e = np.empty(n)
    v[0] = 1.0
    eps = rng.standard_normal(n)
    for s in range(n):
        if s > 0:
            v[s] = omega + alpha * (e[s - 1] ** 2 / mult[s - 1]) + beta * v[s - 1]
        e[s] = np.sqrt(mult[s] * v[s]) * eps[s]
    return e, mult * v


def generate(cfg: DGPConfig):
    """Return a dict with pre-training and stream data.

    Keys
    ----
    X_pre, y_pre        : pre-training features and (contaminated) labels
    X, y_obs, y_clean   : stream features, observed labels, clean labels
    signal              : true conditional mean g_r(x_t) on the stream
    noise_var           : true variance of the label noise on the stream
    regime              : regime label per stream event
    seg_starts          : start index of each stream segment
    is_bad              : bool mask of contaminated stream labels
    """
    if cfg.d < 6:
        raise ValueError("d must be at least 6")
    rng = np.random.default_rng(cfg.seed)
    n_stream = cfg.seg_len * len(cfg.regimes)
    n = cfg.n_pretrain + n_stream

    # Regime label per event; pre-training uses the base signal.
    regime = np.array(["base"] * cfg.n_pretrain + [r for r in cfg.regimes for _ in range(cfg.seg_len)])

    # Features.
    X = _ar1(n, cfg.d, cfg.feature_rho, rng)
    if cfg.variant == "disjoint":
        offsets = np.empty(n)
        # Pre-training: cycle through all bands in blocks so every region is covered.
        bands = list(DISJOINT_OFFSETS.values())
        block = cfg.session_len
        for b in range(0, cfg.n_pretrain, block):
            offsets[b:b + block] = bands[(b // block) % len(bands)]
        offsets[cfg.n_pretrain:] = [DISJOINT_OFFSETS[r] for r in regime[cfg.n_pretrain:]]
        X[:, 0] = offsets + DISJOINT_X0_SD * X[:, 0]
    elif cfg.variant == "shared":
        X[:, 0] = SHARED_X0_SD * X[:, 0]
    else:
        raise ValueError(f"unknown variant {cfg.variant}")

    # Signal.
    signal = f0(X)
    for r in set(cfg.regimes):
        mask = regime == r
        signal[mask] += delta(r, X[mask])
    signal *= cfg.signal_scale

    # Noise: label t sums innovations t+1, ..., t+h (overlapping labels).
    mult = _intraday_multiplier(n + cfg.h, cfg.session_len, cfg.u_shape_amp)
    e, var_e = _garch_innovations(n + cfg.h, cfg.garch_alpha, cfg.garch_beta, mult, rng)
    csum = np.concatenate([[0.0], np.cumsum(e)])
    csum_v = np.concatenate([[0.0], np.cumsum(var_e)])
    idx = np.arange(n)
    noise = (csum[idx + cfg.h + 1] - csum[idx + 1]) / np.sqrt(cfg.h)
    noise_var = (csum_v[idx + cfg.h + 1] - csum_v[idx + 1]) / cfg.h

    y_clean = signal + noise

    # Contamination of observed labels.
    is_bad = rng.random(n) < cfg.bad_tick_rate
    bad = cfg.bad_tick_scale * rng.standard_t(cfg.bad_tick_df, size=n)
    y_obs = y_clean + is_bad * bad

    p = cfg.n_pretrain
    seg_starts = np.arange(len(cfg.regimes)) * cfg.seg_len
    return {
        "X_pre": X[:p],
        "y_pre": y_obs[:p],
        "X": X[p:],
        "y_obs": y_obs[p:],
        "y_clean": y_clean[p:],
        "signal": signal[p:],
        "noise_var": noise_var[p:],
        "regime": regime[p:],
        "seg_starts": seg_starts,
        "is_bad": is_bad[p:],
        "h": cfg.h,
    }
