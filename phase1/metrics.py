"""Prequential scores for Student-t / Gaussian predictive distributions.

A predictive distribution is given by (mu, s, df): location mu, squared scale s,
degrees of freedom df (df = inf means Gaussian with variance s).
"""

import numpy as np
from scipy import special, stats


def log_score(y, mu, s, df):
    """Log predictive density, per event (higher is better)."""
    y, mu, s, df = np.broadcast_arrays(y, mu, s, df)
    out = np.empty(len(y))
    g = np.isinf(df)
    out[g] = stats.norm.logpdf(y[g], mu[g], np.sqrt(s[g]))
    t = ~g
    out[t] = stats.t.logpdf(y[t], df[t], mu[t], np.sqrt(s[t]))
    return out


def _crps_std_t(z, nu):
    """CRPS of a standard Student-t (df nu > 1) at z (Jordan et al., scoringRules)."""
    F = stats.t.cdf(z, nu)
    f = stats.t.pdf(z, nu)
    c = 2.0 * np.sqrt(nu) * special.beta(0.5, nu - 0.5) / ((nu - 1.0) * special.beta(0.5, nu / 2.0) ** 2)
    return z * (2.0 * F - 1.0) + 2.0 * f * (nu + z * z) / (nu - 1.0) - c


def _crps_std_normal(z):
    return z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)


def crps(y, mu, s, df):
    """CRPS per event (lower is better)."""
    y, mu, s, df = np.broadcast_arrays(y, mu, s, df)
    sd = np.sqrt(s)
    z = (y - mu) / sd
    out = np.empty(len(y))
    g = np.isinf(df)
    out[g] = sd[g] * _crps_std_normal(z[g])
    t = ~g
    out[t] = sd[t] * _crps_std_t(z[t], df[t])
    return out


def pit(y, mu, s, df):
    y, mu, s, df = np.broadcast_arrays(y, mu, s, df)
    out = np.empty(len(y))
    g = np.isinf(df)
    out[g] = stats.norm.cdf(y[g], mu[g], np.sqrt(s[g]))
    t = ~g
    out[t] = stats.t.cdf(y[t], df[t], mu[t], np.sqrt(s[t]))
    return out


def coverage(y, mu, s, df, alpha=0.2):
    """Fraction of events inside the central (1 - alpha) interval."""
    u = pit(y, mu, s, df)
    return np.mean((u > alpha / 2.0) & (u < 1.0 - alpha / 2.0))


def block_bootstrap_ci(x, block=500, n_boot=2_000, level=0.95, seed=0):
    """Circular moving-block bootstrap CI for the mean of a dependent series."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    xx = np.concatenate([x, x[:block]])
    csum = np.concatenate([[0.0], np.cumsum(xx)])
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    sums = (csum[starts + block] - csum[starts]).sum(axis=1)
    means = sums / (n_blocks * block)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return x.mean(), lo, hi
