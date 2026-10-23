"""Lean CPU implementation of the online filters (Numba), for deployment and timing.

Exactly the same algorithm as the NumPy reference (phase1/adaptive.py) and the JAX version
(phase1/jax_filters.py): lazy discounting with a discount clock, Student-t robust weight,
overlap inflation, discounted EM noise scale, quantile-tracking calibration factor, and the
bounded drift detector with optional alarm action. Only the K + 1 active leaves are touched
per event, so the cost per event is O(K) with no framework overhead.

Two ways to use it:

    run(...)                        a whole stream in one compiled loop (throughput)
    OnlineFilter.step(...)          one event at a time from Python (real-time use):
                                    tick the clock, update with the label that just
                                    became observable (if any), predict the new event

State lives in NumPy arrays (leaf means m, variances P, last-touch clock values) plus a
small vector of scalars, so it can be saved, copied or shared without pickling a model.
"""

import numpy as np
from numba import njit

from .adaptive import AdaptiveConfig
from scipy import stats

# parameter vector layout
(P_DELTA, P_DELTA_ALARM, P_NU, P_NU_ALARM, P_OFAC, P_DV, P_S2MIN, P_ALPHA, P_Q, P_ETA, P_RAW, P_ZETA,
 P_LAM, P_UBR, P_WARM, P_AWIN, P_ROB, P_DISC, P_SCALE, P_CAL, P_DET, P_ACT, P_FCAST) = range(23)
N_PARAMS = 23
# scalar state layout
(S_CLOCK, S_SIGMA2, S_NEFF, S_DEFF, S_C, S_S, S_UBAR, S_NUPD, S_ALARM_UNTIL) = range(9)
N_STATE = 9


def params_from_config(cfg: AdaptiveConfig, h: int, forecast_scale=False):
    p = np.zeros(N_PARAMS)
    p[P_FCAST] = float(forecast_scale)
    p[P_DELTA], p[P_DELTA_ALARM] = cfg.delta, cfg.delta_alarm
    p[P_NU], p[P_NU_ALARM] = cfg.nu, cfg.nu_alarm
    p[P_OFAC] = h if cfg.overlap_factor is None else cfg.overlap_factor
    p[P_DV], p[P_S2MIN] = cfg.delta_v, cfg.sigma2_min
    p[P_ALPHA], p[P_Q], p[P_ETA] = cfg.alpha, stats.t.ppf(1.0 - cfg.alpha / 2.0, cfg.nu), cfg.eta_c
    p[P_RAW], p[P_ZETA], p[P_LAM] = float(cfg.detector == "raw"), cfg.zeta, cfg.lam
    p[P_UBR], p[P_WARM], p[P_AWIN] = cfg.u_bar_rate, cfg.warmup, cfg.alarm_window
    p[P_ROB], p[P_DISC], p[P_SCALE] = cfg.use_robust, cfg.use_discount, cfg.use_scale
    p[P_CAL], p[P_DET], p[P_ACT] = cfg.use_calibration, cfg.use_detector, cfg.use_alarm_action
    return p


# ---------------------------------------------------------------- compiled kernels
@njit(cache=True, inline="always")
def _in_alarm(sc, p, now):
    return p[P_ACT] > 0 and now <= sc[S_ALARM_UNTIL]


@njit(cache=True)
def _tick(sc, p, now):
    if p[P_DISC] > 0:
        d = p[P_DELTA_ALARM] if _in_alarm(sc, p, now) else p[P_DELTA]
        sc[S_CLOCK] += -np.log(d)


@njit(cache=True)
def _touch_sum(P, cat, Pmax, ids, clock):
    """Apply pending discount to the active leaves; return the sum of their variances."""
    tot = 0.0
    for k in range(ids.shape[0]):
        j = ids[k]
        g = clock - cat[j]
        v = P[j]
        if g > 0.0:                      # no exp() when nothing accumulated (e.g. delta = 1)
            v = v * np.exp(g if g < 700.0 else 700.0)
            if v > Pmax[j]:
                v = Pmax[j]
            P[j] = v
            cat[j] = clock
        tot += v
    return tot


@njit(cache=True)
def _update(m, P, cat, Pmax, sc, p, ids, y, now, mu_f, s_f):
    """Filter update with an observed label. Returns True if an alarm fired.

    mu_f, s_f: the forecast made for this label when it was predicted. With p[P_FCAST] > 0
    the noise scale and the calibration factor learn from the forecast-time residual
    y - mu_f (scaled by s_f). Otherwise they use the residual against the current leaf
    values, which, with overlapping labels, already contain near-copies of this label and
    so understate the forecast error."""
    S0 = _touch_sum(P, cat, Pmax, ids, sc[S_CLOCK])
    mu = 0.0
    for k in range(ids.shape[0]):
        mu += m[ids[k]]
    s = sc[S_C] * (S0 + sc[S_SIGMA2])
    r = y - mu
    z2 = r * r / s
    in_alarm = _in_alarm(sc, p, now)
    nu = p[P_NU_ALARM] if in_alarm else p[P_NU]
    omega = (nu + 1.0) / (nu + z2) if p[P_ROB] > 0 else 1.0
    inv = 1.0 / (S0 + p[P_OFAC] * sc[S_SIGMA2] / omega)       # 1 / s_tilde
    for k in range(ids.shape[0]):
        j = ids[k]
        g = P[j] * inv
        m[j] += g * r
        P[j] = P[j] - P[j] * g
    if p[P_FCAST] > 0:
        r_sc, s_sc = y - mu_f, s_f
        om_sc = (nu + 1.0) / (nu + r_sc * r_sc / s_sc) if p[P_ROB] > 0 else 1.0
    else:
        r_sc, s_sc, om_sc = r, s, omega
    if p[P_SCALE] > 0:
        sc[S_NEFF] = p[P_DV] * sc[S_NEFF] + 1.0
        sc[S_DEFF] = p[P_DV] * sc[S_DEFF] + om_sc * r_sc * r_sc
        v = sc[S_DEFF] / sc[S_NEFF]
        sc[S_SIGMA2] = v if v > p[P_S2MIN] else p[P_S2MIN]
    if p[P_CAL] > 0:
        miss = 1.0 if abs(r_sc) > p[P_Q] * np.sqrt(s_sc) else 0.0
        sc[S_C] *= np.exp(p[P_ETA] * (miss - p[P_ALPHA]))
    sc[S_NUPD] += 1.0
    fired = False
    if p[P_DET] > 0:
        u = z2 if p[P_RAW] > 0 else 1.0 - (omega if omega < 1.0 else 1.0)
        if sc[S_NUPD] == 1.0:
            sc[S_UBAR] = u
        if sc[S_NUPD] <= p[P_WARM]:
            sc[S_UBAR] += (u - sc[S_UBAR]) / sc[S_NUPD]
        else:
            if not in_alarm:
                sc[S_UBAR] += p[P_UBR] * (u - sc[S_UBAR])
            v = sc[S_S] + u - sc[S_UBAR] - p[P_ZETA]
            sc[S_S] = v if v > 0.0 else 0.0
            if sc[S_S] > p[P_LAM]:
                sc[S_ALARM_UNTIL] = now + p[P_AWIN]
                sc[S_S] = 0.0
                fired = True
    return fired


@njit(cache=True)
def _predict(m, P, cat, Pmax, sc, ids):
    S0 = _touch_sum(P, cat, Pmax, ids, sc[S_CLOCK])
    mu = 0.0
    for k in range(ids.shape[0]):
        mu += m[ids[k]]
    return mu, sc[S_C] * (S0 + sc[S_SIGMA2])


@njit(cache=True)
def _step(m, P, cat, Pmax, sc, p, i, ids_new, has_label, ids_lab, y_lab, buf):
    """One event: tick, update with the label that became observable, predict event i.

    buf (2, h): the last h forecasts (mean, scale); slot i % h holds the forecast of event
    i - h until it is overwritten by the forecast of event i."""
    _tick(sc, p, i)
    fired = False
    k = i % buf.shape[1]
    if has_label:
        fired = _update(m, P, cat, Pmax, sc, p, ids_lab, y_lab, i, buf[0, k], buf[1, k])
    mu, s = _predict(m, P, cat, Pmax, sc, ids_new)
    buf[0, k] = mu
    buf[1, k] = s
    return mu, s, fired


@njit(cache=True)
def _run(m, P, cat, Pmax, sc, p, idx, y, h, buf):
    T = idx.shape[0]
    mu = np.empty(T)
    s = np.empty(T)
    fire = np.zeros(T, np.bool_)
    for i in range(T):
        j = i - h
        if j >= 0:
            mu[i], s[i], fire[i] = _step(m, P, cat, Pmax, sc, p, i, idx[i], True, idx[j], y[j], buf)
        else:
            mu[i], s[i], fire[i] = _step(m, P, cat, Pmax, sc, p, i, idx[i], False, idx[i], 0.0, buf)
    return mu, s, fire


@njit(cache=True)
def _run_sgd(m, idx, y, h, step):
    T, K1 = idx.shape
    mu = np.empty(T)
    for i in range(T):
        j = i - h
        if j >= 0:
            r = y[j]
            for k in range(K1):
                r -= m[idx[j, k]]
            for k in range(K1):
                m[idx[j, k]] += step * r
        v = 0.0
        for k in range(K1):
            v += m[idx[i, k]]
        mu[i] = v
    return mu


# ---------------------------------------------------------------- Python API
class OnlineFilter:
    """Adaptive filter state with a compiled per-event step."""

    def __init__(self, w0, bias_index, cfg: AdaptiveConfig, h: int, forecast_scale=False):
        """forecast_scale=True: the noise scale and calibration learn from forecast-time
        residuals (recommended with long, overlapping labels; see _update)."""
        self.h = h
        self.buf = np.zeros((2, max(h, 1)))
        self.m = np.asarray(w0, np.float64).copy()
        P0 = np.full(len(w0), cfg.P0)
        P0[bias_index] = cfg.P0_bias
        self.Pmax = cfg.P_max_factor * P0
        self.P = np.minimum(P0, self.Pmax)   # P <= Pmax always holds, so touches without discount skip the cap
        self.cat = np.zeros(len(w0))
        self.sc = np.zeros(N_STATE)
        self.sc[S_SIGMA2] = cfg.sigma2_init
        self.sc[S_NEFF] = 1.0
        self.sc[S_DEFF] = cfg.sigma2_init
        self.sc[S_C] = 1.0
        self.sc[S_ALARM_UNTIL] = -1.0
        self.p = params_from_config(cfg, h, forecast_scale)
        self._dummy = np.zeros(1, np.int64)

    def run(self, idx, y):
        """Prequential pass over a whole stream; returns (mu, s, fired) arrays."""
        idx = np.ascontiguousarray(idx)                 # any integer dtype; int16 saves memory
        return _run(self.m, self.P, self.cat, self.Pmax, self.sc, self.p, idx, np.asarray(y, np.float64), self.h,
                    self.buf)

    def step(self, i, ids_new, ids_label=None, y_label=0.0):
        """Event i: update with (ids_label, y_label) if given, then predict ids_new."""
        if ids_label is None:
            return _step(self.m, self.P, self.cat, self.Pmax, self.sc, self.p, i, ids_new, False, ids_new, 0.0,
                         self.buf)
        return _step(self.m, self.P, self.cat, self.Pmax, self.sc, self.p, i, ids_new, True, ids_label,
                     float(y_label), self.buf)


def run_sgd(w0, idx, y, h, eta0):
    idx = np.ascontiguousarray(idx)
    return _run_sgd(np.asarray(w0, np.float64).copy(), idx, np.asarray(y, np.float64), h, eta0 / idx.shape[1])


# ---------------------------------------------------------------- chunked streams
@njit(cache=True)
def _run_chunk(m, P, cat, Pmax, sc, p, idx_ext, y_ext, h, n_tail, offset, buf):
    """Steps for local rows n_tail .. end of an extended chunk whose first n_tail rows are
    the previous chunk's last rows (needed for delayed labels). offset = global index of
    local row n_tail."""
    n = idx_ext.shape[0] - n_tail
    mu = np.empty(n)
    s = np.empty(n)
    fire = np.zeros(n, np.bool_)
    for k in range(n):
        il = n_tail + k
        jl = il - h
        if jl >= 0:
            mu[k], s[k], fire[k] = _step(m, P, cat, Pmax, sc, p, offset + k, idx_ext[il], True, idx_ext[jl],
                                         y_ext[jl], buf)
        else:
            mu[k], s[k], fire[k] = _step(m, P, cat, Pmax, sc, p, offset + k, idx_ext[il], False, idx_ext[il], 0.0,
                                         buf)
    return mu, s, fire


@njit(cache=True)
def _sgd_chunk(m, idx_ext, y_ext, h, n_tail, step):
    n = idx_ext.shape[0] - n_tail
    K1 = idx_ext.shape[1]
    mu = np.empty(n)
    for k in range(n):
        il = n_tail + k
        jl = il - h
        if jl >= 0:
            r = y_ext[jl]
            for q in range(K1):
                r -= m[idx_ext[jl, q]]
            for q in range(K1):
                m[idx_ext[jl, q]] += step * r
        v = 0.0
        for q in range(K1):
            v += m[idx_ext[il, q]]
        mu[k] = v
    return mu


def run_stream(make_idx, y, h, chunk, filt=None, sgd=None):
    """Run the adaptive filter and/or SGD over a long stream, chunk by chunk.

    make_idx(a, b) returns the leaf indices of rows a..b-1 (computed on demand, so the
    full index matrix is never held in memory). Exactly equal to an unchunked run.
    filt: OnlineFilter or None. sgd: (w0, eta0) or None. Returns dict of output arrays.
    """
    T = len(y)
    y = np.asarray(y, np.float64)
    out = {}
    if filt is not None:
        out["mu"], out["s"], out["fire"] = np.empty(T), np.empty(T), np.zeros(T, bool)
    if sgd is not None:
        m_sgd = np.asarray(sgd[0], np.float64).copy()
        out["mu_sgd"] = np.empty(T)
    tail_idx = None
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        idx = np.ascontiguousarray(make_idx(a, b))
        lo = max(0, a - h)
        if tail_idx is None:
            idx_ext, n_tail = idx, 0
        else:
            idx_ext, n_tail = np.concatenate([tail_idx, idx]), a - lo
        y_ext = y[a - n_tail:b]
        if filt is not None:
            mu, s, fire = _run_chunk(filt.m, filt.P, filt.cat, filt.Pmax, filt.sc, filt.p, idx_ext, y_ext,
                                     filt.h, n_tail, a, filt.buf)
            out["mu"][a:b], out["s"][a:b], out["fire"][a:b] = mu, s, fire
        if sgd is not None:
            out["mu_sgd"][a:b] = _sgd_chunk(m_sgd, idx_ext, y_ext, h, n_tail, sgd[1] / idx.shape[1])
        tail_idx = idx_ext[-h:] if h > 0 else idx_ext[:0]
    return out


@njit(cache=True)
def robust_t_scale(mu, y, h, nu, delta_v, sigma2_init, sigma2_min, alpha, q, eta_c):
    """Compiled version of baselines.robust_t_scale (same recursion)."""
    T = mu.shape[0]
    s = np.empty(T)
    sigma2, n_eff, d_eff, c = sigma2_init, 1.0, sigma2_init, 1.0
    for i in range(T):
        j = i - h
        if j >= 0:
            sj = c * sigma2
            r = y[j] - mu[j]
            omega = (nu + 1.0) / (nu + r * r / sj)
            n_eff = delta_v * n_eff + 1.0
            d_eff = delta_v * d_eff + omega * r * r
            v = d_eff / n_eff
            sigma2 = v if v > sigma2_min else sigma2_min
            miss = 1.0 if abs(r) > q * np.sqrt(sj) else 0.0
            c *= np.exp(eta_c * (miss - alpha))
        s[i] = c * sigma2
    return s


def robust_scale(mu, y, h, nu=4.0, delta_v=0.995, sigma2_init=1.0, sigma2_min=1e-4, alpha=0.2, eta_c=0.005):
    q = float(stats.t.ppf(1.0 - alpha / 2.0, nu))
    return robust_t_scale(np.asarray(mu, np.float64), np.asarray(y, np.float64), h, nu, delta_v,
                          sigma2_init, sigma2_min, alpha, q, eta_c)
