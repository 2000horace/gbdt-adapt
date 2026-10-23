"""JAX versions of the online filters, for running many seeds and grids on a GPU.

Every filter is a jax.lax.scan over stream events. Hyperparameters, including the
ablation switches, are numbers in a dict, so a whole grid can be vmapped. Seeds of
different ensembles are padded to a common number of leaves.

The step order matches the NumPy reference (phase1/adaptive.py) exactly:
    tick the discount clock -> update with the label of event i - h -> predict event i.

Shapes (single seed):
    w0       (D,)        initial leaf values, padded with zeros
    is_bias  (D,)        True at the bias leaf
    idx      (T, K1)     active leaf indices per event (int32)
    y        (T,)        observed labels
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from .adaptive import AdaptiveConfig


# ---------------------------------------------------------------- parameters
def adaptive_params(cfg: AdaptiveConfig, h: int):
    """Turn an AdaptiveConfig into a dict of floats (vmappable)."""
    f = float
    return dict(
        P0=f(cfg.P0), P0_bias=f(cfg.P0_bias), P_max_factor=f(cfg.P_max_factor),
        delta=f(cfg.delta), nu=f(cfg.nu),
        ofac=f(h if cfg.overlap_factor is None else cfg.overlap_factor),
        sigma2_init=f(cfg.sigma2_init), delta_v=f(cfg.delta_v), sigma2_min=f(cfg.sigma2_min),
        alpha=f(cfg.alpha), q=f(stats.t.ppf(1.0 - cfg.alpha / 2.0, cfg.nu)), eta_c=f(cfg.eta_c),
        detector_raw=f(cfg.detector == "raw"), zeta=f(cfg.zeta), lam=f(cfg.lam),
        u_bar_rate=f(cfg.u_bar_rate), warmup=f(cfg.warmup), alarm_window=f(cfg.alarm_window),
        delta_alarm=f(cfg.delta_alarm), nu_alarm=f(cfg.nu_alarm),
        use_robust=f(cfg.use_robust), use_discount=f(cfg.use_discount), use_scale=f(cfg.use_scale),
        use_calibration=f(cfg.use_calibration), use_detector=f(cfg.use_detector),
        use_alarm_action=f(cfg.use_alarm_action),
    )


def stack_params(param_list):
    """List of param dicts -> dict of arrays with a leading grid axis."""
    return {k: jnp.asarray([p[k] for p in param_list]) for k in param_list[0]}


# ---------------------------------------------------------------- adaptive filter
def adaptive_filter(p, w0, is_bias, idx, y, h):
    """Robust, lazily discounted leaf filter. Returns (mu, s, alarm) arrays of length T."""
    dt = w0.dtype
    D = w0.shape[0]
    P0vec = jnp.where(is_bias, p["P0_bias"], p["P0"]).astype(dt)
    Pmax = p["P_max_factor"] * P0vec
    T = idx.shape[0]

    carry0 = dict(
        m=w0, P=P0vec, cat=jnp.zeros(D, dt), clock=jnp.asarray(0.0, dt),
        sigma2=jnp.asarray(p["sigma2_init"], dt), n_eff=jnp.asarray(1.0, dt),
        d_eff=jnp.asarray(p["sigma2_init"], dt), c=jnp.asarray(1.0, dt),
        S=jnp.asarray(0.0, dt), u_bar=jnp.asarray(0.0, dt), n_upd=jnp.asarray(0.0, dt),
        alarm_until=jnp.asarray(-1.0, dt),
    )

    def touch(st, ids):
        growth = jnp.exp(jnp.minimum(st["clock"] - st["cat"][ids], 700.0))
        return jnp.minimum(st["P"][ids] * growth, Pmax[ids])

    def step(st, i_int):
        # Rows are read inside the scan (no shifted copy of idx, which is the largest array).
        ii = idx[i_int].astype(jnp.int32)
        j = jnp.maximum(i_int - h, 0)
        ij = idx[j].astype(jnp.int32)
        yj = y[j]
        vj = i_int >= h
        i = i_int.astype(dt)
        vf = vj.astype(dt)

        # 1. tick the discount clock
        in_alarm = (p["use_alarm_action"] > 0) & (i <= st["alarm_until"])
        delta_now = jnp.where(in_alarm, p["delta_alarm"], p["delta"])
        clock = st["clock"] + p["use_discount"] * (-jnp.log(delta_now))
        st = dict(st, clock=clock)

        # 2. update with the label of event j = i - h (masked when j < 0)
        P_act = touch(st, ij)
        S0 = P_act.sum()
        mu_j = st["m"][ij].sum()
        s_j = st["c"] * (S0 + st["sigma2"])
        r = yj - mu_j
        z2 = r * r / s_j
        nu_upd = jnp.where(in_alarm, p["nu_alarm"], p["nu"])
        omega = jnp.where(p["use_robust"] > 0, (nu_upd + 1.0) / (nu_upd + z2), 1.0)
        s_tilde = S0 + p["ofac"] * st["sigma2"] / omega
        gain = P_act / s_tilde
        m = st["m"].at[ij].add(vf * gain * r)
        P = st["P"].at[ij].set(jnp.where(vj, P_act - P_act * P_act / s_tilde, st["P"][ij]))
        cat = st["cat"].at[ij].set(jnp.where(vj, clock, st["cat"][ij]))

        do_scale = vj & (p["use_scale"] > 0)
        n_eff = jnp.where(do_scale, p["delta_v"] * st["n_eff"] + 1.0, st["n_eff"])
        d_eff = jnp.where(do_scale, p["delta_v"] * st["d_eff"] + omega * r * r, st["d_eff"])
        sigma2 = jnp.where(do_scale, jnp.maximum(d_eff / n_eff, p["sigma2_min"]), st["sigma2"])

        miss = (jnp.abs(r) > p["q"] * jnp.sqrt(s_j)).astype(dt)
        do_cal = vj & (p["use_calibration"] > 0)
        c = jnp.where(do_cal, st["c"] * jnp.exp(p["eta_c"] * (miss - p["alpha"])), st["c"])

        # detector
        n_upd = st["n_upd"] + vf
        do_det = vj & (p["use_detector"] > 0)
        u = jnp.where(p["detector_raw"] > 0, z2, 1.0 - jnp.minimum(omega, 1.0))
        warm = n_upd <= p["warmup"]
        u_bar_warm = st["u_bar"] + (u - st["u_bar"]) / jnp.maximum(n_upd, 1.0)
        u_bar_live = jnp.where(in_alarm, st["u_bar"], st["u_bar"] + p["u_bar_rate"] * (u - st["u_bar"]))
        u_bar_new = jnp.where(warm, u_bar_warm, u_bar_live)
        S_live = jnp.maximum(0.0, st["S"] + u - u_bar_new - p["zeta"])
        fire = do_det & (~warm) & (S_live > p["lam"])
        S_new = jnp.where(warm, st["S"], jnp.where(fire, 0.0, S_live))
        u_bar = jnp.where(do_det, u_bar_new, st["u_bar"])
        S = jnp.where(do_det, S_new, st["S"])
        alarm_until = jnp.where(fire, i + p["alarm_window"], st["alarm_until"])

        st = dict(m=m, P=P, cat=cat, clock=clock, sigma2=sigma2, n_eff=n_eff, d_eff=d_eff,
                  c=c, S=S, u_bar=u_bar, n_upd=n_upd, alarm_until=alarm_until)

        # 3. predict event i
        P_i = touch(st, ii)
        st = dict(st, P=st["P"].at[ii].set(P_i), cat=st["cat"].at[ii].set(clock))
        mu_i = st["m"][ii].sum()
        s_i = st["c"] * (P_i.sum() + st["sigma2"])
        return st, (mu_i, s_i, fire)

    _, (mu, s, fire) = jax.lax.scan(step, carry0, jnp.arange(T))
    return mu, s, fire


# ---------------------------------------------------------------- baselines
def online_sgd(eta0, w0, idx, y, h):
    step_size = eta0 / idx.shape[1]

    def step(m, i):
        ii = idx[i].astype(jnp.int32)
        j = jnp.maximum(i - h, 0)
        ij = idx[j].astype(jnp.int32)
        r = y[j] - m[ij].sum()
        m = m.at[ij].add((i >= h) * step_size * r)
        return m, m[ii].sum()

    _, mu = jax.lax.scan(step, w0, jnp.arange(idx.shape[0]))
    return mu


def robust_t_scale(mu, y, h, nu=4.0, delta_v=0.995, sigma2_init=1.0, sigma2_min=1e-4,
                   alpha=0.2, eta_c=0.005):
    """Same as baselines.robust_t_scale: predictive t-scale^2 for a point forecaster."""
    dt = mu.dtype
    q = float(stats.t.ppf(1.0 - alpha / 2.0, nu))
    T = mu.shape[0]
    mu_j = jnp.concatenate([jnp.zeros(h, dt), mu[:T - h]])
    y_j = jnp.concatenate([jnp.zeros(h, dt), y[:T - h]])
    valid = jnp.arange(T) >= h

    def step(st, xs):
        sigma2, n_eff, d_eff, c = st
        mj, yj, vj = xs
        sj = c * sigma2
        r = yj - mj
        omega = (nu + 1.0) / (nu + r * r / sj)
        n_new = delta_v * n_eff + 1.0
        d_new = delta_v * d_eff + omega * r * r
        s2_new = jnp.maximum(d_new / n_new, sigma2_min)
        c_new = c * jnp.exp(eta_c * ((jnp.abs(r) > q * jnp.sqrt(sj)).astype(dt) - alpha))
        st_new = tuple(jnp.where(vj, a, b) for a, b in
                       zip((s2_new, n_new, d_new, c_new), (sigma2, n_eff, d_eff, c)))
        return st_new, st_new[3] * st_new[0]

    init = (jnp.asarray(sigma2_init, dt), jnp.asarray(1.0, dt), jnp.asarray(sigma2_init, dt), jnp.asarray(1.0, dt))
    _, s = jax.lax.scan(step, init, (mu_j, y_j, valid))
    return s


def ewma_gaussian_scale(mu, y, h, var_init=1.0, lam=0.995):
    dt = mu.dtype
    T = mu.shape[0]
    mu_j = jnp.concatenate([jnp.zeros(h, dt), mu[:T - h]])
    y_j = jnp.concatenate([jnp.zeros(h, dt), y[:T - h]])
    valid = jnp.arange(T) >= h

    def step(v, xs):
        mj, yj, vj = xs
        r = yj - mj
        v = jnp.where(vj, lam * v + (1.0 - lam) * r * r, v)
        return v, v

    _, s = jax.lax.scan(step, jnp.asarray(var_init, dt), (mu_j, y_j, valid))
    return s


def t_log_score_mean(y, mu, s, nu):
    """Mean Student-t log predictive density (for tuning inside JAX)."""
    return jax.scipy.stats.t.logpdf(y, nu, mu, jnp.sqrt(s)).mean()


# ---------------------------------------------------------------- batched entry points
@partial(jax.jit, static_argnames=("h",))
def adaptive_batch(params, w0, is_bias, idx, y, h):
    """vmap over seeds (leading axis of w0, is_bias, idx, y) and a grid of params.

    params: dict of arrays with leading grid axis G.
    returns mu, s, fire with shape (seeds, G, T).
    """
    over_grid = jax.vmap(adaptive_filter, in_axes=(0, None, None, None, None, None))
    over_seeds = jax.vmap(over_grid, in_axes=(None, 0, 0, 0, 0, None))
    return over_seeds(params, w0, is_bias, idx, y, h)


@partial(jax.jit, static_argnames=("h",))
def adaptive_tune_scores(params, w0, is_bias, idx, y_obs, y_clean, h):
    """Mean log score on clean labels, shape (seeds, G). Avoids moving (seeds, G, T) to host."""
    def one(p, w, b, i, yo, yc):
        mu, s, _ = adaptive_filter(p, w, b, i, yo, h)
        return t_log_score_mean(yc, mu, s, p["nu"])
    over_grid = jax.vmap(one, in_axes=(0, None, None, None, None, None))
    return jax.vmap(over_grid, in_axes=(None, 0, 0, 0, 0, 0))(params, w0, is_bias, idx, y_obs, y_clean)


@partial(jax.jit, static_argnames=("h",))
def sgd_batch(etas, w0, idx, y, h):
    """Returns mu with shape (seeds, G, T)."""
    over_grid = jax.vmap(online_sgd, in_axes=(0, None, None, None, None))
    return jax.vmap(over_grid, in_axes=(None, 0, 0, 0, None))(etas, w0, idx, y, h)


@partial(jax.jit, static_argnames=("h",))
def robust_scale_batch(mu, y, h):
    """mu: (seeds, T) -> s: (seeds, T)."""
    return jax.vmap(robust_t_scale, in_axes=(0, 0, None))(mu, y, h)


@partial(jax.jit, static_argnames=("h",))
def ewma_scale_batch(mu, y, h):
    return jax.vmap(ewma_gaussian_scale, in_axes=(0, 0, None))(mu, y, h)


@partial(jax.jit, static_argnames=("h",))
def sgd_tune_scores(etas, w0, idx, y_obs, y_clean, h, nu=4.0):
    def one(eta, w, i, yo, yc):
        mu = online_sgd(eta, w, i, yo, h)
        s = robust_t_scale(mu, yo, h)
        return t_log_score_mean(yc, mu, s, nu)
    over_grid = jax.vmap(one, in_axes=(0, None, None, None, None))
    return jax.vmap(over_grid, in_axes=(None, 0, 0, 0, 0))(etas, w0, idx, y_obs, y_clean)


def pad_batch(items, dtype=np.float32):
    """Stack per-seed arrays, padding the leaf dimension to the largest ensemble."""
    D = max(len(it["w0"]) for it in items)
    w0 = np.zeros((len(items), D), dtype)
    is_bias = np.zeros((len(items), D), bool)
    for k, it in enumerate(items):
        w0[k, :len(it["w0"])] = it["w0"]
        is_bias[k, it["bias_index"]] = True
    idx = np.stack([it["idx"] for it in items])
    idx = idx.astype(np.int16 if D < 2 ** 15 else np.int32)   # halves the largest array
    return w0, is_bias, idx
