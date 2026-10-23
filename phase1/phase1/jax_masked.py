"""Masked tuning scores, for batches whose streams have different lengths (real data).

Streams are padded at the end to a common length. The filters are causal (a prediction
at event t only uses labels of events <= t - h), so padding after a stream's end never
changes its real predictions; it only has to be left out of the score, which `mask` does.
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import jax_filters as J


def _masked_mean_t_logpdf(y, mu, s, nu, mask):
    lp = jax.scipy.stats.t.logpdf(y, nu, mu, jnp.sqrt(s))
    return jnp.sum(jnp.where(mask, lp, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


@partial(jax.jit, static_argnames=("h",))
def adaptive_tune_scores_masked(params, w0, is_bias, idx, y_obs, y_clean, mask, h):
    """Mean log score over the unmasked events, shape (seeds, G)."""
    def one(p, w, b, i, yo, yc, m):
        mu, s, _ = J.adaptive_filter(p, w, b, i, yo, h)
        return _masked_mean_t_logpdf(yc, mu, s, p["nu"], m)
    over_grid = jax.vmap(one, in_axes=(0, None, None, None, None, None, None))
    return jax.vmap(over_grid, in_axes=(None, 0, 0, 0, 0, 0, 0))(params, w0, is_bias, idx, y_obs, y_clean, mask)


@partial(jax.jit, static_argnames=("h",))
def sgd_tune_scores_masked(etas, w0, idx, y_obs, y_clean, mask, h, nu=4.0):
    def one(eta, w, i, yo, yc, m):
        mu = J.online_sgd(eta, w, i, yo, h)
        s = J.robust_t_scale(mu, yo, h)
        return _masked_mean_t_logpdf(yc, mu, s, nu, m)
    over_grid = jax.vmap(one, in_axes=(0, None, None, None, None, None))
    return jax.vmap(over_grid, in_axes=(None, 0, 0, 0, 0, 0))(etas, w0, idx, y_obs, y_clean, mask)
