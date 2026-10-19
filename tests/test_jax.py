import jax

jax.config.update("jax_enable_x64", True)

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from phase1 import baselines as B
from phase1 import jax_filters as J
from phase1.adaptive import AdaptiveConfig, AdaptiveLeafFilter
from phase1.dgp import DGPConfig, generate
from phase1.ensemble import EnsembleConfig, LeafMap, train_ensemble

H = 10


@pytest.fixture(scope="module")
def setup():
    data = generate(DGPConfig(n_pretrain=5_000, seg_len=1_500, seed=3, h=H, bad_tick_rate=0.02, innov_df=3.0))
    booster = train_ensemble(data["X_pre"], data["y_pre"], EnsembleConfig(n_trees=15))
    lm = LeafMap(booster)
    lm.calibrate_bias(data["X_pre"])
    idx = lm.leaf_index(data["X"])
    return data, lm, idx


CONFIGS = {
    "default": {},
    "alarms": dict(warmup=200, lam=3.0, zeta=0.0, alarm_window=300),
    "raw_detector": dict(warmup=200, detector="raw", lam=20.0, zeta=0.5),
    "no_robust": dict(use_robust=False),
    "no_discount": dict(use_discount=False),
    "no_scale": dict(use_scale=False),
    "no_calibration": dict(use_calibration=False),
    "no_alarm_action": dict(warmup=200, lam=3.0, zeta=0.0, use_alarm_action=False),
    "no_overlap": dict(overlap_factor=1.0),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_adaptive_matches_numpy(setup, name):
    data, lm, idx = setup
    cfg = replace(AdaptiveConfig(P0=3e-4, P0_bias=3e-3, delta=0.999), **CONFIGS[name])
    ref = AdaptiveLeafFilter(lm.w0, lm.bias_index, cfg, H)
    mu_np, s_np, _ = ref.run(idx, data["y_obs"])

    is_bias = np.zeros(lm.size, bool)
    is_bias[lm.bias_index] = True
    p = {k: jnp.asarray(v) for k, v in J.adaptive_params(cfg, H).items()}
    mu, s, fire = J.adaptive_filter(p, jnp.asarray(lm.w0), jnp.asarray(is_bias), jnp.asarray(idx),
                                    jnp.asarray(data["y_obs"]), H)
    np.testing.assert_allclose(np.asarray(mu), mu_np, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(s), s_np, rtol=1e-9, atol=1e-10)
    assert list(np.nonzero(np.asarray(fire))[0]) == ref.alarms
    if name == "alarms":
        assert len(ref.alarms) > 0, "test config should trigger alarms"


def test_sgd_and_scales_match_numpy(setup):
    data, lm, idx = setup
    y = data["y_obs"]
    mu_np = B.online_sgd(lm, idx, y, H, eta0=0.01)
    mu = J.online_sgd(0.01, jnp.asarray(lm.w0), jnp.asarray(idx), jnp.asarray(y), H)
    np.testing.assert_allclose(np.asarray(mu), mu_np, rtol=1e-9, atol=1e-10)

    s_np, _ = B.robust_t_scale(mu_np, y, H)
    s = J.robust_t_scale(jnp.asarray(mu_np), jnp.asarray(y), H)
    np.testing.assert_allclose(np.asarray(s), s_np, rtol=1e-9)

    e_np, _ = B.ewma_gaussian_scale(mu_np, y, H)
    e = J.ewma_gaussian_scale(jnp.asarray(mu_np), jnp.asarray(y), H)
    np.testing.assert_allclose(np.asarray(e), e_np, rtol=1e-9)


def test_batched_grid_matches_single(setup):
    data, lm, idx = setup
    cfgs = [AdaptiveConfig(P0=p0, P0_bias=10 * p0, delta=d) for p0 in (1e-4, 1e-3) for d in (0.999, 1.0)]
    params = J.stack_params([J.adaptive_params(c, H) for c in cfgs])
    item = dict(w0=lm.w0, bias_index=lm.bias_index, idx=idx)
    w0, is_bias, idxb = J.pad_batch([item, item], dtype=np.float64)
    yb = np.stack([data["y_obs"]] * 2)
    mu, s, _ = J.adaptive_batch(params, w0, is_bias, idxb, yb, H)
    assert mu.shape == (2, len(cfgs), len(idx))
    for g, cfg in enumerate(cfgs):
        mu_np, _, _ = AdaptiveLeafFilter(lm.w0, lm.bias_index, cfg, H).run(idx, data["y_obs"])
        np.testing.assert_allclose(np.asarray(mu[1, g]), mu_np, rtol=1e-9, atol=1e-10)
