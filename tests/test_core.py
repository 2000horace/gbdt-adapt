import numpy as np
import pytest
import xgboost as xgb
from scipy import integrate, stats

from phase1 import metrics as M
from phase1.adaptive import AdaptiveConfig, AdaptiveLeafFilter
from phase1.dgp import DGPConfig, generate
from phase1.ensemble import EnsembleConfig, LeafMap, train_ensemble


@pytest.fixture(scope="module")
def small():
    data = generate(DGPConfig(n_pretrain=5_000, seg_len=2_000, seed=1))
    booster = train_ensemble(data["X_pre"], data["y_pre"], EnsembleConfig(n_trees=20))
    lm = LeafMap(booster)
    lm.calibrate_bias(data["X_pre"])
    return data, booster, lm


def test_leaf_map_reproduces_booster(small):
    data, booster, lm = small
    idx = lm.leaf_index(data["X"])
    assert idx.shape[1] == booster.num_boosted_rounds() + 1
    np.testing.assert_allclose(lm.predict(idx), booster.predict(xgb.DMatrix(data["X"])), atol=1e-5)


@pytest.mark.parametrize("nu", [3.0, 4.0, 10.0])
def test_crps_t_matches_integral(nu):
    y, mu, sd = 0.7, -0.2, 1.3
    F = lambda x: stats.t.cdf(x, nu, mu, sd)
    lhs = integrate.quad(lambda x: F(x) ** 2, -np.inf, y)[0]
    rhs = integrate.quad(lambda x: (1 - F(x)) ** 2, y, np.inf)[0]
    assert M.crps(np.array([y]), mu, sd ** 2, nu)[0] == pytest.approx(lhs + rhs, rel=1e-6)


def test_crps_gaussian_matches_integral():
    y, mu, sd = 0.7, -0.2, 1.3
    F = lambda x: stats.norm.cdf(x, mu, sd)
    ref = integrate.quad(lambda x: F(x) ** 2, -np.inf, y)[0] + integrate.quad(lambda x: (1 - F(x)) ** 2, y, np.inf)[0]
    assert M.crps(np.array([y]), mu, sd ** 2, np.inf)[0] == pytest.approx(ref, rel=1e-6)


def test_lazy_discount_equals_stepwise():
    cfg = AdaptiveConfig(P0=1e-3, P_max_factor=1e6, delta=0.99)
    f = AdaptiveLeafFilter(np.zeros(3), 2, cfg, h=1)
    for t in range(50):
        f._tick(t)
    P = f._touch(np.array([0]))[0]
    assert P == pytest.approx(1e-3 * 0.99 ** -50, rel=1e-10)


def test_lazy_discount_respects_cap():
    cfg = AdaptiveConfig(P0=1e-3, P_max_factor=4.0, delta=0.9)
    f = AdaptiveLeafFilter(np.zeros(3), 2, cfg, h=1)
    for t in range(500):
        f._tick(t)
    assert f._touch(np.array([0]))[0] == pytest.approx(4e-3)


def test_bounded_influence():
    """Prop. 1: the change of the prediction shrinks for very large residuals."""
    cfg = AdaptiveConfig(use_detector=False, use_scale=False, use_calibration=False)
    changes = []
    for r in [1.0, 10.0, 100.0, 1_000.0, 10_000.0]:
        f = AdaptiveLeafFilter(np.zeros(5), 4, cfg, h=1)
        idx = np.array([0, 1, 4])
        f.update(idx, r, now=0)
        changes.append(abs(f.log_dpred[-1]))
    assert changes[-1] < changes[-2] < changes[-3]
    # Gaussian version is linear in r.
    cfg_g = AdaptiveConfig(use_robust=False, use_detector=False, use_scale=False, use_calibration=False)
    f = AdaptiveLeafFilter(np.zeros(5), 4, cfg_g, h=1)
    f.update(np.array([0, 1, 4]), 10_000.0, now=0)
    assert abs(f.log_dpred[-1]) > 100 * changes[-1]


def test_single_outlier_never_alarms():
    """Prop. 4: with the bounded statistic and lam > 1, one bad tick cannot alarm."""
    cfg = AdaptiveConfig(warmup=100, lam=2.0)
    f = AdaptiveLeafFilter(np.zeros(3), 2, cfg, h=1)
    rng = np.random.default_rng(0)
    idx = np.array([0, 2])
    for t in range(1_000):
        f._tick(t)
        y = 1e6 if t == 600 else rng.standard_normal()
        f.update(idx, y, now=t)
    assert 600 not in f.alarms


def test_raw_detector_alarms_on_single_outlier():
    cfg = AdaptiveConfig(warmup=100, lam=2.0, detector="raw")
    f = AdaptiveLeafFilter(np.zeros(3), 2, cfg, h=1)
    rng = np.random.default_rng(0)
    idx = np.array([0, 2])
    for t in range(1_000):
        f._tick(t)
        y = 1e6 if t == 600 else 0.1 * rng.standard_normal()
        f.update(idx, y, now=t)
    assert 600 in f.alarms
