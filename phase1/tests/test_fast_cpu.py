from dataclasses import replace

import numpy as np
import pytest

from phase1 import baselines as B
from phase1 import fast_cpu as F
from phase1 import metrics as M
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
    return data, lm, lm.leaf_index(data["X"])


CONFIGS = {
    "default": {},
    "tuned_real_data": dict(P0=1e-3, P0_bias=1e-2, delta=1.0, overlap_factor=10.0),
    "alarms": dict(warmup=200, lam=3.0, zeta=0.0, alarm_window=300, use_alarm_action=True),
    "alarms_no_action": dict(warmup=200, lam=3.0, zeta=0.0),
    "raw_detector": dict(warmup=200, detector="raw", lam=20.0, zeta=0.5, use_alarm_action=True),
    "no_robust": dict(use_robust=False),
    "no_discount": dict(use_discount=False),
    "no_scale": dict(use_scale=False),
    "no_calibration": dict(use_calibration=False),
    "no_detector": dict(use_detector=False),
    "big_cap": dict(P_max_factor=64.0),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_matches_numpy_reference(setup, name):
    data, lm, idx = setup
    cfg = replace(AdaptiveConfig(P0=3e-4, P0_bias=3e-3, delta=0.999), **CONFIGS[name])
    ref = AdaptiveLeafFilter(lm.w0, lm.bias_index, cfg, H)
    mu_np, s_np, _ = ref.run(idx, data["y_obs"])
    mu, s, fired = F.OnlineFilter(lm.w0, lm.bias_index, cfg, H).run(idx, data["y_obs"])
    np.testing.assert_allclose(mu, mu_np, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(s, s_np, rtol=1e-12, atol=1e-13)
    assert list(np.flatnonzero(fired)) == ref.alarms
    if name == "alarms":
        assert len(ref.alarms) > 0


def test_step_api_equals_run(setup):
    data, lm, idx = setup
    cfg = AdaptiveConfig(P0=3e-4, P0_bias=3e-3, delta=0.999)
    mu_run, s_run, _ = F.OnlineFilter(lm.w0, lm.bias_index, cfg, H).run(idx, data["y_obs"])
    f = F.OnlineFilter(lm.w0, lm.bias_index, cfg, H)
    idx64 = idx.astype(np.int64)
    out = []
    for i in range(len(idx)):
        j = i - H
        out.append(f.step(i, idx64[i], idx64[j] if j >= 0 else None, data["y_obs"][j] if j >= 0 else 0.0)[:2])
    out = np.array(out)
    np.testing.assert_array_equal(out[:, 0], mu_run)
    np.testing.assert_array_equal(out[:, 1], s_run)


def test_sgd_matches_numpy(setup):
    data, lm, idx = setup
    np.testing.assert_allclose(F.run_sgd(lm.w0, idx, data["y_obs"], H, 0.01),
                               B.online_sgd(lm, idx, data["y_obs"], H, eta0=0.01), rtol=1e-12, atol=1e-13)


def test_chunked_stream_equals_single_run(setup):
    data, lm, idx = setup
    cfg = AdaptiveConfig(P0=3e-4, P0_bias=3e-3, delta=0.999, warmup=200, lam=3.0, zeta=0.0, use_alarm_action=True)
    mu, s, fired = F.OnlineFilter(lm.w0, lm.bias_index, cfg, H).run(idx, data["y_obs"])
    mu_g = F.run_sgd(lm.w0, idx, data["y_obs"], H, 0.01)
    for chunk in (7, 333, 1_000, len(idx)):
        out = F.run_stream(lambda a, b: idx[a:b], data["y_obs"], H, chunk,
                           filt=F.OnlineFilter(lm.w0, lm.bias_index, cfg, H), sgd=(lm.w0, 0.01))
        np.testing.assert_array_equal(out["mu"], mu)
        np.testing.assert_array_equal(out["s"], s)
        np.testing.assert_array_equal(out["fire"], fired)
        np.testing.assert_array_equal(out["mu_sgd"], mu_g)


def test_compiled_robust_scale_matches(setup):
    data, lm, idx = setup
    mu = B.online_sgd(lm, idx, data["y_obs"], H, eta0=0.01)
    np.testing.assert_allclose(F.robust_scale(mu, data["y_obs"], H), B.robust_t_scale(mu, data["y_obs"], H)[0], rtol=1e-12)


class _ForecastScaleRef(AdaptiveLeafFilter):
    """Independent NumPy reference for forecast_scale=True: the leaf update is unchanged; the
    noise scale and calibration learn from y - mu_forecast with the forecast-time scale."""

    def run(self, idx_stream, y_obs):
        T = len(idx_stream)
        mu, s = np.empty(T), np.empty(T)
        for i in range(T):
            self._tick(i)
            j = i - self.h
            if j >= 0:
                self._update_fc(idx_stream[j], y_obs[j], i, mu[j], s[j])
            mu[i], s[i], _ = self.predict(idx_stream[i])
        return mu, s

    def _update_fc(self, idx, y, now, mu_f, s_f):
        cfg = self.cfg
        P = self._touch(idx)
        S0 = P.sum()
        mu = self.m[idx].sum()
        s = self.c * (S0 + self.sigma2)
        r = y - mu
        nu = cfg.nu_alarm if self._in_alarm(now) else cfg.nu
        omega = (nu + 1.0) / (nu + r * r / s) if cfg.use_robust else 1.0
        s_tilde = S0 + self.ofac * self.sigma2 / omega
        self.m[idx] += P / s_tilde * r
        self.P[idx] = P - P * P / s_tilde
        rf = y - mu_f
        om_f = (nu + 1.0) / (nu + rf * rf / s_f) if cfg.use_robust else 1.0
        if cfg.use_scale:
            self.n_eff = cfg.delta_v * self.n_eff + 1.0
            self.d_eff = cfg.delta_v * self.d_eff + om_f * rf * rf
            self.sigma2 = max(self.d_eff / self.n_eff, cfg.sigma2_min)
        if cfg.use_calibration:
            self.c *= np.exp(cfg.eta_c * (float(abs(rf) > self.q * np.sqrt(s_f)) - cfg.alpha))
        self.n_updates += 1


@pytest.mark.parametrize("h", [1, 10, 37])
def test_forecast_scale_matches_reference(setup, h):
    data, lm, idx = setup
    cfg = AdaptiveConfig(P0=3e-4, P0_bias=3e-3, delta=0.999, use_detector=False)
    mu_ref, s_ref = _ForecastScaleRef(lm.w0, lm.bias_index, cfg, h).run(idx, data["y_obs"])
    mu, s, _ = F.OnlineFilter(lm.w0, lm.bias_index, cfg, h, forecast_scale=True).run(idx, data["y_obs"])
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(s, s_ref, rtol=1e-12, atol=1e-13)
    for chunk in (5, 400):
        out = F.run_stream(lambda a, b: idx[a:b], data["y_obs"], h, chunk,
                           filt=F.OnlineFilter(lm.w0, lm.bias_index, cfg, h, forecast_scale=True))
        np.testing.assert_array_equal(out["mu"], mu)
        np.testing.assert_array_equal(out["s"], s)


def test_forecast_scale_fixes_coverage_with_overlapping_labels():
    """Long overlapping labels: residuals against current leaves understate the forecast error."""
    rng = np.random.default_rng(0)
    T, h, D = 60_000, 400, 64
    e = rng.standard_normal(T + h)
    y = np.convolve(e, np.ones(h) / np.sqrt(h), "valid")[:T]          # heavily overlapping labels
    leaf = np.repeat(rng.integers(0, D - 1, T // 2_000 + 1), 2_000)[:T]    # persistent features, as in LOB data
    idx = np.column_stack([leaf, np.full(T, D - 1)])
    cfg = AdaptiveConfig(P0=1e-2, P0_bias=1e-1, delta=0.9999, overlap_factor=1.0, use_detector=False,
                         eta_c=0.05 / h, delta_v=1 - 1 / (5 * h))
    cov = {}
    for fc in (False, True):
        mu, s, _ = F.OnlineFilter(np.zeros(D), D - 1, cfg, h, forecast_scale=fc).run(idx, y)
        u = M.pit(y[5 * h:], mu[5 * h:], s[5 * h:], cfg.nu)
        cov[fc] = np.mean((u > 0.1) & (u < 0.9))
    assert cov[False] < 0.75 and abs(cov[True] - 0.8) < 0.03
