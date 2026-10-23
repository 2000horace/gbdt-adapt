import numpy as np
import pytest

from phase1 import dgp, dgp_drift
from phase1.dgp import DGPConfig


def _cfg(variant, regimes="ABCDE", **kw):
    return DGPConfig(variant=variant, n_pretrain=2_000, seg_len=1_000, regimes=tuple(regimes), seed=4, **kw)


def test_original_variants_dispatch_unchanged():
    for v in ("disjoint", "shared", "disjoint_drift"):
        a, b = dgp_drift.generate(_cfg(v, "ABACB")), dgp.generate(_cfg(v, "ABACB"))
        for k in ("X", "y_obs", "signal"):
            np.testing.assert_array_equal(a[k], b[k])


@pytest.mark.parametrize("variant", dgp_drift.DRIFT_VARIANTS)
def test_shapes_and_labels(variant):
    d = dgp_drift.generate(_cfg(variant))
    T = 5 * 1_000
    assert d["X"].shape == (T, 6) and d["y_obs"].shape == (T,) and len(d["regime"]) == T
    assert list(d["seg_starts"]) == [0, 1_000, 2_000, 3_000, 4_000]
    assert len(set(d["regime"])) == 5          # every segment is a new, never-repeated state


def test_nonrecurring_changes_abruptly_between_segments():
    cfg = _cfg("nonrecurring")
    d = dgp_drift.generate(cfg)
    a = dgp_drift.coefficients(cfg)
    x, sig = d["X"], d["signal"]
    pieces = dgp_drift._pieces(x)
    base = cfg.signal_scale * dgp.f0(x)
    seg = np.arange(len(x)) // cfg.seg_len
    np.testing.assert_allclose(sig - base, cfg.signal_scale * (a[seg] * pieces).sum(1), atol=1e-12)
    assert not np.allclose(a[0], a[1])


def test_gradual_is_continuous_across_boundaries():
    cfg = _cfg("gradual")
    a = dgp_drift.coefficients(cfg)
    T = len(cfg.regimes) * cfg.seg_len
    tau = (np.arange(T) % cfg.seg_len) / cfg.seg_len
    seg = np.arange(T) // cfg.seg_len
    coef = a[seg] + (a[seg + 1] - a[seg]) * tau[:, None]
    jumps = np.abs(np.diff(coef, axis=0)).max()
    assert jumps < 2 * np.abs(a[1:] - a[:-1]).max() / cfg.seg_len + 1e-12
