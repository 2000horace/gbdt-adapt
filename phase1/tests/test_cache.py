import numpy as np

from phase1 import cache


def _fake():
    T = 50
    data = dict(y_clean=np.random.randn(T), signal=np.random.randn(T),
                regime=np.array(["A"] * 25 + ["B"] * 25), seg_starts=np.array([0, 25]), h=5)
    results = {
        "adaptive": dict(mu=np.random.randn(T), s=np.ones(T), df=np.full(T, 4.0), seconds=1.5),
        "abl:no_robust": dict(mu=np.random.randn(T), s=np.ones(T), df=np.full(T, 4.0), seconds=1.2),
        "frozen+ewma": dict(mu=np.random.randn(T), s=np.ones(T), df=np.full(T, np.inf), seconds=0.1),
    }
    diag = dict(log_r=np.random.randn(T), log_dpred=np.random.randn(T))
    return data, results, diag


def test_roundtrip(tmp_path):
    data, results, diag = _fake()
    cache.save(tmp_path, "abc123", data, results, diag, dict(alarms=[3, 7], n_leaves=10))
    key, d2, r2, g2, meta = cache.load(tmp_path)
    assert key == "abc123" and meta["alarms"] == [3, 7] and d2["h"] == 5
    for k in ("y_clean", "signal", "regime", "seg_starts"):
        np.testing.assert_array_equal(d2[k], data[k])
    assert list(r2) == list(results)
    for name in results:
        for f in ("mu", "s", "df"):
            np.testing.assert_array_equal(r2[name][f], results[name][f])
        assert r2[name]["seconds"] == results[name]["seconds"]
    np.testing.assert_array_equal(g2["log_r"], diag["log_r"])


def test_missing_cache_returns_none(tmp_path):
    assert cache.load(tmp_path) is None


def test_key_depends_on_spec_and_functions():
    def f():
        return 1

    def g():
        return 2

    k = cache.cache_key(dict(a=1), (f,))
    assert k == cache.cache_key(dict(a=1), (f,))
    assert k != cache.cache_key(dict(a=2), (f,))
    assert k != cache.cache_key(dict(a=1), (g,))
