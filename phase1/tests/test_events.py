"""Event-level runner on fake processed stock-days (base-series format)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1 import events as EV
from phase1 import lobster_raw as R


def write_fake_processed(root, tickers=("AAA", "BBB", "CCC"), months=("2019-10", "2020-03"), n_days=8,
                         per_day=6_000, seed=0):
    rng = np.random.default_rng(seed)
    for k, tk in enumerate(tickers):
        for m in months:
            days = pd.bdate_range(f"{m}-01", periods=n_days).strftime("%Y-%m-%d")
            vol = 2e-5
            for d in days:
                n = per_day * (k + 1)
                v = np.empty(n)
                for i in range(n):                       # volatility clustering
                    vol = np.sqrt(1e-12 + 0.05 * (vol * rng.standard_normal()) ** 2 + 0.94 * vol ** 2)
                    v[i] = vol
                r = v * rng.standard_normal(n)
                r[0] = 0.0
                mid = 100.0 * np.exp(np.cumsum(r))
                bq, aq = rng.integers(1, 500, n).astype(float), rng.integers(1, 500, n).astype(float)
                s = {"t": np.sort(rng.uniform(R.OPEN_S + 300, R.CLOSE_S - 300, n)), "mid": mid, "r": r,
                     "rel_spread": np.full(n, 1e-4), "bq1": bq, "aq1": aq, "ofi": rng.normal(0, 100, n)}
                for L in (3, 5, 10):
                    s[f"bdep{L}"], s[f"adep{L}"] = bq * L, aq * L
                for f in ("buy_vol", "sell_vol", "hidden_vol"):
                    s[f] = rng.poisson(20, n).astype(float)
                s["n_exec"], s["n_msg"] = rng.poisson(0.2, n).astype(float), rng.poisson(2, n).astype(float) + 1
                R.save_day(Path(root) / tk / f"{d}.npz", s, {"n_messages": int(s["n_msg"].sum()), "n_window": int(s["n_msg"].sum()),
                                                               "n_kept": n, "n_invalid": 0})
    return root


@pytest.fixture(scope="module")
def proc(tmp_path_factory):
    return write_fake_processed(tmp_path_factory.mktemp("events"))


def test_run_unit(proc):
    cfg = EV.EventRunConfig(root=str(proc), ticker="BBB", month="2020-03", h=20, n_trees=20, chunk=5_000,
                            ens_trials=4, tune_threads=1, trees_range=(10, 40),
                            p0_grid=(1e-4, 1e-3, 1e-2), delta_grid=(0.999, 1.0), sgd_grid=(0.003, 0.03),
                            batch=(("1d", "1h"), ("2d", "1d")), har=("2d", "1h"))
    rows, seconds, meta = EV.run_unit(cfg)
    cov = {r["method"]: r["coverage80"] for r in rows}
    assert 0.7 < cov["adaptive"] < 0.9 and 0.7 < cov["frozen+robust"] < 0.9      # calibrated on the event clock
    assert meta["scale_memory"] in (20, 100, 400)
    names = {r["method"] for r in rows}
    assert {"adaptive", "frozen+robust", "sgd+robust", "har+robust", "refresh[1d/1h]+robust",
            "rolling[2d/1d]+robust"} <= names
    for r in rows:
        assert np.isfinite([r["log_score"], r["crps"], r["mae"], r["rmse"]]).all()
    assert meta["n_eval_days"] == 3 and meta["n_eval_events"] > 0
    assert seconds["rolling[2d/1d]"] > 0 and seconds["adaptive+sgd_updates"] > 0
    # sanity: on stationary fake data the adaptive filter is close to the frozen ensemble
    sc = {r["method"]: r["log_score"] for r in rows}
    assert abs(sc["adaptive"] - sc["frozen+robust"]) < 0.1


def test_batch_cache_reuse(proc, tmp_path):
    cfg = EV.EventRunConfig(root=str(proc), ticker="AAA", month="2019-10", h=20, n_trees=20, chunk=5_000,
                            ens_tune=False,
                            p0_grid=(1e-3,), delta_grid=(1.0,), overlap_grid=("h",), sgd_grid=(0.003,),
                            scale_memory_grid=(5,), batch=(("1d", "1h"),), har=("2d", "1h"),
                            batch_cache=str(tmp_path / "bc"))
    rows1, sec1, _ = EV.run_unit(cfg)
    assert len(list((tmp_path / "bc").rglob("*.npy"))) == 2        # refresh and rolling
    rows2, sec2, _ = EV.run_unit(cfg)
    r1 = {r["method"]: r for r in rows1}
    r2 = {r["method"]: r for r in rows2}
    for name in ("refresh[1d/1h]+robust", "rolling[1d/1h]+robust"):
        assert r2[name]["log_score"] == pytest.approx(r1[name]["log_score"], rel=1e-6)
        assert sec2[name.replace("+robust", "")] == sec1[name.replace("+robust", "")]
    # a different filter grid keeps the batch cache valid
    rows3, _, _ = EV.run_unit(EV.EventRunConfig(**{**cfg.__dict__, "p0_grid": (1e-4,)}))
    assert {r["method"]: r for r in rows3}["rolling[1d/1h]+robust"]["log_score"] == pytest.approx(
        r1["rolling[1d/1h]+robust"]["log_score"], rel=1e-6)


def test_event_spec_conversion():
    assert EV._events("1d", 6_330) == 6_330
    assert EV._events("1h", 19_000) == 3_000         # 6 1/3 trading hours per day (09:35-15:55)
    assert EV._events("5d", 100) == 500
    assert EV._events(1234, 100) == 1234


def test_cli_runs_caches_and_aggregates(proc, tmp_path):
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "run_events.py"), "--root", str(proc), "--out", str(tmp_path / "ev"),
           "--h", "20", "--batch", "1d/1h", "--workers", "2", "--batch-cache", str(tmp_path / "bc"),
           "--ens-trials", "3", "--tune-threads", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "6 units: 0 cached, 6 to run" in r.stdout and "gain of adaptive" in r.stdout
    g = pd.read_csv(tmp_path / "ev" / "gains.csv")
    assert set(g.month) == {"2019-10", "2020-03"} and (g.n == 3).all()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert "6 units: 6 cached, 0 to run" in r.stdout


def test_forward_return_same_day_and_values():
    rng = np.random.default_rng(0)
    df = {"r": rng.standard_normal(300).astype(np.float32), "day": np.repeat([0, 1, 2], 100).astype(np.int32)}
    out = EV.forward_return(df, 10, 1)
    assert out[5] == pytest.approx(np.sum(df["r"][7:16].astype(np.float64)))
    for d in range(3):                                   # the last h events of each day have no label
        assert np.isnan(out[d * 100 + 90:d * 100 + 100]).all() and np.isfinite(out[d * 100:d * 100 + 90]).all()


def test_return_target(tmp_path):
    # fake data where order-flow imbalance predicts the next returns
    root = write_fake_processed(tmp_path / "ev", tickers=("RET",), months=("2020-03",), per_day=5_000)
    for f in sorted((root / "RET").glob("*.npz")):
        z = dict(np.load(f))
        n = len(z["r"])
        ofi = np.random.default_rng(len(str(f))).normal(0, 100, n)
        z["ofi"] = ofi.astype(np.float32)
        r = 1e-5 * np.random.default_rng(1).standard_normal(n)
        r[1:] += 2e-7 * ofi[:-1]                         # signal: today's flow moves the next mid
        r[0] = 0.0
        z["r"] = r.astype(np.float32)
        np.savez(f, **z)
    cfg = EV.EventRunConfig(root=str(root), ticker="RET", month="2020-03", h=20, n_trees=30, chunk=4_000,
                            ens_trials=4, tune_threads=1, trees_range=(10, 40),
                            target="return", p0_grid=(1e-4, 1e-3, 1e-2), delta_grid=(1.0,), sgd_grid=(0.003, 0.03),
                            batch=(("2d", "1d"),), har=("2d", "1h"))
    A = EV.build_arrays(cfg)
    early = (A["day"] < 5) & np.isfinite(A["y"])
    assert np.std(A["y"][early]) == pytest.approx(1.0, rel=1e-6)    # standardized on pre-evaluation days only
    rows, seconds, meta = EV.run_unit(cfg)
    sc = {r["method"]: r for r in rows}
    assert {"zero+robust", "linear+robust", "adaptive", "sgd+robust", "rolling[2d/1d]+robust"} <= set(sc)
    assert sc["zero+robust"]["r2_oos"] == pytest.approx(0.0) and sc["zero+robust"]["corr"] == 0.0
    assert meta["target"] == "return"
    # the flow signal is found by the tree methods
    for m in ("adaptive", "frozen+robust", "linear+robust"):
        assert sc[m]["corr"] > 0.05 and sc[m]["r2_oos"] > 0.0


def test_ensemble_bayesopt(proc):
    cfg = EV.EventRunConfig(root=str(proc), ticker="CCC", month="2019-10", h=20, ens_trials=8, tune_threads=1,
                            trees_range=(10, 60), mcw_range=(5, 500), tune_rows=5_000)
    A = EV.build_arrays(cfg)
    tr = (A["day"] < 3) & np.isfinite(A["y"])
    tu = (A["day"] >= 3) & (A["day"] < 5) & np.isfinite(A["y"])
    ens, info = EV.tune_ensemble(cfg, A["X"][tr], A["y"][tr], A["X"][tu], A["y"][tu].astype(np.float64))
    assert info["ens_trials"] == 8 and info["ens_trial_rows"] == 5_000
    assert cfg.lr_range[0] <= ens["learning_rate"] <= cfg.lr_range[1]
    assert cfg.trees_range[0] <= ens["n_trees"] <= cfg.trees_range[1] and isinstance(ens["n_trees"], int)
    # minimum samples per leaf: chosen on the 5,000-row subsample, scaled to the full training set
    assert ens["min_child_weight"] == pytest.approx(ens["mcw_per_row"] * tr.sum())
    assert 5 <= ens["mcw_per_row"] * 5_000 <= 500
    # deterministic for a fixed seed
    ens2, _ = EV.tune_ensemble(cfg, A["X"][tr], A["y"][tr], A["X"][tu], A["y"][tu].astype(np.float64))
    assert ens2 == ens
    # switching tuning off returns the fixed settings
    fixed, _ = EV.tune_ensemble(EV.EventRunConfig(**{**cfg.__dict__, "ens_tune": False}), A["X"][tr], A["y"][tr],
                                A["X"][tu], A["y"][tu].astype(np.float64))
    assert fixed["learning_rate"] == 0.1 and fixed["n_trees"] == 100 and fixed["min_child_weight"] == 20.0
