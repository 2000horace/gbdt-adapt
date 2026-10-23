"""Raw LOBSTER preprocessing on fake 7z archives in the real layout."""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1 import lobster_raw as R

LEVELS = 10


def _fake_day(rng, n, price0):
    """Messages and a 10-level book consistent enough for the processing logic."""
    t = np.sort(rng.uniform(R.OPEN_S, R.CLOSE_S, n))
    typ = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.45, 0.1, 0.3, 0.1, 0.05])
    size = rng.integers(1, 500, n)
    dirn = rng.choice([-1, 1], n)
    tick = 100
    mid = np.cumsum(rng.choice([-1, 0, 0, 0, 1], n)) * tick + price0
    spread = rng.choice([1, 1, 2], n) * tick
    ask1, bid1 = mid + spread // 2 + tick, mid - spread // 2 - tick
    asz = rng.integers(1, 1000, (n, LEVELS))
    bsz = rng.integers(1, 1000, (n, LEVELS))
    rep = rng.random(n) < 0.4                         # deep-book-only events: top unchanged
    for i in np.flatnonzero(rep):
        if i > 0:
            ask1[i], bid1[i], asz[i, 0], bsz[i, 0] = ask1[i - 1], bid1[i - 1], asz[i - 1, 0], bsz[i - 1, 0]
    bad = rng.random(n) < 0.003                       # empty ask side
    book = np.empty((n, 4 * LEVELS), np.int64)
    for k in range(LEVELS):
        book[:, 4 * k] = ask1 + k * tick
        book[:, 4 * k + 1] = asz[:, k]
        book[:, 4 * k + 2] = bid1 - k * tick
        book[:, 4 * k + 3] = bsz[:, k]
    book[bad, 0], book[bad, 1] = R.EMPTY_PRICE, 0
    msg = pd.DataFrame({0: t, 1: typ, 2: rng.integers(1, 10**8, n), 3: size, 4: ask1, 5: dirn, 6: "null"})
    return msg, pd.DataFrame(book)


def write_fake_raw(root, tickers=(("AAA", 4000), ("BBB", 2500), ("CCC", 1500)), n_days=7, seed=0, only_2019=()):
    """Archives in both naming styles: 2017-19 (per year) and 2020 (2020-01 .. 2021-04)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    specs = [("2017-19", "_data_dwn_32_302__{t}_2019-01-01_2019-12-31_10.7z", "2019-10"),
             ("2020", "_data_dwn_32_302_lob2020_{t}_2020-01-01_2021-04-27_10.7z", "2020-03")]
    for folder, pattern, month in specs:
        days = pd.bdate_range(f"{month}-01", periods=n_days).strftime("%Y-%m-%d")
        for k, (tk, n) in enumerate(tickers):
            if folder == "2020" and tk in only_2019:
                continue
            tmp = root / "_tmp" / folder / tk
            tmp.mkdir(parents=True, exist_ok=True)
            names = []
            for d in days:
                msg, book = _fake_day(rng, n, 1_000_000 * (k + 1))
                base = f"{tk}_{d}_34200000_57600000"
                msg.to_csv(tmp / f"{base}_message_10.csv", header=False, index=False)
                book.to_csv(tmp / f"{base}_orderbook_10.csv", header=False, index=False)
                names += [f"{base}_message_10.csv", f"{base}_orderbook_10.csv"]
            arc = root / folder / pattern.format(t=tk)
            arc.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["7z", "a", "-bd", str(arc.resolve()), *names], cwd=tmp, check=True,
                           capture_output=True)
    return root


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory):
    return write_fake_raw(tmp_path_factory.mktemp("lobster_raw"), only_2019=("CCC",))


def test_archive_discovery_and_ticker_selection(raw_root):
    arcs = R.find_archives(raw_root)
    assert {(a.ticker, a.start, a.end, a.levels) for a in arcs} >= {
        ("AAA", "2019-01-01", "2019-12-31", 10), ("AAA", "2020-01-01", "2021-04-27", 10)}
    assert {a.ticker for a in R.archives_for_month(arcs, "2020-03")} == {"AAA", "BBB"}
    tickers, _ = R.select_tickers(arcs, ["2019-10", "2020-03"], n=5)
    assert tickers == ["AAA", "BBB"]               # CCC lacks 2020; AAA has the larger archives
    with pytest.raises(ValueError, match="CCC"):
        R.select_tickers(arcs, ["2019-10", "2020-03"], n=5, explicit=["CCC"])


def test_list_days(raw_root):
    arc = next(a for a in R.find_archives(raw_root) if a.ticker == "BBB" and a.start.startswith("2020"))
    days = R.list_days(arc)
    assert len(days) == 7 and all(m.endswith("message_10.csv") and b.endswith("orderbook_10.csv") for m, b in days.values())


def _reference(arc, msg_name, book_name, trim=300.0):
    """Independent, row-by-row recomputation of the kept events and flows."""
    def read(name):
        out = subprocess.run(["7z", "e", "-so", str(arc), name], capture_output=True, check=True).stdout.decode()
        return pd.read_csv(io.StringIO(out), header=None)
    m, b = read(msg_name), read(book_name)
    kept, prev, flows = [], None, []
    buy = sell = 0.0
    for i in range(len(m)):
        t, typ, size, d = m.iat[i, 0], m.iat[i, 1], m.iat[i, 3], m.iat[i, 5]
        a, asz, bd, bsz = b.iat[i, 0], b.iat[i, 1], b.iat[i, 2], b.iat[i, 3]
        inwin = R.OPEN_S + trim <= t < R.CLOSE_S - trim
        if inwin and typ in (4, 5):
            buy += size if d == -1 else 0.0
            sell += size if d == 1 else 0.0
        top = (a, bd, asz, bsz)
        valid = 0 < a < R.EMPTY_PRICE and bd > 0 and a > bd and asz > 0 and bsz > 0
        if inwin and valid and (prev is None or top != prev):
            kept.append(i)
            flows.append((buy, sell))
        prev = top
    return m, b, np.array(kept), np.array(flows)


def test_process_day_matches_reference(raw_root, tmp_path):
    arc = next(a for a in R.find_archives(raw_root) if a.ticker == "BBB" and a.start.startswith("2019"))
    d, (msg_name, book_name) = next(iter(R.list_days(arc).items()))
    out = tmp_path / "BBB" / f"{d}.npz"
    res = R.process_day(dict(archive=str(arc.path), message=msg_name, orderbook=book_name, levels=10, out=str(out)))
    assert res["status"] == "ok"
    m, b, kept, flows = _reference(arc.path, msg_name, book_name)
    z = np.load(out)
    assert res["n_kept"] == len(kept) == len(z["t"])
    np.testing.assert_allclose(z["t"], m.iloc[kept, 0].to_numpy())
    np.testing.assert_allclose(z["mid"], 0.5 * (b.iloc[kept, 0] + b.iloc[kept, 2]).to_numpy() / 1e4)
    np.testing.assert_allclose(np.cumsum(z["buy_vol"]), flows[:, 0], rtol=1e-6)
    np.testing.assert_allclose(np.cumsum(z["sell_vol"]), flows[:, 1], rtol=1e-6)
    np.testing.assert_allclose(z["bdep10"], b.iloc[kept, [4 * k + 3 for k in range(10)]].sum(axis=1).to_numpy(), rtol=1e-6)
    assert np.isfinite(z["r"]).all() and z["r"][0] == 0.0
    # resuming: an existing output is not recomputed
    assert R.process_day(dict(archive=str(arc.path), message=msg_name, orderbook=book_name, out=str(out)))["status"] == "exists"


def test_ofi_definition():
    raw = {"t": np.full(4, 40_000.0), "type": np.ones(4, np.int8), "size": np.ones(4), "dir": np.ones(4, np.int8),
           "ask": np.array([101., 101., 100., 102.]) * 1e4, "bid": np.array([99., 100., 99., 99.]) * 1e4,
           "ask_sz": np.array([[5.], [5.], [7.], [3.]]), "bid_sz": np.array([[4.], [6.], [2.], [2.]])}
    s, _ = R.base_series(raw)
    # Cont-Kukanov-Stoikov increments, computed by hand
    # event 1: bid up (+6), ask unchanged (-5 + 5); event 2: bid down (-6), ask down (-7);
    # event 3: bid unchanged (+2 - 2), ask up (+7)
    np.testing.assert_allclose(s["ofi"], [0.0, 6.0, -13.0, 7.0])


@pytest.fixture(scope="module")
def processed(raw_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("events")
    for a in R.find_archives(raw_root):
        if a.ticker == "AAA" and a.start.startswith("2020"):
            for d, (mname, bname) in R.list_days(a).items():
                R.process_day(dict(archive=str(a.path), message=mname, orderbook=bname, out=str(out / "AAA" / f"{d}.npz")))
    return out


def test_event_unit(processed):
    cfg = R.EventConfig(root=str(processed), ticker="AAA", month="2020-03", h=50, pre_days=3)
    u = R.load_event_unit(cfg)
    assert u["X"].shape == (len(u["y_obs"]), len(u["feature_names"])) and len(u["y_obs"]) > 0
    assert np.isfinite(u["y_obs"]).all() and np.isfinite(u["y_pre"]).all()
    assert len(set(u["dates"])) == 4 and u["seg_starts"][0] == 0


def test_event_features_past_only(processed):
    cfg = R.EventConfig(root=str(processed), ticker="AAA", month="2020-03")
    parts = []
    for k, f in enumerate(R.unit_days(cfg)):
        z = dict(np.load(f))
        z["day"] = np.full(len(z["t"]), k, np.int32)
        parts.append(z)
    df = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    X, _ = R.event_features(df)
    t = len(df["t"]) // 2
    df2 = {k: v.copy() for k, v in df.items()}
    rng = np.random.default_rng(0)
    for k in ("r", "ofi", "buy_vol", "sell_vol", "hidden_vol", "n_msg", "n_exec", "bq1", "aq1", "rel_spread"):
        df2[k][t + 1:] = rng.permutation(df2[k][t + 1:])
    X2, _ = R.event_features(df2)
    np.testing.assert_array_equal(X[:t + 1], X2[:t + 1])


def test_target_same_day_only(processed):
    cfg = R.EventConfig(root=str(processed), ticker="AAA", month="2020-03", h=50)
    parts = []
    for k, f in enumerate(R.unit_days(cfg)):
        z = dict(np.load(f))
        z["day"] = np.full(len(z["t"]), k, np.int32)
        parts.append(z)
    df = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rv = R.forward_rv(df, 50, 1)
    last = np.flatnonzero(np.r_[df["day"][1:] != df["day"][:-1], True])
    for e in last:                                   # the last h events of each day have no label
        assert np.isnan(rv[e - 49:e + 1]).all()
    i = 100
    assert rv[i] == pytest.approx(np.sum(df["r"][i + 2:i + 51].astype(np.float64) ** 2))


def test_cli_dry_run_and_run(raw_root, tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = [sys.executable, str(root / "preprocess_lobster_raw.py"), "--raw-root", str(raw_root),
            "--out", str(tmp_path / "ev"), "--months", "2019-10", "2020-03", "--n-tickers", "2"]
    r = subprocess.run(base + ["--dry-run"], capture_output=True, text=True, cwd=root)
    assert "tickers: AAA, BBB" in r.stdout and "28 stock-days planned" in r.stdout
    r = subprocess.run(base + ["--workers", "2"], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "ev").rglob("*.npz"))) == 28
    r = subprocess.run(base + ["--workers", "2"], capture_output=True, text=True, cwd=root)
    assert "28 already done, 0 to process" in r.stdout
