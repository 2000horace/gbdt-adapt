"""LOBSTER loader tests on fake data written in the exact lobster_minutely/3 format."""

import numpy as np
import pandas as pd
import pytest

from phase1 import lobster as L

HEADER = (",Unnamed: 0,Unnamed: 0.1,Unnamed: 0.1.1,timeIndex,timeHMs,timeHMe,volBuyQty,volBuyNotional,"
          "volSellQty,volSellNotional,nrTrades,volBuyQty_lit,volBuyNotional_lit,volBuyNrTrades_lit,"
          "volSellQty_lit,volSellNotional_lit,volSellNrTrades_lit,volBuyQty_hid,volBuyNotional_hid,"
          "volBuyNrTrades_hid,volSellQty_hid,volSellNotional_hid,volSellNrTrades_hid,bidPx,askPx,bidQty,"
          "askQty,pret_1m,symbol,vol,jump_value,is_jump,signed_jump").split(",")


def write_fake(root, tickers=("AAA", "BBB", "CCC"), n_days=90, year=2020, seed=0, bad_frac=0.002):
    """Random-walk mids with GARCH-like volatility, in the real column layout."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(f"{year}-01-02", periods=n_days).strftime("%Y%m%d")
    for t_i, tk in enumerate(tickers):
        price, vol = 50.0 + 30 * t_i, 3e-4
        for d in dates:
            rows = []
            for m in range(L.MINUTES_PER_DAY):
                vol = np.sqrt(1e-9 + 0.1 * (vol * rng.standard_normal()) ** 2 + 0.88 * vol ** 2)
                price *= np.exp(vol * rng.standard_normal())
                half = max(0.005, price * 2e-4)
                bid, ask = round((price - half) * 1e4), round((price + half) * 1e4)
                if rng.random() < bad_frac:                       # crossed or empty book
                    bid, ask = (ask + 100, bid) if rng.random() < 0.5 else (-9999999999, 9999999999)
                hh, mm = divmod(L.OPEN_MINUTE + m, 60)
                buy, sell = rng.poisson(5000), rng.poisson(5000)
                row = dict.fromkeys(HEADER, "")
                row.update({"": m, "timeIndex": L.OPEN_MINUTE + m, "timeHMs": hh * 100 + mm,
                            "timeHMe": hh * 100 + mm + 1, "volBuyQty": buy, "volSellQty": sell,
                            "volBuyQty_hid": buy // 5, "volSellQty_hid": sell // 5,
                            "nrTrades": rng.poisson(30 * (1 + t_i)), "bidPx": bid, "askPx": ask,
                            "bidQty": rng.integers(100, 2000), "askQty": rng.integers(100, 2000),
                            "symbol": tk})
                rows.append(row)
            (root / d).mkdir(exist_ok=True)
            pd.DataFrame(rows, columns=HEADER).to_csv(root / d / f"{d}-{tk}.csv", index=False)
    return list(dates)


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("lobster3")
    write_fake(root)
    return root


def test_listing_and_availability(fake_root):
    dates = L.list_dates(fake_root, 2020)
    assert len(dates) == 90
    assert L.available_tickers(fake_root, 2020, ["AAA", "ZZZ", "BBB"]) == ["AAA", "BBB"]


def test_auto_ranking_by_liquidity(fake_root):
    ranked, liq = L.rank_tickers(fake_root, 2020)
    assert ranked == ["CCC", "BBB", "AAA"]          # fake trade rates grow with the ticker index
    assert liq["CCC"] > liq["BBB"] > liq["AAA"]


def test_unit_shapes_and_alignment(fake_root):
    cfg = L.LobsterConfig(root=str(fake_root), ticker="AAA", year=2020, pre_days=30)
    u = L.load_unit(cfg)
    assert u["X"].shape[0] == len(u["y_obs"]) > 0 and u["X_pre"].shape[0] == len(u["y_pre"]) > 0
    assert u["X"].shape[1] == len(u["feature_names"])
    assert np.isfinite(u["y_obs"]).all() and np.isfinite(u["y_pre"]).all()
    # kept minutes: after the open trim, and label must end before the closing trim
    assert u["minute"].min() >= cfg.trim_open
    assert (u["minute"] + cfg.h <= L.MINUTES_PER_DAY - cfg.trim_close).all()
    assert u["seg_starts"][0] == 0 and len(u["seg_starts"]) == len(set(u["regime"]))


def test_target_matches_direct_computation(fake_root):
    cfg = L.LobsterConfig(root=str(fake_root), ticker="BBB", year=2020, pre_days=30)
    df = L.load_days(cfg)
    rv = L.forward_rv(df, cfg.h, cfg.gap)
    r = df["r"].to_numpy()
    for t in (100, 2_000, 15_000):
        if np.isfinite(rv[t]):
            assert rv[t] == pytest.approx(np.sum(r[t + cfg.gap + 1:t + cfg.h + 1] ** 2))
    # never across the overnight gap: the last h minutes of every day have no label
    last = df["minute"].to_numpy() > L.MINUTES_PER_DAY - 1 - cfg.h
    assert np.isnan(rv[last]).all()


def test_features_are_past_only(fake_root):
    cfg = L.LobsterConfig(root=str(fake_root), ticker="CCC", year=2020)
    df = L.load_days(cfg)
    X, _ = L.features(df)
    t = 10_000
    df2 = df.copy()
    rng = np.random.default_rng(1)
    for c in ("r", "buy", "sell", "trades", "hidden", "bid_qty", "ask_qty", "rel_spread"):
        df2.loc[t + 1:, c] = rng.permutation(df2.loc[t + 1:, c].to_numpy())   # scramble the future
    X2, _ = L.features(df2)
    np.testing.assert_array_equal(X[:t + 1], X2[:t + 1])


def test_bad_books_are_cleaned(fake_root):
    cfg = L.LobsterConfig(root=str(fake_root), ticker="AAA", year=2020)
    df = L.load_days(cfg)
    assert np.isfinite(df["mid"]).all() and (df["mid"] > 0).all()
    assert np.abs(df["r"]).max() < 0.05            # no jumps from crossed / empty books


def test_har_baseline_runs(fake_root):
    cfg = L.LobsterConfig(root=str(fake_root), ticker="AAA", year=2020, pre_days=30)
    u = L.load_unit(cfg)
    mu = L.har_rolling(u["X_pre"], u["y_pre"], u["X"], u["y_obs"], u["har_cols"], cfg.h, every=2_000, window=10_000)
    assert mu.shape == u["y_obs"].shape and np.isfinite(mu).all()
    # HAR should beat the unconditional mean on volatility-clustered data
    assert np.mean((mu - u["y_obs"]) ** 2) < np.mean((u["y_obs"].mean() - u["y_obs"]) ** 2)


def _blank_books(root, ticker, dates):
    """Make a ticker's book empty on the given days (like SPY on Nasdaq)."""
    for d in dates:
        path = L.day_path(root, d, ticker)
        df = pd.read_csv(path)
        df["bidPx"] = np.nan
        df["askPx"] = np.nan
        df.to_csv(path, index=False)


def test_unusable_ticker_is_screened_and_explained(tmp_path):
    dates = write_fake(tmp_path, tickers=("AAA", "BAD", "CCC"), n_days=80)
    _blank_books(tmp_path, "BAD", dates[:70])
    ranked, _ = L.rank_tickers(tmp_path, 2020)
    assert "BAD" not in ranked and set(ranked) == {"AAA", "CCC"}
    with pytest.raises(ValueError, match="mostly empty book: 70"):
        L.load_unit(L.LobsterConfig(root=str(tmp_path), ticker="BAD", year=2020, pre_days=30))


def test_missing_times_and_broken_files(tmp_path):
    dates = write_fake(tmp_path, tickers=("AAA",), n_days=70)
    # day 0: some timeHMs empty (time must come from timeIndex); day 1: unparseable file
    p0 = L.day_path(tmp_path, dates[0], "AAA")
    df = pd.read_csv(p0)
    df.loc[::7, "timeHMs"] = np.nan
    df.to_csv(p0, index=False)
    L.day_path(tmp_path, dates[1], "AAA").write_text("garbage\x00\x01,,\n1,2\n\"unterminated")
    day, why = L.check_day(tmp_path, dates[0], "AAA")
    assert why == "ok" and len(day) == L.MINUTES_PER_DAY
    assert L.check_day(tmp_path, dates[1], "AAA") == (None, "unreadable file")
    ranked, _ = L.rank_tickers(tmp_path, 2020)
    assert ranked == ["AAA"]
    u = L.load_unit(L.LobsterConfig(root=str(tmp_path), ticker="AAA", year=2020, pre_days=30))
    assert np.isfinite(u["y_obs"]).all()
