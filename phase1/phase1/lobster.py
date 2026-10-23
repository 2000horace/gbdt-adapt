"""Real data: LOBSTER one-minute bars (lobster_minutely/3), realized-volatility task.

Folder layout (one CSV per stock per day):

    <root>/<YYYYMMDD>/<YYYYMMDD>-<TICKER>.csv

    3881400

with, among others, the columns
    timeHMs                 bar start as HHMM (930 = 09:30)
    bidPx, askPx            best bid / ask, in dollars x 10,000
    bidQty, askQty          sizes at the best bid / ask
    volBuyQty, volSellQty   buyer- / seller-initiated volume in the bar
    volBuyQty_hid, volSellQty_hid   the hidden part of it
    nrTrades                number of trades

One replication unit = one stock in one calendar year. The first `pre_days` trading days
pre-train the ensemble; the rest of the year is the stream.

Target
------
    r_m      = log mid_m - log mid_{m-1}       within a day (the overnight return is dropped)
    RV_t     = sum_{i = t + gap + 1}^{t + h} r_i^2       (h - gap minutes, same day only)
    y_t      = log(RV_t + floor)

With gap = 1 the first future minute is skipped, so the minute-t bar (whose volume may
overlap the next return, depending on when the book is snapshotted) never leaks into the
target. The label of minute t is then observable exactly h minutes later, matching the
pipeline's label delay h. `floor` = floor_frac x median positive RV on the pre-training
days only (no look-ahead); it keeps log finite in minutes without price changes.

Every feature at minute t uses minutes <= t only. Rolling windows run over the
concatenated trading minutes, so they may span the previous day's close (never the
overnight return). The first `trim_open` and last `trim_close` minutes of each day are
excluded from the sample (auction effects), as are minutes whose label would need
minutes after the close.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 390
OPEN_MINUTE = 9 * 60 + 30

# Liquid US large caps; tickers missing from the data are skipped by the runner.
DEFAULT_TICKERS = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "INTC", "CSCO", "CMCSA", "PFE", "BAC",
    "JPM", "WFC", "C", "T", "VZ", "XOM", "CVX", "KO", "PEP", "MRK",
    "ORCL", "QCOM", "NVDA", "AMD", "MU", "DIS", "HD", "WMT", "PG", "JNJ",
    "UNH", "V", "MA", "NFLX", "ADBE", "CRM", "PYPL", "ABT", "TMO", "MCD",
    "NKE", "SBUX", "BA", "CAT", "GE", "MMM", "HON", "UNP", "UPS", "IBM",
    "TXN", "AVGO", "GILD", "AMGN", "BMY", "ABBV", "LLY", "MDT", "CVS", "F",
    "GM", "SLB", "COP", "OXY", "MO", "PM", "KMI", "GOOG", "COST", "LOW",
)

COLUMNS = ("timeHMs", "timeIndex", "bidPx", "askPx", "bidQty", "askQty", "volBuyQty", "volSellQty",
           "nrTrades", "volBuyQty_hid", "volSellQty_hid")

RV_WINDOWS = (5, 10, 30, 60, 390, 1950)
HAR_WINDOWS = (10, 60, 390, 1950)          # ~ the forecast horizon, an hour, a day, a week


@dataclass
class LobsterConfig:
    root: str = "/nfs/data/lobster_data/lobster_minutely/3"
    ticker: str = "AAPL"
    year: int = 2020
    h: int = 10                  # label delay (minutes); the RV window is h - gap minutes
    gap: int = 1
    trim_open: int = 5
    trim_close: int = 5
    pre_days: int = 50
    max_rel_spread: float = 0.05
    floor_frac: float = 0.01


# ---------------------------------------------------------------- reading
def list_dates(root, year):
    root = Path(root)
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and len(p.name) == 8 and p.name.isdigit() and p.name.startswith(str(year)))


def day_path(root, date, ticker):
    return Path(root) / date / f"{date}-{ticker}.csv"


def available_tickers(root, year, tickers, min_coverage=0.9, dates=None):
    """Tickers with a file on at least `min_coverage` of the year's trading days."""
    dates = dates if dates is not None else list_dates(root, year)
    out = []
    for t in tickers:
        n = sum(day_path(root, d, t).exists() for d in dates)
        if dates and n >= min_coverage * len(dates):
            out.append(t)
    return out


def tickers_on(root, date):
    """Tickers with a file on one date."""
    pre = f"{date}-"
    return sorted(p.name[len(pre):-4] for p in (Path(root) / date).iterdir()
                  if p.name.startswith(pre) and p.name.endswith(".csv"))


def rank_tickers(root, year, dates=None, min_coverage=0.9, n_sample_days=6, min_usable=5):
    """All tickers on at least `min_coverage` of the year's days, most liquid first.

    Liquidity = mean number of trades per day over `n_sample_days` dates spread over the
    year. A ticker must also have at least `min_usable` usable days (see check_day) among
    those sample days, which screens out stocks whose book is mostly empty or invalid.
    The universe differs by year, so this replaces a fixed ticker list.
    """
    dates = dates if dates is not None else list_dates(root, year)
    counts = {}
    for d in dates:
        for t in tickers_on(root, d):
            counts[t] = counts.get(t, 0) + 1
    covered = [t for t, n in counts.items() if n >= min_coverage * len(dates)]
    sample = [dates[i] for i in np.linspace(0, len(dates) - 1, min(n_sample_days, len(dates))).astype(int)]
    liquidity, usable = {}, []
    for t in covered:
        vals, n_ok = [], 0
        for d in sample:
            path = day_path(root, d, t)
            if path.exists():
                vals.append(pd.read_csv(path, usecols=["nrTrades"])["nrTrades"].sum())
                n_ok += check_day(root, d, t)[0] is not None
        liquidity[t] = np.mean(vals) if vals else 0.0
        if n_ok >= min(min_usable, len(sample)):
            usable.append(t)
    return sorted(usable, key=lambda t: (-liquidity[t], t)), liquidity


def _read_day(root, date, ticker):
    path = day_path(root, date, ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda c: c in COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    # time of day: from timeHMs (HHMM) where present, else from timeIndex (minutes after
    # midnight, 570 = 09:30); rows with neither are dropped
    hhmm = pd.to_numeric(df["timeHMs"], errors="coerce")
    from_hhmm = (hhmm // 100) * 60 + hhmm % 100 - OPEN_MINUTE
    from_index = pd.to_numeric(df["timeIndex"], errors="coerce") - OPEN_MINUTE
    df["minute"] = from_hhmm.where(hhmm.notna(), from_index)
    df = df[df["minute"].notna()]
    df["minute"] = df["minute"].astype(int)
    df = df[(df["minute"] >= 0) & (df["minute"] < MINUTES_PER_DAY)]
    df = df.drop_duplicates("minute", keep="last").set_index("minute").reindex(range(MINUTES_PER_DAY))
    df["date"] = date
    return df


def _clean_day(df, max_rel_spread):
    """Mid-price with invalid books (crossed, empty, absurd spread) forward/back filled."""
    bid = df["bidPx"].to_numpy(float) / 1e4
    ask = df["askPx"].to_numpy(float) / 1e4
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = 0.5 * (bid + ask)
        rel_spread = (ask - bid) / mid
        ok = (bid > 0) & (ask > bid) & (ask < 1e5) & (rel_spread < max_rel_spread)
    mid = pd.Series(np.where(ok, mid, np.nan)).ffill().bfill().to_numpy()
    rel_spread = pd.Series(np.where(ok, rel_spread, np.nan)).ffill().bfill().to_numpy()
    out = pd.DataFrame({
        "date": df["date"].to_numpy(),
        "minute": np.arange(MINUTES_PER_DAY),
        "mid": mid,
        "rel_spread": rel_spread,
        "bid_qty": df["bidQty"].to_numpy(float),
        "ask_qty": df["askQty"].to_numpy(float),
        "buy": df["volBuyQty"].fillna(0).to_numpy(float),
        "sell": df["volSellQty"].fillna(0).to_numpy(float),
        "hidden": (df["volBuyQty_hid"].fillna(0) + df["volSellQty_hid"].fillna(0)).to_numpy(float),
        "trades": df["nrTrades"].fillna(0).to_numpy(float),
        "valid_frac": np.full(MINUTES_PER_DAY, ok.mean()),
    })
    r = np.diff(np.log(mid), prepend=np.nan)
    r[0] = 0.0                       # no overnight return
    out["r"] = np.nan_to_num(r)
    return out


def check_day(root, date, ticker, max_rel_spread=0.05):
    """(cleaned day or None, reason). A day is usable if its file exists, at least half of
    its minutes have a quote, and at least half of those quotes are valid."""
    try:
        raw = _read_day(root, date, ticker)
    except Exception:                 # malformed file: an unusable day, never a crash
        return None, "unreadable file"
    if raw is None:
        return None, "missing file"
    if raw["bidPx"].notna().sum() < MINUTES_PER_DAY // 2:
        return None, "mostly empty book"
    day = _clean_day(raw, max_rel_spread)
    if day["valid_frac"].iloc[0] < 0.5 or not np.isfinite(day["mid"]).all():
        return None, "mostly invalid quotes"
    return day, "ok"


def load_days(cfg: LobsterConfig, dates=None, min_days=1):
    """All usable trading days of one stock-year, concatenated in time order."""
    dates = dates if dates is not None else list_dates(cfg.root, cfg.year)
    days, reasons = [], {}
    for d in dates:
        day, why = check_day(cfg.root, d, cfg.ticker, cfg.max_rel_spread)
        reasons[why] = reasons.get(why, 0) + 1
        if day is not None:
            days.append(day)
    if len(days) < min_days:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
        raise ValueError(f"{cfg.ticker} {cfg.year}: only {len(days)} usable days of {len(dates)} ({detail})")
    return pd.concat(days, ignore_index=True)


# ---------------------------------------------------------------- features and target
def _trailing_sum(x, w):
    """Sum of x over the last w rows including the current one (shorter at the start)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - w, 0)]


def features(df):
    """Past-only features at the end of each minute. Returns (matrix, names)."""
    r, r2 = df["r"].to_numpy(), df["r"].to_numpy() ** 2
    vol = df["buy"].to_numpy() + df["sell"].to_numpy()
    flow = df["buy"].to_numpy() - df["sell"].to_numpy()
    cols, names = [], []
    for w in RV_WINDOWS:
        cols.append(np.log(_trailing_sum(r2, w) / w + 1e-12)); names.append(f"lrv_{w}")
    for w in (1, 5, 30):
        cols.append(_trailing_sum(r, w)); names.append(f"ret_{w}")
    for w in (1, 5, 30):
        cols.append(_trailing_sum(flow, w) / (_trailing_sum(vol, w) + 1.0)); names.append(f"ofi_{w}")
    bq, aq = df["bid_qty"].to_numpy(), df["ask_qty"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        cols.append((bq - aq) / (bq + aq)); names.append("depth_imb")
    rs = df["rel_spread"].to_numpy()
    cols.append(np.log(rs + 1e-8)); names.append("log_spread")
    cols.append(np.log(_trailing_sum(rs, 30) / 30 + 1e-8)); names.append("log_spread_30")
    cols.append(np.log1p(df["trades"].to_numpy())); names.append("log_trades_1")
    cols.append(np.log1p(_trailing_sum(df["trades"].to_numpy(), 30))); names.append("log_trades_30")
    cols.append(np.log((_trailing_sum(vol, 30) + 1.0) / (_trailing_sum(vol, 390) / 13.0 + 1.0)))
    names.append("rel_volume_30")
    cols.append(_trailing_sum(df["hidden"].to_numpy(), 30) / (_trailing_sum(vol, 30) + 1.0))
    names.append("hidden_share_30")
    cols.append(df["minute"].to_numpy(float)); names.append("minute_of_day")
    return np.column_stack(cols).astype(np.float64), names


def forward_rv(df, h, gap):
    """RV over minutes t+gap+1 .. t+h of the same day; NaN where that needs another day."""
    r2 = df["r"].to_numpy() ** 2
    n = len(r2)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    t = np.arange(n)
    lo, hi = t + gap + 1, t + h + 1                       # sum over [lo, hi) rows
    ok = hi <= n
    rv = np.full(n, np.nan)
    rv[ok] = c[hi[ok]] - c[lo[ok]]
    # same-day check: the row h ahead must be the same date, h minutes later
    minute = df["minute"].to_numpy()
    same = np.zeros(n, bool)
    same[ok] = minute[t[ok] + h] == minute[t[ok]] + h
    rv[~same] = np.nan
    return rv


# ---------------------------------------------------------------- one unit
def load_unit(cfg: LobsterConfig, dates=None):
    """One stock-year as the dict the pipeline expects (same keys as the synthetic data)."""
    df = load_days(cfg, dates, min_days=cfg.pre_days + 21)
    X, names = features(df)
    rv = forward_rv(df, cfg.h, cfg.gap)
    minute = df["minute"].to_numpy()
    keep = (minute >= cfg.trim_open) & (minute < MINUTES_PER_DAY - cfg.trim_close) & np.isfinite(rv)
    # labels may not reach into the closing trim either
    keep &= minute + cfg.h < MINUTES_PER_DAY - cfg.trim_close + 1

    dates = df["date"].to_numpy()
    day_list = sorted(set(dates))
    if len(day_list) <= cfg.pre_days + 20:
        raise ValueError(f"{cfg.ticker} {cfg.year}: only {len(day_list)} usable days")
    pre_mask = np.isin(dates, day_list[:cfg.pre_days]) & keep
    stream_mask = ~np.isin(dates, day_list[:cfg.pre_days]) & keep

    pos = rv[pre_mask][rv[pre_mask] > 0]
    floor = cfg.floor_frac * (np.median(pos) if len(pos) else 1e-10)
    y = np.log(rv + floor)

    s_dates = dates[stream_mask]
    months = np.array([d[:6] for d in s_dates])
    seg_starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    n_stream = int(stream_mask.sum())
    return {
        "X_pre": X[pre_mask], "y_pre": y[pre_mask],
        "X": X[stream_mask], "y_obs": y[stream_mask], "y_clean": y[stream_mask],
        "signal": np.full(n_stream, np.nan), "noise_var": np.full(n_stream, np.nan),
        "regime": np.array([f"{m[:4]}-{m[4:]}" for m in months]), "seg_starts": seg_starts,
        "is_bad": np.zeros(n_stream, bool), "h": cfg.h,
        "feature_names": names, "har_cols": [names.index(f"lrv_{w}") for w in HAR_WINDOWS],
        "dates": s_dates, "minute": minute[stream_mask], "unit": f"{cfg.ticker}-{cfg.year}",
        "floor": floor,
    }


# ---------------------------------------------------------------- HAR baseline
def har_rolling(X_pre, y_pre, X, y_obs, cols, h, every=2_000, window=40_000, ridge=1e-6):
    """HAR regression of the target on log realized variances over several windows,
    refit every `every` events on the last `window` observable labels (pre-training
    data included while the stream is short). Same schedule as the XGBoost baselines."""
    Za = np.column_stack([np.ones(len(X_pre) + len(X)), np.vstack([X_pre, X])[:, cols]])
    ya = np.concatenate([y_pre, y_obs])
    off = len(X_pre)
    T = len(X)
    mu = np.empty(T)
    for start in range(0, T, every):
        stop = min(start + every, T)
        hi = off + start - h + 1                 # rows with labels observable at `start`
        lo = max(0, hi - window)
        Z, yy = Za[lo:hi], ya[lo:hi]
        ok = np.isfinite(Z).all(axis=1) & np.isfinite(yy)
        A = Z[ok].T @ Z[ok] + ridge * np.eye(Z.shape[1])
        beta = np.linalg.solve(A, Z[ok].T @ yy[ok])
        Zs = np.nan_to_num(Za[off + start:off + stop])
        mu[start:stop] = Zs @ beta
    return mu
