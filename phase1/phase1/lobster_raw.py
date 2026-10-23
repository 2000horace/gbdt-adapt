"""Raw LOBSTER (event level): archives -> compact per-day files -> event-level units.

Archives (one per ticker and period, 10 book levels), e.g.

    <raw_root>/2017-19/_data_dwn_32_302__AAPL_2019-01-01_2019-12-31_10.7z
    <raw_root>/2020/_data_dwn_32_302_lob2020_AAPL_2020-01-01_2021-04-27_10.7z

each holding, per trading day,

    AAPL_2020-03-02_34200000_57600000_message_10.csv     time, type, order id, size, price, direction[, mpid]
    AAPL_2020-03-02_34200000_57600000_orderbook_10.csv   per level: ask px, ask size, bid px, bid size

(prices in dollars x 10,000; row i of the order book is the book after message i).

Step 1, preprocessing (process_day): stream both files of one day out of the archive (no
extraction to disk), keep the events that change the top of the book inside the trading
window, and save compact, feature-agnostic base series per kept event:

    t, mid, r (log mid return), rel_spread, bid/ask size at level 1, bid/ask depth summed
    over levels 1-3, 1-5, 1-10, order-flow imbalance (Cont, Kukanov & Stoikov 2014), and,
    accumulated over ALL events since the previous kept event: buyer- and seller-initiated
    executed volume, hidden executed volume, number of executions, number of messages.

Step 2, units (load_event_unit): one stock over one month; features and target are built
from the base series, so they can change without reprocessing the archives.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_S, CLOSE_S = 34_200.0, 57_600.0          # 09:30 and 16:00 in seconds after midnight
ARCHIVE_RE = re.compile(r"_(?P<ticker>[A-Z][A-Z0-9.]*)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})_(?P<levels>\d+)\.7z$")
INNER_RE = re.compile(r"^(?P<ticker>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<t0>\d+)_(?P<t1>\d+)_(?P<kind>message|orderbook)_(?P<levels>\d+)\.csv$")
EMPTY_PRICE = 9_999_999_999
BASE_FLOAT32 = ("r", "rel_spread", "bq1", "aq1", "bdep3", "adep3", "bdep5", "adep5", "bdep10", "adep10",
                "ofi", "buy_vol", "sell_vol", "hidden_vol", "n_exec", "n_msg")


# ---------------------------------------------------------------- archives
@dataclass(frozen=True)
class Archive:
    path: Path
    ticker: str
    start: str
    end: str
    levels: int
    size: int


def find_archives(raw_root):
    """All ticker archives under raw_root (any sub-folder)."""
    out = []
    for p in Path(raw_root).rglob("*.7z"):
        m = ARCHIVE_RE.search(p.name)
        if m:
            out.append(Archive(p, m["ticker"], m["start"], m["end"], int(m["levels"]), p.stat().st_size))
    return out


def archives_for_month(archives, month):
    """Archives whose date range covers the whole month 'YYYY-MM'."""
    first = f"{month}-01"
    last = (pd.Period(month, "M").end_time.date()).isoformat()
    return [a for a in archives if a.start <= first and a.end >= last]


def list_days(archive: Archive):
    """{date: (message file, orderbook file)} inside an archive (reads the headers only)."""
    res = subprocess.run(["7z", "l", "-slt", str(archive.path)], capture_output=True, text=True, check=True)
    files = {}
    for line in res.stdout.splitlines():
        if line.startswith("Path = "):
            m = INNER_RE.match(line[7:].strip())
            if m:
                files.setdefault(m["date"], {})[m["kind"]] = m.group(0)
    return {d: (f["message"], f["orderbook"]) for d, f in sorted(files.items()) if len(f) == 2}


def select_tickers(archives, months, n, explicit=None):
    """The n tickers with the largest archives (a proxy for message activity) among those
    covering every requested month, ranked by their mean size rank across months."""
    per_month = {m: {a.ticker: a for a in archives_for_month(archives, m)} for m in months}
    common = set.intersection(*(set(v) for v in per_month.values())) if per_month else set()
    if explicit:
        missing = [t for t in explicit if t not in common]
        if missing:
            raise ValueError(f"no archive covering {months} for: {', '.join(missing)}")
        return list(explicit), per_month
    rank = {t: 0.0 for t in common}
    for m, arcs in per_month.items():
        order = sorted(common, key=lambda t: -arcs[t].size)
        for i, t in enumerate(order):
            rank[t] += i / len(months)
    return sorted(common, key=lambda t: (rank[t], t))[:n], per_month


# ---------------------------------------------------------------- streaming
def _stream_csv(archive_path, inner, **kwargs):
    proc = subprocess.Popen(["7z", "e", "-so", str(archive_path), inner],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        df = pd.read_csv(proc.stdout, header=None, **kwargs)
    finally:
        proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"7z failed ({rc}) on {inner}")
    return df


def read_day(archive_path, message_name, orderbook_name, levels=10):
    """Message and order-book columns needed for the base series, as NumPy arrays."""
    msg = _stream_csv(archive_path, message_name, usecols=[0, 1, 3, 5],
                      dtype={0: np.float64, 1: np.int8, 3: np.int64, 5: np.int8})
    L = min(levels, 10)
    cols = [0, 2] + [4 * k + 1 for k in range(L)] + [4 * k + 3 for k in range(L)]
    book = _stream_csv(archive_path, orderbook_name, usecols=cols, dtype=np.float64)
    if len(msg) != len(book):
        raise ValueError(f"{message_name}: {len(msg)} messages but {len(book)} book rows")
    return {
        "t": msg[0].to_numpy(), "type": msg[1].to_numpy(), "size": msg[3].to_numpy(np.float64),
        "dir": msg[5].to_numpy(),
        "ask": book[0].to_numpy(), "bid": book[2].to_numpy(),
        "ask_sz": book[[4 * k + 1 for k in range(L)]].to_numpy(), "bid_sz": book[[4 * k + 3 for k in range(L)]].to_numpy(),
    }


# ---------------------------------------------------------------- one day -> base series
def base_series(raw, trim_open_s=300.0, trim_close_s=300.0):
    """Top-of-book events of one day and the per-event base series (see module docstring)."""
    t, typ, size, dirn = raw["t"], raw["type"], raw["size"], raw["dir"]
    ask, bid, asz, bsz = raw["ask"], raw["bid"], raw["ask_sz"], raw["bid_sz"]
    asz = np.where(np.abs(asz) > 1e9, 0.0, asz)
    bsz = np.where(np.abs(bsz) > 1e9, 0.0, bsz)
    valid = ((ask > 0) & (bid > 0) & (ask < EMPTY_PRICE) & (bid > -EMPTY_PRICE) & (ask > bid)
             & (asz[:, 0] > 0) & (bsz[:, 0] > 0))
    window = (t >= OPEN_S + trim_open_s) & (t < CLOSE_S - trim_close_s)

    top = np.column_stack([ask, bid, asz[:, 0], bsz[:, 0]])
    changed = np.ones(len(t), bool)
    changed[1:] = (top[1:] != top[:-1]).any(axis=1)
    keep = np.flatnonzero(changed & valid & window)

    # flows over ALL messages inside the window, accumulated up to each kept event
    exe = np.isin(typ, (4, 5)) & window
    flows = {
        "buy_vol": np.where(exe & (dirn == -1), size, 0.0),      # resting sell hit: buyer-initiated
        "sell_vol": np.where(exe & (dirn == 1), size, 0.0),
        "hidden_vol": np.where((typ == 5) & window, size, 0.0),
        "n_exec": exe.astype(np.float64),
        "n_msg": window.astype(np.float64),
    }
    out = {}
    for k, x in flows.items():
        c = np.cumsum(x)[keep]
        out[k] = np.diff(c, prepend=0.0)

    a, b = ask[keep], bid[keep]
    qa, qb = asz[keep, 0], bsz[keep, 0]
    mid = 0.5 * (a + b) / 1e4
    r = np.diff(np.log(mid), prepend=np.log(mid[:1]))
    ofi = np.zeros(len(keep))
    if len(keep) > 1:
        ofi[1:] = ((b[1:] >= b[:-1]) * qb[1:] - (b[1:] <= b[:-1]) * qb[:-1]
                   - (a[1:] <= a[:-1]) * qa[1:] + (a[1:] >= a[:-1]) * qa[:-1])
    L = asz.shape[1]
    out.update({
        "t": t[keep], "mid": mid, "r": r, "rel_spread": (a - b) / (0.5 * (a + b)),
        "bq1": qb, "aq1": qa, "ofi": ofi,
        "bdep3": bsz[keep, :min(3, L)].sum(1), "adep3": asz[keep, :min(3, L)].sum(1),
        "bdep5": bsz[keep, :min(5, L)].sum(1), "adep5": asz[keep, :min(5, L)].sum(1),
        "bdep10": bsz[keep, :L].sum(1), "adep10": asz[keep, :L].sum(1),
    })
    meta = {"n_messages": int(len(t)), "n_window": int(window.sum()), "n_kept": int(len(keep)),
            "n_invalid": int((~valid & window).sum())}
    return out, meta


def save_day(path, series, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: (series[k].astype(np.float32) if k in BASE_FLOAT32 else series[k].astype(np.float64))
              for k in series}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(meta))


def process_day(job):
    """job: dict(archive, message, orderbook, levels, out, trim_open_s, trim_close_s).
    Returns a status dict; never raises (errors are reported)."""
    out = Path(job["out"])
    if out.exists():
        return {"out": str(out), "status": "exists"}
    try:
        raw = read_day(job["archive"], job["message"], job["orderbook"], job.get("levels", 10))
        series, meta = base_series(raw, job.get("trim_open_s", 300.0), job.get("trim_close_s", 300.0))
        save_day(out, series, meta)
        return {"out": str(out), "status": "ok", **meta}
    except Exception as err:                    # noqa: BLE001 - report, keep the pool alive
        return {"out": str(out), "status": "error", "error": f"{type(err).__name__}: {err}"}


# ---------------------------------------------------------------- event-level units
@dataclass
class EventConfig:
    root: str = "data/lobster_events"    # processed files: <root>/<TICKER>/<YYYY-MM-DD>.npz
    ticker: str = "AAPL"
    month: str = "2020-03"
    h: int = 500                          # label delay in kept events; RV window is h - gap events
    gap: int = 1
    pre_days: int = 5
    floor_frac: float = 0.01


RV_WINDOWS = (100, 500, 5_000, 50_000, 500_000)
HAR_WINDOWS = (500, 5_000, 50_000, 500_000)        # ~ horizon, x10, x100, x1000 events


def unit_days(cfg: EventConfig):
    d = Path(cfg.root) / cfg.ticker
    return sorted(p for p in d.glob(f"{cfg.month}-*.npz") if not p.name.endswith(".tmp.npz"))


def _trailing_sum(x, w):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - w, 0)]


def event_features(df, rv_windows=RV_WINDOWS):
    """Past-only features at each kept event. df: concatenated base series with 'day'.

    Columns are computed one at a time and written into a preallocated float32 matrix,
    so a 40-million-event unit needs ~5 GB rather than ~10 GB.
    """
    n = len(df["t"])
    r = df["r"].astype(np.float64)
    day = df["day"]
    new_day = np.r_[True, day[1:] != day[:-1]]
    dt = np.where(new_day, 0.0, np.diff(df["t"], prepend=df["t"][0]))       # no overnight gap
    depth1 = df["bq1"].astype(np.float64) + df["aq1"]
    flow = df["buy_vol"].astype(np.float64) - df["sell_vol"]
    vol = df["buy_vol"].astype(np.float64) + df["sell_vol"]

    def imb(k):
        b = (df["bq1"] if k == 1 else df[f"bdep{k}"]).astype(np.float64)
        a = (df["aq1"] if k == 1 else df[f"adep{k}"]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (b - a) / (b + a)

    specs = []
    specs += [(f"lrv_{w}", lambda w=w: np.log(_trailing_sum(r * r, w) / w + 1e-16)) for w in rv_windows]
    specs += [(f"ret_{w}", lambda w=w: _trailing_sum(r, w)) for w in (10, 100, 1_000)]
    specs += [(f"ofi_{w}", lambda w=w: _trailing_sum(df["ofi"], w) / (_trailing_sum(depth1, w) / w + 1.0))
              for w in (10, 100, 1_000)]
    specs += [(f"tfi_{w}", lambda w=w: _trailing_sum(flow, w) / (_trailing_sum(vol, w) + 1.0)) for w in (100, 1_000)]
    specs += [(f"depth_imb_{k}", lambda k=k: imb(k)) for k in (1, 3, 5, 10)]
    specs += [("log_spread", lambda: np.log(df["rel_spread"].astype(np.float64) + 1e-8))]
    specs += [(f"log_secs_{w}", lambda w=w: np.log(_trailing_sum(dt, w) + 1e-6)) for w in (100, 1_000)]
    specs += [("log_msgs_100", lambda: np.log1p(_trailing_sum(df["n_msg"], 100))),
              ("log_execs_100", lambda: np.log1p(_trailing_sum(df["n_exec"], 100))),
              ("hidden_share_1000", lambda: _trailing_sum(df["hidden_vol"], 1_000) / (_trailing_sum(vol, 1_000) + 1.0)),
              ("hours_since_open", lambda: (df["t"] - OPEN_S) / 3600.0)]
    X = np.empty((n, len(specs)), np.float32)
    for k, (_, fn) in enumerate(specs):
        X[:, k] = fn()
    return X, [name for name, _ in specs]


def forward_rv(df, h, gap):
    """RV over kept events t+gap+1 .. t+h of the same day; NaN where that needs another day."""
    r2 = df["r"].astype(np.float64) ** 2
    n = len(r2)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    t = np.arange(n)
    hi = t + h + 1
    ok = hi <= n
    rv = np.full(n, np.nan)
    rv[ok] = c[hi[ok]] - c[t[ok] + gap + 1]
    same = np.zeros(n, bool)
    same[ok] = df["day"][t[ok] + h] == df["day"][t[ok]]
    rv[~same] = np.nan
    return rv


def load_event_unit(cfg: EventConfig):
    """One stock-month at event level, as the dict the pipeline expects."""
    files = unit_days(cfg)
    if len(files) < cfg.pre_days + 3:
        raise ValueError(f"{cfg.ticker} {cfg.month}: only {len(files)} processed days")
    parts = []
    for k, f in enumerate(files):
        with np.load(f) as z:
            part = {key: z[key] for key in z.files}
        part["day"] = np.full(len(part["t"]), k, np.int32)
        parts.append(part)
    df = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    X, names = event_features(df)
    rv = forward_rv(df, cfg.h, cfg.gap)
    keep = np.isfinite(rv)
    pre = (df["day"] < cfg.pre_days) & keep
    stream = (df["day"] >= cfg.pre_days) & keep
    pos = rv[pre][rv[pre] > 0]
    floor = cfg.floor_frac * (np.median(pos) if len(pos) else 1e-12)
    y = np.log(rv + floor)
    dates = np.array([f.stem for f in files])[df["day"]]
    s_dates = dates[stream]
    seg_starts = np.flatnonzero(np.r_[True, s_dates[1:] != s_dates[:-1]])
    n = int(stream.sum())
    return {
        "X_pre": X[pre], "y_pre": y[pre], "X": X[stream], "y_obs": y[stream], "y_clean": y[stream],
        "signal": np.full(n, np.nan), "noise_var": np.full(n, np.nan), "regime": s_dates,
        "seg_starts": seg_starts, "is_bad": np.zeros(n, bool), "h": cfg.h, "feature_names": names,
        "har_cols": [names.index(f"lrv_{w}") for w in HAR_WINDOWS], "dates": s_dates, "t": df["t"][stream],
        "unit": f"{cfg.ticker}-{cfg.month}", "floor": floor,
    }
