"""Event-level experiment on preprocessed raw LOBSTER data: one job per stock-month.

Per unit (days are trading days of the month, in order):

    days 1 .. pre_days                  pre-train an ensemble for tuning
    days pre_days+1 .. pre_days+tune    tune the adaptive filter and SGD on this stream
    days 1 .. pre_days+tune             pre-train the final ensemble
    the remaining days                  evaluate every method prequentially

No step looks at evaluation days before predicting them. Batch baselines use windows and
refit intervals in trading time (1 day = the mean number of events per day on the
pre-evaluation days; 1 hour = 1 day / 6.33, as days run 09:35-15:55). Everything runs on
CPU; the adaptive filter and SGD use the compiled implementation and are processed in
chunks, so the full leaf-index matrix is never held in memory.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import stats

from . import fast_cpu as F
from . import lobster_raw as R
from . import metrics as M
from .adaptive import AdaptiveConfig
from .ensemble import EnsembleConfig, LeafMap, train_ensemble

HOURS_PER_DAY = 6 + 20 / 60          # 09:35 to 15:55


@dataclass
class EventRunConfig:
    root: str = "data/lobster_events"
    ticker: str = "AAPL"
    month: str = "2020-03"
    h: int = 10_000                      # label delay in events; RV over events t+2 .. t+h
    gap: int = 1
    pre_days: int = 3
    tune_days: int = 2
    floor_frac: float = 0.01
    n_trees: int = 100
    max_depth: int = 4
    xgb_threads: int = 1
    chunk: int = 1_000_000
    p0_grid: tuple = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    delta_grid: tuple = (0.99999, 0.999999, 0.9999999, 1.0)
    overlap_grid: tuple = ("1", "sqrt_h", "h", "2h")
    p_max_factor: float = 16.0
    sgd_grid: tuple = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    forecast_scale: bool = True              # noise scale / calibration from forecast-time residuals
    target: str = "logrv"                    # "logrv": log realized variance; "return": mid-price return
    # ensemble tuning by Gaussian-process Bayesian optimisation (bayes_opt) on the tuning days
    ens_tune: bool = True
    ens_trials: int = 30
    ens_seed: int = 0
    lr_range: tuple = (0.005, 0.3)           # learning rate, continuous, log scale
    mcw_range: tuple = (5, 5_000)            # minimum samples per leaf (on the trial subsample), integer, log scale
    trees_range: tuple = (50, 500)           # number of trees, integer, log scale
    tune_rows: int = 1_000_000               # trials train on a random subsample of at most this many events
    tune_threads: int = 4                    # XGBoost threads for ensemble training (one-off, not part of per-event cost)
    min_child_weight: float = 20.0           # used when ens_tune is False
    learning_rate: float = 0.1               # used when ens_tune is False
    scale_memory_grid: tuple = (1, 5, 20)    # noise-scale memory in multiples of h (shared by all methods)
    calib_rate: float = 0.05                 # calibration step eta_c = calib_rate / h (0.005 at h = 10)
    batch_cache: str = ""                    # folder for cached batch predictions ("" = no cache)
    batch: tuple = (("1d", "1h"), ("1d", "1d"), ("5d", "1d"))       # (window, refit every)
    har: tuple = ("5d", "1h")
    skip_batch: bool = False
    nu: float = 4.0


def _overlap(name, h):
    return {"1": 1.0, "sqrt_h": float(np.sqrt(h)), "h": float(h), "2h": 2.0 * h}[name]


def rv_windows(h):
    return tuple(sorted({100, 1_000, h, 10 * h, 100 * h}))


def har_windows(h):
    return (h, 10 * h, 100 * h)                # ~ the horizon, x10, x100


def _events(spec, per_day):
    """'5d' / '1h' / plain integer -> number of events."""
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    n, unit = float(spec[:-1]), spec[-1]
    return max(1, int(round(n * per_day if unit == "d" else n * per_day / HOURS_PER_DAY)))


def _log(cfg, msg):
    print(f"[{time.strftime('%H:%M:%S')}]   {cfg.ticker}-{cfg.month}: {msg}", flush=True)


# ---------------------------------------------------------------- data
LINEAR_FEATURES = ("ofi_10", "ofi_100", "ofi_1000", "tfi_100", "tfi_1000", "depth_imb_1", "depth_imb_3",
                   "depth_imb_5", "depth_imb_10", "ret_10", "ret_100", "ret_1000")


def forward_return(df, h, gap):
    """Mid-price log return over kept events t+gap+1 .. t+h of the same day; NaN otherwise."""
    r = df["r"].astype(np.float64)
    n = len(r)
    c = np.concatenate([[0.0], np.cumsum(r)])
    t = np.arange(n)
    hi = t + h + 1
    ok = hi <= n
    out = np.full(n, np.nan)
    out[ok] = c[hi[ok]] - c[t[ok] + gap + 1]
    same = np.zeros(n, bool)
    same[ok] = df["day"][t[ok] + h] == df["day"][t[ok]]
    out[~same] = np.nan
    return out


def build_arrays(cfg: EventRunConfig):
    files = R.unit_days(R.EventConfig(root=cfg.root, ticker=cfg.ticker, month=cfg.month))
    n_early = cfg.pre_days + cfg.tune_days
    if len(files) < n_early + 2:
        raise ValueError(f"{cfg.ticker} {cfg.month}: only {len(files)} processed days")
    parts = []
    for k, f in enumerate(files):
        with np.load(f) as z:
            part = {key: z[key] for key in z.files}
        part["day"] = np.full(len(part["t"]), k, np.int32)
        parts.append(part)
    df = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    del parts
    X, names = R.event_features(df, rv_windows(cfg.h))
    day = df["day"]
    if cfg.target == "return":
        # mid-price return over the horizon, in units of its standard deviation on the
        # pre-evaluation days (no look-ahead); 'rv' holds the raw return for diagnostics
        rv = forward_return(df, cfg.h, cfg.gap)
        early = (day < n_early) & np.isfinite(rv)
        floor = float(np.std(rv[early]))
        y = rv / floor
    else:
        rv = R.forward_rv(df, cfg.h, cfg.gap)
        early = (day < n_early) & np.isfinite(rv)
        pos = rv[early][rv[early] > 0]
        floor = cfg.floor_frac * (np.median(pos) if len(pos) else 1e-12)
        y = np.log(rv + floor)
    per_day = np.bincount(day)[:n_early].mean()        # events per day, pre-evaluation days only
    dates = [f.stem for f in files]
    return dict(X=X, y=y, rv=rv, day=day, t=df["t"], names=names, dates=dates, per_day=float(per_day),
                floor=floor, har_cols=[names.index(f"lrv_{w}") for w in har_windows(cfg.h)],
                lin_cols=[names.index(f) for f in LINEAR_FEATURES],
                files=[(f.name, f.stat().st_size) for f in files])


def _leaf_idx(lm, X, chunk=500_000):
    out = np.empty((len(X), lm.n_trees + 1), np.int16 if lm.size < 2 ** 15 else np.int32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = lm.leaf_index(X[a:a + chunk])
    return out


def _ensemble(cfg, X, y, ens=None, nthread=None):
    """Train the boosted ensemble. ens: dict(learning_rate, min_child_weight, n_trees);
    defaults to the fixed settings in cfg."""
    ens = ens or dict(learning_rate=cfg.learning_rate, min_child_weight=cfg.min_child_weight, n_trees=cfg.n_trees)
    base = EnsembleConfig(n_trees=int(ens["n_trees"]), max_depth=cfg.max_depth, learning_rate=ens["learning_rate"],
                          min_child_weight=ens["min_child_weight"])
    params = dict(base.params(), nthread=nthread or cfg.xgb_threads)
    booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=base.n_trees)
    lm = LeafMap(booster)
    lm.calibrate_bias(X[: min(len(X), 200_000)])
    return booster, lm, params


def _mean_t_logscore(y, mu, s, nu):
    return float(stats.t.logpdf(y, nu, mu, np.sqrt(s)).mean())


# ---------------------------------------------------------------- tuning on days 4-5
def tune_ensemble(cfg, X_tr, y_tr, X_tu, y_tu):
    """Gaussian-process Bayesian optimisation (package `bayesian-optimization`, `bayes_opt`)
    of the ensemble's learning rate, minimum samples per leaf and number of trees, by the
    frozen ensemble's log score on the tuning days.

    All three are searched on a log10 scale; the two integer parameters are rounded inside
    the objective. cfg.ens_trials evaluations in total: 5 random starting points, then
    guided steps; the Gaussian process includes a small noise term (the score is estimated
    on two days). Trials train on a random subsample of at most cfg.tune_rows events; the
    minimum samples per leaf is returned per training row (mcw_per_row), so that it can be
    scaled to the size of any later training set."""
    if not cfg.ens_tune:
        ens = dict(learning_rate=cfg.learning_rate, min_child_weight=cfg.min_child_weight, n_trees=cfg.n_trees)
        return dict(ens, mcw_per_row=cfg.min_child_weight / len(y_tr)), {}
    from bayes_opt import BayesianOptimization
    rng = np.random.default_rng(cfg.ens_seed)
    n_sub = min(len(y_tr), cfg.tune_rows)
    sub = np.sort(rng.choice(len(y_tr), n_sub, replace=False)) if n_sub < len(y_tr) else np.arange(len(y_tr))
    d_tr = xgb.DMatrix(X_tr[sub], label=y_tr[sub])
    d_tu = xgb.DMatrix(X_tu)
    sk = scale_kwargs(cfg, 5 * cfg.h)

    def decode(log_lr, log_mcw, log_trees):
        return dict(learning_rate=float(10 ** log_lr), min_child_weight=int(round(10 ** log_mcw)),
                    n_trees=int(round(10 ** log_trees)))

    def objective(log_lr, log_mcw, log_trees):
        e = decode(log_lr, log_mcw, log_trees)
        params = dict(EnsembleConfig(n_trees=e["n_trees"], max_depth=cfg.max_depth, learning_rate=e["learning_rate"],
                                     min_child_weight=e["min_child_weight"]).params(), nthread=cfg.tune_threads)
        mu = xgb.train(params, d_tr, num_boost_round=e["n_trees"]).predict(d_tu).astype(np.float64)
        return _mean_t_logscore(y_tu, mu, F.robust_scale(mu, y_tu, cfg.h, **sk), cfg.nu)

    bounds = {"log_lr": tuple(np.log10(cfg.lr_range)), "log_mcw": tuple(np.log10(cfg.mcw_range)),
              "log_trees": tuple(np.log10(cfg.trees_range))}
    opt = BayesianOptimization(f=objective, pbounds=bounds, random_state=cfg.ens_seed, verbose=0,
                               allow_duplicate_points=True)
    opt.set_gp_params(alpha=1e-4)                     # observation noise of the validation score
    n_init = min(5, cfg.ens_trials)
    opt.maximize(init_points=n_init, n_iter=cfg.ens_trials - n_init)
    bp = decode(**opt.max["params"])
    per_row = bp["min_child_weight"] / n_sub
    ens = dict(learning_rate=bp["learning_rate"], n_trees=bp["n_trees"], mcw_per_row=per_row,
               min_child_weight=per_row * len(y_tr))
    edges = [k for k, v, (lo, hi) in (("learning_rate", bp["learning_rate"], cfg.lr_range),
                                       ("min_child_weight", bp["min_child_weight"], cfg.mcw_range),
                                       ("n_trees", bp["n_trees"], cfg.trees_range))
             if v <= lo * 1.05 or v >= hi / 1.05]
    return ens, dict(ens_tune_score=float(opt.max["target"]), ens_trials=len(opt.res),
                     ens_edges=",".join(edges), ens_trial_rows=n_sub)


def scale_kwargs(cfg, memory):
    """Predictive-scale settings on the event clock: memory in events, calibration step ~ 1/h."""
    return dict(nu=cfg.nu, delta_v=1.0 - 1.0 / memory, eta_c=cfg.calib_rate / cfg.h)


def tune(cfg, A):
    day, valid = A["day"], np.isfinite(A["y"])
    tr = (day < cfg.pre_days) & valid
    tu = (day >= cfg.pre_days) & (day < cfg.pre_days + cfg.tune_days) & valid
    y = A["y"][tu].astype(np.float64)
    ens, ens_info = tune_ensemble(cfg, A["X"][tr], A["y"][tr], A["X"][tu], y)
    booster, lm, _ = _ensemble(cfg, A["X"][tr], A["y"][tr], ens, nthread=cfg.tune_threads)

    # 1) noise-scale memory, shared by every method, chosen on the frozen ensemble's forecasts
    mu_frozen = booster.predict(xgb.DMatrix(A["X"][tu]))
    mem_scores = {}
    for k in cfg.scale_memory_grid:
        sk = scale_kwargs(cfg, k * cfg.h)
        mem_scores[k] = _mean_t_logscore(y, mu_frozen, F.robust_scale(mu_frozen, y, cfg.h, **sk), cfg.nu)
    mem = max(mem_scores, key=mem_scores.get) * cfg.h
    sk = scale_kwargs(cfg, mem)

    # 2) adaptive filter grid
    idx = _leaf_idx(lm, A["X"][tu])
    best, best_score = None, -np.inf
    for p0 in cfg.p0_grid:
        for d in cfg.delta_grid:
            for o in cfg.overlap_grid:
                c = AdaptiveConfig(P0=p0, P0_bias=10 * p0, delta=d, P_max_factor=cfg.p_max_factor,
                                   overlap_factor=_overlap(o, cfg.h), nu=cfg.nu, delta_v=sk["delta_v"],
                                   eta_c=sk["eta_c"], use_detector=False)
                mu, s, _ = F.OnlineFilter(lm.w0, lm.bias_index, c, cfg.h, cfg.forecast_scale).run(idx, y)
                sc = _mean_t_logscore(y, mu, s, cfg.nu)
                if sc > best_score:
                    best, best_score = c, sc

    # 3) SGD step
    eta_best, eta_score = None, -np.inf
    for eta in cfg.sgd_grid:
        mu = F.run_sgd(lm.w0, idx, y, cfg.h, eta)
        sc = _mean_t_logscore(y, mu, F.robust_scale(mu, y, cfg.h, **sk), cfg.nu)
        if sc > eta_score:
            eta_best, eta_score = eta, sc
    edges = [n for n, v, g in (("P0", best.P0, cfg.p0_grid), ("delta", best.delta, cfg.delta_grid),
                               ("sgd_eta0", eta_best, cfg.sgd_grid)) if v in (g[0], g[-1]) and not (n == "delta" and v == 1.0)]
    return best, eta_best, mem, dict(adaptive_tune_score=best_score, sgd_tune_score=eta_score,
                                     scale_memory=mem, n_tune_events=int(tu.sum()), grid_edges=",".join(edges),
                                     ens=ens, n_train_rows_tuning=int(tr.sum()), **ens_info)


# ---------------------------------------------------------------- batch baselines (memory-light)
def batch_predict(kind, booster, params, Xp, yp, Xs, ys, h, window, every, n_trees):
    """Refresh or full retrain every `every` stream events on the last `window` observable
    labels (pre-training rows first, then stream rows). Returns predictions and seconds."""
    T, n_pre = len(ys), len(yp)
    mu = np.empty(T)
    t0 = time.perf_counter()
    upd = dict(params, process_type="update", updater="refresh", refresh_leaf=True)
    for a in range(0, T, every):
        b = min(a + every, T)
        hi = n_pre + max(0, a - h + 1)                 # labels observable at stream index a
        lo = max(0, hi - window)
        Xw = np.concatenate([Xp[lo:min(hi, n_pre)], Xs[max(0, lo - n_pre):max(0, hi - n_pre)]])
        yw = np.concatenate([yp[lo:min(hi, n_pre)], ys[max(0, lo - n_pre):max(0, hi - n_pre)]])
        dm = xgb.DMatrix(Xw, label=yw)
        if kind == "rolling":
            model = xgb.train(params, dm, num_boost_round=n_trees)
        else:
            model = xgb.train(upd, dm, num_boost_round=booster.num_boosted_rounds(), xgb_model=booster.copy())
        mu[a:b] = model.predict(xgb.DMatrix(Xs[a:b]))
    return mu, time.perf_counter() - t0


def batch_key(cfg, A, kind, W, E, ens=None):
    """Everything a batch prediction depends on (not the filter or scale settings)."""
    spec = dict(v=1, kind=kind, W=W, E=E, root=str(Path(cfg.root).resolve()), ticker=cfg.ticker, month=cfg.month,
                h=cfg.h, gap=cfg.gap, pre_days=cfg.pre_days, tune_days=cfg.tune_days, floor_frac=cfg.floor_frac,
                n_trees=cfg.n_trees, max_depth=cfg.max_depth, files=A["files"], rv_windows=rv_windows(cfg.h),
                ens={k: float(v) for k, v in (ens or {}).items()})
    hsh = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for mod in ("events.py", "lobster_raw.py", "ensemble.py"):
        src = (Path(__file__).parent / mod).read_text()
        if mod == "events.py":                      # only the batch-relevant functions
            src = src[src.index("def batch_predict"):src.index("def har_predict")] + src[src.index("def _ensemble"):src.index("def _mean_t_logscore")]
        hsh.update(src.encode())
    return hsh.hexdigest()[:16]


def har_predict(Xp, yp, Xs, ys, cols, h, window, every):
    t0 = time.perf_counter()
    mu = R_har(Xp[:, cols].astype(np.float64), yp, Xs[:, cols].astype(np.float64), ys, h, window, every)
    return mu, time.perf_counter() - t0


def R_har(Zp, yp, Zs, ys, h, window, every, ridge=1e-6):
    T, n_pre = len(ys), len(yp)
    mu = np.empty(T)
    for a in range(0, T, every):
        b = min(a + every, T)
        hi = n_pre + max(0, a - h + 1)
        lo = max(0, hi - window)
        Z = np.concatenate([Zp[lo:min(hi, n_pre)], Zs[max(0, lo - n_pre):max(0, hi - n_pre)]])
        yy = np.concatenate([yp[lo:min(hi, n_pre)], ys[max(0, lo - n_pre):max(0, hi - n_pre)]])
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        ok = np.isfinite(Z1).all(axis=1) & np.isfinite(yy)
        beta = np.linalg.solve(Z1[ok].T @ Z1[ok] + ridge * np.eye(Z1.shape[1]), Z1[ok].T @ yy[ok])
        mu[a:b] = np.column_stack([np.ones(b - a), np.nan_to_num(Zs[a:b])]) @ beta
    return mu


# ---------------------------------------------------------------- one unit
def run_unit(cfg: EventRunConfig):
    t_start = time.perf_counter()
    A = build_arrays(cfg)
    n_early = cfg.pre_days + cfg.tune_days
    _log(cfg, f"{len(A['y']):,} events over {len(A['dates'])} days ({A['per_day']:,.0f} per pre-evaluation day)")

    best, eta, mem, tune_info = tune(cfg, A)
    best_ens = tune_info.pop("ens")
    sk = scale_kwargs(cfg, mem)
    _log(cfg, f"tuned: scale memory {mem:,} events, P0={best.P0:g}, delta={best.delta}, "
              f"overlap={best.overlap_factor:g}, SGD eta0={eta}"
              + (f" (at grid edge: {tune_info['grid_edges']})" if tune_info["grid_edges"] else "")
              + f"; ensemble lr={best_ens['learning_rate']:.4g}, trees={best_ens['n_trees']}, "
                f"min samples/leaf={best_ens['mcw_per_row'] * 1e6:.1f} per million rows"
              + (f" (at range edge: {tune_info['ens_edges']})" if tune_info.get("ens_edges") else ""))

    valid = np.isfinite(A["y"])
    pre = (A["day"] < n_early) & valid
    ev = (A["day"] >= n_early) & valid
    Xp, yp = A["X"][pre], A["y"][pre].astype(np.float64)
    Xs, ys = A["X"][ev], A["y"][ev].astype(np.float64)
    t_ev = A["t"][ev]
    floor_share = float(np.mean(A["rv"][ev] == 0.0))
    del A["X"]
    T = len(ys)
    seconds, preds = {}, {}

    t0 = time.perf_counter()
    ens_final = dict(best_ens, min_child_weight=best_ens["mcw_per_row"] * len(yp))
    booster, lm, params = _ensemble(cfg, Xp, yp, ens_final, nthread=cfg.tune_threads)
    params = dict(params, nthread=cfg.xgb_threads)          # batch refits: single-threaded, as timed before
    seconds["pretrain"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    frozen = np.empty(T)
    for a in range(0, T, cfg.chunk):
        frozen[a:a + cfg.chunk] = booster.predict(xgb.DMatrix(Xs[a:a + cfg.chunk]))
    seconds["inference_batched"] = time.perf_counter() - t0
    preds["frozen"] = frozen

    # adaptive filter and SGD, chunked; leaf indices computed per chunk
    t_idx = [0.0]

    def make_idx(a, b):
        t1 = time.perf_counter()
        out = _leaf_idx(lm, Xs[a:b])
        t_idx[0] += time.perf_counter() - t1
        return out

    t0 = time.perf_counter()
    filt = F.OnlineFilter(lm.w0, lm.bias_index, best, cfg.h, cfg.forecast_scale)
    out = F.run_stream(make_idx, ys, cfg.h, cfg.chunk, filt=filt, sgd=(lm.w0, eta))
    seconds["adaptive+sgd_updates"] = time.perf_counter() - t0 - t_idx[0]
    seconds["leaf_indices"] = t_idx[0]
    preds["sgd"] = out["mu_sgd"]

    per_day = A["per_day"]
    hw, he = _events(cfg.har[0], per_day), _events(cfg.har[1], per_day)
    if cfg.target == "return":
        preds["zero"], seconds["zero"] = np.zeros(T), 0.0
        preds["linear"], seconds["linear"] = har_predict(Xp, yp, Xs, ys, A["lin_cols"], cfg.h, hw, he)
    else:
        preds["har"], seconds["har"] = har_predict(Xp, yp, Xs, ys, A["har_cols"], cfg.h, hw, he)

    if not cfg.skip_batch:
        for kind in ("refresh", "rolling"):
            for w_spec, e_spec in cfg.batch:
                W, E = _events(w_spec, per_day), _events(e_spec, per_day)
                name = f"{kind}[{w_spec}/{e_spec}]"
                cache = None
                if cfg.batch_cache:
                    key = batch_key(cfg, A, kind, W, E, ens_final)
                    croot = Path(cfg.batch_cache) if cfg.target == "logrv" else Path(cfg.batch_cache) / f"target_{cfg.target}"
                    cache = croot / f"{cfg.ticker}-{cfg.month}" / f"{kind}_W{W}_E{E}_{key}"
                if cache is not None and cache.with_suffix(".npy").exists():
                    preds[name] = np.load(cache.with_suffix(".npy")).astype(np.float64)
                    seconds[name] = json.loads(cache.with_suffix(".json").read_text())["seconds"]
                    _log(cfg, f"{name}: loaded from batch cache ({seconds[name]:.0f}s when computed)")
                    continue
                preds[name], seconds[name] = batch_predict(kind, booster, params, Xp, yp, Xs, ys, cfg.h, W, E,
                                                           int(ens_final["n_trees"]))
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache.with_suffix(".tmp.npy"), preds[name].astype(np.float32))
                    cache.with_suffix(".tmp.npy").replace(cache.with_suffix(".npy"))
                    cache.with_suffix(".json").write_text(json.dumps(dict(seconds=seconds[name], window=W, every=E)))
                _log(cfg, f"{name}: window {W:,} events, refit every {E:,} events, {seconds[name]:.0f}s")

    # scores
    rows = []

    mse0 = float(np.mean(ys ** 2))
    nz = ys != 0

    def score(name, mu, s, df):
        mse = float(np.mean((ys - mu) ** 2))
        corr = float(np.corrcoef(mu, ys)[0, 1]) if np.std(mu) > 0 else 0.0
        rows.append(dict(method=name, log_score=float(M.log_score(ys, mu, s, df).mean()),
                         crps=float(M.crps(ys, mu, s, df).mean()),
                         coverage80=float(M.coverage(ys, mu, s, df, 0.2)),
                         rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(ys - mu))),
                         corr=corr, dir_acc=float(np.mean(np.sign(mu[nz]) == np.sign(ys[nz]))),
                         r2_oos=1.0 - mse / mse0))

    score("adaptive", out["mu"], out["s"], cfg.nu)
    for name, mu in preds.items():
        score(f"{name}+robust", mu, F.robust_scale(mu, ys, cfg.h, **sk), cfg.nu)

    secs_per_event = np.diff(t_ev)
    meta = dict(unit=f"{cfg.ticker}-{cfg.month}", ticker=cfg.ticker, month=cfg.month, n_eval_events=T, target=cfg.target,
                n_pre_events=int(len(yp)), events_per_day=per_day, n_eval_days=len(A["dates"]) - n_early,
                horizon_median_seconds=float(np.median(secs_per_event[secs_per_event >= 0]) * cfg.h),
                n_alarms=int(out["fire"].sum()), adaptive=asdict(best), sgd_eta0=eta, **tune_info,
                ens_learning_rate=best_ens["learning_rate"], ens_n_trees=int(best_ens["n_trees"]),
                ens_min_child_weight_final=float(ens_final["min_child_weight"]),
                ens_mcw_per_million=float(best_ens["mcw_per_row"] * 1e6),
                floor_share=floor_share,
                har_window=hw, har_every=he, total_seconds=time.perf_counter() - t_start)
    return rows, seconds, meta
