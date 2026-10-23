"""Cache of per-run results, so tables and plots can be regenerated without recomputing.

What is cached (per run directory):
    cache.npz        predictive (mu, s, df) of every method, the data needed for scoring
                     and plotting (clean labels, true signal, regimes, segment starts),
                     and the filter diagnostics (residuals, prediction changes)
    cache_meta.json  cache key, method names, timings, tuned hyperparameters, alarms

Scores are always recomputed from the cached predictions, so changes to metrics.py or
to the plotting code never need a rerun.

The cache key hashes everything that determines the predictions: the run specification
(data, ensemble, tuning and ablation settings, default hyperparameters), the source of
the modules that generate data and run the methods, and the source of the given
functions (e.g. tune, run_all). Change any of these and the cache is invalidated.
"""

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
PACKAGE_DIR = Path(__file__).resolve().parent
CORE_MODULES = ("dgp.py", "ensemble.py", "adaptive.py", "baselines.py")
DATA_KEYS = ("y_clean", "signal", "regime", "seg_starts")


def cache_key(spec: dict, functions=()) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(dict(version=CACHE_VERSION, spec=spec), sort_keys=True, default=str).encode())
    for name in CORE_MODULES:
        h.update((PACKAGE_DIR / name).read_bytes())
    for fn in functions:
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def save(run_dir: Path, key: str, data: dict, results: dict, diagnostics: dict, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    names = list(results)
    arrays = {f"data_{k}": np.asarray(data[k]) for k in DATA_KEYS}
    for i, name in enumerate(names):
        for field in ("mu", "s", "df"):
            arrays[f"m{i}_{field}"] = np.asarray(results[name][field])
    for k, v in diagnostics.items():
        arrays[f"diag_{k}"] = np.asarray(v)
    tmp = run_dir / "cache.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(run_dir / "cache.npz")      # atomic: no half-written cache
    full_meta = dict(key=key, created=time.strftime("%Y-%m-%d %H:%M:%S"), h=int(data["h"]),
                     methods=names, seconds={n: results[n]["seconds"] for n in names},
                     diagnostics=list(diagnostics), **meta)
    (run_dir / "cache_meta.json").write_text(json.dumps(full_meta, indent=4, default=str))


def load(run_dir: Path):
    """Return (key, data, results, diagnostics, meta), or None if there is no cache."""
    meta_path, npz_path = run_dir / "cache_meta.json", run_dir / "cache.npz"
    if not (meta_path.exists() and npz_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    with np.load(npz_path) as z:
        data = {k: z[f"data_{k}"] for k in DATA_KEYS}
        data["h"] = meta["h"]
        results = {}
        for i, name in enumerate(meta["methods"]):
            results[name] = dict(mu=z[f"m{i}_mu"], s=z[f"m{i}_s"], df=z[f"m{i}_df"],
                                 seconds=meta["seconds"][name])
        diagnostics = {k: z[f"diag_{k}"] for k in meta["diagnostics"]}
    return meta["key"], data, results, diagnostics, meta
