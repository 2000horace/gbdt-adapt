# Phase 1: synthetic benchmark (Gate G1)

Fast layer of the adaptation-layer proposal (v2), tested on a synthetic
regime-switching stream against the Gate G1 baselines.

## Layout

    phase1/dgp.py        synthetic data (N1): AR(1) features, regimes A -> B -> A -> C,
                         "disjoint" and "shared" variants, GARCH + intraday noise,
                         overlapping h-step labels, bad ticks
    phase1/ensemble.py   XGBoost pre-training and the leaf-indicator map (+ bias leaf)
    phase1/adaptive.py   robust, lazily discounted leaf filter, noise-scale tracker,
                         calibration factor, bounded-statistic drift detector
    phase1/baselines.py  frozen, online SGD on leaves, XGBoost refresh, rolling retrain,
                         and predictive-scale wrappers (EWMA Gaussian, robust Student-t)
    phase1/metrics.py    log score, CRPS (Student-t / Gaussian), PIT, coverage,
                         block bootstrap
    run_phase1.py        runs everything, scores on clean labels, checks Gate G1, plots
    tests/test_core.py   leaf map, CRPS, lazy discount, Prop. 1 and Prop. 4 checks

## Run

    pip install xgboost numpy scipy pandas matplotlib pytest
    python -m pytest -q tests
    python run_phase1.py --variants disjoint shared --seeds 0 1 2 --tune --ablations --out results

    # short, recurring regimes with heavy tails and 1% bad ticks
    python run_phase1.py --variants disjoint shared disjoint_drift --seeds 0 \
        --seg-len 5000 --regimes ABACBCABACBCABAC --bad-tick-rate 0.01 --innov-df 3 \
        --tune --ablations --out results_short

Variants: "disjoint" (each regime in its own x0 band: covariate shift only, no concept
drift), "shared" (same x0 distribution: concept drift in place), "disjoint_drift"
(own bands, and each regime's change flips sign on every revisit: a region that is
left and later revisited has changed in the meantime).

`--tune` picks hyperparameters for every method on a separate seed (seed + 1000).
Baselines get the same robust Student-t scale wrapper as the method, so the
comparison isolates the effect of adapting the leaf values. `frozen+ewma` is the
naive practitioner baseline (Gaussian, EWMA variance).

## Caching

Each run of `run_phase1.py` stores its predictions in `<out>/<variant>_seed<k>/cache.npz`
(with `cache_meta.json`). Scores, tables and plots are always rebuilt from the cached
predictions, so changing `metrics.py` or any plotting code never requires a rerun.

    python run_phase1.py <same args>             # reuses valid caches, redoes scores and plots
    python run_phase1.py <same args> --replot    # never recomputes; warns if a cache is stale
    python run_phase1.py <same args> --no-cache  # forces a full recomputation

A cache is valid when its key matches. The key hashes the run settings (data, ensemble,
tuning and ablation flags, default hyperparameters), the source of `dgp.py`,
`ensemble.py`, `adaptive.py` and `baselines.py`, and the source of `prepare`, `run_all`,
`run_adaptive` and `tune`. Editing any of these triggers a recomputation on the next
normal run. A full-length run's cache is about 25 MB.

The JAX runner (`run_seeds_jax.py`) writes per-seed results to CSV, which is its cache.

## Outputs

    results/summary.csv     per method: log score, CRPS, 80% coverage, signal MSE, seconds
    results/segments.csv    per regime segment, including early-segment signal MSE
    results/comparisons.csv paired gains of "adaptive" over each baseline in log score, RMSE and
                            MAE (vs the clean label), with block-bootstrap CIs; positive = adaptive better
    results/<run>/          rolling signal MSE plot, influence plot, PIT histograms, config

## Deviations from proposal v2 (to fold back into the text)

* The update uses the current leaf means for the residual, not the stored prediction.
* The noise scale uses the Student-t EM fixed point and does not subtract the summed
  leaf variance: the diagonal sum overstates the variance of the prediction, and
  subtracting it collapsed the scale and destabilised the filter.
* Lazy discounting uses a cumulative discount clock, exact when delta changes in alarms.
* The detector uses u = 1 - min(omega, 1), since omega can exceed 1.

## Many seeds on a GPU (JAX)

    pip install -U "jax[cuda12]" xgboost numpy scipy pandas
    python -m pytest -q tests            # includes JAX-vs-NumPy equivalence tests (float64)

    XLA_PYTHON_CLIENT_PREALLOCATE=false python run_seeds_jax.py \
        --variant disjoint_drift --n-seeds 100 --tune-seeds 10 \
        --batch 25 --workers 16 --xgb --tune-xgb --ablations --out results_jax_dd

What runs where:

* `phase1/jax_filters.py` (GPU): the adaptive filter, SGD on leaves and the predictive-scale
  wrappers, each a `lax.scan` over events, vmapped over seeds and over hyperparameter grids.
  Ablation switches are numbers, so all ablations run in the same vmapped call.
* `phase1/prepare.py` (CPU worker processes): data generation, XGBoost pre-training, leaf
  indices, and the XGBoost baselines (refresh, rolling). `--xgb-device cuda` trains XGBoost
  on the GPU instead. Without `--xgb` these two baselines are skipped (much faster).
* Tuning uses separate seeds (seed_start + 100000 + k): one vmapped call scores 25
  (P0, delta) configs on all tuning seeds; another scores the SGD step sizes.
  `--tune-xgb` also tunes the refresh / rolling windows (CPU, slower).

Outputs in `--out`:

    per_seed.csv   every metric for every method and seed (plus alarm counts)
    means.csv      means over seeds
    gains.csv      gain of "adaptive" over each method: mean, 95% t-interval over seeds,
                   and share of seeds won, for log score, CRPS, RMSE, MAE, signal RMSE/MAE
    timing.csv     per batch: preparation, JAX adaptive, JAX SGD, XGBoost baseline seconds
    config.json    arguments and tuned hyperparameters

Notes:

* float32 (default) matches float64 closely on full streams (log score within ~1e-5,
  identical alarm times); use `--x64` to be safe.
* The leaf dimension is padded to n_trees * 2**max_depth + 1, so `jit` compiles once.
* GPU memory per batch is roughly batch x configs x T x 12 bytes for the outputs plus
  batch x T x (K + 1) x 4 bytes for leaf indices (~0.8 GB for 25 seeds x 8 configs).
  Lower `--batch` if needed.
* The scan is sequential in time, so one batch takes about the same time whether it holds
  1 or 100 filters; large batches are what make the GPU worthwhile. Expect XGBoost
  preparation (with `--xgb`) to dominate the total run time; scale it with `--workers`.
