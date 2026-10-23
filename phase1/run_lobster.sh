#!/usr/bin/env bash
# Phase 2: realized-volatility task on LOBSTER one-minute data, one run per year.
# One unit = one stock in one year: pre-train on its first PRE_DAYS trading days, then
# stream the rest of the year (about 70,000-80,000 minutes). Tuning uses separate stocks.
#
# Usage (from the repo root, with the virtualenv active):
#     chmod +x run_lobster.sh
#     nohup ./run_lobster.sh > run_lobster.log 2>&1 &
#
# Override defaults with environment variables, e.g.
#     YEARS="2019 2020" N_UNITS=30 ./run_lobster.sh
# Reruns reuse cached results (same caching as run_full.sh).

set -euo pipefail
cd "$(dirname "$0")"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

CORES=$(nproc)
DEFAULT_WORKERS=$(( CORES > 33 ? 32 : (CORES > 1 ? CORES - 1 : 1) ))
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
YEARS=${YEARS:-"2017 2018 2019 2020"}
N_UNITS=${N_UNITS:-50}           # evaluated stocks per year
TUNE_UNITS=${TUNE_UNITS:-10}     # separate stocks used for tuning
PRE_DAYS=${PRE_DAYS:-50}
BATCH=${BATCH:-25}
LOBSTER_ROOT=${LOBSTER_ROOT:-/nfs/data/lobster_data/lobster_minutely/3}
OUT=${OUT:-runs/lobster_vol}
BASELINE_CACHE=${BASELINE_CACHE:-runs/baseline_cache}
EXTRA=${EXTRA:-}                 # e.g. --verbose, --report-only

mkdir -p "$OUT"
{
    echo "started:    $(date)"
    echo "host:       $(hostname)"
    echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "settings:   YEARS=$YEARS N_UNITS=$N_UNITS TUNE_UNITS=$TUNE_UNITS PRE_DAYS=$PRE_DAYS BATCH=$BATCH WORKERS=$WORKERS"
    echo "data:       $LOBSTER_ROOT"
    echo "extra args: ${EXTRA:-none}"
    python -c "import jax; print('jax devices:', jax.devices())"
    echo
} | tee -a "$OUT/run_info.txt"

for y in $YEARS; do
    echo
    echo "=== $y: started $(date)"
    python run_seeds_jax.py \
        --source lobster \
        --lobster-root "$LOBSTER_ROOT" \
        --year "$y" \
        --n-seeds "$N_UNITS" \
        --tune-seeds "$TUNE_UNITS" \
        --pre-days "$PRE_DAYS" \
        --batch "$BATCH" \
        --workers "$WORKERS" \
        --xgb --tune-xgb \
        --ablations \
        --baseline-cache "$BASELINE_CACHE" $EXTRA \
        --out "$OUT/$y" 2>&1 | tee -a "$OUT/$y.log"
    echo "=== $y: finished $(date)"
done

echo
echo "all done: $(date)" | tee -a "$OUT/run_info.txt"
