#!/usr/bin/env bash
# Full Phase 1 run: 100 seeds x 3 variants, with XGBoost baselines, tuning and ablations.
#
# Usage (from the repo root, with the virtualenv active):
#     chmod +x run_full.sh
#     nohup ./run_full.sh > run_full.log 2>&1 &
#     tail -f run_full.log
#
# Override defaults with environment variables, e.g.
#     N_SEEDS=20 WORKERS=16 XGB_DEVICE=cuda ./run_full.sh
#
# Rerunning is safe and cheap: results go to a fixed folder (OUT, default runs/phase1_full)
# and run_seeds_jax.py caches tuning and every finished batch there. An interrupted run
# resumes; a finished run only rebuilds its summary tables; a larger N_SEEDS only computes
# the new seeds. Set EXTRA=--report-only to rebuild tables without computing anything,
# or EXTRA=--no-cache to force a full recomputation. EXTRA=--reuse-tuning keeps an earlier
# tuning result after code edits that don't change the numbers.

set -euo pipefail
cd "$(dirname "$0")"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1         # print progress immediately, also under nohup

CORES=$(nproc)
DEFAULT_WORKERS=$(( CORES > 33 ? 32 : (CORES > 1 ? CORES - 1 : 1) ))
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
N_SEEDS=${N_SEEDS:-100}
TUNE_SEEDS=${TUNE_SEEDS:-10}
BATCH=${BATCH:-25}
GPU_CHUNK=${GPU_CHUNK:-0}       # seeds per GPU call; 0 = whole batch, halves automatically on OOM
XGB_DEVICE=${XGB_DEVICE:-cpu}
VARIANTS=${VARIANTS:-"disjoint_drift shared disjoint"}
OUT=${OUT:-runs/phase1_full}
EXTRA=${EXTRA:-}               # e.g. --verbose, --report-only, --reuse-tuning

mkdir -p "$OUT"
{
    echo "started:    $(date)"
    echo "host:       $(hostname)"
    echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "settings:   N_SEEDS=$N_SEEDS TUNE_SEEDS=$TUNE_SEEDS BATCH=$BATCH GPU_CHUNK=$GPU_CHUNK WORKERS=$WORKERS XGB_DEVICE=$XGB_DEVICE"
    echo "variants:   $VARIANTS"
    python -c "import jax; print('jax devices:', jax.devices())"
    echo "extra args: ${EXTRA:-none}"
    echo
} | tee -a "$OUT/run_info.txt"

for v in $VARIANTS; do
    echo
    echo "=== $v: started $(date)"
    python run_seeds_jax.py \
        --variant "$v" \
        --n-seeds "$N_SEEDS" \
        --tune-seeds "$TUNE_SEEDS" \
        --batch "$BATCH" \
        --gpu-chunk "$GPU_CHUNK" \
        --workers "$WORKERS" \
        --xgb --tune-xgb --xgb-device "$XGB_DEVICE" \
        --ablations $EXTRA \
        --out "$OUT/$v" 2>&1 | tee -a "$OUT/$v.log"
    echo "=== $v: finished $(date)"
done

echo
echo "all done: $(date)" | tee -a "$OUT/run_info.txt"
echo "results in $OUT - one folder per variant with gains.csv, means.csv, per_seed.csv, timing.csv, config.json"
