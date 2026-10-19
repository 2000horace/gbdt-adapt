#!/usr/bin/env bash
# 3374662 - node 22
# Full Phase 1 run: 100 seeds x 3 variants, with XGBoost baselines, tuning and ablations.
#
# Usage (from the repo root, with the virtualenv active):
#     chmod +x run_full.sh
#     nohup ./run_full.sh > run_full.log 2>&1 &
#     tail -f run_full.log
#
# Override defaults with environment variables, e.g.
#     N_SEEDS=20 WORKERS=16 XGB_DEVICE=cuda ./run_full.sh

set -euo pipefail
cd "$(dirname "$0")"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

CORES=$(nproc)
DEFAULT_WORKERS=$(( CORES > 33 ? 32 : (CORES > 1 ? CORES - 1 : 1) ))
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
N_SEEDS=${N_SEEDS:-100}
TUNE_SEEDS=${TUNE_SEEDS:-10}
BATCH=${BATCH:-25}
XGB_DEVICE=${XGB_DEVICE:-cpu}
VARIANTS=${VARIANTS:-"disjoint_drift shared disjoint"}
OUT=${OUT:-runs/$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUT"
{
    echo "started:    $(date)"
    echo "host:       $(hostname)"
    echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "settings:   N_SEEDS=$N_SEEDS TUNE_SEEDS=$TUNE_SEEDS BATCH=$BATCH WORKERS=$WORKERS XGB_DEVICE=$XGB_DEVICE"
    echo "variants:   $VARIANTS"
    python -c "import jax; print('jax devices:', jax.devices())"
} | tee "$OUT/run_info.txt"

for v in $VARIANTS; do
    echo
    echo "=== $v: started $(date)"
    python run_seeds_jax.py \
        --variant "$v" \
        --n-seeds "$N_SEEDS" \
        --tune-seeds "$TUNE_SEEDS" \
        --batch "$BATCH" \
        --workers "$WORKERS" \
        --xgb --tune-xgb --xgb-device "$XGB_DEVICE" \
        --ablations \
        --out "$OUT/$v" 2>&1 | tee "$OUT/$v.log"
    echo "=== $v: finished $(date)"
done

echo
echo "all done: $(date)" | tee -a "$OUT/run_info.txt"
echo "results in $OUT/<variant>/ (gains.csv, means.csv, per_seed.csv, timing.csv, config.json)"
