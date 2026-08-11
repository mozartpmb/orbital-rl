#!/bin/bash
# T5 NB campaign — bearings-only training arms, one at a time.
#
# 8 workers x 256 envs saturates the machine (14 physical cores), so arms MUST
# run sequentially. Each arm: 50M steps on puffer_orbital_nav with
# nav_mode=bearings_only, chain-private data dir, wandb group t5-nav-<arm>.
#
# Usage: nb_campaign.sh <arm> [load_model_path]
#   arm            nb1_warm | nb1_fresh_42 | nb1_fresh_7 | nb1_fresh_1337
# The seed is derived from the arm name.
set -uo pipefail
WT=/Users/pete/space_training-extnav
PUF=$WT/pufferlib
ARM=$1
LOAD=${2:-}

case "$ARM" in
    nb1_warm)       SEED=42 ;;
    nb1_fresh_42)   SEED=42 ;;
    nb1_fresh_7)    SEED=7 ;;
    nb1_fresh_1337) SEED=1337 ;;
    *) echo "unknown arm $ARM"; exit 2 ;;
esac

EXTRA=""
if [ -n "$LOAD" ]; then EXTRA="--load-model-path $LOAD"; fi

cd "$PUF" || exit 2
export PYTHONPATH=$PUF
# NAV-H §2.2: OMP=8 is SLOWER than OMP=1 on (N,4,4) chains, and 8 workers x N
# BLAS threads would thrash 14 cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python3 -m pufferlib.pufferl train puffer_orbital_nav \
    --train.device cpu \
    --train.total-timesteps 50000000 \
    --train.seed "$SEED" \
    --env.nav-mode bearings_only \
    $EXTRA \
    --wandb --wandb-project orbital-rl --wandb-group "t5-nav-$ARM" \
    --train.data-dir "$PUF/experiments_extnav/$ARM" \
    > "/tmp/${ARM}.log" 2>&1
echo "train exit=$?"

# Newest checkpoint in the arm's private data dir.
CK=$(ls -t "$PUF"/experiments_extnav/"$ARM"/*/model_*.pt 2>/dev/null | head -1)
echo "ckpt=$CK"
