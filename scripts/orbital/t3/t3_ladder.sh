#!/bin/bash
# T3 recovery ladder, one training seed per invocation:
#   L1: fresh 50M — e=0, same-orbit init, physical gap ±180°, a ∈ 500–800 km
#   L2: 50M warm-started from L1-final — headline (e≤0.05 both, independent ω,
#       LEO 300–800 km, physical gap ±180°)
# Every eval: eval_checkpoint.py "Physical success" line, greedy, held-out seed.
# Usage: t3_ladder.sh <train_seed>
# macOS bash 3.2 compatible.
set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
SEED=$1
LOG=/tmp/t3_ladder_s${SEED}.log
EVAL_SEED=123

T3_ENV="--env.shaping-mode 1 --env.shape-gamma 1.0 --env.phase-gap-mode 1 \
--env.phase-obs-mode 1 --env.episode-cap-steps 3000 --env.cap-terminal-reward 0.0 \
--env.valid-init-only 1"

get_latest_dir() { ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1; }

heval() {  # <ckpt> <soi> <emax> <amin|none> <eps> -> "n/d"
    local ckpt="$1" soi="$2" emax="$3" amin="$4" eps="$5"
    local amin_flag=""
    if [ "$amin" != "none" ]; then amin_flag="--a-min-override $amin"; fi
    cd $PUFFER
    python3 scripts/orbital/eval_checkpoint.py "$ckpt" --episodes $eps --seed $EVAL_SEED \
        --e-max-target $emax --e-max-sat $emax --same-orbit-init $soi \
        --init-phase-gap-max 3.14159 --valid-init-only 1 $amin_flag \
        --shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 \
        --episode-cap-steps 3000 --cap-terminal-reward 0.0 \
        --out-dir /tmp/t3_ladder_eval_s${SEED} 2>/dev/null \
        | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|'
}

last_ckpt() { ls "$1"/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1; }

echo "$(date) ===== t3_ladder seed=$SEED =====" >> $LOG

# ── L1 ──────────────────────────────────────────────────────────────────────
cd $PUFFER
puffer train puffer_orbital --train.seed $SEED --train.total-timesteps 50000000 \
    --train.device cpu $T3_ENV \
    --env.same-orbit-init 1 --env.e-max-target 0.0 --env.e-max-sat 0.0 \
    --env.init-phase-gap-max 3.14159 --env.a-min-override 6.871e6 \
    --train.checkpoint-interval 20 --tag t3_L1_s${SEED} \
    > /tmp/t3_L1_s${SEED}_train.log 2>&1
L1_DIR=$(get_latest_dir)
L1_CKPT=$(last_ckpt "$L1_DIR")
L1_RES=$(heval "$L1_CKPT" 1 0.0 6.871e6 200)
echo "$(date) L1 seed=$SEED dir=$(basename $L1_DIR) ckpt=$(basename $L1_CKPT) heldout=$L1_RES" >> $LOG

if [ -z "$L1_CKPT" ] || [ ! -f "$L1_CKPT" ]; then
    echo "$(date) ABORT seed=$SEED: no L1 ckpt" >> $LOG
    echo "RESULT seed=$SEED L1=FAILED"
    exit 1
fi

# ── L2 (warm from L1) ───────────────────────────────────────────────────────
puffer train puffer_orbital --train.seed $SEED --train.total-timesteps 50000000 \
    --train.device cpu $T3_ENV \
    --env.same-orbit-init 0 --env.e-max-target 0.05 --env.e-max-sat 0.05 \
    --env.init-phase-gap-max 3.14159 \
    --train.checkpoint-interval 20 --tag t3_L2_s${SEED} \
    --load-model-path "$L1_CKPT" \
    > /tmp/t3_L2_s${SEED}_train.log 2>&1
L2_DIR=$(get_latest_dir)
L2_CKPT=$(last_ckpt "$L2_DIR")
L2_RES=$(heval "$L2_CKPT" 0 0.05 none 200)
echo "$(date) L2 seed=$SEED dir=$(basename $L2_DIR) ckpt=$(basename $L2_CKPT) heldout=$L2_RES" >> $LOG

echo "RESULT seed=$SEED L1_dir=$(basename $L1_DIR) L1=$L1_RES L2_dir=$(basename $L2_DIR) L2_ckpt=$(basename $L2_CKPT) L2=$L2_RES"
