#!/bin/bash
# T1/T12 — corrected-dynamics recovery, round 2.
#
# Context: after the true_to_mean() fix, fresh Stage-1 bootstrap fails (0.7%
# at 40M, 0% at 80M) — the buggy burn-teleport was subsidizing exploration.
# Direct gentle re-adaptation of the old Stage-4 policy reached 59% rolling /
# 33.5% greedy (epoch 25), non-monotonic after.
#
# Three arms, sequential:
#   A. Stochastic eval of the direct-re-adapt peak ckpt (diagnostic: how much
#      knowledge is in the sampled policy vs argmax).
#   B. Crystallization: continue from that peak at lr 1e-4 (4x lower), 15M —
#      reduce policy churn, let argmax catch up to the sampled policy.
#   C. Curriculum rebuild: re-adapt the committed STAGE-1 warm-start under
#      corrected dynamics (soi=1, gentle), then run the standard Stage-4
#      recipe from its best ckpt (default profile, 50M) — the original
#      two-stage lineage rebuilt on corrected physics.
#
# macOS bash 3.2 compatible.
set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/t1_readapt_v2.log
READAPT_PEAK=$PUFFER/experiments/puffer_orbital_178640242476/model_puffer_orbital_000025.pt
STAGE1_WARM=$ROOT/models/phase5e/seed42_stage1_warmstart.pt

get_latest_dir() { ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1; }

heval() {  # heval <ckpt> <soi> <eps> [--stochastic] -> prints "n/d"
    local ckpt="$1" soi="$2" eps="$3" extra="${4:-}"
    cd $PUFFER
    python3 scripts/orbital/eval_checkpoint.py "$ckpt" --episodes $eps \
        --e-max-target 0.05 --e-max-sat 0.05 --init-phase-gap-max 3.14159 \
        --same-orbit-init $soi --valid-init-only 1 --legacy-action-space 10 \
        $extra --out-dir /tmp/t1rv2_scan_ignore --seed 42 2>/dev/null \
        | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|'
}

scan_dir() {  # scan_dir <dir> <soi> <eps> -> echoes "best_ckpt best_frac"
    local dir="$1" soi="$2" eps="$3"
    local best_n=-1 best_ckpt="" best_frac=""
    local n_ckpts=$(ls $dir/model_puffer_orbital_*.pt 2>/dev/null | wc -l | tr -d ' ')
    local ckpts=$(ls $dir/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | awk -v last="$n_ckpts" 'NR==1 || NR%5==0 || NR==last{print}')
    for ckpt in $ckpts; do
        local frac=$(heval "$ckpt" "$soi" "$eps")
        local n=${frac%%/*}
        echo "$(date '+%H:%M:%S') scan $(basename $ckpt) soi=$soi -> $frac" >> $LOG
        if [ -n "$n" ] && [ "$n" -gt "$best_n" ] 2>/dev/null; then
            best_n=$n; best_ckpt=$ckpt; best_frac=$frac
        fi
    done
    echo "$best_ckpt $best_frac"
}

echo "$(date) ===== t1_readapt_v2 =====" >> $LOG

# Arm A — stochastic diagnostic of the direct-re-adapt peak
A=$(heval "$READAPT_PEAK" 0 200 --stochastic)
echo "$(date) ARM-A stochastic eval of readapt epoch-25: $A (greedy was 67/200)" >> $LOG

# Arm B — low-lr crystallization from the peak
cd $PUFFER
puffer train puffer_orbital --train.seed 42 --train.total-timesteps 15000000 \
    --train.device cpu --train.learning-rate 1e-4 --train.lr-max 1e-4 \
    --train.ent-coef 0.001 \
    --env.init-phase-gap-max 3.14159 --env.e-max-target 0.05 --env.e-max-sat 0.05 \
    --env.same-orbit-init 0 --env.valid-init-only 1 --env.legacy-action-space 10 \
    --train.checkpoint-interval 5 \
    --load-model-path "$READAPT_PEAK" --tag t1rv2_crystal > /tmp/t1rv2_crystal.log 2>&1
B_DIR=$(get_latest_dir)
read B_BEST B_FRAC <<< "$(scan_dir $B_DIR 0 200)"
echo "$(date) ARM-B crystallization best: $B_BEST -> $B_FRAC" >> $LOG

# Arm C — curriculum rebuild: stage-1 re-adapt then standard stage-4
puffer train puffer_orbital --train.seed 42 --train.total-timesteps 20000000 \
    --train.device cpu --train.learning-rate 5e-4 --train.lr-max 5e-4 \
    --train.ent-coef 0.005 \
    --env.init-phase-gap-max 3.14159 --env.e-max-target 0.05 --env.e-max-sat 0.05 \
    --env.same-orbit-init 1 --env.valid-init-only 1 --env.legacy-action-space 10 \
    --train.checkpoint-interval 5 \
    --load-model-path "$STAGE1_WARM" --tag t1rv2_s1readapt > /tmp/t1rv2_s1readapt.log 2>&1
C1_DIR=$(get_latest_dir)
# warm-start quality: scan stage-1 ckpts at STAGE-4 conditions (soi=0)
read C1_BEST C1_FRAC <<< "$(scan_dir $C1_DIR 0 100)"
echo "$(date) ARM-C stage-1 re-adapt best-for-warmstart: $C1_BEST -> $C1_FRAC" >> $LOG
if [ -n "$C1_BEST" ] && [ -f "$C1_BEST" ]; then
    puffer train puffer_orbital --train.seed 42 --train.total-timesteps 50000000 \
        --train.device cpu \
        --env.init-phase-gap-max 3.14159 --env.e-max-target 0.05 --env.e-max-sat 0.05 \
        --env.same-orbit-init 0 --env.valid-init-only 1 --env.legacy-action-space 10 \
        --train.checkpoint-interval 5 \
        --load-model-path "$C1_BEST" --tag t1rv2_s4 > /tmp/t1rv2_s4.log 2>&1
    C4_DIR=$(get_latest_dir)
    read C4_BEST C4_FRAC <<< "$(scan_dir $C4_DIR 0 200)"
    echo "$(date) ARM-C stage-4 best: $C4_BEST -> $C4_FRAC" >> $LOG
else
    echo "$(date) ARM-C aborted: no stage-1 ckpt" >> $LOG
    C4_BEST=""; C4_FRAC=""
fi

echo "RESULT armA_stoch=$A armB_best=$B_BEST armB=$B_FRAC armC_s1=$C1_FRAC armC_s4_best=$C4_BEST armC_s4=$C4_FRAC"
