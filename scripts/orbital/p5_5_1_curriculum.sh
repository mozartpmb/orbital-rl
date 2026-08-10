#!/bin/bash
# Phase 5.5 Stage 5.5.1 — slight altitude extension warm-start (smoke test).
# Three seeds, 30M steps each, warm-started from per-seed Phase 5b ckpts.
# a in [6.671e6, 8.5e6] m (LEO + ~1300 km headroom).
# e_max preserved at 0.05.
# Env scaling kwargs held at LEO defaults (lvlh=R_EARTH, obs_alt=1.6e6, K=0.001)
# to isolate altitude expansion from observation-distribution shifts.
# Discrete action space coerced to legacy 10-head for warm-start compatibility.

set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/p5_5_1_curriculum.log
echo "$(date) START $@" >> $LOG

# Per-seed warm-start ckpts (MODELS.md canonical Phase 5b multi-seed)
declare_warm() {
    case "$1" in
        42)        echo "$PUFFER/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt" ;;
        31415)     echo "$PUFFER/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt" ;;
        20260423)  echo "$PUFFER/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt" ;;
        *) echo "" ;;
    esac
}

SEEDS_3="42 31415 20260423"
BUDGET=30000000
A_MIN=6.671e6
A_MAX=8.5e6

train_seed() {
    local seed="$1"
    local warm
    warm=$(declare_warm "$seed")
    if [ -z "$warm" ] || [ ! -f "$warm" ]; then
        echo "$(date) ERROR no warm ckpt for seed=$seed (expected: $warm)" >> $LOG
        return 1
    fi

    local tag="p5_5_1_s${seed}"
    echo "$(date) START $tag warm=$warm" >> $LOG
    cd "$PUFFER"
    puffer train puffer_orbital \
        --train.seed "$seed" --train.total-timesteps "$BUDGET" \
        --train.device cpu \
        --env.init-phase-gap-max 3.14159 \
        --env.e-max-target 0.05 --env.e-max-sat 0.05 \
        --env.same-orbit-init 0 --env.valid-init-only 1 \
        --env.a-min-override "$A_MIN" --env.a-max-override "$A_MAX" \
        --env.legacy-action-space 10 \
        --train.checkpoint-interval 5 \
        --tag "$tag" \
        --load-model-path "$warm" > "/tmp/${tag}.log" 2>&1
    local exit_code=$?
    local dir
    dir=$(ls -td "$PUFFER"/experiments/puffer_orbital_*/ 2>/dev/null | head -1)
    echo "$(date) DONE $tag exit=$exit_code dir=$dir" >> $LOG
    echo "$dir"
}

if [ "${1:-}" = "all" ]; then
    for s in $SEEDS_3; do train_seed "$s"; done
elif [ "${1:-}" = "single" ]; then
    train_seed "${2:-42}"
else
    echo "Usage: $0 {single SEED|all}"
fi

echo "$(date) ALL_DONE $@" >> $LOG
