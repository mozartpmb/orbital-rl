#!/bin/bash
# Phase 5.5 Probe 2 — fine-tune protocol investigation at Stage 5.5.1 conditions.
# F-A (aggressive) reuses existing Stage 5.5.1 smoke data.
# F-B (gentle): lr_max=1e-4, ent_coef=0.001, 7M steps.
# F-C (mid-training warm-start): warm from Phase 5b epoch-175 ckpt, F-B hyperparams.
# 3 seeds per protocol.

set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/p5_5_probe2.log
echo "$(date) START $@" >> $LOG

# Per-seed warm-start ckpts:
#  F-B → final (epoch-350) ckpts (canonical Phase 5b)
#  F-C → midpoint (epoch-175) ckpts of the same training runs
declare_warm_final() {
    case "$1" in
        42)        echo "$PUFFER/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt" ;;
        31415)     echo "$PUFFER/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt" ;;
        20260423)  echo "$PUFFER/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt" ;;
        *) echo "" ;;
    esac
}
declare_warm_mid() {
    case "$1" in
        42)        echo "$PUFFER/experiments/puffer_orbital_177750198246/model_puffer_orbital_000175.pt" ;;
        31415)     echo "$PUFFER/experiments/puffer_orbital_177750405236/model_puffer_orbital_000175.pt" ;;
        20260423)  echo "$PUFFER/experiments/puffer_orbital_177750301624/model_puffer_orbital_000090.pt" ;;
        *) echo "" ;;
    esac
}

SEEDS_3="42 31415 20260423"
BUDGET=7000000

train_one() {
    local protocol="$1" seed="$2" warm="$3" lr="$4" ent="$5"
    local tag="p5_5_probe2_${protocol}_s${seed}"
    echo "$(date) START $tag warm=$warm lr=$lr ent=$ent" >> $LOG
    cd "$PUFFER"
    puffer train puffer_orbital \
        --train.seed "$seed" --train.total-timesteps "$BUDGET" \
        --train.device cpu \
        --train.learning-rate "$lr" \
        --train.lr-max "$lr" \
        --train.ent-coef "$ent" \
        --env.init-phase-gap-max 3.14159 \
        --env.e-max-target 0.05 --env.e-max-sat 0.05 \
        --env.same-orbit-init 0 --env.valid-init-only 1 \
        --env.a-min-override 6.671e6 --env.a-max-override 8.5e6 \
        --env.legacy-action-space 10 \
        --train.checkpoint-interval 5 \
        --tag "$tag" \
        --load-model-path "$warm" > "/tmp/${tag}.log" 2>&1
    local exit_code=$?
    local dir
    dir=$(ls -td "$PUFFER"/experiments/puffer_orbital_*/ 2>/dev/null | head -1)
    echo "$(date) DONE $tag exit=$exit_code dir=$dir" >> $LOG
}

# F-B: gentle, final-ckpt warm-start
for seed in $SEEDS_3; do
    warm=$(declare_warm_final "$seed")
    [ -f "$warm" ] || { echo "MISSING warm_final $seed: $warm" >> $LOG; continue; }
    train_one "FB" "$seed" "$warm" 1e-4 0.001
done

# F-C: gentle, mid-training warm-start
for seed in $SEEDS_3; do
    warm=$(declare_warm_mid "$seed")
    [ -f "$warm" ] || { echo "MISSING warm_mid $seed: $warm" >> $LOG; continue; }
    train_one "FC" "$seed" "$warm" 1e-4 0.001
done

echo "$(date) ALL_DONE" >> $LOG
