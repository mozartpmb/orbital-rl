#!/bin/bash
# V3 — F-B+ hyperparameter probe at Stage 5.5.1.
# lr_max=5e-4 (5× F-B), ent_coef=0.005 (5× F-B). Same warm-start as F-B.
set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/p5_5_v3.log
echo "$(date) START $@" >> $LOG

declare_warm() {
    case "$1" in
        42)        echo "$PUFFER/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt" ;;
        31415)     echo "$PUFFER/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt" ;;
        20260423)  echo "$PUFFER/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt" ;;
    esac
}

BUDGET=7000000
for seed in 42 31415 20260423; do
    warm=$(declare_warm "$seed")
    [ -f "$warm" ] || { echo "MISSING $seed: $warm" >> $LOG; continue; }
    tag="p5_5_v3_FBplus_s${seed}"
    echo "$(date) START $tag warm=$warm" >> $LOG
    cd "$PUFFER"
    puffer train puffer_orbital \
        --train.seed "$seed" --train.total-timesteps "$BUDGET" \
        --train.device cpu \
        --train.learning-rate 5e-4 \
        --train.lr-max 5e-4 \
        --train.ent-coef 0.005 \
        --env.init-phase-gap-max 3.14159 \
        --env.e-max-target 0.05 --env.e-max-sat 0.05 \
        --env.same-orbit-init 0 --env.valid-init-only 1 \
        --env.a-min-override 6.671e6 --env.a-max-override 8.5e6 \
        --env.legacy-action-space 10 \
        --train.checkpoint-interval 5 \
        --tag "$tag" \
        --load-model-path "$warm" > "/tmp/${tag}.log" 2>&1
    exit_code=$?
    dir=$(ls -td "$PUFFER"/experiments/puffer_orbital_*/ 2>/dev/null | head -1)
    echo "$(date) DONE $tag exit=$exit_code dir=$dir" >> $LOG
done
echo "$(date) ALL_DONE" >> $LOG
