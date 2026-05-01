#!/bin/bash
# Phase 5e Block II.A curriculum runner.
# Two-stage: Stage 1.0 (same_orbit_init=1) → Stage 4.0 (same_orbit_init=0).
# All stages run with valid_init_only=1.
# macOS-compatible (bash 3.2).

set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/p5e_curriculum.log
echo "$(date) START $@" >> $LOG

SEEDS_5="42 1337 20260423 31415 2718"

# Stage config: returns "e_max budget same_orbit_init warm_from"
stage_config() {
    case "$1" in
        s10) echo "0.05 40000000 1 -" ;;     # Stage 1.0: same_orbit_init bootstrap
        s40) echo "0.05 50000000 0 s10" ;;   # Stage 4.0: random sat init, warm from s10
        *) echo "" ;;
    esac
}

BEST_CKPT_FILE=/tmp/p5e_best_ckpts.txt
get_best_ckpt() { grep "^${1}=" $BEST_CKPT_FILE 2>/dev/null | tail -1 | cut -d= -f2-; }
set_best_ckpt() { echo "${1}=${2}" >> $BEST_CKPT_FILE; }
get_latest_dir() { ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1; }

train_one() {
    local stage="$1" seed="$2"
    local config=$(stage_config $stage)
    local e_max=$(echo $config | awk '{print $1}')
    local budget=$(echo $config | awk '{print $2}')
    local soi=$(echo $config | awk '{print $3}')
    local warm_from=$(echo $config | awk '{print $4}')

    local warm_arg=""
    if [ "$warm_from" != "-" ]; then
        local warm_ckpt=$(get_best_ckpt "${warm_from}_${seed}")
        if [ -z "$warm_ckpt" ] || [ ! -f "$warm_ckpt" ]; then
            echo "$(date) ERROR no warm ckpt for $stage seed=$seed (from $warm_from)" >> $LOG
            return 1
        fi
        warm_arg="--load-model-path $warm_ckpt"
    fi

    local tag="p5e_${stage}_s${seed}"
    echo "$(date) START $tag e_max=$e_max soi=$soi budget=$budget warm=$warm_from" >> $LOG
    cd $PUFFER
    puffer train puffer_orbital --train.seed $seed --train.total-timesteps $budget \
        --train.device cpu --env.init-phase-gap-max 3.14159 \
        --env.e-max-target $e_max --env.e-max-sat $e_max \
        --env.same-orbit-init $soi --env.valid-init-only 1 \
        --train.checkpoint-interval 5 \
        --tag $tag $warm_arg > /tmp/${tag}.log 2>&1
    local exit_code=$?
    local dir=$(get_latest_dir)
    echo "$(date) DONE $tag exit=$exit_code dir=$dir" >> $LOG
    echo "$dir"
}

# Eval all ckpts in dir at e_max=0.20 (the deliverable test); pick best.
scan_best() {
    local dir="$1"
    local best_perf="0"
    local best_ckpt=""
    cd $PUFFER
    for ckpt in $(ls $dir/model_puffer_orbital_*.pt | sort -t_ -k4 -n | awk 'NR%5==0 || NR==1'); do
        local out=$(python3 scripts/orbital/eval_checkpoint.py $ckpt --episodes 50 \
            --e-max-target 0.20 --e-max-sat 0.20 --init-phase-gap-max 3.14159 \
            --valid-init-only 1 --out-dir /tmp/p5e_scan_ignore --seed 42 2>&1 | grep "Success rate" | head -1)
        local pct=$(echo $out | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%')
        if [ -n "$pct" ]; then
            if [ "$(echo "$pct > $best_perf" | bc -l)" = "1" ]; then
                best_perf=$pct; best_ckpt=$ckpt
            fi
        fi
    done
    echo "$best_perf $best_ckpt"
}

run_seed() {
    local seed="$1"
    for stage in s10 s40; do
        local dir=$(train_one $stage $seed)
        if [ -z "$dir" ] || [ ! -d "$dir" ]; then
            echo "$(date) ERROR train failed $stage seed=$seed" >> $LOG; continue
        fi
        local result=$(scan_best $dir)
        local perf=$(echo $result | awk '{print $1}')
        local ckpt=$(echo $result | awk '{print $2}')
        set_best_ckpt "${stage}_${seed}" "$ckpt"
        echo "$(date) SCAN $stage seed=$seed best_perf=${perf}% best_ckpt=$ckpt" >> $LOG
    done
}

if [ "${1:-}" = "all" ]; then
    for seed in $SEEDS_5; do run_seed $seed; done
elif [ "${1:-}" = "single" ]; then
    run_seed "${2:-42}"
else
    echo "Usage: $0 {single SEED|all}"
fi

echo "$(date) ALL_DONE $@" >> $LOG
