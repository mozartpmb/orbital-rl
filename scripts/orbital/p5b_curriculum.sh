#!/bin/bash
# Phase 5b Block A curriculum runner.
# macOS-compatible (bash 3.2): no associative arrays.

set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/p5b_curriculum.log
echo "$(date) START $@" >> $LOG

SEEDS_5="42 1337 20260423 2718 31415"
SEEDS_3="42 1337 20260423"

# Stage config: returns "e_max num_seeds budget warm_from"
stage_config() {
    case "$1" in
        stage_1_0) echo "0.05 5 40000000 -" ;;
        stage_1_1) echo "0.10 3 50000000 stage_1_0" ;;
        stage_1_2) echo "0.20 3 50000000 stage_1_1" ;;
        stage_1_3) echo "0.50 5 80000000 stage_1_2" ;;
        stage_1_4) echo "0.70 5 80000000 stage_1_3" ;;
        *) echo "" ;;
    esac
}

# Track best ckpts via file (key=path, line per stage_seed)
BEST_CKPT_FILE=/tmp/p5b_best_ckpts.txt
get_best_ckpt() {
    grep "^${1}=" $BEST_CKPT_FILE 2>/dev/null | tail -1 | cut -d= -f2-
}
set_best_ckpt() {
    echo "${1}=${2}" >> $BEST_CKPT_FILE
}

get_latest_dir() {
    ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1
}

train_one() {
    local stage="$1" seed="$2"
    local config=$(stage_config $stage)
    local e_max=$(echo $config | awk '{print $1}')
    local budget=$(echo $config | awk '{print $3}')
    local warm_from=$(echo $config | awk '{print $4}')

    local warm_arg=""
    if [ "$warm_from" != "-" ]; then
        local warm_ckpt=$(get_best_ckpt "${warm_from}_${seed}")
        if [ -z "$warm_ckpt" ] || [ ! -f "$warm_ckpt" ]; then
            echo "$(date) ERROR no warm-start ckpt for $stage seed=$seed (expected from $warm_from)" >> $LOG
            return 1
        fi
        warm_arg="--load-model-path $warm_ckpt"
    fi

    local tag="p5b_${stage}_seed${seed}"
    echo "$(date) START $tag e_max=$e_max budget=$budget warm=$warm_from" >> $LOG
    cd $PUFFER
    puffer train puffer_orbital --train.seed $seed --train.total-timesteps $budget \
        --train.device cpu --env.init-phase-gap-max 3.14159 \
        --env.e-max-target $e_max --env.e-max-sat $e_max --env.same-orbit-init 1 \
        --train.checkpoint-interval 5 \
        --tag $tag $warm_arg > /tmp/p5b_${tag}.log 2>&1
    local exit_code=$?
    local dir=$(get_latest_dir)
    echo "$(date) DONE_TRAIN $tag exit=$exit_code dir=$dir" >> $LOG
    echo "$dir"
}

scan_best_ckpt() {
    local dir="$1" e_max="$2"
    local best_perf="0"
    local best_ckpt=""
    # Sample every 25 epochs + final ckpt to keep scan fast
    local final_ckpt=$(ls -t $dir/model_puffer_orbital_*.pt | head -1)
    for CKPT in $dir/model_puffer_orbital_*.pt; do
        local ep=$(basename $CKPT | grep -oE "[0-9]+")
        local mod25=$((10#$ep % 25))
        if [ $mod25 -ne 0 ] && [ "$CKPT" != "$final_ckpt" ]; then continue; fi
        local perf_str=$(cd $PUFFER && python3 scripts/orbital/eval_checkpoint.py "$CKPT" \
            --episodes 50 --init-phase-gap-max 3.14159 \
            --e-max-target $e_max --e-max-sat $e_max --same-orbit-init 1 \
            --seed 42 --out-dir /tmp/p5b_scan_$$ 2>&1 | grep "Success rate" | grep -oE "[0-9]+\.[0-9]+")
        rm -rf /tmp/p5b_scan_$$
        [ -z "$perf_str" ] && continue
        if awk "BEGIN {exit !($perf_str > $best_perf)}"; then
            best_perf=$perf_str
            best_ckpt=$CKPT
        fi
    done
    echo "$best_perf $best_ckpt"
}

run_stage_multiseed() {
    local stage="$1"
    local config=$(stage_config $stage)
    local e_max=$(echo $config | awk '{print $1}')
    local n_seeds=$(echo $config | awk '{print $2}')

    local seeds_list
    if [ "$n_seeds" = "5" ]; then seeds_list="$SEEDS_5"; else seeds_list="$SEEDS_3"; fi

    echo "$(date) === $stage ($n_seeds seeds, e_max=$e_max) ===" >> $LOG
    for SEED in $seeds_list; do
        local dir=$(train_one $stage $SEED)
        if [ -z "$dir" ] || [ ! -d "$dir" ]; then
            echo "$(date) ERROR train_one failed for $stage seed=$SEED" >> $LOG
            continue
        fi
        local scan_result=$(scan_best_ckpt $dir $e_max)
        local best_perf=$(echo $scan_result | awk '{print $1}')
        local best_ckpt=$(echo $scan_result | awk '{print $2}')
        set_best_ckpt "${stage}_${SEED}" "$best_ckpt"
        echo "$(date) SCAN $stage seed=$SEED best_perf=${best_perf}% best_ckpt=$best_ckpt" >> $LOG
    done
}

for STAGE_ARG in "$@"; do
    if [ "$STAGE_ARG" = "full_block_a" ]; then
        for STAGE in stage_1_0 stage_1_1 stage_1_2 stage_1_3 stage_1_4; do
            run_stage_multiseed $STAGE
        done
    else
        run_stage_multiseed $STAGE_ARG
    fi
done

echo "$(date) ALL_DONE $@" >> $LOG
