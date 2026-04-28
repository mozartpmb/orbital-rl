#!/bin/bash
# Phase 5a entry-condition probe (~20 min single-seed).
# Goal: find a viable starting condition before re-running Investigation A.
set -u
LOG=/tmp/p5a_probe.log
echo "$(date) START probe" > $LOG
PUFFER=/Users/pete/space_training/pufferlib
cd $PUFFER

train_probe() {
    local tag="$1"; shift
    local timesteps="$1"; shift
    local phase="$1"; shift
    local e="$1"; shift
    local extra="$@"
    echo "$(date) START $tag phase=$phase e=$e steps=$timesteps extra='$extra'" >> $LOG
    puffer train puffer_orbital --train.seed 42 --train.total-timesteps $timesteps \
        --train.device cpu --env.init-phase-gap-max $phase --env.e-max-target $e \
        --tag $tag $extra > /tmp/p5a_$tag.log 2>&1
    local dir=$(ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1)
    local ckpt=$(ls -t $dir/model_puffer_orbital_*.pt 2>/dev/null | head -1)
    local peak=$(grep -oE "perf +[0-9.]+" /tmp/p5a_$tag.log | awk '{print $2}' | sort -n | tail -1)
    local final=$(grep -oE "perf +[0-9.]+" /tmp/p5a_$tag.log | awk '{print $2}' | tail -1)
    echo "$(date) DONE $tag peak=$peak final=$final ckpt=$ckpt" >> $LOG
    echo "$ckpt"
}

# Probe A: Phase 4 recipe at e=0, then e-ramp to 0.05
A1=$(train_probe "p5a_probe_A_pi6_e0" 10000000 0.5236 0.0)
A2=$(train_probe "p5a_probe_A_pi6_e0p05_warm" 10000000 0.5236 0.05 --load-model-path "$A1")

# Probe B: smaller bounds (3 fresh runs)
train_probe "p5a_probe_B_pi6_e0p02" 10000000 0.5236 0.02
train_probe "p5a_probe_B_pi12_e0p05" 10000000 0.2618 0.05
train_probe "p5a_probe_B_pi12_e0p02" 10000000 0.2618 0.02

# Probe C: bigger phase at e=0
train_probe "p5a_probe_C_pi2_e0" 10000000 1.5708 0.0

echo "$(date) DONE probe" >> $LOG
