#!/bin/bash
# T1 — tightened-terminal-criterion retrain (5 km / 1 m/s success box).
#
# Two-arm smoke, one seed each, then multi-seed the winner:
#   d10 — proven recipe action set (legacy_action_space=10; min burn 5 m/s).
#         Risk: 1 m/s velocity tolerance is below the burn quantum, so closure
#         must be ballistic (zero-shot scan showed mass strandings at this box).
#   d16 — full Discrete(16) incl. M3 fine burns (±1/±2 m/s), which are exactly
#         the control authority a 1 m/s tolerance needs. Risk: bootstrap
#         dilution (Phase 2 d11 / Phase 5c B4 lessons — but those were tested
#         at harder task settings, never as a fresh Stage-1.0 bootstrap).
#
# Same two-stage curriculum as p5e_curriculum.sh, with the tight box in BOTH
# stages. Stage-1 best ckpt is picked by held-out eval at STAGE-4 conditions
# (discipline §6.7: warm-start quality, not in-stage metric).
#
# Usage: t1_tightbox_curriculum.sh <arm d10|d16> <seed>
# macOS-compatible (bash 3.2).

set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
LOG=/tmp/t1_tightbox.log
RADIUS=5000.0
VELTOL=1.0

ARM="${1:?arm required: d10|d16}"
SEED="${2:?seed required}"

case "$ARM" in
    d10) LAS_TRAIN="--env.legacy-action-space 10"; LAS_EVAL="--legacy-action-space 10" ;;
    d16) LAS_TRAIN=""; LAS_EVAL="" ;;
    *) echo "arm must be d10 or d16"; exit 1 ;;
esac

get_latest_dir() { ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1; }

train_stage() {
    local stage="$1" soi="$2" budget="$3" warm_ckpt="$4"
    local warm_arg=""
    [ "$warm_ckpt" != "-" ] && warm_arg="--load-model-path $warm_ckpt"
    local tag="t1box_${ARM}_${stage}_s${SEED}"
    echo "$(date) START $tag soi=$soi budget=$budget warm=$warm_ckpt" >> $LOG
    cd $PUFFER
    puffer train puffer_orbital --train.seed $SEED --train.total-timesteps $budget \
        --train.device cpu --env.init-phase-gap-max 3.14159 \
        --env.e-max-target 0.05 --env.e-max-sat 0.05 \
        --env.same-orbit-init $soi --env.valid-init-only 1 \
        --env.rendezvous-radius-m $RADIUS --env.rel-vel-tol-ms $VELTOL \
        $LAS_TRAIN \
        --train.checkpoint-interval 5 \
        --tag $tag $warm_arg > /tmp/${tag}.log 2>&1
    echo "$(date) DONE $tag exit=$? dir=$(get_latest_dir)" >> $LOG
    get_latest_dir
}

# Scan a run dir's ckpts (every 25 epochs + final) with held-out eval at the
# given same_orbit_init; echoes "best_ckpt best_pct". Uses Physical success.
scan_best() {
    local dir="$1" soi="$2" eps="$3"
    local best_pct="-1"; local best_ckpt=""
    cd $PUFFER
    local ckpts=$(ls $dir/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | awk 'NR==1 || NR%5==0 || NR==last{print}' last=$(ls $dir/model_puffer_orbital_*.pt 2>/dev/null | wc -l))
    for ckpt in $ckpts; do
        local out=$(python3 scripts/orbital/eval_checkpoint.py $ckpt --episodes $eps \
            --e-max-target 0.05 --e-max-sat 0.05 --init-phase-gap-max 3.14159 \
            --same-orbit-init $soi --valid-init-only 1 \
            --rendezvous-radius-m $RADIUS --rel-vel-tol-ms $VELTOL \
            $LAS_EVAL --out-dir /tmp/t1box_scan_ignore --seed 42 2>&1 \
            | grep "Physical success" | head -1)
        local n=$(echo "$out" | sed -E 's|.* ([0-9]+)/([0-9]+) .*|\1|')
        local d=$(echo "$out" | sed -E 's|.* ([0-9]+)/([0-9]+) .*|\2|')
        if [ -n "$n" ] && [ -n "$d" ] && [ "$d" != "0" ]; then
            local pct=$(echo "scale=2; 100*$n/$d" | bc -l)
            echo "$(date) scan $ckpt soi=$soi -> $n/$d ($pct%)" >> $LOG
            if [ "$(echo "$pct > $best_pct" | bc -l)" = "1" ]; then
                best_pct=$pct; best_ckpt=$ckpt
            fi
        fi
    done
    echo "$best_ckpt $best_pct"
}

echo "$(date) ===== T1 tightbox arm=$ARM seed=$SEED box=${RADIUS}m/${VELTOL}m/s =====" >> $LOG

# Stage 1.0 — fresh bootstrap (same_orbit_init=1), 40M
S1_DIR=$(train_stage s10 1 40000000 -)
# Warm-start quality: scan Stage-1 ckpts at STAGE-4 conditions (soi=0)
read S1_BEST S1_PCT <<< "$(scan_best $S1_DIR 0 50)"
echo "$(date) Stage1 best-for-warmstart: $S1_BEST ($S1_PCT% at stage-4 conditions)" >> $LOG
if [ -z "$S1_BEST" ] || [ "$S1_BEST" = "" ]; then
    echo "$(date) ABORT: no stage-1 ckpt scanned" >> $LOG; exit 1
fi

# Stage 4.0 — random init (same_orbit_init=0), 50M, warm from Stage-1 best
S4_DIR=$(train_stage s40 0 50000000 $S1_BEST)
read S4_BEST S4_PCT <<< "$(scan_best $S4_DIR 0 100)"
echo "$(date) Stage4 best: $S4_BEST ($S4_PCT%)" >> $LOG

# Final: 200-episode eval of the best Stage-4 ckpt
cd $PUFFER
python3 scripts/orbital/eval_checkpoint.py $S4_BEST --episodes 200 \
    --e-max-target 0.05 --e-max-sat 0.05 --init-phase-gap-max 3.14159 \
    --same-orbit-init 0 --valid-init-only 1 \
    --rendezvous-radius-m $RADIUS --rel-vel-tol-ms $VELTOL \
    $LAS_EVAL --out-dir /tmp/t1box_final_${ARM}_s${SEED} --seed 42 2>&1 \
    | tail -8 >> $LOG

echo "$(date) ===== T1 tightbox arm=$ARM seed=$SEED COMPLETE. s1=$S1_DIR s4=$S4_DIR best=$S4_BEST =====" >> $LOG
echo "RESULT arm=$ARM seed=$SEED s1_dir=$S1_DIR s1_best=$S1_BEST s4_dir=$S4_DIR s4_best=$S4_BEST s4_pct=$S4_PCT"
