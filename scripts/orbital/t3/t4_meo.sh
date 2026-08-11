#!/bin/bash
# T4-C — MEO lineage, Discrete-20 (3h/6h warps), obs scales for 300–20,200 km.
#   Lineage scales (constant within family): obs_alt_scale_m=2.1e7,
#   lvlh_scale_m=4e7, shape_dv_ref_ms=700.
#   M1: fresh D20 — e=0, same-orbit, ±180°, 500–800 km,          cap 3000
#   M2: e≤0.05 both, 300–800 km,                                  cap 3000
#   M3: e≤0.15, de 0.06, da 400 km, 300–2,000 km,                 cap 3000
#   M4: e≤0.30, de 0.08, da 600 km, 300–8,000 km,                 cap 6000
#   M5: e≤0.50, de 0.10, da 1,000 km, 300–20,200 km,              cap 12000
# M5 joint feasibility (recon §3.4) is 93.0% — success is reported against
# that ceiling, not against 100%. Held-out eval (seed 123, 200 eps) per rung.
# Chain-private data_dir — safe to run concurrently with other chains.
set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
SEED=${1:-42}
LOG=/tmp/t4_meo_s${SEED}.log
EVAL_SEED=123
DATA_DIR=experiments/t4_meo_s${SEED}

BASE="--env.shaping-mode 1 --env.shape-gamma 1.0 --env.phase-gap-mode 1 \
--env.phase-obs-mode 1 --env.cap-terminal-reward 0.0 --env.valid-init-only 1 \
--env.init-phase-gap-max 3.14159 --env.obs-alt-scale-m 2.1e7 --env.lvlh-scale-m 4e7 \
--env.shape-dv-ref-ms 700 --env.legacy-action-space 20"
BASE_EVAL="--init-phase-gap-max 3.14159 --valid-init-only 1 --seed $EVAL_SEED \
--shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 \
--cap-terminal-reward 0.0 --obs-alt-scale-m 2.1e7 --lvlh-scale-m 4e7 \
--shape-dv-ref-ms 700 --legacy-action-space 20"

get_latest_dir() { ls -td $PUFFER/$DATA_DIR/puffer_orbital_*/ 2>/dev/null | head -1; }
last_ckpt() { ls "$1"/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1; }

run_rung() {  # <name> <train_env_flags> <eval_flags> <warm_ckpt|none>
    local name="$1" tflags="$2" eflags="$3" warm="$4"
    local warm_flag=""
    if [ "$warm" != "none" ]; then warm_flag="--load-model-path $warm"; fi
    local before=$(get_latest_dir)
    cd $PUFFER
    puffer train puffer_orbital --train.seed $SEED --train.total-timesteps 50000000 \
        --train.device cpu --train.data-dir $DATA_DIR $BASE $tflags \
        --train.checkpoint-interval 20 --wandb --wandb-project orbital-rl \
        --wandb-group t4-meo-s${SEED} $warm_flag --tag t4_${name}_s${SEED} \
        > /tmp/t4_${name}_s${SEED}_train.log 2>&1
    local dir=$(get_latest_dir)
    if [ "$dir" = "$before" ]; then
        echo "$(date) ABORT $name: no new dir" >> $LOG; echo "RESULT $name=NO_NEW_DIR"; exit 1
    fi
    local ckpt=$(last_ckpt "$dir")
    local res=$(python3 scripts/orbital/eval_checkpoint.py "$ckpt" --episodes 200 \
        $BASE_EVAL $eflags --out-dir /tmp/t4_meo_eval_${name} 2>/dev/null \
        | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|')
    echo "$(date) $name seed=$SEED dir=$(basename $dir) ckpt=$(basename $ckpt) heldout=$res" >> $LOG
    echo "RUNG $name dir=$(basename $dir) heldout=$res"
    LAST_CKPT=$ckpt
}

echo "$(date) ===== t4_meo seed=$SEED =====" >> $LOG
LAST_CKPT=none

run_rung M1 \
  "--env.same-orbit-init 1 --env.e-max-target 0.0 --env.e-max-sat 0.0 --env.a-min-override 6.871e6 --env.a-max-override 7.171e6 --env.episode-cap-steps 3000" \
  "--same-orbit-init 1 --e-max-target 0.0 --e-max-sat 0.0 --a-min-override 6.871e6 --a-max-override 7.171e6 --episode-cap-steps 3000" \
  none

run_rung M2 \
  "--env.e-max-target 0.05 --env.e-max-sat 0.05 --env.episode-cap-steps 3000" \
  "--e-max-target 0.05 --e-max-sat 0.05 --episode-cap-steps 3000" \
  "$LAST_CKPT"

run_rung M3 \
  "--env.e-max-target 0.15 --env.de-max 0.06 --env.da-max-m 400e3 --env.a-min-override 6.671e6 --env.a-max-override 8.371e6 --env.episode-cap-steps 3000" \
  "--e-max-target 0.15 --de-max 0.06 --da-max-m 400e3 --a-min-override 6.671e6 --a-max-override 8.371e6 --episode-cap-steps 3000" \
  "$LAST_CKPT"

run_rung M4 \
  "--env.e-max-target 0.30 --env.de-max 0.08 --env.da-max-m 600e3 --env.a-min-override 6.671e6 --env.a-max-override 14.371e6 --env.episode-cap-steps 6000" \
  "--e-max-target 0.30 --de-max 0.08 --da-max-m 600e3 --a-min-override 6.671e6 --a-max-override 14.371e6 --episode-cap-steps 6000" \
  "$LAST_CKPT"

run_rung M5 \
  "--env.e-max-target 0.50 --env.de-max 0.10 --env.da-max-m 1000e3 --env.a-min-override 6.671e6 --env.a-max-override 26.571e6 --env.episode-cap-steps 12000" \
  "--e-max-target 0.50 --de-max 0.10 --da-max-m 1000e3 --a-min-override 6.671e6 --a-max-override 26.571e6 --episode-cap-steps 12000" \
  "$LAST_CKPT"

echo "RESULT meo seed=$SEED complete; see $LOG"
