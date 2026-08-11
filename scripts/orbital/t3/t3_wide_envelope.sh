#!/bin/bash
# T3 wide-envelope lineage (seed 42): fresh re-ladder under wide obs scales so
# the policy can operate to 8000 km altitude and e <= 0.30.
#   Scales for the WHOLE lineage: obs_alt_scale_m=8e6, lvlh_scale_m=1.5e7,
#   shape_dv_ref_ms=700 (obs semantics must be constant within a lineage;
#   the LEO T3 canonical keeps its own 1.6e6-scale claim untouched).
#   WL1: e=0, same-orbit, +-180 deg, 500-800 km, cap 3000  (fresh)
#   WL2: e<=0.05 both,    +-180 deg, 300-800 km, cap 3000  (warm)
#   WL3: e<=0.15, de 0.06, da 400 km, 300-2000 km, cap 3000 (warm)
#   WL4: e<=0.30, de 0.08, da 600 km, 300-8000 km, cap 6000 (warm)
# Held-out eval (seed 123, 200 eps) after every rung, at that rung's config.
# macOS bash 3.2 compatible. Run ONE instance at a time (latest-dir capture).
set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
SEED=${1:-42}
LOG=/tmp/t3_wide_s${SEED}.log
EVAL_SEED=123
# Chain-private experiment dir: kills the latest-dir attribution race,
# so multiple chains can run concurrently.
DATA_DIR=experiments/t4_wide_s${SEED}

BASE="--env.shaping-mode 1 --env.shape-gamma 1.0 --env.phase-gap-mode 1 \
--env.phase-obs-mode 1 --env.cap-terminal-reward 0.0 --env.valid-init-only 1 \
--env.init-phase-gap-max 3.14159 --env.obs-alt-scale-m 8e6 --env.lvlh-scale-m 1.5e7 \
--env.shape-dv-ref-ms 700"
BASE_EVAL="--init-phase-gap-max 3.14159 --valid-init-only 1 --seed $EVAL_SEED \
--shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 \
--cap-terminal-reward 0.0 --obs-alt-scale-m 8e6 --lvlh-scale-m 1.5e7 --shape-dv-ref-ms 700"

get_latest_dir() { ls -td $PUFFER/$DATA_DIR/puffer_orbital_*/ 2>/dev/null | head -1; }
last_ckpt() { ls "$1"/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1; }

run_rung() {  # <name> <train_env_flags> <eval_flags> <warm_ckpt|none>
    local name="$1" tflags="$2" eflags="$3" warm="$4"
    local warm_flag=""
    if [ "$warm" != "none" ]; then warm_flag="--load-model-path $warm"; fi
    local before=$(get_latest_dir)
    cd $PUFFER
    puffer train puffer_orbital --train.seed $SEED --train.total-timesteps 50000000 \
        --train.device cpu --train.data-dir $DATA_DIR $BASE $tflags --train.checkpoint-interval 20 \
        --wandb --wandb-project orbital-rl --wandb-group t4-wide-s${SEED} \
        $warm_flag --tag t3_${name}_s${SEED} > /tmp/t3_${name}_s${SEED}_train.log 2>&1
    local dir=$(get_latest_dir)
    if [ "$dir" = "$before" ]; then
        echo "$(date) ABORT $name: no new dir" >> $LOG; echo "RESULT $name=NO_NEW_DIR"; exit 1
    fi
    local ckpt=$(last_ckpt "$dir")
    local res=$(python3 scripts/orbital/eval_checkpoint.py "$ckpt" --episodes 200 \
        $BASE_EVAL $eflags --out-dir /tmp/t3_wide_eval_${name} 2>/dev/null \
        | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|')
    echo "$(date) $name seed=$SEED dir=$(basename $dir) ckpt=$(basename $ckpt) heldout=$res" >> $LOG
    echo "RUNG $name dir=$(basename $dir) heldout=$res"
    LAST_CKPT=$ckpt
}

echo "$(date) ===== t3_wide_envelope seed=$SEED =====" >> $LOG
LAST_CKPT=none

run_rung WL1 \
  "--env.same-orbit-init 1 --env.e-max-target 0.0 --env.e-max-sat 0.0 --env.a-min-override 6.871e6 --env.a-max-override 7.171e6 --env.episode-cap-steps 3000" \
  "--same-orbit-init 1 --e-max-target 0.0 --e-max-sat 0.0 --a-min-override 6.871e6 --a-max-override 7.171e6 --episode-cap-steps 3000" \
  none

run_rung WL2 \
  "--env.e-max-target 0.05 --env.e-max-sat 0.05 --env.episode-cap-steps 3000" \
  "--e-max-target 0.05 --e-max-sat 0.05 --episode-cap-steps 3000" \
  "$LAST_CKPT"

run_rung WL3 \
  "--env.e-max-target 0.15 --env.de-max 0.06 --env.da-max-m 400e3 --env.a-min-override 6.671e6 --env.a-max-override 8.371e6 --env.episode-cap-steps 3000" \
  "--e-max-target 0.15 --de-max 0.06 --da-max-m 400e3 --a-min-override 6.671e6 --a-max-override 8.371e6 --episode-cap-steps 3000" \
  "$LAST_CKPT"

run_rung WL4 \
  "--env.e-max-target 0.30 --env.de-max 0.08 --env.da-max-m 600e3 --env.a-min-override 6.671e6 --env.a-max-override 14.371e6 --env.episode-cap-steps 6000" \
  "--e-max-target 0.30 --de-max 0.08 --da-max-m 600e3 --a-min-override 6.671e6 --a-max-override 14.371e6 --episode-cap-steps 6000" \
  "$LAST_CKPT"

echo "RESULT wide_envelope seed=$SEED complete; see $LOG"
