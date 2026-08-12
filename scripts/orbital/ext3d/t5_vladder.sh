#!/bin/bash
# Corrected wide-3D V-ladder (post de_max fix), per diag recommendation.
set -uo pipefail
WT=/Users/pete/space_training-ext3d
PUF=$WT/pufferlib
SEED=${1:-42}
LOG=/tmp/t5_v_s${SEED}.log
RUN="python3 $WT/scripts/orbital/ext3d/puffer_wt.py train puffer_orbital --train.device cpu --train.checkpoint-interval 20 --wandb --wandb-project orbital-rl --wandb-group t5-v3d-s${SEED} --train.seed $SEED"
BASE="--env.shaping-mode 2 --env.shape-gamma 1.0 --env.phase-gap-mode 1 --env.phase-obs-mode 1 --env.cap-terminal-reward 0.0 --env.valid-init-only 1 --env.init-phase-gap-max 3.14159 --env.dim3-mode 1 --env.shape-dv-ref-ms 700 --env.legacy-action-space 30 --env.obs-alt-scale-m 8e6 --env.lvlh-scale-m 1.5e7"
BASE_EVAL="--seed 123 --init-phase-gap-max 3.14159 --valid-init-only 1 --shaping-mode 2 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 --cap-terminal-reward 0.0 --dim3-mode 1 --shape-dv-ref-ms 700 --legacy-action-space 30 --obs-alt-scale-m 8e6 --lvlh-scale-m 1.5e7"
last_ckpt() { ls "$1"/puffer_orbital_*/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1; }
run_rung() { # name steps train_flags eval_flags warm
  local name=$1 steps=$2 tf=$3 ef=$4 warm=$5 wflag=""
  [ "$warm" != "none" ] && wflag="--load-model-path $warm"
  cd $PUF
  $RUN --train.total-timesteps $steps --train.data-dir experiments_ext3d/v_${name}_s${SEED} $BASE $tf $wflag --tag t5_v_${name}_s${SEED} > /tmp/t5_v_${name}_s${SEED}_train.log 2>&1
  local ck=$(last_ckpt $PUF/experiments_ext3d/v_${name}_s${SEED})
  [ -f "$ck" ] || { echo "RESULT ${name}=NO_CKPT"; exit 1; }
  local res=$(python3 scripts/orbital/eval_checkpoint.py "$ck" --episodes 200 $BASE_EVAL $ef --out-dir /tmp/t5_v_eval_${name} 2>/dev/null | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|')
  echo "$(date) $name heldout=$res ckpt=$ck" >> $LOG
  echo "RUNG $name heldout=$res"
  LAST=$ck
}
echo "$(date) ===== t5_vladder seed=$SEED =====" >> $LOG
W2CKPT=$PUF/experiments_ext3d/w3d_W2_s${SEED}/puffer_orbital_*/model_puffer_orbital_000382.pt
W2=$(ls $W2CKPT 2>/dev/null | tail -1)
[ -f "$W2" ] || { echo "RESULT V2=NO_W2_WARMSTART"; exit 1; }
run_rung V2 15000000 "--env.e-max-target 0.05 --env.e-max-sat 0.05 --env.a-min-override 6.671e6 --env.a-max-override 7.171e6 --env.di-max-rad 0.017453 --env.episode-cap-steps 3000" \
                     "--e-max-target 0.05 --e-max-sat 0.05 --a-min-override 6.671e6 --a-max-override 7.171e6 --di-max-rad 0.017453 --episode-cap-steps 3000" "$W2"
run_rung V3 25000000 "--env.e-max-target 0.15 --env.de-max 0.06 --env.da-max-m 400e3 --env.a-min-override 6.671e6 --env.a-max-override 8.371e6 --env.di-max-rad 0.017453 --env.episode-cap-steps 3000" \
                     "--e-max-target 0.15 --de-max 0.06 --da-max-m 400e3 --a-min-override 6.671e6 --a-max-override 8.371e6 --di-max-rad 0.017453 --episode-cap-steps 3000" "$LAST"
run_rung V4 30000000 "--env.e-max-target 0.22 --env.de-max 0.07 --env.da-max-m 500e3 --env.a-min-override 6.671e6 --env.a-max-override 10.371e6 --env.di-max-rad 0.013090 --env.episode-cap-steps 4500" \
                     "--e-max-target 0.22 --de-max 0.07 --da-max-m 500e3 --a-min-override 6.671e6 --a-max-override 10.371e6 --di-max-rad 0.013090 --episode-cap-steps 4500" "$LAST"
run_rung V5 30000000 "--env.e-max-target 0.30 --env.de-max 0.08 --env.da-max-m 600e3 --env.a-min-override 6.671e6 --env.a-max-override 14.371e6 --env.di-max-rad 0.013090 --env.episode-cap-steps 6000" \
                     "--e-max-target 0.30 --de-max 0.08 --da-max-m 600e3 --a-min-override 6.671e6 --a-max-override 14.371e6 --di-max-rad 0.013090 --episode-cap-steps 6000" "$LAST"
echo "RESULT vladder seed=$SEED complete"
