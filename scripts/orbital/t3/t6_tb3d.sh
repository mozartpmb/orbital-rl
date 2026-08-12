#!/bin/bash
# T6 — 3D x tight-box ladder on merged main. Warm from X3 (Discrete-30).
set -uo pipefail
ROOT=/Users/pete/space_training
PUF=$ROOT/pufferlib
SEED=${1:-42}
LOG=/tmp/t6_tb3d_s${SEED}.log
RUN="caffeinate -is puffer train puffer_orbital --train.device cpu --train.checkpoint-interval 20 --wandb --wandb-project orbital-rl --wandb-group t6-tb3d-s${SEED} --train.seed $SEED"
BASE="--env.shaping-mode 2 --env.shape-gamma 1.0 --env.phase-gap-mode 1 --env.phase-obs-mode 1 --env.cap-terminal-reward 0.0 --env.valid-init-only 1 --env.init-phase-gap-max 3.14159 --env.dim3-mode 1 --env.di-max-rad 0.017453 --env.shape-dv-ref-ms 700 --env.legacy-action-space 30 --env.e-max-target 0.05 --env.e-max-sat 0.05 --env.episode-cap-steps 3000"
BASE_EVAL="--seed 123 --init-phase-gap-max 3.14159 --valid-init-only 1 --shaping-mode 2 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 --cap-terminal-reward 0.0 --dim3-mode 1 --di-max-rad 0.017453 --shape-dv-ref-ms 700 --legacy-action-space 30 --e-max-target 0.05 --e-max-sat 0.05 --episode-cap-steps 3000"
last_ckpt() { ls "$1"/puffer_orbital_*/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1; }
run_rung() { # name steps boxr boxv warm
  local name=$1 steps=$2 br=$3 bv=$4 warm=$5 wflag=""
  [ "$warm" != "none" ] && wflag="--load-model-path $warm"
  cd $PUF
  $RUN --train.total-timesteps $steps --train.data-dir experiments/t6_tb3d_${name}_s${SEED} $BASE \
    --env.rendezvous-radius-m $br --env.rel-vel-tol-ms $bv $wflag --tag t6_tb3d_${name}_s${SEED} \
    > /tmp/t6_tb3d_${name}_s${SEED}_train.log 2>&1
  local ck=$(last_ckpt $PUF/experiments/t6_tb3d_${name}_s${SEED})
  [ -f "$ck" ] || { echo "RESULT ${name}=NO_CKPT"; exit 1; }
  local res=$(python3 scripts/orbital/eval_checkpoint.py "$ck" --episodes 200 $BASE_EVAL \
    --rendezvous-radius-m $br --rel-vel-tol-ms $bv --out-dir /tmp/t6_tb3d_eval_${name} 2>/dev/null \
    | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|')
  echo "$(date) $name box=${br}/${bv} heldout=$res ckpt=$ck" >> $LOG
  echo "RUNG $name heldout=$res"
  LAST=$ck
}
echo "$(date) ===== t6_tb3d seed=$SEED =====" >> $LOG
X3=$ROOT/models/t3/seed42_X3_3d_di1deg.pt
run_rung B1 25000000 10000 10 "$X3"
run_rung B2 25000000 5000 2 "$LAST"
run_rung B3 30000000 5000 1 "$LAST"
echo "RESULT tb3d seed=$SEED complete"
