#!/bin/bash
# X3 multi-seed: per seed, fresh XA smoke (di 0.40deg, e=0, 12M) then X3 warm (di 1.0deg, headline, 50M).
set -uo pipefail
WT=/Users/pete/space_training-ext3d
PUF=$WT/pufferlib
SEED=$1
LOG=/tmp/t5_x3_s${SEED}.log
RUN="python3 $WT/scripts/orbital/ext3d/puffer_wt.py train puffer_orbital --train.device cpu --train.checkpoint-interval 20 --wandb --wandb-project orbital-rl --wandb-group t5-3d-s${SEED}"
BASE="--env.shaping-mode 2 --env.shape-gamma 1.0 --env.phase-gap-mode 1 --env.phase-obs-mode 1 --env.episode-cap-steps 3000 --env.cap-terminal-reward 0.0 --env.valid-init-only 1 --env.init-phase-gap-max 3.14159 --env.dim3-mode 1 --env.shape-dv-ref-ms 700 --env.legacy-action-space 30"
last_ckpt() { ls "$1"/puffer_orbital_*/model_puffer_orbital_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1; }
cd $PUF
echo "$(date) === XA seed $SEED ===" >> $LOG
$RUN --train.seed $SEED --train.total-timesteps 12000000 --train.data-dir experiments_ext3d/xa_s${SEED} \
  $BASE --env.same-orbit-init 0 --env.e-max-target 0.0 --env.e-max-sat 0.0 \
  --env.a-min-override 6.871e6 --env.a-max-override 7.171e6 --env.di-max-rad 0.006981 \
  --tag t5_XA_s${SEED} > /tmp/t5_XA_s${SEED}_train.log 2>&1
XA=$(last_ckpt $PUF/experiments_ext3d/xa_s${SEED})
echo "$(date) XA ckpt: $XA" >> $LOG
[ -f "$XA" ] || { echo "RESULT s${SEED} XA=FAILED"; exit 1; }
echo "$(date) === X3 seed $SEED ===" >> $LOG
$RUN --train.seed $SEED --train.total-timesteps 50000000 --train.data-dir experiments_ext3d/x3_s${SEED} \
  $BASE --env.e-max-target 0.05 --env.e-max-sat 0.05 --env.di-max-rad 0.017453 \
  --load-model-path "$XA" --tag t5_X3_s${SEED} > /tmp/t5_X3_s${SEED}_train.log 2>&1
X3=$(last_ckpt $PUF/experiments_ext3d/x3_s${SEED})
[ -f "$X3" ] || { echo "RESULT s${SEED} X3=FAILED"; exit 1; }
RES=$(python3 scripts/orbital/eval_checkpoint.py "$X3" --episodes 200 --seed 123 \
  --e-max-target 0.05 --e-max-sat 0.05 --init-phase-gap-max 3.14159 --valid-init-only 1 \
  --shaping-mode 2 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 --episode-cap-steps 3000 \
  --cap-terminal-reward 0.0 --dim3-mode 1 --di-max-rad 0.017453 --shape-dv-ref-ms 700 --legacy-action-space 30 \
  --out-dir /tmp/t5_x3_eval_s${SEED} 2>/dev/null | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|')
echo "$(date) X3 s${SEED} heldout=$RES ckpt=$X3" >> $LOG
echo "RESULT s${SEED} X3=$RES"
