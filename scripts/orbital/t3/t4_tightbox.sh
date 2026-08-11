#!/bin/bash
# T4-D — tight-success-box ladder at LEO, Discrete-20 (fine radial ±1 m/s).
#   TB1: fresh D20 — e=0, same-orbit, ±180°, 500–800 km, box 30 km/50 m/s
#   TB2: headline (e≤0.05 both, 300–800 km),              box 30 km/50 m/s
#   TB3: headline,                                        box 10 km/10 m/s
#   TB4: headline,                                        box  5 km/ 2 m/s
#   TB5: headline,                                        box  5 km/ 1 m/s
# Quanta context (recon §4): 16-action best |v_rel| floor was 5.02 m/s (10 m/s
# radial quantum binding); with radial ±1 m/s the floor is 0.71 m/s, so TB5 is
# reachable at ~1.4× margin. Each rung 50M warm from the previous; held-out
# eval (seed 123, 200 eps) at the rung's own box. LEO obs scales; dv_ref 300.
# Chain-private data_dir — safe to run concurrently with other chains.
set -uo pipefail
ROOT=/Users/pete/space_training
PUFFER=$ROOT/pufferlib
SEED=${1:-42}
LOG=/tmp/t4_tb_s${SEED}.log
EVAL_SEED=123
DATA_DIR=experiments/t4_tb_s${SEED}

BASE="--env.shaping-mode 1 --env.shape-gamma 1.0 --env.phase-gap-mode 1 \
--env.phase-obs-mode 1 --env.cap-terminal-reward 0.0 --env.valid-init-only 1 \
--env.init-phase-gap-max 3.14159 --env.episode-cap-steps 3000 \
--env.legacy-action-space 20"
BASE_EVAL="--init-phase-gap-max 3.14159 --valid-init-only 1 --seed $EVAL_SEED \
--shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 \
--cap-terminal-reward 0.0 --episode-cap-steps 3000 --legacy-action-space 20"

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
        --train.checkpoint-interval 20 $warm_flag --tag t4_${name}_s${SEED} \
        > /tmp/t4_${name}_s${SEED}_train.log 2>&1
    local dir=$(get_latest_dir)
    if [ "$dir" = "$before" ]; then
        echo "$(date) ABORT $name: no new dir" >> $LOG; echo "RESULT $name=NO_NEW_DIR"; exit 1
    fi
    local ckpt=$(last_ckpt "$dir")
    local res=$(python3 scripts/orbital/eval_checkpoint.py "$ckpt" --episodes 200 \
        $BASE_EVAL $eflags --out-dir /tmp/t4_tb_eval_${name} 2>/dev/null \
        | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+) .*|\1|')
    echo "$(date) $name seed=$SEED dir=$(basename $dir) ckpt=$(basename $ckpt) heldout=$res" >> $LOG
    echo "RUNG $name dir=$(basename $dir) heldout=$res"
    LAST_CKPT=$ckpt
}

echo "$(date) ===== t4_tightbox seed=$SEED =====" >> $LOG
LAST_CKPT=none

run_rung TB1 \
  "--env.same-orbit-init 1 --env.e-max-target 0.0 --env.e-max-sat 0.0 --env.a-min-override 6.871e6 --env.a-max-override 7.171e6" \
  "--same-orbit-init 1 --e-max-target 0.0 --e-max-sat 0.0 --a-min-override 6.871e6 --a-max-override 7.171e6" \
  none

run_rung TB2 \
  "--env.e-max-target 0.05 --env.e-max-sat 0.05" \
  "--e-max-target 0.05 --e-max-sat 0.05" \
  "$LAST_CKPT"

run_rung TB3 \
  "--env.e-max-target 0.05 --env.e-max-sat 0.05 --env.rendezvous-radius-m 10000 --env.rel-vel-tol-ms 10" \
  "--e-max-target 0.05 --e-max-sat 0.05 --rendezvous-radius-m 10000 --rel-vel-tol-ms 10" \
  "$LAST_CKPT"

run_rung TB4 \
  "--env.e-max-target 0.05 --env.e-max-sat 0.05 --env.rendezvous-radius-m 5000 --env.rel-vel-tol-ms 2" \
  "--e-max-target 0.05 --e-max-sat 0.05 --rendezvous-radius-m 5000 --rel-vel-tol-ms 2" \
  "$LAST_CKPT"

run_rung TB5 \
  "--env.e-max-target 0.05 --env.e-max-sat 0.05 --env.rendezvous-radius-m 5000 --env.rel-vel-tol-ms 1" \
  "--e-max-target 0.05 --e-max-sat 0.05 --rendezvous-radius-m 5000 --rel-vel-tol-ms 1" \
  "$LAST_CKPT"

echo "RESULT tightbox seed=$SEED complete; see $LOG"
