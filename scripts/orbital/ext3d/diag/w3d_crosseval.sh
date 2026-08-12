#!/bin/bash
# ext-3d wide lineage diagnostics: cross-evaluate the W3 / W4 checkpoints on
# each other's configs and on a di_max=0 control.
#
# The di=0 control matters because the de_max disc is applied to the
# NODE-RELATIVE (e cos w, e sin w) 2-vector and the plane rotation then sets
# RAAN ~ U(0,2pi), randomising the INERTIAL periapsis longitude. At di_max = 0
# no rotation happens, so the knob is honoured and the same policy sees the
# in-plane task the ladder intended. Everything else is held fixed.
#
# EVAL ONLY. No training is launched.
set -uo pipefail
PUF=/Users/pete/space_training-ext3d/pufferlib
cd $PUF
W3CK=$(ls experiments_ext3d/w3d_W3_s42/*/model_puffer_orbital_*.pt | sort -t_ -k4 -n | tail -1)
W4CK=$(ls experiments_ext3d/w3d_W4_s42/*/model_puffer_orbital_*.pt | sort -t_ -k4 -n | tail -1)
BASE="--episodes 200 --seed 123 --init-phase-gap-max 3.14159 --valid-init-only 1 \
--shaping-mode 2 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 \
--cap-terminal-reward 0.0 --dim3-mode 1 --shape-dv-ref-ms 700 \
--legacy-action-space 30 --obs-alt-scale-m 8e6 --lvlh-scale-m 1.5e7"
CFG_W3="--e-max-target 0.15 --de-max 0.06 --da-max-m 400e3 --a-min-override 6.671e6 --a-max-override 8.371e6 --episode-cap-steps 3000"
CFG_W4="--e-max-target 0.30 --de-max 0.08 --da-max-m 600e3 --a-min-override 6.671e6 --a-max-override 14.371e6 --episode-cap-steps 6000"
DI1="--di-max-rad 0.017453"   # 1.0 deg
DI0="--di-max-rad 0.0"        # plane rotation off -> de_max knob honoured
DI04="--di-max-rad 0.006981"  # 0.40 deg

run() { # label ckpt cfg di outdir
  local lab=$1 ck=$2 cfg=$3 di=$4 out=$5
  local r
  r=$(python3 scripts/orbital/eval_checkpoint.py "$ck" $BASE $cfg $di --out-dir "$out" 2>/dev/null \
      | grep "Physical success" | sed -E 's|.* ([0-9]+/[0-9]+).*|\1|')
  echo "CROSSEVAL $lab = $r"
}

run "W3ck@W3cfg_di1.0_(reference)" "$W3CK" "$CFG_W3" "$DI1"  /tmp/xe_w3_w3_di1
run "W3ck@W3cfg_di0.0_(knob-honoured)" "$W3CK" "$CFG_W3" "$DI0"  /tmp/xe_w3_w3_di0
run "W3ck@W3cfg_di0.4" "$W3CK" "$CFG_W3" "$DI04" /tmp/xe_w3_w3_di04
run "W4ck@W3cfg_di1.0_(REGRESSION)" "$W4CK" "$CFG_W3" "$DI1"  /tmp/xe_w4_w3_di1
run "W3ck@W4cfg_di1.0_(parent-on-child)" "$W3CK" "$CFG_W4" "$DI1"  /tmp/xe_w3_w4_di1
run "W4ck@W4cfg_di0.0_(knob-honoured)" "$W4CK" "$CFG_W4" "$DI0"  /tmp/xe_w4_w4_di0
run "W3ck@W4cfg_di0.0_(knob-honoured)" "$W3CK" "$CFG_W4" "$DI0"  /tmp/xe_w3_w4_di0
