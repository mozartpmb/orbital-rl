#!/bin/bash
# A.A3: Phase 4 conditions multi-seed. e=0, no valid_init_only, 200 eps × 5 seeds.
set -uo pipefail
PUFFER=/Users/pete/space_training/pufferlib
RESULTS=/tmp/p5closure_phase4.csv
echo "seed_dir,phase4_success_n,total" > $RESULTS

CKPTS="
experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt
experiments/puffer_orbital_177765655537/model_puffer_orbital_000375.pt
experiments/puffer_orbital_177765658166/model_puffer_orbital_000250.pt
experiments/puffer_orbital_177765658729/model_puffer_orbital_000350.pt
experiments/puffer_orbital_177765659007/model_puffer_orbital_000275.pt"

cd $PUFFER
for CKPT in $CKPTS; do
  LABEL=$(echo $CKPT | grep -oE 'puffer_orbital_[0-9]+' | head -1)
  RES=$(python3 scripts/orbital/eval_checkpoint.py $CKPT \
    --episodes 200 --e-max-target 0.0 --e-max-sat 0.0 \
    --init-phase-gap-max 3.14159 --valid-init-only 0 \
    --out-dir /tmp/p5closure_phase4_${LABEL} --seed 42 2>&1 | grep "Success rate" | head -1)
  N=$(echo $RES | grep -oE '[0-9]+/[0-9]+' | head -1)
  SUCC=$(echo $N | cut -d/ -f1); TOT=$(echo $N | cut -d/ -f2)
  echo "$LABEL,$SUCC,$TOT" >> $RESULTS
  echo "$LABEL phase4: $RES"
done
echo "DONE"
