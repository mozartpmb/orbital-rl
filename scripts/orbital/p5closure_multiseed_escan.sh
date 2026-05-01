#!/bin/bash
# Multi-seed eccentricity scan: 5 seeds × 6 e_max levels × 200 eps each.
set -uo pipefail
PUFFER=/Users/pete/space_training/pufferlib
RESULTS=/tmp/p5closure_escan.csv
echo "seed_dir,e_max,success_n,total" > $RESULTS

# 5 seed Stage 4.0 ckpts identified earlier
SEED_42_CKPT="experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt"
SEED_A_CKPT="experiments/puffer_orbital_177765655537/model_puffer_orbital_000375.pt"
SEED_B_CKPT="experiments/puffer_orbital_177765658166/model_puffer_orbital_000250.pt"
SEED_C_CKPT="experiments/puffer_orbital_177765658729/model_puffer_orbital_000350.pt"
SEED_D_CKPT="experiments/puffer_orbital_177765659007/model_puffer_orbital_000275.pt"

cd $PUFFER
for SEED_DIR in $SEED_42_CKPT $SEED_A_CKPT $SEED_B_CKPT $SEED_C_CKPT $SEED_D_CKPT; do
  for E in 0.05 0.10 0.20 0.30 0.50 0.70; do
    LABEL=$(echo $SEED_DIR | grep -oE 'puffer_orbital_[0-9]+' | head -1)
    OUT=/tmp/p5closure_escan_${LABEL}_e${E}
    RES=$(python3 scripts/orbital/eval_checkpoint.py $SEED_DIR \
      --episodes 200 --e-max-target $E --e-max-sat $E \
      --init-phase-gap-max 3.14159 --valid-init-only 1 \
      --out-dir $OUT --seed 42 2>&1 | grep "Success rate" | head -1)
    N=$(echo $RES | grep -oE '[0-9]+/[0-9]+' | head -1)
    SUCC=$(echo $N | cut -d/ -f1); TOT=$(echo $N | cut -d/ -f2)
    echo "$LABEL,$E,$SUCC,$TOT" >> $RESULTS
    echo "$LABEL e=$E -> $RES"
  done
done
echo "DONE — results in $RESULTS"
