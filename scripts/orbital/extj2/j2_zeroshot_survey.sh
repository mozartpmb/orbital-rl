#!/bin/bash
# Reproduces every zero-shot number quoted in J2_RUNG_NOTES.md §2.2/§2.3/§3.2.
# Eval only — launches no training, writes nothing outside /tmp and the JSON dir.
#
# ALL CLAIMS ARE IN MEAN ELEMENTS (j2_A_design §1.4). Under j2_mode=1 the env's
# state IS the mean element set; "success" means the MEAN elements are inside
# the box. Benign at 30 km / 50 m/s; NOT an osculating-grade claim at 5 km/1 m/s.
#
#   bash scripts/orbital/extj2/j2_zeroshot_survey.sh
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
export PYTHONPATH=$ROOT/pufferlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
E="python3 $ROOT/scripts/orbital/extj2/j2_rung_eval.py --episodes ${EPS:-200} --seed 123"
X3=$ROOT/models/t3/seed42_X3_3d_di1deg.pt
TB5=$ROOT/models/t3/seed42_TB3D_box5k1.pt
TB1=$ROOT/models/t3/seed42_TB3D_box10k10.pt
Q() { grep -v "^Gym has\|^Please upgrade\|^See the migration\|^\[orbital/j2\] WARNING" ; }

echo "### A. J2 gap vs BOX TIGHTNESS (i_t = 0 => plane channel provably inert)"
echo "###    each box with its OWN parent ckpt so the control sits at ceiling"
for J in 0 1; do
  $E --ckpt "$X3"  --label "A_box30km-50ms_j2=$J"  --j2-mode $J --i-band off \
     --rendezvous-radius-m 30000 --rel-vel-tol-ms 50 --shape-w-match 0.8166667 2>&1 | Q | head -2
done
for J in 0 1; do
  $E --ckpt "$TB1" --label "A_box10km-10ms_j2=$J"  --j2-mode $J --i-band off \
     --rendezvous-radius-m 10000 --rel-vel-tol-ms 10 --shape-w-match 0.35 2>&1 | Q | head -2
done
for J in 0 1; do
  $E --ckpt "$TB5" --label "A_box5km-1ms_j2=$J"    --j2-mode $J --i-band off \
     --rendezvous-radius-m 5000  --rel-vel-tol-ms 1  --shape-w-match 0.35 2>&1 | Q | head -2
done

echo
echo "### B. INCLINATION BAND sweep at the loose box (lvlh_frame_mode=1, raan fixed)"
for B in 5,10 20,30 20,45 25,50 30,60 60,80 80,89; do
  for J in 0 1; do
    $E --ckpt "$X3" --label "B_band($B)_j2=$J" --j2-mode $J --i-band "$B" \
       --lvlh-frame-mode 1 2>&1 | Q | head -2
  done
done

echo
echo "### C. THE LVLH CONFOUND — same scenarios, only the obs frame differs"
for L in 0 1; do
  for J in 0 1; do
    $E --ckpt "$X3" --label "C_band30-60_lvlh${L}_j2${J}" --j2-mode $J \
       --i-band 30,60 --lvlh-frame-mode $L 2>&1 | Q | head -2
  done
done

echo
echo "### D. THE SO(2) LEAK — raan sampled vs fixed, everything else identical"
for R in 0 1; do
  $E --ckpt "$X3" --label "D_raansample=${R}_j2=1" --j2-mode 1 --i-band 30,60 \
     --lvlh-frame-mode 1 --raan-sample $R 2>&1 | Q | head -2
done
