#!/bin/bash
# Phase 5a multi-seed sweep — Investigation A (6 candidates × 5 seeds).
# Sequential because we're CPU-bound on one machine. Each stage emits one
# log line on completion so a Monitor can track progress.

set -u
SEEDS=(42 1337 20260423 2718 31415)
STAGE1_STEPS=10000000
STAGE2_STEPS=15000000
LOG=/tmp/p5a_sweep.log
echo "$(date) START p5a_sweep" > $LOG

PUFFER=/Users/pete/space_training/pufferlib
ROOT=/Users/pete/space_training

cd $PUFFER

build() {
    python3 setup.py build_ext --inplace --force 2>&1 | grep -E "orbital.*\.so" | tail -1 >> $LOG
}

train() {
    local tag="$1"; shift
    local timesteps="$1"; shift
    local phase_max="$1"; shift
    local e_max="$1"; shift
    local extra="$@"
    echo "$(date) START $tag steps=$timesteps phase=$phase_max e=$e_max" >> $LOG
    puffer train puffer_orbital --train.seed "${SEED}" --train.total-timesteps $timesteps \
        --train.device cpu --env.init-phase-gap-max $phase_max --env.e-max-target $e_max \
        --tag $tag $extra > /tmp/p5a_$tag.log 2>&1
    local dir=$(ls -td $PUFFER/experiments/puffer_orbital_*/ 2>/dev/null | head -1)
    local ckpt=$(ls -t $dir/model_puffer_orbital_*.pt 2>/dev/null | head -1)
    local perf=$(grep -oE "perf +[0-9.]+" /tmp/p5a_$tag.log | awk '{print $2}' | sort -n | tail -1)
    echo "$(date) DONE $tag perf_max=$perf ckpt=$ckpt" >> $LOG
    echo "$ckpt"
}

# ===== CANONICAL RECIPE PHASE (LVLH on, shaping on) =====
echo "$(date) --- CANONICAL Stage 1 runs (shared by square/tall/wide) ---" >> $LOG
declare -A CANON_S1
for SEED in "${SEEDS[@]}"; do
    ckpt=$(train "p5a_canon_s1_seed${SEED}" $STAGE1_STEPS 0.5236 0.05)
    CANON_S1[$SEED]=$ckpt
done

echo "$(date) --- FRESH-AT-TARGET runs (no curriculum) ---" >> $LOG
declare -A FRESH_S1
for SEED in "${SEEDS[@]}"; do
    ckpt=$(train "p5a_fresh_seed${SEED}" $STAGE1_STEPS 3.14159 0.30)
    FRESH_S1[$SEED]=$ckpt
done

echo "$(date) --- SQUARE Stage 2 (warm from canon S1) ---" >> $LOG
for SEED in "${SEEDS[@]}"; do
    train "p5a_square_s2_seed${SEED}" $STAGE2_STEPS 1.5708 0.10 --load-model-path "${CANON_S1[$SEED]}"
done

echo "$(date) --- TALL Stage 2 (warm from canon S1) ---" >> $LOG
for SEED in "${SEEDS[@]}"; do
    train "p5a_tall_s2_seed${SEED}" $STAGE2_STEPS 1.5708 0.05 --load-model-path "${CANON_S1[$SEED]}"
done

echo "$(date) --- WIDE Stage 2 (warm from canon S1) ---" >> $LOG
for SEED in "${SEEDS[@]}"; do
    train "p5a_wide_s2_seed${SEED}" $STAGE2_STEPS 0.5236 0.10 --load-model-path "${CANON_S1[$SEED]}"
done

# ===== minus-LVLH variant =====
echo "$(date) --- BUILDING minus-LVLH variant ---" >> $LOG
cp $PUFFER/pufferlib/ocean/orbital/orbital.h $PUFFER/pufferlib/ocean/orbital/orbital.h.bak
cp $PUFFER/pufferlib/ocean/orbital/orbital.py $PUFFER/pufferlib/ocean/orbital/orbital.py.bak
python3 - <<'PY' >> $LOG
p='/Users/pete/space_training/pufferlib/pufferlib/ocean/orbital/orbital.h'
lines=open(p).readlines()
# OBS_DIM line
for i,l in enumerate(lines):
    if l.startswith('#define OBS_DIM'):
        lines[i]='#define OBS_DIM     33              /* Phase 5a A: minus-LVLH */\n'
        break
# Strip LVLH block — find start at "[33-37] LVLH-frame", end at the closing brace of the LVLH `{...}` block (one indent deeper than function)
start=end=None
for i,l in enumerate(lines):
    if start is None and '[33-37] LVLH-frame relative state' in l:
        start=i
    elif start is not None and l.rstrip()=='    }':
        end=i; break
assert start is not None and end is not None, f'LVLH block not found: start={start}, end={end}'
new=lines[:start] + ['    /* Phase 5a A minus-LVLH: block stripped. */\n'] + lines[end+1:]
open(p,'w').writelines(new)
print(f'LVLH block stripped (lines {start+1}-{end+1}, {end-start+1} lines removed)')
PY
sed -i.bak2 's|shape=(38,)|shape=(33,)|' $PUFFER/pufferlib/ocean/orbital/orbital.py
build

echo "$(date) --- minus-LVLH Stage 1 runs ---" >> $LOG
declare -A NOLVLH_S1
for SEED in "${SEEDS[@]}"; do
    ckpt=$(train "p5a_nolvlh_s1_seed${SEED}" $STAGE1_STEPS 0.5236 0.05)
    NOLVLH_S1[$SEED]=$ckpt
done

echo "$(date) --- minus-LVLH Stage 2 runs ---" >> $LOG
for SEED in "${SEEDS[@]}"; do
    train "p5a_nolvlh_s2_seed${SEED}" $STAGE2_STEPS 1.5708 0.10 --load-model-path "${NOLVLH_S1[$SEED]}"
done

# ===== Restore LVLH, build minus-shaping =====
echo "$(date) --- RESTORING LVLH, BUILDING minus-shaping variant ---" >> $LOG
mv $PUFFER/pufferlib/ocean/orbital/orbital.h.bak $PUFFER/pufferlib/ocean/orbital/orbital.h
mv $PUFFER/pufferlib/ocean/orbital/orbital.py.bak $PUFFER/pufferlib/ocean/orbital/orbital.py
rm -f $PUFFER/pufferlib/ocean/orbital/orbital.py.bak2
cp $PUFFER/pufferlib/ocean/orbital/orbital.h $PUFFER/pufferlib/ocean/orbital/orbital.h.bak
sed -i '' 's|^#define BETA_SHAPE   1.0.*|#define BETA_SHAPE   0.0    /* Phase 5a A: minus-shaping */|' $PUFFER/pufferlib/ocean/orbital/orbital.h
build

echo "$(date) --- minus-shaping Stage 1 runs ---" >> $LOG
declare -A NOSHAPE_S1
for SEED in "${SEEDS[@]}"; do
    ckpt=$(train "p5a_noshape_s1_seed${SEED}" $STAGE1_STEPS 0.5236 0.05)
    NOSHAPE_S1[$SEED]=$ckpt
done

echo "$(date) --- minus-shaping Stage 2 runs ---" >> $LOG
for SEED in "${SEEDS[@]}"; do
    train "p5a_noshape_s2_seed${SEED}" $STAGE2_STEPS 1.5708 0.10 --load-model-path "${NOSHAPE_S1[$SEED]}"
done

# ===== Restore canonical =====
echo "$(date) --- RESTORING canonical recipe ---" >> $LOG
mv $PUFFER/pufferlib/ocean/orbital/orbital.h.bak $PUFFER/pufferlib/ocean/orbital/orbital.h
rm -f $PUFFER/pufferlib/ocean/orbital/orbital.h.bak2 $PUFFER/pufferlib/ocean/orbital/orbital.py.bak $PUFFER/pufferlib/ocean/orbital/orbital.py.bak2
build

echo "$(date) DONE p5a_sweep" >> $LOG
