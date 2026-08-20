#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# T11-CONSOL PROBE SIDECAR — the trajectory, not the endpoints
# ═══════════════════════════════════════════════════════════════════════════
# Runs BESIDE t11_consol_campaign.sh, never inside it. The campaign's mid/final
# batteries answer "did it hold"; this answers "when did it move" — which is
# the half a 50M/100M pair cannot give: both prior collapses COMPLETED within
# 50M, and warm-start fine-tuning does its damage in the earliest updates
# (fresh value error on the re-weighted mixture backprops through the shared
# trunk hardest at the start). So: dense early probes, then ~10M cadence.
#
#   probe epochs: 20, 40 (the early-distortion window: 2.6M, 5.2M)
#                 then 80, 160, 240, 320, 400, 480, 560, 640, 720 (~10.5M grid)
#   per probe:    6 active cells x 20 eps, native bearings-only, seed 123
#                 (W1 skipped: excluded from training, 0.0 in all seesaw
#                  states; probing it 11 times buys nothing at 22000 cap)
#   retention:    every probed ckpt copied to web_data/.../probe_ckpts/ —
#                 enables rollback to the last tight-good ckpt AND offline
#                 per-cell KL(pi_t || pi_root) analysis after the run
#
# PRE-REGISTERED ALARMS (advisory — this sidecar never kills the trainer):
#   PROBE ALARM  tight < 65% (~70% of the root's 92.5) on TWO consecutive
#                probes -> the seesaw is repeating at 1e-3; decision point is
#                rollback-to-last-tight-good, not waiting for the mid battery.
#   PROBE NOTE   at >= 40M, E3 still <= 15% -> red-team (a) shaping up: LR too
#                low to RE-ACQUIRE; expected read is extend/anneal or the
#                pre-decided 2e-3 fallback — NOT a capacity conclusion.
#
# PRE-REGISTERED NON-FAILURE (decided before launch, so it cannot be
# improvised into a redesign later): "no seesaw but incomplete wide-cell
# recovery at 100M" is the stability/plasticity trade playing out as designed
# — the answer is extend/anneal, not a new mixture.
#
# Runs nice -n 15, single-threaded. ~20 min/round, ~11 rounds over a 9.6 h
# run: one polite core, no meaningful SPS cost to the 8-worker trainer.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
EVAL=$MAIN/scripts/orbital/extj2/t11_eval.py
# All paths/patterns are env-overridable so the same sidecar can serve any
# campaign of this family (T13 consol, T13b anchor, ...) without collisions.
ARM_DIR=${T11C_ARM_DIR:-$PUF/experiments_t11_consol/consol}
PROBE_DIR=${T11C_PROBE_DIR:-$MAIN/web_data/results/t11_consol/probes}
CKPT_KEEP=$PROBE_DIR/probe_ckpts
LOG=${T11C_PROBE_LOG:-/tmp/t11c_probe.log}
CAMPAIGN_PROG=${T11C_CAMPAIGN_PROG:-/tmp/t11c_progress.log}
COMPLETE_PAT=${T11C_COMPLETE_PAT:-'t11-consol campaign COMPLETE'}
PGREP_PAT=${T11C_PGREP:-'experiments_t11_consol|t11_consol_campaign'}
EPS=${T11C_PROBE_EPS:-20}
SEED=123
CELLS="TIGHT_5k1 E0_j2 E1_j2 E2_j2 E3_j2 LONGRANGE"
PROBE_EPOCHS="20 40 80 160 240 320 400 480 560 640 720"
TIGHT_ALARM=65          # ~70% of the root's 92.5
E3_STALL=15             # vs 6.5 baseline
E3_STALL_EP=305         # ~40M

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
mkdir -p "$PROBE_DIR" "$CKPT_KEEP"

rate_of() {
    python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try: print(int(round(100*json.load(open(sys.argv[1]))['rate'])))
except Exception: print("?")
PY
}

ckpt_for_epoch() {   # first ckpt at/past the epoch, sorted on extracted epoch
    local ep="$1" f n
    for f in $(ls "$ARM_DIR"/*/model_puffer_orbital_nav_*.pt 2>/dev/null \
               | sed 's/.*_\([0-9][0-9]*\)\.pt$/\1 &/' | sort -n | cut -d' ' -f2-); do
        n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$ep" ] && { echo "$f"; return 0; }
    done
    return 0
}

campaign_alive() {
    grep -qF "$COMPLETE_PAT" "$CAMPAIGN_PROG" 2>/dev/null && return 1
    pgrep -f "$PGREP_PAT" >/dev/null 2>&1 && return 0
    return 1
}

say "=========== probe sidecar start (pid $$) ==========="
say "grid: epochs $PROBE_EPOCHS ($EPS eps/cell, cells: $CELLS)"
say "alarms: tight<${TIGHT_ALARM}% x2 consecutive = PROBE ALARM; E3<=${E3_STALL}% at ep>=${E3_STALL_EP} = PROBE NOTE (red-team a)"

prev_tight=999
for EP in $PROBE_EPOCHS; do
    # wait for a ckpt at/past this epoch (or campaign end)
    while :; do
        CK=$(ckpt_for_epoch "$EP")
        [ -n "$CK" ] && break
        campaign_alive || { say "campaign over before epoch $EP — sidecar done"; exit 0; }
        sleep 120
    done
    N=${CK##*_}; N=${N%.pt}; N=$((10#$N))
    STEPS_M=$(( N * 131072 / 1000000 ))
    cp -n "$CK" "$CKPT_KEEP/model_ep$(printf '%06d' "$N").pt" 2>/dev/null
    LINE=""
    for C in $CELLS; do
        OUT=$PROBE_DIR/probe_e${N}_${C}.json
        if [ ! -f "$OUT" ]; then
            nice -n 15 python3 "$EVAL" --ckpt "$CK" --cell "$C" \
                --nav-mode bearings_only --episodes "$EPS" --seed "$SEED" \
                --label "probe_e${N}_${C}" --out "$OUT" \
                > "/tmp/t11c_probe_e${N}_${C}.log" 2>&1
        fi
        LINE="$LINE ${C%%_*}=$(rate_of "$OUT")"
    done
    say "PROBE e${N} (~${STEPS_M}M):$LINE  (${EPS} eps, root: TIGHT=92 E0=97 E1=96 E2=29 E3=6 LONG=40)"
    T=$(rate_of "$PROBE_DIR/probe_e${N}_TIGHT_5k1.json")
    if [ "$T" != "?" ] && [ "$T" -lt "$TIGHT_ALARM" ]; then
        if [ "$prev_tight" != "?" ] && [ "$prev_tight" -lt "$TIGHT_ALARM" ]; then
            say "PROBE ALARM: tight ${prev_tight}% -> ${T}% — two consecutive probes under ${TIGHT_ALARM}%."
            say "  The seesaw is repeating at 1e-3. Decision point: last tight-good ckpt is in $CKPT_KEEP."
        else
            say "PROBE WARN: tight ${T}% under ${TIGHT_ALARM}% (first occurrence — alarm on a second)."
        fi
    fi
    prev_tight=$T
    E3=$(rate_of "$PROBE_DIR/probe_e${N}_E3_j2.json")
    if [ "$N" -ge "$E3_STALL_EP" ] && [ "$E3" != "?" ] && [ "$E3" -le "$E3_STALL" ]; then
        say "PROBE NOTE: E3 ${E3}% at ~${STEPS_M}M — red-team (a): LR may be too low to RE-ACQUIRE."
        say "  This is an LR finding, not capacity. Pre-decided fallback: T11C_LR=2e-3, then staged."
    fi
done
say "=========== probe sidecar done (grid exhausted) ==========="
