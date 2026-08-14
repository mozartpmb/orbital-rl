#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# J2 x ANGLES-ONLY NAV — the two hardest realism axes at once
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT:
#   nohup caffeinate -is bash scripts/orbital/nav/j2nav_campaign.sh \
#       > /tmp/j2nav_stdout.log 2>&1 &
#
# Rung: X3's task and box (30 km / 50 m/s), plus J2 secular drift and an
# inclined target band i_t ~ U(30,60) deg with lvlh_frame_mode=1. That is
# exactly the box A2_j2trained was trained at, so the J2-root floor below is
# measured at its OWN rung and not a harder one.
#
# ── THE FILTER SPEC WAS DECIDED BY MEASUREMENT, NOT CHOSEN ──────────────────
# `nav_j2_mode=1` selects nav_math3d.BatchedBearingMSC6J2 — secular J2 in BOTH
# the state and the covariance STM (finite difference, 12 propagations). From
# the ext-j2 head-to-head (N=128, truth J2 in every arm), at the 24 h operating
# point against a matched two-body control at 458.0 m / NEES 1.00:
#
#   FIXED   J2 state, two-body cov    1202.7 m (2.63x)  NEES 11.37   21.9% band
#   C1-an   J2 state, analytic J2 cov 1234.5 m (2.70x)  NEES 11.46   18.8%
#   C1-fd   J2 state, FD J2 cov        457.3 m (1.00x)  NEES  1.01   87.5%
#
# The point that makes this non-negotiable: putting J2 in the state alone does
# not merely leave the filter overconfident — the overconfident P shrinks the
# Kalman gain, so it ALSO pays 2.63x in position. OrbitalNav therefore REFUSES
# j2_mode=1 with nav_j2_mode=0 rather than letting that arm be run by accident.
# The analytic O(J2) STM removes only ~59% of the two-body STM's error and is
# refused outright by the filter class.
#
# Cost: the probe projected ~217% of a nav step from predict() alone
# (0.603 -> 3.072 ms/tick, 5.09x). The INTEGRATED cost is higher and the
# difference is a real finding, not an accounting error: the wrapper also
# sub-propagates BOTH TRUTH states between decision epochs, and that had to
# become J2-aware too (see the mechanism note below), which the probe could not
# have known because it never ran the wrapper. Measured end to end at B=256,
# interleaved against the two-body arm in the same process:
#
#     X3  (two-body filter)  188.4 ms/step
#     J2X (J2 filter)        837.7 ms/step      ratio 4.45x
#
# So budget ~4.5x a two-body nav arm, i.e. ~7.5 h per 50M arm against rung-1's
# ~101 min, not the ~3.6 h the predict-only projection implied. Absolute
# numbers were taken on a contended machine; the RATIO is the robust part and
# is what the schedule should be built on.
#
# ── THE MECHANISM THAT COST A DAY, WRITTEN DOWN SO IT IS NOT REDISCOVERED ───
# The filter was consistent standalone (NEES ~2 through 24 h) and wildly
# inconsistent in the loop (NEES 25520, azimuth innovation ramping to +38
# sigma). The cause was NOT the filter: its J2 propagation matched the env to
# 0.0 m at 6 h, its FD STM sat 4e-3 from the two-body STM exactly as O(J2)
# predicts, and re-poling the chart EVERY tick changed nothing (the chart is a
# shared basis, so a stale one cancels).
#
# The cause was that `OrbitalNav._nav_step` sub-propagates the truth states
# between decision epochs with the TWO-BODY propagator. tau reaches 360
# sub-steps (6 h), where two-body drifts ~132 km from the env's own J2 truth,
# so the filter was being fed measurements built from a fiction. It is now
# J2-aware when nav_j2_mode=1, which restores in-loop NEES to 1.20 acquired /
# 1.37 at handoff against the two-body reference of 0.66 / 1.02, and flattens
# the azimuth innovation to mean -0.05 +/- 1.0 across 40 h.
#
# The general lesson for any future perturbation: a filter is only as good as
# the truth the harness hands it, and filter-vs-filter validation cannot see a
# harness that is wrong in the same way for both arms.
#
# NOT TAKEN: the element-space covariance optimisation. Identified but
# unmeasured; a mid-campaign swap of the covariance path is exactly the
# compound change this project keeps being burned by.
#
# ── ACQUISITION SURROGATE: UNCHANGED, AND HERE IS ITS BOUNDARY ──────────────
# Measured safe at the shipped 46-minute floor under J2: sigma ratio 1.0002,
# 0/128 decisions differ. It is NOT re-derived here and does not need to be.
#
# RE-MEASURE IT if any arm allows blind windows of roughly 3 h or more. The
# surrogate runs OPTIMISTIC on long arcs — 3.3% at 6 h — and optimistic is the
# dangerous direction: it would hand the policy an acquisition the real solver
# cannot deliver, and the failure would look like a policy result. Every arm
# below acquires at the 46-minute floor, so the boundary is not approached;
# a rung that adds long coasts, a re-acquisition after an extended warp, or a
# tighter gate must re-run scripts/orbital/extj2/j2_acq_surrogate_probe.py
# before its numbers mean anything.
#
# ── THE ASYMMETRY THIS CAMPAIGN EXISTS TO MEASURE ───────────────────────────
# Two roots, each skilled in one axis and naive in the other:
#   NAV root  models/t3/n3dnav_T-BO3.pt        (bearings-only, two-body)
#   J2  root  models/t3/extj2_A2_j2trained.pt  (J2-skilled, truth-state)
# Stage 1 floors BOTH under the combined rung, before any training. Which skill
# is harder to retrofit is the scientific question; the negative-transfer
# history (rung 2's control transferred NOTHING to the blind problem, 13-18%)
# predicts the J2 root floors low under bearings-only. Stage 3 is gated on that
# number so the campaign does not spend 50M re-running rung 1's floor result.
#
# ── WARM START: NO DERIVATION NEEDED, AND THAT IS VERIFIED ──────────────────
# Under j2_mode the newly-live observation slots are obs[29] = cos(i_sat) and
# obs[30] = cos(i_target) (orbital.h:1319-1320); 31-32 stay hard-zeroed.
# n3dnav_T-BO3.pt's encoder columns 29-32 are ALREADY EXACTLY ZERO — measured
# max|w| = 0.000000 on all four — because it inherited them zeroed from
# n3dnav_warm_X3.pt and obs[29-32] were identically 0 throughout its 50M, so
# they took no gradient. The standard column discipline is therefore already
# satisfied and the root is used unmodified. Verified rather than assumed: it
# reproduces its published 200/200 with a bit-identical action stream, md5
# 8df04e6ba5e4, on the ported code.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
NAV_ROOT=$MAIN/models/t3/n3dnav_T-BO3.pt
J2_ROOT=$MAIN/models/t3/extj2_A2_j2trained.pt
PROG=/tmp/j2nav_progress.log
JSON_DIR=$MAIN/web_data/results/j2nav
EVAL=$MAIN/scripts/orbital/nav/eval_relnav3d.py
PORTCHK=$MAIN/scripts/orbital/nav/j2nav_portcheck.py
VERIFY_J2=$MAIN/scripts/orbital/extj2/verify_extj2.py
EPS=200
EVAL_SEED=123
STEPS=50000000
FINAL_CKPT_TAG=000382
WATCHDOG_S=900
# Stage 3 runs only if the J2 root floors at or above this under bearings-only.
# Below it, stage 3 is rung-1's floor experiment again with a different warm
# start, and 50M is too expensive to spend confirming a known negative.
STAGE3_FLOOR_MIN=30

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

STAGES=${J2NAV_STAGES:-0,1,2,3}
want() { [[ ",$STAGES," == *",$1,"* ]]; }
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

BASE_ENV=(
  --env.num-debris-min 0 --env.num-debris-max 0
  --env.e-max-target 0.05 --env.e-max-sat 0.05 --env.same-orbit-init 0
  --env.init-phase-gap-max 3.14159 --env.valid-init-only 1
  --env.gave-up-action terminate --env.max-valid-init-attempts 4096
  --env.obs-alt-scale-m 1.6e6 --env.lvlh-scale-m 6.371e6
  --env.rendezvous-radius-m 30000.0 --env.rel-vel-tol-ms 50.0
  --env.shaping-mode 2 --env.shape-w-lambda 1.0 --env.shape-w-match 0.8166667
  --env.shape-dv-ref-ms 700.0 --env.shape-gamma 1.0
  --env.phase-gap-mode 1 --env.phase-obs-mode 1
  --env.episode-cap-steps 3000 --env.cap-terminal-reward 0.0
  --env.dim3-mode 1 --env.di-max-rad 0.017453 --env.legacy-action-space 30
)
NAV_ENV=(
  --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0
  --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20
  --env.nav-acq-mode crlb_online --env.nav-max-ticks 0
)
J2_ENV=(
  --env.j2-mode 1 --env.nav-j2-mode 1
  --env.i-target-min-rad 0.5235987755982988    # 30 deg
  --env.i-target-max-rad 1.0471975511965976    # 60 deg
  --env.raan-target-sample 0 --env.lvlh-frame-mode 1
)

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: branch '$br', need '$BRANCH_REQ'"; return 1; }
    for f in "$NAV_ROOT" "$J2_ROOT" "$EVAL" "$PORTCHK"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    mkdir -p "$JSON_DIR"; return 0
}

# ── stage 0: three anchors, all of which must pass ──────────────────────────
anchors() {
    local ok=1
    say "START stage 0 anchors"
    cd "$PUF" || return 1

    # (a) nav byte-transparency at THIS config with J2 off — the 19-slot layer
    #     must still be invisible before anything J2 is trusted.
    python3 "$EVAL" --stage anchor --rung X3 --episodes $EPS --seed $EVAL_SEED \
        --ckpt "$NAV_ROOT" --expect 200/200 \
        --out "$JSON_DIR/anchor_navtransparency.json" \
        > /tmp/j2nav_anchor_nav.log 2>&1
    local l; l=$(grep 'HARNESS ANCHOR:' /tmp/j2nav_anchor_nav.log | tail -1)
    say "RESULT anchor nav-transparency (j2 off): ${l}"
    echo "$l" | grep -q PASS || ok=0

    # (b) the shipped J2 battery — the C side is unchanged by this campaign,
    #     and this is what says so.
    if [ -f "$VERIFY_J2" ]; then
        python3 "$VERIFY_J2" > /tmp/j2nav_verify_extj2.log 2>&1
        local rc=$?
        say "RESULT verify_extj2 battery rc=$rc  $(grep -ciE '\[FAIL\]' /tmp/j2nav_verify_extj2.log) FAIL lines"
        [ "$rc" = "0" ] || ok=0
    else
        say "NOTE verify_extj2.py absent — skipping battery"
    fi

    # (c) the ported J2 filter still reproduces the head-to-head decision, and
    #     the OFF path is still bit-identical to the published nav result.
    python3 "$PORTCHK" --n 64 > /tmp/j2nav_portcheck.log 2>&1
    l=$(grep 'PORT CHECK:' /tmp/j2nav_portcheck.log | tail -1)
    say "RESULT MSC6J2Cov port check: ${l}"
    grep -E '^  (6|24) h:' /tmp/j2nav_portcheck.log | while read -r x; do say "  portcheck $x"; done
    echo "$l" | grep -q PASS || ok=0

    python3 "$EVAL" --stage eval --rung X3 --nav-mode bearings_only --acq real \
        --episodes $EPS --seed $EVAL_SEED --ckpt "$NAV_ROOT" \
        --label anchor_offpath --out "$JSON_DIR/anchor_offpath.json" \
        > /tmp/j2nav_anchor_offpath.log 2>&1
    local md5; md5=$(grep -oE 'md5 [0-9a-f]{12}' /tmp/j2nav_anchor_offpath.log | head -1)
    say "RESULT off-path identity (nav_j2_mode=0): $(grep -E 'success [0-9]+/' /tmp/j2nav_anchor_offpath.log | head -1 | sed 's/^ *//')"
    say "  expected md5 8df04e6ba5e4 (published T-BO3 native); got ${md5}"
    echo "$md5" | grep -q 8df04e6ba5e4 || ok=0

    [ "$ok" = "1" ] || { say "ABORT: stage 0 anchors FAILED"; return 1; }
}

# ── eval helper: <arm> <ckpt> <rung> <nav_mode> <acq> ───────────────────────
one_eval() {
    local arm="$1" ck="$2" rung="$3" nav="$4" acq="$5"
    cd "$PUF" || return 1
    local tag="${rung}_${nav}" L=/tmp/j2nav_${arm}_${rung}_${nav}.log
    python3 "$EVAL" --stage eval --rung "$rung" --ckpt "$ck" --nav-mode "$nav" \
        --acq "$acq" --episodes $EPS --seed $EVAL_SEED --label "${arm}/${tag}" \
        --out "$JSON_DIR/${arm}__${tag}.json" > "$L" 2>&1
    say "RESULT $arm $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    g=$(grep -E 'acquisition:' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag ACQ: $g"
    g=$(grep -E 'EPOCH error' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag EPOCH: $g"
}

pct_from_json() {   # success percentage out of a per-arm JSON
    python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(int(round(100*d['success']/max(d['n_valid'],1))))
except Exception: print(-1)
" "$1" 2>/dev/null
}

# ── train <arm> <warm> ──────────────────────────────────────────────────────
train_arm() {
    local arm="$1" warm="$2"
    local dir="$PUF/experiments_j2nav/${arm}"
    if ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
        say "SKIP train $arm (final ckpt already present)"; return 0
    fi
    say "START train $arm (bearings_only + J2 filter, 50M, warm=$(basename $warm))"
    cd "$PUF" || return 1
    # No inner caffeinate: the script already runs under one, and wrapping the
    # trainer again would make $! the wrapper's PID — caffeinate does not
    # forward signals, so the watchdog would orphan the trainer.
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps $STEPS --train.seed 42 \
        --train.data-dir "$dir" --load-model-path "$warm" \
        --env.nav-mode bearings_only \
        "${BASE_ENV[@]}" "${NAV_ENV[@]}" "${J2_ENV[@]}" \
        --wandb --wandb-project orbital-rl --wandb-group "t6-j2nav-${arm}" \
        --tag "t6_j2nav_${arm}" > "/tmp/j2nav_${arm}_train.log" 2>&1 &
    local pid=$!
    say "  trainer pid $pid, log /tmp/j2nav_${arm}_train.log"
    local t_final=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [ "$t_final" -eq 0 ] && ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
            t_final=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
        fi
        if [ "$t_final" -ne 0 ] && [ $(( $(date +%s) - t_final )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: final ckpt saved but trainer alive after ${WATCHDOG_S}s — killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
            sleep 20; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null
    local rc=$? perf
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/j2nav_${arm}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a}"
    ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1
}

last_ckpt() { ls -t "$PUF/experiments_j2nav/$1"/*/model_*${FINAL_CKPT_TAG}.pt 2>/dev/null | head -1; }

# ── the four eval rows every trained arm gets ───────────────────────────────
eval_arm() {
    local arm="$1" ck="$2"
    one_eval "$arm" "$ck" J2X       bearings_only real       # native
    one_eval "$arm" "$ck" J2X       truth         surrogate  # guidance ceiling
    one_eval "$arm" "$ck" J2X-2body bearings_only real       # J2-off retention
    one_eval "$arm" "$ck" X3        bearings_only real       # home retention
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== J2 x nav campaign start (pid $$) ==========="
say "rung: X3 task + box 30km/50m/s, j2_mode=1, i_t ~ U(30,60)deg,"
say "      lvlh_frame_mode=1, di_max=1deg, e<=0.05, D30, shaping2 dv_ref=700"
say "filter: nav_j2_mode=1 = BatchedBearingMSC6J2 (J2 in state AND covariance,"
say "        FD STM). Decided by the ext-j2 head-to-head: two-body covariance"
say "        under a J2 truth leaves NEES 11.37 and pays 2.63x position at 24h,"
say "        because an overconfident P shrinks the gain. OrbitalNav refuses it."
say "acquisition surrogate UNCHANGED (sigma ratio 1.0002, 0/128 decisions at the"
say "        46-min floor). Re-measure if any arm allows blind windows >~3 h —"
say "        the surrogate runs optimistic on long arcs (3.3% at 6 h)."
say "roots: NAV $(basename $NAV_ROOT) (cols 29-32 already exactly zero, verified)"
say "       J2  $(basename $J2_ROOT)"
say "stages requested: $STAGES"
preflight || { say "=========== ABORTED (preflight) ==========="; exit 1; }

if want 0; then
    anchors || { say "=========== ABORTED (stage 0 anchors) ==========="; exit 1; }
    say "---- stage 0 done (anchors) ----"
else say "---- stage 0 SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stage 1: floors in BOTH directions, eval only ───────────────────────────
if want 1; then
    # (a) the NAV-skilled root under J2 + inclined targets
    one_eval "FLOOR-nav"  "$NAV_ROOT" J2X bearings_only real
    one_eval "FLOOR-nav"  "$NAV_ROOT" J2X truth         surrogate
    # (b) the J2-skilled root under bearings-only, at its OWN rung
    one_eval "FLOOR-j2"   "$J2_ROOT"  J2X bearings_only real
    one_eval "FLOOR-j2"   "$J2_ROOT"  J2X truth         surrogate
    NAVF=$(pct_from_json "$JSON_DIR/FLOOR-nav__J2X_bearings_only.json")
    J2F=$(pct_from_json  "$JSON_DIR/FLOOR-j2__J2X_bearings_only.json")
    say "TRANSFER ASYMMETRY: nav-root floor ${NAVF}%  vs  J2-root floor ${J2F}%"
    say "  (which skill is harder to retrofit; rung-2's control transferred"
    say "   NOTHING to the blind problem at 13-18%, so a low J2-root floor is"
    say "   the predicted outcome, not a surprise)"
    say "---- stage 1 done (floors, both directions) ----"
else say "---- stage 1 SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stage 2: the treatment, warm from the NAV root ──────────────────────────
if want 2; then
    if train_arm "T-J2BO-nav" "$NAV_ROOT"; then
        CK=$(last_ckpt "T-J2BO-nav"); cp "$CK" "$MAIN/models/t3/j2nav_T-J2BO-nav.pt" 2>/dev/null
        eval_arm "T-J2BO-nav" "$CK"
        say "---- stage 2 done (T-J2BO-nav, the treatment) ----"
    else
        say "FAIL train T-J2BO-nav (no final ckpt)"
    fi
else say "---- stage 2 SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stage 3: the same treatment from the J2 root — AUTO-GATED ───────────────
# Only informative if the J2 root has something to transfer. If it floors below
# STAGE3_FLOOR_MIN it is starting from the same place rung 1's floor did, and
# 50M spent confirming that is 50M not spent on a new question.
if want 3; then
    J2F=$(pct_from_json "$JSON_DIR/FLOOR-j2__J2X_bearings_only.json")
    if [ "$J2F" -lt 0 ]; then
        say "---- stage 3 SKIPPED: no stage-1(b) floor on disk to gate on ----"
    elif [ "$J2F" -lt "$STAGE3_FLOOR_MIN" ]; then
        say "---- stage 3 SKIPPED BY GATE: J2-root floor ${J2F}% < ${STAGE3_FLOOR_MIN}% ----"
        say "     The J2 root has no bearings-only skill to build on, so this arm"
        say "     would re-run rung 1's floor experiment with a different warm"
        say "     start. Run it deliberately with J2NAV_STAGE3_FORCE=1 if the"
        say "     asymmetry itself is the object of study."
        [ "${J2NAV_STAGE3_FORCE:-0}" = "1" ] && {
            say "     J2NAV_STAGE3_FORCE=1 — running anyway"
            if train_arm "T-J2BO-j2root" "$J2_ROOT"; then
                CK=$(last_ckpt "T-J2BO-j2root")
                cp "$CK" "$MAIN/models/t3/j2nav_T-J2BO-j2root.pt" 2>/dev/null
                eval_arm "T-J2BO-j2root" "$CK"
            else say "FAIL train T-J2BO-j2root (no final ckpt)"; fi
        }
    else
        say "---- stage 3 RUNNING: J2-root floor ${J2F}% >= ${STAGE3_FLOOR_MIN}% ----"
        if train_arm "T-J2BO-j2root" "$J2_ROOT"; then
            CK=$(last_ckpt "T-J2BO-j2root")
            cp "$CK" "$MAIN/models/t3/j2nav_T-J2BO-j2root.pt" 2>/dev/null
            eval_arm "T-J2BO-j2root" "$CK"
            say "---- stage 3 done (T-J2BO-j2root) ----"
        else say "FAIL train T-J2BO-j2root (no final ckpt)"; fi
    fi
else say "---- stage 3 SKIPPED (not in STAGES=$STAGES) ----"; fi

say "=========== j2nav campaign COMPLETE ==========="
