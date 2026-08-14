#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 3D-NAV x ECCENTRICITY — bearings-only rendezvous up the e axis
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT:
#   nohup caffeinate -is bash scripts/orbital/nav/n3dnav_e_campaign.sh \
#       > /tmp/n3dnav_e_stdout.log 2>&1 &
#
# Everything nav-related so far has only ever seen e <= 0.05. The guidance-only
# lineages reach e <= 0.30 (2D WL4, 3D V-ladder). This closes that gap for the
# flagship: 3D + bearings-only + real batch IOD, up the eccentricity axis.
#
# j2_mode stays 0 throughout.
#
# ═══ FOUR DESIGN QUESTIONS, ANSWERED FROM EVIDENCE ═════════════════════════
#
# ── Q0 (not asked, but it governs the other three): CAN the nav lineage go up
#      the e axis at all? NO — not at its own altitude band.
#
# Eccentricity and altitude are NOT independent here. c_reset rejection-samples
# until perigee a(1-e) >= EARTH_KEEPOUT (6.571e6 m), so a high e is only
# reachable at a high a. Measured on the sampler at the X3/rung-1 LEO band
# (a 6.671-7.171e6), 512 draws per cell:
#
#     requested e_max_target   0.05    0.10    0.20    0.30
#     REALIZED mean e_target   0.0228  0.0293  0.0292  0.0285
#
# The knob is inert. A rung labelled "e0.30" at the nav band trains at e ~ 0.03
# and the log would say 0.30. Arithmetic agrees: that band admits e <= 0.084
# even at its upper edge, and e = 0.10 needs a >= 7.301e6.
#
# So the ladder MUST widen the altitude band, which forces the WIDE observation
# normalizers (obs_alt_scale_m 8e6, lvlh_scale_m 1.5e7) — at 1.6e6 the encoder
# would see altitudes 3-9x outside its trained range. Phase 5.5 already paid
# for this lesson once: its reported "cliff at e >= 0.075" was altitude OOD via
# the widening band, not an eccentricity limit.
#
# CONSEQUENCE, and it is the big one: `n3dnav_T-BO3.pt` (rung-1 winner, trained
# at 1.6e6/6.371e6) CANNOT be the warm start. The e-ladder is a wide-normalizer
# lineage and must start from a wide-normalizer parent.
#
# ── Q1 Warm-start chain: LADDER, and t5_vladder.sh is the precedent.
# It chains `run_rung ... "$LAST"` (V2<-W2, V3<-V2, V4<-V3, V5<-V4) and scored
# 200/200, 200/200, 200/200, 199/200. Followed exactly: E0 <- the warm file,
# E1 <- E0, E2 <- E1, E3 <- E2.
#
# The chain's ROOT changes for the reason in Q0: E0 warm-starts from
# seed42_V2_wide3d.pt (3D, di 1 deg, wide normalizers, e <= 0.05, published
# 200/200) rather than from the rung-1 nav winner. That makes E0 exactly
# "rung 1, re-run in the wide normalization": one knob, truth -> bearings-only,
# at fixed everything else. Rung 1 showed 50M is enough for that step.
#
# ── Q2 Is the acquisition machinery valid at wide e? Two halves, opposite
#      directions, and one of them needed a fix.
#
# The TRAINING surrogate defaults to `crlb_online`, which accumulates the exact
# Fisher information of the REALIZED arc. It has no eccentricity parameter and
# cannot go stale; its calibration to a real filter runs through
# `BLSRatioTable`, whose cells include NAV-G's wide-eccentric G4 (e = 0.30).
#
# The REAL solver was the risk. `ray_init3` seeds a CIRCULAR velocity and the
# shipped lattice brackets it at +/-0.20 v_c. NAV-G section 2 diagnosed exactly
# this on the arm it REJECTED: the range-parameterized bank "works at LEO,
# fails at e >= 0.2", because a circular prior is wrong by ~v_c*e = 1730 m/s at
# e = 0.30 — outside every component's basin. Batch least squares re-linearizes
# and survived G4 where the bank did not, so it was a risk, not a certainty.
# `RealAcq3D` now widens the tangential bracket to 1.15 * e_max, and stage 0b
# gates the campaign on the two agreeing. MEASURED at E3, 25 eps, untrained
# warm start:
#     acquired/ep  surrogate 0.31  real 0.28   (gate |diff| <= 0.20)
#     latency med  surrogate 2700 s  real 2760 s  (gate rel <= 0.50)
#     real solver failures 0                      (gate 0)   -> PASS
#
# Also tightened here: the blind range prior's inner shell is `a_min*(1-e_max)`,
# which at e_max = 0.30 is 4.67e6 m — INSIDE THE EARTH — while the env
# rejection-samples until perigee >= EARTH_KEEPOUT. This campaign floors it via
# `--env.nav-r-min-m 6.571e6`, narrowing the log-range prior by 25% at E3.
# Deliberately NOT changed in the wrapper's default: the floor also binds at
# e_max = 0.05 (6.337e6 -> 6.571e6), so making it the default would silently
# move the ALREADY-PUBLISHED rung-1 and tight-box bearings-only action streams.
# Promote it at the next deliberate re-baseline, not as a side effect of this.
#
# ── Q3 Shaping: 0.35, NOT rung-1's 0.8166667 — and that follows the
#      coordinator's own rule once Q0 moves the parent.
#
# The rule is "inherit the nav parent's value, one knob per rung". Q0 changes
# the parent from the rung-1 nav winner to seed42_V2_wide3d.pt, and the entire
# V-ladder trained with orbital.py's DEFAULT shape_w_match = 0.35 at
# dv_ref = 700. So 0.35 is the inherited value here. Using rung-1's 0.8166667
# would change the reward and the observation normalizers in the same run,
# which is the compound change every T3 collapse in this project traces to.
# Flagged loudly because it contradicts the campaign brief's stated value.
#
# ── The rungs. Per-rung (de_max, da_max, a-band, di_max, cap) follow
# t5_vladder.sh; E3 IS V5 exactly, E1/E2 interpolate between validated
# neighbours. From E1 on, de_max governs the CHASER's eccentricity and
# OVERRIDES e_max_sat, so e_max_sat is deliberately absent rather than
# set-and-ignored. Measured at 1024 draws per rung:
#
#   rung  e_max  de_max  da_max   a-band 1e6   di      cap   realized e_t   gave_up
#   E0    0.05   -       -        6.671-7.171  1.00d   3000  0.023 (p90 0.042)  0
#   E1    0.10   0.05    300 km   6.671-7.871  1.00d   3000  0.041 (p90 0.081)  0
#   E2    0.20   0.065   450 km   6.671-9.871  0.75d   4500  0.085 (p90 0.166)  0
#   E3    0.30   0.08    600 km   6.671-14.371 0.75d   6000  0.126 (p90 0.257)  0
#
# Realized e is ~0.42x the requested cap, because e is drawn uniformly on
# [0, e_max] and then perigee-filtered. Quote the cap as a SETTING and the
# realized distribution as the RESULT — they are not the same number, and the
# difference is exactly the kind of gap this project keeps having to catch.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
WS_E=$MAIN/models/t3/n3dnav_warm_E0.pt
PROG=/tmp/n3dnav_e_progress.log
JSON_DIR=$MAIN/web_data/results/n3dnav_e
EVAL=$MAIN/scripts/orbital/nav/eval_relnav3d.py
EPS=200
ACQCHECK_EPS=25
EVAL_SEED=123
STEPS=50000000
FINAL_CKPT_TAG=000382
WATCHDOG_S=900
WM=0.35                      # see Q3

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

STAGES=${N3DNAV_E_STAGES:-0,0b,1,2,3,4,5}
want() { [[ ",$STAGES," == *",$1,"* ]]; }
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# The 3D task, wide normalizers, loose box — identical in every rung.
BASE_ENV=(
  --env.num-debris-min 0 --env.num-debris-max 0 --env.same-orbit-init 0
  --env.init-phase-gap-max 3.14159 --env.valid-init-only 1
  --env.gave-up-action terminate --env.max-valid-init-attempts 4096
  --env.obs-alt-scale-m 8e6 --env.lvlh-scale-m 1.5e7
  --env.rendezvous-radius-m 30000.0 --env.rel-vel-tol-ms 50.0
  --env.shaping-mode 2 --env.shape-w-lambda 1.0 --env.shape-w-match $WM
  --env.shape-dv-ref-ms 700.0 --env.shape-gamma 1.0
  --env.phase-gap-mode 1 --env.phase-obs-mode 1 --env.cap-terminal-reward 0.0
  --env.dim3-mode 1 --env.legacy-action-space 30
)
NAV_ENV=(
  --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0
  --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20
  --env.nav-acq-mode crlb_online --env.nav-max-ticks 0
  # Blind range prior floored at EARTH_KEEPOUT (see the Q2 note). Set per
  # campaign rather than changed in the wrapper default, because the floor also
  # binds at e_max = 0.05 and would move the published rung-1 / tight-box
  # action streams.
  --env.nav-r-min-m 6.571e6
)
E0_ENV=(--env.e-max-target 0.05 --env.e-max-sat 0.05
        --env.a-min-override 6.671e6 --env.a-max-override 7.171e6
        --env.di-max-rad 0.017453 --env.episode-cap-steps 3000)
E1_ENV=(--env.e-max-target 0.10 --env.de-max 0.05 --env.da-max-m 300e3
        --env.a-min-override 6.671e6 --env.a-max-override 7.871e6
        --env.di-max-rad 0.017453 --env.episode-cap-steps 3000)
E2_ENV=(--env.e-max-target 0.20 --env.de-max 0.065 --env.da-max-m 450e3
        --env.a-min-override 6.671e6 --env.a-max-override 9.871e6
        --env.di-max-rad 0.013090 --env.episode-cap-steps 4500)
E3_ENV=(--env.e-max-target 0.30 --env.de-max 0.08 --env.da-max-m 600e3
        --env.a-min-override 6.671e6 --env.a-max-override 14.371e6
        --env.di-max-rad 0.013090 --env.episode-cap-steps 6000)

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: branch '$br', need '$BRANCH_REQ'"; return 1; }
    [ -f "$WS_E" ] || { say "ABORT: missing $WS_E"; return 1; }
    mkdir -p "$JSON_DIR"; return 0
}

anchor() {
    say "START stage 0 harness anchor (E0, expect the PUBLISHED V2 200/200)"
    cd "$PUF" || return 1
    python3 "$EVAL" --stage anchor --rung E0 --episodes $EPS --seed $EVAL_SEED \
        --ckpt "$WS_E" --expect 200/200 --out /tmp/n3dnav_e_anchor.json \
        > /tmp/n3dnav_e_anchor.log 2>&1
    local rc=$? line; line=$(grep 'HARNESS ANCHOR:' /tmp/n3dnav_e_anchor.log | tail -1)
    say "RESULT anchor rc=$rc ${line}"
    grep -E '^\s+\[(PASS|FAIL)\]' /tmp/n3dnav_e_anchor.log | while read -r l; do say "  anchor $l"; done
    echo "$line" | grep -q PASS || { say "ABORT: harness anchor FAILED"; return 1; }
}

acqcheck() {
    # Stage 0b. The surrogate trains the policy; the real IOD scores it. If
    # they disagree at wide e, every downstream number measures the surrogate.
    say "START stage 0b surrogate-vs-real acquisition at E3 (the top rung)"
    cd "$PUF" || return 1
    python3 "$EVAL" --stage acqcheck --rung E3 --episodes $ACQCHECK_EPS \
        --seed $EVAL_SEED --ckpt "$WS_E" \
        --out "$JSON_DIR/acqcheck_E3.json" > /tmp/n3dnav_e_acqcheck.log 2>&1
    local rc=$? line; line=$(grep 'ACQUISITION AGREEMENT:' /tmp/n3dnav_e_acqcheck.log | tail -1)
    say "RESULT acqcheck rc=$rc ${line}"
    grep -E 'acquired/ep|latency med|solver failures' /tmp/n3dnav_e_acqcheck.log \
        | while read -r l; do say "  acqcheck $(echo $l)"; done
    echo "$line" | grep -q PASS || { say "ABORT: acquisition agreement FAILED at E3"; return 1; }
}

# train_arm <rung> <env-array-name> <warm>
train_arm() {
    local rung="$1" envvar="$2[@]" warm="$3"
    local envflags=("${!envvar}")
    local dir="$PUF/experiments_n3dnav_e/${rung}"
    if ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
        say "SKIP train $rung (final ckpt already present)"; return 0
    fi
    say "START train $rung (bearings_only, 50M, w_match=$WM, warm=$(basename $warm))"
    cd "$PUF" || return 1
    # No inner caffeinate: the whole script runs under one, and wrapping the
    # trainer again would make $! the wrapper's PID — caffeinate does not
    # forward signals, so the watchdog would orphan the trainer.
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps $STEPS --train.seed 42 \
        --train.data-dir "$dir" --load-model-path "$warm" \
        --env.nav-mode bearings_only \
        "${BASE_ENV[@]}" "${NAV_ENV[@]}" "${envflags[@]}" \
        --wandb --wandb-project orbital-rl --wandb-group "t6-n3dnave-${rung}" \
        --tag "t6_n3dnave_${rung}" > "/tmp/n3dnav_e_${rung}_train.log" 2>&1 &
    local pid=$!
    say "  trainer pid $pid, log /tmp/n3dnav_e_${rung}_train.log"
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
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/n3dnav_e_${rung}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $rung rc=$rc last=${perf:-n/a}"
    ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1
}

last_ckpt() { ls -t "$PUF/experiments_n3dnav_e/$1"/*/model_*${FINAL_CKPT_TAG}.pt 2>/dev/null | head -1; }

# one_eval <arm> <ckpt> <rung> <nav_mode> <acq>
one_eval() {
    local arm="$1" ck="$2" rung="$3" nav="$4" acq="$5"
    cd "$PUF" || return 1
    local tag="${rung}_${nav}" L=/tmp/n3dnav_e_${arm}_${rung}_${nav}.log
    python3 "$EVAL" --stage eval --rung "$rung" --ckpt "$ck" --nav-mode "$nav" \
        --acq "$acq" --episodes $EPS --seed $EVAL_SEED --label "${arm}/${tag}" \
        --out "$JSON_DIR/${arm}__${tag}.json" > "$L" 2>&1
    say "RESULT $arm $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    g=$(grep -E 'acquisition:' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag ACQ: $g"
    g=$(grep -E 'EPOCH error' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag EPOCH: $g"
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== 3D-nav x eccentricity campaign start (pid $$) ==========="
say "task: dim3=1 loose box 30km/50m/s D30 shaping2 w_match=$WM dv_ref=700 gamma=1.0"
say "      WIDE normalizers obs_alt_scale=8e6 lvlh_scale=1.5e7 (REQUIRED: the e"
say "      knob is inert at the nav band — realized e_t 0.023->0.029 for"
say "      e_max 0.05->0.30 there; perigee >= keepout couples e to a)"
say "rungs: E0 e0.05 | E1 e0.10 | E2 e0.20 | E3 e0.30 (de_max/da_max/a-band/di/cap"
say "       from t5_vladder.sh; E3 == V5 exactly; realized e_t 0.023/0.041/0.085/0.126)"
say "warm:  $(basename $WS_E) <- seed42_V2_wide3d.pt cols 29-32 zeroed (LADDER chain"
say "       E0<-warm, E1<-E0, E2<-E1, E3<-E2, per t5_vladder.sh which scored 200/200/200/199)"
say "shaping w_match=$WM inherited from the V-ladder parent, NOT rung-1's 0.8166667"
say "eval: $EPS eps @ held-out seed $EVAL_SEED, native (bearings-only, real IOD) + truth,"
say "      at each rung's own e AND the rung below (retention)"
say "stages requested: $STAGES"
preflight || { say "=========== ABORTED (preflight) ==========="; exit 1; }
if want 0; then
    anchor || { say "=========== ABORTED (anchor) ==========="; exit 1; }
    say "---- stage 0 done (anchor) ----"
else say "---- stage 0 SKIPPED (not in STAGES=$STAGES) ----"; fi
if want 0b; then
    acqcheck || { say "=========== ABORTED (acqcheck) ==========="; exit 1; }
    say "---- stage 0b done (surrogate vs real acquisition) ----"
else say "---- stage 0b SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stage 1: the floor. Untrained warm start, zero-shot at every rung. ──────
if want 1; then
    for R in E0 E1 E2 E3; do
        one_eval "FLOOR" "$WS_E" "$R" bearings_only real
        one_eval "FLOOR" "$WS_E" "$R" truth         surrogate
    done
    say "---- stage 1 done (floor rows, all rungs) ----"
else say "---- stage 1 SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stages 2-5: the ladder. Each rung evaluates at its own e AND the one
# below, so a retention loss is visible the rung it happens on. ─────────────
run_rung() {  # <stage> <rung> <env-array> <warm-source: file path or rung> <rung-below|->
    local st="$1" rung="$2" envvar="$3" wsrc="$4" below="$5"
    if ! want "$st"; then say "---- stage $st SKIPPED (not in STAGES=$STAGES) ----"; return 0; fi
    # Resolve the warm start LAZILY. Eager `$(last_ckpt E0)` at the call site
    # would be evaluated even when stage 2 is skipped by the selector, silently
    # handing the trainer an empty --load-model-path (i.e. a FRESH run wearing
    # a ladder rung's name). Resolve here and refuse if it is missing.
    local warm="$wsrc"
    if [ ! -f "$warm" ]; then warm=$(last_ckpt "$wsrc"); fi
    if [ ! -f "$warm" ]; then
        say "ABORT stage $st ($rung): warm start '$wsrc' not found. The ladder"
        say "      is a CHAIN — rung $rung starts from the previous rung's"
        say "      checkpoint, so re-run the stage that produces it, or point"
        say "      N3DNAV_E_STAGES at the whole chain."
        return 1
    fi
    if train_arm "$rung" "$envvar" "$warm"; then
        local CK; CK=$(last_ckpt "$rung")
        cp "$CK" "$MAIN/models/t3/n3dnav_e_${rung}.pt" 2>/dev/null
        one_eval "$rung" "$CK" "$rung" bearings_only real
        one_eval "$rung" "$CK" "$rung" truth         surrogate
        if [ "$below" != "-" ]; then
            one_eval "$rung" "$CK" "$below" bearings_only real
            one_eval "$rung" "$CK" "$below" truth         surrogate
        fi
        say "---- stage $st done ($rung) ----"
    else
        say "FAIL train $rung (no final ckpt)"
    fi
}

run_rung 2 E0 E0_ENV "$WS_E" -
run_rung 3 E1 E1_ENV E0 E0
run_rung 4 E2 E2_ENV E1 E1
run_rung 5 E3 E3_ENV E2 E2

say "=========== 3D-nav x eccentricity campaign COMPLETE ==========="
