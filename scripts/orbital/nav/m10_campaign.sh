#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# MAJOR-10 — is the NORMAL-axis fine burn (rows 20/21) guidance-critical?
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/nav/m10_campaign.sh \
#       > /tmp/m10_stdout.log 2>&1 &
# The n3dnav rung-1 campaign's first launch died because it was spawned inside
# an agent process tree and was reaped when that task closed.
#
# ── WHAT MAJOR-10 ACTUALLY SAYS ─────────────────────────────────────────────
# `_FINE = [12,13,14,15,18,19]` is the correct set of IN-PLANE fine burns, but
# rows 20/21 are normal +/-1 m/s — also fine burns, and by N3D-B 3.3 the 3D
# OBSERVABILITY TREATMENT (1 m/s normal beats 1 m/s prograde by 2.5-8x in
# information at every terminal cell). An ablation that blocks only the
# in-plane set "leaves the treatment axis wide open, and its result is
# uninterpretable". The named sets now exist in nav_math.ACTION_SETS
# (fine_inplane / fine_normal / fine_all / normal_all / combined) and EVERY ARM
# BELOW STATES WHICH SET IT BLOCKS.
#
# And the requirement that makes this a campaign rather than an eval:
#   "A T-BO-normal arm therefore needs a truth control at the same box, or the
#    plane leg's guidance cost will be misread as an information result."
#
# ── THE BLOCKER THIS PREP FOUND (read before trusting any earlier plan) ─────
# The obvious tool was the existing `nav_block_fine_below_m` + `nav_block_set`
# interlock. It CANNOT do this job, for a reason that is invisible unless you
# look: `step()` reads
#       if self._block_below > 0.0 and self._nav_mode != 'truth':
# so the ablation is SKIPPED ENTIRELY IN TRUTH MODE. Measured, asking it to
# block fine_normal at TB5-3D over 12 episodes:
#       bearings_only  intercept rate 0.8008   4->2/12   (it works)
#       truth          intercept rate  n/a     12/12     (SILENTLY INERT)
# `_d_block` is not even allocated in truth mode. Had the campaign used it, the
# truth control would have shown NO deficit and MAJOR-10 would have been closed
# with exactly the misreading it warns against, inverted: "no truth effect =>
# the normal axis is an INFORMATION result". That interlock is a NAVIGATION
# interlock ("do not make fine burns when you cannot see"), which is
# meaningless when you can always see — right for NAV-F, wrong for MAJOR-10.
#
# So this campaign uses a new, separate, DEFAULT-OFF kwarg `nav_ablate_rows`:
# the named rows coast, unconditionally, in EVERY nav mode, at every
# separation. Off is bitwise-inert (gated). In bearings_only the new knob
# reproduces the old one exactly (both 0.8008), so it is a faithful
# generalisation and not a different intervention.
#
# ── SELF-RED-TEAM ───────────────────────────────────────────────────────────
#
# (a) WHAT NAV-F's T-BO-act TAUGHT, and whether it repeats here.
#     NAV-F blocked the IN-PLANE fine burns below 10 km and found native
#     92.5% and — decisively — TRUTH 94.0% against T-BO's 97.0%. The deficit
#     surviving into truth mode is what proved fine burns guidance-critical
#     rather than info-critical. The pre-registered expectation here is the
#     same pattern on the normal axis, and THIS PREP ALREADY MEASURED IT
#     ZERO-SHOT (n=40, TB5-3D, sampling mask):
#         bearings_only  40/40 -> 18/40      truth  40/40 -> 20/40
#     The truth deficit is not smaller than the native one. If the trained arms
#     reproduce that, the normal axis is guidance-critical and the "3D
#     observability treatment" framing of N3D-B 3.3 does NOT survive as an
#     information claim at this box. The campaign is built to be able to say
#     the opposite too: if the TRUTH arm recovers to ~baseline after training
#     while the bearings-only arm does not, the deficit IS informational.
#
# (b) WHY BOTH TRAINING-SIDE AND EVAL-SIDE MASKING ARE NEEDED. They answer
#     different questions and neither substitutes for the other.
#       * EVAL-side (sampling mask, --mask-rows): removes the rows from the
#         POLICY'S CHOICE SET at argmax. Right for the FLOOR, where the policy
#         was trained WITH the rows: "do your best without them".
#       * TRAINING-side (env ablation, --env.nav-ablate-rows): the rows coast.
#         Right for the TREATMENT, where the question is what is ACHIEVABLE
#         without the axis, not what this particular policy degrades to.
#     They are NOT interchangeable, and the gap is measured, not asserted. Used
#     as a floor, the env ablation makes the policy SPIN on a dead row —
#     80.8% of its decisions were intercepted — which conflates "lost the axis"
#     with "wasted the episode":
#         bearings_only floor: sampling mask 18/40   env ablation 13/40
#         truth         floor: sampling mask 20/40   env ablation 13/40
#     Both are reported. For the TRAINED arms the two must CONVERGE (a trained
#     ablated policy should have learned not to emit an inert row), and that
#     convergence is itself a gate: stage 2/3 report `nav_ablate_rate`, which
#     should fall toward 0.
#
# ── THE MEASUREMENT ─────────────────────────────────────────────────────────
# Success rate alone cannot separate "lost the plane axis" from "got worse", so
# every eval also reports the IN-BOX RELATIVE VELOCITY decomposed along the
# target's h_hat (cross-track — exactly what a normal burn controls and nothing
# else does) versus in-plane. Zero-shot at TB5-3D the split already moves hard:
#       bearings_only  cross-track 0.432 -> 3.053 m/s  (share 0.119 -> 0.447)
#       truth          cross-track 0.352 -> 2.677 m/s  (share 0.094 -> 0.468)
# i.e. against a 1 m/s box, removing normal +/-1 leaves a ~2.7-3.1 m/s
# cross-track residual the policy cannot null — WITH PERFECT INFORMATION.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/nav/eval_relnav3d.py
VERIFY=$MAIN/scripts/orbital/nav/verify_extnav.py
WS_TB=$MAIN/models/t3/n3dnav_warm_TB5.pt
SHIPPED=$MAIN/models/t3/n3dnav_T-BO3D-TB5.pt
TB_PARENT=$MAIN/models/t3/seed42_TB3D_box5k1.pt
PROG=/tmp/m10_progress.log
JSON_DIR=$MAIN/web_data/results/m10
EPS=200
EVAL_SEED=123
STEPS=${M10_STEPS:-50000000}
FINAL_CKPT_TAG=000382          # 50M at the shipped 8w x 256 shape
WATCHDOG_S=900
ABLATE=fine_normal             # THE SET THIS CAMPAIGN BLOCKS. Rows 20/21.

M10_SEED=${M10_SEED:-42}
SEED_SFX=""
[ "$M10_SEED" != "42" ] && SEED_SFX="_s${M10_SEED}"

STAGES=${M10_STAGES:-0,1,2,3}
want() { [[ ",$STAGES," == *",$1,"* ]]; }

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# The 3D task, identical in every stage. TB5-3D is the box where fine burns
# bind (N3D-B 5: out-of-plane VELOCITY is the new binding constraint, and the
# tolerance here is 1 m/s).
BASE_ENV=(
  --env.num-debris-min 0 --env.num-debris-max 0
  --env.e-max-target 0.05 --env.e-max-sat 0.05 --env.same-orbit-init 0
  --env.init-phase-gap-max 3.14159 --env.valid-init-only 1
  --env.gave-up-action terminate --env.max-valid-init-attempts 4096
  --env.obs-alt-scale-m 1.6e6 --env.lvlh-scale-m 6.371e6
  --env.shaping-mode 2 --env.shape-w-lambda 1.0
  --env.shape-dv-ref-ms 700.0 --env.shape-gamma 1.0
  --env.phase-gap-mode 1 --env.phase-obs-mode 1
  --env.episode-cap-steps 3000 --env.cap-terminal-reward 0.0
  --env.dim3-mode 1 --env.di-max-rad 0.017453
  --env.legacy-action-space 30
)
NAV_ENV=(
  --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0
  --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20
  --env.nav-acq-mode crlb_online --env.nav-max-ticks 0
)
TB5_BOX=(--env.rendezvous-radius-m 5000.0 --env.rel-vel-tol-ms 1.0)
# shape_w_match 0.35 throughout: the TB3D ladder's value, which n3dnav_warm_TB5
# inherits. An arm must use its OWN parent's value or the reward moves
# alongside the intervention.
WM=0.35

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: $MAIN on '$br', need '$BRANCH_REQ'"; return 1; }
    for f in "$EVAL" "$VERIFY" "$WS_TB" "$SHIPPED" "$TB_PARENT"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    mkdir -p "$JSON_DIR"; return 0
}

anchor() {
    say "START stage 0 anchors"
    cd "$PUF" || return 1
    python3 "$EVAL" --stage anchor --episodes $EPS --seed $EVAL_SEED \
        --ckpt "$WS_TB" --box TB5-3D --shape-w-match $WM --expect 194/200 \
        --out /tmp/m10_anchor.json > /tmp/m10_anchor.log 2>&1
    local rc=$? line; line=$(grep 'HARNESS ANCHOR:' /tmp/m10_anchor.log | tail -1)
    say "RESULT anchor rc=$rc ${line}"
    grep -E '^\s+\[(PASS|FAIL)\]' /tmp/m10_anchor.log | while read -r l; do say "  anchor $l"; done
    echo "$line" | grep -q PASS || { say "ABORT: harness anchor FAILED"; return 1; }
    # the 2D lineage must also be untouched by the wrapper change
    python3 "$VERIFY" --stage v1 --eps 100 > /tmp/m10_v1.log 2>&1
    say "RESULT v1 $(grep -E 'legacy Orbital|ACTION STREAM' /tmp/m10_v1.log | head -2 | tr '\n' ' ')"
    say "---- stage 0 done ----"; return 0
}

# one_eval <tag> <ckpt> <nav_mode> <box> [extra flags...]
one_eval() {
    local tag="$1" ck="$2" nav="$3" box="$4"; shift 4
    cd "$PUF" || return 1
    case "$tag" in
        F*) [ -s "$JSON_DIR/${tag}.json" ] && { say "SKIP eval $tag (floor, present)"; return 0; } ;;
        *)  tag="${tag}${SEED_SFX}" ;;
    esac
    local L=/tmp/m10_${tag}.log
    python3 "$EVAL" --stage eval --ckpt "$ck" --nav-mode "$nav" --acq real \
        --box "$box" --shape-w-match $WM --episodes $EPS --seed $EVAL_SEED \
        --label "$tag" --out "$JSON_DIR/${tag}.json" "$@" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    for pat in 'CROSSTRACK' 'inside the' 'env-ablation intercepted' 'causes:'; do
        g=$(grep -i "$pat" "$L" | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    done
}

# train_arm <arm> <nav_mode>
train_arm() {
    local arm="$1" nav="$2"
    local dir="$PUF/experiments_m10/${arm}${SEED_SFX}"
    if ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
        say "SKIP train $arm (final ckpt present)"; return 0
    fi
    say "START train $arm nav=$nav seed=$M10_SEED BLOCKING SET=$ABLATE (rows 20/21)"
    cd "$PUF" || return 1
    # NO inner caffeinate: the script runs under one, and wrapping again makes
    # $! the wrapper's PID; caffeinate does not forward signals, so the
    # watchdog would kill the wrapper and orphan the trainer on 14 cores.
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps $STEPS --train.seed "$M10_SEED" \
        --train.data-dir "$dir" --load-model-path "$WS_TB" \
        --env.nav-mode "$nav" --env.shape-w-match $WM \
        --env.nav-ablate-rows "$ABLATE" \
        "${BASE_ENV[@]}" "${NAV_ENV[@]}" "${TB5_BOX[@]}" \
        --wandb --wandb-project orbital-rl --wandb-group "t7-m10-${arm}${SEED_SFX}" \
        --tag "t7_m10_${arm}${SEED_SFX}" > "/tmp/m10_${arm}${SEED_SFX}_train.log" 2>&1 &
    local pid=$!; say "  trainer pid $pid"
    local t_final=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [ "$t_final" -eq 0 ] && ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
            t_final=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
        fi
        if [ "$t_final" -ne 0 ] && [ $(( $(date +%s) - t_final )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 20
            pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; break
        fi
    done
    wait "$pid" 2>/dev/null
    local rc=$? perf
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/m10_${arm}${SEED_SFX}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a}"
    ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1
}
last_ckpt() { ls -t "$PUF/experiments_m10/$1${SEED_SFX}"/*/model_*${FINAL_CKPT_TAG}.pt 2>/dev/null | head -1; }

# ═══════════════════════════════════════════════════════════════════════════
say "=========== MAJOR-10 campaign start (pid $$) ==========="
say "BLOCKED SET: $ABLATE = rows 20/21 = normal +/-1 m/s (state this in every claim)"
say "box TB5-3D 5 km / 1 m/s; task dim3=1 di_max=1deg e<=0.05 D30 shaping2 w_match=$WM"
say "warm $(basename $WS_TB); steps $STEPS; seed $M10_SEED${SEED_SFX:+ (suffixed)}"
say "stages requested: $STAGES"
preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED ----"; fi

# ── stage 1: floors, eval only ─────────────────────────────────────────────
if want 1; then
  say "START stage 1 (floors: the SHIPPED T-BO3D-TB5, zero-shot without rows 20/21)"
  for BOX in TB5-3D TB4-3D; do
    # baseline, for the paired comparison at each box
    one_eval "F_base_bo_${BOX}"   "$SHIPPED" bearings_only "$BOX"
    one_eval "F_base_tr_${BOX}"   "$SHIPPED" truth         "$BOX"
    # SAMPLING mask: the clean "do your best without them" floor
    one_eval "F_mask_bo_${BOX}"   "$SHIPPED" bearings_only "$BOX" --mask-rows $ABLATE
    one_eval "F_mask_tr_${BOX}"   "$SHIPPED" truth         "$BOX" --mask-rows $ABLATE
    # ENV ablation: the same intervention the TREATMENT trains under, so the
    # floor and the treatment are measured through the same mechanism too.
    one_eval "F_abl_bo_${BOX}"    "$SHIPPED" bearings_only "$BOX" --ablate-rows $ABLATE
    one_eval "F_abl_tr_${BOX}"    "$SHIPPED" truth         "$BOX" --ablate-rows $ABLATE
  done
  say "---- stage 1 done ----"
else say "---- stage 1 SKIPPED ----"; fi

# ── stage 2: treatment — bearings-only, trained WITHOUT rows 20/21 ─────────
if want 2; then
  say "START stage 2 (T-BO-normal: bearings-only, trained with $ABLATE ablated)"
  if train_arm "T-BOnormal" bearings_only; then
      ck=$(last_ckpt T-BOnormal); say "  ckpt $ck"
      for BOX in TB5-3D TB4-3D; do
        one_eval "s2_BOnorm_bo_${BOX}" "$ck" bearings_only "$BOX" --ablate-rows $ABLATE
        one_eval "s2_BOnorm_tr_${BOX}" "$ck" truth         "$BOX" --ablate-rows $ABLATE
        # convergence gate: a TRAINED ablated policy should no longer emit the
        # inert rows, so the sampling mask and the env ablation must agree.
        one_eval "s2_BOnorm_msk_${BOX}" "$ck" bearings_only "$BOX" --mask-rows $ABLATE
      done
      say "---- stage 2 done ----"
  else say "---- stage 2 FAILED (no final ckpt) ----"; fi
else say "---- stage 2 SKIPPED ----"; fi

# ── stage 3: THE CONTROL — same training, TRUTH state ──────────────────────
# MAJOR-10's explicit requirement. Separates guidance-necessity from
# info-necessity: if the truth arm ALSO fails to recover, the normal axis is
# guidance-critical (NAV-F's T-BO-act pattern). If truth recovers and
# bearings-only does not, the deficit is informational.
if want 3; then
  say "START stage 3 (T-truth-normal: TRUTH state, trained with $ABLATE ablated)"
  say "  this is MAJOR-10's required control, not an extra"
  if train_arm "T-truthnormal" truth; then
      ck=$(last_ckpt T-truthnormal); say "  ckpt $ck"
      for BOX in TB5-3D TB4-3D; do
        one_eval "s3_TRnorm_tr_${BOX}" "$ck" truth         "$BOX" --ablate-rows $ABLATE
        one_eval "s3_TRnorm_bo_${BOX}" "$ck" bearings_only "$BOX" --ablate-rows $ABLATE
      done
      say "---- stage 3 done ----"
      say "READ THE MATRIX: if s3 (truth, trained ablated) stays well below the"
      say "  F_base_tr baseline, the normal axis is GUIDANCE-critical and N3D-B"
      say "  3.3's observability framing does not survive as an INFORMATION"
      say "  claim at this box. If s3 recovers to baseline while s2 does not,"
      say "  the deficit IS informational. Report the cross-track velocity"
      say "  decomposition alongside — success rate alone cannot tell them apart."
  else say "---- stage 3 FAILED (no final ckpt) ----"; fi
else say "---- stage 3 SKIPPED ----"; fi

say "=========== MAJOR-10 campaign end ==========="
