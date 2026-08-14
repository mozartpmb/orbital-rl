#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ext-j2 RUNG 1 — J2 as a disturbance the policy must absorb
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH THIS FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/extj2/j2_rung_campaign.sh \
#       > /tmp/j2_rung_stdout.log 2>&1 &
# The n3dnav rung-1 campaign's first launch died at 22:42 because it was
# spawned inside an agent process tree and was reaped when that task closed.
# Nothing in this script prevents that; only the launching shell does.
#
# ── EVERY CLAIM HERE IS IN MEAN ELEMENTS ────────────────────────────────────
# Under j2_mode=1 the env's state IS the mean element set. No mean/osculating
# conversion happens anywhere — not at reset, not at a burn, not at the success
# test (j2_A_design §1.4; J2_DESIGN_NOTES.md §3). "Success" means the MEAN
# elements are inside the box. At the 30 km / 50 m/s box that is benign: the
# relative-state error of the mean-element approximation is 83 m / 0.094 m/s
# per orbit at a 5 km separation, common-mode along-track and largely
# cancelling. At a 5 km / 1 m/s box the VELOCITY term is 9.4% of tolerance per
# orbit, so a tight-box J2 arm must NOT also claim osculating-grade terminal
# fidelity. Every eval label carries the "MEAN ELEMENTS" tag for this reason.
#
# ── WHY TIGHT-BOX IS THE EXPERIMENT (read this before anything else) ────────
# The loose box is the warm-up; the tight-box recovery is the headline. The
# zero-shot survey (scripts/orbital/extj2/j2_zeroshot_survey.sh, output
# committed beside it) measured the J2 gap as a monotone function of box
# tightness, at i_t = 0 so the PLANE channel is provably inert and only the
# in-plane secular drift is active, each box with its OWN parent checkpoint so
# the control sits at ceiling:
#
#   box                 ckpt                  j2=0     j2=1     gap
#   30 km / 50 m/s      seed42_X3_3d_di1deg   200/200  200/200    0.0 pp
#   10 km / 10 m/s      seed42_TB3D_box10k10  199/200  113/200  -43.0 pp
#    5 km /  1 m/s      seed42_TB3D_box5k1    194/200    0/200  -97.0 pp
#
# At the loose box J2 is a NON-EVENT at an equatorial target and reaches at
# most -10.5 pp through the inclination-dependent plane channel. At the tight
# boxes it is decisive, and it is NOT the plane channel — it is present at
# i_t = 0 where that channel cannot act at all. The failure mode is ALWAYS
# safety_cap, median episode length exactly the cap: the policy never crashes,
# it just never closes the box against a secular drift it cannot null.
#
# WHERE THE DRIFT COMES FROM, which is also not what the design predicted.
# Omega-dot = -k(a,e)*cos(i) depends on BOTH a and i, and the chaser and target
# differ in both — the transfer task ITSELF puts them at different altitudes.
# Decomposing |Omega-dot_s - Omega-dot_t| over 2000 sampled draws:
#
#   delta-a only (planes cloned)   median 0.4379 deg/day   closed form 0.4551
#   delta-i only (orbits cloned)   median 0.0347 deg/day   closed form 0.0392
#
# The ALTITUDE difference dominates by 13x. di_max is a 7% correction on top.
# Over a 3000-substep (50 h) cap that injects a mean 0.692 deg of relative
# inclination = 183% of the 0.3775 deg free-plane zone = 92 m/s = 19% of the
# 478 m/s budget. The J2 signal is order-unity in the units that bind; what the
# loose box lacks is not signal but HEADROOM to demonstrate it in.
#
# Hence the ladder: stages 3a and 3b walk the box down 30km/50 -> 10km/10 ->
# 5km/1, each warm-started from the previous stage's child, so the arm never
# has to bootstrap from a 0/200 floor in one jump.
#
# ── WHAT THE PRE-LAUNCH MEASUREMENTS SAY (read this before re-scoping) ──────
# At the LOOSE box with inclined targets (lvlh_frame_mode=1, raan fixed),
# which is the only way J2 reaches the loose box at all:
#
#   i_t band     j2=0     j2=1     gap
#   U(20,30)     200/200  200/200    0.0 pp
#   U(20,45)     200/200  191/200   -4.5 pp
#   U(25,50)     199/200  186/200   -6.5 pp
#   U(30,60)     180/200  159/200  -10.5 pp
#
# Stages 0-2 are exactly the specified loose-box design (the warm-up). Stages
# 3a/3b are the tight-box ladder (the headline) and are NOT in the default
# stage list -- pass J2_RUNG_STAGES=0,1,2,3a,3b at launch to run everything.
#
# ── ONE CONFOUND THAT HAD TO BE CLOSED BEFORE STAGE 1 MEANT ANYTHING ────────
# obs[33-36] — the policy's PRIMARY rendezvous channel — was built by rotating
# the INERTIAL x,y offset by the in-plane angle omega+theta. That is the LVLH
# frame only at i_t = Omega_t = 0, which every shipped lineage pinned. Turning
# on inclined targets makes it wrong by a mean 1.51 / worst 4.37 obs units
# (Box(-2,2) channels), 239% relative. Measured cost, zero-shot at U(30,60):
#       legacy frame  113/200 = 56.5%   (36 collisions, 43 stranded)
#       fixed  frame  180/200 = 90.0%   ( 0 collisions,  1 stranded)
# i.e. the broken frame costs 33.5 pp and MANUFACTURES failure modes. Without
# lvlh_frame_mode=1 the "J2 gap" would have read 56.5 -> 51.0 = -5.5 pp, which
# would have HALVED the apparent J2 effect while tripling the noise. Every arm
# below therefore runs lvlh_frame_mode=1. It is free: verified bitwise
# identical to the legacy block at i = Omega = 0, and all three checkpoint
# anchors reproduce with IDENTICAL action-stream md5s.
#
# ── AND ONE THAT IS DELIBERATELY LEFT OPEN ──────────────────────────────────
# raan_target_sample = 0 in every TRAINING arm. Omega_t is GAUGE under J2
# (the potential is axisymmetric about z; measured: the differential nodal rate
# is independent of Omega_t to 5.0% across a half-split), so sampling it adds
# no task content. It does cost: obs[18] is the Earth-conjunction bearing, and
# Earth is static at the origin, so that channel reduces to atan2(s_y, s_x) —
# the chaser's ABSOLUTE INERTIAL LONGITUDE. Sampling Omega_t decorrelates it
# from everything the warm-start policy learned, for nothing. Measured cost at
# U(30,60) j2=1: 159/200 = 79.5% -> 138/200 = 69.0%. It runs as ONE diagnostic
# eval cell in stage 1 (the SO(2) leak detector) and nowhere else.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/extj2/j2_rung_eval.py
GATES_C=$MAIN/scripts/orbital/extj2/j2_gates.c
SAMP_C=$MAIN/scripts/orbital/extj2/j2_sampler_gates.c
VERIFY=$MAIN/scripts/orbital/extj2/verify_extj2.py
WARM=$MAIN/models/t3/seed42_X3_3d_di1deg.pt
TB_WARM=$MAIN/models/t3/seed42_TB3D_box5k1.pt
TB1_REF=$MAIN/models/t3/seed42_TB3D_box10k10.pt
TRIP=$MAIN/scripts/orbital/extj2/j2_flatline_check.py
PROG=/tmp/j2_rung_progress.log
JSON_DIR=$MAIN/web_data/results/extj2_rung
EPS=200
EVAL_SEED=123
STEPS=50000000
FINAL_CKPT_TAG=000382        # 50M at the shipped 8w x 256 shape
WATCHDOG_S=900               # 15 min after the final ckpt appears

# Which stages to run. Stage 0 (anchors) runs unless explicitly excluded and
# its failure ALWAYS aborts — no stage list can turn that off.
#   J2_RUNG_STAGES=0,3 bash scripts/orbital/extj2/j2_rung_campaign.sh
STAGES=${J2_RUNG_STAGES:-0,1,2}      # launch: 0,1,2,3a,3b
want() { [[ ",$STAGES," == *",$1,"* ]]; }

# The rung band. 30-60 deg keeps sin(2i) in [0.866, 1.0] — the J2 plane channel
# within 13% of its maximum everywhere — and keeps 63.43 deg (the critical
# inclination, where omega-dot = 0) OUTSIDE the band so a second, unrelated
# degeneracy is not mixed into the arm. Overridable: J2_BAND=25,50 gives a
# 99.5% control ceiling instead of 90.0%, at 6.5 pp of headroom instead of 10.5.
BAND=${J2_BAND:-30,60}
BAND_LO_RAD=$(python3 -c "import math,sys;print(math.radians(float('$BAND'.split(',')[0])))")
BAND_HI_RAD=$(python3 -c "import math,sys;print(math.radians(float('$BAND'.split(',')[1])))")

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# ── the X3 task, identical in every stage; only the J2 knobs and box move ───
BASE_ENV=(
  --env.num-debris-min 0 --env.num-debris-max 0
  --env.e-max-target 0.05 --env.e-max-sat 0.05 --env.same-orbit-init 0
  --env.init-phase-gap-max 3.14159 --env.valid-init-only 1
  --env.gave-up-action terminate --env.max-valid-init-attempts 4096
  --env.obs-alt-scale-m 1.6e6 --env.lvlh-scale-m 6.371e6
  --env.shaping-mode 2 --env.shape-w-lambda 1.0 --env.shape-w-match 0.8166667
  --env.shape-dv-ref-ms 700.0 --env.shape-gamma 1.0
  --env.phase-gap-mode 1 --env.phase-obs-mode 1
  --env.episode-cap-steps 3000 --env.cap-terminal-reward 0.0
  --env.dim3-mode 1 --env.di-max-rad 0.017453
  --env.legacy-action-space 30
)
J2_ENV=(
  --env.j2-mode 1
  --env.i-target-min-rad "$BAND_LO_RAD" --env.i-target-max-rad "$BAND_HI_RAD"
  --env.raan-target-sample 0
  --env.lvlh-frame-mode 1
)
X3_BOX=(--env.rendezvous-radius-m 30000.0 --env.rel-vel-tol-ms 50.0)
B1_BOX=(--env.rendezvous-radius-m 10000.0 --env.rel-vel-tol-ms 10.0)
TB5_BOX=(--env.rendezvous-radius-m 5000.0 --env.rel-vel-tol-ms 1.0)

preflight() {
    local br
    br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    if [ "$br" != "$BRANCH_REQ" ]; then
        say "ABORT: $MAIN is on branch '$br', need '$BRANCH_REQ'. The trainer"
        say "       runs whatever the MAIN checkout has checked out."
        return 1
    fi
    for f in "$WARM" "$TB_WARM" "$TB1_REF" "$EVAL" "$GATES_C" "$SAMP_C" \
             "$VERIFY" "$TRIP"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    mkdir -p "$JSON_DIR"
    return 0
}

# ── stage 0: every downstream number depends on these ───────────────────────
anchor() {
    say "START stage 0 anchors"
    cd "$MAIN" || return 1
    local rc=0

    cc -O2 -flto -lm -I pufferlib/pufferlib/ocean/orbital "$GATES_C" \
       -o /tmp/j2_gates_bin 2>/tmp/j2_gates_build.log \
       && /tmp/j2_gates_bin > /tmp/j2_gates_run.log 2>&1
    local g=$?
    say "  RESULT j2_gates rc=$g $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/j2_gates_run.log | tail -1)"
    [ $g -ne 0 ] && rc=1

    cc -O2 -flto -lm -I pufferlib/pufferlib/ocean/orbital "$SAMP_C" \
       -o /tmp/j2_samp_bin 2>/tmp/j2_samp_build.log \
       && /tmp/j2_samp_bin > /tmp/j2_samp_run.log 2>&1
    local s=$?
    say "  RESULT j2_sampler_gates rc=$s $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/j2_samp_run.log | tail -1)"
    [ $s -ne 0 ] && rc=1

    python3 "$VERIFY" --stage a1,a2,a3,a4 --eps 100 > /tmp/j2_verify.log 2>&1
    local v=$?
    say "  RESULT verify_extj2 rc=$v $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/j2_verify.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/j2_verify.log | while read -r l; do say "    $l"; done
    [ $v -ne 0 ] && rc=1

    if [ $rc -ne 0 ]; then
        say "ABORT: stage 0 anchors FAILED (see /tmp/j2_gates_run.log,"
        say "       /tmp/j2_samp_run.log, /tmp/j2_verify.log)"
        return 1
    fi
    say "---- stage 0 done (anchors clean) ----"
    return 0
}

# ── one_eval <tag> <ckpt> <j2> <band> <lvlh> <raan> <box_r> <box_v> <w_match> ─
one_eval() {
    local tag="$1" ck="$2" j2="$3" band="$4" lvlh="$5" raan="$6"
    local br="$7" bv="$8" wm="$9"
    cd "$MAIN" || return 1
    local L=/tmp/j2_rung_${tag}.log
    python3 "$EVAL" --ckpt "$ck" --label "$tag" --episodes $EPS --seed $EVAL_SEED \
        --j2-mode "$j2" --i-band "$band" --lvlh-frame-mode "$lvlh" \
        --raan-sample "$raan" --rendezvous-radius-m "$br" --rel-vel-tol-ms "$bv" \
        --shape-w-match "$wm" --out "$JSON_DIR/${tag}.json" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    g=$(grep -E 'causes:' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    g=$(grep -E 'MEAN-ELEMENT' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    # The claim boundary, in the progress log as well as the eval output, so it
    # survives being read out of context. Only j2=1 rows are mean-element
    # claims; j2=0 rows are two-body, where mean == osculating identically.
    if [ "$j2" = "1" ]; then
        # The mean-element approximation's RELATIVE-state error is 83 m /
        # 0.094 m/s per orbit at 5 km separation (j2_A_design §1.4). As a
        # fraction of the box's VELOCITY tolerance that is 0.19%/orbit at
        # 50 m/s, 0.94%/orbit at 10 m/s and 9.4%/orbit at 1 m/s — so the
        # caveat is quoted at the strength this box actually earns, never
        # borrowed from the tightest one.
        local slip
        slip=$(awk -v b="$bv" 'BEGIN{printf "%.2f", 0.094/b*100}')
        say "  $tag CLAIM: meets the ${br}m/${bv}m/s box IN MEAN ELEMENTS. No"
        say "      $tag mean/osculating conversion anywhere; the osculating"
        say "      $tag VELOCITY slip is ${slip}% of this box's tolerance per orbit."
        if [ "${bv%.*}" -le 1 ]; then
            say "      $tag At this box that is the binding caveat: the honest"
            say "      $tag statement is 'meets the box in mean elements', FULL STOP"
            say "      $tag — never 'osculating-grade rendezvous'."
        fi
    fi
}

# ── train <arm> <warm> <box-array-name> <extra...> ──────────────────────────
train_arm() {
    local arm="$1" warm="$2" boxvar="$3[@]"; shift 3
    local box=("${!boxvar}")
    local dir="$PUF/experiments_extj2/${arm}"
    if ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
        say "SKIP train $arm (final ckpt already present)"
        return 0
    fi
    say "START train $arm (warm=$(basename $warm) box=${box[1]}/${box[3]} band=$BAND)"
    cd "$PUF" || return 1
    # NO inner `caffeinate`: the whole script runs under one, and wrapping the
    # trainer again would make $! the wrapper's PID. caffeinate does not forward
    # signals, so the watchdog would kill the wrapper and leave the trainer
    # ORPHANED on 14 cores under the next stage.
    python3 -m pufferlib.pufferl train puffer_orbital \
        --train.device cpu --train.total-timesteps $STEPS --train.seed 42 \
        --train.data-dir "$dir" --load-model-path "$warm" \
        "${BASE_ENV[@]}" "${J2_ENV[@]}" "${box[@]}" "$@" \
        --wandb --wandb-project orbital-rl --wandb-group "t6-j2-${arm}" \
        --tag "t6_j2_${arm}" > "/tmp/j2_rung_${arm}_train.log" 2>&1 &
    local pid=$!
    say "  trainer pid $pid, log /tmp/j2_rung_${arm}_train.log"

    local t_final=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [ "$t_final" -eq 0 ] \
           && ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
            t_final=$(date +%s)
            say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
        fi
        if [ "$t_final" -ne 0 ] \
           && [ $(( $(date +%s) - t_final )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: final ckpt saved but trainer alive after ${WATCHDOG_S}s — killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null
            kill -TERM "$pid" 2>/dev/null
            sleep 20
            pkill -KILL -P "$pid" 2>/dev/null
            kill -KILL "$pid" 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null
    local rc=$? perf
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/j2_rung_${arm}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a}"
    ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1
}

last_ckpt() {
    ls -t "$PUF/experiments_extj2/$1"/*/model_*${FINAL_CKPT_TAG}.pt 2>/dev/null | head -1
}

# ── flatline <arm> <floor_json> ─────────────────────────────────────────────
# cap_terminal_reward=0 + shape_gamma=1 means a capped episode pays NOTHING and
# still collects the telescoped Phi_T - Phi_0. If the warm start never samples a
# success, the +10 is never seen and PPO has nothing to climb — the T3 red-team
# #1 mechanism. At 5 km / 1 m/s the J2-blind policy fails 200/200 on safety_cap,
# so this is live. Name it in the log rather than discover it two stages later.
# A FLATLINE verdict is a FINDING: it never aborts the campaign.
flatline() {
    local arm="$1" fj="$2" floor out
    floor=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['rate'])" \
            "$fj" 2>/dev/null)
    if [ -z "${floor:-}" ]; then
        say "RESULT FLATLINE-CHECK $arm: SKIPPED (no floor json at $fj)"
        return 0
    fi
    out=$(python3 "$TRIP" --log "/tmp/j2_rung_${arm}_train.log" --floor "$floor" \
          --after-steps 20e6 --margin 0.05 --label "$arm" 2>&1)
    say "RESULT $out"
    return 0
}

# ── ladder_arm <arm> <warm> <boxvar> <box_r> <box_v> <ref_ckpt> <ref_wm> ────
# The tight-box ladder rung. Floor rows run BEFORE training so the tripwire has
# a floor to compare against even if the trainer dies, and so the campaign is
# self-contained (no reliance on the pre-launch survey's numbers).
#   <ref_ckpt>/<ref_wm>: the box's own published parent, evaluated at i_t = 0
#   under J2. Reproduces the survey row for continuity (10 km: 113/200;
#   5 km: 0/200) and pins the "J2-blind at this box" number in-campaign.
ladder_arm() {
    local arm="$1" warm="$2" boxvar="$3" br="$4" bv="$5" refck="$6" refwm="$7"
    if [ ! -f "$warm" ]; then
        say "RESULT $arm ABORTED: warm start missing ($warm)."
        say "      $arm needs the previous ladder stage's child. Run the stages"
        say "      in order, or pass the full list J2_RUNG_STAGES=0,1,2,3a,3b."
        return 1
    fi
    say "START $arm (box ${br}m/${bv}m/s, warm=$(basename $warm))"
    # the bootstrap floor the arm must actually beat: ITS OWN warm start,
    # at this box, under J2 and the sampler
    one_eval "${arm}_floor_chain" "$warm"  1 "$BAND" 1 0 "$br" "$bv" 0.8166667
    # and the box's published parent, i_t = 0, for continuity with the survey
    one_eval "${arm}_floor_ref"   "$refck" 1 off     1 0 "$br" "$bv" "$refwm"
    if train_arm "$arm" "$warm" "$boxvar"; then
        flatline "$arm" "$JSON_DIR/${arm}_floor_chain.json"
        local ck; ck=$(last_ckpt "$arm")
        say "  $arm ckpt $ck"
        one_eval "${arm}_native"    "$ck" 1 "$BAND" 1 0 "$br" "$bv" 0.8166667
        one_eval "${arm}_retention" "$ck" 0 "$BAND" 1 0 "$br" "$bv" 0.8166667
        say "---- $arm done ----"
        return 0
    fi
    # Training that produces no final ckpt is still worth a tripwire read: the
    # log may show a clean flatline rather than a crash, and that is the finding.
    flatline "$arm" "$JSON_DIR/${arm}_floor_chain.json"
    say "---- $arm FAILED (no final ckpt) — see the flatline check above ----"
    return 1
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== ext-j2 rung campaign start (pid $$) ==========="
say "ALL CLAIMS ARE IN MEAN ELEMENTS (no mean/osculating conversion anywhere)"
say "task: dim3=1 di_max=1.0deg e<=0.05 LEO gap=pi D30 shaping2 dv_ref=700"
say "      gamma=1.0 phase modes 1 cap=3000 no debris, truth state (nav is OUT)"
say "J2:   j2_mode=1 i_t ~ U($BAND) deg, raan fixed, lvlh_frame_mode=1"
say "warm: $(basename $WARM)"
say "stages requested: $STAGES"

preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED (explicitly excluded) ----"; fi

# ── stage 1: the floor and its controls. Eval only. ─────────────────────────
if want 1; then
  say "START stage 1 (zero-shot J2 gap + controls, eval only)"
  # the published reference point
  one_eval "s1_ref_X3"            "$WARM" 0 off    1 0 30000 50 0.8166667
  # A0: inclination sampling WITHOUT J2. Isolates "i_t sampled" from "J2".
  # Without this the comparison is confounded (j2_A_design §4.4).
  one_eval "s1_A0_incl_j2off"     "$WARM" 0 $BAND  1 0 30000 50 0.8166667
  # A0-legacy: the same, with the UNFIXED LVLH frame. Quantifies the confound
  # that would otherwise be silently folded into the J2 number.
  one_eval "s1_A0_incl_lvlhbug"   "$WARM" 0 $BAND  0 0 30000 50 0.8166667
  # A1: the floor. Only j2_mode differs from A0.
  one_eval "s1_A1_incl_j2on"      "$WARM" 1 $BAND  1 0 30000 50 0.8166667
  # A1-equatorial: J2 with the plane channel PROVABLY INERT. Separates the
  # in-plane secular drift from the plane channel.
  one_eval "s1_A1_equatorial"     "$WARM" 1 off    1 0 30000 50 0.8166667
  # SO(2) leak detector: raan sampled. Diagnostic only, never trained on.
  one_eval "s1_A1_raansampled"    "$WARM" 1 $BAND  1 1 30000 50 0.8166667
  # tight-box reference cells, to keep the box-sensitivity table on the record
  one_eval "s1_TB5_j2off"     "$TB_WARM" 0 off 1 0 5000 1 0.35
  one_eval "s1_TB5_j2on"      "$TB_WARM" 1 off 1 0 5000 1 0.35
  say "---- stage 1 done ----"
else say "---- stage 1 SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stage 2: the treatment. Loose box, truth state, sampler on. ─────────────
if want 2; then
  say "START stage 2 (J2-trained arm, 50M warm from X3)"
  if train_arm "A2_j2trained" "$WARM" X3_BOX; then
      ck=$(last_ckpt A2_j2trained)
      say "  ckpt $ck"
      # native: the condition it trained under
      one_eval "s2_A2_native"    "$ck" 1 $BAND 1 0 30000 50 0.8166667
      # retention: does learning J2 cost anything at j2_mode=0?
      one_eval "s2_A2_retention" "$ck" 0 $BAND 1 0 30000 50 0.8166667
      # and at the ORIGINAL pinned-plane X3 rung, the lineage's home cell
      one_eval "s2_A2_X3home"    "$ck" 0 off   1 0 30000 50 0.8166667
      say "---- stage 2 done ----"
  else
      say "---- stage 2 FAILED (no final ckpt) ----"
  fi
else say "---- stage 2 SKIPPED (not in STAGES=$STAGES) ----"; fi

# ── stages 3a/3b: THE TIGHT-BOX LADDER — the headline experiment ────────────
# Not in the default stage list; pass J2_RUNG_STAGES=0,1,2,3a,3b at launch.
# Each rung warm-starts from the PREVIOUS rung's child, so the policy walks
# 30km/50 -> 10km/10 -> 5km/1 instead of bootstrapping from a 0/200 floor in
# one jump. shape_w_match stays 0.8166667 throughout: the chain's parent is the
# stage-2 child, which is X3 lineage, and an arm must inherit its OWN parent's
# value — mixing in the TB3D ladder's 0.35 here would move the reward alongside
# the box, and every T3-era collapse in this project traces to a compound
# change. (The 0.35 lineage still appears, correctly, in the *_floor_ref rows,
# which evaluate the TB3D parents under their own training config.)
#
# The zero-shot floors these arms start from, from the pre-launch survey:
#   10 km / 10 m/s under J2:  113/200 = 56.5%   (safety_cap 87)
#    5 km /  1 m/s under J2:    0/200 =  0.0%   (safety_cap 200)
# The 5 km rung is the one at genuine risk of not bootstrapping — hence the
# tripwire, and hence the 10 km rung existing at all.
if want 3a; then
  say "START stage 3a (tight-box ladder rung 1: 10 km / 10 m/s)"
  W3A=$(last_ckpt A2_j2trained)
  if [ -z "${W3A:-}" ]; then
      say "RESULT 3a ABORTED: stage 2 child not found. Run stage 2 first."
  else
      ladder_arm "A3a_j2_box10k10" "$W3A" B1_BOX 10000 10 "$TB1_REF" 0.35
  fi
else say "---- stage 3a SKIPPED (not in STAGES=$STAGES) ----"; fi

if want 3b; then
  say "START stage 3b (tight-box ladder rung 2: 5 km / 1 m/s — the headline)"
  W3B=$(last_ckpt A3a_j2_box10k10)
  if [ -z "${W3B:-}" ]; then
      say "RESULT 3b ABORTED: stage 3a child not found. Run stage 3a first."
  else
      ladder_arm "A3b_j2_box5k1" "$W3B" TB5_BOX 5000 1 "$TB_WARM" 0.35
      # The claim boundary, once more, where a reader of the log will hit it.
      say "NOTE 3b: every j2=1 row above is a MEAN-ELEMENT claim. At 5 km / 1 m/s"
      say "     3b: the osculating velocity slip is 9.4% of tolerance per orbit,"
      say "     3b: so the honest statement is 'meets the box in mean elements',"
      say "     3b: full stop — never 'osculating-grade rendezvous'."
      say "NOTE 3b: if 3b flatlined, that is a FINDING, not a failed campaign:"
      say "     3b: it would be the first measured case of J2 defeating the"
      say "     3b: warm-start ladder, and the tripwire names where it stalled."
  fi
else say "---- stage 3b SKIPPED (not in STAGES=$STAGES) ----"; fi

say "=========== ext-j2 rung campaign end ==========="
