#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# T11-TIGHT — THE 6/7 ATTEMPT: bootstrap the tight box back into the generalist
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/extj2/t11_tight_campaign.sh \
#       > /tmp/t11t_stdout.log 2>&1 &
#
# TO KILL A RUN BY HAND, KILL THE WORKERS FIRST. `pkill -f` on the trainer's
# command line does NOT match its rollout workers — they spawn as
# `python -c "from multiprocessing.spawn import ..."` and carry none of the
# launch flags — so killing the parent alone ORPHANS eight processes that keep
# spinning at ~170% CPU each. Use `pkill -TERM -P <pid>` then `kill -TERM <pid>`.
# The watchdog below already does exactly that, in that order.
#
# ── WHY A LADDER AND NOT MORE MIXTURE WEIGHT ────────────────────────────────
# The rung-B child floors at ~0% on TIGHT. That is a BOOTSTRAP failure, not an
# under-weighting: a cell returning no reward contributes no gradient, so
# raising its mixture weight from 0.10 to 0.25 multiplies zero by 2.5. The
# project has hit this three times now (Phase 3 stage 1, R4 stage 1, TB5-3D),
# and the fix has been the same every time — a ladder that hands the policy a
# box it can already sometimes hit, then tightens. Hence 30 km (already held)
# -> 10 km / 10 m/s -> 5 km / 1 m/s.
#
# ── SELF-RED-TEAM ───────────────────────────────────────────────────────────
#
# (a) THE MEAN-vs-OSCULATING BOUNDARY IS *CROSSED* AT TB5, AND IT IS LABELLED.
#     `propagate_orbit_j2` applies SECULAR RATES ONLY — Omega-dot, omega-dot,
#     M-dot. There are no short-period terms, so every element in this sim is a
#     MEAN element and the Cartesian conversion treats mean as osculating. The
#     first-order Brouwer short-period radial signature that is being neglected:
#
#         a = 6.671e6 m  ->  amplitude 3301 m  (peak-peak 6602 m)
#         a = 6.921e6 m  ->  amplitude 3182 m  (peak-peak 6363 m)
#         a = 7.171e6 m  ->  amplitude 3071 m  (peak-peak 6142 m)
#
#     Against the boxes:
#         30 km box : 0.21x  — small, which is why the E-cells never needed
#                              the caveat said loudly
#         10 km box : 0.64x  — MEAN-ELEMENT LABEL REQUIRED
#          5 km box : 1.27x  — THE NEGLECTED PHYSICS IS LARGER THAN THE BOX
#
#     So a TB5 "success" is a statement about mean elements. Against an
#     osculating truth the short-period radial term alone would carry the
#     chaser out of a 5 km box and back twice per orbit. This does NOT
#     invalidate the rung — the ladder is a real control problem in the sim's
#     own dynamics — but every tight row must be labelled MEAN ELEMENTS, per
#     J2_RESULTS discipline, and no tight number may be quoted as a rendezvous
#     capability claim without that label. The evaluator already stamps
#     `state='MEAN ELEMENTS'` into every JSON; stage 4 echoes it into the log.
#
# (b) THE FUEL FLOOR AT THE TIGHT BOX WAS TOO LOW, AND IS RAISED. MEASURED.
#     The tight box needs a terminal fine-burn train (~27 x 1 m/s at TB5) that
#     the transfer-only feasibility estimate does not price. Infeasible mass for
#     transfer + 27 m/s over 1024 draws in the tight band:
#
#         352.8 m/s (fuel 0.1130, the mixture floor) -> 15.8%   ABOVE G4's 12%
#         418.9 m/s (fuel 0.1327)                    ->  2.1%   <- shipped
#         499.0 m/s (fuel 0.1560)                    ->  0.1%
#
#     TB5 headroom at the 353 m/s floor is p10 = 10.1 m/s, i.e. one episode in
#     six can complete the TRANSFER and then cannot afford the fine-burn train
#     that the box actually requires. Training a bootstrap rung on 15.8%
#     unwinnable episodes is poison. The tight TRAINING cells therefore ship
#     fuel_min = 0.133 (419-656 m/s, a 1.57x spread that still carries
#     budget-awareness signal). Stage 4 evaluates the tight cell BOTH ways so
#     the gain stays comparable to the existing T11 lineage:
#         TIGHT_5k1_T  (0.133) — the training distribution
#         TIGHT_5k1    (0.113) — the shipped mixture cell, whose ~84%
#                                feasibility ceiling is stated, not hidden
#
# (c) CAP 3000 IS KEPT. It is the setting proven at TB5 in the narrow pairing,
#     and 3000 substeps is 50 h — about 30 orbits — for a same-band transfer,
#     so the cap is not the binding constraint here; fuel is (see (b)). Changed
#     only if a stage-2/3 run shows safety_cap dominating the cause histogram,
#     which the RESULT lines print every time.
#
# ── WHAT WOULD MAKE THIS A FAILED CAMPAIGN ──────────────────────────────────
# Not "TIGHT stays low". The failure that matters is TIGHT going up while
# E0-E3/LONGRANGE come down — buying the sixth cell by breaking the five that
# already work. Stage 4 is the verdict, and stage 5 is the gated remedy.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/extj2/t11_eval.py
GATES=$MAIN/scripts/orbital/extj2/t11_gates.py
VERIFY=$MAIN/scripts/orbital/extj2/verify_extj2.py
ROOT_B=$MAIN/models/t3/t11_generalist_rungB.pt
PROG=/tmp/t11t_progress.log
JSON_DIR=$MAIN/web_data/results/t11_tight
EPS=200
EVAL_SEED=123
STEPS_T1=${T11T_STEPS_T1:-50000000}
STEPS_T2=${T11T_STEPS_T2:-50000000}
STEPS_MIX=${T11T_STEPS_MIX:-50000000}
WATCHDOG_S=900
NAV_MAX_TICKS=${T11T_NAV_MAX_TICKS:-120}

# Flatline tripwire: an arm that has not beaten its own floor by this margin
# after this many steps is not learning, and 30 more hours will not change that.
TRIP_AFTER=${T11T_TRIP_AFTER:-20000000}
TRIP_MARGIN_PP=${T11T_TRIP_MARGIN_PP:-5}
TRIP_EPS=${T11T_TRIP_EPS:-60}

T11T_SEED=${T11T_SEED:-42}
SEED_SFX=""
[ "$T11T_SEED" != "42" ] && SEED_SFX="_s${T11T_SEED}"
STAGES=${T11T_STAGES:-0,1,2,3,4,5}
want() { [[ ",$STAGES," == *",$1,"* ]]; }

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

EPOCH_SLACK=${T11T_EPOCH_SLACK:-3}
final_ckpt() {
    local dir="$1" want_ep="$2"
    ls -t "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null | while read -r f; do
        local n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$((want_ep - EPOCH_SLACK))" ] && { echo "$f"; return 0; }
    done
    return 0
}
arm_dir()   { echo "$PUF/experiments_t11_tight/$1${SEED_SFX}"; }
last_ckpt() { final_ckpt "$(arm_dir "$1")" "$2"; }

# The checkpoint nearest a given step count, for the tripwire.
ckpt_at() {
    local dir="$1" ep="$2" f n
    for f in $(ls "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null | sort); do
        n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$ep" ] && { echo "$f"; return 0; }
    done
    return 0
}

json_rate() {   # <json> -> success rate as an integer percent, or empty
    python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try: print(int(round(100*json.load(open(sys.argv[1]))['rate'])))
except Exception: pass
PY
}

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: $MAIN on '$br', need '$BRANCH_REQ'"; return 1; }
    for f in "$EVAL" "$GATES" "$VERIFY" "$ROOT_B"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    python3 - <<'PY' || { say "ABORT: tight cells not resolvable"; return 1; }
import sys
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/extj2')
import t11_cells as T
for n in ('TIGHT_10k10', 'TIGHT_5k1_T'):
    assert n in T.ALL_CELLS, n
assert abs(sum(c['weight'] for c in T.TABLE) - 1.0) < 1e-9, 'mixture weights moved'
assert len(T.CELLS) == 7, 'mixture gained a row'
PY
    mkdir -p "$JSON_DIR"; return 0
}

# ── stage 0: anchors + the T11 gate battery ────────────────────────────────
anchor() {
    say "START stage 0 (anchors, both families, + the T11 gates)"
    cd "$MAIN" || return 1
    python3 "$VERIFY" --stage a1,a2,a3,a4,a5 --eps 100 > /tmp/t11t_anchor.log 2>&1
    local ra=$?
    say "  RESULT anchors rc=$ra $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/t11t_anchor.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t11t_anchor.log | while read -r l; do say "    $l"; done
    python3 "$GATES" > /tmp/t11t_gates.log 2>&1
    local rg=$?
    say "  RESULT t11 gates rc=$rg $(grep -oE '[0-9]+/[0-9]+ gates pass' /tmp/t11t_gates.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t11t_gates.log | while read -r l; do say "    $l"; done
    if [ $ra -ne 0 ] || [ $rg -ne 0 ]; then say "ABORT: stage 0 FAILED"; return 1; fi
    say "---- stage 0 done ----"; return 0
}

# ── one_eval <tag> <ckpt> <cell|all> <nav_mode> [extra...] ─────────────────
one_eval() {
    local tag="$1" ck="$2" cell="$3" nav="$4"; shift 4
    cd "$MAIN" || return 1
    case "$tag" in
        F_*) [ -s "$JSON_DIR/${tag}.json" ] && { say "SKIP eval $tag (floor, present)"; return 0; } ;;
        *)   tag="${tag}${SEED_SFX}" ;;
    esac
    local L=/tmp/t11t_${tag}.log
    python3 "$EVAL" --ckpt "$ck" --cell "$cell" --nav-mode "$nav" \
        --episodes $EPS --seed $EVAL_SEED --label "$tag" \
        --out "$JSON_DIR/${tag}.json" "$@" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    for pat in 'causes:' 'FUEL AUDIT' 'dv used' 'per-cell'; do
        g=$(grep -F "$pat" "$L" | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    done
}

# ── train_tight <arm> <warm> <steps> <cell> <floor_json> ───────────────────
# Single-cell, COMPLETE spec. bug-#15's lesson: every cell field goes on the
# command line explicitly. A field left off does not error — it silently falls
# back to the ini default, and the arm trains on a cell nobody described.
train_tight() {
    local arm="$1" warm="$2" steps="$3" cell="$4" floor_json="${5:-}"
    local dir; dir=$(arm_dir "$arm")
    local want_ep=$(( steps / 131072 ))
    if [ -n "$(final_ckpt "$dir" "$want_ep")" ]; then
        say "SKIP train $arm (final ckpt present)"; return 0
    fi
    say "START train $arm seed=$T11T_SEED steps=$steps cell=$cell warm=$(basename "$warm")"
    cd "$PUF" || return 1

    # Read the COMPLETE cell spec out of the single source of truth and emit it
    # as explicit --env flags, so the log records exactly what was trained.
    local CELLFLAGS
    CELLFLAGS=$(python3 - "$cell" <<'PY'
import sys
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/extj2')
import t11_cells as T
c = T.ALL_CELLS[sys.argv[1]]
print(' '.join([
    f"--env.episode-cap-steps {int(c['cap'])}",
    f"--env.rendezvous-radius-m {c['box_r']}",
    f"--env.rel-vel-tol-ms {c['box_v']}",
    f"--env.a-min-override {c['a_min']}",
    f"--env.a-max-override {c['a_max']}",
    f"--env.e-max-target {c['e_max_target']}",
    f"--env.e-max-sat {c['e_max_sat']}",
    f"--env.de-max {c['de_max']}",
    f"--env.da-max-m {c['da_max']}",
    f"--env.di-max-rad {c['di_max']}",
    f"--env.di-min-rad {c['di_min']}",
    f"--env.di-phase-mode {int(c['di_phase'])}",
    f"--env.j2-mode {int(c['j2'])}",
    f"--env.nav-j2-mode {int(c['j2'])}",
    f"--env.i-target-min-rad {c['i_t_min']}",
    f"--env.i-target-max-rad {c['i_t_max']}",
    f"--env.fuel-frac-min {c['fuel_min']}",
    f"--env.fuel-frac-max {c['fuel_max']}",
]))
PY
) || { say "  ABORT: could not resolve cell $cell"; return 1; }
    say "  cell spec: $CELLFLAGS"

    # NO inner caffeinate: the script runs under one, and wrapping again makes
    # $! the wrapper's PID; caffeinate does not forward signals, so the watchdog
    # would kill the wrapper and orphan the trainer.
    # shellcheck disable=SC2086
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps "$steps" --train.seed "$T11T_SEED" \
        --train.data-dir "$dir" --load-model-path "$warm" \
        --env.num-debris-min 0 --env.num-debris-max 0 \
        --env.same-orbit-init 0 --env.init-phase-gap-max 3.14159 \
        --env.valid-init-only 1 --env.gave-up-action terminate \
        --env.max-valid-init-attempts 4096 \
        --env.obs-alt-scale-m 8.0e6 --env.lvlh-scale-m 6.371e6 \
        --env.shaping-mode 2 --env.shape-w-lambda 1.0 \
        --env.shape-w-match 0.8166667 --env.shape-dv-ref-ms 700.0 \
        --env.shape-gamma 1.0 --env.phase-gap-mode 1 --env.phase-obs-mode 1 \
        --env.cap-terminal-reward 0.0 --env.dim3-mode 1 \
        --env.lvlh-frame-mode 1 --env.raan-target-sample 0 \
        --env.legacy-action-space 31 \
        --env.nav-mode bearings_only \
        --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0 \
        --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20 \
        --env.nav-acq-mode crlb_online --env.nav-max-ticks "$NAV_MAX_TICKS" \
        --env.t11-mixture 0 --env.cell-mixture-mode 0 \
        $CELLFLAGS \
        --wandb --wandb-project orbital-rl --wandb-group "t11t-${arm}${SEED_SFX}" \
        --tag "t11t_${arm}${SEED_SFX}" > "/tmp/t11t_${arm}${SEED_SFX}_train.log" 2>&1 &
    local pid=$!; say "  trainer pid $pid"

    local t_final=0 tripped=0
    local trip_ep=$(( TRIP_AFTER / 131072 ))
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        # ── flatline tripwire ──────────────────────────────────────────────
        if [ "$tripped" -eq 0 ] && [ -n "$floor_json" ] && \
           [ -n "$(ckpt_at "$dir" "$trip_ep")" ]; then
            tripped=1
            local fl tc tr
            fl=$(json_rate "$JSON_DIR/${floor_json}.json")
            tc=$(ckpt_at "$dir" "$trip_ep")
            if [ -n "$fl" ] && [ -n "$tc" ]; then
                python3 "$EVAL" --ckpt "$tc" --cell "$cell" --nav-mode bearings_only \
                    --episodes $TRIP_EPS --seed $EVAL_SEED --label "trip_${arm}" \
                    --out "$JSON_DIR/trip_${arm}${SEED_SFX}.json" \
                    > "/tmp/t11t_trip_${arm}.log" 2>&1
                tr=$(json_rate "$JSON_DIR/trip_${arm}${SEED_SFX}.json")
                say "  TRIPWIRE $arm at ~$((TRIP_AFTER/1000000))M: ${tr:-?}% vs floor ${fl}% (+${TRIP_MARGIN_PP}pp required)"
                if [ -n "$tr" ] && [ "$tr" -lt "$((fl + TRIP_MARGIN_PP))" ]; then
                    say "  RESULT $arm TRIPWIRE FIRED — not learning; killing the arm."
                    say "         A cell that has not moved off its floor in $((TRIP_AFTER/1000000))M will not"
                    say "         move in the remaining $(( (steps-TRIP_AFTER)/1000000 ))M. Read this as a BOOTSTRAP failure:"
                    say "         the next lever is a looser intermediate box, not more steps."
                    pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
                    sleep 20
                    pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
                    return 1
                fi
            fi
        fi
        if [ "$t_final" -eq 0 ] && [ -n "$(final_ckpt "$dir" "$want_ep")" ]; then
            t_final=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
        fi
        if [ "$t_final" -ne 0 ] && [ $(( $(date +%s) - t_final )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 20
            pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; break
        fi
    done
    [ -n "$(final_ckpt "$dir" "$want_ep")" ] || { say "  RESULT $arm NO FINAL CKPT"; return 1; }
    say "  RESULT $arm trained OK"; return 0
}

CELLS_ALL="E0_j2 E1_j2 E2_j2 E3_j2 W1_driftwait TIGHT_5k1 LONGRANGE"
GREEN_CELLS="E0_j2 E1_j2 E2_j2 E3_j2 LONGRANGE"

# ═══════════════════════════════════════════════════════════════════════════
say "=========== T11-tight campaign start (pid $$) ==========="
say "goal: bootstrap the tight box into the T11 generalist, then price the cost"
say "ladder: 30 km (held) -> 10 km/10 m/s -> 5 km/1 m/s, single-cell, COMPLETE spec"
say "root:   $(basename $ROOT_B)"
say "fuel:   tight cells use fuel_min 0.133 (419 m/s), NOT the mixture 0.113 —"
say "        at 353 m/s 15.8% of TB5 episodes cannot afford the fine-burn train"
say "STATE:  ALL TIGHT ROWS ARE MEAN-ELEMENT CLAIMS. At the 5 km box the"
say "        neglected J2 short-period radial signature is 1.27x the box."
say "stages requested: $STAGES   seed $T11T_SEED${SEED_SFX:+ (suffixed)}"
preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED ----"; fi

# ── stage 1: floors (eval only) ────────────────────────────────────────────
if want 1; then
  say "START stage 1 (floors: rung-B child zero-shot at both tight boxes)"
  one_eval "F_rungB_TIGHT_10k10" "$ROOT_B" TIGHT_10k10 bearings_only
  one_eval "F_rungB_TIGHT_5k1_T" "$ROOT_B" TIGHT_5k1_T bearings_only
  one_eval "F_rungB_TIGHT_5k1"   "$ROOT_B" TIGHT_5k1   bearings_only
  say "  READ: these are the numbers stages 2/3 must beat, and the tripwire's"
  say "        reference. ~0% at 5 km is EXPECTED and is the whole premise; a"
  say "        non-trivial floor at 10 km is what makes the ladder viable."
  say "---- stage 1 done ----"
else say "---- stage 1 SKIPPED ----"; fi

# ── stage 2: T-rung 10 km / 10 m/s ─────────────────────────────────────────
if want 2; then
  say "START stage 2 (T-rung 10 km/10 m/s, ${STEPS_T1} steps, warm from rung-B child)"
  if train_tight "T10k10" "$ROOT_B" "$STEPS_T1" TIGHT_10k10 "F_rungB_TIGHT_10k10"; then
      ck=$(last_ckpt T10k10 $(( STEPS_T1 / 131072 ))); say "  ckpt $ck"
      one_eval "s2_T10k10"    "$ck" TIGHT_10k10 bearings_only
      one_eval "s2_T10k10_tr" "$ck" TIGHT_10k10 truth
      say "---- stage 2 done ----"
  else say "---- stage 2 FAILED (tripwire or no ckpt) ----"; fi
else say "---- stage 2 SKIPPED ----"; fi

# ── stage 3: T-rung 5 km / 1 m/s ───────────────────────────────────────────
if want 3; then
  say "START stage 3 (T-rung 5 km/1 m/s, ${STEPS_T2} steps, warm from stage-2 child)"
  WARM3=${T11T_WARM3:-$(last_ckpt T10k10 $(( STEPS_T1 / 131072 )))}
  if [ -z "$WARM3" ]; then
      say "  SKIP: no stage-2 child. Rung 3 warm-starts from the 10 km rung BY"
      say "        DESIGN — starting it from rung B would be the same bootstrap"
      say "        failure stage 2 exists to fix. Override with T11T_WARM3."
  else
      say "  warm: $(basename "$WARM3")"
      if train_tight "T5k1" "$WARM3" "$STEPS_T2" TIGHT_5k1_T "F_rungB_TIGHT_5k1_T"; then
          ck=$(last_ckpt T5k1 $(( STEPS_T2 / 131072 ))); say "  ckpt $ck"
          one_eval "s3_T5k1"    "$ck" TIGHT_5k1_T bearings_only
          one_eval "s3_T5k1_tr" "$ck" TIGHT_5k1_T truth
          say "---- stage 3 done ----"
      else say "---- stage 3 FAILED (tripwire or no ckpt) ----"; fi
  fi
else say "---- stage 3 SKIPPED ----"; fi

# ── stage 4: THE VERDICT — full 7-cell battery on the stage-3 child ────────
if want 4; then
  say "START stage 4 (VERDICT: full 7-cell battery + fuel decomposition)"
  CK4=${T11T_CK4:-$(last_ckpt T5k1 $(( STEPS_T2 / 131072 )))}
  if [ -z "$CK4" ]; then say "  SKIP: no stage-3 child"; else
    say "  ckpt $(basename "$CK4")"
    for C in $CELLS_ALL; do
      one_eval "s4_${C}"    "$CK4" "$C" bearings_only
      one_eval "s4_${C}_tr" "$CK4" "$C" truth
    done
    # the tight cell at its TRAINING fuel range, and the lean/rich slices
    one_eval "s4_TIGHT_5k1_T"      "$CK4" TIGHT_5k1_T bearings_only
    one_eval "s4_tight_lean"       "$CK4" TIGHT_5k1_T bearings_only --fuel-fixed 0.133
    one_eval "s4_tight_rich"       "$CK4" TIGHT_5k1_T bearings_only --fuel-fixed 0.200
    say "  READ (gain): TIGHT_5k1_T vs stage-1 floor F_rungB_TIGHT_5k1_T."
    say "  READ (comparability): TIGHT_5k1 is the SHIPPED mixture cell at fuel"
    say "        0.113 — its ceiling is ~84% because 15.8% of episodes there"
    say "        cannot afford the fine-burn train. Do not read it as skill."
    say "  READ (cost): E0-E3 + LONGRANGE vs the rung-B lineage's own numbers."
    say "  STATE: every tight row above is a MEAN-ELEMENT claim (see header (a))."
    say "---- stage 4 done ----"
  fi
else say "---- stage 4 SKIPPED ----"; fi

# ── stage 5: GATED rehearsal re-mix ────────────────────────────────────────
# Auto-decides. Runs only if stage 4 shows the tight training broke a cell that
# was green, which is the one outcome that makes 6/7 not real.
if want 5; then
  say "START stage 5 (gated: rehearsal re-mix — runs only if retention dropped)"
  CK5=${T11T_CK5:-$(last_ckpt T5k1 $(( STEPS_T2 / 131072 )))}
  if [ -z "$CK5" ]; then say "  SKIP: no stage-3 child"; else
    NEED=0; WORST=""
    for C in $GREEN_CELLS; do
      r=$(json_rate "$JSON_DIR/s4_${C}${SEED_SFX}.json")
      if [ -z "$r" ]; then
        say "  RESULT gate: s4_${C} missing — cannot decide, treating as NO-GO"
      elif [ "$r" -lt 90 ]; then
        NEED=1; WORST="$WORST ${C}=${r}%"
        say "  RESULT gate: ${C} at ${r}% is BELOW the 90% retention floor"
      else
        say "  RESULT gate: ${C} at ${r}% holds"
      fi
    done
    if [ "$NEED" -eq 0 ]; then
      say "  RESULT stage 5 NOT REQUIRED — every green cell held >= 90%."
      say "         The tight skill was bought without breaking the envelope,"
      say "         which is the result that makes 6/7 real."
    else
      say "  RESULT stage 5 REQUIRED — retention dropped:$WORST"
      say "         Running ${STEPS_MIX} of the ORIGINAL 7-cell mixture from the"
      say "         stage-3 child. This is rehearsal, not a new rung: the point"
      say "         is to recover the green cells WITHOUT discarding the tight"
      say "         skill, so the re-eval below has to show BOTH."
      dir=$(arm_dir "REMIX"); want_ep=$(( STEPS_MIX / 131072 ))
      if [ -n "$(final_ckpt "$dir" "$want_ep")" ]; then
        say "  SKIP train REMIX (final ckpt present)"
      else
        cd "$PUF" || exit 1
        python3 -m pufferlib.pufferl train puffer_orbital_nav \
          --train.device cpu --train.total-timesteps "$STEPS_MIX" --train.seed "$T11T_SEED" \
          --train.data-dir "$dir" --load-model-path "$CK5" \
          --env.num-debris-min 0 --env.num-debris-max 0 \
          --env.same-orbit-init 0 --env.init-phase-gap-max 3.14159 \
          --env.valid-init-only 1 --env.gave-up-action terminate \
          --env.max-valid-init-attempts 4096 \
          --env.obs-alt-scale-m 8.0e6 --env.lvlh-scale-m 6.371e6 \
          --env.shaping-mode 2 --env.shape-w-lambda 1.0 \
          --env.shape-w-match 0.8166667 --env.shape-dv-ref-ms 700.0 \
          --env.shape-gamma 1.0 --env.phase-gap-mode 1 --env.phase-obs-mode 1 \
          --env.cap-terminal-reward 0.0 --env.dim3-mode 1 --env.j2-mode 1 \
          --env.nav-j2-mode 1 --env.lvlh-frame-mode 1 --env.raan-target-sample 0 \
          --env.legacy-action-space 31 \
          --env.i-target-min-rad 0.5235987755982988 \
          --env.i-target-max-rad 1.0471975511965976 \
          --env.nav-mode bearings_only \
          --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0 \
          --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20 \
          --env.nav-acq-mode crlb_online --env.nav-max-ticks "$NAV_MAX_TICKS" \
          --env.t11-mixture 1 \
          --wandb --wandb-project orbital-rl --wandb-group "t11t-REMIX${SEED_SFX}" \
          --tag "t11t_REMIX${SEED_SFX}" > "/tmp/t11t_REMIX${SEED_SFX}_train.log" 2>&1 &
        rpid=$!; say "  trainer pid $rpid"
        tf=0
        while kill -0 "$rpid" 2>/dev/null; do
          sleep 60
          if [ "$tf" -eq 0 ] && [ -n "$(final_ckpt "$dir" "$want_ep")" ]; then
            tf=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
          fi
          if [ "$tf" -ne 0 ] && [ $(( $(date +%s) - tf )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: killing $rpid"
            pkill -TERM -P "$rpid" 2>/dev/null; kill -TERM "$rpid" 2>/dev/null; sleep 20
            pkill -KILL -P "$rpid" 2>/dev/null; kill -KILL "$rpid" 2>/dev/null; break
          fi
        done
      fi
      ck=$(final_ckpt "$dir" "$want_ep")
      if [ -z "$ck" ]; then say "  RESULT REMIX NO FINAL CKPT"; else
        for C in $CELLS_ALL; do one_eval "s5_${C}" "$ck" "$C" bearings_only; done
        one_eval "s5_TIGHT_5k1_T" "$ck" TIGHT_5k1_T bearings_only
        say "  READ: rehearsal succeeded only if the green cells are back >= 90%"
        say "        AND s5_TIGHT_5k1_T is still well above the stage-1 floor."
        say "        Green recovered but tight collapsed = the mixture simply"
        say "        overwrote the skill again, and 6/7 needs a different lever."
      fi
    fi
    say "---- stage 5 done ----"
  fi
else say "---- stage 5 SKIPPED ----"; fi

say "=========== t11-tight campaign COMPLETE ==========="
