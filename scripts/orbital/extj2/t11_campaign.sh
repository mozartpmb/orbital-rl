#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# T11 — THE GENERALIST: one policy, J2 x wide-e x bearings-only
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/extj2/t11_campaign.sh \
#       > /tmp/t11_stdout.log 2>&1 &
#
# TO KILL A RUN BY HAND, KILL THE WORKERS FIRST. `pkill -f` on the trainer's
# command line does NOT match its rollout workers -- they are spawned as
# `python -c "from multiprocessing.spawn import ..."` and carry none of the
# launch flags -- so killing the parent alone ORPHANS eight processes that keep
# spinning at ~170% CPU each. Two of those sets survived a T11 smoke and drove
# the box to load 78, which is what made the first smoke look "slow" when it was
# starved. Use:
#     pkill -TERM -P <trainer_pid>   # children first
#     kill  -TERM    <trainer_pid>   # then the parent
# The watchdog below already does exactly this, in that order; the trap is only
# for manual kills.
#
# ── WHAT CHANGED IN THE WORLD MODEL, AND WHY THE FAMILY IS WIDE ─────────────
# GEN_MATRIX called the narrow/wide normalizer split "the largest single effect
# in the matrix" (99 pp, against 51 pp for adding J2 itself) and recommended
# restating the E-ladder in NARROW normalizers. The T11 recon found two things:
#
#   1. That restatement is UNIMPLEMENTABLE. e and altitude are coupled by the
#      perigee keepout, so inside the widest band narrow can represent
#      (|obs0| p99 <= 1) the realized eccentricity tops out at p90 = 0.112 —
#      against E3's 0.257. Reaching E3-class e needs |obs0| to 4.92, i.e. 2.5x
#      OUTSIDE the declared Box(-2,2). gave_up was 0.0000 throughout: the limit
#      is representability, not feasibility.
#
#   2. The barrier is REMOVABLE EXACTLY. `Default.encode_observations` is
#      Sequential(Linear(38,128), GELU) on the RAW observation vector, so a
#      normalizer change is compensated by rescaling 7 weight columns. Verified
#      bit-identical action streams on a truth lineage (91/100 -> 91/100, md5
#      e81c41eb204c) and a bearings-only lineage (60/60 -> 60/60, md5
#      724f345ba8ee), against 1/100 and 0/60 untransplanted.
#
# So the family is WIDE for representability AND keeps the best warm roots by
# transplant. And the "family" is TWO knobs, not one:
#     obs_alt_scale_m = 8.0e6    (wide)   bound by the e requirement
#     lvlh_scale_m    = 6.371e6  (narrow) bound by the tight box
#
# ── ROOT SELECTION: A DISCREPANCY, RESOLVED BY MEASUREMENT ──────────────────
# The design brief named A3b-j2 as the rung-A warm start. A3b-j2 is a TRUTH-only
# lineage with no navigation training, so rooting a bearings-only rung there
# would discard the entire nav capability. Rung A therefore evaluates BOTH
# candidate roots zero-shot in stage 1 and the floors decide:
#     t11_root_j2bonav_wide  J2BO-nav transplanted   (nav + J2, X3-band e)
#     t11_root_eE3_wide      e-E3 transplanted       (nav + wide-e, no J2)
# A3b-j2 transplanted is kept as the rung-B generalist root, which is what its
# 41.8% off-diagonal lead actually argues for.
#
# ── THE MIXTURE ─────────────────────────────────────────────────────────────
# scripts/orbital/extj2/t11_cells.py is the single source of truth for the cell
# table; this script never restates it. Weights follow GEN_MATRIX's nesting
# result — tight superset loose (99%) and J2 superset two-body (91%) are
# ONE-DIRECTIONAL, so subset regimes get weight ZERO — and drift-and-wait gets
# the largest single weight (0.20) because no 30-row head can express it at all.
#
# ── FUEL ────────────────────────────────────────────────────────────────────
# fuel_frac ~ U(0.113, 0.20) -> 353-656 m/s, sampled per episode. The floor is
# MEASURED: at 245 m/s (the originally proposed 0.08) 46-59% of ordinary cells
# are Dv-infeasible, and `valid_init_only` cannot catch it because it rejects on
# perigee only. All three fuel-bonus normalization sites now divide by THIS
# episode's budget; left at the compile-time constant the drawn budget would
# become a reward multiplier that obs[6] makes visible to the policy.
#
# ── UNDER-TRAINING, STATED UP FRONT ─────────────────────────────────────────
# Rung B is 200M over 7 cells = ~28M/cell, against the 50M each specialist got
# for ONE cell. The generalist is under-trained relative to every specialist it
# is compared to. That is the honest cost of the comparison and belongs in the
# result, not in a footnote.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/extj2/t11_eval.py
GATES=$MAIN/scripts/orbital/extj2/t11_gates.py
VERIFY=$MAIN/scripts/orbital/extj2/verify_extj2.py
RESCALE=$MAIN/scripts/orbital/extj2/rescale_ckpt_normalizers.py
ROOT_A_J2BO=$MAIN/models/t11/t11_root_j2bonav_wide.pt
ROOT_A_EE3=$MAIN/models/t11/t11_root_eE3_wide.pt
ROOT_B=$MAIN/models/t11/t11_root_a3b_wide.pt
PROG=/tmp/t11_progress.log
JSON_DIR=$MAIN/web_data/results/t11
EPS=200
EVAL_SEED=123
STEPS_A=${T11_STEPS_A:-50000000}
STEPS_B=${T11_STEPS_B:-200000000}
WATCHDOG_S=900

# ── THE DAY-WARP TICK CAP — THIS NUMBER IS LOAD-BEARING, NOT A TUNING KNOB ──
# The nav wrapper sub-steps the filter once per minute of tau when
# nav_max_ticks=0, so row 30's tau=1440 costs 1440 EKF updates in ONE decision.
# That lifts the mean over a uniform policy from 22.0 ticks/decision (rows 0-29)
# to 67.7, and MEASURED throughput on the mixture falls 4.6x:
#
#     rows 0-29, K=0    22.5 env-steps/s        (the pre-T11 baseline)
#     rows 0-30, K=0     4.4 env-steps/s        <-- rung B = 5.4 DAYS
#     rows 0-30, K=120  19.8 env-steps/s        <-- rung B ~ 29 h
#
# Scaled onto a real trainer (j2nav T-J2BO-nav observed 2.2K SPS bearings-only),
# K=0 puts rung B at ~129 h and rung A at ~32 h. K=120 is what makes the
# campaign affordable at all.
#
# K is safe here ONLY because MAJOR-7 replaced the old "cap tau, tick at 60 s"
# bug with a fixed COUNT and an adaptive INTERVAL: n=min(tau,K),
# dt=tau*60/n, so filter and truth stay on one clock. Divergence vs K on a
# pure-day-warp policy (the worst case that exists):
#
#     K=0 0.000 | K=30 0.109 | K=60 0.041 | K=120 0.003 | K=240 0.003
#
# 120 is the knee — it ties the uncapped filter's divergence and 240's, at 11.6x
# the speed. It also leaves 28 of 31 action rows BIT-IDENTICAL (only tau=180,
# 360, 1440 are capped at all), and the warm-start roots, which trained at K=0,
# measure UNCHANGED under it: E0_j2 zero-shot 36/40 at K=0 and 36/40 at K=120,
# same 274.5 m/s of dv, at 2.25x the wall clock.
NAV_MAX_TICKS=${T11_NAV_MAX_TICKS:-120}

T11_SEED=${T11_SEED:-42}
SEED_SFX=""
[ "$T11_SEED" != "42" ] && SEED_SFX="_s${T11_SEED}"
STAGES=${T11_STAGES:-0,1,2,3,4}
want() { [[ ",$STAGES," == *",$1,"* ]]; }

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# An epoch is 131072 steps at the shipped 8w x 256 shape; derive the expected
# final epoch instead of hardcoding a tag that silently disarms the watchdog if
# the shape ever moves.
EPOCH_SLACK=${T11_EPOCH_SLACK:-3}
final_ckpt() {
    local dir="$1" want_ep="$2"
    ls -t "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null | while read -r f; do
        local n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$((want_ep - EPOCH_SLACK))" ] && { echo "$f"; return 0; }
    done
    return 0
}

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: $MAIN on '$br', need '$BRANCH_REQ'"; return 1; }
    for f in "$EVAL" "$GATES" "$VERIFY" "$RESCALE" "$ROOT_A_J2BO" "$ROOT_A_EE3" "$ROOT_B"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    mkdir -p "$JSON_DIR"; return 0
}

# ── stage 0: anchors + the T11 gates ───────────────────────────────────────
anchor() {
    say "START stage 0 (anchors, both families, + the T11 gates)"
    cd "$MAIN" || return 1
    python3 "$VERIFY" --stage a1,a2,a3,a4,a5 --eps 100 > /tmp/t11_anchor.log 2>&1
    local ra=$?
    say "  RESULT anchors rc=$ra $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/t11_anchor.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t11_anchor.log | while read -r l; do say "    $l"; done
    python3 "$GATES" > /tmp/t11_gates.log 2>&1
    local rg=$?
    say "  RESULT t11 gates rc=$rg $(grep -oE '[0-9]+/[0-9]+ gates pass' /tmp/t11_gates.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t11_gates.log | while read -r l; do say "    $l"; done
    # the transplant must still be bit-identical on this build
    python3 "$RESCALE" "$MAIN/models/t3/extj2_A3b_j2_box5k1.pt" /tmp/t11_tp_check.pt \
        --alt-from 1.6e6 --alt-to 8.0e6 --lvlh-from 6.371e6 --lvlh-to 6.371e6 \
        --verify --verify-mode truth --verify-eps 100 > /tmp/t11_transplant.log 2>&1
    local rt=$?
    say "  RESULT transplant bit-identity rc=$rt $(grep -E '\[(PASS|FAIL)\]' /tmp/t11_transplant.log | head -1 | sed 's/^ *//')"
    [ $ra -ne 0 ] || [ $rg -ne 0 ] || [ $rt -ne 0 ] && { say "ABORT: stage 0 FAILED"; return 1; }
    say "---- stage 0 done ----"; return 0
}

# ── one_eval <tag> <ckpt> <cell|all> <nav_mode> [extra] ─────────────────────
one_eval() {
    local tag="$1" ck="$2" cell="$3" nav="$4"; shift 4
    cd "$MAIN" || return 1
    case "$tag" in
        F_*) [ -s "$JSON_DIR/${tag}.json" ] && { say "SKIP eval $tag (floor, present)"; return 0; } ;;
        *)   tag="${tag}${SEED_SFX}" ;;
    esac
    local L=/tmp/t11_${tag}.log
    python3 "$EVAL" --ckpt "$ck" --cell "$cell" --nav-mode "$nav" \
        --episodes $EPS --seed $EVAL_SEED --label "$tag" \
        --out "$JSON_DIR/${tag}.json" "$@" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    for pat in 'causes:' 'FUEL AUDIT' 'dv used' 'per-cell'; do
        g=$(grep -F "$pat" "$L" | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    done
}

# ── train <arm> <warm> <steps> <mixture 0|1> ───────────────────────────────
train_arm() {
    local arm="$1" warm="$2" steps="$3" mix="$4"
    local dir="$PUF/experiments_t11/${arm}${SEED_SFX}"
    local want_ep=$(( steps / 131072 ))
    if [ -n "$(final_ckpt "$dir" $want_ep)" ]; then
        say "SKIP train $arm (final ckpt present)"; return 0
    fi
    say "START train $arm seed=$T11_SEED steps=$steps mixture=$mix warm=$(basename $warm)"
    cd "$PUF" || return 1
    # NO inner caffeinate: the script runs under one, and wrapping again makes
    # $! the wrapper's PID; caffeinate does not forward signals, so the watchdog
    # would kill the wrapper and orphan the trainer.
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps "$steps" --train.seed "$T11_SEED" \
        --train.data-dir "$dir" --load-model-path "$warm" \
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
        --env.t11-mixture "$mix" ${T11_RUNGA_ENV:-} \
        --wandb --wandb-project orbital-rl --wandb-group "t11-${arm}${SEED_SFX}" \
        --tag "t11_${arm}${SEED_SFX}" > "/tmp/t11_${arm}${SEED_SFX}_train.log" 2>&1 &
    local pid=$!; say "  trainer pid $pid"
    local t_final=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [ "$t_final" -eq 0 ] && [ -n "$(final_ckpt "$dir" $want_ep)" ]; then
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
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/t11_${arm}${SEED_SFX}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a} (expected final epoch ~$want_ep)"
    [ -n "$(final_ckpt "$dir" $want_ep)" ]
}
last_ckpt() { final_ckpt "$PUF/experiments_t11/$1${SEED_SFX}" "$2"; }

CELLS_ALL="E0_j2 E1_j2 E2_j2 E3_j2 W1_driftwait TIGHT_5k1 LONGRANGE"

# ═══════════════════════════════════════════════════════════════════════════
say "=========== T11 generalist campaign start (pid $$) ==========="
say "family: obs_alt_scale_m 8.0e6 (WIDE, for e) + lvlh_scale_m 6.371e6 (NARROW, for the tight box)"
say "fuel:   U(0.113, 0.20) -> 353-656 m/s, sampled per episode; floor is MEASURED"
say "shaping: w_match 0.8166667 / dv_ref 700 (same gradient as 0.35/300, later saturation)"
say "rung A $STEPS_A steps; rung B $STEPS_B steps over 7 cells = ~$((STEPS_B/7/1000000))M/cell"
say "UNDER-TRAINING IS EXPECTED: specialists got 50M for ONE cell each."
say "stages requested: $STAGES   seed $T11_SEED${SEED_SFX:+ (suffixed)}"
preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED ----"; fi

# ── stage 1: floors — every root, zero-shot, per cell ──────────────────────
if want 1; then
  say "START stage 1 (floors: transplanted roots zero-shot on every cell)"
  for C in $CELLS_ALL; do
    one_eval "F_j2bonav_${C}" "$ROOT_A_J2BO" "$C" bearings_only
    one_eval "F_eE3_${C}"     "$ROOT_A_EE3"  "$C" bearings_only
    one_eval "F_a3b_${C}"     "$ROOT_B"      "$C" truth
  done
  say "---- stage 1 done ----"
  say "READ: the rung-A root is whichever of j2bonav/eE3 floors higher on the"
  say "      E0/E1 J2 cells; that choice is DATA, not the brief's A3b-j2 (which"
  say "      is truth-only and would discard the nav capability)."
else say "---- stage 1 SKIPPED ----"; fi

# ── stage 2: rung A — the wide pairing proof ───────────────────────────────
if want 2; then
  say "START stage 2 (rung A: J2 + inclined in the wide pairing)"
  ROOT_A=${T11_ROOT_A:-$ROOT_A_J2BO}
  say "  root: $(basename $ROOT_A)  (override with T11_ROOT_A after reading stage 1)"
  if train_arm "rungA" "$ROOT_A" "$STEPS_A" 0; then
      ck=$(last_ckpt rungA $(( STEPS_A / 131072 ))); say "  ckpt $ck"
      for C in E0_j2 E1_j2; do
        one_eval "s2_rungA_${C}"    "$ck" "$C" bearings_only
        one_eval "s2_rungA_${C}_tr" "$ck" "$C" truth
      done
      say "  GATE: rung A must reach the J2BO-nav headline (94%) at E0_j2 and"
      say "        E1's own 97% at E1_j2, or the wide pairing is wrong and"
      say "        everything downstream moves."
      say "---- stage 2 done ----"
  else say "---- stage 2 FAILED ----"; fi
else say "---- stage 2 SKIPPED ----"; fi

# ── stage 3: rung B — the generalist mixture ───────────────────────────────
if want 3; then
  say "START stage 3 (rung B: the 7-cell generalist, fuel sampled)"
  WARM_B=${T11_WARM_B:-$ROOT_B}
  if [ -n "$(last_ckpt rungA $(( STEPS_A / 131072 )))" ] && [ -z "${T11_WARM_B:-}" ]; then
      WARM_B=$(last_ckpt rungA $(( STEPS_A / 131072 )))
      say "  root: rung A child (gating satisfied)"
  else
      say "  root: $(basename $WARM_B)"
  fi
  if train_arm "rungB" "$WARM_B" "$STEPS_B" 1; then
      ck=$(last_ckpt rungB $(( STEPS_B / 131072 ))); say "  ckpt $ck"
      for C in $CELLS_ALL; do
        one_eval "s3_gen_${C}"    "$ck" "$C" bearings_only
        one_eval "s3_gen_${C}_tr" "$ck" "$C" truth
      done
      say "---- stage 3 done ----"
  else say "---- stage 3 FAILED ----"; fi
else say "---- stage 3 SKIPPED ----"; fi

# ── stage 4: the fuel-efficiency audit ─────────────────────────────────────
# The user's efficiency question gets its measurement from the start, not as a
# retrofit: per cell, Dv used against the cell's own direct-burn reference, and
# against the episode's SAMPLED budget.
if want 4; then
  say "START stage 4 (fuel-efficiency audit across the budget range)"
  ck=$(last_ckpt rungB $(( STEPS_B / 131072 )))
  if [ -z "$ck" ]; then say "  SKIP: no rung B child"; else
    for C in $CELLS_ALL; do
      one_eval "s4_lean_${C}" "$ck" "$C" bearings_only --fuel-fixed 0.113
      one_eval "s4_rich_${C}" "$ck" "$C" bearings_only --fuel-fixed 0.200
    done
    say "  READ: efficiency is the Dv-used ratio between the lean and rich"
    say "        arms on the SAME cell. A policy that ignores scarcity spends"
    say "        the same in both; one that responds spends less when lean."
    say "---- stage 4 done ----"
  fi
else say "---- stage 4 SKIPPED ----"; fi

say "=========== t11 campaign COMPLETE ==========="
