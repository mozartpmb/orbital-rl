#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# W1xNAV — bootstrap the drift-and-wait cell under bearings-only navigation
# The signal W1 was trained on in T11 was FICTIONAL on ~2 episodes in 3.
# This campaign trains on a signal that is measured to agree with the real
# batch IOD, and gates that agreement so it cannot silently regress.
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/ext_recon/w1nav_campaign.sh \
#       > /tmp/w1nav_stdout.log 2>&1 &
#
# TO KILL BY HAND, WORKERS FIRST: `pkill -TERM -P <pid>` then `kill -TERM <pid>`.
# `pkill -f` does not match the rollout workers and orphans eight of them.
#
# ── WHAT WAS MEASURED, AND WHY THE FIX IS A ONE-LINE FLAG ───────────────────
# The premise going in was "~24 h BLIND windows beyond the surrogate's validated
# boundary". Two things turned out to be wrong with that, and the second is the
# whole campaign.
#
# 1. A DAY-WARP IS NOT BLIND. `_tick3` runs predict/update/accumulate on EVERY
#    tick, and nav_max_ticks turns tau=1440 into n=min(1440,K) ticks at
#    dt = 1440*60/n. At the shipped K=120 that is 120 bearings at 12-minute
#    spacing, not zero bearings. Nothing in the stack produces an unmeasured
#    24 h arc.
#
# 2. THE ARC LENGTH WAS NEVER THE PROBLEM. THE TICK SPACING IS. Measured,
#    surrogate vs the real `bls3d` batch IOD, same env, same episodes, same
#    seed, same gate (0.20) and same 2700 s floor — the strict apples-to-apples
#    the drop-in `RealAcq3D` exists to make possible:
#
#      day-warp dt   w0 obs   6 h arc    24 h arc   48 h arc
#          60 s        45     +14.6%      +0.0%      +0.0%     <- K=0
#         180 s        15         --     +20.8%         --     <- K=480
#         720 s         4     +60.4%     +66.7%     +50.0%     <- K=120, T11's
#
#    `both-no` was 0.0% at EVERY point: the real solver DOES acquire on these
#    geometries. So this is not "acquisition is impossible at drift-and-wait
#    arcs" (which would demand an observability-aware policy) — it is the
#    surrogate being wrong, and only when the ticks are sparse. Self-red-team
#    (ii) separated those two and the answer is unambiguous.
#
#    At 60 s spacing the surrogate is EXACT at both 24 h and 48 h: 100% vs
#    100%, 0.0% disagreement. The "unmeasured beyond 6 h" caveat resolves the
#    opposite way from the fear — longer arcs at proper cadence are EASIER,
#    because the batch solver gets more observations.
#
# THIS IS A SELF-INFLICTED WOUND AND SHOULD BE RECORDED AS ONE. nav_max_ticks
# =120 was introduced in the T11 tick-cap work, justified by measuring FILTER
# divergence across the day-warp (0.003 at K=120, better than K=30's 0.109).
# That measurement was correct and irrelevant: it measured the filter, and the
# thing that broke was the SURROGATE, which no nav_* diagnostic reports. W1 then
# trained inside T11 at K=120 with weight 0.20 and scored 0.0/200. The recorded
# explanation was "incompetent root + unvalidated signal"; the signal is now
# measured, and it was 50-67% optimistic. The root may still be the binding
# constraint — this campaign is what separates them.
#
# ── THE FILTER IS NOT THE BINDING PROBLEM (task 2, answered) ────────────────
# The estimate is healthy through day-warps: position error IMPROVES across a
# warp (8298 -> 1675 m) and sits at ~730 m after 8 of them. What degrades is the
# COVARIANCE, and only at long cumulative horizons at the capped cadence:
#     K=120:  NEES med 1.2 (24 h) -> 1.7 (48 h) -> 5.3 (96 h) -> 13 (192 h),
#             81% of rows outside the 6-dof band by 192 h, both seeds.
#     K=0:    NEES med 0.93 at 24 h, 12.5% out of band, pos 596 m.
# Two reasons this does not gate the campaign. First, uncapping the ticks —
# which W1A already requires — improves it as well, so the fix is shared.
# Second, NAV-F dropped the sigma channel from the observation (obs[29-32] are
# hard zero), so NO covariance reaches the policy: an overconfident P is a
# filter-health and Kalman-gain issue (~5% divergence), not a fictional signal.
# It would become one immediately if anyone re-enabled that channel, which is
# why W1B gates it anyway.
#
# ── THE COST, STATED PLAINLY ────────────────────────────────────────────────
# nav_max_ticks=0 is what makes the signal real, and it is expensive: the T11
# tick-cap measurement put K=0 at 4.4 env-steps/s against 19.8 at K=120 on the
# mixture, ~4.5x. W1 is the worst case for that ratio because day-warps are its
# whole strategy. Budget ~4-5x the wall clock of an equivalent capped rung and
# do not be surprised by it — the alternative is training on a signal that is
# wrong two times in three, which is the thing that already cost one W1 rung.
#
# ── SCOPE ───────────────────────────────────────────────────────────────────
# Root bootstrap only: get W1 competent under nav, in its OWN cell. Re-mixing
# into the 7-cell generalist is a separate campaign and a separate argument;
# the T11 seesaw says nothing about that until this root exists.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/extj2/t11_eval.py
GATES=$MAIN/scripts/orbital/extj2/t11_gates.py
VERIFY=$MAIN/scripts/orbital/extj2/verify_extj2.py
W1GATES=$MAIN/scripts/orbital/ext_recon/w1nav_gates.py
ROOT=$MAIN/models/t3/j2wait_W1_driftwait.pt
PROG=/tmp/w1nav_progress.log
JSON_DIR=$MAIN/web_data/results/w1nav
EPS=200
EVAL_SEED=123
STEPS=${W1_STEPS:-50000000}
WATCHDOG_S=1800

# THE load-bearing flag. 0 = 60 s tick spacing = the only cadence at which the
# acquisition signal is measured to be real. Do not raise it to buy throughput
# without re-running W1A at the value you pick.
NAV_MAX_TICKS=${W1_NAV_MAX_TICKS:-0}

W1_SEED=${W1_SEED:-42}
SEED_SFX=""
[ "$W1_SEED" != "42" ] && SEED_SFX="_s${W1_SEED}"
STAGES=${W1_STAGES:-0,1,2,3}
want() { [[ ",$STAGES," == *",$1,"* ]]; }

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

EPOCH_SLACK=${W1_EPOCH_SLACK:-3}
ARM_DIR="$PUF/experiments_w1nav/w1nav${SEED_SFX}"
final_ckpt() {
    local dir="$1" want_ep="$2"
    ls -t "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null | while read -r f; do
        local n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$((want_ep - EPOCH_SLACK))" ] && { echo "$f"; return 0; }
    done
    return 0
}
ckpt_at() {
    local dir="$1" ep="$2" f n
    for f in $(ls "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null \
               | sed 's/.*_\([0-9][0-9]*\)\.pt$/\1 &/' | sort -n | cut -d' ' -f2-); do
        n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$ep" ] && { echo "$f"; return 0; }
    done
    return 0
}

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: $MAIN on '$br', need '$BRANCH_REQ'"; return 1; }
    for f in "$EVAL" "$GATES" "$VERIFY" "$W1GATES" "$ROOT"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    mkdir -p "$JSON_DIR"; return 0
}

# ── stage 0: anchors + T11 battery + THE TWO W1 GATES ──────────────────────
anchor() {
    say "START stage 0 (anchors + T11 gates + W1A/W1B at the TRAINING cadence)"
    cd "$MAIN" || return 1
    python3 "$VERIFY" --stage a1,a2,a3,a4,a5 --eps 100 > /tmp/w1nav_anchor.log 2>&1
    local ra=$?
    say "  RESULT anchors rc=$ra $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/w1nav_anchor.log | tail -1)"
    python3 "$GATES" > /tmp/w1nav_t11gates.log 2>&1
    local rg=$?
    say "  RESULT t11 gates rc=$rg $(grep -oE '[0-9]+/[0-9]+ gates pass' /tmp/w1nav_t11gates.log | tail -1)"
    # The gate that would have caught T11's W1 zero, run at the cadence this
    # campaign actually trains with — not at a convenient one.
    python3 "$W1GATES" --stage w1a,w1b --n 24 --seeds 11 4242 --hours 24 \
        --max-ticks "$NAV_MAX_TICKS" > /tmp/w1nav_w1gates.log 2>&1
    local rw=$?
    say "  RESULT W1 gates rc=$rw (nav_max_ticks=$NAV_MAX_TICKS) $(grep -oE '[0-9]+/[0-9]+ W1xnav gates pass' /tmp/w1nav_w1gates.log | tail -1)"
    grep -E '^\s+\[(PASS|FAIL)\] W1' /tmp/w1nav_w1gates.log | while read -r l; do say "    $l"; done
    if [ $ra -ne 0 ] || [ $rg -ne 0 ] || [ $rw -ne 0 ]; then
        say "ABORT: stage 0 FAILED — training on an unvalidated acquisition"
        say "       signal is exactly what produced T11's W1 zero."
        return 1
    fi
    say "---- stage 0 done ----"; return 0
}

one_eval() {
    local tag="$1" ck="$2" cell="$3" nav="$4"; shift 4
    cd "$MAIN" || return 1
    tag="${tag}${SEED_SFX}"
    local L=/tmp/w1nav_${tag}.log
    python3 "$EVAL" --ckpt "$ck" --cell "$cell" --nav-mode "$nav" \
        --episodes $EPS --seed $EVAL_SEED --label "$tag" \
        --out "$JSON_DIR/${tag}.json" "$@" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' "$L" | head -1 | sed 's/^ *//')"
    local g
    for pat in 'causes:' 'FUEL AUDIT'; do
        g=$(grep -F "$pat" "$L" | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    done
}

# ── stage 1: floors ────────────────────────────────────────────────────────
if want 0 || want 1 || want 2 || want 3; then :; fi
say "=========== W1xnav campaign start (pid $$) ==========="
say "root:      $(basename $ROOT) (truth-mode drift-and-wait specialist)"
say "cadence:   nav_max_ticks=$NAV_MAX_TICKS  <- the load-bearing flag"
say "           at K=120 the surrogate was measured 50-67% OPTIMISTIC on W1"
say "           arcs; at K=0 it is 0.0% off at both 24 h and 48 h."
say "steps:     $STEPS   seed $W1_SEED"
say "STATE:     W1 rows are MEAN-ELEMENT claims under J2 (secular only)."
preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED ----"; fi

if want 1; then
  say "START stage 1 (floors: truth root zero-shot, truth and native BO)"
  one_eval "F_root_W1_truth" "$ROOT" W1_driftwait truth
  one_eval "F_root_W1_bo"    "$ROOT" W1_driftwait bearings_only
  say "  READ: the truth row is the specialist's own capability; the BO row is"
  say "        what nav costs it cold. The gap is what stage 2 has to close,"
  say "        and it is the FIRST honest measurement of it — T11's W1xnav"
  say "        number was taken against a fictional acquisition signal."
  say "---- stage 1 done ----"
else say "---- stage 1 SKIPPED ----"; fi

# ── stage 2: the nav rung ──────────────────────────────────────────────────
if want 2; then
  WANT_EP=$(( STEPS / 131072 ))
  if [ -n "$(final_ckpt "$ARM_DIR" "$WANT_EP")" ]; then
    say "SKIP train (final ckpt present)"
  else
    say "START stage 2 (W1xnav rung: $STEPS steps from the truth specialist)"
    cd "$PUF" || exit 1
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps "$STEPS" --train.seed "$W1_SEED" \
        --train.data-dir "$ARM_DIR" --load-model-path "$ROOT" \
        --train.checkpoint-interval "${W1_CKPT_INT:-20}" \
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
        --env.episode-cap-steps 22000 \
        --env.rendezvous-radius-m 30000.0 --env.rel-vel-tol-ms 50.0 \
        --env.a-min-override 6.671e6 --env.a-max-override 7.171e6 \
        --env.e-max-target 0.05 --env.e-max-sat 0.05 \
        --env.di-max-rad 0.0872665 --env.di-min-rad 0.0349066 \
        --env.di-phase-mode 1 \
        --env.i-target-min-rad 0.5235987755982988 \
        --env.i-target-max-rad 1.0471975511965976 \
        --env.fuel-frac-min 0.113 --env.fuel-frac-max 0.20 \
        --env.nav-mode bearings_only \
        --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0 \
        --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20 \
        --env.nav-acq-mode crlb_online --env.nav-max-ticks "$NAV_MAX_TICKS" \
        --env.t11-mixture 0 --env.cell-mixture-mode 0 \
        --wandb --wandb-project orbital-rl --wandb-group "w1nav${SEED_SFX}" \
        --tag "w1nav${SEED_SFX}" > "/tmp/w1nav${SEED_SFX}_train.log" 2>&1 &
    PID=$!; say "  trainer pid $PID"
    t_final=0
    while kill -0 "$PID" 2>/dev/null; do
      sleep 60
      if [ "$t_final" -eq 0 ] && [ -n "$(final_ckpt "$ARM_DIR" "$WANT_EP")" ]; then
        t_final=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
      fi
      if [ "$t_final" -ne 0 ] && [ $(( $(date +%s) - t_final )) -ge $WATCHDOG_S ]; then
        say "  WATCHDOG: killing $PID"
        pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null; sleep 20
        pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null; break
      fi
    done
    say "---- stage 2 done ----"
  fi
else say "---- stage 2 SKIPPED ----"; fi

# ── stage 3: the W1 home rung + retention ──────────────────────────────────
if want 3; then
  say "START stage 3 (W1 home rung + retention across the generalist cells)"
  CK=${W1_CK:-$(final_ckpt "$ARM_DIR" $(( STEPS / 131072 )))}
  if [ -z "$CK" ]; then say "  SKIP: no checkpoint"; else
    say "  ckpt $(basename "$CK")"
    one_eval "s3_W1_bo"    "$CK" W1_driftwait bearings_only
    one_eval "s3_W1_truth" "$CK" W1_driftwait truth
    MID=$(ckpt_at "$ARM_DIR" $(( STEPS / 2 / 131072 )))
    [ -n "$MID" ] && one_eval "s3_mid_W1_bo" "$MID" W1_driftwait bearings_only
    # retention: this root never saw these cells, so these are a BASELINE for
    # the later re-mix argument, not a regression check.
    for C in E0_j2 E3_j2 TIGHT_5k1 LONGRANGE; do
      one_eval "s3_ret_${C}" "$CK" "$C" bearings_only
    done
    say "  READ (the point of the campaign): s3_W1_bo vs F_root_W1_bo is the"
    say "        first W1xnav number taken on a VALIDATED acquisition signal."
    say "        If it is still ~0, the root/bootstrap is the binding constraint"
    say "        and the surrogate was a confound, not the cause — which is a"
    say "        real answer and closes the over-determination in T11's writeup."
    say "  READ (retention): the truth specialist never saw E*/TIGHT/LONGRANGE,"
    say "        so these rows are a BASELINE for a future re-mix, not a"
    say "        regression. Do not read them as forgetting."
    say "---- stage 3 done ----"
  fi
else say "---- stage 3 SKIPPED ----"; fi

say "=========== w1nav campaign COMPLETE ==========="
