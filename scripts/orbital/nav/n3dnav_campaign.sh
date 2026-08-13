#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 3D-NAV CAMPAIGN, RUNG 1 (X3) — one unattended script, end to end
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs under nohup + caffeinate so it survives turn boundaries and device
# sleep. Every stage appends a RESULT line to $PROG; per-arm JSON lands in
# web_data/results/n3dnav/. Nothing here commits to git — the operator commits
# after reading $PROG, so a half-finished stage can never land a partial row.
#
# ── THE RUNG ────────────────────────────────────────────────────────────────
# X3, exactly as scripts/orbital/ext3d/t5_x3_seeds.sh trained and scored it:
#
#   dim3_mode 1, di_max_rad 0.017453 (1.0 deg relative inclination)
#   e_max_target 0.05, e_max_sat 0.05, LEO, init_phase_gap_max pi
#   success box: rendezvous_radius_m 30000, rel_vel_tol_ms 50   <- the DEFAULT
#   shaping_mode 2, w_lambda 1.0, w_match 0.8166667, dv_ref 700, gamma 1.0
#   phase_gap_mode 1, phase_obs_mode 1, episode_cap_steps 3000
#   legacy_action_space 30, no debris
#
# n3d_C section 5 puts every first-generation nav rung — N0-truth, N0-recon,
# N1-rb3d, N2-bo3d — at X3, and the 30 km / 50 m/s box is the reason: it is the
# SAME box as the 2D nav lineage, so NB1's 98.0-99.5% is a like-for-like
# comparator rather than a different experiment. Tightening the box and adding
# the plane in the same campaign would confound the two.
#
# ── THE ARMS ────────────────────────────────────────────────────────────────
#   T-truth3d   eval only. The truth-trained X3 policy flown on a
#               bearings-only-3D estimate, zero-shot. THE FLOOR: what the
#               guidance policy is worth when the target state stops being
#               given to it. No training.
#   N1-rb3d     50M warm-start, range + two angles. THE CONTROL. Range is
#               MEASURED, so a maneuver buys the estimator nothing. If the
#               treatment and this move together the effect is not
#               observability — it is guidance, the box, or the warm start.
#   T-BO3       50M warm-start, angles only. THE TREATMENT.
#
# ── THE WARM START ──────────────────────────────────────────────────────────
# models/t3/n3dnav_warm_X3.pt, shared by BOTH trained arms so they are
# behaviourally identical at t=0 (NAV-F discipline). Derivation:
#
#   src = models/t3/seed42_X3_3d_di1deg.pt   (200/200 held-out at X3, seed 123)
#   for col in 29 30 31 32: zero_obs_column.py --col $col
#
# Only 29-32. n3d_REDTEAM MAJOR-14: obs[21-28] are the ext-3d plane/ledger
# block and were LIVE during that checkpoint's training, so those encoder
# columns are trained, not random — zeroing them would destroy the warm start.
# obs[29-32] are hard-zeroed by orbital.h and therefore received exactly zero
# gradient, so they still hold random init. Zeroing them is provably a no-op
# today (the arms never light those slots) and it is what keeps the comparison
# clean if a later arm turns on the Sigma channel, which under dim3 lives on
# obs[29] rather than obs[21].
#
# Verified, not assumed: the derived file reproduces the source checkpoint's
# 200/200 with a bit-identical action stream (md5 62f66b07eadf).
#
# ── WATCHDOG ────────────────────────────────────────────────────────────────
# Known failure mode, observed in this project: a trainer can hang in teardown
# AFTER saving its final checkpoint (NAV-F's T_BOS spun 2.25 h post-save and
# exited rc=143 only when killed). So: once the final checkpoint exists, give
# the trainer $WATCHDOG_S to exit on its own, then SIGTERM/SIGKILL it and carry
# on. Safe because the skip-logic keys on the same checkpoint — a re-run of
# this script will not retrain a stage whose final checkpoint is on disk.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=ext-3dnav
WS=$MAIN/models/t3/n3dnav_warm_X3.pt
PROG=/tmp/n3dnav_campaign_progress.log
JSON_DIR=$MAIN/web_data/results/n3dnav
EPS=200
EVAL_SEED=123
STEPS=50000000
FINAL_CKPT_TAG=000382        # 50M steps at the shipped 8w x 256 shape
WATCHDOG_S=900               # 15 min after the final ckpt appears

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

X3_ENV=(
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
  --env.dim3-mode 1 --env.di-max-rad 0.017453
  --env.legacy-action-space 30
)
NAV_ENV=(
  --env.nav-sensor-dt 60.0 --env.nav-noise-mult 1.0
  --env.nav-acq-min-sec 2700.0 --env.nav-acq-gate 0.20
  --env.nav-acq-mode crlb_online --env.nav-max-ticks 0
)

# ── preflight ───────────────────────────────────────────────────────────────
preflight() {
    local br
    br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    if [ "$br" != "$BRANCH_REQ" ]; then
        say "ABORT: $MAIN is on branch '$br', need '$BRANCH_REQ'."
        say "       The trainer runs whatever the MAIN checkout has checked out;"
        say "       training the wrong branch produces a result for code that is"
        say "       not the code under test."
        return 1
    fi
    [ -f "$WS" ] || { say "ABORT: warm start $WS missing"; return 1; }
    mkdir -p "$JSON_DIR"
    return 0
}

# ── harness anchor: the gate every eval number depends on ───────────────────
anchor() {
    say "START harness anchor"
    cd "$PUF" || return 1
    python3 "$MAIN/scripts/orbital/nav/eval_relnav3d.py" --stage anchor \
        --episodes $EPS --seed $EVAL_SEED --ckpt "$WS" \
        --out /tmp/n3dnav_anchor.json > /tmp/n3dnav_anchor.log 2>&1
    local rc=$?
    local line
    line=$(grep 'HARNESS ANCHOR:' /tmp/n3dnav_anchor.log | tail -1)
    say "RESULT anchor rc=$rc ${line}"
    grep -E '^\s+\[(PASS|FAIL)\]' /tmp/n3dnav_anchor.log \
        | while read -r l; do say "  anchor $l"; done
    echo "$line" | grep -q PASS || { say "ABORT: harness anchor FAILED"; return 1; }
    return 0
}

# ── train <arm> <nav_mode> ──────────────────────────────────────────────────
train_arm() {
    local arm="$1" nav="$2"
    local dir="$PUF/experiments_n3dnav/${arm}"
    if ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
        say "SKIP train $arm (final ckpt already present)"
        return 0
    fi
    say "START train $arm (nav_mode=$nav, ${STEPS} steps, warm=$(basename $WS))"
    cd "$PUF" || return 1
    # NO inner `caffeinate` wrapper. The whole script already runs under
    # `caffeinate -is`, so the machine stays awake for the entire campaign —
    # and wrapping the trainer again would make $! the PID of CAFFEINATE, not
    # of python3. caffeinate does not forward signals to its child, so the
    # watchdog would kill the wrapper and leave the hung trainer ORPHANED,
    # burning 14 cores underneath the next stage. That is worse than the hang
    # it was added to fix.
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps $STEPS --train.seed 42 \
        --train.data-dir "$dir" \
        --load-model-path "$WS" \
        --env.nav-mode "$nav" "${X3_ENV[@]}" "${NAV_ENV[@]}" \
        --wandb --wandb-project orbital-rl --wandb-group "t6-n3dnav-${arm}" \
        --tag "t6_n3dnav_${arm}" > "/tmp/n3dnav_${arm}_train.log" 2>&1 &
    local pid=$!
    say "  trainer pid $pid, log /tmp/n3dnav_${arm}_train.log"

    # ── watchdog ────────────────────────────────────────────────────────────
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
            say "  WATCHDOG: final ckpt saved but trainer still alive after ${WATCHDOG_S}s — killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null      # the vector workers first
            kill -TERM "$pid" 2>/dev/null
            sleep 20
            pkill -KILL -P "$pid" 2>/dev/null
            kill -KILL "$pid" 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null
    local rc=$?
    local perf
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/n3dnav_${arm}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a}"
    ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1
}

last_ckpt() {
    local dir="$PUF/experiments_n3dnav/$1"
    ls -t "$dir"/*/model_*${FINAL_CKPT_TAG}.pt 2>/dev/null | head -1 \
        || ls -t "$dir"/*/model_*.pt 2>/dev/null | head -1
}

# ── evaluate <arm> <ckpt> <native nav_mode> <acq> ───────────────────────────
evaluate() {
    local arm="$1" ck="$2" nav="$3" acq="$4"
    [ -n "$ck" ] && [ -f "$ck" ] || { say "FAIL eval $arm: no checkpoint"; return 1; }
    cd "$PUF" || return 1
    say "START eval $arm ckpt=$ck (native=$nav acq=$acq)"
    # native
    python3 "$MAIN/scripts/orbital/nav/eval_relnav3d.py" --stage eval \
        --ckpt "$ck" --nav-mode "$nav" --acq "$acq" --episodes $EPS \
        --seed $EVAL_SEED --label "${arm}/native" \
        --out "$JSON_DIR/${arm}_native.json" \
        > "/tmp/n3dnav_${arm}_eval_native.log" 2>&1
    say "RESULT $arm NATIVE: $(grep -E 'success [0-9]+/' /tmp/n3dnav_${arm}_eval_native.log | head -1 | sed 's/^ *//')"
    say "  $arm ACQ: $(grep -E 'acquisition:' /tmp/n3dnav_${arm}_eval_native.log | head -1 | sed 's/^ *//')"
    say "  $arm EPOCH: $(grep -E 'EPOCH error' /tmp/n3dnav_${arm}_eval_native.log | head -1 | sed 's/^ *//')"
    # truth-mode on the SAME checkpoint: the guidance ceiling for this policy,
    # so the nav tax is a within-checkpoint difference, not a cross-arm one.
    python3 "$MAIN/scripts/orbital/nav/eval_relnav3d.py" --stage eval \
        --ckpt "$ck" --nav-mode truth --episodes $EPS \
        --seed $EVAL_SEED --label "${arm}/truth" \
        --out "$JSON_DIR/${arm}_truth.json" \
        > "/tmp/n3dnav_${arm}_eval_truth.log" 2>&1
    say "RESULT $arm TRUTH:  $(grep -E 'success [0-9]+/' /tmp/n3dnav_${arm}_eval_truth.log | head -1 | sed 's/^ *//')"
    cp "$ck" "$MAIN/models/t3/n3dnav_${arm}.pt" 2>/dev/null
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== 3D-nav campaign start (pid $$) ==========="
say "rung X3: dim3=1 di_max=1.0deg e_t<=0.05 box 30km/50m/s cap 3000 D30 shaping2/dvref700"
say "warm start $WS  (seed42_X3_3d_di1deg.pt with encoder cols 29-32 zeroed)"
say "branch required $BRANCH_REQ; eval $EPS eps @ held-out seed $EVAL_SEED"
preflight || { say "=========== ABORTED (preflight) ==========="; exit 1; }
anchor    || { say "=========== ABORTED (harness anchor) ==========="; exit 1; }
say "---- stage 0 done (anchor) ----"

# ── stage 1: T-truth3d, the floor. Eval only, no training. ──────────────────
evaluate "T-truth3d" "$WS" bearings_only real
say "---- stage 1 done (T-truth3d, the floor) ----"

# ── stage 2: N1-rb3d, the control ───────────────────────────────────────────
if train_arm "N1-rb3d" rb_ekf; then
    evaluate "N1-rb3d" "$(last_ckpt N1-rb3d)" rb_ekf surrogate
else
    say "FAIL train N1-rb3d (no final ckpt)"
fi
say "---- stage 2 done (N1-rb3d, the control) ----"

# ── stage 3: T-BO3, the treatment ───────────────────────────────────────────
if train_arm "T-BO3" bearings_only; then
    evaluate "T-BO3" "$(last_ckpt T-BO3)" bearings_only real
else
    say "FAIL train T-BO3 (no final ckpt)"
fi
say "---- stage 3 done (T-BO3, the treatment) ----"

say "=========== 3D-nav campaign COMPLETE ==========="
