#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 3D-NAV x TIGHT BOX — the NAV-F regime in 3D, plus rung-1 multi-seed
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH THIS FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/nav/n3dnav_tb_campaign.sh \
#       > /tmp/n3dnav_tb_stdout.log 2>&1 &
# The rung-1 campaign's first launch died at 22:42 because it was spawned
# inside an agent process tree and was reaped when that task closed (the
# machine never slept — pmset confirmed). Nothing in this script prevents that;
# only the launching shell does.
#
# ── WHAT THIS ADDS TO RUNG 1 ────────────────────────────────────────────────
# Rung 1 (N3DNAV_RESULTS.md) showed bearings-only training closing a 42-point
# gap at the LOOSE box (30 km / 50 m/s) — the 2D nav lineage's box. It did NOT
# touch the tight boxes, and that is where the interesting physics is: N3D-B
# section 2.3 predicts the 3D angles-only dividend lands on VELOCITY (2-4x),
# and NAV-F section 2.6 established that a rendezvous box binds on sigma_vel
# rather than sigma_range. TB5-3D's 1 m/s tolerance is therefore the first box
# where the estimator's velocity error is the binding constraint rather than a
# rounding term. Every eval here reports it (`navvel_err_inbox_*`).
#
# ── THE BOXES ───────────────────────────────────────────────────────────────
# The 3D task is held FIXED and only the box moves, exactly as the TB3D ladder
# (scripts/orbital/t3/t6_tb3d.sh) moved it. Published held-out scores, seed
# 123, 200 eps, from /tmp/t6_tb3d_s42.log:
#     B1  10 km / 10 m/s   199/200
#     B2   5 km /  2 m/s   199/200      <- TB4-3D here
#     B3   5 km /  1 m/s   194/200      <- TB5-3D here, the flagship
#
# ── THE ARMS ────────────────────────────────────────────────────────────────
#   stage 1  T-truth3d-TB  eval only. The truth-trained TB3D policy flown
#                          bearings-only with the real IOD, at TB5-3D and
#                          TB4-3D. The floor.
#   stage 2  N1-rb3d-TB5   50M warm, rb_ekf, trained AT TB5-3D. The control:
#                          range is MEASURED, so a maneuver buys the estimator
#                          nothing. Evaluated native AND CROSS-MODE
#                          (bearings-only + real IOD) — rung 1's addendum
#                          showed the cross-mode row is what makes attribution
#                          airtight, so it is in the script this time rather
#                          than bolted on afterwards.
#   stage 3  T-BO3D-TB5    50M warm, bearings_only, trained AT TB5-3D. The
#                          treatment.
#   stage 4  rung-1 multi-seed: T-BO3 at the X3 LOOSE box, seeds 7 and 1337.
#                          After the flagship, so TB5 lands first.
#   stage 5  TB5 treatment multi-seed: T-BO3D-TB5 at seeds 7 and 1337,
#                          identical to stage 3 but for --train.seed.
#
# ── WARM STARTS: TWO FILES, AND THEY ARE NOT INTERCHANGEABLE ────────────────
# TB stages (2, 3): models/t3/n3dnav_warm_TB5.pt
#     = seed42_TB3D_box5k1.pt with encoder columns 29-32 zeroed.
#     Only 29-32: measured over 300 steps x 256 envs at this exact config,
#     max |obs[29:33]| = 0 (hard-zeroed by orbital.h) while
#     max |obs[21:29]| = 2.0/2.0/0.033/0.038/0.038/0.44/0.68/0.88 — the ext-3d
#     plane and delta-v-ledger block is LIVE, so those encoder columns carry
#     trained weights and zeroing them would destroy the warm start
#     (n3d_REDTEAM MAJOR-14). 29-32 got exactly zero gradient and still hold
#     random init.
#     VERIFIED: reproduces the parent's published 194/200 at TB5-3D with a
#     bit-identical action stream, md5 721def1b9d72.
#
# Stage 4:            models/t3/n3dnav_warm_X3.pt   (rung 1's file, unchanged)
#
# ── ONE CONFIG SUBTLETY THAT WOULD OTHERWISE BE A SILENT CONFOUND ───────────
# The TB3D ladder trained with orbital.py's DEFAULT shape_w_match = 0.35 at
# dv_ref = 700. The rung-1 nav campaign used 0.8166667 (the value that makes
# w_match/dv_ref match the 0.35-at-300 lineage). Those are different regimes,
# and a nav arm must inherit its OWN parent's:
#     TB stages (2,3) -> 0.35        (matches seed42_TB3D_box5k1.pt)
#     stage 4         -> 0.8166667   (matches rung-1 T-BO3, which it replicates)
# Getting this backwards would change the reward alongside the observation
# pipeline, and every T3 collapse in this project traces to a compound change.
# It does not affect success classification (that is a physics branch), only
# the shaping the policy trains against.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
WS_TB=$MAIN/models/t3/n3dnav_warm_TB5.pt
WS_X3=$MAIN/models/t3/n3dnav_warm_X3.pt
TB_PARENT=$MAIN/models/t3/seed42_TB3D_box5k1.pt
PROG=/tmp/n3dnav_tb_progress.log
JSON_DIR=$MAIN/web_data/results/n3dnav_tb
EVAL=$MAIN/scripts/orbital/nav/eval_relnav3d.py
EPS=200
EVAL_SEED=123
# Which stages to run. Default: all. Set to resume after an interruption
# (skip-logic already prevents retraining a finished arm, but this also skips
# the finished arm's EVALS, which the skip-logic does not cover):
#   N3DNAV_TB_STAGES=0,5 bash scripts/orbital/nav/n3dnav_tb_campaign.sh
# Stage 0 (the anchor) runs unless explicitly excluded, and its failure always
# aborts — no stage list can turn that off.
STAGES=${N3DNAV_TB_STAGES:-0,1,2,3,4,5}
want() { [[ ",$STAGES," == *",$1,"* ]]; }
STEPS=50000000
FINAL_CKPT_TAG=000382        # 50M steps at the shipped 8w x 256 shape
WATCHDOG_S=900               # 15 min after the final ckpt appears

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# ── the 3D task, identical in every stage; only the box and w_match move ────
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
X3_BOX=(--env.rendezvous-radius-m 30000.0 --env.rel-vel-tol-ms 50.0)

preflight() {
    local br
    br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    if [ "$br" != "$BRANCH_REQ" ]; then
        say "ABORT: $MAIN is on branch '$br', need '$BRANCH_REQ'. The trainer"
        say "       runs whatever the MAIN checkout has checked out."
        return 1
    fi
    for f in "$WS_TB" "$WS_X3" "$TB_PARENT"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    mkdir -p "$JSON_DIR"
    return 0
}

# ── stage 0: the anchor every downstream number depends on ──────────────────
anchor() {
    say "START harness anchor (TB5-3D, expect the PUBLISHED 194/200)"
    cd "$PUF" || return 1
    python3 "$EVAL" --stage anchor --episodes $EPS --seed $EVAL_SEED \
        --ckpt "$WS_TB" --box TB5-3D --shape-w-match 0.35 --expect 194/200 \
        --out /tmp/n3dnav_tb_anchor.json > /tmp/n3dnav_tb_anchor.log 2>&1
    local rc=$? line
    line=$(grep 'HARNESS ANCHOR:' /tmp/n3dnav_tb_anchor.log | tail -1)
    say "RESULT anchor rc=$rc ${line}"
    grep -E '^\s+\[(PASS|FAIL)\]' /tmp/n3dnav_tb_anchor.log \
        | while read -r l; do say "  anchor $l"; done
    echo "$line" | grep -q PASS || { say "ABORT: harness anchor FAILED"; return 1; }
    return 0
}

# ── train <arm> <nav_mode> <w_match> <box-array-name> <warm> ────────────────
train_arm() {
    local arm="$1" nav="$2" wm="$3" boxvar="$4[@]" warm="$5" seed="${6:-42}"
    local box=("${!boxvar}")
    local dir="$PUF/experiments_n3dnav_tb/${arm}"
    if ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1; then
        say "SKIP train $arm (final ckpt already present)"
        return 0
    fi
    say "START train $arm (nav=$nav seed=$seed w_match=$wm box=${box[1]}/${box[3]} warm=$(basename $warm))"
    cd "$PUF" || return 1
    # NO inner `caffeinate`: the whole script runs under one, and wrapping the
    # trainer again would make $! the wrapper's PID. caffeinate does not
    # forward signals, so the watchdog would kill the wrapper and leave the
    # trainer ORPHANED on 14 cores under the next stage.
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps $STEPS --train.seed "$seed" \
        --train.data-dir "$dir" --load-model-path "$warm" \
        --env.nav-mode "$nav" --env.shape-w-match "$wm" \
        "${BASE_ENV[@]}" "${NAV_ENV[@]}" "${box[@]}" \
        --wandb --wandb-project orbital-rl --wandb-group "t6-n3dnavtb-${arm}" \
        --tag "t6_n3dnavtb_${arm}" > "/tmp/n3dnav_tb_${arm}_train.log" 2>&1 &
    local pid=$!
    say "  trainer pid $pid, log /tmp/n3dnav_tb_${arm}_train.log"

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
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/n3dnav_tb_${arm}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a}"
    ls "$dir"/*/model_*${FINAL_CKPT_TAG}.pt >/dev/null 2>&1
}

last_ckpt() {
    ls -t "$PUF/experiments_n3dnav_tb/$1"/*/model_*${FINAL_CKPT_TAG}.pt \
        2>/dev/null | head -1
}

# ── one_eval <arm> <ckpt> <tag> <nav_mode> <acq> <box> <w_match> ────────────
one_eval() {
    local arm="$1" ck="$2" tag="$3" nav="$4" acq="$5" box="$6" wm="$7"
    cd "$PUF" || return 1
    local out="$JSON_DIR/${arm}__${tag}.json"
    python3 "$EVAL" --stage eval --ckpt "$ck" --nav-mode "$nav" --acq "$acq" \
        --box "$box" --shape-w-match "$wm" --episodes $EPS --seed $EVAL_SEED \
        --label "${arm}/${tag}" --out "$out" \
        > "/tmp/n3dnav_tb_${arm}_${tag}.log" 2>&1
    local L=/tmp/n3dnav_tb_${arm}_${tag}.log
    say "RESULT $arm $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    g=$(grep -E 'inside the' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag BOX: $g"
    g=$(grep -E 'acquisition:' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag ACQ: $g"
    g=$(grep -E 'EPOCH error' $L | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $arm $tag EPOCH: $g"
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== 3D-nav x tight-box campaign start (pid $$) ==========="
say "task (all stages): dim3=1 di_max=1.0deg e<=0.05 LEO gap=pi D30 shaping2"
say "                   dv_ref=700 gamma=1.0 phase modes 1 cap=3000 no debris"
say "boxes: TB5-3D 5km/1m/s (published 194/200), TB4-3D 5km/2m/s (199/200),"
say "       X3 30km/50m/s (rung 1, published 200/200)"
say "warm TB stages: $(basename $WS_TB) <- seed42_TB3D_box5k1.pt cols 29-32 zeroed,"
say "                verified bit-identical action stream md5 721def1b9d72"
say "warm stage 4:   $(basename $WS_X3) (rung-1 file, unchanged)"
say "shape_w_match: 0.35 for TB stages (matches the TB3D ladder), 0.8166667 for"
say "               stage 4 (matches rung-1 T-BO3, which it replicates)"
say "eval: $EPS eps @ held-out seed $EVAL_SEED, native + truth (+ cross-mode on the control)"
say "stages requested: $STAGES"
preflight || { say "=========== ABORTED (preflight) ==========="; exit 1; }
anchor    || { say "=========== ABORTED (harness anchor) ==========="; exit 1; }
say "---- stage 0 done (anchor) ----"

# ── stage 1: the floor, eval only ───────────────────────────────────────────
if want 1; then
for BOX in TB5-3D TB4-3D; do
    one_eval "T-truth3d-TB" "$WS_TB" "${BOX}_bo"    bearings_only real "$BOX" 0.35
    one_eval "T-truth3d-TB" "$WS_TB" "${BOX}_truth" truth         surrogate "$BOX" 0.35
done
say "---- stage 1 done (T-truth3d-TB, the floor) ----"
else say "---- stage 1 SKIPPED (not in STAGES=$STAGES) ----"
fi

# ── stage 2: the control, trained at TB5-3D ─────────────────────────────────
if ! want 2; then say "---- stage 2 SKIPPED (not in STAGES=$STAGES) ----"
elif train_arm "N1-rb3d-TB5" rb_ekf 0.35 TB5_BOX "$WS_TB"; then
    CK=$(last_ckpt "N1-rb3d-TB5"); cp "$CK" "$MAIN/models/t3/n3dnav_N1-rb3d-TB5.pt" 2>/dev/null
    for BOX in TB5-3D TB4-3D; do
        one_eval "N1-rb3d-TB5" "$CK" "${BOX}_rb"    rb_ekf        surrogate "$BOX" 0.35
        one_eval "N1-rb3d-TB5" "$CK" "${BOX}_truth" truth         surrogate "$BOX" 0.35
        # the cross-mode row: this checkpoint flown BLIND. Rung 1 showed it is
        # what separates "estimates are fine" from "bearings-only is fine".
        one_eval "N1-rb3d-TB5" "$CK" "${BOX}_xbo"   bearings_only real "$BOX" 0.35
    done
    say "---- stage 2 done (N1-rb3d-TB5, the control) ----"
else
    say "FAIL train N1-rb3d-TB5 (no final ckpt)"
fi

# ── stage 3: the treatment, trained at TB5-3D ───────────────────────────────
if ! want 3; then say "---- stage 3 SKIPPED (not in STAGES=$STAGES) ----"
elif train_arm "T-BO3D-TB5" bearings_only 0.35 TB5_BOX "$WS_TB"; then
    CK=$(last_ckpt "T-BO3D-TB5"); cp "$CK" "$MAIN/models/t3/n3dnav_T-BO3D-TB5.pt" 2>/dev/null
    for BOX in TB5-3D TB4-3D; do
        one_eval "T-BO3D-TB5" "$CK" "${BOX}_bo"    bearings_only real "$BOX" 0.35
        one_eval "T-BO3D-TB5" "$CK" "${BOX}_truth" truth         surrogate "$BOX" 0.35
    done
    say "---- stage 3 done (T-BO3D-TB5, the treatment) ----"
else
    say "FAIL train T-BO3D-TB5 (no final ckpt)"
fi

# ── stage 4: rung-1 multi-seed. LAST, so the flagship lands first. ──────────
# Replicates rung-1 T-BO3 exactly — X3 loose box, n3dnav_warm_X3.pt,
# w_match 0.8166667 — changing only --train.seed. Seed 42 is already published
# at 200/200 (N3DNAV_RESULTS.md).
want 4 || say "---- stage 4 SKIPPED (not in STAGES=$STAGES) ----"
for SEED in 7 1337; do
    want 4 || break
    ARM="T-BO3-X3-s${SEED}"
    if train_arm "$ARM" bearings_only 0.8166667 X3_BOX "$WS_X3" "$SEED"; then
        CK=$(last_ckpt "$ARM"); cp "$CK" "$MAIN/models/t3/n3dnav_${ARM}.pt" 2>/dev/null
        one_eval "$ARM" "$CK" "X3_bo"    bearings_only real "X3" 0.8166667
        one_eval "$ARM" "$CK" "X3_truth" truth         surrogate "X3" 0.8166667
    else
        say "FAIL train $ARM (no final ckpt)"
    fi
    say "---- stage 4 seed $SEED done ----"
done

# ── stage 5: TB5 treatment multi-seed ───────────────────────────────────────
# Replicates stage 3 exactly — same warm start n3dnav_warm_TB5.pt, same
# w_match 0.35, bearings_only, 50M, TB5-3D box — changing only --train.seed.
# Seed 42 is stage 3 itself (98.0% @TB5-3D / 99.5% @TB4-3D).
#
# No cross-mode row here: flying a checkpoint in a mode it was not trained for
# is a CONTROL-arm diagnostic. On the control it answers "does estimate-training
# with range measured transfer to the blind problem" (it does not — 13-18% at
# TB5). On the treatment it would just re-run the native row under a different
# name, since bearings-only IS its native mode.
want 5 || say "---- stage 5 SKIPPED (not in STAGES=$STAGES) ----"
for SEED in 7 1337; do
    want 5 || break
    ARM="T-BO3D-TB5-s${SEED}"
    if train_arm "$ARM" bearings_only 0.35 TB5_BOX "$WS_TB" "$SEED"; then
        CK=$(last_ckpt "$ARM"); cp "$CK" "$MAIN/models/t3/n3dnav_${ARM}.pt" 2>/dev/null
        for BOX in TB5-3D TB4-3D; do
            one_eval "$ARM" "$CK" "${BOX}_bo"    bearings_only real "$BOX" 0.35
            one_eval "$ARM" "$CK" "${BOX}_truth" truth         surrogate "$BOX" 0.35
        done
    else
        say "FAIL train $ARM (no final ckpt)"
    fi
    say "---- stage 5 seed $SEED done ----"
done

say "=========== 3D-nav x tight-box campaign COMPLETE ==========="
