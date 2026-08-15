#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ext-j2wait — DRIFT AND WAIT: trading time for fuel with J2
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/extj2/j2wait_campaign.sh \
#       > /tmp/j2wait_stdout.log 2>&1 &
# The n3dnav rung-1 campaign's first launch died because it was spawned inside
# an agent process tree and was reaped when that task closed. Nothing in this
# script prevents that; only the launching shell does.
#
# ── THE EXPERIMENT ──────────────────────────────────────────────────────────
# When the plane gap exceeds what the Delta-v budget can buy directly, does the
# policy dip its semi-major axis, let differential nodal precession rotate the
# relative node, and come back — instead of burning?
#
# Direct plane change costs 2*v_c*sin(di/2) = 134 m/s per degree at LEO, so the
# 478 m/s budget buys 3.57 deg and no more. The scenario band is U(2, 5) deg,
# which STRADDLES that break-even ON PURPOSE: below 3.57 deg direct burning is
# affordable, above it the maneuver is impossible without drift. The experiment
# is therefore not "can it drift" but "does it switch strategy where the
# physics says it must".
#
# ── EVERY PLANE NUMBER HERE IS A *NODE-DOMINANT* PLANE ERROR ────────────────
# Say it that way in every claim. Differential nodal precession rotates Omega
# and NOTHING else, so only the node component of a relative-plane error is
# drift-correctable; a pure inclination difference is untouchable no matter how
# long you wait, and cos(di_rel) = cos i_s cos i_t + sin i_s sin i_t cos(dOm)
# is minimised at dOm = 0 where it equals |i_s - i_t| exactly — a hard floor
# (j2_plane_change E). Under the shipped uniform-phase sampler the correctable
# fraction averages E|cos phi| = 2/pi = 63.7% and is sometimes ~0, which would
# mix "the policy failed to drift" with "there was nothing to drift for". So
# `di_phase_mode=1` puts the rotation axis within 30 deg of the node axis:
# measured node fraction p05 0.864, p50 0.968. Claims read "node-dominant plane
# gap", never "plane error".
#
# ── ALL CLAIMS ARE IN MEAN ELEMENTS ─────────────────────────────────────────
# Under j2_mode=1 the env's state IS the mean element set; no mean/osculating
# conversion happens anywhere. At the 30 km / 50 m/s box the mean-element
# relative-state error is 0.19% of the velocity tolerance per orbit — benign,
# but it is a mean-element claim and nothing more.
#
# ── THE MEASURED FLOOR (what this must beat) ────────────────────────────────
# The J2-trained A2 policy, head-expanded to 31, zero-shot at U(2,5) node-
# dominant gaps with cap 22000 and the day-warp available, 60 episodes:
#     0/60 success — 59 safety_cap, 1 stranded. IT CAPS OUT, IT DOES NOT
#     BURN DRY: median Dv spent is 50.0 m/s of a 478 m/s budget.
#     Median episode runs the FULL 366.7 h cap over 544 decisions,
#     0 of them day-warps (the zero-init row is never argmax).
#     Decomposition: impulses buy +0.154 deg of plane, while precession
#     LOSES 1.533 deg — the policy dips 62.3 km for 117.3 h for PHASING
#     reasons and the resulting differential precession opens the plane
#     because nothing is steering the node.
# That last line is the experiment in miniature: the mechanism is already
# running, unsteered. The question is whether training aims it.
#
# ── SELF-RED-TEAM, addressed before the script was written ──────────────────
#
# (a) STALL DOMINANCE. cap_terminal_reward=0 and shape_gamma=1 mean a capped
#     episode banks only the shaping it accumulated, and Phi's whole range is
#     W_lambda + W_match = 1.8167. Measured over a FULL 22000-step cap driven
#     entirely by day-warps at this scenario distribution (64 envs), the
#     do-nothing gain is +0.665 — 95x the 6000-cap value of 0.006, and that
#     growth is NOT a leak: over 15.3 days precession really does close plane
#     error for free, which is the phenomenon under test. It is unfarmable
#     because shape_gamma=1 telescopes (an episode's shaping total is
#     Phi_T - Phi_0 regardless of path, so looping buys nothing).
#     Against it, a success pays 10*gamma^n. The threshold: stall wins only if
#     10*gamma^n < 1.8167, i.e. n > 340 decisions. With day-warps this design
#     runs ~15-80 decisions (gamma^n = 0.93-0.67), a 4-20x margin. WITHOUT the
#     day-warp the floor policy already runs 544 decisions/episode
#     (gamma^544 = 0.065, terminal worth 0.65) — BELOW the 1.8167 stall value.
#     So the day-warp is not a convenience: at a 22000-step cap it is what
#     keeps success worth more than stalling. The flatline tripwire arms
#     regardless.
#
# (b) DOES Phi REWARD PRECESSION-BOUGHT CLOSURE THE SAME AS IMPULSE-BOUGHT?
#     Yes, structurally: Phi(mode 2)'s plane term is
#     Dv_pl = v_t * ||h_hat_s - h_hat_t||, a pure function of STATE. It cannot
#     see what caused the closure, so a degree bought by waiting scores exactly
#     as a degree bought by burning — while the Dv the burn consumed is
#     separately visible through the fuel-dependent obs. That asymmetry is the
#     gradient the experiment relies on. Gate A5d re-runs the J-A5-style
#     do-nothing bound at the new 22000-step horizon (above).
#
# (c) CREDIT HORIZON at gamma = 0.995 per DECISION (not per sub-step):
#         full 22000-substep cap via day-warp (row 30): 15.3 decisions, 0.9263
#         via 6 h warp  (row 17):                       61.1 decisions, 0.7361
#         via 1 h warp  (row 11):                      366.7 decisions, 0.1591
#     A realistic drift-and-wait episode is ~12 day-warps + ~20 dip burns +
#     ~20 endgame decisions ~= 50, gamma^50 = 0.78. Healthy. The same episode
#     flown on hour-warps is ~370 decisions and the terminal is worth 1.59 —
#     under the 1.8167 stall value, i.e. the credit horizon ALONE would kill
#     it. No gamma bump is proposed: the day-warp fixes the horizon at the
#     source, and raising gamma_train for one rung would make this arm's
#     returns incomparable with every other arm in the lineage.
#
# ── TRAINING BUDGET: 100M, NOT 50M, AND HERE IS THE ARITHMETIC ──────────────
# PPO's budget is counted in DECISIONS. The floor policy spends 544 decisions
# per episode, so 50M steps = 92k EPISODES — against the ~2.8M episodes the
# X3/A2 arms saw at ~18 decisions/episode. That is 30x fewer task instances,
# and this task has MORE to discover, not less. If training converges to
# day-warp usage the count improves to ~50 decisions/episode (1M episodes at
# 50M), but the early, warm-start-shaped phase is exactly where the episode
# starvation bites. 100M doubles both ends. Wall cost is modest because wall
# time tracks SUB-steps, not decisions: at the floor's 40.4 substeps/decision,
# 100M decisions is 4.0e9 substeps against the A2 arm's 3.6e9 for its 50M.
# Override with J2WAIT_STEPS if the launch window is tight.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/extj2/j2wait_eval.py
VERIFY=$MAIN/scripts/orbital/extj2/verify_extj2.py
TRIP=$MAIN/scripts/orbital/extj2/j2_flatline_check.py
EXPAND=$MAIN/scripts/orbital/extj2/expand_ckpt_30_to_31.py
PARENT=$MAIN/models/t3/extj2_A2_j2trained.pt
WARM=$MAIN/models/t3/j2wait_warm_A2.pt
PROG=/tmp/j2wait_progress.log
JSON_DIR=$MAIN/web_data/results/j2wait
EPS=200
EVAL_SEED=123
STEPS=${J2WAIT_STEPS:-100000000}
CAP=22000
DI_MIN=2.0
DI_MAX=5.0
WATCHDOG_S=900

# The trainer emits one checkpoint per epoch and an epoch is 131072 steps at
# the shipped 8w x 256 shape (the nav lineage's 50M runs ended at epoch 382 =
# 50e6/131072 rounded up). Rather than hardcode a tag that silently disarms the
# watchdog and the skip-logic if the shape ever moves, DERIVE the expected
# epoch and match any checkpoint at or past it minus a small slack.
EXPECT_EPOCH=$(( STEPS / 131072 ))
EPOCH_SLACK=${J2WAIT_EPOCH_SLACK:-3}
# any model_*NNNNNN.pt whose epoch >= EXPECT_EPOCH - EPOCH_SLACK
final_ckpt() {
    local dir="$1" want=$(( EXPECT_EPOCH - EPOCH_SLACK ))
    ls -t "$dir"/*/model_puffer_orbital_*.pt 2>/dev/null | while read -r f; do
        local n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$want" ] && { echo "$f"; return 0; }
    done
    return 0
}

J2_SEED=${J2_SEED:-42}
SEED_SFX=""
[ "$J2_SEED" != "42" ] && SEED_SFX="_s${J2_SEED}"

STAGES=${J2WAIT_STAGES:-0,1,2}
want() { [[ ",$STAGES," == *",$1,"* ]]; }

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# The task. Truth state — NAV STAYS OUT, one new variable per campaign.
BASE_ENV=(
  --env.num-debris-min 0 --env.num-debris-max 0
  --env.e-max-target 0.05 --env.e-max-sat 0.05 --env.same-orbit-init 0
  --env.init-phase-gap-max 3.14159 --env.valid-init-only 1
  --env.gave-up-action terminate --env.max-valid-init-attempts 4096
  --env.obs-alt-scale-m 1.6e6 --env.lvlh-scale-m 6.371e6
  --env.shaping-mode 2 --env.shape-w-lambda 1.0 --env.shape-w-match 0.8166667
  --env.shape-dv-ref-ms 700.0 --env.shape-gamma 1.0
  --env.phase-gap-mode 1 --env.phase-obs-mode 1
  --env.episode-cap-steps $CAP --env.cap-terminal-reward 0.0
  --env.dim3-mode 1 --env.j2-mode 1 --env.lvlh-frame-mode 1
  --env.i-target-min-rad 0.5235987755982988
  --env.i-target-max-rad 1.0471975511965976
  --env.raan-target-sample 0
  --env.di-max-rad 0.08726646259971647     # 5.0 deg
  --env.di-min-rad 0.03490658503988659     # 2.0 deg
  --env.di-phase-mode 1
  --env.rendezvous-radius-m 30000.0 --env.rel-vel-tol-ms 50.0
  --env.legacy-action-space 31
)

preflight() {
    local br
    br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    if [ "$br" != "$BRANCH_REQ" ]; then
        say "ABORT: $MAIN is on branch '$br', need '$BRANCH_REQ'. The trainer"
        say "       runs whatever the MAIN checkout has checked out."
        return 1
    fi
    for f in "$EVAL" "$VERIFY" "$TRIP" "$EXPAND" "$PARENT"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    if [ ! -f "$WARM" ]; then
        say "warm start absent; building it from $(basename $PARENT)"
        python3 "$EXPAND" "$PARENT" "$WARM" >> "$PROG" 2>&1 \
            || { say "ABORT: head expansion failed"; return 1; }
    fi
    mkdir -p "$JSON_DIR"
    return 0
}

anchor() {
    say "START stage 0 anchors (a1-a5)"
    cd "$MAIN" || return 1
    python3 "$VERIFY" --stage a1,a2,a3,a4,a5 --eps 100 > /tmp/j2wait_anchor.log 2>&1
    local rc=$?
    say "  RESULT anchors rc=$rc $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/j2wait_anchor.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/j2wait_anchor.log | while read -r l; do say "    $l"; done
    [ $rc -ne 0 ] && { say "ABORT: stage 0 anchors FAILED"; return 1; }
    say "---- stage 0 done ----"
    return 0
}

# one_eval <tag> <ckpt> <di_min> <di_max> <extra flags...>
one_eval() {
    local tag="$1" ck="$2" dmin="$3" dmax="$4"; shift 4
    cd "$MAIN" || return 1
    case "$tag" in
        F*) [ -s "$JSON_DIR/${tag}.json" ] && { say "SKIP eval $tag (floor, already present)"; return 0; } ;;
        *)  tag="${tag}${SEED_SFX}" ;;
    esac
    local L=/tmp/j2wait_${tag}.log
    python3 "$EVAL" --ckpt "$ck" --label "$tag" --episodes $EPS --seed $EVAL_SEED \
        --cap $CAP --di-min-deg "$dmin" --di-max-deg "$dmax" --di-phase-mode 1 \
        --out "$JSON_DIR/${tag}.json" "$@" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' $L | head -1 | sed 's/^ *//')"
    local g
    for pat in 'causes:' 'initial NODE-DOMINANT' 'DECOMPOSITION' 'Dv used' 'dip ' 'ALL EPISODES'; do
        g=$(grep -F "$pat" "$L" | head -1 | sed 's/^ *//')
        [ -n "$g" ] && say "  $tag $g"
    done
    say "  $tag CLAIM: node-dominant plane gap, MEAN ELEMENTS."
}

train_arm() {
    local arm="$1" warm="$2"
    local dir="$PUF/experiments_j2wait/${arm}${SEED_SFX}"
    if [ -n "$(final_ckpt "$dir")" ]; then
        say "SKIP train $arm (final ckpt already present for seed $J2_SEED)"
        return 0
    fi
    say "START train $arm seed=$J2_SEED steps=$STEPS cap=$CAP warm=$(basename $warm)"
    cd "$PUF" || return 1
    # NO inner caffeinate: the whole script runs under one, and wrapping the
    # trainer again would make $! the wrapper's PID. caffeinate does not forward
    # signals, so the watchdog would kill the wrapper and orphan the trainer.
    python3 -m pufferlib.pufferl train puffer_orbital \
        --train.device cpu --train.total-timesteps $STEPS --train.seed "$J2_SEED" \
        --train.data-dir "$dir" --load-model-path "$warm" \
        "${BASE_ENV[@]}" \
        --wandb --wandb-project orbital-rl --wandb-group "t7-j2wait-${arm}${SEED_SFX}" \
        --tag "t7_j2wait_${arm}${SEED_SFX}" \
        > "/tmp/j2wait_${arm}${SEED_SFX}_train.log" 2>&1 &
    local pid=$!
    say "  trainer pid $pid, log /tmp/j2wait_${arm}${SEED_SFX}_train.log"
    local t_final=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [ "$t_final" -eq 0 ] && [ -n "$(final_ckpt "$dir")" ]; then
            t_final=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
        fi
        if [ "$t_final" -ne 0 ] && [ $(( $(date +%s) - t_final )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: final ckpt saved but trainer alive after ${WATCHDOG_S}s — killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 20
            pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; break
        fi
    done
    wait "$pid" 2>/dev/null
    local rc=$? perf
    perf=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/j2wait_${arm}${SEED_SFX}_train.log" \
           | grep -oE 'perf +[0-9.]+' | tail -1)
    say "END train $arm rc=$rc last=${perf:-n/a} (expected final epoch ~$EXPECT_EPOCH)"
    [ -n "$(final_ckpt "$dir")" ]
}

last_ckpt() { final_ckpt "$PUF/experiments_j2wait/$1${SEED_SFX}"; }

flatline() {
    local arm="$1" fj="$2" floor out
    floor=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['rate'])" "$fj" 2>/dev/null)
    if [ -z "${floor:-}" ]; then say "RESULT FLATLINE-CHECK $arm: SKIPPED (no floor json)"; return 0; fi
    out=$(python3 "$TRIP" --log "/tmp/j2wait_${arm}${SEED_SFX}_train.log" --floor "$floor" \
          --after-steps 40e6 --margin 0.05 --label "${arm}${SEED_SFX}" 2>&1)
    say "RESULT $out"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== ext-j2wait campaign start (pid $$) ==========="
say "ALL CLAIMS: MEAN ELEMENTS, and NODE-DOMINANT plane gaps (not plane errors)"
say "task: J2 on, i_t ~ U(30,60) deg, node-dominant di_rel ~ U($DI_MIN,$DI_MAX) deg,"
say "      cap $CAP substeps = 366 h = 15.3 d, Discrete-31 (row 30 = day-warp),"
say "      box 30 km / 50 m/s, truth state (nav is OUT), steps $STEPS"
say "direct-burn break-even: 478 m/s budget buys 3.57 deg, so the band STRADDLES"
say "      feasibility on purpose — the experiment is whether the policy switches"
say "stages requested: $STAGES     seed: $J2_SEED${SEED_SFX:+ (suffixed $SEED_SFX)}"
say "expected final epoch ~$EXPECT_EPOCH (STEPS/131072), slack $EPOCH_SLACK"

preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED ----"; fi

# ── stage 1: the floors. Eval only. ────────────────────────────────────────
if want 1; then
  say "START stage 1 (floors, eval only)"
  # F1 the floor this arm must beat: warm start, day-warp available
  one_eval "F1_floor_warm"      "$WARM"   $DI_MIN $DI_MAX
  # F2 the same policy with the day-warp NOT exposed — isolates the row from
  #    the training. Expect identical (row 30 is zero-init, never argmax).
  one_eval "F2_floor_nodaywarp" "$PARENT" $DI_MIN $DI_MAX --no-daywarp
  # F3 the lineage's HOME rung, to prove the warm start is healthy and that
  #    F1's 0% is the gap's doing and not a broken checkpoint.
  one_eval "F3_home_narrow"     "$WARM"   -1 1.0
  say "---- stage 1 done ----"
else say "---- stage 1 SKIPPED ----"; fi

# ── stage 2: the treatment ─────────────────────────────────────────────────
if want 2; then
  say "START stage 2 (drift-and-wait arm)"
  if train_arm "W1_driftwait" "$WARM"; then
      flatline "W1_driftwait" "$JSON_DIR/F1_floor_warm.json"
      ck=$(last_ckpt W1_driftwait); say "  ckpt $ck"
      # native: the condition it trained under, with the full decomposition
      one_eval "s2_W1_native"     "$ck" $DI_MIN $DI_MAX
      # ABLATION: the trained policy with the day-warp removed. If performance
      # collapses, the day-warp is load-bearing; if it does not, the policy
      # found the maneuver some other way and the row was a convenience.
      one_eval "s2_W1_nodaywarp"  "$ck" $DI_MIN $DI_MAX --no-daywarp
      # retention at the lineage's home rung
      one_eval "s2_W1_home"       "$ck" -1 1.0
      # and the far end of the band alone, where direct burning is IMPOSSIBLE
      one_eval "s2_W1_hard"       "$ck" 4.0 5.0
      say "---- stage 2 done ----"
      say "HEADLINE SHAPE: 's2_W1_native bought X deg of NODE-DOMINANT plane"
      say "      alignment from precession at Y m/s, where direct burning the"
      say "      same X deg costs Z = 2 v sin(X/2) m/s' — read X/Y/Z off the"
      say "      DECOMPOSITION and 'Dv used' lines above. MEAN ELEMENTS."
  else
      flatline "W1_driftwait" "$JSON_DIR/F1_floor_warm.json"
      say "---- stage 2 FAILED (no final ckpt) — see the flatline check ----"
  fi
else say "---- stage 2 SKIPPED ----"; fi

say "=========== ext-j2wait campaign end ==========="
