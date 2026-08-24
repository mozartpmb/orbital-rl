#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# T13b — CONSOLIDATION WITH A TIGHT-CELL ANCHOR
# T13 showed LR alone does not hold the tight box. This adds the one thing
# T13's tight defense lacked: a signal that survives reward silence.
# ═══════════════════════════════════════════════════════════════════════════
#
# ── WHAT T13 MEASURED, AND WHY IT POINTS HERE ───────────────────────────────
# T13 (LR 1e-3, consolidation weights, warm from the tight child) reproduced
# the seesaw in SOFT form: tight 95 -> 80 -> 55 across e20/e40/e80 while the
# wides surged (E2 85, LONGRANGE 90, E3 30). Two readings follow immediately:
#
#   * Red-team (a) is NOT firing. E3 climbing 6.5 -> 30 and E2 -> 85 says
#     1e-3 re-acquires the wide cells fine. The LR fallbacks (2e-3, staged)
#     are not needed and would be the wrong lever.
#   * The tight loss is a CLOCK failure, not an overwrite. Probe JSONs show
#     tight failures are ~all safety_cap timeouts, and successful-episode
#     median length stretches 1617 -> 2248 -> 2499 against a 3000 cap at
#     stable dv (~1.2-1.3x). The transfer still works; the wide gradient is
#     smearing the terminal fine-burn endgame until convergence outruns the
#     clock.
#
# THE MECHANISM THAT MAKES THAT SELF-REINFORCING: T13's tight defense is
# REWARD-MEDIATED ONLY. Under sparse terminal reward the tight gradient exists
# only while tight episodes still succeed — so as successes fall, the defense
# weakens, which costs more successes. That is precisely the anchor-free
# failure mode the research synthesis predicted (§1, "tight-cell-only anchor").
#
# ── THE ONE NEW MECHANISM ───────────────────────────────────────────────────
# A dense supervised anchor on tight-cell states only: per PPO update, add
# lambda * CE(pi_root || pi_theta) on a minibatch drawn from a FIXED dataset of
# tight-cell states collected by rolling the root. It survives reward silence
# because it does not depend on reward at all, and it cannot cap wide-cell
# recovery because it never touches a wide-cell state.
#
# Design per the synthesis: forward CE against the root's FULL 31-way softmax
# at tau=1 (Rusu Table 1: KL >= NLL >> MSE; Kickstarting uses teacher probs
# unmodified — the tau=0.01 folklore is for unnormalised Q-value teachers), and
# lambda CONSTANT with no decay, because the anchor IS the init (CLEAR-end)
# rather than a teacher to be weaned off.
#
# VARIANT CHOSEN: replay-BC, not cell-masked KL. Cell-masked KL is more
# faithful to on-policy states, but the cell id lives in the C env inside each
# rollout WORKER and nothing per-step-per-env crosses the multiprocessing
# boundary today (`info` is aggregated into stats, not aligned to buffer rows).
# Threading it means a new shared-memory channel + a new buffer tensor + binding
# changes — three files across a process boundary, in a project whose bug
# history is mostly silent plumbing errors. Replay-BC needs a tensor loaded at
# init and nothing else. Rejected on diff size and risk.
#
# ── SELF-RED-TEAM ───────────────────────────────────────────────────────────
#
# (i) DOES THE REPLAY DATASET GO STALE as pi_theta drifts? The states are
#     frozen; pi_theta moves. Two reasons this is second-order for a DEFENSE,
#     as opposed to for learning a new skill. First, the anchor's job is to
#     hold pi_theta NEAR the root on the root's own state distribution — if
#     pi_theta has not drifted, the states are exactly right; if it has, they
#     are exactly the states we want to pull it back toward. Staleness of the
#     TARGET would be fatal, and the target is frozen by construction. Second,
#     the synthesis's own objection to offline data (DAgger O(T*eps) vs BC
#     O(T^2*eps)) is about LEARNING a policy from a teacher's distribution;
#     here pi_theta STARTS as the generating policy, so the covariate shift at
#     t=0 is exactly zero and grows only as fast as the anchor is failing.
#     Stated, not hand-waved: if a future run shows tight holding on the anchor
#     set while degrading in-env, that is this effect, and the fix is periodic
#     dataset refresh from the current policy (DAgger-style), not more lambda.
#
# (ii) LAMBDA TOO HIGH CAPS TIGHT AT ROOT LEVEL. The anchor pulls toward the
#     root, so a large lambda forbids the tight cell from getting BETTER than
#     92.5% — and the consolidation mixture might otherwise buy a few points.
#     It also fights the PPO gradient on genuinely improved behaviour. Default
#     is therefore the LOW end of the synthesis's range: 0.02 of 0.01-0.05.
#     The trade is explicit: we are buying "tight does not collapse" at the
#     price of "tight probably does not improve much". Given T13's 95->55,
#     that is the right side of the trade, but it IS a trade.
#
# (iii) LSTM HIDDEN STATE — where a silent bug would live, so this is settled
#     by reading the trainer rather than by intuition. `train()` forwards its
#     minibatches with `lstm_h=None, lstm_c=None`, and `evaluate()` advances
#     buffer rows every `bptt_horizon` steps REGARDLESS of episode boundaries.
#     So the trainer ALREADY evaluates every 64-step window from a ZERO hidden
#     state wherever it sits in an episode. The anchor dataset is therefore cut
#     on that same cadence and both networks are forwarded zero-init. Reaching
#     for R2D2 stored-state/burn-in machinery here would have anchored a
#     DIFFERENT function than the one PPO's gradient shapes — silently, with
#     every number still looking plausible. Gate A3 pins the window shape.
#
# ── GATES BEHIND THIS SCRIPT ────────────────────────────────────────────────
#   A1a instrument determinism (pristine vs pristine) — needed because
#       pufferl leaves torch/numpy/random UNSEEDED, so a naive weight-hash
#       comparison can never pass and would have been a false failure
#   A1b anchor_lambda=0 is BIT-IDENTICAL to the pre-anchor trainer
#   A2  CE(anchor||anchor) == 0, i.e. the term is a divergence and adds no
#       gradient at step 0 — the correct initial condition
#   A3  dataset windows are (N, bptt_horizon, obs_dim)
# ═══════════════════════════════════════════════════════════════════════════
#
# LAUNCH FROM AN INTERACTIVE SESSION SHELL, NOT FROM AN AGENT.
#   nohup caffeinate -is bash scripts/orbital/extj2/t15s_dagger_campaign.sh \
#       > /tmp/t15s_stdout.log 2>&1 &
#
# TO KILL BY HAND, KILL THE WORKERS FIRST: `pkill -f` does not match the
# rollout workers (they spawn as `python -c "from multiprocessing.spawn ..."`
# and carry none of the launch flags), so killing the parent alone orphans
# eight processes spinning at ~170% CPU each. Use `pkill -TERM -P <pid>` then
# `kill -TERM <pid>` — which is what the watchdog below does, in that order.
#
# ── THE QUESTION, AND WHY THE PREVIOUS ANSWER DOESN'T SETTLE IT ─────────────
# T11-tight measured a bidirectional seesaw: the tight ladder recovered the
# tight box (0 -> 98.5%) and collapsed E2/E3/LONGRANGE; the rehearsal re-mix
# recovered those and re-zeroed the tight box. That was written up as skills
# interfering "structurally at this capacity". It does not show that.
#
# BOTH SWINGS RAN AT THE FULL ACQUISITION LR (1e-2, annealed). That is the rate
# for learning a skill, not for maintaining one: the majority gradient re-learns
# its own cells from scratch early and bulldozes the minority skill before
# rehearsal can defend it. The back swing then gave the tight box 0.10 weight —
# the bootstrap problem the ladder existed to solve in the first place. So the
# measured failure is OPTIMIZATION/CURRICULUM, and its two mechanisms are
# exactly the two knobs this run changes.
#
# The representation evidence points the other way, at the same 128 hidden
# units: the TB5-3D specialist holds tight AND loose simultaneously, and the
# E-ladder child holds four wide-band skills at once. Six is not obviously out
# of reach for a network that demonstrably does four, and separately does
# tight+loose.
#
# ONE COHERENT CHANGE SET — root, weights, LR. Nothing else moves:
#   root    the tight child (the skill hardest to rebuild is the one already
#           intact; wide cells rebuild readily under majority gradient, which
#           is the asymmetry both swings showed)
#   weights consolidation (TIGHT 0.10 -> 0.25; the minority stops being one)
#   LR      1e-2 -> 1e-3, annealed (maintenance regime)
# NO l2_init: Phase 4's R3 measured it HURTING at full LR, and adding a second
# untested mechanism would make an ambiguous result uninterpretable. ent_coef
# stays at the shipped 0.01.
#
# ── THE ACTUAL QUESTION IS THE TRAJECTORY, SO BOTH CHECKPOINTS ARE EVALUATED ─
# A single end-of-run battery cannot tell "converging hold" from "slower
# seesaw" — both can look identical at 100M. The mid checkpoint (~50M) is
# therefore evaluated on the FULL battery too, and the pair is the result:
#   tight high at 50M and still high at 100M, wides climbing  -> HOLD
#   tight high at 50M and falling by 100M                     -> slower seesaw
#   tight already gone at 50M                                 -> LR didn't help
#   wides flat at BOTH                                        -> LR too low (a)
#
# ── SELF-RED-TEAM ───────────────────────────────────────────────────────────
#
# (a) THE SHARPEST RISK: 1e-3 MAY BE TOO LOW FOR THE WIDE CELLS. This run is
#     framed as maintenance, but E2 (29.0%) and E3 (6.5%) are not being
#     maintained — they are being RE-ACQUIRED from a collapsed state, and
#     re-acquisition is what full LR is for. A maintenance LR could hold the
#     tight box and simply never rebuild E3, which would look like a capacity
#     result while actually being an LR result — the exact confusion this
#     campaign exists to end.
#     THE INSTRUMENT: the 50M mid-battery. If E2/E3 have not moved materially
#     off their 29.0/6.5 baseline by 50M, the LR is too low, and the finding is
#     about LR, not capacity.
#     THE FALLBACK, decided in advance so it is not improvised at 3 a.m.:
#     re-run at T15S_LR=2e-3; if that also splits the difference, run a STAGED
#     LR (2e-3 for the first ~30M to re-acquire the wides, then 1e-3 to
#     consolidate) via T15S_LR + T15S_LR2/T15S_LR_SWITCH. Staged is second, not
#     first, because it is two changes and this experiment is worth keeping to
#     one.
#
# (b) FUEL FLOORS ARE PER-CELL IN THE TABLE, as the shipped mixture already
#     does it. The tight cell carries its MEASURED floor (0.133 = 419 m/s);
#     every other cell keeps 0.113, which was measured against its own band.
#     At 0.113 the tight cell's terminal fine-burn train is unaffordable in
#     15.8% of episodes — training a consolidation run on unwinnable episodes
#     would present as interference while actually being infeasibility, which
#     is precisely the misreading this campaign cannot afford. The final
#     battery evaluates the tight cell at BOTH floors so the number stays
#     comparable to the T11 lineage.
#
# (c) ENTROPY AT LOW LR — MEASURED IN THE SMOKE, AND THE FLOOR RE-CALIBRATED.
#     The worry: a 10x smaller step with ent_coef unchanged lets the policy
#     sharpen without the entropy bonus pushing back, and a collapsed policy
#     holds everything it already does while learning nothing new — which would
#     MIMIC a successful consolidation at the mid battery and then flatline.
#     The 2M smoke, this exact command, entropy per epoch:
#
#         0.698 0.735 0.709 0.702 0.712 0.706 0.717 0.715
#         0.706 0.708 0.721 0.707 0.709 0.705 0.720
#
#     STABLE — flat to +/-0.02 over 2M, drifting very slightly UP. No collapse
#     at 1e-3 over this horizon.
#     Two things that changes. First, the floor: an initial draft used 0.8,
#     taken from the ~1.62 seen at ACQUISITION LR, and it would have fired on
#     epoch 1 of every run — an alarm that is always on is not an alarm. The
#     floor is now 0.45, well under the measured 0.71 operating point but far
#     above zero. Second, a fact worth carrying: the tight child ARRIVES at
#     0.70 against 1.62 mid-acquisition, i.e. consolidation starts from an
#     already-sharpened converged specialist. That is expected, and it is also
#     why "entropy is low" cannot by itself be read as collapse here.
#     ent_coef stays 0.01: changing it would add a second variable.
#
# ── W1_driftwait IS EXCLUDED FROM TRAINING, AND SAID SO ─────────────────────
# Weight 0.0. It scored 0.0 in all three seesaw states — dead gradient in both
# directions — and its 22000-step cap costs disproportionate wall-clock per
# unit of learning. The target here is the SIX-skill consolidation. It is still
# EVALUATED in every battery, so the zero stays on the record instead of
# quietly leaving it.
#
# ── THE BASELINE IS NOT RE-MEASURED ─────────────────────────────────────────
# The warm root's own battery is already on record from T11-tight at the same
# 200 eps / seed 123 / native BO, so re-running it would burn ~1.5 h to
# reproduce numbers we hold. It is printed below and every RESULT line is read
# against it.
set -uo pipefail

MAIN=/Users/pete/space_training
PUF=$MAIN/pufferlib
BRANCH_REQ=main
EVAL=$MAIN/scripts/orbital/extj2/t11_eval.py
GATES=$MAIN/scripts/orbital/extj2/t11_gates.py
VERIFY=$MAIN/scripts/orbital/extj2/verify_extj2.py
ROOT=${T15S_ROOT:-$MAIN/models/t3/t13b_anchor_final.pt}
PROG=/tmp/t15s_progress.log
JSON_DIR=$MAIN/web_data/results/t15seed
EPS=200
EPS_W1=${T15S_EPS_W1:-200}
EPS_SIX=${T15S_EPS_SIX:-50}
EVAL_SEED=123
LEG1_STEPS=${T15S_LEG1_STEPS:-150000000}
LEG2_STEPS=${T15S_LEG2_STEPS:-75000000}
STEPS=$LEG1_STEPS
MID_STEPS=${T15S_MID_STEPS:-50000000}
WATCHDOG_S=900
NAV_MAX_TICKS=${T15S_NAV_MAX_TICKS:-120}

# THE load-bearing knob. 1e-3 = maintenance. See red-team (a) for the fallbacks.
LR=${T15S_LR:-1e-3}

# ── THE ONE NEW MECHANISM vs T13 ────────────────────────────────────────────
# Everything else in this script is T13's, byte for byte, so any difference in
# outcome is attributable to the anchor and nothing else.
# ── TWO TEACHERS, TWO JOBS, TWO LAMBDAS ─────────────────────────────────────
# ACQUISITION: w1nav_child teaches W1, which this root scores 0.0/25 on. The
#   root has SIX skills; W1 is the single cell at exactly zero, and the
#   bootstrap law (f_k(0)=0) says sparse reward alone never leaves that saddle.
# DEFENSE: the root anchors its own TIGHT cell, CLEAR-style — the anchor IS the
#   init, so CE=0 at step 0 and it adds no gradient until the policy drifts.
#
# LAMBDA IS SET ON MEASURED CE, NOT ON THE LITERATURE'S RAW NUMBERS. The
# kickstarting-vs-CLEAR comparison (~0.5 decayed vs 0.01-0.05 constant) is
# lambda RELATIVE TO THE POLICY LOSS, and our two teachers' CE magnitudes
# differ by ~200x:
#     CE(w1nav_child || root) on W1 states = 10.23   (measured, gate T2c)
#     CE(root || root) on TIGHT states     =  0.00   (rises only as it drifts)
# What must be sane is lambda x CE. At lambda 0.30 the acquisition term would
# be 3.07 against a PPO policy loss of O(0.01-0.1) — 30-300x dominance from
# step 0, which is not kickstarting but pure distillation, and would drag the
# six skills toward a W1 specialist. 0.05 gives ~0.51, comparable-to-above the
# policy term, weaning to ~0.10 by 30M.
ACQ_LAMBDA=${T15S_ACQ_LAMBDA:-0.05}
ACQ_LAMBDA_END=${T15S_ACQ_LAMBDA_END:-0.01}
ACQ_DECAY=${T15S_ACQ_DECAY:-30000000}
DEF_LAMBDA=${T15S_DEF_LAMBDA:-0.02}
ANCHOR_MB=${T15S_ANCHOR_MB:-4}
ACQ_DATA=${T15S_ACQ_DATA:-$MAIN/models/t15/anchor_w1_k0.pt}
ACQ_CKPT=${T15S_ACQ_CKPT:-$MAIN/models/t3/w1nav_child.pt}
DEF_DATA=${T15S_DEF_DATA:-$MAIN/models/t15/anchor_tight_k0.pt}
DEF_CKPT=${T15S_DEF_CKPT:-$MAIN/models/t3/t13b_anchor_final.pt}
NAV_MAX_TICKS=${T15S_NAV_MAX_TICKS:-0}
FILTER_IMPL=${T15S_FILTER_IMPL:-c}
FUZZ=$MAIN/scripts/orbital/ext_recon/navc_fuzz.py
PREFLIGHT=$MAIN/scripts/orbital/ext_recon/navc_preflight.py
ROOTGATE=$MAIN/scripts/orbital/ext_recon/root_gate.py
T15GATES=$MAIN/scripts/orbital/ext_recon/t15_gates.py
KERNEL_DIR=$PUF/pufferlib/ocean/orbital_nav
LR2=${T15S_LR2:-}                 # staged-LR second leg (unset = single LR)
LR_SWITCH=${T15S_LR_SWITCH:-30000000}

# Tripwire: ADVISORY by default, and that is a deliberate choice — see below.
TRIP_AT=${T15S_TRIP_AT:-25000000}
TRIP_EPS=${T15S_TRIP_EPS:-20}
TRIP_FLOOR=${T15S_TRIP_FLOOR:-46}   # half the root's 92.5% tight
TRIP_FATAL=${T15S_TRIP_FATAL:-0}
ENT_FLOOR=${T15S_ENT_FLOOR:-0.45}   # measured operating point 0.71; see red-team (c)

T15S_SEED=${T15S_SEED:-7}
SEED_SFX=""
[ "$T15S_SEED" != "42" ] && SEED_SFX="_s${T15S_SEED}"
STAGES=${T15S_STAGES:-0,1,2,3}
want() { [[ ",$STAGES," == *",$1,"* ]]; }

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

EPOCH_SLACK=${T15S_EPOCH_SLACK:-3}
ARM_DIR="$PUF/experiments_t15seed/consol${SEED_SFX}"
final_ckpt() {
    local dir="$1" want_ep="$2"
    ls -t "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null | while read -r f; do
        local n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$((want_ep - EPOCH_SLACK))" ] && { echo "$f"; return 0; }
    done
    return 0
}
# The FIRST checkpoint at or past an epoch — used for the mid battery and the
# tripwire, where "nearest at or after" is what we want, not "latest".
ckpt_at() {
    # Sort on the EXTRACTED epoch, not on field positions in the full path:
    # `sort -t_ -kN` happens to work for today's directory layout and silently
    # reorders the moment a path component gains an underscore, which would
    # hand the mid battery and the tripwire the wrong checkpoint without
    # erroring.
    local dir="$1" ep="$2" f n
    for f in $(ls "$dir"/*/model_puffer_orbital_nav_*.pt 2>/dev/null \
               | sed 's/.*_\([0-9][0-9]*\)\.pt$/\1 &/' | sort -n | cut -d' ' -f2-); do
        n=${f##*_}; n=${n%.pt}
        [ "$((10#$n))" -ge "$ep" ] && { echo "$f"; return 0; }
    done
    return 0
}
json_rate() {
    python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try: print(int(round(100*json.load(open(sys.argv[1]))['rate'])))
except Exception: pass
PY
}

preflight() {
    local br; br=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)
    [ "$br" = "$BRANCH_REQ" ] || { say "ABORT: $MAIN on '$br', need '$BRANCH_REQ'"; return 1; }
    for f in "$EVAL" "$GATES" "$VERIFY" "$ROOT" "$ACQ_DATA" "$ACQ_CKPT" "$DEF_DATA" "$DEF_CKPT"; do
        [ -f "$f" ] || { say "ABORT: missing $f"; return 1; }
    done
    # T15b DELIBERATELY BREAKS T15's "defense teacher == root" invariant, and
    # this guard is relaxed on purpose rather than silently. In T15 the defense
    # anchor was pure CLEAR: the anchor IS the init, CE=0 at step 0, defend what
    # you already have. T15b's root has TIGHT at 85.5, so anchoring to the root
    # would defend the DEGRADED level. The defense teacher therefore stays
    # `t13b_anchor_final` (TIGHT 95.0) — the better behaviour — which makes this
    # anchor mildly ACQUISITION-flavoured for TIGHT rather than pure defense.
    #
    # Two consequences, both measured and both acceptable:
    #   * CE at step 0 is no longer 0 — it is 0.035 (t13b teacher vs current
    #     policy on TIGHT states), so the pull is real but gentle.
    #   * The anchor is measured SATURATED at that CE, so it cannot by itself
    #     restore TIGHT; the recovery has to come from TIGHT's reward share,
    #     which is why TIGHT keeps 34.8% of steps in this mixture.
    # The guard now checks the teacher is COMPETENT on the cell it defends,
    # which is the property that actually matters.
    if [ "$(shasum -a256 "$DEF_CKPT" | cut -d" " -f1)" = \
         "$(shasum -a256 "$ROOT" | cut -d" " -f1)" ]; then
        say "  NOTE defense teacher == root (pure CLEAR semantics, CE=0 at step 0)"
    else
        say "  NOTE defense teacher != root BY DESIGN — anchoring TIGHT to the"
        say "       better behaviour (t13b 95.0) rather than the root's 85.5."
        python3 "$ROOTGATE" --root "$DEF_CKPT" --cell TIGHT_5k1 \
            --nav-mode bearings_only --expect 95 --tol 15 --episodes 40 \
            --filter-impl "$FILTER_IMPL" > /tmp/t15s_defgate.log 2>&1 \
          || { say "ABORT: defense teacher fails root_gate on TIGHT_5k1"; return 1; }
        say "  RESULT defense-teacher root_gate: $(grep -oE '[0-9]+/[0-9]+ root gates pass' /tmp/t15s_defgate.log | tail -1)"
    fi
    if [ "$(shasum -a256 "$ACQ_CKPT" | cut -d" " -f1)" = \
         "$(shasum -a256 "$ROOT" | cut -d" " -f1)" ]; then
        say "ABORT: acquisition-anchor teacher IS the root — it cannot teach W1"; return 1
    fi
    python3 - <<'PY' || { say "ABORT: consolidation table not resolvable"; return 1; }
import sys
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/extj2')
import t11_cells as T
# T15d REUSES mixture 5 (T15C weights) — verified unchanged at the MEAN
# decisions/ep (290): W1 44.5%, TIGHT 35.3%, wides 20.2% against targets
# 45.9/34.3/19.7. The median swung 307 -> 63 only because the episode
# distribution is BIMODAL and the median flips mode as success crosses
# 50%; total steps depend on the MEAN, which moved 347 -> 290.
w = {n: c['weight'] for n, c in T.T15C_CELLS}
assert abs(sum(w.values()) - 1.0) < 1e-12, w
# weights are SAMPLING weights; the 44.4/34.8 figures are the resulting
# per-STEP shares, RE-SOLVED at this root because W1's decisions/episode fell
# 536 -> 177 once it became competent (success eps are 46 decisions, failures
# 359). A fixed weight therefore delivers a DECREASING gradient share exactly
# as the skill starts working — a built-in brake, and a candidate explanation
# for the 31.5 plateau alongside the lambda wean.
assert abs(w['W1_driftwait'] - 0.19) < 1e-12 and abs(w['TIGHT_5k1'] - 0.07) < 1e-12, w
assert dict(T.T15C_CELLS)['TIGHT_5k1']['fuel_min'] == 0.133
# the shipped mixture must be untouched by the variant's existence
s = {n: c['weight'] for n, c in T.CELLS}
assert len(T.CELLS) == 7 and abs(s['TIGHT_5k1'] - 0.10) < 1e-12, s
tf = dict(T.CONSOL_CELLS)['TIGHT_5k1']['fuel_min']
assert abs(tf - 0.133) < 1e-12, tf
PY
    mkdir -p "$JSON_DIR"; return 0
}

anchor() {
    say "START stage 0 (anchors, both families, + the T11 gate battery)"
    cd "$MAIN" || return 1
    python3 "$VERIFY" --stage a1,a2,a3,a4,a5 --eps 100 > /tmp/t15s_anchor.log 2>&1
    local ra=$?
    say "  RESULT anchors rc=$ra $(grep -oE '[0-9]+/[0-9]+ checks pass' /tmp/t15s_anchor.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t15s_anchor.log | while read -r l; do say "    $l"; done
    python3 "$GATES" > /tmp/t15s_gates.log 2>&1
    local rg=$?
    say "  RESULT t11 gates rc=$rg $(grep -oE '[0-9]+/[0-9]+ gates pass' /tmp/t15s_gates.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t15s_gates.log | while read -r l; do say "    $l"; done
    grep -E '^\s+\[(PASS|FAIL)\] G9' /tmp/t15s_gates.log | while read -r l; do say "    $l"; done
    if [ $ra -ne 0 ] || [ $rg -ne 0 ]; then say "ABORT: stage 0 FAILED"; return 1; fi
    say "---- stage 0 done ----"; return 0
}

# ── one_eval <tag> <ckpt> <cell|all> <nav_mode> [extra...] ─────────────────
one_eval() {   # one_eval <tag> <ckpt> <cell> <nav> [eps] [extra...]
    local tag="$1" ck="$2" cell="$3" nav="$4"; shift 4
    local EPS="${EPS}"
    case "${1:-}" in ''|*[!0-9]*) : ;; *) EPS="$1"; shift ;; esac
    cd "$MAIN" || return 1
    tag="${tag}${SEED_SFX}"
    local L=/tmp/t15s_${tag}.log
    python3 "$EVAL" --ckpt "$ck" --cell "$cell" --nav-mode "$nav" \
        --episodes $EPS --seed $EVAL_SEED --label "$tag" \
        --out "$JSON_DIR/${tag}.json" "$@" > "$L" 2>&1
    say "RESULT $tag: $(grep -E 'success [0-9]+/' "$L" | head -1 | sed 's/^ *//')"
    local g
    for pat in 'causes:' 'FUEL AUDIT' 'dv used'; do
        g=$(grep -F "$pat" "$L" | head -1 | sed 's/^ *//'); [ -n "$g" ] && say "  $tag $g"
    done
}

CELLS_ALL="E0_j2 E1_j2 E2_j2 E3_j2 W1_driftwait TIGHT_5k1 LONGRANGE"

# the warm root's own battery, from T11-tight (200 eps, seed 123, native BO).
# A function, not `declare -A`: this machine's only bash is 3.2, which parses
# associative-array subscripts as arithmetic and dies under set -u at launch.
base_of() {
    case "$1" in
        E0_j2) echo 97.0;;  E1_j2) echo 96.5;;  E2_j2) echo 29.0;;
        E3_j2) echo 6.5;;   W1_driftwait) echo 0.0;;
        TIGHT_5k1) echo 92.5;;  LONGRANGE) echo 40.0;;  *) echo "?";;
    esac
}

battery() {      # battery <prefix> <ckpt>
    local pre="$1" ck="$2"
    for C in $CELLS_ALL; do
        one_eval "${pre}_${C}" "$ck" "$C" bearings_only
        say "  ^ ${C} baseline (tight child) was $(base_of "$C")%"
    done
}


# ── the training invocation, shared by both legs ───────────────────────────
train_leg() {   # train_leg <arm> <root> <steps> <mixture> <anchor_specs_json>
    local arm="$1" root="$2" steps="$3" mix="$4" specs="$5"
    local dir="$PUF/experiments_t15seed/${arm}${SEED_SFX}"
    local want_ep=$(( steps / 131072 ))
    if [ -n "$(final_ckpt "$dir" "$want_ep")" ]; then
        say "SKIP train $arm (final ckpt present)"; echo "$dir"; return 0
    fi
    say "START train $arm: $steps steps, mixture=$mix, root $(basename "$root")"
    say "  anchors: $specs"
    cd "$PUF" || return 1
    python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps "$steps" --train.seed "$T15S_SEED" \
        --train.data-dir "$dir" --load-model-path "$root" \
        --train.learning-rate "$LR" --train.checkpoint-interval "${T15S_CKPT_INT:-20}" \
        --train.anchor-specs "$specs" \
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
        --env.nav-filter-impl "$FILTER_IMPL" --env.t11-mixture "$mix" \
        --wandb --wandb-project orbital-rl --wandb-group "t15seed-${arm}${SEED_SFX}" \
        --tag "t15s_${arm}${SEED_SFX}" > "/tmp/t15s_${arm}${SEED_SFX}_train.log" 2>&1 &
    local pid=$!; say "  trainer pid $pid"
    local tf=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [ "$tf" -eq 0 ] && [ -n "$(final_ckpt "$dir" "$want_ep")" ]; then
            tf=$(date +%s); say "  final ckpt present; watchdog armed (${WATCHDOG_S}s)"
        fi
        if [ "$tf" -ne 0 ] && [ $(( $(date +%s) - tf )) -ge $WATCHDOG_S ]; then
            say "  WATCHDOG: killing $pid"
            pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 20
            pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; break
        fi
    done
    echo "$dir"
}

# ABBREVIATED battery. W1 at full precision because it IS the claim; the six
# others at 50 eps because the only question asked of them is "held or
# collapsed", for which +/-14pp is ample.
abbrev_battery() {   # abbrev_battery <prefix> <ckpt>
    local pre="$1" ck="$2"
    one_eval "${pre}_W1_driftwait" "$ck" W1_driftwait bearings_only "$EPS_W1"
    for C in E0_j2 E1_j2 E2_j2 E3_j2 TIGHT_5k1 LONGRANGE; do
        one_eval "${pre}_${C}" "$ck" "$C" bearings_only "$EPS_SIX"
    done
}

# ═══════════════════════════════════════════════════════════════════════════
say "=========== T15 SEED CHECK (seed $T15S_SEED) start (pid $$) ==========="
say "CLAIM UNDER TEST: 'kickstart anchor + one DAgger refresh teaches W1"
say "                  in-mixture from zero'. Currently single-seed (42)."
say "THIS IS A SHAPE-REPRODUCTION CHECK, NOT ENDPOINT REPLICATION."
say "  pass = W1 off zero to ~25-35 on leg 1, a further ~+10-15 on leg 2,"
say "         with the six other cells holding. Exact numbers are NOT the claim."
say ""
say "*** REDUCED-PRECISION SIDE ROWS ***  The six non-W1 cells are evaluated at"
say "    $EPS_SIX episodes, not 200. That is a HOLD-vs-COLLAPSE spot check with a"
say "    ~+/-14pp CI. These numbers must never be quoted as capability claims;"
say "    the s42 arc's 200-ep batteries remain the citable rows. W1 runs at"
say "    $EPS_W1 eps because W1 IS the claim under test."
say ""
say "PRE-REGISTERED READINGS (all three are useful; none is a failure):"
say "  (a) both legs reproduce shape -> mechanism seed-robust, write-up goes to 2 seeds"
say "  (b) one leg reproduces, the other does not -> the mechanism SPLITS;"
say "      report WHICH half, which is more informative than either pure result"
say "  (c) W1 stays ~0 at seed $T15S_SEED -> mechanism is seed-SENSITIVE and the"
say "      whole T15 arc inherits a seed-sensitivity caveat"
say ""
say "leg 1:  T15 recipe — root $(basename $ROOT), mixture 3, $LEG1_STEPS steps,"
say "        ACQ lambda $ACQ_LAMBDA->$ACQ_LAMBDA_END over $ACQ_DECAY (T15's own schedule)"
say "leg 2:  T15b refresh — root = leg-1 child, dataset REBUILT from the SEED-$T15S_SEED"
say "        student's own visited states, ACQ lambda 0.20 constant, $LEG2_STEPS steps"
say "seed:   $T15S_SEED   suffix '${SEED_SFX}' on every artifact (no collision with s42)"
say "STATE:  all rows MEAN-ELEMENT claims under J2"
say "stages requested: $STAGES"
preflight || exit 1
if want 0; then anchor || exit 1; else say "---- stage 0 SKIPPED ----"; fi

# ── stage 1: LEG 1 — the kickstart, at seed 7 ──────────────────────────────
if want 1; then
  SPECS1=$(python3 - <<PYJ
import json
print(json.dumps([
 {"name":"w1_acq","data":"$ACQ_DATA","ckpt":"$ACQ_CKPT",
  "lambda":$ACQ_LAMBDA,"lambda_end":$ACQ_LAMBDA_END,
  "decay_steps":$ACQ_DECAY,"minibatch":$ANCHOR_MB},
 {"name":"tight_def","data":"$DEF_DATA","ckpt":"$DEF_CKPT",
  "lambda":$DEF_LAMBDA,"minibatch":$ANCHOR_MB}]))
PYJ
)
  D1=$(train_leg leg1 "$ROOT" "$LEG1_STEPS" 3 "$SPECS1")
  CK1=$(final_ckpt "$D1" $(( LEG1_STEPS / 131072 )))
  if [ -z "$CK1" ]; then say "---- stage 1 FAILED (no ckpt) ----"; else
    say "  leg-1 ckpt $(basename "$CK1")"
    abbrev_battery "leg1" "$CK1"
    say "  READ leg 1: W1 off zero to ~25-35 reproduces the KICKSTART half."
    say "---- stage 1 done ----"
  fi
else say "---- stage 1 SKIPPED ----"; fi

# ── stage 2: build the SEED-7 refresh dataset from leg 1's own student ─────
# This cannot be prepared in advance: the states must come from the seed-7
# student, which does not exist until stage 1 finishes. Built here, gated here.
if want 2; then
  CK1=${T15S_CK1:-$(final_ckpt "$PUF/experiments_t15seed/leg1${SEED_SFX}" $(( LEG1_STEPS / 131072 )))}
  if [ -z "$CK1" ]; then say "  SKIP stage 2: no leg-1 child"; else
    SSET=$MAIN/models/t15/anchor_w1_student_s${T15S_SEED}.pt
    AGG=$MAIN/models/t15/anchor_w1_dagger_s${T15S_SEED}.pt
    say "START stage 2 (rebuild the anchor from the SEED-$T15S_SEED student)"
    python3 "$MAIN/scripts/orbital/extj2/build_tight_anchor.py" \
        --ckpt "$CK1" --out "$SSET" --cell W1_driftwait \
        --windows 900 --horizon 64 --envs 64 --max-ticks "$NAV_MAX_TICKS" \
        --filter-impl "$FILTER_IMPL" > /tmp/t15s_build.log 2>&1 \
      || { say "ABORT: state collection failed"; exit 1; }
    say "  RESULT $(grep -E 'windows' /tmp/t15s_build.log | head -1 | sed 's/^ *//')"
    python3 - "$MAIN/models/t15/anchor_w1_k0.pt" "$SSET" "$AGG" <<'PYA'
import sys, torch
t = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
n = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
torch.save({'obs': torch.cat([t['obs'], n['obs']], 0), 'cell': 'W1_driftwait',
            'horizon': 64, 'ckpt': 'SEED-CHECK: teacher-visited + seed student'},
           sys.argv[3])
PYA
    # decisions/ep on the seed-7 student, for the leg-2 mixture choice
    MEAN=$(python3 "$MAIN/scripts/orbital/ext_recon/t15c_w1_failure_modes.py" \
              --ckpt "$CK1" --eps 60 --n 32 2>/dev/null \
            | python3 -c "
import sys,re
t=sys.stdin.read()
sr=re.search(r'success (\d+)/(\d+)',t); ds=re.findall(r'decisions     median +(\d+)',t)
if sr and len(ds)>=2:
    p=int(sr.group(1))/int(sr.group(2)); print(int(p*int(ds[1])+(1-p)*int(ds[0])))
else: print(0)")
    say "  RESULT seed-$T15S_SEED student W1 decisions/ep MEAN = ${MEAN:-?}"
    # mixture 5 was solved at mean 283-347; outside that band fall back to 3
    if [ -n "$MEAN" ] && [ "$MEAN" -ge 200 ] && [ "$MEAN" -le 450 ]; then
        LEG2_MIX=5; say "  -> leg 2 uses mixture 5 (mean in the band it was solved for)"
    else
        LEG2_MIX=3; say "  -> leg 2 falls back to mixture 3 (mean outside the band)"
    fi
    echo "$LEG2_MIX" > /tmp/t15s_leg2_mix
    T15_AGG="$AGG" T15_STUDENT="$CK1" T15_SSET="$SSET" \
      python3 "$MAIN/scripts/orbital/ext_recon/t15b_gates.py" > /tmp/t15s_dgates.log 2>&1
    say "  RESULT D-gates $(grep -oE '[0-9]+/[0-9]+ T15b anchor gates pass' /tmp/t15s_dgates.log | tail -1)"
    grep -E '^\s+\[FAIL\]' /tmp/t15s_dgates.log | while read -r l; do say "    $l"; done
    say "---- stage 2 done ----"
  fi
else say "---- stage 2 SKIPPED ----"; fi

# ── stage 3: LEG 2 — one DAgger refresh, at seed 7 ─────────────────────────
if want 3; then
  CK1=${T15S_CK1:-$(final_ckpt "$PUF/experiments_t15seed/leg1${SEED_SFX}" $(( LEG1_STEPS / 131072 )))}
  AGG=$MAIN/models/t15/anchor_w1_dagger_s${T15S_SEED}.pt
  LEG2_MIX=$(cat /tmp/t15s_leg2_mix 2>/dev/null || echo 5)
  if [ -z "$CK1" ] || [ ! -f "$AGG" ]; then say "  SKIP stage 3: leg-1 child or dataset missing"; else
    SPECS2=$(python3 - <<PYJ
import json
print(json.dumps([
 {"name":"w1_acq","data":"$AGG","ckpt":"$ACQ_CKPT",
  "lambda":0.20,"lambda_end":0.20,"decay_steps":0,"minibatch":$ANCHOR_MB},
 {"name":"tight_def","data":"$DEF_DATA","ckpt":"$DEF_CKPT",
  "lambda":$DEF_LAMBDA,"minibatch":$ANCHOR_MB}]))
PYJ
)
    D2=$(train_leg leg2 "$CK1" "$LEG2_STEPS" "$LEG2_MIX" "$SPECS2")
    CK2=$(final_ckpt "$D2" $(( LEG2_STEPS / 131072 )))
    if [ -z "$CK2" ]; then say "---- stage 3 FAILED (no ckpt) ----"; else
      say "  leg-2 ckpt $(basename "$CK2")"
      abbrev_battery "leg2" "$CK2"
      say "  ── VERDICT (read against the PRE-REGISTERED readings) ──"
      say "  s42 reference shape: W1 0.0 -> 31.5 (kickstart) -> 45.0 (one refresh)"
      say "  (a) both legs in band  -> mechanism seed-robust, 2 seeds"
      say "  (b) only one in band   -> the mechanism SPLITS; name the half"
      say "  (c) W1 ~0 after leg 1  -> seed-SENSITIVE; the arc takes the caveat"
      say "  The six side rows are $EPS_SIX-ep spot checks, NOT capability claims."
      say "---- stage 3 done ----"
    fi
  fi
else say "---- stage 3 SKIPPED ----"; fi

say "=========== t15seed-check COMPLETE ==========="
