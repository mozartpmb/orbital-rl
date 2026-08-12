#!/bin/bash
# ── NAV-F dual-control campaign, remaining stages, end to end ────────────────
#
# Runs unattended under nohup + caffeinate so it survives turn boundaries and
# device sleep. Every stage appends a RESULT line to $PROG; checkpoints are
# copied into models/t3/ and per-arm JSON into web_data/results/navf/.
#
#   stage 1  T-BO      eval battery (training already complete)
#   stage 2  T-BO+Sigma  train -> eval
#   stage 3  T-BO-act    train -> eval
#
# Nothing here commits to git — the operator commits after reading $PROG, so a
# half-finished stage can never land a partial row.
set -uo pipefail

WT=/Users/pete/space_training-extnav
PUF=$WT/pufferlib
WS=$WT/models/t3/extnav_TB5_warmstart_col21zero.pt
PROG=/tmp/navf_campaign_progress.log
EPS=200
STEPS=50000000

export PYTHONPATH=$PUF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

# train <arm> <extra env flags...>
train() {
    local arm="$1"; shift
    local dir="$PUF/experiments_extnav/navf_${arm}"
    if ls "$dir"/*/model_*000382.pt >/dev/null 2>&1; then
        say "SKIP train $arm (final ckpt already present)"
        return 0
    fi
    say "START train $arm"
    cd "$PUF" || return 1
    caffeinate -is python3 -m pufferlib.pufferl train puffer_orbital_nav \
        --train.device cpu --train.total-timesteps $STEPS --train.seed 42 \
        --env.rendezvous-radius-m 5000 --env.rel-vel-tol-ms 1.0 \
        --env.legacy-action-space 20 \
        --load-model-path "$WS" \
        "$@" \
        --wandb --wandb-project orbital-rl --wandb-group "t5-navf-${arm}" \
        --train.data-dir "$dir" > "/tmp/navf_${arm}.log" 2>&1
    local rc=$?
    say "END train $arm rc=$rc last=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "/tmp/navf_${arm}.log" | grep -oE 'perf +[0-9.]+' | tail -1)"
    return $rc
}

# evaluate <arm> [extra navf_eval flags...]
evaluate() {
    local arm="$1"; shift
    local ck
    ck=$(ls -t "$PUF"/experiments_extnav/navf_${arm}/*/model_*.pt 2>/dev/null \
         | grep 000382 | head -1)
    if [ -z "$ck" ]; then
        ck=$(ls -t "$PUF"/experiments_extnav/navf_${arm}/*/model_*.pt 2>/dev/null | head -1)
    fi
    if [ -z "$ck" ]; then say "FAIL eval $arm: no checkpoint"; return 1; fi
    say "START eval $arm ckpt=$ck"
    cp "$ck" "$WT/models/t3/extnav_navf_${arm}.pt"
    cd "$PUF" || return 1
    python3 "$WT/scripts/orbital/nav/navf_eval.py" \
        --ckpt "$ck" --label "$arm" --boxes tb5,tb4 --eps $EPS \
        --json-out "/tmp/navf_${arm}.json" "$@" \
        > "/tmp/navf_${arm}_eval.log" 2>&1
    local rc=$?
    mkdir -p "$WT/web_data/results/navf"
    cp "/tmp/navf_${arm}.json" "$WT/web_data/results/navf/" 2>/dev/null
    # pull the two headline numbers straight out of the eval log
    local n5 n4 b5 z5
    n5=$(grep -oE "${arm}/tb5 +success [0-9]+/[0-9]+ = +[0-9.]+%" "/tmp/navf_${arm}_eval.log" | head -1)
    n4=$(grep -oE "${arm}/tb4 +success [0-9]+/[0-9]+ = +[0-9.]+%" "/tmp/navf_${arm}_eval.log" | head -1)
    b5=$(grep -A1 "PRIMARY" "/tmp/navf_${arm}_eval.log" | head -1)
    say "END eval $arm rc=$rc"
    say "RESULT $arm TB5: ${n5:-?}"
    say "RESULT $arm TB4: ${n4:-?}"
    say "RESULT $arm PRIMARY: ${b5:-?}"
    return $rc
}

say "=========== NAV-F campaign start (pid $$) ==========="

# ── stage 1: T-BO eval (its training finished earlier) ──────────────────────
evaluate T_BO
say "---- stage 1 done ----"

# ── stage 2: T-BO+Sigma ─────────────────────────────────────────────────────
train T_BOS --env.nav-mode bearings_only --env.nav-sigma-channel 1
evaluate T_BOS --sigma-channel
say "---- stage 2 done ----"

# ── stage 3: T-BO-act ───────────────────────────────────────────────────────
train T_BOACT --env.nav-mode bearings_only --env.nav-block-fine-below-m 10000
evaluate T_BOACT --block-fine-below-m 10000
say "---- stage 3 done ----"

say "=========== NAV-F campaign COMPLETE ==========="
