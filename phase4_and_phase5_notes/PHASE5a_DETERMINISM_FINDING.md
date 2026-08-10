# Phase 5a — Determinism Finding (mid-execution halt)

**Date:** 2026-04-28
**Status:** Phase 5a paused after Investigation D and the start of Investigation A. Awaiting decision on how to proceed.

---

## TL;DR

Training is **not deterministic at fixed seed** under the current vec-env backend. The same code + same config + same seed produces a ~70pp spread in Stage 1 final performance across runs. This invalidates the Phase 5a spec's assumption that single-seed first-pass comparisons are meaningful, and partially weakens some Phase 4.5 attribution claims.

Phase 5a's Investigation D (apoapsis-bias) completed. Investigations A, B, C are paused pending a decision on how to handle the noise floor.

---

## What I observed

While running Phase 5a Investigation A, the shared Stage 1 at `(phase_gap_max=π/6, e_max_target=0.05)`, seed 42, ended at **0.1% perf** (peak ~1% mid-training). Phase 4 R4 Stage 1 at `(π/6, e=0)`, seed 42 had reached 77% at the same step count.

I initially read this as "adding 5% target eccentricity tanks Stage 1." Before continuing the rest of A on that interpretation, I ran a sanity check: re-train the *exact* Phase 4 R4 Stage 1 recipe (e=0, π/6, seed 42, byte-identical code).

**Sanity-check results at fixed seed = 42, identical config:**

| Run | When | cwd | Final/peak perf @ 10M |
|---|---|---|---|
| Phase 4 R4 Stage 1 (`puffer_orbital_jro86awn`) | 2026-04-24 | pufferlib/ | **77.0%** |
| Phase 5a sanity #1 | today | space_training/ (root) | **5% final, 10% peak** |
| Phase 5a sanity #2 (killed at 7.3M) | today | pufferlib/ | **40.3% peak** |

That's a 70pp spread across runs at fixed seed, same code, same config.

**Eval of the Phase 4 ckpt itself today** at the same conditions: 71/100 (71%) — within noise of its training-time 77%. So the *checkpoint* is reproducible; only *training* is not.

---

## Root cause

`pufferlib/pufferlib/config/default.ini`:
```ini
backend = Multiprocessing
torch_deterministic = True
```

- `torch_deterministic=True` makes cuDNN/cuBLAS ops deterministic. It does not affect vec-env worker scheduling.
- `backend = Multiprocessing` forks worker processes for parallel env stepping. The order in which `vecenv.recv()` returns step results depends on OS scheduling of those workers.
- PPO's minibatch ordering, advantage normalization, and gradient noise are sensitive to this ordering at fixed seed.

This is documented PufferLib behavior: deterministic vec stepping requires `backend = Serial`, which is single-process and ~10× slower SPS.

---

## Implications

### 1. Phase 5a's premise needs revision

Spec §3.5 (Investigation A): *"Single seed is sufficient for first-pass ordering."*
Spec §4.4 (Investigation B): *"Compare arms on Stage 2 final success rate"* (single seed).
Spec §5.3 (Investigation C): explicitly multi-seed.

With ~70pp single-seed variance, single-seed comparisons cannot reliably distinguish curriculum orderings (square vs tall vs wide differ by maybe 5-10pp at convergence; below the noise floor). The investigation as specified is under-powered.

### 2. The original Phase 5a Stage 1 result becomes ambiguous

The 1% perf at `(π/6, e=0.05)` could be:
- **(a)** Genuine: e=0.05 makes the rendezvous reward-landscape harder (eccentric targets demand orbit-shape matching, not just position matching).
- **(b)** Bootstrap unlucky: same recipe at e=0 also produced 5% in the first sanity, well within "unlucky bootstrap" range.

To distinguish, would need 3+ seeds at each condition.

### 3. Phase 4 narrative needs a noise-floor caveat

The Phase 4 multi-seed table (42: 77%, 1337: 0%, 20260423: 89.5%) was framed as "Stage 1 fresh has ~1/3 retry rate." Better framing now: **the seed isn't determinative; the random worker scheduling is.** Re-running seed 1337 might produce 0% one time and 80% the next.

Phase 4's "shipped result" of 79.6% multi-seed mean at 180° (across the 2 successful training-seed Stage 3 ckpts) is still valid — those are evaluated checkpoints, eval-time noise is a different question and is well-bounded (±5pp per seed via the 50-ep argmax protocol).

### 4. Phase 4.5 attribution claims are partially weakened

| Phase 4.5 claim | Standing |
|---|---|
| **A (no LVLH): 13% peak vs 77%** | Likely still right — average over many runs would still show LVLH helps — but 13% as a single point is not significantly different from a bad bootstrap with LVLH (sanity #1 hit 5%). Strong claim of "load-bearing" needs multi-seed re-test. |
| **B (no shaping): 3.6% peak** | Same caveat. 3.6% is below the worst sanity-with-shaping result (5%), so direction is right, but margin is small. |
| **D (DAPO under curriculum): rolling +5pp, eval -7pp** | DAPO's run was warm-started from Phase 4 Stage 2 ckpt — not affected by Stage 1 bootstrap variance. Claim stands. |
| **C (D7→D10 surgery: 6% smoke)** | Smoke test is eval (deterministic), not training. Claim stands. |

The headline finding — "LVLH and shaping are required, DAPO is inert, surgery doesn't transfer" — is probably still correct in direction, but the standalone-seed effect-sizes I quoted should be treated as point estimates with wide error bars.

---

## Options to proceed

| Option | Compute | Pros | Cons |
|---|---|---|---|
| **1. Multi-seed (3 seeds) for every A & B run** | ~15-18 hr | Statistically robust; matches the spec's intent | 3× compute |
| **2. Switch vec backend to Serial** | ~50+ hr | Bit-deterministic; clearest interpretation | ~10× slower SPS; not how Phase 4 was run, so backwards-comparison is shaky |
| **3. Accept noise, repeat 3× same-seed, report median** | ~3× compute | Same total cost as option 1, simpler analysis | Still doesn't resolve "is variance from seed or scheduler?" |
| **4. Tighten thresholds (≥10pp gaps), single seed, document variance** | 1× compute (as specced) | Cheap; preserves spec timeline | Many comparisons will land "within noise"; under-informative |

**My recommendation:** **option 1 + relaxed thresholds (option 4 modifications).** Run each Phase 5a A and B candidate at 3 seeds, require ≥10pp mean delta to call a winner. Investigation C is already multi-seed. Investigation D is already done. Total ~12-15 hr.

This treats Phase 5a as the noise-aware sibling of Phase 4.5: bigger error bars, fewer false-confident conclusions, but real data feeding into the Phase 5 main spec.

---

## Investigation D (already complete)

Apoapsis-bias check on Phase 4 ckpt at e=0.2 yielded 1/200 successes (Phase 4 ckpt was trained at e=0; failure expected). The single success was at target true anomaly = 140°, which is "near apoapsis" by the ±π/4 window — but N=1 is anecdotal. **Status: untestable with current Phase 4 ckpt; defer the apoapsis-bias question to Phase 5 main as a sanity check after some training at e>0.**

Output: `plots/p5a_D_apoapsis_hist.png`, `logs/orbital/p5a_D_apoapsis_eval/`, `scripts/orbital/p5a_apoapsis_bias.py`.

---

## What's saved on disk

- `experiments/puffer_orbital_177738218703/` — Phase 5a A shared Stage 1 (1% perf, kept for now)
- `experiments/puffer_orbital_<sanity1_id>/` — sanity #1 at e=0 (kept; ~5% perf)
- `logs/orbital/p5a_D_apoapsis_eval/` — Investigation D eval traces (200 eps)
- `plots/p5a_D_apoapsis_hist.png` — apoapsis histogram (essentially empty due to 1 success)
- `logs/orbital/p5a_A_hypotheses_preregistered.md` — hypotheses recorded before any A results
- `scripts/orbital/p5a_apoapsis_bias.py` — analysis script

No commits yet; working tree has new files only (no edits to env/config code).

---

*Halted 2026-04-28 mid-execution to surface the determinism issue. Resume after option selection.*
