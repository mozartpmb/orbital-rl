# Phase 5a — Interim Findings (Multi-Seed Sweep + Entry-Condition Probe)

**Date:** 2026-04-28
**Status:** Investigation A halted mid-sweep after canonical Stage 1 + fresh-at-target sweeps. Followed by an 8-run entry-condition probe. Both inform a revised Phase 5 main strategy. Investigations B and C pending entry-condition resolution.

---

## TL;DR

- The Phase 5a entry condition `(phase_gap_max=π/6, e_max_target=0.05)` is **not bootstrappable from random init** — 5 seeds collapse uniformly to ~1.4% perf.
- **e>0 from random init is unsolvable under the current recipe** (chaser hard-pinned to circular orbit). Even (π/12, 0.02) fresh peaks 3.4%.
- **e=0 still works** as a fresh start: (π/6, 0) reaches 80% in one run; phase-gap > π/6 also fails fresh.
- **Warm-start from e=0 → e>0 transiently succeeds** (peak 88.9% at e=0.05, even 100% at e=0.025) but **collapses to 0% by end of training** — same R5-style collapse pattern.
- **The bottleneck is now checkpoint preservation, not curriculum design.** The viable policy state exists during training; we just don't capture it.
- The determinism-finding's variance scare (~70pp at fixed seed) **only manifests when learning is actively succeeding**. At infeasible difficulty, all runs collapse to similar low scores (std 0.5pp at mean 1.4%). This is a useful methodological caveat for future variance estimation.

---

## Investigation A — partial sweep results

### Canonical recipe Stage 1 — 5 seeds × 10M at (π/6, 0.05)

| Seed | Peak | Final |
|---|---|---|
| 42 | 1.1% | ~0% |
| 1337 | 0.8% | ~0% |
| 20260423 | 1.5% | ~0% |
| 2718 | 1.6% | ~0% |
| 31415 | 2.2% | ~0% |
| **mean ± std** | **1.4 ± 0.5 pp** | — |

### Fresh-at-target — 5 seeds × 10M at (π, 0.30)

| Seed | Peak |
|---|---|
| 42 | 0.2% |
| 1337 | 0.3% |
| 20260423 | 0.5% |
| 2718 | 0.3% |
| 31415 | 1.2% |
| **mean ± std** | **0.5 ± 0.4 pp** |

H2 (DORAEMON null hypothesis) confirmed.

### What I halted before

Stage 2 forks (square/tall/wide × 5 seeds) and minus-LVLH/minus-shaping × 5 seeds Stage 1+2 sweeps. All would have warm-started from these failed Stage 1 ckpts → ~3 hr predicted-doomed compute. Halted before that to investigate the entry-condition problem first.

---

## Entry-condition probe (8 runs, 1 seed = 42 each)

| Probe | Setting | Peak | Final | Verdict |
|---|---|---|---|---|
| A1 | (π/6, 0.0) fresh, 10M | **84.6%** | 80.2% | ✅ recipe baseline still works |
| A2 | (π/6, 0.05) warm from A1, 10M | **88.9%** | 0.0% | ⚠️ peak feasible, collapses |
| B1 | (π/6, 0.02) fresh, 10M | 3.4% | 0.2% | ❌ |
| B2 | (π/12, 0.05) fresh, 10M | 1.9% | 0.1% | ❌ |
| B3 | (π/12, 0.02) fresh, 10M | 3.4% | 0.3% | ❌ |
| C | (π/2, 0.0) fresh, 10M | 5.6% | 0.0% | ❌ — **even π/2 e=0 fresh fails!** |
| E1 | (π/6, 0.025) warm from A1, 5M | **100%** | 0.2% | ⚠️ peak even higher, same collapse |
| E2 | (π/6, 0.05) warm from E1-collapsed, 5M | 0.5% | 0.1% | ❌ chained from collapsed |

### Three load-bearing observations

**1. Random-init bootstrap is narrow.**
Only `(small phase, e=0)` works fresh. (π/6, 0) → 84% peak; (π/2, 0) → 5.6% peak fresh. Phase 4's curriculum was load-bearing for *phase gap*, not just for warm-starting at higher difficulty. This was implicit in Phase 4 but the C probe makes it explicit.

**2. e>0 from random init is impossible under current recipe.**
Even (π/12, 0.02) caps at 3.4% peak. The mechanism is that the chaser starts on a circular orbit (`sat.orbit.e = 0` hard-coded at `orbital.h:758`). With e_target = 0.02, the target's velocity at apoapsis vs perigee differs by ~300 m/s (vs `REL_VEL_TOL = 50 m/s`). Random-policy episodes terminate with the chaser nowhere near a feasible relative-velocity state, so the terminal +10 reward is never sampled.

**3. e=0 → e>0 warm-start transiently solves it, then collapses.**
Probe A2 hit **88.9%** at e=0.05; probe E1 hit **100%** at e=0.025. Both are higher than the e=0 baseline. The Phase 4 e=0 policy DOES generalize to e>0 — for a few hundred-thousand steps. Then PPO's continued training on the new reward landscape destabilizes the value head and entropy crashes to 0. Same R5-pattern as Phase 4.5's warm-start ablations.

The peak ckpt is the policy we want. We just don't have it — `checkpoint_interval = 200` epochs (~13M steps) means we only get the end-of-training ckpt.

---

## Variance update vs the determinism finding

PHASE5a_DETERMINISM_FINDING claimed ~70pp variance at fixed seed across runs. That variance is real *when learning is feasible*. But:

- At (π/6, 0.05) where learning fails: std = 0.5pp across 5 seeds.
- At (π, 0.30) where learning fails: std = 0.4pp.

**Variance manifests only when the learning signal is non-zero.** This is methodologically useful: high variance is a *progress* signature, not a *failure* signature. If a candidate's first 3 seeds all land below 5% with sub-2pp std, declaring it a non-starter at that point is justified. Conversely, two single-seed runs at 80% and 5% might still be "the same algorithm," because variance widens with progress.

---

## Phase 5 main: revised entry strategy

The original Phase 5a spec / addendum's Investigation A candidates all started from `(π/6, 0.05)` fresh. That's now ruled out as an entry condition. A workable Phase 5 curriculum looks like this:

```
Stage 0 (e=0 phase ramp, Phase 4 territory):
  (π/6, 0)     — fresh, 10M   [validated: 80% peak]
  (π/2, 0)     — warm, 15M    [Phase 4 reaches 78%]
  (π,   0)     — warm, 15M    [Phase 4 reaches 81% multi-seed]

Stage 1 (e ramp, new):
  (π,   0.02)  — warm, ~5M    [needs early-eval-and-keep-best]
  (π,   0.05)  — warm, ~5M
  (π,   0.10)  — warm, ~5M
  (π,   0.20)  — warm, ~5M
  (π,   0.30)  — warm, ~5M
```

This effectively prepends the full Phase 4 curriculum as a prerequisite, and adds an eccentricity ramp afterward. The order (phase first, eccentricity second) is forced by the probe data — phase ramp is bootstrappable from random, eccentricity ramp is not.

But this curriculum needs the **collapse mitigation** before it's runnable end-to-end. Two options, in priority order:

### Mitigation 1: tight checkpoint interval + best-ckpt selection

Lower `checkpoint_interval` from 200 epochs (~13M steps) to **10 epochs (~650k steps)**, eval each ckpt during training, keep the best one. The early-peak ckpt (88-100%) becomes the warm-start for the next stage.

Estimated cost: 20× the disk and ~5% wall-time overhead per training run. Cheap relative to the savings.

This is the addendum's §6.2 "eval-during-training" recommendation made concrete.

### Mitigation 2: shorter Stage 2 runs

Probe E1 hit 100% peak somewhere in 5M; probe A2 hit 88.9% peak somewhere in 10M. If we cut each e-ramp stage to 2-3M steps and capture the end-of-training ckpt at the right moment, we sidestep the collapse without infrastructure changes.

Risk: still relies on the right step count being chosen pre-run rather than discovered. Probably the worse mitigation.

### Mitigation 3 (out of scope): policy stabilization

Add KL-regularization or auxiliary loss to prevent the value head from drifting. Phase 4.5 already established that R3-stack components don't fix this (DAPO regressed at eval, others worse). Skip.

---

## Recommendation

Before resuming Investigation A, **adopt Mitigation 1 (tight checkpoint interval) into the recipe**, then re-run a single warm-start ramp probe to confirm the best-ckpt-capture works:

```
A1 ckpt → tight-checkpoint training at (π/6, 0.05) for 10M
   → eval each saved ckpt at (π/6, 0.05) post-hoc
   → expect early-peak ckpt achieves ≥80%
```

If this confirms, the new Phase 5 main spec uses tight-checkpoint-with-best-selection across all stages. Investigation A's revised candidates then start from the validated Stage 0 endpoint (Phase 4 curriculum at e=0, 180° ckpt) and ramp eccentricity from there.

Estimated compute:
- Tight-checkpoint A2 redo: ~6 min
- Re-running revised Investigation A's 6 candidates × 5 seeds at the new entry: ~3-4 hr
- Investigation B (DT comparison) at the new entry × 5 seeds: ~6-8 hr
- Investigation C: now mostly validated by the entry-condition probe; ~30 min for confirmation seeds

Total: ~10-12 hr to land a multi-seed Phase 5a deliverable, with infrastructure for Phase 5 main.

---

## Files saved on disk

### From sweep:
- `pufferlib/experiments/puffer_orbital_17773843{1094,2999,3870,5130,6386}/` — canonical Stage 1 ckpts (failed)
- `pufferlib/experiments/puffer_orbital_17773848{7656,9011}/` — fresh-at-target seeds 42, 1337
- `pufferlib/experiments/puffer_orbital_177738510057/` etc. — remaining fresh seeds
- `/tmp/p5a_sweep.log` — sweep timeline

### From probe:
- `pufferlib/experiments/puffer_orbital_177738621627/model_puffer_orbital_000077.pt` — **A1 (e=0 baseline, 80%)** ⭐ keeper
- `pufferlib/experiments/puffer_orbital_177738633600/` — A2 (e=0.05 collapsed)
- `pufferlib/experiments/puffer_orbital_177738643911/` — B1
- `pufferlib/experiments/puffer_orbital_177738656077/` — B2
- `pufferlib/experiments/puffer_orbital_177738667340/` — B3
- `pufferlib/experiments/puffer_orbital_177738678411/` — C
- `pufferlib/experiments/puffer_orbital_177738695961/` — E1 (e=0.025 collapsed, peak 100%)
- `pufferlib/experiments/puffer_orbital_177738702181/` — E2 (collapsed-chained)
- `/tmp/p5a_probe.log` — probe timeline

### Code state
Unchanged. Sweep was halted before any minus-LVLH/minus-shaping rebuild step. `git diff HEAD --stat` is empty.

---

## Status

| Investigation | Status |
|---|---|
| D — apoapsis bias | Deferred to Phase 5 main (Phase 4 ckpt was untestable at e=0.2) |
| A — curriculum order | **Halted, entry condition revised, awaiting mitigation** |
| B — dt comparison | Pending A's revised entry |
| C — Stage 1 reliability | Largely answered by the probe — see #1 above |

Next action: pick Mitigation 1 vs alternative, validate with A2-redo at tight checkpoint, then resume Investigation A from the validated entry condition.

---

*Halted 2026-04-28 after sweep + entry-condition probe. ~50 min total compute spent. Revealed two structural issues worth resolving before Phase 5 main: (a) e>0 random-init is unsolvable, requires Phase 4 curriculum prepend; (b) PPO training collapses on warm-start to wider-distribution, requires checkpoint-preservation infrastructure.*
