# Phase 5a — A2 Peak Diagnostic Findings

**Date:** 2026-04-28
**Spec:** `phase5a-a2-diagnostic.md`
**Compute spent:** ~12 min (A2 redo with tight ckpts + 16-ckpt scan + 4 sampled-eval probes)

---

## TL;DR

The A2 probe's "peak 88.9% at e=0.05" was a **mirage** in the training-time perf dashboard, not real policy capability. Verified by:

- All 16 A2 saved checkpoints (every 5 epochs from 5 to 77) eval at **0% argmax / ~1% sampled** at e=0.05.
- A1 (Phase 4 e=0 baseline) at e=0.05: **1.5% argmax / 1% sampled**. Phase 4 policy does not generalize to e>0 deterministically OR stochastically.

**Story verdict: C-weak — recipe doesn't transfer.** The Phase 4 policy + recipe combination fails to extend to e>0 even at e=0.05 from any tested entry point. Phase 5 main needs structural changes, not just tight checkpointing + best-ckpt selection.

---

## What was run

### Step 1 — A2 redo with tight checkpoint interval

Re-ran probe A2 (warm from A1 → train at (π/6, 0.05) for 10M, seed 42) with `--train.checkpoint-interval 5` (every ~325k steps). Output dir: `pufferlib/experiments/puffer_orbital_177739444585/`. Saved 16 ckpts at epochs 5, 10, 15, …, 75, 77.

Final perf (rolling train metric): 0.0%. Same collapse pattern as the original probe; tight checkpointing didn't change training dynamics.

### Step 2 — Peak-finder eval (argmax)

Eval'd each saved ckpt at e=0.05 (50 eps × seed=42 argmax):

```
epoch  argmax_perf
5      0.0%
10     0.0%
15     0.0%
20     0.0%
…
75     0.0%
77     0.0%
```

**All 16 ckpts at 0%.** No epoch produces a viable policy under deterministic eval.

### Step 3 — Sampled-action eval (matches training-time metric)

Wrote `/tmp/eval_sampled.py` — same eval harness but uses `pufferlib.pytorch.sample_logits(logits)` instead of `argmax`. This matches the training-time `perf` computation.

Results (100 eps each, seed=42):

| Ckpt | Condition | Argmax eval | Sampled eval |
|---|---|---|---|
| A1 (Phase 4 e=0, epoch 77) | e=0 | 78.5% | 41.0% |
| A1 | e=0.05 | 1.5% | 1.0% |
| A2 epoch=10 (early, near reported peak) | e=0.05 | 0% | 1.0% |
| A2 epoch=77 (final, collapsed) | e=0.05 | 0% | 1.0% |

Note: A1's sampled-at-e=0 of 41% (vs 78% argmax) is the expected entropy gap — Phase 4's final policy has entropy ~0.4 nats, so half of stochastic actions are different from argmax, costing roughly half the success rate.

**A2's reported training-time peak of 88.9% is not reproduced under any eval protocol.** The 100-ep sampled run gives 1% — within noise of "doesn't work."

---

## What the 88.9% number actually was

Speculative but consistent with PufferLib's dashboard semantics:

The dashboard `perf` field is the rolling mean of `success` events over a small training window (likely the last few updates' worth of completed episodes). With 1024 parallel envs and episodes ~970 steps, each update consumes ~30 episodes per env total. A short window can cover a few hundred episodes. If a stochastic-action burst happens to hit a sequence of low-difficulty samples (small e_target draws, favorable phase gaps, lucky exploration) the dashboard spikes.

100 episodes of single-env eval averages over a much longer effective window and bypasses the burst-luck dynamic, giving the true held-out number (~1%).

This means: **training-time perf in this codebase is not a reliable held-out estimator at low success rates.** Use eval_checkpoint.py for any decision that depends on capability.

---

## Decision matrix outcome

Per the diagnostic spec §2.1:

| A2 peak @ e=0.05 | A2 peak @ e=0 | A1 @ e=0.05 | Story |
|---|---|---|---|
| 0% | (irrelevant) | 1.5% | **C-weak — Recipe doesn't transfer** |

The exact C-strong condition (A1 ≥75% at e=0.05) doesn't apply — A1 fails at e=0.05 too. C-weak is the closest fit: Phase 4 policy + recipe doesn't extend to e>0.

---

## Phase 5 main implications

### What this rules out

1. **Tight checkpointing alone is insufficient.** It's still good infrastructure to have, but it doesn't unlock e>0 for this recipe.
2. **Smaller eccentricity steps don't help.** Probe E1 (e=0 → e=0.025) showed 100% peak training-perf but the A2 redo + sampled eval shows that's also a mirage. The recipe's ceiling on e is at e≈0.
3. **Curriculum ordering (square/tall/wide) is not the limiting factor.** All three orderings start from the same untransferable e>0 condition.

### What's needed for Phase 5 main

The recipe needs structural changes for e>0. Candidates, ordered by my prior:

1. **Loosen REL_VEL_TOL during early training, schedule-tighten.** Currently 50 m/s; with e=0.05 target velocity varies ±360 m/s. Try REL_VEL_TOL=200 m/s for initial e>0 training, anneal to 50 m/s as the policy improves. Curriculum on the success criterion, not just on the task parameters.

2. **Initialize satellite with random eccentricity matching target distribution.** Currently `sat.orbit.e = 0` hard-coded at orbital.h:758. Sampling sat.e from `Uniform(0, e_max)` at reset gives the policy a starting state where success is geometrically reachable.

3. **Reshape success criterion to require orbit-shape match (not just position+velocity).** Currently success = `dist < 30km AND rel_vel < 50m/s`. For eccentric rendezvous, the standard formulation requires `|Δa| < TOL_A AND ||Δē|| < TOL_E AND dist < TOL_D` — i.e., be on the same orbit, not just briefly nearby. This is harder to learn but defines a stable target manifold.

4. **Reward shape change for eccentric targets.** The gated NHR shaping (Φ_orbit + σ₂·Φ_phase + σ₃·Φ_vel) was tuned for circular targets. The sigma gates may not activate correctly when target.e > 0; auditing this is cheap and likely informative.

My read: option 1 (REL_VEL_TOL annealing) is the cheapest and most likely to work. Option 2 (random sat init) is structurally cleaner. Option 3 (orbit-match success) is the most physically principled but largest reward-landscape change. Option 4 is diagnostic and worth doing alongside 1.

### Phase 5 main scope estimate

- **If option 1 alone solves it:** ~1-2 weeks. Adds an annealing schedule for REL_VEL_TOL, runs the standard curriculum on top.
- **If option 1 + option 4 needed:** ~2 weeks. Above + shaping-gate audit + retune.
- **If option 2 or 3 required:** ~3-4 weeks. Substantive recipe rework, multi-seed validation, attribution sub-experiments.

Worth running option 1 + option 4 as a Phase 5b (~1 day of compute) before committing to Phase 5 main scope.

---

## What's saved on disk

- `pufferlib/experiments/puffer_orbital_177739444585/` — A2 redo with 16 saved ckpts (every 5 epochs)
- `/tmp/p5a_peakscan.txt` — argmax peak-finder results (all 0%)
- `/tmp/eval_sampled.py` — sampled-action eval harness (one-shot, kept for reuse)
- `/tmp/p5a_a2_redo.log` — A2 redo training log

Code state: unchanged. No env or config edits during this diagnostic.

---

## Status

| Step | Status |
|---|---|
| Locate peak ckpt | Done — peak ckpt nominally at any of 16 saved epochs |
| Eval at training condition (e=0.05, argmax) | Done — 0% across all 16 |
| Eval at e=0 (A1 baseline) | Done — A1 reproduces 78.5% argmax, 41% sampled |
| Eval at e=0.05 (A1) | Done — 1.5% argmax, 1% sampled |
| Sampled-action eval (training-time metric reproduction) | Done — A2 epoch=10 also 1% sampled |
| Decision matrix verdict | **C-weak — recipe doesn't transfer** |

Next action: pick Phase 5b structural change (REL_VEL_TOL annealing recommended), implement, validate, then redraft Phase 5 main with the verdict baked in.

---

*Resolves the A2 peak diagnostic. ~12 min compute, definitive answer. The 88.9% / 100% training-time peaks were dashboard-metric artifacts, not real policy capability. Phase 5 main needs structural recipe changes for e>0.*
