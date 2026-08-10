# Phase 5a Addendum — Variance-Aware Protocol Revision

> **Status:** 2026-04-28. Patch to `orbital-rl-phase5a-spec.md` after the determinism finding (PHASE5a_DETERMINISM_FINDING.md). Updates the run protocol for Investigations A, B, C; folds Phase 4.5 retesting into Investigation A; revises decision rules to use variance-aware thresholds; documents what changed and why.

---

## 0. What changed and why

The Phase 5a spec assumed single-seed first-pass comparisons were sufficient because Phase 4 had ~11pp eval variance and ablation effects were expected to be ≥15pp. That assumption is invalidated by the determinism finding: under PufferLib's Multiprocessing vec backend, *training* (not eval) has ~70pp variance at fixed seed across runs of identical config. The variance is from OS scheduling of vec workers affecting PPO minibatch ordering; this is a documented behavior of multi-process PPO across implementations (Stable-Baselines3 #369, OpenAI Baselines, IsaacGym), not a PufferLib-specific bug.

This is consistent with the "Edge of the Statistical Precipice" (Agarwal et al. 2021) finding that fixed-seed reproducibility doesn't address whether an algorithm performs well under similar conditions; the variance *is* the algorithm's behavior. The standard remediation (Henderson 2018; Agarwal 2021) is multi-seed protocol with explicit variance estimation.

The addendum below adopts that standard, calibrated to our observed ~70pp variance floor.

---

## 1. Seed count: 5 minimum, 10 if borderline

The original spec used 1 seed for A and B, 3 for C. Revised:

- **Core comparisons (A, B, C): 5 seeds minimum.**
- **Borderline outcomes: extend to 10 seeds for the borderline pair only, not blanket.**
- **Investigation D: unchanged** — already deferred to Phase 5 main.

5 seeds is the field reference (Henderson 2018 et seq.) and gives 95% CI of ±35pp width at 70pp variance — sufficient to detect 15pp effects but not smaller ones. 10 seeds shrinks CI to ±25pp width and is the right escalation when 5 leaves a comparison ambiguous. Going beyond 10 has diminishing marginal value relative to compute cost.

The seeds: `42, 1337, 20260423, 2718, 31415`. (Adding two more if needed: `1729, 6022`.)

---

## 2. Investigation A: expanded candidate set (folds Phase 4.5 retests in)

The original A had four candidates: square, tall, wide, fresh-at-target. Revised to **six candidates**, adding Phase 4.5's A and B ablations as natural Phase 5a tests.

| Pattern | Stage 1 | Stage 2 | What's tested |
|---|---|---|---|
| **Square** | (π/6, 0.05) | (π/2, 0.10) | Joint annealing both axes |
| **Tall** (phase-first) | (π/6, 0.05) | (π/2, 0.05) | Phase axis first |
| **Wide** (eccentricity-first) | (π/6, 0.05) | (π/6, 0.10) | Eccentricity axis first |
| **Fresh-at-target** (control) | (π, 0.30) from scratch | — | DORAEMON null hypothesis |
| **Square minus LVLH** | as Square | as Square | Re-tests Phase 4.5 Ablation A under joint-randomized curriculum |
| **Square minus shaping** | as Square | as Square | Re-tests Phase 4.5 Ablation B under joint-randomized curriculum |

**Why fold P4.5 retests into A.** Phase 4.5's A and B claims are based on single-run point estimates; under the new variance floor, those estimates have wide error bars. Retesting them as standalone experiments would require running at Phase 4's conditions (e=0, point-curriculum), which doesn't tell us about Phase 5's recipe. By running no-LVLH and no-shaping under the Phase 5a curriculum candidate (square), we get:
- A multi-seed Phase 4.5 attribution claim
- Direct evidence about whether LVLH/shaping matter under joint-randomized curriculum, which is the forward-relevant question

The "minus LVLH" and "minus shaping" variants pair with square specifically (rather than all three patterns) because square is the simplest pattern to ablate cleanly. If square wins the curriculum-ordering comparison, the ablation results inform Phase 5 main directly. If a different pattern wins, we know whether LVLH and shaping help under at least one viable curriculum, which is enough.

### 2.1 Compute budget for Investigation A

Six candidates × 5 seeds × ~30 min each = ~15 hours of compute. Fits in a long day.

If a candidate's first-3-seed mean is clearly below the worst-other-candidate's first-3-seed mean (gap > 25pp), halt that candidate at 3 seeds and do not run seeds 4-5. This is permissible because the goal is *winner identification*, not *statistical certainty for losers*. Saves compute on clear losers.

### 2.2 Decision rule for A

**Replace "single seed, ≥10pp delta" with variance-aware thresholding.**

For each pair of candidates, compute mean and std of Stage 2 final success rate across the 5 seeds. Declare a winner if:

```
mean(A) - mean(B) > 1.5 × pooled_std(A, B)
```

Where `pooled_std` is the standard pooled estimate `sqrt((std_A² + std_B²) / 2)`. The 1.5× multiplier corresponds roughly to a one-sided p<0.10 test at n=5; tighter than nothing, looser than 95%-CI, appropriate for first-pass decisions.

If multiple candidates are tied within this threshold, pick the simpler curriculum (square > tall ≈ wide > fresh-at-target), and document the tie in the writeup.

If *no* candidate passes the threshold against fresh-at-target, that's a meta-finding: either the variance is too high to distinguish curriculum effects, or curriculum doesn't help at this difficulty level. Either way, halt Phase 5 main and re-think before drafting it.

### 2.3 Pre-registered hypotheses (revised from original spec)

Original H1-H3 still apply. Adding:

- **H4 — minus-LVLH significantly underperforms square.** Phase 4.5's directional finding holds under multi-seed protocol.
- **H5 — minus-shaping significantly underperforms square.** Same.
- **H6 — minus-LVLH and minus-shaping have similar performance to each other.** Both ablations remove signals that help bootstrap; symmetric removal predicts symmetric loss.

If H4 or H5 fail, the Phase 4.5 attribution was based on noise, and the recipe simplifies (we can drop LVLH or shaping for Phase 5).

---

## 3. Investigation B: dt revisit, multi-seed

The original B had two arms (DT=60, DT=30). Revised: same two arms, but **5 seeds each** under the winning curriculum from Investigation A.

### 3.1 Compute budget for B

Two arms × 5 seeds × ~30 min for DT=60, ~60 min for DT=30 (doubled steps to match real-time horizon) = roughly 7-8 hours.

**Sequencing.** B runs after A, because it uses A's winning curriculum pattern. So B's compute starts only after the A decision lands.

### 3.2 Decision rule for B (revised)

Same variance-aware thresholding as A:

```
mean(B-30) - mean(B-60) > 1.5 × pooled_std(B-30, B-60)
```

If B-30 wins by this threshold: switch Phase 5 main to DT=30, accept the doubled compute cost.

If within threshold (no clear winner): stay at DT=60, document negative result. The Phase 3 DT=30 regression remains *plausibly* explained by hyperparameter mismatch, but we don't have positive evidence for DT=30 helping. Default to simpler.

If B-30 regresses by threshold: definitively rule out DT=30 with retuned hyperparameters. Stop pursuing dt as a lever.

---

## 4. Investigation C: Stage 1 reliability, expanded

The original C ran 3 seeds of Stage 1 only. Revised: **5 seeds, plus same-seed re-runs to estimate within-seed variance**.

### 4.1 What the expanded protocol measures

The determinism finding raised a question we don't yet have data on: how much of the Phase 4 inter-seed variance was actually intra-seed (scheduler) variance? Concrete approach:

- 5 seeds × 1 run each (inter-seed variance estimate)
- Best-performing seed × 3 additional runs (intra-seed variance estimate at one seed)

Total: 8 runs of Stage 1 only, ~5 min each on the winning curriculum's small initial bounds. ~40 min compute.

### 4.2 What this tells us

Compare `std(across 5 different seeds)` to `std(across 4 runs of best seed)`:

- **If std(seeds) ≈ std(same-seed):** seeds are not meaningfully different from re-rolls. Phase 5 main should treat all training-time multi-seed protocols as multi-roll-of-same-distribution, and pre-registered seed selection adds no value.
- **If std(seeds) > std(same-seed):** seeds differ meaningfully even at the noise floor. Multi-seed protocols are doing real work. Standard practice continues.
- **If std(seeds) < std(same-seed):** unexpected; would suggest seeds are *more* consistent than scheduler noise, which doesn't match the determinism finding. Investigate.

This is a small experiment with high information value about how to interpret all subsequent multi-seed results. ~40 min compute is cheap relative to that information.

### 4.3 Decision rule for C

| Best-seed Stage 1 mean across 4 runs | Verdict |
|---|---|
| ≥ 70% | Curriculum bootstrap is robust at chosen pattern. Phase 5 main proceeds with normal compute budget. |
| 40-70% | Curriculum bootstrap is variable. Phase 5 main needs explicit retry budget (run each stage 2-3× and pick best). |
| < 40% | Curriculum bootstrap is unreliable. Investigate before Phase 5 main — possibly try smaller initial bounds, or different curriculum pattern. |

---

## 5. Total revised compute budget

| Investigation | Original (single-seed) | Revised (5-seed) |
|---|---|---|
| A — curriculum order (6 candidates) | 1.5 hr | ~15 hr (with early-halt on clear losers, possibly 12 hr) |
| B — dt comparison | 3 hr | 7-8 hr |
| C — Stage 1 reliability | 10 min | 40 min |
| D — apoapsis bias | 5 min (deferred) | deferred to Phase 5 main |
| **Total** | **~5 hr** | **~22 hr** |

Realistically a 2-3 day commitment, can be parallelized somewhat (A's seeds can run in parallel if compute allows; B can prep while A finishes). Within scope.

---

## 6. What this means for Phase 5 main

The variance finding has three forward implications beyond Phase 5a:

### 6.1 Multi-seed throughout

Phase 5 main should adopt 5-seed minimum for every condition tested, with 10-seed escalation for borderline comparisons. This is non-negotiable going forward.

### 6.2 Eval-during-training is now first-class

With high variance, end-of-training point estimates are noisy. *Trajectories* are richer signal. Phase 5 main should:

- Eval every 500k steps during training (~20 evals per stage)
- Plot per-seed success-rate curves, not just final-checkpoint numbers
- Detect collapses early (e.g., if entropy or success rate trending wrong way for 2M steps, halt and diagnose)

This requires a small infrastructure change (log eval every 500k, not just at checkpoint boundaries) but no new code.

### 6.3 Adaptive curriculum stages, not time-boxed ones

The DORAEMON paper's empirical finding — gradually expanding distribution outperforms training at full distribution — is well-established. The mechanism it implies for Phase 5 main: **stage transitions should be gated on success rate, not on step count**. With high training variance, a fixed "10M steps then move on" rule produces wildly different starting points for the next stage depending on which run we're on. A "70% on current bounds → expand" rule produces consistent starting conditions across runs.

This is a more invasive change for Phase 5 main than the others. Worth doing because it directly addresses the variance issue at its mechanistic level. Phase 5a doesn't need to test this — the literature already has — but Phase 5 main should adopt it.

---

## 7. What's NOT changing

For clarity:

- **Investigation D (apoapsis bias)**: deferred to Phase 5 main, unchanged.
- **Phase 4.5 attribution claims for C and D**: not retested. C was a deterministic eval (smoke test on a checkpoint), D was warm-started from a fixed checkpoint with deterministic eval. Both stand at original confidence.
- **Phase 4 final result (79.6% multi-seed mean on Stage 3 checkpoints)**: unchanged. Checkpoint eval is deterministic and the protocol was multi-seed at that level.
- **The recipe itself (Discrete(10) + LVLH + gated shaping + Phase 4 hyperparameters)**: unchanged for Phase 5a entry conditions. Phase 5a may *find* that LVLH or shaping doesn't help (H4/H5), in which case Phase 5 main updates the recipe — but Phase 5a doesn't pre-emptively change it.

---

## 8. Updated sequencing within Phase 5a

1. **Investigation A** (~15 hr) — six candidates × 5 seeds, Stages 1-2. Run first because B depends on A's winner.
2. **Investigation C** (~40 min) — runs after A's winner is identified, uses that pattern's Stage 1 only.
3. **Investigation B** (~7-8 hr) — runs after A's winner, uses that pattern.
4. **Investigation D** — deferred to Phase 5 main.

A→C→B sequence rather than A→B→C because C is cheap, gives us variance information that can refine B's protocol if needed, and doesn't require B's results.

---

## 9. Output: revised deliverable

The Phase 5a writeup gains three sections beyond the original spec's four:

5. **Variance characterization** — measured intra-seed and inter-seed variance estimates. Used to inform Phase 5 main confidence intervals.
6. **Phase 4.5 retests** — multi-seed results for no-LVLH and no-shaping conditions. Updates Phase 4.5 attribution claims with proper error bars.
7. **Adaptive curriculum recommendation** — based on Investigation A results, whether to use time-boxed stages (current Phase 4 pattern) or success-gated stages (DORAEMON-style) in Phase 5 main.

This is more substantive than the original spec's "decision-support" framing. Phase 5a now produces a methodological foundation as well as the four decisions, which is appropriate for a project that has discovered its variance floor mid-execution.

---

## 10. What this addendum is *not*

- **Not a wholesale spec rewrite.** The original spec's four investigations and their hypotheses largely stand. The addendum patches protocol (single-seed → multi-seed) and decision rules (fixed thresholds → variance-aware), and folds Phase 4.5 retesting into Investigation A.
- **Not a retrospective rewrite of Phase 4 or 4.5 docs.** PHASE4_FINDINGS and PHASE4_5_FINDINGS stand as historical record. A separate "epistemic caveat note" will append to PHASE4_5_FINDINGS once Phase 5a results land, updating which claims got confirmed and which got revised. That note is bookkeeping, not protocol.
- **Not a switch to Serial backend.** The variance floor is a property of how PPO + Multiprocessing actually behaves; running Serial would produce a different algorithm whose results don't compare to anything else we've run. Multi-seed protocol is the correct remediation.

---

*Author: 2026-04-28. Revises orbital-rl-phase5a-spec.md to multi-seed protocol after determinism finding. Successor: Phase 5a results writeup, then Phase 5 main spec.*
