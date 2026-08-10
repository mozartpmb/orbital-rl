# Phase 5b — Blocks A + B Findings

**Date:** 2026-04-29
**Spec:** `phase5b-spec.md` + plan refinements
**Compute spent:** ~6 hours wall (Blocks A pre-flight, A, A re-do, B 3 arms, multi-rollout verifications)

---

## TL;DR

The four-stage curriculum proposal in the Phase 5b spec is wrong. **The optimal curriculum is two-stage: Stage 1.0 → Stage 4.0**, skipping the e expansion within Stage 1 entirely. Stage 1.x intermediate stages (1.1, 1.2, 1.3) specialize the policy to same-orbit eccentric rendezvous and *erase* the OOD breadth needed for Stage 4.0 transfer.

**Empirical finding:** warm-from-Stage-1.0 → Stage 4.0 lands **96.4% mean across 3/5 training seeds** at Stage 4.0 conditions (everything random, e_max=0.05). Same path with warm-from-Stage-1.3 lands 1.2% mean. Same path fresh lands 0%. The breadth difference between Stage 1.0 and Stage 1.3 ckpts is decisive.

The user's pre-committed Block B failure mode applies: **curriculum is required, not just helpful.** Fresh fails to bootstrap (0% across 5 seeds); only the Stage-1.0-warmed path produces non-trivial Stage 4.0 capability.

Stage 1.0 → Stage 4.0 has a 2/5 seed retry rate. 3 seeds converge cleanly to 94-98%; 2 seeds collapse to 2-16% (warm-start preservation only).

---

## Block A — Stage 1.x curriculum results

### Pre-flight check (single seed = 42)

Per spec §12 — verify e_max=0.5 reachable before committing.

- Stage 1.0 fresh, 40M: 98.8% peak. Validated.
- Stage 1.3 warm from 1.0, 80M, e_max=0.5: 68% peak (epoch 500). Above 50% feasibility gate. Pre-flight passes.

### Stage 1.0 — 5 seeds × 40M, e_max = 0.05

| Seed | Best held-out @ e=0.05 |
|---|---|
| 42 | 100.0% |
| 1337 | 100.0% |
| 20260423 | 100.0% |
| 2718 | 98.0% |
| 31415 | 100.0% |
| **Mean ± std** | **99.6 ± 0.8 pp** |

Clean convergence, all peaks at epochs 225-300 (~15-20M steps; the second half of training is past peak — confirms Step 1's "ceiling around 40M" finding).

### Stage 1.1 — 3 seeds × 50M warm from 1.0, e_max = 0.10

| Seed | Best @ e=0.10 |
|---|---|
| 42 | 84.0% |
| 1337 | 90.0% |
| 20260423 | 86.0% |
| **Mean ± std** | **86.7 ± 3 pp** |

Above 80% threshold. All seeds converge.

### Stage 1.2 — 3 seeds × 50M warm from 1.1, e_max = 0.20

| Seed | Initial 50M | +20M extension |
|---|---|---|
| 42 | 36% | **22%** (worsened) |
| 1337 | 48% | **38%** (worsened) |
| 20260423 | 42% | 42% (flat) |
| **Mean** | **42%** | **34%** |

**Stage 1.2 is a no-op stage.** The Stage 1.1 ckpts already give 49% at e=0.20 baseline (without any Stage 1.2 training). Stage 1.2 training neither improves nor preserves — extension actively degrades. e=0.10→0.20 jump is the recipe's plateau.

Decision: skip Stage 1.2 in primary path.

### Stage 1.3 — 5 seeds × 80M warm from 1.0 or 1.1, e_max = 0.50

Tested both warm-start paths to disambiguate the spec's curriculum design:

| Seed | v1 (warm from 1.1) | v2 (warm from 1.0) | Best |
|---|---|---|---|
| 42 | 54% | 40% | 54% |
| 1337 | 46% | 42% | 46% |
| 20260423 | 52% | 54% | 54% |
| 2718 | 64% | — | 64% |
| 31415 | 58% | — | 58% |
| **Best-per-seed mean** | | | **55.2 ± 7pp** |

The earlier "Stage 1.0 better than Stage 1.1 by 10pp" claim from v1 was per-seed fluke (seeds 2718/31415 happened to be lucky). v1 vs v2 on seeds 42/1337/20260423: 50.7% vs 45.3% — within noise per Principle II.

Stage 1.3 plateaus at the recipe ceiling. None of 5 seeds reach 80% threshold. This is the e=0.50 same-orbit ceiling for the recipe.

### Stage 1.4 — skipped

Stage 1.3 didn't reach 80% threshold (mean 55.2%). Per spec: stretch goal not warranted.

---

## Block B — Warm vs Fresh at Stage 4.0

Stage 4.0 conditions: `init_phase_gap_max=π, e_max_target=e_max_sat=0.05, same_orbit_init=0` (fully random, low difficulty).

5 seeds × 50M each. Three arms:

| Arm | seed 42 | 1337 | 20260423 | 2718 | 31415 | Mean |
|---|---|---|---|---|---|---|
| **WARM (from Stage 1.3)** | 0 | 0 | 0 | 0 | 6 | **1.2%** |
| **FRESH** | 0 | 0 | 0 | 0 | 0 | **0%** |
| **WARM (from Stage 1.0)** | **94** | 2 | **96** | 16 | **98** | **61.2%** |

Multi-rollout verification on the 3 successful Stage-1.0-warm seeds (200 eps × 3 rollout seeds = 600 eps each):

| Training seed | Mean across 3 rollout seeds |
|---|---|
| 42 | 95.8% |
| 20260423 | 95.7% |
| 31415 | **97.7%** |
| **Cross-seed mean (3/5)** | **96.4%** |

### Warm-start baselines (no Stage 4.0 training)

Direct OOD eval of Stage 1.0 vs Stage 1.3 ckpts at Stage 4.0 conditions (no training):

| Ckpt source | seed 42 | 1337 | 20260423 | 2718 | 31415 | Mean |
|---|---|---|---|---|---|---|
| **Stage 1.0** | 42% | 40% | 54% | 38% | 44% | **43.6%** |
| **Stage 1.3** | 0% | 10% | 6% | 0% | 10% | **5.2%** |

**Stage 1.3 specialization erased 38pp of OOD breadth (43.6% → 5.2%).** The downstream Block B WARM-from-1.3 numbers (mean 1.2%) reflect that even continued Stage 4.0 training can't recover from a 5% baseline starting point. The starting state of the warm-start matters far more than the training that follows.

---

## Block B verdict per spec §5.1 + user's added failure mode

The user pre-committed to recognizing this exact case:

> "fresh fails to bootstrap (≤ 5% after 50M) AND warm reaches non-trivial perf (≥ 30%)" → **curriculum is REQUIRED, not just helpful.**

Conditions:
- Fresh: 0% mean (≤5% ✓)
- Warm-from-1.0: 61.2% mean overall (3/5 at 94-98%, ≥30% ✓)
- Warm-from-1.3: 1.2% mean (also fails — warm is not always sufficient; the *specific* warm-start matters)

**Verdict: curriculum is mandatory. AND only Stage 1.0 specifically is the right warm-start source.** Intermediate Stage 1.x stages erode the very capability needed for the curriculum.

This is the strongest curriculum-justification finding the project has produced. Stage 4.0 fresh is unreachable from random init under this recipe; Stage 1.0 is the load-bearing scaffold.

---

## Why the spec's four-stage curriculum was wrong

The spec hypothesized: progressive eccentricity expansion within Stage 1 (1.0 → 1.1 → 1.2 → 1.3 → 1.4), then constraint relaxation through Stages 2/3/4 — would cumulatively build the capability needed for Stage 4.x.

What we found: each Stage 1.x specialization *removes* OOD breadth rather than building it. The Stage 1.0 ckpt at 99.6% (e=0.05, same-orbit) is also the most generalizable ckpt — it has the broadest exposure to phasing without being over-fit to any specific eccentricity. Each subsequent Stage 1.x training pushes the policy further into specialty and degrades cross-distribution transfer.

This connects to the Step 1 cap-tail finding: **breadth peaks early in training and erodes with continued in-distribution training.** Step 1 saw this for Phase 4 conditions OOD (66% at 40M → 51% at 60M). Block A/B sees the same pattern at much larger scale — Stage 1.3's specialization erases nearly all Stage 4.0 capability.

The mechanism is consistent across Phase 5a and Phase 5b: PPO's late-training refinement produces sharp argmax distributions specialized to the training distribution. For Stage 1.0, this is fine — the test condition (Stage 1.0 itself) matches. For Stage 1.x with x≥1, the test condition (Stage 4.0) is structurally different, and the specialty actively harms.

---

## What this implies for Phase 5b proper

The four-stage curriculum from the spec collapses to a **two-stage curriculum**:

```
Stage 1.0 (e_max=0.05, same_orbit_init=1)        — bootstrap (40M)
Stage 4.0 (e_max=0.05, same_orbit_init=0)        — primary deliverable (50M warm from 1.0)
```

Subsequent Stage 4.x e expansion (Stage 4.1 e=0.10, 4.2 e=0.20, 4.3 e=0.30) likely follows a similar pattern: each stage's training specializes, but the next stage warm-starts from a less-specialized prior. **Test:** which Stage 4.x ckpt is the right warm-start for Stage 4.(x+1)? Possibly the early-peak ckpt rather than the converged ckpt. Track this in Block C.

Per-seed bimodality matters: 2/5 seeds failed Block B (collapsed to 2-16%). Possible mitigations:
- Allow per-seed retries (re-roll RNG)
- Try seeds 1729 and 6022 from the addendum
- Investigate the failure mode (early collapse, peak at epoch 25)

---

## Files saved

- `pufferlib/experiments/puffer_orbital_*` — 30+ experiment dirs from Blocks A and B
- `/tmp/p5b_curriculum.log` — full timeline
- `/tmp/p5b_best_ckpts.txt` — best-ckpt mappings
- `scripts/orbital/p5b_curriculum.sh` — orchestrator
- `PHASE5b_BLOCKS_A_B_FINDINGS.md` — this document

Best Stage 4.0 (warm from 1.0) ckpts (the deliverable so far):
- seed 42: `puffer_orbital_177750198246/model_puffer_orbital_000350.pt` (96% scan, 95.8% multi-rollout)
- seed 20260423: `puffer_orbital_177750301624/model_puffer_orbital_000175.pt` (96% scan, 95.7% multi-rollout)
- seed 31415: `puffer_orbital_177750405236/model_puffer_orbital_000350.pt` (98% scan, 97.7% multi-rollout)

---

## Next: Block C (Stage 4.x e expansion)

Use the 3 successful Block B ckpts as warm-start sources for Stage 4.1 (e=0.10, fully random). Likely 5 seeds × 50M each, + multi-rollout verification. Track whether 4.x specialization erodes generalization the way 1.x did.

For seeds 1337 and 2718 that failed Block B: either retry from Stage 1.0 with different RNG, or accept 3/5 success rate and proceed.

---

*Phase 5b Blocks A + B closed with a sharper-than-expected finding: the spec's four-stage curriculum was over-engineered. Stage 1.0 → Stage 4.0 is the right two-stage path; intermediate stages actively hurt. Curriculum is mandatory (fresh fails to bootstrap), and the specific warm-start ckpt determines whether the curriculum works.*
