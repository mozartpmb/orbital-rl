# Phase 5b Clarify Experiments

**Date:** 2026-04-30 · ~1.5 hr compute · validates / corrects three Phase 5b claims.

## What was tested

Three follow-up experiments on the Phase 5b findings, prompted by reviewer questions:

1. **Exp 1: 5 more seeds at Stage 4.2** — does the e=0.20 ceiling hold beyond the original 3 seeds?
2. **Exp 2: 2× budget at Stage 4.2** — is the failure "genuine recipe ceiling" or "slow bootstrap that needs more time"?
3. **Exp 3: early-peak vs late-peak Stage 1.0 → Stage 4.0** — is the "specialization erodes breadth" mechanism real?

---

## Exp 1 — Stage 4.2 ceiling holds at 8 seeds

5 new seeds (1337, 2718, 1729, 6022, 7919) × 50M warm from the best Stage 4.0 ckpt (seed 31415):

| Seed | Best perf @ e=0.20 |
|---|---|
| 1337 | 2% |
| 2718 | 4% |
| 1729 | 4% |
| 6022 | 0% |
| 7919 | 2% |
| **Mean** | **2.4%** |

Combined with the original Block C (3 seeds at 0/0/2%): **0/8 seeds bootstrap at Stage 4.2.** Stage 4.2 ceiling claim has solid empirical support.

All failed seeds peak at epoch 25 — same warm-start preservation pattern from Block B's 2/5 collapse. The training never improves on the warm-start at e=0.20.

---

## Exp 2 — Train-longer doesn't fix Stage 4.2

Stage 4.2 with 100M budget (vs the original 50M), seed 42, warm from Stage 4.0:

- **Best perf: 4% at epoch 25** (same as 50M budget).

Train-longer playbook: when a stage looks stalled, double the budget before declaring failure. That's how we resolved the cap-tail finding in Step 1 (40M → 99.7%). At Stage 4.2, **100M doesn't change anything** — peak is at the same epoch (25), and the value is the same (4%). This isn't "slow bootstrap"; the recipe genuinely stops producing improvement at e=0.20.

This is a meaningful negative finding. Train-longer is the project's go-to first response, and it sometimes resolves apparent ceilings into "just under-trained" outcomes. Stage 4.2 is a real ceiling.

---

## Exp 3 — Specialization erodes breadth, but small effect

The original Phase 5b writeup claimed "Stage 1.3 specialization erased 38pp of OOD breadth (43.6% → 5.2%)." That comparison conflated two effects:

- **Effect 1 (specialization on same distribution):** training past peak slightly degrades OOD generalization
- **Effect 2 (training-distribution shift):** training at a different e_max than the test distribution causes a large performance gap

The 38pp drop from Stage 1.0 (43.6% OOD) to Stage 1.3 (5.2% OOD) was **mostly Effect 2** (Stage 1.3 was trained at e=0.5, Stage 4.0 evaluates at e=0.05 — different distributions). Not Effect 1.

To isolate Effect 1, compare ckpts from **the same training run** (Stage 1.0 seed 42) at multiple epochs. Same training distribution; only the amount of in-dist training varies.

| Epoch | In-dist (Stage 1.0) | OOD (Stage 4.0) |
|---|---|---|
| 100 (~6.5M) | 68% | 8% |
| 150 (~10M) | 79% | 13% |
| 200 (~13M) | 94% | 33% |
| 225 (~15M) | 97% | 42% |
| 250 (~16M) | 100% | 49% |
| **275 (~18M)** | 100% | **53%** ← OOD peak |
| 300 (~20M) | 100% | 51% |
| 305 (~20M) | 100% | 47% |

**Pattern:** In-dist climbs to 100% by epoch 250, plateaus. OOD continues climbing to a peak at epoch 275 (53%), then declines to 47% by epoch 305.

**Effect 1 magnitude:** ~**6pp OOD erosion** over ~50 epochs of post-in-dist-convergence training (~3M steps). Small but real.

Compare to Effect 2 magnitude: 38pp Stage 1.0 → Stage 1.3 (different distributions). Effect 2 dominates by ~6×.

### Implication for Phase 5b's central claim

The "specialization erodes breadth" mechanism is real but smaller than the writeup made it sound. The bigger effect is **training-distribution mismatch** — Stage 1.3 ckpts are bad for Stage 4.0 transfer because Stage 1.3 trained at a different e (0.5 vs 0.05), not because Stage 1.3 was over-trained relative to Stage 1.0.

This refines the Phase 5b curriculum recommendation:
- "Don't over-train any single stage" remains true (~6pp on the table)
- "Don't change training distribution unnecessarily before transition" is the bigger lesson
- The "two-stage curriculum (1.0 → 4.0)" recommendation is correct, but the *reason* it's correct is "Stage 4.0's task is closer to Stage 1.0's distribution than to Stage 1.3's distribution," not "Stage 1.3 over-specialized."

---

## Original Exp 3 design flaw (acknowledged)

The first Exp 3 attempt used Stage 1.0 epoch 50 (very early, baseline 0%) vs epoch 250 (peak, baseline 49%). That was testing "pre-convergence vs peak," not "peak vs post-peak." The result (0% trained from epoch-50 vs 96% from epoch-250) just confirms you need a converged warm-start, not whether specialization erodes breadth. The ckpt-scan version above is the right test.

This is an example of the methodological lesson from Step 1's cap-tail probes: it's easy to design a test that doesn't isolate the hypothesis. **Always ask: "what would a null result of this experiment look like, and is my test capable of producing it?"** A peak-vs-pre-convergence test couldn't have shown the breadth-erosion effect either way.

---

## Updated mechanism story

Three effects in Phase 5b's curriculum behavior, in order of magnitude:

1. **Training-distribution mismatch (largest, 38pp)**: warm-starting from a ckpt trained on a different distribution costs heavily. Stage 1.3 (e=0.5) → Stage 4.0 (e=0.05) is a -38pp transfer gap.
2. **Random-init bootstrap failure (large, 50-100pp)**: fresh-from-random Stage 4.0 = 0%. Curriculum is required regardless of warm-start specifics.
3. **Specialization on same distribution (small, ~6pp)**: training past in-dist convergence mildly erodes OOD breadth.

The Phase 5b two-stage curriculum (Stage 1.0 → Stage 4.0) is the right answer because it minimizes Effect 1 (Stage 1.0 distribution is closest to Stage 4.0 of all candidates) and addresses Effect 2 (the warm-start). Effect 3 is in the noise compared to the others.

---

## What this changes about Phase 5b's recommendations

| Recommendation | Status |
|---|---|
| Stage 1.0 → Stage 4.0 two-stage curriculum | **Confirmed** (right reasons clarified) |
| Stage 4.2 (e=0.20) ceiling | **Confirmed** at 0/8 seeds |
| Train-longer at Stage 4.2 | **Confirmed not to help** |
| "Specialization erodes breadth" framing | **Refined** — small effect, distribution mismatch is the bigger one |
| "Don't over-train any stage" | Still good practice (~6pp on the table) but secondary to "don't change training distribution unnecessarily" |

---

## Phase 6 implications

The training-distribution-mismatch finding suggests a different curriculum strategy than progressive specialization:

- **Don't curriculum through eccentricity:** stay at the lowest e_max possible while expanding constraint relaxation (same_orbit_init, etc.). Test: does Stage 1.0 (e=0.05 same-orbit) → Stage 4.0 (e=0.05 fully random) stay at the e=0.05 distribution and avoid Effect 1?
- **Probe the actual mechanism behind Stage 4.2's ceiling:** is it reward-reachability (success criterion unreachable at e≥0.20 for trained policies) or shaping mis-specification? The Phase 5b audit found σ₃ activation matters; an analog for Stage 4.2 is worth running.
- **Test if the ceiling is breakable** with: REL_VEL_TOL annealing, larger discrete-action Δv, finer phase-gap sampling, etc.

The capability surface eval showed Stage 4.0 ckpts can solve 14-20% of e=0.20 cases without training there. So the policy *can* handle some e=0.20 trajectories — it just can't train to specialize on e=0.20. That gap (between "evaluation succeeds sometimes" and "training collapses") is the right Phase 6 starting question.

---

## Files

- All clarify ckpts under `pufferlib/experiments/puffer_orbital_*` (timestamps 1777512* through 1777521*)
- `/tmp/p5b_clarify.sh` — orchestrator
- `/tmp/p5b_curriculum.log` — full timeline

---

*Three clarify experiments validated 2 of Phase 5b's claims (Stage 4.2 ceiling, train-longer-doesn't-help) and refined a third (specialization-erodes-breadth is real but small; the bigger effect was training-distribution mismatch). Updated mechanism story improves Phase 5b's narrative without changing its primary deliverable.*
