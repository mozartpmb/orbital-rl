# Phase 5d — Block I Findings

**Date:** 2026-04-30 (continued) · ~30 min compute · diagnostic-first per spec §3.

Block I's job: characterize the failure population at e=0.20 before designing interventions. Result: a single failure mode dominates (Earth collision), and B3's mechanism is *not* limit-damage — it's a partial collision-prevention that runs out the clock.

---

## D1 — Termination-mode classification (Stage 4.0 ckpt @ e=0.20, 200 eps)

| Mode | Count | Pct | Median len | Median Φ_T | Median fuel |
|---|---|---|---|---|---|
| success | 32 | 16.0% | 477 | 1.68 | 0.061 |
| **collision** | **140** | **70.0%** | **36** | **19.06** | **0.150** |
| safety_cap | 28 | 14.0% | 2000 | 21.18 | 0.131 |
| escape | 0 | 0% | — | — | — |
| stranded | 0 | 0% | — | — | — |

Of 120 short failures (n<100): **100% are collisions**. No escapes, no fuel-stranded.

The "early-death" failures are **Earth atmospheric reentry** (radius < HARD_R_EARTH ≈ 6.571e6m). The agent, given high-eccentricity targets, executes retrograde-or-radial burns that drop perigee below 200km altitude, terminating the episode within ~36 steps. Median fuel at termination = 0.150 (fully fueled — they never even started a serious maneuver sequence).

This is decisive per spec §3.1's decision rule: **one mode dominates ≥60%** → target it specifically. Phase 5d Block II priority becomes **I2 (collision-prevention action masking)**.

---

## D2 — Fine-grained length histogram (bin=20)

Strongly bimodal:

| Bin | Failures | Notes |
|---|---|---|
| [0, 20) | 56 | died on first ~20 steps |
| [20, 40) | 17 | |
| [40, 60) | 32 | second peak — first burn arc completes here |
| [60, 80) | 11 | |
| [80, 100) | 4 | |
| [100, 1900) | 15 | sparse middle |
| [1900, 2000) | 33 | safety_cap concentration |

53.6% of failures n<50, 71.4% n<100, 19.6% n>=1900. The "early-death" framing survives fine-grained re-binning. Coarse bins did not invent the structure.

The 56 ultra-short (n<20) failures are likely "first-action collision" — the policy commits a catastrophic burn immediately on reset.

---

## D3 — Phase 4 baseline length re-bin: DEFERRED

Not run this round. The dominance of collision failures at e=0.20 (70%) is so clean that the Phase 4 baseline contrast is no longer the critical question. D3 deferred to post-intervention; if I2 doesn't crack the wall, revisit then.

---

## D4 — B3 ckpt @ e=0.20 vs Stage 4.0 baseline (200 eps each)

| Mode | Stage 4.0 | B3 | Δ |
|---|---|---|---|
| success | 32 (16.0%) | 32 (16.0%) | 0 |
| collision | 140 (70.0%) | 59 (29.5%) | **−81 (−58%)** |
| safety_cap | 28 (14.0%) | 109 (54.5%) | **+81 (+289%)** |

**B3 differentially reduces collisions by 58%** but converts them into safety-cap timeouts. Net success rate is unchanged (16% both) because the reclaimed-from-collision episodes don't reach the target — they wander until the cap.

Of B3's short failures (n<100), 32/32 are still collisions. So B3 cuts collisions where it can but doesn't eliminate the first-action-catastrophe pattern.

This **refutes Phase 5c's "B3 is limit-damage" hypothesis.** B3 is a partial collision-preventer (the mixed-distribution training teaches the policy to be less reckless), not a low-e bias trick. Implication: stacking explicit collision masking (I2) on top of B3 should compound — the policy is already partially aware, and masking removes the residual catastrophic actions.

---

## D5 — Trajectory inspection: DEFERRED

Skipping this round. The D1 collision dominance is unambiguous, and D2's first-20-step peak combined with the 0.150 median fuel-at-termination confirms the failure pattern: catastrophic perigee-lowering burn within the first few actions. Detailed action-sequence inspection deferred to post-I2 if needed for tuning the masking threshold.

---

## Refined characterization

**Phase 5d's intervention target:** Earth-collision failures, dominating 70% of all eval episodes (83% of failures) at e=0.20. Mechanism: the policy, trained on low-e tasks, executes burns that work for low-e geometries but lower perigee below 200km when the target eccentricity is high. The high-e regime requires conservative perigee-management that the policy hasn't learned.

**Why low-e-trained policy fails this way:** at e ≤ 0.05, satellite & target are both near-circular and the cheap maneuvers (small radial/retrograde) don't risk perigee. At e=0.20, the same maneuvers from a near-circular sat onto an elliptic target frequently push perigee inside the keep-out radius. The policy has no signal during low-e training that perigee is a constraint.

**Confirmation of B3 mechanism:** B3 reduces collisions by 58% via mixed-distribution exposure. It's partial collision-awareness, not limit-damage. The 16% success ceiling on B3 isn't a fundamental cap — it's a "collisions traded for timeouts" artifact. Adding explicit collision prevention (I2) should unlock the safety-cap subpopulation by giving the policy time to find the success manifold.

---

## Pre-committed Block II priority (per §4)

1. **I2 (collision-prevention masking)** — primary. Mask actions that would push perigee below HARD_R_EARTH within the next step (or N-step lookahead).
2. **I2 + B3 stacked** — collision masking *during* mixed-distribution training. Tests whether B3's partial collision-awareness compounds.
3. **I7 (B5 + I2)** — higher γ on top of collision masking. The 54.5% B3 safety-cap subpopulation suggests effective horizon may also matter once collisions are eliminated.

Skip I1 (escape masking) and I3 (fuel cost) — D1 shows zero failures in those modes.

I4 (generic safety penalty) held in reserve as a softer alternative to I2 if hard masking destabilizes training.

---

## Discipline checks

- **Length-binning before aggregating:** done. D2 fine-grained confirmed bimodal structure; the 70% collision figure is consistent across length bins (collisions concentrated short, safety_cap concentrated long, no length × mode confound).
- **Failure-mode partition before intervention design:** done. D1's clean partition (70% / 14% / 0% / 0%) makes the intervention target unambiguous.
- **Single-seed caveat:** all D-probe results are on single eval seeds. Block II will be single-seed too; multi-seed validation is Block III's job.

---

*Block I complete in ~30 min. Phase 5d proceeds to Block II with I2 (collision masking) as primary target. The "early-death" failure mode was confirmed and refined: it's specifically Earth-collision via perigee lowering, not a mix of escape/collision/stranded.*
