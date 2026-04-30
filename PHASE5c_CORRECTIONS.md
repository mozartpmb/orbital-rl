# Phase 5c — Corrections and Refinements

**Date:** 2026-04-30 (continued) · ~30 min compute · four follow-up probes that **substantially refine** Phase 5c's mechanism story.

This document corrects three claims in `PHASE5c_FINDINGS.md` and adds new evidence. Net effect: Phase 5c's quantitative findings stand (B3 = 18%, ceiling holds at 0/8 seeds), but the **mechanism story shifts** from "shaping direction reversed" to "trajectory-length-mixing artifact + early-death failure mode."

---

## Probe 4 — B3 binned eval

Eval the B3-best ckpt at varying `e_max` (100 eps each):

| e_max | B3 success |
|---|---|
| 0.05 | **94%** |
| 0.10 | 54% |
| 0.15 | 23% |
| 0.20 | 18% |

**Steep monotonic drop with e** — supports the "limit damage via low-e bias" framing. But 54% at e=0.10 is competitive with Phase 5b Stage 4.1 (65%, *trained* at e=0.10). B3 is doing more than pure damage limitation; it's preserving *and* extending capability somewhat.

The 18% at e=0.20 reflects mostly the policy succeeding on the easier samples (e closer to 0) within the [0, 0.20] uniform distribution. Some genuine high-e capability is present but limited.

---

## Probe 1 — Math re-examination (the big correction)

The original A1 finding: failures had cum r = -0.07 vs successes -0.59 → "shaping reversed." But this was a **Simpson's paradox artifact**.

### Detailed math check

Reconciliation via telescoping: `cum r_orbit = W·[(γ−1)·Σ_internal_Φ + γ·Φ_T − Φ_0]`. Two terms: a "drag" term proportional to (trajectory length) × (mean Φ), and a "boundary" term from initial vs terminal Φ.

For one success ep (847 active steps, Φ goes 22 → 1.7):
- Drag: -0.86, Boundary: -0.21, Total: -1.07. ✓ matches A1.

For one failure ep (221 active steps, Φ goes 23 → 1.3):
- Drag: -0.12, Boundary: -0.22, Total: -0.34. ✓ matches A1.

The math IS correct. The math is also correctly *length-dependent*: longer trajectories pay more drag.

### Length-binned analysis

Re-binning the 200-ep eval by trajectory length:

| len bin | n_succ | succ cum_r | po_T succ | n_fail | fail cum_r | po_T fail | Direction |
|---|---|---|---|---|---|---|---|
| [0, 100) | 2 | -0.05 | 5.98 | 83 | -0.02 | 20.32 | ✓ correct |
| [100, 300) | 11 | -0.20 | 1.22 | 3 | -0.08 | 5.74 | ✓ correct |
| [300, 700) | 6 | -0.59 | 1.43 | 2 | -0.35 | 1.98 | ✓ correct |
| [700, 2000) | 13 | -1.44 | 1.99 | 10 | -0.89 | 2.54 | ✓ correct |

**Within each length bin, successes have MORE negative cumulative shaping than failures.** The original "direction reversed" finding came from mixing populations of different length distributions:

- 83 short failures (len <100, terminal Φ ~20) have very small cumulative drag (~-0.02 each).
- 13 long successes (len 700-2000, terminal Φ ~2) have large cumulative drag (~-1.44 each).
- Population medians: success -0.59 (pulled down by long ones), failure -0.07 (pulled up by short ones).

**The shaping direction is NOT reversed within comparable trajectory lengths.** The recipe's per-step shaping is correctly oriented (more drag for going further into orbital transfer); it just isn't strong enough to overcome the discount on the terminal reward at long horizons.

### Failure-mode distribution

| Terminal Φ_orbit | Count | Median length |
|---|---|---|
| [0, 2) | 10 | 1447 (long) |
| [2, 5) | 13 | 961 |
| [5, 10) | 17 | 52 (short) |
| [10, 20) | 31 | 49 (short) |
| [20, 50) | **60** | 54 (short, ~46% of failures) |

**46% of failures terminate early with high Φ_orbit.** These are escape/collision/stranded events, not cap-timeouts. The "cap-timeout" framing in the previous writeup was wrong. Most failures are catastrophic-action terminations, not slow-climb plateaus.

The remaining 23 failures (18%) reach Φ_T ≤ 5 — i.e., they get close to target shape but miss the position+velocity dual criterion. These are "near-miss" failures, qualitatively different from the early-death majority.

---

## Probe 2 — B4/B5 peak ckpts on held-out

| Intervention | Training-time peak | Held-out at peak ckpt |
|---|---|---|
| B5 (γ=0.998) | 26% | **14% at epoch 5** (warm-start, then collapses to 0%) |
| B4 (Discrete(14)) | 25% | not eval'd cleanly (env reverted to D10; would need rebuild) |

B5 fine scan: peak held-out is at **epoch 5** (the warm-start ckpt itself, before any training). Training degrades from there. The 26% training-time peak is the dashboard mirage we've seen before — rolling-window noise, not real held-out capability.

This means B5 didn't even *briefly* improve on the warm-start. Same pattern as B6 stage 1.

---

## Probe 3 — B2 (continuous gates) empirical test

B2 was originally skipped on theoretical grounds. Testing per the "discipline: test, don't theorize" point.

| | Result |
|---|---|
| Training peak | 25% |
| Held-out best | **0%** |

**Same collapse pattern as B4/B5/B6.** Empirically confirmed: smoothing the gates doesn't help. The skip was empirically validated, but only by accident — should have tested earlier per discipline.

---

## Mechanism story — corrected

What's TRUE:

1. **Phase 5c interventions don't crack the wall.** B3 plateaus at 18%, others ~0%. This stands.
2. **B3 partially extends capability**: 94% at e=0.05 → 54% at e=0.10 → 23% at e=0.15 → 18% at e=0.20. Mostly limit-damage but with some real extension.
3. **Failure mode is mostly early death**, not cap-timeout. 46% of failures terminate at length ~54 with high Φ.

What's WRONG in the previous writeup:

1. ~~"Shaping direction is reversed at e=0.20"~~ — Simpson's paradox artifact. When binned by length, direction is correct in every bin.
2. ~~"Failures cap-timeout at 2000 steps with modest Φ ≈ 18"~~ — failures mostly die early at length ~54 with Φ > 20. The "modest Φ ≈ 18" was the population mean, conflating early-death high-Φ failures with rare near-miss low-Φ failures.
3. ~~"γ-discount drag asymmetry flips at high e"~~ — partially: at e=0, failures DID terminate early (high Φ + crashes), giving small drag. At e=0.20, failures STILL terminate early in 46% of cases, giving small drag too. The "drag asymmetry" framing was incomplete.

What the data actually supports:

The recipe at e=0.20 has a different problem than I diagnosed:

- **The agent dies early in 46% of failures** (length ~54, Φ ≥ 20). Likely because action choices at high e push satellite into invalid orbits (escape, collision, fuel-stranded).
- **17% of failures are near-miss** (Φ_T ≤ 5, length ≥ ~700). These are policy-quality failures: the agent gets close to target shape but doesn't satisfy position+velocity simultaneously.
- **The remaining ~37% are mid-Φ early-late failures** — somewhere between "died early but got partway" and "near-miss."

The interventions tried (B1-B6) address mostly the *near-miss* failure mode (longer horizon, larger actions, etc.). They don't address the *early-death* failure mode, which dominates.

**The right Phase 6 intervention candidate (per this corrected analysis):** something that prevents catastrophic actions during exploration. Options:
- **Action masking** at high e (forbid burns that would cause escape/collision per the propagated trajectory)
- **Fuel cost shaping** (currently agents can fuel-strand themselves; per-step fuel reward might prevent over-burning)
- **Soft fail recovery** (if escape/collision detected mid-burn, abort the burn instead of terminating the episode)
- **Reset-distribution adjustment**: avoid sampling task instances that are physically infeasible at the agent's current capability

---

## What this changes for Phase 6

The previous PHASE5c_FINDINGS proposed Phase 6 fixes targeting "direction inversion":
- Distance-based Φ
- Continuous reachability potential
- Time penalty

These would help with the *near-miss* failure mode (improve the policy's path to success). But they wouldn't address the dominant *early-death* failure mode.

**Better Phase 6 priority list:**

1. **Diagnose the early-death failures specifically.** What action sequences lead to escape/collision/stranded at high e? (Trajectory inspection on the 60 short-failure eps.)
2. **Action-masking or invalid-action soft-penalties** to prevent catastrophic burns.
3. **Reset-distribution conditioning** so the agent isn't asked to recover from physically infeasible setups during early training.
4. **Then** reformulate Φ if near-miss failures still dominate after early-death is fixed.

The corrected diagnostic provides a much more actionable Phase 6 target.

---

## Methodological lesson

The original A1 result was a textbook Simpson's paradox: aggregating cumulative shaping across ALL episodes (different lengths, different outcomes) produced a population-level "direction reversed" finding that wasn't true at the per-bin level.

**Two preventable mistakes:**

1. **Didn't bin by trajectory length.** A1 reported population medians without checking for length-confounding. The fix is to *always* bin by every relevant dimension before claiming a relationship.
2. **Confused training-time mean Φ with episode-level Φ traversal.** I claimed "Φ ≈ 18 throughout the failure" based on Φ_init ≈ 19 and Φ_final ≈ 18 means. But the AVERAGE Φ across all failures is computed over a mixed population — most failures end at Φ > 20 (early death), a few end at Φ < 5 (near-miss).

Both are length-related. The right discipline: bin by length AND failure mode before reporting per-bin numbers.

This is a non-trivial diagnostic skill. The Phase 5c diagnostic suite was designed to identify the failure mechanism but didn't ask the right binning questions. Adding "always check Simpson's paradox at population vs subgroup level" to the project's diagnostic playbook.

---

## Updated B3 binned interpretation

Probe 4's e-binned results (B3 ckpt at varying e_max) look like:
- e=0.05: 94% — preserved low-e capability (matches Stage 4.0's 96.4%)
- e=0.10: 54% — partial extension
- e=0.15: 23% — partial extension
- e=0.20: 18% — partial extension

The 94→54 drop is sharp. B3 isn't just preserving low-e; it's degrading at e=0.10 too. The "limit damage" framing was approximately right. But 54% at e=0.10 is worth noting: B3 can probably be a starting point for a Phase 6 retry at *e=0.10* even if it doesn't reach e=0.20.

Phase 6 strategy could be:
- Use B3-style mixed-dist training to extend Stage 4.0 capability to e=0.10 (54% is below the 65% Phase 5b achieved with dedicated training, but multi-seed B3 might reach 65%+).
- Then attempt e=0.20 separately with the early-death-prevention fixes from above.

---

*Phase 5c findings stand (B3=18%, ceiling holds), but the mechanism story shifts from "direction reversed" (wrong) to "early-death + length-mixing artifact" (correct). Phase 6 priority list updated: address early-death failures before reformulating Φ.*
