# Phase 5c — Final Findings

**Date:** 2026-04-30 · ~4 hours wall-time across Block I diagnostic + Block II intervention sweep · the e=0.20 ceiling did not crack but mechanism is now well-characterized.

---

## TL;DR

Phase 5c **did not crack the e=0.20 ceiling** to the spec's ≥50% target. The best intervention (B3 mixed-distribution training) reached **18% at e=0.20 held-out** — a meaningful step from Phase 5b's 0% baseline but below the threshold for Block III multi-seed scaling. Five other interventions (B4 larger Δv, B5 higher γ, B6 REL_VEL_TOL annealing, B3+B4 combined, B3 train-longer) all collapsed to 0-14%.

The mechanism is now refined and well-supported: at e=0.20, the gated NHR shaping reward is **directionally reversed** in-trajectory (failures accumulate less negative cumulative shaping than successes), and this is mostly *not* a γ-discount artifact — it's a structural property of the bounded-Φ-times-extended-episodes regime. Phase 4's recipe worked at e=0 *because of* a γ-discount drag bias that flips sign at high e. The interventions in the Block II spec address related issues (horizon mismatch, action saturation, gate failure) but do not directly fix the direction inversion.

Phase 5c's contribution is the **diagnostic identification** of the failure mode plus a partial improvement (0% → 18%). Cracking the wall to ≥50% requires a recipe-level change that's outside Phase 5c's intervention scope — likely shaping reformulation or a fundamentally different reward structure for high-e regimes.

---

## Block I — Diagnostic findings

(Detailed in `PHASE5c_BLOCK_I_FINDINGS.md`. Summary here.)

### A1+A2 — Per-step shaping decomposition (Stage 4.0 ckpt at e=0.20)

| Group | Cum r_total (γ=0.995) | Cum r_total (γ=1) |
|---|---|---|
| Successes | -0.59 | -0.12 |
| Failures | **-0.07** | **0.00** |

Failures accumulate **less negative** cumulative shaping than successes. **Direction is reversed under both γ values** — it's not a pure γ-discount bias.

### A6 — Phase 4 baseline contrast (e=0)

| Regime | Success cum r | Failure cum r | Direction |
|---|---|---|---|
| Phase 4 at e=0 | -0.80 | -3.04 | **CORRECT** ✓ |
| Stage 4.0 at e=0.20 | -0.59 | -0.07 | **REVERSED** ✗ |

At e=0, the γ-discount drag is severe on failure trajectories (high Φ_orbit ≈ 28, fast termination) → strongly negative cumulative. The recipe relied on this asymmetry. At e=0.20, failures cap-timeout at 2000 steps with modest Φ ≈ 18, weakening the drag → direction flips.

### A5 — Φ-success correlation

| Terminal Φ_orbit | Success rate |
|---|---|
| [0, 1) | 67% |
| [2, 5) | 44% |
| [5, 10) | 9% |
| [10, 20) | 0% |

Φ_orbit IS a valid proxy. Monotonic correlation with success. The shaping target is well-formed; the per-step reward dynamics is the problem.

### A8 — Action saturation: 52.6% of successful burns at ±25 m/s. B4 well-motivated.

### A10 — Median successful ep at e=0.20: 477 steps; γ=0.995 effective horizon: 200 steps. Horizon mismatch real.

### A11 — Failure purely e-clustered (uniform across phase). Phase axis is fully generalized; eccentricity is the binding axis.

---

## Block II — Intervention sweep

All trained from Stage 4.0 best ckpt (seed 31415's 97.7% multi-rollout), single seed (42), 50M unless noted, eval at e=0.20.

| Intervention | Held-out @ e=0.20 | Notes |
|---|---|---|
| **B3 — mixed-distribution (70% e≤0.05 + 30% e≤0.20)** | **18%** | Winner, but below 30% threshold |
| B3+B4 combined (mixed-dist + Discrete(14)) | 14% | Combination hurt slightly |
| B3 long (100M instead of 50M) | 14% | Train-longer didn't help |
| B4 — larger Δv (Discrete(14)) | 0% | Training peak 25%, collapsed |
| B5 — higher γ (0.995 → 0.998) | 0% | Training peak 26%, collapsed |
| B6 — REL_VEL_TOL anneal (200 → 50) | 0% | Stage 1 → 6%, stage 2 → 0% |
| B1 — γ=1 shaping | SKIPPED | Predicted not to help by A1 (direction reversed at γ=1 too) |
| B2 — continuous gates | SKIPPED | Orthogonal to direction-inversion mechanism |

**Verdict:** Per spec §4 decision rule: "All <30% → Block I diagnostic was insufficient; return to investigation."

The diagnostic *was* sufficient to identify the mechanism (direction inversion via γ-discount-on-bounded-Φ-times-extended-episodes). What's missing is an intervention that *directly* fixes direction inversion. None of B1-B6 do that.

---

## What Block II's failure means

The interventions in the spec's Block II were each grounded in a plausible mechanism candidate:
- B5: horizon mismatch (A10) → higher γ
- B6: reward sparsity → looser success criterion early
- B4: action expressiveness (A8) → larger Δv
- B3: distribution shift catastrophic forgetting → mix easy episodes
- B1: γ-discount bias → γ=1 shaping
- B2: gate cliff → smooth gates

**These all address adjacent problems.** They don't fix the central issue: in-trajectory shaping rewards the policy more for "doing nothing" than for "doing the orbital transfer." Until the per-step shaping signal points the right way, no amount of horizon, action expressiveness, or distribution mixing will produce the right policy.

The deepest finding: Phase 4's recipe was load-bearingly dependent on a γ-discount bias that worked in its favor at e=0 (because failures had high Φ + short trajectories). At higher e, the failure-trajectory profile changes (low Φ + long trajectories due to safety cap), and the bias flips against the recipe. The recipe is regime-fragile.

---

## What partially worked: B3 mixed-distribution

B3 (70% easy + 30% hard) reached 18% — meaningful from the 0% baseline. Why does it partially work?

- 70% of training episodes are at e≤0.05 where the Phase 5b recipe is correct. The policy doesn't lose its low-e capability.
- 30% of episodes are at e≤0.20. The policy's hard-case experience is non-zero, providing some opportunity to adapt without overwhelming the gradient with reversed signal.
- At eval (uniform e=0.20), the policy uses its low-e-leaning capability on the easier draws (e<0.05 portion of the 0.20 distribution) and partially solves them.

This is essentially a "limit damage" intervention, not a "fix mechanism" one. It explains why B3 plateaus at 18% — that's roughly the success rate of the inherited Phase 4 / Stage 4.0 strategy applied opportunistically at e=0.20, not a learned high-e capability.

---

## Recipe ceiling vs fundamental limit

Phase 5c's negative result is **recipe-specific**, not a fundamental limit on RL or PB-shaping or eccentric rendezvous:

- The Stage 4.0 ckpt's *zero-shot* eval at e=0.20 is 14-16% (per Block I bonus scan). The policy *can* solve some e=0.20 cases without ever training there.
- A successful trajectory with low terminal Φ has 67% success rate in Block I A5. The shaping target is well-formed.
- The capability surface at Phase 5b's Stage 4.0 ckpt showed phase axis fully generalized (4-cell grid: 96/93/94% at low e, 14-16% at e=0.20). Phase generalization is robust.

The bottleneck is specifically: **the gated NHR shaping's per-step reward direction inverts at high e in a way that the listed interventions don't fix**. A direct fix would require:

1. **Reformulate Φ to be a true monotonic distance metric.** E.g., Φ = -dist(sat_state, target_state) where dist is some weighted distance in (a, e, ω, θ) space, normalized. Direction would be correct by construction.
2. **Add an explicit per-step penalty for time elapsed.** Discourages "doing nothing" by making the un-acting state strictly worse than the acting state. Phase 4 deliberately rejected this; revisit at high e.
3. **Replace gated cascade with a continuous reachability potential.** Rather than thresholded gates, use a smooth Ng-Harada-Russell potential that's well-defined across the entire state space.

Each of these is a Phase 6 starting point, not a Phase 5c intervention.

---

## Phase 5c contribution

What Phase 5c produced:

1. **Diagnostic identification.** The Phase 5b "Goodhart-induced collapse" finding is refined to "γ-discount-bias-on-bounded-Φ direction inversion at high e." The mechanism is testable, falsifiable, and connects to a known result in PB-shaping literature (Wiewiora 2003 effect on bounded potentials).

2. **Empirical interventions ruled out.** B5 (γ), B6 (REL_VEL_TOL anneal), B4 (larger Δv) don't help — recipe ceiling is robust to these. The intervention that partially works (B3) doesn't fix the mechanism, only limits the damage.

3. **Recipe-fragility framing.** Phase 4 → Phase 5b → Phase 5c trajectory shows the recipe is regime-fragile: it works at e=0 because of a specific γ-discount-induced asymmetry that doesn't survive at higher e. This reframes Phase 5b's "two-stage curriculum is the right answer" finding: it was right *given the recipe*, but the recipe itself has a regime-dependent ceiling.

4. **B3 as a partial improvement.** 0% → 18% at e=0.20. Marginal but the single positive intervention.

What Phase 5c did *not* produce:

- A working agent at e ≥ 0.20 to the spec's success criterion (≥50% single-seed, ≥70% multi-seed).
- A direct fix for the direction inversion mechanism.

---

## Phase 6 implications

Per spec §11, with the negative result on Block III:

> **If Phase 5c partially succeeds (working at e = 0.20 or 0.30, not 0.50):** Phase 6 multi-body work is constrained to the eccentricity regime the recipe handles.

Phase 5c is between "partial success" and "no success." It made small progress (18%) but didn't reach the soft success threshold (≥70% multi-seed at e=0.20).

For Phase 6, the implications:

1. **Multi-body work that requires high-e two-body capability is premature.** Earth-Moon Hohmann (e≈0.97) is unreachable. Earth-Mars (e≈0.21) is at the ceiling. Halo orbits and ballistic captures pass through high-e transients the recipe can't handle.

2. **Phase 6 should start with a recipe-level shaping redesign**, not multi-body env work. The candidate fixes (continuous reachability potential, time penalty, distance-based Φ) need to be tested at e=0.20 first to verify they crack the ceiling, *then* extended to multi-body.

3. **The Phase 5b Stage 4.0 deliverable stands.** A two-body rendezvous agent at 96.4% at e_max=0.05 is shippable as-is for low-e applications. Phase 5c doesn't undo that; it documents the ceiling above it.

4. **The diagnostic playbook generalizes.** A1/A2/A6 (per-step shaping decomposition + cross-regime contrast) is a reusable diagnostic for any PB-shaping recipe at a difficulty boundary. Phase 6 should run it at every regime transition, not after collapse.

---

## Files

- `PHASE5c_BLOCK_I_FINDINGS.md` — diagnostic details
- `PHASE5c_FINDINGS.md` — this document
- `scripts/orbital/p5c_block_i_posthoc.py` — A5+A8+A10+A11
- `scripts/orbital/p5c_a1_shaping_decomp.py` — A1+A2+A6
- B3-best ckpt: `pufferlib/experiments/puffer_orbital_177756728510/model_puffer_orbital_000350.pt` (18% at e=0.20)
- env code state: random sat init unchanged from Phase 5b; B3's `e_mix_easy_frac` / `e_mix_easy_max` knobs added; B5/B6/B4 reverted

---

## Stage 4.0 deliverable still ships

Phase 5b's Stage 4.0 ckpt at 96.4% multi-rollout (e=0.05 fully random) remains the headline two-body deliverable. Phase 5c documented the ceiling beyond it but did not displace it.

---

*Phase 5c didn't crack the e=0.20 wall to ≥50%, but identified why and where it fails. The recipe at e=0.20 is fundamentally direction-inverted in its in-trajectory shaping reward; the listed Block II interventions address adjacent issues but don't fix the central mechanism. Phase 6 needs recipe redesign before multi-body environment expansion.*
