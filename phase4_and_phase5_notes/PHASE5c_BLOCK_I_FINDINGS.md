# Phase 5c Block I — Diagnostic Findings

**Date:** 2026-04-30 · ~30 min compute (post-hoc on existing eval data + 2 fresh evals).

## TL;DR — refined mechanism story

**The shaping signal direction is regime-dependent on γ-discount.** At e=0, the γ=0.995 discount on Φ_orbit accumulation creates a "background drag" that's *larger* on failures (high Φ + early termination dynamics) than on successes (low Φ + faster completion), making cumulative shaping correctly ordered (successes > failures). At e=0.20, failure trajectories cap-timeout at 2000 steps with modest Φ ≈ 18 — not high enough for the discount drag to dominate — while successes still pay the trajectory-traversal shaping cost. The direction *flips*: failures end up with less-negative cumulative shaping than successes.

This is a structural regime change, not a Goodhart-induced collapse in the active-gradient-hacking sense. The Phase 5b `PREDICTIONS` finding ("Goodhart-induced collapse") was directionally right (the policy *is* optimizing toward the wrong reward) but the mechanism is more nuanced: the recipe relied on a γ-discount accounting bias that works at e=0 but inverts at high e. **The recipe at e=0 worked *because of* the γ-discount bias, not despite it.**

This re-frames the Phase 5b finding: it's not "the gates fail at high e and Goodhart kicks in." It's "the γ-discount-bias on PB shaping was always doing load-bearing work, and that load-bearing work breaks at high e."

## Diagnostic results

### A1 — Per-step shaping decomposition (Stage 4.0 ckpt at e=0.20)

200 eps eval, 32 successes / 168 failures.

| | Cum r_total (γ=0.995) | Cum r_total (γ=1.0) | γ-bias |
|---|---|---|---|
| Successes | median -0.59 | -0.12 | -0.42 |
| Failures | median -0.07 | 0.00 | -0.06 |
| Direction at γ=0.995 | **REVERSED** | REVERSED | — |

Φ_orbit dynamics:
- Successes: init 16 → final 2 (large drop, ~14 units of |Φ| traversed)
- Failures: init 19 → final 18 → max 22 (near-constant, agent doesn't transfer)

### A2 — Counterfactual (built into A1's success vs failure comparison)

The smoking-gun test failed: failures accumulate **less negative** cumulative shaping than successes. PPO is correctly maximizing the shaping signal it sees — the signal itself is the problem.

### A5 — Φ_orbit-success correlation

| Terminal Φ_orbit bin | Success rate |
|---|---|
| [0, 1) | 67% |
| [1, 2) | 67% |
| [2, 5) | 44% |
| [5, 10) | 9% |
| [10, 20) | 0% |
| [20, 50) | 0% |

Φ is a **valid proxy** when the policy can get there. Monotonic decrease. The shaping target is well-formed; the path-dependent reward dynamics is the problem.

### A6 — Phase 4 baseline contrast (the diagnostic-by-contrast)

200 eps Phase 4 ckpt at e=0, 156 successes / 44 failures.

| | Cum r_total (γ=0.995) | Cum r_total (γ=1.0) |
|---|---|---|
| Successes | median -0.80 | -0.14 |
| Failures | **median -3.04** | 0.00 |
| Direction at γ=0.995 | **CORRECT** ✓ | REVERSED |

At e=0, failures accumulate **much more negative** shaping than successes (-3.04 vs -0.80). The recipe works because the γ-discount drag is severe on failure trajectories (high Φ_orbit ≈ 28).

**Comparison side-by-side:**

| Regime | Success cum r | Failure cum r | Direction |
|---|---|---|---|
| Phase 4 at e=0 | -0.80 | **-3.04** | CORRECT (gap -2.2pp) |
| Stage 4.0 at e=0.20 | -0.59 | **-0.07** | REVERSED (gap +0.5pp) |

The shaping signal direction is genuinely regime-dependent. Phase 4's success at e=0 was contingent on this regime working in its favor.

### A8 — Action saturation in successful e=0.20 episodes

Median **52.6%** of burns are at the maximum (±25 m/s prograde or retro). Half of all successful burns saturate the action space → the agent is action-bottlenecked. **B4 (larger Δv actions) is well-motivated.**

### A10 — Effective horizon mismatch

| Metric | Value |
|---|---|
| γ = 0.995 effective horizon | 200 steps |
| Median successful ep length at e=0.20 | **477 steps** |
| Successes >200 steps (past effective horizon) | 25/32 (78%) |
| Successes >800 steps | 8/32 (25%) |

At γ=0.995, the discounted +10 terminal reward at t=477 is worth ~10·0.995^477 ≈ +0.9 at t=0. The cumulative in-trajectory shaping for a success (~-0.6) is competitive with the discounted terminal. **The horizon mismatch is real and large** — the agent literally can't see the success reward through the discount. **B5 (higher γ) is well-motivated.**

### A11 — Failure clustering by (phase × e)

| | e ∈ [0, 0.05) | e ∈ [0.05, 0.10) | e ∈ [0.10, 0.20) |
|---|---|---|---|
| 0–60° | 50% success | 33% | **0%** |
| 60–120° | 33% | 27% | **0%** |
| 120–180° | 30% | 26% | **0%** |

Failure is **purely eccentricity-clustered**, uniform across phase. e ≥ 0.10 is the wall regardless of phase gap. Confirms phase axis is well-generalized; eccentricity is the binding axis.

### Skipped diagnostics

- **A3 (V-landscape)** + **A4 (Q-landscape)** — would localize the problem in the value function vs policy, but not load-bearing for choosing interventions. The mechanism is identified; deferred unless Block II results are puzzling.
- **A7 (LVLH degradation)** — could check observation saturation at high e, but unlikely to be the dominant mechanism given the clear shaping-direction finding. Deferred.
- **A9 (Lambert benchmark)** — A8 already establishes action saturation; Lambert provides additional context but isn't load-bearing.

---

## Mechanism verdict

**Regime-dependent γ-discount bias on PB shaping.** The Phase 4 recipe's gated multi-stage potential Φ has been computing per-step shaping deltas as `β·(γ^τ·Φ(s') − Φ(s))` with γ=0.995. This produces a "background drag" on the per-step reward proportional to `(1−γ)·|Φ|`. At e=0:

- Failure trajectories have high Φ (~28) and short duration (early termination via escape/collision/stranded). Background drag accumulates aggressively per-step → cumulative is very negative.
- Success trajectories have Φ dropping from ~16 to ~2 quickly. Less drag accumulation → less negative cumulative.
- Difference: -3 vs -0.8. Successes are clearly preferred. PPO trains correctly.

At e=0.20:

- Failure trajectories have *modest* Φ (~18) but very long duration (cap timeout at 2000 steps). Drag accumulation per-step is moderate but spread across many steps. The active-mask filtering (warp sub-steps zero out) brings cum_r near zero on average.
- Success trajectories still pay the Φ-traversal cost, ~-0.6.
- Difference: success -0.6 vs failure -0.07. **Failures are preferred under shaping.** PPO correctly maximizes shaping; in doing so, it abandons orbit-shape change.

The mechanism is *not* "active gradient hacking via gate failure." It's **shaping-direction inversion via γ-discount-on-bounded-|Φ|-times-extended-episodes**. The Phase 5b PREDICTIONS finding identified the symptom (Goodhart-style behavior) but mis-attributed the mechanism (gate failure instead of γ-discount inversion).

**Falsifiable prediction:** B5 (higher γ → 0.998) should partially fix the direction (longer horizon makes terminal reward more salient + reduces background drag bias). B6 (REL_VEL_TOL annealing) should also help (more frequent terminals anchor the value function, breaking the direction inversion). B1 (γ=1 shaping only) will *not* fix it because the γ=1 cumulative is also reversed (per A1 + A6 data).

---

## Block II priority (refined)

Per the falsifiable prediction:

1. **B5 (γ → 0.998)** — most direct address of the horizon + drag-bias mechanism. Cheap (config change). Run first.
2. **B6 (REL_VEL_TOL annealing)** — provides more frequent terminal anchoring at high e. Long-deferred from Phase 5b Step 1. Run second.
3. **B4 (larger Δv)** — A8 says 52% action saturation. Likely helps regardless of mechanism. Run third.
4. **B3 (mixed-distribution training)** — keeps shaping in a regime where direction is correct. Theory-motivated. Run fourth.
5. **B2 (continuous gates)** — orthogonal to direction-inversion mechanism. Likely no effect. Run only if 1-4 don't work.
6. **B1 (γ=1 shaping)** — explicitly contradicted by A1 + A6 (direction also reversed at γ=1). **Skip** unless other interventions fail.

Compute estimate: 5 interventions × ~1.5 hr = ~7.5 hr (down from 10 hr if all 6 needed).

---

## Files

- `scripts/orbital/p5c_block_i_posthoc.py` — A5+A8+A10+A11 combined
- `scripts/orbital/p5c_a1_shaping_decomp.py` — A1+A2 shaping decomposition + A6 reuse
- `logs/orbital/p5c_s40_at_e020/` — 200-ep Stage 4.0 ckpt eval at e=0.20
- `logs/orbital/p5c_p4_at_e0/` — 200-ep Phase 4 baseline at e=0

---

*Block I refines the Phase 5b PREDICTIONS finding: the mechanism is regime-dependent γ-discount bias, not active gradient hacking. The recipe relied on this bias working in its favor; it inverts at high e. Block II's intervention priority shifts: B5 (higher γ) is the lead candidate. B1 (γ=1 shaping) is contradicted by the data and skipped.*
