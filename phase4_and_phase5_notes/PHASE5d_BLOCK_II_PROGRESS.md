# Phase 5d — Block II Progress (interim)

**Date:** 2026-04-30 (continued) · ~30 min compute on first two interventions.

Block I produced a clean verdict: 70% of failures at e=0.20 are Earth-collision via perigee lowering. Block II's first interventions test soft collision penalty (I4) under two configurations.

---

## Results so far

| Intervention | Config | Held-out (best ckpt) | Held-out (final) | Notes |
|---|---|---|---|---|
| Stage 4.0 baseline | (untrained at e=0.20) | 16% | — | Phase 5b deliverable, 70% collision |
| Phase 5c B3 | e_mix_frac=0.7, e_mix_max=0.05 | 16% | 16% | 30% collision, 55% safety_cap |
| **I4 alone (w=1.0)** | collision_penalty_w=1.0 | 3% (ep 5, ≈ warm-start) | 0% | Penalty too strong; monotonic collapse |
| **I4 + B3 (w=0.1)** | both stacked | 12% (ep 5) → 6% → 0% | 0% | Slower collapse, same trajectory |

**Pattern matches Phase 5c B5/B6:** training-time perf shows 0.0% throughout (rolling-window noise from terminal -10s); held-out scan shows monotonic decline from warm-start ckpt. The Stage 4.0 attractor is fragile to any reward perturbation.

I4's intended mechanism (penalize burns that lower perigee below EARTH_KEEPOUT) fires correctly in unit test but, applied to PPO training, biases gradient against burning generally. Even at w=0.1 with B3's mixed-distribution prior, the policy degrades rather than learns to burn-then-avoid-collision.

---

## Diagnosis

The collision penalty is *correlated with task difficulty* (high-e tasks need risky burns; low-e tasks don't), so it doubles as a difficulty discount: PPO learns "avoid penalties = avoid hard tasks." Under PPO's advantage estimator, this produces a global capability degradation, not a targeted collision-avoidance.

Two paths forward:

### Path A — True hard masking (I2)

Mask catastrophic actions at the logit level before sampling. The policy never sees the bad action; its value function is unperturbed. Expensive to plumb (modify policy forward + rollout collection) but the only way to get a clean signal.

Cost: ~3-4 hours engineering, then re-run the same training comparison.

### Path B — Accept the ceiling, characterize it carefully

Phase 5c established that the Stage 4.0 → e=0.20 transition is fragile to ALL recipe-level interventions tested (B1-B6, plus I4 alone, I4+B3). This is now 8+ failed interventions with the same dashboard-mirage pattern.

The honest reading: **the recipe ceilings near e=0.10-0.15.** Phase 5b's deliverable (e ≤ 0.05 fully random at 96.4%, e ≤ 0.10 partial extension at 65%) is what the recipe supports. Phase 5d's interventions confirm the ceiling rather than crossing it.

This is Phase 5d spec §10 Closure B: "documented ceiling with full mechanism attribution." The mechanism story (Block I findings doc) is solid. The ceiling story:

- Stage 4.0 attractor is at low-e capability
- Any reward modification (B1, B5, B6, I4) destabilizes it without leading to high-e capability
- B3 (mixed-distribution) preserves it and slightly extends collision-avoidance behavior, but doesn't extend success
- Hard masking might cross the wall but is an architectural intervention that exits the "potential-shaping recipe" that Phase 4-5 was scoped around

---

## Recommendation (per spec §4.3 decision rule)

Block II so far: 0/2 interventions reach ≥30%. Spec rule: "If all interventions stay below 30%: Block I diagnostic missed something; return to investigation. Pre-commit: ≤ 1 round."

**Pre-commit invocation:** Block I's failure-mode partition is correct (collision dominates). The interventions failing isn't a Block I problem — it's a recipe-fragility problem (Phase 5c's repeated discovery). The "missing diagnostic" isn't Block I; it's a recipe-architecture analysis that's outside Phase 5's scope.

**Pause Block II for user review.** The remaining Block II items (I2 hard masking, I3 fuel cost, I4 generic, I5/I6/I7 retests) all require the same warm-started training that fails identically. Before committing 6+ more failing experiments, the right move is to:

1. Decide between Path A (engineer hard masking) and Path B (close Phase 5 with documented ceiling).
2. If Path A: invest the 3-4 hour engineering cost.
3. If Path B: write up Phase 5 closure consolidating Phase 5b deliverable + Phase 5c/5d ceiling findings.

This pause respects the user's stated preference for staged experiments → analysis → proposal (memory: feedback_user_staged_experiments).

---

## Code state

- I4 implementation in `orbital.h` and `orbital.py` is correct and commitable. Default `collision_penalty_w=0.0` (off), so canonical recipe is unaffected.
- Two collapsed experiments under `experiments/puffer_orbital_177760385930/` (I4 alone) and `experiments/puffer_orbital_177760496851/` (I4+B3). Can be deleted after writeup or kept as ceiling evidence.

*Block II paused at 2/7 interventions with consistent collapse pattern. Awaiting user direction on Path A vs Path B.*
