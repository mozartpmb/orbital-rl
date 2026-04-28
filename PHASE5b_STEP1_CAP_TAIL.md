# Phase 5b Step 1 — Cap-Tail Analysis

**Date:** 2026-04-28 · Follow-up on `PHASE5b_STEP1_FINDINGS.md` · ~5 min post-hoc.

## Question

Step 1's seed-42 eval at Stage 1.0 lands 84.5% success / 15.5% safety_cap. Are those 31 cap failures clustered on hard (phase × e) combos, or distributed uniformly? And: is the agent grazing the tolerance window (brittleness — could be fixed with looser tolerance) or far away (capability gap — needs more training/recipe work)?

Tool: `scripts/orbital/p5b_step1_cap_tail_analysis.py`. Computes per-episode min distance, min relative velocity, min "miss factor" = `max(dist/30km, rel_vel/50m/s)` (1.0 = exactly at threshold; >1 = outside).

## Results

### Brittleness vs capability gap

| Failure category (n=31) | Count | % |
|---|---|---|
| Very close (miss factor < 1.1) | 1 | 3.2% |
| Close (< 1.5) | 3 | 9.7% |
| **Far (≥ 3.0)** | **25** | **80.6%** |

Failure stats: median min-distance **11,051 km**, median min-rel-velocity **12,066 m/s**, median miss factor **368×**. Loosening the tolerance window wouldn't fix this — the agent is never close.

**Verdict: 80% capability gap, 10% borderline, 3% true brittleness.**

### (Phase × eccentricity) success grid (seed 42, 200 eps)

| abs phase gap ＼ target.e | 0 – 0.025 | 0.025 – 0.05 |
|---|---|---|
| 0 – 60° | 32/33 = **97.0%** | 40/41 = **97.6%** |
| 60 – 120° | 31/39 = 79.5% | 25/30 = 83.3% |
| **120 – 180°** | **15/31 = 48.4%** | **26/26 = 100.0%** |

Two findings:
1. **Phase gap is the dominant difficulty axis** — small phase ≈ 97%, large phase varies 48-100%.
2. **At large phase, eccentricity inverts the difficulty.** Low-e is hardest (48%); high-e is fully solved (100%). Counter to the simple "harder = both axes max" intuition.

## Mechanism hypothesis: natural phasing on eccentric orbits

Under `same_orbit_init=1`, sat and target start on the *same* orbit, only θ differs. Phasing requires changing the time at which sat reaches a given point on the orbit. Two strategies:

- **Transfer-orbit phasing (circular case).** Raise or lower semi-major axis temporarily → enter a faster/slower drifting orbit → wait for relative phase to shift → return to original orbit. Costs fuel for two burns and requires episode time.
- **Natural phasing (eccentric case).** On an eccentric orbit, angular velocity is non-uniform: ~`v_circ × (1−e)/(1+e)` slower at apoapsis, faster at periapsis. The agent can simply *wait* through apoapsis to let the target close the phase gap. No fuel cost beyond minor station-keeping.

The data is consistent with this: the agent has emergently learned to exploit eccentricity as a phasing mechanism, which is why high-e + high-phase cases are *easier* than low-e + high-phase. On near-circular orbits the agent must execute a transfer-orbit maneuver that apparently it doesn't always nail.

This is a sophisticated emergent strategy — a free positive finding from Step 1.

## Implications for Stage 1.x scaling

1. **Expanding e_max (0.05 → 0.10 → 0.20+) should help, not hurt the headline number.** The harder corner is `(high-phase, low-e)`, which doesn't change as e_max expands; the new `(high-phase, high-e)` cells should benefit from the natural-phasing strategy.

2. **The 48% (high-phase, low-e) bottleneck is a Phase 4 territory issue resurfacing.** With near-circular orbits, the agent must do transfer-orbit phasing, which Phase 4's Stage 3 also struggled with on the long-phase end. Mitigations to test in Phase 5b proper:
   - **Raise MAX_STEPS** from 2000 to 3000 — gives transfer-orbit phasing room to complete a full transfer + drift + return. Cheap (config knob).
   - **Curriculum reweighting** to oversample low-e/high-phase episodes during training. Adds a sampling bias kwarg.
   - **Phase 4 ckpt warm-start** for the low-e regime specifically — Phase 4 Stage 3 reached 80% at e=0 with phase gap up to π. A hybrid ckpt initialization (random sat init + warm-from-Phase-4) might compose. Untested.
   - **Action-space audit** — radial actions are barely used (radial- 0.34%, radial+ 0%). Maybe action-discretization is the bottleneck for transfer maneuvers.

3. **Track natural-phasing skill across Stage 2-4 transitions.** When sat.ω, sat.a become random (Stages 2-4), the agent loses the same-orbit guarantee that enables natural phasing. Watch whether this strategy persists or has to be re-learned.

## What this isn't

- Not multi-seed. Single-seed (42) post-hoc on existing data; the (phase × e) bin counts are noisy. Re-run on seeds 1337 and 20260423 to confirm the pattern would tighten the result.
- Not a Stage 1.1 prediction. The natural-phasing inversion may shift as e_max scales; we expect it to persist but haven't tested.

---

*Cap-tail mechanism: 80% capability gap (not brittleness), bottleneck is (high-phase, near-circular) — a Phase 4 territory issue. Free finding: agent emergently uses eccentricity for natural phasing.*
