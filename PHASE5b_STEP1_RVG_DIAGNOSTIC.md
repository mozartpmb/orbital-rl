# Phase 5b Step 1 — R-vs-G Diagnostic

**Date:** 2026-04-28 · §3 of phase5b-step1.md · 5 min post-hoc on existing Audit logs.

## Result

| Metric | e=0 | e=0.05 | e=0.10 |
|---|---|---|---|
| Success rate | 84% | 1.5% | 4.0% |
| Mean total Δv per ep (m/s) | 129 | 106 | 109 |
| Burn frequency (%) | 4.5% | **0.7%** | 0.8% |
| Coast action share | 1.0% | **35.1%** | 35.9% |
| Warp action share | 96.6% | 64.3% | 63.6% |
| Active burn actions | mostly prograde+, retro++ | prograde+ + retro++ at <0.5% each | same |

## Interpretation

The two metrics move differently at e>0:

- **Mean Δv: nearly the same** at e=0 vs e>0 (129 vs 106; within 20%). When the policy *does* burn, magnitudes are normal.
- **Burn frequency: 6× lower** at e>0. The policy is coasting 35× more often (1% → 35%) and warping less (96.6% → 64%).

Neither pure-R nor pure-G fits the decision matrix cleanly. The combination — same Δv per burn, much fewer burns — suggests **OOD policy passivity** rather than active maneuvering: the inherited Phase 4 policy doesn't know what to do at e>0, so its argmax collapses to "coast" instead of trying maneuvers.

Important caveat: this diagnostic ran on a **frozen Phase 4 ckpt**, not on a live training run. The spec's R-vs-G matrix was designed for live-training behavior where shaping gradient should affect what the policy explores. A frozen ckpt's argmax behavior is a different signal — it tells us the inherited policy is OOD, not whether shaping would generate gradient if training resumed.

## Decision

Per spec §8 risk-table mitigation: **"Default to R interpretation (simpler hypothesis matching audit findings). Run §4-§5 anyway; the validation run itself further diagnoses the mechanism."**

The §5.1 Stage 1.0 validation run (live training under random sat init at e_max=0.05) will produce action-distribution time-series data that makes the R-vs-G call definitively. If the policy continues coasting under random sat init, gradient-absence is real. If it starts maneuvering, the previous passivity was OOD-policy artifact.

Proceeding with §4 implementation as planned.
