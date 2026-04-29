# Phase 5b-0 — Pre-flight check

**Date:** 2026-04-29
**Goal:** Verify the recipe scales to e_max = 0.5 in Stage 1 (same-orbit) before committing to full Block A.

## Run sequence (single seed = 42)

1. **Stage 1.0 from scratch** (e_max = 0.05, same_orbit_init = 1, π phase gap, 40M)
   - Final training-time peak: 98.8% (final 98.4%)
   - Run dir: `pufferlib/experiments/puffer_orbital_177748113676`
2. **Stage 1.3 warm** (e_max = 0.50, warm from Stage 1.0 final ckpt, 80M)
   - Final training-time peak: 58.1%
   - Run dir: `pufferlib/experiments/puffer_orbital_177748261397`
   - **Held-out scan at e=0.5:**

| Epoch (~Msteps) | Held-out @ e=0.5 |
|---|---|
| 100 (~13M) | 30% |
| 200 (~26M) | 46% |
| 300 (~39M) | 50% |
| 400 (~52M) | 52% |
| **500 (~65M)** | **68%** (peak) |
| 600 (~78M) | 66% |
| 611 (final) | 66% |

## Decision

**Pre-flight passes.** Held-out 68% at e=0.5 is well above the 50% feasibility gate. **Full Block A curriculum is feasible up to e=0.5.**

## Budget implications for Block A

- Stage 1.3 (e=0.5) needs 50-80M to plateau around 67% from a *direct* Stage 1.0 warm-start. The proper Stage 1.0→1.1→1.2→1.3 progression should converge faster and possibly higher.
- Late-run dip from peak 68% (epoch 500) to 66% (final) suggests slight specialization past convergence — consistent with the post-extend characterization's "breadth ceiling at ~40M" finding.
- Per-stage budgets should target convergence, not pre-budgeted step counts. **Use the eval-scanner-driven transition rule strictly** — extend by +20M when below 80% threshold, capture peak ckpt rather than end ckpt.

## Notable absence

The spec's "if Stage 1.3 still climbing at 80M, extend to 120M" rule didn't quite apply here — the run wasn't monotonically climbing at 80M (peaked at epoch 500, slight dip after). So the conclusion is "feasible but plateaued around 67% under direct warm-start with this budget." The full curriculum should produce higher numbers than this single skipped-stage test.

---

*Phase 5b-0 done in ~22 min wall (40M Stage 1.0 + 80M Stage 1.3 warm + 7-ckpt scan). Block A can proceed.*
