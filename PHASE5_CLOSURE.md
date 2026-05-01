# Phase 5 — Closure (Phases 5a-5e)

**Date:** 2026-05-01 · Phase 5 closes with **Closure A** (working agent at high eccentricity).

---

## Deliverable

A 2D coplanar orbital rendezvous agent that handles arbitrary phase gap and eccentricity up to e=0.50 with ≥84% success, with graceful degradation through e=0.70 (~65%).

**Recipe:** Phase 5b's two-stage curriculum (Stage 1.0 same_orbit_init=1 → Stage 4.0 random sat init) at e_max=0.05, with `valid_init_only=1` rejection-sampling enabled throughout.

**Multi-seed validation (Phase 5e Block II.A):**

| e_max | 5-seed mean ± std | Best |
|---|---|---|
| 0.05 | (deferred multi-scan) | 97.5% |
| 0.20 | **90.2% ± 2.1%** | 92.0% |
| 0.30 | (single-seed) | 93.0% |
| 0.50 | (single-seed) | 84.5% |
| 0.70 | (single-seed) | 64.5% |

5/5 seeds reach >85% at e=0.20. No collapse seed.

---

## Phase-by-phase summary

| Phase | Outcome |
|---|---|
| 5a | Process discipline (multi-seed, train-longer-null, capability surface) |
| 5b | Two-stage curriculum delivered Stage 4.0 ckpt at 96.4% e=0.05, partial extension to e=0.10 (65%) |
| 5c | 6 reward-side interventions tested at e=0.20; best B3 at 18%. Mechanism story revised twice. |
| 5d | 2 more reward-side interventions tested; same collapse pattern. **Late finding: 64% of e=0.20 inits had sub-surface perigee — physically doomed.** Added `valid_init_only=1`; same Stage 4.0 ckpt jumped from 16% → 93.5% at e=0.20. |
| 5e | Block I env-validation suite (E1-E6) confirmed env correct at high e; Block II multi-seed retrain confirmed single-seed result generalizes. Closure. |

---

## The discipline lesson

Phase 5 had **three** mechanism stories that turned out to be measurement artifacts on contaminated populations:

1. Phase 5b post-extend "60M shows erosion" — single noisy datapoint.
2. Phase 5c "shaping direction reversed" — Simpson's paradox in length-aggregation.
3. Phase 5d "high-e wall" — unphysical curriculum samples.

Pattern: each time, an aggregate statistic on a contaminated population looked structural until disaggregated. **The cost of skipping the underlying-distribution check: ~3 spec rounds and ~50 hours of compute investigating phantom mechanisms.** The cost of the check: 5 minutes of physics + 2 minutes of code.

The discipline rules added to the project playbook over Phases 5b-5e:

- **Phase 5b:** train-longer null first when stages stall; multi-seed at headline numbers; capability surface eval.
- **Phase 5c:** length-bin before aggregating any length-dependent statistic.
- **Phase 5d:** failure-mode partition before mechanism stories.
- **Phase 5e:** verify the underlying task distribution is one the recipe could possibly succeed on, before claiming a recipe ceiling from a series of failed interventions.

---

## Artifacts kept in the codebase

Available infrastructure, off by default:

- `enable_action_mask` (Phase 5d Path A): hard logit masking via 48-dim obs + Default policy mask consumer. Verified working end-to-end. Not needed for the Phase 5 deliverable but available for future high-e or constrained-action work.
- `collision_penalty_w` (Phase 5d I4): soft penalty when burns place sat on sub-keepout-perigee orbit. Verified working but ineffective during training (collapses warm-start). Off by default.
- `valid_init_only` (Phase 5d, validated by Phase 5e): rejection-sample sat & target inits with perigee ≥ EARTH_KEEPOUT. **The actual fix.** Default 0 to preserve historical reproducibility; `1` is canonical for Phase 5+ training.
- 2D Lambert solver (`scripts/orbital/p5e_e1_lambert.py`) — for future task-difficulty analysis.
- Env validation suite (`scripts/orbital/p5e_e[2-6]_*.py`) — re-runnable tests for Kepler precision, round-trip, LVLH, Φ_orbit, action effect.

---

## Recommended next steps

1. **Optional:** full multi-seed × e-scan (5 × 6 = 30 evals) to publish the formal capability surface table with mean ± std at every e_max. Phase 5e Block II.D defers this; ~3 hours compute.
2. **Phase 6:** Phase 5 unblocks Phase 6 work — debris re-enable, multi-body extension, time warp expansion, etc. The recipe is solid; future phases can build on it.
3. **Documentation:** the Phase 5 series is publishable as-is. The diagnostic depth is the contribution; portfolio writeup waits for project completion.

---

## Files

- Findings: `PHASE5b_*.md` (Phase 5b deliverable + multiseed), `PHASE5c_FINDINGS.md` + `PHASE5c_CORRECTIONS.md`, `PHASE5d_FINDINGS.md`, `PHASE5e_BLOCK_I_FINDINGS.md`, `PHASE5e_BLOCK_II_FINDINGS.md`, `PHASE5_CLOSURE.md` (this).
- Stage 4.0 best ckpt: `pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt` (Phase 5e seed 42).
- Original Phase 5b deliverable: `pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt` (works under `valid_init_only=1`).

---

*Phase 5 closed. The portfolio piece: a working orbital rendezvous agent at e ≤ 0.50 ≥ 84%, with diagnostic depth across five spec rounds.*
