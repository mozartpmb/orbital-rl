# Phase 5b Step 1 — Findings

**Date:** 2026-04-28
**Spec:** `phase5b-step1.md` + `phase5b-audit-plan.md` (audit findings)
**Compute spent:** ~30 min total (1 train + 1 ckpt scan + 3-rollout eval + post-hoc analyses)

---

## TL;DR

The Phase 5b env change (random satellite eccentricity + `same_orbit_init=1` Stage 1 constraint) **bootstraps cleanly without REL_VEL_TOL annealing**. Held-out multi-rollout eval at the spec's Stage 1.0 conditions (e_max_sat = e_max_target = 0.05, init_phase_gap_max = π) lands **81.7% mean across 3 rollout seeds at 20M steps**, climbing to **99.7% mean at 40M steps** with the same recipe — well above the spec's ≥50% success threshold.

**Update 2026-04-28 +1hr:** the 15.5% safety_cap tail at 20M training was *under-converged training*, not a structural recipe issue. See `PHASE5b_STEP1_CAP_TAIL.md` for the full investigation arc (P1/P2/P3 probe mechanisms → train-longer experiment overturning all three).

Three structural improvements came along with the env change:

1. **σ₃ went from structurally dead to active.** Phase 5b audit found σ₃ firing ≤2% at all eccentricities, even in Phase 4's working regime. Step 1's Stage 1.0 sees σ₃ at **45.8% activation** — the gated cascade is genuinely three-component now.
2. **Goodhart ratio collapsed from 0.114 → 0.007.** Cumulative shaping is now <1% of terminal magnitude (vs >10% in audit). No Goodhart concerns.
3. **The recipe doesn't suffer the R5-style end-of-training collapse.** Training-time perf went 89.3% peak → 84.1% final (vs Phase 4.5 R5's collapse to 0%). The same-orbit constraint stabilizes warm-end-of-training dynamics.

REL_VEL_TOL annealing was not implemented; with these results it isn't needed to make Stage 1.0 work. Reserved for Phase 5b proper if Stage 1.x or Stage 2+ benefit from it.

---

## The five-number deliverable (per spec §5.5)

### #1 — R-vs-G diagnostic

Pre-Step-1 finding: frozen Phase 4 ckpt at e=0.05 burns 6× less than at e=0 (0.7% vs 4.5%) but with similar Δv per burn. Interpreted as **OOD-policy passivity**, defaulted to "R — reachability" hypothesis since the simpler interpretation matched audit findings.

**Resolution by Step 1.B:** under random sat init (which broadens the input distribution), the trained policy burns 15.8% of steps — 22× higher than the frozen Phase 4 policy at the same condition. The earlier passivity was OOD artifact, not a fundamental gradient-absence problem. **R was the right framing.**

### #2 — Stage 1.0 held-out success rate

Spec §5.2 thresholds: ≥50% bootstrap, ≥70% decisive.

| Rollout seed | Held-out eval @ Stage 1.0 (200 eps argmax) |
|---|---|
| 42 | 84.5% (169/200) |
| 1337 | 83.5% (167/200) |
| 20260423 | 77.0% (154/200) |
| **mean ± std** | **81.7% ± 3.3pp** |

**Decisive.** Well above the 70% threshold. Recipe genuinely solves same-orbit eccentric rendezvous at e_max=0.05.

Training-time peak: 89.3% (epoch ~149). Held-out peak: 84% at epoch 150. **5pp train/eval gap** — far smaller than the 88pp gap that prompted the audit. The PHASE5a A2 mirage was a difficulty-cliff artifact (variance manifests when learning is feasible; collapses when it isn't); at this difficulty the metric is reliable.

### #3 — REL_VEL_TOL annealing contribution

**Skipped** per plan §1.E ("Skip §5.1 if Step 1.B is decisive ≥70%; annealing won't dramatically change a working recipe").

If Phase 5b proper hits a stage where annealing is needed (e.g., Stage 1.x at e_max ≥ 0.20, or Stage 2/3/4 transitions), revisit the implementation. Implementation cost ~1 hr engineering: add `rel_vel_tol_initial`, `_final`, `_anneal_envsteps` kwargs; track `env->global_step_count`; recompute live `env->rel_vel_tol` per c_reset.

### #4 — σ₃ activation rate

| Condition | σ₂ activation | σ₃ activation |
|---|---|---|
| Audit: A1 frozen at e=0 (84% success) | 24.2% | 1.6% |
| Audit: A1 frozen at e=0.05 (1.5% success) | 17.0% | 0.7% |
| Audit: A1 frozen at e=0.10 (4.0% success) | 20.1% | 0.2% |
| **Step 1: trained policy at Stage 1.0 (84.5% success)** | **75.0%** | **45.8%** |

σ₃ went from "structurally dead at all conditions" to "fires nearly half the time." Per spec §5.4 (>5% threshold): random sat init incidentally fixes σ₃'s deadness. **Free improvement.**

The mechanism: with `same_orbit_init=1`, sat and target start with identical (a, e, ω). Φ_orbit is small from the outset (Δa=0, Δē=0), so σ₂ stays active throughout the episode. As the policy closes the phase gap, Φ_phase drops below EPS_PHASE=0.3 → σ₃ activates. The cascade was designed for "after orbit-shape closes, focus on phase, then velocity." On Stage 1.0 the orbit-shape is *already* closed at init, so the cascade rolls forward as designed.

This also means: σ₃'s activation depends on the task structure being "orbit-shape close, phase gap remaining." If Stage 2-4 generalize (orbit-shape differs at init), σ₃ may go dormant again. Re-check σ₃ activation at each stage transition.

### #5 — Trajectory inspection

| Metric | Audit @ e=0.05 (frozen A1) | Step 1.0 trained |
|---|---|---|
| Mean total Δv per ep | 106 m/s | **161 m/s** |
| Mean burn frequency | 0.7% | **15.8%** |
| Coast action share | 35.1% | 19.4% |
| Warp action share | 64.3% | 75.0% |
| Termination mode (success / cap / other) | 1.5% / 93% / 5.5% | **84.5% / 15.5% / 0%** |

Burn-frequency jump is the clearest signal: the trained policy is *actively maneuvering* at this difficulty, not coasting. The action distribution shows prograde+ (2.76%), retro++ (2.56%), radial- (0.34%) — consistent with phasing maneuvers (raise/lower a temporarily to drift) interspersed with warp coast.

Goodhart ratio (cumulative shaping / |terminal|): 0.007 mean, 0% of eps over the Phase 4 plan's 0.1 threshold. Shaping is well below the threshold; no Goodhart concerns at Stage 1.0.

The 15.5% safety_cap tail is notable — the policy fails on a fraction of episodes by running out the clock, not by crashing. Worth investigating in Phase 5b proper (likely correlated with sampled e_target in the high-end of [0, 0.05] or with adverse phase gaps), but doesn't block Step 1.

---

## What Step 1 establishes

### About the recipe

The Phase 4 recipe (LVLH obs, gated NHR shaping, warp action, Discrete(10), Phase 4 hyperparameters) **carries forward unchanged** to the Phase 5b task definition. The single env change — random sat init + same_orbit_init Stage 1 constraint — is sufficient to restore bootstrap. The recipe wasn't broken; it was specialized to a corner of the task space.

### About the audit's mechanism diagnosis

The Phase 5b audit hypothesized the failure mechanism was reward reachability, not shaping mis-specification. Step 1 confirms this:

- Reachability fix (random sat init + same_orbit_init) → recipe bootstraps.
- Shaping was never anti-correlated with the goal — σ₃'s deadness was state-distribution-dependent, not a structural flaw.
- The audit's recommendation to NOT do a wholesale shaping rewrite was correct.

### About the dashboard `perf` metric

At Stage 1.0 difficulty, training-time `perf` and held-out eval agree within 5pp. The 88pp gap from PHASE5a's A2 was a difficulty-cliff artifact: rolling-window noise on rare easy-task draws when the recipe was failing systematically. **At feasible difficulty, the metric is reliable.** Still recommended to track held-out eval as a secondary signal, but the metric isn't broken.

### About curriculum design

The four-stage curriculum proposed in `phase5b-step1.md` §2 (same-orbit → ω-match → transfer → fully general) has its first stage validated. Each stage progressively decouples one of (a, e, ω) between sat and target. Stage 1 worked on the first try, suggesting the decomposition is sensible. Phase 5b proper tests Stages 2-4.

---

## Phase 5b proper — proposed scope

With Step 1 landed, Phase 5b proper draft becomes concrete. Recommended scope:

### 1. Multi-seed validation of Stage 1.0 (~6 hr compute)

Per the addendum's variance-aware protocol: 5 training seeds × **40M steps** × Stage 1.0 conditions (updated from 20M after train-longer experiment). Goal: 99.7% mean is the new headline; we want to know inter-seed std. If std < 2pp, Stage 1.0 is robust; if > 5pp, recipe is fragile.

### 2. Stage 1.x — eccentricity expansion (~10 hr compute)

Within Stage 1, expand e_max from 0.05 → 0.10 → 0.20 → 0.30 → 0.50, success-rate-gated transitions (≥70% before advancing). 5 seeds at the final eccentricity.

Hypothesis: σ₃ activation will be the leading indicator of whether each sub-stage converges. If σ₃ stays active, recipe scales. If it drops, gate retune may be needed.

### 3. Stage 2.0 — ω-match rendezvous (~3 hr compute)

Constraint relaxation: sat.ω becomes random while sat.{a, e} = target.{a, e}. Tests whether the policy can match argument-of-periapsis through a maneuver, not just phase. Single-seed first.

### 4. Stage 3.0 — transfer rendezvous (~5 hr compute)

Constraint relaxation: sat.a random, sat.{e, ω} = target.{e, ω}. Hohmann + apsidal-correction territory. Multi-seed if Stage 2 succeeds.

### 5. Stage 4.0 — fully general (~5 hr compute)

All sat orbital parameters random. The deliverable Phase 5 was originally targeting.

### 6. REL_VEL_TOL annealing (conditional, ~2 hr if needed)

If any of Stages 1.x, 2, 3, 4 fail to bootstrap, implement REL_VEL_TOL annealing and re-run that stage. Conditional intervention.

### 7. Capability surface eval (~3 hr compute)

Grid over (target.a, target.e, target.ω, phase_gap), eval ≥600 eps per grid point per the addendum's variance protocol. Output: a heatmap of where the agent succeeds vs fails.

### 8. Attribution sub-experiments (~3 hr compute)

On the Stage 4 ckpt: strip-LVLH, strip-shaping, fix-circular-sat. Confirms which recipe components carry across the difficulty surface.

**Total Phase 5b proper estimate: ~50 hr compute over ~1-2 weeks elapsed** (updated from 35 hr after train-longer experiment showed 40M-per-stage is the right horizon). A realistic deliverable cycle.

### Engineering deferred — what the train-longer result removes from scope

The cap-tail probes (Probes #1, #2, #3 in `PHASE5b_STEP1_CAP_TAIL.md`) initially suggested mitigations like adaptive entropy, curriculum reweighting, two-pass training, and policy distillation. The train-longer experiment (extending to 40M steps) showed those aren't needed: under-convergence was the binding mechanism, not argmax brittleness. **Phase 5b proper drops these from the engineering plan.** Stratified e sampling remains a useful tool for Stage 1.x e_max scaling but isn't required for Stage 1.0.

---

## What's saved on disk

- `pufferlib/experiments/puffer_orbital_177741559081/` — §5.3 training run, 16 ckpts (every 10 epochs)
- `pufferlib/experiments/puffer_orbital_177741559081/model_puffer_orbital_000150.pt` — peak ckpt @ 84% held-out (canonical Stage 1.0 ckpt)
- `logs/orbital/p5b_step1_eval_seed{42,1337,20260423}/` — multi-rollout held-out evals
- `logs/orbital/p5b_audit_A1_e{000,005,010}/` — pre-Step-1 audit baselines (kept for comparison)
- `/tmp/p5b_step1_peak_scan.txt` — full ckpt scan
- `scripts/orbital/p5b_step1_rvg_diagnostic.py` — R-vs-G diagnostic tool
- `scripts/orbital/p5b_audit2_gate_activations.py` — Φ-port + activation analysis (reused at Step 1)
- `scripts/orbital/p5b_audit4_termination_modes.py` — termination classifier
- This file: `PHASE5b_STEP1_FINDINGS.md`

### Code state for commit

- `orbital.h` — struct fields `e_max_sat`, `same_orbit_init`; c_reset reorganized; sat init handles e≥0 via solve_kepler. Phase 4-compatible (defaults preserve old behavior).
- `binding.c` — two new kwargs plumbed through.
- `orbital.py` — two new constructor params, defaults match Phase 4.
- `orbital.ini` — defaults `e_max_sat = 0.0`, `same_orbit_init = 0`.
- `eval_checkpoint.py` — `--e-max-sat` and `--same-orbit-init` flags added.

REL_VEL_TOL annealing not implemented (deferred). All Phase 4 hyperparameters / recipe components unchanged.

---

## Status

| Step | Status |
|---|---|
| 1.A — peak ckpt scan | Done. epoch 150 @ 84% held-out. |
| 1.B — multi-rollout eval | Done. 81.7% mean (84.5/83.5/77.0). |
| 1.C — σ₃ check | Done. 45.8% activation (was dead). |
| 1.D — termination + R-vs-G | Done. 84.5% success, 15.5% cap, 0% else. |
| 1.E — annealing | Skipped (decisive without it). |
| 1.F — writeup | This document. |

Step 1's five-number deliverable is captured. Next action: commit the Phase 5b env changes, then draft Phase 5b proper using the proposed scope above.

---

*Step 1 lands cleanly. The Phase 5b env redefinition (random sat init + Stage 1 same-orbit constraint) restores bootstrap on the eccentric-rendezvous task and, as a free side effect, activates the dead σ₃ gate. Phase 5b proper can proceed as a multi-stage curriculum from this validated entry point.*
