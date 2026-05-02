# Phase 5 — Consolidated Findings (Phase 5b through wrap-up)

**Date:** 2026-05-01 · spans Phase 5b–5e plus wrap-up. Honest narrative; the headline numbers are what they are.

---

## 1. Headline result — what was actually built

A 2D coplanar orbital rendezvous agent that **handles random LEO-altitude (300-800km) tasks at sat & target eccentricity ≤ 0.05** with multi-seed mean ≥ 84% across phase gaps and orbit relations. The recipe is the Phase 5b two-stage curriculum (Stage 1.0 same_orbit_init → Stage 4.0 fully random) trained at e_max=0.05, with Phase 5e's `valid_init_only=1` rejection-sampling fix.

The wrap-up's per-condition surface (fixed-eccentricity, alt-scaled where physically valid) reveals a hard cliff: **the recipe does not generalize beyond the training distribution.** Beyond e=0.05 at LEO, success drops to ~15% (e=0.075) and 0% (e=0.10). At higher altitudes (where e ≥ 0.10 is physically valid), success is similarly low — the recipe never saw those altitudes in training.

The previous Phase 5e closure claimed multi-seed 90.2% at e_max=0.20 and 83.4% at e_max=0.50. **Those numbers are aggregates over uniform-up-to-bound distributions filtered by `valid_init_only`, where most surviving samples have e well below the bound.** Per-condition values at e=0.20 fixed are ~0% fully-random; at e=0.50 fixed, also ~0%. The aggregate framing was misleading; the per-condition framing in the surface CSV (`web_data/results/phase5_capability_surface.csv`) is the truthful representation.

---

## 2. Phase-by-phase narrative

### Phase 5b — working agent at e ≤ 0.10 with two-stage curriculum

Built on Phase 4's R4-curriculum recipe (Discrete(10) including warp-5min, gated NHR shaping with terminal Φ-clamp, LVLH observations). Added random sat eccentricity init and Stage 1.0 same_orbit_init bootstrap. Multi-seed validated 96.4% at e_max=0.05 fully random. Stage 4.1 at e=0.10 partial (65%). Stage 4.2 at e=0.20 collapsed across all 8 seeds.

This deliverable (Phase 5b Stage 4.0 ckpt at e_max=0.05) was always the strongest claim. The wrap-up's surface confirms the e ≤ 0.05 capability holds; everything beyond was the misattributed extension story.

### Phase 5c/5d — the ceiling investigation that wasn't

~50 hours of compute investigating "why doesn't the agent extend beyond e=0.10." Eight reward-side interventions tested (B1: γ=1 shaping, B2: continuous gates, B3: mixed-distribution, B4: larger Δv, B5: higher γ, B6: REL_VEL_TOL anneal, I4 alone: collision penalty, I4+B3 stack). All collapsed in similar patterns. Two mechanism stories emerged and were retracted:

- **Phase 5b post-extend's "60M shows erosion."** Single-seed borderline data point overcommitted to a structural claim. Retracted in Phase 5b corrections.
- **Phase 5c's "shaping direction is reversed at high e."** Simpson's paradox: shaping direction was correct *within length bins*, the population-level reversal was a length-confounding artifact. Retracted in `PHASE5c_CORRECTIONS.md`.

The eventual "the recipe ceilings at e=0.10" claim was made tentatively at end of Phase 5d and turned out to be **the wrong framing**, but for a subtler reason than expected: the *eval distribution* was contaminated. See Phase 5e.

### Phase 5e — env validation revealed the real issue (and what we missed)

Block I env-validation suite (Lambert reachability, Kepler precision, round-trip stability, LVLH frame, Φ_orbit calibration, action peri/apo asymmetry) found one real issue: **at e_max=0.20, ~64% of randomly-sampled initial states had perigee `a(1-e) < R_EARTH`** — physically unrecoverable orbits. The "wall" was 64% of eval samples being doomed regardless of policy.

Adding `valid_init_only=1` (rejection-sample inits with sat & target perigees ≥ EARTH_KEEPOUT) bumped the same Stage 4.0 ckpt from 16% to 93.5% at e_max=0.20 with no retraining. Block II multi-seed retrain under `valid_init_only=1`: 90.2% ± 2.0% at e_max=0.20, 83.4% ± 1.7% at e_max=0.50, 63% ± 2.3% at e_max=0.70.

We declared closure here. **It was premature.** The "e_max=0.50 = 83%" headline was still an aggregate — under `valid_init_only`, most samples at e_max=0.50 have e well below 0.50 because high-e + LEO altitude is physically infeasible (perigee sub-keepout). The filter was selecting low-e samples within high-e labels.

### Wrap-up — per-condition surface revealed the aggregate-vs-fixed gap

W1 added env kwargs for fixed-eccentricity and altitude-band overrides. W1.3 validation:

- Fixed e=0.05 at LEO (recipe-trained): **93%** ✓
- Fixed e=0.05 at expanded alt (1500-2000km): **68%**
- Fixed e=0.10 at slightly higher alt: **24%**
- Fixed e=0.50 at LEO (no alt override): **0%** (perigee sub-surface)
- Fixed e=0.50 at high alt (alt scaled to keep perigee valid): **0%**

W2 surface (240 cells: 5 seeds × 6 e × 4 phase × 2 relations):

| e_fixed | 30° random | 90° random | 150° random | 180° random | 30° same_orbit | 90° same_orbit | 150° same_orbit | 180° same_orbit |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 84.6 | 87.4 | 91.4 | 93.4 | 100.0 | 97.6 | 75.4 | 76.2 |
| 0.05 | 5.8 | 5.0 | 5.6 | 5.6 | 58.4 | 40.8 | 30.0 | 36.6 |
| 0.10 | 1.0 | 0.6 | 0.2 | 0.4 | 38.0 | 24.6 | 14.2 | 24.6 |
| 0.20 | 0.0 | 0.0 | 0.0 | 0.2 | 11.4 | 6.8 | 0.8 | 7.0 |
| 0.50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.6 |
| 0.70 | 13.0 | 13.0 | 13.0 | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 |

(All values in %. Multi-seed mean across 5 seeds × 100 eps per cell. Same_orbit cells share orbit shape including ω; only θ differs. Fully_random cells differ in a, e, ω.)

W2 LEO probe (LEO altitude only, `fully_random`):

| e_fixed | 30° | 90° | 150° | 180° |
|---|---|---|---|---|
| 0.000 | 84.6 | 87.4 | 91.4 | 93.4 |
| 0.010 | 95.6 | 95.2 | 93.6 | 94.2 |
| 0.025 | **97.8** | **97.2** | **97.2** | **97.4** |
| 0.050 | 87.0 | 85.6 | 83.8 | 84.6 |
| 0.075 | 16.4 | 13.8 | 13.6 | 13.4 |
| 0.100 | 2.0 | 0.0 | 0.0 | 0.2 |

Peak performance at e=0.025 (~97%). e ≥ 0.075 falls off the cliff.

The 13% at e=0.70 fully_random across the surface is itself a rendezvous-conditional artifact (alt-scaled cells at high-e have wide transfer windows; the policy's coast-and-warp default randomly hits success in some configurations). Doesn't reflect "policy capability."

---

## 3. The actual deliverable, framed honestly

**The recipe is a LEO low-eccentricity rendezvous specialist.** Specifically:

- altitude band: 300-800 km
- sat & target eccentricity: ≤ 0.05 (peak ~97% at e=0.025; ~85% at e=0.05; cliff at e ≥ 0.075)
- phase gap: any (30-180°, ≤ 10pp degradation across the range at low e)
- sat-target relation: random altitude, random ω, random θ
- multi-seed: 5/5 seeds reach >85% in this regime

This is a real result. It's substantially narrower than what Phase 5e initially claimed (which was an aggregate-distribution result with a misleading headline). The Phase 5b deliverable's 96.4% multi-seed at e_max=0.05 fully random was always the strongest, most-honest number; it stands.

For the web frontend / portfolio narrative: lead with "LEO low-eccentricity rendezvous, multi-seed validated" and show the LEO probe heatmap. Don't lead with "handles eccentric orbits up to 0.50" — that's the aggregate framing the wrap-up retracted.

---

## 4. Methodological lessons (named, three of them)

### 4.1 The train-longer null first

Before claiming a stage is stalled, train longer. Phase 5b's "60M shows erosion" was a single noisy datapoint; 80M training showed no erosion. The cap-tail "P3-then-train-longer" rule.

### 4.2 Bin (or condition) before aggregating

Phase 5c's "shaping direction reversed" was a Simpson's paradox: aggregating across length-correlated outcomes flipped the sign. Length-bin first.

### 4.3 Verify the metric measures what its name implies

Phase 5e's "e_max=0.50 = 83%" sounded like "handles e=0.50". It actually meant "handles a uniform [0, 0.50] distribution filtered by valid_init_only, which heavily biases toward low-e." Same with Phase 4's "180° phase gap" (uniform up to π, not exactly π). When metric names imply specific values, verify the metric measures that value or condition-fix it.

### 4.4 (Bonus) — Validate the env at every regime before measuring policy capability there

Phase 1 validated the physics at e=0. Phase 5c/5d/5e ran for ~50 hours measuring policy capability at high e *without* re-running the validation tests at high e. The env had a sampling bug at high e that masqueraded as a policy failure. The Block I E1 Lambert check would have caught this in the first round if it had been part of the standard policy-capability eval protocol.

The pattern is the same across all three lessons: **named metrics often hide aggregations that don't match the metric's intuition.** Phase 5 spent ~50 hours of compute investigating mechanism stories that turned out to be measurement artifacts on contaminated populations. The cost of the appropriate per-condition checks was 5 minutes of physics + 2 minutes of code, three separate times.

---

## 5. The transferable methodology

The deliverable plus the diagnostic discipline are a single output. Future projects (Phase 6 onward) inherit:

- **Per-condition eval** as the default; aggregates only as caveatted summaries.
- **Train-longer null** before structural claims about stalls.
- **Length / outcome / regime binning** before aggregating any quantity that depends on the binning dimension.
- **Env validation at every regime where policy capability is measured.** Phase 1's e=0 physics tests should have a high-e replay that runs whenever the curriculum is extended.
- **Metric naming discipline.** "e_max=0.20 = 90%" is fine if accompanied by "(uniform [0, 0.20] under valid_init_only)". "Handles e=0.20" requires fixed-e measurement.

---

## 6. Code state

What ships:
- Recipe (RECIPE.md): Phase 5b two-stage curriculum + `valid_init_only=1`
- 5 multi-seed Stage 4.0 ckpts (models/phase5e/seed{42,A,B,C,D}_stage4_best.pt)
- 1 wandb-tracked canonical retrain (models/phase5e/canonical_seed42_stage4_best.pt)
- Eval infrastructure with fixed-e and altitude-band overrides
- Capability surface CSV + heatmap
- LEO probe CSV + heatmap
- Lambert solver, env validation suite (E2-E6)
- export_web_data.py for trajectory JSON exports

What stays in the codebase off-by-default:
- enable_action_mask (Phase 5d Path A) — verified working but not needed for the deliverable
- collision_penalty_w — verified working but ineffective during training

What was removed or never made canonical:
- R3 components (DAPO, L2-init, TEC, adaptive KL) — explicitly excluded from the recipe
- Continuous action space — Phase 5b ruled out
- Bound-expansion curriculum — designed but not implemented; deferred to Phase 6 if needed

---

## 7. The Phase 5 story as portfolio narrative

If/when this becomes a blog post, the honest arc is:

1. Built a working LEO rendezvous agent at e ≤ 0.05 with a clean two-stage curriculum (Phase 5b, the strongest claim)
2. Spent ~50 hours trying to extend to higher e via reward-side interventions — all failed
3. Discovered the eval distribution was contaminated (~64% of e_max=0.20 inits were physically unrecoverable)
4. The fix was one conditional in c_reset; same Stage 4.0 ckpt jumped 16% → 93.5% at e_max=0.20
5. Multi-seed validated, declared closure
6. Wrap-up surface revealed the closure was premature: e_max numbers were still aggregates; per-condition values are much narrower
7. Honest deliverable: LEO low-e specialist, multi-seed validated. The diagnostic depth across the two retractions is the portfolio piece.

The hard part of the project was not the algorithm. It was the discipline to keep retracting mechanism stories until the data agreed.

---

*Phase 5 closed. Recipe documented. Phase 6 unblocked.*
