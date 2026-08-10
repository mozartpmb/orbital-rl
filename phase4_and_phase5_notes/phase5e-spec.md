# Phase 5e — Code Investigation and Bound-Expansion Curriculum

> **Status:** 2026-04-30 (continued). Phase 5d Block II tested 2 reward-side interventions; both collapsed in the same pattern as Phase 5c's 6 prior reward-side interventions. The natural conclusion would be "recipe ceilings at e ≤ 0.10," but that conclusion is overcommitted: every failed intervention so far has been in the reward-side class. This spec pivots to two genuinely different intervention classes — env-level code investigation and bound-expansion curriculum — that test whether the wall is recipe-architectural or whether reward-side intervention design was the wrong attack surface.

---

## 0. The framing problem with declaring a ceiling now

After 8+ failed interventions across Phase 5c and Phase 5d Block II, the natural next step looks like "close Phase 5 with documented ceiling." But the ceiling claim has a structural issue: every tested intervention is in the reward-side class.

**Reward-side interventions tested:**
- B1: γ=1 for shaping
- B2: continuous gates
- B3: mixed-distribution training (the one partial success at 18%)
- B4: larger Δv actions
- B5: higher γ
- B6: REL_VEL_TOL annealing
- I4 alone: collision soft penalty
- I4+B3: stacked with mixed-distribution

All eight share a feature: they modify the per-step reward signal without changing the env physics, the task distribution structure, or the agent's action selection mechanism. PPO's response across all eight has been similar — destabilization or collapse of the warm-start.

**Untested intervention classes:**
- Env-level: is the env behaving correctly at e=0.20? Are tasks even physically winnable?
- Task-distribution-level: bound-expansion curriculum vs discrete stages
- Policy-level: action masking, demonstration bootstrapping, KL constraints

Phase 5e tests two of these (env-level investigation, bound-expansion curriculum) before considering the reward-side intervention class exhausted.

---

## 1. Two independent hypotheses

### 1.1 The env-might-be-broken hypothesis

We've never validated env behavior at high e the way we validated it at e=0 (the Phase 1 physics tests). The Phase 4 → Phase 5 progression assumed env correctness across the eccentricity range, but several env subsystems behave differently at high e:

- Kepler propagation (Newton-Raphson on Kepler's equation, well-known to converge slower at high e)
- Cartesian↔elements conversion (eccentricity vector geometry, true anomaly recovery)
- LVLH frame computation (rotates non-uniformly when chaser has e>0)
- Φ_orbit normalization (||Δē|| has different magnitudes at different e for "physically close" orbits)
- Episode termination (escape detection via E ≥ 0 may misfire at high-e periapsis-apoapsis swings)
- **Reset distribution physics (random sampling at high e may generate unwinnable tasks)**

The last one is the load-bearing concern. If random sat init at e=0.20 samples (sat, target) pairs where the minimum required Δv exceeds the fuel budget, those tasks are unwinnable from spawn. The policy can never solve them, no matter the recipe. This would explain:
- Why interventions don't help (can't solve unwinnable tasks)
- Why early-death dominates (escape/collision are the only "completions" of unwinnable tasks)
- Why B3 helps marginally (70% e=0.05 episodes are mostly winnable, providing clean signal anchor)

### 1.2 The bound-expansion-curriculum hypothesis

The Phase 5b two-stage curriculum (Stage 1 same_orbit_init=1 → Stage 4 same_orbit_init=0) introduces a discrete task-structure change that the policy has to bridge. The bridge appears to be where the warm-start fragility surfaces.

A continuous bound-expansion curriculum replaces the discrete flag with continuous distance bounds. Sat orbital state is sampled relative to target orbital state with bounds that expand smoothly as the policy improves:

- |sat.a − target.a| ≤ Δa_max, expanding from 1 km to ALT_MAX
- |sat.ω − target.ω| ≤ Δω_max, expanding from 1° to 180°
- e_max_sat = e_max_target = e_max, expanding from 0.05 to target

This is one continuous curriculum axis (bounds), not multiple discrete stages. The Phase 4-5 stage structure becomes a special case of a more general framework.

**Why this might help:**
- No discrete transition where warm-start fragility currently breaks
- Bound-expansion is a curriculum on action consequences (tight bounds → small possible state changes per action → less catastrophic-action risk)
- DORAEMON-style adaptive curriculum becomes single-axis (track success, expand bounds)
- The "almost-same-orbit" regime that Phase 4-5 never trained on becomes a natural training distribution

**Why this might not help:**
- If the env-might-be-broken hypothesis is right, bound-expansion doesn't fix unwinnable-by-construction tasks
- Bound-expansion adds complexity to env kwargs, with new hyperparameters (expansion schedule)
- The two-stage curriculum was at least *empirically validated* to bootstrap; bound-expansion's bootstrap behavior is unknown

---

## 2. Phase 5e structure

Three blocks, sequential:

- **Block I — Env-level code investigation (~3 hours).** Audits to verify env correctness at e=0.20. Most consequential check: Lambert reachability of randomly-sampled task instances.
- **Block II — Bound-expansion implementation and validation (~3-4 hours engineering + ~2 hours training).** Implement the bound-expansion env, verify it reproduces same_orbit_init at tight bounds, run a single-seed e=0.05 → e=0.20 training pipeline.
- **Block III — Decision and follow-up.** Based on Blocks I and II results, decide between (a) multi-seed validation of the working recipe, (b) further intervention testing, or (c) Phase 5 closure with documented ceiling.

Total ~10-12 hours compute. Smaller than Phase 5c or Phase 5d full budgets because Block I is mostly post-hoc analysis and Block II is bounded by single-seed validation.

---

## 3. Block I — Code investigation

### 3.1 Probe E1 — Lambert reachability check (most important)

For 200 randomly-sampled (sat_init, target) pairs at each of e_max ∈ {0.05, 0.10, 0.20, 0.30}, compute the minimum-fuel Δv to rendezvous via the Lambert problem. Compare to the chaser's fuel budget (15% fuel fraction → ~480 m/s Δv).

**The question:** what fraction of randomly-sampled tasks are unwinnable by minimum-fuel solution alone (i.e., minimum required Δv > fuel budget)?

**Implementation:** Write a Lambert solver for the 2D coplanar case (closed-form for circular endpoints; iterative for general endpoints). Apply to sampled (sat, target) pairs.

**Decision rule:**
- If <5% of e=0.20 tasks are unwinnable: the random-init distribution is fine; ceiling isn't from infeasibility.
- If 10-30% of e=0.20 tasks are unwinnable: significant fraction; explains some failures; reset distribution should be filtered.
- If >50% of e=0.20 tasks are unwinnable: the env is asking the impossible; this is the leading explanation for the ceiling.

**Cost:** ~1.5 hours including Lambert solver implementation. Possibly cheaper if PufferLib or another library has a Lambert solver to reuse.

### 3.2 Probe E2 — Kepler propagation precision at high e

Validate orbital stability over many periods at e ∈ {0.05, 0.10, 0.20, 0.50}. For each e:

1. Initialize an orbit at known (a, e, ω, θ).
2. Propagate forward 100 orbital periods using the env's Kepler step.
3. Check that orbital elements (a, e, ω) are unchanged within numerical tolerance.

**Why it matters:** if Kepler propagation has accumulating error at high e, the policy's expected-vs-actual state mismatch grows during long episodes. This wouldn't necessarily explain training collapse, but it would degrade policy execution.

**Decision rule:**
- |Δa|, |Δe|, |Δω| < 1e-6 after 100 periods: Kepler is fine.
- Errors growing measurably at high e: instrumentation issue; Newton-Raphson may need more iterations or better initial guess.

**Cost:** ~30 minutes.

### 3.3 Probe E3 — Cartesian↔elements round-trip at high e

For 100 sampled orbital states at e=0.20 (varying ω, θ), perform the round-trip: (a, e, ω, θ) → (x, y, vx, vy) → (a, e, ω, θ). Check that recovered elements match input.

**Why it matters:** if the round-trip has error at high e, the env's internal representation is inconsistent. The agent's actions (which are computed in inertial then converted back to elements) could be silently mis-applied.

**Cost:** ~15 minutes. Use existing C functions; just add a test harness.

### 3.4 Probe E4 — LVLH frame inspection at high e

Compute the LVLH-frame relative-state observation across one full orbital period for two representative configurations:

- Config A: sat.e = 0.05, target.e = 0.05, identical orbits (chaser at θ=0, target at θ=π)
- Config B: sat.e = 0.20, target.e = 0.20, identical orbits (same θ values)

Plot the relative-state observation across the period. Check for:

- Spurious oscillations not present in Config A
- Sign discontinuities (frame flipping)
- Saturation against normalization limits
- Non-physical magnitudes

**Why it matters:** if LVLH frame degrades at high e, the policy's input is partly noise — and no recipe modification fixes a corrupted observation.

**Cost:** ~30 minutes. May require adding LVLH logging hooks to the env.

### 3.5 Probe E5 — Φ_orbit calibration check

For a hand-designed "almost-rendezvous" state at e=0.20 (sat and target have identical (a, e, ω) and θ differs by 1°), compute Φ_orbit. Should be close to 0. If it's not, Φ_orbit's normalization is mis-calibrated for high e.

For a "very-far" state at e=0.20 (sat near target apoapsis, target near sat apoapsis, perpendicular ω), compute Φ_orbit. Should be large but bounded.

**Why it matters:** Φ_orbit is the dominant shaping component. If its dynamic range is wrong at high e (e.g., always 5+ even for near-rendezvous states), the σ₂ gate threshold (EPS_ORBIT = 2) is mis-tuned for high-e training.

**Cost:** ~15 minutes.

### 3.6 Probe E6 — Action effect at high e

The agent's prograde/retrograde actions apply Δv in the velocity direction. At periapsis, velocity is high; at apoapsis, it's low. The same discrete action produces different Δa depending on orbital position.

For a chaser at e=0.20:

- Compute Δa from a +25 m/s prograde burn at periapsis vs at apoapsis.
- Compute Δa from a +5 m/s prograde burn at periapsis vs at apoapsis.

**Why it matters:** if the action's effect varies by 10× across an orbit, the policy needs to learn position-dependent action selection. The current discrete action set might be ill-suited for this — at apoapsis, +25 m/s might be barely enough; at periapsis, the same burn might be too much.

**Cost:** ~15 minutes.

### 3.7 Block I deliverable

A document `PHASE5e_BLOCK_I_FINDINGS.md` covering all six probes. Each probe's outcome with concrete numbers. Headline finding: "the env is correct at e=0.20" or "the env has issue X at e=0.20." If the latter, the issue determines what comes next.

Pre-committed responses based on E1 outcome:

**E1 result A — <5% unwinnable.** Env is correct; the ceiling is recipe-architectural. Proceed to Block II as designed.

**E1 result B — 10-30% unwinnable.** Reset distribution filter is needed: reject task instances where minimum Δv exceeds, say, 2× the fuel budget. Re-train Phase 5b/5c interventions with the filter to see if the ceiling lifts.

**E1 result C — >50% unwinnable.** The env was asking the impossible. The ceiling claim was correct in observation but wrong in mechanism — it wasn't "recipe ceilings at e=0.10," it was "env distribution becomes mostly-impossible at e=0.10." Reset distribution filter is mandatory. Re-evaluate everything with filtered distribution.

---

## 4. Block II — Bound-expansion curriculum

### 4.1 Implementation

Replace the discrete `same_orbit_init` flag with continuous bounds:

```c
// In Env struct
float delta_a_max;      // max |sat.a - target.a| at reset
float delta_e_max;      // max |sat.e - target.e| at reset
float delta_omega_max;  // max |sat.omega - target.omega| at reset
```

Reset logic:

```c
// First, sample target orbit as before
target.a = sample_a();
target.e = uniform(0, e_max_target);
target.omega = uniform(0, 2*PI);

// Then sample sat orbit relative to target, bounded
sat.a = target.a + uniform(-delta_a_max, +delta_a_max);
sat.e = clamp(target.e + uniform(-delta_e_max, +delta_e_max), 0, e_max_sat);
sat.omega = target.omega + uniform(-delta_omega_max, +delta_omega_max);
sat.theta = uniform(0, 2*PI);  // phase always random
```

**Reproduction check:** at delta_a_max = 1 km, delta_e_max = 0.001, delta_omega_max = 0.01°, this should approximately reproduce same_orbit_init=1 behavior. Verify by training one Phase 5d-style run with these tight bounds; should reproduce Phase 5b's Stage 1.0 result (≥99% at e_max=0.05).

**Implementation cost:** ~2 hours. Modify orbital.h reset, add kwargs to binding.c and orbital.py, validate with a single training run that the tight-bound case reproduces same_orbit_init=1.

### 4.2 Single-seed bound-expansion training

Single-seed (seed 42) training pipeline:

```
Phase 1: e_max=0.05, tight bounds (Δa≤1km, Δω≤1°, Δe≤0.001) — should reach ≥99%
Phase 2: e_max=0.05, expanded bounds (Δa≤100km, Δω≤45°, Δe≤0.05) — track success
Phase 3: e_max=0.05, full random (Δa≤ALT_MAX, Δω≤180°, Δe≤0.05) — should match Stage 4.0
Phase 4: e_max=0.10, tight bounds — bootstrap eccentric same-orbit at higher e
Phase 5: e_max=0.10, expanded bounds — extend to fully random
Phase 6: e_max=0.20, tight bounds — the critical bootstrap test
Phase 7: e_max=0.20, expanded bounds — the deliverable test
```

Each phase is success-rate-gated: advance when held-out eval at the current bounds reaches ≥80% mean (or plateaus over 10 evals). Train-to-convergence per Principle I.

**Cost estimate:** Phases 1-3 should run quickly given the recipe is known to work at e=0.05. Phase 4 onward is uncharted; ~30-50M steps per phase realistic. Total ~3-5 hours of training.

### 4.3 Decision points within Block II

After each phase:

- **Reaches ≥80% at the phase's bounds:** advance.
- **Plateaus below 80%:** don't advance; investigate. Likely candidates: bound expansion was too aggressive; eccentricity ceiling within this curriculum class.
- **Collapses (peak then degrades):** known fragility; pause and consider whether Phase 5e Block I's E1 result explains it.

### 4.4 Block II deliverable

A document `PHASE5e_BLOCK_II_FINDINGS.md` reporting:

- Per-phase training trajectory (steps to convergence, final success rate)
- Bound-expansion behavior: smooth or discontinuous performance changes
- Comparison to Phase 5b/5c at equivalent task conditions
- Whether the e=0.20 ceiling is breached

**Pre-commit decision rule:**

- If Phase 7 reaches ≥50% at e=0.20: bound-expansion is the working recipe. Proceed to Block III multi-seed.
- If Phase 6 (e=0.20, tight bounds) reaches ≥50% but Phase 7 (full random) doesn't: bound-expansion enables eccentric phasing but not eccentric transfer. Different result; intermediate-stage strategy revealed.
- If Phase 4 (e=0.10, tight bounds) doesn't reach ≥80%: bound-expansion doesn't fix the higher-e bootstrap issue. The ceiling is real and survives this intervention class.

---

## 5. Block III — Conditional follow-up

Three branches based on Blocks I and II:

### 5.1 Branch A — Bound-expansion works, Block I shows env is fine

Multi-seed validation (5 seeds) of the bound-expansion recipe at e=0.20. If multi-seed lands ≥70% mean, eccentricity expansion to e=0.30, 0.50, 0.70 (Phase 5c stretch territory).

This is the working-Phase-5 outcome.

### 5.2 Branch B — Bound-expansion partially works (e=0.20 tight but not full random)

Implement Path C from the Phase 5d corrections doc: action masking with full mechanism. Test on the bound-expansion-Phase-7-stalled ckpt.

This combines bound-expansion (which addresses curriculum design) with action masking (which addresses early-death). If both classes are needed, this is where they stack.

### 5.3 Branch C — Block I reveals env issue (significant unwinnable fraction)

Implement reset distribution filter that rejects unwinnable task instances. Re-run Phase 5d and key Phase 5b experiments with filtered distribution.

The "ceiling" might lift dramatically once the env distribution is corrected.

### 5.4 Branch D — Nothing works

Both Block I and Block II reveal no actionable issues, and the recipe genuinely ceilings at e=0.10 across all tested intervention classes. Phase 5 closes with Phase 5b's deliverable + comprehensive ceiling attribution from Phase 5c, Phase 5d, and Phase 5e.

This is acceptable. The portfolio piece is the diagnostic depth.

---

## 6. Why Phase 5e is structurally different from Phase 5d

Phase 5d's Block II tested 7 reward-side interventions before the spec's "≤ 1 round of return to investigation" clause was triggered. Phase 5e steps outside that class:

- **Code investigation (Block I)** is post-hoc env validation, not training intervention.
- **Bound-expansion curriculum (Block II)** is task-distribution-level, not reward-side.

Both classes are genuinely orthogonal to what's been tested. If they don't crack the wall, we have evidence across three intervention classes (reward-side, env-level, task-distribution-level) that the ceiling is real.

---

## 7. The pre-committed acknowledgments

Phase 5 has now had:

- Phase 5b post-extend's structural-from-borderline-data overcommitment
- Phase 5c's Simpson's paradox in original analysis
- Phase 5d's reward-side intervention saturation

Phase 5e will likely have its own surprise. Pre-committed responses:

**Surprise A — Block I reveals an env issue I haven't predicted.** Investigate, fix, re-evaluate everything affected. Don't assume the issue's scope; map it.

**Surprise B — Bound-expansion bootstraps weirdly (e.g., works at tight bounds but collapses on first expansion step).** Diagnose with the same length-binning and failure-mode discipline that Phase 5d Block I established. Don't commit to mechanism stories from the first observation.

**Surprise C — Bound-expansion works trivially up to e=0.50 because the underlying issue was the discrete stage transition all along.** Embrace it. Validate multi-seed. The "ceiling" was a curriculum-design artifact.

**Surprise D — Lambert reachability check reveals random init at e=0.20 is mostly unwinnable.** This would re-frame the entire Phase 5c/5d work as "we were trying to train a policy on tasks that are physically unsolvable." Significant lesson; mostly-good news (the recipe wasn't broken; the eval was).

The discipline: report what's true, not what fits the narrative we're building.

---

## 8. What's locked, what's open

### Locked

- Random sat init in some form (specific bound-expansion implementation depends on Block II)
- Two-stage / bound-expanded curriculum from Phase 5b validated principle
- Train-longer null first when stages stall (Phase 5b Step 1)
- Length-binning before aggregating (Phase 5c corrections)
- No structural claims from <2σ data
- 5-seed multi-seed at headline stages
- Phase 4 condition eval at major stages
- Failure-mode partitioning before mechanism stories

### Open (Phase 5e will determine)

- Whether the env behaves correctly at high e (Block I, especially E1)
- Whether bound-expansion replaces or complements the discrete two-stage curriculum
- Whether reset-distribution filtering is needed (depends on E1 outcome)
- Whether action masking is still the right next intervention (depends on Block II outcome)

### Out of scope

- Phase 6 multi-body work (still Phase 5)
- Debris (still on user hold)
- New algorithmic interventions outside bound-expansion (unless Block I reveals env issues)

---

## 9. Compute and time budget

| Block | Activity | Compute | Wall |
|---|---|---|---|
| I | 6 probes, mostly post-hoc | ~3 hours | 1 day |
| II | Bound-expansion impl + 7 phases of training | ~5 hours | 3-4 days |
| III | Conditional, depends on results | ~5-15 hours | 3-7 days |
| **Total** | | **~13-23 hours** | **~1-2 weeks** |

Smaller scope than Phase 5c or 5d. Block I is fast post-hoc work; Block II's compute is bounded by the curriculum's per-phase training. Block III is the variable-cost component depending on which branch fires.

---

## 10. The deeper question Phase 5e implicitly asks

Phase 5d concluded "8 interventions failed; recipe ceilings at e ≤ 0.10." Phase 5e starts from "8 interventions in the same class failed; what about other classes?"

This is a calibration question about how to draw structural conclusions from a series of failed experiments. The lesson, if Phase 5e goes well: don't conclude "X is structurally impossible" from a series of failed interventions that share a common feature, until at least one intervention without that feature has been tested.

The lesson, if Phase 5e goes poorly: even orthogonal intervention classes don't crack the wall, and the recipe genuinely has a ceiling at the e ≤ 0.10 regime. Then "the recipe ceilings" becomes a defensible structural claim.

Either way, the experimental discipline produces stronger deliverables than premature closure.

---

## 11. Sequencing

In order:

1. **Block I E1 (Lambert reachability).** ~1.5 hours. Most consequential single check; might reframe everything.
2. **Block I E2-E6 (other env probes).** ~2 hours. Run in parallel with Block II prep.
3. **Block II implementation.** ~2 hours. Replace `same_orbit_init` flag with continuous bounds.
4. **Block II Phase 1-3 (validation at e=0.05).** ~1 hour. Confirms the bound-expansion env reproduces Phase 5b results.
5. **Block II Phase 4-7 (e=0.10 → e=0.20 expansion).** ~2-3 hours. The actual test.
6. **Block III (whichever branch).** ~5-15 hours.
7. **Phase 5 closure writeup.** ~1 day.

If at any point during Block II the curriculum is clearly stalling (e.g., Phase 4 plateaus below 50% after 50M steps), pause and reassess before continuing through Phases 5-7.

---

## 12. What this spec is NOT

- Not a Phase 6 plan. Phase 5 stays open.
- Not a commitment to find a working agent at e=0.20. If both intervention classes (env-level + task-distribution) fail, that's strong evidence for the ceiling.
- Not a redo of Phase 5d. The Phase 5d failure-mode partition (Block I) and the Phase 5d intervention findings (Block II) are inputs to Phase 5e, not work to be repeated.
- Not a complete intervention exhaustion plan. Action masking, demonstration bootstrapping, and other policy-level interventions remain as Phase 5f possibilities if Phase 5e doesn't crack the wall.
- Not a portfolio writeup. That waits for Phase 5 closure.

---

*Author: 2026-04-30. Phase 5e spec, after Phase 5d Block II's 2/2 reward-side intervention failures. Pivoting to env-level investigation and task-distribution-level intervention, the two intervention classes most likely to be missing from the recipe-ceiling argument.*
