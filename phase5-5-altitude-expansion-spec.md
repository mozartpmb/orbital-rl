# Phase 5.5 — Altitude Expansion Spec

> **Status:** 2026-05-02. Written after the Phase 5 verification investigation revealed that the recipe's apparent eccentricity ceiling was confounded with altitude OOD: at LEO altitudes (300-800 km, the trained band), eccentricity is geometrically bounded to e ≤ 0.084. Achieving the original Phase 5 ambition ("any orbit to any orbit at any eccentricity") requires training the recipe at higher altitudes where high-e is physically valid. This spec scopes that work, with explicit lessons-learned from Phase 5b-e and pre-committed protocols against the failure patterns those phases established.

---

## 0. Why this spec exists

### 0.1 The honest framing

After Phase 5e and the verification investigation, the recipe is:
- Working at LEO 300-800 km, e ≤ 0.05 fully random — Phase 5b's deliverable, multi-seed at 96.4% (3/5 successful seeds).
- *Unknown* at higher altitudes, because it's never been trained there.
- *Bounded* at LEO eccentricities by physics (e ≤ 0.084 max at LEO altitudes).

The original Phase 5 ambition was a two-body any-orbit any-eccentricity rendezvous agent. With the current LEO-bound recipe, that ambition is geometrically unreachable: high-e orbits don't exist at LEO altitudes.

Either we accept the LEO deliverable as Phase 5's output, or we extend training to altitudes where high-e is physically valid. Phase 5.5 is the latter.

### 0.2 What this spec is not

- **Not Phase 6.** Phase 6 is multi-body (cislunar, n-body integration). Phase 5.5 is two-body altitude/eccentricity expansion.
- **Not a clean restart.** It builds on Phase 5b's working recipe, the env-fix from `phase5-env-fix-spec.md` (prerequisite), and the methodological infrastructure from Phases 5b-e.
- **Not a guaranteed success.** The recipe might not extend to MEO altitudes without significant changes. The pre-commits in §10 explicitly cover this.

### 0.3 Prerequisites

1. `phase5-env-fix-spec.md` landed and validated. Specifically: cap raise to 4096+, realized-init metadata, gave-up handling, validation instrumentation.
2. Phase 5b's working recipe artifacts: the canonical Stage 4.0 ckpts (seeds 31415, 42, 20260423).
3. The methodological infrastructure: `eval_checkpoint.py` with full env-kwarg support, multi-seed orchestrator, capability surface eval pipeline.

---

## 1. What's known about altitude regime changes

The Phase 4-5 recipe was trained at LEO with a 60s timestep, a 5-min warp action, and Δv discretization (5/10/25 m/s prograde, ±10 m/s radial). Several things change at higher altitudes:

### 1.1 Physics that's known to scale

**Orbital periods scale as a^(3/2).** At a=10000 km (MEO ~3600 km altitude), period is ~166 min, ~2× LEO. At a=42164 km (GEO), period is 1436 min, ~16× LEO.

**Orbital velocities scale as a^(-1/2).** At MEO, v_circ is ~4500 m/s vs LEO ~7700 m/s. At GEO, v_circ is ~3074 m/s.

**The same Δv produces different Δa at different altitudes.** From vis-viva, Δa ≈ 2a²·Δv/μ for small Δv. So at 2× the altitude, the same Δv produces ~4× the Δa. The Δv discretization tuned for LEO produces gross over-shoots at MEO/GEO.

**Action effects at peri vs apo asymmetry grows with e.** Phase 5e Block I E6 measured 1.5× at e=0.20, 3× at e=0.50. At MEO with e=0.50, both effects compound: large Δa per Δv, high asymmetry.

### 1.2 Recipe parameters that may need adjustment

| Parameter | LEO value | Concern at MEO/HEO |
|---|---|---|
| `DT` (timestep) | 60s | At MEO, 60s is 0.6% of period (vs 1.1% at LEO). May be too fine. |
| `WARP_DT` | 300s (5 min) | At GEO, 300s is 0.35% of period. Need 1-hour or 6-hour warp. |
| `MAX_STEPS` | 2000 | At GEO, 2000 steps × 60s = 33 hr ≈ 1.4 GEO orbits. Too short for some maneuvers. |
| Δv discretization | 5/10/25 m/s | Produces overshoots at high altitude. May need 1/2/5/10/25 or different scaling. |
| `ALT_MAX` (shaping norm) | LEO range | Φ_orbit normalization will saturate or distort at MEO. |

### 1.3 Physics that doesn't scale

- Kepler's equation Newton-Raphson convergence is fine at any altitude (validated up to e=0.50 in Block I E2).
- Cartesian↔elements round-trip is fine (Block I E3).
- Gated NHR shaping logic is altitude-agnostic in form, but its tuning depends on Φ_orbit's range.

### 1.4 What's genuinely unknown

- Whether the recipe's curriculum (warm-start from Stage 1.0) transfers to higher altitudes.
- Whether multi-altitude training requires per-altitude adapters or works uniformly.
- Whether the warp action's "skipping" behavior creates new failure modes at long-period orbits.
- Whether the action discretization mismatch causes training collapse, plateaus, or graceful degradation.

These are the questions the pre-experiments below answer before committing to a full curriculum.

---

## 2. Pre-experiments: probe before commit

The pattern from Phase 5c/5d/5e: committing to a long curriculum without first validating env behavior produced 50+ hours of compute investigating what turned out to be env bugs. Phase 5.5 starts with cheap probes.

### 2.1 P1 — Env validation suite at high altitude

Re-run Phase 5e Block I's six probes (E2 Kepler, E3 round-trip, E4 LVLH, E5 Φ_orbit, E6 action effects) at altitudes a ∈ {7000, 12000, 26000, 42000} km. Each probe is ~15-30 min.

**Specifically check at MEO/GEO:**
- Kepler precision over 100 periods at high altitude (does Newton-Raphson convergence behave the same when periods are 16× longer?)
- LVLH frame at GEO (does the rotating-frame computation handle long-period orbits?)
- Φ_orbit at MEO with e=0.30 (where ω-mismatch term can be larger; does Φ saturate or behave?)
- Action effect at MEO: 25 m/s prograde at a=10000 km. Compare to LEO. (Expected ~4× larger Δa.)

**Total cost:** ~3 hours, no training.

**Pre-committed responses:**
- All probes pass: env is correct at high altitude. Proceed to P2.
- Probe fails: identify and fix before any training. The Phase 5e pattern.
- Probe shows scaling concerns (e.g., warp underflows long-period orbits): note, plan for env adjustment in §3.

### 2.2 P2 — Phase 5b ckpt zero-shot at high altitude

Eval the canonical Phase 5b Stage 4.0 ckpt (seed 31415) at:
- LEO baseline (control): a ∈ [6671, 7171] km, e ∈ [0, 0.05]
- MEO: a ∈ [10000, 11000] km, e ∈ [0, 0.05]
- MEO-eccentric: a ∈ [10000, 11000] km, e ∈ [0, 0.30]
- GEO: a ∈ [42000, 42500] km, e ∈ [0, 0.05]
- GEO-eccentric: a ∈ [42000, 42500] km, e ∈ [0, 0.30]

200 episodes per cell × 3 rollout seeds. With env-fix's high cap, no contamination. ~30 min compute.

**Pre-committed responses (this is informational, not deciding):**
- LEO baseline reproduces 96.4%: confirms the ckpt and pipeline are correct.
- MEO at low-e: gives us a starting point. If 0%: recipe is altitude-specific. If non-trivial (>20%): some altitude-generalization exists.
- MEO/GEO at high-e: tells us how much of the agent's "skill" transfers across altitudes.

This isn't the deliverable; it's data for designing the curriculum.

### 2.3 P3 — Action effect calibration

For each of altitudes {LEO, MEO, GEO}, compute the Δa produced by each Δv action in the current Discrete(10) set. Tabulate. Compare to the Φ_orbit gate threshold and the rendezvous tolerance.

**The question:** at MEO, does 5 m/s prograde produce too small a Δa to be useful (lost in Φ_orbit's bin width)? Does 25 m/s produce too large a Δa (overshoots target)?

**Pre-committed responses:**
- Existing Discrete(10) is adequate at MEO/GEO: continue with current action space.
- Δv overshoots at MEO/GEO: introduce smaller Δv actions (1/2 m/s) for fine control.
- Δv undershoots at MEO/GEO: introduce larger Δv actions (50/100 m/s).
- Both: redesign to be altitude-relative (Δv / v_circ) — bigger change, may require its own validation.

### 2.4 P4 — Warp duration adequacy

Run a single coast trajectory at GEO. Measure how many warp-5min actions the current ckpt takes before doing anything else.

**Pre-committed responses:**
- < 10 warps: current 5-min warp is fine.
- 10-50 warps: marginal. Consider adding 30-min or 1-hour warp.
- > 50 warps: current warp is structurally inadequate at GEO. Add longer warps before training.

### 2.5 What P1-P4 produce

A `PHASE5_5_PRE_FINDINGS.md` document covering:
- Env correctness at high altitude (pass/fail per probe).
- Existing recipe's transfer behavior across altitudes (informational).
- Required env changes before training (concrete list, prioritized).

Total cost: ~4 hours including writing.

---

## 3. Env modifications (probably needed)

Based on P1-P4 findings, expect to need some subset of:

### 3.1 Altitude bounds as kwargs (definitely needed)

The hardcoded `300e3 + (rand/RAND_MAX) * 500e3` in orbital.h must accept the existing `a_min_override`/`a_max_override` kwargs (they exist but aren't wired through). This was already partially infrastructure; finish it.

### 3.2 Action discretization (likely needed)

If P3 shows Δv overshoots at MEO, add smaller actions. If undershoots, add larger. The existing Discrete(10) actions are positional in the action vector (action 0 = coast, 9 = warp-5min); adding actions requires either weight surgery on existing ckpts (Phase 4.5 pattern) or fresh training from a no-action-change ckpt.

**Risk note:** Phase 4.5 Block C found that cross-phase action-table surgery doesn't transfer (D7→D10 from Phase 3 ckpts collapsed). If we add actions, we need to test whether warm-start from Phase 5b ckpts works with the new action set. May need fresh training.

### 3.3 Warp action duration (probably needed)

If P4 shows GEO needs longer warps, add a warp-1hr or warp-6hr action. Same surgery considerations as 3.2.

### 3.4 Φ_orbit normalization (possibly needed)

If P1 shows Φ_orbit saturates at MEO, the normalization (currently `|Δa|/ALT_MAX + ||Δē||`) needs to scale with altitude. Options:
- Per-episode normalization: divide Δa by `target.a` instead of `ALT_MAX`.
- Multi-altitude `ALT_MAX`: scale with the max altitude in the curriculum.

Risk: changing the shaping changes the recipe. Phase 5b's recipe was explicitly tuned with the current normalization. Modifying it might disrupt training even at LEO.

### 3.5 MAX_STEPS expansion (probably needed)

GEO orbits at 60s timestep need more steps for some maneuvers. Either expand MAX_STEPS to ~10000 (needs more memory for trajectory log) or add altitude-dependent step limits.

---

## 4. Curriculum design

This is the core of Phase 5.5. Three options, with pre-committed evaluation against the patterns Phase 5b established.

### 4.1 What we learned from Phase 5b's curriculum

- **Intermediate "specialist" stages erode breadth.** Stage 1.3 (e=0.5 same-orbit) gave 5.2% Stage 4 OOD vs Stage 1.0's 43.6%. Specialization at hard tasks hurts subsequent transfer.
- **Two-stage curriculum (Stage 1.0 → Stage 4.0 directly) outperformed four-stage.** Skipping intermediate Stage 1.x was the right call.
- **Curriculum is required, not just helpful.** Block B showed fresh training at Stage 4.0 collapses (0%); only warm-from-Stage-1.0 produces working agents.

These lessons constrain the curriculum design for altitude expansion.

### 4.2 Option A — Multi-altitude sampling from start

Train with altitudes sampled from a wide range from the start (e.g., a ∈ [6671, 26000] km), no curriculum stages, no warm-start.

**Pros:**
- Simplest design.
- No specialization erosion (no specialist stages).
- The policy learns altitude-invariant skills if such skills exist.

**Cons:**
- Phase 5b Block B showed fresh training at Stage 4.0 (LEO, fully random) collapses. Adding altitude variance probably makes it harder, not easier. Likely 0% at start.
- No bootstrap mechanism.

**Likelihood of success:** Low. The Phase 5b finding "curriculum is required" probably extends to "curriculum is even more required when expanding scope."

### 4.3 Option B — Altitude bound expansion (analogous to e_max expansion)

Start with Phase 5b's working LEO ckpt. Expand altitude bounds incrementally:
- Stage 5.5.0: warm-start from Phase 5b. Re-validate at LEO with env-fix.
- Stage 5.5.1: a ∈ [6671, 8500] km (LEO + slight extension), e_max=0.05. Train to convergence.
- Stage 5.5.2: a ∈ [6671, 12000] km (LEO + low MEO), e_max=0.10. Train to convergence.
- Stage 5.5.3: a ∈ [6671, 26000] km (LEO + full MEO), e_max=0.30. Train to convergence.
- Stage 5.5.4: a ∈ [6671, 42500] km (LEO + MEO + GEO), e_max=0.50. Train to convergence.

Eccentricity expands in concert because the altitude expansion makes higher e physically valid.

**Pros:**
- Builds on working LEO ckpt, no fresh-training collapse risk.
- Each stage is incrementally further from training distribution; gradient stays useful.
- Mirrors Phase 5b's working two-stage pattern (warm-start at one level, expand to next).

**Cons:**
- Phase 5b's lesson: each stage's specialization may erode breadth needed for the next. Stage 5.5.2 might give a worse warm-start for Stage 5.5.3 than Stage 5.5.1 would.
- The "broader earlier stage as better warm-start" hypothesis from Phase 5b suggests we might want to skip stages.

**Mitigation for the breadth-erosion risk:**
- Eval each Stage 5.5.x ckpt at the *next* stage's distribution before training Stage 5.5.x+1. If Stage 5.5.1 ckpt gives meaningful Stage 5.5.3 OOD performance, consider skipping Stage 5.5.2.
- Test the "early peak ckpt as warm-start" hypothesis: at each stage, try warm-starting Stage X+1 from both Stage X's converged ckpt AND from Stage X's mid-training ckpt (e.g., 50% through). The early ckpt may transfer better.

**Likelihood of success:** Medium-high. The altitude bound is a continuous variable like e_max; the bound-expansion pattern worked for Phase 5b's e_max curriculum.

### 4.4 Option C — Altitude-mixed sampling with bias schedule

Train with altitudes sampled from a wide range from the start, but with a sampling distribution that's initially biased toward LEO and gradually shifts toward uniform across the full range.

E.g., at training start: P(LEO) = 0.9, P(MEO) = 0.1, P(GEO) = 0.0.
Mid-training: P(LEO) = 0.5, P(MEO) = 0.3, P(GEO) = 0.2.
Late-training: P(LEO) = 0.33, P(MEO) = 0.33, P(GEO) = 0.34.

This is closer to DORAEMON-style adaptive curriculum than discrete stages.

**Pros:**
- No "specialization at intermediate stages" because there are no stages.
- Smooth transition; gradients stay useful throughout.
- Can warm-start from Phase 5b's LEO ckpt.

**Cons:**
- More complex to implement (sampling distribution as a function of training step).
- Hyperparameter design space is larger (the bias schedule).
- Hard to debug if it doesn't work; what's the "stage" that's failing?

**Likelihood of success:** Unknown. It's a different paradigm than Phase 5b's discrete stages, so its failure modes are different.

### 4.5 Recommended choice: Option B with breadth-preservation discipline

Use bound-expansion stages, but apply Phase 5b's lessons:
- Eval each stage at the next stage's distribution (OOD check) before committing to that stage's converged ckpt as warm-start.
- Test mid-training ckpts as warm-starts in addition to converged ckpts.
- Skip intermediate stages if direct warm-start from an earlier stage gives comparable OOD performance.

This is essentially "Phase 5b's two-stage pattern, applied to altitude expansion, with explicit testing of stage-skipping."

### 4.6 Pre-committed budget per stage

Following Phase 5b: 30-50M steps per stage. Use Step 1's "train-longer null" — if a stage plateaus, run another 20M before declaring it stalled. Multi-seed at the headline stages (5 seeds), single-seed for intermediate exploration.

---

## 5. Validation protocol

### 5.1 Per-stage validation

For each Stage 5.5.x:
- Held-out eval at the stage's training distribution (200 eps × 3 rollout seeds × 5 training seeds = 3000 eps).
- OOD eval at Phase 4 conditions (LEO, e=0). Confirms no regression on the original task.
- OOD eval at the next stage's distribution. Tells us how good a warm-start the stage's ckpt is.
- Diagnostic: trajectory plots from 3-5 successful and 3-5 failed episodes per stage. Visual inspection for mechanism.

**Decision criterion:** advance to next stage when held-out at current stage ≥ 75% multi-seed mean. Don't advance on weaker results — fix or document first.

### 5.2 Multi-seed at headline stages

Run 5-seed retrains at:
- Stage 5.5.0 (re-validate Phase 5b in fixed-env): 5 seeds × 20M steps. Confirms baseline.
- Stage 5.5.4 (final deliverable): 5 seeds × 50M steps. Confirms recipe's actual capability at the full altitude/eccentricity range.

Headline numbers: `mean ± std (n_seeds_succeeded / n_seeds_total)` per Phase 5b's bimodality lesson.

### 5.3 Final capability surface

Once the curriculum is complete, run the proper capability surface across:
- Altitude bands: LEO (300-800 km), MEO (3000-5000 km), MEO-high (10000-12000 km), GEO (42000-42500 km).
- Eccentricities: physically valid per altitude band. (LEO: e ≤ 0.08. MEO: e ≤ 0.30. MEO-high: e ≤ 0.50. GEO: e ≤ 0.85.)
- Phase gaps: 30°, 90°, 150°, 180°.
- Sat/target relations: same_orbit, fully_random.

This is finally a meaningful "any orbit to any orbit" surface, not contaminated by altitude OOD.

Cell count: ~4 alt bands × ~4 e levels (per alt) × 4 phases × 2 relations = ~128 cells. 5 seeds × 600 eps each = ~400K episodes. ~4 hours compute.

### 5.4 Phase 5.5 success criteria

- **Soft success:** Stage 5.5.4 reaches ≥ 60% multi-seed mean across the full altitude range with e_max scaled per altitude. Final surface shows graceful degradation, no catastrophic corners.
- **Hard success:** Stage 5.5.4 reaches ≥ 80% multi-seed mean. Surface looks uniform across altitudes.
- **Documented limit:** Stage 5.5.x stalls at altitude X. The "recipe extends to X but not beyond" is a publishable result with mechanism attribution from the stage's collapse pattern.

---

## 6. Methodological discipline (carried forward from Phases 5b-e)

### 6.1 Train-longer null

Before declaring any stage stalled, run +20M steps. Multiple Phase 5b iterations confirmed this resolves "P1/P2/P3 mechanism" speculation. Cheapest control experiment in the project.

### 6.2 Length-bin / condition-bin before aggregating

Any analysis that aggregates across episodes (mean Δv, success rate, terminal Φ) must bin by relevant covariates first. Phase 5c's Simpson's paradox reversal came from population-level aggregation hiding within-bin truth.

### 6.3 Verify metric vs implementation

For every kwarg, every threshold, every metric — read the code. Compare to the name. The five-instances pattern (`perf`, cumulative shaping, e_max, valid_init_only, diagonal-only surface) means every named quantity should be assumed misleading until verified.

### 6.4 Multi-seed before structural claims

5 seeds minimum for any "the recipe does X" claim. Single-seed results are screening signals at best. Phase 5b's bimodality (3/5 vs 2/5 on the same recipe) is sufficient evidence that single-seed numbers are unreliable.

### 6.5 Test the curriculum's necessity

For any new curriculum design, run the analog of Phase 5b's Block B: fresh training, warm from intermediate stage, warm from Stage 1.0. Stronger than "curriculum helps" — establishes whether curriculum is required.

### 6.6 Per-seed variance is information

Report failure-rate alongside mean. Bimodality at the recipe edge is data, not noise.

### 6.7 OOD-eval at every stage

Each stage's ckpt must be evaluated at Phase 4 conditions (no regression) and at the next stage's distribution (warm-start quality). This catches specialization-erosion early.

### 6.8 Save mid-training ckpts

Phase 5b's "early peak ckpt may be better warm-start than converged ckpt" hypothesis was identified but not tested. Phase 5.5 should test it explicitly. Save ckpts at 25%, 50%, 75%, 100% of each stage; eval at next-stage OOD; pick best for warm-start.

### 6.9 Trajectory inspection

For every stage, look at 3-5 successful and 3-5 failed trajectories. Mechanism stories should be visually verifiable, not just statistical.

---

## 7. Risks and mitigations

### 7.1 Risk: env modifications break Phase 5b reproducibility

If env-fix or Phase 5.5 env mods change reset behavior at low-e LEO, Phase 5b's 96.4% deliverable may not reproduce.

**Mitigation:** before any Phase 5.5 training, re-validate Phase 5b's deliverable on the env-fix code (Stage 5.5.0). If it doesn't reproduce within ~3pp of 96.4%, identify the change and either revert or document.

### 7.2 Risk: action discretization mismatch causes training collapse at MEO

If 25 m/s overshoots at MEO, the agent might learn to coast forever (the only "safe" action). Phase 5b Block B-style fresh-training collapse at the new altitude.

**Mitigation:** P3 (action calibration) tells us this in advance. If overshoots are detected, redesign action space before training Stage 5.5.x. If overshoots show up only mid-curriculum, drop back to a previous ckpt and reconsider.

### 7.3 Risk: Φ_orbit normalization saturates at high altitude

If Δa at MEO routinely exceeds ALT_MAX, Φ_orbit saturates at 1.0 and the shaping signal disappears.

**Mitigation:** P1 (Φ_orbit calibration probe) tells us this. If saturation occurs, consider per-episode normalization (Δa / target.a) or per-altitude ALT_MAX.

### 7.4 Risk: warp action inadequacy at GEO

If GEO orbits need 24 hours to phase but the longest warp is 5 minutes, the agent has to chain 288 warps for one orbital period. Likely catastrophic for credit assignment.

**Mitigation:** P4 measures this. If GEO warps are inadequate, add longer warps (1hr, 6hr). Note: this creates the same "actions added mid-training" issue as 7.2.

### 7.5 Risk: per-seed bimodality worsens at higher altitude

Phase 5b had 2/5 seeds collapse at the e=0.05 LEO regime. At higher altitudes, this rate may grow. We could see 1/5 working seeds, or 0/5.

**Mitigation:** budget for retries. Run 5 seeds; if <2 succeed at any stage, run 5 more. If still <2, that's a recipe-fragility finding to document.

### 7.6 Risk: scope creep

Phase 5.5 is altitude expansion. It is not action-space redesign, shaping redesign, observation redesign, or recipe-overhaul. If P1-P4 reveal that significant non-altitude changes are needed, that's a different scope (Phase 5.7 or similar) that should be explicitly split out, not tackled simultaneously.

**Mitigation:** keep the scope tight. If the curriculum stalls and the fix requires non-altitude recipe changes, write a separate spec for that work. Don't pile changes onto an already-running Phase 5.5.

---

## 8. Sequencing

Phase 5.5 has a natural dependency chain:

**Pre-flight (1 day, no compute):**
1. Land env-fix from `phase5-env-fix-spec.md`. Verify with V1-V4.
2. Snapshot canonical Phase 5b ckpts. Tag them.

**Pre-experiments (1 day, ~4 hours compute):**
3. Run P1-P4. Document findings in `PHASE5_5_PRE_FINDINGS.md`.
4. Decide: env mods needed? If yes, scope and implement. If no, proceed.

**Curriculum (3-7 days, ~10-30 hours compute, depending on how stages go):**
5. Stage 5.5.0 (re-validate Phase 5b in fixed env).
6. Stage 5.5.1 (LEO + slight altitude extension).
7. Stage 5.5.2 (LEO + low MEO).
8. Stage 5.5.3 (LEO + MEO + low eccentricity).
9. Stage 5.5.4 (full LEO+MEO+GEO with e scaled by altitude).

Each stage gated on success criteria (§5.4); if a stage fails, write a stage-specific findings doc and decide whether to debug, skip, or stop.

**Validation (1 day, ~4 hours compute):**
10. Run final capability surface (§5.3).
11. Multi-seed validate Stage 5.5.4 (§5.2).
12. Document `PHASE5_5_FINDINGS.md`.

**Closure (1 day, no compute):**
13. Update `RECIPE.md` with the altitude-expanded recipe.
14. Update `PHASE5_FINDINGS.md` (or create Phase 5 final consolidated findings).
15. Phase 5 closes.

Total: ~1-2 weeks elapsed. ~20-40 hours compute. The variance is mostly in how the curriculum goes.

---

## 9. Pre-committed acknowledgments

This is the seventh-or-so "Phase 5 thing" being scoped after multiple retractions. The pattern is real. Pre-commit to:

### 9.1 Surprise A: Pre-experiments reveal env issues that change the scope

If P1-P4 show that the recipe needs significant non-altitude changes (action space redesign, shaping reconfiguration, etc.), Phase 5.5 spec should be paused and scope-revised. Don't cargo-cult through with a spec that doesn't match the env's actual needs.

### 9.2 Surprise B: Curriculum stalls at intermediate stage

If Stage 5.5.2 plateaus below 75% after train-longer + multi-seed, the recipe doesn't extend to MEO at the planned eccentricity range. Document the stall, the mechanism (with proper binning, no Simpson's paradox), and the conclusion. "The recipe extends to LEO + slight altitude extension but not to MEO without further changes" is a valid Phase 5.5 outcome.

### 9.3 Surprise C: Per-seed variance dominates the result

If 5-seed mean has ±25pp std at any headline stage, the recipe is fragile in that regime. Report it explicitly. Don't collapse to a clean number that hides the bimodality.

### 9.4 Surprise D: The capability surface still has a hidden confound

If the final surface shows weird structure (cliffs, corners) that doesn't match the curriculum's training distribution, suspect a hidden confound. Possibilities: distribution mismatch in eval cells vs training cells, env behavior at corner cases, metric mis-specification. Apply the §6.3 discipline (verify metric vs implementation) before drawing conclusions.

### 9.5 Surprise E: It works trivially

If Stage 5.5.4 hits 90%+ multi-seed on the first try, suspect a measurement issue. The Phase 5e pattern. Verify with explicit checks: does the surface use fixed eccentricities? Are doomed inits filtered (gave_up=0)? Are altitudes within the trained range per cell? Don't ship until verified.

### 9.6 The discipline applies at every stage

Each stage gets length-binned analysis, multi-seed numbers, OOD checks at adjacent stages, trajectory inspection. No single-snapshot "we found the mechanism" stories without orthogonal evidence. The cost of doing this discipline is ~10% more wall time per stage; the benefit is not retracting Phase 5.5 in another spec.

---

## 10. What this spec is NOT

- **Not a guarantee that altitude expansion works.** Could fail at any stage. The pre-commits in §9 explicitly cover failure cases.
- **Not Phase 6.** Multi-body work waits for two-body altitude/eccentricity to be solid.
- **Not a complete intervention sweep.** Action masking, demonstration bootstrapping, bound-expansion-as-continuous-curriculum (Option C) remain available as Phase 5.6 or Phase 5.7 if Phase 5.5 stalls.
- **Not portfolio writing.** That follows Phase 5 closure.

---

## 11. Closure conditions

Phase 5.5 is complete when:

1. Pre-flight validations pass (env-fix landed, Phase 5b reproducible).
2. P1-P4 pre-experiments documented in `PHASE5_5_PRE_FINDINGS.md`.
3. Stage 5.5.0 reproduces Phase 5b's deliverable in the env-fix code.
4. Stage 5.5.1 through 5.5.4 each documented with stage-specific findings (success or stall).
5. Final capability surface produced (§5.3).
6. Multi-seed validated at Stage 5.5.4 (§5.2).
7. `PHASE5_5_FINDINGS.md` written.
8. `RECIPE.md` updated.

If the curriculum stalls at any stage, items 5-7 still apply, framed around "documented capability and limit" rather than "shipping the full deliverable."

---

## 12. The portfolio framing this enables

If Phase 5.5 succeeds at any meaningful level (Stage 5.5.3 = LEO+MEO+e≤0.30 or better), the deliverable shifts from "LEO low-e specialist" to "two-body any-orbit rendezvous agent within trained envelope." The portfolio framing becomes:

> Built a two-body orbital rendezvous agent that handles arbitrary phase gaps, eccentricities, and altitudes from LEO through GEO. Trained via curriculum learning with explicit attention to specialization-erosion trade-offs. The recipe extends from a LEO low-e bootstrap through altitude × eccentricity expansion stages, with multi-seed validation at the headline stages. Documented capability surface across the full envelope.

This is a stronger portfolio claim than the LEO-only deliverable. If Phase 5.5 stalls, the LEO-only framing remains, augmented with "documented attempt at altitude expansion with mechanism attribution."

Either way, the portfolio piece is shippable. The honest accounting of what works and what doesn't is itself a deliverable.

---

*Author: 2026-05-02. Phase 5.5 altitude expansion spec. Builds on env-fix prerequisite. Pre-experiments → curriculum → validation → closure. Multi-week effort, 20-40 hours compute. Designed against Phase 5b-e methodological lessons; pre-committed against the patterns that produced earlier retractions.*
