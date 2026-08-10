# Phase 5d — Continuing Toward Working High-Eccentricity Rendezvous

> **Status:** 2026-04-30 (continued). Phase 5 is not done. Phase 5c's interventions reached 18% at e=0.20 and produced a corrected mechanism story (Simpson's paradox in original A1; early-death dominates failures, not direction inversion). This spec lays out the next round of diagnostic + intervention work to push Phase 5 toward a working agent at eccentricities that matter.

---

## 0. Why Phase 5d exists

The deliverable Phase 5 is supposed to ship — a 2D orbital rendezvous agent that handles arbitrary phase and eccentricity — does not yet exist. Phase 5b shipped a working agent at e ≤ 0.05 and partial extension to e = 0.10. Phase 5c attempted to extend past e = 0.20 and produced 18% at the target. Neither is sufficient for the phrase "working 2D orbital rendezvous" to be honest.

Phase 5d continues Phase 5. There is no Phase 6 pivot in this spec. Phase 5 stays open until either:
- The agent works at the eccentricity range that matters (target: ≥70% multi-seed at e ≥ 0.30, stretch ≥ 0.50)
- We've exhausted the intervention space and ship "best result + documented limitations" with full attribution of why higher-e failed

Both outcomes are publishable. The discipline: don't fudge the framing to declare premature success.

---

## 1. What Phase 5c actually established (revised reading)

Phase 5c's net findings, after the corrections doc:

**Established with high confidence:**
- B3 (mixed-distribution training) reaches 18% at e=0.20, single-seed. Best Phase 5c result.
- B4/B5/B6 collapse to 0% at e=0.20. Tested interventions on those mechanisms don't work.
- The Phase 4 recipe's per-step shaping is directionally correct within trajectory-length bins (corrected from Phase 5c's first claim).
- 46% of failures at e=0.20 terminate early (length ~54, terminal Φ_orbit ≥ 20). Catastrophic-action failures, not slow plateaus.
- 17% of failures are near-miss (long episodes, low terminal Φ, fail dual criterion).
- The remaining ~37% are intermediate.

**Established with low confidence (worth re-verifying):**
- "B3 is mostly limit-damage" — based on the e-binned eval showing steep monotonic drop. Plausible but the alternative ("B3 partially fixes a specific failure mode") wasn't directly tested.
- The 46% figure for early-death — was the Simpson's-paradox catch applied to *this* statistic? If failures were re-binned by length, the "46% early-death" framing might also have an aggregation hidden in it.
- The Phase 4 baseline contrast (A6 from the original Phase 5c findings) — only the e=0.20 within-bin analysis was done in the corrections. Phase 4 baseline data wasn't re-binned.

**Not established at all:**
- Multi-seed variance on B3. The 18% is single-seed.
- What kind of early-death dominates (escape vs collision vs stranded vs other).
- Whether early-death and near-miss failures respond to different interventions.
- Whether the Block II interventions (B2, B4, B5) that were strangled by 0% success would actually work *given* early-death prevention.

The "not established" list is what Phase 5d's first block should resolve before any new training runs.

---

## 2. Phase 5d structure

Three blocks, each gated on the previous. Designed to be aggressively diagnostic-first because the project has now twice (Phase 5b post-extend, Phase 5c original) committed to mechanism stories that turned out to be Simpson-paradox-confounded or otherwise overcommitted. Phase 5d front-loads the "make sure we know what's actually happening" work.

- **Block I — Failure-mode characterization (~1-2 hours).** Five diagnostic probes that finish characterizing what failures look like at e=0.20.
- **Block II — Targeted intervention sweep (~6-10 hours).** Four interventions that target the dominant failure mode identified in Block I, plus re-test of strangled Block II interventions from Phase 5c.
- **Block III — Multi-seed validation and eccentricity expansion (~10-15 hours).** The best intervention from Block II validated across seeds; eccentricity expansion to e=0.30, 0.50, ideally 0.70.

Total ~25-30 hours compute. Comparable to Phase 5b and Phase 5c full budgets.

---

## 3. Block I — Failure-mode characterization

### 3.1 Probe D1 — Termination-mode classification

Take the Stage 4.0 ckpt at e=0.20 (Phase 5b's deliverable, 14% zero-shot). Run 200 evaluation episodes. For each terminating episode, log the *termination reason*:

- `escape`: specific orbital energy ≥ 0 (hyperbolic trajectory)
- `collision`: distance to Earth < hard_radius (atmospheric reentry / impact)
- `stranded`: fuel = 0 with rendezvous incomplete
- `safety_cap`: hit MAX_STEPS = 2000
- `success`: rendezvous achieved

Cross-tabulate with trajectory length and terminal Φ_orbit. Specifically: of the ~46% short-failure subpopulation, what fraction is escape vs collision vs stranded?

**Why this matters:** the proposed Phase 5d interventions all target specific termination modes. Action masking on prograde burns helps escape; action masking on retrograde near Earth helps collision; fuel cost shaping helps stranded. Without knowing which dominates, we'd be designing interventions blind.

**Cost:** ~10 minutes. Eval logs already capture termination reason; this is post-hoc analysis.

**Decision rule:**
- If one mode dominates (≥60% of short failures): target it specifically in Block II.
- If multiple modes are comparable (each 20-40%): generic safety intervention rather than mode-specific.
- If the short-failure subpopulation isn't actually as large as Phase 5c reported (the 46% might itself be Simpson-confounded): re-examine the failure-mode partition.

### 3.2 Probe D2 — Verify the 46% short-failure framing

Re-bin the 200-episode eval by trajectory length using *fine-grained* bins (every 100 steps, not the [0, 100) / [100, 300) / [300, 700) / [700, 2000) coarse bins used in the corrections doc).

Plot the histogram of terminal lengths separated by success vs failure. Look for:
- Is the "early death" peak actually at length ~54, or is that the median of a broader distribution?
- Is there a *bimodal* failure distribution (clear short-failure peak + clear long-failure peak), or is it a continuous distribution that looks bimodal under coarse binning?
- What's the precise short-failure fraction at finer bins?

**Why this matters:** the corrections doc's Simpson's paradox catch revealed that coarse aggregation can hide structure. The "46% short-failure" number was derived from the same kind of coarse-binned analysis that produced the original direction-inversion artifact. We should verify it survives fine-grained re-examination.

**Cost:** ~10 minutes.

**Possible outcomes:**
- Bimodal confirmed → "early-death vs near-miss" framing is right.
- Continuous distribution → the failure types may not be cleanly separable; intervention design needs to address the full distribution.
- Different peak structure → revise the failure-mode framing.

### 3.3 Probe D3 — Re-bin Phase 4 baseline (e=0) under length

The corrections doc fixed e=0.20 analysis but didn't apply the same length-binning to the Phase 4 e=0 baseline. The original A6 finding ("Phase 4 at e=0 has correctly-oriented direction: success cum r = -0.80, failure cum r = -3.04") might be Simpson-confounded too.

Re-bin Phase 4 e=0 eval data by trajectory length. Compute success vs failure cumulative shaping per bin.

**Why this matters:** if Phase 4 at e=0 also shows direction-correct within length bins (and the population statistic was a length-asymmetry artifact in both regimes), the "regime-fragile recipe" framing was wrong in both directions. The recipe is consistent across regimes, and the e=0.20 problem isn't shaping mis-specification at all — it's purely the early-death issue.

If Phase 4 at e=0 shows different within-bin patterns (e.g., direction is correct at e=0 but reversed at e=0.20 *within bins*), then there *is* a regime-specific shaping issue that survives the Simpson correction.

**Cost:** ~10 minutes. Phase 4 ckpt eval logs presumably exist; just need to re-process.

### 3.4 Probe D4 — B3's 18% by failure mode

Take B3's 18% checkpoint. Run 200 episodes at e=0.20. For each episode, log termination mode and trajectory length.

Compare to the Stage 4.0 ckpt's failure distribution (from D1).

**Hypothesis to test:** B3's 18% is "limit damage via low-e capability preservation," meaning B3's failure profile should look similar to Stage 4.0's at e=0.20 — same short-failure rate, just slightly fewer of them. The successes should be on the easier samples (within-distribution e closer to 0).

**Alternative hypothesis:** B3 differentially reduces early-death failures (e.g., the mixed-distribution training teaches the policy to avoid catastrophic actions even at higher e). If true, B3's failure profile would have *fewer* early deaths than Stage 4.0's, with the additional successes coming from preventing catastrophic actions.

**Why this matters:** if B3 differentially reduces early-death, that's a free signal that mixed-distribution training itself is part of the early-death prevention — and combining B3 with explicit early-death prevention (action masking) might compound benefits. If B3 is purely limit-damage, the early-death prevention work needs to start from scratch.

**Cost:** ~15 minutes.

### 3.5 Probe D5 — Trajectory inspection of representative failures

For each failure mode identified in D1 (escape / collision / stranded / safety_cap), pick 3-5 representative trajectories. Inspect:
- What action sequences led to the catastrophic terminal state?
- Were there clear warning signs (e.g., orbital energy approaching escape) that the policy ignored?
- At the moment of the catastrophic action, what did the observation look like? (Was the policy in OOD state?)

**Why this matters:** intervention design requires understanding the *behavior* that produces failures, not just statistics. If escape failures consistently come from "too many prograde burns at high apogee," the intervention is "don't allow prograde burns when sat is at apogee with energy > threshold." If escape failures come from "single large prograde burn from near-circular state," the intervention is "burn limit on first action from each rest state."

**Cost:** ~30 minutes for inspection of 12-20 trajectories.

### 3.6 Block I deliverable

A document `PHASE5d_BLOCK_I_FINDINGS.md` with:
- Termination-mode partition of the failure population (D1)
- Fine-grained length distribution (D2)
- Phase 4 baseline re-bin verdict (D3)
- B3 vs Stage 4.0 failure profile comparison (D4)
- Trajectory inspection summaries per failure mode (D5)

Ends with a refined characterization: "Phase 5d's intervention target is X failure mode, dominating Y% of failures, with mechanism Z." If the data doesn't support a single dominant target, the deliverable is "Phase 5d needs to address a mixture of failure modes; combination interventions required."

---

## 4. Block II — Targeted intervention sweep

Designed assuming Block I produces actionable verdicts. If Block I reveals the failure modes are unclear or different from expected, Block II's first item changes.

The interventions are organized around the failure modes likely to dominate:

### 4.1 Mode-specific interventions

**I1 — Action masking (escape prevention).** Before each step, compute what each available action would do to specific orbital energy. Mask actions that would push energy ≥ 0 within the step. Implementation: add a `valid_actions` mask passed alongside the obs; PPO's action sampling respects the mask.

Cost: ~2 hours engineering. Targeted at escape-mode early death.

**I2 — Action masking (collision prevention).** Same pattern: mask actions that would push satellite altitude below hard-radius within the step.

Cost: ~1 hour additional (extends I1 infrastructure). Targeted at collision-mode early death.

**I3 — Fuel cost shaping.** Add a per-burn shaping cost proportional to Δv squared (penalizing inefficient burns). Reset adds it as a separate component, not modifying the existing shaping.

Cost: ~30 min. Targeted at stranded-mode early death.

**I4 — Generic physical-safety penalty.** Soft penalty (e.g., -1.0) added to the reward when an action would cause escape/collision within the next 10 steps under current physics. Doesn't mask the action; just discourages it.

Cost: ~1 hour. Targeted at all early-death modes generically. Less invasive than masking.

### 4.2 Re-tested Phase 5c interventions

If Block I confirms early-death is the dominant blocker, the Phase 5c interventions (B2, B4, B5) might actually work *once early-death is suppressed*. Test by re-running them on top of the best mode-specific intervention from §4.1.

**I5 — B2 + best early-death prevention.** Continuous gates with action masking.

**I6 — B4 + best early-death prevention.** Discrete(14) larger Δv with action masking.

**I7 — B5 + best early-death prevention.** Higher γ (0.998) with action masking.

These three are cheap to add once §4.1 has produced a working early-death prevention.

### 4.3 Block II deliverable

A document `PHASE5d_BLOCK_II_FINDINGS.md` reporting:
- Per-intervention success rate at e=0.20 (single seed, 200 ep eval)
- Failure-mode shifts (does intervention X actually reduce mode Y failures?)
- Combination tests (do I1+I3 compound?)

**Decision rule:**
- If any single intervention or combination reaches ≥ 50% at e=0.20: candidate for Block III multi-seed.
- If multiple interventions reach 30-50%: combine them and re-test.
- If all interventions stay below 30%: Block I diagnostic missed something; return to investigation. Pre-commit to: ≤ 1 round of "return to investigation" before declaring the recipe ceilinged at e=0.20.

The "≤ 1 round" pre-commit is important. The project has cycled through "diagnostic → intervention → revise diagnostic → intervention" twice (Phase 5b cap-tail, Phase 5c). If Phase 5d's first round doesn't produce a workable intervention, the second round is the last attempt before we accept the ceiling and ship Phase 5 with documented limitations.

---

## 5. Block III — Multi-seed validation and eccentricity expansion

Conditional on Block II producing a working intervention.

### 5.1 Multi-seed validation at e=0.20

5 seeds × the chosen intervention from Block II. Train to convergence (per Principle I, train-to-convergence not pre-budgeted step count). Eval at e=0.20 with multi-rollout protocol.

Target: ≥ 70% mean across seeds.

### 5.2 Eccentricity expansion

If §5.1 lands, expand to:
- e=0.30 (Stage 4.3 — Phase 5b's original target)
- e=0.50 (Stage 4.4 — Phase 5b's stretch goal)
- e=0.70 (Molniya territory — Phase 5d's stretch goal)

Each is a sub-stage. Train to convergence per stage. Multi-seed at e=0.30 and e=0.50.

### 5.3 Capability surface eval

Full grid eval at the final ckpt: phase × e × (sat-target orbit relation) × phase_gap. Same methodology as Phase 5b Block D.

### 5.4 Phase 4 condition check

Per the post-extend lesson: at every major stage, eval at Phase 4 conditions (sat circular, target circular different a). Verify high-e training hasn't destroyed low-e capability.

### 5.5 Block III deliverable

`PHASE5d_FINDINGS.md` final report. Recipe diff from Phase 5b. Capability surface heatmap. Phase 5 deliverable confirmed (or documented limit if ceiling holds at some intermediate e).

---

## 6. What's locked, what's open

### Locked

- Random sat init + same_orbit_init logic from Phase 5b Step 1
- Two-stage curriculum (Stage 1.0 → Stage 4.0) from Phase 5b Block B
- Train-longer null first when stages stall (Phase 5b Step 1)
- No structural claims from <2σ data (Phase 5b post-extend)
- Length-binning *before* aggregating any length-dependent statistic (Phase 5c corrections)
- 5-seed multi-seed at headline stages (Phase 5a addendum)
- Eval-during-training (post-hoc scan or in-loop) per stage
- Phase 4 condition eval at every major stage

### Open (Block I will determine)

- Which failure mode dominates at e=0.20
- Whether the failure-mode framing itself survives fine-grained re-examination
- Whether the Phase 4 baseline contrast survives length-binning
- Whether B3 is doing more than limit-damage

### Out of scope

- Phase 6 multi-body work (still!)
- Debris (still on user hold)
- New algorithmic interventions outside the Block II list (R3 stays dead unless Block I demands)
- Continuous action space (Phase 5b ruled out)

---

## 7. Pre-committed acknowledgment of what could go wrong

The project has now had three distinct "we were wrong about the mechanism" moments in Phase 5:

1. Phase 5b post-extend's "60M shows erosion" was overcommitted from one borderline data point.
2. Phase 5c's original "shaping direction reversed" was a Simpson's paradox artifact.
3. Phase 5c's intervention set ranked B1 first when A1 turned out not to be the dominant mechanism.

It would be naïve to assume Phase 5d's mechanism story (early-death dominates) won't have its own surprises. **Pre-committed responses to plausible Phase 5d surprises:**

**Surprise A — Block I shows failure modes are mixed (no dominant mode).** Run I4 (generic physical-safety penalty) instead of mode-specific masking. If that doesn't work, the problem is structural and we accept the ceiling.

**Surprise B — Block I shows the 46% early-death framing is itself an artifact of coarse binning.** Revise the failure-mode characterization. Probably means iterating Block I until the failure population is well-understood. Pre-commit: ≤ 2 hours additional Block I work before declaring failure-mode partitioning intractable and trying intervention by trial-and-error.

**Surprise C — Block II's interventions reduce early-death but the policy still doesn't reach success.** Means there's a near-miss component that intervention I*-doesn't address. Combine with B3-style mixed-distribution training as the near-miss handler.

**Surprise D — Even with everything stacked, Phase 5d doesn't crack the wall.** Then the recipe genuinely has a ceiling at the e=0.10-0.15 regime. We ship Phase 5 with that documented limit, multi-seed validate Phase 5b's deliverable cleanly, and write up the negative finding with full mechanism attribution. **This is acceptable.** The portfolio piece is "I built a working agent up to e=0.10 and identified a recipe-level ceiling above that, with detailed diagnostic showing why."

The discipline: **don't fudge to claim Phase 5 success at e ≥ 0.30 if the recipe genuinely ceilings at e ≤ 0.15.** The project's value is in the diagnostic depth, not in the headline number.

---

## 8. Specific things to NOT skip

Three discipline items that the project has recurrently been tempted to skip:

**8.1 — Length-binning before aggregating.** Phase 5c's Simpson's paradox catch was the third or fourth time the project caught itself overaggregating. Phase 5d's any-time analysis includes length binning by default. If a Block I or II analysis produces a population statistic, it should explicitly say either "length-binning was done and direction is consistent" or "this is an aggregate; subgroup analysis pending."

**8.2 — Failure-mode partitioning before mechanism stories.** Don't propose "the recipe fails because of X" without first checking which failures the proposed mechanism would and wouldn't explain. If 46% of failures are early-death and the proposed mechanism is "near-miss policy quality," the mechanism only addresses 17% of failures — important caveat.

**8.3 — Multi-seed before Block III commitment.** Phase 5c's B3 was single-seed 18%. Phase 5d's Block III commits to a 5-seed validation only after Block II identifies a candidate. Single-seed results are screening signals, not deliverables.

---

## 9. Compute and time budget

| Block | Activity | Compute | Wall |
|---|---|---|---|
| I | 5 diagnostic probes, mostly post-hoc | ~2 hours | 1 day |
| II | 7 interventions, single-seed | ~10 hours | 3-4 days |
| III | 5-seed multi-seed + e expansion + capability surface | ~15 hours | 1 week |
| Writeup | Findings doc + Phase 5 closure | minimal compute | 1 day |
| **Total** | | **~27 hours** | **~2 weeks** |

Comparable to Phase 5b and 5c. No urgency; happens in the time it takes.

---

## 10. The Phase 5 closure question

Phase 5d is intended to be the final Phase 5 chunk. Phase 5 closes when one of:

**Closure A — Working agent at high e.** Block III lands with multi-seed ≥ 70% at e=0.30, and capability surface shows graceful degradation. Phase 5 ships a working "any orbit to any orbit" agent. This is the goal.

**Closure B — Documented ceiling.** Phase 5d's interventions don't crack the wall. Phase 5 ships Phase 5b's deliverable (e ≤ 0.05 fully-random, partial extension to e ≤ 0.10) with a comprehensive diagnostic of why higher-e is unreachable under this recipe. The portfolio piece is the diagnostic depth.

**Closure C — Partial extension.** Phase 5d cracks e=0.20 but ceilings before e=0.50. Phase 5 ships an agent for the e ≤ 0.20 range with documented limit beyond.

All three are publishable. The discipline: report what's true.

If even Phase 5d doesn't produce closure (interventions cycle without converging on either a working recipe or a clear ceiling story), that's a meta-failure that warrants stepping back and asking whether the project's framing is right. But that's contingent on actually running Phase 5d, not pre-emptive.

---

## 11. What this spec is NOT

- Not a Phase 6 spec. Phase 5 stays open.
- Not a commitment to find a working high-e agent. Block III is conditional on Block II producing a candidate.
- Not a commitment to multi-week investigation. The "≤ 1 round of return to investigation" pre-commit in §4.3 limits scope.
- Not a redefinition of the recipe. The Phase 5b recipe (random sat init, gated NHR shaping, LVLH obs, etc.) carries forward unchanged. Phase 5d adds *training-time* interventions (action masking, fuel cost shaping, etc.) without modifying the core observation/reward/curriculum design.
- Not a portfolio-narrative document. Blog/tweet writeups are post-Phase-5.

---

## 12. Sequencing

In order:

1. **Block I (one day, mostly post-hoc).** Run all 5 probes. Refine failure-mode story.
2. **Block II first round (3-4 days).** Run mode-specific interventions targeting the dominant failure mode. If one reaches threshold: skip to Block III. If not: run combinations.
3. **Block II second round (if needed, 2-3 days).** Combinations and re-tested Phase 5c interventions stacked on early-death prevention.
4. **Block III (1 week).** Multi-seed validation + eccentricity expansion + capability surface.
5. **Phase 5 closure writeup.** Findings, recipe documentation, lessons.

If at any point during Block II it becomes clear that interventions aren't moving the needle (e.g., 4-5 interventions tested, all 0-20%), invoke the "≤ 1 round of return to investigation" provision and escalate to deeper recipe questions before continuing.

---

*Author: 2026-04-30. Phase 5d spec, after Phase 5c corrections. Successor: PHASE5d_BLOCK_I_FINDINGS.md, then iterative.*
