# Orbital RL — Phase 5a Spec: Pre-Investigations

> **Status:** 2026-04-26. Bridge between Phase 4.5 (recipe attribution closed) and Phase 5 (full capability surface across phase gap × eccentricity). Phase 5a's job is to lock four decisions that materially shape Phase 5 before we commit to a full curriculum. Phase 5 spec is drafted after 5a results land.

---

## 0. TL;DR

Phase 5's goal is a single agent that handles `phase_gap ∼ Uniform(0, π)` jointly with `e_target ∼ Uniform(0, e_max)` for `e_max ≥ 0.3`, with stretch toward Molniya-class eccentricities. Before designing that curriculum, four upstream decisions need empirical data, not priors:

1. **Curriculum ordering** — square / tall / wide / fresh-at-target. Pick by EV.
2. **Timestep** — DT=60 vs DT=30 with retuned hyperparameters. The Phase 3 regression interpretation was probably wrong.
3. **Stage 1 bootstrap with joint sampling** — does fresh Stage 1 with `phase_gap ∼ Uniform(0, π/6)` AND `e_target ∼ Uniform(0, 0.05)` converge as reliably as Phase 4's point-curriculum Stage 1 (~2/3 seeds)?
4. **Eccentricity-aware reward sanity check** — does the chaser learn to always rendezvous at apoapsis (where relative velocities are lowest)?

Phase 5a runs these as four small experiments, then reports a recipe-and-curriculum decision that feeds into Phase 5's main spec. Total compute estimate: ~6 hours wall on M3 Max.

---

## 1. Why Phase 5a exists

Phase 4 → Phase 4.5 → Phase 5 is the third successive phase where we've discovered the previous phase's framing was structurally off. Phase 4's "only R4 worked" claim was wrong about marginal contribution. Phase 4.5's investigation showed LVLH and shaping are load-bearing for bootstrap. Each correction was data-discovered, not prior-discovered.

Phase 5 is a larger commitment than either — it's a full curriculum across an expanded difficulty surface, with attribution sub-experiments, and a deliverable capability artifact. Drafting the main Phase 5 spec without first running the upstream investigations means committing to a curriculum that may need to be discarded mid-phase. Cheap to run the investigations first, expensive to re-run a curriculum.

The four decisions below all have material impact on the main Phase 5 design. None of them are answerable from existing data:

- **Ordering:** Phase 4 used point-curriculum at single phase gaps. Joint-randomized curriculum is a different shape; Phase 4 evidence doesn't transfer.
- **Timestep:** Phase 3's DT=30 regression was confounded with a hyperparameter mismatch. Re-test cleanly.
- **Stage 1:** Phase 4.5 confirmed Phase 4-style Stage 1 has ~1/3 retry rate. Joint-sampled Stage 1 may be different (better or worse).
- **Reward sanity:** New regime, untested rendezvous behavior at eccentric targets. Cheap to verify.

---

## 2. Conceptual framework: what the curriculum is doing

Drawing on recent literature (DORAEMON, ICLR 2024; Goldilocks RL, 2026; the goal-conditioned RL curriculum survey) the standard framing for our problem is:

We have a task distribution `T(φ)` parameterized by difficulty parameters `φ = (phase_gap_max, e_target_max)`. The goal is a policy that succeeds on the maximum-entropy task distribution at full bounds. Curriculum learning expands the sampling distribution over `φ` from a small initial range (where bootstrap is feasible) to the full range (where the deliverable lives), gated on success rate.

DORAEMON's empirical finding directly applies: **gradually expanding the distribution outperforms training on the maximum-entropy distribution from scratch, even when both end at the same distribution.** This rules out "just train on the full random box from random init" as a viable Phase 5 approach.

The remaining question is *how* to expand. The four candidate orderings — square, tall, wide, fresh-at-target — are different schedules over the parameter rectangle. Phase 5a picks the empirically winning one rather than committing on prior reasoning alone.

---

## 3. Investigation A — curriculum ordering

### 3.1 The four candidates

Each starts at `(phase_gap_max, e_target_max) = (π/6, 0.05)` (small initial bounds, both axes random within bounds) and ends at `(π, 0.3)` (full bounds for Phase 5 main).

| Pattern | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| **Square** | (π/6, 0.05) | (π/2, 0.10) | (π, 0.20) | (π, 0.30) |
| **Tall** (phase-first) | (π/6, 0.05) | (π/2, 0.05) | (π, 0.05) | then anneal e: (π, 0.10), (π, 0.20), (π, 0.30) |
| **Wide** (eccentricity-first) | (π/6, 0.05) | (π/6, 0.10) | (π/6, 0.20) | (π/6, 0.30) | then anneal phase: (π/2, 0.30), (π, 0.30) |
| **Fresh-at-target** (control) | direct (π, 0.30) from scratch | — | — | — |

The fresh-at-target run is the DORAEMON-style control: train on the full distribution from scratch, no curriculum. Expected to fail or underperform per the DORAEMON results, but worth confirming on our env.

### 3.2 Run protocol

For Phase 5a's purpose (decision-making, not deliverable), each candidate runs only **the first two stages** of its curriculum, on **one seed**. Fresh-at-target runs ~10M steps from scratch.

| Pattern | Compute |
|---|---|
| Square: Stages 1-2, ~25M steps | ~30 min |
| Tall: Stages 1-2, ~25M steps | ~30 min |
| Wide: Stages 1-2, ~25M steps | ~30 min |
| Fresh-at-target: ~10M steps | ~12 min |
| **Total** | **~1.5 hours** |

Two stages is enough to surface the asymmetry. If Stage 2 of any candidate collapses (fails to maintain Stage 1's performance under the wider distribution), that pattern is dead — the rest of its curriculum doesn't need testing.

### 3.3 Decision rule

Rank the four candidates by Stage 2 final success rate at the full Stage 2 sampling distribution (200 episodes × 3 rollout seeds = 600 episodes per candidate). Pick the winner. If two candidates are within noise (±5pp), pick the simpler curriculum (square > tall ≈ wide > fresh-at-target).

If all four fail (no candidate exceeds 50% Stage 2), Phase 5 is in trouble and we re-think before committing to a main spec. This is unlikely given Phase 4.5 results but the gate exists.

### 3.4 Pre-registered hypotheses

To prevent post-hoc rationalization of whichever result we get:

- **H1 — Wide (eccentricity-first) wins.** Phase 2's e=0.3 wall was hit by warm-starting eccentricity onto a phase-solved policy; wide avoids this pattern.
- **H2 — Fresh-at-target fails.** DORAEMON evidence + Phase 4 evidence (fresh-at-180° returned 0%) suggests the joint full distribution is not bootstrappable from random init.
- **H3 — Square underperforms tall and wide.** Joint expansion mixes well-understood phase-gap difficulty with the harder eccentricity axis; if eccentricity is the binding constraint, isolating it (wide) helps.

If H2 fails (fresh-at-target works), that's a bigger finding than the curriculum question — the entire premise that "curriculum is necessary" weakens. Worth investigating before committing to Phase 5.

If H1 and H3 both fail (square wins), that's evidence that eccentricity isn't qualitatively harder than phase gap on this env, simplifying the Phase 5 narrative.

### 3.5 What this is *not*

- Not a final selection of curriculum stages. Phase 5 main spec will refine the chosen pattern's stage boundaries based on observed learning curves, not just Stage 1-2 evidence.
- Not a multi-seed confirmation. Single seed is sufficient for first-pass ordering. If the chosen pattern's main run collapses on different seeds in Phase 5, we revisit.

---

## 4. Investigation B — timestep (DT=60 vs DT=30)

### 4.1 The case for revisiting

Phase 3 documented DT=30 collapses at e=0.30 (0%) and e=0.125 (2.7%). The interpretation in PROGRESS.md was "halving dt doubles episode length, halves terminal-reward density, compounds sparse reward."

This interpretation fails the Phase 4 evidence test. Phase 4's R4 curriculum produces episodes with `episode_length ≈ 970` decision steps, far longer than DT=30 would have produced under Phase 3 conditions. Long episodes are not the binding mechanism; Phase 4 has them and works.

Plausible alternative: Phase 3's DT=30 test was confounded. The hyperparameter set was tuned for DT=60 — shaping coefficients, action Δv magnitudes, normalization factors, and γ all baked in implicit dt assumptions. Halving dt without retuning these is a different test than "does DT=30 help."

### 4.2 Why dt might matter at high eccentricity

Eccentric orbits have variable speeds along the orbit. At periapsis (e=0.3, perigee at 350 km altitude), orbital velocity is ~8.0 km/s; at apoapsis, ~5.6 km/s. Decision spacing of 60s at periapsis covers ~480 km of arc, vs ~336 km at apoapsis. Tight rendezvous tolerances near periapsis may demand finer decision granularity than 60s provides.

This is a hypothesis, not a fact. Worth ~3 hours to test.

### 4.3 Run protocol

Two parallel curricula, each running the **winning pattern from Investigation A**, Stages 1-2, single seed:

- **Arm B-60:** DT=60 (current default), Phase 4.5 hyperparameters as specified in PHASE4_5_FINDINGS.
- **Arm B-30:** DT=30, retuned hyperparameters per the protocol below.

The DT=30 retune is the load-bearing piece. Specifically:

| Parameter | DT=60 | DT=30 (retuned) | Rationale |
|---|---|---|---|
| `gamma` | 0.995 | 0.9975 | Preserve effective horizon: γ^N constant for halved N → γ' = γ^(1/2) |
| `BETA_SHAPE` | 1.0 | 1.0 | Per-step shaping is already velocity-of-Φ; no rescaling needed |
| `ent_coef` | 0.01 | 0.01 | Independent of dt |
| `total_timesteps` Stage 1 | 10M | 20M | Match real-time training horizon, not step count |
| `total_timesteps` Stage 2 | 15M | 30M | Same |
| Action Δv table | unchanged | unchanged | Fixed Δv per burn is dt-independent in physical effect |
| warp action τ | 5 (5min) | 10 (5min) | Preserve real-time skip; sub-step count doubles |

The action Δv interpretation is worth flagging: a 10 m/s prograde burn applied every 60s is the same physical impulse as a 10 m/s prograde burn applied every 30s. The difference is the *rate* at which burns can be issued. DT=30 doubles maximum burn rate per orbit, which may be exactly the resolution needed at high eccentricity.

### 4.4 Decision rule

Compare arms on Stage 2 final success rate at the full Stage 2 distribution. If B-30 beats B-60 by ≥ 5pp, switch Phase 5 main to DT=30. If within noise, stay at DT=60 (simpler default, cheaper compute). If B-30 regresses by ≥ 5pp, conclude DT=30 doesn't help even when retuned, and document the negative result (this also retroactively explains Phase 3's regression).

### 4.5 Risks

- The retune may be incomplete. A perfect retune is hard; some interaction we haven't identified may still favor one dt over the other. Mitigation: if results are within noise, the conclusion is "no clear winner at the retune effort we tried," which is enough for Phase 5 purposes (default to DT=60).
- Doubling step count doubles compute. Already budgeted (~1.5x normal Stage 1+2 runtime).
- Sub-step count for warp doubles, slightly increasing per-step cost. Verify SPS doesn't drop below 100k.

### 4.6 What this is *not*

- Not a comprehensive dt sweep. We test one alternative (DT=30). DT=15 or DT=120 are not on the table for Phase 5a.
- Not a verdict on Phase 3's interpretation. We're testing whether dt matters now, with current machinery, at the new task. Whether Phase 3's specific interpretation was right or wrong is secondary.

---

## 5. Investigation C — Stage 1 bootstrap reliability with joint sampling

### 5.1 The question

Phase 4.5 confirmed Phase 4-style Stage 1 (point-curriculum at 30°, e=0) had a 1/3 seed-failure rate. Phase 5's Stage 1 is structurally different: `phase_gap ∼ Uniform(0, π/6)` AND `e_target ∼ Uniform(0, 0.05)`, both random per episode.

Joint sampling could go either way:
- **More reliable bootstrap:** the easiest cases (small phase gap, small e) are guaranteed sampled in every batch; the policy gets dense gradient signal even when harder cases fail.
- **Less reliable bootstrap:** the additional eccentricity randomness adds variance to the reward signal, slowing convergence.

We don't know which. Phase 4.5 only tested phase-gap-only random sampling, not joint.

### 5.2 Run protocol

Run **Stage 1 only** of the winning Investigation A pattern, fresh, **on three seeds** (42, 1337, 20260423). Same as Phase 4.5 Ablation A/B — first-pass attribution doesn't need more.

Compute: ~3 stages × ~3 min each = ~10 minutes wall.

### 5.3 Decision rule

| Stage 1 success on seed 42 | Stage 1 success on seed 1337 | Stage 1 success on seed 20260423 | Verdict |
|---|---|---|---|
| ≥ 70% | ≥ 70% | ≥ 70% | Joint sampling robust. Phase 5 Stage 1 budget = 1× retry. |
| 2/3 succeed | | | Joint sampling on par with Phase 4. Phase 5 Stage 1 budget = 1.5× retry. |
| 1/3 succeeds | | | Joint sampling worse. Need to investigate before Phase 5. Possibly try smaller initial bounds. |
| 0/3 succeed | | | Joint sampling fails to bootstrap. Phase 5 needs structural change before main spec. |

### 5.4 What this resolves

This is the only Phase 5a investigation that directly threatens the main Phase 5 plan. If Stage 1 bootstrap fails reliably, we don't have a curriculum to start, period. Worth knowing before drafting the main spec.

---

## 6. Investigation D — eccentric-target reward sanity check

### 6.1 The concern

At high eccentricity, relative velocity at apoapsis is much lower than at periapsis (factor of (1+e)/(1−e) for circular chaser). The chaser's optimal rendezvous strategy may be "match the target's apoapsis position and let velocities equalize naturally there." If the agent learns to *always* rendezvous at apoapsis regardless of where the target currently is, that's a degenerate strategy that game-hacks the success criterion.

Two questions:

1. Does the recipe's rendezvous tolerance (`30 km` position, `50 m/s` relative velocity) implicitly favor apoapsis encounters?
2. If yes, is this a problem we want to fix, or accept?

The first question is empirical; the second is design.

### 6.2 Run protocol

Take the Phase 4 Stage 3 ckpt (seed 42, `nyul1pl8`), evaluate it on a custom eval scenario:

- Generate 200 episodes with `e_target = 0.2`, `phase_gap = π/2`, all other parameters at training defaults.
- Log the target's true anomaly at the exact step rendezvous succeeds.
- Histogram: rendezvous-anomaly distribution.

If rendezvous concentrates near anomaly = π (apoapsis), the agent has learned the apoapsis-bias strategy.

Compute: ~5 minutes (no training, just eval with extended logging).

### 6.3 Interpretation and design response

| Histogram pattern | Interpretation | Phase 5 design response |
|---|---|---|
| Roughly uniform | No bias. Reward is well-posed. | No change. |
| Concentrated near apoapsis | Apoapsis bias. Reward is reachability-favored at apoapsis. | Either: (a) accept, document, move on; or (b) fix by adding rendezvous-anomaly diversity to success criterion. |
| Concentrated near periapsis | Unexpected. Investigate. | Halt Phase 5 main, diagnose. |

The (b) option for fixing apoapsis bias is non-trivial — modifying the success criterion changes the reward landscape, which feeds into the recipe. If we choose (b), Phase 5 needs a recipe-change evaluation before the main curriculum can run. The (a) option is simpler: document the bias, design the eval surface to test rendezvous at varying target anomalies, and let the agent's behavior speak for itself.

My prior is (a). Apoapsis bias is physically natural — real spacecraft engineers preferentially rendezvous at apoapsis for the same reason. If our agent rediscovers this, that's a positive finding, not a flaw. But the empirical check is cheap and answers the question.

### 6.4 What this is *not*

- Not testing the agent's capability at eccentricity (Phase 4 ckpt was trained at e=0; expect failures). Testing only the *anomaly* of successful rendezvous, not the success rate.

---

## 7. Phase 5a deliverable

A short markdown writeup (~2 pages) summarizing:

1. **Curriculum order winner** — square/tall/wide/fresh, with comparison table.
2. **dt decision** — DT=60 or DT=30, with statistical justification.
3. **Stage 1 bootstrap reliability** — seed retry rate for the chosen curriculum.
4. **Apoapsis-bias finding** — apoapsis bias yes/no, design response.

This four-decision report is the input to the Phase 5 main spec, which gets drafted afterward.

---

## 8. Sequencing within Phase 5a

The investigations are roughly independent except for one dependency: Investigation B uses the Investigation A winner. So the order is:

1. **Investigation D** (~5 min) — fastest, gives us a finding we'll want to know before committing.
2. **Investigation A** (~1.5 hr) — picks the curriculum order.
3. **Investigation B** (~3 hr) — uses A's winner; can run during writeup of A.
4. **Investigation C** (~10 min) — uses A's winner; runs after B.

Total wall time: ~5-6 hours, doable in a single afternoon plus an evening.

---

## 9. What Phase 5a is *not*

- **Not the main Phase 5 work.** Phase 5 main spec gets drafted after 5a results. Phase 5a is decision-support.
- **Not an attempt to solve the e=0.3 wall.** Phase 5a's eccentricity bounds are ≤0.3. Pushing past 0.3 (toward Molniya-class) is a Phase 5 main-spec question once the recipe is locked.
- **Not a re-test of R3 components.** Phase 4.5's verdict on DAPO and the implicit verdicts on the other R3 components (L2-init, LayerNorm, adaptive-KL, TEC) carry forward.
- **Not multi-seed confirmation of A or B.** Single-seed first-pass for ordering and dt; multi-seed confirmation happens in Phase 5 main if the chosen pattern requires it.

---

## 10. Out-of-scope but worth flagging for Phase 5 main

These are things the Phase 5 main spec needs to address but aren't Phase 5a's job:

- **Eccentricity ceiling.** How far past 0.3 to push (0.4? 0.5? Molniya 0.7?). Depends on Phase 5a's mid-curriculum results — if e=0.3 is reachable cleanly, we know the recipe scales; if it's marginal, stretch ceiling is academic.
- **Capability surface evaluation.** Grid spacing, episode counts, rollout seeds, eval-during-training cadence. Phase 5 main needs this fully specified — ~150 episodes per grid point gave 11pp std in Phase 4.5; aim for 5pp std → ~600 episodes per point.
- **Attribution sub-experiments at the end of Phase 5 main.** Strip-LVLH, strip-shaping, vary-warp on the new capability frontier. Tells us whether the recipe ingredients carry to higher difficulty or only worked at e=0.
- **Phase 6 readiness checklist.** Eval pipeline parameterized over arbitrary task parameters; curriculum scheduler abstracted; trajectory logging extensibility for n-body bodies; reward decomposition modularized.

---

## 11. Research notes — what informed this spec

Brief annotations on findings that updated the Phase 5a design:

- **DORAEMON (ICLR 2024, Tiboni et al.)**: gradually expanding the sampling distribution outperforms training on the maximum-entropy distribution from scratch. Confirms curriculum > fresh-at-target. Provides an automatic curriculum framework (expand entropy gated on success rate) that may generalize Phase 5's hand-coded stages. *Worth considering for Phase 5 main if the hand-coded stages prove brittle.*
- **Goldilocks RL (Mahrooghi, Lotfi, Abbe 2026)**: sample at the frontier of current capability. Implies that within each Phase 5 stage, weighting episodes toward marginally-failing difficulty rather than uniform-random would help. *Out of scope for Phase 5a but worth Phase 5 main consideration.*
- **Curriculum Learning for RL Domains (survey, Narvekar et al. 2020)**: reverse-curriculum (starting state difficulty) is a related but different approach, less applicable here because our task parameter is geometric, not state-distribution-based.
- **DAPO and on-policy plasticity literature**: already incorporated into Phase 4.5 verdicts; Phase 5a doesn't revisit.

The conceptual upgrade from prior phases: **task parameter sampling** (DORAEMON / Goldilocks framing) is a sharper mental model than **stage curriculum** (Phase 4 framing). The Phase 5 main spec should adopt this lens explicitly.

---

*Author: 2026-04-26. Pre-Phase-5 decision-support spec. Successor: Phase 5 main spec, drafted after 5a results land.*
