# Phase 5b Step 1 — Task Redefinition and Initial Validation

> **Status:** 2026-04-28. Draft after the audit findings (PHASE5b_AUDIT_FINDINGS.md). This doc covers the immediate next chunk of work: redefining the rendezvous task to its general form, restructuring the curriculum to separate phasing-on-eccentric-orbits from orbit-shape change, validating that the new recipe can bootstrap, and isolating REL_VEL_TOL annealing as the intervention to attribute. Compute estimate: ~3-5 hours, possibly more depending on training duration.

---

## 0. Reframing what Phase 5 is actually trying to do

The audit findings closed two questions and opened one. Closed: the dashboard `perf` metric is rolling-window noise (use eval_checkpoint.py); σ₃ is structurally dead in the current recipe but the rest of the gated shaping is fine; failure mode at e>0 is uniformly safety-cap timeout (reward reachability is the binding constraint). Open: Phase 4's recipe is a circular-chaser specialist that doesn't transfer to eccentric targets, and the proposed fixes (REL_VEL_TOL annealing, random satellite eccentricity initialization) address reachability but represent different commitments.

Phase 5b Step 1 commits to a sharper framing of the task. The deliverable Phase 5 is targeting is:

> **Rendezvous from any orbit to any other orbit, with whatever phase and eccentricity, provided it is physically possible with given fuel.**

The Phase 4 recipe is a special case of this — circular chaser, circular target, varying phase gap. It is not a foundation we extend; it's the simplest corner of a larger task cube. Phase 5b Step 1 redefines the env to the general task and trains from scratch under it. This is more work upfront and produces a more capable agent, with a cleaner narrative about what was actually being learned.

The implication: the Phase 4 ckpt becomes a historical reference, not a warm-start source. Phase 5 trains fresh. Compute is cheap; correctness is not.

---

## 1. Task redefinition

### 1.1 Random satellite eccentricity at reset

Currently `orbital.h:758` hardcodes `sat.orbit.e = 0`. Replace with:

```c
// Sample chaser's initial orbital state from the same distribution as target
sat.orbit.e = uniform(0, env->e_max_sat);
sat.orbit.omega = uniform(0, 2 * M_PI);
// sat.orbit.a remains uniform within altitude band as before
// sat.orbit.theta remains uniform [0, 2π) as before
```

This adds two new env config kwargs: `e_max_sat` (analogous to `e_max_target`) and possibly `omega_sat_random` if we want a flag. Keep `omega_sat` always randomized when `e_max_sat > 0`; ω is undefined for circular orbits anyway.

The chaser's mass, fuel, and discrete-action repertoire stay unchanged. Only the *initial orbital state* generalizes.

### 1.2 What this changes about the observation space

The agent already has access to its own orbital elements (sat.a, sat.e, sat.ω in some encoding) via the existing observation space. Phase 4 observation included these but the values were degenerate (sat.e always 0, sat.ω undefined). Under random sat init, these become live signals that the policy learns to use. No observation-space changes required, but the policy's effective input distribution broadens.

This is the change with the deepest implications for learning. Phase 4's network never had to use sat.e or sat.ω as input — they were constants. Under random sat init, they're informative. The policy effectively has more input channels to learn from, which is data augmentation in the goal-conditioned-RL sense (DORAEMON-style state distribution broadening).

### 1.3 Implications for the recipe

Most of the Phase 4 recipe carries forward unchanged:

- Discrete(10) action space (coast, prograde ±5/±10/±25, retro ±5/±10/±25, radial in/out, warp 5min)
- LVLH-frame observations (38-dim)
- Gated NHR shaping (Φ_orbit + σ₂·Φ_phase + σ₃·Φ_vel) with terminal Φ-clamp
- Phase 4 hyperparameters (gamma=0.995, lr=0.01, ent_coef=0.01, minibatch=8192, num_envs=1024)

What changes:

- `e_max_sat` becomes a curriculum axis alongside `e_max_target` and `init_phase_gap_max`.
- The success criterion (rendezvous within position tolerance + relative velocity tolerance) is unchanged. This is still position+velocity matching, not orbit-shape matching. We're not committing to the Audit's option 3 (require `|Δa|<TOL` and `|Δē|<TOL`); position+velocity is what real rendezvous requires.

### 1.4 What this is *not*

- Not a change to the success criterion. Still position+velocity matching.
- Not a change to the action space.
- Not a change to the observation space dimension. Same 38 floats; some of them just have more dynamic range now.
- Not abandonment of Phase 4 work. The Phase 4 recipe components (LVLH obs, gated shaping, warp action, curriculum) all carry forward; the env is what generalizes.

---

## 2. Curriculum: separate phasing skills from orbit-shape change

The current curriculum mental model — "phase gap × eccentricity" — collapses two distinct skills into a difficulty box. Under the new task definition, separate them explicitly.

### 2.1 The four-stage curriculum

| Stage | sat init | target init | Phase gap | What's learned |
|---|---|---|---|---|
| **1 — Same-orbit rendezvous** | sat.e ∈ [0, e_max], sat.ω = target.ω, sat.a = target.a | target.e ∈ [0, e_max], target.ω random | Uniform(0, π) | Phase matching on arbitrary eccentric orbits — chaser and target on the *same orbit*, different positions |
| **2 — ω-match rendezvous** | sat.e = target.e, sat.a = target.a, sat.ω random | target.e ∈ [0, e_max], target.ω random | Uniform(0, π) | ω matching while phasing — same orbit shape and size, different orientation |
| **3 — Transfer rendezvous** | sat.e = target.e, sat.a random in band, sat.ω = target.ω | target.e ∈ [0, e_max], target.a random, target.ω random | Uniform(0, π) | Semi-major axis transfer with same eccentricity |
| **4 — Fully general** | sat.{e, ω, a} all random | target.{e, ω, a} all random | Uniform(0, π) | Arbitrary-to-arbitrary rendezvous |

Each stage holds two of the (sat.e ↔ target.e, sat.ω ↔ target.ω, sat.a ↔ target.a) constraints fixed and varies the third. By Stage 4 all three are decoupled.

The eccentricity bound `e_max` itself is also a curriculum dimension within each stage. Within Stage 1, we'd start at e_max=0.05 and expand toward e_max=0.5+ before transitioning to Stage 2.

### 2.2 Physics callout — Stage 1 is the hardest version of the simplest skill

Phasing on a circular orbit has uniform angular velocity: a phasing maneuver (raise or lower semi-major axis temporarily) produces predictable Δθ over time. Phasing on an eccentric orbit is harder: the chaser moves faster at periapsis, slower at apoapsis, so a phasing burn at the wrong true anomaly produces unpredictable phase drift.

Stage 1 *will* be hard. Don't expect it to converge as fast as Phase 4's circular-orbit phase ramp. Treat slow Stage 1 convergence as expected, not as evidence of a bug.

### 2.3 Eccentricity expansion within each stage

Within each stage, we expand the eccentricity bound on a sub-curriculum:

- Stage 1.0: e_max = 0.05
- Stage 1.1: e_max = 0.10
- Stage 1.2: e_max = 0.20
- Stage 1.3: e_max = 0.30
- Stage 1.4: e_max = 0.50 (Molniya-class territory)

Move to next sub-stage when current sub-stage hits ≥ 70% held-out eval. This is the success-rate-gated stage transition recommended in the Phase 5a addendum (DORAEMON-style adaptive curriculum), now applied to the eccentricity dimension within each stage.

### 2.4 Same-eccentricity vs same-orbit-shape distinction

Stage 1 is "sat and target on the *exact same orbit* (same a, e, ω), different θ." This is more constrained than "sat and target with same eccentricity but different a/ω." We start at the most constrained version because it isolates phasing as the only learning task. Stages 2-4 progressively relax constraints.

---

## 3. The R-vs-G diagnostic (5 min, run before any other work)

Before implementing the task changes, settle the question raised in our analysis of the audit: is the e>0 failure dominated by reward reachability (R — agent maneuvers but never finds success) or gradient absence (G — agent doesn't maneuver because shaping signal is too weak)?

### 3.1 What to measure

From existing audit logs (200 episodes of A1 ckpt at e=0.05, 50 episodes at e=0):

For each set, compute and compare:

1. **Mean total Δv per episode.** Sum of |Δv| over all burn actions in the episode.
2. **Burn frequency.** Fraction of steps that were non-coast actions (action != 0 and != warp).
3. **Action distribution.** Histogram across actions 0-9.

Compare e=0 (recipe works, 84% success) to e=0.05 (recipe fails, 1.5% success).

### 3.2 Decision rule

| Mean Δv at e=0.05 vs e=0 | Burn frequency at e=0.05 vs e=0 | Mechanism | Phase 5b implication |
|---|---|---|---|
| Similar (within 30%) | Similar (within 30%) | **R — Reachability.** Agent is trying. The fixes (random sat init + REL_VEL_TOL anneal) address the right thing. | Proceed with Step 1 plan as written. |
| Much lower at e=0.05 (>50% lower) | Much lower (>50% lower) | **G — Gradient absence.** Agent isn't trying because shaping signal is too weak. Reachability fixes alone won't help. | Pause. Add shaping changes (raise EPS_ORBIT, retune σ₂ threshold) before validation run. |
| Much higher (>50% higher) | High burn rate | **G' — Random thrashing.** Agent burning randomly without convergence. Suggests entropy collapse or value miscalibration. | Pause. Investigate entropy and value loss curves before adding interventions. |

### 3.3 Compute

5 minutes of post-hoc analysis on existing logs. No training. Do this first; it might change what Step 1 looks like.

---

## 4. Implementation

### 4.1 Env changes (orbital.h)

1. Replace the `sat.orbit.e = 0` line at orbital.h:758 with random sampling per §1.1.
2. Add `e_max_sat` env kwarg, plumbed through binding.c and orbital.py.
3. Add `omega_sat_random` flag (default true when `e_max_sat > 0`).
4. Verify via standalone C test (`orbital.c`) that random sat init produces valid orbits at extreme cases (e=0.5, ω=π, etc.) — no NaNs, no escape on init, propagation stable.

### 4.2 REL_VEL_TOL annealing schedule

REL_VEL_TOL is currently 50 m/s, a physical constant defining success. For annealing, make it a config-controlled curriculum parameter:

```ini
# In orbital.ini
rel_vel_tol_initial = 200.0
rel_vel_tol_final = 50.0
rel_vel_tol_anneal_fraction = 0.5  # anneal over first 50% of total_timesteps
```

Schedule: linear interpolation from initial to final over the first `anneal_fraction` of training steps; constant at final for the rest.

This becomes the testable intervention. We run validation both with and without it (§5.3).

### 4.3 Eval-during-training infrastructure

Per the audit's recommendation, replace the dashboard `perf` mirage with held-out eval. Specifics:

- Eval every 200k steps for the first 5M steps of any run (catches early peaks).
- Eval every 500k steps thereafter.
- Eval condition matches training condition (same e_max, same phase_gap_max).
- Eval uses argmax actions, 50 episodes, fixed eval seed for reproducibility within a run (different across runs).
- Log to wandb under separate metric name `held_out_success` to distinguish from the dashboard `perf`.

The training loop change is small (insert eval call every N steps, log result). One-time engineering cost ~1 hour.

**Mirage guard.** If `dashboard_perf - held_out_success > 10pp`, log a warning. Catches future drift between the two metrics.

### 4.4 Curriculum scheduler

Stage transitions are gated on held-out success rate, not on step count:

```python
# Pseudocode
if held_out_success >= 0.70 and current_stage < final_stage:
    advance_stage()
```

For Step 1's validation run we don't need a full multi-stage pipeline yet — we just need Stage 1.0 (smallest bounds) to bootstrap. The full scheduler is a Phase 5 main item.

---

## 5. Validation protocol

### 5.1 Stage 1.0 single-seed validation run

Train from scratch under the new env (random sat init enabled, REL_VEL_TOL annealing on, all Phase 4 recipe components), Stage 1.0 bounds:

- `e_max_sat = e_max_target = 0.05`
- `init_phase_gap_max = π` (full phase gap, since sat and target are on the same orbit only different θ)
- Constraint: sat.a = target.a, sat.ω = target.ω (Stage 1's defining constraint)
- Total timesteps: 20M (longer than Phase 4 because the task is harder and random sat init broadens the state distribution)
- Single seed (42) for first-pass

Compute estimate: ~25 minutes wall time. This is the run that tells us whether the new recipe bootstraps at all on the same-orbit rendezvous task.

### 5.2 Success criteria

| Held-out eval at end of Stage 1.0 training | Verdict |
|---|---|
| ≥ 50% | Recipe bootstraps on same-orbit eccentric rendezvous. Proceed to multi-seed validation, then Stage 1.1 (e_max=0.10). |
| 20-50% | Partial bootstrap. Investigate trajectories, check σ₃ activation rate, possibly extend training to 30M before declaring failure. |
| < 20% | Bootstrap failure. Run R-vs-G diagnostic on the new logs. Possibly add shaping changes before re-running. |

### 5.3 REL_VEL_TOL annealing attribution

After §5.1's run lands a result, run one ablation: same recipe, random sat init still on, but REL_VEL_TOL fixed at 50 m/s (no annealing). Single seed. Compare to §5.1.

| §5.1 success rate | §5.3 success rate (no anneal) | Verdict |
|---|---|---|
| ≥ 50% | ≥ 50% | REL_VEL_TOL anneal isn't contributing under this recipe. Drop it. |
| ≥ 50% | < 30% | REL_VEL_TOL anneal is doing real work. Keep in Phase 5b recipe. |
| < 50% | regardless | §5.1 didn't bootstrap; ablation is meaningless until §5.1 succeeds. |

### 5.4 σ₃ activation check

In the Stage 1.0 run's logs, compute σ₃ activation rate across the training trajectories. Compare to the audit's e=0 result of 1.6%.

| σ₃ activation in §5.1 logs | Verdict |
|---|---|
| > 5% | Random sat init incidentally fixes σ₃'s deadness. Recipe is genuinely three-component now. Free improvement. |
| ≤ 5% | σ₃ remains dead. EPS_PHASE retune is still needed; defer to Phase 5b Step 2 or document as known limitation. |

This is post-hoc, no extra runs. ~5 minutes of analysis.

### 5.5 What §5 collectively tells us

Five numbers come out of Step 1:
1. R-vs-G diagnostic result (mechanism dominant at e>0 in current recipe).
2. Stage 1.0 held-out success rate (does the new recipe bootstrap?).
3. REL_VEL_TOL annealing contribution (is the anneal pulling weight?).
4. σ₃ activation rate under random sat init (does generalizing the env fix the dead gate?).
5. Trajectory inspection — are agents executing recognizable phasing maneuvers, apsidal burns, or some emergent strategy?

These five numbers determine the Phase 5b spec's full scope. If §5.1 lands ≥ 50%, Phase 5b is straightforward extension through the four-stage curriculum. If it doesn't, we go back to investigation.

---

## 6. Compute summary

| Activity | Estimate |
|---|---|
| §3 R-vs-G diagnostic (post-hoc on existing logs) | 5 min |
| §4 Implementation (env changes + eval-during-training infra) | ~3-4 hours engineering |
| §5.1 Stage 1.0 validation run | 25 min training + 5 min eval |
| §5.3 REL_VEL_TOL ablation | 25 min training + 5 min eval |
| §5.4 σ₃ check | 5 min analysis |
| **Total** | **~5 hours wall, mostly implementation** |

---

## 7. What this is *not*

- **Not a full Phase 5b spec.** Step 1 settles whether the redefined task bootstraps. Phase 5b spec covers the full four-stage curriculum, multi-seed protocol, capability surface eval, attribution sub-experiments — drafted after Step 1's results land.

- **Not a commitment to four-stage curriculum if simpler works.** If Stage 1 → Stage 4 collapses to two effective stages (e.g., Stage 1 generalizes to Stage 4 without intermediate training), we use the simpler curriculum. The four stages are decomposition aids, not mandatory stations.

- **Not abandonment of the Phase 4 recipe.** The recipe (LVLH obs, gated shaping, warp action, Discrete(10) actions) is preserved. Only the env's chaser-init distribution generalizes.

- **Not a multi-seed run.** Step 1 is single-seed first-pass. Multi-seed validation is Phase 5b proper.

- **Not addressing Phase 6 (multi-body) hooks.** Step 1 stays in two-body. Phase 6 readiness items (eval pipeline parameterization, dynamics interface abstraction) are Phase 5 main concerns, not Step 1.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Random sat init produces unstable orbits at high e (e.g., e > 0.7 → numerical issues with Kepler propagator). | Low | Standalone C test in §4.1. Cap e_max_sat at 0.5 for Stage 1.0; expand later. |
| Stage 1 (same-orbit rendezvous on eccentric orbits) is harder than Phase 4 (circular-orbit phasing) and 20M steps isn't enough. | Medium | Pre-committed: extend to 30M before declaring failure. Document expected difficulty in §2.2. |
| Eval-during-training infrastructure introduces a bug that affects training (e.g., env state contamination during eval). | Medium | Eval uses a separate env instance, not the training envs. Verify with a unit test that training is byte-identical with vs without eval-during-training enabled. |
| REL_VEL_TOL annealing schedule is wrong (anneals too fast or too slow). | Medium | Pre-registered schedule (linear over first 50%) is principled but untested. If §5.1 fails, try alternative schedules (cosine, step) before declaring REL_VEL_TOL anneal a failure. |
| The R-vs-G diagnostic is ambiguous (e.g., some signal suggests R, some G). | Medium | Default to R interpretation (it's the simpler hypothesis matching audit findings). Run §4-§5 anyway; the validation run itself further diagnoses the mechanism. |
| The four-stage curriculum's Stage transitions don't compose (e.g., Stage 1 policy is too specialized to transfer to Stage 2). | Medium | Step 1 doesn't test stage transitions — that's Phase 5b proper. If transitions fail later, revisit curriculum decomposition. |

---

## 9. After Step 1 lands

Phase 5b spec gets drafted with Step 1's five numbers baked in. The key questions Phase 5b addresses:

1. Multi-seed reproducibility of Stage 1.0 result.
2. Stage 1.1, 1.2, 1.3, 1.4 (eccentricity expansion within Stage 1).
3. Stage transitions (Stage 1 → Stage 2 → Stage 3 → Stage 4).
4. Capability surface evaluation across the (sat.e, target.e, phase_gap, sat.a/target.a, sat.ω/target.ω) space.
5. Attribution sub-experiments on the new capability frontier.
6. Phase 6 hooks (eval pipeline, curriculum scheduler, dynamics interface).

Phase 5b is a 2-3 week effort once Step 1 lands. Step 1 is the gate that decides whether to commit that effort.

---

## 10. The sequencing logic

To be explicit about why this is "Step 1" and not "the full plan":

The project has now had three successive moments where the plan needed to change after data came in (Phase 3→4 reframe, Phase 4→4.5 reframe, Phase 5a's entry-condition discovery, Phase 5a→5b mechanism resolution). At each, drafting the full subsequent plan before getting the data wasted effort because the plan needed restructuring after the data landed.

Step 1's deliverable — "does the redefined task bootstrap, and which interventions matter" — is decision-blocking for everything that follows. Drafting the full Phase 5b spec before getting this answer means committing to a curriculum we may have to discard. The discipline of "small focused step, then plan the next chunk" has worked three times in this project; doing it again here is the same playbook.

If Step 1 lands cleanly (≥50% on Stage 1.0 with the new recipe), Phase 5b is a straightforward extension through the four-stage curriculum. If it doesn't, Phase 5b is a different shape and Step 1 told us what shape that is.

---

*Author: 2026-04-28. Pre-Phase-5b implementation and validation. Successor: full Phase 5b spec, drafted after Step 1's five-number deliverable lands.*
