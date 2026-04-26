# Orbital RL — Phase 4 Technical Spec

> **Status:** 2026-04-23. Supersedes `orbital-rl-phase4-plan.md` (2026-04-21) where the two documents conflict. The prior plan framed four interventions (time-warp, physics-based reward, continuous actions, scale-up) as a menu. Research has since produced a structural diagnosis of the 50% plateau that changes what those interventions *mean* and which ones are worth doing. This document course-corrects accordingly.

---

## 0. Executive summary

The 50% ceiling at 180° phase-gap rendezvous is not a ceiling. It is the expected behavior of vanilla PPO on a task whose observation representation aliases at the orbital frequency, whose shaping and discount factors are mis-specified for the effective decision horizon, and whose hyperparameter defaults (10 PPO epochs, symmetric clipping, no entropy floor, no plasticity mitigation) were calibrated for dense-reward MuJoCo control — the exact opposite of our regime. The research report compiled on 2026-04-23 identifies three compounding root causes and maps each to cheap, well-validated fixes from published work in 2023–2026.

This spec is organized around those three root causes, not around the four-item intervention list. It folds in what the Phase 4 plan got right (time-warp, gated multi-stage shaping) and explicitly kills or demotes what the plan got wrong (continuous-action revisit, aggressive action-space expansion, un-ablated joint deployment). It also adds two categories of fix the plan did not contain — representation change (CW/Hill frame) and plasticity mitigation (LayerNorm + L2-init + DAPO clip-higher + adaptive-KL) — which the research indicates are higher-leverage than anything already on the list.

**The one-sentence reframe:** stop trying to make a distance-reward-on-inertial-state PPO agent learn phasing orbits, and start building an environment where phasing orbits are the path of least resistance for a correctly-regularized PPO agent.

---

## 1. First-principles framework

### 1.1 The three invariants worth defending

Every decision in Phase 4 should be checked against three invariants. If a proposed change violates one, it needs a strong justification or it gets rejected.

**Invariant 1 — The agent's observation must be Markov for the task.** A 180° phase-gap rendezvous requires decisions conditioned on relative geometry (where is the target *relative to me, right now*, and how is that changing). Raw orbital elements plus a sin/cos of true anomaly do not satisfy this cleanly: reward structure aliases at the orbital period, and a fixed-step discount sees the same apparent state at ω = 0° and ω = 180° of the target. If the state is not Markov in a relative frame, the recurrent network has to recover the relative frame implicitly from inertial history — which is exactly the kind of representational load that causes rank collapse under PPO. Make the state Markov before asking the network to do anything else.

**Invariant 2 — The reward function must not punish the optimal trajectory.** A fuel-efficient 180° rendezvous requires a phasing orbit: temporarily move *away from* the target's current location so that orbital-period mismatch closes the phase gap over several revolutions. Any shaping whose gradient points "reduce Euclidean distance now" punishes the first half of the only fuel-feasible maneuver class. This is the single most important thing the original plan got right (§4.1 of phase4-plan.md), and it remains true: replacing distance-based shaping with a physics-aware decomposition is non-negotiable.

**Invariant 3 — PPO's trust region must remain trustworthy.** The symptom set (entropy < 0.5 nats by epoch 40, stable value loss, 76% mid-episode abort, stochastic eval matching greedy eval within noise) is a textbook signature of representation collapse → entropy collapse → local-optimum lock-in. Published 2024 work (Moalla et al., Klein et al.) shows this is a *specific* PPO-on-sparse-reward failure mode with *specific* fixes. Fighting it with curriculum or warm-starts does not work; fighting it with regularization (LayerNorm + L2-init) and trust-region adjustments (adaptive KL, asymmetric clip) does.

### 1.2 Reasoning principles going forward

These are rules for how we evaluate proposed interventions, not interventions themselves. Violating them is the path back to the 50% plateau.

**Principle A — Diagnose before intervening.** Every intervention in Phase 3 was a proposed solution to an *assumed* failure mode (too coarse actions, too coarse timestep, too narrow action space). None of those were actually the failure mode. The diagnostic data were available the whole time — entropy curves, mid-episode termination distribution, value-loss dynamics — but the interventions weren't being checked against them. Going forward: if a proposed change doesn't have a hypothesis linking it to a measurable diagnostic, do not run it. If an ablation doesn't produce a diagnostic signature consistent with the hypothesis, do not trust its success.

**Principle B — Prefer architectural and representational changes over algorithmic ones.** Representation changes (CW frame, better reward decomposition, LayerNorm) are *sticky* — they carry forward through every subsequent experiment. Algorithmic tweaks (ent_coef, LR schedule, clip range) are fragile — they need to be re-tuned whenever something upstream changes. When both options are available, invest the engineering time in the sticky version.

**Principle C — Do not expand the action or policy space mid-curriculum.** The research report is unambiguous: no published PPO technique reliably expands Discrete(N) → Discrete(M) or Discrete → Continuous across curriculum boundaries without substantial custom machinery. Phase 3 confirmed this twice (D7→D11 regression; Box(2) collapse). The correct move is to fix the action space at the start of a training run and never touch it mid-run.

**Principle D — Do not warm-start across distribution shifts without explicit anti-forgetting machinery.** The Phase 2→3 warm-start collapses (Stage 2 collapse at e=0.10, Stage 3g regression from 50% to 24%) are the same bug documented in Wołczyk et al. (ICML 2024). Vanilla PPO fine-tuning across curriculum boundaries is known to fail. If we warm-start, we use BC-kickstart + value pretraining + LR warmup. Otherwise, we train from scratch at the target difficulty.

**Principle E — Attribution is not optional.** The Phase 4 plan proposed shipping time-warp and physics-reward together because "they compose." This is correct in principle and wrong in practice. If the combined run succeeds, we have no idea which intervention did the work, which means we also have no idea what to do next when the *next* plateau appears. Every new intervention gets a minimum-viable ablation: a 2–5M step debug run with just that one change, measured against the same metrics as the main run. The compute cost is trivial compared to running a multi-day curriculum and discovering you can't explain the result.

**Principle F — True success rate is the only metric. Shaped reward is a diagnostic.** Goodharting on shaped reward is a known failure mode (the research report highlights it explicitly). We should plot true terminal-success rate at every checkpoint, separately from shaped return, and compare trajectories. If shaped return is climbing but success rate is flat, the shaping is wrong and we stop before burning more compute.

**Principle G — "How would you explain this result to a hiring manager at APL?" is a legitimate filter.** This is a portfolio project with a specific audience. A clean diagnostic narrative ("the 50% plateau was entropy collapse; here's the fix with a citation trail") is worth more than a 10-percentage-point bump with no explanation. When two approaches are roughly equivalent on expected-success, pick the one with the cleaner story.

### 1.3 What this framework discourages

To make the reframe concrete, here are things the original plan contemplated that this spec actively discourages:

- **Continuous actions (Box(2)) in any form, including warm-started.** Phase 3's attempt failed; the research report explains exactly why (Gaussian mass near zero + clamp pathology + entangled magnitude-timing) and confirms there is no published fix that ports cleanly to PPO. Discrete(9) is the productive baseline on this env. If we want finer control at handover, use a richer discretization trained from scratch, not Box.
- **Scale-up as an early intervention.** Doubling LSTM width or training steps on a broken setup produces a more confidently broken setup. Scale-up is on the table, but only after the representation/shaping/plasticity fixes are in and attribution has been done.
- **Mid-curriculum weight surgery.** The D7→D9 surgery in Phase 3f (which briefly hit 50%) was the kind of move that works once and then poisons your ability to reproduce the baseline on a fresh seed. Mid-curriculum surgery is banned going forward — see §5.
- **Joint-shipping interventions to save compute.** See Principle E.
- **Shaping weights tuned by hand with no calibration protocol.** The Phase 4 plan's "tune offline" was underspecified. This spec replaces it with a percentile-calibrated protocol from published rendezvous-RL work.

---

## 2. Diagnosis: the three compounding root causes

This is the load-bearing section. Every intervention in §3 maps to a root cause here.

### 2.1 Root cause 1: observation representation aliases at the orbital frequency

**Symptom.** Mid-episode give-up at 76% of failures, with entropy collapsed to <0.5 nats. The agent has converged to a deterministic policy that terminates early because the value function cannot distinguish "I am 300 steps into a correctly-executed phasing orbit" from "I am 300 steps into a failed direct-intercept attempt." Both states look similar in the current observation (sat elements, target elements, relative geometry in inertial frame), but they have drastically different values.

**Mechanism.** Orbital motion is periodic. In an inertial-or-near-inertial frame, the agent at `θ_sat = 0°` on its orbit looks nearly identical to the agent at `θ_sat = 360°` on the same orbit. The relative geometry between sat and target oscillates on the orbital period. A critic trying to estimate `V(s)` from these observations is fitting a function that repeats every ~90 minutes of sim time — with a 60s timestep, that's 90 observations that the critic needs to map to very different values depending on where we are in the episode. The research report notes this is exactly the regime in which the Federici/Zavoli/Furfaro orbital-RL school rejected inertial-frame observations in favor of rotating frames (Clohessy-Wiltshire / Hill / LVLH for circular targets, Tschauner-Hempel for elliptic, Modified Equinoctial Elements for multi-revolution transfers).

**Consequence for our env.** Our Box(33) observation is heavily inertial-frame. Although we encode sin/cos of ω and sin/cos of relative phase angle (good), the underlying sat and target elements are absolute. The critic has to implicitly subtract one from the other to get the Markov-relative-state that the task actually depends on — and that subtraction is part of the learning problem. Under PPO's representation-collapse dynamics, this is exactly the kind of implicit computation that gets sacrificed first when the trunk's rank drops.

**Fix in §3.1:** switch to CW/Hill-frame relative state as primary observation. Keep absolute elements as auxiliary.

### 2.2 Root cause 2: reward and discount are mis-specified for the effective decision horizon

**Symptom.** Distance-based shaping (`c_dist = 0.05`) plateaus at 50% regardless of shaping weight (we tried 0.01–0.2). Removing shaping entirely collapses to 0%. Adding phase-based shaping (c_phase=2.0, Stage 3d) collapsed to 0%. Balanced shaping (Stage 3e) got 37%.

**Mechanism, part 1 — reward gradient mismatch.** A 180° phase-gap phasing orbit requires the chaser to *increase* Euclidean distance to the target during the drift phase (moving to a lower orbit, drifting under higher mean motion, then climbing back). Distance-closing shaping assigns negative reward to exactly this trajectory. Phase-closing shaping assigns negative reward to the initial plane-change burn. Neither is a Lyapunov function on Kepler orbits. The only correct shaping signal is one that respects orbital-element geometry — which is what the gated orbit-match → phase-match → velocity-null decomposition in §3.2 provides.

**Mechanism, part 2 — SMDP correctness.** Even if the shaping were geometrically correct, the per-step discount is wrong. Ng-Harada-Russell potential-based shaping `F = γΦ(s') − Φ(s)` is policy-invariant only when `dt` is constant. If we add time-warp actions (which is on the plan), or if we aggregate sub-steps internally, the correct form is `F = γ^τ Φ(s_{t+τ}) − Φ(s_t)` where τ is the elapsed sub-step count. The per-step TD target and GAE similarly need `γ^τ` factors. Naively keeping the constant-dt form under variable dt produces a systematic bias that, for γ=0.99 and τ=30, is approximately `(1 − 0.99^30)/(1 − 0.99) ≈ 26×` the per-sub-step potential — enough to completely dominate the terminal reward and lock in a wrong policy.

**Mechanism, part 3 — terminal-state boundary condition.** Ng-Harada-Russell requires `Φ(s_terminal) = 0`. If we leave nonzero Φ at the terminal state (including time-limit truncation), the policy has a direct incentive to terminate early inside a locally-attractive low-Φ basin — which is a plausible contributor to our 76% mid-episode abort rate.

**Fix in §3.2:** gated multi-stage shaping with `1 − cos(θ)` phase term, percentile-calibrated thresholds, SMDP-correct discount on all time-warp sub-steps, hard Φ=0 clamp at termination.

### 2.3 Root cause 3: PPO is undergoing representation → entropy → policy collapse

**Symptom.** Entropy falls below 0.5 nats (out of max 2.197 for Discrete(9)) by epoch 40. Value loss is stable. Policy converges to a deterministic branch. Warm-start fine-tuning regresses rather than improving.

**Mechanism.** This is the failure mode characterized in two NeurIPS 2024 papers (Moalla et al., *No Representation, No Trust*; Klein et al., *A Study of Plasticity Loss in On-Policy Deep RL*). The sequence is:
1. Early in training, the critic's feature representation collapses in rank — features that distinguish valuable states degrade faster than features that distinguish common states.
2. Rank collapse degrades the value estimates used to compute advantages.
3. Degraded advantages make PPO's clipping-based trust region less informative (the clipping bounds the *ratio*, not the *magnitude* of the update, and when advantages are noisy the clip no longer prevents harmful updates).
4. Entropy collapses because the policy follows noisy advantages into whatever local optimum shows positive advantage first.
5. Warm-starting on top of a collapsed representation inherits all the above.

**Mechanism, secondary — symmetric clipping.** PPO's standard clip `(1 − ε, 1 + ε)` bounds the upside ratio at 1.2 but still allows the downside ratio to drop to zero. For low-probability exploratory actions (which is *every* action once entropy has collapsed), this is asymmetric: they cannot gain probability fast enough to recover. DAPO (Yu et al., 2025) reports that raising `ε_high` to 0.26–0.28 while keeping `ε_low = 0.2` eliminates entropy collapse on reasoning-RL tasks at 50% of training steps.

**Mechanism, tertiary — 1024 envs × 10 PPO epochs.** Bharthulwar et al. (2025) on 1M-parallel-env PPO recommend *reducing* epochs when envs are plentiful, because each epoch's updates on nearly-identical rollouts compound the rank-collapse dynamics rather than adding diversity. Our 1024 × 10 is a known pathology.

**Fix in §3.3:** LayerNorm + L2-init for plasticity. DAPO clip-higher. Adaptive-KL LR scheduling. Target-entropy controller with floor and bump. Drop PPO epochs to 3. Drop clip range to 0.15.

### 2.4 What the diagnosis means for the Phase 4 plan

Re-reading `orbital-rl-phase4-plan.md` against this diagnosis:

| Plan item | Verdict | Reason |
|---|---|---|
| (1) Time-warp actions | **Do, but reframe** | Correct intuition (horizon collapse), but the effect is 3rd-order compared to fixing representation and plasticity. Also needs SMDP-correct discount to avoid the γ^τ bias (§2.2). |
| (2) Physics-based reward | **Do, with stricter protocol** | Correct direction. Gate calibration needs to follow a published protocol (§3.2), not hand-tuning. Needs the `1 − cos(θ)` phase term and Φ=0 terminal clamp. |
| (3) Continuous actions revisit | **Kill.** | Research report is explicit: no published PPO technique reliably expands discrete → continuous mid-curriculum. Phase 3 confirmed. If fine Δv control is needed at handover, use Discrete(13) fine rungs trained from scratch. |
| (5) Scale-up | **Defer to Phase 5** | Scaling a broken setup produces a confidently broken setup. Gate on §3.1–§3.3 showing improvement. |
| (4) Demonstration bootstrap | **Move to top of backlog** | The research report positively identifies BC-kickstart as the standard fix for the warm-start collapse we keep hitting. If we ever warm-start again, we use BC-kickstart. |

Three new categories of fix the plan did not contain:

- **Representation change (CW/Hill frame).** Not in plan. Highest-leverage change per the research.
- **Plasticity mitigation (LayerNorm + L2-init + DAPO + adaptive-KL).** Not in plan. Directly attacks the measured entropy collapse.
- **Episode-length sufficiency check.** Not in plan. If `max_episode_steps × dt < 1.5 × T_target_period`, a phasing orbit cannot physically complete and the agent correctly learns to give up. Must be verified before anything else.

---

## 3. Interventions

Ordered by leverage. Each intervention has: a hypothesis linking it to the diagnosis, a concrete implementation, a measurable success criterion, and a risk register. No intervention ships without an ablation plan.

### 3.1 Intervention A — CW/Hill-frame relative state

**Hypothesis.** The critic's rank collapse is partly caused by having to implicitly compute relative geometry from absolute elements. Moving to a frame in which the task's natural invariants are explicit will reduce the representational load and should improve value-function stability, which should in turn reduce entropy collapse severity.

**Rationale from research.** Federici, Zavoli, and Furfaro's entire publication line uses CW/Hill or Modified Equinoctial Elements. Federici & Zavoli (Acta Astronautica 2022) specifically report that MLP matches LSTM when state is Markov in these frames. Chen, Phillips & Copp (AAS 23-026) use HCW throughout their ARPOD rendezvous work. The empirical case is strong enough that deviating from it requires a positive justification.

**Implementation.**

1. Add a rotating frame centered on the target, with axes:
   - `x̂` — radial (target-outward)
   - `ŷ` — along-track (target velocity direction, approximately)
   - In 2D, these are sufficient; z-axis is implicit.
2. Primary relative-state observation fields (append to existing Box):
   - `dx, dy` — chaser position in target's LVLH frame, normalized by `R_earth`
   - `dvx, dvy` — chaser velocity in target's LVLH frame, normalized by target orbital velocity
   - `n_target` — target mean motion (scalar, normalized)
3. Keep the existing absolute-element observations as auxiliary for now — do not remove them in the same run as adding relative state, so we can ablate cleanly.
4. Observation dimension grows from 33 to 38 (5 new fields).

**Implementation details (`orbital.h`).** The CW-frame transformation is straightforward at each step:

```c
// Target state (already propagated this step)
double theta_t = env->target_body.orbit.theta + env->target_body.orbit.omega;
double cos_t = cos(theta_t), sin_t = sin(theta_t);

// Rotation matrix inertial → LVLH (target's x̂ points target-outward from Earth)
// In 2D: LVLH x-axis = target's radial; y-axis = target's along-track
double dx_inertial = sat_x - tgt_x;
double dy_inertial = sat_y - tgt_y;
double dvx_inertial = sat_vx - tgt_vx;
double dvy_inertial = sat_vy - tgt_vy;

// Project into LVLH
double dx_lvlh =  cos_t * dx_inertial + sin_t * dy_inertial;
double dy_lvlh = -sin_t * dx_inertial + cos_t * dy_inertial;
double dvx_lvlh =  cos_t * dvx_inertial + sin_t * dvy_inertial;
double dvy_lvlh = -sin_t * dvx_inertial + cos_t * dvy_inertial;
// Plus target rotational-frame correction for velocity (n × r term)
double n_tgt = sqrt(MU / pow(env->target_body.orbit.a, 3));
dvx_lvlh += n_tgt * dy_lvlh;
dvy_lvlh -= n_tgt * dx_lvlh;
```

The `n × r` correction is easy to get wrong — it's the velocity of the LVLH frame itself rotating, and omitting it gives a "pseudo-velocity" that doesn't null when the chaser is at rest in LVLH. Test this with a stationary-at-target scenario: after the burns that zero relative position and velocity, `dvx_lvlh` and `dvy_lvlh` should remain near zero across many propagation steps.

**Success criterion for this change alone (2M step ablation).**
- Greedy eval at 30° phase gap remains ≥ 95% (no regression on easy cases).
- Greedy eval at 90° phase gap ≥ 75% (current checkpoint is 74%).
- Critic value loss decreases faster than baseline over first 1M steps.
- Entropy decay rate slows (entropy at epoch 40 ≥ 0.8 nats, vs current <0.5).

**Risk register.**

| Risk | Likelihood | Mitigation |
|---|---|---|
| LVLH transformation bug silently gives wrong state. | Medium | Unit-test: stationary-in-LVLH trajectory must have `dvx_lvlh ≈ dvy_lvlh ≈ 0`. Also test `dx,dy` match known CW closed-form solutions for small perturbations. |
| Observation-space growth (33→38) triggers weight-shape mismatch vs existing checkpoints. | High (by construction) | Accept it. This run is from scratch; warm-starting from Phase 3 checkpoints across this change is Principle D-forbidden. |
| CW frame is singular when target is on a degenerate orbit (e.g., target itself escape-bound). | Low | Target init parameters already restrict to bounded circular-ish orbits; assert this at init. |
| Keeping both absolute and relative state makes observation high-dim for no benefit. | Medium | Run ablation after the combined-fix run: remove absolute elements, see if performance holds. Defer to Phase 5. |

### 3.2 Intervention B — Gated multi-stage shaping (the physics-based reward, done correctly)

**Hypothesis.** Distance-based shaping punishes the phasing-orbit strategy. A potential function decomposed into orbit-match → phase-match → velocity-null stages, with soft gates, calibrated thresholds, and SMDP-correct discounting, should give PPO a gradient signal that points toward fuel-optimal phasing rather than against it.

**Rationale from research.** Chen, Phillips & Copp (AAS 23-026) use exactly this structure for ARPOD docking, with piecewise-gated reward across three radial bands. Federici, Scorsoglio, Zavoli & Furfaro (Acta Astronautica 2022) demonstrate that ε-constraint curriculum on terminal tolerances is the most frequently-cited plateau-breaker. Li et al. (Advances in Space Research 2024) show exponential-acceleration terminal rewards outperform linear ones for terminal-region convergence. All of these are compatible; we combine them.

**The potential function.**

```
Φ(s) = −[ w₁ · Φ_orbit(s) · σ₁(s)
        + w₂ · Φ_phase(s) · σ₂(s)
        + w₃ · Φ_vel(s)  · σ₃(s) ]

where:
  Φ_orbit(s) = |Δa|/TOL_A + ‖Δē‖      # orbital-element error
  Φ_phase(s) = 1 − cos(Δθ)             # smooth across ±π wrap, NOT |Δθ|/π
  Φ_vel(s)   = ‖v_rel_lvlh‖ / REL_VEL_TOL

  σ₁(s) = 1                                        # orbit-match always active
  σ₂(s) = sigmoid((ε_orbit − Φ_orbit(s)) / τ_orbit)   # activates when orbits close
  σ₃(s) = σ₂(s) · sigmoid((ε_phase − Φ_phase(s)) / τ_phase)  # activates when phased

with τ_k = 0.1 · ε_k (smooth gates)
```

Reward per step:

```
r_step = β · (γ^τ · Φ(s') − Φ(s))
```

where τ is the number of sub-steps elapsed between decisions (1 for normal actions, 5 for warp-5min, 30 for warp-30min — see §3.4).

**Why `1 − cos(Δθ)` instead of `|Δθ|/π`.**
- Smooth everywhere, including at the ±π wrap point.
- Bounded in [0, 2], convenient normalization.
- Derivative is `sin(Δθ)`, which is continuous through the wrap.
- `|Δθ|/π` has a cusp at Δθ = 0 and a jump at Δθ = ±π. Both cause advantage-estimator variance spikes, which compound the rank-collapse dynamics of Root Cause 3.

**Why `γ^τ` not `γ`.** Policy invariance of Ng-Harada-Russell shaping under variable dt. Derivation: for an SMDP with sub-step count τ, the correct Bellman operator uses `γ^τ`, and the shaping telescoping argument only preserves the optimal policy when the same `γ^τ` appears in the shaping term. Using a constant `γ` with variable τ introduces bias of order `(1 − γ^τ)/(1 − γ) · Φ`, which at γ=0.99, τ=30 is 26× the per-sub-step potential — easily enough to dominate terminal reward.

**Why clamp Φ=0 at termination.** NHR policy-invariance's boundary condition. Leaving nonzero Φ at terminal states lets the agent gain free reward by terminating in a locally-attractive Φ basin — which is a plausible contributor to 76% mid-episode abort.

**Terminal reward (replaces current ±10).**

```
On success (rendezvous):
  r_term = 10.0 · (0.5 + 0.5 · fuel_remaining)     # fuel-efficiency bonus
         + 50.0 · exp(−‖d_final‖ / d_scale)       # exponential precision bonus
         + 50.0 · exp(−‖v_rel_final‖ / v_scale)   # exponential velocity-null bonus

  (d_scale = 5 km, v_scale = 10 m/s)

On failure (collision, escape, stranded, safety cap):
  r_term = -10.0
```

The exponential-acceleration terminal reward from Li et al. 2024 gives strong gradient pull in the terminal region where linear distance rewards flatten out. The one-time success bonus of ~100 dominates accumulated shaping (max per-episode |G_shape| ≈ 10), which is the ratio the research report identifies as the correct scale.

**Gate calibration protocol.** Do not hand-tune. Run the protocol.

1. Starting from the current Stage 3f 50%-checkpoint (before any Phase 4 changes), run 200 episodes at 180° phase gap.
2. Record the *minimum* value of `Φ_orbit(s)` reached in each episode (best orbit-match achieved).
3. Set `ε_orbit` at the 60th percentile of these minima. (If 60% of random-to-50% rollouts cross `ε_orbit`, stage 2's gate activates in a meaningful fraction of episodes.)
4. Among the subset of episodes that crossed `ε_orbit`, record the minimum value of `Φ_phase(s)` reached after the first crossing.
5. Set `ε_phase` at the 60th percentile of those minima.
6. Target activation rates: σ₁ active 100%, σ₂ active in ≥50% of episodes, σ₃ active in ≥20% of episodes. If activation rates fall below these, loosen thresholds (go to 70th or 80th percentile).

**Initial weights.**
- `w₁ = w₂ = w₃ = 0.05` (equal; research recommends slight up-weighting of later stages but we start equal for attribution).
- `β = 1.0` (the `c_shape = 0.05` factor is absorbed into `w_k`).

**Success criterion for this change alone (5M step ablation).**
- Greedy eval at 180° phase gap ≥ 55% (incremental improvement over 50% baseline).
- True success rate tracks shaped return (no Goodharting).
- Trajectory plots show visible phasing-orbit structure on at least one successful run: a ≠ a_target during a drift phase that precedes the terminal approach.

**Risk register.**

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gate thresholds are poorly calibrated (gates never open or always open). | Medium | Log σ_k activation rates to trajectory files. Re-run calibration protocol if activation rates fall outside target bands. |
| Weights create a new local optimum (e.g., agent learns to sit at the σ₂ gate edge). | Medium | Monitor trajectory patterns; if agent oscillates around gate boundaries, widen τ_k (softer gates). |
| γ^τ not correctly applied under time-warp (if 3.4 ships together). | High | Unit test: run a fixed warp-30 action, verify `r_step = β · (γ^30 · Φ_next − Φ_prev)` numerically. Explicit test case. |
| Terminal Φ clamp missed in one of the four termination branches. | High | Terminal-state assertion: log `Φ(s_terminal)` at every episode end, alert if nonzero. |
| Exponential terminal bonus lets agent "collect the bonus early" via repeated approaches. | Low | Bonus is applied once at terminal success. Cannot be reaccessed. |
| Shaping dominates terminal reward (Goodhart). | Medium | Hard check at end of every training run: accumulated |G_shape| per episode must be ≤ 0.1 · |R_terminal|. If violated, halve all w_k. |

### 3.3 Intervention C — PPO plasticity mitigation

**Hypothesis.** The entropy collapse symptom is caused by representation rank collapse. Direct regularization of the representation (LayerNorm + L2-init) plus trust-region adjustments (asymmetric clip, adaptive KL, entropy floor) will prevent the collapse, keeping the agent in an explorable region long enough for learning to find the phasing-orbit basin.

**Rationale from research.** Moalla et al. (NeurIPS 2024) identify rank collapse → trust-region failure → performance collapse as the canonical PPO-on-sparse-reward failure mode. Klein et al. (NeurIPS 2024) find that for PPO specifically, plasticity injection and neuron resets do *not* work; LayerNorm + L2-init does. Yu et al. (DAPO, 2025) show asymmetric clip eliminates entropy collapse on reasoning RL. IsaacLab's standard configuration (adaptive KL, target 0.008–0.015) is the industry default for avoiding distribution-shift collapse at curriculum boundaries.

**Implementation — network changes.**

1. Add `LayerNorm` after each linear layer in both encoder and LSTM input projection:
   ```python
   self.encoder = nn.Sequential(
       nn.Linear(obs_dim, 128),
       nn.LayerNorm(128),
       nn.Tanh(),
       nn.Linear(128, 128),
       nn.LayerNorm(128),
       nn.Tanh(),
   )
   ```
2. Add L2-init regularizer (regenerative regularization toward init weights):
   ```python
   # After network construction, store init weights
   self.init_params = {k: v.detach().clone() for k, v in self.named_parameters()}
   # During loss:
   l2_init_loss = sum(((p - self.init_params[k])**2).sum()
                      for k, p in self.named_parameters())
   total_loss = ppo_loss + lambda_l2 * l2_init_loss    # λ = 1e-5
   ```
3. If using LSTMWrapper, evaluate removing it first (Federici/Zavoli evidence: MLP suffices when state is Markov, which §3.1 aims to achieve). If kept, add LayerNorm on LSTM inputs and outputs.

**Implementation — PPO hyperparameter changes.**

| Hyperparam | Current | New | Why |
|---|---|---|---|
| `clip_range` (ε_low, ε_high) | 0.2, 0.2 | 0.2, 0.28 | DAPO: asymmetric clip prevents entropy collapse. |
| `n_epochs` | 10 | 3 | Klein et al.: more epochs accelerates rank collapse; Bharthulwar et al.: co-scale down with high env count. |
| `learning_rate` | fixed 3e-4 | adaptive KL | Target KL = 0.012, factor 2, min 1e-6, max 1e-2. Standard IsaacLab config. |
| `ent_coef` | 0.005 | floor 0.01, bump to 0.03 | Target-entropy controller, floor at 0.01, bump to 0.03 when `H(π) < 1.1 nats` (50% of max for Discrete(9)). |
| `vf_coef` | 0.5 | 0.5 | Unchanged. |
| `gamma` | 0.99 | 0.99 | Unchanged, but now `γ^τ` in shaping (§3.2). |
| `gae_lambda` | 0.95 | 0.95 | Unchanged; see below for variable-dt handling. |

**Adaptive-KL implementation pseudocode:**

```python
# After each minibatch update
with torch.no_grad():
    new_log_probs = policy.log_prob(batch.actions, batch.obs)
    kl = (batch.old_log_probs - new_log_probs).mean()

# After full epoch
if kl > 2.0 * kl_target:
    lr /= 2.0
elif kl < 0.5 * kl_target:
    lr *= 2.0
lr = clip(lr, lr_min, lr_max)
```

**Target-entropy controller pseudocode:**

```python
# Track rolling mean entropy (last 10 updates)
H_mean = running_mean(policy_entropy)
if H_mean < 1.1:  # below 50% of log(9) = 2.197
    ent_coef_effective = 0.03
elif H_mean < 1.76:
    ent_coef_effective = 0.015
else:
    ent_coef_effective = 0.01   # floor
```

**GAE under variable dt.** When §3.4 ships, GAE needs `γ^τ_t`:

```
δ_t = r_t + γ^{τ_t} V(s_{t+τ_t}) − V(s_t)
A_t = δ_t + γ^{τ_t} λ A_{t+1}
```

This is a few-line change in the GAE function but easy to get wrong. Unit-test with τ=1 throughout — should exactly match vanilla GAE.

**Success criterion for this change alone (5M step ablation, no §3.4 time-warp).**
- Entropy at epoch 40 ≥ 1.1 nats (current: <0.5).
- Median per-update KL ≤ 0.02.
- Mid-episode abort rate on failures ≤ 40% (current: 76%).
- Greedy eval at 90° phase gap regresses by no more than 5pp vs current 74%.

**Risk register.**

| Risk | Likelihood | Mitigation |
|---|---|---|
| L2-init pulls weights back toward init too hard, preventing learning. | Medium | Start at λ=1e-5. If training stalls, reduce to 1e-6. If unstable, increase to 1e-4. |
| LayerNorm changes gradient magnitudes; existing LR no longer optimal. | Medium | Adaptive KL handles this by construction — LR self-adjusts. |
| Dropping to 3 PPO epochs reduces sample efficiency. | Medium | Expected and accepted. If variance is too high, increase minibatch size rather than epochs. |
| Asymmetric clip produces instabilities. | Low | DAPO reports it's strictly better than symmetric in their regime; unlikely to break us. |
| Adaptive-KL bug wedges LR at min or max. | Medium | Assert `lr_min < lr < lr_max` after each update; alert on clipping. |
| Entropy controller oscillates between bumped and unbumped. | Medium | Use running mean over 10 updates, not per-update — hysteresis by construction. |
| Removing LSTM breaks Markov assumption for partial-observability we didn't realize we had. | Medium | Ablate: train with and without LSTM under §3.1 relative state. If MLP matches or beats LSTM, remove it. |

### 3.4 Intervention D — Time-warp actions (SMDP-correct)

**Hypothesis.** Long-horizon phasing maneuvers require ~500 decisions of which ~450 are coast. Giving the agent temporal abstraction collapses the effective horizon, which improves credit assignment and reduces the number of updates over which entropy can collapse.

**Rationale from research.** TempoRL (Biedenkapp et al. ICML 2021), FiGAR (Lakshminarayanan et al. 2017), and the general frame-skip literature show action-repeat / skip actions reliably improve long-horizon sparse-reward tasks. For Keplerian dynamics the analytic propagator makes warp exact, not approximate, so there's no fidelity-accuracy tradeoff.

**Reframe from Phase 4 plan.** The original plan proposed warp-5min and warp-30min. Two changes:
1. **Drop warp-30min as a first-pass action.** It risks skipping over terminal conjunctions and creates larger advantage-estimator variance. Start with warp-5min only. Add warp-30min later only if diagnostics show the agent is using warp-5min constantly.
2. **Make SMDP correctness non-negotiable.** γ^τ in shaping, γ^τ in TD target, γ^τ in GAE. Sub-step collision checking.

**Action space change.** Discrete(9) → Discrete(10), adding:
- `action 9`: warp-5min (dt = 300s, no burn, `τ = 5`)

The original plan's warp-30min is deferred. If the agent wants longer warps, we'll see it pinning warp-5min and we'll add warp-30min then.

**Implementation (`orbital.h`).**

```c
void c_step(OrbitalEnv *env, int action) {
    double step_dt;
    int tau;
    if (action == 9) {
        step_dt = 300.0;
        tau = 5;
        // No impulse applied.
    } else {
        step_dt = DT;  // 60.0
        tau = 1;
        apply_impulse(env, action);
    }

    // Sub-step propagation + collision check
    int n_sub = tau;
    double sub_dt = step_dt / n_sub;
    for (int k = 0; k < n_sub; k++) {
        propagate_orbit(&env->sat.orbit, sub_dt);
        propagate_orbit(&env->target_body.orbit, sub_dt);
        for (int i = 0; i < env->num_bodies; i++) {
            propagate_orbit(&env->bodies[i].orbit, sub_dt);
        }
        // Collision check at every sub-step (60s resolution preserved)
        if (collision_check(env)) {
            terminate_collision(env);
            return;
        }
    }

    env->last_tau = tau;  // exposed to Python for GAE γ^τ handling
    // ... rest of step function, with Φ_curr - γ^τ * Φ_prev shaping
}
```

**SMDP handling in Python (`orbital.py` + PPO loop).**

1. Env exposes `info["tau"]` every step.
2. GAE computation uses `γ^tau` as per §3.3.
3. Shaping computation uses `γ^tau` as per §3.2.
4. Episode statistics account for real-time: "mean episode length" should report both in decision-steps and in sim-time seconds.

**Success criterion for this change alone (5M step ablation, with §3.1–§3.3 already in place).**
- Mean decision-steps per successful episode drops from ~800 to <300.
- Mean sim-time per successful episode within 2x of baseline (i.e., we're not wasting time; warps replace coasts).
- Greedy eval at 180° phase gap ≥ 60%.
- Trajectory plots show visible phasing structure with warps concentrated during drift phases.

**Risk register.**

| Risk | Likelihood | Mitigation |
|---|---|---|
| SMDP discount bug silently produces biased advantages. | High | Unit test: τ=1 everywhere must exactly match vanilla GAE. Numerical test: hand-compute expected `r_step` for a fixed (action=9, Φ_prev, Φ_next) triple. |
| Warp skips intermediate conjunction. | Low | Sub-step collision check at 60s resolution preserves the guarantee. |
| Agent spams warp without learning phasing. | Medium | Monitor warp-action usage rate. If >50% of actions are warp, either the shaping is wrong or warp-5 is too coarse — diagnose before adjusting. |
| SPS drops too much under sub-stepping. | Medium | Benchmark first. Current SPS ≈ 550k. Target post-warp: ≥ 100k. If below, reduce sub-step resolution (e.g., check collisions every 120s instead of 60s inside warp). |
| Entropy-bonus interactions: warp action shares entropy with burn actions, so agent can satisfy entropy floor by randomizing among warps and never burning. | Medium | Separate-head entropy tracking: if ever warp-head entropy is high but burn-head entropy is low, the controller hides the problem. Track per-action-class entropy explicitly. |

### 3.5 What we are *not* doing (and why)

**Continuous actions (Box(2) or scaled variants).** Killed per §2.4 and Principle C. If fine Δv control is needed at terminal handover, use Discrete(13) with rungs `{±2.5, ±5, ±10, ±25}` m/s, trained from scratch. Do not revisit Box in Phase 4.

**Scale-up (LSTM 128→256, 50M steps).** Deferred to Phase 5 per Principle A (diagnose before intervening). If §3.1–§3.4 show clear improvement but plateau below 70%, then scaling is the next lever. Scaling before diagnosis just trains the pathology harder.

**Mid-curriculum weight surgery (D7→D9 style).** Banned per Principle C. If we need a new action space, we train from scratch at that action space.

**Warm-starting across §3.1 representation change.** Banned per Principle D. §3.1 changes observation dimension; warm-starting weights into a larger-input network via surgery is exactly the bug that caused Stage 3g regression. New representation → new training run.

**Warm-starting within this Phase 4 without BC-kickstart.** If we need to chain curriculum stages within Phase 4, use BC-kickstart + value pretraining. Vanilla fine-tune is banned.

**Debris.** User hold in effect. Debris re-enters the picture only after §3.1–§3.4 produce a clean baseline without debris.

---

## 4. Attribution protocol

Per Principle E, every intervention gets an ablation. The minimum set before declaring Phase 4 complete:

**Ablation runs (each 5M steps, 180° phase gap, from scratch where noted):**

| Run | §3.1 Relative state | §3.2 Gated shaping | §3.3 Plasticity | §3.4 Time-warp | Expected | Purpose |
|---|---|---|---|---|---|---|
| R0 — baseline replication | no | old distance | no | no | 50% ±5pp | Verify Stage 3f reproduces on fresh seed |
| R1 — relative state only | yes | old distance | no | no | 55–60% | §3.1 attribution |
| R2 — shaping only | no | yes | no | no | 55–60% | §3.2 attribution |
| R3 — plasticity only | no | old distance | yes | no | 50–55% + better entropy | §3.3 attribution |
| R4 — warp only | no | old distance | no | yes | 50–55% | §3.4 attribution (likely weakest alone) |
| R5 — R1 + R2 | yes | yes | no | no | 65%+ | Primary combined |
| R6 — R1 + R2 + R3 | yes | yes | yes | no | 70%+ | Plasticity marginal contribution |
| R7 — all four | yes | yes | yes | yes | 70–75% | Full stack |

**Gate before running R7 (full stack):** R6 must beat R5 by ≥3pp. If not, §3.3 isn't helping and we should diagnose why before adding §3.4 on top.

**Gate before declaring Phase 4 done:** R0 must reproduce 50% ±5pp on a fresh seed. If R0 produces 30% or 65%, the Stage 3f baseline is noise and we need to establish a better baseline before any attribution is meaningful.

**Compute budget.** 8 runs × 5M steps ≈ 8 × 20 min on M3 Max ≈ 3 hours total. Trivial compared to the multi-day full-stack runs. Not optional.

**Diagnostic logging required on every run:**
- Entropy per update (running mean).
- Per-update KL (median + max).
- Value loss + explained variance.
- True success rate (not shaped return) per checkpoint.
- Mid-episode abort rate per checkpoint.
- Dormant-neuron fraction per checkpoint (for §3.3 attribution).
- For §3.4 runs: per-action-class entropy, warp-usage rate.
- For §3.2 runs: per-episode accumulated |G_shape|, gate activation rates.

---

## 5. Reproducibility and anti-fragility

The Phase 3 experience revealed that our best checkpoint (Stage 3f @ 50%) is a pointy maximum, not a stable basin. Phase 4 should not reproduce that pathology.

**Reproducibility checks.**
- Every reported success rate should be an average over 3 seeds, not a single lucky run.
- Best-of-checkpoint eval (current practice) is allowed only during development. Final reported numbers must be end-of-training eval, averaged across seeds.
- Checkpoint every 10 epochs and keep all. Peak-checkpoint vs final-checkpoint divergence >10pp is a warning sign that the run is on a pointy max.

**Seed protocol.**
- Three seeds per attribution run: 42, 1337, 20260423.
- Report mean and min across seeds. If min is below target, the result is not robust.

**Version pinning.**
- Python, PyTorch, PufferLib versions frozen at the start of Phase 4.
- Hyperparameters committed to `orbital.ini` before each run.
- Random seeds logged.
- Environment version (observation dim, action dim, reward formulation) tagged in wandb run name.

**Checkpoint hygiene.**
- Never fine-tune across environment changes (obs dim, action dim, reward formulation). Always train from scratch after structural env changes.
- If fine-tuning within-environment (stage to stage in a curriculum), use BC-kickstart + value pretraining per Wołczyk et al.

---

## 6. Sequencing and milestones

Given attribution requirements and the research-report-informed prioritization, here's the actual order of operations.

### Week 1 — Foundation and diagnosis
- **Day 1.** Verify episode-length sufficiency: `max_episode_steps × dt ≥ 1.5 × T_target_period`. This is cheap and must be done before anything else. If violated, fix immediately — §3.4 partially addresses this, but even the baseline should allow physical phasing-orbit completion.
- **Day 1.** Set up attribution framework: wandb project, seed protocol, diagnostic logging hooks.
- **Day 2.** Run R0 (baseline reproduction). Three seeds. If not 50% ± 5pp, stop and investigate.
- **Days 3–4.** Implement §3.1 (CW/Hill frame). Unit-test LVLH transformation. Run R1. Evaluate.
- **Days 5–6.** Implement §3.2 (gated shaping) with calibration protocol. Run R2. Evaluate.
- **Day 7.** Run R5 (R1 + R2). Evaluate. If R5 ≥ 65%, we have the primary result.

### Week 2 — Plasticity and time-warp
- **Days 8–9.** Implement §3.3 (plasticity). Run R3 standalone. Evaluate.
- **Day 10.** Run R6 (R1 + R2 + R3). Evaluate. Compare to R5; if R6 − R5 ≥ 3pp, plasticity is marginally useful.
- **Days 11–12.** Implement §3.4 (time-warp with SMDP correctness). Unit-test γ^τ in GAE and shaping. Run R4 standalone. Evaluate.
- **Day 13.** Run R7 (full stack). Evaluate.
- **Day 14.** Write up results. Trajectory plots for each successful run. Pick best checkpoint across seeds.

### Milestone gates

| Gate | Check | If failed |
|---|---|---|
| End Week 1 | R0 reproduces 50% baseline, R5 ≥ 60% | Stop; diagnose why §3.1 + §3.2 didn't move the needle |
| Mid Week 2 | R6 − R5 ≥ 3pp | §3.3 isn't helping; skip R7 or diagnose further |
| End Week 2 | R7 ≥ 65% @ 180°, with seed variance ≤ 10pp | Phase 4 complete; move to Phase 5 |

### Portfolio artifacts (for APL/Lincoln Lab/Blue Origin audience)

- Clean diagnosis writeup (this document + research report + trajectory plots).
- Side-by-side: distance-shaped agent failing to phase, vs. gated-shaped agent executing a visible phasing orbit.
- Training curves showing entropy collapse under baseline, entropy preserved under §3.3.
- Attribution table (R0–R7) with citation chain back to published papers.

The narrative: "I hit a 50% plateau, diagnosed it as a combination of three published failure modes, and broke through with cited fixes." That's a stronger story than "I got 75% with a large combined change."

---

## 7. Appendix: research findings summary

Condensed from the 2026-04-23 research report. Full citations in the report itself.

**Orbital RL invariants (Federici, Zavoli, Furfaro, Chen, Copp, Li 2022–2026).**
- Rotating frame (CW/Hill/LVLH/MEE) is the standard, not optional.
- ε-constraint curriculum on terminal tolerances is the most-cited plateau-breaker.
- Action parameterizations that cannot violate constraints (discrete, or bounded-by-construction) are preferred over penalty-enforced continuous.
- Episode length must physically accommodate the maneuver class.

**PPO failure modes (Moalla et al. NeurIPS 2024, Klein et al. NeurIPS 2024, Yu et al. DAPO 2025).**
- Rank collapse → trust-region failure → entropy collapse is a known cascade.
- LayerNorm + L2-init works; plasticity injection and neuron resets don't (for PPO specifically).
- Asymmetric clip (`ε_high` > `ε_low`) eliminates entropy collapse on sparse reward.
- More PPO epochs with many parallel envs accelerates the cascade, not mitigates it.

**Shaping (Chen Copp 2023, Li et al. 2024, Ng-Harada-Russell 1999, and followups).**
- Gated multi-stage decomposition outperforms single-potential shaping.
- `1 − cos(θ)` outperforms `|θ|/π` for angular errors.
- Exponential terminal bonus outperforms linear for final-approach convergence.
- Terminal Φ=0 clamp is load-bearing; omitting it causes abort-early pathologies.
- SMDP-correct γ^τ is required under variable dt.

**Warm-start (Wołczyk et al. ICML 2024, VC-PPO arXiv 2025).**
- BC-kickstart (KL penalty against frozen pre-trained policy, anneal 0.5→0.05 over 20–30% of fine-tune) is the standard fix.
- Value-function pretraining before policy updates prevents first-epoch collapse.
- LR warmup from 1/10 of from-scratch LR prevents distribution-shift blowup.

**Action-space expansion (Farquhar et al. ICML 2020 and negative results).**
- No published PPO technique reliably expands action space mid-curriculum.
- If forced, use negative logit init on new actions, freeze trunk initially, bump entropy, BC-kickstart against pre-expansion policy. Even then, works unreliably.

---

## 8. Appendix: glossary

- **CW / Hill / LVLH.** Clohessy-Wiltshire frame, a rotating reference frame centered on the target with axes aligned radial/along-track/cross-track. Equivalent names for the same thing in 2D coplanar settings.
- **SMDP.** Semi-Markov Decision Process. An MDP where actions take variable time; requires γ^τ discounting to preserve Bellman equation properties.
- **Ng-Harada-Russell (NHR) shaping.** Potential-based shaping `F = γΦ(s') − Φ(s)` that preserves optimal policy. Requires Φ(terminal) = 0.
- **DAPO.** Asymmetric PPO clip variant (Yu et al. 2025): `ε_low = 0.2`, `ε_high = 0.26–0.28`.
- **L2-init.** Regenerative regularization toward initial weights: `L_reg = λ · ||θ − θ_init||²`. Preserves plasticity.
- **Rank collapse.** The effective rank of a neural network's feature representation drops during training, degrading downstream predictions. Precedes and predicts performance collapse in PPO.
- **BC-kickstart.** Behavioral-cloning-based fine-tuning: add `KL(π_new || π_0)` penalty against frozen pre-trained policy, anneal over early fine-tune.

---

*Last updated: 2026-04-23. Supersedes orbital-rl-phase4-plan.md where conflicts exist.*
