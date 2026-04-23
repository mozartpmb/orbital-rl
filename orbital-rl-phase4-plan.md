# Orbital RL Phase 4 — Planning Document

> Status: 2026-04-21. Phase 3 concluded at **50% greedy eval @ 180° phase gap** (target 60%). This doc plans the next four interventions the user approved: **(1) time-warp actions**, **(2) physics-based reward**, **(3) continuous actions revisit**, **(5) scale-up**. Item 4 (demonstration bootstrap) is deferred.

---

## 1. Executive Summary

**Where we are.** The rendezvous agent solves up to ~90° phase-gap robustly (74% @ 90°) and delivers a respectable 50% success at the full 180° phase gap with Discrete(9) actions and distance-based shaping. We missed the 60% plan target by 10 points after exhausting the "easy" knobs: curriculum, shaping reweighting, action granularity, LR floor. wandb shows the remaining failure mode is *not* fuel exhaustion — it is mid-episode entropy collapse into suboptimal phasing strategies.

**Best checkpoint:** `experiments/puffer_orbital_3o45m2t5/model_puffer_orbital_000070.pt` — 50% @ 180° greedy, Discrete(9).

**Why we stopped tuning.** Three separate shaping formulations (distance-only, phase-only, balanced) all converged to 37–50%. The ceiling is structural, not a hyperparameter. Two root causes: (a) 180° rendezvous requires a *phasing orbit* — temporarily moving *away* from the target's current location — which any monotonic distance shaping actively punishes; and (b) the 60-second timestep forces the agent to take thousands of near-identical coast decisions per episode, which dilutes credit assignment and lets entropy collapse around bad strategies.

**Plan ahead.** Ship (1) and (2) together: time-warp gives the agent temporal abstraction, physics-based reward gives it a gradient that respects orbital mechanics instead of punishing phasing. Then revisit (3) continuous actions now that we have a trained narrow-discrete policy to warm-start from, and (5) scale up once the env delivers signal.

---

## 2. Current Code State (post-Phase 3)

### 2.1 Environment summary

| Aspect | Value |
|---|---|
| Observation space | `Box(float32, 33)` |
| Action space | `Discrete(9)` |
| Timestep | 60 s sim time |
| Success tolerance | `RENDEZVOUS_RADIUS=30 km`, `REL_VEL_TOL=50 m/s` |
| Shaping | distance-only potential form, `c_dist=0.05`, `c_phase=0.0` |
| Debris | disabled (`num_debris_min=num_debris_max=0`) |
| Target eccentricity | `e_max_target=0.0` (circular) |

### 2.2 Key source files

| Path | Role | Notable recent edits |
|---|---|---|
| `pufferlib/pufferlib/ocean/orbital/orbital.h` | C env: structs, physics, obs/reward/termination | Discrete(9), fine ±5 m/s burns, `dist_prev`/`dphase_prev` cache fields, distance-only shaping |
| `pufferlib/pufferlib/ocean/orbital/orbital.py` | PufferEnv wrapper | `single_action_space = Discrete(9)` |
| `pufferlib/pufferlib/ocean/orbital/binding.c` | C↔Python glue, trajectory export | `init_phase_gap_max` kwarg, `vec_get_trajectory` |
| `scripts/orbital/expand_ckpt_actions_7_to_9.py` | Weight surgery D7→D9 | `old_to_new = [0, 2, 3, 5, 6, 7, 8]`; rows 1,4 zero-init |
| `scripts/orbital/eval_all_ckpts.py` | Greedy-eval every ckpt in run dir | Takes `--e-max-target`, `--init-phase-gap-max` |

### 2.3 Action table (current, Discrete(9))

```
0: coast
1: prograde   5 m/s   (fine — new in Phase 3f)
2: prograde  10 m/s
3: prograde  25 m/s
4: retrograde 5 m/s   (fine — new in Phase 3f)
5: retrograde 10 m/s
6: retrograde 25 m/s
7: radial out 10 m/s
8: radial in  10 m/s
```

### 2.4 Shaping (current)

```c
const double c_phase = 0.0;
const double c_dist  = 0.05;
env->rewards[0] += (float)(
    c_phase * (fabs(env->dphase_prev) - fabs(dp)) / M_PI +
    c_dist  * (env->dist_prev - dist_curr)        / RENDEZVOUS_RADIUS
);
```

Potential-based (Φ_curr − Φ_prev) so stuck = 0 reward.

---

## 3. Phase 3 Experimental Results

| Stage | Phase gap | Start from | Run ID | Best ckpt | Greedy eval | Notes |
|---|---|---|---|---|---|---|
| 1 | 30° | **fresh** | `q0jsaz88` | 115 | **99%** | Strong shaping + loose tolerance broke the bootstrap |
| 2 | 90° | Stage 1 ckpt 115 | `r6g60p2y` | 077 | 67% | 10M steps |
| 2-ext | 90° | Stage 2 ckpt 077 | `ehgfxjyb` | 039 | **74%** | +5M extension, mean_len 799→551 |
| 2b | 135° | Stage 2-ext ckpt 039 | `cumxvwd8` | 077 | 53% | Intermediate step before full 180° |
| 3 | 180° | Stage 2-ext ckpt 039 | `0ys77hbc` | 090 | 46% | Direct 90°→180° jump |
| 3c | 180° | Stage 2b ckpt 077 | — | — | ~46% | 2b→180° path no better than direct |
| 3d | 180° | Stage 3 ckpt 090 | `rt1t505c` | — | **0%** | Phase shaping (c_phase=2, c_dist=0.01) — collapse |
| 3e | 180° | Stage 3 ckpt 090 | `nnnyi1g2` | — | 37% | Balanced (c_phase=1, c_dist=0.02) |
| 3f | 180° | surgically D9'd 3 ckpt 090 | `3o45m2t5` | **070** | **50%** | +Discrete(9), reverted shaping |
| 3g | 180° | Stage 3f ckpt 070 | `73evcbf6` | — | 24–37% | Lock-in retrain (ent_coef=0.002) regressed |

**Plateau is real.** Four different approaches (longer training, curriculum branch, reshaping, action granularity) all landed within 37–53% @ 180°.

### 3.1 Diagnostic findings (from `/tmp/diag_180.py`)

- 0/100 episodes ended stranded (fuel exhausted)
- 76% of failures were mid-episode — agent gave up before hitting tolerance
- Mean end_fuel ≈ mean start_fuel → agent coasts most of the episode
- Entropy collapses to < 0.5 nats by ~40 epochs → policy locks into a deterministic branch
- Value-loss stable and low → critic is not confused, policy is just timid

This matches a textbook "local optimum + entropy collapse" pattern rather than a fuel/precision issue.

---

## 4. Why We Hit a Ceiling — Research & Analysis

### 4.1 Orbital mechanics vs. our shaping

For a 180° rendezvous, the textbook fuel-optimal plan is a **phasing orbit**: lower (or raise) the semi-major axis by ~Δa to change orbital period by ΔT, let the phase drift close over N orbits, then re-circularize. This is the *only* efficient way to close 180° with our Δv budget.

During the phasing orbit, satellite–target distance *grows* before it shrinks. Our distance-based potential shaping assigns **negative reward** to this correct first half of the maneuver. The agent is learning to avoid the one trajectory that actually works, and is instead searching for impossibly direct intercepts.

### 4.2 Timestep / temporal abstraction

A 92-minute LEO orbit at dt=60s is **92 decision steps per revolution**. A 5-orbit phasing maneuver is ~460 steps of mostly-coast. PPO's credit assignment across hundreds of identical coast decisions is weak — advantages are tiny, gradients are noisy, and entropy collapses long before the terminal reward propagates.

### 4.3 Reward gradient mismatch

Distance `||r_sat − r_target||` is **not a Lyapunov function** on Kepler orbits. Two satellites on adjacent orbits can be arbitrarily far apart right now yet converge a quarter-orbit later; two co-located satellites with different velocities are about to diverge. Pythagorean distance punishes correct maneuvers and rewards coincidences. The *right* potential is something like phase-match quality on the target orbit, scaled by how far from the target orbit the satellite currently sits.

### 4.4 Action / tolerance interaction (partially solved)

Discrete(9) added ±5 m/s fine burns, which lifted us from 46% to 50%. But the fundamental issue at 180° isn't precision on the final approach — it's strategy selection during the first 300 steps. More granularity here has diminishing returns.

---

## 5. Item 1 — Time-Warp Actions

### 5.1 Motivation

A 180° phasing maneuver takes ~500 steps of which ~450 are trivial Kepler coast. If we give the agent actions that advance the sim by larger dt without burning, (a) the effective horizon collapses, (b) credit assignment on the post-warp state becomes direct, (c) entropy is spent on strategic decisions instead of thousands of redundant coasts. The Kepler propagator is analytic — warping 10 minutes is as accurate as stepping 10 × 60s.

### 5.2 Proposed action space (Discrete 11)

```
 0: coast  (dt = 60s, unchanged)
 1: pro  5 m/s      (dt = 60s)
 2: pro 10 m/s      (dt = 60s)
 3: pro 25 m/s      (dt = 60s)
 4: retro 5 m/s
 5: retro 10 m/s
 6: retro 25 m/s
 7: radial +10 m/s
 8: radial -10 m/s
 9: warp 5 min      (dt = 300s, no burn)
10: warp 30 min     (dt = 1800s, no burn)
```

Rationale for just two warp steps: 5-min covers fine-grained phase closure; 30-min covers a full "wait a quarter-orbit" decision. A third 2-hour warp is tempting but risks under-sampling debris/near-target conjunctions (and we can add it later if the agent asks for it by pinning warp-30 often).

### 5.3 Implementation (`orbital.h`)

In the action switch inside `c_step`:

```c
double step_dt = DT;  // default 60s
if (action == 9)  step_dt = 300.0;
if (action == 10) step_dt = 1800.0;

// ACTION_DV[action] is {0,0,0} for warps — no impulse applied.
apply_impulse(env, action);            // no-op for warp actions
propagate_orbit(&env->sat.orbit, step_dt);
propagate_orbit(&env->target_body.orbit, step_dt);
for (i…) propagate_orbit(&env->bodies[i].orbit, step_dt);
```

### 5.4 Collision safety during warps

Warps skip intermediate positions. For v1 we substep the collision check:

```c
if (step_dt > DT) {
    int n_sub = (int)(step_dt / DT);
    double sub_dt = step_dt / n_sub;
    for (int k = 0; k < n_sub; k++) {
        propagate_orbit(&env->sat.orbit, sub_dt);
        propagate_orbit(&env->target_body.orbit, sub_dt);
        for (i…) propagate_orbit(&env->bodies[i].orbit, sub_dt);
        if (collision_check()) { terminate(-10); return; }
    }
} else {
    propagate_orbit(…, step_dt);
}
```

This gives exact Kepler positions at every 60s boundary inside the warp, at the cost of warping being ~5× (warp-5) or ~30× (warp-30) slower than a normal step. Net training throughput still ~2× better than no-warp because the agent takes far fewer decisions per episode.

### 5.5 Reward accounting

No per-step reward penalty exists today, so warps don't need time-scaling. If we add a small coast penalty later (see §6.3), warp-30 should accumulate `30 * per_step_penalty` so the agent can't dodge it.

### 5.6 Observation design

No new obs dims. The agent sees the post-warp state; orbital elements are continuous through Kepler so discontinuity is only in Cartesian.

### 5.7 Training plan

1. Warm-start from Stage 3f ckpt `3o45m2t5/model_puffer_orbital_000070.pt` using weight surgery D9→D11 (extend decoder rows 9, 10 zero-init).
2. `init_phase_gap_max = π` (180°), no debris, `e_max_target=0.0`.
3. 15M steps, `ent_coef=0.01` (up from 0.005 — encourage reuse of new actions).
4. Eval every 10 ckpts.

### 5.8 Success criteria

- Warped agent reaches ≥ 60% @ 180° (plan target).
- Trajectory plots show the characteristic "drop to lower orbit → warp several times → raise" phasing pattern.
- Mean decision-steps per successful episode drops from ~800 → < 200.

### 5.9 Risks

- Agent spams warp-30 and skips over the terminal conjunction. *Mitigation: terminal check runs at each sub-step.*
- Warp breaks the existing policy's timing intuition. *Mitigation: warm-start + high entropy gives exploration of new actions without forgetting the old ones.*
- Numerical drift if a warp coincides with a close passage to Earth. *Mitigation: Kepler is analytic; drift is bounded; Earth escape detection still fires.*

---

## 6. Item 2 — Physics-Based Reward

### 6.1 Motivation

Replace the distance-based potential with something that **does not punish phasing orbits**. The shaping should reward: (a) matching the target *orbit* (a, e, ω), (b) closing the *phase* once on the right orbit, and (c) nulling *relative velocity* at handover. All three are physically meaningful and compose naturally into the terminal success condition.

### 6.2 Proposed potential function

Let the potential be a weighted sum of three terms:

```
Φ(s) = −w_orbit * d_orbit(s)
       − w_phase * d_phase(s) * gate_phase(s)
       − w_vel  * d_vel(s)  * gate_vel(s)
```

where

- `d_orbit = |Δa|/ALT_MAX + |Δē|/1.0`  — distance in orbital-element space (semi-major axis + eccentricity vector). Scale-independent, zero iff orbits match.
- `d_phase = |wrap(θ_sat − θ_target)| / π`  — phase mismatch, zero iff phased.
- `d_vel   = ||v_sat − v_target|| / REL_VEL_TOL`  — relative velocity, already in tolerance units.
- `gate_phase = sigmoid((ORBIT_MATCH_THRESHOLD − d_orbit) / 0.1)` — activates only when orbits already close.
- `gate_vel   = sigmoid((PHASE_MATCH_THRESHOLD − d_phase) / 0.2)` — activates only when phase already close.

Reward per step = `β · (Φ(s') − Φ(s))` (potential-based, Ng et al. 1999 form with γ=1 to avoid the stuck-gets-rewarded pathology). Weights initial guess: `w_orbit=1.0`, `w_phase=0.5`, `w_vel=0.3`, `β=0.1`.

### 6.3 Why this fixes phasing

During a phasing orbit the agent *intentionally* changes `a` to a phasing value, drifts for several orbits closing `d_phase`, then restores `a`. Under `d_orbit` alone, the excursion is a round-trip: shaping cost during the first impulse is cancelled by shaping gain from the second. During drift, `d_orbit` is constant so shaping is zero. Meanwhile `d_phase` closes monotonically — positive shaping accumulates exactly during the part of the trajectory our old reward punished. Velocity term only kicks in at the final approach, so it doesn't distort strategy selection.

### 6.4 Optional coast penalty (small)

Add `-0.0005` per step (so ~-1 over a 2000-step episode, small vs ±10 terminal). Discourages infinite waiting without reshaping the strategic decisions. Warp actions pay the scaled penalty.

### 6.5 Implementation (`orbital.h`)

Add to env struct:

```c
double sat_a_prev;          // already exists from Phase 2 shaping
double sat_ex_prev, sat_ey_prev;  // already exists
double sat_theta_prev;      // NEW: for phase delta tracking (use θ_sat − θ_target)
double rel_vel_prev;        // NEW: for velocity-null tracking
```

In `c_step`, after propagation, before `write_traj_record`:

```c
double d_orbit_curr = fabs(env->sat.orbit.a - env->target_body.orbit.a) / ALT_MAX
                    + sqrt((sat_ex_c - tgt_ex)^2 + (sat_ey_c - tgt_ey)^2);
double d_phase_curr = fabs(wrap_pi(env->sat.orbit.theta - env->target_body.orbit.theta)) / M_PI;
double d_vel_curr   = rel_vel / REL_VEL_TOL;

double gate_p = sigmoid((ORBIT_MATCH - d_orbit_curr) / 0.1);
double gate_v = sigmoid((PHASE_MATCH - d_phase_curr) / 0.2);

double phi_curr = -(W_ORBIT*d_orbit_curr
                  + W_PHASE*d_phase_curr*gate_p
                  + W_VEL  *d_vel_curr  *gate_v);

env->rewards[0] += (float)(BETA * (phi_curr - env->phi_prev));
env->phi_prev = phi_curr;
```

### 6.6 Training plan

1. Rebuild C, run `orbital_test` (physics unchanged, only reward).
2. Regression: eval Stage 3f ckpt with new reward active at e_max=0°. If it drops below 60%, the new potential is miscalibrated for easy cases — tune weights.
3. Warm-start Stage 3f ckpt → 180°, new reward, 15M steps.
4. Expect `episode_return` to climb more steadily than in Stage 3 (smoother signal).

### 6.7 Interaction with time-warp (§5)

These two items compose. Ship them together in one training run (D11 + new reward). Reward-accounting for warps: scale coast-penalty term by `dt/60`; shaping difference Φ(s')−Φ(s) is already dt-agnostic.

### 6.8 Success criteria

- ≥ 60% @ 180° (plan target).
- `d_phase` trace on a successful episode shows monotone closing during drift phase.

### 6.9 Risks

- Gates are poorly calibrated (fire too early or never). *Mitigation: log Φ components in trajectory file, tune offline.*
- Weight balance creates a new local optimum. *Mitigation: keep Stage 3f ckpt as a fallback baseline; run regression eval first.*

---

## 7. Item 3 — Continuous Actions (Revisit)

### 7.1 Prior finding

Phase 2 trial: Box(2) continuous actions trained **from scratch** collapsed to 24% greedy / 33% stochastic @ e_max=0.10 after 15M steps, entropy flatlined. Root cause identified in memory: **broader action space dilutes the random-exploration productive rate**. In Discrete(7), ~6/7 random actions are "productive" (non-coast). In Box(2) with Gaussian init, the vast majority of samples land in low-magnitude directions that barely move the orbit. PPO can't bootstrap.

### 7.2 The fix: warm-start, don't train from scratch

We now have a trained Discrete(9) policy (Stage 3f ckpt). Convert it to a continuous policy via weight surgery:

- **Decoder**: replace `Linear(128, 9)` + softmax with `Linear(128, 2 × 2)` + Gaussian (mean, log_std for prograde-Δv and radial-Δv).
- **Initialization**: compute the expected value of each action under the trained discrete policy at common rollout states, fit the Gaussian head to match (1-layer least squares). Optionally warm-start just the mean head; initialize log_std to -2 (tight initial std) to preserve deterministic-ish policy while allowing gradient.

This is the same pattern that worked for Discrete(7)→Discrete(9): preserve the policy's current behavior, then let PPO relax the action space on-policy.

### 7.3 Expected benefits

- Finer Δv control → smaller fuel overshoots, tighter terminal approach.
- Continuous action gradient — the policy gradient directly moves Δv; under Discrete(9) the agent must choose among fixed rungs.
- Smoother burn schedules → plots should look less "pulse-train" and more "textbook impulse."

### 7.4 Training plan

1. Build surgery script `scripts/orbital/discrete9_to_box2.py`.
2. Keep Discrete(9) in the env; add a `CONTINUOUS=1` env flag that switches action decoding to reading two floats (prograde and radial Δv in m/s, clipped to [−25, 25]).
3. 15M steps warm-start, `ent_coef=0.01`, `init_phase_gap_max=π`.

### 7.5 Sequencing

**Do this AFTER items (1) + (2) land.** Continuous actions on top of an agent that already handles 180° via phasing + physics reward gives us a much cleaner signal about what continuous *adds*, versus stacking three experimental changes at once.

### 7.6 Success criteria

- ≥ 65% @ 180° (improvement over Discrete(9) + warp + new reward).
- Mean total Δv drops by ≥ 10% compared to Discrete(9) baseline at same success rate.

### 7.7 Risks

- Warm-start from Discrete distribution produces degenerate Gaussian (mean on top of a mode, tiny std). *Mitigation: initialize log_std=-2 (std≈0.14 normalized), not -5.*
- Continuous still collapses despite warm-start. *Mitigation: fallback to Discrete(11) with very fine rungs (±2.5, ±5, ±10, ±25) — captures 80% of the benefit without the distributional change.*

---

## 8. Item 5 — Scale-Up

### 8.1 What to scale

The policy we've been training is pufferlib's default: `Default` (MLP encoder, Linear decoder) + `LSTMWrapper(hidden_size=128)`. For the complexity of 180° rendezvous with temporal abstraction, 128 hidden is small.

### 8.2 Proposed scale axes

| Axis | From | To | Cost |
|---|---|---|---|
| LSTM hidden size | 128 | 256 | ~2× compute, linear on wall-clock |
| Encoder hidden | 128 | 256 | minor |
| Total training steps per stage | 10–15M | 30–50M | ~3× wall-clock |
| num_envs | default (~1024) | 4096 | better gradient variance |

Keep the architecture family the same (pufferlib.Default + LSTMWrapper); just larger.

### 8.3 When to scale

**After items (1), (2), (3) are evaluated.** Scale-up should confirm improvements from architecture rather than simultaneously develop them. If the D11+warp+new-reward run plateaus again, scaling is the right next lever; if it already hits 60%+, scale gives a cheap push to 70%+.

### 8.4 Compute cost

Current training: ~20 min for 15M steps on Apple Silicon. Scaling to 256 LSTM + 50M steps is ~2 hours — still tractable locally. If we want to iterate fast, also try CPU→MPS (user has M3 Max) for a throughput bump.

### 8.5 Success criteria

- ≥ 70% @ 180° (stretch target above the original 60% plan).
- Regression-safe: scale-up doesn't break earlier stages (90°, 30°).

### 8.6 Risks

- Larger LSTM with same data → overfits to curriculum artifacts. *Mitigation: drop LR min_ratio from 0.2 to 0.1, extend training, monitor eval/train gap.*
- MPS inconsistency. *Mitigation: verify SPS and gradient numerics match CPU on a short run before committing.*

---

## 9. Implementation Ordering

Recommended sequence, accounting for what composes and what isolates a signal:

1. **Ship (1) + (2) together** — time-warp actions AND physics-based reward in a single training run. These are jointly necessary for the phasing strategy, and evaluating them separately splits a coupled gain across two under-signal runs. Target: 60% @ 180°.
2. **If (1+2) plateaus:** run (5) scale-up on the same code — LSTM 256, 30–50M steps. Target: 65–70%.
3. **Run (3) continuous actions last** — warm-start from the best checkpoint produced by (1+2) or (1+2+5). Target: +5–10% on fuel efficiency without losing success rate.
4. **Deferred:** (4) demonstration bootstrap from a Hohmann-solver policy. Only pursue if (1)+(2)+(3)+(5) still plateau below 60%.

Each step gates the next on the regression eval against the previous checkpoint.

---

## 10. Evaluation Protocol (applies to every stage)

1. `./orbital_test` passes — physics is untouched unless we explicitly change it.
2. `python3 orbital.py` SPS ≥ 300k (≥ 150k with warps due to sub-stepping).
3. **Regression eval** at e_max=0, 90° phase gap on Stage 2-ext ckpt with the new reward/action code — must stay ≥ 70%. Catches "new reward breaks the easy case."
4. Training: 15M steps (30–50M if scaled), `ent_coef=0.01`, `checkpoint-interval=10`, wandb group labeled by phase/item.
5. Greedy eval every 10 ckpts at target phase gap. Best-of-ckpt wins.
6. Trajectory plots on 5 successful + 5 failed episodes. Verify qualitative behavior matches mechanism (phasing orbit visible for 180°).
7. Save best ckpt to named artifact; memory-log the config.

---

## 11. Risks & Overarching Mitigations

| Risk | Mitigation |
|---|---|
| Shipping two changes at once masks which one helped | Trajectory plots + ablation: if 1+2 hits 60%, also run 1-only and 2-only as 5M-step debug runs to attribute credit. |
| Time-warp sub-stepping tanks SPS below training-viable | Benchmark first. If <100k SPS, reduce warp-30 to warp-15 or increase sub-step spacing to 120s. |
| Physics-reward gates create flat regions where Φ′=Φ | Add a small linear term `-0.01 * d_orbit` outside the gate so there's always *some* gradient even far from the goal. |
| Continuous-action surgery fails to reproduce discrete policy behavior | Quantitative check: roll 1000 states through both policies, compare action distribution in Δv-space; adjust if KL > 0.5. |
| Scale-up changes numerics and the new Phase 3 reward behaves differently | Always warm-start from a passing D9 ckpt; never retrain from scratch at new scale. |

---

## 12. Open Questions for the User

- **Compute cap.** Are we happy with local M3 training runs (~2 hrs for a full scaled stage), or do we want to move to a rented GPU before (5)?
- **Target metric.** Plan says 60% @ 180°; happy to re-pin to 70% if we commit to all four items, or keep 60% as pass-fail?
- **Debris.** Still on hold. If a scaled agent hits 70% clean, do we re-enable debris for the final evaluation pass, or hold that for Phase 5?

---

*Author: Claude, 2026-04-21. Superseded by actual runs — update this doc as stages complete.*
