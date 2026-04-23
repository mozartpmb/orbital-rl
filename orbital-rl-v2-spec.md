# Orbital RL — V2 Technical Spec
## Fuel Efficiency → Elliptical Orbits → Rendezvous

---

## Overview

Three sequential upgrades to the Orbital RL environment, each building on the last. The goal is to move from the current "spiral and arrive" agent toward one that performs near-optimal multi-impulse transfers to elliptical target orbits, and ultimately rendezvouses with a target object at a specific orbital position.

**Current baseline (v1):**
- 100% eval success (circular-to-circular with debris)
- 32% Hohmann fuel efficiency (2–3× more Δv than optimal)
- "Many small burns" spiral strategy
- No eccentricity control, no position matching

**Target (v2 complete):**
- Near-Hohmann fuel efficiency on circular transfers (~70%+)
- Reliable transfers to elliptical target orbits (e up to ~0.3)
- Rendezvous: match orbital position within tolerance
- Warm-start path: each phase feeds the next

---

## Phase 1: Fuel Efficiency Shaping

### Problem

The agent succeeds reliably but wastes fuel. It spirals with dozens of small prograde burns instead of learning the two-impulse Hohmann pattern. The current reward function gives a flat +10 for any success regardless of cost.

### Changes

#### 1A. Fuel-Proportional Success Reward

Replace the flat terminal reward:

```
Current:   +10.0 on success
Proposed:  +10.0 * (0.5 + 0.5 * fuel_remaining) on success
```

This maps to a reward range of **+5.0** (arrived on fumes) to **+10.0** (arrived with full tanks). Success is always positive, but efficient success is twice as rewarding. The 0.5 floor ensures that "arrive inefficiently" is still far better than any failure (-10), so the agent won't sacrifice reliability for efficiency.

**Implementation:** In `orbital.h`, in the terminal success block inside `step()`:

```c
// Current
rewards[0] = 10.0f;

// New
rewards[0] = 10.0f * (0.5f + 0.5f * env->sat.fuel);
```

One line change.

#### 1B. Trim Action Space to 7

Remove normal burn actions (5 and 6). In 2D coplanar orbits, normal burns have no useful orbital effect but cost fuel. The Stage 2 agent already learned to avoid them (0% usage), but removing them shrinks the policy's search space.

**New action table:**

| Action | Description | Δv (m/s) |
|--------|-------------|----------|
| 0 | Coast | 0 |
| 1 | Prograde small | +10 |
| 2 | Prograde large | +50 |
| 3 | Retrograde small | −10 |
| 4 | Retrograde large | −50 |
| 5 | Radial in | +10 |
| 6 | Radial out | −10 |

**Implementation:**
- `orbital.h`: Remove cases 5/6 from the action switch, renumber radial actions from 7/8 → 5/6, change `NUM_ACTIONS` from 9 → 7
- `orbital.py`: Update `Discrete(9)` → `Discrete(7)`
- `orbital.ini`: No change needed (PPO doesn't care about action count)

**Note:** This means we cannot warm-start from the v1 checkpoint (policy head shape changes). Train from scratch — this is fine since Phase 1 is also validating the new reward function.

#### 1C. Observation Improvement — Sinusoidal True Anomaly

Replace `obs[2] = θ/π` with `(sin(θ), cos(θ))`. This eliminates the discontinuity at θ = 2π → 0 and gives a clean [-1, 1] representation. Adds 1 observation dimension (24 → 25).

**Implementation:**
- `orbital.h` in the observation-writing block:
  ```c
  // Current
  obs[2] = (float)(theta / M_PI);

  // New
  obs[2] = (float)sin(theta);
  obs[3] = (float)cos(theta);
  // Shift all subsequent indices by +1
  ```
- `orbital.py`: Update `Box(low, high, shape=(25,))`, update index comments
- `binding.c`: Update `OBS_SIZE` if defined as a constant

#### 1D. (Optional) Dense Shaping — Δa Progress

If the sparse fuel-proportional reward alone doesn't push the agent toward two-impulse transfers after ~10M steps, add a small dense shaping reward:

```c
float da_prev = fabs(env->sat.orbit.a - env->target.a);  // before step
// ... step physics ...
float da_curr = fabs(env->sat.orbit.a - env->target.a);  // after step
float shaping = 0.01f * (da_prev - da_curr) / env->a_scale;
rewards[0] += shaping;
```

This gives a tiny positive reward for getting closer to the target semi-major axis and a tiny negative for drifting away. The 0.01 scale keeps it much smaller than the terminal reward so it guides but doesn't dominate.

**Hold this in reserve.** Try 1A + 1B + 1C first. Only add 1D if training stalls.

### Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Eval success rate | 96–100% | ≥95% |
| Fuel efficiency (% of Hohmann) | 32–61% | ≥65% |
| Agent strategy | Many small burns | Recognizable 2-impulse pattern |
| Mean episode length | 43–94 steps | Similar or lower |

### Validation

1. Train from scratch with new reward + action space, no debris, ~10M steps
2. `eval_checkpoint.py` on 50 episodes — check success rate and mean Δv
3. `plot_trajectory.py` — visually inspect for two-burn structure
4. Compare mean total Δv to mean Hohmann Δv across eval episodes
5. If efficiency ≥ 65%: warm-start debris training, confirm no regression
6. If efficiency < 50% after 10M steps: add dense shaping (1D), retrain

---

## Phase 2: Elliptical Target Orbits

### Problem

The agent only transfers between circular orbits (e ≈ 0). Real orbital maneuvers often target eccentric orbits (Molniya, GTO, etc.). Elliptical targets require the agent to learn *where* on its orbit to burn — burns at periapsis increase apoapsis (and vice versa), which is how you shape eccentricity. This is qualitatively different from circular-to-circular transfers.

### Changes

#### 2A. Randomize Target Eccentricity

During `reset()`, sample `target.e` from a range instead of fixing it near zero:

```c
// Current
env->target.e = 0.0;

// New — uniform in [0, e_max], curriculum-controlled
env->target.e = random_float() * env->e_max_target;
```

Add `e_max_target` as a configurable parameter (exposed in `orbital.ini`). Start curriculum at 0.1, increase to 0.3 as training progresses.

**Curriculum schedule (manual or automated):**

| Phase | e_max_target | Training Steps | Warm-Start From |
|-------|-------------|----------------|-----------------|
| 2a | 0.0 (circular only) | — | Phase 1 checkpoint |
| 2b | 0.1 | 10M | Phase 2a |
| 2c | 0.2 | 10M | Phase 2b |
| 2d | 0.3 | 10M | Phase 2c |

If the agent handles 0.3 after this curriculum, skip intermediate stages next time and just train on [0, 0.3] directly from the Phase 1 checkpoint.

#### 2B. Tighten Eccentricity Tolerance

Current success condition: `e < 0.01`. For elliptical targets, the success condition becomes:

```c
// Current
int success = (fabs(sat_a - target_a) < 10e3) && (sat_e < 0.01);

// New
int success = (fabs(sat_a - target_a) < 10e3) &&
              (fabs(sat_e - target_e) < 0.01);
```

This requires the agent to match the target eccentricity, not just keep its own eccentricity low.

#### 2C. Add Argument of Periapsis to Observations

For elliptical orbits, the *orientation* of the ellipse matters. Two orbits with the same (a, e) but different argument of periapsis (ω) are different orbits. In 2D, ω is the angle from the reference direction to periapsis.

**New observation fields** (insert after target eccentricity):

| Index | Field | Normalization |
|-------|-------|---------------|
| 8 | Satellite ω | (sin(ω), cos(ω)) pair |
| 9 | (cos component) | |
| 10 | Target ω | (sin(ω_target), cos(ω_target)) pair |
| 11 | (cos component) | |

This adds 4 observation dimensions (25 → 29 after Phase 1's sin/cos anomaly change).

**Implementation:**
- `orbital.h`: Add `double omega` to the `Orbit` struct. Compute during cartesian-to-elements conversion (ω = atan2 of eccentricity vector). Write sin/cos pair to observations.
- Track ω through burns: when applying impulse and reconverting to elements, ω changes. The existing cartesian→elements conversion should already handle this if it computes the full state.
- `orbital.py`: Update observation space shape and index documentation.

#### 2D. Randomize Target ω

During `reset()`, sample `target.omega` uniformly in [0, 2π). The agent must learn to orient its orbit's periapsis to match.

**Success condition update:**

```c
// Angular difference, wrapped to [-π, π]
double dw = fmod(sat_omega - target_omega + M_PI, 2*M_PI) - M_PI;

int success = (fabs(sat_a - target_a) < 10e3) &&
              (fabs(sat_e - target_e) < 0.01) &&
              (target_e < 0.005 || fabs(dw) < 0.1);
              // ω tolerance only matters when target is non-circular
```

The `target_e < 0.005` guard skips ω matching for near-circular targets where ω is undefined/meaningless.

### Success Criteria

| Metric | Target |
|--------|--------|
| Eval success (circular targets, e=0) | ≥95% (no regression) |
| Eval success (elliptical, e ∈ [0, 0.1]) | ≥85% |
| Eval success (elliptical, e ∈ [0, 0.3]) | ≥75% |
| Fuel efficiency | ≥50% of bi-elliptic optimal |
| Learned behavior | Burns concentrated at apoapsis/periapsis |

### Validation

1. Eval on 50 circular-only episodes — must match or beat Phase 1 performance
2. Eval on 50 episodes with e_target ∈ [0, 0.1] — track success and Δv
3. Eval on 50 episodes with e_target ∈ [0, 0.3] — track success and Δv
4. Plot trajectories — check that burns cluster at apsides (periapsis/apoapsis), which indicates the agent learned *where* to burn, not just *how hard*
5. Add debris, warm-start, confirm avoidance still works

---

## Phase 3: Rendezvous

### Problem

The current success condition matches orbit *shape* (a, e) but not *position* on the orbit. Two objects in identical orbits but at different true anomalies are far apart. Real rendezvous requires closing the phase gap — the angular separation between the satellite and target.

This is the hardest phase because the optimal strategy is counterintuitive: to catch something ahead of you in the same orbit, you *slow down* (drop to a lower, faster orbit), let it come to you, then raise back up. The agent must learn that going backward is the way forward.

### Changes

#### 3A. Target Object Instead of Target Orbit

Replace the abstract target orbit with a physical target body that moves:

```c
// In Env struct
// Current:  Orbit target;
// New:
typedef struct {
    Orbit orbit;
    double hard_radius;   // e.g. 10 m (small satellite)
    double rendezvous_radius;  // e.g. 1 km — success threshold
} TargetBody;
```

The target body is Kepler-propagated each step, just like debris. It moves along its orbit at the natural orbital velocity — the agent must intercept it.

#### 3B. Phase Angle Observation

Add the phase angle (angular separation) between satellite and target to the observation space:

| Index | Field | Normalization |
|-------|-------|---------------|
| new | sin(Δθ_phase) | [-1, 1] |
| new+1 | cos(Δθ_phase) | [-1, 1] |
| new+2 | Target true anomaly sin | [-1, 1] |
| new+3 | Target true anomaly cos | [-1, 1] |

Where `Δθ_phase = θ_satellite - θ_target` (angular separation in inertial frame, accounting for ω differences if orbits are elliptical).

Total observation dimensions: ~33 (29 from Phase 2 + 4 new).

#### 3C. Rendezvous Success Condition

```c
// Compute physical distance between satellite and target
double dx = sat_x - target_x;
double dy = sat_y - target_y;
double dist = sqrt(dx*dx + dy*dy);

// Also check relative velocity for a "stable" rendezvous
double dvx = sat_vx - target_vx;
double dvy = sat_vy - target_vy;
double rel_vel = sqrt(dvx*dvx + dvy*dvy);

int success = (dist < env->target_body.rendezvous_radius) &&
              (rel_vel < 10.0);  // < 10 m/s relative velocity
```

The relative velocity constraint prevents "flyby" solutions where the agent passes through the rendezvous sphere at high speed. A real rendezvous requires arriving slow enough to dock. The 10 m/s threshold is generous — real docking approach rates are < 1 m/s — but it prevents the worst degenerate strategies.

**Note on orbit matching:** The rendezvous condition replaces the orbit-shape matching entirely. The agent doesn't need to be in the exact same orbit as the target — it just needs to be in the same place at the same time, moving at roughly the same velocity. The orbit matching emerges naturally from the physics.

#### 3D. Phasing Reward Shaping

Rendezvous from sparse reward alone is very hard. The agent needs to discover a multi-step strategy (lower orbit → coast → raise orbit) where intermediate states look *worse* (farther from target orbit). Dense shaping is likely necessary here.

**Approach 1 — Phase angle closing (simple):**

```c
float phase_prev = compute_phase_angle(sat_prev, target_prev);
float phase_curr = compute_phase_angle(sat_curr, target_curr);
float phase_shaping = 0.02f * (fabs(phase_prev) - fabs(phase_curr));
rewards[0] += phase_shaping;
```

Rewards closing the phase gap. Problem: this fights the "go to a lower orbit first" strategy because dropping down initially *increases* the phase angle. May cause local optima.

**Approach 2 — Distance closing (more robust):**

```c
float dist_prev = compute_distance(sat_prev, target_prev);
float dist_curr = compute_distance(sat_curr, target_curr);
float dist_shaping = 0.02f * (dist_prev - dist_curr) / scale;
rewards[0] += dist_shaping;
```

Rewards reducing physical distance. More robust because the phasing orbit *does* reduce distance over time, even though it temporarily increases phase angle. But it can still create local optima for direct-approach strategies.

**Approach 3 — Potential-based shaping (theoretically clean):**

```c
// Potential function: negative distance to target
float phi_prev = -compute_distance(sat_prev, target_prev) / scale;
float phi_curr = -compute_distance(sat_curr, target_curr) / scale;
float shaping = 0.05f * (phi_curr - phi_prev);
// Potential-based shaping provably preserves optimal policy
```

Potential-based shaping (Ng et al. 1999) is guaranteed not to change the optimal policy — it only changes the speed of learning. This is the safest option if we're worried about shaping distorting behavior.

**Recommendation:** Start with Approach 2 (distance closing) at low scale (0.01–0.02). If it creates degenerate direct-approach strategies, switch to Approach 3. If neither works, try curriculum (start with small phase gaps, increase over training).

#### 3E. Curriculum for Phase Gap

Rendezvous difficulty scales with initial phase separation. Start easy:

| Stage | Initial Phase Gap | Training |
|-------|-------------------|----------|
| 3a | 0°–30° (very close, trivial) | 5M steps |
| 3b | 0°–90° | 10M steps |
| 3c | 0°–180° (full difficulty) | 15M steps |

Each stage warm-starts from the previous.

### Success Criteria

| Metric | Target |
|--------|--------|
| Eval success (small phase gap, < 30°) | ≥90% |
| Eval success (medium phase gap, < 90°) | ≥75% |
| Eval success (large phase gap, < 180°) | ≥60% |
| Learned behavior | Visible phasing orbit in trajectory plots |
| No regression | Debris avoidance still functional |

### Validation

1. Eval suites at each phase gap range
2. Trajectory plots — look for the phasing orbit pattern: lower orbit → coast → raise back
3. Compare to analytical phasing orbit Δv (there's a closed-form solution for circular co-altitude rendezvous)
4. Debris eval — confirm avoidance isn't compromised
5. Fuel efficiency — phasing adds Δv cost, but agent shouldn't use more than ~2× analytical minimum

---

## Implementation Order & Dependencies

```
Phase 1: Fuel Shaping (~2 days)
│
│  1. Modify reward in orbital.h (1 line)
│  2. Trim action space to 7 (orbital.h + orbital.py)
│  3. Add sin/cos true anomaly (orbital.h + orbital.py)
│  4. Train from scratch, no debris, 10M steps
│  5. Eval: success rate + fuel efficiency
│  6. If efficiency < 50%: add dense Δa shaping, retrain
│  7. Warm-start debris training, eval
│
Phase 2: Elliptical Orbits (~3 days)
│  Depends on: Phase 1 checkpoint
│
│  1. Add ω to Orbit struct + cartesian↔elements conversion
│  2. Add ω sin/cos observations (satellite + target)
│  3. Randomize target eccentricity + ω in reset()
│  4. Update success condition for (a, e, ω) matching
│  5. Curriculum: e_max 0.0 → 0.1 → 0.2 → 0.3
│  6. Eval at each eccentricity range
│  7. Trajectory plots — verify apsidal burn clustering
│  8. Add debris, confirm no regression
│
Phase 3: Rendezvous (~4 days)
│  Depends on: Phase 2 checkpoint
│
│  1. Add TargetBody struct, propagate each step
│  2. Add phase angle + target anomaly observations
│  3. Replace orbit-matching success with distance + velocity check
│  4. Add distance-closing shaping reward
│  5. Curriculum: phase gap 30° → 90° → 180°
│  6. Train, eval at each phase gap range
│  7. Trajectory plots — look for phasing orbits
│  8. If stuck: try potential-based shaping or tighter curriculum
│  9. Add debris, final eval suite
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fuel shaping makes agent too conservative (won't burn when needed) | Low | Medium | 0.5 floor on fuel multiplier ensures any success > any failure |
| Elliptical targets too hard to match ω precisely | Medium | Medium | Loosen ω tolerance, or only enforce ω when e > threshold |
| Rendezvous reward shaping creates local optima | High | High | Potential-based shaping (Approach 3), aggressive curriculum |
| Action space too small for elliptical control | Low | Medium | Radial burns give eccentricity control; add medium Δv actions if needed |
| Observation space grows too large (33 dims) | Low | Low | Still small for an MLP — 64×64 hidden layers handle this fine |
| Phase 2/3 training destabilizes debris avoidance | Medium | Medium | Always eval with debris after each phase; fine-tune if regressed |
| Warm-starting fails across phases | Medium | High | Each phase is trainable from scratch if needed; just slower |

---

## Observation Space Evolution

For reference, how the obs vector grows across phases:

| Index | Phase 1 (25) | Phase 2 (29) | Phase 3 (~33) |
|-------|-------------|-------------|---------------|
| 0 | sat a | sat a | sat a |
| 1 | sat e | sat e | sat e |
| 2 | sin(θ) | sin(θ) | sin(θ) |
| 3 | cos(θ) | cos(θ) | cos(θ) |
| 4 | v_r | v_r | v_r |
| 5 | v_t | v_t | v_t |
| 6 | fuel | fuel | fuel |
| 7 | target a | target a | target a |
| 8 | target e | target e | target e |
| 9 | — | sin(sat ω) | sin(sat ω) |
| 10 | — | cos(sat ω) | cos(sat ω) |
| 11 | — | sin(target ω) | sin(target ω) |
| 12 | — | cos(target ω) | cos(target ω) |
| 13 | — | — | sin(Δθ_phase) |
| 14 | — | — | cos(Δθ_phase) |
| 15 | — | — | sin(θ_target) |
| 16 | — | — | cos(θ_target) |
| 9/13/17+ | body fields (4 × 4) | body fields (4 × 4) | body fields (4 × 4) |

Body field block starts at index 9 (Phase 1), 13 (Phase 2), or 17 (Phase 3). Each body contributes 4 floats: Δr, Δθ, closing rate, keepout radius.

---

*Spec written: 2026-04-13*
