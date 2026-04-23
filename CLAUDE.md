# CLAUDE.md — Orbital RL Environment for PufferLib

## Project Overview

Building a custom PufferLib Ocean environment that trains an RL agent to perform fuel-optimal orbital maneuvers while avoiding debris. Written in C for performance (targeting 1M+ steps/sec), bound to Python via PufferLib's C API pattern.

This is both a learning project and a portfolio piece targeting space industry roles (APL, Lincoln Lab, Blue Origin). The goal is a working demo that shows RL + orbital mechanics + systems-level C engineering.

## Repository Structure

This is a fork of [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) on the `3.0` branch. The custom environment lives in:

```
pufferlib/ocean/orbital/
├── orbital.h          # C environment: structs, physics, init/reset/step
├── orbital.c          # Standalone C main() for testing without Python
├── binding.c          # Python C API binding (PufferLib pattern)
├── orbital.py         # PufferEnv wrapper defining obs/action spaces
└── __init__.py

pufferlib/config/ocean/
└── orbital.ini        # Training hyperparameters

scripts/orbital/
├── plot_trajectory.py # Matplotlib viz from logged .npz files
└── export_logs.py     # Helper to extract/convert trajectory logs
```

Also need to register in:
- `pufferlib/ocean/environment.py` (add lazy import)
- `setup.py` (add extension path)

## Reference Examples

Study these existing Ocean envs to understand the pattern:
- **Squared** (`pufferlib/ocean/squared/`) — simplest example, tutorial walkthrough in docs
- **Target** (`pufferlib/ocean/target/`) — multi-agent, continuous state, multi-discrete actions
- **Pong** (`pufferlib/ocean/pong/`) — slightly more complex single-agent

The official tutorial is at https://puffer.ai/docs.html under "Writing Your Own Environment."

## Environment Design

### Scenario (v1)
Single satellite performs fuel-optimal transfer from initial circular orbit to target circular orbit, avoiding debris. 2D coplanar orbits only (no inclination).

### Observation Space (~24 floats, Box)

| Field | Count | Description |
|-------|-------|-------------|
| Satellite orbital state | 6 | a (normalized), e, θ, v_r, v_θ, fuel (0-1) |
| Target orbit | 2 | a_target (normalized), e_target |
| Closest N bodies (N=4) | 16 | Δr, Δθ, closing rate, keepout_radius per body |
| **Total** | **24** | |

All values normalized roughly to [-1, 1]. The body observations are sorted by distance (closest first). The agent doesn't know if a body is Earth, debris, or a planet — it just sees "thing with a keep-out radius at distance X." This makes the policy transfer naturally to new scenarios with different body types.

### Action Space (Discrete, 9 actions)

```
0: Coast (no burn)
1: Prograde   small  (Δv = 10 m/s)
2: Prograde   large  (Δv = 50 m/s)
3: Retrograde small  (Δv = 10 m/s)
4: Retrograde large  (Δv = 50 m/s)
5: Normal+    small  (Δv = 10 m/s)
6: Normal-    small  (Δv = 10 m/s)
7: Radial in  small  (Δv = 10 m/s)
8: Radial out small  (Δv = 10 m/s)
```

### Reward Function

Start naive. Only add complexity if the agent can't learn from this.

```
Per-step:
  0                                     # nothing. just coast.

Terminal:
  +10.0   if agent reaches target orbit (|a - a_target| < tol and e < tol)
  -10.0   if agent collides with anything (enters hard-body radius)
  -10.0   if agent goes out of bounds (orbit exceeds altitude ceiling)
  -10.0   if agent runs out of fuel (can't maneuver = mission failure)
```

No timeout — the episode ends for physical reasons only: you arrived, you crashed, you escaped, or you're stranded. Every failure mode is something that would actually kill a real mission.

**Planned reward upgrades (add one at a time, only if needed):**
1. Fuel bonus: scale terminal success by `(1.0 - total_fuel_used)` to prefer efficient transfers
2. Time penalty: `-0.001` per step to discourage coasting forever
3. Potential shaping: `+c * (Δa_prev - Δa)` for dense learning signal if sparse reward is too hard
4. Keep-out zone penalty: smooth ramp when inside an object's keep-out radius

### Thrust & Fuel Model

**How burns work:**
Each action applies an instantaneous impulse (Δv) to the satellite's velocity vector. This models chemical rocket burns, which are short compared to orbital periods. The Δv is applied in the satellite's local orbital frame: prograde (along velocity), radial (toward/away from Earth), or normal (out of plane, limited in 2D).

**Fuel consumption — Tsiolkovsky rocket equation:**
```
Δm/m₀ = 1 - exp(-|Δv| / (Isp * g₀))

where:
  m₀    = current total mass (dry mass + remaining fuel)
  Δm    = fuel consumed by this burn
  Isp   = specific impulse (engine efficiency), ~300s for typical bipropellant
  g₀    = 9.80665 m/s²
  Isp*g₀ = exhaust velocity ≈ 2942 m/s
```

Bigger burns cost disproportionately more fuel (exponential, not linear). A 50 m/s burn costs more than 5× a 10 m/s burn in fuel mass. This naturally incentivizes efficient maneuver planning.

**Fuel budget:**
```c
#define FUEL_FRACTION 0.15    // fuel is 15% of initial mass
#define ISP 300.0             // specific impulse (seconds)
#define G0 9.80665            // standard gravity (m/s²)
```

15% fuel fraction gives roughly 480 m/s of total Δv budget. A Hohmann transfer between 400km and 800km costs ~110 m/s, so the agent has ~4x the theoretical minimum. Enough to succeed with a decent policy, but not enough to brute-force with random burns.

**Running out of fuel:**
When fuel hits zero, the agent can still coast but cannot burn. If it hasn't reached the target orbit, it's stranded — terminal failure (-10). This is realistic: a satellite with no fuel is space junk.

**Fuel in observation space:**
Already included as a normalized [0, 1] float. The agent can see how much fuel it has left and should learn to budget.

### Collision & Keep-Out Model

Every physical object (Earth, debris, future planets) uses the same two-radius model:

```c
typedef struct {
    Orbit orbit;
    double hard_radius;     // physical body — collision = episode over
    double keepout_radius;  // exclusion zone — agent should avoid
    int is_static;          // 1 = fixed at origin (Earth), 0 = orbiting
} Body;
```

**Object types and radii:**

| Object | Hard Radius | Keep-Out Radius | Notes |
|--------|-------------|-----------------|-------|
| Earth  | 6,371 km (R_earth) | ~6,571 km (200km altitude) | Reentry = death, atmosphere = drag |
| Debris | 1 m (effectively point) | 5 km | Based on real conjunction screening thresholds |
| Moon (v2) | 1,737 km | 10,000 km | Future cislunar scenarios |

**Collision check logic (per step):**
```
For each body:
  d = distance(satellite, body)
  if d < body.hard_radius  → terminal, reward = -10, episode over
  if d < body.keepout_radius → (future: penalty, for now: nothing)
```

This is generic — adding a new body (planet, station, another satellite) is just pushing another `Body` onto the array with its radii. The agent doesn't need to know *what* it's avoiding, just that there are circles it shouldn't enter.

**Earth specifically:** the satellite can't go below ~200km without atmospheric drag destroying it. So Earth's keep-out radius isn't just the surface — it's the surface plus a survivable altitude floor. Entering the keep-out zone is "you're decaying," entering the hard radius is "you burned up."

### Episode Termination & Bounds

No arbitrary step limit. Episodes end for physical reasons:

| Condition | Trigger | Reward |
|-----------|---------|--------|
| **Success** | \|a - a_target\| < 10 km and e < 0.01 | +10.0 |
| **Collision** | Distance to any body < hard_radius | -10.0 |
| **Escape** | Specific orbital energy ≥ 0 (hyperbolic trajectory) | -10.0 |
| **Stranded** | Fuel = 0 and not in target orbit | -10.0 |
| **Safety cap** | 2000 steps reached (prevents infinite coasting) | -10.0 |

**Escape detection (physics-based "out of bounds"):**

Instead of an arbitrary altitude ceiling, use the satellite's specific orbital energy:

```c
// Specific orbital energy (vis-viva)
double E = 0.5 * v² - MU / r;

// E < 0  → bound orbit (elliptical) — normal
// E = 0  → parabolic escape — exactly enough energy to leave
// E > 0  → hyperbolic escape — gone forever
```

If E ≥ 0, the satellite is on an escape trajectory and will never return. This is the physically rigorous definition of "out of bounds" — it doesn't depend on an arbitrary altitude, it works regardless of what bodies exist in the scene, and it generalizes perfectly to multi-body scenarios. A satellite in a highly eccentric orbit with apoapsis at 50,000 km is still bound (E < 0) and could come back. A satellite that got a huge prograde burn and hit escape velocity is gone (E ≥ 0).

For v2 with multiple gravitational bodies, we can extend this to check against the Hill sphere of the primary body — the region where its gravity dominates. The Hill sphere is the largest sphere centered at the secondary body within which the sum of gravitational and centrifugal forces is directed toward that body. Earth's Hill sphere extends to about 1.5 million km, but for near-Earth orbit scenarios, escape velocity is the practical bound.

**Practically bounded orbits that are still "lost":**

There's a subtlety: a satellite could be on a bound orbit (E < 0) but with semi-major axis so large that it takes days to complete one orbit and will never reach the target within any reasonable timeframe. The safety cap at 2000 steps handles this — it's not a timeout, it's "you're in a useless orbit and wasting time." We could also check if the agent's remaining Δv budget is insufficient to reach the target orbit (reachability check), but that's an optimization for later.

### Orbital Timing & Phasing

All non-static bodies orbit according to Kepler propagation each step. This means:

**The target position changes over time.** Even though the target *orbit* (a, e) is fixed, the target *location on that orbit* advances each step. For v1 we're matching orbit shape only (not rendezvous), so the agent doesn't need to time its arrival. But the debris IS moving, which means:

**Debris creates dynamic windows.** A piece of debris that's blocking the transfer path right now might be on the other side of its orbit in 30 minutes. The agent can learn to **wait for a clear window** before burning — which is a genuinely sophisticated strategy that emerges naturally from the physics. This is one of the most interesting things about this environment: the optimal policy isn't just "burn prograde twice" (Hohmann), it's "burn prograde twice, but *when* matters because of debris phasing."

**What the agent sees:** The observation includes relative distance, bearing, and closing rate for nearby bodies. As debris orbits, these values change every step, giving the agent real-time information about whether a conjunction is approaching or receding. A smart policy will learn to coast when debris is nearby and burn when the path is clear.

### Episode Parameters
- **Timestep:** 60 seconds of sim time per step
- **Init orbit altitude:** random 300-800 km (a = R_earth + altitude)
- **Target orbit altitude:** random 300-800 km (different from init)
- **Debris:** 4-8 bodies in random orbits within the altitude band
- **Success tolerance:** |a - a_target| < 10 km and e < 0.01
- **Fuel fraction:** 0.15 (15% of initial mass, ~480 m/s Δv budget)
- **Engine Isp:** 300s (typical bipropellant thruster)
- **Escape bound:** specific orbital energy ≥ 0 (vis-viva)
- **Safety cap:** 2000 steps (~33 hours orbital time)

## Trajectory Logging & Visualization

### Data Logging (build from day 1)

The C environment should write trajectory data to a ring buffer every step. The Python wrapper dumps this to disk periodically. This gives us data for both debugging and later animation.

**Per-step log record:**
```c
typedef struct {
    int episode_id;
    int step;
    float sim_time;          // seconds since episode start
    float sat_x, sat_y;     // satellite Cartesian position
    float sat_vx, sat_vy;   // satellite velocity
    float sat_a, sat_e;     // orbital elements
    float sat_theta;        // true anomaly
    float fuel_remaining;
    int action_taken;        // 0-8
    float reward;
    float delta_v_applied;   // magnitude of burn this step
    float min_debris_dist;   // closest conjunction distance
    // debris positions (for rendering)
    float debris_x[MAX_DEBRIS];
    float debris_y[MAX_DEBRIS];
} TrajectoryRecord;
```

**Logging strategy:**
- C side: write each step's data into a flat buffer (`TrajectoryRecord log[MAX_STEPS]`)
- Python side: at episode end, dump the buffer to a `.jsonl` or `.npz` file
- Log selectively during training: every N episodes, or on terminal success/failure
- Store with metadata: episode reward, whether agent succeeded, total Δv used

**Output directory structure:**
```
logs/orbital/
├── training/
│   ├── ep_000100.npz      # periodic snapshots during training
│   ├── ep_001000.npz
│   └── ep_010000.npz
├── eval/
│   ├── best_transfer.npz  # best fuel-efficient transfer found
│   └── debris_avoidance.npz
└── metadata.json           # training config, reward weights, etc.
```

### Basic Visualization (for debugging, Phase 1-2)

**Terminal/ASCII viz (simplest, works in C standalone):**
Print a polar grid with satellite position marked. Good enough to sanity-check orbits:
```
     .  .  .  .  .
  .        *        .     * = debris
 .    S              .    S = satellite
.          E          .   E = Earth
 .              T    .    T = target orbit marker
  .        *        .
     .  .  .  .  .
```

**Matplotlib viz (Phase 2, Python side):**
A simple script that reads a logged `.npz` file and plots:
- Earth (blue circle)
- Satellite trajectory (colored by time, green→red)
- Debris orbits (gray ellipses)
- Target orbit (dashed blue)
- Burn events (red arrows showing thrust direction/magnitude)
- Conjunction events (yellow warning markers)

This doesn't need to be in the training loop — just a standalone `plot_trajectory.py` that reads log files. Keep it decoupled.

### Future Animation (not now, but design for it)

The logged trajectory data is everything needed to later build:
- Animated matplotlib/manim orbits for a blog post
- Three.js WebGL visualization for personal website
- Side-by-side: early training (random burns) vs converged policy (clean Hohmann)
- Training montage: overlay 100 episode trajectories at step 1K, 10K, 100K, 1M

The key is that the C logging struct captures everything. As long as we have (x, y) for satellite + all debris at every step, plus action/reward metadata, any viz can be built after the fact.

## Physics Implementation

### Constants
```c
#define MU 3.986004418e14    // Earth μ (m³/s²)
#define R_EARTH 6.371e6      // Earth radius (m)
#define DT 60.0              // Timestep (seconds)
#define MAX_BODIES 16        // Earth + debris + future planets
#define MAX_STEPS 2000       // safety cap (generous, ~33 hrs orbital time)
#define ALT_CEILING 1.6e6    // 1600 km altitude ceiling (m above surface)
#define FUEL_FRACTION 0.15   // fuel = 15% of initial mass
#define ISP 300.0            // specific impulse (seconds)
#define G0 9.80665           // standard gravity (m/s²)
#define VE (ISP * G0)        // exhaust velocity ≈ 2942 m/s
```

### C Structs

```c
typedef struct {
    double a;       // semi-major axis (m)
    double e;       // eccentricity
    double M;       // mean anomaly (rad)
    double theta;   // true anomaly (rad) — derived from M
} Orbit;

typedef struct {
    Orbit orbit;
    double dry_mass;    // mass without fuel (kg), stays constant
    double fuel_mass;   // remaining fuel (kg), decreases with burns
} Satellite;
// total mass = dry_mass + fuel_mass
// fuel fraction = fuel_mass / (dry_mass + fuel_mass)
// when fuel_mass = 0, agent can coast but not burn

// Generic physical body — debris, planets, stations all use this
typedef struct {
    Orbit orbit;            // orbital elements (for Earth: unused, it's at origin)
    double hard_radius;     // physical body radius — collision = episode over
    double keepout_radius;  // exclusion zone — penalty/warning region
    int is_static;          // 1 = fixed at origin (Earth), 0 = orbiting
} Body;

// Trajectory logging — one per step, dumped at episode end
typedef struct {
    int step;
    float sim_time;
    float sat_x, sat_y;
    float sat_vx, sat_vy;
    float sat_a, sat_e, sat_theta;
    float fuel;
    int action;
    float reward;
    float delta_v;
    float min_conj_dist;           // closest approach to any body
    float body_x[MAX_BODIES];      // positions of all bodies this step
    float body_y[MAX_BODIES];
} TrajectoryRecord;

typedef struct {
    Satellite sat;
    Orbit target;
    Body bodies[MAX_BODIES];       // bodies[0] = Earth, rest = debris/planets
    int num_bodies;
    int step;
    double total_fuel_used;

    // Trajectory log buffer
    TrajectoryRecord log[MAX_STEPS];
    int log_enabled;         // flag: set from Python to enable/disable logging

    // PufferLib buffer pointers (set during init)
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
} Env;
```

**Collision check (in step function):**
```c
for (int i = 0; i < env->num_bodies; i++) {
    double d = distance(sat_x, sat_y, body_x[i], body_y[i]);
    if (d < env->bodies[i].hard_radius) {
        // collision — terminal, reward = -10
        env->rewards[0] = -10.0f;
        env->terminals[0] = 1;
        return;
    }
    // keepout zone violations tracked for future reward shaping
}
```

### Core Functions Needed

1. **Kepler propagation:** Advance mean anomaly M += n*dt where n = sqrt(μ/a³). Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson (3-5 iterations). Convert E → true anomaly θ.

2. **Elements ↔ Cartesian:** Convert (a, e, θ) to (x, y, vx, vy) for distance calcs. r = a(1-e²)/(1+e*cos(θ)). Standard perifocal frame equations.

3. **Impulse application with fuel consumption:**
   - Convert Δv from local orbital frame to inertial, add to velocity vector
   - Compute fuel consumed: `Δm = m_total * (1 - exp(-|Δv| / VE))`
   - Subtract Δm from fuel_mass. If fuel_mass would go negative, clamp burn to remaining fuel.
   - Convert updated velocity back to orbital elements

4. **Collision/conjunction check:** For each Body, propagate its orbit (if not static), compute Cartesian distance to satellite, check against hard_radius (collision) and keepout_radius (warning). Earth is bodies[0] with is_static=1 (always at origin).

### Simplifications for v1
- 2D only (coplanar orbits, no inclination/RAAN/AoP)
- Impulsive maneuvers (instantaneous Δv, not continuous thrust)
- Two-body dynamics only (no J2, no drag, no third-body)
- Fixed debris orbits (no debris maneuvers or fragmentation)

## Build & Run Commands

```bash
# Compile C standalone (for testing physics without Python)
cd pufferlib/ocean/orbital
gcc -O2 -fsanitize=address -lm orbital.c -o orbital_test
./orbital_test

# Build Python extension
python setup.py build_ext --inplace --force

# Test that it steps
puffer eval puffer_orbital --train.device cpu

# Train
puffer train puffer_orbital --train.device cpu

# Train on MPS (Apple Silicon) — may or may not help for tiny models
puffer train puffer_orbital --train.device mps

# Debug build (address sanitizer + Python)
DEBUG=1 python setup.py build_ext --inplace --force
```

## Development Phases

### Phase 1: Pure C Physics (CURRENT)
- [ ] Write orbital.h with structs and physics functions
- [ ] Write orbital.c with main() that runs a test scenario
- [ ] Verify: satellite in circular orbit stays circular (no drift)
- [ ] Verify: prograde burn raises orbit (compare to analytic Δv)
- [ ] Verify: Hohmann transfer Δv matches textbook value
- [ ] Add TrajectoryRecord logging to step function
- [ ] Add basic ASCII terminal rendering (polar grid with sat/debris/target)
- [ ] Compile with address sanitizer, fix any memory issues

### Phase 2: PufferLib Binding
- [ ] Write binding.c following Squared example
- [ ] Write orbital.py with Gymnasium spaces
- [ ] Add to setup.py and environment.py
- [ ] Wire up trajectory log buffer — Python can read it at episode end
- [ ] Verify: builds, steps, doesn't crash
- [ ] Verify: observations are in expected range
- [ ] Write `plot_trajectory.py` — reads .npz, renders orbit plot with matplotlib

### Phase 3: Training & Reward Iteration
- [ ] Train without debris first (pure transfer task)
- [ ] Monitor reward curves — should see clear learning signal
- [ ] Log trajectories at checkpoints (ep 100, 1K, 10K, 100K)
- [ ] Compare learned Δv to theoretical Hohmann Δv
- [ ] Use plot_trajectory.py to visually inspect early vs late training behavior
- [ ] Add debris, iterate on conjunction penalty weight
- [ ] Tune hyperparameters (lr, gamma, etc.) if needed

### Phase 4: Polish
- [ ] Terminal or raylib rendering
- [ ] README with physics explanation + results
- [ ] Training curve plots
- [ ] Side-by-side trajectory plots: untrained vs trained agent
- [ ] GIFs/animations of learned behavior (for website/portfolio)

## Common Pitfalls (from PufferLib docs)

- **Zero observations/rewards/terminals at start of step.** If you don't memset, values carry over from previous step.
- **Observation scale matters.** Keep everything roughly in [-1, 1]. Raw orbital radii in meters will break training.
- **Type mismatches.** Make sure C buffer types match Python Gymnasium space dtypes.
- **Resets:** Environment must handle its own resets internally when terminal conditions are hit.
- **Mac multiprocessing:** May need `__main__` guard. If hanging, try `--vec.backend Serial` first.

## Validation Benchmarks

Use these to verify physics correctness before training:

| Test | Expected Result |
|------|----------------|
| Circular orbit at 400km, no burns, 1000 steps | a unchanged, e ≈ 0 |
| Hohmann LEO (400km) → MEO (2000km) | Δv ≈ 678 m/s total |
| Hohmann LEO (400km) → GEO (35786km) | Δv ≈ 3935 m/s total |
| Prograde burn increases a | a_after > a_before |
| Retrograde burn decreases a | a_after < a_before |
| 10 m/s burn fuel cost | Δm/m₀ ≈ 0.0034 (0.34%) |
| 50 m/s burn fuel cost | Δm/m₀ ≈ 0.0169 (1.69%) — ~5× the 10 m/s cost |
| Total Δv budget (15% fuel fraction) | ~480 m/s |
| Orbit above 1600 km altitude | Still bound (E < 0), episode continues |
| Burn past escape velocity | E ≥ 0, episode terminates (escape) |
| Orbit below 200 km altitude | Episode terminates (Earth collision) |
| Exhaust all fuel, not at target | Episode terminates (stranded) |

## Key Design Decisions Log

Track decisions here as they're made so both Claude Code and Claude.ai Project stay in sync:

- **2D coplanar for v1** — eliminates 3 orbital elements, halves physics complexity
- **Discrete actions** — PPO works better, simpler reward shaping
- **9 actions with 2 thrust levels** — gives agent granularity without explosion of action space
- **60s timestep** — long enough for meaningful orbital arcs, short enough for fine control
- **Normalize obs to [-1,1]** — critical for training stability
- **Naive reward function first** — terminal only: +10 success, -10 collision/OOB/stranded. Add shaping only if needed.
- **Generic Body model with hard_radius + keepout_radius** — Earth, debris, future planets all use same struct. Agent sees "obstacle at distance with radius," not "this is debris vs planet."
- **Earth = bodies[0], is_static=1** — always at origin. Debris/planets orbit normally. Uniform collision check loop.
- **No timeout — physical termination only** — collision, escape (E ≥ 0), stranded (no fuel), or success. Safety cap at 2000 steps only prevents infinite coasting.
- **Escape detection via specific orbital energy** — E = ½v² - μ/r. If E ≥ 0, satellite is on hyperbolic escape trajectory. Physics-based, no arbitrary altitude ceiling, generalizes to multi-body scenarios.
- **All non-static bodies orbit each step** — debris moves, creating dynamic conjunction windows. Agent can learn to wait for clear paths before burning. This emergent timing strategy is one of the most interesting aspects of the env.
- **v1 matches orbit shape, not position** — agent needs to match (a, e) of target, not rendezvous at a specific point on the orbit. Rendezvous (position matching + phasing) is a v2 goal.
- **Tsiolkovsky fuel model** — exponential cost, not linear. 50 m/s burn costs ~5× a 10 m/s burn in fuel. Naturally incentivizes efficient maneuvers.
- **15% fuel fraction (~480 m/s Δv)** — ~4× theoretical Hohmann minimum for 400→800km. Enough for a good policy, not enough for random burns.
- **Isp = 300s** — typical bipropellant thruster. Could increase to ~3000s for electric propulsion in a continuous-thrust v2.
- **Log trajectories from C side** — flat struct buffer, dumped to .npz from Python at episode end
- **Decouple visualization** — standalone plot_trajectory.py reads log files, not embedded in training loop

## Future Roadmap (v2+)

Not in v1 scope, but documented here so the design stays forward-compatible.

### Time Warp Actions

**Problem:** Long-duration scenarios (cislunar, interplanetary) require thousands of coast steps where the agent does nothing. This makes episodes impossibly long and wastes compute.

**Solution:** Add discrete warp actions that advance the sim by large dt without burning:

```
9:  Warp 1 hour    (dt = 3600s)
10: Warp 6 hours   (dt = 21600s)
11: Warp 1 day     (dt = 86400s)
12: Warp 1 week    (dt = 604800s)
```

**Implementation:** Call the same Kepler propagator with a larger dt. The math is analytic (exact at any dt), so there's no accuracy loss. All bodies advance, fuel doesn't change, one line of C:

```c
case ACTION_WARP_6HR:
    propagate_all_orbits(env, 21600.0);
    break;
```

**Why this works for two-body:** Kepler's equation gives exact positions at any time. No numerical integration, no accumulated error. Warping 6 hours is as accurate as stepping 360 times at 60s.

**Why this works for n-body (with caveats):** The sim internally sub-steps the warp (e.g., break a 6-hour warp into 360 one-minute RK4 steps). The agent acts once, the sim crunches silently. Slower than analytic Kepler but still much faster than making the agent sit through each step.

**Open issues to solve when implementing:**

1. **Collision during warp.** A warp skips intermediate positions — the satellite and debris could pass within meters during the skipped interval and you'd never know. Solutions:
   - Sub-sample a few intermediate points and check distances (simple, slightly conservative)
   - Compute closest approach analytically from the two orbits' geometry (exact, more math)
   - For short warps in LEO this is negligible — orbits are ~90 min so a 1-hour warp barely misses anything

2. **Reward accounting.** If per-step penalties are added later, warps must accumulate the proportional penalty (6-hour warp = 360× step penalty), otherwise the agent learns to warp instead of coast to dodge the clock.

3. **Observation discontinuity.** After a big warp, the world looks completely different. The agent needs to handle this — debris positions, relative geometries all change dramatically. Not a code issue (obs are recomputed fresh), but a learning challenge since different actions have wildly different magnitudes of effect on the state.

**What this enables:**
- Cislunar transfers (3-5 day duration) in ~100 agent steps
- Interplanetary transfers (months) in ~200-500 agent steps
- Agent learns *when* to pay attention vs *when* to skip — itself a meaningful strategic skill
- Mars transfer becomes: burn → warp weeks → course correct → warp weeks → insertion burn

### N-Body Gravity

Switch from analytic Kepler to numerical integration (RK4/RK45) summing gravitational forces from all massive bodies. Enables:
- **Cislunar scenarios** (Earth-Moon-Sun, 3 bodies) — Lagrange points, low-energy transfers, ballistic capture
- **Interplanetary transfers** (Earth-Sun-Mars) — combined with warp actions
- Hill sphere replaces vis-viva for escape detection in multi-body systems
- Compute cost: 10-100× slower per step than Kepler, still feasible on M3 Max with sub-million steps/sec

**Best first n-body target:** Earth-Moon system. Transfer times are short (3-5 days), three-body dynamics create interesting features, directly relevant to Artemis program / APL work.

### Other v2+ Ideas
- **3D orbits** — add inclination changes (extremely expensive maneuvers, interesting Δv tradeoffs)
- **Rendezvous** — match position on orbit, not just orbit shape (requires phasing)
- **Multi-agent pursuit-evasion** — inspector vs evader via self-play (PettingZoo)
- **Continuous thrust** — model electric propulsion (spiral transfers, Isp ~3000s)
- **Realistic conjunction data** — ingest TLE/CDM data from Space-Track.org
- **WASM browser demo** — compile C to WebAssembly like PufferLib's existing demos
- **Station-keeping** — maintain orbit against J2 perturbation drag
- **Constellation management** — multiple agents maintaining a formation

---

*Last updated: March 26, 2026*
*Origin: Planning conversation on claude.ai*
