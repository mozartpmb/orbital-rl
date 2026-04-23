# Orbital RL — PufferLib Ocean Environment
## V1 Project Scope & Build Plan

---

## The Simplest Interesting V1

**Scenario:** A single satellite must perform a fuel-optimal Hohmann-like transfer from an initial circular orbit to a target circular orbit, while avoiding a small field of debris objects in fixed orbits.

**Why this scope:** It's simple enough to train in minutes on your M3 Max, but complex enough to be non-trivial — the agent has to discover when to burn, how hard, and in what direction, while routing around conjunction threats. A classical Hohmann transfer is the "known optimal" baseline you can compare against, so you can actually measure if your agent is learning something real.

---

## State / Action / Reward Design

### Observation Space (~20 floats)

```
Satellite state (6):
  - semi-major axis (normalized)          a
  - eccentricity                          e
  - true anomaly                          θ
  - radial velocity                       v_r
  - tangential velocity                   v_θ
  - fuel remaining (0-1)                  fuel

Target orbit (2):
  - target semi-major axis (normalized)   a_target
  - target eccentricity                   e_target

Closest N debris objects (N × 3, e.g. N=4 → 12):
  - relative distance (normalized)        Δr
  - relative bearing                      Δθ
  - closing rate                          ḋr
```

Total: ~20 floats. Small, fast, no CNN needed — just an MLP.

### Action Space (Discrete, ~9 actions)

```
0: No burn (coast)
1: Prograde   small   (Δv = 10 m/s)
2: Prograde   large   (Δv = 50 m/s)
3: Retrograde small   (Δv = 10 m/s)
4: Retrograde large   (Δv = 50 m/s)
5: Normal     small   (Δv = 10 m/s)
6: Normal     large   (Δv = 10 m/s)
7: Radial in  small   (Δv = 10 m/s)
8: Radial out small   (Δv = 10 m/s)
```

Discrete is simpler for PPO and for a v1. You can go continuous later.

### Reward Function

Start as simple as possible. Only add complexity when the naive version demonstrably fails.

```
Per-step:   0 (nothing)

Terminal:
  +10.0   reached target orbit
  -10.0   collided with a body (entered hard_radius)
  -10.0   out of bounds (orbit altitude > 1600 km ceiling)
  -10.0   stranded (fuel = 0 and not in target orbit)
```

No timeout, no shaping. Every episode ends for a physical reason. If the agent can learn orbit transfers from this alone, the physics and obs space are correct.

### Collision & Keep-Out Model

Every physical object uses a generic `Body` struct with two radii:
- **hard_radius** — physical body. Entering = collision = episode over (-10 reward)
- **keepout_radius** — exclusion zone. For future penalty/shaping, not enforced in v1

| Object | Hard Radius | Keep-Out Radius |
|--------|-------------|-----------------|
| Earth  | 6,371 km    | 6,571 km (200km altitude floor) |
| Debris | ~1 m        | 5 km |

The agent's observation includes each nearby body's keepout_radius as a feature, so it can learn to avoid them even before we add explicit penalties. Adding new body types (Moon, stations, other satellites) is just appending to the array.

### Thrust & Fuel

**Impulse burns:** Each action applies an instantaneous Δv (chemical rocket model). The velocity changes immediately in the satellite's local orbital frame.

**Tsiolkovsky fuel consumption:**
```
fuel_consumed = total_mass * (1 - exp(-|Δv| / (Isp * g₀)))

Isp = 300s (bipropellant thruster)
exhaust velocity = Isp * g₀ ≈ 2942 m/s
```

Key property: fuel cost is **exponential**, not linear. A 50 m/s burn costs ~5× the fuel of a 10 m/s burn. This naturally teaches the agent that big burns are expensive.

**Fuel budget:** 15% fuel fraction → ~480 m/s total Δv. A Hohmann transfer 400→800km costs ~110 m/s, so the agent has ~4× the theoretical minimum. Room to learn, but not to waste.

**Out of fuel:** Agent can still coast but can't burn. If not in target orbit → terminal failure (stranded).

### Episode Structure

No arbitrary step limit. Episodes end for physical reasons:

| Condition | Reward |
|-----------|--------|
| Reached target orbit (a and e within tolerance) | +10.0 |
| Collision with any body (d < hard_radius) | -10.0 |
| Escape trajectory (specific orbital energy E ≥ 0) | -10.0 |
| Fuel exhausted, not at target | -10.0 |
| Safety cap: 2000 steps (~33 hrs) | -10.0 |

**Escape detection** uses vis-viva energy: `E = ½v² - μ/r`. If E ≥ 0, the satellite is on a hyperbolic trajectory and will never return. This is the physics-based "out of bounds" — no arbitrary altitude ceiling. It generalizes to any scenario regardless of what bodies exist.

**Debris orbits over time.** All non-static bodies are Kepler-propagated each step. This means debris that's blocking your transfer path right now might be clear in 30 minutes. The agent can discover that **waiting for a window** before burning is an optimal strategy — one of the most interesting emergent behaviors we hope to see.

**v1 matches orbit shape, not position.** The agent needs to match (a, e) of the target orbit, not rendezvous at a specific point. Full rendezvous (matching phasing) is a v2 upgrade.

- **Reset:** randomize initial/target orbit radii within 300-800 km altitude band, scatter 4-8 debris objects in random orbits

---

## Orbital Mechanics Core (the C physics)

You need ~4 functions, totaling maybe 150-200 lines of C:

### 1. Kepler Propagator
```c
// Advance orbit by dt seconds under two-body dynamics
// Input:  orbital elements (a, e, M0, ...) + dt
// Output: updated mean anomaly M, then solve for true anomaly θ
//
// M = M0 + n * dt          where n = sqrt(μ / a³)
// Solve Kepler's equation:  M = E - e * sin(E)  via Newton-Raphson
// Convert E → θ:           tan(θ/2) = sqrt((1+e)/(1-e)) * tan(E/2)
```

This is the heart of the sim. Newton-Raphson converges in 3-5 iterations for typical eccentricities. You've seen this math in your astronomy coursework.

### 2. Orbital Elements ↔ Cartesian Conversion
```c
// Convert (a, e, θ) → (x, y, vx, vy) for distance calculations
// r = a(1-e²) / (1 + e*cos(θ))
// In perifocal frame:
//   x = r * cos(θ),    y = r * sin(θ)
//   vx = -sqrt(μ/p) * sin(θ)
//   vy =  sqrt(μ/p) * (e + cos(θ))
```

### 3. Impulse Maneuver Application
```c
// Apply delta-v in local orbital frame (prograde/normal/radial)
// Convert Δv from orbital frame to inertial, add to velocity vector
// Convert back to orbital elements
//
// This is where burns happen — literally just vector addition
```

### 4. Conjunction Check
```c
// For each debris object:
//   propagate debris orbit to current time
//   compute distance to satellite
//   return minimum distance and closing rate
//
// Simple Euclidean distance in Cartesian coords
```

**Simplification for v1:** Work in 2D (coplanar orbits). This eliminates inclination, RAAN, and argument of periapsis — cutting the problem from 6 orbital elements to 3 (a, e, θ). The physics is identical, just projected onto a plane. You can add 3D later.

---

## File Structure (inside PufferLib)

```
pufferlib/ocean/orbital/
├── orbital.h          # C structs + init/reset/step/render functions
├── orbital.c          # main() for standalone C testing
├── binding.c          # Python C API binding (PufferLib pattern)
├── orbital.py         # PufferEnv wrapper (observation/action spaces)
└── __init__.py

pufferlib/config/ocean/
└── orbital.ini        # hyperparameters

scripts/orbital/
├── plot_trajectory.py # Matplotlib orbit visualizer (reads .npz logs)
└── export_logs.py     # Log extraction helpers
```

Plus one line added to `pufferlib/ocean/environment.py`.

---

## Trajectory Logging

Log trajectory data from C every step into a flat buffer. Python dumps to `.npz` at episode end. This gives us debugging data now and animation data later.

```c
typedef struct {
    int step;
    float sim_time;
    float sat_x, sat_y;           // Cartesian position
    float sat_a, sat_e, sat_theta; // orbital elements
    float fuel;
    int action;
    float reward;
    float delta_v;                // burn magnitude
    float min_debris_dist;
    float debris_x[MAX_DEBRIS];
    float debris_y[MAX_DEBRIS];
} TrajectoryRecord;
```

Store `TrajectoryRecord log[MAX_STEPS]` in the Env struct. Toggle with `log_enabled` flag so logging doesn't slow down bulk training (only log every N episodes or during eval).

**What this enables later (without rebuilding):**
- Matplotlib orbit plots with burn arrows and debris positions
- Animated trajectories (manim, matplotlib.animation, or Three.js)
- Side-by-side: untrained agent (random burns) vs converged agent (clean transfer)
- Training montage: overlay many trajectories at different training checkpoints
- Website portfolio piece with interactive WebGL visualization

---

## Basic Visualization (for debugging)

**Phase 1 — ASCII in C standalone:**
A simple polar grid printed to terminal each step. Enough to visually confirm orbits are circular, burns change trajectory, debris positions make sense.

**Phase 2 — `plot_trajectory.py` (matplotlib):**
Standalone script that reads a `.npz` log file and renders:
- Earth (blue circle at origin)
- Satellite path (line colored green→red over time)
- Debris orbits (gray dashed ellipses)
- Target orbit (blue dashed circle)
- Burn events (red arrows at burn locations)
- Conjunction warnings (yellow markers when close to debris)

Fully decoupled from training — just reads log files.

---

### The C Header (orbital.h) — Core Structs

```c
#include <math.h>
#include <string.h>

#define MAX_DEBRIS 8
#define MU 3.986e14          // Earth gravitational parameter (m³/s²)
#define R_EARTH 6.371e6      // Earth radius (m)
#define CONJ_THRESHOLD 5000  // 5 km conjunction warning
#define DT 60.0              // 60 second timestep
#define MAX_STEPS 500

typedef struct {
    double a;       // semi-major axis (m)
    double e;       // eccentricity
    double M;       // mean anomaly (rad)
    double theta;   // true anomaly (rad) — derived from M
} Orbit;

typedef struct {
    Orbit orbit;
    double fuel;    // remaining fuel (0-1)
} Satellite;

typedef struct {
    Satellite sat;
    Orbit target;
    Orbit debris[MAX_DEBRIS];
    int num_debris;
    int step;

    // PufferLib buffer pointers (set during init)
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
} Env;

// Core API
void init(Env *env, float *obs, float *actions,
          float *rewards, unsigned char *terminals);
void reset(Env *env);
void step(Env *env);
```

---

## Build Sequence (what you actually do, in order)

### Phase 1: Pure C (1-2 days)
1. Write `orbital.h` with the structs and physics functions
2. Write `orbital.c` with a `main()` that runs a simple scenario
3. Print satellite position each step, verify orbits look correct
4. Manually test: hardcode a prograde burn, confirm it raises the orbit
5. Compile with `-fsanitize=address` to catch memory bugs early

### Phase 2: PufferLib Binding (half day)
1. Write `binding.c` following the Squared example pattern
2. Write `orbital.py` defining obs/action spaces as Gymnasium Box/Discrete
3. Add to `setup.py` extension paths and `environment.py`
4. `python setup.py build_ext --inplace --force`
5. Run `puffer eval puffer_orbital` to verify it steps without crashing

### Phase 3: Training (1-2 days of iteration)
1. Start with no debris, just orbit transfer — sanity check
2. Train: `puffer train puffer_orbital --train.device cpu`
3. Watch reward curves. If flat → debug obs normalization & reward scale
4. Once transfer works, add debris and iterate on reward shaping
5. Compare learned policy's fuel usage to theoretical Hohmann Δv

### Phase 4: Polish (1 day)
1. Add terminal rendering (ASCII orbit diagram, or hook into raylib)
2. Record training curves and agent behavior GIFs
3. Write up results: learned Δv vs optimal Δv, debris avoidance behavior
4. Push PR or publish standalone repo with README

---

## What "Done" Looks Like

- Agent reliably transfers between random orbit pairs
- Uses <2x the fuel of a theoretical Hohmann transfer
- Successfully avoids debris (zero conjunctions after training)
- Trains to convergence in <30 minutes on M3 Max
- Clean repo with README explaining the physics + RL design

---

## Stretch Goals (v2+)

### Time Warp Actions (high priority for v2)
Add discrete warp actions (1hr, 6hr, 1day, 1week) that advance the simulation without burning. Uses the same Kepler propagator with a larger dt — analytic, exact, no accuracy loss. This solves the long-episode problem for cislunar and interplanetary scenarios, turning a 9-month Mars transfer into ~200-500 agent decisions. Key subtlety: need collision detection during warps (sub-sample intermediate points or compute closest approach analytically).

### N-Body Gravity
Switch to numerical integration (RK4) for Earth-Moon-Sun system. Enables cislunar transfers, Lagrange point navigation, low-energy ballistic capture. Best first target: Earth-Moon (3-5 day transfers, directly relevant to Artemis/APL). Combined with warp actions, interplanetary transfers become tractable.

### Other Ideas
- **3D orbits** — add inclination changes, which are extremely expensive maneuvers
- **Rendezvous** — match position on orbit, not just orbit shape (requires phasing)
- **Multi-agent pursuit-evasion** — inspector vs evader via self-play (PettingZoo)
- **Continuous thrust** — model low-thrust electric propulsion (spiral transfers)
- **Realistic conjunction data** — ingest actual TLE/CDM data from Space-Track.org
- **WASM browser demo** — compile C to WebAssembly like PufferLib's existing demos
- **Station-keeping** — maintain orbit against J2 perturbation drag
- **Constellation management** — multiple agents maintaining a formation
- **Station-keeping** — agent maintains orbit against J2 perturbation drag
