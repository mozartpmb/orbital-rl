# Phase 5 Recipe — Reproducible Spec

The recipe ships a 2D coplanar orbital rendezvous agent at LEO altitudes (300-800 km) with sat & target eccentricity ≤ 0.05. **For the per-condition capability surface, see `PHASE5_FINDINGS.md` §1 — this document specifies what runs, not what generalizes.**

---

## Environment

### Action space — `Discrete(10)`

| idx | name | dv (m/s) | direction | use |
|---|---|---|---|---|
| 0 | coast | 0 | — | wait, default |
| 1 | prograde fine | +5 | along velocity | small phasing |
| 2 | prograde small | +10 | along velocity | small transfer |
| 3 | prograde medium | +25 | along velocity | larger transfer |
| 4 | retrograde fine | −5 | against velocity | brake fine |
| 5 | retrograde small | −10 | against velocity | brake small |
| 6 | retrograde medium | −25 | against velocity | brake larger |
| 7 | radial out | +10 | away from Earth | rare |
| 8 | radial in | −10 | toward Earth | rare |
| 9 | warp 5 min | 0 | — | propagate 5×60s without burning |

### Observation space — `Box(38,)` (or 48 with `enable_action_mask=1`, off by default)

| index | content |
|---|---|
| 0-6 | sat orbital state: a (norm), e, sin θ, cos θ, vr/v_circ, vt/v_circ, fuel_frac |
| 7-8 | target a (norm), target e |
| 9-12 | sin/cos sat ω, sin/cos target ω |
| 13-16 | sin/cos (θ_sat - θ_target), sin/cos target θ |
| 17-32 | closest 4 bodies: each with Δr, Δθ, closing_rate, keepout_radius |
| 33-37 | LVLH-frame relative state: dx_l, dy_l, dvx_l, dvy_l, n_target |

LVLH frame is critical (Phase 4 finding). Without it the policy can't generalize across target inertial position.

### Termination

| reason | trigger | reward |
|---|---|---|
| success | \|sat.a − target.a\| < 30km AND \|relvel\| < 50 m/s AND \|θ_sat − θ_target\| < small | +10 |
| collision | distance to body < hard_radius (R_EARTH for Earth) | −10 |
| escape | specific orbital energy ≥ 0 | −10 |
| stranded | fuel_mass = 0 with no rendezvous | −10 |
| safety_cap | step ≥ MAX_STEPS = 2000 | −10 |

### Reward shaping — gated NHR with terminal Φ-clamp

Per non-terminal step:

```
r_step = β · (γ^τ · Φ(s_{t+1}) − Φ(s_t))
β = BETA_SHAPE = 1.0
γ = 0.995 (matches PPO discount)
τ = 1 (5 under warp action)
Φ(s) = −(W_ORBIT · Φ_orbit · σ₁ + W_PHASE · Φ_phase · σ₂ + W_VEL · Φ_vel · σ₃)
```

Component weights all 0.01. Gates:

```
σ₁ = 1 (always active)
σ₂ = sigmoid((EPS_ORBIT − Φ_orbit) / TAU_ORBIT)   # opens as orbit closes
σ₃ = σ₂ · sigmoid((EPS_PHASE − Φ_phase) / TAU_PHASE)  # opens as phase aligns
```

`EPS_ORBIT = 2.0`, `EPS_PHASE = 0.3`. `TAU_ORBIT = 0.1·EPS_ORBIT`, `TAU_PHASE = 0.1·EPS_PHASE`. Smooth sigmoid (not threshold step).

Terminal Φ-clamp: at terminal, append `β·(0 − Φ_prev)` to reward to undo accumulated bias. (See orbital.h:c_step for exact form.)

### Env config (at training)

```python
Orbital(
    num_envs=1024,
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05,
    init_phase_gap_max=3.14159,        # uniform [-π, π]
    same_orbit_init=<stage-dependent>,
    valid_init_only=1,                  # MANDATORY — Phase 5e fix
    e_mix_easy_frac=0.0, e_mix_easy_max=0.05,
    collision_penalty_w=0.0,            # default off
    enable_action_mask=0,               # default off
)
```

Default obs is 38-dim. `enable_action_mask=1` switches to 48-dim with last 10 dims = action validity mask, but the deliverable doesn't use it.

---

## Curriculum — two stages, both at e_max=0.05

### Stage 1.0 — bootstrap (same_orbit_init=1)

```bash
puffer train puffer_orbital --train.seed <S> --train.total-timesteps 40000000 \
    --train.device cpu --env.init-phase-gap-max 3.14159 \
    --env.e-max-target 0.05 --env.e-max-sat 0.05 \
    --env.same-orbit-init 1 --env.valid-init-only 1 \
    --train.checkpoint-interval 5 --tag stage_1_0_seed<S>
```

40M steps, ~7 min wall on M3 Max. The same_orbit_init flag forces sat to start on identical orbit shape as target; only θ differs. This bootstraps the rendezvous timing problem cleanly.

Expected: convergence to ≥99% on the same_orbit task by epoch 250-300.

### Stage 4.0 — fully random (same_orbit_init=0), warm-started from 1.0

```bash
puffer train puffer_orbital --train.seed <S> --train.total-timesteps 50000000 \
    --train.device cpu --env.init-phase-gap-max 3.14159 \
    --env.e-max-target 0.05 --env.e-max-sat 0.05 \
    --env.same-orbit-init 0 --env.valid-init-only 1 \
    --train.checkpoint-interval 5 --tag stage_4_0_seed<S> \
    --load-model-path <stage_1_0_best_ckpt>
```

50M steps, ~7 min wall. Sat init is fully random within altitude × eccentricity bounds.

Expected: convergence to 86-92% multi-seed at e_max=0.05 fully random by epoch 250-325. Held-out at e_max=0.20 (still aggregate, not fixed-e) reaches 90% multi-seed; fixed e=0.05 LEO reaches ~85%.

### Stage transition — best-ckpt by held-out

After each stage, scan ckpts every 25 epochs at the next-stage eval condition (typically e_max=0.20 fully random with valid_init_only=1, 50 eps each). Pick best. Use as warm-start for next stage.

The "train-longer null first" rule applies if a stage stalls below 80%. Extend the training budget by 50% (e.g., 60M for Stage 1.0) before declaring a problem.

### Multi-seed protocol

5 seeds: {42, 1337, 20260423, 31415, 2718} (matching Phase 5a addendum convention).

Run all 5 in parallel where compute permits. Aggregate mean ± std at the held-out eval. Phase 5e showed 5/5 seeds reach >85% at e_max=0.20; no collapse seed.

---

## Hyperparameters

### PPO config (defaults from `pufferlib/config/ocean/orbital.ini`)

```
total_timesteps = depends on stage (40M / 50M)
gamma = 0.995
learning_rate = 0.01 (annealed via lr_min=1e-6, lr_max=1e-2)
minibatch_size = 8192
ent_coef = 0.01
clip_coef_low = 0.2
clip_coef_high = 0.2
kl_target = 0.0   (no KL constraint)
target_entropy_controller = False  (R3 stack disabled)
l2_init_coef = 0.0  (R3 disabled)
vf_clip_coef = (default)
```

Disabled R3 components: DAPO, L2-init, TEC, adaptive KL. They were tested in Phase 4 R3 spec and *hurt* training. Don't re-enable without strong evidence.

### Env physics constants

```
MU = 3.986004418e14 m³/s²
R_EARTH = 6.371e6 m
ALT_MIN = 200e3 m  (200 km — atmospheric collision floor)
ALT_MAX = 1.6e6 m   (1600 km — scenario altitude ceiling)
EARTH_KEEPOUT = R_EARTH + ALT_MIN = 6.571e6 m
DT = 60 s  (per agent step)
WARP_TAU = 5  (warp action advances 5×DT)
MAX_STEPS = 2000  (~33 hr orbital time)
FUEL_FRAC = 0.15  (15% fuel mass; ~480 m/s Δv budget)
ISP = 300 s  (bipropellant)
G0 = 9.80665 m/s²
SUCCESS_TOL_A = 10000 m  (semi-major axis match)
REL_VEL_TOL = 50 m/s  (terminal velocity match)
```

---

## Capability — per-condition values (from W2 surface)

See `web_data/results/phase5_capability_surface_agg.csv` for the full surface. Key points:

- **e=0.0 LEO:** 75-100% across (phase × relation). Phase 4 condition.
- **e=0.025 LEO fully_random:** **97.2-97.8%** across phase gaps. Peak performance.
- **e=0.05 LEO fully_random:** 84-87%. The deliverable.
- **e=0.075 LEO fully_random:** 13-16%. Recipe falls off here.
- **e=0.10 LEO:** ~0%. Cliff.
- **e ≥ 0.10 with alt scaled to keep perigee valid:** 0-7%. Recipe doesn't generalize to high altitudes either.

The aggregate "e_max=0.20 = 90%" number from Phase 5e Block II is the result of `valid_init_only=1` filtering uniform [0, 0.20] samples to retain only the low-e ones. It's a real number for that distribution but doesn't reflect "handles e=0.20."

---

## Limits

### What this recipe does not do

- High eccentricity (≥ 0.075). Fall-off is sharp; not gradual.
- Altitudes outside 300-800km. Recipe never trained outside LEO.
- Phase 4 conditions specifically (e=0, target circular at different a) — 4-5pp below the original Phase 4 R4 baseline of 96.4%. Generalization tax of training under random sat eccentricity.
- 3D orbital mechanics. 2D coplanar only.
- Continuous thrust. Impulsive Δv only.
- Active debris. Debris is supported in the env but `num_debris_*=0` in the recipe.
- Time horizons longer than ~33 hours (MAX_STEPS=2000 cap).

### What this recipe gets but is not validated

- Generalization to slightly different fuel budgets. Untested but probably small effect.
- Generalization to different scenario altitudes (e.g., MEO 1500-3000km). Untested; per the W2 surface, mostly fails outside the trained band.

---

## Reproduction

```bash
# Train all 5 seeds (multi-seed deliverable)
bash scripts/orbital/p5e_curriculum.sh all

# Eval one seed at the canonical e_max=0.20 with valid_init_only filter
cd pufferlib && python3 scripts/orbital/eval_checkpoint.py \
    ../models/phase5e/seed42_stage4_best.pt \
    --episodes 200 --e-max-target 0.20 --e-max-sat 0.20 \
    --init-phase-gap-max 3.14159 --valid-init-only 1

# Per-condition surface (240 cells × 100 eps × 5 seeds, parallelized 4-way)
python3 scripts/orbital/p5wrap_surface_eval.py --episodes 100 --workers 4

# LEO probe (per-eccentricity at trained altitude band)
python3 scripts/orbital/p5wrap_leo_probe.py
```

Expected results:
- Multi-seed Stage 4.0 best at e_max=0.20: ~90% (aggregate)
- LEO probe at e=0.025: ~97% (in-distribution peak)
- LEO probe at e=0.075: ~14% (post-cliff)
- W2 surface at fixed e=0.20 fully_random: ~0%

---

## What's NOT in the recipe (explicit exclusions)

- **R3 components.** Tested in Phase 4 R3 spec. All hurt training (DAPO, L2-init, TEC, adaptive KL). Disabled in `orbital.ini`; don't re-enable without strong evidence.
- **REL_VEL_TOL annealing.** Tested as Phase 5c B6. Didn't help; no effect on the e=0.20 wall.
- **Continuous actions.** Phase 5b ruled out. Discrete(10) including warp-5min works; continuous Box(2) collapses fresh-train and provides no benefit.
- **Bound-expansion curriculum.** Designed in Phase 5e Block III but not implemented. Deferred — `valid_init_only` did the job.
- **Action masking (`enable_action_mask`).** Verified working end-to-end in Phase 5d. Not needed for the deliverable. Available infrastructure for future high-e or constrained-action work.
- **Soft collision penalty (`collision_penalty_w`).** Verified working as a single-step penalty. Ineffective during training (collapses warm-start). Available infrastructure.
- **Mixed-distribution training (B3, `e_mix_easy_*`).** Tested in Phase 5c. Reaches 18% at e_max=0.20 single-seed. Not needed once the env's `valid_init_only` fix is applied.

These exclusions are deliberate. Each represents an intervention class that was tested and determined unnecessary for the recipe's deliverable.
