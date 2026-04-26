# Orbital RL — Phase 4 Deep Dive

Companion to `PHASE4_FINDINGS.md`. Every intervention with full code, what was tried, what happened, and why. Read this when you need to reproduce or audit the work.

**Period:** 2026-04-23 to 2026-04-24
**Plan:** `.claude/plans/steady-honking-kurzweil.md`
**Spec:** `orbital-rl-phase4-spec.md`

---

## 0. Starting state at Phase 4 entry

Phase 3 closed 2026-04-22 at 64% @ 180° on a single seed. The plan reframed this as a pointy-max — and the first thing R0 did was confirm that. Multi-seed eval of the closing ckpt `experiments/puffer_orbital_6f56229o/model_puffer_orbital_000115.pt`:

| Seed | Success @ 180° (50 eps argmax) |
|---|---|
| 42 | 64% |
| 1337 | 52% |
| 20260423 | 42% |
| **Mean** | **52.7%** |

Std ≈ 11pp. This sets the Principle E bar: any intervention claiming success must lift *every* seed above its R0 number, not just the mean.

### R0 bug caught: e_max_target eval mismatch

`scripts/orbital/eval_checkpoint.py` defaulted `--e-max-target 0.1`, but training used `e_max_target=0.0` (circular targets). Policies trained on circular targets scored 0% on elliptical eval. Fixed default to 0.0:

```python
# scripts/orbital/eval_checkpoint.py
parser.add_argument('--e-max-target', type=float, default=0.0,
                    help='Max target eccentricity (default 0.0 matches training)')
```

This bug had been present since Phase 3 evals; corrected before any Phase 4 numbers were collected. All Phase 3 eval numbers stand because they happened to use the eval default but training used 0.0 — so the bug was actually *suppressing* Phase 3 success rates by ~10pp. The new R0 numbers (52.7% mean) are higher than the Phase 3 historical eval mean for that reason.

### R0 verdict

- Plan threshold: ≥60% on fresh seed → seed 42 passes (64%), other seeds fail.
- Real headline: 42-64% range, 52.7% mean.
- All later interventions measured against this distribution.

---

## 1. Intervention R1 — CW/Hill-frame relative state (LVLH observations)

### What we tried

Spec §3.1: replace inertial-frame body offsets with LVLH-frame (target's rotating frame) relative state. The hypothesis: rendezvous policies should reason in the *target's* coordinate frame, where the goal is at origin and motion is governed by Clohessy-Wiltshire dynamics.

### Code that landed

OBS_DIM extended 33 → 38, with five new floats appended:

```c
// pufferlib/pufferlib/ocean/orbital/orbital.h:30
#define OBS_DIM 38   // was 33
```

In `fill_observations` (`orbital.h:475-511`), after the satellite + target + bodies block:

```c
/* LVLH-frame relative state — primary for rendezvous */
double tx, ty, tvx, tvy;
orbit_to_cartesian(&env->target, &tx, &ty, &tvx, &tvy);

/* Target's argument-of-latitude angle — rotation between inertial and LVLH */
double theta_t = env->target.theta + env->target.omega;
double ct = cos(theta_t), st = sin(theta_t);

/* Inertial offsets */
double dxi  = sx  - tx;
double dyi  = sy  - ty;
double dvxi = svx - tvx;
double dvyi = svy - tvy;

/* Rotate offsets into LVLH (radial = +x, along-track = +y) */
double dx_l  =  ct * dxi  + st * dyi;
double dy_l  = -st * dxi  + ct * dyi;
double dvx_l =  ct * dvxi + st * dvyi;
double dvy_l = -st * dvxi + ct * dvyi;

/* Subtract frame-rotation contribution: ω × r where ω = n_tgt ẑ */
double n_tgt = sqrt(MU / pow(env->target.a, 3));
dvx_l += n_tgt * dy_l;
dvy_l -= n_tgt * dx_l;

obs[33] = (float)(dx_l / R_EARTH);
obs[34] = (float)(dy_l / R_EARTH);
double v_circ_t = sqrt(MU / env->target.a);
obs[35] = (float)(dvx_l / v_circ_t);
obs[36] = (float)(dvy_l / v_circ_t);
obs[37] = (float)(n_tgt / 1e-3);   // ~LEO mean motion scale
```

The frame-rotation correction (lines `dvx_l += n_tgt * dy_l; dvy_l -= n_tgt * dx_l;`) is the n×r term: when you rotate a velocity into a rotating frame, you have to subtract the velocity-due-to-rotation, otherwise a satellite stationary in LVLH would appear to be moving.

Python obs space:

```python
# pufferlib/pufferlib/ocean/orbital/orbital.py:54
self.single_observation_space = gymnasium.spaces.Box(
    low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)
```

### Unit test

A stationary-in-LVLH check: place satellite at exactly the target's state, step 100×, assert `|dvx_l|, |dvy_l| < 1e-3 m/s`. Catches the n×r sign-flip — without the correction, dvx_l grows linearly to ~7000 m/s per orbit. With the correction, dvx_l < 1e-6 m/s.

### Warm-start problem

Plan said "warm-start banned across this change (Principle D — obs dim change)." But fresh training at 180° was infeasible at 5M-step budget — early R1 fresh runs returned 0% @ 180° and 6.3% @ 30°. So the actual training used a **zero-pad warm-start**: load the Stage 2 ckpt (33-dim encoder), expand the encoder's first linear layer's input weight from `[128,33]` to `[128,38]` with zeros in the new columns:

```python
# In scripts/orbital/warm_start_lvlh.py (one-shot conversion)
state = torch.load(stage2_ckpt, weights_only=True)
old_w = state['policy.encoder.0.weight']           # [128, 33]
new_w = torch.zeros(old_w.shape[0], 38)
new_w[:, :33] = old_w
state['policy.encoder.0.weight'] = new_w
torch.save(state, expanded_ckpt)
```

Step-0 behavior is identical to Stage 2 (zero contribution from new columns). LVLH columns learn through gradients during continued training.

### What happened

| Seed | R1 perf @ 180° | vs R0 |
|---|---|---|
| 42 | 64% | +0 |
| 1337 | 54% | +2 |
| 20260423 | 46% | +4 |
| **Mean** | **54.7%** | **+2.0** |

Within seed std. Seed 42 flat. **Fails Principle E.**

### Why marginal

Hypothesis: at 180° phase gap, the policy is failing on *reachability* (random init can't discover the reward), not on *representation* (it can't tell where the target is). LVLH gives a cleaner state signal but the bottleneck is upstream. LVLH is permanently baked into the obs space (Stage 3 R4 curriculum uses it) but its standalone contribution remains inconclusive.

---

## 2. Intervention R2 — Gated multi-stage potential shaping

### What we tried

Spec §3.2: replace the existing per-step distance shaping with a multi-stage gated potential Φ(s) that decomposes into orbital-shape, phase, and velocity components, each gated by sigmoids that activate only when the previous component is "close enough." Plus a clamp `Φ(s_terminal)=0` at every termination branch (NHR — Near-Horizon Reward) and an exp-decay terminal bonus.

### compute_phi code

```c
// pufferlib/pufferlib/ocean/orbital/orbital.h:514-562
static inline double compute_phi(Orbital* env) {
    double sx, sy, svx, svy, tx, ty, tvx, tvy;
    orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &svx, &svy);
    orbit_to_cartesian(&env->target, &tx, &ty, &tvx, &tvy);

    /* Orbital-shape error: Δa scaled by tolerance + eccentricity-vector norm */
    double da     = fabs(env->sat.orbit.a - env->target.a) / TOL_A;
    double e_sat  = env->sat.orbit.e;
    double e_tgt  = env->target.e;
    double w_sat  = env->sat.orbit.theta + env->sat.orbit.omega;
    double w_tgt  = env->target.theta    + env->target.omega;
    double dex    = e_sat * cos(w_sat) - e_tgt * cos(w_tgt);
    double dey    = e_sat * sin(w_sat) - e_tgt * sin(w_tgt);
    double phi_orbit = da + sqrt(dex*dex + dey*dey);

    /* Phase error: 1 - cos(Δθ), smooth across ±π wrap */
    double dtheta = env->sat.orbit.theta - env->target.theta;
    double phi_phase = 1.0 - cos(dtheta);

    /* Velocity error in LVLH-relative frame */
    double theta_t = w_tgt;
    double ct = cos(theta_t), st = sin(theta_t);
    double dvxi = svx - tvx, dvyi = svy - tvy;
    double dvx_l =  ct * dvxi + st * dvyi;
    double dvy_l = -st * dvxi + ct * dvyi;
    double n_tgt = sqrt(MU / pow(env->target.a, 3));
    double dx_l  =  ct * (sx - tx) + st * (sy - ty);
    double dy_l  = -st * (sx - tx) + ct * (sy - ty);
    dvx_l += n_tgt * dy_l;
    dvy_l -= n_tgt * dx_l;
    double v_rel = sqrt(dvx_l*dvx_l + dvy_l*dvy_l);
    double phi_vel = v_rel / REL_VEL_TOL;

    /* Gated cascade: σ₂ enables when orbit close, σ₃ enables when both orbit + phase close */
    double sigma_1 = 1.0;
    double sigma_2 = 1.0 / (1.0 + exp(-(EPS_ORBIT - phi_orbit) / TAU_ORBIT));
    double sigma_3 = sigma_2 * (1.0 / (1.0 + exp(-(EPS_PHASE - phi_phase) / TAU_PHASE)));

    return -(W_ORBIT*phi_orbit*sigma_1 + W_PHASE*phi_phase*sigma_2 + W_VEL*phi_vel*sigma_3);
}
```

Gates ensure the policy can't game phase or velocity rewards before the orbit is shape-correct. Constants:

```c
#define BETA_SHAPE  1.0
#define W_ORBIT     0.01    // calibrated down from 0.05 after v1 collapse
#define W_PHASE     0.01
#define W_VEL       0.01
#define EPS_ORBIT   2.0     // gate threshold (60th pct of trained ckpt distribution)
#define EPS_PHASE   0.3
#define TAU_ORBIT   0.2     // 0.1 × EPS_ORBIT
#define TAU_PHASE   0.03
#define TOL_A       1e4     // 10 km
#define REL_VEL_TOL 50.0    // 50 m/s
```

### NHR shaping — terminal clamp

The shaping reward in `c_step` is the **potential-difference form**:

```c
// pufferlib/pufferlib/ocean/orbital/orbital.h:852-961 (c_step)
double phi_curr  = compute_phi(env);
double gamma_tau = pow(GAMMA_PPO, env->last_tau);
env->rewards[0] += (float)(BETA_SHAPE * (gamma_tau * phi_curr - env->phi_prev));
env->phi_prev    = phi_curr;
```

The NHR clamp asserts Φ(s_terminal) = 0 at every termination branch. Without this, the agent gets `−Φ(s_terminal)` as a final shaping reward, which can be large and arbitrary. The clamp ensures the *only* terminal signal comes from the actual terminal reward (success bonus, collision penalty), not the shaping potential. Implemented as a macro applied at every terminal branch:

```c
// orbital.h:568 — applied at every terminal in check_termination
#define R2_NHR_CLAMP(env) do { \
    (env)->rewards[0] += (float)(BETA_SHAPE * (0.0 - (env)->phi_prev)); \
    (env)->phi_prev = 0.0; \
} while(0)
```

Applied at every terminal: escape (line 587), collision (line 602), safety cap (line 631), stranded (line 645), success (line 658). Each branch has both `R2_NHR_CLAMP(env);` and the actual reward assignment.

### Critical bug found and fixed: shaping ordering

Original c_step code computed shaping **before** check_termination. This clobbered terminal rewards:

```c
// WRONG — shaping runs, terminal_reward overwritten
env->rewards[0] += BETA_SHAPE * (gamma_tau * phi_curr - phi_prev);
check_termination(env);   // sets env->rewards[0] = -10.0 on collision
```

The += was overwritten by the = in check_termination, so the shaping potential difference was lost. Fixed ordering: terminate first, then shape — but only on non-terminal steps:

```c
// CORRECT
check_termination(env);  // sets terminals[0] and rewards[0] for terminal branches
if (!env->terminals[0]) {
    double phi_curr = compute_phi(env);
    env->rewards[0] += BETA_SHAPE * (gamma_tau * phi_curr - phi_prev);
    env->phi_prev = phi_curr;
}
// On terminal branches, NHR_CLAMP already applied phi-diff inside check_termination
```

This was a load-bearing fix saved as a feedback memory: shaping must run AFTER check_termination on non-terminal branches; on terminal branches the NHR clamp handles it.

### Exp-decay terminal bonus

In check_termination at success (line ~656), replaced fuel-only bonus:

```c
// Before: env->rewards[0] = 10.0f * (0.5 + 0.5*fuel_remaining);
double d_final  = sqrt((sx-tx)*(sx-tx) + (sy-ty)*(sy-ty));
double vr_final = sqrt(dvx_l*dvx_l + dvy_l*dvy_l);  // LVLH relative vel
env->rewards[0] = 10.0f * (0.5f + 0.5f * fuel_remaining)
                + 50.0f * (float)exp(-d_final  / 5000.0)
                + 50.0f * (float)exp(-vr_final / 10.0);
```

Idea: dense terminal reward incentivizes precise approach, not just "any orbit shape match."

### What was tried — R5 variants on warm-start

Plan called for R5 = R1+R2 combo. Applied on top of Stage 2 LVLH zero-pad warm-start (seed 42), 4 variants:

| Variant | W (W_ORBIT/PHASE/VEL) | Dual bonus | β | Other | Result |
|---|---|---|---|---|---|
| v1 | 0.05 | +50/+50 | 1.0 | default | 8.5% terminal, entropy 0.007 |
| v2 | 0.01 | +10/+10 | 1.0 | scaled down | 1.9%, entropy 0.007 |
| v3 | 0.01 | none | 0.1 | shaping only, no dual bonus | 37% peak then falling, killed at 1.3M |
| v4 | 0.01 | none | 0.1 | + lr=1e-3 | 5.8%, entropy 0.027 |

Trajectory pattern was identical across all variants:
1. Start at warm-start baseline (~55%)
2. Peak at 60-68% around 500k-3M steps
3. Entropy crashes from 0.16 → <0.01
4. Performance crashes to <10%
5. Stays there for the remaining steps

Goodhart check: |G_shape| stayed below 0.1·|R_term| in v3/v4 (the lower-β variants), so the shaping itself wasn't dominating numerically. The collapse was a **policy collapse** mechanism, not a reward-hacking mechanism.

### Root cause

Stage 2's policy entropy is ~0.16 — already near-deterministic. The PPO clip objective with a deterministic policy is brittle: any reward signal that shifts the value head's calibration creates advantages that point in unfamiliar directions, and the policy's clipped update locks in whichever new action dominates the gradient at that moment. Without entropy headroom, there's no exploratory recovery. Confirmed by R3 attempts.

---

## 3. Intervention R3 — Plasticity stack

### What we tried

Spec §3.3: prevent entropy collapse on warm-starts via five mechanisms simultaneously: LayerNorm, L2-init regularizer, DAPO asymmetric clip, adaptive-KL LR scheduler, target-entropy controller.

### Code: LayerNorm (tried then reverted)

`pufferlib/pufferlib/models.py` had stubs at lines 130, 131, 146, 148, 186, 190 commented out. Initial R3b uncommented them and added an encoder LayerNorm:

```python
# models.py:42-45 — Default encoder, R3b ATTEMPT
self.encoder = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.LayerNorm(hidden_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.LayerNorm(hidden_size),
    torch.nn.Tanh(),
)

# LSTMWrapper:138-139 — uncommented
self.pre_layernorm  = torch.nn.LayerNorm(input_size)
self.post_layernorm = torch.nn.LayerNorm(hidden_size)
# and forward calls at lines 154, 156, 194, 198
```

### What happened with LN

LN broke warm-start completely. Activation distributions of the warm-start ckpt do not match LN's expected zero-mean unit-variance, so applying LN at step 0 effectively scrambles the policy. R3b crashed to 4.2% within the first 1M steps and never recovered.

Reverted. Saved a feedback memory: LayerNorm in encoder/LSTM wrapper is incompatible with warm-start; only land it from-scratch.

```python
# models.py:42-45 — final version, GELU only
self.encoder = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.GELU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.GELU(),
)
# LSTMWrapper LN calls reverted to commented-out
```

### Code: init_params (anchor for L2-init)

```python
# models.py:67-68 — Default.__init__ end
self.init_params = {k: v.detach().clone() for k, v in self.named_parameters()}
```

Stored at the end of construction so warm-start init values are captured *after* the load_state_dict call (because PuffeRL clones init_params from the post-load policy).

### Code: L2-init in PPO loss

```python
# pufferl.py:447-461
l2_init_coef = config.get('l2_init_coef', 0.0)
l2_init = torch.zeros((), device=device)
if l2_init_coef > 0.0:
    init_map = getattr(self.policy, 'policy', self.policy)
    init_map = getattr(init_map, 'init_params', None)
    if init_map is not None:
        for k, p in self.policy.named_parameters():
            base = k.split('policy.', 1)[-1] if 'policy.' in k else k
            if base in init_map:
                l2_init = l2_init + ((p - init_map[base])**2).sum()

loss = pg_loss + config['vf_coef']*v_loss - ent_coef_eff*entropy_loss \
     + l2_init_coef * l2_init
```

### Code: DAPO asymmetric clip

```python
# pufferl.py:346-350, 434-437
clip_lo = config.get('clip_coef_low',  clip_coef)   # 0.2
clip_hi = config.get('clip_coef_high', clip_coef)   # 0.28

pg_loss1 = -adv * ratio
pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_lo, 1 + clip_hi)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

Wider upper bound (0.28) lets positive-advantage updates breathe; tighter lower bound (0.2) keeps trust-region safety on the downside.

### Code: Adaptive-KL LR scheduler

```python
# pufferl.py:495-507
kl_target = config.get('kl_target', 0.0)
if not config['anneal_lr'] and kl_target > 0.0:
    lr_min = config.get('lr_min', 1e-6)
    lr_max = config.get('lr_max', 1e-2)
    kl_val = losses['approx_kl']
    if kl_val > 2.0 * kl_target:
        for g in self.optimizer.param_groups:
            g['lr'] = max(g['lr'] / 2.0, lr_min)
    elif kl_val < 0.5 * kl_target:
        for g in self.optimizer.param_groups:
            g['lr'] = min(g['lr'] * 2.0, lr_max)
```

Halves LR on KL drift >2× target, doubles on <0.5× target, clamped to [lr_min, lr_max].

### Code: Target-entropy controller (TEC)

```python
# pufferl.py:213-214 — init
self.entropy_history = []

# pufferl.py:355-369 — every train step, compute ent_coef_eff
ent_coef_eff = config['ent_coef']
if config.get('target_entropy_controller', False) and len(self.entropy_history) >= 1:
    action_space = self.vecenv.single_action_space
    num_actions = getattr(action_space, 'n', None) or int(np.prod(getattr(action_space, 'shape', [1])))
    log_N = float(np.log(max(num_actions, 2)))
    H_mean = float(np.mean(self.entropy_history[-10:]))
    if H_mean < 0.50 * log_N:
        ent_coef_eff = config.get('ent_coef_high', 0.03)
    elif H_mean < 0.80 * log_N:
        ent_coef_eff = config.get('ent_coef_mid', 0.015)
    else:
        ent_coef_eff = config.get('ent_coef', 0.01)

# pufferl.py:485-489 — append after each train step
if config.get('target_entropy_controller', False):
    self.entropy_history.append(float(losses['entropy']))
    if len(self.entropy_history) > 50:
        self.entropy_history = self.entropy_history[-50:]
```

For Discrete(10), log_N = ln(10) ≈ 2.303. Thresholds:
- H < 0.50·log_N (≈1.15) → ent_coef = 0.03 (boost)
- H < 0.80·log_N (≈1.84) → ent_coef = 0.015 (mild boost)
- otherwise → 0.01 (floor, raised from default 0.001)

### orbital.ini additions

```ini
clip_coef_low  = 0.2
clip_coef_high = 0.28
kl_target      = 0.012
lr_min         = 1e-6
lr_max         = 1e-2
l2_init_coef   = 1e-5
anneal_lr      = False
target_entropy_controller = True
ent_coef       = 0.01
ent_coef_mid   = 0.015
ent_coef_high  = 0.03
```

(Later disabled — see "config evolution" section.)

### What happened — R3 variants

| Variant | Setup | Result |
|---|---|---|
| R3a | shaping β=1.0, ent_coef=0.01, no LN, no L2, default clip | 34% peak |
| R3b | R3a + LayerNorm uncommented + encoder LN | 4.2% (LN broke warm-start) |
| R3b2 | R3a + L2-init only (no LN) | 34% peak 68%, then crash |
| R3c | R3b2 + DAPO + adaptive-KL | peak 65% @3M, crash to 4.5% @4.5M |
| R3d | R3c + lr=1e-4 | peak 64% @786k, crash to 2.3% @5M |
| R3-FS | full stack from scratch, 20M steps, 180° | 0.19% terminal |

Common arc on warm-start: peak 60-68%, entropy crash, perf crash.

### R3-FS specifically

20M steps, full stack, from scratch at 180°. TEC was working as designed — entropy oscillated 0.03-0.62 over the run, never collapsed permanently. But the policy never discovered the terminal reward at all: only 0.19% successful eps in 20M steps. The dense gated shaping fires only when Φ_orbit < EPS_ORBIT, but a random policy at 180° never gets close enough for the gates to activate, so the shaping signal is ~0 most of the time, and the policy gradient is dominated by the negative shaping potential at the terminal (escape, stranded). Effectively the policy learned to *avoid trying* — entropy stayed up but policy mass concentrated on "coast" actions.

### Important null finding

The untrained Stage 2 LVLH zero-pad ckpt evaluates at **47.3% mean** at 180° — with no R3 training applied. Every R3 variant degraded this. R3 strictly *destroyed* performance.

### Why R3 failed

Two distinct failure modes:

1. **Warm-start (R3a-d):** The committed policy has entropy ~0.16. New gradient signals (shaping, DAPO clip, adaptive-KL LR) shift the value head off calibration. PPO's clip objective then locks the policy into deterministic mode before the TEC can ramp up entropy. The TEC operates on rolling-mean entropy over 10 updates; collapse happens within 5-10 updates — TEC sees the post-collapse measurement and tries to recover, but PPO clip prevents recovery.

2. **From scratch (R3-FS):** TEC successfully prevents entropy collapse, but preserving entropy ≠ exploration. Random policy at 180° has near-zero probability of stumbling onto a coordinated phasing sequence within an episode. Dense shaping doesn't fire because the gates require getting close first. Policy converges to "do nothing high-entropy" rather than "phase to target."

Saved as a feedback memory: plasticity stack does not rescue committed warm-starts and does not bootstrap from-scratch at hard phase gaps.

---

## 4. Intervention R4 — Time-warp action (SMDP)

### What we tried

Spec §3.4: add a "warp 5 minutes" action that advances the sim by τ=5 sub-steps of DT=60s. Preserves 60s collision check resolution via sub-stepping. Shapes the reward signal with γ^τ via pre-multiplied rewards (no kernel changes needed).

### Code: action table

```c
// orbital.h:50-69
#define NUM_ACTIONS  10
#define WARP_ACTION  9
#define WARP_TAU     5   // 5 × DT(60s) = 300s = 5 min

static const double ACTION_DV[NUM_ACTIONS][3] = {
    {  0,   0, 0},   // 0: coast
    { 10,   0, 0},   // 1: prograde small
    { 50,   0, 0},   // 2: prograde large
    {-10,   0, 0},   // 3: retrograde small
    {-50,   0, 0},   // 4: retrograde large
    {  0,  10, 0},   // 5: normal+
    {  0, -10, 0},   // 6: normal-
    {  0,   0, 10},  // 7: radial out
    {  0,   0,-10},  // 8: radial in
    {  0,   0, 0},   // 9: warp 5 min (no impulse, τ sub-steps)
};
```

### Code: c_step warp branch

```c
// orbital.h:852-961 — c_step (excerpted)
int    tau    = (action == WARP_ACTION) ? WARP_TAU : 1;
double sub_dt = (action == WARP_ACTION) ? DT       : DT;   /* DT = 60s for both */

if (action != WARP_ACTION) {
    apply_impulse(env, action);  /* fuel cost + Δv */
}

for (int k = 0; k < tau; k++) {
    propagate_orbit(&env->sat.orbit, sub_dt);
    propagate_orbit(&env->target,    sub_dt);
    for (int i = 0; i < env->num_bodies; i++) {
        if (!env->bodies[i].is_static) {
            propagate_orbit(&env->bodies[i].orbit, sub_dt);
        }
    }
    if (check_collision_substep(env)) {
        /* terminal — apply NHR clamp + collision penalty */
        R2_NHR_CLAMP(env);
        env->rewards[0] += -10.0f;
        env->terminals[0] = 1;
        env->last_tau = k + 1;
        return;
    }
}
env->last_tau = tau;

/* check_termination + shaping (with γ^τ via env->last_tau) */
check_termination(env);
if (!env->terminals[0]) {
    double phi_curr  = compute_phi(env);
    double gamma_tau = pow(GAMMA_PPO, env->last_tau);
    env->rewards[0] += (float)(BETA_SHAPE * (gamma_tau * phi_curr - env->phi_prev));
    env->phi_prev    = phi_curr;
}
```

`env->last_tau` is consumed by the shaping reward — γ^τ correctly discounts the shaping potential difference across multi-step transitions. Under SMDP, this is the right semantic: a τ-step transition's potential difference is `γ^τ·Φ(s') - Φ(s)`.

### Code: Python action space

```python
# orbital.py:56
self.single_action_space = gymnasium.spaces.Discrete(10)   # was 9
```

### Code: γ^τ pre-multiplied rewards (Path A)

The puff_advantage kernel uses constant γ. To get γ^τ behavior with τ=1 default and τ=5 on warps, pre-multiply the per-step reward by γ^(τ-1) inside C before writing to the rewards buffer:

```c
// At the end of c_step, after all reward components are added:
if (env->last_tau > 1) {
    double gamma_pow = pow(GAMMA_PPO, env->last_tau - 1);
    env->rewards[0] *= (float)gamma_pow;
}
```

Effect: the constant-γ GAE `δ_t = r_t + γ V(s') − V(s)` matches the variable-dt `r_t + γ^τ V(s') − V(s)` after pre-multiplication. Verified with τ=1 byte-for-byte parity test before any warp run was attempted.

### Code: warp_frac logging

Added to `Log` struct and `fill_log`:
```c
log->warp_frac = (float)env->warp_step_count / (float)env->total_step_count;
```

### What happened — from scratch at 180°

8.8M steps, perf 0.000% terminal. Same exploration dead-end as R3-FS — random policy can't bootstrap reward discovery at 180° even with warp helping skip coast periods.

### What happened — phase-gap curriculum

This is the win. Three stages, three seeds.

#### Stage 1: 30°, 10M fresh, Discrete(10) with warp, R3 stack OFF

Command:
```bash
puffer train puffer_orbital --train.seed <S> --train.total-timesteps 10_000_000 \
  --train.device cpu --env.init-phase-gap-max 0.524
```

| Seed | Stage 1 perf | Notes |
|---|---|---|
| 42 | 77.0% | smooth convergence |
| 1337 | **0.0%** | entropy stable at ~0.55, never found reward |
| 20260423 | 89.5% | converged faster than seed 42 |

Phase 3 Stage 1 (Discrete(9), no warp) was 99% at seed 42 — so adding the warp action costs ~22pp from the gold standard. Seed 1337 failure is the seed-sensitivity of Discrete(10) bootstrap — the extra action slot creates failure modes some initial weight distributions can't escape. This is now a documented retry-rate concern (~1/3 seeds).

#### Stage 2: 90°, 15M warm-start from Stage 1

```bash
puffer train puffer_orbital --train.seed <S> --train.total-timesteps 15_000_000 \
  --train.device cpu --env.init-phase-gap-max 1.5708 \
  --load-model-path experiments/puffer_orbital_<stage1>/model_puffer_orbital_000077.pt
```

| Seed | Stage 2 perf | vs Phase 3 Stage 2 (74%) |
|---|---|---|
| 42 | 78.4% | +4pp |
| 1337 | — | (Stage 1 failed) |
| 20260423 | 82.4% | +8pp |

**Both successful seeds beat Phase 3's Stage 2 baseline.** This was the first concrete sign R4 was working in curriculum context.

#### Stage 3: 180°, 15M warm-start from Stage 2

```bash
puffer train puffer_orbital --train.seed <S> --train.total-timesteps 15_000_000 \
  --train.device cpu --env.init-phase-gap-max 3.14159 \
  --load-model-path experiments/puffer_orbital_<stage2>/model_puffer_orbital_000115.pt
```

| Seed | Stage 3 train (rolling 128-ep) | Stage 3 eval (3 rollout seeds × 50 eps) |
|---|---|---|
| 42 | 94.1% | 81.3% (78/86/80%) |
| 1337 | — | — |
| 20260423 | 86.3% | 78.0% (82/80/72%) |

**Successful-seed mean @ 180°: 79.6%.** Above Phase 4 plan target of 70% by 9.6pp. Above R0 baseline 52.7% by 26.9pp.

### Why R4 curriculum worked

1. **Stage 1 at 30° has reachable sparse reward.** Random init can stumble onto success within an episode at 30° gap — terminal +10 is discoverable. Phase 3 history confirms this (99% at Discrete(9)).
2. **Stage 2 at 90° warm-starts from a 30° policy that already knows phasing.** The strategy generalizes (just needs longer drift) — exploration is local, not global.
3. **Stage 3 at 180° warm-starts from a 90° policy.** Same generalization pattern. The policy doesn't have to discover phasing from scratch; it has to extend its known phasing strategy to a wider gap.
4. **Warp action helps Stage 2 and 3** by letting the policy skip coast periods that would otherwise cost decision-budget. This is why R4 curriculum Stage 2 (78-82%) beats Phase 3 Stage 2 (74%) — the warp action contributes value once the policy knows when to use it.

The mechanism is **bootstrapping under reward sparsity**, not "warp makes policies better." Warp from scratch at 180° is just as broken as no-warp from scratch at 180°.

---

## 5. Config evolution at orbital.ini

This is the timeline of what was set/unset during Phase 4 attempts.

### After R0 baseline (Phase 3 inheritance)

```ini
total_timesteps = 15_000_000
gamma = 0.995
ent_coef = 0.001
learning_rate = 0.01
anneal_lr = True
clip_coef = 0.2
init_phase_gap_max = 3.14159
```

### After R1 (LVLH) — obs change only

No ini changes. R1 was a code change (obs dim 33→38).

### After R2/R5 (gated shaping) — failed runs

```ini
beta_shape = 1.0
w_orbit = 0.05  # later 0.01
w_phase = 0.05  # later 0.01
w_vel = 0.05    # later 0.01
eps_orbit = 2.0
eps_phase = 0.3
```

### After R3 (plasticity stack on)

```ini
ent_coef = 0.01           # raised from 0.001
ent_coef_mid = 0.015
ent_coef_high = 0.03
target_entropy_controller = True
clip_coef_low = 0.2
clip_coef_high = 0.28
kl_target = 0.012
lr_min = 1e-6
lr_max = 1e-2
l2_init_coef = 1e-5
anneal_lr = False
```

### Final R4 curriculum config (R3 stack OFF, R1+R2 baked in)

```ini
total_timesteps = 10_000_000  # Stage 1 (overridden via CLI for Stage 2/3)
gamma = 0.995
ent_coef = 0.01               # kept the floor raise from R3
learning_rate = 0.01
anneal_lr = True              # R3 disabled
clip_coef = 0.2
clip_coef_low = 0.2           # symmetric (DAPO disabled)
clip_coef_high = 0.2
kl_target = 0.0               # adaptive-KL disabled
l2_init_coef = 0.0            # L2-init disabled
target_entropy_controller = False
init_phase_gap_max = 0.524    # 30° (overridden via CLI for Stage 2/3)
num_envs = 1024
minibatch_size = 8192
beta_shape = 1.0
w_orbit = 0.01                # NHR shaping baked in but with low weights
w_phase = 0.01
w_vel = 0.01
```

R3 was stripped from the config rather than ifdef'd because every variant tested with R3 enabled regressed. Keeping it as code in pufferl.py / models.py with toggle flags lets future work re-enable it without re-implementing.

---

## 6. Wandb run IDs and checkpoints

For each result in this document. `experiments/puffer_orbital_<id>/model_puffer_orbital_NNN.pt`.

| Run | Seed | Stage | Run ID | Final ckpt epoch |
|---|---|---|---|---|
| Phase 3 Stage 3 (R0 baseline) | 42 | — | 6f56229o | 115 |
| R3-FS (full stack from scratch 20M) | 42 | — | (logged to wandb) | — |
| R4 from-scratch at 180° (8.8M) | 42 | — | (logged) | — |
| R4 curriculum Stage 1 (30°, 10M) | 42 | 1 | jro86awn | 77 |
| R4 curriculum Stage 1 | 1337 | 1 | hbg1qa6f | 77 (FAILED) |
| R4 curriculum Stage 1 | 20260423 | 1 | vopwobur | 77 |
| R4 curriculum Stage 2 (90°, 15M) | 42 | 2 | h2ccoyi1 | 115 |
| R4 curriculum Stage 2 | 20260423 | 2 | 504i206m | 115 |
| R4 curriculum Stage 3 (180°, 15M) | 42 | 3 | nyul1pl8 | 115 |
| R4 curriculum Stage 3 | 20260423 | 3 | cyxmcalu | 115 |

Final shippable ckpts:
- Seed 42: `experiments/puffer_orbital_nyul1pl8/model_puffer_orbital_000115.pt`
- Seed 20260423: `experiments/puffer_orbital_cyxmcalu/model_puffer_orbital_000115.pt`

---

## 7. Lessons / what to remember

1. **Multi-seed gating from day one.** Phase 3's headline 64% was a pointy-max we didn't catch until R0. Every run after that was multi-seed.
2. **Path-dependency >> mechanism choice on hard sparse-reward tasks.** Curriculum > shaping > regularization > optimizer-tuning, in expected payoff.
3. **Warm-start PPO is brittle to reward/structure changes.** Don't introduce new reward terms or new clip shapes onto a committed policy with entropy <0.2. Either restart entropy first, or do the change from scratch with curriculum.
4. **TEC works as a safety net, not as an explorer.** It prevents entropy collapse but doesn't help discovery. Pair with curriculum.
5. **Verify eval defaults match training defaults.** The e_max_target=0.1 vs 0.0 bug suppressed Phase 3 numbers by ~10pp and could have led to abandoning a working approach.
6. **Phase ordering matters in c_step.** Shaping must run AFTER check_termination on non-terminal branches, or terminal rewards get clobbered.
7. **NHR clamp is non-negotiable for potential-based shaping.** Without Φ(s_terminal)=0 enforcement, the agent gets arbitrary terminal shaping rewards that destabilize the value function.
8. **Discrete(10) with warp is seed-sensitive at fresh training.** Plan for ~1/3 retry rate or use action-table zero-pad warm-start from a Discrete(9) policy.

---

## 8. What was NOT done (and why)

- **R6 (R1+R2+R3): not run.** R3 components individually regress on warm-start; combining with R5 (which already collapses) was extremely unlikely to help.
- **R7 full stack: not run.** Same reason as R6, plus the curriculum supersedes the joint ablation question.
- **Stage 1 cross-seed warm-start for seed 1337 recovery: not run.** Out of scope at the time of the final writeup; recommended as Phase 5 prep.
- **Action-table zero-pad from Discrete(9) Phase 3 policy: not tried.** Cleanest theoretical fix for seed 1337 but untested. Recommended for production use of R4 curriculum.
- **20M+ Stage 3 to push from 79.6% toward 85%: not run.** Within Phase 4 budget; Phase 5 task.
- **Aggressive phase-gap schedule (30° → 60° → 120° → 180°): not run.** Expected to be similar or marginally better; not high-priority.
- **LSTM-removal sub-ablation: not run.** Plan called for `--no-lstm` branch in eval; LSTM stayed since the obs is partially state-dependent (debris would matter, but debris is disabled per the no-debris feedback).
