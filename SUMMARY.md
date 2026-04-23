# Orbital RL — Project Summary

*Last updated: 2026-04-21*

A custom PufferLib Ocean environment that trains a PPO agent to perform fuel-efficient orbital maneuvers in 2D Keplerian dynamics. Written in C for performance (~550k SPS on 1024 parallel envs, CPU-only). Target: portfolio piece for space industry roles (APL, Lincoln Lab, Blue Origin).

---

## Project State

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Circular-to-circular transfers | **Complete** — 100% eval success, with and without debris |
| 2 | Elliptical target orbits (match a, e, ω) | **Partially complete** — solved at e ≤ 0.15; ceiling ~45% at e=0.30 |
| 3 | Rendezvous with a propagated target body | Not started |

**Current best policy:** `experiments/puffer_orbital_15nb996c/model_puffer_orbital_000382.pt` (Stage 3a, scaled-tolerance success criterion).

| Target eccentricity | Greedy eval success (50 eps) |
|---------------------|-----------------------------:|
| e_max = 0.10 | 100% |
| e_max = 0.20 |  68% |
| e_max = 0.30 |  40% |

---

## Environment

- **Physics:** 2D Keplerian, two-body, analytic Kepler propagation (no drag, no J2). 60s timestep.
- **Satellite state:** (a, e, M, θ, ω) plus Tsiolkovsky fuel (Isp=300s, 15% fuel fraction → ~480 m/s Δv budget).
- **Observation:** Box(29) — sat elements, target elements, sin/cos of ω for both, 4 nearest bodies (distance, bearing, closing rate, keepout radius).
- **Action:** Discrete(7) — coast, prograde ±10/±25 m/s, radial ±10 m/s.
- **Reward:** fuel-scaled terminal (`+5..+10` on success, −10 on collision/escape/stranded/safety cap) + dense per-step shaping (`c_shape=0.05 × Δ|Δa|/TOL_A + Δ|Δē|/TOL_E`).
- **Success condition:** `|Δa| < 10 km` and `|Δē| < (0.01 + 0.05·e_target)` (scaled tolerance).

---

## Phase 1 — Circular Transfers

### What we did
- Fuel-scaled success reward (`+5..+10` as a function of remaining fuel).
- Trimmed action space from Discrete(9) → Discrete(7), removing the two normal (out-of-plane) burns that were silently ignored in 2D.
- Replaced raw true-anomaly observation (`θ/π`, discontinuous at 0/2π) with sin/cos of θ.

### Result
- **100% eval success** at circular target transfers, ~14% mean fuel usage, ~25 step mean episode length.
- Trajectories show clean two-impulse Hohmann-like burns near apsides.

---

## Phase 2 — Elliptical Targets

### What we did (additive on top of Phase 1)
- Added argument of periapsis ω as a field of `Orbit`; `orbit_to_cartesian` rotates perifocal→inertial; `cartesian_to_elements` recovers ω from the eccentricity vector.
- Randomized target ω; exposed `e_max_target` as a kwarg (curriculum over target eccentricity).
- Success condition now uses `|Δē|` on the eccentricity vector (not scalar |Δe|), giving ω information directly.
- Added sin/cos of sat ω and target ω to observation → Box(29).
- Extended trajectory logging with `sat_omega`, `target_omega`.

### Curriculum & results

Stages trained as warm-start chains from Phase 1 debris checkpoint:

| Stage | e_max_target | Training | Result |
|-------|--------------|----------|--------|
| 1 | 0.03 | warm-start, 50M | 99.8% |
| 2 | 0.10 | warm-start from stage 1 | **Collapsed** to 0% (see below) |
| 3a | curriculum 0.03 → 0.125 with dense shaping | warm-start + `c_shape=0.05` | **100%** at 0.125, also solved stage 2's collapse |
| 3b → 0.20 | warm-start from 3a | | ~68% |
| 3c → 0.30 | warm-start from 3a or 3b | | ceiling ~45% |

### The collapse at stage 2 → fix with dense shaping
- Stage 1 at e=0.03 succeeded by luck: fixed tolerance `SUCCESS_TOL_E=0.01` at target e=0.03 gives an ω window of arcsin(0.01/0.03) ≈ 19°. Phase 1's "burn prograde at apsides" policy randomly lands in that window often enough to bootstrap.
- At e=0.10 the window shrinks to 5.7°. Zero initial successes → value function collapses to uniform −10 → advantages vanish → PPO gradient ≈ 0 → policy drifts to deterministic coast → all 50 eval episodes hit the 2000-step safety cap (reward −10).
- **Fix**: dense shaping `r_step = c_shape × (Δ|Δa|/TOL_A + Δ|Δē|/TOL_E)` with `c_shape=0.05`. Simple difference form (not γΦ'−Φ, to avoid free-reward-for-stuck pathologies). Gives PPO a gradient signal even without terminal successes. Phase 2 Stage 3a then solved with 100% at e=0.125.

### The e=0.30 wall (unsolved)

Four independent attacks; all plateau at ~45%:

| Attempt | Change | Result |
|---------|--------|--------|
| Fresh-at-0.30 | Skip curriculum, train from scratch at hard e | ~40–45% ceiling |
| **d11 action space** | Expand Discrete(7) → Discrete(11) with ±3 m/s fine actions | **6.7% at e=0.10** — broader action space dilutes random-policy productive rate, PPO can't bootstrap. Reverted. |
| **Scaled tolerance** | `tol_e = 0.01 + 0.05·e_target` widens ω window with e | +9pp at direct eval (36→45% at e=0.30) but warm-start fine-tune regresses further (45→31%). Code change kept. |
| **DT=30 timestep** | Halve simulation DT (60s → 30s) for finer timing resolution | Collapsed at BOTH e=0.30 (0%) AND e=0.125 (2.7%). Finer timing doesn't help if episodes double in length — sparse reward compounds. Reverted. |
| **Continuous actions (Box(2))** | Replace Discrete(7) with Box(2) in [-1,1] scaled to ±25 m/s per axis | **24% greedy / 33% stochastic at e=0.10** — entropy collapsed (σ≈0.08) after 15M steps. Same dilution problem as d11, plus continuous entangles magnitude + timing. Reverted. |

### Root cause of the wall
The binding constraint at e=0.30 is **not** action granularity or simulation fidelity. It's **PPO's bootstrap rate** at the intersection of:
- Tight ω tolerance (~5.7° window at e=0.30 even with scaled tol).
- Sparse terminal reward regime (without dense shaping, zero initial successes).
- Warm-start fragility (a policy that solved e=0.15 doesn't transfer gracefully to e=0.30's tighter regime; fine-tuning degrades it).

Anything that widens the action/observation distribution (d11, Box(2), DT=30 doubling episode length) strictly regresses. The productive baseline on this env is Discrete(7) with dense shaping and curriculum warm-start.

---

## Key Learnings

1. **Broader action spaces regress PPO on sparse-reward envs.** Confirmed twice (d11, Box(2)). With Discrete(7), ~4/7 actions are productive → 57% productive-exploration rate. Broadening dilutes this below PPO's bootstrap threshold.
2. **Dense shaping can unstick sparse-reward collapse.** A potential-based per-step reward on (a, ē) proximity gave the agent gradient signal that terminal-only shaping couldn't. Use difference form (Φ_curr − Φ_prev), not γΦ'−Φ, to avoid stuck-far-from-goal earning free reward.
3. **Warm-start fine-tuning between curriculum stages is fragile.** A policy tuned for e_max=0.125 does not cleanly transfer to e_max=0.30 — the new regime's ω tolerance is tight enough that the warm-started policy's lucky alignments no longer succeed, and PPO can degrade rather than improve.
4. **Simulation fidelity is a double-edged sword.** DT=30 looked like it should help (tighter ω windows hit more precisely), but doubling episode length halves terminal-reward density per training step. Finer-grained physics hurt learning.
5. **Debris contaminates early learning.** Debris masks the clean features of optimal transfers (apsidal burns, Hohmann structure). All current training uses `num_debris=0`; debris is deferred indefinitely.

---

## Code Changes Kept

- `orbital.h`: `effective_tol_e = 0.01 + 0.05 * target.e` in success check (physical, +9pp at hardest e).
- `orbital.h`: dense shaping `r_step = 0.05 × (Δ|Δa|/TOL_A + Δ|Δē|/TOL_E)` in `c_step`.
- `orbital.h`: ω as a first-class orbital element, recovered from eccentricity vector.
- `orbital.py`: Box(29) observation, Discrete(7) action space.
- `scripts/orbital/eval_all_ckpts.py`: handles both Discrete and Box action spaces via dispatch on `forward_eval` return type.

### Reverted (known-bad on this env)
- Discrete(11) action expansion.
- Box(2) continuous actions.
- DT=30 timestep.
- Debris in training/eval (indefinite user hold).

---

## What's Next

### Recommended: Phase 3 — Rendezvous
Replace abstract target orbit with a propagated target body. Success = position distance < rendezvous_radius AND relative velocity < tolerance. Dense shaping = distance-closing.

**Why now:**
- Phase 3's tolerance structure (position + velocity, not (a, e, ω)) sidesteps the e=0.30 wall entirely. The unsolved portion of Phase 2 is not on Phase 3's critical path.
- Starting curriculum uses circular target body (e_body=0) with phase gap 0°→30°→90°→180°. The interesting emergent behavior is the *phasing orbit* — drop low, coast, raise back up — independent of target eccentricity.
- Warm-start from the Phase 2 3a checkpoint.

### Alternative / deferred
- **Apsidal-burn shaping** (Option C from prior recommendations). Per-step bonus proportional to `cos(ω_err)` at periapsis/apoapsis to reward burns at the right true anomaly. Directly attacks the e=0.30 wall if Phase 3 reveals the same problem recurs with eccentric target bodies.
- **Re-enable debris** after orbital mechanics are solid. The user's hold is until the agent has clean transfer structure; debris avoidance is v2.
