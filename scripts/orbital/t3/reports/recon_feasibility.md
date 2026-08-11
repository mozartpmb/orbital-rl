All analysis complete. Writing the report.

---

# T3 Recon: Analytic Feasibility Envelope for Rendezvous Under Corrected Physics

**Scripts:** `/Users/pete/space_training/scripts/orbital/t3/feasibility_tables.py`, `ecc_dv_cost.py`, `joint_feasibility.py`, `phasing_expert.py`
**CSVs:** `/Users/pete/space_training/web_data/results/t3_{phasing_table, discount_table, discount_horizons, headline_feasibility, dv_surcharge, ecc_ladder, ecc_dv_cost, ecc_box_relief, joint_feasibility, shaping_gate, shaping_penalty, donothing_floor, velocity_quanta, oracle_driftcheck, phasing_expert}.csv`

---

## 0. Premise correction (load-bearing — everything downstream depends on it)

**`MAX_STEPS = 2000` is a SIM-STEP cap, not an agent-decision cap.** `env->step++` sits *inside* the warp sub-step loop (`orbital.h:1288`), and `check_termination` (which tests `env->step >= MAX_STEPS`) runs on every sub-step. Empirically verified against the built extension:

| repeated action | decisions to `TERM_SAFETY_CAP` |
|---|---|
| 0 coast (τ=1) | 2000 |
| 9 warp-5min (τ=5) | 400 |
| 10 warp-30min (τ=30) | 67 |
| 11 warp-1hr (τ=60) | 34 |

Consequences:
- **Episode wall-clock is fixed at 2000 × 60 s = 33.33 h = 21.6 LEO orbits, regardless of warp usage.** Warps buy *decisions*, not *time*. The task brief's "MAX_STEPS = 2000 agent decisions" is not what the code does.
- The binding constraint on phasing is therefore **sim time**, and no cell in §1 can ever exceed 2000 *decisions* (decisions ≤ sim steps always).
- Warps cannot be used to stall the clock — the −10 timeout still arrives on schedule.
- `check_termination` runs every sub-step, so **warp granularity never causes a missed success**. A warp-1hr flying through the 30 km box still triggers `TERM_SUCCESS`. The only cost of a coarse warp is inability to burn during it.

Two further code facts established by reading `c_reset`:
- **ω is sampled independently ~U(0, 2π) for chaser and target whenever e > 0.** `e_max = 0.05` is therefore not a small perturbation — it randomizes the *eccentricity vector* over a disc. This dominates the Δv budget (§3).
- The safety-cap branch pays no Φ-clamp but the **per-step shaping still leaks** (the code's own KNOWN ISSUE comment). Quantified in §2.4.

Oracle cross-check of the drift formula (`orbital_math.propagate_cartesian`, exact f&g): linearized `1.5(da/a)n` over-predicts by 0.9 % at da = 50 km, 3.6 % at 200 km, 5.9 % at 340 km. **All tables below use the exact `|n(a) − n(a+da)|`.**

---

## 1. Phasing time / steps / decisions / Δv (a = 6771 km, T = 92.41 min, v_c = 7672.6 m/s, Δv budget = 478.1 m/s)

| gap° | da km | drift °/hr | drift hr | **sim steps** | Δv open+close | Δv/budget | dec: coast | dec: w5 | dec: w5+30 | dec: w5+30+60 |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 50 | 2.57 | 11.7 | 795 | 57 | 0.12 | 799 | 167 | 37 | 25 |
| 30 | 100 | 5.08 | 5.9 | 449 | 113 | 0.24 | 454 | 102 | 32 | 26 |
| 30 | 200 | 9.99 | 3.0 | 277 | 227 | 0.47 | 287 | 67 | 22 | 18 |
| 30 | 340 | 16.56 | 1.8 | 208 | 385 | 0.81 | 224 | 64 | 34 | 32 |
| 90 | 50 | 2.57 | 35.1 | **2198 ⚠** | 57 | 0.12 | 2202 | 446 | 81 | 45 |
| 90 | 100 | 5.08 | 17.7 | 1157 | 113 | 0.24 | 1162 | 242 | 52 | 34 |
| 90 | 200 | 9.99 | 9.0 | 637 | 227 | 0.47 | 648 | 140 | 35 | 25 |
| 90 | 340 | 16.56 | 5.4 | 426 | 385 | 0.81 | 441 | 105 | 40 | 34 |
| 180 | 50 | 2.57 | 70.2 | **4304 ⚠** | 57 | 0.12 | 4307 | 867 | 152 | 81 |
| 180 | 100 | 5.08 | 35.4 | **2219 ⚠** | 113 | 0.24 | 2224 | 456 | 91 | 55 |
| 180 | 200 | 9.99 | 18.0 | 1178 | 227 | 0.47 | 1188 | 248 | 53 | 34 |
| 180 | 340 | 16.56 | 10.9 | 752 | 385 | 0.81 | 767 | 171 | 51 | 40 |

⚠ = exceeds the 2000 sim-step horizon. **No cell exceeds the Δv budget** (worst is 81 % at da = 340 km). **No cell exceeds 2000 decisions except coast-only** at 90°/50 km and 180°/{50,100} km — and those are already horizon-infeasible.

Decision counts use a greedy largest-warp-that-does-not-overshoot mix ending in τ=1 coast steps; the mix is lumpy (the remainder after the last 60/30/5 division is spent one step at a time), which is why 30°/340 km costs more decisions than 30°/200 km.

**Minimum |da| to close 180° inside the horizon** (this is the sharp constraint):

| horizon (sim steps) | min \|da\| | Δv round trip | % budget |
|---|---|---|---|
| 1000 | 217.0 km | 245.8 m/s | 51 % |
| 1500 | 142.7 km | 161.7 m/s | 34 % |
| **2000 (shipped)** | **106.3 km** | **120.5 m/s** | **25 %** |
| 3000 | 70.4 km | 79.8 m/s | 17 % |
| 4000 | 52.7 km | 59.7 m/s | 12 % |

So 180° phasing at LEO costs **~25 % of the Δv budget and the entire episode** at the shipped horizon. It is feasible, but with almost no margin for a suboptimal policy.

**Free phasing from the initial mismatch.** In the headline distribution a_sat and a_target are drawn independently from a 500 km band, so |Δa|_init averages 167 km — a drift orbit the agent already owns for free. The optimal plan is *loiter on the initial orbit until the transfer window, then one Hohmann*, whose Δv is exactly the unavoidable orbit-match cost. MC over 200 k headline samples (`t3_headline_feasibility.csv`):

| horizon | feasible frac (zero Δv surcharge) | p50 steps | p90 | p99 | Δv mean |
|---|---|---|---|---|---|
| 2000 | **61.6 %** | 1440 | 8590 | 85 700 | 91.2 m/s |
| 4000 | 79.4 % | — | — | — | — |
| 6000 | 85.9 % | — | — | — | — |

The 38 % tail is the near-degenerate case a_sat ≈ a_target (drift rate → 0). Buying a phasing orbit rescues **100 % of it** at a mean surcharge of 57 m/s (p90 102 m/s) — `t3_dv_surcharge.csv`. **Nothing in the headline distribution is physically infeasible.**

**Capture window** — with the 50 m/s box allowing residual |Δa| up to 88.2 km, the agent should deliberately *not* fully circularize:

| residual da | \|v_rel\| | along-track closure | dwell in 30 km box |
|---|---|---|---|
| 20 km | 11.3 m/s | 2.03 km/step | 29.5 sim steps |
| 40 km | 22.7 m/s | 4.05 km/step | 14.8 sim steps |
| 60 km | 34.0 m/s | 6.05 km/step | 9.9 sim steps |
| 88 km | 49.9 m/s | 8.83 km/step | 6.8 sim steps |

Terminal timing precision required is ~1 sim step, available in every action set. Combined with per-sub-step termination checking, **the endgame is not the hard part.**

---

## 2. Discount visibility

`orbital.ini [train]`: **`gamma = 0.995`**, `total_timesteps = 50M`, `learning_rate = 0.01`, `minibatch_size = 8192`, `ent_coef = 0.01` (fixed, no schedule; `target_entropy_controller = False`, `ent_coef_mid/high` present but inert), `anneal_lr = True`, `clip 0.2/0.2`, `kl_target = 0.0`, `l2_init_coef = 0.0`. **`bptt_horizon` is not overridden → default 64** (`config/default.ini:52`), with `gae_lambda = 0.90`, `vf_coef = 2.0`, `update_epochs = 1`, `max_minibatch_size = 32768`. `[env] num_envs = 1024`, `init_phase_gap_max = 0.524` (30°) — the shipped training config is Stage-1, not the headline 180°.

### 2.1 Horizons

| γ | N at 50 % | **N at 10 %** | **N at 1 %** | 1/(1−γ) |
|---|---|---|---|---|
| 0.995 | 138 | **459** | **919** | 200 |
| 0.997 | 231 | 766 | 1531 | 333 |
| 0.999 | 693 | 2303 | 4605 | 1000 |

### 2.2 Terminal value 10·γ^N for the 180° mission

| γ | da | action set | N | 10·γ^N | % |
|---|---|---|---|---|---|
| 0.995 | 200 km | coast only | 1188 | 0.026 | **0.26 %** |
| 0.995 | 200 km | legacy D10 (warp-5min) | 248 | 2.885 | 28.9 % |
| 0.995 | 200 km | D16 (w5+30) | 53 | 7.667 | 76.7 % |
| 0.995 | 200 km | **D16 (w5+30+60)** | **34** | **8.433** | **84.3 %** |
| 0.995 | 340 km | coast only | 767 | 0.214 | 2.1 % |
| 0.995 | 340 km | legacy D10 | 171 | 4.244 | 42.4 % |
| 0.997 | 200 km | coast only | 1188 | 0.282 | 2.8 % |
| 0.997 | 200 km | D16 (w5+30+60) | 34 | 9.029 | 90.3 % |
| 0.999 | 200 km | coast only | 1188 | 3.046 | 30.5 % |
| 0.999 | 200 km | D16 (w5+30+60) | 34 | 9.666 | 96.7 % |

### 2.3 Verdict on the legacy action space

**Strictly by the discount criterion, legacy D10 is *not* discount-starved at 180°** — warp-5min puts N at 248, giving 28.9 % visibility, comfortably above the 10 % line (N = 459). What *is* discount-starved is the **coast-only regime** (0.26 %), and that is where a policy without a warp habit actually lives.

The real barrier is **exploration length, not discount**. The phasing maneuver requires an uninterrupted run of non-burn decisions:

| action set | N non-burn decisions | P(non-burn) uniform | P(run) |
|---|---|---|---|
| coast only | 1178 | 1/9 | 10⁻¹¹²⁴ |
| legacy D10 | 248 | 2/10 | 10⁻¹⁷³ |
| **D16 w5+30+60** | **34** | **4/16** | **10⁻²¹** |

Discrete(16) with warp-1hr improves the exploration exponent by **~150 orders of magnitude** and lifts terminal visibility from 29 % to 84 %. It is **necessary but not sufficient** — 10⁻²¹ is still unreachable from uniform random, so the shaping must supply the gradient (§2.5).

### 2.4 Per-decision vs semi-MDP (γ^τ) discounting — state the convention explicitly

The env's shaping uses **γ^τ** (`orbital.h:1336`); the trainer's GAE uses **flat γ per decision**. They disagree, and the disagreement matters:

- **Flat per-decision γ (trainer, current).** A warp-1hr and a coast are discounted identically. A warp advances 60× more sim time for the same discount, so **warps are strictly attractive** — this is precisely the property that makes 180° learnable (84 % vs 0.26 %). The safety cap being in sim steps means this cannot be exploited to stall.
- **Semi-MDP γ^τ in GAE.** Warps become *neutral* (equivalent per unit sim time), not punitive. But then 10·γ^S = **0.027 regardless of action set** at 180°/200 km — every action set becomes discount-starved and the task is unlearnable.

**Recommendation: keep flat per-decision γ in GAE. Do NOT "fix" the trainer to semi-MDP.** Instead change the *env* to match: use γ¹ in the shaping term rather than γ^τ. Two reasons:

1. **PBRS validity.** Potential-based shaping is policy-invariant only when the shaping discount equals the MDP discount. With shaping γ^τ and GAE γ¹, the shaping is *not* potential-based for τ > 1 and injects a bias.
2. **It removes a warp reward leak.** With Φ constant, each decision pays β·|Φ|·(1 − γ^τ) > 0 (Φ ≤ 0, so γ^τΦ′ is *less* negative than Φ). At τ = 60 that is **+0.0434 per decision of free reward for doing nothing**, vs **+0.00084** under γ¹ — a **52× reduction**.

Do-nothing shaping floor over a full timeout episode (γ = 0.995, warp-1hr, 33 decisions):

| \|Δa\| scale | \|Φ\| | undiscounted leak | timeout return (−10 + leak) | flips positive? |
|---|---|---|---|---|
| 50 km | 0.05 | 0.43 | −9.57 | no |
| 167 km (headline mean) | 0.167 | 1.43 | −8.57 | no |
| 500 km | 0.5 | 4.29 | −5.71 | no |
| 1000 km | 1.0 | 8.57 | −1.43 | no |
| 5000 km | 5.0 | 42.9 | **+32.9** | **YES** |

At LEO the leak is a 14 % softening of the timeout penalty — tolerable but not free. **Beyond ~1000 km of Δa it dominates and beyond ~1200 km it flips the timeout positive.** This is the same mechanism as the previously-documented Φ-clamp leak, surviving in the per-step term. Any wide-band rung (§3) must fix this before training.

### 2.5 The shaping is structurally anti-phasing under correct physics (the central finding)

σ₂ = sigmoid((2 − φ_orbit)/0.2), φ_orbit = |Δa|/10 km:

| \|Δa\| | φ_orbit | σ₂ | implied \|v_rel\| | drift °/hr | status |
|---|---|---|---|---|---|
| 10 km | 1.0 | 0.993 | 5.7 | 0.52 | gate OPEN |
| 20 km | 2.0 | 0.500 | 11.3 | 1.03 | half open |
| 30 km | 3.0 | 6.7e−3 | 17.0 | 1.54 | closed |
| 50 km | 5.0 | 3.1e−7 | 28.3 | 2.57 | **DEAD** |
| **88 km (50 m/s box edge)** | 8.8 | 1.7e−15 | 49.9 | 4.48 | **DEAD** |
| **200 km (drift orbit)** | 20 | 8.2e−40 | 113 | 9.99 | **DEAD** |
| 340 km | 34 | 3.3e−70 | 193 | 16.6 | **DEAD** |

**The phase-reward gate opens only for |Δa| < ~29 km. Every drift orbit capable of closing 180° inside the horizon requires |Δa| ≥ 106 km. σ₂ is therefore identically zero (≤ 10⁻¹⁷) for 100 % of the phasing maneuver.** Even the success box itself tolerates |Δa| up to 88 km, where the gate is already dead — so the gate is mis-keyed against the env's own terminal criterion by a factor of ~9.

Worse, the gate creates a **strict local optimum that is a guaranteed failure state**:

- Φ(Δa = 200 km, Δθ = 180°) = −0.01·20 − 0 = **−0.20**
- Φ(Δa = 0, Δθ = 180°) = −0 − 0.01·2·0.99995 = **−0.02**

Circularizing onto the target orbit pays **+0.18 of shaping immediately**. But at Δa = 0 the relative drift rate is zero, so Δθ is **frozen forever**, distance is frozen at a·Δθ, Φ is frozen, and the episode is a guaranteed −10 timeout with no remaining gradient. **The shaping's greedy gradient points directly into an absorbing failure state.** Conversely, opening the drift orbit costs −0.18 up front and refunds only γ^N later:

| action set (180°, da = 200 km, γ = 0.995) | N | γ^N | net entry cost | % of terminal |
|---|---|---|---|---|
| coast only | 1091 | 0.0042 | −0.199 | 1.99 % |
| legacy D10 | 227 | 0.321 | −0.136 | 1.36 % |
| D16 (w5+30) | 47 | 0.790 | −0.042 | 0.42 % |
| **D16 (w5+30+60)** | **29** | **0.865** | **−0.027** | **0.27 %** |

Warp-1hr shrinks the un-refunded penalty **7.4×** — another independent argument for it — but does not remove the local optimum.

**Root cause:** the stage ordering *orbit → phase → velocity* was correct under the **bugged** dynamics, where `true_to_mean` inversion let a burn teleport phase at Δa ≈ 0 (gate open). Under correct physics the causality is reversed: **phase is only controllable while the orbit is deliberately mismatched.** The shaping and the bug were co-designed; removing the bug leaves the shaping pointed backwards. Recommended minimum fix: **ungate φ_phase (or re-key EPS_ORBIT to the rel-vel box, |Δa| ≈ 88 km ⇒ EPS_ORBIT ≈ 8.8 rather than 2.0)** so phase closure earns reward *during* the drift.

---

## 3. Eccentricity extension

### 3.1 Geometry (perigee must clear R_EARTH + 200 km)

| band (km alt) | e_max at a_hi | e_max valid ∀a in band | e_max both apsides in band | T at a_hi | sim steps/orbit | orbits in 2000 steps |
|---|---|---|---|---|---|---|
| 300–800 | **0.0837** | 0.0150 | 0.0361 | 101 min | 101 | 19.9 |
| 300–2000 | 0.215 | 0.0150 | 0.113 | 127 min | 127 | 15.7 |
| 300–8000 | 0.543 | 0.0150 | 0.366 | 286 min | 286 | 7.0 |
| 300–20200 (MEO) | 0.753 | 0.0150 | 0.599 | 718 min | 718 | 2.78 |
| 300–35786 (GEO) | 0.844 | 0.0150 | 0.727 | 1436 min | 1436 | 1.39 |

Confirms the 0.084 LEO ceiling. **But geometry is not the binding constraint.**

### 3.2 Δv is the binding constraint, and ω-randomization is why

Matching eccentricity vectors costs Δv_e ≈ v_c·|Δē|/2. Since ω ~ U(0,2π) independently, |Δē| is large even at modest e. MC (`t3_ecc_dv_cost.csv`, 200 k samples, budget 478.1 m/s):

| band | e_max | Δv_a mean | Δv_e mean | Δv p50 | Δv p90 | % ≤ budget | % ≤ 75 % budget |
|---|---|---|---|---|---|---|---|
| 300–800 | 0.000 | 91 | 0 | 80 | 187 | 100.0 | 100.0 |
| 300–800 | 0.025 | 91 | 69 | 152 | 264 | 100.0 | 99.5 |
| **300–800 | 0.050 (headline)** | 91 | 138 | 222 | 359 | **99.1** | 89.9 |
| 300–2000 | 0.050 | 275 | 132 | 377 | 708 | 65.5 | 46.8 |
| 300–2000 | 0.084 | 275 | 222 | 473 | 816 | 50.9 | 30.8 |
| 300–2000 | 0.150 | 275 | 396 | 651 | 1055 | **26.8** | 13.8 |
| 300–8000 | 0.300 | 783 | 676 | 1384 | 2405 | **5.2** | 2.5 |
| 300–20200 | 0.500 | 1107 | 910 | 1881 | 3440 | **2.8** | 1.3 |

**Widening the altitude band to unlock e simultaneously makes the a-transfer unaffordable** (Δv_a = 783 m/s at 300–8000 vs a 478 m/s budget). And **independent ω makes e itself unaffordable** — at e = 0.15 the e-vector match alone averages 396 m/s.

### 3.3 The 30 km / 50 m/s box does NOT rescue you

Two coplanar ellipses with the same a but different e-vectors always intersect, so one might rendezvous by timing alone. Measured |v_rel| at the crossing (`t3_ecc_box_relief.csv`):

| e_max | crossing exists | \|v_rel\| p10/p50/p90 | **% < 50 m/s** | % < 1 m/s |
|---|---|---|---|---|
| 0.010 | 100 % | 19.4 / 53.6 / 95.3 | 45.6 % | 0.4 % |
| 0.025 | 100 % | 48.4 / 134.0 / 238.3 | 10.6 % | — |
| 0.050 | 100 % | 96.8 / 268.1 / 476.9 | **3.5 %** | — |
| 0.084 | 100 % | 162.7 / 450.5 / 802.4 | 1.5 % | — |

At the headline e = 0.05, only 3.5 % of pairs allow a burn-free intersection rendezvous. **The e-vector match must actually be paid.**

### 3.4 The fix: bound |Δē|, not e

Sample the target's e-*vector* within `de_max` of the chaser's, so both orbits can be strongly eccentric while the matching maneuver stays affordable. Joint MC (Δv *and* horizon simultaneously, phasing surcharge included — `t3_joint_feasibility.csv`, n = 3000):

| band | e_max | de_max | ff | da_max | horizon | % Δv ok | **% FEASIBLE** | Δv p50 | Δv p90 | steps p50 | steps p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300–800 | 0.05 | — | 0.15 | — | 2000 | 99.0 | **99.0** | 242 | 373 | 1229 | 1884 |
| 300–800 | 0.00 | — | 0.15 | — | 2000 | 100.0 | **100.0** | 110 | 187 | 1229 | 1884 |
| 300–800 | 0.084 | — | 0.15 | — | 3000 | 85.2 | 85.2 | 319 | 516 | 1312 | 2673 |
| 300–2000 | 0.15 | — | 0.15 | 400 | 3000 | 50.2 | 50.2 | 477 | 787 | 1414 | 2728 |
| 300–8000 | 0.30 | — | 0.15 | 600 | 6000 | 24.4 | 24.4 | 731 | 1297 | 2073 | 5477 |
| 300–20200 | 0.50 | — | 0.15 | 1000 | 12000 | 16.6 | 16.6 | 941 | 1782 | 3712 | 11443 |
| **300–2000** | **0.15** | **0.06** | 0.15 | 400 | 3000 | 100.0 | **100.0** | 249 | 338 | 1372 | 2707 |
| **300–8000** | **0.30** | **0.08** | 0.15 | 600 | 6000 | 98.5 | **98.5** | 259 | 387 | 1996 | 5416 |
| **300–20200** | **0.50** | **0.10** | 0.15 | 1000 | 12000 | 93.0 | **93.0** | 247 | 439 | 3503 | 11311 |
| 300–2000 | 0.15 | — | **0.30** | 400 | 3000 | 98.9 | 98.9 | 477 | 787 | 1414 | 2728 |
| 300–8000 | 0.30 | — | **0.40** | 600 | 6000 | 95.1 | 95.1 | 731 | 1297 | 2073 | 5477 |

**The headline task (LEO 300–800, e ≤ 0.05, ±180°) is 99.0 % feasible within 2000 sim steps and 478 m/s.** Post-fix training failure is a *learning* problem, not a physics problem.

Two independent levers unlock wide e:
- **`de_max` (preferred, free):** bound the e-vector mismatch. Takes MEO/e=0.5 from 16.6 % → 93.0 % feasible with the existing 15 % fuel fraction, and holds Δv p50 at ~250 m/s across all three rungs. Preserves the interesting skill (operating on eccentric orbits: varying r and v, apsidal timing) without demanding an unaffordable e-vector rotation.
- **Raise `FUEL_FRAC`** (0.15 → 0.30/0.40, i.e. 1051/1503 m/s) if fully independent e-vectors are wanted. Realistic for an upper stage but changes the mission's character.

Also required: **bound |a_sat − a_target| (`da_max`) independently of the altitude band.** `same_orbit_init=1` is not a substitute — it also forces identical e *and* ω.

### 3.5 Horizon and warp scaling

Scale-free relations (verify against the table above):
- orbits to phase: **N_orb = Δθ / (3π·f)**, f = da/a
- Δv round trip: **Δv = v_c(a)·f**
- affordable f: **f_max = Δv_avail / v_c(a)** — grows with altitude as v_c falls

| band | v_c at a_hi | affordable da (60 % budget) | sim steps to phase 180° | required MAX_STEPS | dec (w5) | dec (w5+30+60) |
|---|---|---|---|---|---|---|
| LEO-narrow | 7460 | 276 km | 873 | 1500 | 177 | 18 |
| LEO-wide | 6857 | 348 km | 1022 | 2000 | 207 | 26 |
| LEO/MEO | 5266 | 783 km | 1751 | 3000 | 353 | 34 |
| MEO | 3861 | 1970 km | 3230 | 5500 | 649 | 61 |
| MEO/GEO | 3075 | 3930 km | 5130 | 8500 | 1026 | 86 |

**MAX_STEPS must scale with the orbital period** (∝ a^{3/2}); phasing is *cheap in orbits* at high altitude (2.1 orbits at GEO vs 11 at LEO, because f_max is larger) but *expensive in wall time*.

**Warp-set sizing rule.** A uniform-random Discrete(16) policy has mean τ = (13·1 + 5 + 30 + 60)/16 = **6.75**, so a random episode is MAX_STEPS/6.75 decisions. Keeping that inside the γ = 0.995 10 % horizon (N ≤ 459) requires **mean τ ≳ MAX_STEPS/450**:

| MAX_STEPS | mean τ needed | warp set | achieved mean τ | random-policy decisions |
|---|---|---|---|---|
| 3000 | 6.7 | {5, 30, 60} (current D16) | 6.75 | 444 ✓ |
| 6000 | 13.3 | {5, 30, 60, 180} | 16.9 | 355 ✓ |
| 12000 | 26.7 | {5, 30, 60, 180, 360} | 36.0 | 333 ✓ |

Also: `bptt_horizon = 64` covers **all 34–55 decisions** of an optimally-warped LEO 180° mission, but only ~5 % of a 1178-decision coast-only one. Warps fix the LSTM credit window too.

---

## 4. Success-box velocity quanta

Greedy nulling with quantum q on an axis leaves |residual| ≤ q/2.

| action set | min tangential | min radial | best \|v_rel\| | 30 km/50 m/s box | 5 km/1 m/s box | margin (50 m/s) | Δa quantum |
|---|---|---|---|---|---|---|---|
| legacy D10 | 5 m/s | **10 m/s** | 5.59 m/s | ✅ 8.9× margin | ❌ | 8.9× | 8.82 km |
| **D16 (shipped)** | 1 m/s | **10 m/s** | 5.02 m/s | ✅ 10.0× margin | ❌ | 10.0× | 1.76 km |
| D16 + radial ±1 (proposed) | 1 m/s | 1 m/s | 0.71 m/s | ✅ 70.7× | ✅ 1.4× | 70.7× | 1.76 km |

**Findings:**
- The 30 km / 50 m/s box is **generous for both action sets** — ~9–10× margin. It is not the reason the task is hard.
- Discrete(16) added fine *tangential* burns (1, 2 m/s) but **no fine radial burn**. The radial quantum is still 10 m/s in both sets, so it is the binding constraint for both. D16's improvement over D10 on |v_rel| is only 5.59 → 5.02 m/s.
- **The 5 km / 1 m/s tight box is unreachable with either shipped action set.** Single-step nulling floors at 5.02 m/s. It requires adding radial ±1 m/s (→ Discrete(18)), which achieves 0.71 m/s. *Caveat:* the quantum bound is on single-step nulling — because the LVLH frame rotates, "vernier pairs" of 10 m/s radial burns at different anomalies can synthesize sub-m/s corrections, but at ~20 m/s of budget per sub-m/s correction. Noting the constraint per instruction; not designing for it.
- Δa side: the 50 m/s box tolerates |Δa| ≤ 88.2 km (10× the 8.82 km legacy quantum); the 1 m/s box tolerates only 1.76 km, exactly D16's 1 m/s tangential quantum — marginal.
- Position side is never binding: a 5 km box at 1 m/s rel-vel has a 167 sim-step dwell.

---

## 5. Expert-baseline attempt — INCONCLUSIVE, do not cite

`scripts/orbital/t3/phasing_expert.py` implements a closed-loop TRIM→DRIFT→CLOSE→TERMINAL controller against the real C env (state via `orbital_math.recover_states`). After fixing an inverted drift-direction sign it flies real trajectories (median 850–1800 sim steps, 38–300 decisions) but **converges to 300–500 km / 300–450 m/s, never entering the box; 0/8 on every scenario**, exhausting the Δv budget. This is a controller-tuning failure, not evidence about the env. A sibling T3 agent owns `scripts/orbital/t3/expert_controller.py`; defer to that.

**One reusable trap discovered:** the env auto-resets on terminal, so the `obs` returned by the terminating `env.step()` belongs to the **next** episode. Any harness computing final distance/rel-velocity from the post-step obs gets garbage (we observed 13 347 km / 15 433 m/s). Classify on `last_terminal_cause`, and cache geometry from the step *before* terminal.

---

## 6. Recommended configuration

### Single recommended config (immediate — the 180° LEO retry)

| knob | value | justification |
|---|---|---|
| **Action set** | **Discrete(16) as shipped** (keep warp-30min and warp-1hr) | 84.3 % terminal visibility vs 28.9 % (D10) and 0.26 % (coast); exploration exponent improves ~150 orders of magnitude; shrinks the drift-entry shaping penalty 7.4×; brings the mission inside `bptt_horizon = 64`. Do **not** train with `legacy_action_space = 10`. |
| **γ (trainer GAE)** | **0.995, flat per-decision — unchanged** | Flat-γ is what makes warps attractive and 180° learnable. Semi-MDP γ^τ in GAE collapses every action set to 0.027 terminal value. N = 34 ≪ N₁₀% = 459, so there is no need for 0.997/0.999. |
| **γ (env shaping)** | **change γ^τ → γ¹** (`orbital.h:1336`) | Restores PBRS policy-invariance (currently broken for τ > 1) and cuts the do-nothing warp leak 52× (0.0434 → 0.00084 per decision). One-line change. |
| **Shaping structure** | **ungate φ_phase**, or at minimum re-key `EPS_ORBIT` 2.0 → 8.8 (tie φ_orbit's tolerance to the rel-vel box, 2·a·v_tol/v_c = 88 km, not `SUCCESS_TOL_A` = 10 km) | σ₂ ≤ 10⁻¹⁷ for 100 % of any viable drift orbit; the gate makes "circularize and coast" a strict local optimum that is an absorbing failure state (Φ: −0.20 → −0.02 for free, then Δθ frozen forever). The orbit→phase→velocity ordering is a fossil of the `true_to_mean` bug. |
| **MAX_STEPS** | **3000** (from 2000) | 2000 leaves the p90 headline mission at 1884 sim steps — 6 % margin, no slack for a suboptimal policy. 3000 costs nothing (it is a *time* cap; warps keep decisions at ~50) and lifts free-loiter feasibility 61.6 % → ~72 %, joint feasibility 99.0 % → 99.5 %. |
| **Success box** | 30 km / 50 m/s unchanged | 10× quantum margin; not the bottleneck. |
| **`ent_coef`** | 0.01 unchanged | With P(non-burn) = 4/16, entropy must not be so high that 34-decision warp runs are broken up. |

### Two new env kwargs required for the ladder

1. **`da_max_m`** — bound |a_sat − a_target| independently of the altitude band. Without it, widening the band to unlock e makes the transfer Δv-infeasible (300–8000 km: Δv_a = 783 m/s vs a 478 m/s budget).
2. **`de_max`** — sample the target's eccentricity *vector* within `de_max` of the chaser's, instead of drawing e and ω independently. This is the single highest-leverage change for the "wide e" goal: MEO/e = 0.5 goes 16.6 % → 93.0 % feasible at the existing fuel fraction.

### Curriculum ladder: (alt band, e_max, de_max, da_max, warp set, MAX_STEPS, γ)

| rung | alt band | e_max | de_max | da_max | warp set | MAX_STEPS | γ | phase gap | feasible | opt. decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| **L0** | 300–800 | 0.0 | — | — | {5,30,60} | 3000 | 0.995 | ±30° | 100 % | ~25 |
| **L1** | 300–800 | 0.0 | — | — | {5,30,60} | 3000 | 0.995 | ±180° | 100 % | ~34 |
| **L2** | 300–800 | 0.05 | 0.05 | — | {5,30,60} | 3000 | 0.995 | ±180° | 99.0 % | ~50 |
| **L3** | 300–2000 | 0.15 | 0.06 | 400 km | {5,30,60} | 3000 | 0.995 | ±180° | **100 %** | ~60 |
| **L4** | 300–8000 | 0.30 | 0.08 | 600 km | {5,30,60,**180**} | 6000 | 0.995 | ±180° | **98.5 %** | ~90 |
| **L5** | 300–20200 | 0.50 | 0.10 | 1000 km | {5,30,60,180,**360**} | 12000 | 0.995 | ±180° | **93.0 %** | ~130 |

Ladder rationale: **eccentricity is unlocked by climbing in altitude** (perigee clearance gives e_max 0.084 → 0.75, and falling v_c makes e-vector matching cheaper: Δv_e for |Δē| = 0.1 is 384 m/s at LEO but 154 m/s at MEO), while **|Δa| and |Δē| are bounded to keep Δv affordable** and **MAX_STEPS + the warp set scale with the orbital period** to keep both the optimal mission (≤ 130 decisions) and a random-policy episode (≈ 330–440 decisions) inside the γ = 0.995 10 % horizon (N = 459). L4/L5 also require the §2.4 leak fix first — at |Δa| ~ 1000 km the do-nothing shaping floor reaches +8.6, nearly cancelling the −10 timeout.

**Gate between rungs on multi-seed success ≥ 60 %, not on a single seed** (per the Phase 4 R0 finding that single-seed peaks are not robust).

### Bottom line

Under corrected physics the headline task is **99.0 % feasible** within the shipped horizon and Δv budget — the physics is not the blocker. The blockers, in order of severity:

1. **The shaping gate σ₂ is dead for 100 % of any viable phasing maneuver and makes "circularize and coast" a locally-optimal absorbing failure state.** This is a direct fossil of the `true_to_mean` bug and is the leading explanation for post-fix flatlining.
2. **Exploration length**: 34 consecutive non-burn decisions (D16 + warp-1hr) is reachable only with a shaping gradient that rewards phase closure *during* the drift — which fix 1 supplies.
3. **γ^τ/γ¹ mismatch** between env shaping and trainer GAE: breaks PBRS validity and pays a 52×-inflated do-nothing warp bonus.
4. **MAX_STEPS = 2000** leaves only 6 % margin at p90; 3000 is nearly free.
5. **Wide e is Δv-gated, not geometry-gated**, because ω is randomized independently — fixed by a `de_max` kwarg, not by widening bands.