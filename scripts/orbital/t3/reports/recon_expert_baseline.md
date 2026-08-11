# T3 — Scripted Classical-GNC Expert: the task IS feasible in the discrete action space

**Verdict: 99.2% success (496/500) at headline conditions under corrected dynamics, obs-only, using nothing but the env's 16 discrete actions, at 40% of the fuel budget (median).** Feasibility is not the constraint. Every downstream RL failure is a credit-assignment/exploration problem, not a reachability problem.

---

## 1. Deliverables

| Path | Contents |
|---|---|
| `/Users/pete/space_training/scripts/orbital/t3/expert_controller.py` | The expert + eval harness. Re-runnable, seeded, no env vars needed. `--suite`, `--episodes N --seed S`, `--same-orbit`, `--verbose`, `--trace`, `--csv`. |
| `/Users/pete/space_training/scripts/orbital/t3/t3_expert_model.py` | Exact Python replica of `orbital.h` dynamics + obs decode (private copy; `orb_model.py` is a duplicate other lanes also touch). |
| `/Users/pete/space_training/scripts/orbital/t3/probe_model_match.py` | Validation of the replica against the C env (P1–P4 below). |
| `/Users/pete/space_training/web_data/results/expert_baseline.csv` | 1000 per-episode rows across 9 conditions. Columns: `condition, seed, episode, gap_deg, gapM_deg, e_sat, e_tgt, alt_sat_km, alt_tgt_km, da_km, success, cause, steps, decisions, dv_used, plan_dv, n_plans, n_burns, n_warps, n_coast, min_d_km, hohmann_dv, evec_dv, shape_lb_dv`. |

Nothing committed. Nothing outside `scripts/orbital/t3/` and `web_data/results/` modified.

---

## 2. State source: observations only

`decode_obs()` inverts `fill_observations()` exactly — chaser `(a,e,θ,ω)` from `obs[0,1,2/3,9/10]`, target from `obs[7,8,15/16,11/12]`, fuel from `obs[6]`. **No ground-truth accessor is used by the controller.** `last_episode_result()` is read only after termination, for scoring.

Replica validated against the C env (`probe_model_match.py`):

| Check | Result |
|---|---|
| P1 coast/warp propagation, 2016 sim steps | max chaser position error **0.432 m** (float32-obs floor) |
| P2 every burn action (1–8, 12–15), 120 burns at random true anomalies | max post-burn+propagate error **0.517 m**, 5.4e-4 m/s; fuel matches Tsiolkovsky to <2e-6 mass fraction |
| P3 drift sign, chaser 197.3 km **below** target | measured **+10.2239 °/hr** vs analytic `n_s − n_t` = **+10.2240 °/hr** (chaser below ⇒ drifts *ahead*) |
| P4 Δv budget from full tank | **478.13 m/s** |

---

## 3. Controller (state machine)

Two nested loops. Both classical, no learned components, no tuned gates in the inner loop.

**Inner — matched-orbit targeting (every decision).**
Error metric in Δv units:
```
X = ( A, Ex, Ey ) = ( n·(a − a_g)/2 , v·(ē − ē_g)/2 )        [m/s]
V(X) = max(|A|, |E|)
```
A prograde burn `δv` at inertial position angle φ moves `X` by `δv·(1, cos φ, sin φ)`; an antipodal burn pair therefore nulls `X` for exactly `|δv₁|+|δv₂| = max(|A|,|E|)`, so **V is the exact minimum two-impulse Δv cost-to-go**. Control law = one-step lookahead on that value: evaluate all 12 burn quanta by *exact* impulse prediction, pick the one maximising `ΔV − |Δv|·(1−slack)`, `slack = 0.12`. This is a certainty-equivalent optimal-cost policy, and it reproduces the Hohmann split with no tuning: burn 1 stops exactly where the induced `|E|` would exceed the remaining `|A|`; a pure eccentricity-vector error makes it wait for the correct apsis. Post-burn perigee is rejected below 200 km altitude.

**Outer — along-track phase tracking (every decision).**
`Δλ = λ_sat − λ_tgt` (mean longitudes) drifts at `n(a) − n(a_t)`. First-order receding-horizon law:
```
da_cmd = −Δ · a_t / (1.5 · n_t · T_ctrl) ,   T_ctrl = K · t_remaining,  K = 0.65
a_g    = a_t + da_cmd     (clamped: |da| ∈ [3 km, 340 km], perigee ≥ 200 km alt,
                           round-trip Δv ≤ 0.75 · Δv_remaining, 6 km deadband)
```
`Δ` is the signed phase still to close. `PLAN` chooses short-way vs long-way-round by Δv cost; `_track_goal` re-checks direction feasibility every step and flips to the long way when the perigee floor forbids the required drift. `|da_cmd|` is **floored at 3 km rather than driven to zero** — a 3 km radial offset is well inside the 30 km box, so the residual drift itself sweeps Δλ through zero and the env samples the box for thousands of seconds.

**HOLD.** A forward scan of the exact model finds the closest approach; if it clears 26 km / 42 m/s the controller stops burning and warps to it, re-scanning every 6 decisions.

**Warps.** Largest of {1 hr, 30 min, 5 min} that does not overshoot the next favourable burn phase (or 300 s when |Δλ| < 14°). Warps are safe: the C env runs `check_termination` every sub-step.

---

## 4. Results

### 4.1 Headline (LEO 300–800 km, `e_max_target=e_max_sat=0.05`, gap ±180°, `valid_init_only=1`, full 16-action env)

| seed | episodes | success |
|---|---|---|
| 42 | 200 | **99.0%** (198) |
| 7 | 100 | 99.0% |
| 1337 | 100 | 100.0% |
| 20260810 | 100 | 99.0% |
| **pooled** | **500** | **99.2% (496/500)** |

`same_orbit_init=1`, seed 42, 100 eps: **99.0%**.

**Terminal-cause histogram (500 headline):** `success 496, safety_cap 4`. Zero collisions, zero escapes, zero stranded, zero hyperbolic. (`same_orbit`: `success 99, safety_cap 1`.)

### 4.2 By initial phase gap (pooled 500; gap = mean-longitude difference Δλ, the physically meaningful along-track offset — `gapM_deg` in the CSV is the env's `init_phase_gap_max` knob, which differs from Δλ because ω is randomized)

| \|Δλ₀\| | n | success | median Δv | median steps |
|---|---|---|---|---|
| 0–45° | 110 | 98.2% | 150 m/s | 1609 |
| 45–90° | 132 | 100.0% | 182 | 1778 |
| 90–135° | 116 | 99.1% | 197 | 1883 |
| 135–180° | 142 | 99.3% | 224 | 1922 |

Cost rises monotonically with gap, success does not — the small-gap bin is *slightly worse* because a small gap that must be closed downward can be blocked by the perigee floor (§5.2).

### 4.3 By eccentricity and altitude separation (pooled 500)

| e_target | n | success | median Δv | | \|Δa\| | n | success | median Δv |
|---|---|---|---|---|---|---|---|---|
| 0–0.0125 | 140 | 100.0% | 173 | | 0–100 km | 124 | 100.0% | 175 |
| 0.0125–0.025 | 148 | 100.0% | 173 | | 100–200 km | 174 | 98.9% | 181 |
| 0.025–0.0375 | 105 | 98.1% | 203 | | 200–300 km | 134 | 99.3% | 187 |
| 0.0375–0.051 | 107 | 98.1% | 246 | | 300–600 km | 68 | 98.5% | 219 |

Eccentricity costs Δv (`v·|Δē|/2` dominates the transfer budget) but barely costs success.

### 4.4 Cost and effort (successes, pooled headline)

| metric | p10 | median | p90 | max |
|---|---|---|---|---|
| Δv used | 105 | **189 m/s** | 300 | 409 (budget 478.1) |
| fuel budget consumed | 22% | **40%** | 63% | 86% |
| sim steps | 1592 | **1868** | 1948 | (cap 2000) |
| agent decisions | 229 | **352** | 454 | |

**Δv vs analytic comparators**

| comparator | median ratio | p90 |
|---|---|---|
| two-impulse orbit-match lower bound `max(Hohmann, v·\|Δē\|/2)` | **1.30×** | 1.84× |
| circular Hohmann alone (ignores ē and phasing) | 1.87× | 5.19× |
| the expert's own analytic Hohmann+phasing plan | **0.99×** | 1.07× |

The last row is the important one: the executed Δv lands within 1% of the analytic prediction, i.e. burn quantization and finite-burn arc losses are ≈ 1%, and the 1.30× over the shape-only bound is precisely the **cost of phasing**. For scale, the T1 Lambert baseline's time-matched comparator on the same IC distribution has median 225 m/s; the expert's 189 m/s median is below it because the expert is free to use the full 33-h budget for a cheap drift orbit whereas that comparator was constrained to the RL policy's episode length.

**Action mix (headline, all decisions):** burns 16.4%, warps 42.8%, coast 40.8%. Burn breakdown is Hohmann-like: clusters of 25/10/5 m/s at one apsis, the complementary cluster ~half an orbit later, then 1–2 m/s trims. This is usable imitation data.

### 4.5 Time-vs-fuel frontier (200 eps, seed 42; `T3_K` env var sets the horizon gain)

| K | success | median steps | median Δv |
|---|---|---|---|
| 0.30 | 94.5% | 1305 | 249 m/s |
| 0.45 | 98.0% | 1612 | 221 m/s |
| **0.65 (default)** | **99.0%** | 1875 | **190 m/s** |
| 0.75 | 95.0% | 1944 | 174 m/s |

Phasing is the whole trade: buying phase with a large drift orbit is fast and expensive, with a small one slow and cheap. The 2000-step cap sets where you can sit on that curve.

### 4.6 Success-box tightening (100 eps, seed 42; controller aim scaled to the box)

| box | success | median closest approach |
|---|---|---|
| 30 km / 50 m/s (headline) | **99%** | 64.7 km ← never needs to get closer |
| 20 km / 20 m/s | 99% | 36.7 km |
| 10 km / 10 m/s | **97%** | 12.0 km |
| 5 km / 5 m/s | **53%** | 6.1 km |
| 5 km / 1 m/s | **18%** | 4.9 km |

**This is a direct, pre-training answer for task #8 (retrain at 5 km / 1 m/s): that box is at the action-quantization floor and is not reliably reachable even by an optimal scripted controller.** Derivation, confirmed by direct measurement at a = 6921 km:

- smallest burn = 1 m/s ⇒ `Δa = 2Δv/n = 1825 m` (measured 1824.6 m), `Δe = 2Δv/v = 2.64e-4` (measured 2.636e-4)
- best achievable parked residual is therefore `|Δa| ≤ 912 m`, `|Δē| ≤ 1.3e-4`
- ⇒ irreducible relative position ≈ `|Δa| + a·|Δē|` ≈ **1.8 km**, irreducible inertial relative velocity ≈ `0.5·v·|Δa|/a + v·|Δē|` ≈ **1.5 m/s**

5 km/1 m/s asks for a velocity tolerance *below* the quantum. 10 km/10 m/s is ~6× the floor and works. If a tightened criterion is wanted, **10 km / 10 m/s is the tightest defensible target**; anything tighter needs sub-1 m/s burn actions.

---

## 5. What limited it (evidence-backed, ranked)

**5.1 The 2000-step budget is the binding constraint.** All 4 headline failures and the single same-orbit failure are `safety_cap`. The mechanism is arithmetic, not tuning: closing phase requires `Δλ = 1.5·n·(da/a)·T`, so at LEO with the 340 km drift cap the *fastest* possible π-closure is ~39,000 s, and the *cheap* closure needs most of the 120,000 s. The fuel-efficient operating point consumes a median 1868 of 2000 sim steps (93%). This is why the K-frontier exists and why 3 of the 4 failures also ran the tank down: forced onto a fast/expensive drift they exhausted Δv before the clock.

**5.2 The 200 km-altitude perigee floor removes one phasing direction on some episodes.** When `a_t` is low and `e_t` is high, the target's own perigee sits only ~10 km above the floor (e.g. failure ep 99: `a_t` = 379 km alt, `e_t` = 0.025 ⇒ perigee 210 km alt). A drift orbit *below* the target is then unavailable, so a short-way closure that needs `da < 0` must instead go the long way round — the 2π trip that exhausted the tank (Δv used 470/478). This is an **operational choice of the expert, not an env constraint**: `check_termination` only fires at `hard_radius = R_EARTH` (6371 km); `EARTH_KEEPOUT` is used only by `valid_init_only` for init screening. Dropping the floor to ~6450 km would remove these failures. Flagging it because it also means an RL policy is *allowed* to fly through 200 km altitude and will discover that.

**5.3 Burn quantization does not bind at headline conditions** (17× position margin, 33× velocity margin over the 1.8 km / 1.5 m/s floor) and only binds below a 10 km / 10 m/s box.

**5.4 Fuel does not bind at headline conditions** — median 40% of budget, p90 63%.

---

## 6. Debugging history (the two mistakes worth recording)

1. **Over-conservative perigee filtering in the planner.** Filtering candidate drift orbits on `min(a_s,a_p)·(1 − max(e_s,e_leg) − Δa/2a) ≥ EARTH_KEEPOUT` — a transfer-transient bound — rejected *every* candidate on many episodes and dropped the controller into a do-nothing fallback (65% success). The transient is correctly handled online by the burn law's own per-burn perigee guard; filtering only the *sustained* drift orbit took it to 93%.

2. **A discrete "start the closing transfer NOW" trigger is unusable here (91.5% → the smooth law fixes it).** Two independent failure modes:
   - The closure *duration* jumps by a half orbital period whenever the required burn apsis flips, so `Δλ_final(t)` is piecewise-linear **with ~10–15° jumps**; the zero crossing is frequently straddled by a jump and skipped, costing a full relative revolution (a 5–15° residual = 600–1800 km, then a trim cycle that usually cannot fit in the remaining clock).
   - An analytic half-period closure model under-predicts the true closure by ~2.5× (measured: predicted 47 steps, actual 125), firing the trigger a whole revolution late. Replacing it with a closed-loop rollout of the CLOSE controller through the exact model fixed the *estimate* but not the *jumps*.
   - The fix that actually worked was to delete the trigger: a continuous first-order phase-tracking law whose commanded drift offset shrinks with the remaining phase, floored at 3 km. No discrete event to miss; the terminal sweep through Δλ = 0 happens at ≤ 5 m/s of along-track rate, giving thousands of seconds inside the box.
   - Sub-plot: the threshold was set at `0.5·DT`, finer than the 60 s decision grid, so `t_align` stepped straight past zero and wrapped `2π`. Any event threshold must be ≥ DT.

---

## 7. Cross-lane finding: the env's shaping actively fights the optimal solution

Measured by running the expert in the live env and reading `rewards[0]` (100 headline episodes, seed 42):

| quantity | value |
|---|---|
| mean episode return (env reward incl. shaping) | **+8.668** |
| mean terminal-step reward | **+7.753** |
| ⇒ mean **net** shaping contribution over an entire optimal episode | **+0.915** (≈ 5e-4 per sim step) |
| Φ_orbit at t = 0 (gate opens below 2.0) | median **16.6** |
| fraction of expert **sim-steps** with the σ₂ gate open (Φ_orbit < 2, i.e. \|Δa\| ≲ 20 km) | **14.1% mean, 0.2% median** |

This is direct quantitative confirmation of the mechanism hypothesis in the task brief. The optimal policy **deliberately holds |Δa| between 3 and 340 km for essentially the whole episode** — that *is* the phasing maneuver. Over that entire interval:
- σ₂ ≈ 0, so `W_PHASE·Φ_phase·σ₂` is switched off — the shaping cannot see the one quantity the expert is actually driving to zero;
- σ₃ ≈ 0, so the velocity term is off too;
- the only live term, `W_ORBIT·Φ_orbit`, **penalizes the drift orbit**, i.e. the shaping gradient points *against* the optimal maneuver for ~86% of the trajectory (median: ~99.8%).

The net shaping payout along the globally-optimal trajectory is +0.92 against a +7.75 terminal — a nearly flat, occasionally adversarial signal. Also relevant to discounting: the expert reaches the terminal at a median 1868 sim steps / 352 decisions; at γ = 0.995 per decision that is 0.995³⁵² ≈ 0.17 (workable), but per sub-step 0.995¹⁸⁶⁸ ≈ 8e-5 (not). **Time-warp actions are load-bearing for terminal-reward visibility, and the expert uses them for 42.8% of its decisions.**

---

## 8. Reproduction

```bash
cd /Users/pete/space_training
python3 scripts/orbital/t3/probe_model_match.py                       # replica validation
python3 scripts/orbital/t3/expert_controller.py --suite --seed 42 \
        --csv web_data/results/expert_baseline.csv                    # 200 headline + 100 same-orbit
python3 scripts/orbital/t3/expert_controller.py --episodes 100 --seed 7
T3_K=0.45 python3 scripts/orbital/t3/expert_controller.py --episodes 200   # time/fuel frontier
```
Runtime ≈ 15–30 s per 100 episodes. Deterministic given the seed (the C env's IC stream depends only on the reset count, not on actions, so failure episodes are reproducible by index).