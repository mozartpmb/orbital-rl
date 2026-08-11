All artifacts written, nothing committed, no files outside `scripts/orbital/t3/` and `web_data/results/` touched.

---

# T3 — Forensic audit of the reward shaping under corrected dynamics

**Method.** Every number below comes from driving the real C env (`Orbital(...)` exactly as `eval_checkpoint.py` builds it) and recomputing Φ and its components in Python from the env's own observation vector, which carries `a,e,θ,ω` for chaser and target exactly. **Validation: the Python legacy Φ reproduces the realised C reward to 5.5e-08 over 2261 non-terminal steps**, so the reimplementation is provably faithful and the counterfactual designs are evaluated on identical, real trajectories.

Deliverables:
- `/Users/pete/space_training/scripts/orbital/t3/shaping_trace.py` (self-contained; run from `/Users/pete/space_training/pufferlib`)
- `/Users/pete/space_training/scripts/orbital/t3/shaping_trace_output.txt` (full console log)
- `/Users/pete/space_training/web_data/results/t3_shaping_trace.csv` (2268 per-decision rows × all designs × 3 discount forms)
- `.../t3_shaping_design_comparison.csv`, `.../t3_omega_audit.csv`, `.../t3_phase_gap_sweep.csv`, `.../t3_phasing_feasibility.csv`, `.../t3_burn_phase_discontinuity.csv`

---

## 0. Sign check (empirical, not derived)

`phase_gap_fixed = g` sets `tgt_M = sat_M + g`, i.e. **target ahead / chaser behind** by `g`; the wrapped gap the code sees is `θ_s − θ_t = −g`.

| burn | Δa | drift rate ġ = d(λ_s−λ_t)/dt | gap after 6 h |
|---|---|---|---|
| retrograde 25×4 (**lower a**) | −184.2 km | **+8.689 °/hr** | −90.0° → **−37.5°** (closing) |
| prograde 25×4 (raise a) | +196.9 km | −8.688 °/hr | −90.0° → −142.5° (opening) |

**Confirmed: a chaser behind the target must LOWER its orbit.** In code terms, with `g = wrap(λ_s − λ_t)` (chaser minus target), the required drift orbit is `da* = +k·g`, `k > 0` (g<0 → da<0). The `-k*gap` sign in the S-C brief is correct only if `gap` is defined target-minus-chaser.

---

## 1. Q1 — The drift-orbit valley

### 1.1 The expert path is real and cheap; the env is not fuel- or time-infeasible

Scripted expert (perigee-aware N-revolution phasing: open → drift → close → fine-phase → null). **All four runs reach `TERM_SUCCESS`.**

| scenario | plan | decisions / sim steps | Δv used | terminal reward |
|---|---|---|---|---|
| 90° gap, 750–800 km | N=18, da=−66 km, 2×35.1 m/s | 41 / 1772 | 70 m/s | +9.672 |
| 150° gap | N=18, da=−110 km, 2×59.0 m/s | 41 / 1756 | 118 m/s | +9.172 |
| 180° gap | N=18, da=−133 km, 2×71.2 m/s | 50 / 1873 | 142 m/s + fine round | +8.423 |
| 69 km transfer **+** 120° gap | Hohmann 2×18.9 + N=18, da=−88 km | 62 / 1906 | ~136 m/s | +8.495 |

Feasibility grid (`t3_phasing_feasibility.csv`, 200 km perigee floor, ≤1450 steps, ≤460 m/s) — **every (altitude, gap) cell is feasible**, worst case 245 m/s:

```
alt\gap   -180  -135   -90   -45   -15    15    45    90   135   180
   300    166u  206u  245u   41d   13d   13u   43u   84u  126u  166u
   400    165u  205u   87d   43d   14d   14u   42u   84u  125u  165u
   500    175u  130d   86d   43d   14d   14u   42u   83u  132u  175u
   600    174d  129d   85d   42d   14d   15u   45u   88u  132u  174d
   700    173d  128d   91d   45d   15d   15u   44u   88u  131u  173d
   800    184d  137d   90d   45d   15d   15u   44u   87u  130u  184d
```
(d = lower/catch-up, u = raise/lap-around). **Two structural constraints fall out:** (a) below ~500 km altitude the perigee floor forbids lowering for large gaps, so the only route is *raising and lapping the long way* — 2–3× the drift time; (b) the min-Δv solution consumes ~1750–1900 of the 2000 available sim steps, so **MAX_STEPS is a binding constraint and the fuel/time trade is live**. The task is learnable in principle; it is the reward that is broken.

### 1.2 Per-leg shaping return, expert @ 180° gap

Two views. **γ_shape = 1** (pure telescoping) isolates the *information* content — for any policy the undiscounted total is exactly Φ(s_T)−Φ(s_0), so per-leg numbers show where the signal lives. **γ^τ** is what the code actually does.

**γ_shape = 1, β = 1 — expert_g180 (50 decisions / 1873 sim steps, SUCCESS):**

| leg | n | sim steps | legacy | S-A | S-B | S-C | S-R (t̂go) | **S-R3 (rec.)** |
|---|---|---|---|---|---|---|---|---|
| A_open | 5 | 5 | **−0.1125** | −0.1127 | +0.1238 | +0.2023 | −0.0029 | −0.1132 |
| B_drift | 37 | 1742 | **+0.0000** | +0.1128 | +0.8710 | **−0.3193** | **+0.9972** | **+0.9934** |
| C_close | 5 | 5 | +0.1235 | +0.0109 | **−0.9948** | +0.2879 | +0.0025 | +0.1064 |
| F_open | 1 | 1 | −0.0060 | −0.0022 | +0.8838 | −0.0274 | −0.0001 | −0.0037→−0.0009 |
| F_drift | 1 | 60 | +0.0000 | +0.0000 | +0.0296 | +0.0004 | −0.0021 | +0.0010 |
| TERMINAL (NHR clamp) | 1 | 60 | +0.0150 | +0.0112 | +0.0866 | +0.1761 | +0.0054 | +0.0133 |
| **TOTAL** | 50 | 1873 | **+0.0200** | +0.0200 | +1.0000 | +0.3200 | +1.0000 | **+1.0000** |
| **worst single step** | | | **−0.0466** | −0.0466 | **−0.7660** | −0.0436 | **−0.0021** | **−0.0408** |

Same structure at 90° / 150° / transfer+120° (`t3_shaping_design_comparison.csv`).

### 1.3 What this says about the legacy design

1. **The drift leg — 1742 of 1873 sim steps (93 %) — earns exactly 0.0000.** σ₂ = sigmoid((2 − Φ_orbit)/0.2) with Φ_orbit = |Δa|/10 km = 13.3 is `sigmoid(−56.5) ≈ 0`. Φ_legacy is *constant* through the entire phasing maneuver. Measured gate activation: σ₂ > 0.01 on **1/41, 1/41, 6/50, 13/62** decisions (2.4 %, 2.4 %, 12 %, 21 %) — the shaping is blind exactly where the task lives.
2. **The only non-zero legacy signal on the phasing task is a penalty for doing it.** Opening the drift orbit costs −0.1125; the *entire* phase-reward budget the design can ever pay is `W_PHASE·max(1−cosΔθ) = 0.01·2 = 0.02`. So legacy charges **5.6× the total available phase reward as an entry fee** at 133 km, **15× at 300 km**. That is the valley, quantified.
3. **The worst single adverse step (−0.0466) is 2.3× the entire episode's shaping return (+0.0200).** The signal-to-cliff ratio is below unity.
4. **γ^τ discount leak.** `c_step` uses `pow(0.995, last_tau)` — γ per *sim sub-step* — while PPO discounts per *agent decision*. A frozen state therefore pays `(γ^τ−1)Φ = +0.26|Φ|` per warp-1hr. Measured leak (γ^τ total minus γ_shape=1 total):

   | policy | legacy leak | legacy total (γ^τ) |
   |---|---|---|
   | expert 90° | +0.5081 | +0.5302 |
   | expert 180° | +1.0109 | +1.0435 |
   | coast-forever | +0.1999 | +0.1999 |
   | warp-1hr-forever | +0.1714 | +0.1714 |
   | **park-far-and-warp** (6× prograde-25 → Δa=+278 km, then warp) | **+2.5911** | **+2.3094** |

   **Under the shaping as implemented, essentially 100 % of the legacy shaping return is discount leak, and a deliberately useless "burn away and wait" policy harvests 2.2× more of it than a physically optimal 180° phasing expert that actually succeeds.** The leak is proportional to |Δa| × elapsed sim time, i.e. the design literally pays to be far from the target and to wait there.
5. **Do-nothing floor.** Coast-forever and warp-forever return exactly **0.0000** under γ_shape=1 (all designs), and **+0.1999 / +0.1714** under γ^τ for legacy. Requirement "≈0 for do-nothing" is met only by γ_shape = 1.

### 1.4 Verdict on the three candidate redesigns

- **S-A (phase-first gates) — rejected.** Total signal identical to legacy (+0.0200) and the same worst step (−0.0466), because Φ_phase is capped at `W_PHASE·2 = 0.02`; gating the orbit term off during the drift removes the penalty but adds no reward. Worse, it inherits Φ_phase's burn-teleport (§2.7): its worst step against the park-farm adversary is **−0.2761**, 14× its own expert-path total.
- **S-B (realised-drift t̂go) — rejected as specified.** Right idea, pathological instantiation. Using the *current* drift rate means t̂go collapses from the cap to a finite value the instant any Δa exists (A_open **+1.0076**, F_open +0.8838) and blows back up to the cap when Δa is removed (**C_close −0.9948**). The mandatory final circularisation burn costs the entire potential range: worst step **−0.766**, i.e. **77 % of the whole shaping budget in one action**. It also pays the park-farm adversary +0.7290.
- **S-C (guidance corridor `|Δa − da*(Δφ)|`) — rejected, structural flaw.** The expert *holds* Δa constant while the gap shrinks, but `da*(Δφ) = k·Δφ` shrinks with the gap, so the corridor error grows monotonically through the drift: **B_drift = −0.3193**, the leg that matters is penalised again. No choice of k fixes this (a time-to-go-normalised da\* has the same defect unless the expert is forced to finish exactly at the horizon). Total +0.3200 with worst step −0.0436, but the sign of the drift leg is wrong.

**Winner: an admissible time-to-go potential.** S-R (`|Δλ|/ġ_max + |v_rel|/A_max`) puts **99.5 %** of its signal on the drift leg with a worst step of −0.0021, but starves the transfer sub-task (transfer leg −0.0022, i.e. no credit for a 69 km Hohmann) and inherits the periodic |v_rel| ripple of an eccentric phasing orbit (25 → 310 m/s within one drift). **S-R3** replaces the |v_rel| term with a linearised orbit-match Δv and is the recommendation (§3).

---

## 2. Q2 — The ω question. Yes, it is a live second bug, and there is a third.

### 2.1 How ω is initialised (`c_reset`, orbital.h:1015–1056)

```
ω_target = (e_tgt > 0) ? U(0, 2π) : 0
ω_sat    = ω_target                      if same_orbit_init                (or e_sat_fixed & soi)
         = U(0, 2π)                      if e_max_sat > 0                  ← headline path
         = 0                             otherwise (e_sat = 0, Phase-4 default)
```
`omega_offset_fixed > −10` overrides `ω_target = ω_sat + offset` (used only by the W1 surface eval).

**Under the headline config (`e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0`) the two ω are independent uniforms.** Measured over 3000 resets: |ω_s − ω_t| **mean 89.87°, median 88.86°, p90 162.5°, max 180.0°** — i.e. uniform on [0°, 180°]. At e=0 (Phase 4 / `orbital.ini` defaults) both are exactly 0.

### 2.2 What `init_phase_gap_max` actually parameterises

```c
double tgt_M = env->sat.orbit.M + phase_gap;   // orbital.h:1126
```
It is a **mean-anomaly** difference — not true anomaly, not true longitude. At e=0 all three coincide, so it worked as intended in Phase 3/4. At e>0 with independent ω it is **inert**:

| `init_phase_gap_max` | mean \|true-longitude gap\| | max | mean \|Δθ\| (what Φ_phase sees) |
|---|---|---|---|
| 10° (e=0.05) | **89.88°** | 179.96° | 5.18° |
| 30° (e=0.05) | **90.02°** | 179.88° | 15.24° |
| 90° (e=0.05) | **90.40°** | 179.96° | 45.58° |
| 180° (e=0.05) | **91.05°** | 179.71° | 91.12° |
| 30° (e=0) | 15.12° | 29.89° | 15.12° |
| 180° (e=0) | 90.70° | 179.33° | 90.70° |

**Consequence: every phase-gap curriculum stage ever run with `e_max_sat > 0` was a no-op — the agent always faced the full ±180° physical distribution.** (Phase 3's 30°→90°→180° ladder ran at e=0 and was real; the Phase 5 / 5.5 / T1 ladders at e=0.05 were not.) This alone invalidates "phase-gap curriculum" as a described mechanism for every post-Phase-4 run.

### 2.3 Which angle each consumer uses

| consumer | formula | uses |
|---|---|---|
| `compute_phi` Φ_phase (orbital.h:723-725) | `1 − cos(θ_s − θ_t)` | **raw true anomaly** |
| obs[13],[14] (orbital.h:568-570) | `sin/cos(θ_s − θ_t)` | **raw true anomaly** |
| obs[15],[16] | `sin/cos(θ_t)` | **raw true anomaly** |
| obs[9-12] | `sin/cos ω_s`, `sin/cos ω_t` | ω only (recovery requires a product term) |
| obs[33-37] LVLH | rotation by `θ_t + ω_t` | **true longitude — correct** |
| `check_termination` success box | Cartesian distance / rel-velocity | correct |

### 2.4 The misdirection, quantified

Φ_phase = 0 means θ_s = θ_t, which places the chaser at inertial longitude `θ_t + ω_s` while the target sits at `θ_t + ω_t` — a residual true-longitude error of exactly **ω_s − ω_t**. Chord miss `2a·sin(|Δω|/2)` at a = 7000 km:

| config | mean miss | median miss | fraction of episodes where Φ_phase = 0 is inside the 30 km box |
|---|---|---|---|
| **headline** (e=0.05, soi=0) | **8894 km** | **9800 km** | **0.17 %** |
| Phase-4 (e=0, soi=0) | 0.0 km | 0.0 km | 100 % |
| Stage-1 (e=0.05, soi=1) | 0.0 km | 0.0 km | 100 % (at t=0 only — see §2.5) |

**Yes: for every e>0 episode the shaping's phase objective points at a location a median 9800 km from the actual rendezvous point.** This is a live second bug, and it has been active in every headline eval and every Phase-5/5.5 training run.

### 2.5 ω diverges even from same-orbit inits

`cartesian_to_elements` re-derives ω from the eccentricity vector after every burn, so `ω_s ≠ ω_t` develops from the first Δv even when `same_orbit_init = 1`. Measured across the scripted 180° episode:

| e at init | ω_s − ω_t start → end | range over episode | largest single-decision Δω_s |
|---|---|---|---|
| 0.00 | 0.00° → +9.21° | [0.0°, +9.2°] | 4.85° (one 25 m/s burn) |
| 0.03 | 0.00° → **−159.96°** | **[−162.3°, +103.9°]** | **97.66°** (one 25 m/s burn) |

### 2.6 The third bug: burns teleport the shaping's phase coordinate

Isolating a single burn against a matched pure-coast control (`t3_burn_phase_discontinuity.csv`, 40 seeds × burn phase, 750–800 km):

| action | Δv (m/s) | e at burn | median \|Δ(θ_s−θ_t)\| vs coast | max | median ΔΦ_phase (range 0–2) | median physical displacement vs coast |
|---|---|---|---|---|---|---|
| 3 | +25 | **0.00** | **86.51°** | 175.99° | **0.669** | 0.99 km |
| 2 | +10 | **0.00** | **86.51°** | 175.98° | 0.668 | 0.40 km |
| 12 | **+1** | **0.00** | **86.51°** | 175.98° | 0.668 | **0.040 km** |
| 7 | radial +10 | 0.00 | 88.04° | 179.02° | 0.744 | 0.45 km |
| 3 | +25 | 0.02 | 14.19° | 19.51° | 0.246 | 0.99 km |
| 3 | +25 | 0.05 | 5.17° | 7.65° | 0.090 | 0.99 km |

**From a circular chaser, a 1 m/s burn moves the shaping's phase coordinate by a median 86.5° and Φ_phase by 33 % of full scale, while moving the spacecraft 40 metres.** The jump magnitude is *independent of Δv* at e=0 — it is purely the relocation of the periapsis reference that defines θ's origin. This is the same class of defect as the `true_to_mean` bug (a representation artifact fired by `cartesian_to_elements` on every burn), it corrupts **obs[13],[14]** on every burn as well as Φ_phase, and it has never been documented.

---

## 3. Q3 — Recommendation: one design, concrete constants

### 3.1 The potential (`shaping_mode = 1`, "phase-time")

```
λ_s = M_s + ω_s ,  λ_t = M_t + ω_t          # MEAN longitude — not true anomaly
Δλ  = wrap(λ_s − λ_t)                        # ∈ [−π, π]

v_t      = sqrt(MU / a_t)
Δē       = |e_s·(cos ω_s, sin ω_s) − e_t·(cos ω_t, sin ω_t)|
Δv_match = 0.5 · v_t · hypot((a_s − a_t)/a_t, Δē)      # linearised 2-impulse (Edelbaum/Gauss)
                                                        # NOT |Δa|/1855 + Δē·v — that double-counts,
                                                        # because the burn that removes the phasing
                                                        # orbit's Δa also removes the e it created

Φ(s) = −[ W_LAMBDA · |Δλ|/π  +  W_MATCH · min(1, Δv_match / DV_REF) ]

W_LAMBDA = 1.0        W_MATCH = 0.35        DV_REF = 300.0 m/s
```
No sigmoid gates anywhere — cliffs are impossible by construction. Φ ∈ [−1.35, 0].

### 3.2 The shaping reward — **γ_shape must be 1**

```
r_shape = BETA_SHAPE · ( Φ(s') − Φ(s) )        BETA_SHAPE = 1.0
```
i.e. replace `pow(0.995, env->last_tau)` with `1.0` in mode 1. Rationale: the current `γ^τ` is per *sim sub-step* while PPO discounts per *agent decision*; the mismatch is the entire source of the +2.59 park-farm leak (§1.3.4). Per-decision γ (0.995¹) shrinks the leak but does not remove it — coast-forever still harvests `(1−γ)·|Φ|·2000`. Only γ_shape = 1 gives exactly zero. The cost is that PBRS is no longer *formally* policy-invariant under γ_RL = 0.995 — but it already is not (the γ^τ/decision mismatch broke that), and the residual bias favours reaching the goal sooner, which is aligned. Terminal handling is unchanged: NHR clamp `r += β(0 − Φ_prev)` at state-caused terminals, **no clamp at the safety cap** (keep the Φ-clamp-leak fix).

### 3.3 Measured against the stated requirements

| requirement | measured (β=1, γ_shape=1) |
|---|---|
| monotone along the expert path | drift leg **+0.9944** (180°), **+0.6681** (transfer+120°); only two adverse legs, both the mandatory burn sequences |
| ≈0 return for do-nothing | coast-forever **0.0000**, warp-1hr-forever **0.0000** (exact) |
| no cliff > 1/10 of the +10 terminal (i.e. 1.0) | **worst adverse step −0.0408** → **24× margin**; = 4.1 % of the episode's own shaping return (legacy: 233 %) |
| bounded for adversaries | park-far-and-warp **+0.1341** (legacy γ^τ: +2.3094; S-B: +0.7290; S-R: +0.3659) |
| transfer sub-task gets credit | T_xfer leg **+0.0340** (S-R: −0.0022; legacy: +0.0477 but that is leak) |
| ratio drift-reward : drift-entry-cost | **8.8 : 1 in favour** (legacy: **5.6 : 1 against**) |

Full per-scenario numbers in `t3_shaping_design_comparison.csv` (`gamma_form = "gamma_shape=1"`, `design = "S_R3"`).

### 3.4 Should phase use θ+ω? **Yes — use mean longitude λ = M + ω.**

- fixes the §2.4 misdirection (median 9800 km → 0)
- fixes the §2.6 burn teleport: λ is continuous through a burn by construction (M is re-derived consistently with the new ω)
- drift is exactly linear in λ, so |Δλ| has no equation-of-centre ripple (true longitude u = θ+ω carries a ±2e ≈ ±5.7° ≈ 700 km ripple at e=0.05; measured, it costs ~2× the worst adverse step vs λ)
- the residual λ↔u offset shrinks to zero as the Δē term drives e-matching, and the final 30 km is the terminal reward's job, not the shaping's

### 3.5 Plumbing (copy the `rendezvous_radius_m` pattern exactly — **do NOT implement here**)

`unpack()` (`env_binding.h:615`) hard-fails on a missing key, so every layer must carry the kwarg; `orbital.py`'s default guarantees presence. Ints and floats only (no strings — see `gave_up_action`'s enum encoding if a string is ever needed).

1. **`orbital.h`** — add to `struct Orbital` next to `rendezvous_radius_m`:
   `int shaping_mode;  double shape_w_lambda;  double shape_w_match;  double shape_dv_ref_ms;  double shape_gamma;`
2. **`orbital.h::compute_phi`** — branch at the top: `if (env->shaping_mode == 1) { …new Φ…; return; }`, legacy body untouched below.
3. **`orbital.h::c_step`** (the non-terminal shaping block) — `double gamma_tau = (env->shape_gamma >= 1.0) ? 1.0 : pow(env->shape_gamma, (double)env->last_tau);`
4. **`binding.c::my_init`** — five lines after `env->rel_vel_tol_ms = …`:
   `env->shaping_mode = (int)unpack(kwargs, "shaping_mode");` + the four doubles.
5. **`orbital.py::Orbital.__init__`** — signature defaults `shaping_mode=0, shape_w_lambda=1.0, shape_w_match=0.35, shape_dv_ref_ms=300.0, shape_gamma=0.995` and pass all five into `binding.vec_init(...)`.
6. **`pufferlib/config/ocean/orbital.ini` `[env]`** — `shaping_mode = 0`, `shape_w_lambda = 1.0`, `shape_w_match = 0.35`, `shape_dv_ref_ms = 300.0`, `shape_gamma = 0.995`. Training runs override with `--env.shaping-mode 1 --env.shape-gamma 1.0`.
7. **`pufferlib/scripts/orbital/eval_checkpoint.py`** — add the five params to `evaluate()` with identical defaults, forward into `Orbital(...)`, add CLI flags. Defaults keep every legacy-checkpoint eval byte-identical.

**Separate, optional, and breaking:** the same fix applied to **obs[13],[14]** (→ `sin/cos(Δλ)`) and **obs[15],[16]** (→ `sin/cos(λ_t)`) removes the §2.6 observation teleport. This changes observation *semantics* and invalidates warm starts, so gate it behind its own `phase_obs_mode` kwarg (default 0) and only enable it for from-scratch runs.

### 3.6 Two non-shaping findings that the recovery plan must absorb

- **`init_phase_gap_max` is inert at e>0** (§2.2). Any curriculum that stages the phase gap must either run at e=0 or be re-plumbed to control the *mean-longitude* gap (`tgt_λ = sat_λ + gap`, i.e. set `tgt_M = sat_M + gap + ω_s − ω_t`). Otherwise every stage is the 180° stage.
- **MAX_STEPS = 2000 is binding.** The min-Δv phasing solution uses 1750–1900 sim steps (~90 % of budget); at gaps ≥135° below 500 km altitude the perigee floor forces the slow "raise and lap" route. Expect a hard time/fuel Pareto edge, and consider whether the safety cap should rise to ~3000 for the ≥135° curriculum stage.