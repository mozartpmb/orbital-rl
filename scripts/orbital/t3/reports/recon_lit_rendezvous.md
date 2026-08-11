# T3 Literature Survey — RL for Orbital Rendezvous/Phasing + Classical Phasing GNC

Nothing committed. Scripts in `/Users/pete/space_training/scripts/orbital/t3/` (`phasing_design_table.py`, `write_lit_bib.py`, `pdf_extract.py`); CSVs in `/Users/pete/space_training/web_data/results/` (`t3_literature_survey.csv`, `t3_phasing_design_table.csv` 216 rows, `t3_phasing_capture_window.csv`).

---

## 0. Headline finding (read this first)

**The literature's hard leg is not our hard leg.** Essentially all deep-RL spacecraft rendezvous work operates on CW/Hill dynamics over 1–3 orbits at ranges of 100 m – 30 km, i.e. *terminal approach*. The Stanford ART benchmark uses an OCP horizon of literally `[1, 3]` orbits at 416 km; OrbitZoo's flagship "Hohmann transfer" task is a **30 km** altitude change. Nobody publishes end-to-end RL closing a ±180° co-orbital phase gap over tens-to-hundreds of revolutions.

**But classical GNC solved exactly our problem, and the fix maps 1:1 onto our broken `compute_phi()`.** The technique is **target semi-major-axis augmentation** (Naasz 2002 → Lanktun → Locoche → Holt/Baresi/Armellin 2024 → arXiv 2606.12108): bias the *commanded* semi-major axis by an amount proportional to phase error, so that a controller which only ever drives orbital elements toward a target automatically produces the drift orbit and converges on `a` and the fast variable **simultaneously**. Our `phi_orbit = |a_sat − a_target|/10km` is the *unaugmented* version, which is precisely the formulation those papers say fails to converge in phase.

Second finding, from first-principles computation (not literature): **our Δv budget is not the binding constraint — the horizon is.** At 550 km, closing 180° costs **25 m/s** if you have ~158 hours (δa = −23 km, 100 revs) and **130 m/s** if you only have the 33.3 hours a coast-only episode allows (δa = −116 km, 20 revs). The Δv-optimal maneuver is a *shallow, long* drift. The warp actions are not an efficiency convenience; they are what makes the fuel-optimal solution reachable at all.

---

## (a) Annotated bibliography

Full machine-readable version with per-entry "what transfers" fields: `web_data/results/t3_literature_survey.csv` (27 entries). Condensed below, grouped by what they actually contribute.

### A1. Solves phasing — classical (the load-bearing group)

**Holt, Baresi, Armellin, "Reinforced Lyapunov controllers for low-thrust lunar transfers," *Astrodynamics* 8(4):633–656, 2024.** <https://doi.org/10.1007/s42064-024-0212-x> (full text extracted locally)
RL learns the *state-dependent weights* of a Lyapunov/Q-law rather than the control itself: 100% Monte Carlo convergence, 9–12% improvement over the hand-tuned law, dual-actor (Earth-frame / Moon-frame). Critically, §6 "Guidance with phasing" states plainly that the Lyapunov law **cannot** achieve rendezvous criteria without augmenting the target semi-major axis, and gives the equation (Lanktun form):
```
a*_T = [ (1/a_T)^(3/2) + W_L · sgn(ΔL)·arccos(cos ΔL) / exp(N_tt) ]^(-2/3)
N_tt = (W_tt/2π)·sqrt(a³/µ) · Σ_X sgn(W_X)·( δ(X,X_T) / max_ν(Ẋ) )²      (revs-to-go)
```
`W_L ≈ 1` nominal; `W_tt` "can significantly alter the impact of this phasing element." Note the structure: the augmentation is applied to **1/a^(3/2) ∝ mean motion**, is proportional to the phase error, and **decays as exp(−N_tt)** so it vanishes as you converge. This is the single most transferable equation in the survey.

**arXiv:2606.12108, "Constrained Lyapunov Stabilization based on Gauss Variational Equations: From Spacecraft Orbital Transfers to Rendezvous."** <https://arxiv.org/abs/2606.12108>
Modern, explicit outer-loop statement with saturation and a stability claim:
```
n_mod   = clip( n_des − λ·zᵀSᵀP̄(z − z_des), 0.25·n_des, 4·n_des )     z = phase state on unit circle
a_des,mod = clip( (µ/n_mod²)^(1/3), 0.8·a_des, 1.2·a_des )
```
Locally input-to-state stable at `z_des`. Demonstrated on −2 rev and +3 rev phase gaps, both raising and lowering. The ±20% saturation on commanded `a` is a directly reusable drift-orbit sizing limit. **Use this as our expert controller** (it is ~30 lines and needs no Lambert solve).

**Naasz, MS thesis, Virginia Tech, 2002** — origin of "use `a` as a proxy for period and bias the target `a` by the phase error," with mean-motion control giving asymptotic convergence.

**Vallado *Fundamentals of Astrodynamics and Applications* ch.6 / Curtis ch.6, co-orbital phasing.** Canonical two-impulse phasing: `T_ph = T(1 − Δθ/2πN)`, `a_ph = (µ(T_ph/2π)²)^(1/3)`, `Δv = 2|√(µ(2/a − 1/a_ph)) − √(µ/a)|`. Worked GEO relocation example (0°E → 137.2°W): Δv falls 4.228 → 0.569 km/s going from N=0 to N=1 rev, then rises again — i.e. **Δv is monotone-decreasing in allowed revolutions until the drift orbit gets so shallow the geometry constraint binds**. Same structure holds in our LEO band (computed below).

**ISS/Soyuz operational phasing profiles** (Soyuz/Progress short-profile literature; Burlison, "Rendezvous and Docking: A User's Guide"). The operational numbers: 2-day profile tolerates ~270° initial phase angle; 4-orbit profile ~20°; 1-orbit profile ~0.4°. Chaser inserts *below* the target so it catches up, then NH/NC height-adjust burns at apsides tune the catch-up rate. **This is the canonical answer to "how long does phasing take": days, not hours, for large gaps.**

### A2. Deep RL that actually targets phasing (n = 1)

**"Propulsionless planar phasing of multiple satellites using deep reinforcement learning," *Adv. Space Res.* 67(11):3667, 2021.** <https://doi.org/10.1016/j.asr.2020.09.025>
The closest published analogue to our hard leg. A2C and PPO rephase the relative argument of latitude of three coplanar LEO satellites using **differential drag** (action = cross-sectional area). Two lessons transfer: (i) phasing *is* learnable by PPO when the action is a slow drift-rate knob rather than an impulse; (ii) they needed a **surrogate propagation model** to make multi-week horizons trainable — i.e. temporal compression was a prerequisite, exactly like our warp actions.

### A3. RL with coast-duration / temporal abstraction in the action

**"Optimal Multi-impulse Linear Rendezvous via Reinforcement Learning," *Space: Sci. Technol.* 2023, art. 0047.** <https://spj.science.org/doi/10.34133/space.0047>
Action = **(impulse Δv, coasting period Δt) jointly**. Explicit astrodynamics precedent for coast duration being part of the action vector. Reward reflects fuel, transfer time, relative state, dynamics model. CW/linear relative motion only.

**Biedenkapp et al., "TempoRL: Learning When to Act," ICML 2021.** <https://arxiv.org/abs/2106.05262>
Factorize the policy into *(what action, how long to hold it)*, with the skip-length head **conditioned on the chosen action**. Reported far more stable than FiGAR/DAR as max skip length grows. **Our three warp actions are a hand-coded, action-independent version of this.**

**Sharma et al., "FiGAR," ICLR 2017** <https://arxiv.org/abs/1702.06054> — second policy head emitting a repetition count from a discrete set. Cheapest possible upgrade for us: keep the 16 actions, add a 4-way duration head.

**Sutton, Precup, Singh, "Between MDPs and semi-MDPs," *AIJ* 112, 1999.** Confirms our shaping term `β(γ^τ·Φ(s′) − Φ(s))` is the correct SMDP-consistent form. That part of the env is right.

**"Hierarchical Deep Reinforcement Learning for cubesat guidance and control," *Control Eng. Practice*, 2024.** Hierarchical actor-critic (HAC) + DDPG, policies at **different time resolutions**, hardware-in-the-loop validated. Closest HRL-for-rendezvous precedent; still short-range.

### A4. Reward design that reportedly worked at long horizon

**LaFarge, Miller, Howell, Linares, AIAA SciTech 2020, "Guidance for Closed-Loop Transfers using RL with Application to Libration Point Orbits."** <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2020_AIAA_LafMilHowLin.pdf> (full text extracted)
The best-documented reward design in the astro-RL literature and the strongest template for us:
```
r = η·exp(−λk)   if |δr| < δr_max and |δv| < δv_max      k = ‖δρ‖ to nearest reference state (KD-tree)
  = b            arrival                                  b = 25
  = p_i = −10    impact
  = p_d = −4     deviate beyond threshold → EPISODE TERMINATED
η = i/(n·ξ) + 1  (ramps reward along the reference; kills the "stationkeep at departure orbit" local optimum)
```
State = `[x, y, ẋ, ẏ, m, δρ(4), C_st, C_ref]` (11-D): dynamical state + **relative state w.r.t. nearest reference point** + an **energy-like invariant** for both agent and reference. Removing the Jacobi constant "does not prohibit convergence, but the resulting policy is less optimal."
Hyperparameters: **γ = 0.9** (recommended range 0.85–0.9) over ~150-step episodes, 100k training episodes, λ = 3600 (range 1300–4000), actor lr 1e-4, critic lr 2e-3, KL target 3e-3, actor 120/60/30 tanh.
Two results that mirror our own experience exactly: (1) they trained **60 identically-configured agents and selected the good ones**, because "multiple identically configured agents can converge to drastically different policies" — our seed-1337 Stage-1 failure is a known-normal phenomenon, not a bug; (2) their reported "failure" mode is *reaching 2 km / 2 cm/s but not the 30 km / 50 cm/s arrival box within the step cap* — i.e. **cap-timeout is also their dominant non-success**, same as ours.

**Federici & Zavoli, "Densely rewarded RL for robust low-thrust trajectory optimization," *Adv. Space Res.* 72(4), 2023** (and Zavoli & Federici, JGCD 44(8) 2021 / <https://arxiv.org/abs/2008.08501>). States the sparse-vs-dense tradeoff cleanly for astrodynamics: sparse terminal rewards generalize better but cost far more training; dense shaping trains fast but **biases toward the shaping designer's assumed solution**. This is a precise description of our failure: the shaping encoded "stay on the target orbit," which was only optimal because the bug let phase be teleported for free.

**Ng, Harada, Russell, ICML 1999.** Potential-based shaping preserves the optimal policy only in the limit; with finite training it changes *which* optimum is found first. Our Φ is potential-based, so it does not change the true optimum — it changes what is findable. Correct diagnosis of the mechanism.

### A5. Terminal-approach RL (dominant, mostly not transferable except in style)

- **Gaudet, Linares, Furfaro** — 6-DOF hovering / terminal adaptive guidance via reinforcement meta-learning (<https://arxiv.org/abs/1911.08553>, and the 6-DOF landing line in *Adv. Space Res.* 2020). Sensor-to-thrust, recurrent (meta-RL) policies adapting to unknown dynamics. Horizon: seconds-to-minutes. What transfers: shaping toward a *guidance law* (velocity-field/glideslope tracking) rather than toward the goal state.
- **Hovell & Ulrich, *JSR* 58(2), 2021** <https://carleton.ca/spacecraft/wp-content/uploads/sites/229/JSR-2021.pdf>. "Deep **guidance**, not deep control": the policy emits velocity commands that a conventional controller tracks. Lowers learning burden, enables sim-to-real onto a granite table. Direct analogue for us: **let the policy emit a commanded δa / drift rate; let a classical law realize it.**
- **"Robust trajectory design and guidance for far-range rendezvous…," *Aerosp. Sci. Technol.* 2025** (<https://doi.org/10.1016/j.ast.2025.109930>; ADR variant <https://arxiv.org/abs/2411.01021>). Despite "far-range" in the title, **the trajectory (i.e. the phasing) is designed by PSO / a nonlinear optimizer, and PPO only supplies the closed-loop correction around it.** Reports 22.31% Δv reduction vs best benchmark. The architecture split is worth copying if the pure-RL route stalls.
- **Guffanti, Gammelli, D'Amico, Pavone, "ART," IEEE Aero 2024 / arXiv:2310.13831.** ROE state, transformer trained on 400k optimal trajectories to warm-start a sequential convex optimizer (keeping hard-constraint guarantees). OCP horizon `[1,3]` orbits at 416 km. Confirms the field's horizon norm.
- **arXiv:2511.11402, transformer(GTrXL)+PPO multi-phase trajectory optimization.** Reaches 1.02–1.03× optimal on benchmarks, 1.8%/1.5%/0.3° orbital-insertion error. Relevant only for its *observation* trick: **augment the observation with global normalized time and an explicit phase index**, and smooth target transitions over 5 steps to avoid observation discontinuities at regime boundaries.

### A6. State representation

- **"RL-based station keeping using relative orbital elements," *Adv. Space Res.* 2025** (<https://doi.org/10.1016/j.asr.2025.05.031>) and **OrbitZoo** (<https://arxiv.org/abs/2504.04160>). ROE decouple secular from short-period; `δa` drives the secular drift of relative mean longitude `δλ`. OrbitZoo additionally recommends **equinoctial** elements to dodge circular/equatorial singularities — relevant to us since our headline distribution is `e ~ U(0, 0.05)` where `θ`/`ω` are ill-conditioned.
- OrbitZoo explicitly names the open problems in orbital RL as **sparse/delayed feedback, credit assignment, reward misspecification, and curriculum**. We are squarely inside their stated open-problem set.

### A7. Sparse reward / curriculum / warm-start pathology

- **Florensa et al., "Reverse Curriculum Generation," CoRL 2017** <https://arxiv.org/abs/1707.05300>. Expand the **start-state distribution** outward from the goal, gated on measured success rate (keep starts whose success ∈ [0.1, 0.9]). Our fixed 30°→90°→180° schedule is a coarse hand-made version; the adaptive version is strictly better and trivially cheap here (sample `init_phase_gap` from a moving frontier).
- **Andrychowicz et al., HER, NeurIPS 2017** <https://arxiv.org/abs/1707.01495>, plus **"Impulsive maneuver strategy for multi-agent orbital pursuit-evasion under sparse rewards," *AST* 2024** (DDPG+HER+hierarchical net on *impulsive orbital* maneuvers — existence proof outside robotics). HER is a natural fit because our goal is part of the observation: an episode ending 40° short is a perfect demonstration for a 40°-smaller gap. **Cost: requires an off-policy learner; PufferLib PPO is not.**
- **Abbas et al., "Loss of Plasticity in Continual Deep RL," CoLLAs 2023** <https://arxiv.org/abs/2303.07507>; **Dohare et al., *Nature* 632, 2024** <https://www.nature.com/articles/s41586-024-07711-7>. Directly explains our R3/R5 result that warm-started continuation collapses. Only intervention that maintained plasticity indefinitely: **continual backprop** (reinitialize a small fraction of least-used units every step). Also: shrink-and-perturb, ReDo dormant-unit resets, full head re-init. ~20 lines in a PufferLib training loop.
- **"Comparing Behavioural Cloning and RL for Spacecraft G&C Networks," arXiv:2507.19535 (ESA ACT).** BC nails the optimal control *structure* in deterministic settings and is dataset-bound; RL is better under stochasticity and can beat suboptimal demos. Explicitly frames IL as a **warm start for (meta-)RL**.
- **Tipaldi, Iervolino, Massenio, *Annual Reviews in Control* 54, 2022.** The survey to cite for "the field is terminal-phase dominated."

### A8. Dynamics-change / re-training after a sim fix

There is **no astrodynamics paper on retraining after a simulator correction.** The nearest usable framing is domain/dynamics randomization (OpenAI dexterous manipulation 2018; sim-to-real surveys arXiv:2009.13303, arXiv:2008.06686) and offline DR (DROPO). Federici & Zavoli's disturbance-randomized training is the astro instance. Practical implication for us: **randomize the physics we're least sure of (e.g. a small randomized along-track perturbation per burn, ±1σ) so no future correction of that magnitude can invalidate a checkpoint again.** That converts our bug class from "invalidates everything" to "already in the training distribution."

---

## Quantitative grounding computed for our env (`t3_phasing_design_table.py`)

At 550 km circular, `v = 7589 m/s`, `T = 95.5 min`, `n = 226.2°/hr`. Drift rate `dλ/dt = −1.5·n·(δa/a)`:

| δa | drift | Δv (2-impulse round trip) | hours to close 180° | capture window (‖Δr‖<30 km) |
|---|---|---|---|---|
| 10 km | 0.49 °/hr | 11.0 m/s | 367 hr | 60.8 min = 12.2 warp-5min steps |
| 23 km | 1.13 °/hr | 25.2 m/s | 160 hr | 26.4 min = 5.3 warp-5min steps |
| 50 km | 2.45 °/hr | 54.5 m/s | 73 hr | 12.2 min = 2.4 warp-5min steps |
| 100 km | 4.90 °/hr | 108 m/s | 37 hr | 6.1 min = **1.2** warp-5min steps |
| 116 km | 5.69 °/hr | 126 m/s | 32 hr | 5.2 min = **1.05** warp-5min steps |
| 200 km | 9.80 °/hr | 215 m/s | 18 hr | 3.0 min = **0.6** warp-5min steps |

Cheapest phasing design that fits a 2000-decision episode, by horizon model (`Δθ = 180°`, 550 km):

| horizon model | max sim time | N revs | δa | Δv | Δv/budget |
|---|---|---|---|---|---|
| coast-only (60 s/decision) | 33.3 hr | 20 | −116 km | **130 m/s** | 27% |
| warp-5min | 167 hr | 100 | −23 km | **25.4 m/s** | 5.3% |
| warp-1hr | 2000 hr | 100 | −23 km | 25.4 m/s | 5.3% |

Five consequences, all actionable:

1. **The Δv budget (478 m/s) is never binding.** Even the worst case (300 km altitude, 90° gap, coast-only horizon, forced to phase the *long way up* because dropping perigee below 200 km would reenter) costs 178 m/s = 37% of budget. Any shaping term that trades Δv against time is optimizing the wrong resource.
2. **Time is binding, so warp usage is the whole game.** The Δv-optimal solution is 5× cheaper than the coast-only-reachable one. The previously-observed "96.8% of actions are warp-5min" was the policy doing the right thing for the wrong reason.
3. **Fast drift destroys terminal observability.** At the δa needed for a coast-only 180° close (−116 km), the 30 km capture window is 5.2 min ≈ **one** warp-5min step. The policy literally cannot resolve arrival. Shallow drift (δa ≈ −20 km) gives a 26-min window = 5 warp steps. **Shallow-and-slow is both fuel-optimal and controllability-optimal** — a rare alignment, and the shaping should push there hard.
4. **The gate is set exactly wrong.** `σ₂ = sigmoid((2.0 − φ_orbit)/0.2)` with `φ_orbit = |Δa|/10km + |Δe⃗|` opens only for `|Δa| ≲ 20 km`. The Δv-optimal drift is δa ≈ −23 km — **just outside** the gate; the horizon-limited drift (−116 km → φ_orbit ≈ 11.6) is far outside. Under corrected dynamics the shaping switches off the phase reward for precisely the maneuver that is optimal, while `φ_orbit` simultaneously penalizes it. Hypothesis confirmed quantitatively.
5. **The success box is position-limited, not velocity-limited.** 30 km ⇒ 0.248° of phase. 50 m/s rel-vel is satisfied by δa up to 91 km or |Δe| up to 0.0066, whereas 30 km position requires |Δe| ≲ 0.0043 and δa ≈ 0. So the terminal task is: *null δa and arrive at phase zero at the same instant.* That is a two-variable simultaneous-convergence problem — exactly what a-augmentation was invented for.

---

## (b) Five most transferable design patterns, ranked

### 1. Phase-augmented target semi-major axis in the shaping potential (**do this first**)
*Sources:* Naasz 2002; Lanktun via Holt et al. 2024 §6; arXiv:2606.12108.
Replace `φ_orbit = |a_sat − a_target|/10km` with `|a_sat − a*_target(Δθ, N_tt)|/10km`, where
```
a*_target = ( (1/a_target)^{3/2} + W_L · sgn(Δθ)·arccos(cos Δθ) / exp(N_tt) )^{-2/3}
Δθ  = wrap(θ_sat − θ_target) ∈ (−π, π]
N_tt = revs-to-go estimate = |Δθ| / (2π · |1.5·(δa/a)|)  ... or a fixed budget-based proxy
a*_target clipped to [0.97·a_target, 1.03·a_target]     (±3% ≈ ±207 km at 550 km; ±20% of the Lyapunov paper is far too loose for LEO)
```
Concrete parameters for our env: choose `W_L` so that at `Δθ = 180°` and `N_tt` large the commanded offset lands near the Δv-optimal **δa ≈ −25 km**, and at `Δθ = 30°` near **δa ≈ −5 km**; a linear surrogate `δa_cmd = −clip(K·Δθ, 5 km, 40 km)` with `K ≈ 25 km / π rad` reproduces the table above to within 10% and is easier to reason about than the exponential form. Sign convention: chaser behind ⇒ lower `a`.
Consequences: (i) `φ_orbit` and `φ_phase` stop being antagonistic — a correct drift now *reduces* `φ_orbit`; (ii) the `σ₂` gate can be **kept** (it now opens along the correct trajectory) but its threshold should be re-derived against `a*` rather than `a`; (iii) at `Δθ → 0`, `a* → a_target`, so the terminal condition is unchanged and the NHR clamp is untouched.

### 2. Shape toward a *reference trajectory*, not toward the *goal state*
*Sources:* LaFarge et al. 2020; Gaudet/Furfaro velocity-field shaping; far-range RPO 2025 (PSO nominal + PPO correction).
If (1) is not enough, generate the classical phasing reference at reset (it's closed-form: `T_ph`, `a_ph`, burn epochs — no Lambert needed) and shape on distance to the *nearest point of that reference*, LaFarge-style: `r = η·exp(−λ·k)`, `η = i/(n·ξ) + 1` to ramp reward along the path and kill the "hover on the initial orbit" local optimum. Suggested scaling for us: normalize `k` by (30 km, 5 m/s) then `λ ∈ [3, 10]`; terminate the episode on gross deviation (their `p_d = −4`) — early termination on divergence is a large part of why their signal is learnable, and we currently have no such term. This is also the honest fallback framing for the portfolio: *RL provides the closed-loop correction around a classically-designed nominal*, which is what the 2025 far-range paper actually publishes.

### 3. Make coast duration a first-class, *factored* action (semi-MDP done properly)
*Sources:* TempoRL (ICML 2021); FiGAR (ICLR 2017); Space Sci.Tech. 0047 (Δv, Δt joint action); propulsionless-phasing 2021 (surrogate propagation).
Our 16 discrete actions entangle "what burn" with "how long to wait" — 9/10/11 are coast-only, so any burn is forced to a 60 s step. Two upgrades, in increasing cost:
- **Cheap (FiGAR):** add a second discrete head `τ ∈ {1, 5, 30, 60}` sub-steps applied to *any* action, giving 16×4 combinations without 64 logits. Keeps the `γ^τ` SMDP shaping already implemented.
- **Better (TempoRL):** condition the τ-head on the sampled action. Reported to be robust as max skip grows, which matters because our useful τ range spans 60×.
Design constraint from the capture-window table: **the policy must retain a τ=1 (60 s) option near arrival** — at δa = −116 km the capture window is 5 min. A curriculum that only ever offers warp-5min at 116 km drift is unsolvable by construction.

### 4. Adaptive reverse curriculum on `init_phase_gap`, with plasticity maintenance at every stage boundary
*Sources:* Florensa et al. CoRL 2017; Abbas et al. 2023; Dohare et al. *Nature* 2024; LaFarge's 60-agent selection.
- Replace the hard 30°→90°→180° ladder with a **success-gated moving frontier**: sample `init_phase_gap_max` from the frontier, expand when rolling success ∈ [0.1, 0.9], contract otherwise. This removes the discrete distribution shocks that our R3/R5 warm-start experiments died on.
- At *any* warm-start or distribution shift, apply **shrink-and-perturb** (`w ← α·w + σ·ε`, α ≈ 0.4–0.8) on the trunk and **re-init the policy head**, or run continual-backprop-style reinitialization of the least-used units. Our documented finding "untrained warm-start ckpt beats every trained variant" is textbook plasticity loss.
- Budget **≥5 seeds and select**, per LaFarge's explicit 60-agent practice. Our seed-1337 Stage-1 zero is expected variance, not a defect.

### 5. Observe relative orbital elements (and an energy invariant), not raw elements
*Sources:* RL-ROE station-keeping 2025; ART (ROE); OrbitZoo (equinoctial); LaFarge (Jacobi constant in the observation "less optimal without it"); arXiv:2511.11402 (global normalized time + phase index in the observation).
Add to the observation, cheaply and non-breaking:
- `δa/a` (dimensionless relative semi-major axis) — **this is the control variable for phasing**, and it is currently only implicit;
- `δλ` = relative mean longitude (wrapped), and its rate `d(δλ)/dt = −1.5·n·δa/a` — these two form the 2-D phase plane in which the optimal policy is a simple bang-coast-bang;
- relative eccentricity vector `δe⃗ = (δe_x, δe_y)` instead of scalar `e` (removes the `ω`/`θ` singularity at `e→0`, which our headline `e ~ U(0,0.05)` distribution sits on top of);
- specific orbital energy (or `a`-derived invariant) for chaser and target — LaFarge's Jacobi-constant analogue;
- **normalized time-remaining** `(MAX_STEPS − step)/MAX_STEPS` and **cumulative sim-time-remaining**, because with warp actions the step count and the elapsed time decouple and the MDP is otherwise non-stationary from the agent's point of view. This is likely a real Markov violation in the current env.

*Runner-up, not in the top 5 but worth costing:* **HER** (relabel a 140°-closed episode as a successful 140° episode). It is the single highest-leverage sparse-reward tool and there is orbital-impulsive precedent (AST 2024), but it requires an off-policy learner and would mean leaving the PufferLib PPO path.

---

## (c) What nobody has shown working — where we are off the map

1. **End-to-end deep RL closing a large (≥90°, up to ±180°) co-orbital phase gap over tens-to-hundreds of revolutions with impulsive burns.** No paper found does this. Every "rendezvous" RL result is either (a) CW/Hill terminal approach, ≤3 orbits, ≤30 km, or (b) has the phasing designed by a classical/global optimizer with RL only doing closed-loop correction. The one paper that genuinely targets phasing (Adv. Space Res. 2021) uses *differential drag*, not impulses, and needed a surrogate propagator.
2. **RL discovering the drift-orbit strategy on its own.** In every classical treatment, the phase-closing behavior is *installed* by a-augmentation or a reference trajectory. The RL literature that succeeds at rendezvous either inherits it (Holt 2024) or sidesteps it (terminal-only). There is no published demonstration of the drift-then-brake structure *emerging* from a reward function.
3. **Time-warp / variable-Δt macro-actions in an astrodynamics RL environment where the horizon spans 100+ revolutions.** TempoRL/FiGAR are Atari/control-suite; the (Δv, Δt) rendezvous paper is CW-linear and few-impulse. Nobody has reported the failure modes of a 60× skip range against a terminal capture window of a few minutes — which is the precise regime we are in.
4. **Retraining/transfer after a *simulator correction*.** No astrodynamics literature. Dynamics randomization is the only adjacent tool, and it is used prospectively (for robustness) never retrospectively (to recover from a discovered sim bug). Our T1 finding — canonical ckpt 97.5% → 13%, best gentle re-adapt 33.5%, fresh training flat — is, as far as this survey found, an *unpublished* and reportable result about reward-hacking of a dynamics bug.
5. **Combined shaping + curriculum + plasticity interaction on a warm start.** Each is studied alone. Our own R3/R5 results (any reward reshape on a committed warm start collapses to <10%; plasticity intervention is a prerequisite) are a genuine negative result with no literature counterpart.

**Reportable framing:** items 1, 3, 4 and 5 are all legitimate "here is what the literature doesn't cover, here is what we measured" contributions for the Draper narrative — the limitations-first framing already chosen. The honest positioning of the *technical* recovery is: **install the classical structure (a-augmentation) into the shaping and the observation, let RL do what RL is good at (closed-loop correction, robustness, timing under uncertainty), and measure against the classical expert** — which is exactly what the two strongest papers in this survey (Holt 2024, far-range RPO 2025) each independently concluded.