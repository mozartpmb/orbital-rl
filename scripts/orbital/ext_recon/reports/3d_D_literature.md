## 3D-D — Literature: 3D RL rendezvous, relative orbital elements, plane-change guidance

Artifacts (mine; siblings also writing into the same dirs — no filename collisions):
`/Users/pete/space_training/scripts/orbital/ext_recon/{ext_plane_change_table.py, ext_oop_window.py, ext_u_lattice.py, ext_j2_nodal.py, ext_pdftext.py}` · extracted paper text `ext_chernick.txt`, `ext_issfd2011.txt`, `ext_lafarge2022.txt` · CSVs `/Users/pete/space_training/web_data/results/{ext_3d_planechange.csv, ext_3d_oop_windows.csv}`.

---

### 0. Headline (read first)

1. **Full 3D adds exactly TWO relative-state dimensions, not six.** In the D'Amico quasi-nonsingular ROE, the 6-D relative state is `δα = [δa, δλ, δe_x, δe_y, δi_x, δi_y]`. Our `phase_obs_mode=1` observation already *is* the 2D specialization: `obs[13,14] = sin/cos(Δλ)` with `λ = M+ω` **is δλ** with `i=Ω=0, η≈1`. 3D appends only the relative inclination vector `δi = [i_s−i_t, (Ω_s−Ω_t)·sin i_t]`.
2. **The Δv budget hard-caps the campaign scope.** 478 m/s buys **3.54° of pure plane change at 300 km**, and our measured headline policy already spends p50 235 m/s on phasing → residual headroom ≈ 243 m/s ⇒ **δi_max ≈ 1.8° at LEO if everything left is spent on the plane**. Any "interesting 3D" claim above ~2° is budget-infeasible without changing the fuel model.
3. **The interesting δi regime is 0.03–0.25°, because that's where the plane error is the same order as the success box.** Cross-track miss = `a·|δi|`; nulling cost = `v·|δi|`. At 550 km: δi=0.043° = the 5 km tight box (free, no burn needed); δi=0.25° = 30 km far-field box = 33 m/s.
4. **Our warp actions structurally destroy out-of-plane burn precision — measured.** One normal impulse must fire at `u* = atan2(δi_y, δi_x)`; efficiency is `cos(Δu)`, so ≤1% loss needs `|Δu| ≤ 8.11°`. **warp-1h advances u by 226.2° at 550 km ⇒ worst-case cosine efficiency 0.000** (can land antipodal, sign-flipped). warp-5min → 0.987, coast-60s → 0.999. Mitigation exists and is measured (below).
5. **Without J2, the 3D task is a strictly additive one-shot cost with no strategy. With J2 it acquires the *same* drift-orbit structure that made 2D interesting.** Differential nodal precession `δΩ̇/Ω̇ = −3.5·δa/a`: a 100 km drift orbit at i=51.6°, 550 km yields **0.326° of free ΔΩ over the 33.3 h clock (= 33.8 m/s equivalent), 0.98° / 101 m/s over a 200 h clock** — a large fraction of the entire affordable δi range, for free, on a drift orbit the policy is already opening for phasing. This is the strongest research framing available for ext-3d.

---

### (a) Annotated bibliography

#### A1 — ROE formalism (the load-bearing group)

**Chernick & D'Amico, "Closed-Form Optimal Impulsive Control of Spacecraft Relative Motion Using Reachable Set Theory," JGCD 44(1):25, 2021** — [arXiv:2002.07832](https://arxiv.org/abs/2002.07832) (full text extracted to `ext_chernick.txt`). *The single most transferable paper in this survey.* Gives verbatim:
- qns-ROE definition (Eq. 3): `δa=(a_d−a_c)/a_c`; `δλ_e = M_d−M_c + η(ω_d−ω_c + (Ω_d−Ω_c)cos i_c)`; `δe_{x,y} = e_d cos/sin ω_d − e_c cos/sin ω_c`; `δi_x = i_d−i_c`; `δi_y = (Ω_d−Ω_c) sin i_c`.
- **Singularity statement, critical for us:** "*valid for circular chief orbits (e_c = 0) but becomes singular for strictly equatorial chief orbits (i_c = 0)*". **Our env is exactly i=0.**
- Control input matrix Γ (Eq. 6), out-of-plane column: `Δδi_x = η cos θ /(na(1+e cos ν)) · δv_N`, `Δδi_y = η sin θ /(...) · δv_N`. In-plane column has **zero** entries on δi.
- Decoupling: at e≈0 "*in-plane maneuvers (radial, tangential) affect only δa, δλ, δe, and out-of-plane maneuvers (normal) affect only δi*." **At nonzero e this breaks** — normal burns perturb δe. Fix given: redefine `δe' = [e_d−e_c, ω_d−ω_c+(Ω_d−Ω_c)cos i_c]` (Eq. 5), under which decoupling is exact at arbitrary e. **Directly relevant to our e≤0.30 / e≤0.50 wide envelopes.**
- Impulse counts: "*To control a single 2D state, such as the out-of-plane ROE, one maneuver is required if the desired pseudo-state lies in both S and S\*, and two if just in S\*.*"

**Spurmann & D'Amico, "Proximity Operations of On-Orbit Servicing Spacecraft Using an Eccentricity/Inclination Vector Separation," ISSFD 2011** — [PDF](https://issfd.org/ISSFD_2011/S2-Formation.Flying-FF/S2_P4_ISSFD22_PF_034.pdf) (extracted to `ext_issfd2011.txt`). Operational statement of the same algebra:
- "*In-plane and out-of-plane relative orbit control problems are decoupled. One cross-track maneuver is necessary and sufficient to control the relative inclination vector, while two in-plane maneuvers are necessary and sufficient to correct relative eccentricity vector and relative semi-major axis.*" ⇒ **3 impulses total for a full 6-D reconfiguration.**
- Maneuver placement (Eq. 18): cross-track burns at `u_N = atan2(δi_y, δi_x)` (and +π), with `δv_N = n·a·|δi|`; radial burns at `u_R = atan2(δe_y, δe_x)`. Their worked plan: `δv_N = 0.162 m/s` per burn.
- "*radial maneuvers are two times more expensive than along-track pulses for changes of the relative eccentricity vector and do not affect the semi-major axis.*"
- **e/i-vector separation passive safety:** min separation perpendicular to flight `≥ f(δe, δi)`; for bounded motion (δa=0) minimum collision risk at **parallel or anti-parallel δe/δi vectors** (φ=ϑ or φ=ϑ+π). Also: along-track pulses are inherently *less* safe than radial because `δa < 2δe` shrinks the min separation.
- Origin paper: D'Amico & Montenbruck, JGCD 2006 ([SLAB](https://slab.stanford.edu/)); flight heritage GRACE, TanDEM-X, PRISMA ([DLR PRISMA](https://www.dlr.de/en/rb/research-operation/research-projects/flight-dynamics-navigation-and-orbital-sustainability/gnss-technology-and-navigation/past-projects/prisma-formation-flying-misson)).

**Gaias & Ardaens/D'Amico, "Impulsive Maneuvers for Formation Reconfiguration Using Relative Orbital Elements," JGCD** — [10.2514/1.G000189](https://arc.aiaa.org/doi/10.2514/1.G000189); **"Trajectory Design for Proximity Operations: The ROE Perspective," JGCD** — [10.2514/1.G006175](https://arc.aiaa.org/doi/10.2514/1.G006175); **MPC on D'Amico ROE, Astrodynamics 2024** — [10.1007/s42064-024-0214-8](https://link.springer.com/article/10.1007/s42064-024-0214-8). Supporting; MPC paper notes ROE constraints reduce fuel vs CW/HCW formulations.

#### A2 — RL with ROE / 3D orbital state

**Tafanidis, Banerjee, Satpute, Nikolakopoulos, "Reinforcement learning-based station keeping using relative orbital elements," *Adv. Space Res.* 76:750–763, 2025** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0273117725004533) / [ADS](https://ui.adsabs.harvard.edu/abs/2025AdSpR..76..750T/abstract). SAC observing ROE between actual and desired trajectory, outputting a finite-pulse maneuver plan; decentralized, deployable on space-grade compute. **The only found precedent for ROE-as-RL-observation.** Station-keeping, not rendezvous.

**Guffanti, Gammelli, D'Amico, Pavone, "ART" (IEEE Aero 2024)** — [arXiv:2310.13831](https://arxiv.org/abs/2310.13831); **Takubo, Gammelli, Pavone, D'Amico, "Agile Tradespace Exploration for Space Rendezvous Mission Design via Transformers"** — [arXiv:2510.03544](https://arxiv.org/abs/2510.03544). ROE state, transformer warm-starts a sequential convex optimizer (hard constraints retained). OCP horizon `[1,3]` orbits at 416 km. Learning does *not* produce the maneuvers.

**OrbitZoo, arXiv:2504.04160** — [v3 HTML](https://arxiv.org/html/2504.04160v3). Full 3D, PPO/DDPG/DQN. Action = **polar parameterization (thrust magnitude, deviation angle, azimuthal angle)** — the cleanest published 3D thrust-direction encoding. Explicitly: "*equinoctial elements avoid such issues and are often preferred in RL applications for their robustness*" vs Keplerian singularities at circular/equatorial. Names the open problems as sparse/delayed feedback, credit assignment, reward misspecification, curriculum.

#### A3 — RL that touches plane change / 3D transfer

**Kolosa, PhD, Western Michigan, "A Reinforcement Learning Approach to Spacecraft Trajectory Optimization," 2020** — [abstract](https://scholarworks.wmich.edu/dissertations/3542/). DDPG, actor outputs thrust magnitude from an orbital-element state. Three problems: generalized orbit change, SMA change, **inclination change**. Abstract wording is telling: the first two were solved "*with no prior knowledge of the environment's dynamics*", whereas inclination change was framed as a *robustness/generalization* test with randomized initial states. Low-thrust continuous; the plane change is just another element error, never a timed impulsive decision.

**ERAU dissertation, "Orbital Maneuvers and Interplanetary Trajectory Design via Reinforcement Learning"** — [commons.erau.edu/edt/924](https://commons.erau.edu/edt/924/). PPO + SAC over orbit-raising, **inclination change**, combined maneuvers, asteroid rendezvous; Gauss variational equations in **modified equinoctial elements**.

**Multi-phase Transformer-RL, arXiv:2511.11402** — [HTML](https://arxiv.org/html/2511.11402). Full 3D ECI. GTrXL (6 blocks, 384-d, 8 heads, 256 memory) + PPO. **Reported orbital insertion errors: a 1.8%, e 1.5%, i 0.3°.** Two directly reusable tricks: (i) observation augmented with **normalized global time τ and explicit phase index φ**; (ii) **targets transition smoothly over the first 5 steps of each phase** rather than discontinuously.

**Federici & Zavoli / Zavoli & Federici** — [arXiv:2008.08501](https://arxiv.org/pdf/2008.08501), *Adv. Space Res.* 72(4) 2023. MEE + GVE, 3D low-thrust. States the sparse-vs-dense tradeoff: dense shaping biases toward the designer's assumed solution.

**"Revisiting Space Mission Planning: RL-Guided Multi-Debris Rendezvous," arXiv:2409.16882** — [HTML](https://arxiv.org/html/2409.16882v1). **The decisive negative datapoint.** State = all six Keplerian elements (i and Ω present) + Cartesian positions of all debris. But the action space is *which debris next*; **the plane change is computed by Izzo's Lambert solver, not learned.** Reward `R_t = −T_t/T_max`, +1 on completion.

**Terminal-approach 3D RL (style only, horizon ≤3 orbits, ≤30 km):** Gaudet/Linares/Furfaro 6-DOF meta-RL [arXiv:1911.08553](https://arxiv.org/abs/1911.08553); Hovell & Ulrich "deep *guidance*, not deep control" [JSR 58(2) 2021](https://carleton.ca/spacecraft/wp-content/uploads/sites/229/JSR-2021.pdf); "Optimal Multi-impulse Linear Rendezvous via RL," [Space: Sci. Technol. 0047](https://spj.science.org/doi/10.34133/space.0047) (action = joint (Δv, Δt)); SE(3) rendezvous with rotating target ([RG](https://www.researchgate.net/publication/392845799)); Tipaldi et al., *Annual Reviews in Control* 54, 2022 (survey: field is terminal-phase dominated).

#### A4 — Classical combined-maneuver guidance

Vallado ch.6 / Curtis ch.6; [Braeunig orbital mechanics](http://www.braeunig.us/space/orbmech.htm); [Pressbooks ch.7 Maneuvering](https://oer.pressbooks.pub/lynnanegeorge/chapter/chapter-7-manuvering/); [HawkLogic Δv tool](https://hawklogicsystems.com/tools/delta-v-maneuver). Canonical results: simple plane change `Δv = 2v sin(Δi/2)`; combined maneuver by law of cosines `Δv_k = √(v_i² + v_f² − 2 v_i v_f cos Δi_k)`; optimal split minimizes `Σ Δv_k` s.t. `Σ Δi_k = Δi`. **I computed the split numerically for our bands rather than trusting the folk rule — see (b)#4; the folk rule is wrong for LEO→LEO.** For pure inclination correction, [operational practice](https://issfd.org/2015/files/downloads/papers/166_Junker.pdf) is paired burns near `u = ±90°` from the node with small cosine-efficiency losses off-optimum.

#### A5 — Curriculum / dimensionality expansion

**Farquhar, Gustafson et al., "Growing Action Spaces," ICML 2020** — [arXiv:1906.12266](https://arxiv.org/abs/1906.12266), [PMLR v119](https://proceedings.mlr.press/v119/farquhar20a.html). Curriculum of progressively growing action spaces; off-policy value estimation for multiple action spaces simultaneously, transferring data/values/representations from restricted to full action set. **Off-policy — not directly usable under PufferLib PPO** (see (b)#5 for the on-policy substitute).

**LaFarge, Howell, Folta, AIAA SciTech 2022, "An Autonomous Stationkeeping Strategy for Multi-Body Orbits"** — [PDF](https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2022_AIAA_LafHowFol.pdf) (extracted `ext_lafarge2022.txt`). Three transferable items: (i) action ∈ R⁵ = `[f, u_x, u_y, u_z, τ]` with direction L2-normalized and τ (thrust time) as a *learned action component*; a cited alternative (Federici) uses a **trig encoding with the cosine replaced by a sign function, "extrapolated to the spatial case by including two additional parameters to model out-of-plane direction"**; (ii) a **separate "timing environment"/agent that learns *when* to maneuver** — the RL precedent for treating maneuver location as its own decision; (iii) episodes use a **randomized coast arc `Δt ~ U(Δt_min, Δt_max)`**, explicitly to "*expose the agent to many locations along the orbit, and discourage policies that rely on one particular maneuver frequency*." Also, again: "*many agents are trained in parallel… a specific controller is selected*" — multi-seed selection is the norm.

Adjacent, weak: [Frontiers curriculum for high-DOF manipulators](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1066518/full); [2D→3D thermal-control policy transfer at 10% of 3D training cost](https://www.sciencedirect.com/science/article/abs/pii/S1290072923004799); [multi-objective RL for libration transfers](https://ntrs.nasa.gov/api/citations/20210018983/downloads/Multi_Objective_Reinforcement_Learning_for_Low_Thrust_Transfer_Design_between_Libration_Point_Orbits.pdf). CR3BP work uses transfer learning across *fidelity* (CR3BP → ephemeris), never across *dimensionality*.

---

### (b) Five most transferable design choices, with parameters

#### 1. Adopt qns-ROE as the relative-state observation. 3D costs +2 dims.
Exact mapping against our 38-dim obs (`orbital.h:600–757`):

| ROE | present today? | where |
|---|---|---|
| `δa` | **implicit** — network must subtract `obs[0]−obs[7]` | add `obs[N] = (a_s−a_t)/a_t` explicitly (free win, not 3D-specific) |
| `δλ` | ✅ **exact** | `obs[13,14] = sin/cos(Δλ)`, `λ=M+ω` — our T3 fix independently rediscovered ROE `δλ_e` at `η→1, i=Ω=0` |
| `δe_x, δe_y` | **implicit** — ingredients in `obs[1,8]` (e) + `obs[9-12]` (sin/cos ω) + `obs[16]=cos(ω_s−ω_t)` | add `δe_x = e_s cos ω_s − e_t cos ω_t`, `δe_y = …sin…`. Note our *shaping* already uses `|Δē|` — the potential is ROE-correct, the observation isn't |
| `δi_x, δi_y` | ❌ **the only genuinely new state** | `δi_x = i_s−i_t`, `δi_y = (Ω_s−Ω_t) sin i_t` |
| LVLH `obs[33-37]` | 2D (x,y,vx,vy,+1) | add `z, vz`; the `+1` pad slot is already there |

Normalization: scale `δi` so it shares units with the LVLH channels — `obs = a·δi_{x,y} / lvlh_scale_m` (i.e. observe the **cross-track offset in metres**, not radians). That makes δi commensurate with the success box automatically at every altitude lineage (LEO 1.5e7, wide 1.5e7, MEO 4e7) with no new scale kwarg. Absolute `i_t` must also be observed (as `sin/cos i_t`) because it is the conditioning factor on `δi_y`.

**Trap, from Chernick verbatim: qns-ROE is singular at `i_c = 0` — exactly where our 2D env sits.** At `i_t=0`, `δi_y ≡ 0` for any ΔΩ and Ω itself is undefined. A naive "warm-start 2D → ramp δi" curriculum starts *on the singularity*. **Curriculum must ramp absolute inclination away from zero FIRST, then ramp δi**: stage 0 sample `i_t ∈ [20°, 100°]` with `δi = 0` (pure re-anchoring, task unchanged), only then open `δi`. Alternative: use non-singular/equinoctial `(h,k) = tan(i/2)·(cos Ω, sin Ω)` — note `(h,k)` *is* the inclination vector in equinoctial form, so this is the same object, just conditioned at i=0 instead of at i=90°.

#### 2. Out-of-plane control law: one normal impulse, at `u* = atan2(δi_y, δi_x)`, `Δv_N = n·a·|δi|`.
Necessary and sufficient (Spurmann/D'Amico verbatim; Chernick reachable-set proof). Scaling law to build the expert and the potential on:
```
cross-track miss = a·|δi|        Δv_null = v·|δi| = n·a·|δi|
⇒ Δv per km of cross-track offset = n  (1.10 m/s/km @550 km, 0.15 m/s/km @20,200 km)
```
**Preserve decoupling at high e** by switching the observed/shaped eccentricity state to Chernick Eq. (5) `δe' = [e_d−e_c, ω_d−ω_c+(Ω_d−Ω_c)cos i_c]` — otherwise normal burns perturb `δe` (nonzero third column of Γ at e≠0) and the wide-e lineages (WL4 e≤0.30, M5 e≤0.50) get a hidden cross-coupling that the potential can't see. Also budget-relevant: radial burns cost **2×** along-track for the same `δe`.

#### 3. Scope δi from the budget. Recommended curriculum ladder (measured, `ext_3d_planechange.csv`).
Max pure plane change on the *whole* 478 m/s: 3.54° @300 km · 3.67° @800 · 5.20° @8000 · 7.08° @20,200. Realistic ceiling after phasing (p50 235 m/s spent) ≈ **1.8° at LEO**.

| rung | δi | cross-track @550 km | Δv_null | % budget | rationale |
|---|---|---|---|---|---|
| 3D-0 | 0 (i_t ∈ [20°,100°]) | 0 | 0 | 0 | de-singularize; must reproduce 200/200 |
| 3D-1 | ≤0.04° | ≤4.8 km | ≤5.3 m/s | 1.1% | **inside the 5 km tight box — a free rung; tests obs plumbing only** |
| 3D-2 | ≤0.10° | ≤12 km | ≤13 m/s | 2.8% | inside 30 km far box, outside 5 km |
| 3D-3 | ≤0.25° | ≤30 km | ≤33 m/s | 6.9% | **δi ≈ box — the phase transition; this is the headline rung** |
| 3D-4 | ≤0.5° | ≤60 km | ≤66 m/s | 13.9% | plane cost enters the same order as a Hohmann leg |
| 3D-5 | ≤1.0° | ≤121 km | ≤132 m/s | 27.7% | stretch; competes directly with phasing |
| — | 2.0° | 242 km | 265 m/s | 55.4% | **infeasible with p50 phasing spend — do not claim** |

Note the free-rung logic: at 550 km the 5 km box corresponds to `δi = 0.0414°` and the 30 km box to `0.2484°`. State it explicitly in any writeup — otherwise "we did 3D at δi=0.03°" reads as a stunt when it is actually the correct first rung.

#### 4. The optimal combined-maneuver split is ≈50/50 for LEO→LEO, NOT "all at apoapsis."
Measured (`ext_3d_planechange.csv`, 300→800 km, Hohmann-only = 274.26 m/s):

| δi | opt fraction at burn 1 | combined total | separate (Hohmann then plane) | saving |
|---|---|---|---|---|
| 0.1° | 0.4774 | 274.58 | 287.28 | 4.4% |
| 0.5° | 0.4767 | 282.14 | 339.33 | 16.9% |
| 1.0° | 0.4743 | 304.55 | 404.39 | 24.7% |
| 2.0° | 0.4649 | 381.18 | 534.50 | 28.7% |
| 5.0° | 0.4033 | 715.29 | 924.68 | 22.6% |

The apoapsis rule is a GTO→GEO rule (large `v_p/v_a` ratio); in our band the two burn velocities differ by ~4%, so the split is near-even. **An expert controller or a potential that hard-codes "plane change at apoapsis" is 17–25% fuel-suboptimal in our band.** Correct plan structure per Chernick/Spurmann is **3 impulses** (2 in-plane for `δa+δe`, 1 normal for `δi` at `u*`), collapsing to 2 only when `u*` coincides with an in-plane burn epoch. Build the expert that way; report policy Δv against the 3-impulse bound, not against Hohmann+separate-plane-change.

#### 5. Solve the burn-window/warp conflict with a node-targeting macro-action — do NOT delete the long warps.
Measured (`ext_3d_oop_windows.csv`). Cosine-efficiency window: ≤1% loss ⇒ `|Δu| ≤ 8.11°`; ≤5% ⇒ 18.19°; ≤10% ⇒ 25.84°. Argument-of-latitude advanced per action:

| alt | 60 s | 5 min | 30 min | 1 h | 3 h | 6 h |
|---|---|---|---|---|---|---|
| 550 km, Δu° | 3.77 | 18.85 | 113.1 | 226.2 | 678.5 | 1357 |
| worst-case cos-eff | 0.999 | 0.987 | 0.551 | **0.000** | **0.000** | **0.000** |
| 20,200 km, cos-eff | 1.000 | 1.000 | 0.991 | 0.966 | 0.706 | 0.000 |

But the quanta *mix* — reachable u-residues mod 360 within N decisions (`ext_u_lattice.py`, worst-case error = half the max gap):

| decisions | 550 km, full set | 550 km, warps only (no 60 s) |
|---|---|---|
| 3 | max gap 22.6° | 22.6° |
| 6 | **1.94°** | 7.57° |
| 12 | 1.70° | 1.89° |

So the machinery already exists — the policy must learn a coarse-then-fine u descent costing ~6 decisions, and **the 60 s coast quantum is load-bearing** (it is the difference between a 1.94° and a 7.57° worst-case landing at 6 decisions). Three recommendations, in order:
- **(a) Keep every warp.** They gave 84.3% terminal-reward visibility and ~150 orders of exploration exponent; removing them re-creates the flatline.
- **(b) Add a node-targeting macro-action** (strongest, ~20 lines): *"coast to `u = u*(δi)` then apply normal burn ±q."* This is the 3D analogue of warp and is exactly classical practice (Spurmann: cross-track maneuvers "always located at the extreme northern or southern latitudes"). It converts a 6-decision timing search into one action and removes lattice aliasing entirely. RL precedent for factoring maneuver-location as its own decision: LaFarge 2022's separate timing agent; TempoRL/FiGAR for the general factorization.
- **(c) Jitter warp durations ±10% per episode** (LaFarge's `Δt ~ U(Δt_min, Δt_max)`, stated purpose: prevent policies keyed to one maneuver frequency). Cheap insurance against u-lattice/period commensurability, which is a live hazard because our altitude band is narrow.

**Normal burn quanta:** at 550 km, `1 m/s normal = 0.912 km cross-track (0.00755°)`, `10 m/s = 9.12 km`. Since the tight box is 5 km, **a ±1 m/s normal quantum is the right fine resolution and ±10 m/s the right coarse one** — i.e. mirror the existing radial ±1/±10 pair. Do not add a 25 m/s normal action; it overshoots every box.

**Bonus recommendation (J2).** If the campaign can afford one physics change, add J2 nodal precession only (`Ω̇ = −1.5 J2 (R_E/p)² n cos i`). Measured payoff (`ext_j2_nodal.py`): at 550 km, i=51.6°, a 100 km drift orbit accrues **0.326° free ΔΩ over 33.3 h (≡33.8 m/s) and 0.977° over 200 h (≡101 m/s)** — comparable to the *entire* affordable δi range. Without it, δi is a constant and the 3D extension is a bolt-on cost; with it, the drift orbit does double duty (phase *and* plane) and the task inherits the drift-then-brake structure the 2D campaign already solved. This is also the mechanism the whole e/i-separation literature exists to manage, so it buys direct literature alignment for the Draper framing.

---

### (c) What nobody has shown — the off-the-map list

1. **A policy that learns the plane change itself, impulsively, for rendezvous.** Every 3D astro-RL result found is one of: (i) low-thrust continuous 3D thrust-direction where inclination is just another element error inside a norm (Kolosa, ERAU, Federici/Zavoli, OrbitZoo, arXiv:2511.11402); or (ii) **RL sequences targets, a classical solver computes the plane change** (multi-debris RL, arXiv:2409.16882, verbatim: the action is "the next debris to be targeted", Lambert does the rest). **No paper found in which a policy chooses when and how much to burn normal to null a relative inclination vector.**
2. **Impulsive out-of-plane burn *timing* learned under coarse temporal abstraction.** The measurement above — that warp-1h yields cosine efficiency 0.000 at LEO while the mixed-quantum lattice recovers to 1.94° in 6 decisions — has no literature counterpart. TempoRL/FiGAR are Atari/control-suite; the (Δv, Δt) rendezvous paper is CW-linear and few-impulse.
3. **ROE (specifically `δi`) as an RL *observation for a rendezvous agent*.** RL-ROE exists for station keeping (Tafanidis 2025, SAC); ART/Takubo use ROE to warm-start a convex solver. Nobody feeds `δi` to a policy that must close a large phase gap and a plane error jointly.
4. **A planar→spatial curriculum in astrodynamics RL, and the `i=0` singularity trap inside it.** CR3BP work transfers across *fidelity* (LaFarge/Howell 2022: CR3BP→ephemeris), never across *dimensionality*. Growing Action Spaces (ICML 2020) is the ML-side analogue but is off-policy. **The specific failure — that the natural warm-start point (i=0) is exactly where qns-ROE is singular and `δi_y ≡ 0` regardless of ΔΩ — is documented nowhere and is a genuine, cheap-to-demonstrate negative result.**
5. **The LEO combined-maneuver split as an RL design constraint.** That the optimal split is ~50/50 rather than apoapsis-loaded in our altitude band is textbook-derivable, but it is stated nowhere as a *shaping/expert* constraint, and it is exactly the kind of assumption a shaping designer would get wrong (Federici & Zavoli's "dense shaping biases toward the designer's assumed solution", which is precisely the mechanism that sank this project's 2D shaping).
6. **Joint phase+plane closure where J2 differential nodal drift is the free plane-change mechanism.** The e/i-separation literature manages J2 drift as a *disturbance to be cancelled*; nobody has posed it to an RL agent as a *resource to be exploited* on a drift orbit already opened for phasing.

Items 1, 2, 4, and 6 are reportable in the limitations-first Draper framing regardless of whether ext-3d succeeds.

Sources: [arXiv:2002.07832](https://arxiv.org/abs/2002.07832) · [ISSFD 2011 Spurmann/D'Amico](https://issfd.org/ISSFD_2011/S2-Formation.Flying-FF/S2_P4_ISSFD22_PF_034.pdf) · [JGCD 10.2514/1.G000189](https://arc.aiaa.org/doi/10.2514/1.G000189) · [JGCD 10.2514/1.G006175](https://arc.aiaa.org/doi/10.2514/1.G006175) · [Astrodynamics 10.1007/s42064-024-0214-8](https://link.springer.com/article/10.1007/s42064-024-0214-8) · [AdSpR 2025 RL-ROE](https://www.sciencedirect.com/science/article/pii/S0273117725004533) · [arXiv:2310.13831](https://arxiv.org/abs/2310.13831) · [arXiv:2510.03544](https://arxiv.org/pdf/2510.03544) · [OrbitZoo arXiv:2504.04160](https://arxiv.org/html/2504.04160v3) · [Kolosa 2020](https://scholarworks.wmich.edu/dissertations/3542/) · [ERAU edt/924](https://commons.erau.edu/edt/924/) · [arXiv:2511.11402](https://arxiv.org/html/2511.11402) · [arXiv:2008.08501](https://arxiv.org/pdf/2008.08501) · [arXiv:2409.16882](https://arxiv.org/html/2409.16882v1) · [arXiv:1906.12266](https://arxiv.org/abs/1906.12266) / [PMLR v119](https://proceedings.mlr.press/v119/farquhar20a.html) · [LaFarge/Howell/Folta 2022](https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2022_AIAA_LafHowFol.pdf) · [arXiv:1911.08553](https://arxiv.org/abs/1911.08553) · [JSR 58(2) 2021](https://carleton.ca/spacecraft/wp-content/uploads/sites/229/JSR-2021.pdf) · [Space Sci.Tech. 0047](https://spj.science.org/doi/10.34133/space.0047) · [ISSFD 2015 GTO ascent](https://issfd.org/2015/files/downloads/papers/166_Junker.pdf) · [JPL MEE](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf) · [DLR PRISMA](https://www.dlr.de/en/rb/research-operation/research-projects/flight-dynamics-navigation-and-orbital-sustainability/gnss-technology-and-navigation/past-projects/prisma-formation-flying-misson)