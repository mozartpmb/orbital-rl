## N3D-C — Integration architecture: `OrbitalNav` under `dim3_mode=1`

**Status: recon complete, 4 probes run, all read-only. Nothing modified, nothing committed.**
Scripts: `/Users/pete/space_training/scripts/orbital/ext_recon/n3d/{n3d_c1_decode.py, n3d_c2_prop6.py, n3d_c3_gauge.py, n3d_c4_tau30.py}`
CSVs: `/Users/pete/space_training/web_data/results/{n3d_decode_completeness.csv, n3d_prop6_validation.csv, n3d_filter6_cost.csv, n3d_gauge_sensitivity.csv, n3d_tau30_action_mix.csv}`

---

### 0. Bottom line

1. **Obs-only decode is EXACT at `e_max_target = 0`, and broken at `e_target > 0`** — and the failure is a single missing d.o.f., the chaser's RAAN. Measured against the running C env: rebuilding the C's own `obs[33-36]` from the obs-only decode gives **0.22 m / 4.4e-5 m/s at e_t = 0** (= float32 obs quantisation, i.e. exact) and **6.1e5 m / 740 m/s at e_t = 0.05** (the X3 rung), **3.6e6 m / 3877 m/s at e_t = 0.30** (V5). Supplying Ω_s restores exactness at every e_t (**0.24 m / 7.5e-5 m/s at e_t=0.05, 0.41 m / 3.2e-4 m/s at e_t=0.30**).
2. **The right fix is a read-only C getter**, not a numerical solve and not the trajectory accessor. The trajectory accessor is unusable per-step (measured: `vec_get_trajectory` returns 0 rows mid-episode because `last_traj_records` is only set at terminals; one env per call; float32). The obs-only route to Ω_s exists but requires a per-step 1-D nonlinear solve on `obs[33,34]` whose sensitivity **collapses to zero exactly at e_t = 0** (measured 1.2e-5 m/rad at e_t=0 → 6.0e5 at 0.05 → 3.3e6 at 0.30).
3. **The gauge is safe.** Building `PlaneGauge` from the *estimated* target plane instead of the truth moves the phase channel `obs[13,14]` by **≤ 0.006° per 1° of plane-estimate tilt** (measured, gain ∝ i_chaser). The plane channels `obs[21,22]` take the tilt at **gain ≈ 0.7 : 1** — they are the fragile pair, not the phase channel.
4. **The 6-state port is a slicing change, validated.** The dimension-generic f&g agrees with the independent universal-variable oracle to **3.1e-8 m @ dt=60 s** and **3.7e-4 m @ dt=21600 s**, and at `i = 0` it reproduces the shipped 2-D `propagate_cartesian` **bit-exactly (max abs diff 0.0)** — so the 2D-nav bit-exactness anchor survives even a single shared code path.
5. **Cost is ~5× the 2D nav lineage, and it is still not binding** — but the driver is not the state dimension, it is τ. Measured mean τ for the shipped X3 policy is **79.61 sub-steps/decision** (2D canonical: 32.50), with **26.0% of decisions at τ=180/360**. Projected ~10K SPS at the shipped 8w×256 shape ⇒ **~80 min per 50M rung** (2D: 16.5 min).
6. **Three MUST-FIX defects found in the shipped wrapper** before any 3D run: an `IndexError` at Discrete-30, a silent filter/truth time desync in `nav_max_ticks`, and an `obs[21]` collision between the ext-3d plane channel and the NAV-F Σ-channel.

---

### 1. Slot table — what the 3D-nav wrapper must rebuild from the ESTIMATE

Source of truth: `fill_observations`, `/Users/pete/space_training/pufferlib/pufferlib/ocean/orbital/orbital.h:913-1209`. Current 2D set is `nav_math.TARGET_SLOTS_T3 = (7,8,11,12,13,14,16,33,34,35,36,37)` (`nav_math.py:74`), 12 slots. **Under dim3 it becomes 19 slots.**

Inputs: chaser state is TRUTH throughout (self-knowledge: GPS/IMU). Target estimate `x̂_t = (r̂_t, v̂_t) ∈ ℝ⁶`; derived `â_t` (vis-viva), `ê_t = (v̂×ĥ)/μ − r̂` (3-vec), `ĥ_t = (r̂_t×v̂_t)/|·|`, `v̂_c = √(μ/â_t)`.

| slot | C source | rebuild formula from estimate | singularity / burn-continuity check |
|---|---|---|---|
| 7 | `:948` | `(â_t − R_E)/S_a` | none |
| 8 | `:949` | `‖ê_t‖` | none (norm, no angle) |
| 11,12 | `:954-955` | `sin/cos ϖ̂_t`, **ϖ̂_t := atan2(ê_ty, ê_tx)** | ⚠ **DO NOT compute ω̂_t from the node.** C writes raw `target.omega` with `i_t = Ω_t = 0` *exactly*, so its value is the inertial periapsis longitude. An estimate carries `î_t ≈ 1e-6 rad` of noise → the node direction is random → a node-referenced ω̂ is garbage. Reproduce the **value**, not the formula. Ill-conditioned as `‖ê_t‖→0`; C has the same degeneracy, so match it (`ω=0` when `e<1e-9`). |
| 13,14 | `:977-991` | `sin/cos(λ̂_s − λ̂_t)`, λ in the gauge `gauge_from_orbit(target ESTIMATE)` (`:739`), `λ = M + ϖ_gauge` | ✅ measured safe: gauge tilt → Δλ gain **≤0.006 deg/deg** at i_s=1°, 0.000 at i_s=0.02° (`n3d_gauge_sensitivity.csv`). Compute `M̂_t` via `nav_math.mean_from_true` (the corrected map, `nav_math.py:109`) — never the C's inverted `true_to_mean`. |
| 16 | `:993` | `cos(ω_s − ϖ̂_t)` — grandfathered mixed-frame channel; use the *same* ϖ̂_t as 11,12 for self-consistency | ⚠ not a physical angle under 3D (`ω_s` is node-referenced, `ϖ̂_t` is inertial). Preserved only for warm-start compatibility. |
| **21,22** | `:1113-1114` | `δî = Δi_rel·n̂`, `n̂ = (ĥ_t×ĥ_s)/‖·‖`, `Δi_rel = atan2(‖ĥ_t×ĥ_s‖, ĥ_t·ĥ_s)`; project on chaser RTN `R̂=r̂_s, T̂=ĥ_s×R̂`; ÷`di_scale` | ⚠ **the fragile pair.** Plane-estimate error enters at **gain 0.66–0.81 deg/deg** (measured). Guard `‖ĥ_t×ĥ_s‖ > 1e-300 → δî = 0⃗` exactly as C. `δî·N̂_s ≡ 0` analytically — assert it. Exactly invariant under prograde/radial burns; moves only under normal burns. |
| **23** | `:1120` | `(ē_s − ê_t)·ĥ_t / de_scale` | ē_s from `orb_evec` on chaser truth (`:553`) — the *element* route, not the Cartesian one (BLOCKER-2 variant V3). |
| **24,25** | `:1130-1131` | `(r_s − r̂_t)·ĥ_t / lvlh_scale`; `(v_s − v̂_t)·ĥ_t / v̂_c` | frame-rotation term drops out exactly (ω⃗_LVLH ∥ ĥ_t). Cleanest channel: measured decode error **1 mm**. |
| **26** | `:1143` | `v̂_c·‖ĥ_s − ĥ_t‖ / dv_ref` | chord form, exact (not small-angle). |
| 27 | `:1144` | **NOT REBUILT** — `V_e·ln(m/m_dry)/dv_ref` is chaser-only | leave the C value untouched. |
| **28** | `:1145` | `(dv_rem − dv_pl − dv_in)/dv_ref`, `dv_in = 0.5·v̂_c·hypot(δa_rel, ‖Δê‖)`, `δa_rel=(a_s−â_t)/â_t` | ⚠ mixes chaser truth with an estimate. Under a blind 4-decade range prior `â_t` is wildly wrong → this saturates. It is the only slot where a diverged estimate produces a *plausible-looking* feasibility signal. |
| 29-32 | `:1146-1149` | hardcoded `0.0f` — **verified min=max=0 over 512 resets at X3 and V5** | free; see §6c for the Σ-channel/J2 contention. |
| 33-36 | `:1186-1189` | as `fill_target_obs_t3` today (`nav_math.py:301`), but the **in-plane projection only** — C uses `(x,y)` of both bodies and drops z | ✅ keep `theta_t = atan2(ŷ_t, x̂_t)` (the Cartesian route the 2D wrapper already uses). C computes `theta_t = target.theta + target.omega`, which equals `atan2(y,x)` **only because `i_t = Ω_t = 0`**; with a tilted estimate the element route diverges. Reproduce the value. |
| 37 | `:1190` | `√(μ/â_t³)/1e-3` | none |

**Clamp semantics (easy to miss, breaks the recon gate).** C applies `obs_clamp2` (±2, also traps NaN — `orbital.h:907`) to slots **21-28 only**; slots 7/8/11-16/33-37 are written raw. The wrapper's single `nav_clip = 4.0` (`orbital_nav.py:120,269-274`) must therefore become **±2 on 21-28, 4.0 elsewhere**, or `recon` ≠ `truth` whenever the C clamp fires. Measured clamp-hit rate at reset: **0/512 at both X3 and V5** (max |obs21..28| = 0.985), but 3d_REDTEAM m3 measured 3.24 mid-episode at the widest rung, so replicate it.

---

### 2. Decode completeness — verdict: **incomplete by exactly one d.o.f.; add a getter**

**What obs *does* contain (measured, `n3d_decode_completeness.csv`, 256 envs/config, obs and truth pinned to the same epoch):**

| quantity | route | error (median / max) |
|---|---|---|
| `Δi_rel` | `di_scale·hypot(obs21,obs22)` | **1.3e-8 / 7.2e-8 deg** |
| chaser argument of latitude `u = ω_s+θ_s` | `atan2(−obs22, obs21)` | **4.7e-6 / 2.2e-5 deg** |
| cross-track separation | `obs24·lvlh_scale` | **1.0e-3 / 8.9e-3 m** |
| chaser `θ_s` | `obs[2,3]` (node-referenced, 3D-clean) | **2.8e-6 deg** |

**What it does not contain: Ω_s.** By construction — the entire ext-3d block was designed SO(3)-invariant, so `obs[21-28]` are exactly invariant under rotation about ĥ_t and cannot carry Ω_s. The observable constraint is
`Δλ = (M_s + ω_s + Ω_s) − (M_t + ω_t)` with `ω_t` pinned by `obs[11,12]`,
leaving the one-parameter family **Ω_s → Ω_s+α, M_t → M_t+α**. At `e_t = 0` that family is a *rigid rotation of both bodies about ĥ_t* — physically irrelevant, and every relative channel is invariant. At `e_t > 0` the target's mean-anomaly advance is not a rotation, so the family is broken by `obs[33-36]` — but only at strength O(a_t·e_t).

| e_target | obs-only recon vs C's own obs[33,34] | vs obs[35,36] | d‖LVLH‖/dΩ_s | best-case Ω_s from a float32 solve |
|---|---|---|---|---|
| 0.000 | **0.22 m** | **4.4e-5 m/s** | 1.2e-5 m/rad | unresolvable (family is exact) |
| 0.005 | 6.0e4 m | 74.5 m/s | 6.1e4 m/rad | 1.8e-5 rad |
| 0.020 | 2.4e5 m | 299 m/s | 2.4e5 m/rad | 4.5e-6 rad |
| 0.050 (**X3**) | 6.1e5 m | 740 m/s | 6.0e5 m/rad | 1.9e-6 rad |
| 0.150 | 1.9e6 m | 2247 m/s | 1.7e6 m/rad | 5.0e-7 rad |
| 0.300 (**V5**) | 3.6e6 m | 3877 m/s | 3.3e6 m/rad | 3.2e-7 rad |
| with Ω_s supplied | **0.20–0.41 m** at every e_t | **4e-5 – 3e-4 m/s** | — | — |

(The `t3decode_tgt_pos_err ≈ 1e7 m` row in the CSV is the *absolute inertial* error and is the harmless rotation gauge — it is ~1e7 m even at e_t = 0. The LVLH columns above are the physically meaningful ones.)

**Recommendation.**
- **Add `vec_get_state(handle, out_f64)`** to `/Users/pete/space_training/pufferlib/pufferlib/ocean/orbital/binding.c` alongside the three existing custom methods (`binding.c:6-17`): `(num_envs, 15) float64` = sat `{a,e,M,θ,ω,i,Ω}`, `fuel_frac`, target `{a,e,M,θ,ω,i,Ω}`. Pure read, no state mutation, ~40 lines, mirrors `vec_get_episode_init_info`. This makes the decode *trivial and exact* (there is no decode at all) and removes the ill-conditioned `atan2` chain from the hot path.
- **Keep the obs-only decoder as a gate, not a dependency:** at `e_target = 0` it must agree with the getter to < 1 m. That gate catches scale-constant errors exactly the way `nav_mode='recon'` does today.
- **Corollary the ladder should exploit:** rung 1 at `e_max_target = 0` needs **no C change at all** — obs-only + `i_s` from `obs[21,22]` + `Ω_s := 0` is exact to 0.22 m. The getter is only required from the e>0 rung onward. That is a clean two-stage risk profile.
- **Do not use the trajectory accessor.** Measured: `vec_get_trajectory` returns `records = 0` mid-episode (it reports `env->last_traj_records`, set only at terminals; `binding.c:114-124`); it is per-env-index, float32, and requires `log_enabled=1` which costs 352 B per sub-step (`orbital.h:1504`).

---

### 3. Port plan

**3a. `propagate_cartesian` → 3-vectors** (`nav_math.py:167`). Dimension-generic by index change alone: `r0 = ‖r‖`, `σ₀ = r·v/√μ`, and f/g/ḟ/ġ are scalars multiplying whole vectors. Validated in `n3d_c2_prop6.py`:

| dt | max ‖Δr‖ vs `orbital_math3d.propagate_universal` | max ‖Δv‖ |
|---|---|---|
| 60 s | 3.10e-8 m | 5.06e-10 m/s |
| 300 s | 9.14e-7 m | 4.02e-9 m/s |
| 1800 s | 2.48e-5 m | 3.14e-8 m/s |
| 3600 s | 5.52e-5 m | 4.56e-8 m/s |
| 10800 s (τ=180) | 1.33e-4 m | 1.13e-7 m/s |
| 21600 s (τ=360) | 3.66e-4 m | 2.94e-7 m/s |

**Equatorial rows reproduce the shipped 2D routine bit-exactly (max abs diff `0.000e+00`, 300 draws).** So a single generic code path is admissible; the conservative option (dispatch on `dim3_mode`) is still recommended so the 2D anchor is *structural* rather than empirical, matching the `orbit_to_cartesian` value-gate precedent (`orbital.h:577-592`).

**3b. Filter state 4 → 6.** Keep the Aidala–Hammel property that makes NAV-G's design work (measurement = state components ⇒ **linear** update ⇒ no premature covariance collapse). 3D MSC:
`y = [β, ε, β̇cos ε, ε̇, ρ̇/ρ, ln ρ]`, measurement `(β,ε) = (y₀,y₁)`, `H = [e₁;e₂]`.
- **Pole choice matters.** `ε = ±90°` is the chart singularity. Fix the spherical pole per episode at `p̂ = unit(u⃗₀ × ĥ_t)` (ε₀ = 0 by construction, pole in the orbit plane), plus a **re-pole guard at |ε| > 60°** implemented as the same similarity transform `set_cart` already uses. Do not pole at ẑ: a purely cross-track separation is reachable late in a plane-matching episode.
- Keep the `log(max(ρ, 1 m))` guard (`nav_math.py:478`) and the `_LOG_RHO_MAX` clamp.
- `range_prior_intervals` (`nav_math.py:656`) is already written in dot products — the LOS-ray-vs-annulus quadratic is dimension-generic; only the `sat_cart[:, 0]**2 + sat_cart[:, 1]**2` line needs the z term.
- `BatchedRangeBearingEKF` → range + 2 angles, 6-state Cartesian: mechanical.

**3c. Warp sub-propagation** (`orbital_nav.py:464-491`). Unchanged in structure: sub-propagate both truth 6-states at the nav cadence, pin the last sub-tick to the env's own epoch (`sc[last] = sat_c[last]`) so no integration drift accumulates. One batched `propagate_cartesian_nd` call for the stacked (chaser, target) pair as today.

**3d. Cost** (`n3d_filter6_cost.csv`, OMP=1, float64):

| B | prop 6D/4D (ms) | FD-STM 6D/4D (ms) | cov chain 6×6/4×4 (ms) | **tick ratio 6/4** |
|---|---|---|---|---|
| 256 | 0.115 / 0.085 | 0.870 / 0.371 | 0.041 / 0.026 | **2.25×** |
| 1024 | 0.482 / 0.146 | 2.250 / 0.701 | 0.158 / 0.101 | **2.84×** |
| 4096 | 1.195 / 0.621 | 10.149 / 2.696 | 0.689 / 0.403 | **3.29×** |

Combined with the measured **mean τ = 79.61** (`n3d_tau30_action_mix.csv`; 15 greedy episodes of `seed42_X3_3d_di1deg.pt`, 15/15 success, 250 decisions, 16.7 decisions/episode):

- nav wall-clock ratio vs 2D = (79.61/32.50) × 2.25 ≈ **5.5×** at the shipped 8w×256 shape.
- env cost also rises: `env_ms = 0.070 + 0.348·τ` ⇒ 11.38 → 27.8 ms/decision at B=1024 (2.44×).
- Projected total ≈ 4.9× ⇒ **~10K SPS, ~80 min per 50M rung** (2D nav60: 50.4K SPS, 16.5 min). Not binding.
- **Useful framing:** total nav ticks per episode = simulated seconds / 60, *independent of the action mix*. The per-agent-step cost jumps only because the 3D policy packs 2.45× more sim-time into each decision.
- Levers, in order: (i) **adaptive-dt tick capping** (§6b) — `K=12` cuts ticks 6.6×; (ii) analytic 3D STM (Shepperd/Goodyear) — FD-STM is **87%** of the tick (2.250 of 2.566 ms at B=1024) and 12 of 13 propagations are the FD probes; (iii) C port.

**3e. Action mix, for the record** (X3 greedy, Discrete-30): warp-30m 13.2%, **warp-6h 13.2%**, **warp-3h 12.8%**, retro-25 10.0%, warp-1h 7.6%, combined −25/−25 7.6%, combined +25/+25 6.0%, combined +25/−25 6.0%. **Plane + combined rows (20-29) = 29.2% of all decisions.**

---

### 4. Eval-harness deltas (`/Users/pete/space_training/scripts/orbital/nav/eval_relnav.py`)

1. `recover_states_t3` (`:178`) → `recover_states_3d`, or bypassed entirely by the getter. `build_obs_t3` (`:209`) → the 19-slot table of §1 **with the ±2 clamp on 21-28**.
2. `_obs_scales` / `make_env` (`:233,297`) must thread `dim3_mode`, `di_max_rad`, `obs_di_scale_rad`, `obs_de_scale`, `legacy_action_space=30`, `shaping_mode=2`, `shape_dv_ref_ms`.
3. **Real batch IOD at eval** (`run_bo`, `:421`; `_bo_blind_state`, `_bo_seed_mpc`) goes from 4 unknowns / 1 angle to **6 unknowns / 2 angles**. The extra angle is *information*, not just cost — see §5's headline hypothesis. `_bo_env_annulus` (`:365`) and the range-prior quadratic need the z term.
4. `nav_surrogate.ObservabilityMap` (`nav_surrogate.py:126`) is keyed on `(sep_km, orbits, dv_ms)` from `ext_bo_observability.csv`, which is a **2D CRLB**. Under 3D it becomes *conservative* rather than calibrated (normal burns are ⊥ to a predominantly along-track close-range LOS — NAV-F §2.5 measures the best/worst direction spread at 48×, and radial is the worst direction). Ship rung 1 on the existing table with that stated, re-derive a 3D CRLB for the headline.
5. `nb_eval.py` / `navf_metrics.py` counterfactual-information machinery: the per-burn replay is filter-agnostic and ports as-is once the filter is 6-state.

---

### 5. Ladder skeleton, arms, headline, anchors

**Warm start: `/Users/pete/space_training/models/t3/seed42_X3_3d_di1deg.pt`** (Discrete-30, obs 38, 3/3 seeds at 200/200). Its box is the *default* 30 km / 50 m/s — the **same box as the 2D nav lineage**, so the 98.0–99.5% NB1 numbers are a like-for-like comparator. No head surgery, no obs-dim change: the wrapper touches only the contents of 19 slots. Seeds 7 and 1337 exist for multi-seed replication (`seed{7,1337}_X3_3d_di1deg.pt`).

| rung | env | nav_mode | why |
|---|---|---|---|
| **N0-truth** | X3, dim3 | `truth` | regression anchor: must be byte-identical to plain `Orbital` |
| **N0-recon** | X3, dim3, **e_target = 0** | `recon` | isolates the 19-slot encode/decode layer with **zero C changes** and zero estimation error; gate = obs reproduced to ≤1 ulp, closed loop 200/200 with an md5-identical action stream |
| **N0-recon-e05** | X3 (e=0.05) | `recon` + getter | proves the getter; same gate |
| **N1-rb3d** | X3 | `rb_ekf` (range + 2 angles) | **essential control** — range is measured, so a burn buys no information. If N1 ≡ N2 the effect is not observability |
| **N2-bo3d** | X3 | `bearings_only` (2 angles, 6-state MSC) | the treatment. zero-shot from X3 → warm 50M → 3 fresh seeds (42/7/1337), mirroring the NB1 protocol |
| **N3-wide** (stretch) | V-ladder V2→V5 | `bearings_only` | e≤0.30, 300–8000 km; needs the getter and the re-derived 3D acquisition table |

**Headline claim, if it lands:** *one policy flies a full 3D rendezvous — including the plane change — on angles-only bearings, at N% against N% on truth (zero truth-tax), from a 4-decade blind range prior.* Draper framing: this is GN&C, not guidance; the sensor is a star-tracker-class angles sensor with no range at all; and the limitation is stated up front (two-body only, no J2, batch-IOD acceptance gates still certify multi-1000-km epoch errors — NAV-REDTEAM's unfixed finding carries over verbatim).

**The one genuinely new scientific hypothesis this combination creates** (pre-register it, it is falsifiable and cheap): NAV-F's dual-control null was *"the policy learned to coast through the blind window instead of burning for observability"*, and NAV-F §2.5 measured that the **worst** burn direction (radial, 90°/270° from LOS) is 48× weaker than the best. At close range the LOS is predominantly along-track, so **a normal burn is very nearly ⊥ LOS by construction** — and the 3D task *forces* 29.2% of decisions to be normal/combined burns for guidance reasons anyway. Prediction: **3D-nav acquires with materially less latency and less Δv than 2D-nav, without the policy ever "learning" observability** — the plane maneuver is a free observability maneuver. Metric: acquisition latency and pre-acquisition Δv, N2-bo3d vs NB1 (2D), matched box and noise. A null here is as informative as a positive.

**Regression anchors (all must be re-run; the first three are existing stages of `/Users/pete/space_training/scripts/orbital/nav/verify_extnav.py`):**
1. **v1** — plain `Orbital` untouched: legacy 26/200 + T3 canonical 200/200 by direct replay, md5-identical action streams.
2. **v1b** — `OrbitalNav(nav_mode='truth')` at dim3 == plain dim3 `Orbital`, md5-identical, 200 eps, `seed42_X3`.
3. **v2/v3** — the 2D `recon`/`rb_ekf` rows stay **bit-identical** (the 6-state port measured 0.0 diff at i=0; if a shared code path is used, this anchor is what proves it).
4. **ext-3d anchors** — `ext_rt3d_a2_bitexact.csv` and the 425,477-row T3 canonical trajectory anchor, re-run after **any** `binding.c` change (the getter recompiles the TU).
5. **NB1 reproduction** — `models/t3/extnav_nb1_fresh_42.pt` re-evaluated → 99.0% ± sampling noise, proving the 2D nav lineage is untouched.
6. **New:** obs-only decode vs getter at `e_target = 0` must agree to < 1 m (the scale-constant gate of §2).

---

### 6. MUST-FIX defects found in the shipped wrapper

**(a) `IndexError` at Discrete-30 — hard crash, first step.** `nav_math.ACTION_TAU` and `ACTION_DV_MAG` are length **20** (`nav_math.py:40,50`); `orbital.h` is `NUM_ACTIONS 30`. Confirmed: `nm.ACTION_TAU[np.array([26])]` → `IndexError: index 26 is out of bounds for axis 0 with size 20`. Extend both to 30 from `orbital.h:78-137,166-175`. Note `ACTION_DV_MAG` for rows 26-29 is `hypot(25,25) = 35.355`, **not** 50 — `apply_impulse` takes the norm of the assembled vector and p̂ ⟂ n̂ exactly (`orbital.h:797-818`).

**(b) `nav_max_ticks` silently desyncs the filter from truth.** `orbital_nav.py:443-444` does `tau = np.minimum(tau, self._nav_max_ticks)` and then ticks that many times **at dt = 60 s**, while the C env advances the full τ. With `nav_max_ticks=60` and action 17 (τ=360) the estimate ends up **5 hours stale** and is then encoded against the current-epoch chaser. NAV-H §2.4 called this "inert for every shipped config" — true at Discrete-16, **false at Discrete-30, where τ=180/360 is 26.0% of decisions.** Fix: fixed measurement *count* with adaptive interval — `n_ticks = min(τ, K)`, `dt_tick = τ·60/n_ticks` — which also delivers the §3d cost lever. The 2D `nav300` arm (dt=300) was validated at 99.5% @10× noise, so a stretched cadence is precedented; dt=1800 s at τ=360 is not, and needs its own measurement.

**(c) `obs[21]` collision: ext-3d plane channel vs NAV-F Σ-channel.** `_write_sigma_channel` writes `self.observations[:, 21]` (`orbital_nav.py:308`) on the premise that 21-32 are identically zero — true in 2D, **false under dim3**, where `obs[21] = δî·R̂_s`. Move the Σ-channel to **`obs[29]`** (verified zero: `obs[29:33] == 0` over 512 resets at X3 and V5) and retarget `/Users/pete/space_training/scripts/orbital/nav/zero_obs_column.py` to column 29. This contends with the J2 reservation (`orbital.h:156`, `3d_C §2 G`): proposed split — **29 = nav Σ-channel, 30 = cos i_s, 31 = cos i_t (J2), 32 free.** Flag to the J2 lane before either ships.

**(d) Minor.** `obs[15]` (episode clock) must stay excluded from the rebuilt set under dim3 exactly as today (`nav_math.py:71-74`) — overwriting it hands the policy a fake deadline. And `nav_clip` must become slot-dependent per §1.

---

### 7. Residual risks / open questions for the orchestrator

- **The plane channels are the weak link, not the phase channel.** `obs[21,22]` take plane-estimate error at gain ≈0.7:1, and 3d_C §4.8 puts the tight box's plane tolerance at 0.0075°. At the T3 box (30 km / 50 m/s) this is comfortable; at TB4/TB5 it is not, and a 3D-nav-at-the-tight-box rung should not be assumed to follow from the T3-box result.
- **The acquisition surrogate is now mis-calibrated in the safe direction.** It was fitted to a 2D CRLB; 3D adds a second angle and normal burns are ⊥ LOS. Expect the real batch solver at eval to acquire *earlier* than the training surrogate — which inverts the usual sim-to-eval risk but still breaks the "surrogate validated end-to-end" claim until re-measured.
- **`obs[28]` under a blind prior** is the one channel where a diverged estimate produces a confidently wrong feasibility margin. Worth a dedicated diagnostic (fraction of decisions with `obs[28] < 0` while unacquired).
- **The 3D policy's 16.7 decisions/episode** is a very short credit path relative to a 22-hour mission; combined with the τ=180/360 usage it means the nav gap per *decision* is enormous. That is the real reason to measure blind-gap distributions rather than only mean nav error.