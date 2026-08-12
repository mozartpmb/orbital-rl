GO. Design memo, prototype, and measured comparison below.

---

# N3D-A — 3D bearings-only filter: design + prototype + go/no-go

**VERDICT: GO.** The recommended filter — **BO3-BLS-MSC** (plane-seeded angles-only batch acquisition → 6-state modified-spherical recursive filter) — runs at every geometry in the combined envelope with **zero divergences, 1 acquisition failure in 96 runs, and NEES in-bounds 0.74–0.98**, and its coplanar controls reproduce NAV-G's own 2D numbers to **0.78–1.12×**. Cost at 1024 envs is **3.42× the shipped 2D filter tick** (6.98 vs 2.04 ms), which is inside the ext-nav SPS budget's existing slack. **But the headline NAV-G caveat is FALSE as stated:** 3D does *not* improve range observability by opening an independent "plane" channel — the orbit plane and the range ambiguity are **the same degree of freedom** (measured corr → 1.000), and the plane's *relative* precision is exactly the range's. 3D buys **1.2–1.8× on range and 2–4× on velocity**, and **nothing at all** in the far field.

---

## 1. Design decisions (each with its measurement)

### 1.1 Measurement + frame — az/el in an epoch-frozen frame poled on the chaser's orbit normal

Two angles of the LOS in an **inertially fixed** frame whose +z is `ĥ_c(t₀)` (chaser orbit normal at epoch), +x the chaser radial at epoch. Noise isotropic on the sphere: `R = diag(σ_β²/cos²el, σ_β²)`.

Why this and not chaser-LVLH or raw ECI RA/Dec:
- **`az` IS the shipped 2D bearing β and `el` IS the out-of-plane LOS angle.** At Δi=0 the self-test measures `el ≡ 0` and `ėl ≡ 0` to **0.0 exactly (< 1e-15 rad)**, so the 6-state filter collapses onto the shipped 4-state modified-polar filter component-for-component. That reduction is a regression test, not a claim.
- **Frozen ⇒ no frame-rate terms in the transition and no chaser-attitude model.** A co-rotating LVLH reference injects a deterministic orbital-rate bearing sweep that dominates the signal and couples chaser velocity into `H`.
- **Gimbal risk is real and bounded, not assumed away.** `el → ±90°` requires the out-of-plane separation to dominate the in-plane separation. The env caps this: a rendezvous holding at separation ρ cannot carry Δi > asin(ρ/r), so at ρ=10 km, Δi ≤ 0.085° and `el_max` stayed ≤ 27° across every prototype geometry. The guard (re-pole at |el| > 60°) is specified but never fired.

### 1.2 State — 3D modified spherical coordinates (MSC), **not** Cartesian-6, **not** ROE

`y = [az, el, ω_a, ω_e, ρ̇/ρ, ln ρ]` with `ω_a = ȧz·cos el`, `ω_e = ėl`.

| candidate | verdict | evidence |
|---|---|---|
| **MSC-6 (chosen)** | ship | Measurement is *exactly* (y₀,y₁) ⇒ `H = [e₁;e₂]`, zero linearisation error in the update; the weak direction is isolated in one scalar `ln ρ`. Prototype: NEES 0.435–0.823, in-bounds 0.74–0.98 at every geometry. |
| Cartesian-6 EKF, bearings-only | **reject** | Measured: NEES **3.9 – 7.9e17**, position error 36 m → 1.4e7 m, **8/12 seeds diverge** at the 180° gap. Same confidently-wrong failure NAV-G measured in 2D, worse in 6 dof. |
| **ROE / CW / YA (linearised relative motion)** | **reject, on two measured grounds** | (a) *Validity*: CW propagation error vs exact two-body, over the arcs the filter actually runs — **2.2 % of ρ at the 5 km box, 3.3 % at 10 km, 55.6 % at 300 km, 5699 % at the 180° gap** (`n3d_param_linearity.csv`; the CW STM is validated to 2.8e-6 m at 1 m separation, so these are model error, not code error). (b) *Observability*: any homogeneous linear relative model has the exact symmetry `x_rel → k·x_rel`, so range is structurally unobservable in it — NAV-F §2.1 measured the linear arm's scaled-trajectory bearing difference as identically zero. **The O(ρ/r) nonlinearity that ROE deletes is our entire observability budget**, and it is ~8× larger than the achievable range precision at the tight box. Recovering it means carrying second-order terms (Sullivan & D'Amico), valid only for ρ/r ≪ 1 — our envelope reaches ρ/r = 2.0. |

### 1.3 Cost at 1024 envs (`n3d_cost.csv`, OMP=1, same process, same machine)

| op | B=1024 | µs/env |
|---|---|---|
| `propagate3` f&g (6 Newton) | 0.242 ms | 0.236 |
| `stm_fd3` (12N+1 props) | 2.290 ms | 2.236 |
| MSC6 predict (13 props + FPFᵀ + Q) | 6.400 ms | 6.250 |
| MSC6 update (2×2 solve + Joseph) | 0.517 ms | 0.505 |
| **FULL 3D tick** | **6.979 ms** | **6.815** |
| shipped 2D `BatchedBearingMPC` tick | 2.041 ms | 1.993 |
| shipped 2D range+bearing EKF tick | 1.489 ms | 1.454 |

**Ratios: 3.42× the shipped 2D MPC, 4.69× the RB-EKF.** Predicted by structure (13/9 × (6/4)² = 3.25×) and measured at 3.42× — the extra is the 6×6 chains, not overhead. Against NAV-H's nav60 budget this moves the sensor tick from 2.04 → 6.98 ms/decision; at the measured 20.9 decisions/episode this is the *only* new cost, and the two documented levers (4 workers × 512 envs; validated analytic STM) are untouched.

---

## 2. The 3D observability structure — NAV-G's caveat is refuted

`n3d_crlb.csv`, 54 cells: whole-arc FIM on the target's epoch 6-state, `F = Σ Φᵀ HᵀR⁻¹H Φ`, projected onto range-along-LOS / plane-normal tilt / (a, e, u) / velocity.

### 2.1 Range: the 3D dividend is 1.2–1.8×, and zero far-field

σ_range (BO), LEO 400 km, σ_β = 1 mrad @ 60 s:

| family (ρ₀, arc) | Δi=0 drift | Δi max drift | Δi=0 +1 m/s | Δi max +1 m/s |
|---|---|---|---|---|
| TB 5 km box, 185 min | 896 m (17.9 %) | 726 m @0.040° (14.5 %) | 8.81 m | **4.86 m @0.040°** (1.81×) |
| G1 10 km, 277 min | 222.9 m (2.19 %) | 230 m @0.080° | 27.7 m | **16.2 m** (1.71×) |
| G2 300 km, 185 min | 711 m | 459 m @2.0° (1.55×) | — | — |
| G3 180° gap, 139 min | 898.7 m | **898.8 m @1.0°** (1.000×) | 898.8 m | 898.8 m |

Drift-only, relative inclination is worth **≤ 1.55×**; with a 1 m/s burn, **≤ 1.81×**; at the 180° phase gap, **nothing to 4 significant figures**. The NAV-F crossover (maneuver worthless beyond ρ ≈ 300–500 km) is unchanged by going 3D.

### 2.2 The plane is not an independent channel — it is the range ambiguity, re-projected

Two regimes, both exact:

```
σ_plane  ≈  max(  Δi_rel · (σ_ρ/ρ)  ,  c·ρ̄σ_β/(r√N) ),  c ≈ 1.1–2.0
```

The first branch is confirmed to **0.995–1.022× over 14 independent cells** (TB and G1, drift-only and burn, Δi from 0.002° to 0.08°), with `corr(range, plane) = 0.999–1.000` in exactly those cells. Physically: the near-null scale direction `r_t = r_c + k·ρ⃗` tilts the target's plane whenever the LOS has out-of-plane content, so plane error and range error are one degree of freedom.

At **Δi = 0 exactly**, `corr = 0.0000` and `σ_plane(BO) = σ_plane(RB)` to all digits — the out-of-plane dof decouples completely and is purely angular. **So the plane is well determined precisely when it is nearly zero, and degrades in lock-step with range as soon as it is not.** (G1 drift: 2.66e-5° at Δi=0 → 1.81e-3° at Δi=0.08°, a 68× degradation; RB stays flat at 2.7e-5°.)

### 2.3 Where 3D actually pays: **velocity**, and it pays at the box that matters

σ_vel (BO), same cells: TB 5 km drift **1.014 → 0.270 m/s** (3.8×) from Δi=0 → 0.040°; with 1 m/s burn **0.0086 → 0.0024 m/s** (3.6×). G1 drift 0.250 → 0.088 (2.9×), with burn 0.031 → 0.0069 (4.5×). NAV-F §2.6 established that a rendezvous box binds on **σ_vel, not σ_range** — so the 3D dividend lands on the binding constraint. (Drift-only σ_vel at the 5 km box is 1.014 m/s against a 1 m/s tolerance; adding Δi=0.04° takes it to 0.270 m/s, i.e. inside the box without a burn.)

---

## 3. Acquisition in 3D

**(a) The analytic range prior lifts verbatim — it is dimension-agnostic.** `|r_c + ρû|² = R²` is the same scalar quadratic in 3D; the annulus becomes a spherical shell and the LOS stays a ray. Verified against brute-force ray scans over 200 random geometries: **interval-count agreement 200/200, worst relative support error 2.4e-4, 41 % of geometries bimodal** (`n3d_acq_prior.csv`). No code change beyond 2- → 3-vectors.

**(b) "Solve the plane first" is NOT a shortcut — measured, and it fails for a specific reason.** A candidate plane normal `n̂` hands you the whole range history in closed form, `ρ_k = −(n̂·r_c,k)/(n̂·û_k)`, which is the strongest possible version of "3D escapes the 2D degeneracy". Its conditioning (`n3d_acq_plane.csv`):

| geometry | Δi | ρ_oop (median) | measured amplification d(ρ)/ρ per rad of tilt | tilt for 1 % range |
|---|---|---|---|---|
| TB 5 km | 0.020° | 1.62 km | 2 931 | **3.4e-6 rad** |
| G1 10 km | 0.040° | 3.24 km | 1 466 | 6.8e-6 rad |
| G1 10 km | 0.005° | 0.40 km | 11 730 | 8.5e-7 rad |
| G2 300 km | 1.00° | 80.9 km | 58.1 | 1.7e-4 rad |
| G3 180° | 0.751° | 60.7 km | 78.7 | 1.3e-4 rad |
| **G1 coplanar** | 0.000° | 0 | **3.6e12** | 2.8e-15 rad |

Amplification tracks the predicted `(r_t+ρ)/ρ_oop` within 1.4–4×, and **the coplanar column recovers the classical `D₀ = L₁·(L₂×L₃) ≡ 0` singularity as a limit rather than a special case** — the honest answer to "does 3D genuinely escape the 2D IOD singularity": *it escapes it continuously, at a rate set by ρ_oop, and the escape is far too slow to be useful*. Required plane precision at close range is ~1e-6 rad; the plane's own bearings-only accuracy there (§2.2) is 1.5e-5 rad — **10–20× too coarse**. Plane-first cannot bootstrap range.

**(c) Therefore the cheap route is the right route, and it costs the same as 2D.** Keep the validated 2D lattice — log-range grid × (3 tangential × 3 radial) — and **seed out-of-plane velocity at zero**; the out-of-plane *position* comes free and exactly from the measured elevation because the seed sits on the LOS ray. Sizing: true `v_oop = v_c·sin(Δi)` is **0–134 m/s** across the whole envelope (Δi ≤ 1°), against a lattice step of `0.02·v_c = 153 m/s` — smaller than one step everywhere. Measured consequence: the 27-node lattice (`BO3-BLS-FULL`) **produced bit-identical results to the 9-node lattice at all 8 geometries × 12 seeds**; its extra nodes never win the global minimum (best fn=0 cost beats best fn≠0 by 1.3–27× at every geometry). **The 3D acquisition is the 2D acquisition plus two locally-convergent dof, at 1.0× the node count.** The red-team's 17–90× acquisition-cost blocker and its surrogate remedy are therefore unchanged by the 3D lift.

Gates carried over unchanged (χ², ambiguity margin ≥ 16, σ_LOS/ρ ≤ 0.20, arc grown ×1.6 with warm start). **Window must start at 45 obs, not 90** — at w₀=90 the grid nodes are already useless Gauss-Newton starts and acquisition fails outright at the tight box (measured: w₀=45 acquires at w=72 with 35.8 m epoch error; w₀=90 exhausts the arc and returns nothing).

---

## 4. Prototype results (go/no-go datum)

`n3d_filter.csv` — 5 arms × 8 geometries × 12 noise seeds, settled = last quarter of the arc, medians over seeds.

### 4.1 Truth-tax: bearings-only vs range+bearing, in the same 6-state code

| geometry | Δi | RB3-EKF pos | BO3-BLS-MSC pos | tax | RB vel | BO vel | tax | NEES | in-bounds |
|---|---|---|---|---|---|---|---|---|---|
| **N2b 5 km box + 1 m/s, coplanar** | 0° | 4.55 m | 13.81 m | **3.0×** | 0.0046 | 0.0146 | 3.2× | 0.534 | 0.94 |
| **N2 5 km box + 1 m/s, 3D** | 0.020° | 4.76 m | **5.09 m** | **1.1×** | 0.0046 | 0.0055 | 1.2× | 0.435 | 0.83 |
| N0 10 km drift, coplanar | 0° | 6.15 m | 1 376 m | 224× | 0.0056 | 1.557 | 278× | 0.823 | 0.86 |
| N1 10 km drift, 3D | 0.040° | 5.42 m | 856 m | 158× | 0.0055 | 0.971 | 177× | 1.55 | 0.74 |
| N4b 300 km, coplanar | 0° | 45.9 m | 993 m | 21.6× | 0.049 | 1.137 | 23.3× | 0.653 | 0.98 |
| N4 300 km, 3D | 1.00° | 30.7 m | 865 m | 28.1× | 0.037 | 0.961 | 25.8× | 0.624 | 0.97 |
| N3b 180° gap, coplanar | 0° | 1 632 m | 3 487 m | 2.1× | 1.908 | 3.659 | 1.9× | 0.798 | 0.94 |
| N3 180° gap, 3D | 0.751° | 1 417 m | 3 476 m | 2.5× | 1.545 | 3.679 | 2.4× | 0.799 | 0.94 |

Failed arms, for contrast: `BO3-EKF` (naive Cartesian, blind) — 36 m to 1.36e7 m, NEES 3.9–7.9e17, 8/12 seeds diverge at N3. `BO3-MSC-o` (oracle 30 % range prior, no batch) — 15.7 m to 8.1e4 m, NEES 7–229, in-bounds 0.04–0.40. **The batch acquisition beats the oracle-initialised recursive filter at 6 of 8 geometries**, the same re-linearisation effect NAV-G measured in 2D.

### 4.2 Paired 3D-vs-coplanar (identical δa, burns, arc, seeds — only the plane differs)

| pair | arm | coplanar pos | 3D pos | gain | coplanar vel | 3D vel | gain |
|---|---|---|---|---|---|---|---|
| **N2b→N2 (5 km box, burns)** | BO3-BLS-MSC | 13.81 m | 5.09 m | **2.72×** | 0.0146 | 0.0055 | **2.67×** |
| N2b→N2 | RB3-EKF *(control)* | 4.55 m | 4.76 m | 0.96× | 0.0046 | 0.0046 | 1.01× |
| N0→N1 (10 km drift) | BO3-BLS-MSC | 1 376 m | 856 m | 1.61× | 1.557 | 0.971 | 1.60× |
| N4b→N4 (300 km) | BO3-BLS-MSC | 993 m | 865 m | 1.15× | 1.137 | 0.961 | 1.18× |
| N3b→N3 (180° gap) | BO3-BLS-MSC | 3 487 m | 3 476 m | 1.00× | 3.659 | 3.679 | 0.99× |

The RB control moves by 0.96–1.01× while BO moves by 2.72×, so **the gain is specific to angles-only, not a geometry artefact.** **Honesty flag:** the CRLB gain at that cell is only **1.21×** (8.81 → 7.28 m), so ~2.2× of the 2.72× is acquisition/filter behaviour, not information. **Do not quote 2.72× as an information result.**

### 4.3 Acquisition latency and cost

Latency **44–114 min** (median per geometry), 1 failure in 96 runs (N2b, 1/12). Grid 1 440–4 320 nodes, wall **0.30–1.62 s** per acquisition batched over nodes. Epoch-state error at acquisition: 15 m (5 km box) / 719–735 m (10 km drift) / 2.3–10.3 km (300 km, 180° gap).

---

## 5. Validation anchors

| anchor | result |
|---|---|
| `propagate3` vs the independent universal-variable oracle (`orbital_math3d`) | **4.3e-12** relative, 180 cases across e ≤ 0.30, i ≤ 69°, dt ∈ {60, 600, 3600} s |
| coplanar `propagate3` == shipped `nav_math.propagate_cartesian` | **0.000e+00** (bit-identical) |
| Δi=0 3D CRLB vs the independent 2D CRLB (`ext_bo_filter.crlb_range_sigma`) | **ratio 1.0000** at all four families (896/896, 222.9/222.9, 711.1/711.1, 898.7/898.7 m) |
| coplanar `el`, `ėl` | **0.0 exactly** (< 1e-15 rad, < 1e-18 rad/s) |
| MSC[0]==az, MSC[1]==el | 0.000e+00 |
| analytic az/el Jacobian vs central differences | 2.9e-9 relative |
| STM symplectic residual | 1.0e-6 (the 1 m / 1e-3 m/s FD truncation floor, matching shipped `stm_fd`) |
| 3D CRLB vs sibling lane N3D-B's independently-formulated CRLB | **1.0000–1.0008** across the shared grid (`n3d_crlb_xcheck.csv`) |
| CW STM used in the ROE-rejection probe | 2.8e-6 m at 1 m separation, error quadratic in ρ (0.028 % @100 m, 1.39 % @5 km) |
| σ_range at TB 5 km / 185 min / drift, independent reproduction of NAV-F §2.6 | **896 m, 1.014 m/s** vs their 937 m, 1.06 m/s |

---

## 6. Caveats (stated, not buried)

1. **One CRLB cell is at the conditioning floor.** At Δi=0 **and** δa=0 (exactly degenerate co-orbital drift, no burn) my pipeline returns a finite 2.23e6 m where the sibling lane's nondimensionalised pipeline returns `inf` (cond 7.5e15 vs 1.1e14). Same class as the NAV-G/NAV-F disagreement. Treat that one cell as unbounded; every other cell agrees to 1.0008×.
2. **RB3-EKF is not NAV-G's RB-EKF.** It estimates 6 states where theirs estimates 4, so at the wide geometries it is worse than their baseline (N3b 1 632 m vs their 415 m; N4b 45.9 vs 8.2 m). The *bearings-only* coplanar controls do reproduce their BO-BLS-MPC column (0.78–1.12×), which is the comparison that licenses the 3D numbers.
3. **Truth here is the oracle, not the C env.** No closed-loop measurement through `orbital_nav`; the wrapper's batched 6-state path is benchmarked (`n3d_cost.py`) but not run in a policy loop.
4. **No J2, no drag, no FOV/occultation, no measurement association, single target.** Real angles-only relative nav leans on J2 to break the scale symmetry; we have none, so our problem is *harder* than reality here — defensible framing, state it. (J2 is the natural next realism step and its slot is reserved in the ext-3d obs block.)
5. **Δi is capped by ρ, not chosen freely.** Every geometry enforces Δi ≤ asin(ρ/r) and skips infeasible cells rather than silently becoming a different geometry — the failure mode that produced ext-3d's `de_max` bug.
6. **The gimbal guard was specified but never exercised** (max |el| 27°). A geometry with dominant out-of-plane separation would fire it and is untested.
7. **12 seeds per cell, single geometry family per separation.** Adequate for a go/no-go, not for a headline distribution.

---

## 7. Recommendation for the 3D-nav campaign

- **Ship `BO3-BLS-MSC`.** MSC-6 recursive stage + 2D-lattice batch acquisition with zero-seeded out-of-plane velocity. Zero new search dimensions, 1.0× the 2D acquisition node count, 3.42× the 2D per-tick cost.
- **Reuse the ext-nav surrogate unchanged.** The red-team's acquisition-cost blocker and its CRLB-conditioned surrogate remedy transfer without modification, because the 3D acquisition cost is the 2D acquisition cost.
- **Do not build a plane-first acquisition stage.** Measured dead (§3b).
- **Do not add an out-of-plane observability action.** The 3D dividend is 1.2–1.8× on range and lands mostly on velocity; the existing Δv set already dominates it (a 1 m/s burn is worth 10²–10⁴× where Δi is worth 1.8×).
- **The one genuinely new scientific claim to go after:** at the tight box, relative inclination and a maneuver are *substitutes* for range observability. The paired result (angles-only tax 3.0× → 1.1× at Δi=0.02°, RB control flat) says a 3D-nav policy has a second lever the 2D policy did not — it can buy navigation information by *not fully nulling the plane* as well as by burning. That is a sharper dual-control experiment than NAV-F's, it is falsifiable with the counterfactual-information metric already designed, and it is exactly the "guidance wants X→0, navigation wants X≠0" story Draper reviewers recognise — now with two X's.

---

## Files (all new; nothing existing modified, nothing committed)

| path | contents |
|---|---|
| `/Users/pete/space_training/scripts/orbital/ext_recon/n3d/n3d_core.py` | batched 3D f&g propagator + STM, pole frame, az/el model + analytic Jacobian, MSC encode/decode, geometry builders, 10-check self-test (`python3 n3d_core.py`) |
| `…/n3d_crlb.py` | 3D observability map + 2D cross-check (`--xcheck`, `--quick`) |
| `…/n3d_filter.py` | prototype: 5 arms, 8 geometries, 3D batch acquisition (`--seeds N --scenarios … --arms …`) |
| `…/n3d_acq.py` | range-prior lift verification + plane-first conditioning probe |
| `…/n3d_param.py` | ROE/CW linearity-validity probe (CW STM validated at 1 m) |
| `…/n3d_cost.py` | batched 6-state filter benchmark vs shipped 2D at B=1024 |
| `…/n3d_report.py` | table renderer (`python3 n3d_report.py [filter\|crlb]`) |
| `/Users/pete/space_training/web_data/results/n3d_crlb.csv` | 54 rows: CRLB over family × Δi × Δv |
| `…/n3d_filter.csv`, `…/n3d_filter_conv.csv` | 40 rows summary, 1 276 rows per-step convergence traces |
| `…/n3d_acq_plane.csv`, `…/n3d_acq_prior.csv` | 42 rows plane conditioning, 200 rows prior verification |
| `…/n3d_param_linearity.csv`, `…/n3d_cost.csv` | 4 rows CW validity, 13 rows cost benchmark |