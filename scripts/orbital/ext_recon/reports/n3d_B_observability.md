All probes complete. Final deliverable:

# N3D-B — 3D angles-only observability map + training-surrogate design

**Artifacts.** Scripts (all new, nothing existing modified, nothing committed): `/Users/pete/space_training/scripts/orbital/ext_recon/n3d/n3d_crlb3d.py` (machinery + validation), `n3d_b_xcheck.py`, `n3d_b_map.py`, `n3d_b_tightbox.py`, `n3d_b_rank1.py`, `n3d_b_burnaxis.py`, `n3d_b_j2.py`; console logs `n3d_b_{map,tightbox,rank1,j2}_run.txt` in the same dir. CSVs in `/Users/pete/space_training/web_data/results/`: `n3d_obs_map.csv` (1225), `n3d_obs_map_wide.csv` (162), `n3d_tightbox_dualcontrol.csv` (420), `n3d_tightbox_anglesplit.csv` (72), `n3d_rank1_scale_model.csv` (1800), `n3d_burn_axis_placement.csv` (192), `n3d_crlb_xcheck.csv` (18), `j2_n3d_observability.csv` (144).

---

## 0. VERDICT (headline)

1. **The plane channel is free, and it is free for a reason that generalises.** Out-of-plane position error is measurement-limited everywhere — `σ_plane = √2·σ_β/√(Σ_k ρ_k⁻²)`, median ratio **0.999** over 1221 map cells, 4 decades of separation, every ΔI. It never binds: worst case **0.0038 of the plane position box** (264× margin). It is never singular, even where range is.
2. **But the whole 6×6 covariance is one number.** `Σ ≈ σ_k²·x_rel x_relᵀ + floor`, `σ_k = σ_range/ρ₀` — the classical scale family. Validated component-by-component against the exact CRLB over 1800 cells: median model/CRLB **0.987 (pos_plane), 1.002 (vel_plane), 0.999 (vel_inplane), 1.004 (vel_range)**. **So the answer to "plane bolt-on or full 3D table" is neither: the plane needs no table entry at all, it is a projection.** What *does* need a new axis is the scalar σ_k, because ΔI moves it by up to 2000×.
3. **The 2D map is mostly conservative but not safely so.** At ΔI=0 the 3D 2-angle CRLB equals the shipped 2D 1-angle CRLB **to 4 decimals** (structural: coplanar two-body decouples the FIM exactly into in-plane 4-dof ⊕ out-of-plane 2-dof). At ΔI>0 it is conservative by up to **262×** — but **6.1% of comparable cells are optimistic**, worst **2.8×** (5 km, 1 orbit, ΔI=1°, 5 m/s: map says 39 m, truth 110 m). Once a burn has already broken the ambiguity, adding relative inclination *costs* information (two more dof to estimate). A Δv-conditioned map with no ΔI axis hands the policy unearned information in exactly the corner the wide-3D envelope lives in.
4. **The 3D tight box is LESS live than 2D, but still live — and the tension is deferred, not removed.** Drift-only σ_range at the terminal (δa = v_tol/1.5n = 601 m ⟺ v_tol = 1 m/s): **47% of the 5 km box at ΔI=0 (reproducing NAV-F's 52%), 43% at the velocity-implied plane tolerance 0.0075°, 24% at the position-implied 0.0417°.** A plane residual at the box's own tolerance buys a 2× relief — real, bounded, and it evaporates as guidance nulls the plane. **The structural change: the exactly-unobservable configuration is now codimension-2 (δa=0 AND δi=0), not codimension-1.**
5. **The new binding constraint is out-of-plane VELOCITY.** σ_vplane exceeds the plane velocity box (0.998 m/s) in **9.5% of tight-box cells, worst 2.19×** — at 1-orbit arcs whenever a real plane residual exists. Mechanism is exact: `σ_vplane = σ_k · v_c·sin(Δi_rel)` (predicted 1.90 vs measured 1.902 m/s). **≥3 orbits (~4.5 h) of arc is required to bring it inside tolerance.** This is the 3D twin of NAV-F §2.6's "the box binds on σ_vel, not σ_range".
6. **In 3D the observability maneuver is the normal burn — the one guidance must make anyway.** At NAV-F's measured information peak (20% into the arc), 1 m/s normal beats 1 m/s prograde by **2.5–8× in information** at every terminal cell, and is **5.5× more robust to mis-placement** (worst σ over placements 0.05–0.75: normal 66 m, prograde 365 m). This is the strongest dual-control result available in this campaign: **the guidance-required action and the navigation-required action are the same action.**
7. **J2 does not rescue anything, and the standing caveat is wrong at these scales.** Measured, not asserted: J2 changes σ_range by **≤5% at every operationally relevant geometry**. It helps only in the single exactly-degenerate cell (δa=0, δi=0, drift-only): 286 km → 19.5 km, still 4× the box. The caveat repeated in NAV-F §5 / NAV-G §5 ("real LEO leans on J2, our sim is harder than reality") is **false for ≤3-orbit arcs at δa ≤ 30 km / δi ≤ 1°** — Keplerian curvature already dominates wherever it matters. Keep the framing for multi-day IROD; drop it for this env's regime.

---

## 1. Machinery and validation

**Measurement model, basis-free.** Two focal-plane angles give `H = ρ⁻¹[ê₁ᵀ 0; ê₂ᵀ 0]`, so `HᵀR⁻¹H = (σ_β ρ)⁻²·[P_⊥ 0; 0 0]`, `P_⊥ = I − uuᵀ`. Information depends only on `P_⊥` — independent of focal-plane roll, so no az/el pole singularity. Classical az/el with `σ_az = σ_β/cos el` is exactly this object. The 2D machinery is the rank-1 restriction `P_⊥ → n̂n̂ᵀ`.

**Conditioning.** Everything is computed in the nondimensional state `D⁻¹x`, `D = diag(a,a,a,v_c,v_c,v_c)`. Not cosmetic: NAV-F §4.1's correction to NAV-G ("exactly singular" was an artifact of a badly-scaled differenced FIM) is precisely this failure mode. The scaled 6×6 returns a finite 221,879 m where the shipped 2D pipeline trips its own `cond > 1e16` guard and reports `inf`.

**Validation (`python3 n3d_crlb3d.py`):**

| check | result |
|---|---|
| C1 RK4(J2 off) vs universal-variable oracle, 1 h | 1.13e-10 rel |
| C2 STM symplecticity (scaled, parameter-free) | 2.08e-06 |
| **C3 anchor** — coplanar equatorial 3D vs `ext_bo_filter.crlb_range_sigma` | ratio **1.0000** at 5 km/1 m/s and 200 km/5 m/s; z ≡ 0 exactly |

**Independent cross-check vs sibling lane N3D-A** (`n3d_b_xcheck.py`, `n3d_crlb_xcheck.csv`): both CRLBs fed identical truth states. N3D-A = explicit az/el about a pole frame + f&g-in-ΔE propagation + raw FIM; N3D-B = basis-free `P_⊥` + universal variables + scaled FIM. **17 matched cells, σ_range ratio B/A ∈ [0.9998, 1.0008], median 1.0000; σ_vel identical to the same precision.** The one non-matching cell is the exactly-degenerate co-orbital coplanar drift case where both are at their numerical guard (A: 2.23e6 m at cond 1e17; B: unbounded at scaled-cond 1e14) — the operational conclusion is identical.

**Defect found in the shipped 2D design map.** `ext_bo_observability.csv`, LEO block, `sep=5 km, orbits=0.25, dv=0`: `sigma_los_bo_m = 0.0, rel_sigma_bo = 0.0` at `fim_cond = 9.94e15` — the FIM was numerically singular but slipped under `crlb_range_sigma`'s `cond > 1e16` guard, and inverting it returned a meaningless *zero*, i.e. the CSV asserts **perfect** range observability in one of the least observable cells in the table. Runtime is *defended* (`ObservabilityMap.__init__` maps `v <= 0.0 → _INF_REL`), so no shipped behaviour is wrong, but any human or script reading the CSV directly is misled. Recommend tightening the guard to `cond > 1e12` on a *scaled* FIM.

**Methodology note (a trap worth recording).** The first map run reused one long roll for every arc checkpoint, which places the burn at a fixed fraction of the *longest* arc — silently turning every short-arc row into drift-only. Both sweeps now re-roll per arc, mirroring `ext_bo_observability.build`. Symptom was a table where the dv column was bit-identical to dv=0 at 1 orbit.

---

## 2. The map (Q1: what does the plane cost?)

`n3d_obs_map.csv` — LEO a=6771 km, i_t=51.6°, σ_β=1 mrad, 60 s: separation {5,10,50,200,1000,5000,13000} km × arc {0.25,0.5,1,2,3} orbits × Δv {0, 1, 5, 25} m/s × axis {prograde, normal} × ΔI_rel {0, 0.01, 0.05, 0.25, 1.0}°. `n3d_obs_map_wide.csv` — a=12000 km, e ∈ {0, 0.30}, separation {10, 1000, 12000} km, ΔI {0, 0.1, 1.0}°, arcs {0.5, 1.0, 1.5}.

### 2.1 σ_plane is the measurement floor, universally

Drift-only, 1-orbit arc; last column is `σ_plane / (√2·ρ̄σ_β/√N)`:

| sep | ΔI=0 | ΔI=0.01° | ΔI=0.05° | ΔI=0.25° | ΔI=1.0° | ratio-to-floor |
|---|---|---|---|---|---|---|
| 5 km | 0.73 m | 0.78 | 0.98 | 1.90 | 3.34 | 1.41 → 0.43 |
| 50 km | 7.31 | 7.68 | 7.73 | 8.48 | 12.8 | 1.41 → 1.32 |
| 1000 km | 146 | 146 | 146 | 150 | 167 | 1.41 |
| 13000 km | 1622 | 1622 | 1622 | 1622 | 1622 | 1.41 |

Against σ_range in the same cells the plane is **3× to 300,000× better** (`plane_over_range` spans 3e-6 to 0.34). The `√2` is a degrees-of-freedom factor (2 free parameters in the decoupled out-of-plane channel); it is an *upper bound* — rich geometries beat it (0.43 at 5 km/ΔI=1°). Using the **information mean** of ρ (`√(Σρ_k⁻²)⁻¹`) rather than the arithmetic mean matters by 2× on drifting geometries; median ratio over the whole map with the correct combination is **0.999**.

**Consequence for the acquisition gate.** `σ_plane/ρ ≤ 0.20` requires only `N ≥ 2/0.04 = 50` observations — **50 minutes, geometry-independent**. The range gate always binds first. **NAV-G's shipped gate (`σ_LOS/ρ ≤ 0.20`, 45-tick floor) carries over to 3D unchanged.**

### 2.2 Relative inclination is an observability resource

σ_range (m), LEO, 1-orbit arc, drift-only — and this is the whole story of why a 3D table is needed:

| sep | ΔI=0 | 0.01° | 0.05° | 0.25° | 1.0° |
|---|---|---|---|---|---|
| 5 km | 681,895 | 4,064 | 1,532 | 302 | **68** |
| 10 km | 715,006 | 7,363 | 2,297 | 659 | 142 |
| 50 km | 218,040 | 35,439 | 7,460 | 2,323 | 851 |
| 200 km | 58,483 | 54,547 | 26,918 | 6,376 | 2,423 |
| 1000 km | 15,142 | 15,140 | 15,100 | 14,188 | 8,615 |
| 5000 km | 7,232 | 7,232 | 7,232 | 7,232 | 7,225 |
| 13000 km | 4,722 | 4,722 | 4,722 | 4,722 | 4,722 |

**168× at 5 km for 0.01° of relative inclination**, decaying to nothing beyond ~1000 km — the same crossover NAV-F measured for the maneuver (ρ* ≈ 300–500 km). This is not merely "bigger ρ": at *matched* mean separation the inclined geometry is ~500× better (5 km/ΔI=0.05° at ρ̄=6.4 km gives 1532 m; 10 km/ΔI=0 at ρ̄=10 km gives 715,006 m). Relative inclination plays the role in 3D that δa plays in 2D — permanent, bounded, and free — **but it is bounded by the same success box.**

Wide envelope confirms and extends: at a=12000 km, ρ=10 km, drift-only, e=0 → 1.04e6 m at ΔI=0, 771 m at 0.1°, 71 m at 1.0°; e=0.30 → 9,826 / 557 / 52 m. **Eccentricity helps by ~100× at ΔI=0** (NAV-G §2.3 measured the same sign at far field; this extends it to the terminal).

### 2.3 Reuse verdict for the shipped 2D map

639 cells comparable at matched (sep, arc, Δv):

- **ΔI = 0: exact.** The coplanar two-body FIM is block-diagonal in (in-plane 4-dof ⊕ out-of-plane 2-dof), the in-plane block is bit-for-bit the 2D single-bearing block, and the second angle contributes nothing to it. Verified numerically to ratio 1.0000.
- **Conservative up to 262×** wherever ΔI>0 and the arc is drift-dominated.
- **Optimistic in 6.1% of cells, worst 2.8×** — all in the corner (close separation, ΔI ≥ 0.25°, burn present) where the two extra out-of-plane dof consume information the 2D problem never had to pay for. Example: 5 km, 1 orbit, 5 m/s, ΔI=1° — map 39.0 m, truth 110.0 m.

**Recommendation: do not reuse the 2D table for the 3D lineage, even with a plane bolt-on.** Not because the plane is expensive (it is free) but because the *scalar* σ_k acquires a fifth axis that changes it by three orders of magnitude, and the direction of the error is not uniformly safe. Details in §4.

---

## 3. Dual control at the tight box (Q2)

`n3d_tightbox_dualcontrol.csv` — LEO 500 km, ρ₀ = 5 km, i_t = 51.6°. δa swept as NAV-F's map δa = v_tol/(1.5n); ΔI grid anchored on `ext_3d_box_plane_tol.csv` (5 km/1 m/s box ⇒ ΔI ≤ **0.0417°** position, **0.0075°** velocity; ⇒ plane position box 4.98 km, plane velocity box 0.998 m/s).

### 3.1 Range: the collapse survives, softened

Drift-only σ_range as a **fraction of the 5 km box**, 1-orbit arc:

| v_tol ⇒ δa | ΔI=0 | 0.0075° | **0.0417°** | 0.10° | 0.25° | 1.0° |
|---|---|---|---|---|---|---|
| 0.0 ⇒ 0 m | **∞** | 1.01 | 0.34 | 0.16 | 0.06 | 0.01 |
| 0.5 ⇒ 301 m | 0.92 | 0.67 | 0.29 | 0.16 | 0.06 | 0.01 |
| **1.0 ⇒ 601 m** | **0.47** | **0.43** | **0.24** | 0.14 | 0.06 | 0.01 |
| 2.0 ⇒ 1203 m | 0.25 | 0.24 | 0.18 | 0.12 | 0.06 | 0.01 |
| 50 ⇒ 30070 m | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.05 |

At 3 orbits every cell divides by ~4 (v_tol=1, ΔI=0: 0.47 → 0.07).

The operationally consistent terminal state is the (v_tol=1, ΔI ∈ [0.0075°, 0.0417°]) block: **σ_range = 43%→24% of the box**. NAV-F's 2D TB5 figure was 52%. So:

> **The 3D tight-box experiment is roughly 2× less live than the 2D one at the same box, and the liveness is a decreasing function of the plane residual the policy has not yet nulled.** The dual-control tension is *deferred to later in the approach* rather than removed — during the plane leg nav is easy (ΔI ≥ 0.25° ⇒ σ_range ≤ 6% of box), and it goes hard only once the plane is closed.

Structurally: **exact unobservability now requires δa = 0 *and* δi = 0 simultaneously** — codimension 2. The (v_tol=0, ΔI=0) cell is the only `∞` in the table; 0.0075° of residual inclination regularises it to 1.01× box at 1 orbit and 0.23× at 3.

### 3.2 Plane: does *not* collapse — but its velocity channel binds

| v_tol | ΔI | orb | σ_plane | /pos-box | σ_vplane | **/vel-box** |
|---|---|---|---|---|---|---|
| 1.0 | 0 | 1 | 1.05 m | 0.0002 | 0.0012 m/s | 0.001 |
| 1.0 | **0.0417°** | 1 | 1.22 m | 0.0002 | **1.316 m/s** | **1.32** |
| 1.0 | 0.0417° | 3 | 0.98 m | 0.0002 | 0.314 m/s | 0.31 |
| 0.0 | 0.0417° | 1 | 0.93 m | 0.0002 | **1.902 m/s** | **1.90** |
| 1.0 | 1.0° | 1 | 3.75 m | 0.0008 | 1.752 m/s | 1.75 |

**Plane position: never binds** — max 0.0038 of box over 419 cells. **Plane velocity: exceeds tolerance in 9.5% of cells, worst 2.19×.** The mechanism is the scale family, exactly:
```
σ_vplane = σ_k · v_c·sin(Δi_rel)        σ_k = σ_range/ρ₀
predicted 0.3436 × 5.54 = 1.903 m/s     measured 1.902 m/s
```
The asymmetry between plane position and plane velocity is a *phase* effect: my geometry puts the out-of-plane offset at zero at epoch (rotation about r̂), so the scale ambiguity has no position to scale but full velocity to scale. N3D-A's `f_oop` convention puts it at maximum at epoch. **The two conventions bracket the orbital phase; a training surrogate must not assume either.**

**Design consequence: the 3D-nav terminal needs ≥3 orbits (~4.5 h) of continuous bearing arc, or a burn, before the out-of-plane velocity estimate is inside the plane box.** Against a warp-heavy policy (NAV-F §3.1: 66.8% of terminal decisions are 1-hour warps), that is 4–5 consecutive warps of uninterrupted staring — achievable, but it is a real pre-registered failure mode to monitor.

### 3.3 The normal burn is the 3D observability maneuver

`n3d_burn_axis_placement.csv` — 1 m/s, four axes × eight placements. At the terminal (v_tol=1 m/s, ΔI=0, 1 orbit), post-burn σ_range (m) and information gain:

| placement | prograde | radial | **normal** | ⊥LOS in-plane |
|---|---|---|---|---|
| 0.00 | 2354 ×1 | 2354 ×1 | 2354 ×1 | 2354 ×1 |
| 0.05 | 728 ×10.5 | 133 ×314 | **131 ×324** | 137 ×295 |
| 0.10 | 197 ×143 | 70 ×1132 | **66 ×1302** | 70 ×1141 |
| **0.20** | 24.3 ×9416 | 20.4 ×1.34e4 | **15.2 ×2.39e4** | 16.9 ×1.93e4 |
| 0.35 | **11.5 ×4.2e4** | 25.1 ×8768 | 12.5 ×3.6e4 | 16.7 ×2.0e4 |
| 0.50 | **9.1 ×6.7e4** | 33.7 ×4871 | 13.6 ×3.0e4 | 20.7 ×1.3e4 |
| 0.95 | 622 ×14 | 178 ×174 | 167 ×198 | 161 ×213 |

Three readings, all load-bearing:

- **A burn at the start of the arc is worth exactly ×1** — NAV-F §2.5's 2D finding reproduced in 3D, unchanged.
- **Early (≤20% in), normal dominates**: 5.5× better than prograde at frac 0.05, 3.0× at 0.10, 1.6× (2.5× in information) at 0.20. Mechanism: prograde information accumulates as (remaining arc)² through secular along-track drift; normal produces an *immediate* bounded out-of-plane excursion, already ⊥ LOS.
- **Late (≥35%), prograde overtakes** (best single number in the table is prograde at 50%, 9.1 m). But **robustness to mis-placement**, which is what an RL policy on a 60 s decision grid actually has: over placements 0.05–0.75 the worst-case σ is **65.9 m for normal vs 365 m for prograde — 5.5×**. Prograde swings 99→9→545 m; normal stays inside 33–22 m over 0.10–0.75.

> **NAV-F §2.5 corrected for the 3D lineage.** Its recommendation ("radial ±1 m/s is the worst-directed 1 m/s burn; the along-track fine burns are 1.3–4.7× better") holds *within the plane*. In 3D there is a strictly better axis, it already exists in Discrete-30 (normal ±1 m/s), it is the axis the plane leg of guidance must fire anyway, and it is the *robust* choice under placement uncertainty. This makes the 3D dual-control hypothesis structurally easier to satisfy — and correspondingly harder to *detect*, because the treatment action is confounded with a guidance action. **The detection metric must be counterfactual information gain per burn (NAV-F §3.3 metric 1), not action-mix; action-mix is invalid in 3D by construction.**

### 3.4 Marginal value of the second angle (Q2d)

`n3d_tightbox_anglesplit.csv`, σ_range with only one of the two focal-plane angles retained:

| δa | ΔI | burn | both | in-plane only | out-of-plane only |
|---|---|---|---|---|---|
| 601 m | 0 | — | 2354 | **rank-deficient** | rank-deficient |
| 601 m | 0 | 1 m/s nor | 15.2 | rank-deficient | 198 |
| 601 m | 0.0417° | — | 1197 | rank-deficient | 5449 |
| 601 m | 0.0417° | 1 m/s nor | 16.0 | rank-deficient | **17.4** |
| 601 m | 1.0° | — | 68.4 | rank-deficient | 114 |

The in-plane angle alone can never see the two out-of-plane dof, so the 6-state FIM is structurally singular — **the second angle is not optional in 3D; it is what makes the 6-state estimable at all.** Once a plane residual and a normal burn exist, the out-of-plane angle alone recovers range to within **1.09×** of the two-angle bound: the plane channel is not a passenger, it carries most of the terminal range information.

---

## 4. Surrogate spec (Q3)

### 4.1 Runtime path — keep `crlb_online`, lift to 6 states

The shipped 2D surrogate defaults to `acq_mode='crlb_online'` because the table's co-orbital δa=0 parametrisation is catastrophically wrong for the geometry the policy actually flies (`nav_surrogate.py` docstring: G1 SINGULAR vs 223 m truth, G2 9.2×). **Every one of those arguments holds a fortiori in 3D, and there is now a second: the ΔI axis moves σ_k by up to 2000× at close range.** Recommendation: `crlb_online` remains the sole default; the table is a design/reporting artifact and a degraded fallback.

Lift, mechanically:

| element | 2D shipped | 3D |
|---|---|---|
| state | 4 (x,y,vx,vy) | 6 |
| `Phi` | `nm.stm_fd` (8 props) | 6×6 central-difference (12 props) |
| `H` per tick | `Hr = [-dy/r², dx/r², 0, 0]` | `M[:3,:3] = P_⊥/(σ_β ρ)²`, `P_⊥ = I − uuᵀ` — **basis-free, no az/el frame, no pole guard** |
| FIM accumulation | raw | **nondimensionalised by `D = diag(a,a,a,v_c,v_c,v_c)`** (`Φ̂ = D⁻¹ΦD`, `M̂ = DMD`); mandatory, see §1 |
| singular guard | `cond > 1e16` on raw | `cond > 1e14` on **scaled** (≈2 decades more usable range; also fixes the `σ=0.0` defect) |
| gate | `σ_LOS/ρ ≤ 0.20`, 45-tick floor | **unchanged** — the plane never binds the gate (§2.1) |
| `Σ` at handoff | `ratio²·Φ F⁻¹ Φᵀ`, eigen-floored at 1e-12·λ_max | unchanged in form, 6×6 |

**Cost.** N3D-A measured (`n3d_filter6_cost.csv`) tick6/tick4 = **2.25× at B=256, 2.84× at B=1024, 3.29× at B=4096** — at B=1024, 2.57 ms/tick vs 0.90 ms. Charged only while a row is unacquired (~7% of ticks at the canonical policy's 680-tick mean episode), so the marginal training cost is ~0.12 ms/step ≈ well inside budget. No new blocker.

### 4.2 The rank-1 scale model — what the table would have to carry

`n3d_rank1_scale_model.csv`, 1800 component-cells (5 v_tol × 5 ΔI × 3 burn × 2 sep × 2 arc × 6 components):

```
Σ  ≈  σ_k² · x_rel x_relᵀ  +  floor² · P_⊥-block
σ_k    = σ_range / ρ₀                                    (the ONLY tabulated scalar)
floor  = √2 · σ_β / √(Σ_k ρ_k⁻²)                          (position; × n_target for velocity)
σ_ê    = hypot( σ_k·|x_rel·ê| , floor )                   for any direction ê
```

| component | n | p05 | median | p95 | within 1.5× |
|---|---|---|---|---|---|
| pos_range | 300 | 1.000 | 1.000 | 1.023 | 100% |
| pos_inplane | 300 | 0.448 | 0.793 | 1.279 | 73% |
| pos_plane | 300 | 0.535 | 0.987 | 1.054 | **89%** |
| vel_range | 300 | 0.499 | 1.004 | 1.531 | 80% |
| vel_inplane | 300 | 0.826 | 0.999 | 1.335 | 98% |
| vel_plane | 300 | 0.736 | 1.002 | 1.168 | **96%** |

`pos_inplane` is the weak component (floor-dominated at epoch for an along-track geometry; the `√2` constant is crude there) — it is also the least consequential, being 3–5 orders below σ_range. **This is why the plane needs no table: it is `σ_k` projected onto `ê_p`, plus a closed-form floor.** In `crlb_online` mode it comes out of `Φ F⁻¹ Φᵀ` exactly and requires *no code at all*.

### 4.3 Table spec (`table` mode / design artifact)

If the table is built — recommended for reporting and as the degraded fallback, **not** the default:

**Axes (4, log-interpolated, single scalar value `log₁₀ σ_k`):**

| axis | grid | interpolation |
|---|---|---|
| separation | 5, 10, 50, 200, 1000, 5000, 13000 km | `log₁₀ sep` |
| arc | 0.25, 0.5, 1, 2, 3 orbits | `log₁₀ orbits` |
| Δv | 0, 1, 5, 25 m/s | `log₁₀(1+Δv)` |
| **ΔI_rel** | **0, 0.01, 0.05, 0.25, 1.0 deg** | **`log₁₀(1 + ΔI/0.01°)`** — 3 decades of dynamic range at close separation |

Δv is **not** split by axis in the table: at 35% placement the axis effect is ≤1.7× (map), which is inside the table's own 2× fidelity, and the axis effect that matters (§3.3) is a *placement* interaction the table cannot represent. Instead, **conditioning variables passed to `rel_sigma()`**:

1. `Δv_total` realised inside the acquisition arc (unchanged, load-bearing — a Δv-independent surrogate destroys the observability claim).
2. **`Δv_normal` realised, separately** — so a policy that fires the plane leg is credited for the information it actually bought (2.5–8× at 20% placement), and not credited for information it did not.
3. **`Δi_rel` realised** — legitimate and load-bearing for the same reason Δv is: the policy's plane burns change it. Read from the C env's own `δı⃗` (obs slots 21/22 decode) rather than recomputed.
4. `δa` realised — implicit and exact in `crlb_online`; **the documented reason the table cannot be the default** (`nav_surrogate.py` §"Two acquisition modes").

Plane bolt-on for table mode, one line, no new grid:
```
sigma_plane_pos = hypot( sigma_k * |x_rel_pos . e_p| , sqrt2 * sigma_b / sqrt(sum 1/rho_k^2) )
sigma_plane_vel = hypot( sigma_k * v_c * sin(di_rel)  , n_target * (same floor) )
```

### 4.4 Calibration ratios (CRLB → filter)

The CRLB is the information floor; the shipped batch+recursive filter sits above it by a geometry-dependent factor. 2D measured (`ext_bo_filter.csv`, 24 seeds/cell, `BLSRatioTable`): drift 5.50 / 1.73 / 1.97 / 3.62 at ρ/r = 0.0057 / 0.127 / 0.841 / 2.004; burn 2.95 / 5.16 at ρ/r = 0.0022 / 0.0118.

**N3D-A has not yet produced 3D filter settled-error measurements** (`n3d_filter6_cost.csv` is timing only, `n3d_prop6_validation.csv` is propagation fidelity). The spec is therefore a procedure plus a prior, not a number:

- **Prior:** carry the 2D `BLSRatioTable` unchanged for the range channel. Justification: the ratio is a re-linearisation/consistency penalty, not a geometric one, and the 3D lift changes neither the estimator family nor the ambiguity structure.
- **New channel to measure:** `ratio_plane = settled out-of-plane RMSE / σ_plane_CRLB`. **Predicted ≈ 1.0–1.5** — the out-of-plane update is linear and well-conditioned, so a filter should nearly attain the bound, unlike range (2.9–5.5×). If measured `ratio_plane > 2`, the filter has a plane-channel defect, not an information limit. This is a falsifiable prediction, and it is cheap.
- **Interpolant:** `ratio(log₁₀(ρ/r), has_burn)` per channel, exactly the shipped 2D form, with a third split on `Δi_rel > 0` if the measured plane ratio moves by >1.5× across it.
- **Blocking dependency for the 3D-nav campaign:** the 6-state bearings-only filter prototype must report settled position/velocity RMSE decomposed into the **(range, in-plane, plane)** triad, over ≥6 geometry classes × ≥24 noise seeds. Reporting only a scalar RMSE would make the plane ratio unmeasurable and the surrogate uncalibratable in exactly the channel that is new.

### 4.5 Validation gate (QQ vs prototype)

Per geometry class, from `AcquisitionSurrogate.log_draws` extended to 6 states, against prototype errors on matched geometries:

1. **Whitened-residual QQ.** `z = L⁻¹(x̂ − x_true)`, `L = chol(Σ)`. Report QQ per triad component (range / in-plane / plane, position and velocity) — **6 plots, not one** — plus KS statistic. **Gate: KS p > 0.05 on each of the 6, per class.**
2. **NEES.** 6-dof, `cov_inflate` chi-square bounds. **Gate: in-bounds fraction ∈ [0.90, 1.00]** (the 2D shipped filter achieves 0.91–0.99).
3. **Scale ratio.** Predicted 1σ / realised RMSE per component. **Gate: ∈ [0.80, 1.25]** (2D shipped surrogate: "reproduces all six measured geometries within 1.0–1.2×").
4. **Anisotropy.** Predicted vs realised eigenvector of the dominant error direction. **Gate: angle to `x_rel` < 15°** — this is what falsifies the rank-1 model on real filter output rather than on the CRLB.

**Geometry classes (a 3-way factorial, ≥8 cells):**
`separation` {terminal ≤10 km, near 10–100 km, mid 100–1000 km, far ≥1000 km} × `plane` {coplanar ΔI=0, ΔI ≤ box tol 0.0417°, ΔI > box tol} × `actuation` {drift, in-plane burn, **normal burn**}. The normal-burn cells are non-negotiable: 3D-E §5's coverage ablation showed dropping the normal-burn policies made two seeded bug classes *undetectable*, and the same logic applies here — a validation set with no normal burns cannot see a plane-channel calibration error.

### 4.6 Integration findings for the 3D-nav wrapper

- **Obs-slot collision, concrete.** `OrbitalNav`'s `nav_sigma_channel` writes **`obs[21]`** (`_write_sigma_channel`, chosen because slots 21–32 are identically zero in every 2D/T3 config). Under `dim3_mode=1`, **`obs[21]` and `obs[22]` are the `δı⃗·R̂_s` / `δı⃗·T̂_s` plane channels** (`orbital.h:1113–1114`) — the exact quantities the plane leg of guidance flies on. **The Σ channel must move to `obs[29..32]`, which `orbital.h:1146–1149` reserves as `0.0f` under 3D.** Four free slots: `σ_LOS/ρ`, `√tr P`, time-since-update, nav-valid — the full NAV-F §3.4 `T-BO+Σ` arm fits without a dim change.
- **`nm.TARGET_SLOTS_T3` is a 2D slot list** and must be re-derived for the 3D layout (slots 7–15, 21–27 are target-derived under `dim3_mode=1`); the `fill_target_obs_t3` re-encode path needs a 3D twin.
- **NAV-F §3.2's arm table transfers unchanged**, with one addition: a **`T-BO−normal`** arm (normal burns masked inside the terminal box) is now the sharpest necessity test, because §3.3 makes the normal axis the treatment and the guidance action simultaneously.

---

## 5. J2 (designed-for-later)

`j2_n3d_observability.csv`, 144 rows. Truth roll and 6×6 STM both under the same RK4 flow (J2 on/off), so the comparison is like-for-like; the J2-off arm doubles as an RK4 control (**1.20e-8 rel vs the universal-variable oracle over one orbit; J2-STM symplecticity 1.68e-9**).

| i_t | δa | ΔI | arc | ΔΩ̇·T (deg/orbit) | two-body | J2 | gain |
|---|---|---|---|---|---|---|---|
| 51.6° | 0 | 0 | 3 orb | 0 | 285,574 m | 19,525 m | **214×** |
| 97.4° | 0 | 0 | 3 orb | 0 | 281,886 m | 12,846 m | **482×** |
| 51.6° | 0 | 0.0417° | 3 orb | 2.9e-4 | 683.7 | 682.8 | 1.00 |
| 51.6° | 601 m | 0 | 3 orb | 9.6e-5 | 343.3 | 346.8 | 0.98 |
| 51.6° | 601 m | 0.0417° | 3 orb | 3.8e-4 | 285.5 | 284.9 | 1.00 |
| 51.6° | 30 km | 1.0° | 3 orb | 1.2e-2 | 49.7 | 49.4 | 1.01 |

**J2 gain is within ±5% in 34 of 36 cells.** It matters only in the one cell that is exactly degenerate under two-body — and even there leaves 19.5 km, four times the box. Differential nodal drift at δa ≤ 30 km / δi ≤ 1° is ~10⁻³ deg/orbit; over 3 orbits that is ~1.7 km of extra out-of-plane displacement, which is already dominated by the Keplerian curvature term wherever curvature is doing any work at all.

**Design consequences:**
- The standing caveat ("real LEO leans on J2, our two-body sim is harder than reality — defensible framing, but state it") should be **retired for this env's regime** and replaced with the measured statement. It remains true for multi-day IROD and for the mean→osculating literature (Ardaens & Gaias 2019, ARGON/AVANTI), which operate over spans 10–100× longer.
- **This probe measured matched-model J2 (truth and filter both J2).** It says nothing about *mis-modelled* J2 — a filter propagating two-body against a J2 truth — which is the failure mode that actually bites, and which the sibling J2 lane's `j2_nav_modelerr.csv` addresses. Do not read this as "J2 is safe to ignore in the filter."
- If `j2_mode=1` ships, the observability map does **not** need regeneration; the `cos i_s` / `cos i_t` obs slots (28/29) and the surrogate are unaffected at ≤3-orbit arcs.

---

## 6. Limitations

- Single target, single observer, no FOV / occultation / pointing / association model. A FOV constraint would bite acquisition first, and would interact with the normal-burn recommendation (out-of-plane motion moves the target across the focal plane fastest).
- `n3d_obs_map.csv`'s LEO block is δa = 0 by construction (mirroring the 2D map for cell-for-cell comparability). That is the degenerate case; the δa dependence lives in `n3d_tightbox_dualcontrol.csv`. **Do not read σ_range off the map for a drifting geometry** — this is the same trap that makes the 2D table unusable at runtime.
- One geometry family per (separation, ΔI); NAV-F §2.3's 100 km resonance shows observability is not a pure function of separation. Not geometry-randomised.
- ΔI_rel enters via a rotation about r̂ at epoch, so the out-of-plane excursion is **zero at epoch and maximal a quarter orbit later**. N3D-A's `f_oop` convention is the opposite phase. The two bracket the effect; neither is "the" answer, and the plane position/velocity asymmetry in §3.2 is a direct consequence of the choice. A phase-randomised sweep would firm this up and is the single highest-value follow-on.
- Burn model is a single impulse per arc. Real policies fire repeatedly; information from multiple burns is not additive in general.
- CRLB is a *bound*. Every number here is optimistic by the calibration ratio of §4.4, which for the range channel is measured at 2.9–5.5× in 2D and **is not yet measured at all in 3D**.
- `σ_β = 1 mrad`, 60 s cadence throughout. NAV-G's 2D sweep found error exactly linear in σ_β; that scaling was assumed, not re-measured, here.