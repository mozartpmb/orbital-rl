## 3D-E deliverable — independent 3D oracle, invariants, thresholds, harness plan

### Files (all new, nothing existing modified, nothing committed)

| path | what | status |
|---|---|---|
| `/Users/pete/space_training/scripts/orbital/ext_recon/orbital_math3d.py` | the 3D two-body oracle + 56-check validation battery | **56/56 PASS**, 6.1 s |
| `/Users/pete/space_training/scripts/orbital/ext_recon/ext_invariants3d.py` | the 3D invariant battery (`INVARIANTS`, importable by the future harness) + a mutation test that proves each check fires | **17/17 quiet on the correct model; 15/15 seeded bugs caught; zero dead checks**, 31 s |
| `/Users/pete/space_training/web_data/results/ext_3d_oracle_validation.csv` | per-check value/threshold/verdict | written |
| `/Users/pete/space_training/web_data/results/ext_3d_invariant_mutation_matrix.csv` | bug × invariant residual matrix | written |

Run: `python3 scripts/orbital/ext_recon/orbital_math3d.py` and `python3 scripts/orbital/ext_recon/ext_invariants3d.py` (add `--quick` / `--steps 40` for a fast pass).

---

### 1. The oracle — and why it can catch the env's bugs

Deliberately a different algebra from anything the env will plausibly contain, so agreement is evidence rather than a shared assumption:

- **Propagation: universal variables + Stumpff c2/c3**, bracketed-Newton on a strictly increasing `F(χ)` (`F'(χ)=r>0`), bracket grown geometrically. Conic-agnostic (ellipse / near-parabolic seam / hyperbola all one path), any sign of `dt`, many revolutions. The 2D oracle is ellipse-only f&g in `ΔE`; the env is element-wise Kepler. **Three independent formulations.**
- **Elements: signed `atan2` against the orbit's own frame vectors** (`signed_angle(u,w,axis)`). There is not one `acos`+if-sign branch in the file — that is precisely the bug class it must indict (B2/B3/B4 below).
- **Degenerate cases resolved by one convention that makes `coe2rv ∘ rv2coe` the identity**, which is the only definition that can be regression-tested: equatorial ⇒ `RAAN:=0` with node `:= x̂` and `argp` = signed angle about `ĥ` (this single formula gives the standard longitude-of-periapsis at `i=0` **and** its mirror at `i=π`); circular ⇒ `argp:=0`, `nu:=` argument of latitude.
- **Anomaly solvers are monotone-bracketed**, not fixed-iteration Newton — the env's 5-iteration budget is itself something the oracle must be able to indict.

**Validation results (all PASS):**

| case | measured |
|---|---|
| coe→rv→coe→rv round trip, 11 regimes incl. hyperbolic | 8.4e-15 rel (non-degenerate) |
| universal-variable vs mean-anomaly element propagation | 1.2e-11 rel |
| universal-variable vs **scipy DOP853** (`rtol 1e-13`) — third independent path | 3.1e-11 rel |
| coast invariants: energy / h-vector / e-vector | 1.9e-12 / 4.5e-12 / 3.2e-12 |
| semigroup: 1×3600 s vs 60×60 s, and time reversal | 2.0e-11 rel |
| **2D cross-check vs `nav/orbital_math.py`** (required 1e-9) | coe2rv **2.8e-16**; rv2coe a **1.4e-15**, e **7.8e-16**, `argp+nu` vs `omega+theta` **2.3e-13**; universal vs f&g propagation **5.2e-12** |
| circular-equatorial, polar (`h_z/h` = 6e-17), Molniya (perigee/apogee/`i`/`argp` exact; period vs 12 h sidereal 4.4e-4), retrograde `i=120°` (node direction 0.0), **retrograde-equatorial `i=π`** (round trip 2.8e-16, motion verified clockwise) | PASS |
| pure plane change at the node: `|Δv|` vs `2v sin(Δi/2)` 1.7e-16, `Δi` realised 8.3e-17, RAAN and `a` unchanged exactly | PASS |

### 2. Findings that change ext-3d's design

**F1 — `i = acos(h_z/|h|)` loses half the mantissa near `i≈0` and `i≈π`.** Measured 1.4e-8 rad at `i~1e-8`, which leaks straight into every plane-invariance test (my own V8 "in-plane burn changes `i`" failed at 1.5e-8 before the fix). Oracle now uses `i = atan2(hypot(h_x,h_y), h_z)`. **Recommendation: the 3D env must do the same** — otherwise its own reported `i` is noisier than the invariant threshold, and near-equatorial scenarios (the ones adjacent to the validated 2D lineage) are exactly where the campaign starts.

**F2 — classical elements have an irreducible ~2e-11 round-trip error, and the current env architecture pays it on every burn.** `orbital.h` stores elements and does `orbit_to_cartesian → burn → cartesian_to_elements` per impulse. Measured sweep (`V9`, in the CSV):

| `e` | 0 | 1e-13 | 1e-12 | **1e-11** | 1e-10 | ≥1e-8 |
|---|---|---|---|---|---|---|
| rel err | 1.7e-15 | 2.0e-13 | 2.0e-12 | **2.0e-11** | 1.4e-15 | ~1.3e-15 |

Identical curve versus `sin(i)` (peak **1.9e-11** at `sin i = 1e-11`). The peak sits at the degeneracy switch point and is *optimal*, not a tuning failure: keeping `argp` when `e` is tiny costs `d²/2` with `d ~ 1e-16/e`; collapsing to `argp=0` costs `O(e)`; they cross at `e ≈ 1.7e-11`. **Recommendation: ext-3d should hold Cartesian state and derive elements for obs/reward only, or use equinoctial elements.** In 2D this was tolerable; in 3D you acquire a second seam (`sin i`) that near-equatorial and near-circular training draws will sit on, and it sets the floor for the entire invariant battery (I7's clean residual is 4.3e-5 m, purely this).

**F3 — `nav/orbital_math.py:60` `true_to_mean` is still the pre-fix inverted version** (its docstring says so explicitly; `mean_from_true` is the correct one). Post-`f55d9cb` it therefore *mis*-mirrors the C. `eval_relnav.py:161-189` already works around it with a local `_mean_anomaly`. Inert today, but ext-nav will reuse this module — **any new code must call `mean_from_true`, never `true_to_mean`.** My V7 cross-check is unaffected (it never compares `M`).

**F4 — `apply_impulse`'s basis is non-orthogonal** (`prograde = v̂`, `radial = r̂`, equal only at `e=0`). The 3D lift must take `normal = ĥ` (orthogonal to both) and must compute `|Δv|` as the norm of the *assembled vector*, never a sum or quadrature of components — otherwise the fuel ledger is wrong, not just the log (seeded as B9, caught by I9/I11/I15).

---

### 3. Invariant list, with thresholds

Implemented as `INVARIANTS` in `ext_invariants3d.py` — `(label, fn(traj), threshold, applies(scenario, policy))`. Thresholds below are for a **double-precision** log; `FLOAT32_THRESHOLDS` in the same file gives the ≥8× float32-noise variants for reading the C env's float32 trajectory buffer (mirrors the 2D harness's convention so the two reports are directly comparable).

| id | invariant | thresh (double) | thresh (f32 log) | clean residual |
|---|---|---|---|---|
| I1 | energy invariant across coast, scaled by `μ/r` | 1e-12 | 1e-6 | 7.3e-16 |
| **I2** | **h-VECTOR across coast, all 3 components, `|Δh|/|h|`** | 1e-12 | 1e-6 | 7.9e-16 |
| **I3** | **`ĥ` direction across coast (rad)** — plane orientation, separated from magnitude | 1e-12 | 1e-6 | 3.4e-16 |
| I4 | RAAN across coast (rad) — the form that appears in the obs vector | 1e-10 | 1e-5 | 8.9e-16 |
| I5 | e-VECTOR across coast (apsidal line must not precess) | 1e-11 | 1e-6 | 1.4e-15 |
| I6 | step-local oracle re-sim (burn + coast), rel position | 1e-9 | 10 m | 6.3e-12 |
| I7 | impulse moves position (m) — element round-trip witness | 1e-3 | 5 m | 4.3e-5 |
| **I8** | **in-plane burn tilts the plane (rad)** — *the killer regression* | 1e-9 | 1e-6 | 6.4e-12 |
| I8b | same event as `max(|Δi|,|ΔRAAN|)` (the obs-visible form) | 1e-9 | 1e-6 | 6.4e-12 |
| I9 | burn `ΔE = v·Δv + |Δv|²/2` **and `Δh_vec = r × Δv` (3 comps)** | 1e-6 | 1e-4 | 3.2e-8 |
| **I15** | **realised vs commanded Δv direction (rad)** — commanded-frame fidelity | 1e-4 | 1e-3 | 3.9e-8 |
| I10 | argument-of-latitude continuity across an in-plane impulse — the 3D lift of the 2D `θ+ω` invariant | 1e-9 | 1e-5 | 3.6e-15 |
| I10b | `r̂` rebuilt from the **logged** `(i, RAAN, argp+nu)` vs the logged position | 1e-9 | 1e-5 | 1.1e-15 |
| I11 | logged `|Δv|` vs realised `|v_post − v_pre|` | 1e-5 | 1e-3 | 1.2e-9 |
| I12 | logged elements vs logged state (`a`, `e`, `i` + full state rebuilt through the oracle) | 1e-9 | 1e-5 | 3.9e-15 |
| **I13** | **2D-compat anchor**: `i=0`, no normal burns ⇒ `z=vz=0` exactly and in-plane motion matches the validated 2D oracle | 1e-9 | 1e-6 | 1.0e-15 |
| I14 | angle domains (`i∈[0,π]`, `RAAN/argp/nu∈[0,2π)`), `a>0`, `e∈[0,1)`, finiteness | 0 | 0 | 0 |

Normalization notes that matter (each was a false alarm before fixing): **I9's energy term must be scaled by `|v||Δv|`, not `|v·Δv|`** — a pure normal burn has `v·Δv = 0` by construction, so the natural-looking normalizer divides by noise on exactly the manoeuvre class this extension is about; **I9's h-term must be scaled by `|r||Δv|`, not `|Δh_pred|`** — a purely radial burn has `r×Δv = 0` exactly. **I12 must not compare `RAAN`/`argp`/`nu` individually** — a degeneracy-convention difference between env and oracle is a reporting choice, not a physics bug; rebuild the state instead.

### 4. Mutation test — proof the battery fires

15 seeded 3D bug classes in a naive classical-element reference env (`Model3D`), which is deliberately the architecture `orbital.h` would grow into. **Every bug caught; every invariant earns its place.**

| bug | caught by |
|---|---|
| B1 normal burn along `ẑ` instead of `ĥ` | I6, **I15** |
| B2 RAAN from `acos`, no `n_y` sign fix | I6, I7, I8, I8b, I9, I11, I15 |
| B3 `argp` from `acos`, no `e_z` sign fix | I6, I7, I9, I10, I11, I15 |
| B4 `nu` from `acos`, no `r·v` sign fix | I6, I7, I9, I10, I11, I15 |
| B5 `nu→M` half-angle inverted (**2026-08-10 bug's 3D twin**) | I6, I13 — *note: energy/h checks stay silent, exactly as in 2D* |
| B6 `R1(+i)` (mirrored plane) | 10 checks |
| B7 3-1-3 rotation order reversed | 10 checks |
| B8 `h = v×r` (normal reversed) | I6, I8b, I10, I11, I15 |
| B9 `|Δv|` as sum of frame components | I6, I9, I11, I15 |
| B10 `i`/RAAN not refreshed after a burn | I6, I7, I9, I10, I11, I15 |
| **B11 plane rotates during COAST** (stray nodal rate) | **I2, I3, I4, I5**, I6, I13 |
| B12 normal axis forced to point north | I6, I15 |
| B13 radial axis as the x-y projection of `r` ("2D lifted") | I6, I8, I8b, I10, I15 |
| B14 RAAN left on `(-π,π]` | **I14 only** |
| **B15 `a` decays during COAST** (stray perturbation) | **I1**, I2 |

Two structural lessons encoded here: **B5's 3D twin is invisible to every conservation check** — only oracle re-sim and the 2D anchor see it, which is why I6 is non-negotiable; and **B11/B15 are invisible to everything except the coast-time vector invariants**, which is the direct answer to "why a 3-component h check rather than `|h|`". Both are the "someone left a J2/drag term in the two-body propagator" class — highly live, since J2 is on the roadmap.

One seeded bug was discarded as a *mathematical no-op*: "normal axis taken after the in-plane Δv is added" — `r × (v + dv_pro·v̂ + dv_rad·r̂) ∥ r × v`, so the direction is unchanged. Worth knowing before someone writes a defect report about it.

### 5. Test matrix — and evidence each cell is load-bearing

**Scenarios (8):** `C0 circ-equatorial` (the 2D anchor), `C1 circ i=28.5°`, `C2 e=0.05 i=51.6°`, `C3 polar i=90°`, `C4 retrograde i=116°`, `C5 Molniya e=0.70 i=63.4°`, `C6 near-equatorial i=0.001°`, `C7 near-polar i=89.999°`.

**Policies (7):** `uniform`, `inplane_only`, `normal_only`, **`normal_at_node`**, **`normal_at_antinode`**, `burn_burst`, `coast_heavy`. The node/anti-node pair separates a plane-change implementation that works at the node from one that works anywhere — the classic partial-correctness failure.

Coverage ablation (measured, `evaluate()` restricted):

| bug | full matrix | prograde scenarios only | in-plane policies only |
|---|---|---|---|
| B12 normal-forced-north | 2 checks fire | **0 — undetectable** | **0** |
| B1 normal-along-`ẑ` | 2 | 2 | **0 — undetectable** |
| B13 radial-inertial | 5 | 5 | 5 |

I.e. **dropping the retrograde scenarios or the normal-burn policies creates real blind spots**; the matrix is not decoration.

### 6. Harness plan — `scripts/orbital/ext_recon/fuzz_dynamics3d.py`

Mirrors `t3/fuzz_dynamics.py` structurally so the two reports read the same:

1. **Static/unit stage first** (no env needed): `check_L` 3D analogue — `nu→M→(Kepler)→E→nu` identity over `e ≤ 0.5`; `check_L2` — the env's 5-iteration Newton residual vs `e`, converted to along-track metres; **new `check_L3`** — `coe2rv ∘ rv2coe` identity of the env's *own* conversion pair over the 11 regimes, reported against the F2 conditioning curve.
2. **Drive the real C env** with the 8×7 matrix, `--episodes-per-cell` as in 2D; pull the per-sub-step trajectory via `binding.vec_get_trajectory`; call `INVARIANTS` from `ext_invariants3d.py` unchanged, with `FLOAT32_THRESHOLDS`.
3. **Reuse the 2D `Acc`** accumulator (max/p99/count/worst-context), same CSV shape, into `web_data/results/ext_3d_fuzz_dynamics.csv` + `ext_3d_fuzz_coverage.csv`.
4. **Regression anchors, both mandatory before any 3D training claim:** (a) `i=0`, normal burns masked ⇒ the 3D env reproduces the current 2D env **bit-exactly** on the T3 canonical config (this is the runtime-flagged/legacy-default discipline applied to geometry); (b) I13 green on the `C0 × {inplane_only, coast_heavy}` cells.
5. **Carry over the 2D checks that are dimension-independent** and are *not* in this list because they are not 3D-specific: `D` Tsiolkovsky bookkeeping, `E` Δv-table match under fuel clamp, `F` warp `τ` bookkeeping + zero Δv on warps, `H` observation cross-check (extended to the new `sin/cos i`, `sin/cos RAAN`, relative-plane-angle channels and the **6-component** LVLH block), `K` terminal-cause consistency with a 3-component miss distance, `M` reward = shaping delta, `Q` collision tunnelling (now a 3D chord test).

**What the env must expose for this to run** (the current `TRAJ_COLS` is 2D-only): `sat_z, sat_vz, sat_i, sat_raan`, `target_z, target_vz, target_i, target_raan`, and body `z`. The commanded `(dv_pro, dv_rad, dv_nor)` triple can stay implicit via the `action` column + table lookup, as in 2D — but `delta_v` must remain the *realised* magnitude, or I6/I11 misread every fuel-clamped burn as a dynamics error (this cost me a false 4e-4 alarm during development).

**Not yet covered / open:** no check exists for a 3D **Lambert/two-impulse comparator** (the T1 physics-consistency anchor) — that needs a 3D Lambert solver and should be a separate deliverable; and the oracle has no J2, so the moment a perturbation is deliberately added, I1–I5 must be re-specified as *bounded secular rates* rather than invariants.