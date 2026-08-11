All probes complete. Nothing in the repo modified, nothing committed.

---

# ext-3d adversarial review — SHIP WITH CHANGES

Probes: `scripts/orbital/ext_recon/redteam3d/rt3d_{common,p1_sampler,p2_a2_bitexact,p3_phi,p4_obs,p5_macro,p6_numerics,p7_phasegap}.py`
CSVs: `web_data/results/ext_rt3d_*.csv` (14 files). Geometry written independently of `orbital.h` and of `orbital_math3d.py`.

**Headline: two blockers, six majors. The Φ design survives its adversarial tests essentially intact — the failures are in sampling, anchoring, coordinates, and the ladder.** Also: reports A, B and C disagree with each other on the combiner, on `di_max`, on the obs layout, and on the normal quanta, and the prompt's synthesis silently picks a different winner in each case. Three of those picks are wrong.

---

## BLOCKER 1 — the `di_max` sampler in 3d_A §4 is catastrophically wrong at i_t > 0

**Attack.** The reports specify two different samplers. 3d_C §4.4(a) gives prose (`ĥ_s = R(δ,n̂)ĥ_t`); **3d_A §4 gives the actual `c_reset` C code an implementer will paste**: add an area-uniform disc to `ī = (sin i cosΩ, sin i sinΩ)`, then `inc = asin(|ī|)`. Neither report ever measured 3d_A's version. The `ī` representation is a *projection*: it under-measures pure-inclination differences by `cos i_t` and is degenerate at i = 90°.

**Measurement** (`ext_rt3d_sampler.csv`, 40k draws/cell, realized = true angle between ĥ's):

| i_t | 3d_A ī-disc: max/knob | frac over knob | Δv_max @ knob=1° |
|---|---|---|---|
| 0° | 1.00× | 0.0% | 132 m/s |
| 28.5° | 1.14× | 12.5% | 152 |
| 51.6° (ISS) | 1.64× | **38.0%** | 217 |
| 80° | 10.0× | **82.7%** | 1324 (2.8× budget) |
| 90° | 10.7× | 49.5% | 1417 |
| 98° | **21.4×** | **100%** | **2816 m/s = 5.9× budget** |

3d_C's rotation sampler: max/knob = 1.0000, 0.0% over, **at every i_t** — exactly as claimed.

Worse: `inc = asin(si)` cannot produce i > 90°. Both 3d_A §4 and 3d_B §4 open `i_min/i_max` to **0–98° at every rung**. At i_t = 98° the chaser folds to ~82°, giving a **median 16.0° relative inclination — 2158 m/s, 4.5× the entire budget, every episode unconditionally infeasible.**

**Fix.** Delete 3d_A §4's `c_reset` block. Sample `ĥ_s = R(δ,n̂)·ĥ_t` with `δ = di_max·√U`, `n̂ = û₁cosφ + û₂sinφ` uniform in the target plane, then recover `(inc, raan)` from ĥ_s via `atan2`. Pre-flight gate: realized `Δi_rel` max ≤ `di_max` and 0.0% over, measured at i_t ∈ {0, 51.6, 90, 98}°.

---

## BLOCKER 2 — anchor A2 ("Φ mode2 @ Δi=0 == mode1 bit-exact") is false as specced

**Attack.** Shipped mode 1 (`orbital.h:805-818`) uses `sqrt(da*da + de*de)` with `de` from *element-derived* 2-vectors `(e cosω, e sinω)`. 3d_C §3 specs `hypot(δa, Δē)` with `Δē` from the *inertial 3-vector* `(v×h)/μ − r̂`. Two different FP paths.

**Measurement** (`ext_rt3d_a2_bitexact.csv`, 200k / 20k draws at the L2 distribution):

| variant | double mismatch | one-step **reward** f32 mismatch | max abs ΔΦ |
|---|---|---|---|
| **V0 = the literal 3d_C spec** (hypot + Cartesian ē) | **87.71%** | **0.010%** (max rel 1.1e-7) | 6.9e-15 |
| V1 sqrt + Cartesian ē | 87.72% | 0.010% | 6.9e-15 |
| V2 hypot + element ē | 4.99% | 0.000% | 2.2e-16 |
| **V3 sqrt + element ē** | **0.0000%** (0/200000) | **0.0000%** (0/60000) | 0.0 |

The break survives the float32 cast: 1 reward in 10⁴ differs, so a 3000-step episode diverges with ~0.3 expected mismatches and a 200-episode anchor with ~60. `Δv_pl` at Δi=0 *is* exactly 0.0 (ĥ = (0,0,±1) bitwise from both element and Cartesian paths), so the plane term is not the problem — it is `hypot()` vs `sqrt(x*x+y*y)` and the ē path.

**Fix (exact formulation exists — do not downgrade the anchor).** In `compute_phi` mode 2: build `ē` from elements (`e·(cosΩcosω − sinΩsinωcos i, …)`), compute `de = sqrt(dx*dx+dy*dy+dz*dz)`, then `dv_in = 0.5*v_t*sqrt(da_rel*da_rel + de*de)` — **not** `hypot`. Also run A2 with mode 1's *own* weights; see MAJOR 6.

---

## MAJOR 1 — `Δλ = M+ω+Ω` is frame-VARIANT at Δi ≠ 0; `frame_randomize` is a broken test

**Attack.** Ω is measured from the inertial x̂. When the two planes differ, a global SO(3) rotation moves Ω_s and Ω_t by *different* amounts, so `λ_s − λ_t` is not a function of the physical relative state. 3d_C §4.4(c) proposes `frame_randomize=1` as an obs-leak regression on the premise that every channel is invariant.

**Measurement** (`ext_rt3d_frame_invariance.csv`, 4000 random SO(3) rotations/cell, target gauge i_t=Ω_t=0):

| Δi | ‖ΔΔλ‖ p50 | p90 | max | max in Φ units | target-gauge fix |
|---|---|---|---|---|---|
| 0° | 0 | 0 | 0 | 0 | 2.3e-11° |
| 0.25° | 0.079° | 0.359° | 34.07° | 0.189 | 3.7e-12° |
| **1.0°** | **0.297°** | **1.353°** | **59.83°** | **0.332** | 8.8e-12° |
| 2.0° | 0.616° | 2.909° | 135.3° | 0.752 | 2.2e-11° |

Plane angle and LVLH ρ are invariant to 1e-15 / 1e-8 m, as claimed. Only Δλ moves — and 0.332 Φ units is **8× the worst adverse step**. So `frame_randomize=1` changes the task, not just the frame: a success-rate move under it is a false positive for "obs leak", and an implementer chasing it will "fix" the wrong thing.

**Fix.** Either (a) restrict `frame_randomize` to a rotation about ĥ_t only (the true gauge freedom once the design pins i_t=Ω_t=0), or (b) adopt the target-plane gauge: express both bodies in `(n̂_t, m̂_t, ĥ_t)` before forming λ. Measured invariant to 2e-11° at every Δi — validated in the probe, ~15 lines of C.

---

## MAJOR 2 — the 38-slot relayout invalidates every validated downstream harness

**Attack.** 3d_C §2 claims "dim stays 38 — `orbital.py`, the .ini, `eval_checkpoint.py` and the 48-dim action-mask path all work unchanged." True of the *shape*, false of the *semantics*. Verified consumers that index by slot:

- `scripts/orbital/t3/redteam/rt_common.py:59-95` `ObsView` — reads `obs[2,3]` (sin/cos θ_s), `obs[9,10]`/`[11,12]` (sin/cos ω_s, ω_t), `obs[15,16]`. This is the Φ replica that "matched the C shaping to 3.6e-9" and made every T3 counterfactual exact.
- `scripts/orbital/nav/orbital_math.py::decode_obs` → used by `t3/expert_controller.py:144,451` — the 99.2%/100-of-100 scripted expert, which the design lists as a *pre-flight feasibility gate*.
- `scripts/orbital/nav/eval_relnav.py:204-226` `build_obs_t3` — hardcodes `out[13],[14],[15],[16]` plus `om.build_obs`'s `[7],[8],[11],[12],[33-37]`. This is the EKF **injector**; it must write estimated state back into obs.

vNext moves or deletes every one of those. This is the T3 red-team's own MAJOR ("`phase_obs_mode=1` silently broke the expert's obs decoder, 12/12 → 0/12 — would have poisoned every imitation arm") at 20 slots instead of 4. It also voids 3d_A §6's free warm-start rung-0 and reduces anchor A3 to "the legacy layout still exists behind a value gate" — i.e. `fill_observations` needs two complete layouts, which is not in 3d_A §8's LOC estimate. And reserving slots 30-37 "so both extensions share one 38-dim shape" buys nothing when the nav layer's encoder has to be rewritten regardless.

**Fix.** Either (a) take 3d_A §3.1's obs-*repurposing* (3D block into the dead slots 21-32, keep 0-16 and 33-37 semantics) and accept the two documented wrinkles (raw ω → ϖ, LVLH ω⃗ from h/r²) — this preserves the whole harness stack and the warm-start; or (b) keep the relayout but add it to the ladder as explicit work: port `ObsView`, `decode_obs`, `build_obs_t3`, re-run `expert_controller --t3` to its 100/100, and re-run the Φ-replica match to 1e-8, *before* X0. Option (a) is strictly cheaper and the project's own history favours it.

---

## MAJOR 3 — `di_max` = 1.0° (X3) and 0.75° (3d_B) rest on a screen the action space cannot execute

**Attack.** 3d_B §1 recommends `di_max = 0.75°` citing "screen cost 0.5–3.7pp" — that is its **combined-burn** column, which assumes one impulse in a combined direction. `ACTION_DV[20][3]` in `orbital.h:78-98` is **entirely single-axis**: every one of the 20 rows has at most one non-zero component. There is no executable combined impulse; two impulses one sub-step apart cost `D + P` in Tsiolkovsky, not `hypot(D,P)`.

**Measurement.** Independent joint screen with `di=0` controls (`ext_rt3d_ladder_plane_attribution.csv`, 20k/cell; absolute levels are conservative vs the T3 measured record, so read the *deltas*):

| rung | di | screen(seq) | screen(comb) | **plane cost vs di=0, sequential** |
|---|---|---|---|---|
| X1 | 0.05° | 100.0% | 100.0% | +0.00pp |
| X2 | 0.25° | 100.0% | 100.0% | +0.00pp |
| **X3** | **1.00°** | 80.3% | 95.4% | **−15.97pp** |
| X3b (3d_B) | 0.75° | 86.0% | 95.9% | **−10.29pp** |
| X4 | 1.00° | 58.3% | 78.3% | **−22.12pp** |
| X5 | 2.00° | 76.0% | 93.8% | **−19.19pp** |

3d_B's own `−1pp` criterion at the L2 band gives **0.40° sequential** vs 1.20° combined. Its recommendation quotes the combined number.

`ext_rt3d_crank_granularity.csv` also shows hypot understates the executable cost by 20.9–29.3% at D=100 m/s.

**Fix — pick one.** (i) Set `di_max = 0.40°` at the LEO headline rung and re-derive the ladder from the sequential column; or (ii) **add combined actions** — e.g. `{+25,0,±25}`, `{−25,0,±25}`, `{+25,0,±10}`, `{−25,0,±10}` → Discrete-30. That recovers 15pp at X3 and 20pp at X4 for ~6 rows of table, and it makes the physics 3d_B identifies as "the single biggest lever in the 3D task" actually reachable. Do not ship 1.0° with a single-axis action set.

---

## MAJOR 4 — X1 and X2 are provable no-op rungs

**Attack.** The success test is instantaneous and simultaneous (`orbital.h:968`: `dist < radius && rel_vel < tol`). For two orbits separated by δ, at argument-of-latitude u from the **relative node**: `|Δr| = 2r sin(δ/2)|sin u|`, `|Δv| = 2v sin(δ/2)|cos u|`. Capturing *at the node* makes the position term exactly zero, so the velocity tolerance alone bounds the free plane error.

**Measurement** (`ext_rt3d_box_vs_ladder.csv`):

| box | alt | plane error that is FREE | frac of disc draws needing **no** plane burn |
|---|---|---|---|
| 30 km / 50 m/s | 550 km | **0.3775°** | di 0.05° → **100%**; 0.25° → **100%**; 0.75° → 25.3%; 1.0° → 14.3% |
| 30 km / 50 m/s | 20200 km | 0.7397° | di 0.75° → **97.3%**; 1.0° → 54.7% |
| 10 km / 10 m/s | 550 km | 0.0755° | di 0.25° → 9.1% |
| 5 km / 1 m/s | 550 km | 0.0075° | di 0.05° → 2.3% |

X1 (0.05°) and X2 (0.25°) are **100% inside the box** — solvable with zero normal burns. Their Φ contribution confirms it: `Δv_pl` p90 = 6.2 and 31.2 m/s, so the entire plane sub-task is worth `0.817×31.2/700 = 0.036` of a 1.817 Φ range (**2.0%**) and 0.36% of the +10 terminal (`ext_rt3d_phi_saturation.csv`). Two of six rungs cost training time and can prove nothing. Note also 3d_C §4.8 has the binding constraint backwards — it calls position "not binding"; at the node position is the term that vanishes.

**Fix.** Collapse X1+X2 into one rung at `di_max = 0.40°` (just past the 0.3775° free zone) — or keep X1 explicitly labelled as an actions-exist smoke test with a **0/200 normal-burn-usage** expectation, not a gate. And note that 3d_B's "constant `di_max = 0.75` from the headline rung upward" is a 97.3% no-op at MEO.

---

## MAJOR 5 — the Δē over-credit is `e/2`, not 2.5%

**Attack.** 3d_C §3.1 reports the 3-vector `‖Δē‖` over-credit as "2.5% at e=0.05, Δi=1°… bounded, monotone, no action needed." It was measured at one eccentricity. The ladder goes to e ≤ 0.30 (X4) and e ≤ 0.50 (X5).

**Measurement** (`ext_rt3d_phi_overcredit.csv`, 8k/cell, in-plane geometry held *identical* so the entire residual is the plane tilt):

| e | phantom in-plane cost, p50 / max, as a fraction of the plane leg | phantom Δv p90 @ di=2° |
|---|---|---|
| 0.05 | 1.77% / 2.50% | 5.3 m/s |
| 0.15 | 5.31% / 7.50% | 15.9 |
| 0.30 | 10.63% / **15.00%** | 31.8 |
| **0.50** | **17.71% / 25.00%** | **53.1 m/s** |

Exactly `e/2` at the maximum, independent of Δi. Under exact telescoping it cannot be *farmed*, but it breaks the design's own headline calibration ("one potential unit per m/s of Δv-to-go") by up to 25% and biases the policy toward the normal axis at the high-e rungs.

**Fix.** Either compute `Δē` after rotating ē_s into the target plane about the relative node (the residual in-plane error — 6 lines), or state the bound as `e/2` in the design and gate on it. Do not carry "2.5%" forward.

---

## MAJOR 6 — X5's `dv_ref = 900` breaks the design's own ≥5:1 terminal-dominance gate

**Attack.** 3d_C §3 pins `W_m/dv_ref = 1.1667e-3` invariant and §3.1 requires `Φ_range/terminal ≥ 5:1`. At `dv_ref = 900` the weight rule forces `W_m = 1.05`.

**Measurement** (`ext_rt3d_a2_bitexact.csv`, part 2): `Φ_range = 1 + 1.05 = 2.050` → **10/2.050 = 4.88 : 1 — fails**. And the reason for raising `dv_ref` does not exist: measured `min(1,·)` saturation at each rung's real init distribution (`ext_rt3d_phi_saturation.csv`, 20k draws, full `c_reset` sampling incl. `da_max`/`de_max` discs and the perigee filter):

| rung | dv_ref | Δv₃ p90 | frac saturated (L1) | design gate <5% |
|---|---|---|---|---|
| X3 LEO di1.0 | 700 | 348 | 0.000% | pass |
| X4 wide di1.0 | 700 | 352 | 0.235% | pass |
| X5 MEO di2.0 | 900 | 445 | 0.510% | pass |
| **X5 MEO di2.0** | **700** | 445 | **0.945%** | **pass** |

**Fix.** Keep `dv_ref = 700, W_m = 0.8166667` at every rung including X5. Both gates pass simultaneously. Separately: `W_m = 0.817` is a rounding of `0.35·700/300 = 0.8166667` — a 4.1e-4 relative violation of the invariant and a gratuitous extra A2 break. Use the exact value.

---

## MINOR

**m1 — the coast-to-node macro is unnecessary and adds risk; DEFER.** `ext_rt3d_crank_granularity.csv` (greedy crank with timing quantization, 400 trials/cell), realized Δv / ideal `2v sin(Δi/2)`, p90:

| alt | τ=1 | **τ=5 (warp-5min, action 9)** | τ=30 | τ=60 |
|---|---|---|---|---|
| 550 km, Δi=1.0° | 0.98× | **1.00×** | 1.24× | 1.56× |
| 550 km, Δi=0.25° | 1.00× | **1.03×** | 1.60× | 2.23× |
| 8000 km, Δi=1.0° | 0.99× | 0.99× | 1.01× | 1.09× |
| 20200 km, Δi=2.0° | 1.00× | 1.00× | 1.00× | 1.00× |

The τ≤5 actions already exist and are near-optimal; the barrier is real only at τ≥30. (My greedy re-targeting agrees with 3d_C's own `eff 0.986` at τ=5 = +1.4%; 3d_B's "+15% at 5-min granularity" is the outlier of the three.) Either way that is 2–20 m/s on a 478 m/s budget. Against that, the macro costs (`ext_rt3d_macro_economics.csv`): at X5 the one-period clamp is **τ = 718 sub-steps vs the warp set's max of 360** — a new discount extreme (γ^N terminal visibility 0.920 vs 0.846 best existing warp), confounding "3D plane skill" with "a longer warp"; and the Δi→0 clamp to τ=1 is a **48× (LEO) / 359× (MEO)** discontinuity in sim-seconds-per-decision, sited exactly in the endgame. **Defer action 26.** If longer warps are wanted at MEO, add a plain warp-12hr so the ablation stays separable; if the macro is ever shipped, clamp to τ=5, not 1.

**m2 — the specced KS test cannot detect the inert phase-gap knob.** `ext_rt3d_phasegap.csv`: unpatched (`tgt_M += ω_s − ω_t`) under 3D sampling gives realized-gap error p50 = **90.4°**, max 180° — fully inert, the ANOM-4 re-run. But **KS vs uniform is 0.0083 unpatched and 0.0079 patched**: the realized distribution stays uniform, merely decorrelated from the request. 3d_C §4.4(b)'s "realized-gap error 0.000° **and** KS ≤ 0.06" only works because of the first clause; the KS clause is decorative. Gate on per-draw `|realized − requested|` max, never on KS alone. The `ϖ` patch itself is exact: 0.00000° p50 and max at every Δi.

**m3 — obs 16/19 exceed the declared `Box(-2,2)`.** `ext_rt3d_obs_saturation.csv`: slot 16 max 2.35 (X4) / 2.52 (X5 @900) / 3.24 (X5 @700), clip 0.02–0.14%; slot 19 min −2.78. 3d_C §2's declared ranges `[0,~0.7]` and `~[−1,1]` are wrong at the wide rungs. Either widen the Box, clamp in `fill_observations`, or scale slots 16-19 by `478/dv_ref` rather than `1/dv_ref`.

**m4 — the joint `min(1,·)` kills both gradients at once.** Saturating on `Δv_in + Δv_pl` means 0.24% (X4) / 0.95% (X5) of episodes start with *both* axes gradient-dead. Two separate saturations would keep one alive. Small, and `shape_match_squash=1` already exists as the escape hatch — but the default-0 joint form is strictly the worse of the two.

**m5 — signed-zero trap in `raan = atan2(hx, −hy)`.** My probe hit it accidentally: an exactly-equatorial ĥ = (0.0, −0.0, 1.0) passed through a rotation becomes (0.0, +0.0, 1.0), and `atan2(0.0, −0.0) = π` vs `atan2(0.0, +0.0) = 0`. 3d_A §2.2 guards it with `hxy == 0.0` and §4 guards with `si > 1e-12`; 3d_C's rotation sampler as written does not. Guard it there too.

**m6 — "worst adverse step = W_m·25/dv_ref exactly" is arithmetically wrong.** A 25 m/s tangential impulse moves `δa/a` *and* `|Δē|` each by `2Δv/v`, so `Δv_match` moves by `√2·Δv = 35.36`, giving `0.35·35.36/300 = 0.0413` — which is the quoted 0.0408, but the stated formula gives 0.0292. Under mode 2 the same √2 applies (weight rule preserved): worst adverse step **0.0413 + ~0.0009 λ-jump = 0.042, margin 238×**. Number fine, formula wrong; fix before it propagates into a gate.

**m7 — X0 with a Discrete-26 head is not a bit-exact anchor.** One 1 m/s normal action sets `inc = 1.318e-4 rad` (0.00755°) permanently, and the `inc==0.0 && raan==0.0` value gate never fires again that episode. X0 as specced ("fresh nets, obs vNext") *will* take exploratory normal burns. Keep A1/A2/A3 as scripted tests with actions 20-25 masked (`legacy_action_space=20`); X0 is a behavioural reproduction, label it as such.

---

## NON-ISSUES (attacked; the design holds)

**n1 — L1 is right, but for the opposite reason both reports give.** 3d_C justifies L1 as "a bonus for the physically-optimal combined maneuver, unfarmable by telescoping"; 3d_B calls the sum "over-penalizes plane error by up to 41% and re-creates exactly the failure mode that sank T1/T3." **Both are wrong: `ACTION_DV` has no combined-direction action**, so the 1.41× bonus is never realizable and the quadrature saving is not achievable. The executable cost-to-go *is* the L1 sum; `hypot` would understate it by 20.9–29.3% (`ext_rt3d_crank_granularity.csv`, A3 block). Keep L1 — and record this reason, so nobody "corrects" it back to hypot per 3d_B's §1 directive. (If MAJOR 3's option (ii) is taken and combined actions are added, this flips and must be re-derived.)

**n2 — the potential is unfarmable under every adversary I could construct.** `ext_rt3d_phi_adversary.csv`. Normal-sign dithering (+q, −q one sub-step apart, 3.90° of along-track motion at LEO): net ΔΦ per cycle **−7.96e-5** (q=1), **−8.09e-4** (q=10), **−2.08e-3** (q=25) — strictly negative, a 34:1 loss of fuel to potential. The γ<1 leak on a `shape_gamma=1` (non-PBRS) delta: Abel summation gives `Σγᵏ(Φ_{k+1}−Φ_k) = γ^{N−1}Φ_N − Φ_0 + (1/γ−1)Σγᵏ Φ_k`; since Φ ≤ 0 by construction the correction is **≤ 0** — a penalty for dawdling at low Φ, never an oscillation income. Max discounted shaping return ≤ `−Φ_0` = 1.817, dominance 5.50:1.

**n3 — the saturation worry is empirically unfounded.** < 1% at every rung (table in MAJOR 6). 3d_C's "LEO Δi≤0.25° at dv_ref 300 → 1.22, saturates" does not reproduce on the real `c_reset` init distribution (X2 Δv₃ p90 = 31 m/s).

**n4 — the 2D closure argument (3d_A §5.1 step 5) holds bitwise.** In-plane burn at i=0 with `z = vz = 0.0`: `hx = y·0.0 − 0.0·vy` and `hy = 0.0·vx − x·0.0` are each a difference of exact zeros → **0 non-zero events in 200,000 draws** across all prograde/retrograde/radial quanta (`ext_rt3d_numerics.csv` F1). The equatorial branch fires deterministically.

**n5 — 3d_E F1 absorbed.** `acos(h_z/|h|)` error vs `atan2(hypot(h_x,h_y), h_z)`: 1.0e-9 vs 4.1e-25 at i=1e-9 (2.4e15× worse); 1.25e-14 vs 5.2e-18 at i=1e-2. 3d_A §2.2 already uses atan2. Correct.

**n6 — 3d_E F2 is a seam phenomenon no rung occupies.** Per-burn `coe2rv∘rv2coe` residual p99 = 8.6–9.7e-9 m across X0/X3/X4/X5 geometries; over a 400-burn episode as a random walk that is **1.9e-7 m = 4e-11 of the 5 km box**. The 2e-11 *relative* floor sits at `e ≈ 1.7e-11` / `sin i ≈ 1e-11`, and the ladder is never there (X0/X1 have e = 0.0 exactly → the `e < 1e-10` branch; X3+ have e ≳ 1e-3; one 1 m/s normal burn puts `sin i` at 1.3e-4). Hold Cartesian state if you like, but it is not load-bearing here.

**n7 — action masking.** `obs[38+a]`, a ∈ 0..9, `obs_dim = 48`. Extending to Discrete-26 without extending the mask leaves actions 20-25 unmasked — same status as 10-19 today, and defensible (a normal burn raises |v| and cannot drop perigee below the floor). The claim "the 48-dim action-mask path works unchanged" is true only in that sense; say so.

---

## Verdict: **SHIP WITH CHANGES**

Required before any implementation starts:

1. Replace the `di_max` sampler with the exact rotation form; gate on max/knob = 1.000 at i_t ∈ {0, 51.6, 90, 98}°. **[BLOCKER 1]**
2. Spell `Δv_in` as `0.5*v_t*sqrt(da_rel*da_rel + de*de)` with element-derived ē; A2 must be re-stated as bit-exact under mode 1's own weights, and verified at 0/200000. **[BLOCKER 2]**
3. Fix or scope `frame_randomize` to the ĥ_t gauge (or move Φ to the target-plane gauge). **[MAJOR 1]**
4. Choose: 3d_A obs-repurposing (preferred), or budget the decoder/encoder port + expert re-validation into the ladder before X0. **[MAJOR 2]**
5. `di_max = 0.40°` at the LEO headline rung, **or** add combined-direction actions and re-derive. **[MAJOR 3]**
6. Collapse X1/X2; no rung gates below 0.3775° at the 30 km/50 m/s box. **[MAJOR 4]**
7. State the Δē over-credit as `e/2`; fix it if X4/X5 are in scope. **[MAJOR 5]**
8. `dv_ref = 700, W_m = 0.8166667` at every rung. **[MAJOR 6]**
9. Defer action 26. **[m1]**
10. Gate the phase-gap knob on per-draw realized error, not KS. **[m2]**

The Φ mode-2 core (L1 combiner, coefficient 1.0 on the plane term, equinoctial λ, `shape_gamma = 1`, clamp-nowhere) survives every adversarial test I could build and should ship as designed. The failures are all upstream of it.