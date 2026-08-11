# 3D-C — Observation & shaping design for the ext-3d extension

**Scripts (new, read-only recon):** `/Users/pete/space_training/scripts/orbital/ext_recon/ext_3dc_shaping_math.py`, `/Users/pete/space_training/scripts/orbital/ext_recon/ext_3dc_knob_warp_j2.py`
**CSVs:** `web_data/results/ext_3d_{levers,longitude_continuity,node_crank,maneuver_ledger,plane_affordability,box_plane_tol,saturation,inert_knob,warp_node,j2_raan}.csv`
Both scripts carry their own independent 3D Kepler element↔Cartesian machinery (no import from `orbital.h` or `orbital_math.py`), so every number below is an independent cross-check, per the T3 f&g-oracle discipline. Nothing in the repo was modified.

---

## 0. Bottom line

1. **Coordinates.** Use *equinoctial mean longitude* `λ = M + ω + Ω` for phase and the *relative-inclination vector* `δı⃗ = Δi_rel·n̂`, `n̂ = (ĥ_t×ĥ_s)/|·|` for plane. The textbook ROE relative longitude `δλ = δu + δΩ·cos i_t` **is a burn-teleport coordinate in 3D** — measured jump **43.6° / 136.4° / 46.4°** (at i_t = 28.5/51.6/97.4°) and *independent of burn size* (identical at 1, 10 and 25 m/s). That is the exact signature of the 2D ANOM-2 defect that cost this project five months. Equinoctial λ jumps **linearly in Δv**, ≤ 1.15·(Δv_n/v) rad — i.e. it is burn-*continuous*, not teleporting (0.216° worst case at 25 m/s, i=97.4°; ≤1.2e-3 in potential units).
2. **Coefficients.** Measured Gauss levers (§A): tangential → `∂(δa/a)/∂Δv = 2/v` and `∂|Δē|/∂Δv = 2/v`; normal → `∂Δi/∂Δv = 1/v`. **The plane term does not halve.** The task's candidate `0.5·v·hypot(Δa/a, |Δē|, 2·|Δī|)` has the *right* coefficient (0.5 × 2 = 1.0·v). The design rule that produces it: **the coefficient on each element error is the inverse of that element's best Gauss lever, so Φ measures remaining Δv in true m/s and one m/s of correctly-aimed burn buys exactly one unit of Δv-to-go.**
3. **Combiner.** Reject `hypot`-across-the-plane-term; use **L1 sum inside one squash**. `hypot3` attenuates the in-plane gradient by `x/‖·‖` whenever the plane error is large (5× at Δi = 2°, δa = 100 km) — that is a *soft gate*, defect class #1 in continuous clothing. L1 keeps ∂Φ/∂Δv = W_m/dv_ref exactly on both axes and gives combined (tangential+normal) burns up to a 1.41× potential bonus, which is the physically-optimal maneuver. Telescoping (γ_shape = 1) makes the bonus unfarmable.
4. **Verified monotone.** Scripted 5-leg 3D maneuver (180° gap, Δi = 1.0°, e = 0.02, i_t = 51.6°, 355 m/s = 74% of budget): total ΔΦ **+1.0736**, telescoping exact to 1e-16, **drift leg +1.0000**, **plane leg +0.1504**, **worst adverse step −0.0408** (245× margin vs the +10 terminal), coast-to-node leg **exactly 0.0000**.
5. **The binding constraint is Δv, not learning.** At LEO a 1° plane change costs **133 m/s = 28% of the whole 478 m/s budget**; the budget affords only **3.5° of pure plane change**. `di_max` must be a first-class, disc-sampled kwarg (the `de_max` pattern), and 3D LEO rungs live at **Δi ≤ 1°**.

---

## 1. Coordinate audit (why each choice; evidence)

| quantity | rejected | adopted | evidence |
|---|---|---|---|
| phase | `θ_s − θ_t` (2D legacy) | — | ANOM-1/2, T3 §4.4 |
| phase | ROE `δλ = δu + δΩ cos i_t` | **equinoctial `Δλ = (M+ω+Ω)_s − (M+ω+Ω)_t`** | normal-burn jump **43.6–136.4°, size-independent** vs equinoctial **0.0019–0.216°, exactly ∝ Δv** (`ext_3d_longitude_continuity.csv`) |
| plane | `Δi = i_s − i_t`, `ΔΩ` | **`δı⃗ = Δi_rel·n̂`, `Δi_rel = atan2(\|ĥ_t×ĥ_s\|, ĥ_t·ĥ_s)`, `n̂ = (ĥ_t×ĥ_s)/\|·\|`** | `i`,`Ω` individually singular at i→0; `δı⃗ → 0⃗` continuously, no undefined angle |
| ecc | `(e cos ω, e sin ω)` per-body | **inertial e-vectors `ē = (v×h)/μ − r̂`, `Δē = ē_s − ē_t` (3-vector)** | reduces *bit-exactly* to the 2D formula when coplanar (regression anchor A2) |
| in-plane phase ref | `θ_s`, `ω_s` (obs[2,3,9,10]) | **drop** — replaced by relative vectors projected into chaser RTN | both singular at e=0, both rotation-variant, both teleport on burns; env is SO(3)-invariant so they carry zero task information |

**Why `ĥ` is the right plane handle:** `ĥ` is *exactly invariant* under both existing action axes. Prograde: `Δh = r×(Δv v̂) ∥ h`. Radial: `Δh = r×(Δv r̂) = 0`. So the plane channels are untouched by every legacy action and move only, and proportionally, under a normal burn. `δı⃗ ⟂ ĥ_s` exactly, so its `N̂_s` component is identically zero and **two floats are lossless** (for Δi < 90°, far outside any affordable envelope — guard and assert).

**Why `Δī` projected into the *chaser's* RTN and not the target's:** `atan2(δı⃗·T̂_s, δı⃗·R̂_s)` is exactly *how far the chaser must coast along its own orbit to reach the relative node* — the burn-timing signal, delivered without any "are you at the node" gate. `δı⃗·T̂_s > 0` ⇒ node ahead.

**Latent 2D bug found in passing (fix in vNext):** `fill_observations` builds the LVLH frame-rotation term with `n_tgt = sqrt(μ/a³)` instead of the instantaneous `ḟ_t = h_t/r_t²`. At e = 0.3 (WL4's realized envelope) these differ by up to **1.96×** at perigee, so obs[35,36] are a nonstandard (though deterministic) frame. Use `ω⃗_LVLH = (h_t/r_t²)·N̂_t` in vNext. Breaking obs change — 3D is a fresh lineage anyway.

---

## 2. Observation layout vNext-3D — exact 38-slot map

Frames: `R̂_s = r̂_s`, `N̂_s = ĥ_s`, `T̂_s = N̂_s × R̂_s` (chaser RTN); target RTN likewise. Scales: `S_a = obs_alt_scale_m`, `S_L = lvlh_scale_m`, `v_c(a) = √(μ/a)`, `dv_ref = shape_dv_ref_ms`, new kwargs `obs_da_scale` (default 0.10), `obs_de_scale` (default `max(de_max, 0.05)`), `obs_di_scale_rad` (default `max(di_max, radians(0.25))`).

| # | field | formula | scale | burn-continuity |
|---|---|---|---|---|
| **A. chaser absolute (5)** | | | | |
| 0 | chaser size | `(a_s − R_E)/S_a` | ~[0,1] | ∝Δv (vis-viva) |
| 1 | chaser shape | `e_s` | [0,1) | ∝Δv (‖ē‖, no angle) |
| 2 | radial rate | `v_r,s / v_c(a_s)` | ~[−e,e] | position fixed at burn; v jumps by exactly Δv |
| 3 | speed | `v_t,s / v_c(a_s)` | ~[0.5,1.5] | same |
| 4 | fuel | `m_f/(m_d+m_f)` | [0,1] | monotone |
| **B. target absolute (2)** | | | | |
| 5 | target size | `(a_t − R_E)/S_a` | ~[0,1] | constant |
| 6 | target shape | `e_t` | [0,1) | constant |
| **C. relative elements, chaser-RTN (9)** | | | | |
| 7 | `δa` | `((a_s−a_t)/a_t)/obs_da_scale` | ~[−1,1] | ∝Δv |
| 8 | `sin Δλ` | `Δλ = wrap(λ_s−λ_t)`, `λ = M+ω+Ω` | [−1,1] | jump ≤1.15·Δv_n/v, **linear in Δv** (§B) |
| 9 | `cos Δλ` | ″ | [−1,1] | ″ |
| 10 | `\|Δλ\|/π` | ″ | [0,1] | ″ — this is the exact Φ argument |
| 11 | `Δē·R̂_s / obs_de_scale` | `Δē = ē_s − ē_t` (inertial 3-vec) | ~[−1,1] | `ē` is a smooth function of (r,v) |
| 12 | `Δē·T̂_s / obs_de_scale` | ″ | ~[−1,1] | ″ |
| 13 | `Δē·N̂_s / obs_de_scale` | ″ (magnitude `O(e·Δi)`) | ~[−0.1,0.1] | ″ |
| 14 | `δı⃗·R̂_s / obs_di_scale` | `δı⃗ = Δi_rel·n̂` | ~[−1,1] | 0 under in-plane burns (exact); ∝Δv_n |
| 15 | `δı⃗·T̂_s / obs_di_scale` | ″ | ~[−1,1] | ″ |
| **D. Δv ledger (4)** | | | | |
| 16 | in-plane cost-to-go | `0.5·v_t·hypot(δa, ‖Δē‖) / dv_ref` | [0,~0.7] | Lipschitz in Δv |
| 17 | plane cost-to-go | `v_t·‖ĥ_s−ĥ_t‖ / dv_ref` | [0,~0.7] | ″ |
| 18 | Δv remaining | `V_e·ln((m_d+m_f)/m_d) / dv_ref` | [0,~0.7] | monotone |
| 19 | margin | `(obs18 − obs16 − obs17)` | ~[−1,1] | ″ |
| **E. clock (2)** | | | | |
| 20 | time remaining | `(cap_steps − step)/cap_steps` | [0,1] | n/a — red-team #11 requirement, kept |
| 21 | timescale | `n_t / 1e-3` | ~[0.1,1.2] | constant |
| **F. LVLH 6-D, target RTN (6)** | | | | |
| 22–24 | `ρ_R, ρ_T, ρ_N` | `Rᵀ(r_s − r_t) / S_L` | ~[−1,1] | position continuous |
| 25–27 | `ρ̇_R, ρ̇_T, ρ̇_N` | `Rᵀ(v_s−v_t) − ω⃗_L×ρ⃗`, `ω⃗_L = (h_t/r_t²)N̂_t`, `/v_c(a_t)` | ~[−1,1] | v jumps by exactly Δv |
| **G. J2 / absolute orientation (2)** | | | | |
| 28 | `cos i_s` | — | [−1,1] | 0 unless `j2_mode=1` |
| 29 | `cos i_t` | — | [−1,1] | 0 unless `j2_mode=1` |
| **H. reserved (8)** | | | | |
| 30–37 | `0.0f` | — | — | held for ext-nav estimator channels (residual, √tr P, time-since-update, sensor-valid) so both extensions share one 38-dim shape |

Notes: (i) dim stays **38** — `orbital.py`, the .ini, `eval_checkpoint.py` and the 48-dim action-mask path all work unchanged. (ii) Under two-body the env is SO(3)-invariant, so **G is exactly zero-information** and is written only under `j2_mode=1`, where the equator becomes physical (J2 is axisymmetric, so `Ω` remains gauge and only `cos i` is needed). (iii) Resolution check: the 5 km/1 m/s tight box demands `Δi_rel < 0.0075°`; at `obs_di_scale = 0.25°` that reads 0.030 — comfortably resolved in float32, and the *fine* out-of-plane endgame signal is carried by slots 24/27 anyway (global vs local division of labour, mirroring Δλ vs `ρ_T`).

---

## 3. Shaping — `shaping_mode = 2`

```
λ      = M + ω + Ω                                  # equinoctial mean longitude
Δλ     = wrap_pi(λ_s − λ_t)
v_t    = sqrt(MU / a_t)                             # per-episode CONSTANT (as in mode 1)
δa     = (a_s − a_t)/a_t
Δē     = ‖ē_s − ē_t‖                                # inertial e-vectors, 3-D
s_i    = ‖ĥ_s − ĥ_t‖ = 2 sin(Δi_rel/2)              # chord — EXACT single-impulse geometry

Δv_in  = 0.5 · v_t · hypot(δa, Δē)                  # unchanged from mode 1
Δv_pl  = 1.0 · v_t · s_i                            # coefficient 1.0, NOT 0.5
Δv3    = Δv_in + Δv_pl                              # L1, NOT hypot-3

Φ = −[ W_λ·|Δλ|/π  +  W_m·min(1, Δv3/dv_ref) ]      Φ ∈ [−(W_λ+W_m), 0]

W_λ = 1.0    W_m = 0.817    dv_ref = 700 m/s        (LEO 3D rung)
```

**Coefficient derivation (first principles, confirmed numerically in `ext_3d_levers.csv`).** Gauss variational equations near-circular: a tangential impulse gives `δ(a/a) = 2Δv/v` and `δē = (2Δv/v)(cos u, sin u)`; a radial impulse gives `δē = (Δv/v)(sin u, −cos u)` (lever 1/v — worse, so ignored); a normal impulse gives `δı⃗ = (Δv/v)(cos u, sin u)` (lever 1/v, the *only* lever). Measured: tangential 2.0007 / 2.0001, radial 0.0001 / 1.0000, normal 1.0000 on (δa/a, |Δē|, Δi) respectively, all ×v/Δv. **Coefficient = 1/(best lever)** ⇒ 0.5v, 0.5v, 1.0v. Equivalently, in the task's `0.5·v·hypot(…, 2·|Δī|)` form the factor 2 is exactly `lever_tangential/lever_normal`, so the candidate's coefficient is correct. Independently, the exact optimal single-impulse plane rotation is `2v sin(Δi/2) = v·‖ĥ_s−ĥ_t‖` and splitting into k impulses costs `2kv sin(Δi/2k) > 2v sin(Δi/2)` — single impulse at the node is optimal, and the chord form is exact rather than small-angle.

**Why the plane term does not halve.** The tangential axis has a *lever of 2* on both `δa` and `δē` — two impulses at opposite `u` can trade one against the other, which is what the shared `0.5·v·hypot` encodes. The normal axis has a lever of 1 on `δı⃗` and no such trade: a second normal impulse at the opposite node undoes the first. Physically: raising `a` and rotating `ē` are energy/shape changes that a tangential burn drives twice per rev; a plane rotation is a rigid rotation of the velocity vector whose cost is the chord `2v sin(Δi/2)`, achieved once.

**Why L1 and not `hypot3`.** `∂hypot(x,y,z)/∂x = x/hypot`. With `z = 2s_i` at Δi = 2° (`z = 0.0698`) and `x = δa` for 100 km (`0.0148`), the in-plane gradient is attenuated **4.8×** while the plane error persists — a continuous re-implementation of defect class #1. L1 keeps `∂Φ/∂Δv_in = ∂Φ/∂Δv_pl = W_m/dv_ref` exactly. The price is that L1 over-counts a *combined* burn (`t + n ≥ √(t²+n²)`, up to 1.41×) — which is a **bonus for the physically-optimal combined maneuver**, and is unfarmable because `shape_gamma = 1` makes the episode total telescope to `Φ_T − Φ_0` regardless of path. Both forms are in the ledger CSV; totals agree to 0.4%, so this is a gradient-shape choice, not a magnitude choice.

**Weight rule.** Hold **`W_m/dv_ref = 1.167e-3 per (m/s)`** invariant (= mode 1's 0.35/300) so the "one potential unit per m/s of Δv-to-go" calibration is preserved across rungs; move `dv_ref` *only* to control `min(1,·)` saturation. Saturation screen (`ext_3d_saturation.csv`): LEO Δi≤0.25° at dv_ref 300 → **1.22, saturates**; at dv_ref 700 → 0.67 ok; wide Δi≤1° → 0.59 ok; MEO Δi≤2° at dv_ref 900 → 0.48 ok. Report `frac(Δv3 > dv_ref)` at init for every rung; require < 5%.

### 3.1 Monotonicity — analytic + measured ledger (`ext_3d_maneuver_ledger.csv`)

Scenario: `a_t = 6871 km`, `e = 0.02`, `i_t = 51.6°`, `Ω_t = 40°`, `Δλ_0 = 180°`, `Δi_rel,0 = 1.0°`. Reference maneuver = coast-to-node → plane-crank (25 m/s quanta) → drift-open (δa = −200 km) → drift 17.9 h → drift-close.

| leg | ΔΦ | Δv | note |
|---|---|---|---|
| L1a coast to relative node | **−0.0000** | 0 | plane term *exactly* constant on a coast |
| L1b plane crank ×6 | **+0.1504** | 133 m/s | +0.029 per 25 m/s quantum, flat rate |
| L2 drift-open ×5 | **−0.1789** | 111 m/s | the entry fee, paid in 5 steps of −0.041 |
| L3 drift ×6 slices | **+1.0000** | 0 | +0.1667 per 30° of Δλ, perfectly linear |
| L4 drift-close ×5 | **+0.1022** | 111 m/s | |
| **total** | **+1.0736** | 355 m/s (74% budget) | telescoping check `Φ_T−Φ_0` matches to 1e-16 |

- **worst adverse step −0.0408** = `W_m·25/dv_ref` exactly ⇒ **245× margin** vs the +10 terminal, 3.8% of the episode's own shaping return (mode-1 2D: −0.0408, 24×).
- **drift-reward : drift-entry-fee = 5.6 : 1 in favour** at δa = 200 km (mode-1 2D reported 8.8:1 at δa = 133 km; the ratio scales as 1/δa and is a *choice*, not a defect).
- **Q: does holding a drift orbit while phasing keep the plane term constant?** **Yes, exactly.** `ĥ` is a constant of Keplerian motion and `v_t = √(μ/a_t)` is a per-episode constant, so `Δv_pl` is invariant under coasting/warping for both bodies. Measured: 0.0383° for all 6 drift slices, ΔΦ contribution exactly 0. **Zero false gradient.** (This fails under J2 — see §4.6.)
- **Q: does the plane term reward cranking at the node?** **Yes, with credit/Δv = |cos ψ| exactly** (`ext_3d_node_crank.csv`, Δi = 1°, 10 m/s burns): ψ = 0° → +1.000, 15° → 0.963, 30° → 0.856, 45° → 0.687, 60° → 0.471, **90° → −0.038** (a small *penalty*), 180° → +1.000 with the opposite sign. Off-node burns earn a bounded second-order penalty `|δ|²sin²ψ/(2Δi)` capped by the norm's Lipschitz constant, so it can never exceed `W_m·Δv/dv_ref` — no cliff, no gate, and the "burn at the node" policy falls out of the geometry with no explicit node detector.
- **Quantization corner:** at Δi = 0.05° a 10 m/s normal burn *overshoots* (10 m/s ≡ 0.075° of rotation) and credit/Δv falls to 0.33 even at ψ = 0. The endgame therefore needs **normal quanta ≤ 1 m/s** (1 m/s ≡ 0.0075°, which is exactly the 5 km/1 m/s box tolerance — see §4.5).
- **Side effect of a normal burn on the in-plane terms:** `|v'|² = v² + Δv_n²` ⇒ `δa/a = (Δv_n/v)²` and `δe ≈ (Δv_n/v)²`. At 133 m/s that is 2.1 km / 3.0e-4 ⇒ 1.65 m/s of new in-plane cost against 133 m/s of plane cost removed: a **1.2% second-order tax**, visible in the ledger as the 5th crank's +0.0280 vs the first's +0.0293.
- **Small over-reward, documented:** `‖Δē‖` (3-vector) contains an `O(e·Δi)` out-of-plane component that a plane burn also removes. At e = 0.05, Δi = 1° that is 3.3 m/s of the 133 m/s leg = **2.5% over-credit**. Bounded, monotone, no action needed beyond stating it.
- **Terminal dominance:** `Φ` range `W_λ+W_m = 1.817` vs the +10 terminal = **5.5:1** (mode 1: 7.4:1). Gate criterion: report `Φ_range / terminal` each rung, require ≥ 5:1; the pre-registered fallback is `W_m = 0.35 @ dv_ref = 700` (range 1.35, 7.4:1) at the cost of halving the match rate.

---

## 4. Self-audit against the five T3 reward-defect classes

**4.1 Gate class — two live risks, both mitigated in the proposal.**
(a) *`min(1, ·)` is a gate.* Above `dv_ref` the entire match term has zero gradient — the 2D audit already measured 80.7% saturation at L3 inits. In 3D a single degree of Δi contributes 133 m/s, so saturation is the default unless `dv_ref` is re-sized. **Mitigation:** the §3 saturation screen per rung + a flagged `shape_match_squash = 1` alternative `g(x) = x/(1+x)` (bounded, strictly monotone, no dead zone, no corner). Default 0 so the mode-1 anchor stays bit-exact.
(b) *`hypot3` is a soft gate* (4.8× in-plane attenuation at Δi = 2°). **Rejected in favour of L1.** This is the single most important 3D-specific finding of this audit.
(c) No sigmoid gates anywhere; `|Δλ|/π` has constant gradient everywhere.

**4.2 γ^τ income class — clean, but check the 3D adversary.** `shape_gamma = 1.0` (already the T3 default) ⇒ exact telescoping ⇒ do-nothing return is exactly `Φ_T − Φ_0`, and Φ is bounded in `[−1.817, 0]`. Novel 3D adversary: "crank the plane, never rendezvous" harvests at most `W_m·Δv_pl/dv_ref = 0.156` at Δi = 1° — but it must *spend 133 m/s of real fuel* to collect it, and the credit exactly equals the Δv spent, so there is no free lunch and no back-and-forth farming (telescoping). Park-far-and-warp is unchanged from 2D (bounded at ~+0.13). Worst-case episode harvest without success ≈ +1.0 (Δλ ending near 0) vs success +5…+10 ⇒ ≥5:1 dominance preserved. **Do NOT re-enable the NHR clamp** — "clamp-nowhere" must carry forward, otherwise a normal burn that increases Δi and then dies pays a refund.

**4.3 Wrong-coordinate class — the biggest 3D risk; four candidates audited, three rejected.** ROE `δλ` (teleports 43.6–136.4°, size-independent — quantified above), `i`/`Ω` separately (singular at i→0), `ω`/`θ` (singular at e→0, teleport on every burn), true longitude `θ+ω+Ω` (carries the ±2e equation-of-centre ripple, same argument as 2D §3.4). Adopted: equinoctial `λ`, `δı⃗`, inertial `Δē`. **Success criterion is Cartesian (`|Δr|`, `|Δv_rel|`) and therefore coordinate-free — it extends to 3D with no change and remains the ground truth the eval classifier keys on.** New sub-risk: `atan2` on `δı⃗` is undefined at Δi = 0, but `‖δı⃗‖ → 0` so the *vector* obs is continuous; only the derivative has a corner, bounded by the norm's Lipschitz constant. Add an assert for `Δi_rel > 90°` (out of every affordable envelope; `sin` would fold).

**4.4 Inert-knob class — REAPPEARS in two places; both must be fixed before any 3D curriculum.**
(a) *Independent `(i, Ω)` sampling is an inert knob* (`ext_3d_inert_knob.csv`, n = 200k per cell): with `i_s, i_t ~ U(0, i_max)` and `Ω ~ U(0,2π)`, the realized `Δi_rel` has **p50 = 0.70·i_max, p90 = 1.24·i_max, max = 2.0·i_max, and 22.2% of draws exceed the knob** — at *every* value of `i_max` (0.25°/1°/5°/30°), i.e. the knob controls neither the bound nor the shape. **Fix:** sample the *relative* i-vector directly — `ĥ_s = R(δ, n̂)·ĥ_t` with `δ = di_max·√U` (area-uniform disc) and `n̂` uniform in the target plane. Measured: p90 = 0.95·knob, **max = knob exactly, 0.0% over**. This is literally the `de_max` disc pattern applied to `δı⃗`; call it `di_max_rad`.
(b) *`phase_gap_mode = 1` becomes inert in 3D as written.* It sets `tgt_M += ω_s − ω_t`. In 3D it must become `tgt_M += (ω_s + Ω_s) − (ω_t + Ω_t)` to control the *equinoctial* Δλ; left unpatched, the knob is inert whenever `Ω_s ≠ Ω_t` — an exact re-run of ANOM-4. Verify with the red-team's own test: realized-gap error 0.000° and KS ≤ 0.06 vs the intended distribution.
(c) *Absolute `i_t`, `Ω_t` are pure gauge under two-body.* Sampling them adds zero task diversity. **Set `i_t = Ω_t = 0` and put all relative inclination in the chaser**, then add a `frame_randomize=1` *audit* flag that applies a random global SO(3) rotation to both bodies. A policy whose success rate moves under `frame_randomize` has a rotation-variant obs leak — a one-command invariance regression test.

**4.5 Warp-barrier class — a new, sharp 3D instance.** Efficient plane change requires being near a node; a warp steps *past* it. Worst-case node overshoot at LEO (T = 94.5 min, `ext_3d_warp_node.csv`): coast τ=1 → ψ = 3.8° (eff 0.999), **warp-5min τ=5 → 19.1° (0.986)**, warp-30min → 114° (0.542), **warp-1hr/3hr/6hr → ≥229° (0.000)**. So at LEO the plane leg is only viable at τ ≤ 5, while the phasing leg *needs* the long warps for terminal-reward visibility under γ = 0.995 — the two sub-tasks demand opposite granularity. At MEO the conflict evaporates (warp-1hr → ψ = 30°, eff 0.966). **Recommended fix:** add a **"coast to next relative-node crossing"** macro-action whose τ the env computes from `δı⃗` and clamps to one orbital period (and to 1 when `Δi_rel` is below the box tolerance). This is exactly how real GNC schedules a maneuver, it removes the barrier without touching γ, and it is a clean ablation. Fallback if it is not implemented: run 3D LEO with the τ ≤ 5 warp set only and accept the smaller credit window, or start the 3D ladder at MEO. Also carry forward `cap_terminal_reward = 0` — the red-team #1 blocker is unchanged and would bite harder here.

**4.6 Beyond the five: J2 — evidence says DEFER.** (`ext_3d_j2_raan.csv`.) Secular J2 would keep the propagator analytic (constant rates on Ω, ω, M ⇒ warps stay exact, which is load-bearing) and would make ΔΩ correctable by drifting — attractive because it turns a plane error into a phasing-like problem. But the economics at LEO do not support it: at i = 51.6°, δa = 200 km gives `Δ(Ω̇) = 0.485°/day`, so **1° of ΔΩ costs 49.5 h of clock and 222 m/s (open+close)** versus **104 m/s and ~0 h for a direct plane change**; J2 only wins above ~2° of ΔΩ, by which point the clock exceeds the 12000-step (200 h) cap. Co-scheduling with the phasing drift is nearly free but yields at most **0.36° of ΔΩ over a 180°-closing drift** (drift rates differ 495:1). Shaping-hazard assessment: under J2 the plane term is **no longer constant during the phasing drift** (H1-recurrence risk), but the perturbation is bounded at **≤ 37 m/s ≡ ≤ 0.043 of potential vs the drift leg's +1.000 = 4.3%**, sign-indefinite — annoying, not a valley. **Verdict: J2 is a fidelity upgrade, not a task-enrichment one; scope it as an optional later rung (`j2_mode=1`) with its own shaping re-audit, and reserve obs 28/29 for it now.** If it is adopted, declare the env's truth to be *mean* elements (no osculating↔mean conversion) so the propagator stays exact and self-consistent.

**4.7 Beyond the five: the budget is the real constraint.** (`ext_3d_plane_affordability.csv`.) Δv per degree: LEO-300 **134.9**, LEO-800 130.1, 2000 km 120.4, 8000 km 91.9, MEO-20200 67.6, GEO 53.7 m/s/deg. Max pure plane change on the 478 m/s budget: **3.5° at LEO**, 7.1° at MEO, 8.9° at GEO; capping the plane leg at 28% of budget gives **1.0° at LEO**, 2.0° at MEO. Bi-elliptic plane change does not help (breaks even only above Δi ≈ 38.9°). **Therefore the 3D task is "relative-inclination correction at Δi ≤ 1° at LEO" — which is the realistic operational regime (co-planar-ish rendezvous, RAAN maintenance), not a limitation to apologise for.** State `di_max` per rung from this table, and run an analytic joint Δv+horizon MC screen (the 2D `joint_feasibility.py` pattern) before each rung, so a low success rate is never confused with an infeasible envelope.

**4.8 Beyond the five: the terminal box sets the plane tolerance.** (`ext_3d_box_plane_tol.csv`.) The out-of-plane *velocity* is the binding constraint, not position: at LEO the required `Δi_rel` is **0.376° for 30 km/50 m/s**, **0.0752° for 10 km/10 m/s**, **0.00752° = 27 arcsec for 5 km/1 m/s**. A 1 m/s normal quantum rotates the plane by exactly 0.0075° — **the 3D tight box lands precisely on the actuation floor**, the same signature as the 2D radial-quantum finding (T3 §8.3: 0.71 m/s floor vs 1.0 m/s tolerance, 95.5%). Predict TB-3D at 5 km/1 m/s will show the same benign cap-timeout tail; require normal ±1 m/s in the action set for any box below 10 km/10 m/s.

---

## 5. Env-surface implications (for the implementation lane)

- **Actions.** `ACTION_DV` already carries a `dv_normal` column (all zeros) and `apply_impulse` already accepts and discards `dv_nor` — the plumbing is one basis vector. Append without renumbering: **20/21 = normal ±1, 22/23 = normal ±10, 24/25 = normal ±25** ⇒ Discrete(26); optional **26 = coast-to-next-relative-node** (§4.5). Keep `legacy_action_space=20` as the default so both T3 anchors stay bit-exact. The normal axis `ĥ = (r×v)/‖r×v‖` is *exactly* orthogonal to both existing axes (`v̂`, `r̂`) at any e, so adding it perturbs no existing action semantics — note that `v̂` and `r̂` themselves are oblique at e>0, a pre-existing quirk that is harmless.
- **New kwargs** (legacy defaults preserved bit-exactly): `dim3_mode` (0/1), `shaping_mode=2`, `di_max_rad` (<0 = off), `obs_da_scale`, `obs_de_scale`, `obs_di_scale_rad`, `shape_match_squash` (0=min, 1=rational), `frame_randomize`, `j2_mode`. Follow the `rendezvous_radius_m` plumbing pattern through `orbital.h` → `binding.c::my_init` → `orbital.py` → `orbital.ini` → `eval_checkpoint.py` (`unpack()` hard-fails on a missing key).
- **Regression anchors (all three must pass before any training run).** **A1 dynamics:** 3D propagator with `i = Ω = 0` reproduces the 2D env trajectories to ≤1 ULP over a full episode (fuzz against the extended f&g oracle, `fuzz_dynamics.py` pattern). **A2 shaping:** `Φ(shaping_mode=2)` at `Δi = 0` equals `Φ(shaping_mode=1)` bit-exactly. **A3 checkpoints:** legacy 26/200 stays bit-exact and the T3 canonical stays 200/200 under default flags.
- **Ladder (gate ≥ 60% multi-seed per rung, Physical-success line, 200 held-out, greedy, seed disclosed).** X0 coplanar anchor (reproduce T3 headline through the 3D code path, obs vNext, fresh nets) → X1 `di_max = 0.05°` (inside the velocity box already — teaches only that normal actions exist) → X2 `0.25°` → X3 `1.0°` at LEO 300–800 km, `dv_ref = 700`, `W_m = 0.817` → X4 wide envelope (300–8000 km, e ≤ 0.30, `di_max` 1.0°) → X5 MEO (`di_max` 2.0°, `dv_ref` 900, cap 12000, where the warp-barrier disappears).
- **Pre-flight measurements to re-run the T3 way, before training:** do-nothing shaping ≈ 0 (`probe_shaping_leak.py`), `frac(Δv3 > dv_ref)` at init < 5%, realized-`Δi_rel` distribution vs `di_max` (KS), realized-Δλ distribution vs `phase_gap` (KS), `frame_randomize` invariance, and a scripted 3D expert (extend `expert_controller.py` with a node-crank leg) to establish the H4-equivalent feasibility floor before blaming the learner.