**J2-A — J2 realism design memo (secular mean-element upgrade)**

Probes (new, read-only, nothing existing modified, nothing committed):
`/Users/pete/space_training/scripts/orbital/ext_recon/n3d/j2a_core.py` (independent element/cartesian machinery + secular-J2 mean propagator + Cowell J2 integrator), `j2a_probe.py` (A–G), `j2a_oracle2.py` (corrected oracle protocol).
CSVs: `web_data/results/j2_{rates,warp_exact,shaping_ledger,donothing,channel,meantruth,nav_modelerr,oracle,oracle_mutations,oracle2,oracle2_mutations,oracle2_lambda}.csv`.
All J2 numbers below are from machinery that shares no code with `orbital.h` or `orbital_math3d.py`.

---

## 0. Bottom line

**GO, narrow scope.** J2 is cheap (≈10 lines in the propagator + 1 kwarg + 2 obs slots + 1 sampler rule), stays exactly closed-form under every warp (verified), and does not break shaping_mode 2 (drift-leg monotonicity preserved 36/36 slices, worst adverse step unchanged at −0.0354).

**But 3d_C §4.6's verdict — "fidelity upgrade, not task-enrichment" — is wrong at LEO, and I can now say exactly why.** §4.6 priced J2 as *an alternative way to buy a plane change* (49.5 h and 222 m/s for 1° of ΔΩ vs 104 m/s direct — correct, and it does lose). It never priced J2 as *a disturbance coupled to the leg the policy already flies*. Differential nodal precession during the phasing drift is **not optional and not free-standing**: the ratio of RAAN accrued to phase closed is a pure geometric constant,

> **dΩ/dλ = −3.5 · J2 · (R_E/p)² · cos i — independent of the drift altitude offset δa.**

At LEO-500 / i=51.6° that is **0.1162°/rad**, i.e. **0.286° of relative inclination injected by one 180° phasing drift** (measured in the scripted ledger: Δi_rel 0.2037° → 0.4898°, +0.286° — matching the analytic value to 3 digits from a completely separate code path). That is **76% of the 0.3775° free-plane zone at the 30 km/50 m/s box** and **3.8× the 0.0752° tolerance at 10 km/10 m/s**. X3 currently scores 3/3 seeds at 200/200; under J2 the drift leg alone consumes most of its plane margin. That is a real capability delta with a clean control arm, and it is the *only* reason to do this.

Three conditions gate that entirely, and two of them are violated by the shipped ext-3d defaults:

1. **`j2_mode=1` requires `dim3_mode=1`.** (Not a style rule — see §1.4.)
2. **`i_t = 0` makes J2 a plane no-op.** At an equatorial target, Δi_rel = i_s regardless of Ω_s, so differential Ω̇ cannot move the plane term at all: measured Δ(dv_pl) = **+0.00 m/s** and ΔΦ = **−0.00000** over a full 6000-step cap at every Δi. The current sampler hard-defaults `i_t = Ω_t = 0` (correct under two-body, where the plane is pure gauge). **Under J2 the equator is physical and `i_t` must be sampled > 0**, or the whole upgrade is inert — the 4th instance of this project's inert-knob class, caught before implementation this time.
3. The channel is **LEO-only within the proven caps** (§2.3).

---

## 1. Physics

### 1.1 The rates (Vallado 9-38…9-40), mean elements, secular only

```
n  = sqrt(MU/a³)          p = a(1−e²)          k = 1.5·n·J2·(R_EQ/p)²
Ω̇  = −k·cos i
ω̇  = +0.5·k·(4 − 5 sin²i)
Ṁ  =  n + 0.5·k·√(1−e²)·(2 − 3 sin²i)
ȧ  = ė = i̇ = 0        (secular J2 has no secular rate on a, e, i)
J2 = 1.08263e-3       R_EQ = 6.378137e6 m
```

**`R_EQ ≠ R_EARTH`.** The env's `R_EARTH = 6.371e6` is the *mean* radius (used for altitudes and the keepout). J2's reference radius is the *equatorial* radius, 6378.137 km. Using the env constant biases every rate by (6371/6378.137)² = **−0.22%**. Add a separate `#define J2_R_EQ 6.378137e6` and say so in the comment — this is a 7 km inconsistency that already exists in the env's altitude bookkeeping and should not be laundered into the dynamics. (The oracle can only barely see this error — §3.3.)

Measured table (`j2_rates.csv`, 98 rows):

| alt km | i° | Ω̇ °/day | ω̇ °/day | (Ṁ−n)/n |
|---|---|---|---|---|
| 400 | 28.5 | −7.104 | +11.565 | 9.49e−4 |
| 400 | 51.6 | −5.021 | +3.755 | 1.14e−4 |
| 400 | 97.4 | +1.041 | −3.706 | −6.85e−4 |
| 8000 | 51.6 | −0.361 | +0.270 | 2.52e−5 |
| 20200 | 51.6 | −0.042 | +0.031 | 7.37e−6 |

### 1.2 Exactness under warps — verified, including the Ṁ ordering

The rates depend only on (a, e, i), which secular J2 leaves invariant ⇒ **the rates are constant for the whole life of an element set** ⇒ the map is closed-form at any dt and the warp actions (τ = 5/30/60/180/360) stay exact. The existing `propagate_orbit` ordering (`M += rate·dt` → `fmod` → `solve_kepler` → `eccentric_to_true`) needs no restructuring: the *only* change is which rate multiplies dt, plus two new linear angle updates ahead of it.

Additivity measured in double precision, 1 call of τ·DT vs τ calls of DT (`j2_warp_exact.csv`):

| τ | \|Δpos\| j2=1 | \|Δpos\| j2=0 (baseline) |
|---|---|---|
| 5 / 30 / 60 | 8.0e−9 / 1.8e−8 / 4.3e−8 m | 0 |
| 180 / 360 | 8.0e−9 / 2.3e−8 m | 0 |
| 3000 | 2.6e−7 m | 7.5e−8 m |

The residual is `fmod`-accumulation float noise of the *same class and order* the two-body path already shows at τ=3000, not a J2 defect. Nanometres over 50 hours.

**Anchor:** `j2_mode=0` reproduces the current propagator **bit-exactly** — worst |Δ| = **0.000e+00 rad** on M, Ω, ω over 20,000 random draws (a ∈ 300–8000 km, e ≤ 0.30, i ≤ 60°). Achieved by branching, not by adding a zero.

### 1.3 The equatorial trap (must be in the implementation, or the gauge anchor silently dies)

At `i = 0`, Ω is undefined/gauge but Ω̇ = −k is *maximal*. The physical quantity is ϖ̇ = Ω̇ + ω̇ = **+k**. Two failure modes:

- Propagating Ω normally drives `target.raan` off 0.0 after one step, which silently disables `gauge_from_orbit`'s `identity` fast path — λ switches from the bit-exact `M+ω` form to the cartesian round-trip form mid-episode. Φ moves by FP noise, but the A2-class anchor is gone.
- Naively "skipping Ω̇ because Ω is gauge" while keeping ω̇ = 2k gives ϖ̇ = 2k — **wrong by exactly 2×**.

**Correct special case:** `if (o->inc == 0.0) { o->omega += (om + Om)*dt; /* raan stays 0.0 */ } else { normal }`. Validated against the Cowell integrator, which independently resolves the equatorial degeneracy the same way: measured equatorial ω̇ = **1.5713e−6 rad/s** vs (Ω̇+ω̇)_secular = **1.5586e−6** (0.8%, inside the ω̇ floor). This keeps `dim3_mode=0` orbits exactly equatorial and the identity gauge alive.

### 1.4 What is deliberately omitted, and the honest truth statement

Omitted: **all short-period and m-daily osculating terms**, **long-period terms**, J2², J3+, drag, third-body, SRP. Keeping only secular rates is what buys closed-form warps, and warps are load-bearing for this project.

> **Truth statement.** Under `j2_mode=1` the env's state *is* the mean element set. No mean↔osculating conversion is performed anywhere — not at reset, not at a burn, not at the success test. A burn is applied to the mean state through the osculating Gauss response (`orbit_to_cartesian → +Δv → cartesian_to_elements`), an O(J2) inconsistency that is stated, not corrected.

Quantified (`j2_meantruth.csv`) — re-flying the same numbers through a full-J2 Cowell integrator that reads them as *osculating*:

| quantity | 1 orbit | 3 orbits |
|---|---|---|
| absolute position divergence | 29.7 km | 89.1 km |
| relative-state error @ 5 km separation | **83 m / 0.094 m/s** | 249 m / 0.282 m/s |
| relative-state error @ 30 km | 498 m / 0.56 m/s | 1.49 km / 1.69 m/s |
| relative-state error @ 2000 km | 34.9 km / 39.3 m/s | 105 km / 118 m/s |

The absolute error is 360× the relative error at the 5 km box: it is **common-mode along-track and cancels in the relative state**, scaling linearly at **1.66% of the separation per orbit**. So mean-element truth is benign exactly where the success classifier looks — with one caveat worth writing down: at the 5 km/1 m/s tight box the *velocity* term is **9.4% of tolerance per orbit** (28% over three), so a tight-box J2 rung must not also claim osculating-grade terminal fidelity.

`a_mean − a_osc` at LEO-500 is **+5.6 km** — larger than the 5 km box. That is a definitional offset, not an error, but it is the reason §3 exists.

---

## 2. Shaping re-audit under J2 (current mode-2 formulas, verbatim)

Re-derived against the shipped algebra (`Δv_in = 0.5·v_t·√(δa_rel² + ‖Δē₃‖²)`, `Δv_pl = 1.0·v_t·‖ĥ_s−ĥ_t‖`, L1 sum, `min(1,·)`, `W_λ=1.0, W_m=0.817, dv_ref=700`), not against §4.6's estimate.

### 2.1 The ledger (`j2_shaping_ledger.csv`) — 3d_C §3.1 scenario, re-scripted

| leg | ΔΦ j2=0 | ΔΦ j2=1 | Δ |
|---|---|---|---|
| L1a coast-to-node | −0.0000 | +0.0002 | +0.0002 |
| L1b plane crank ×6 (150 m/s) | +0.1236 | +0.1238 | +0.0001 |
| L2 drift-open ×5 | −0.1735 | −0.1739 | −0.0004 |
| **L3 drift (1110 steps, 18.5 h)** | **+0.9913** | **+0.9464** | **−0.0450** |
| L4 drift-close ×5 | −0.0317 | −0.0325 | −0.0008 |
| **total** | **+0.9097** | **+0.8639** | **−0.0458 (−5.0%)** |
| worst adverse step | **−0.0354** | **−0.0354** | 0.0000 |
| Δi_rel after crank / after drift | 0.204° / 0.204° | 0.203° / **0.490°** | **+0.286°** |

**Verdicts against the T3 gates:**

- **Drift-leg monotonicity: PRESERVED.** 36/36 30-step slices strictly positive under J2 (min **+0.01249**, max +0.02610) vs (min +0.01512, max +0.02712) two-body. **Zero sign flips.** The drift leg remains an unbroken climb.
- **Worst adverse step: UNCHANGED at −0.0354** = `W_m·25/dv_ref`, still set by the burn quantum, not by J2 ⇒ the **245× margin vs the +10 terminal survives intact**. Φ's range (1.817) and the 5.5:1 terminal dominance are untouched (Φ is bounded by construction).
- **§4.6's ≤4.3% estimate: confirmed at 5.0%**, and the −0.0450 is fully explained by the plane term: 0.286° × 134 m/s/deg = 38.3 m/s ⇒ 0.817·38.3/700 = **0.0447**. Residual −0.0003 ⇒ the Δλ and in-plane (Δē rotating at differential ω̇) perturbations together are **≤ 0.26 m/s of Δv-equivalent** — negligible, as predicted (e·Δω ≈ 9.5e−5 at e=0.02).
- **`L1a coast-to-node = exactly 0` is LOST** (now +0.0002). §4.6's H1-recurrence flag was right in kind, wrong in size.

### 2.2 Do-nothing leak (`j2_donothing.csv`) — the gate that must be restated

T3's pre-flight gate is "do-nothing shaping ≈ 0", and under two-body it is **exactly** 0.00000 in every cell. Under J2, coasting a *full cap* at δa = 0:

| alt / cap | max +ΔΦ | min ΔΦ |
|---|---|---|
| LEO-500 / 6000 | **+0.00537** (i_t=51.6°, Δi=1°) | −0.03292 (i_t=97.4°, Δi=1°) |
| 8000 km / 6000 | +0.00042 | −0.00073 |
| MEO-20200 / 12000 | +0.00003 | −0.00013 |
| any alt, **i_t = 0** | **0.00000** | 0.00000 |

Mechanism: at δa = 0 but Δi ≠ 0 the two orbits have different i ⇒ different Ω̇ ⇒ Δi_rel relaxes (prograde, i_t=51.6°: 1.000° → 0.901° over 100 h, **13.2 m/s of free plane correction**) or grows (retrograde, i_t=97.4°: 1.000° → 1.189°, **−25.2 m/s**).

**Restate the gate as: `|do-nothing ΔΦ| ≤ 0.006 over a full cap` (0.06% of the terminal, 16% of one adverse step), sign-indefinite, never exactly 0.** Not farmable: `shape_gamma = 1` telescopes, and the positive branch is a physical drift the agent cannot reverse or re-harvest.

### 2.3 The new channel — differential nodal drift (`j2_channel.csv`)

The clean result: **ΔΩ per radian of Δλ closed is `−3.5·J2·(R_E/p)²·cos i`, independent of δa** (Ω̇ ∝ a^−7/2, n ∝ a^−3/2 ⇒ ratio = (7/3)·(Ω̇/n)). Converting to relative inclination (Δi_rel ≈ ΔΩ·sin i) gives **1.75·J2·(R_E/p)²·sin 2i per radian** — **maximal at i = 45°, identically zero at i = 0 and i = 90°.**

| alt km | i° | Δi_rel per rad of λ | per 180° | per 360° | Δv-equiv @360° | %budget | drift h @δa=200 km |
|---|---|---|---|---|---|---|---|
| 500 | 28.5 | 0.0784°/rad | 0.246° | 0.493° | 65.5 m/s | 13.7% | 36.1 |
| 500 | **45.0** | **0.0935** | 0.294 | **0.588** | **78.1** | **16.3%** | 36.1 |
| 500 | 51.6 | 0.0911 | **0.286** | 0.572 | 76.1 | 15.9% | 36.1 |
| 500 | 97.4 | 0.0239 | 0.075 | 0.150 | 20.0 | 4.2% | 36.1 |
| 8000 | 51.6 | 0.0208 | 0.065 | 0.131 | 12.0 | 2.5% | **228** |
| 20200 | 51.6 | 0.0061 | 0.019 | 0.038 | 2.6 | 0.5% | **1060** |

**Is this a usable new channel?** Partly, and the honest answer has three parts:

1. **Mostly it is a forced disturbance, not a choice.** The sign of the accrued ΔΩ is locked to the sign of the phase change, which is set by the geometry. At 180° gap the policy pays 0.286° of plane error to close phase — **76% of the free-plane zone**, on top of whatever `di_max` sampled.
2. **The one genuine decision is the go-around.** Closing gap *g* the short way vs *g − 2π* the long way flips the sign and rescales the magnitude, for **zero extra Δv** (same drift orbit, same open/close burns) at the cost of clock: a full 2π go-around at LEO/δa=200 km is 36.1 h = 2168 steps, comfortably inside a 6000-step cap, and buys up to **0.572° ≡ 76 m/s ≡ 16% of budget**. This is real, operationally authentic (it is how RAAN is actually managed) and directly measurable as a policy behaviour.
3. **It only reaches half the error space and only at LEO.** J2 moves ΔΩ at fixed i, a 1-D subspace of the 2-D δı⃗; the `di_max` disc sampler draws n̂ uniformly, so on average only the projection is correctable. And a 2π go-around needs 228 h at 8000 km / 1060 h at MEO — outside the 6000-step (100 h) and 12000-step (200 h) caps. **The J2 channel is a LEO phenomenon inside the proven horizons.**

Upper bound if a drift orbit is simply *held* for a whole cap (δa=200 km, 6000 steps, LEO, i_t=51.6°): the plane term grows by **198 m/s** (Δi_rel 0 → 1.489°) — 41% of the budget. §4.6's "≤37 m/s" was right for one 180° drift and 5.4× optimistic for a full-cap hold.

---

## 3. Oracle + fuzz protocol

The universal-variable oracle is two-body and cannot check J2. The independent cross-check is **Cowell numerical integration of the full J2 acceleration** (DOP853, rtol 1e-12) — a different algebra (force model vs analytic rates) that produces *osculating* elements, so the protocol must compare orbit-averaged rates, never element values.

### 3.1 The protocol (and the trap it must avoid)

The naive version — fit the secular rates from the integration and compare to `secular_rates(a_IC, e_IC, i_IC)` — **fails**, and instructively (`j2_oracle.csv`): Ω̇ stalls at a 2.5e−3 floor that does not improve with integration length, and the Ṁ check is useless (relerr **1.1 – 1000**). Cause: reading the mean IC as osculating injects `⟨a_osc⟩ − a_IC = +5.6 km` at LEO (+8.2e−4 relative), which is 3.5× amplified in Ω̇ (∝ a^−7/2) and **12× larger than the entire J2 correction to Ṁ**.

**Corrected protocol** (`j2a_oracle2.py`, `j2_oracle2.csv`):

1. Integrate N complete orbits with Cowell J2 from the element set read as osculating.
2. Convert every sample to osculating elements (atan2-only, degeneracy conventions that make coe2rv∘rv2coe the identity).
3. **Recover the mean elements by orbit-averaging** `a, e, i` over an integer number of orbits. M is linear in t to O(J2), so uniform-in-time sampling is uniform-in-M and the Brouwer short-period terms average to zero at first order ⇒ `⟨a_osc⟩ = a_mean + O(J2²)`.
4. Fit Ω̇, ω̇, Ṁ by least squares on the unwrapped angles (LSQ averages out the periodic terms).
5. Compare against `secular_rates(a_mean, e_mean, i_mean)`. **Compare the J2 *correction* `Ṁ − n`, never Ṁ itself** (the correction is 1e−4 of Ṁ; comparing Ṁ hides a 100% error in the correction inside a 1e−4 relative agreement).
6. **At e → 0, check `λ̇ = Ω̇ + ω̇ + Ṁ`, not ω̇ and Ṁ separately** — their split is gauge and the individual checks return relerr 400–1000 on a *correct* model.
7. **At i = 0, check `ϖ̇ = Ω̇ + ω̇`** (validated at 0.8%, §1.3).

Measured agreement floors:

| case | N | Ω̇ | ω̇ | (Ṁ−n) | (λ̇−n) |
|---|---|---|---|---|---|
| LEO-500 i=51.6 e=0.02 | 20 / 60 | 6.3e−4 / 7.6e−4 | 1.0e−2 / 2.7e−3 | 3.8e−2 / 4.8e−3 | 1.5e−2 |
| LEO-800 i=97.4 e=0.05 | 60 | 7.8e−4 | 1.4e−3 | 7.1e−3 | 3.3e−3 |
| **WIDE-8000 i=45 e=0.30** | **60** | **4.1e−4** | **7.7e−4** | **2.8e−4** | 1.3e−3 |
| MEO-20200 i=55 e=0.01 | 60 | 1.6e−5 | 7.7e−4 | 1.7e−2 | 5.0e−4 |
| LEO-500 i=28.5 e≈0 | 60 | 2.5e−3 | *(400–500, gauge)* | *(1000, gauge)* | **3.3e−3** |

### 3.2 Mutation table — proof the protocol indicts real errors

12 seeded errors, scored as relative deviation from the *measured* rate (`j2_oracle2_mutations.csv`):

| seeded error | LEO-500 e=0.02 N=20 | WIDE-8000 e=0.30 N=60 | margin over floor |
|---|---|---|---|
| *(correct)* | 6.3e−4 / 1.0e−2 / 3.9e−2 | 4.1e−4 / 7.7e−4 / 2.8e−4 | — |
| Ω̇ sign flipped | **2.00** | **2.00** | ≥3000× |
| Ω̇ factor 0.75 not 1.5 | **0.500** | **0.500** | ≥790× |
| Ω̇ uses sin i not cos i | **0.262** | 2.9e−4 *(i=45°: sin=cos — blind)* | 416× at LEO |
| ω̇ sign flipped | **1.99** | **2.00** | ≥198× |
| ω̇ factor 2× | **0.980** | **0.999** | ≥97× |
| ω̇ uses the Ṁ bracket (2−2.5s²) | **0.505** | **0.500** | ≥50× |
| Ṁ correction omitted | **1.000** | **1.000** | ≥26× |
| Ṁ correction sign flipped | **2.04** | **2.00** | ≥52× |
| Ṁ uses the ω̇ bracket (4−5s²) | **5.16** | **2.00** | ≥132× |

**Three documented blind spots — state them, do not paper over them:**

- **`R_ENV` vs `R_EQ`** scores only **2.6–2.9e−3** (a 0.22% physical effect). Margin ~4× at LEO N=20, ~6× at WIDE N=60. Detectable but weak: **fix the radius by inspection and convention, not by the oracle.**
- **`p` vs `a` in (R/p)²** is invisible at low e (1.4e−3 at e=0.02, ~2× margin) and obvious at high e (**0.172**, 420× margin at e=0.30). **The fuzz matrix must contain an e ≥ 0.3 cell** — which the V5 envelope already justifies.
- **Missing `√(1−e²)` in Ṁ** likewise: 3.9e−2 at e=0.02 (indistinguishable from the floor) vs **4.8e−2 at e=0.30 with a 170× margin**. Same cell covers it.

### 3.3 Required fuzz matrix

Minimum 5 cells × N=60: `LEO-500 i=51.6 e=0.02` (sin/cos discrimination), `LEO-800 i=97.4 e=0.05` (retrograde sign of Ω̇), `WIDE-8000 i=45 e=0.30` (p-vs-a and √(1−e²)), `LEO-500 i≈0 e=0.05` (equatorial ϖ̇ special case), `LEO-500 i=28.5 e≈0` (λ̇-only cell). Plus the two structural checks the existing 3D battery already provides and that J2 must *not* break: **I2/I3 (ĥ across coast) and I5 (e-vector across coast) will now fire by design** — under J2 the plane *does* rotate and the apse line *does* precess during a coast. Those two invariants must be re-thresholded to the predicted secular rates under `j2_mode=1` (they become *rate* checks, not *constancy* checks), and left as constancy checks at `j2_mode=0`. This is exactly the B11/B15 mutation class from 3d_E §4 — the battery was explicitly built anticipating "someone left a J2 term in the two-body propagator", so it will now indict the correct behaviour unless re-thresholded. **This is the single largest test-side cost of the upgrade.**

---

## 4. Kwargs, anchors, obs, ladder

### 4.1 Surface

```c
/* orbital.h */
#define J2_COEF   1.08263e-3
#define J2_R_EQ   6.378137e6     /* equatorial — NOT R_EARTH (mean, 6.371e6) */

int j2_mode;                     /* 0 = off (bit-exact anchor), 1 = secular mean-element J2 */

static inline void propagate_orbit_j2(Orbit* o, double dt, int j2_mode) {
    if (!j2_mode) { propagate_orbit(o, dt); return; }   /* verbatim legacy path */
    double n  = sqrt(MU / (o->a * o->a * o->a));
    double p  = o->a * (1.0 - o->e * o->e);
    double rp = J2_R_EQ / p;
    double k  = 1.5 * n * J2_COEF * rp * rp;
    double si2 = sin(o->inc) * sin(o->inc);
    double Om  = -k * cos(o->inc);
    double om  =  0.5 * k * (4.0 - 5.0 * si2);
    double Md  =  n + 0.5 * k * sqrt(1.0 - o->e * o->e) * (2.0 - 3.0 * si2);
    if (o->inc == 0.0) {                 /* §1.3: Ω is gauge, ϖ̇ = Ω̇+ω̇; keep raan == 0.0
                                          * so gauge_from_orbit's identity path survives */
        o->omega = wrap_2pi(o->omega + (om + Om) * dt);
    } else {
        o->raan  = wrap_2pi(o->raan  + Om * dt);
        o->omega = wrap_2pi(o->omega + om * dt);
    }
    o->M += Md * dt;                     /* same ordering as legacy from here down */
    o->M = fmod(o->M, 2.0*M_PI); if (o->M < 0.0) o->M += 2.0*M_PI;
    o->theta = eccentric_to_true(solve_kepler(o->M, o->e), o->e);
}
```

Three call sites only (`c_step` sub-step loop: bodies, target, satellite). Thread `j2_mode` through `binding.c::my_init` → `orbital.py` → `orbital.ini` → `eval_checkpoint.py::unpack()` (hard-fails on a missing key). **No action-table change** — Discrete(30) is unchanged; J2 needs no new actions.

**Preconditions to assert at init:** `j2_mode=1 ⇒ dim3_mode=1`; `j2_mode=1 ⇒ i_target_rad > 0` (else the plane channel is provably inert, §2.2); `j2_mode=1 ⇒ num_debris = 0` (inherited).

**Sampler change:** `i_target_rad` must become a *sampled* per-episode quantity under `j2_mode=1` (recommend uniform over {28.5°, 45°, 51.6°, 63.4°, 97.4°} or U(20°, 100°)), not the fixed 0 gauge. Correspondingly `frame_randomize` degrades from an SO(3) invariance test to an **SO(2)-about-ẑ** test — still a valid leak detector, and it should be re-run in that reduced form because a policy that is rotation-variant about ẑ is still broken.

### 4.2 Anchors

- **J-A1 (bit-exact):** `j2_mode=0` reproduces the current propagator exactly. Measured 0.000e+00 over 20,000 draws.
- **J-A2 (bit-exact):** T3 canonical + legacy eval unchanged under default flags (`j2_mode=0`) — the 425,477-row trajectory check re-run.
- **J-A3 (equatorial closure):** `j2_mode=1, dim3_mode=1, i_t=0` ⇒ `raan == 0.0` exactly for every body at every step, identity gauge never disengages, and ϖ̇ matches the oracle to 1%.
- **J-A4 (warp additivity):** τ·DT in one call vs τ calls of DT, all τ in the action table, ≤1e−6 m.
- **J-A5 (inertness guard):** `j2_mode=1, i_t=0` ⇒ Δ(dv_pl) = 0 over a full cap. This anchor exists to make the §2.2 inert-knob failure loud instead of silent.

### 4.3 Obs — correction to the 3d_C reservation

3d_C §2 reserved "obs 28–29" for `cos i_s`, `cos i_t`. **The shipped ext-3d code uses obs[28] for the Δv feasibility margin**; the genuinely free reserved slots are **obs[29]–obs[32]**. Therefore:

```
obs[29] = cos i_s        written only under j2_mode=1, else 0.0f
obs[30] = cos i_t        ″
obs[31], obs[32]         remain reserved
```

Rationale unchanged and still correct: J2 is axisymmetric, so Ω stays gauge and only `cos i` is needed to make the equator observable. Dim stays 38; every validated decoder, the scripted expert, the EKF injector and the 2D→3D warm-start path are untouched. **Consider also `obs[31] = Ω̇_s/|Ω̇|_ref`** if the go-around behaviour (§2.3-2) is to be learnable rather than merely discoverable — the rate is a smooth function of (a,e,i) the net could infer, but it is 2 lines and it is the exact quantity the decision keys on. Recommend deferring it to a second arm so the attribution stays clean.

### 4.4 Which rung re-runs under J2 — recommend exactly one: **X3**

`X3` = LEO 300–800 km, e ≤ 0.05, ±180° phase, ΔI_rel ≤ 1.0°, currently **3/3 seeds at 200/200**.

Why X3 and not the nav rung, despite the task's prior: **the shipped nav stack is 2D, and J2 is close to a no-op there.** Measured model error of a J2-ignorant relative filter over a 105-min acquisition window (`j2_nav_modelerr.csv`):

| geometry | \|Δρ\| error | \|Δρ̇\| error |
|---|---|---|
| coplanar, co-altitude (the NB1 arms) | **145 – 598 m** | 0.07 – 0.68 m/s |
| Δi = 1° (3D) | **1.29 – 3.04 km** | 2.66 – 4.07 m/s |
| δa = 200 km drift orbit | **7.9 – 39.5 km** | 8.8 – 45 m/s |

At the 2D nav rung, J2 injects a few hundred metres — under the 5 km box, invisible against the filter's own error. **J2 only becomes a navigation story once the pair is non-coplanar** (1.3–3.0 km = 26–61% of the tight box) **or on a drift orbit.** So: J2 belongs to the *3D-nav* rung, which does not exist yet. Do not spend it on the 2D nav arms.

**Recommended experiment — J2-X3, 4 arms, 2 seeds each:**

| arm | config | purpose |
|---|---|---|
| **A0** | X3 as shipped, `j2_mode=0`, `i_t` sampled | isolates "i_t sampling" from "J2" — without this the comparison is confounded |
| **A1** | A0 + `j2_mode=1`, X3 checkpoint **zero-shot** | measures the disturbance the trained policy already absorbs |
| **A2** | A1, **retrained** from the X3 checkpoint | the fidelity/capability demonstration |
| **A3** *(optional)* | A2 with cap raised to 6000 and the go-around reachable | tests whether the policy *exploits* J2 rather than merely rejecting it |

**Pre-registered predictions** (falsifiable, worth being wrong about):

- **A0 ≈ 200/200.** Sampling i_t alone changes nothing under two-body (pure gauge) — if it does, there is a rotation-variant obs leak and the `frame_randomize` SO(2) test should have caught it.
- **A1 = 55–85%.** The drift injects 0.286° into a 0.3775° box; the residual is one 25 m/s crank (0.187°) away, and the policy already owns the crank action, so partial absorption is expected. A1 < 40% would mean the policy is timing-blind to the plane and would itself be the interesting result.
- **A2 ≥ 95%**, with **median Δv up +30 to +45 m/s** and a visible **post-drift re-crank** in the action trace. A2 restoring X3 performance at ~zero Δv cost would mean the policy found the go-around (surprising at cap 3000) or pre-biased the plane before drifting (the smart answer, and the one worth a figure).
- **A3:** go-around usage measurable as a bimodal `sign(δa)` distribution vs the phase gap; expected uptake low unless the budget is tightened, since the direct crank is affordable at ΔI ≤ 1°. **If the demonstration must show J2 being *used* rather than *rejected*, tighten the Δv budget or raise `di_max` to 1.5–2°** — at the current envelope the policy has no reason to bother, and that is a scoping decision, not a result.

**Do NOT re-run V5 under J2.** The wide envelope reaches 8000 km where the channel needs 228 h — outside the cap. J2 there is pure fidelity with no measurable behavioural delta, and it would burn a 100-h-episode rung for nothing.

### 4.5 Scope and cost

| item | cost |
|---|---|
| propagator + kwarg + preconditions + obs[29,30] + sampler | ~40 lines, 1 day |
| anchors J-A1…J-A5 | half a day |
| J2 oracle (`j2a_core.py` already contains the integrator) + 5-cell fuzz matrix | 1 day |
| **re-thresholding I2/I3/I5 to secular rates under `j2_mode=1`** | **1 day — the largest test-side cost, and the one most likely to be under-budgeted** |
| J2-X3, 4 arms × 2 seeds | ~1 GPU-day |

**Recommendation: GO on J2-1 (physics + anchors + oracle) and J2-2 (J2-X3 arms A0–A2). DEFER J2-3 (nav under J2) until 3D-nav exists**, at which point J2 becomes the natural unmodeled-dynamics stressor: give the filter matched secular J2 as the baseline and run a "−J2 filter" ablation whose model error is already tabulated above (1.3–3.0 km at Δi=1°). That ablation — *the measured navigation cost of one unmodeled dynamic, with a matched control* — is the Draper-relevant artifact here, more than the rate formulas themselves.

**Two corrections to carry back into the record:** 3d_C §4.6's "fidelity upgrade, not task-enrichment" holds at MEO/wide and fails at LEO (0.286° per 180° drift = 76% of the plane tolerance); and 3d_C §2's "obs 28–29 reserved" is stale — obs[28] is occupied, use obs[29]/obs[30].