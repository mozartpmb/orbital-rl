Reconciliation complete. All measurements done and cross-validated. Final report:

---

# Angles-only rendezvous observability — theory, our regime, and the RL campaign design

## VERDICT (headline)

**Maneuver-for-observability is REQUIRED only at the tight terminal box, and the requirement is created by the guidance objective itself.** Three regimes, all measured:

| regime | separation | drift-only range σ/ρ | 1 m/s burn worth | verdict |
|---|---|---|---|---|
| far / mid field | ρ ≥ 500 km | ≤ 2% | **×1.00** | **UNNECESSARY** — Keplerian nonlinearity gives range for free |
| near field | 30–300 km | 0.8–19% | ×1.6–1800 | helpful, sharply decreasing |
| terminal box | ρ ≤ 10 km, Δa nulled | 26–520% | **×10³–10⁶** | **REQUIRED** |

**The campaign's headline hypothesis must be run at the TIGHT box (TB4/TB5), not the T3 headline box.** At 30 km/50 m/s the drift-only angles-only nav error is 1.8% of the tolerance — nav error can never cost the agent reward, so the experiment is guaranteed null for trivial reasons. At 5 km/1 m/s it is 52% of the tolerance and the hypothesis is live.

**The sharpest new result:** observability at close range is governed by the **semi-major-axis difference δa, not the separation**. And a rendezvous policy's terminal job is to null δa. Nulling relative velocity to `v_tol` leaves `δa = v_tol/(1.5n)`, so:

| success box | v_tol | implied δa | drift-only σ_range | **as fraction of box** | info gain of one 1 m/s burn |
|---|---|---|---|---|---|
| T3 headline 30 km/50 m/s | 50 | 29.4 km | 541 m | **1.8%** | ×1.37 |
| TB3 10 km/10 m/s | 10 | 5.9 km | 682 m | **6.8%** | ×21.6 |
| TB4 5 km/2 m/s | 2 | 1.18 km | 1372 m | **27%** | ×2.7e3 |
| TB5 5 km/1 m/s | 1 | 588 m | 2589 m | **52%** | ×1.4e4 |
| (hypothetical 5 km/0.1 m/s) | 0.1 | 59 m | 25.3 km | **506%** | ×1.9e6 |

Angles-only observability collapses *exactly* as the agent succeeds. That tension — guidance wants δa→0, navigation wants δa≠0 or a burn — is a genuine dual-control problem, it is quantified above, and it is the campaign's real story.

---

## 1. The criteria and what the literature actually says

### 1.1 The unobservable family (verified)

Under **linear** relative dynamics — CW/Hill, Yamanaka–Ankersen, any homogeneous LTV — the relative state propagates as `r(i) = Φ_rr(i)r₀ + Φ_rv(i)v₀`, and the LOS unit vector is

```
i_LOS(i) = [Φ_rr(i)r₀ + Φ_rv(i)v₀] / ‖Φ_rr(i)r₀ + Φ_rv(i)v₀‖
```

which is invariant under `(r₀,v₀) → k(r₀,v₀)` for any scalar k>0. The entire ray produces a bit-identical bearing history; range is structurally unobservable. Unobservability requires **four conjoint conditions** — linear dynamics, single sensor, sensor at the CoM, coasting flight — and **breaking any one restores observability**. That taxonomy is the useful frame: maneuvers break the fourth; camera CoM-offset plus attitude motion breaks the third (Klein & Geller's zero-Δv route); **nonlinear dynamics breaks the first, which is our case**.

- **D. C. Woffinden, D. K. Geller, "Observability Criteria for Angles-Only Navigation," IEEE Trans. Aerospace and Electronic Systems 45(3), July 2009, pp. 1194–1208, DOI 10.1109/TAES.2009.5259193.** *(Note: 2009 Vol. 45, not 2008 Vol. 44.)* Abstract, verbatim: *"with angle measurements alone, the relative position and velocity cannot be determined for systems with linear dynamics. However, with a calibrated thrust maneuver, observability can be guaranteed for all possible relative trajectories."*
- Woffinden & Geller, "Optimal Orbital Rendezvous Maneuvering for Angles-Only Navigation," **JGCD 32(4), 2009, 1382–1387, DOI 10.2514/1.45006**.
- Woffinden PhD thesis, Utah State, 2008. Prior art: **R. J. V. Chari, MIT S.M. thesis, 2001** — the origin of the "nonlinearity slowly reduces range uncertainty" observation.

### 1.2 What a maneuver does — the criterion

A calibrated impulse enters as an inhomogeneous forcing term, so `r(t) = p(t) + w(t)` with `p` the ballistic propagation of the true IC and `w(t) = Σ_j Φ_rv(t,t_j)Δv_j`. The hypothesis `k·x₀` propagates to `k·p(t) + w(t)`, **not** `k·r(t)`. Indistinguishability requires `w_i ∥ p_i`, giving

```
UNOBSERVABLE ⟺ [Σ_j Φ_rv(t_i,t_j)Δv_j] × [Φ_rr(t_i,t_0)r_0 + Φ_rv(t_i,t_0)v_0] = 0  ∀ t_i
```

Woffinden's own modern phrasing (Cavesmith/Woffinden/Collins, AAS 24-168, NASA NTRS 20240000556, verbatim): *"observability is guaranteed when the change in position due to a calibrated thrust acceleration is not aligned with the natural line-of-sight profile."* Note the criterion is **initial-condition dependent**, which is why Grzymisch frames the problem as solving for unobservable *maneuver sets*.

### 1.3 Unobservable maneuvers, and the geometry

- **J. Grzymisch, W. Fichter, "Observability Criteria and Unobservable Maneuvers for In-Orbit Bearings-Only Navigation," JGCD 37(4), 2014, 1250–1259, DOI 10.2514/1.62476.** Abstract, verbatim: *"unobservable impulsive maneuvers can only occur on a departure trajectory and that all impulsive maneuvers performed on an approach trajectory are observable"*; *"unobservable constant thrust maneuvers are possible during approach trajectories."*
- Grzymisch & Fichter, "Analytic Optimal Observability Maneuvers for In-Orbit Bearings-Only Rendezvous," **JGCD 37(5), 2014, 1658–1664, DOI 10.2514/1.G000612**. *(Both 2014 Vol. 37.)*
- Grzymisch & Fichter, "Optimal Rendezvous Guidance with Enhanced Bearings-Only Observability," JGCD 38(6), 2015, 1131–1140, DOI 10.2514/1.G000822.

**Optimal direction is ⊥ LOS** (confirmed three ways: Gong et al. review, GUIBEAR, Woffinden), but Woffinden's is a *trade* not pure 90°: perpendicular maneuvers maximise angular signature but cost more Δv and deviate more from the baseline.

**The most citable limitation in the field** — an ESA team (Serra et al., GUIBEAR, 11th ESA GNC, Sopot 2021, DOI 10.5281/zenodo.6471464, CC-BY) implementing Grzymisch's own cost, verbatim: *"the above observability cost does not correlate perfectly with the observability of the range."* Relevant to the lead-with-limitations framing in the Draper notes.

### 1.4 Scaling laws

Derived from the STM (no published closed form found in either recon): `sin θ_obs(t) ≈ ‖P_⊥ Φ_rv(t,t_m)Δv‖/ρ(t)`, hence **linear in |Δv|, ∝ sin α (max at 90° from LOS), ∝ 1/ρ, linear in Δt for short arcs** (super-linear only once `n·Δt = O(1)`).

**Confirmed in our own simulator** (`ext_angles_burn_threshold.csv`, A1 = 5 km LEO, 90 min arc): local exponent of ambiguity-breaking angle vs Δv = **1.05, 1.01, 0.96** at Δv = 0.03/0.1/0.3 m/s — linear to 1–5% — rolling to 0.85/0.68/0.49 at 1/3/10 m/s as the geometry perturbation stops being small.

### 1.5 Nonlinearity breaks the ambiguity — established, and used in flight

The relevant successor line is the D'Amico / ROE school, where the ambiguity **condenses into a single element**, the relative mean argument of latitude δλ (≈ range) — the orbital analogue of Aidala & Hammel's modified-polar TMA coordinates.

- Gaias, D'Amico, Ardaens, "Angles-Only Navigation to a Noncooperative Satellite Using Relative Orbital Elements," JGCD 37(2), 2014, 439–451, DOI 10.2514/1.61494. Verbatim: *"the ROD fitting was able to identify the relative orbit shape except for a scale factor related to the ambiguity in aδa and aδu."*
- **Sullivan & D'Amico, "Nonlinear Kalman Filtering for Improved Angles-Only Navigation Using Relative Orbital Elements," JGCD 40(9), 2017, 2183–2199, DOI 10.2514/1.G002719** — the maneuver-free filter. Their bearing model, Eqs. (20)–(21): `tan β ≈ −δa/δλ − |δλ|/2 + (δe_x/δλ)cos u + (δe_y/δλ)sin u`. **Read the scale-invariance directly off it**: under `δα → kδα` every ratio is invariant and the *only* scale-carrying term is the orbit-curvature term `|δλ|/2 = ρ/(2r)`. That one term is the entire mechanism.
- **Ardaens & Gaias, Adv. Space Res. 63(12), 2019, 3884–3899, DOI 10.1016/j.asr.2019.03.001** — the best citation for our crux. Verbatim: *"exploits the small discrepancies which can be observed between a linear and a more advanced relative motion model... in the vicinity of a family of collinear solutions coming from the linear theory"*; *"an observation time span of a few maneuver-free orbits is enough."* Validated on ARGON (2012) and AVANTI (2016) flight data. This is literally "sweep the k-family, break the tie with the nonlinear model."
- **StarFOX / Starling, arXiv:2406.06748** — flight demonstration of **maneuver-free** angles-only convergence, **1.3% of target range (1σ)** single-observer, 0.6% multi-observer.
- **Zhou et al., arXiv:2604.24451** — pure two-body, no J2, no maneuvers, ρ/r ≈ 1.4%: 0.18% range error over one orbit. The cleanest existence proof for our exact dynamics.
- Willis & D'Amico (AAS 20-493; Adv. Space Res. 73(11), 2024, 5484–5500): the second-order angular contribution scales as **θ₂ ~ δr/r** — at 10 km / 7000 km, ≈0.1°. Also verbatim: *"Woffinden's dilemma applies specifically to linear models in Cartesian coordinates... the relative state is observable for linear models in curvilinear coordinates."*

**Caveat that cuts in our favour:** in real LEO, J2 (via the nonlinear mean→osculating map) is the workhorse, and pure-two-body second-order IROD *"fails under unmodelled perturbations in all but a narrow range of scenarios."* **Our two-body-only sim is a friendlier environment for this than reality** — state it, don't hide it.

### 1.6 Terrestrial analogue

**Nardone & Aidala, "Observability Criteria for Bearings-Only Target Motion Analysis," IEEE Trans. AES-17(2), March 1981, 162–166, DOI 10.1109/TAES.1981.309141.** Verbatim: *"for certain types of maneuvers the estimation process remains unobservable, even when the associated bearing rate is nonzero."* Order condition (Song 1996): an N-th order target needs an (N+1)-th order observer. Sufficiency correction: Fogel & Gavish, IEEE AES 24(3), 1988, 305–308.

**The one real asymmetry:** in TMA the straight line is *exact*, so nonlinearity offers no escape. In orbit the straight line is a *truncation*, and Keplerian curvature breaks it. That is the crux of our regime.

---

## 2. Quantified applicability to OUR regime (all measured)

Method: forward simulation only for the family divergence (no STM, no differencing); direct whole-arc differencing for the Fisher bound. Vectorised f&g propagator validated against the oracle `propagate_cartesian` to **1.6e-13** relative; measured bearing numerical floor **3.4e-9 mrad**, i.e. 8–11 orders below the 1 mrad sensor σ.

### 2.1 The literal classical family: bearing histories of scaled trajectories

`ext_angles_scale_ambiguity.py`. Truth vs `x_c0 + k(x_t0 − x_c0)`, both under exact two-body, k=2, σ_β=1 mrad @ 60 s. The **LINEAR arm is identically zero with no burn** (the classical symmetry, reproduced in our own code); the nonlinear arm is not.

| geometry | ρ₀ | ρ/r | t to \|Δβ\|>1 mrad | t to batch 3σ | Δβ @1 h | Δβ @50 h |
|---|---|---|---|---|---|---|
| A1 5 km box | 5.0 km | 7.4e-4 | 29 min | **29 min** | 0.45 mrad | 9.0 mrad |
| A2 10 km drift | 10.2 km | 1.5e-3 | 63 min | **27 min** | 0.78 | 44.5 |
| A3 100 km | 101.6 km | 0.015 | 6 min | **8 min** | 3.99 | 325 |
| A4 300 km drift | 301.5 km | 0.045 | 3 min | **5 min** | 13.1 | 170 |
| A5 180° gap | 13 573 km | 1.96 | 1 min | **2 min** | 2674 | — |
| A6 wide e=0.30 | 14 216 km | 1.24 | 2 min | **3 min** | 175 | — |

Batch detection (accumulated Mahalanobis) beats single-sample crossing by up to 10× — the right statistic.

**Classical theory vs full Kepler, with a 1 m/s burn** (Δβ at 1 h, mrad) — this is the crossover, stated cleanly:

| geometry | nonlinear | linear (classical) | ratio |
|---|---|---|---|
| A1 5 km | 36.79 | 36.49 | **1.01** — classical theory quantitatively exact |
| A2 10 km | 13.08 | 13.89 | 0.94 |
| A3 100 km | 2.61 | 1.43 | 1.83 — theory failing |
| A4 300 km | 13.35 | 0.384 | **34.7** — nonlinearity dominates |
| A5 13 573 km | 2674 | 6.1e-4 | **4.4e6** — maneuver contributes nothing |

**Caveat, stated plainly:** the naive k-scaling overstates observability, because an estimator re-fits the other 3 dof. Hence §2.2.

### 2.2 The honest measure: profile likelihood and the Fisher bound

`ext_angles_observability_profile.py`. `D(k) = √(min over the other 3 dof of Σ(Δβ/σ)²)`; `D≥3` = 3σ rejection. **Drift-only relative range σ (CRLB), σ_range/ρ:**

| geometry | ρ/r | 15 min | 30 min | 90 min (1 orbit) | 360 min | 1440 min | 3000 min |
|---|---|---|---|---|---|---|---|
| A1 5 km | 7.4e-4 | 481 | 44.7 | 0.747 | 0.0669 | 0.00991 | 0.00392 |
| A2 10 km | 1.5e-3 | 103 | 9.44 | 0.156 | 0.0151 | 0.00241 | 0.00144 |
| A3 100 km | 0.015 | 10.9 | 0.999 | 0.0170 | 0.00215 | 0.00135 | 0.00134 |
| A4 300 km | 0.045 | 3.06 | 0.298 | 0.00752 | 0.00164 | 0.00150 | 0.00127 |
| A5 180° gap | 1.96 | 0.0135 | 0.00222 | 1.76e-4 | 3.87e-5 | 1.86e-5 | 1.26e-5 |
| A6 wide e=0.30 | 1.24 | 0.523 | 0.054 | 0.00200 | 1.57e-4 | 9.36e-5 | 3.41e-5 |

**Time-to-observability (drift-only, no maneuver ever):**

| geometry | ρ/r | arc to σ/ρ ≤ 10% | orbits | to ≤ 1% | orbits |
|---|---|---|---|---|---|
| A1 5 km | 7.4e-4 | 286 min | **3.1** | 1431 min | 15.5 |
| A2 10 km | 1.5e-3 | 117 min | **1.3** | 491 min | 5.3 |
| A3 100 km | 0.015 | 56 min | 0.60 | 128 min | 1.4 |
| A4 300 km | 0.045 | 42 min | 0.45 | 83 min | 0.89 |
| A5 180° gap | 1.96 | 15 min | **0.16** | 17 min | 0.18 |
| A6 wide | 1.24 | 25 min | 0.11 | 53 min | 0.24 |

Two clean scaling regimes: **σ ∝ T^−3.4±0.2** while the ambiguity is being broken (remarkably consistent, −3.35 to −3.73 across A1–A4), settling to T^−1.3…−0.3 once observable. With a burn present, range is pinned within 30–90 min and then flattens (T^−0.11) — the classical picture.

**Two independent methods agree.** `1/σ_rel` (local Fisher) vs `D(k=2)` (global profile) agree to **0.7% at A1/90 min** (1.338 vs 1.329) and within a factor of 2 in **29/36** configurations. They diverge only at ρ/r > 1, in the direction physics predicts (the likelihood is nowhere near quadratic there). Independent cross-check against NAV-G's separate pipeline: my range+bearing CRLB **4.9 m vs their 4.7 m** (5 km/1 orbit), **15.8/16.8 m vs their 16.5 m** (100/300 km).

### 2.3 Where the maneuver stops mattering — the crossover

`ext_angles_crossover.py`. Information gain `G = (σ_drift/σ_burn)²` of one 1 m/s burn, 20% into the arc, best of radial/prograde:

| ρ | ρ/r | G @90 min | G @360 min | drift-only σ/ρ @90 min |
|---|---|---|---|---|
| 2 km | 3.0e-4 | 1.1e5 | 3418 | 0.775 |
| 5 km | 7.4e-4 | 3.4e4 | 2319 | 0.747 |
| 10 km | 1.5e-3 | 1.1e4 | 1621 | 0.745 |
| 30 km | 4.4e-3 | 1811 | 668 | 0.789 |
| 100 km | 0.015 | 1169 | 810 | 2.081 † |
| 200 km | 0.030 | 13.4 | 8.6 | 0.522 |
| 300 km | 0.044 | **1.56** | 1.25 | 0.176 |
| 500 km | 0.074 | **0.90** | 0.99 | 0.060 |
| ≥1000 km | ≥0.15 | **1.00** | 1.00 | ≤0.017 |

**Crossover ρ\* ≈ 300–500 km (ρ/r ≈ 0.05).** † The 100 km row is a geometry-specific resonance in this co-orbital family, not a trend — observability is not a pure function of separation (see §2.4).

### 2.4 The governing parameter is δa, not ρ — and this reconciles a 500× disagreement

The literature recon ran its own two-body separability experiment at **δa = 0 exactly** (target on the chaser's own circular orbit, pure constant along-track offset) and fitted `ε_min ≈ 2σ(r/ρ)²/√N`, predicting ε_min ≈ 384 at 5 km — hopeless. I measured 0.747. **Both are right.** `ext_angles_da_dependence.py`, ρ = 5 km, LEO, 1-orbit arc, drift-only:

| δa | along-track drift | σ_range/ρ | vs δa=0 |
|---|---|---|---|
| 0 | 0 | **159** | 1× |
| 10 m | 0.09 km/orbit | 31.2 | 5× |
| 100 m | 0.94 | 2.97 | 54× |
| 400 m | 3.77 | 0.748 | **213×** |
| 1 km | 9.4 | 0.311 | 512× |
| 10 km | 94 | 0.040 | 3940× |

At δa = 0 my measurement **matches their formula within 0.41–1.21×** at the 1-orbit arc. The 500× gap was entirely δa. Physical statement: **bearings-only range observability at close range is set by the relative drift rate, not by separation** — and our policy phases on drift orbits (|Δa| ~ 100–340 km) for most of the mission, so it is nowhere near the degenerate case *except at terminal capture*, which is precisely where the tight box needs it. That is the headline table in the verdict.

### 2.5 Burn geometry and timing

`ext_angles_burn_placement.py`. **Timing** (1 m/s ⊥ LOS, swept through the arc; A1 5 km, 6 h): gain **1.06 at burn_frac 0.0** → **1758 at 0.20** → 74 at 0.95. **A maneuver at the start of the arc is worth nothing** — it merely redefines the initial state; information comes from the pre/post-burn contrast. `d(info)/d(epoch)` is sharply peaked at ~20% into the arc. This is the quantity an "is it burning where it's informative" metric must correlate against.

**Direction** (1 m/s at 25% of arc, angle from instantaneous LOS): at A1 (5 km, 6 h) gain ranges **740× (270°) to 3.5e4 (210°)** — a 48× spread, but **even the worst direction is transformative**. At A2 (10 km, 6 h): 5.9× to 97.8×. At A3 (100 km, 3 h): **0.85–1.25× — a 1 m/s burn changes nothing.**

**Correction to NAV-G's recommendation #2** ("Discrete-20's radial ±1 m/s already suffices"): it suffices in *magnitude*, but radial (90°/270° from LOS) is the **worst** direction at the multi-hour arcs the policy actually flies, and the along-track fine burns already in Discrete-16 (actions 12–15) are 1.3–4.7× better. Since even the worst direction gives ≥740× at 5 km, the practical recommendation stands — but do not describe radial ±1 m/s as "the observability action."

### 2.6 Navigation margin against the actual success boxes

`ext_angles_box_margin.py`, bearings-only CRLB, σ_β = 1 mrad @ 60 s, one burn at 20% of arc:

| ρ | arc | burn | σ_range | σ_vel | vs TB5 (5 km / 1 m/s) |
|---|---|---|---|---|---|
| 5 km | 180 min | none | 937 m | **1.06 m/s** | **velocity exceeds tolerance** |
| 5 km | 180 min | 1 m/s prograde | 9.1 m | 0.0087 m/s | 115× margin |
| 5 km | 300 min | none | 428 m | 0.485 m/s | marginal |
| 30 km | 180 min | none | 5681 m | 6.4 m/s | vs 30 km/50 m/s: 5–8× margin ✓ |

**Note the binding constraint is relative velocity, not range** — the classical literature focuses on range observability, but a rendezvous box with a velocity tolerance binds on σ_vel first.

---

## 3. The RL campaign: falsifiable design

### 3.1 Baseline behaviour (measured, `ext_angles_policy_actions.py`, 30 greedy episodes each)

**T3 headline ckpt (30 km/50 m/s):** **zero** decisions below 10 km; closest approach p50 58 km. Nav error is 1.8% of tolerance. **The hypothesis is untestable here.**

**TB5 ckpt (5 km/1 m/s, Discrete-20):** 27.9% of decisions and **31.1% of sim time** below 10 km; median dwell **301 min**; median longest burn-free gap inside 10 km **180 min** (p90 300); closest approach p50 2.66 km. Action mix inside 10 km:

| action | | share |
|---|---|---|
| 11 warp-1 hr | | **66.8%** |
| 14/15 prograde/retrograde 2 m/s | | 28.7% |
| 10 warp-30 min | | 1.9% |
| 7/8 radial ±10 m/s | | 1.6% |
| **18/19 radial ±1 m/s** | | **0.0%** |

Two things follow. (i) The policy already burns ~29% of decisions inside 10 km for guidance reasons, so **presence-of-burn is not a valid metric** — only placement, timing, direction, and counterfactual information are. (ii) There is unambiguous behavioural headroom: actions 18/19 are never used, and 66.8% of terminal decisions are hour-long no-burn warps.

### 3.2 Arms (one variable at a time, common warm start, all evaluated under bearings-only)

| arm | trained on | tests |
|---|---|---|
| **T-truth** | truth state (= existing TB5 ckpt) | control; also gives the zero-shot floor |
| **T-RB** | range+bearing EKF estimates | **essential control** — range is measured, so a burn buys *no* information; if T-RB ≡ T-BO the effect is not observability |
| **T-BO** | bearings-only estimates | the treatment |
| **T-BO+Σ** | bearings-only + covariance channel in obs | see §3.4 |
| **T-BO−act** | bearings-only, burn actions masked inside 10 km | necessity |

Environment: TB4 (5 km/2 m/s) and TB5 (5 km/1 m/s), LEO, Discrete-20, sensor decoupled from guidance rate (`--sensor-dt 60`, per T3's own finding that welding nav to decisions starves the filter through 1-hr warps).

### 3.3 Detection metrics (ranked)

1. **Counterfactual information gain per burn (primary).** At each decision epoch, replay the filter forward under (a) the action taken and (b) coast, recording ΔσF_range at episode end. Regress burn probability on that difference. **Positive significant coefficient for T-BO and null for T-truth is the cleanest possible detection.** The CRLB machinery for this exists (`ext_angles_burn_placement.py`).
2. **Information-weighted placement in matched states.** Bin by (ρ, δa, time-to-go, Δv remaining); test whether the distribution of `G = (σ_drift/σ_burn)²` over T-BO's burns stochastically dominates T-truth's. Predicted signature: shift of burn epochs toward the ~20%-into-arc peak, and shortening of the 180-min burn-free gaps inside 10 km.
3. **Direction signature.** Angular distribution of burns relative to instantaneous LOS. Interpretable single plot; expect a shift out of the radial (worst, 90°/270°) quadrant.
4. **Excess Δv, paired episodes.** Predicted effect is **small**: 1 m/s suffices, against a 235 m/s median mission and a 478 m/s budget. Expect 1–5 m/s (0.4–2%) excess, concentrated at ρ<10 km. A *large*, mission-wide excess indicates degraded guidance, not observability-seeking. **Power note: needs ≥200 paired episodes; do not run this underpowered.**
5. **Filter covariance response.** Δtr(P) over K updates following each burn vs matched no-burn windows.
6. **Performance.** Success under bearings-only for every arm, plus T-truth zero-shot. If T-truth zero-shot ≈ T-BO, training bought nothing.

### 3.4 Design facts the campaign must not get wrong

- **Run at TB4/TB5, never the T3 headline box** (§3.1). This is the single highest-risk design error available.
- **OBS_DIM = 38 has no uncertainty channel.** The agent must infer its own nav uncertainty from estimate jitter through the LSTM — a weak, indirect signal. Include the **T-BO+Σ** arm (append σ_range/ρ or tr P) or the null result will be uninterpretable.
- **Observability is nearly free in this env.** No fuel term in the reward; median mission leaves 243 m/s spare; the shaping Δv_match term measures *remaining* Δv, not *spent*. So if the behaviour does not emerge, it is **not** because it was too expensive — which makes a null result informative.
- **`cap_terminal_reward = 0` makes timing out free.** A bearings-only agent may learn to stall rather than risk a bad capture. Monitor the timeout rate as a pre-registered confound.
- **Acquisition is a separate phase** (NAV-G: 44–114 min, 0/144 failures). Log acquisition-phase burns separately or they will contaminate the steady-state statistic.
- **Reward-term guidance from the only two prior RL works.** Wijayatunga/Armellin/Holt (arXiv:2411.01021) use `η = ȳ_balᵀȳ` — one extra ballistic propagation, O(1), viable in a 1M-step/s C loop — as a **one-sided penalty against a nominal, never a reward**. Yuan & Li (Acta Astronautica 129, 2022, 107812, DOI 10.1016/j.ast.2022.107812) use FIM det / covariance trace and get 2.4× the Δv and 35× the nav error. **Do not use naive per-step `log det(FIM)`**: for bearings-only relative nav the velocity columns of H are identically zero, so those blocks are permanently zero and log det is −inf. Use the prior-regularised cumulative form, or E-optimality `−1/(λ_min+c)`, which targets the least-observable direction — which here *is* range.

### 3.5 Honest prior on the outcome

Positive at TB5, small in Δv terms, visible as **timing and gap-shortening rather than more burns or new actions**. The mechanism is real and quantified (drift-only σ_vel = 1.06 m/s vs a 1 m/s tolerance; one 1 m/s burn buys ×1.4e4), the actuation exists, and the fuel cost is negligible. The risk is not that the effect is absent but that it is **too small to resolve against a policy that already burns 29% of the time inside 10 km** — which is exactly why metrics 1–3 (counterfactual, placement, direction) matter more than metric 4.

---

## 4. Corrections to sibling recon (NAV-G, `reports/nav_G_filter.md`)

1. **"At ρ ≲ 50 km a purely drifting chaser has an exactly singular Fisher information for range" is wrong as a structural claim.** At the same geometry and arc (5 km, 1 orbit) I measure σ/ρ = 0.747 with FIM condition **4.1e8**; their pipeline reports singular with cond 9.9e15. Diagnosis: their FIM chains a numerically differenced STM across the arc; mine differences the whole-arc map directly. **The two pipelines agree on the well-conditioned range+bearing problem (4.9 vs 4.7 m; 15.8 vs 16.5 m) and diverge only where conditioning is marginal** — the signature of a numerical artifact, not a physics disagreement. It matters: "exactly singular" implies no arc can fix it, whereas **3.1 orbits of pure drift does** (§2.2).
2. **Radial ±1 m/s is the worst-directed 1 m/s burn** at multi-hour arcs, not the canonical observability maneuver (§2.5). Their conclusion (no new action needed) survives; the rationale should change.
3. Their crossover estimate ("≈1000 km separation") is close; I measure **300–500 km** with a monotone sweep.

## 5. Limitations of this work

- **Dwell statistics are sampled at decision epochs**, with a warp's full τ·60 s attributed to the bin at its *start*. With 66.8% of terminal decisions being 1-hr warps this biases the bin assignment; the campaign should re-measure at sub-step resolution. (It also explains why the T3 policy's minimum *sampled* separation is 58 km while captures occur at ~29 km — capture fires mid-warp.)
- Profile-likelihood `D(k)` at ρ/r > 1 is optimiser-limited and non-monotone in k (multimodal likelihood); it is reliable as corroboration there, not as a precise number. The CRLB is optimiser-free and is the primary metric.
- Single geometry family per separation; §2.3's 100 km resonance shows observability is not a pure function of ρ. A geometry-randomised sweep would firm up the crossover.
- 2D coplanar sits exactly on the classical IOD coplanar singularity `D₀ = L₁·(L₂×L₃) ≡ 0` (Miller & Frueh, 8th European Conf. on Space Debris, ESA 2021), so three-observation Gauss/Laplace is structurally unsolvable; ≥4, realistically 6–12 observations are needed. Expect a and e to recover cleanly and in-track phase to be the weak direction — the same object as δλ in the ROE literature.
- No J2/drag: in real LEO J2 is the dominant ambiguity-breaker. **Our two-body sim is harder than reality here** — a defensible framing, but state it.
- Literature: I verified citation metadata via search and abstracts; the literature agent obtained fuller text for several sources. **Neither of us obtained Woffinden's or Grzymisch's actual equations** (IEEE/AIAA paywalls, and Grzymisch's dissertation is Shaker-Verlag commercial only). The criterion statements above come from abstracts plus three independent secondary restatements that agree. The mechanism behind Grzymisch's departure/approach asymmetry was not retrieved by either of us.

## 6. Artifacts

Scripts, `/Users/pete/space_training/scripts/orbital/ext_recon/` (all new; no existing file modified, nothing committed by me — the orchestrator's own `git add` swept several into commits `359edfc`/`6968919`):
`ext_angles_scale_ambiguity.py` · `ext_angles_observability_profile.py` · `ext_angles_burn_placement.py` · `ext_angles_box_margin.py` · `ext_angles_crossover.py` · `ext_angles_da_dependence.py` · `ext_angles_policy_actions.py`

CSVs, `/Users/pete/space_training/web_data/results/`:
`ext_angles_scale_ambiguity.csv` · `ext_angles_dbeta_trace.csv` · `ext_angles_burn_threshold.csv` · `ext_angles_profile.csv` · `ext_angles_crlb.csv` · `ext_angles_dv_sizing.csv` · `ext_angles_burn_geometry.csv` · `ext_angles_box_margin.csv` · `ext_angles_crossover.csv` · `ext_angles_da_dependence.csv` · `ext_angles_box_vs_observability.csv` · `ext_angles_policy_actions.csv` · `ext_angles_policy_dwell.csv`

Console logs: `ext_angles_profile_run.txt`, `ext_angles_scale_run_quick.txt`. Every script is standalone (`python3 <name>.py`, `--quick` where supported); the policy probe runs from `/Users/pete/space_training/pufferlib`.