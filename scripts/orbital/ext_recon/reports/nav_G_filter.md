## NAV-G — Bearings-only filter design: **GO** (measured)

**Verdict:** angles-only relative navigation is viable for this env at every shipped envelope. The recommended filter delivers **17.6 m / 0.019 m/s** at the terminal-box geometry vs the range+bearing baseline's 5.7 m / 0.006 m/s — a 3.1× accuracy cost, and 284×/52× *inside* the tight 5 km/1 m/s success box. Zero divergences and zero acquisition failures in 144 runs. The one structural risk is real and quantified: **at ρ ≲ 50 km a purely drifting chaser has an exactly singular Fisher information for range** — a single 1 m/s burn (already in the Discrete-20 action set) removes it.

---

### 1. Which pathologies actually transfer, and which filter

**The classic theorems do not apply as stated, and this is the load-bearing fact.** Nardone–Aidala (1981) unobservability is about constant-velocity target + observer: `x_rel → k·x_rel` is dynamically closed. Woffinden–Geller (2007) / Grzymisch–Fichter (2014) is the same statement for *linearized* relative motion (CW/YA are linear + homogeneous ⇒ scaling is a symmetry). Our filter state is the **target's absolute state on a Keplerian orbit about a known center, seen from a known absolute chaser position**. Scaling the relative vector changes the target's radius, hence its mean motion (n ∝ a^−3/2), hence the bearing history. The symmetry is broken by gravity's nonlinearity at strength O(ρ/r). So the question is conditioning, not observability — measured filter-independently by CRLB below.

**Measured comparison** (`ext_bo_filter.py`, 24 noise seeds/cell, σ_β = 1 mrad, 60 s cadence, truth from `orbital_math`; settled = last quarter of arc, median over seeds; `-o` = oracle epoch range, isolating steady state from acquisition):

| geometry (ρ̄, ρ/r) | RB-EKF | BO-EKF blind | BO-UKF blind | BO-EKF-o | BO-MPC-o | BO-RPB | **BO-BLS-MPC** |
|---|---|---|---|---|---|---|---|
| G6 5 km box, 1 m/s burns (15 km, .002) | 5.7 m | 40.5 m (NEES 9.5) | 340 km, div .96 | 16.8 m | 17.2 m | 18.5 m | **17.6 m, NEES 0.57/0.98** |
| G5 10 km, 5 m/s burns (80 km, .012) | 6.3 m | 704 m (NEES 155) | 350 km | 53.7 m | 59.8 m | 49.8 m | **56.6 m, 0.50/0.99** |
| G1 10 km drift (38 km, .006) | 6.2 m | 23.6 km | 302 km | 1018 m | 1264 m | 1075 m | **1226 m, 0.78/0.91** |
| G2 300 km drift (858 km, .127) | 8.2 m | 1256 km | 338 km | 3415 m | 2215 m | 37.0 km | **1229 m, 0.60/0.97** |
| G3 180° gap (13 566 km, 2.0) | 415 m | 13 547 km | 2218 km, div .38 | 5173 m | 8500 m | 4015 m | **3257 m, 0.59/0.97** |
| G4 wide e=0.30 (9771 km, .84) | 6.1 m | 7124 km | 6570 km | 2759 km | 158 km | 8333 km | **4503 m, 0.75/0.98** |

Read-outs:
- **Plain EKF, blind: fails everywhere.** Not "range collapse" — outright confidently-wrong convergence (NEES 10⁴–10¹⁴). Only survives when the chaser burns, and even then is inconsistent.
- **UKF is strictly worse than the EKF**, everywhere. A decades-wide range prior makes sigma points straddle the Earth and go hyperbolic; propagating the exact flow through them is meaningless. Do not use.
- **Modified polar (Aidala–Hammel) confirms its claim in our dynamics**: same accuracy as Cartesian EKF, better NEES at every geometry, and **17.5× better at the wide-eccentric envelope** (158 km vs 2759 km). Its update is exactly linear (H = e₁), which is what kills the premature-collapse mode. It does *not* solve initialization: its ln ρ / β̇ / ρ̇/ρ priors are only sane once range is bracketed.
- **Range-parameterized bank works at LEO, fails at e ≥ 0.2 and mid-field.** Cause is diagnosed, not mysterious: each component carries a *circular* velocity prior, wrong by ≈ v_c·e ≈ 1730 m/s at e = 0.30 — outside every component's linearization basin. It would need a 2-D (range × velocity) lattice.
- **Recommendation: `BO-BLS-MPC`** — batch angles-only acquisition handed to a modified-polar recursive filter. Only blind arm that works at every geometry with a consistent covariance (NEES in-bounds 0.91–0.99), and it *beats* the oracle-initialized recursive filters at G2/G3/G4 because it re-linearizes, which no one-pass filter can.

### 2. Observability map — the design table (`ext_bo_observability.py`, CRLB, filter-independent)

σ of the epoch range along the LOS, LEO 400 km, σ_β = 1 mrad, 60 s cadence:

| separation | arc | Δv=0 | 1 m/s | 5 m/s | 25 m/s | (range+bearing ref) |
|---|---|---|---|---|---|---|
| 5 km | 1 orbit | **∞ (singular FIM)** | 9.1 m | 5.2 m | 4.5 m | 4.7 m |
| 10 km | 1 orbit | **∞** | 27.1 m | 12.7 m | 9.1 m | 7.3 m |
| 50 km | 1 orbit | 222 551 m | 527 m | 142 m | 65 m | 16.5 m |
| 200 km | 1 orbit | 58 557 m | 8 495 m | 2 053 m | 589 m | 16.5 m |
| 5 000 km | 1 orbit | 7 232 m | 7 240 m | — | 7 414 m | 16.5 m |
| 13 000 km | 1 orbit | 4 722 m | 4 724 m | — | 4 759 m | 16.5 m |

Three decisions fall out:
1. **The observability maneuver is worth 4–6 orders of magnitude at close range and nothing at far range** (crossover ≈ 1000 km separation). At ρ ≥ 5000 km drift-only angles-only is already good — the 180° phase-gap regime needs no special handling.
2. **1 m/s captures most of the benefit** (10 km/1 orbit: 1→27.1 m, 5→12.7 m, 25→9.1 m). **Do not add an observability-maneuver action; Discrete-20's radial ±1 m/s already suffices.**
3. Eccentricity *helps* far-field and is irrelevant close-in once a burn exists (a=12 000 km, 10 km sep, 25 m/s: 5.4 m at e=0, 4.6 m at e=0.3, 6.7 m at e=0.5).

**Sensor-spec sensitivity** (`ext_bo_noise_sweep.csv`, BO-BLS-MPC, 12 seeds) — error is exactly linear in σ_β while RB-EKF is flat (range-dominated):

| σ_β | 0.1 mrad | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|
| G6 (5 km box) | 1.8 m | 17.6 m | 53.3 m | 181.9 m | 755 m |
| G5 (10 km) | 5.2 m | 52.4 m | 157 m | 522 m | 1560 m |
| NEES in-bounds | 0.98 | 0.98 | 0.98 | 0.98 | 0.97–1.00 |

The tight 5 km/1 m/s box survives to ~30 mrad on position; velocity (1.74 m/s at 30 mrad, 10 km) is the first thing to break. At 0.1 mrad **angles-only beats range+bearing** (1.8 vs 2.0 m) — the 50 m range sensor is the limiting element at close range.

### 3. Initialization without range — spec (implemented, `range_prior_intervals`)

1. **Analytic prior from the scenario distribution, not a guess.** The target's radius is bounded to an annulus [r_min, r_max] by the altitude band × eccentricity. The bearing is a ray from the known chaser position; `|r_c + ρû|² = R²` is a quadratic, so the feasible range set is exact and closed-form. Measured spans: 0.1–4055 km (LEO, 40 547×), 0.1–26 660 km (wide, 266 597×).
2. **The prior is bimodal whenever the ray cuts the inner circle** (near-side / far-side branches) — 2 intervals in 4 of 6 geometries. This is the structural reason a single Gaussian is the wrong object, independent of any nonlinearity argument.
3. **Multi-hypothesis over range only.** Every other direction of the cost is near-quadratic. Dense log grid (ratio 1.15, ≤160 nodes/interval) × a coarse 3×3 velocity lattice (tangential 0.8/1.0/1.2 v_c, radial ±0.3 v_c — required at e ≥ 0.2), scored by forward propagation alone (no STM, no linearization).
4. **Binned multi-start, not top-k.** Refine the cheapest node in each of 8 equal log-range bins. A globally-cheapest top-k selection systematically misses the true basin — a wrong-range node with a lucky velocity out-scores the true range with a circular guess. This exact failure produced confidently-wrong 3800 km solutions at 10 km true separation.
5. **Three acquisition gates, arc grown ×1.6 with warm start until all pass**: χ² on the residual (cost ≤ dof + 3√(2·dof)) catches the wrong basin; **ambiguity margin** (best cost at a range differing by >1.2× must be ≥16 worse) catches genuine short-arc multi-modality that no covariance test can see; σ_LOS/ρ ≤ 0.20 catches "right basin, too early." Covariance handed off is 4× inv(N), since the CRLB is optimistic at finite noise.
6. **Measured acquisition latency: 44 min (45 obs) with burns, 71–114 min drift-only. 0/144 acquisition failures.** Against 3000–12 000 sub-step episodes (50–200 h) that is 0.4–4% of the clock.

### 4. Integration plan for ext-nav

- **Drop-in surface.** `BearingMPC`/`BearingEKF` expose `predict/update/mean_cov`; `eval_relnav.run()` needs only the update-signature change (drop `rho`) and the acquisition preamble. `SENSOR_DT` cadence machinery, `build_obs`/`build_obs_t3` re-encode, and the hyperbolic-estimate guards all carry over unchanged.
- **Blind window must be handled explicitly.** Recommend holding a coast/warp until the acquisition gate passes, plus a "nav-valid" observation bit if the policy is retrained. Feeding the policy an unacquired mixture mean risks an early wrong burn.
- **The interesting coupling for the RL half:** at ρ ≲ 50 km, coasting makes range unobservable. A warp-heavy policy (96.8% warp-5min historically) will blind itself. Training *on the estimated state* is exactly the mechanism that should teach the 1 m/s observability burn — that is the headline result to go after, and the actuation for it already exists.

### 5. Caveats to carry forward

- Truth here is the oracle, not the C env. T2/T4 already established oracle↔env agreement at 0.29 m over live steps, but the closed-loop number must be re-measured through `eval_relnav.py`.
- **No J2, no drag.** Real angles-only relative nav leans on J2 to break the scale symmetry; we have none, so our observability comes purely from the nonlinear gravity gradient plus chaser maneuvers. Our problem is *harder* than reality here — defensible framing, but state it.
- No FOV/occultation/pointing model, no measurement association, single target. A FOV constraint is the obvious next realism step and would bite acquisition first.
- All arms share q_a = 1e-13 for fair comparison (shipped LEO default is 1e-11); the RB-EKF baseline column is therefore not bit-identical to the T2 headline.
- 2D favors us mildly on dof accounting (4 bearings is the algebraic minimum, the planar analogue of Gauss's 3-observation method). Porting to ext-3d should *improve* observability — the LOS rays through a known observer determine the orbit plane strongly — but that needs its own probe.

### Files

| path | contents |
|---|---|
| `/Users/pete/space_training/scripts/orbital/ext_recon/ext_bo_filter.py` | prototype: 8 filters, analytic range prior, batch acquisition, CRLB, 6 geometries. `python3 ext_bo_filter.py [--quick\|--seeds N\|--scenarios G5,G6\|--sigma-scale X]` |
| `…/ext_bo_observability.py` | CRLB design map over separation × arc × Δv × eccentricity |
| `…/ext_bo_report.py` | compact table renderer for the results CSV |
| `…/ext_bo_full_run.txt`, `…/ext_bo_observability_run.txt` | captured console output of the runs quoted above |
| `/Users/pete/space_training/web_data/results/ext_bo_filter.csv` | 48 rows: 6 geometries × 8 filters, 24 seeds each |
| `…/ext_bo_convergence.csv` | 3414 rows: per-step error traces (median seed) |
| `…/ext_bo_observability.csv` | 158 rows: CRLB map |
| `…/ext_bo_noise_sweep.csv` | 30 rows: σ_β ∈ {0.1, 1, 3, 10, 30} mrad |

No existing file modified; nothing committed.