# T2 — Relative Navigation (measurement model + EKF + policy on estimated state)

> **Status:** 2026-08-10. The canonical rendezvous policy now flies on an *estimated* target state instead of truth, closing the loop through a range/bearing EKF. **Success is unchanged at nominal and 3× sensor noise (97.5% → 97.5% → 97.5%), and drops 1 pp at 10× (96.5%).** Filter consistency passes: NEES in-bounds 91.7%, NIS 94.7% against 95% chi-square bounds. Entirely eval-side; the C environment is untouched.

---

## TL;DR

- **Headline (200 eps each, seed 42, greedy, e_max=0.05 sat+target, phase gap ±π, `valid_init_only=1`):**
  truth 97.5% · EKF @1× 97.5% · EKF @3× 97.5% · EKF @10× **96.5%**.
- **Filter consistency (100-episode open-loop MC, nominal noise):** NEES in-bounds **0.917** (4.4% below / 3.9% above — balanced, so no systematic over- or under-confidence), NIS in-bounds **0.947** against a theoretical 0.95.
- **Accuracy at nominal noise (closed loop, 200 eps):** median settled (step ≥ 10) position RMSE **53 m**, velocity RMSE **0.058 m/s**. Terminal error at rendezvous **≈12 m / 0.011 m/s**, against a 30 km / 50 m/s success box. The episode-mean RMSE (1130 m) is entirely the first ~15 steps of acquisition.
- **Port fidelity:** the Python reconstruction of the 38-dim observation matches the C environment's own to **1.2e-7** (float32 epsilon), and the `recon` control — same rebuild path, zero estimation error — reproduces the truth run **exactly** (195/200, identical RMSE to the last digit). The nav result is therefore attributable to estimation error and nothing else.
- **Breaking point is the filter, not the policy.** Degradation only becomes material at 30× (87.5%), 100× (74.0%), 300× (55.0%) — and at those levels the EKF itself is inconsistent (NEES in-bounds 0.54 / 0.31 / 0.12) and diverges to a hyperbolic estimate in 5.5% / 17% / 30.5% of episodes.

---

## Architecture

**Chaser-nav-known, target-tracked** — the standard RPO split. The chaser knows its own state (GPS/INS); everything about the target is estimated.

- **Sensor suite.** One measurement per `env.step()`: range `rho = |r_t - r_c|` at σ_ρ = 50 m, and bearing `beta = atan2(dy, dx)` of the line-of-sight **in the inertial frame** at σ_β = 1 mrad — i.e. what a star-tracker-registered optical sensor delivers with known chaser attitude. Inertial rather than LVLH-referenced because it keeps `H` free of chaser-velocity terms; with attitude known the two carry identical information.
- **Filter state.** `x = [x_t, y_t, vx_t, vy_t]`, the target's **absolute** inertial Cartesian state. The chaser position enters only through the measurement model, as a known quantity.
- **Dynamics.** Exact two-body Kepler propagation of the mean via Lagrange f & g coefficients solved in the eccentric-anomaly difference ΔE (`orbital_math.propagate_cartesian`). This is analytically the same map the C env applies to the target, verified to 7.3e-15 relative against the env's element-wise propagation and to 0.29 m against the live environment over 99 steps including warp actions. Covariance via a numerically differenced STM (central differences, h = 1 m / 1e-3 m/s), symplectic to 9.7e-7.
- **Timing.** Warp actions advance τ sub-steps of 60 s per `env.step()` (action 9 → τ=5, 10 → 30, 11 → 60). The harness reads τ from the action it just issued and propagates the filter by τ·60 s, so the measurement cadence tracks the agent's own time-warping.
- **Tuning.** Process noise is the continuous white-noise-acceleration model, `Q = q_a · [[Δt³/3 I, Δt²/2 I],[Δt²/2 I, Δt I]]`. Because the dynamics are exact there is *no model error for Q to absorb* — it is purely a covariance floor, and it was selected on NEES rather than guessed (table below). Initialization is single-measurement, no acquisition phase: position by inverting the measurement (so P₀'s position block is the exact `J R Jᵀ`), velocity by a prograde circular-orbit guess at the measured position, with σ_v0 = 100 m/s taken from the **measured** RMS of that guess's error (107.6 m/s) rather than tuned.
- **Closed loop.** The target-derived observation slots — `[7] a_t`, `[8] e_t`, `[11,12] sin/cos ω_t`, `[13,14] sin/cos(θ_sat−θ_t)`, `[15,16] sin/cos θ_t`, `[33-37]` the LVLH block — are recomputed from (chaser truth, target **estimate**) and handed to the policy. Chaser-only slots stay truth. The policy is the unmodified canonical checkpoint, argmax, LSTM state reset per episode.

### Process-noise selection (40 eps per row, NEES in-bounds is the criterion)

| q_a (m²/s³) | NEES in-bounds | below | above | median NEES | NIS in-bounds |
|---:|---:|---:|---:|---:|---:|
| 1e-13 | 0.902 | 0.031 | 0.067 | 0.947 | 0.947 |
| 1e-12 | 0.918 | 0.031 | 0.051 | 0.879 | 0.947 |
| **1e-11** | **0.919** | 0.038 | 0.043 | 0.813 | 0.947 |
| 1e-10 | 0.914 | 0.049 | 0.037 | 0.684 | 0.947 |
| 1e-9 | 0.912 | 0.071 | 0.017 | 0.548 | 0.948 |
| 1e-8 | 0.891 | 0.099 | 0.010 | 0.444 | 0.947 |

The sweep behaves exactly as a zero-model-error problem should: raising q_a walks the filter monotonically from slightly overconfident (6.7% of samples above the upper bound at 1e-13, only 3.1% below) to conservative (9.9% below and 1.0% above at 1e-8), with median NEES sliding from 0.95 to 0.44. **q_a = 1e-11** is the balance point — the tails are symmetric (3.8% / 4.3%) — and it was adopted. σ_v0 was swept over {100, 200, 300} m/s and moved NEES by < 0.001, since it only affects the first few steps.

---

## Validation (before any closed-loop run)

Port correctness, from `python3 scripts/orbital/nav/orbital_math.py`:

| Check | Result |
|---|---|
| elements → Cartesian → elements round trip (2000 draws, double) | a 3.3e-15, e 1.6e-12, ω 5.7e-13, θ 5.7e-13 (relative) |
| f&g Cartesian propagation vs env element propagation (e ∈ [0, 0.3], Δt ∈ {60, 1800, 3600} s) | 7.3e-15 relative position |
| STM symplecticity, max\|FᵀJF − J\| (e ∈ [1e-12, 1e-3]) | 9.7e-7 |
| STM linearization vs finite 100 m / 0.1 m/s probe | 1.2e-7 relative |
| **38-dim observation rebuilt from states recovered out of a truth obs**, 100 env steps | **max \|err\| 1.2e-7** (worst dim `[33]`) |
| Target propagation vs the **live** env, 99 steps, actions 0-11 incl. warps | 0.29 m position, 3.1e-4 m/s velocity |

The last two are the ones that matter. The observation residual sits at float32 epsilon, which is the quantization of the observation vector itself — the port has no error of its own. The live-env check confirms both the dynamics and the τ·60 s timing convention, and its 0.29 m residual is the same float32 quantization (≪ σ_ρ = 50 m).

Open-loop filter Monte Carlo — 100 episodes, 20 347 filter updates, mean episode 909 sim steps, policy driven by truth so the filter is validated on exactly the trajectory distribution it will later close the loop on:

| Metric | Value | Reference |
|---|---:|---|
| NEES in-bounds | **0.917** (below 0.044 / above 0.039) | 0.95 theoretical, χ²(4) bounds [0.121, 2.786] |
| NEES in-bounds, step ≥ 10 | 0.920 | — |
| NEES median | 0.724 | 1.0 |
| NIS in-bounds | **0.947** (below 0.026 / above 0.027) | 0.95 theoretical, χ²(2) bounds [0.025, 3.689] |
| NIS median | 0.705 | 1.0 |
| Position RMSE, settled (step ≥ 10) | 142 m (episode mean) | — |
| Velocity RMSE, settled | 0.159 m/s (episode mean) | — |
| Position RMSE, whole episode | 1140 m mean / 1025 m median | — |
| Terminal error at rendezvous | 11.9 m, 0.011 m/s | 30 km / 50 m/s success box |

NIS lands within 0.3 pp of its theoretical 95%, which says the innovation covariance `S` is right. NEES is 3 pp short of theory with symmetric tails, which is what a nonlinear filter with a coarse single-measurement init gives; the shortfall is not concentrated in the acquisition transient (in-bounds is 0.920 after step 10, essentially unchanged).

Plots in `plots/relnav/`: `nees.png`, `nis.png`, `rmse_vs_time.png`, `err_3sigma_ep{1,2,3}.png`, `success_vs_noise.png`. The error-vs-±3σ traces show the error inside the envelope throughout, with the σ envelope breathing once per orbit as the range geometry cycles the radial/along-track observability split.

---

## Headline: policy on estimated state

200 episodes per condition, identical initial-condition sequence across conditions (the env's RNG stream is consumed only by `c_reset`, so this is a paired comparison).

| Condition | σ_ρ | σ_β | Success | Settled pos RMSE (med) | Settled vel RMSE (med) | NEES in-bounds |
|---|---:|---:|---:|---:|---:|---:|
| truth (control) | — | — | **97.5%** (195/200) | 46 m | 0.053 m/s | — |
| recon (rebuild, zero est. error) | — | — | **97.5%** (195/200) | 46 m | 0.053 m/s | — |
| EKF nominal (1×) | 50 m | 1 mrad | **97.5%** (195/200) | 53 m | 0.058 m/s | 0.905 |
| EKF 3× | 150 m | 3 mrad | **97.5%** (195/200) | 147 m | 0.172 m/s | 0.890 |
| EKF 10× | 500 m | 10 mrad | **96.5%** (193/200) | 543 m | 0.573 m/s | 0.780 |

Extended degradation curve (beyond the requested set, to locate the actual failure point):

| Condition | Success | Settled pos RMSE (med) | Episodes with diverged estimate | NEES in-bounds |
|---|---:|---:|---:|---:|
| EKF 30× | 87.5% | 1.68 km | 5.5% | 0.542 |
| EKF 100× | 74.0% | 10.0 km | 17.0% | 0.314 |
| EKF 300× | 55.0% | 108 km | 30.5% | 0.117 |

Failure modes shift with noise. At ≤10× every non-success is a safety-cap timeout (5, 5, 5, 5, 7 across truth/recon/1×/3×/10×) and there are zero collisions. Collisions appear only once the filter starts diverging: 3 at 30×, 7 at 100×, 10 at 300×, alongside 21/42/72 safety caps.

## Interpretation

The policy is essentially insensitive to navigation error up to roughly 500 m of position uncertainty, and the reason is a scale mismatch: the success box is 30 km / 50 m/s, while the converged filter delivers 53 m / 0.06 m/s at nominal noise and 543 m / 0.57 m/s at 10×. Nav error only bites when it becomes an appreciable fraction of the terminal criterion, which is why the curve is flat to 3×, bends at 10×, and falls off from 30× onward. The rendezvous geometry helps enormously: bearing error maps to cross-range error as ρ·σ_β, so the sensor gets monotonically better as the chaser closes, and the filter's worst moment (5 km position error at ~10 000 km range) is exactly when the agent is doing coarse phasing and does not need precision. That the failure mode at ≤10× is *entirely* safety-cap timeouts with zero collisions says the degraded observation makes the agent slower to close the box, not more dangerous. Beyond 30× the limiting element is no longer the guidance policy at all but the estimator: NEES in-bounds collapses to 0.54 and the EKF is pushed to a hyperbolic target estimate in 5.5% of episodes, so the reported success rate there is measuring filter divergence, not policy competence. The honest headline is that guidance→GN&C costs this system nothing at a realistic sensor suite, and the first thing that would need work to fly a noisier sensor is the filter's initialization and divergence protection, not the policy.

## Limitations / future work

- **Chaser navigation is assumed perfect** (GPS/INS truth). A real chaser has its own state error, which is common-mode in the relative measurement and would need either an augmented filter state or a directly relative formulation.
- **No measurement dropouts or outliers.** Every step delivers a valid range and bearing. Real optical/radar tracking has occultation, eclipse-driven gaps, and false associations; a residual-gating (χ²) test and coasting behavior are unmodeled.
- **No burn execution error.** Impulses are applied exactly as commanded. Real thrusters have magnitude and pointing error, which is both a guidance disturbance and a genuine source of process noise — the one thing that would make a non-trivial `Q` physically meaningful here.
- **No sensor bias or misalignment.** Both channels are zero-mean; range bias and bearing boresight misalignment are the usual first-order error sources and are not in the state.
- **Single-measurement acquisition.** No dedicated acquisition arc, no batch initial-orbit determination, no divergence detection or filter reset. Adding any of these is the direct route to extending the flat part of the degradation curve past 10×.
- **The policy was never trained on estimated state.** These are zero-shot robustness numbers. Fine-tuning on filtered observations (or on an explicit uncertainty input) is untested and would likely recover part of the 30×/100× loss.
- **Two-body, coplanar, no debris**, matching the environment. Real relative navigation contends with J2, drag, and out-of-plane geometry that changes the observability structure.

## Files

| Path | Contents |
|---|---|
| `scripts/orbital/nav/orbital_math.py` | Port of the env's orbital math; f&g Cartesian propagator; STM; observation reconstruction; self-tests (`python3 orbital_math.py`) |
| `scripts/orbital/nav/ekf.py` | Range/bearing measurement model and the target-state EKF (Joseph form, exact-dynamics propagation) |
| `scripts/orbital/nav/eval_relnav.py` | Harness — stages `sanity` / `qsweep` / `validate` / `eval` / `all` |
| `web_data/results/relnav_results.csv` | Per-condition success, RMSE, NEES/NIS, divergence, terminal-cause counts |
| `plots/relnav/*.png` | NEES, NIS, RMSE-vs-time, error-vs-±3σ traces, success-vs-noise |

Reproduce end to end (~70 s, run from `pufferlib/`):

```
python3 ../scripts/orbital/nav/orbital_math.py                   # port validation
python3 ../scripts/orbital/nav/eval_relnav.py --stage all \
    --qsweep-eps 40 --validate-eps 100 --eval-eps 200
```

## Note on an environment detail found along the way

`true_to_mean()` in `orbital.h` applies `tan(E/2) = sqrt((1+e)/(1-e))·tan(θ/2)` — the E→θ map rather than its inverse. It is reached only from `cartesian_to_elements()`, i.e. after a burn, so it perturbs the *chaser's* mean anomaly but never the target's (the target's `M` is sampled at reset and only ever advanced by `n·dt`). This harness is unaffected: it reads the chaser state as truth every step and never propagates it, and the target — the only body the filter propagates — follows exact two-body motion, confirmed to 0.29 m against the live env. The port reproduces the C behavior verbatim rather than correcting it. Flagged here only because it is real, and because it would matter to anyone propagating the chaser off-line.
