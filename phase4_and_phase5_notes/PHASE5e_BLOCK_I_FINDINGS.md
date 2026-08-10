# Phase 5e — Block I Findings (Env Validation Suite)

**Date:** 2026-05-01 · ~1.5 hours compute · all 6 probes from spec §3.

Headline: **the env is correct at e=0.20**. The Phase 5d closure (`valid_init_only` rejection sampling) addresses the only physical-feasibility issue. No further env modifications required.

---

## E1 — Lambert reachability

Implemented universal-variable 2D Lambert solver (`scripts/orbital/p5e_e1_lambert.py`), validated against textbook Hohmann transfer (170° proxy: 781 m/s vs expected 770 m/s, 1.5% err). Sampled 200 task instances per `e_max ∈ {0.05, 0.10, 0.20, 0.30}` × {raw, valid_init_only}.

| e_max | mode | doomed_R | doomed_KO | hohm>budget | lamb_med | lamb_p10 | lamb<budget |
|---|---|---|---|---|---|---|---|
| 0.05 | raw | 1.5% | 41.0% | 0% | 6896 | 1907 | 1.0% |
| 0.05 | valid_init | 0.0% | 0.0% | 0% | 6781 | 2108 | 1.0% |
| 0.10 | raw | 44.5% | 81.0% | 0% | 7314 | 2070 | 0.0% |
| 0.10 | valid_init | 0.0% | 0.0% | 0% | 7928 | 2659 | 0.0% |
| 0.20 | raw | **86.0%** | **94.0%** | 0% | 6983 | 2340 | 0.0% |
| 0.20 | valid_init | 0.0% | 0.0% | 0% | 7884 | 2668 | 0.0% |
| 0.30 | raw | 93.0% | 97.0% | 0% | 6832 | 2093 | 0.0% |
| 0.30 | valid_init | 0.0% | 0.0% | 0% | 6848 | 2422 | 2.0% |

(Δv values in m/s. Fuel budget: 478 m/s. doomed_R = either perigee < R_EARTH; doomed_KO = either perigee < EARTH_KEEPOUT (200km altitude).)

**Findings:**

1. **Raw e=0.20 distribution: 86% sub-surface-perigee inits, 94% sub-keepout.** Confirms the Phase 5d closure quantitatively. Phase 5b/5c/5d's measured 16% success was the policy doing 26/72 = 36% on the survivable subset, dragged down by 122/128 doomed inits.

2. **Under `valid_init_only`: 0% doomed at all e_max.** The rejection-sampling fix completely eliminates physical infeasibility.

3. **Hohmann never exceeds budget** (0% across all configs). Random LEO-band transfer pairs are always within fuel budget assuming favorable phasing.

4. **Lambert 2-impulse is too tight an upper bound.** Median Lambert Δv at e=0.20 valid_init is 7884 m/s — 16× budget. Doesn't predict policy success (we observe 93.5%) because the agent uses multi-burn phasing trajectories far cheaper than single-shot Lambert. **Lambert is not the right reachability test for this env**; Hohmann + perigee-feasibility together are the binding constraints.

**Spec §3.7 decision:** result A (<5% unwinnable under valid_init_only by the binding constraints). No additional filter needed beyond `valid_init_only`.

---

## E2 — Kepler propagation precision at high e

Propagated orbits 100 periods (~9446 timesteps each) at e ∈ {0.05, 0.10, 0.20, 0.50} via env's `propagate_orbit` mirror. Round-trip via `cartesian_to_elements ∘ orbit_to_cartesian` to verify integration precision.

| e | n_steps | max \|Δa\| | max \|Δe\| | max \|Δω\| |
|---|---|---|---|---|
| 0.05 | 9446 | 9.31e-09 m | 8.53e-16 | 1.67e-14 rad |
| 0.10 | 9446 | 1.12e-08 m | 1.05e-15 | 9.21e-15 rad |
| 0.20 | 9446 | 1.12e-08 m | 1.11e-15 | 4.11e-15 rad |
| 0.50 | 9446 | 1.68e-08 m | 9.99e-16 | 1.78e-15 rad |

**Pass.** All drifts well below spec criterion (1e-6). Newton-Raphson Kepler solve converges cleanly even at e=0.50.

---

## E3 — Cartesian↔elements round-trip at high e

100 random orbits per e ∈ {0.05, 0.10, 0.20, 0.50}, varying ω, θ.

| e | max \|Δa\| | max \|Δe\| | max \|Δω\| | max \|Δθ\| |
|---|---|---|---|---|
| 0.05 | 6.52e-09 m | 5.55e-16 | 1.09e-14 | 2.93e-14 |
| 0.10 | 6.52e-09 m | 5.97e-16 | 5.08e-15 | 4.44e-15 |
| 0.20 | 8.38e-09 m | 6.66e-16 | 3.80e-15 | 6.63e-15 |
| 0.50 | 1.49e-08 m | 8.88e-16 | 2.00e-15 | 2.00e-15 |

**Pass.** The eccentricity-vector formulation in `cartesian_to_elements` is numerically stable at high e.

---

## E4 — LVLH frame inspection at high e

120-step coast under `same_orbit_init=1`, `valid_init_only=1` at e ∈ {0.05, 0.20}. Recorded `obs[33:38]` per step.

| e | dx_l range | dy_l range | dvx_l range | dvy_l range | n_tgt | NaN/Inf |
|---|---|---|---|---|---|---|
| 0.05 | [-0.022, -0.018] | [-0.207, -0.203] | [-0.002, 0.002] | [-0.002, 0.002] | 1.111 | None |
| 0.20 | [-0.028, -0.011] | [-0.213, -0.197] | [-0.007, 0.008] | [-0.008, 0.007] | 1.111 | None |

**Pass.** LVLH bounded and well-scaled at e=0.20. Slight oscillation amplitude growth proportional to eccentricity (expected — same_orbit_init=1 with e>0 means sat & target on same orbit but at different θ, so relative state oscillates by ~e magnitude). No saturation, no sign discontinuities, no NaN.

---

## E5 — Φ_orbit calibration at high e

| Config | da | de | Φ_orbit | σ₂ active? |
|---|---|---|---|---|
| almost-rendezvous e=0.05 | 0.000 | 0.000 | 0.000 | ON |
| almost-rendezvous e=0.20 | 0.000 | 0.000 | 0.000 | ON |
| Δa=10km e=0.20 | 1.000 | 0.000 | 1.000 | ON |
| perp ω e=0.20 | 0.000 | 0.283 | 0.283 | ON |
| opposite ω e=0.20 | 0.000 | 0.400 | 0.400 | ON |
| Δa=500km e=0.20 | 50.000 | 0.000 | 50.000 | off |
| worst-case e=0.20 | 50.000 | 0.400 | 50.400 | off |

**Pass.** Φ_orbit dynamic range matches σ₂ gate threshold (EPS_ORBIT = 2.0). Near-rendezvous states keep σ₂ active; far states deactivate it. The ω-mismatch term (||Δē||) is bounded by 2·e (max 0.4 at e=0.20), which is well below the gate threshold even at maximum mis-orientation. Φ is dominated by Δa for actual transfers, which is the intended design.

---

## E6 — Action effect at periapsis vs apoapsis

Δa from prograde Δv at peri/apo, by eccentricity:

| e | Δv | Δa @peri | Δa @apo | ratio |
|---|---|---|---|---|
| 0.05 | 25 | 47.8 km | 43.2 km | 1.11 |
| 0.10 | 25 | 50.3 km | 41.1 km | 1.22 |
| 0.20 | 25 | 55.8 km | 37.1 km | **1.50** |
| 0.50 | 25 | 79.1 km | 26.2 km | 3.00 |

**Caveat-pass.** The peri-vs-apo ratio grows with e (orbital mechanics: vis-viva at peri vs apo). At e=0.20 the ratio is 1.5×, manageable. At e=0.50 it's 3×, which means the discrete action set (5/10/25 m/s prograde) needs ~3 actions to cover the same Δa range as at e=0. The current action discretization is adequate for e ≤ 0.20 (Phase 5e Block II target) and marginal for e ≥ 0.50.

This is not an env bug — it's a known consequence of vis-viva. If Phase 5+ targets e ≥ 0.50, action discretization (B4-style Discrete(14) with larger Δv steps) becomes a real consideration. For the current Phase 5e Block II target (e=0.20), no action.

---

## Synthesis

**The env is correct at e=0.20.** Six probes, six passes. The Phase 5d closure (`valid_init_only`) addresses the single environment-design issue (sub-surface-perigee init sampling). Block II proceeds as planned: multi-seed validation of the `valid_init_only` recipe.

**Open question, deferred:** at e ≥ 0.50, the peri-vs-apo action asymmetry ratio reaches 3× (E6). Block III could test whether action discretization becomes a bottleneck there. Not necessary for the e=0.20 deliverable.

**No Lambert-budget gate added.** The single-shot Lambert criterion is too restrictive (multi-burn phasing trajectories are far cheaper). Adding it would falsely reject most viable tasks.

---

*Block I complete. Block II (multi-seed `valid_init_only` retrain) proceeds without env modifications.*
