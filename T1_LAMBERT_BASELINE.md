# T1 — Lambert two-impulse Δv baseline

Phase 5e canonical seed-42 checkpoint, 200 episodes (`/tmp/compat_check`), LEO 300-800 km, e ≤ 0.05 on both bodies, phase gap ±180°, `valid_init_only=1`. 195/200 episodes reached the 30 km / 50 m/s success box; every ratio below is over those only.

Replaces the circular-Hohmann surrogate in `bonus_stats.json` (median 2.49×), which ignores phasing, eccentricity matching, and the fact that rendezvous constrains position *and* velocity.

**Headline: the ratio distribution is not physically admissible.** Given the same mission clock it used, the policy sits at parity with the classical two-impulse optimum (median 1.08×) — but 45% of successful episodes come in *below* that optimum and 13% below half of it, which nothing can do against a fixed-time two-impulse solution. Running that down found a bug in the environment's burn path: `true_to_mean()` in `orbital.h` inverts the eccentric-anomaly half-angle map backwards, so every impulse teleports the satellite along-track by ≈ 2e·sin θ radians — median 24° of free phase shift per episode. Measurements below.

## Solver validation

Izzo/Lancaster (λ, x) formulation, 2D coplanar prograde, zero- and multi-revolution, bisection on x with a guarded near-parabolic branch. Chosen over the universal-variable form in `p5e_e1_lambert.py`, which is singular at Δν = π (A = 0) — exactly the Hohmann geometry validation case (a) requires.

| check | expected | solver | verdict |
|---|---|---|---|
| circular 6771 → 7171 km, Δν = π, TOF = Hohmann half-period | 217.02 m/s (vis-viva) | 217.02 m/s | err 0.00e+00% |
| circular 6771 → 42164 km (LEO→GEO), same geometry | 3856.7 m/s | 3856.7 m/s | err 2.36e-14% |
| round-trip: propagate v1 for TOF, compare to r2 (400 random LEO pairs, TOF 1200-9000 s) | 0 m | median 2.8e-08 m, max 3.9e-04 m | pass |
| same, multi-revolution N=1, both branches (800 solves) | 0 m | max 1.4e-02 m | pass |
| Kepler propagator vs. independent RK4 (h = 0.25 s) over 6000 s | 0 m | median 7.0e-07 m, max 5.2e-05 m | pass |
| Kepler propagator vs. logged target trajectory (20 episodes) | 0 m | 7.2 m worst over 100 steps; 29 m median over the full episode | pass — the full-episode residual is the 30 m along-track drift implied by float32 logging of the initial state |

## Comparators

Each minimizes total two-impulse Δv = |v_req(t_dep) − v_chaser(t_dep)| + |v_target(t_arr) − v_arr(t_arr)| by dense grid search plus four rounds of local refine, over departure wait, time of flight, and revolution count. The target is propagated two-body from its logged t0 state; the chaser coasts two-body until departure.

| comparator | departure wait | TOF | revs |
|---|---|---|---|
| IMMEDIATE | 0 (depart at t0) | 600 s … 3·P_target | 0-3 |
| FREE-DEPARTURE | 0 … 1.2·P_target | 600 s … 2·P_target | 0-1 |
| TIME-MATCHED | 0 … T_episode | 600 s … T_episode, wait + TOF ≤ T_episode | 0-20 |

T_episode is the wall-clock the policy itself consumed on that episode (median 9.1 target orbits, p90 17.6). TIME-MATCHED exists because the first two windows turned out to be the binding constraint, not the solver — see below.

## Δv, successful episodes (m/s)

| quantity | median | mean | p10 | p90 |
|---|---|---|---|---|
| policy | 235.0 | 235.9 | 157.0 | 320.0 |
| Lambert IMMEDIATE | 805.3 | 767.4 | 268.5 | 1257.8 |
| Lambert FREE-DEPARTURE | 710.0 | 722.4 | 236.5 | 1249.2 |
| Lambert TIME-MATCHED | 226.0 | 318.3 | 144.9 | 503.4 |
| circular-Hohmann surrogate | 87.9 | 99.4 | 42.4 | 171.8 |

## Ratio distribution (policy Δv / baseline Δv), successful episodes only

| comparator | n | median | mean | p10 | p90 |
|---|---|---|---|---|---|
| policy / Lambert IMMEDIATE | 195 | 0.31× | 0.43× | 0.17× | 0.84× |
| policy / Lambert FREE-DEPARTURE | 195 | 0.35× | 0.50× | 0.17× | 1.03× |
| policy / Lambert TIME-MATCHED | 195 | 1.08× | 1.03× | 0.41× | 1.55× |
| policy / circular-Hohmann surrogate (prior reference) | 195 | 2.51× | 3.19× | 1.21× | 6.49× |

Best free-departure wait: median 0 min, p90 65 min. Revolution count chosen by TIME-MATCHED: 0×48, 1×32, 2×11, 3×5, 4×6, 5×8, 6×10, 7×8, 8×12, 9×7, 10×7, 11×8, 12×9, 13×6, 14×2, 15×4, 16×3, 17×2, 18×6, 20×1 (revs×episodes).

The circular-Hohmann row reproduces `bonus_stats.json`'s 2.49× median (2.51× here) — an independent check that this episode set and Δv accounting line up with the project's existing numbers.

## Why so many ratios fall below 1.0×: the env's burn path teleports phase

The policy beats the time-matched classical optimum on 45% of successful episodes, and beats it by more than 2× on 13% (p10 of the ratio is 0.41×). More impulses can shave a few percent off a two-impulse schedule in some geometries; they cannot halve it. Neither can the 30 km / 50 m/s success box, which is only ~4 s of along-track slack at LEO speeds. So this was run down, and it is an environment bug, not a solver error.

`true_to_mean()` in `orbital.h` (~line 339) is the forward E → θ half-angle map, not its inverse — the √(1−e) and √(1+e) factors are attached to the wrong terms:

```c
/* orbital.h — true_to_mean(): θ → M */
double x = sqrt(1.0 - e) * cos(theta / 2.0);   /* should be sqrt(1+e) */
double y = sqrt(1.0 + e) * sin(theta / 2.0);   /* should be sqrt(1-e) */
double E = 2.0 * atan2(y, x);
return E - e * sin(E);
```

`cartesian_to_elements()` calls it to rebuild M after every impulse, and `apply_impulse()` calls that on every burn. The result is that each burn displaces the satellite **along-track** by ≈ 2e·sin θ radians — the impulse teleports it forward or backward along its own orbit, sign selectable by where in the orbit the agent burns. At e = 0.03 that is ±3.4° (±54 s of orbital position) per burn; at e = 0.10, ±11°. Coasting and warping are exact: `propagate_orbit()` advances M directly and never round-trips through θ.

Measured on the logged trajectories, per episode, as the mismatch between consecutive log rows and the n·Δt they should differ by on the post-burn orbit:

| | median | mean | p10 | p90 |
|---|---|---|---|---|
| burns per episode | 29 | 29 | 17 | 42 |
| net phase teleport (deg) | 24.5 | 28.9 | 5.7 | 63.6 |
| total \|teleport\| (deg) | 39.8 | 44.6 | 15.6 | 82.5 |
| same, summed over non-burn steps (deg) | 6.45e-04 | 6.25e-04 | — | — |

Non-burn steps are clean to 6e-04°, which isolates the defect to the impulse path. Across successful episodes the correlation between |net teleport| and policy Δv / time-matched Lambert Δv is -0.30: the episodes that teleport most are exactly the episodes where the policy most outperforms the classical optimum.

Phase matching is the expensive part of this rendezvous task — at 300-800 km the synodic period is 25-30 orbits, so the classical schedule must either wait it out or buy the phase with propellant. The bug lets the policy buy it with neither. This does not invalidate the learning result (the policy solved the environment it was given), but any claim about fuel efficiency against real orbital mechanics is unsupported until `true_to_mean()` is fixed and the checkpoints are re-trained and re-evaluated.

## Caveats

- **The comparison favors the policy.** The policy only has to close to the 30 km / 50 m/s success box; Lambert is held to exact rendezvous (0 m, 0 m/s), which additionally requires matching the target's eccentricity vector. The policy-equivalent classical cost is therefore *lower* than reported, and the true ratios are *higher*.
- The policy's Δv is quantized into 5 and 10 m/s impulses along four fixed directions (prograde, retrograde, radial in, radial out); Lambert uses two continuous impulses of any magnitude in any direction. Part of the gap is action-space granularity, not policy quality.
- Two impulses is not the unconstrained optimum. Three-impulse and bi-elliptic schedules beat Lambert for some geometries, so these baselines are upper bounds on the classical optimum and the ratios are correspondingly lower bounds.
- TIME-MATCHED uses each episode's own duration as the budget, so a policy that dawdles is handed a cheaper baseline to be measured against; it is a conservative construction, not a generous one.
- Grid-search minima with local refine, not certified global optima. The reported value is the best over all rev counts and both multi-rev branches.
- The baselines are computed under correct two-body dynamics; the policy was trained and evaluated under the buggy impulse path. The two sides of every ratio are therefore not playing the same game, which is the point of the section above.

## Interpretation

The Lambert baseline the project was missing now exists and is exact (1e-14 against analytic Hohmann geometry at Δν = π, sub-µm on round-trip), and its first use was to falsify the measurement it was built to make: 13% of successful episodes spend less than half the fixed-time two-impulse optimum, which is skill no policy has, and the residual traces to one inverted half-angle formula in `orbital.h:true_to_mean()` that hands the agent a free along-track jump of ≈ 2e·sin θ on every burn — median 24° of net phase per episode, on a task whose dominant cost *is* phasing. The defensible reading of the parity headline (median 1.08× the time-matched optimum, 2.51× the circular-Hohmann surrogate the project has been quoting) is that it is an upper bound on the classical cost and a lower bound on the policy's true expense, with an unknown amount of the gap donated by the bug rather than earned by the policy. The fix is two swapped `sqrt` factors; the cost is that every Δv number in Phases 3-5 was measured in a world where burns move you, so the checkpoints need re-training and re-evaluation before any fuel claim is portfolio-ready.

Per-episode data: `web_data/results/lambert_baseline.csv`. Generated by `scripts/orbital/t1_lambert_baseline.py`.
