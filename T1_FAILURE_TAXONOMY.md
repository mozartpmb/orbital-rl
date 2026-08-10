# T1 — Named Failure-Mode Taxonomy at the Headline Regime

> **Status:** 2026-08-10. 1100 greedy-eval episodes of the canonical checkpoint at the headline conditions. Aggregate 96.5% success; 37 safety-cap timeouts and 1 collision. The timeouts decompose into **three named modes** with a single shared mechanism: the policy's phasing strategy is a fixed ±200 km drift orbit, and every timeout is a way that strategy misses its one alignment window inside the 2000-step clock. **Not one of the 37 timeouts ever entered the 30 km position sphere** — no failure is a tolerance near-miss.

---

## TL;DR

| Mode | n | % of failures | % of all episodes | One-line signature |
|---|---:|---:|---:|---|
| **M1 — Drift-orbit lock** (transfer never initiated) | 24 | 64.9% | 2.18% | ≤3 burns, ≤30 m/s, `a` unchanged for 2000 steps; \|Δa_init\| ≥ 348 km in every case |
| **M2 — Window beyond the horizon** | 6 | 16.2% | 0.55% | Drift orbit correctly opened (\|Δa\| ≈ 186 km), then **zero** phase alignments before the cap |
| **M3 — Closure burst truncated** | 7 | 18.9% | 0.64% | 17–38 burns, closure burst live at the cap, closest approach 30–152 km and still converging |
| **M4 — Floor clip** (collision, not a timeout) | 1 | — | 0.09% | Canonical downward drift orbit opened from a 306 km start; perigee driven to −7 km |

Failure is almost entirely a function of one variable: **the initial semi-major-axis separation.** Below \|Δa\| = 340 km the cap-failure rate is 1.2% (12/1004). At \|Δa\| ≥ 340 km it is 26.0% (25/96), and 24 of those 25 are M1.

---

## Data

- **Checkpoint:** `pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt` (canonical Phase 5 policy), greedy argmax, `--legacy-action-space 10`.
- **Conditions:** LEO 300–800 km, `e_max_sat = e_max_target = 0.05`, `init_phase_gap_max = π`, `valid_init_only 1`, no debris. Success = dist < 30 km **and** rel-vel < 50 m/s vs the propagated target.
- **Episodes:** 1100 total across 4 rollout seeds.

| Source dir | seed | n | success | safety_cap | collision | success rate |
|---|---:|---:|---:|---:|---:|---:|
| `/tmp/compat_check` | 42 | 200 | 195 | 5 | 0 | 97.5% |
| `/tmp/taxonomy_s123` | 123 | 300 | 287 | 12 | 1 | 95.7% |
| `/tmp/taxonomy_s777` | 777 | 300 | 292 | 8 | 0 | 97.3% |
| `/tmp/taxonomy_s9001` | 9001 | 300 | 288 | 12 | 0 | 96.0% |
| **total** | | **1100** | **1062** | **37** | **1** | **96.5%** |

All per-episode metrics are in `web_data/results/failure_taxonomy.csv` (38 rows = 37 caps + 1 collision). Generator: `scripts/orbital/t1_failure_taxonomy.py`.

Units note: one trajectory record = one 60 s sim step. The cap is 2000 records (33.3 h). Agent decisions are fewer than records because action 9 (warp) expands to 5 sub-steps, each of which is logged and each of which is success-checked — so warp never skips over a success.

---

## The policy the failures are failing at

Before naming the modes: the success population shows the learned policy is a **three-phase phasing maneuver**, not a Hohmann transfer.

1. **Open a drift orbit.** A short early burn sequence (median ~7 burns, median first burn at 26% of episode length) pushes \|Δa\| to a canonical **196 km (IQR 137–242 km)**. The sign of the drift orbit matches the sign of the initial Δa in **92%** of successes — the policy pushes Δa *away* from zero, deliberately setting up a differential mean motion.
2. **Coast/warp.** The phase gap sweeps at a median **194°/1000 records** (IQR 152–247) under that Δa. The median success spends most of the episode on action 9 (warp).
3. **Terminal closure burst.** Median 25 burns inside the final 300 records (median burst start at 93% of episode length), simultaneously nulling Δa, Δe and relative velocity.

That gives roughly **one usable alignment window per episode**, and the whole failure taxonomy is about missing it.

**Success-population contrast stats (n = 1062):**

| Quantity | median | p5 | p95 | max |
|---|---:|---:|---:|---:|
| Episode length (records) | 809 (13.5 h) | 111 | 1762 | 1995 |
| Terminal distance (km) | 26.2 | 8.0 | 29.9 | 30.0 |
| Terminal rel-vel (m/s) | 28.6 | 8.1 | 48.7 | 50.0 |
| Total Δv (m/s) | 230 | 135 | 355 | 440 |
| Burn count | 28 | 16 | 47 | 62 |
| Fuel used (% of budget) | 46.1 | 26.6 | 72.7 | 91.4 |
| Plateau \|Δa\| (km) | 196 | — | — | — |

Two facts from this table matter for the taxonomy:

- **The clock is genuinely binding.** 13.2% of successes take > 1500 records and 3.7% take > 1800; the maximum is 1995 out of 2000. The success-time distribution runs right into the cap, so a timeout tail is expected, not anomalous.
- **Fuel is never binding.** The worst cap failure used 72.7% of its Δv budget — exactly the success p95. Hypothesis (d), "fuel-depleted late", has zero instances.

Terminal geometry is tight but not marginal: successes land at median 26 km / 29 m/s, i.e. they use most of the 30 km box but only ~57% of the 50 m/s velocity box. In this regime relative velocity is dominated by along-track phase separation (`v_rel ≈ 2·v_orb·sin(Δφ/2)`), so **position is the binding constraint** — ~1 m/s of rel-vel per km of range. That is why hypothesis (b), "near-miss velocity", does not occur: closing to 30 km automatically brings rel-vel to ~30 m/s.

---

## M1 — Drift-orbit lock (24/37, 64.9%)

**Rule:** ≤3 burns and ≤30 m/s total Δv, with semi-major axis unchanged.

**Signature:**

| Quantity | M1 (n=24) | success (n=1062) |
|---|---:|---:|
| \|Δa_init\| (km), median | 412 (range 348–479) | 167 |
| Burn count, median | **0** (18/24 episodes burn zero times) | 28 |
| Total Δv (m/s), median | **0** (max 30 across all 24) | 230 |
| \|Δa_final − Δa_init\| (km), max | **57** | — |
| Phase alignments per episode, median | 2 (all 24 have ≥1) | 2 |
| Range at closest alignment (km), median | 411 (range 159–732) | 24 |

**Mechanism.** When the initial semi-major-axis separation is already near the top of the achievable range (≈350–500 km, the tail of the triangular Δa distribution induced by two independent U(300, 800) km altitudes), the policy emits action 9 (warp) for essentially every one of its ~400 agent decisions. Across all 24 M1 episodes the pooled action histogram is **47,991 warps and 9 burns**; 18 of the 24 are 100% warp with a literally empty burn list. The satellite is born on a large drift orbit, so the phase gap closes fast (median 2 alignments per episode) — but at each alignment the satellite sweeps past the target at a radial offset equal to its uncorrected Δa, a median of 411 km, an order of magnitude outside the 30 km box. The policy's learned closure trigger appears to require the flyby offset to be inside its canonical ~200 km drift band; at 400+ km it never fires, and the agent watches two clean alignment windows go by without acting. This is a coverage failure, not a physics wall: a 450 km LEO transfer costs ≈240 m/s against a ≈480 m/s budget, and 71 of the 96 episodes with \|Δa\| ≥ 340 km do succeed.

**The cliff:**

| \|Δa_init\| band (km) | n | success | cap | collision | cap rate |
|---|---:|---:|---:|---:|---:|
| 50–300 | 932 | 919 | 12 | 1 | 1.3% |
| 300–350 | 87 | 86 | 1 | 0 | 1.1% |
| 350–400 | 46 | 39 | 7 | 0 | **15.2%** |
| 400–450 | 28 | 16 | 12 | 0 | **42.9%** |
| 450–500 | 7 | 2 | 5 | 0 | **71.4%** |

(No episode is drawn with \|Δa\| < 50 km — the reset enforces a minimum separation — and 500 km is the hard ceiling from the 300–800 km altitude band.) The 8.7% of episodes with \|Δa\| ≥ 340 km carry 25 of the 37 timeouts (68%).

---

## M2 — Window beyond the horizon (6/37, 16.2%)

**Rule:** drift orbit established (4–11 burns, canonical \|Δa\|), fewer than 10 burns in the final 300 records.

**Signature:**

| Quantity | M2 (n=6) | success |
|---|---:|---:|
| \|Δa_init\| (km), median | 116 | 167 |
| Plateau \|Δa\| (km), median | 186 | 196 |
| Burn count, median | 10.5 (all in the first ~half) | 28 |
| Total Δv (m/s), median | 52 | 230 |
| Fuel used (% budget), median | 10.2 | 46.1 |
| **Phase alignments before cap** | **0 in all 6** | 2 |
| Phase gap at cap (deg) | 5.9 – 46.8 | ~0 |
| Closest approach (km), median | 2088 | 24 |

**Mechanism.** The policy executes step 1 of its strategy correctly — a ~50 m/s prograde burn sequence that opens a 163–209 km drift orbit — and then waits. The phase gap never reaches zero inside 2000 records: all six episodes have exactly zero alignments. Extrapolating each episode's own terminal drift rate, they needed **32 to 333 additional records (0.5–5.6 h, median 174 records ≈ 2.9 h, i.e. ~9% more clock)** to reach alignment, plus another ~150 records of closure burst. The proximate cause is a mis-sized drift orbit: the policy picks a roughly constant ±200 km Δa regardless of how much phase it has to close, so when the initial phase gap is on the long side for the direction it chose, one sweep does not fit in the 33 h budget. Note that the drift direction is set by the sign of Δa_init (92% of the time), not by which direction is the shorter way around — so the policy sometimes commits to closing 320° when 40° was available.

---

## M3 — Closure burst truncated (7/37, 18.9%)

**Rule:** ≥10 burns inside the final 300 records.

**Signature:**

| Quantity | M3 (n=7) | success |
|---|---:|---:|
| Burn count, median | 33 (range 17–38) | 28 |
| Total Δv (m/s), median | 245 (range 170–355) | 230 |
| Fuel used (% budget) | 34–73 | 46.1 |
| Closest approach (km) | **30.4 – 152.0** | 24 |
| Rel-vel at closest (m/s) | 26 – 191 | ~29 |
| Time of closest approach (frac of episode) | **0.99 median** | — |
| Terminal distance (km), median | 94 | 26 |

**Mechanism.** These are the true "ran out of clock" failures. The full three-phase maneuver executes — drift orbit at ~203 km, wait, then a 17–38 burn closure burst in the last ~200 records — but the cap arrives mid-burst. Closest approach occurs at 99% of episode length in the median case, i.e. **the satellite is still converging when the episode is killed**. The worst case is `taxonomy_s123/ep_0000076`: closest approach **30.4 km at 26.0 m/s** — 0.4 km outside the position box while comfortably inside the velocity box, a normalized joint distance to the success set of 1.01. `taxonomy_s777/ep_0000125` ends at 67.0 km / 45.8 m/s and `taxonomy_s777/ep_0000196` at 37.9 km / 90.7 m/s. Fuel is not the limit (max 73% of budget used). These would convert to successes with a few hundred more sim steps, and they are the population that a cap raise would recover.

---

## M4 — Floor clip (1 episode, 0.09%; collision, not a timeout)

`taxonomy_s123/ep_0000226`. Satellite starts at 306 km altitude, target at 378 km, Δa_init = −71.8 km. The policy applies its sign-preserving drift-orbit rule: 18 retrograde burns (90 m/s) driving Δa from −72 km toward the canonical −223 km. From a 306 km start that puts perigee at **−7.0 km** — inside the Earth — and the episode terminates on collision at record 62. This is the same ±200 km drift-orbit heuristic as every other episode, applied without an altitude-floor check. The env has an action-validity mask (`enable_action_mask`, obs[38..47], which would have blanked burns whose post-burn perigee falls below `EARTH_KEEPOUT`) but it is disabled in this configuration — the policy is 38-dim-obs and has no perigee guard. Prevalence is low because it requires both a bottom-of-band start and a downward Δa sign, but it is a mechanism, not noise.

---

## Hypotheses tested and rejected

- **(b) "near-miss velocity"** — entered the 30 km sphere with rel-vel never simultaneously < 50 m/s. **Zero instances.** Across all 37 timeouts, 0 episodes ever entered 30 km, 2 entered 60 km, 5 entered 100 km. Because rel-vel scales with along-track separation at ~1 m/s per km in LEO, the velocity gate is never the binding constraint; it is slack by ~40% at the moment the position gate closes.
- **(c) "slow spiral"** — monotonic approach that runs out of clock. Not the shape of any failure. The policy does not spiral; it drifts ballistically and then closes in a burst. M3 is the nearest thing, but its distance trace is non-monotonic (it sweeps out to 13,000+ km before the window) and its final approach is fast, not slow.
- **(d) "fuel-depleted late"** — **zero instances.** Max fuel consumed by any cap failure is 72.7% of budget, equal to the success p95. No cap episode was ever unable to burn.
- **(a) "phasing never converged"** — real, and it splits into two mechanistically distinct halves: M1 (never tried to converge; alignment happened and was ignored) and M2 (converging correctly but the window is past the horizon). Keeping them merged would hide the fact that M1 is a Δa-coverage hole and M2 is a clock/drift-sizing problem.

---

## Implications

1. **The dominant hole is a Δa-coverage hole, not a timing hole.** 65% of failures are the policy declining to act on the largest transfers in the distribution. Fixing M1 (curriculum oversampling of \|Δa\| ≥ 350 km, or an explicit Δa-magnitude feature) addresses roughly two-thirds of the residual failure mass and would move the headline from 96.5% toward ~98.7%.
2. **A cap raise buys at most 35% of the failures.** Only M3 and M2 are clock-limited. M3 (7 episodes) needs a few hundred more records; M2 (6 episodes) needs 32–333 records to reach alignment plus ~150–200 for the closure burst. A cap of 2500 records would plausibly recover all 7 M3 and 5 of 6 M2 — about **+1.1pp**, from 96.5% to ~97.6% — while leaving M1 (65% of failures) completely untouched.
3. **The drift orbit is fixed-size and sign-locked.** Both M2 and M4 trace to the policy choosing a ~±200 km drift orbit whose sign follows Δa_init rather than the shorter phasing direction, and without regard to altitude floor. A phase-gap-conditioned drift magnitude (or simply allowing the sign to flip) is the single highest-leverage policy change.
4. **Enabling the action mask removes M4** at zero policy cost, but requires a 48-dim-obs checkpoint.

---

## Files

- `web_data/results/failure_taxonomy.csv` — one row per non-success episode (38 rows), all metrics plus `assigned_mode`.
- `scripts/orbital/t1_failure_taxonomy.py` — metric extraction and mode assignment. Rerun with:
  `python3 scripts/orbital/t1_failure_taxonomy.py DIR [DIR ...] --failures-only --out web_data/results/failure_taxonomy.csv`
- Trajectory sources: `/tmp/compat_check`, `/tmp/taxonomy_s123`, `/tmp/taxonomy_s777`, `/tmp/taxonomy_s9001` (1100 `.npz`).
