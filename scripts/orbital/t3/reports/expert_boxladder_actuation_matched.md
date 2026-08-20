# Expert box ladder, actuation-matched — the quantum was not the cliff

**Verdict: the review's hypothesis is refuted for the actuator and confirmed for
the controller constants.** Granting the scripted GNC expert the Discrete-20
fine-radial rows changes its tight-box score by **exactly nothing** — arm B is
bit-identical to arm A, and rows 18/19 are emitted **zero times** in 800
episodes. Scaling the two non-scaling controller constants to the box is worth
**+12.0 pp** at 5 km / 1 m/s. Even so the expert reaches 33.2% against the
policy's 97–98%, and **56% of its remaining failures never get within 5 km at
all**, so the expert-vs-policy gap at TB5 is not an actuation artifact.

Measured 2026-08-20 · `scripts/orbital/t3/expert_boxladder.py` · 200 episodes
per row per seed (42 and 7), upgraded from the published n=100 single-seed ·
per-row CSV in `web_data/results/expert_boxladder/`.

---

## Results

`±95%` is the binomial interval on the pooled row. `FAILED-ep vrel_inbox` is the
minimum |v_rel| reached while already inside the position box, **over failed
episodes only** — successes terminate the instant both tolerances are met, so
their recorded minimum is censored at `box_v` by construction and pooling them
pins the statistic to the tolerance. `nf` is how many failed episodes ever
entered the position box; `fine` counts emissions of rows 18/19.

| arm | box | seed | n | success | ±95% | dv med | vrel_inbox med / p10 / min | nf | fine |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** D16 control | 5 km / 5 m/s | 42 | 200 | 50.0% | 6.9% | 181 | 6.36 / 5.82 / 5.69 | 2 | 0 |
| **A** | 5 km / 5 m/s | 7 | 200 | 60.0% | 6.8% | 180 | 6.10 / 5.78 / 5.70 | 2 | 0 |
| **A** | 5 km / 5 m/s | **both** | 400 | **55.0%** | 4.9% | 180 | 6.10 / 5.69 / 5.69 | 4 | 0 |
| **A** | 5 km / 1 m/s | 42 | 200 | 16.5% | 5.1% | 175 | 1.96 / 1.41 / 1.11 | 69 | 0 |
| **A** | 5 km / 1 m/s | 7 | 200 | 26.0% | 6.1% | 175 | 1.89 / 1.36 / 1.06 | 70 | 0 |
| **A** | 5 km / 1 m/s | **both** | 400 | **21.2%** | 4.0% | 175 | 1.90 / 1.35 / 1.06 | 139 | 0 |
| **B** D20 quantum | 5 km / 5 m/s | both | 400 | **55.0%** | 4.9% | 180 | 6.10 / 5.69 / 5.69 | 4 | **0** |
| **B** | 5 km / 1 m/s | both | 400 | **21.2%** | 4.0% | 175 | 1.90 / 1.35 / 1.06 | 139 | **0** |
| **C** D20 + scaled | 5 km / 5 m/s | 42 | 200 | 58.5% | 6.8% | 180 | — | 0 | 0 |
| **C** | 5 km / 5 m/s | 7 | 200 | 70.0% | 6.4% | 178 | — | 0 | 0 |
| **C** | 5 km / 5 m/s | **both** | 400 | **64.2%** | 4.7% | 179 | — (no entered-and-failed) | 0 | 0 |
| **C** | 5 km / 1 m/s | 42 | 200 | 31.5% | 6.4% | 183 | 2.05 / 1.34 / 1.05 | 52 | 0 |
| **C** | 5 km / 1 m/s | 7 | 200 | 35.0% | 6.6% | 180 | 2.16 / 1.19 / 1.10 | 66 | 0 |
| **C** | 5 km / 1 m/s | **both** | 400 | **33.2%** | 4.6% | 182 | 2.15 / 1.22 / 1.05 | 118 | 0 |
| **H16** headline | 30 km / 50 m/s | 42 | 200 | **99.0%** | 1.4% | 190 | — | 0 | 0 |
| **H20** headline | 30 km / 50 m/s | 42 | 200 | **99.0%** | 1.4% | 190 | — | 0 | 0 |

Causes are binary throughout: every non-success is `safety_cap`. No collisions,
no strandings, no escapes in 2,800 episodes.

### Arm A reproduces — the harness has not drifted

Published `recon_expert_baseline §4.6`: **53%** at 5 km / 5 m/s and **18%** at
5 km / 1 m/s (n=100, single seed). Arm A at seed 42 gives **50.0%** and
**16.5%**; pooled over both seeds, **55.0% ± 4.9%** and **21.2% ± 4.0%**. Both
published values sit inside the pooled intervals. The gate passes.

The seed spread is worth noting on its own: 50.0% vs 60.0% at tight5 and 16.5%
vs 26.0% at tight1 between two seeds of n=200. The published single-seed n=100
numbers carry roughly ±10 pp of seed noise, so **no difference below ~10 pp
should be read off them**.

### Arm B is bit-identical to arm A, and rows 18/19 are never emitted

Every arm-B cell equals its arm-A counterpart exactly — success, dv, causes, and
the whole |v_rel| distribution. `fine = 0` across all 800 arm-B episodes.

The instrument is not vacuous; it was tested directly. Calling `_burn` on a
synthetic pure-e-vector residual with rows 18/19 armed returns `None` at a
1.13 m/s residual and, at 3.77 m/s, returns **row 13 — fine *tangential* −1 m/s
— not the fine radial**. That is correct control, and it explains the null:

- **D16 already has 1 m/s authority.** Rows 12–15 are the M3 sub-5 m/s burns,
  ±1 and ±2 m/s **tangential**, and they are inside the 16-action space.
- At an apsis a tangential impulse moves the eccentricity vector **twice** as
  far per m/s as a radial one (δe = 2Δv/v against Δv/v). The expert's gain law
  `(V0 − V2) − |Δv|·(1−slack)` therefore always prefers row 12/13 over row
  18/19, and the fine radial rows are dominated rather than unused-by-oversight.

So the premise behind the review — that the published tight rows were measured
against a ±10 m/s radial floor — **does not describe this controller**. The
expert already had 1 m/s resolution; it simply had it on the tangential axis.
`orbital.h:192-194`'s "5.02 m/s with 16 actions → 0.71 m/s with fine radial" is
a statement about a radial-only capability analysis and is not a statement about
`expert_controller.py`. (The pooled `vrel_inbox` minimum of 5.00 at the 5 m/s
box in an earlier cut of this table looks like it corroborates 5.02, but it does
not — that number is censoring at `box_v = 5.0`, and it is why this report
quotes the failed-episode distribution instead.)

### Arm C — the constants were a real confound, but not the whole story

Scaling moves tight5 **55.0% → 64.2%** (+9.2 pp) and tight1 **21.2% → 33.2%**
(+12.0 pp). Both exceed the ±4–5 pp intervals, and both hold across seeds
(tight1: 16.5→31.5 at seed 42, 26.0→35.0 at seed 7).

**The scaling, stated.** `A_DEADBAND` = 6 km and `DA_FLOOR` = 3 km are absolute
metres fixed at the 30 km design point, i.e. 0.20 and 0.10 of `box_r`. Arm C
holds those fractions: `min(default, frac · box_r)` → **1.0 km and 0.5 km** at a
5 km box. `SAFE_R`/`SAFE_V` already scaled (0.87/0.84 of the box); these two did
not, and `DA_FLOOR` had no environment override at all until this change.

The reason to suspect `DA_FLOOR` was dimensional, not empirical. A parked
semi-major-axis offset `da` implies a relative speed |Δv| ≈ ½·v·da/a, so at
a = 7×10⁶ m (v = 7546 m/s) the unscaled 3 km floor implies **1.62 m/s on its
own** — already outside a 1 m/s tolerance before any guidance error. Scaled to
0.5 km it implies 0.27 m/s, inside. The measured failed-episode |v_rel| in arm A
(median 1.90, p10 1.35) brackets that 1.62 m/s prediction, which is the evidence
that the constant, not the quantum, was binding.

### Where the tight-box failures actually live

| arm | box | failures | never reached 5 km | reached it, |v_rel| too high |
|---|---|---:|---:|---:|
| A / B | 5 km / 5 m/s | 180 | **176 (97.8%)** | 4 (2.2%) |
| A / B | 5 km / 1 m/s | 315 | **176 (55.9%)** | 139 (44.1%) |
| C | 5 km / 5 m/s | 143 | **143 (100%)** | 0 |
| C | 5 km / 1 m/s | 267 | **149 (55.8%)** | 118 (44.2%) |

At the 5 m/s box the velocity tolerance is essentially free: 97.8% of failures
never get inside 5 km, and under arm C *every* episode that reached the position
box captured. **The tight5 cliff is a position problem, not a velocity one** —
which no amount of finer velocity authority can address.

At the 1 m/s box the failure mode splits almost evenly, and the same 176
episodes fail to approach in both boxes. Of the 21.2% baseline: 44% of the field
never reaches 5 km, 35% reaches it and cannot null |v_rel| below 1 m/s, 21%
succeeds. Arm C converts part of the second group (85 → 133 successes) while
barely moving the first (176 → 149).

---

## Self-red-team: is the expert disadvantaged by decision cadence?

**No, and the code says why.** The concern is that the expert coasts through the
box on a long warp and is never sampled inside it, whereas a τ=1 policy would
be. The measurement confirms the premise: only **28.8%** of the expert's in-box
decisions are τ=1 — it is inside the box on a warp most of the time.

But that carries no penalty, because the success test is **not** evaluated on
the decision grid. `orbital.h` runs the sub-step loop
`for (int k = 0; k < tau; k++) { … if (check_termination(env)) …}` with
`check_termination` — which contains the `at_target` position-and-velocity test
at `orbital.h:1825` — executed **after every sub-step**, explicitly so a warp
"never skips past a conjunction". A 60-minute warp that passes through a
satisfying instant terminates successfully at that instant. The expert and the
policy are therefore scored by the identical instantaneous-in-box criterion at
identical 60 s resolution, and neither is advantaged by its cadence.

Two further checks that the comparison is fair:

- **The headline control did not move.** H16 and H20 both score 99.0% with
  identical causes (198 success, 2 safety_cap) and identical median dv (190 m/s).
  Opening the action space to 20 changes nothing where the box is loose, which
  is what it should do and confirms no unintended coupling from the ctor change.
- **The expert aims inside whatever box it is given.** `SAFE_R`/`SAFE_V` are set
  to 0.87/0.84 of the enforced box in `run()`, so the tight rows are not the
  30 km controller being scored against a 5 km ruler.

## What this means for the expert-vs-policy comparison

The published tight-box expert rows should be **corrected upward for the
constants and left alone for the quantum**: 5 km / 5 m/s **55.0% → 64.2%** and
5 km / 1 m/s **21.2% → 33.2%**, with the seed-noise caveat that the original
n=100 single-seed numbers carry ~±10 pp.

The gap to the tight-box policies (97–98% at TB5) survives the correction with
room to spare, and the failure decomposition says why it is a guidance gap
rather than an actuation one: over half the expert's remaining tight-box
failures never close to 5 km in the first place. Whether a differently-tuned
scripted law could close that is untested here — arm C scaled two constants on a
dimensional argument, not a re-tune, and the +12 pp it bought suggests the
controller has more headroom of the same kind.

## Files

- `scripts/orbital/t3/expert_controller.py` — `BURN_ACTIONS_D16/D20`, `acts`
  histogram widened to 30 (rows 18/19 would `IndexError` a 16-slot list),
  `legacy_action_space` plumbed to the ctor, `T3_DA_FLOOR` override added,
  box/actuation/scaling CLI flags, and per-episode terminal-geometry columns
  (`best_vrel_inbox`, `best_vrel`, `best_d_km`, `n_inbox`, `n_inbox_tau1`,
  `n_fine_rad`).
- `scripts/orbital/t3/expert_boxladder.py` — arm driver and aggregator.
- `web_data/results/expert_boxladder/*.csv` — 14 rows × 200 episodes.

Not touched: `pufferlib/` (no C or env change of any kind),
`scripts/orbital/extj2/`, and every root `*.md`.

```bash
for A in A B C; do for S in 42 7; do
  OMP_NUM_THREADS=1 python3 scripts/orbital/t3/expert_boxladder.py --arm $A --seed $S --episodes 200 &
done; done
for A in H16 H20; do
  OMP_NUM_THREADS=1 python3 scripts/orbital/t3/expert_boxladder.py --arm $A --seed 42 --episodes 200 &
done; wait
python3 scripts/orbital/t3/expert_boxladder.py --report
```
