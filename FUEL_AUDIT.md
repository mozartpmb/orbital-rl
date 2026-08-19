# Fuel audit — per-episode Δv against the optimal-transfer baseline

200 held-out episodes per lineage, seed 123, greedy argmax, each at the exact
config its published score was produced under. Script
`scripts/orbital/nav/fuel_audit.py`; per-episode rows in
`web_data/results/fuel_audit.csv` (800). Δv budget is 478 m/s.

Extends the T1 Lambert baseline (`scripts/orbital/t1_lambert_baseline.py`)
across the current flagship lineages. Bearings-only lineages fly on their own
estimate with the real batch IOD, as trained.

## What the comparators are, and what they are not

**Lambert time-matched** — the minimum two-impulse transfer over (departure
wait, TOF, revolutions) inside the same mission clock the policy actually
spent. It is the right optimum for the 2D lineage and is quoted there as a
ratio.

**Box-credited Lambert** — the same schedule with its second impulse reduced by
the box's velocity tolerance. Lambert is held to *exact* rendezvous (0 m,
0 m/s); the policy only has to reach the success box, and the box grants
`|v_rel| < rel_vel_tol` on arrival, so a schedule stopping short by exactly
that tolerance still passes the env's own test. T1 named this bias and could
not remove it. At the 50 m/s box it is most of the second impulse — which is
why the raw ratio can sit *below* 1.0 with nothing wrong. The 30 km position
slack is **not** credited, so this remains a conservative comparator.

**For the 3D lineages the raw ratio is not like-for-like.** It divides *total*
Δv, plane change included, by a coplanar baseline that cannot buy a plane
change at all. Read the decomposition and the in-plane box-credited ratio
instead; the raw column is kept only so the inflation is visible rather than
hidden.

**Plane floor** — `2·v_c·sin(Δi_rel/2)` at the realized per-episode Δi_rel
(drawn as `di_max·√U`, so quoting it at `di_max` would understate the ratio by
~1.4×). This is the pure-inclination optimum at *circular* speed: it is an
upper bound on the true optimum, because a combined tangential+normal impulse
and a plane change taken near apogee of an eccentric orbit both beat it.

## The table

| lineage | success | Δv med (p90) m/s | raw ratio med/p75/p90 | in-plane box-credited med/p90 | decomposition med: tan / plane / fine | plane vs floor med | burns med | Lambert med |
|---|---|---|---|---|---|---|---|---|
| **T3-canonical** 2D, box 30 km / 50 m/s | 200/200 | 235 (341) | 1.18 / 1.34 / 1.54 | **1.55** / 2.34 | 235 / 0 / 0 | — | 10 | 198 |
| **TB5-3D** 3D + bearings-only, 5 km / 1 m/s | 196/200 | 363 (392) | 2.22 / 2.92 / 3.72 | **1.81** / 2.99 | 294 / 138 / 14 | **1.48×** (floor 92) | 27.5 | 162 |
| **E3** 3D + bearings-only, e→0.30, 30 km / 50 m/s | 190/200 | 250 (325) | 1.42 / 1.69 / 2.15 | **1.79** / 3.25 | 225 / 25 / 0 | **0.64×** (floor 56) | 13 | 169 |
| **J2-A3b** 3D + J2, 5 km / 1 m/s | 198/200 | 315 (391) | 1.86 / 2.21 / 2.80 | **1.41** / 2.12 | 242 / 133 / 16 | **1.36×** (floor 103) | 24 | 173 |

Median realized Δi_rel: 0.700° (TB5-3D), 0.553° (E3), 0.775° (J2-A3b).
Median episode length: 7.9 / 13.2 / 11.4 / 14.8 target orbits.

## Is there a fuel-efficiency case worth a training intervention?

The bar is set by R5: reward-reshaping a *committed* policy collapses it (all
four variants fell below 10%, and plasticity was the prerequisite), so any
fuel bonus costs a fresh-ish rung, not a fine-tune. Each lineage against that
bar:

- **T3-canonical — no.** 1.18× a time-matched two-impulse optimum and 1.55×
  the box-credited one, on an all-tangential budget with 10 burns and a perfect
  score. The residual headroom is real but is the same order as the
  comparator's own slack, and two-impulse Lambert is itself an upper bound on
  the classical optimum, so the true gap is smaller than 1.55× suggests.
- **TB5-3D — the only arguable case, and it is small.** The plane leg runs
  1.48× its circular floor, i.e. ~46 m/s median of identified excess on a
  478 m/s budget (~10%), and the terminal burn train is 27.5 burns against 10
  for T3. That is a real inefficiency with a named mechanism, but ~10% of
  budget on a 98%-success lineage does not obviously repay a fresh ladder.
- **E3 — no, and the number is interesting.** Its plane spend is **0.64× the
  circular floor**: the policy is beating a pure-inclination change, which is
  what combined tangential+normal impulses and apogee plane changes on
  eccentric orbits are supposed to buy. Nothing to reshape here.
- **J2-A3b — no.** The lowest box-credited ratio of the four (1.41×) despite
  carrying both a plane leg and J2 secular drift.

**Overall: no lineage clears the bar.** The single largest identified excess is
TB5-3D's plane leg at ~10% of budget. If a fuel intervention is ever taken, the
evidence points at the tight-box plane leg specifically — not at a global
fuel bonus, and not on a committed checkpoint.

## Validation carried with the numbers

**The burn path is clean.** T1's headline was falsified by its own diagnostic:
an inverted `true_to_mean` half-angle map handed the agent ~24°/episode of free
along-track phase, which is what made 13% of its episodes cost less than half
the fixed-time optimum. That map is fixed, and this audit re-measures the
teleport per episode rather than assuming it: **0.0006° / 0.0010° / 0.0017°**
per episode for T3-canonical / TB5-3D / E3 — six orders below the pre-fix
figure. So the ratios here are admissible where T1's were not.

Two corrections were needed to get that right, and both would have produced a
false alarm:

1. **T1's diagnostic is planar.** It reads the in-plane anomaly as
   `atan2(y, x) − ω`, which is the true anomaly only for an equatorial orbit.
   Applied unchanged to the inclined lineages it reported **125°/episode** on
   TB5-3D — pure frame mismatch. Taking the anomaly in the orbit plane (which
   reduces to T1's expression exactly at `inc = raan = 0`) gives 0.0010°.
2. **J2-A3b shows 3.35°/episode and that is physics, not a defect.** Under
   `j2_mode=1` the orbit is not Keplerian, so a diagnostic built on two-body
   propagation necessarily registers the secular drift. The discriminator is
   *where* it sits: measured per step it is **0.00224°/step at burns vs
   0.00258°/step on coast** — uniform, and marginally larger away from burns.
   A burn-path defect concentrates at burns; this does not. Magnitude also
   matches the expected J2 secular rate for this band to within the spread of
   the 30-60° inclination sampling.

**In-plane projection.** For the 3D lineages the Lambert reference is computed
after rotating both bodies into the target's orbit plane (a rotation, not a
projection of a tilted orbit onto the equator). The dropped out-of-plane
component is at most **1.7%** of state magnitude across all episodes.

## Caveats

- Two-impulse Lambert is an upper bound on the classical optimum — three-impulse
  and bi-elliptic schedules beat it in some geometries — so every ratio here is
  a lower bound on the policy's true excess.
- Time-matched hands a policy that dawdles a cheaper baseline to be measured
  against; it is a conservative construction, not a generous one.
- The plane floor is the *circular* pure-inclination optimum, which combined
  impulses and eccentric-orbit apogee burns legitimately beat. It bounds the
  plane spend from above, which is why E3's 0.64× is not an anomaly.
- Ratios and decompositions are over successful episodes only.
- Single evaluation seed (123) per lineage; these are distributions over
  scenarios, not over training seeds.

---

## Addendum — the T11 generalist against the same comparator (2026-08-18)

`models/t3/t11_generalist_rungB.pt` at its five green mixture cells, 200 held-out
episodes each at seed 123, real batch IOD. Same box-credited time-matched Lambert
comparator as the four specialist rows above.

**Why re-measure something T11 already reports.** `T11_GENERALIST.md` quotes a
uniform "~1.28× fuel efficiency", but that is a **different reference** — the
cell's own linearised direct-burn estimate — and is not comparable to a
box-credited Lambert ratio. Putting the generalist in this table required running
it through this script's comparator, not copying its number across. Each cell is
built by `t11_eval.env_kwargs`, the generalist's **own** eval path including its
acquisition configuration, because the quantity being audited is the fuel the
shipped policy actually spends and swapping the acquisition front-end would change
the trajectory before the comparator ever ran. The two red cells (W1_driftwait,
TIGHT_5k1) are omitted: both score 0.0%, so there are no successes to audit.

| lineage | success | Δv med (p90) m/s | raw ratio med | **in-plane box-credited med / p90** | decomposition med: tan / plane / fine | burns med | Lambert med |
|---|---|---|---|---|---|---|---|
| **T3-canonical** 2D, 30 km / 50 m/s | 200/200 | 235 (340) | 1.18 | **1.55** / 2.34 | 235 / 0 / 0 | 10 | 151 |
| **TB5-3D** 3D + BO, 5 km / 1 m/s | 196/200 | 363 (391) | 2.22 | **1.81** / 2.99 | 294 / 138 / 14 | 28 | 161 |
| **E3** 3D + BO, e→0.30, 30 km / 50 m/s | 190/200 | 250 (325) | 1.42 | **1.79** / 3.25 | 225 / 25 / 0 | 13 | 123 |
| **J2-A3b** 3D + J2, 5 km / 1 m/s | 198/200 | 315 (391) | 1.86 | **1.41** / 2.12 | 242 / 133 / 16 | 24 | 171 |
| T11 generalist @ E0_j2 | 194/200 | 334 (441) | 1.89 | **2.15** / 3.62 | 285 / 100 / **0** | 16 | 130 |
| T11 generalist @ E1_j2 | 200/200 | 308 (403) | 2.00 | **2.44** / 4.27 | 252 / 100 / **0** | 14 | 106 |
| T11 generalist @ E2_j2 | 199/200 | 308 (396) | 1.83 | **2.04** / 3.24 | 255 / 110 / **0** | 14 | 119 |
| T11 generalist @ E3_j2 | 195/200 | 319 (411) | 1.75 | **2.00** / 3.49 | 260 / 100 / **0** | 15 | 131 |
| T11 generalist @ LONGRANGE | 198/200 | 319 (431) | 1.84 | **1.95** / 3.42 | 252 / 122 / **0** | 14 | 131 |

### Reading

**The generalist pays a consistent fuel premium.** Its box-credited ratio spans
**1.95–2.44 (median 2.04)** against **1.41–1.81** for every specialist — worse
than the worst specialist in all five cells. The premium is not explained by
harder geometry: the generalist's cells have *lower* Lambert references
(106–131 m/s versus 123–171), so it is spending more against an easier in-plane
optimum. Nor is it a capability trade — success is 97.0–100% across the five.

**It spends zero on fine burns, in every cell.** `dv_fine` is **0 m/s** for all
five generalist cells, while the two tight-box specialists spend 14 and 16 m/s
there. The generalist is a coarse-burn policy. That is consistent with its two
red cells being the tight box and drift-and-wait, and it connects to a result
measured independently: `NAVF_RESULTS.md` and `N3DNAV_RESULTS.md` §MAJOR-10 both
found the fine actuation rows to be **guidance-critical, not informational**
(masking rows 20/21 drops TB5-3D from 98.0% to 51.0% even with perfect state). A
policy that never uses them is not equipped for a tight box — which is what the
mixture result already showed from the other side.

**Verdict against the R5 bar: no fuel intervention.** The excess over the best
specialist is ~0.5 on a box-credited ratio, roughly 30–60 m/s of median spend
against a ~478 m/s budget — real, consistent, and still far too small to justify
what it would cost. `R5` established that reward-reshaping a committed policy
collapses it (all four variants below 10%), so a fuel bonus cannot be bolted onto
`t11_generalist_rungB` and would require a fresh-ish rung at 200M steps. The
cheaper lever already exists and is already in the policy: T11 sampled the fuel
budget per episode and got **graded budget-awareness** (88–97% on a 353 m/s lean
tank, 99–100% on a 656 m/s rich one). Widening or re-weighting that sampling
costs nothing extra at train time and targets the same quantity. If fuel ever
does become the objective, the measurement to take first is whether the missing
fine-burn repertoire — not the coarse schedule — is where the excess lives.

### Note on this script

The CSV writer used to **truncate** `web_data/results/fuel_audit.csv` to whatever
`--lineages` named, so auditing one lineage silently destroyed the others. A
4-episode smoke run during this work deleted 800 rows of measured data, which
survived only because they happened to be committed. The writer now merges:
rows for the lineages just measured are replaced, every other lineage is carried
through (`1800 rows: 1000 new, 800 carried`).
