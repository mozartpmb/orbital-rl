# J2 rung campaign — secular perturbation training (2026-08-14)

**Headline: tight-box precision under J2 recovered from a 0/200 floor to
198/200 = 99.0% (5 km / 1 m/s), via a two-rung warm-start ladder. All
j2_mode=1 rows are MEAN-ELEMENT claims** (osculating velocity slip at the
1 m/s box is 9.4% of tolerance per orbit — the honest statement is "meets the
box in mean elements", never "osculating-grade rendezvous").

Provenance: branch `ext-j2` (propagator, 2026-08-13) + `ext-j2-rung`
(sampler + campaign, merged 2026-08-14), both fully anchored on the main
build before launch (22-gate physics battery, bitwise anchor trio, LVLH-frame
md5-identity, sampler-off no-op, nav byte-transparency). Campaign
`scripts/orbital/extj2/j2_rung_campaign.sh`, wall-clock ≈ 70 min total.

## What J2 does to the task (measured, stage-1 survey, all in-campaign)

| cell | result |
|---|---|
| X3 reference (equatorial, j2=0) | 200/200 |
| inclined targets U(30°,60°), j2=0, corrected LVLH frame | 90.0% |
| same, with the legacy (non-LVLH-at-inclination) frame | 56.5% — the frame bug, not J2 |
| inclined, j2=1 | 79.5% (−10.5 pp = the loose-box J2 gap) |
| equatorial, j2=1 | 200/200 — J2 exactly free when the plane channel is inert |
| **5 km / 1 m/s, j2=1, zero-shot** | **0/200** — collapse; failures 100% safety_cap |

Mechanism (from the sampler non-inertness gate, 2000 draws): median
differential nodal precession 0.436°/day, of which **the altitude difference
contributes 13× more than the plane offset** — the transfer task itself
generates the J2 signal. Over a ~2-day episode that injects ~0.69° of
relative inclination = 183% of the free-plane zone ≈ 92 m/s if corrected
naively.

## The recovery ladder

Chained warm starts (X3 → loose → 10 km/10 m/s → 5 km/1 m/s), inclined
targets U(30°,60°), corrected LVLH frame, `lvlh_frame_mode=1`, 50M steps per
rung, seed 42; flatline tripwire armed at every rung (floor + 5 pp after 20M).

| rung | floor (chain / old lineage) | native j2=1 | retention j2=0 |
|---|---|---|---|
| loose 30 km / 50 m/s | 79.5% zero-shot | **200/200** | 200/200 (+ 200/200 on X3 home) |
| 10 km / 10 m/s | 75.5% / 56.5% | **200/200** | 200/200 |
| **5 km / 1 m/s** | 13.5% / **0/200** | **198/200 = 99.0%** | 196/200 = 98.0% |

No rung flatlined (tripwire crossings at 1.7M and 11.9M steps). Each rung
lifted the next rung's floor (0/200 → 13.5% at the final box) — the
bootstrap risk the red-team named (capped episodes pay nothing under
`cap_terminal_reward=0`) never bound because the ladder kept the success
signal alive.

## Bugs found by this campaign's prep (both fixed, both default-off-safe)

1. **The 12th metric-vs-implementation bug:** the inertial-ϖ correction
   (itself the fix for the 11th) subtracted the chaser's *absolute* RAAN —
   correct only in the pinned-node gauge every prior lineage happened to use.
   Under target-plane sampling the e-disc knob silently became 5.55× its
   setting. Fixed with the relative node; bit-exact for all shipped lineages.
2. **The LVLH observation is only LVLH at equatorial targets** — at 30–60°
   inclination the frame error reaches 239% on the primary rendezvous
   channels. Fixed behind `lvlh_frame_mode=1` (md5-identical at i=Ω=0);
   at inclined targets it alone moves zero-shot 56.5% → 90.0%. Without this
   fix the J2 gap would have been confounded ~2:1 by a frame artifact.
3. Known and deliberately unfixed: obs[18] is absolute inertial longitude,
   which caps inclined-target capability above ~60° regardless of J2
   (present at j2_mode=0); fixing it would change legacy-anchor semantics.
   The sampler band U(30°,60°) stays below the cliff and outside both channel
   zeros (0°, 90°) and the critical inclination (63.43°).

## Multi-seed — 3/3 (2026-08-14)

Full recovery chain re-trained per seed (loose → 10 km/10 m/s → 5 km/1 m/s):

| seed | loose native | 10 km/10 native | **5 km/1 native** | 5 km/1 retention (j2=0) |
|---|---|---|---|---|
| 42 | 200/200 | 200/200 | **198/200 = 99.0%** | 98.0% |
| 7 | 200/200 | 200/200 | **199/200 = 99.5%** | 99.0% |
| 1337 | 200/200 | 200/200 | **198/200 = 99.0%** | 99.0% |

Pooled at the tight box: **595/600 (99.2%), 0.5 pp spread, zero flatlines**
(tripwire crossings 1.2–11.9M steps). The J2 recovery is a capability, not a
seed artifact.

## Claim boundaries

- Mean elements only, everywhere j2_mode=1 (no mean↔osculating layer exists).
- Truth-state navigation: J2 × angles-only nav campaign in flight (its prep
  found and fixed the 13th metric-vs-implementation-class defect: the nav
  harness sub-propagated truth states two-body under a J2 env — the filter
  was consistently tracking a fiction; filter-vs-filter validation is
  structurally blind to a harness wrong the same way for both arms).
- Single inclination band U(30°,60°), Ω gauge verified flat (5% split, a leak
  detector not a knob).

Checkpoints `models/t3/extj2_{A2_j2trained,A3a_j2_box10k10,A3b_j2_box5k1}.pt`;
per-cell JSON `web_data/results/extj2_rung/`; wandb `t6-j2-*`; full analysis
`J2_RUNG_NOTES.md` + `scripts/orbital/extj2/j2_zeroshot_survey.txt`.

---

# J2 × angles-only navigation (2026-08-15, `j2nav_campaign.sh`)

**The capability product: one policy flying bearings-only with real batch IOD
in the J2-perturbed, inclined-target world — 96.0% from a 32.0% floor, with
truth-mode at 200/200.** Loose box (30 km / 50 m/s), targets U(30°,60°),
`lvlh_frame_mode=1`, `MSC6J2Cov(stm_j2='fd')` (the measured filter winner),
50M warm-started from the nav root (`n3dnav_T-BO3.pt`).

## Floors — the transfer asymmetry, measured before training

| root | blind (BO + real IOD) | truth | anatomy |
|---|---|---|---|
| nav-skilled (no J2 exposure) | 32.0% | 47.0% | J2+inclination costs 53 pp of capability; the estimate-flying skill survives the world-shift (15 pp gap) |
| J2-skilled (no estimate exposure) | 29.5% | 100% | full capability at home, full 70.5 pp observability price |

Same ~30% floor, opposite anatomy — and consistent with the program's core
finding: estimate-robustness is the hard-to-retrofit skill. The stage-3
auto-gate (retrain the J2 root only if its blind floor ≥ 30%) correctly
closed at 29.5%.

## Treatment

| row | result |
|---|---|
| **native — bearings-only, J2, inclined** | **192/200 = 96.0%** (floor 32.0%) |
| truth, same ckpt | 200/200 — the 4 pp gap is pure navigation |
| retention: J2 off, bearings-only | 191/200 = 95.5% |
| retention: X3 home rung (equatorial) | **166/200 = 83.0%** — real forgetting, −17 pp vs parent |

The X3-home dip is the campaign's honest cost: adapting to the inclined
J2 world traded 17 pp at the legacy equatorial rung. Not hidden, not fixed
by re-labeling; a rehearsal-mix rung would likely recover it (untested).

## Provenance & the 13th bug

Trainer hung in teardown post-checkpoint; the campaign's own watchdog killed
it and continued autonomously (rc=143, final perf 0.999 — the NAV-F lesson,
now automated). Campaign prep found and fixed the **13th
metric-vs-implementation-class defect**: `OrbitalNav` sub-propagated truth
states two-body under a J2 env — the filter tracked a fiction with perfect
consistency; standalone (filter-vs-filter) validation is structurally blind
to a harness wrong the same way for both arms. The fix restored in-loop NEES
from 25,520 to 1.20–1.37. Integrated J2-nav cost: 4.45× the two-body nav
tick (~7.5 h per 50M arm).

Claim boundaries: mean elements everywhere under J2; single seed (42);
inclination band U(30°,60°); loose box only (tight-box J2 × nav untested);
acquisition-surrogate validity boundary at ~3 h blind windows (measured
optimistic 3.3% at 6 h). Checkpoint `models/t3/j2nav_T-J2BO-nav_final.pt`;
JSON `web_data/results/j2nav/`; wandb `t6-j2nav-*`.
