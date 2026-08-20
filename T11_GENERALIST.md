# T11 — the generalist (2026-08-18, `t11_campaign.sh`)

**One policy flies J2 × the full eccentricity range × bearings-only
navigation × long-range transfer at 97–100% per cell with uniform ~1.28×
fuel efficiency and graded budget-awareness — five of the seven mixture
cells. The two failures (drift-and-wait, tight box) are both instances of
the project's most-replicated mechanism, and both are stated, bounded, and
diagnosed.** Single seed (42), mean-element claims under J2 throughout.

## Construction

- **Root**: `extj2_A3b_j2_box5k1` (J2 tight, the matrix's strongest
  generalist-adjacent checkpoint) **transplanted** into the wide normalizer
  family — the GEN_MATRIX's 99-pp "normalizer barrier" is a parameterization
  artifact: the encoder is linear on raw obs, so a 7-column weight rescale
  reproduces the source bit-identically (gate-verified every launch). Head
  expanded 30→31 (day-warp row, logits seeded from the hour-warp row —
  zero-init measured vacuous at P=3.9e−9).
- **Rung A** (50M): J2 + inclined targets in the wide pairing, bearings-only
  → 100.0%/99.5% native at E0/E1_j2 (truth 99.0/98.5). Bug #15 en route: a
  single-cell rung must specify its cell as completely as the mixture
  sampler does (di/fuel/cap/band fell to ini defaults; 50M of plane-free
  training scored 0.998 rolling and 50% at eval — train/eval split).
- **Rung B** (200M): 7-cell weighted mixture, **fuel budget sampled
  U(0.113, 0.20) ≈ 353–656 m/s per episode** (scarcity pressure in lieu of
  any fuel-bonus reward term — reward-reshape is the known-collapse path),
  per-episode cell sampler C-side (caps 3000–22000 mixed in one batch,
  obs-clock per episode's own cap).
- Pre-launch red-team catches, both measured-free on narrow cells before
  shipping: MAJOR-16 OOP seed ceiling (16.4% of mixture seeds sat on a fixed
  numerical clip), and the range-prior ceiling — **the constructor-derived
  prior excluded 39.1% of true target radii under the mixture** (two in five
  episodes navigated with the target outside the prior's universe; now
  mixture-honest only when the mixture is on, single-cell evals untouched;
  containment gate G8 added).

## The envelope table (200 eps/cell, held-out seed 123, real batch IOD)

| cell | native BO | truth | fuel eff. | lean tank (353 m/s) | rich tank (656 m/s) |
|---|---|---|---|---|---|
| E0_j2 | 97.0% | 96.5% | 1.28× | 88.0% | 99.0% |
| E1_j2 | 100.0% | 97.5% | 1.29× | 97.0% | 100.0% |
| E2_j2 | 99.5% | 98.0% | 1.28× | 96.0% | 99.0% |
| E3_j2 | 97.5% | 97.0% | 1.23× | 88.5% | 99.0% |
| LONGRANGE | 99.0% | 100.0% | 1.28× | 96.5% | 99.0% |
| **W1_driftwait** | **0.0%** | **0.0%** | — | 0.0% | 0.0% |
| **TIGHT_5k1** | **0.0%** | **0.0%** | — | 0.0% | 0.0% |

Budget-awareness reads: graded capability with tank size (88–97% lean →
99–100% rich on the same cells), and adaptive spending — the policy uses
margin when it has margin (1.48× efficiency at E3-rich vs 1.23× at the
sampled mix). The fuel skill came from scenario pressure, not reward
surgery.

## The two zeros — one mechanism, diagnosed

Both fail by timeout (cap 198–200/200), both identical under truth state
(nav exonerated), and both are the bootstrap mechanism measured four times
before (T3 flatline, W4, R5, MAJOR-10): **a cell whose effective warm root
is incompetent never bootstraps under sparse terminal reward in-mixture;
rehearsal weight is not a substitute for a competent root.** Drift-and-wait
was never in this lineage (the specialist is a separate branch); the tight
box skill eroded during the loose-cells-only rung A. The rolling-perf
arithmetic confirms both cells sat at ~0 for their entire 40M/20M shares
(0.756 final ≈ 0.95 × the other cells' weight).

Additional honest confound on W1×nav: the mixture's drift-and-wait cell is
the first-ever bearings-only drift-and-wait attempt, and its day-warp blind
windows (~24 h) sit **beyond the acquisition surrogate's validated boundary**
(measured optimistic 3.3% at 6 h, unmeasured beyond) — the spec's own
re-measure gate was not run for this regime. The W1 zero is therefore
over-determined: incompetent root AND unverified training signal. Any
future attempt must re-validate or replace the surrogate at day-warp arcs
first.

## Deployment reading

The shipped capability set is: **generalist checkpoint for the
J2/eccentric/long-range envelope + specialist checkpoints for drift-and-wait
(truth) and the tight box (bearings-only) + gap-conditioned selection** —
a standard flight-software architecture (mode table), with the boundary
measured rather than guessed. A 7/7 single-policy attempt is possible
(bootstrap the two roots first, then re-mix) but is gated on the surrogate
re-validation above and ~1.5–2 days of compute.

Checkpoints `models/t3/t11_{rungA_j2wide,generalist_rungB}.pt`; per-cell
JSON `web_data/results/t11/` (incl. all floors); wandb `t11-*`;
campaign `scripts/orbital/extj2/t11_campaign.sh`; design + red-team
`scripts/orbital/ext_recon/reports/t11_generalist_design.md`.

---

# T11-tight — the 6/7 attempt, and the interference seesaw it measured (2026-08-19)

**The tight-box skill is fully bootstrappable into the generalist lineage
(0/200 → 98.5% via the proven ladder), but this training recipe cannot HOLD
the tight box and the wide envelope simultaneously: the interference is
bidirectional, and both directions are now measured. Both swings ran at the
full acquisition LR (1e-2), so this is an optimization/curriculum result —
NOT a capacity limit (see Reading).**
Campaign `t11_tight_campaign.sh`, single seed, mean-element claims under J2.

## The three-state seesaw (200 eps/cell, held-out seed 123, native BO)

| cell | generalist (rung B) | + tight ladder | + 50M rehearsal re-mix |
|---|---|---|---|
| E0_j2 | 97.0 | 97.0 | 93.5 |
| E1_j2 | 100.0 | 96.5 | 95.5 |
| E2_j2 | 99.5 | **29.0** | **98.5** |
| E3_j2 | 97.5 | **6.5** | **84.5** |
| LONGRANGE | 99.0 | **40.0** | **99.5** |
| TIGHT_5k1 | 0.0 | **92.5** (98.5 @ training fuel-floor) | **0.0** |
| W1_driftwait | 0.0 | 0.0 | 0.0 |

- **Forward swing**: two 50M tight rungs (10 km/10 m/s → 5 km/1 m/s,
  complete-cell specs per the bug-#15 discipline; flatline tripwires never
  fired — perf 0.972 within 1 h from a 1.5% floor) recover the tight box
  fully but collapse the wide-band cells (E2/E3/LONGRANGE), while the
  narrow-band E0/E1 hold.
- **Back swing**: the auto-gated rehearsal re-mix (7 cells, tight at 0.10)
  recovers the wide cells but re-zeroes the tight box — a *maintenance*
  failure this time: the skill existed at re-mix start and 10% rehearsal
  weight could not defend it against 90% wide gradient. Truth rows match
  native rows throughout (capability, not navigation).
- E3's partial recovery (84.5 vs its 97.5 peak) says 50M of re-mix is thin
  for the longest-episode cell; the direction of all other cells says more
  re-mix would deepen the tight loss, not fix it.

## Reading

Two swings measured in opposite directions establish that **this training
recipe cannot hold both skills at once**. They do **not** establish a
capacity limit, and an earlier draft of this section said they did — the
correction matters enough to state plainly.

**Both swings ran at the full acquisition learning rate** (1e-2, annealed),
which is the rate for *learning a skill*, not for *maintaining* one. At that
rate the majority gradient re-learns its cells from scratch early in the run
and bulldozes the minority skill before rehearsal can defend it; the back
swing then hands the tight box a 0.10 weight, which is the bootstrap problem
the ladder existed to solve. So what is measured is an
**optimization/curriculum failure**, and the two mechanisms — full-LR
re-learning and an under-weighted minority — are exactly the two things a
consolidation run changes.

The representation evidence points the other way. At the same 128 hidden
units, the **TB5-3D specialist holds tight and loose boxes simultaneously**,
and the **E-ladder child holds four wide-band skills at once**. A network
that demonstrably represents four wide-band skills, and separately
represents tight+loose together, is not obviously out of room for six.

**Unresolved: optimization vs capacity.** Both swings were run at
acquisition LR, so neither can distinguish them. The discriminating
experiment is consolidation — low LR, consolidation weights, warm from the
root holding the hardest-to-rebuild skill — and it is the next run, not a
conclusion already reached.

Meanwhile the honest deployment shape is what the fleet already provides —
**the generalist for the envelope, the tight child for proximity operations,
gap-conditioned selection** — justified by measurement, and not contingent
on how the capacity question resolves.

Boundary notes carried from this campaign's red-team: every tight row under
J2 is a **mean-element claim** — and the honest boundary number is the
*relative* mean-vs-osculating slip: **83 m and 0.094 m/s per orbit at 5 km
separation (9.4% of the velocity tolerance per orbit, 28% over three)**,
per `j2_A_design.md`/`J2_DESIGN_NOTES.md`. The absolute short-period terms
are common-mode at this separation and cancel in the relative state; an
earlier draft here compared the 3.3 km absolute amplitude to the 5 km
*relative* box — a frame error (and the amplitude arithmetic also mixed
amplitude with peak-to-peak; inclination-weighted for the trained 30–60°
band the absolute term is ~0.4–1.2 km regardless). Corrected 2026-08-20.
And the tight cell's fuel floor was
raised (0.113 → 0.133) after measuring that the terminal fine-burn train is
unpriced by transfer-only feasibility (15.8% → 2.1% infeasible mass).

Checkpoints `models/t3/t11t_{tight_child,remix_child}.pt`; JSON
`web_data/results/t11_tight/`; wandb `t11t-*`. See also `HOLD_EVAL.md`
(the box certifies arrival, not station-keeping — hold-30 ≈ 21% unprompted)
and `FUEL_AUDIT.md` §generalist (coarse-burn policy, dv_fine = 0, box-credited
2.04× median vs specialists' 1.41–1.81×).
