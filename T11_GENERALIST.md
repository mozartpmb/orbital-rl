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

---

# T13 — consolidation at maintenance LR: the seesaw becomes a stable compromise (2026-08-20)

**The discriminating experiment ran, and neither pure hypothesis survived.
At LR 1e-3 (10× below acquisition), warm from the tight child, tight weight
0.25: the wide envelope consolidates COMPLETELY and PERMANENTLY — five cells
at 94.0–99.5% held from 50M through 100M — while the unanchored tight skill
decays early (92.5 → ~40% by 21M) and then PLATEAUS (42.5% at 50M → 37.5% at
100M, n=200 each, ~1 SE apart). No seesaw to zero: the acquisition-LR re-mix
had tight at 0/200 by 50M; maintenance LR holds a six-skill compromise
indefinitely.** Campaign `t11_consol_campaign.sh`, single seed 42,
mean-element claims under J2, per-cell probes every ~2.6–10.5M
(`web_data/results/t11_consol/probes/`).

## The three-attempt table (200 eps/cell, held-out seed 123, native BO)

| cell | tight child (root) | remix @1e-2 | **consol @1e-3 (mid 50M → fin 100M)** |
|---|---|---|---|
| E0_j2 | 97.0 | 93.5 | 95.5 → **95.0** @1.026× |
| E1_j2 | 96.5 | 95.5 | 99.5 → **98.0** @1.052× |
| E2_j2 | 29.0 | 98.5 | 99.5 → **99.5** @1.021× |
| E3_j2 | 6.5 | 84.5 | 93.5 → **94.0** @1.098× |
| LONGRANGE | 40.0 | 99.5 | 99.5 → **99.5** @0.998× |
| TIGHT_5k1 | 92.5 | **0.0** | 42.5 → **37.5** @1.468× |
| W1_driftwait | 0.0 | 0.0 | 0.0 (excluded by design, evaluated) |

Reads that survive the statistics:

1. **Strict dominance over the re-mix**: every live cell equal or better,
   and tight 0.0 → 37.5. A third state, not a repeat of either swing.
2. **The mechanism split**: LR was the seesaw's mechanism (no collapse to
   zero at 1e-3) but LR alone is half a fix — reward-mediated defense of a
   minority skill still leaks ~50pp before stabilizing. The probe mechanism
   read: tight failures are ~all safety_cap with success episode length
   stretching 1617 → 2499 vs cap 3000 at stable Δv — the wide gradient
   smears the terminal fine-burn endgame, it does not overwrite the
   transfer. Decline stops as the wides saturate (~21M) and the LR anneals.
3. **An unasked-for fuel result**: wide-cell Δv fell 14–21% (mean ~18%) vs
   the rung-B generalist — efficiencies now 1.00–1.10× the linearized
   direct-burn reference (LONGRANGE 0.998×), bought with 9–24% longer
   episodes at ~1.6pp success cost. Not inherited: the root ran 1.08–1.38×
   on the same cells. Consolidation at low LR polished fuel nobody asked it
   to polish.
4. Budget-awareness survives inside the degraded skill (tight lean 26.0% →
   rich 57.5%), and the fin truth row (46.0 vs 37.5 native) shows a small
   nav component in the tight loss — new, previously truth ≈ native.
5. **Probe-level honesty**: the n=20 probe tail (35/25/25/25/15 past e400)
   over-reads as decay; Wilson CIs all overlap and the n=200 anchors
   (42.5 → 37.5) say plateau. The final probe missed the final battery by
   22.5pp — probe direction is quotable, probe levels are not.

**Next (running)**: T13b = same run + a tight-cell-only replay anchor
(CE to the root's softmax on 76,800 stored tight states, λ=0.02, bit-inert
when off) — the dense defense the literature says reward-mediated rehearsal
lacks. If tight holds ≥~90 while the wides repeat their T13 trajectory,
the 6/7 single-policy generalist exists in one checkpoint.

Checkpoints `models/t3/t13_consol_{mid50M,final}.pt`; JSON + probes
`web_data/results/t11_consol/`; wandb `t11c-consol`; research packet
`scripts/orbital/ext_recon/reports/t13_research_synthesis.md`.

---

# T13b — the anchor closes it: one policy holds all six skills (2026-08-21)

**CONSOLIDATION HOLDS. T13 + a tight-cell-only replay anchor (CE to the
root's 31-way softmax on 76,800 stored tight states, λ=0.02 constant,
measured ≤3% SPS cost, bit-identical when off) holds every trained skill
simultaneously for 100M steps: TIGHT 90.5% (root 92.5, statistically
indistinguishable at n=200) with the five wide cells at 94.5–100% — and
96.0% on the tight cell at its training fuel floor, ABOVE the root.** The
interference seesaw is closed as an optimization problem: LR alone bought a
compromise (T13: tight → 37.5%); LR + a dense supervised anchor on the
minority skill buys the full hold. 128 hidden units were never out of room
— the user's pushback on "structural at this capacity" is now a measured
result. **SEED-CONFIRMED 2026-08-21: seed 7 reproduces the hold — final
TIGHT 92.5% (exactly the root), E0 99.0, E1 100.0, E2 99.5, E3 93.5,
LONGRANGE 100.0; tight fuel slices 91.5/95.5/97.5 lean/floor/rich (seed 42:
91.5/96.0/97.5); probe record flat 80–90 throughout, same as seed 42.**
Two-seed tight finals 90.5/92.5 vs root 92.5. Mean-element claims under J2
throughout. Seed-7 JSONs `web_data/results/t13b_anchor/*_s7.json`, ckpt
`models/t3/t13b_anchor_s7_final.pt`.

## The four-recipe table (200 eps/cell, held-out seed 123, native BO)

| cell | root (tight child) | remix @1e-2 | T13 consol @1e-3 | **T13b + anchor (mid 50M → fin 100M)** |
|---|---|---|---|---|
| E0_j2 | 97.0 | 93.5 | 95.0 | 99.5 → **99.0** |
| E1_j2 | 96.5 | 95.5 | 98.0 | 100.0 → **99.5** |
| E2_j2 | 29.0 | 98.5 | 99.5 | 98.5 → **98.5** @1.068× |
| E3_j2 | 6.5 | 84.5 | 94.0 | 91.0 → **94.5** @1.121× |
| LONGRANGE | 40.0 | 99.5 | 99.5 | 100.0 → **100.0** |
| TIGHT_5k1 | 92.5 | 0.0 | 37.5 | 89.0 → **90.5** (96.0 @training floor) |
| W1_driftwait | 0.0 | 0.0 | 0.0 | 0.0 (excluded by design, evaluated) |

- **The anchor defended without taxing**: wide-cell trajectories and fuel
  profiles match T13's within noise (the anchor never touches wide-cell
  minibatches); probe record flat at 80–90 across all 94M (T13's: 95→15).
- Budget-awareness intact in the held skill: lean 91.5 → floor 96.0 →
  rich 97.5. Truth rows match native (tight 90.0, E3 95.0) — capability,
  not navigation.
- Mechanism read: T13 measured reward-mediated defense leaking ~50pp
  before plateauing; the anchor's dense signal (independent of tight
  successes) is the difference, exactly as the CLEAR/kickstarting
  literature predicted (research packet §1).

**Deployment**: if seed 7 confirms, the mode-table's generalist/tight-child
split collapses into ONE checkpoint (`models/t3/t13b_anchor_final.pt`) for
everything except drift-and-wait — whose validated-signal bootstrap is the
running 7/7 arc (`ext-w1nav`: the day-warp "blind window" was a tick-cap
artifact; surrogate exact at K=0).

Checkpoints `models/t3/t13b_anchor_{mid50M,final}.pt`; JSON + probes
`web_data/results/t13b_anchor/`; wandb `t13b-anchor`; anchor build
`scripts/orbital/extj2/build_tight_anchor.py` + gates
`t13b_anchor_gates.py`; campaign `t13b_anchor_campaign.sh`.

---

# T15 — the 7/7 attempt: all seven skills in one policy, the seventh at partial strength (2026-08-22)

**One 128-hidden policy now carries every skill this project has built:
E0 99.0, E1 99.5, E2 98.5, E3 95.5, LONGRANGE 99.5, TIGHT 85.5 (95.5 at
its training fuel floor), and W1 drift-and-wait at a stable 31.5% — up
from an unclimbable 0.** Warm from `t13b_anchor_final` with TWO anchors:
a defense anchor on TIGHT (λ=0.02, as T13b) and a **kickstart anchor** on
W1 (CE to `w1nav_child`, λ=0.05→0.01 over 30M) — the first measured use
of an anchor for *acquisition* in this project. `t11_mixture=3` with
sampling weights set by measured step share (W1 .25 / TIGHT .10 /
wides .65 → step shares .614/.286/.10); K=0 global (the first
honest-acquisition-signal run of this family — the K=120 surrogate was
measured 50–67% optimistic even on E-cells); C filter kernel; 150M in
~4.5 h. Single seed 42, mean-element claims under J2.

## The read

- **The kickstart worked as designed and stopped where predicted.** W1:
  0 → ~30 by 40M (anchor-driven), then plateau 31.5–34.0 through 150M
  after the anchor weaned — the covariate-shift boundary the design
  pre-registered (the teacher's states stop covering the student's own
  failure distribution). Mechanically: the bootstrap saddle f(0)=0 that
  blocked every prior in-mixture attempt is permanently escaped;
  `t15_remix_final` is itself a W1-competent root.
- **The defense held under the heaviest gradient it has faced**: TIGHT
  88.5 (mid) → 85.5 (fin) with W1 owning 61% of gradient steps — a few
  points under T13b's 90.5, far above every unanchored outcome. At the
  training floor: 95.5, matching both T13b seeds. Budget-awareness
  graded (83.0 lean → 97.5 rich).
- **The wides are untouched by the W1-heavy re-mix**: 95.5–99.5, E3's
  best number in any mixture (95.5, truth 92.0).
- Pre-registered next step (not a redesign): **DAgger refresh** — rebuild
  the W1 anchor from the *student's* visited states labeled by the
  teacher, warm from `t15_remix_final`, continue. The probe-level lesson
  repeated: n=20 probes read TIGHT 75–85 while the n=200 batteries said
  88.5/85.5 — probe direction only, never levels.

Checkpoints `models/t3/t15_remix_{mid50M,final}.pt`; anchors
`models/t15/anchor_{w1,tight}_k0.pt`; JSON + probes
`web_data/results/t15_remix/`; campaign
`scripts/orbital/extj2/t15_remix_campaign.sh` (its launch glue died on
first run — four stale template references; the STAGES=none dry-run
under /bin/bash before first launch is now a standing rule).

---

# T15b — DAgger iteration 1: W1 steps to 45, the six untouched (2026-08-22)

**The pre-registered DAgger refresh broke T15's plateau: W1 31.5 → 45.0
(mid 47.0) with the other six flat — TIGHT 86.0 (vs 85.5), wides
94.0–99.0. The iterated-DAgger shape is now measured twice: each refresh
converts the CURRENT failure distribution into supervision, buys a step
(+13.5 this iteration), and saturates as the student's distribution
shifts under it.** 75M from `t15_remix_final`, aggregated 900+900
teacher/student anchor states (teacher label quality MEASURED first:
+0.219 nats entropy on student states, 2.3× more correction — nowhere
near noise), λ_acq 0.20 constant by the λ×CE calculus, `t11_mixture=4`
weights re-solved for the self-attenuation mechanism (W1 decisions/ep
fell 536 → 177 on reaching competence, silently halving its gradient
share at fixed weights — a trap for any long-episode curriculum).
Single seed, mean elements under J2.

Findings carried: (1) the defense anchor is **saturated** — per-step
agreement 0.035 vs a ~10pp closed-loop gap to its teacher means part of
the tight skill lives in recurrent state that per-step CE cannot
supervise; TIGHT's remaining points need reward share or a dedicated
rung, not more λ. (2) TIGHT's mid-run dip (80.5) annealed away by the
final (86.0) — mid-battery dips at active LR are not verdicts. (3) The
tight fuel-floor slice moved −7.0pp (95.5 → 88.5) while native TIGHT was
flat — slice-level noise or a real lean-shift; flagged, not resolved.
(4) Probe levels under-read batteries again (probe band 30–40 vs battery
45–47) — direction only, every time.

Checkpoints `models/t3/t15b_dagger_final.pt`; dataset
`models/t15/anchor_w1_dagger.pt`; JSON `web_data/results/t15b_dagger/`;
campaign `t15b_dagger_campaign.sh`.

---

# T15c/T15d — the DAgger iteration curve (2026-08-23)

**Three iterations of the same mechanical recipe carried W1 from 31.5% to
70.0% while the six other skills stayed flat-to-improving — and the curve's
shape was PREDICTED, not discovered.**

| iteration | W1 fin | yield | TIGHT fin | wides |
|---|---|---|---|---|
| T15 (kickstart) | 31.5 | — | 85.5 | 94.5–100 |
| T15b (iter 1) | 45.0 | +13.5 | 86.0 | 94.0–99.0 |
| T15c (iter 2) | 59.0 | +14.0 | 84.5 | 95.0–99.5 |
| T15d (iter 3) | 70.0 | +11.0 | 87.0 | 96.0–99.5 |
| T15e (iter 4) | **74.0** | +4.0 | 86.0 (**92.5** @floor) | 96.0–99.5 |

Mechanism findings, each measured before it was needed:
- **CE does not forecast yield; novelty does.** Iteration 2's available
  correction halved while its yield held — the falsification that killed
  the "curve is bending" read. Nearest-neighbor novelty of each fresh
  state generation is flat (1.20/1.13/1.13): every refresh reaches a
  roughly constant-size new region and converts a constant slice.
- **The residual is a single mechanism throughout**: 100% of W1 failures
  acquire, keep their fuel, and run out the clock. Each iteration converts
  progressively slower geometries — success cap-usage p90 marched 0.51 →
  0.70 across iterations, predicting frontier exhaustion ≈ iteration 4.
  Cap-raising is falsified as a fix (obs[15] clock rescale = OOD, 43.3 →
  8.3%).
- **Stopping rule pre-committed before iteration 3's number existed**:
  first of yield < +7, W1 ≥ 85, or success cap-p90 ≥ 0.95 (the leading
  trigger — a 7-minute checkpoint measurement instead of a 6-hour run).
- Gate discipline held under pressure: the D2 label-quality floor was
  deliberately NOT retuned when the model behind it died (no observed
  failures to calibrate against; sub-floor = warn, never retune-to-pass).
  The agent also self-caught a median-vs-mean error on the bimodal
  decisions/ep statistic (success ~42 vs fail ~470-600 decisions — step
  share needs the mean).
- TIGHT held 84.5–87.0 across the whole era (defense anchor + 34–35%
  reward share), floor-slice 88.5–90.0; wides never moved.

**THE ARC RETIRED BY ITS OWN RULE (2026-08-23 19:40).** Iteration 4
closed 19.0% of the remaining gap (4.0 of 21.0) — under the pre-committed
20% mechanism-health floor, with the original +7 absolute floor agreeing.
The rule amendment (fraction-closed replaces absolute yield; cap-p90
trigger struck as untriggerable after the teacher was measured flying
inside the same cap) was recorded BEFORE iteration 4's number existed
(`web_data/results/t15e_dagger/rule_amendment.log`). Iteration 4's other
correction: there was never a cap wall — the 91.7% teacher works at
cap-p90 0.682, so the residual was always teachable and the ceiling is
the teacher itself; the constant-fractional-gap-closure model (22.7 →
30.4 → 34.4 → 19.0%) called the landing within a point (predicted 76–77,
mid 74.5, fin 74.0).

**FINAL STATE — one 128-hidden policy, all seven skills** (T15e, 200
eps/cell, seed 123, native BO, mean elements under J2): E0 97.0, E1 99.0,
E2 99.5, E3 96.0 (truth 94.0), LONGRANGE 99.0, **TIGHT 86.0** (92.5 at
its training fuel floor — the original specialist's exact number; lean
83.0 / rich 97.0), **W1 drift-and-wait 74.0** against the specialist
teacher's 91. TIGHT's era trend was gently rising (84.5 → 87.0 mid-era
peak) on reward share alone — the dedicated-rung option retired itself.
For W1-critical operations the mode table remains available
(`w1nav_child` at 91); for everything else the single checkpoint is the
deliverable: **`models/t3/t15e_dagger_final.pt`**.

Mean-element claims under J2. Checkpoints
`models/t3/t15{b,c,d,e}_dagger_final.pt`; JSON
`web_data/results/t15{b,c,d,e}_dagger/`.

**MECHANISM SEED CHECK (2026-08-24, `t15seed_check_campaign.sh`): the
acquisition mechanism is TWO-SEED confirmed.** At seed 7, the kickstart
took W1 0 → 33.5% (seed 42: 31.5) and one DAgger refresh stepped it
+14.0 to 47.5% (seed 42's iteration-2 step: +14.0 exactly), with the six
other cells holding 84–100% at spot precision and the drift-and-wait
fuel signature intact (0.74–0.77×). Shape-reproduction standard,
pre-registered readings, leg-2 dataset built and D-gated at runtime from
the seed-7 student. **The chain's ENDPOINT (74.0) remains single-seed by
design** — replicating it would need the full path-dependent chain, and
the user's own framing settled why that conflates method-reliability
with endpoint reproduction. Open: structural-W1 (teaching the slow
geometries the 91% teacher itself fails) is a future arc; envelope
expansion (high-e, wider Δi) is the next one.
