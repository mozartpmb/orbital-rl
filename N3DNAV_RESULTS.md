# 3D-nav rung 1 — bearings-only navigation at the X3 rung

**The combined capability: 3D guidance (1° plane envelope) flown on an
angles-only estimate with real batch initial-orbit-determination — the
flagship extension both prior campaigns (ext-3d, ext-nav) were building
toward.**

Campaign 2026-08-13, branch `ext-3dnav`. Design gates: n3d_REDTEAM.md (4
BLOCKERs closed pre-training, Phase A), harness anchor (19-slot estimated-obs
layer byte-identical to the C env at zero estimation error — the campaign
script re-proves this at stage 0 and aborts on failure).

## Rung

X3 exactly as `t5_x3_seeds.sh` scored it: `dim3_mode 1`, `di_max_rad` 1.0°,
`e ≤ 0.05`, LEO, `init_phase_gap_max π`, box **30 km / 50 m/s**, shaping_mode 2,
Discrete-30, no debris. Same box as the 2D nav lineage, so NB1 (98.0–99.5%) is
a like-for-like comparator. Warm start for all trained arms:
`models/t3/n3dnav_warm_X3.pt` = `seed42_X3_3d_di1deg.pt` with encoder columns
29–32 (hard-zeroed obs, zero-gradient, still random init) zeroed — verified
bit-identical action stream to the parent (md5 `62f66b07eadf`).

All evals: 200 episodes, held-out seed 123. "Native" = the arm's own nav mode;
bearings-only arms use the REAL batch IOD (`bls_acquire_adaptive3`, geodesic
residual, 3-axis velocity lattice), not the training surrogate.

## The table

| arm | trained on | native eval mode | native | truth (same ckpt) |
|---|---|---|---|---|
| **T-truth3d** (floor, eval-only) | — (truth X3 ckpt) | bearings-only, real IOD | **116/200 = 58.0%** | 200/200 |
| **N1-rb3d** (control) | range+bearing estimates, 50M | rb_ekf | 200/200 = 100.0% | 200/200 |
| **N1-rb3d cross-mode** (addendum) | ″ | bearings-only, real IOD | **121/200 = 60.5%** | — |
| **T-BO3** (treatment) | bearings-only estimates, 50M | bearings-only, real IOD | **200/200 = 100.0%** | 200/200 |

## Verdict

**Bearings-only-in-the-loop training closes the entire 42-point gap at the
first 3D-nav rung — 58.0% → 100.0% — with zero truth-tax, and the attribution
is airtight.**

1. **The floor**: the truth-trained 3D policy is worth 58.0% when the target
   state stops being given to it (causes: collisions + strandings — it
   maneuvers on garbage during the blind window).
2. **The control**: with range measured, flying on estimates is lossless
   (100.0%) — estimates per se have no deficit. And its checkpoint flown blind
   collapses to 60.5% ≈ the floor, with the floor's failure signature
   (67 collisions, 9 stranded) and 29% more Δv (317 vs 246 m/s median):
   50M steps of estimate-training with range measured transfers **nothing**
   to the bearings-only problem. This is the 3D replica of NAV-F's T-RB null.
3. **The treatment**: T-BO3 flies the full rung at 200/200 on the real
   acquisition pipeline — every episode acquires at exactly the 46.0-min
   sim-time floor (p90 = median = 2760 s), 0 solver failures, epoch error
   median 10.1 km, Δv median 246 m/s, truth-mode also 200/200 (**zero
   truth-tax in both directions**).
4. Consistent with NAV-F (2D, tight box): the learned skill is disciplined
   blind-window behaviour and acting on a converged estimate — here it is
   sufficient for a perfect score at the 2D-lineage box even with the 3D
   plane channel in the loop.

## Rung 2 — the tight boxes (2026-08-13, campaign `n3dnav_tb_campaign.sh`)

The NAV-F regime in 3D: boxes where navigation actually binds. Same X3 3D
settings with `rendezvous_radius_m`/`rel_vel_tol_ms` overridden per box;
warm start `models/t3/n3dnav_warm_TB5.pt` derived from
`seed42_TB3D_box5k1.pt` (columns 29–32 only; verified bit-identical action
stream, reproduces the parent's published 194/200). Shaping inherited from
the TB3D lineage (`shape_w_match 0.35`), NOT the rung-1 value — each arm
trains against its own parent's reward. Stage-0 anchor re-proven at the
tight box before any arm ran.

| arm | TB5-3D (5 km / 1 m/s) | TB4-3D (5 km / 2 m/s) |
|---|---|---|
| **Floor** — truth ckpt flown blind | 92/200 = **46.0%** (truth 194/200 = 97.0%) | 107/200 = **53.5%** (truth 200/200) |
| **Control** — trained on rb_ekf @TB5 | 99.0% in-mode · truth 99.5% · **blind 26/200 = 13.0%** | 100% in-mode · truth 100% · **blind 36/200 = 18.0%** |
| **Treatment** — trained bearings-only @TB5 | **196/200 = 98.0%** · truth 199/200 = 99.5% | **199/200 = 99.5%** · truth 200/200 |

Verdict, three parts:

1. **The gap closes entirely at the tightest box** — 46.0% → 98.0% on the
   real IOD, and the treatment's truth-mode (99.5%) *exceeds* its parent's
   97.0%: bearings-only training at TB5-3D improved the guidance skill it
   was warm-started with. Failure census across all 400 treatment
   bearings-only episodes: 1 collision, 3 safety-cap, 1 cap (TB4).
2. **Negative transfer, measured twice.** The control is lossless in its own
   mode but collapses to 13.0%/18.0% flown blind — far BELOW the untrained
   floor (46.0%/53.5%). At tight tolerances, training on always-reliable
   estimates actively unlearns blind-window caution. Estimates-per-se are
   not merely insufficient (rung 1's 60.5% ≈ floor); at rung 2 they are
   harmful. The attribution to observability structure could not be sharper.
3. **Navigation is over-solved at capture; guidance is the binding
   constraint.** Inside the 5 km box the treatment's estimator velocity
   error is 0.001 m/s (p90 0.005) against the 1 m/s tolerance — three
   orders of magnitude of margin — while in-box |v_rel| medians 3.2 m/s
   until the capture maneuver. n3d_B's "the 3D dividend lands on velocity"
   framing is moot at this box: there is no velocity-information deficit to
   buy back. What bearings-only training buys is guidance discipline under
   the estimate stream, and that is sufficient — the same mechanism NAV-F
   identified in 2D, confirmed at the tightest 3D boxes.

## Multi-seed — rung-1 treatment replicated 3/3

T-BO3 at X3 (loose box), seeds 7 and 1337, same warm start and config as
seed 42: **both 200/200 native AND 200/200 truth.** With the original run,
the rung-1 result is 3/3 seeds at 100% — including seed 1337, the seed that
failed Phase 4's Stage-1 fresh training outright. Checkpoints
`models/t3/n3dnav_T-BO3-X3-s{7,1337}.pt`.

## Multi-seed — tight-box treatment replicated 3/3 (stage 5)

T-BO3D-TB5 re-trained from the same warm start at seeds 7 and 1337:

| seed | TB5-3D bo | TB5-3D truth | TB4-3D bo | TB4-3D truth |
|---|---|---|---|---|
| 42 | 196/200 = 98.0% | 99.5% | 199/200 = 99.5% | 100% |
| 7 | 196/200 = 98.0% | 99.0% | 199/200 = 99.5% | 100% |
| 1337 | 197/200 = 98.5% | 98.5% | 198/200 = 99.0% | 100% |

**Pooled: 589/600 (98.2%) at TB5-3D, 596/600 (99.3%) at TB4-3D; spread
0.5 pp.** The tight-box number is a capability, not a seed artifact —
and seed 1337 (which failed Phase 4 Stage-1 fresh training outright) is
indistinguishable from the others here. Checkpoints
`models/t3/n3dnav_T-BO3D-TB5-s{7,1337}.pt`.

## Rung 3 — the eccentricity ladder (2026-08-14, `n3dnav_e_campaign.sh`)

Extends the flagship along eccentricity. Two prep findings shaped the design:
the `e_max` knob is **inert at the nav altitude band** (perigee-keepout
rejection caps realized e ≈ 0.03 regardless of setting — the Phase 5.5
"eccentricity cliff" failure class), so rungs widen the altitude band per the
validated V-ladder tuples and re-root the warm start in the wide-normalizer
lineage (`seed42_V2_wide3d.pt`, columns 29–32 zeroed, bit-identical to parent).
**Realized e runs ~0.42× the cap** (E3: mean 0.126, p90 0.257) — the cap is a
setting, the realized distribution is the result. Stage-0b gated
surrogate-vs-real acquisition agreement at E3 before any training (PASS).
Chain: E0←warm, E1←E0, E2←E1, E3←E2; `shape_w_match 0.35` (V-ladder parent's
own value). Bearings-only rows use the real batch IOD with the widened
velocity bracket (1.15·e_max); `nav_r_min_m` floored at the keepout shell
per-campaign (the wrapper default would silently move published streams).

| rung | e cap (realized mean / p90) | floor blind / truth | trained BO | trained truth | retention (BO, rung below) |
|---|---|---|---|---|---|
| E0 | 0.05 (0.023 / 0.042) | 70.5% / 100% | **200/200** | 200/200 | — |
| E1 | 0.10 (0.041 / 0.081) | 70.5% / 96.0% | **198/200 = 99.0%** | 200/200 | 99.5% |
| E2 | 0.20 (0.085 / 0.166) | 19.5% / 24.0% | **191/200 = 95.5%** | 200/200 | 99.5% |
| E3 | 0.30 (0.126 / 0.257) | 5.5% / 6.0% | **190/200 = 95.0%** | 200/200 | 97.5% |

Verdict: **bearings-only rendezvous holds at 95.0–100% across a realized
eccentricity distribution reaching p90 ≈ 0.26** — from floors as low as 5.5%.
The floor structure splits into two regimes: at E0/E1 the deficit is
navigation (truth floors 96–100%); at E2/E3 the untrained deficit is
capability itself (truth floors 24%/6%), and the ladder builds both at once —
trained truth rows are 200/200 at every rung, so the residual 4.5–5 pp
bearings-only gap at E2/E3 is pure navigation under eccentric geometry.
Retention never drops below 97.5%, and E3 *improved* E2's own score.
Checkpoints `models/t3/n3dnav_e_E{0..3}.pt`.

## MAJOR-10 — the normal-axis ablation (2026-08-15, `m10_campaign.sh`)

The last open red-team item: is the fine normal axis (±1 m/s, rows 20/21) an
*information* instrument (N3D-B's observability framing) or a *guidance*
requirement? Design: masked zero-shot floors + trained-ablated arms with a
truth control, at TB5-3D and TB4-3D. Instrument note: the pre-existing
`nav_block_*` interlock silently no-ops in truth mode — using it would have
fabricated a "no truth deficit" reading in the deciding control arm; the
campaign added an unconditional `nav_ablate_rows` instead (bit-inert off).

| condition | TB5-3D (1 m/s) | TB4-3D (2 m/s) |
|---|---|---|
| baseline (bo / truth) | 98.0% / 99.5% | 99.5% / 100% |
| **masked zero-shot** (bo / truth) | **51.0% / 56.0%** | 81.0% / 82.5% |
| trained-ablated, bearings-only | 0–0.5% | 0–1.0% |
| trained-ablated, TRUTH control | 0–0.0% | 1.0–1.5% |

Verdict, two parts:
1. **Guidance-critical, not informational.** The zero-shot truth deficit
   (43.5 pp) equals the native one (47 pp), the deficit scales with
   velocity-box tightness (TB4 loses only ~18 pp), and the mechanism is
   physical: without normal ±1 the cross-track velocity residual is
   2.7–3.1 m/s against a 1 m/s tolerance *with perfect information*.
   N3D-B's "2.5–8× more information than prograde" framing does not survive
   as an information claim at this box.
2. **Training cannot route around it — it collapses instead**, identically
   under bearings-only and truth (rolling perf 0.000/0.003 from warm starts
   that scored 51–56% masked). Consistent with the W4/R-literature
   value-collapse mechanism: the ablated task carries large physically-
   infeasible mass at the 1 m/s box, and infeasible episodes shift the
   optimum rather than adding noise. The collapse is itself the third
   replication of that mechanism in this project.

With NOTE-22's calibration pass, the n3d_REDTEAM ledger is now fully closed
(4 BLOCKERs, 13 MAJORs, 4 NOTEs — every item either fixed, measured, or
closed with a campaign). JSON `web_data/results/m10/`; wandb `t6-m10-*`.

## Caveats
- Both boxes share eval seed 123 (paired scenarios across arms — valid for
  arm-vs-arm deltas).
- The IOD acceptance gates still certify large epoch errors (max 465.9 km
  here; red-team MAJOR-2) — latency and success are honest, the "acquired"
  label is not a precision claim.
- N1-rb3d native used the surrogate acquisition (as trained); its cross-mode
  row used the real IOD, same as the other bearings-only rows.

## Provenance

- Campaign: `scripts/orbital/nav/n3dnav_campaign.sh` (stage-0 harness anchor,
  per-stage RESULT lines, trainer-pid watchdog); progress log archived at
  `/tmp/n3dnav_campaign_progress.log` during the run.
- One infrastructure incident: the first launch (21:42) was killed at 22:42
  with the spawning agent's process tree — machine did not sleep, no crash;
  relaunched 23:20 from the session shell; floor eval reproduced
  **bit-identically** (md5 `984582acd215` both runs).
- Checkpoints: `models/t3/n3dnav_{warm_X3,T-truth3d,N1-rb3d,T-BO3}.pt`;
  per-arm JSON: `web_data/results/n3dnav/`; wandb groups `t6-n3dnav-*` in
  `orbital-rl`.
- Phase A/B engineering (getter, 19-slot layer + leakcheck, MSC pole fix,
  analytic STM, sim-time acquisition floor, 3D batch IOD corrections):
  commits `61e7bf8..69ebeb3` on `ext-3dnav`; red-team report
  `scripts/orbital/ext_recon/reports/n3d_REDTEAM.md`.
