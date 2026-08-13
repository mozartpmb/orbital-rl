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

## Caveats

- Single training seed (42) per arm; single rung (X3, 30 km / 50 m/s). The
  2D lineage's tight boxes (TB4/TB5) with the plane channel are the natural
  next rung and NOT covered by this table.
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
