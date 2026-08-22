# W1xnav: the surrogate, the filter, and the flag that was breaking both

**Bottom line: the acquisition surrogate is fine. `nav_max_ticks=120` was
breaking it — and the filter with it. Both are fixed by one flag, measured.**

## The framing was wrong, and that mattered

The task was posed as "~24 h BLIND windows". A day-warp is not blind.
`_tick3` runs predict -> update -> acq.accumulate -> acq.ready on EVERY tick,
and `nav_max_ticks` turns tau=1440 into `n = min(1440, K)` ticks at
`dt = 1440*60/n`:

| K | ticks per day-warp | dt | real-IOD window `w0 = 2700/dt` |
|---|---|---|---|
| 0 | 1440 | 60 s | 45 obs |
| 480 | 480 | 180 s | 15 obs |
| 120 (shipped) | 120 | 720 s | **4 obs**, against 6 unknowns |

So the variable was never arc length. It was tick spacing.

## 1. Surrogate vs the REAL batch IOD

`RealAcq3D` is a declared drop-in for `AcquisitionSurrogate`, so both arms ran
in the same `OrbitalNav`, same episodes, same seeds, same gate (0.20) and same
2700 s floor — self-red-team (i) satisfied by construction, not by argument.
n = 24 x 2 seeds per cell.

| day-warp dt | 6 h arc | 24 h arc | 48 h arc |
|---|---|---|---|
| 60 s (K=0) | +14.6% | **+0.0%** | **+0.0%** |
| 180 s (K=480) | -- | +20.8% | -- |
| 720 s (K=120) | +60.4% | **+66.7%** | +50.0% |

`both-no` was **0.0% at every single point**: the real solver DOES acquire on
these geometries. Self-red-team (ii) therefore resolves cleanly to "the
surrogate is wrong", NOT "acquisition is impossible at drift-and-wait arcs".
The latter would have demanded an observability-aware policy; it is not what is
happening.

The "unmeasured beyond 6 h" caveat resolves the OPPOSITE way from the fear:
at a proper cadence longer arcs are EASIER (more observations), and the
surrogate is exact at 24 h and 48 h.

The 3.3% anchor did not reproduce on W1 geometry — at 60 s / 6 h we measure
+14.6%. The anchor was taken on a different geometry set; W1's node-dominant
2-5 deg plane gaps are harder. Stated rather than smoothed over.

## 2. The filter (task 2): NOT the binding problem

The estimate was never fragile. Position error IMPROVES across a day-warp
(8298 -> 1675 m through one, conditioned on rows acquired before it), and
acquisition goes 26% -> 100% across the same warp: the warp is where
drift-and-wait acquisition HAPPENS.

What degrades is the covariance — and only at the capped cadence:

| arc | K=120 | K=0 |
|---|---|---|
| 48 h | NEES 1.64, 41.7% out of band, pos 1003 m | NEES 1.28, 25.0%, 336 m |
| 96 h | NEES 6.50, 66.7%, 745 m | NEES 1.15, 6.2%, 125 m |
| **192 h (W1's own ~8-day horizon)** | **NEES 15.26, 81.2%, 729 m** | **NEES 0.91, 6.2%, 76 m** |

(K=120 reproduced across two seeds: NEES 15.26 / 11.03, 81.2% / 81.2%.)

At 60 s ticks, at the full 8-day horizon with 8 day-warps, NEES is 0.91 —
dead centre of the 6-dof band [0.206, 2.408] — with 76 m of position error.
**The "11x overconfident at 24 h" finding is an artifact of the tick cap, not
a property of MSC6J2Cov.** No filter change is needed.

Also relevant and load-bearing: NAV-F dropped the sigma channel from the
observation (obs[29-32] are hard zero), so no covariance reaches the policy.
Even at K=120 the covariance error was a filter-health/Kalman-gain issue
(~5% divergence), not a fictional signal. The fictional signal was the
surrogate's acquisition flag.

## 3. Recommendation: (e), not (a)-(d)

**Train W1 at `nav_max_ticks=0`.** None of the four listed candidates is the
right fix:

- **(a) recalibrate the surrogate** — treats a symptom. The surrogate is
  already exact where the cadence is right, and a correction table would
  encode the tick cap into the signal model forever.
- **(b) real IOD in training** — dominated. It costs 0.84-1.62 s per 3D solve
  (17-90x a training step) to buy a signal that `K=0` provides for free, and
  `K=0` fixes the covariance too, which real-IOD does not.
- **(c) filter changes** — unnecessary; see the K=0 column above.
- **(d) declare out of scope** — unjustified: `both-no = 0%` says these
  geometries are solvable.

**Cost, stated plainly:** the T11 tick-cap measurement put K=0 at 4.4
env-steps/s against 19.8 at K=120 on the mixture (~4.5x). W1 is the worst case
for that ratio because day-warps are its entire strategy. Budget ~4-5x an
equivalent capped rung. The alternative is training on a signal that is wrong
two times in three.

## The self-inflicted part, on the record

`nav_max_ticks=120` came from my own T11 tick-cap work. It was justified by
measuring FILTER divergence across the day-warp (0.003 at K=120 vs 0.109 at
K=30) and by checking that the warm-start roots scored identically. Both
measurements were correct and both were beside the point: they measured the
filter and the policy, and the thing that broke was the SURROGATE — which no
`nav_*` diagnostic reports, because every one of them measures the filter.

W1 then trained inside T11 at K=120 with weight 0.20 and scored 0.0/200. The
recorded post-mortem was "incompetent root AND unverified training signal,
over-determined". The signal half is now measured: it was 50-67% optimistic.
Whether the root half was also binding is exactly what the prepared campaign
separates — and if W1xnav is still ~0 at a validated signal, that is a real
answer that closes the over-determination rather than leaving it open.

## Gates (new, and they are the ones that were missing)

- **W1A** surrogate/real disagreement <= 5% **at the cadence being trained**,
  not at a convenient one. PASS at K=0 (0.0%); would FAIL at K=120 (66.7%).
- **W1B** NEES median in band and out-of-band <= 35% at the trained horizon.
  PASS at K=0 (0.93 median, 12.5%).

Both are pure measurement — no trainer or filter code changed, so every
existing lineage is untouched by construction (self-red-team (iii)).

## Addendum (2026-08-21): the gate this campaign did NOT have

The first launch aborted at the floors stage — `F_root_W1_truth` read **0/200
in 9 s**. The root `j2wait_W1_driftwait.pt` is a 96.0% specialist trained in the
NARROW normalizer family (obs_alt_scale_m 1.6e6) and this campaign runs WIDE
(8.0e6). The GEN_MATRIX normalizer barrier, straight through the middle of my
own campaign. Fixed by transplant (`rescale_ckpt_normalizers.py`,
1.6e6 -> 8.0e6), which reproduces 96/100 truth through this exact harness with
the drift-and-wait fuel signature intact (0.678x vs direct ref); the campaign
now roots at `models/t3/w1nav_root_wide.pt` and the truth floor is LOAD-BEARING
(<50% aborts).

**What I got wrong, precisely.** Stage 0 gated the filter (L2), the acquisition
signal (W1A/W1B), the C kernel (fuzz + mutation battery) and the env anchors. It
gated everything except the thing the whole campaign is built on. And the row
that would have shown it — the root's own home-cell floor — was ADVISORY, so it
printed the failure and streamed past. The campaign specified its CELL
completely, field by field, per the bug-#15 discipline, and specified its ROOT
as a file path.

The irony is worth recording: I wrote the transplant tool, for this barrier,
during T11. Knowing a failure mode is not the same as gating it.

**Standing pattern, now implemented in `root_gate.py`:**

- **R1, static, ~0.1 s.** The wide/narrow families differ by 5x on the
  alt-scaled observation columns, so a net trained against 5x smaller inputs
  carries ~5x larger weights there — directly measurable from the checkpoint
  with no env and no eval:

      j2wait_W1_driftwait.pt  (narrow)           0.5004
      w1nav_root_wide.pt      (wide/transplant)  2.4235   <- 4.84x
      t11_generalist_rungB.pt (wide)             2.1756
      t11t_tight_child.pt     (wide)             2.0359

  4.84 against a theoretical 5.00. **This mismatch was detectable from the file
  alone, before any env was constructed.** Verified both ways: R1 rejects the
  root that failed the launch and names the fix, and passes the transplanted
  one.

- **R2, authoritative, one short eval.** A root with a published home-cell score
  must reproduce it *through the campaign's own harness*, as a HARD gate.
  Measured on the shipped root: 25/25 against a published 96%. R2 generalises
  past this instance — it catches normalizer family, head size, action-space
  width, cell misparameterisation and plain checkpoint corruption without
  needing to know which one went wrong, because it asks the only question that
  matters: does this root still work where it is known to work.

**The rule for future campaign templates: any warm-start campaign must specify
its root's FAMILY — normalizers, head size, action space — as completely as it
specifies its cell, and must run that root through its own harness as a gate,
never as a floor row that prints and continues.**
