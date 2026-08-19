# Hold-duration eval — does the tight box *hold* after capture?

**Verdict: no. Both tight-box flagships capture reliably (96–98%) and then leave.
Continuous residency falls to 40.9–43.9% by 15 minutes and 21.5–21.6% by 30. The
failure is unanimous and one-sided: `left_pos = 0` in every cell of both arms —
position is held, relative *velocity* is what leaves.**

Measured 2026-08-18 · `scripts/orbital/nav/hold_eval.py` · 200 held-out episodes
per arm at seed 123 · per-arm JSON in `web_data/results/hold_eval/`.

---

## Why this exists

Every success number this project publishes is an **instant**. The C env
terminates the moment position and relative velocity are simultaneously inside
the box (`orbital.h:1825`), and nothing measures the next minute. A policy that
crosses the box at speed and coasts out the far side scores identically to one
that parks. For a rendezvous criterion that is the difference between a capture
and a flyby, so **"98% success" and "98% captured" are not the same claim**, and
only the first has been evidenced until now.

## Method — real continued rollout, not a proxy

The box thresholds enter the C env in exactly one place. They appear in the
struct (`orbital.h:500-501`), the config load (`2012-2013`), the compile-time
defaults (`119-120`) and the `at_target` test (`1825-1826`) — and **nowhere in
the observation encoder or the shaping potential**. So shrinking the box to
1 µm / 1 nm/s makes success unreachable while leaving the policy's inputs
bit-identical: the episode simply keeps running and occupancy is scored offline
against the real 5 km / 1 m/s box. No C change, no proxy.

**Capture is anchored on the env's own criterion, not on my sampling.**
`orbital.h:2504` runs the termination check *every substep* ("so the warp never
skips past a conjunction"), so capture happens at 60 s resolution while a policy
mid-warp only decides every 360 min. Detecting entry on the decision grid
therefore misses real captures — measured directly, one seed whose real-box
episode succeeded at decision 56 was not seen as inside until decision 155. So
each episode runs **twice at the same seed**: once at the real box, which
terminates exactly when the env says captured and yields the capture substep
via `last_episode_result`, and once at the dead box, which supplies what happens
afterwards. The window is anchored on the first run and scored on the second.

## The table

`hold_N` = fraction of captured episodes continuously inside the box (position
**and** velocity) at every decision epoch from capture through N minutes of sim
time. `occ_N` = mean fraction of epochs inside over that window.

| arm | checkpoint | mode | capture | hold 5 | occ 5 | hold 15 | occ 15 | hold 30 | occ 30 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A3b-j2** | `extj2_A3b_j2_box5k1` | truth | 98.0% | **70.0%** | 80.0% | **43.9%** | 48.4% | **21.6%** | 26.1% |
| **TB5-3D** | `n3dnav_T-BO3D-TB5` | bearings-only | 96.0% | **56.1%** | 64.2% | **40.9%** | 44.0% | **21.5%** | 26.8% |

Capture rates reproduce the published headlines (99.0% and 98.0%), which is the
check that the dead-box harness has not changed the task.

### Why it fails

| arm | window | left on position | left on **velocity** | left on both | episode ended |
|---|---|---:|---:|---:|---:|
| A3b-j2 | 5 / 15 / 30 min | **0 / 0 / 0** | 9 / 36 / 75 | 0 / 1 / 16 | 0 / 0 / 0 |
| TB5-3D | 5 / 15 / 30 min | **0 / 0 / 0** | 10 / 51 / 123 | 8 / 17 / 27 | 0 / 0 / 0 |

**Not one episode across either arm left the box on position alone.** The 5 km
position tolerance is held comfortably; the 1 m/s velocity tolerance is not.
That is the same axis `NAVF_RESULTS.md` identified as binding at TB5 and the
same one `N3DNAV_RESULTS.md` summarised as "navigation is over-solved, guidance
binds" — the estimator knows the relative velocity to 0.001 m/s, and the policy
still cannot null it and keep it nulled.

### The policies are not station-keeping at all

Post-capture `ACTION_TAU` histogram — the policy's own decision cadence after it
has been declared successful:

| arm | τ = 1 min | τ = 5 | τ = 30 | τ = 60 |
|---|---:|---:|---:|---:|
| A3b-j2 | 265 | — | 40 | 195 |
| TB5-3D | 160 | 1 | 336 | 1 |

A3b-j2 answers capture by warping an **hour**; TB5-3D by warping **30 minutes**.
Neither is holding a station — they coast, and the box is a place they pass
through. This is the mechanism behind the hold numbers, and it is not a defect
in the policies: nothing in the reward ever asked them to stay. The episode ends
at capture, so there has never been a gradient pointing at the next minute.

---

## Instrument limitations, stated

**1. `unresolved` epochs.** Because the post-capture cadence is 30–60 min, many
episodes have no decision boundary inside the shorter windows. Those are scored
**`unresolved` — never as held or not-held** — and excluded from the denominator,
because the instrument genuinely has nothing to say there and rounding either way
would manufacture a number.

| arm | 5 min | 15 min | 30 min |
|---|---|---|---|
| A3b-j2 | 30 scored, 166 unresolved | 66 / 130 | 116 / 80 |
| TB5-3D | 41 scored, 151 unresolved | 115 / 77 | 191 / 1 |

The 5-minute column rests on 30–41 episodes and should be treated as indicative;
**the 30-minute column is the solid one** (116 and 191 episodes scored), and it
is also the worst.

**2. The bearings-only arm's anchor is only partly matched.** The harness checks,
on every episode, that the dead box did not change behaviour by comparing the two
passes' pre-capture action streams. A3b-j2 (truth) passes **200/200 — exact**.
TB5-3D passes **147/200**. The cause was found by that check: the nav wrapper
seeds its measurement RNG once at construction and advances it per step, so a
pass that terminates at capture and a pass that runs past it desync. Re-seeding
`_rng` per episode lifted the match from 90/200 to 147/200 and moved the headline
materially (hold-5 42.1% → 56.1%), so it was load-bearing. The residual 53
episodes carry an anchor from a slightly different noise realization. That is a
real caveat on the TB5-3D row — and the reason to trust the qualitative result
anyway is that the **exactly-instrumented arm (A3b-j2, 200/200) shows the same
decay and the identical `left_pos = 0` signature**.

**3. Truth-mode arms are unaffected by (2)** — they draw no measurement noise,
which is exactly the asymmetry that made the cause identifiable, and A3b-j2
re-ran bit-identical after the fix.

---

## What this means for the criteria

The success box as currently defined certifies **arrival, not capture**. If a
future rung wants a capture claim, the criterion has to carry a duration, and
these numbers say where to set it: at 5 minutes the flagships are already at
56–70%, so a hold requirement is a genuinely harder task and not a formality.
The cheapest honest options, in order of cost:

1. **Report capture-plus-hold alongside success** — no retraining, just publish
   this table beside the headline. It narrows the claim to what is evidenced.
2. **Extend the episode past first entry and reward residency** — the real fix,
   and it needs a fresh-ish rung: `R5` established that reward-reshaping a
   committed policy collapses it, so this cannot be bolted onto these
   checkpoints.
3. **Tighten the velocity tolerance instead** — cheaper, but it attacks the
   symptom. The measurement says the box is not being held because the policy
   leaves, not because 1 m/s is too loose.

## Reproduce

```bash
for A in TB5-3D A3b-j2; do
  OMP_NUM_THREADS=1 python3 scripts/orbital/nav/hold_eval.py --arm $A --episodes 200 &
done; wait
python3 scripts/orbital/nav/hold_eval.py --report
```
