# The generalization matrix — six flagship checkpoints × seven regime cells

**Design input for the generalist-mixture rung.** 42 cells + a 20-cell
mode-matched control, 100 held-out episodes each (seed 123), measured
2026-08-15. Script `scripts/orbital/nav/gen_matrix.py`, per-cell JSON in
`web_data/results/gen_matrix/`.

**The one-line read:** the observation *normalizer family* is a harder barrier
than any dynamics difference we have — and the two dynamics barriers we do have
are **one-directional**, so the mixture should be built from the superset
regimes, not the union of all six.

---

## Protocol

`eval_relnav3d.rollout` verbatim — num_envs=1, greedy argmax, LSTM zeroed per
episode, gave-up inits excluded from the denominator. Reusing the published
protocol means the diagonal is directly comparable to each lineage's headline,
which is the only available check that the harness is not itself the story. It
holds: T-BO3 100 (pub ~100), TB5 98 (pub 97), e-E3 94, J2BO-nav 94 (pub 96),
A3b-j2 98, W1 97.

**Action heads are never resized.** `W1-driftwait` carries a 31-row head (row 30
= the 1440-substep day-warp); the other five carry 30. The env's action space is
set from the *checkpoint*, not the cell: 30-head checkpoints in the W1 cell
simply have no day-warp, which is reported rather than engineered around.
Masking row 30 outside its home cell was considered and rejected — masking
measures a crippled policy, and the question is what each checkpoint actually
does when dropped into a foreign regime.

**Modes are labelled.** Two lineages (A3b-j2, W1-driftwait) are truth-state
control policies with no navigation training; two cells (J2-tight, W1-plane) are
truth-only. Everything else runs bearings-only with real batch-IOD acquisition.

---

## Table 1 — success matrix

`*` home cell · `!` OOD (reason listed below) · bearings-only unless noted

| checkpoint | X3-loose | TB5-3D | E1 | E3 | J2X-loose | J2-tight ᵀ | W1-plane ᵀ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **T-BO3** rung-1 nav | **100.0*** | 1.0! | 3.0! | 0.0! | 34.0! | 0.0! | 0.0! |
| **T-BO3D-TB5** tight nav | 99.0 | **98.0*** | 1.0! | 2.0! | 15.0! | 0.0! | 0.0! |
| **e-E3** eccentric nav | 0.0! | 0.0! | 97.0 | **94.0*** | 0.0! | 0.0! | 0.0! |
| **J2BO-nav** J2×nav | 83.0! | 0.0! | 1.0! | 0.0! | **94.0*** | 0.0! | 0.0! |
| **A3b-j2** J2 tight, truth | 91.0! | 59.0! | 0.0! | 0.0! | 100.0 | **98.0*** | 1.0! |
| **W1-driftwait** drift-and-wait, truth | 12.0! | 0.0! | 0.0! | 0.0! | 35.0 | 0.0! | **97.0*** |

ᵀ truth-only cell (all rows). Rows A3b-j2 and W1-driftwait run truth everywhere.

### Row means, with the mode confound controlled

The raw row means are not mode-matched: the two truth lineages see the target
exactly in all seven cells, while the four nav lineages carry real estimation
error in five. So every nav row was re-run in forced truth mode (20 extra
cells) to ask whether the ranking is about generalization or about who was
allowed to see the target.

| checkpoint | all | off-diagonal | all (truth) | off-diag (truth) |
|---|---:|---:|---:|---:|
| T-BO3 | 19.7% | 6.3% | 21.4% | 8.3% |
| T-BO3D-TB5 | 30.7% | 19.5% | 30.9% | 19.3% |
| e-E3 | 27.3% | 16.2% | 28.4% | 16.5% |
| J2BO-nav | 25.4% | 14.0% | 26.6% | 14.3% |
| **A3b-j2** | **49.9%** | **41.8%** | **49.9%** | **41.8%** |
| W1-driftwait | 20.6% | 7.8% | 20.6% | 7.8% |

**The confound is ~1pp.** Handing the nav rows perfect state moves them by
+2.0, −0.2, +0.3, +0.3 pp. A3b-j2's 2.1× off-diagonal lead over the next
checkpoint is a real generalization gap, not an artifact of running truth-mode.

---

## Table 2 — the three most informative failure signatures

Ranked by what they tell you about *mechanism*, not by how bad the number is.
A cause breakdown separates "never arrived" (safety_cap) from "spent the budget
and stranded" from "flew into something" (collision) — three very different
kinds of wrong.

### Signature A — the normalizer barrier has two opposite death modes

| cell | mode | rate | mean len | dv_med | causes | min ρ |
|---|---|---:|---:|---:|---|---:|
| T-BO3 → E1 (narrow policy, wide cell) | bearings-only | 3.0% | 64 | 331 | **collision 25**, safety_cap 68, stranded 4 | 637 km |
| T-BO3D-TB5 → E1 (narrow, wide) | truth | 0.0% | 99 | 402 | **collision 33**, safety_cap 46, stranded 21 | — |
| e-E3 → X3-loose (wide policy, narrow cell) | bearings-only | 0.0% | 148 | 341 | **safety_cap 100** | 186 km |
| e-E3 → X3-loose | **truth** | 0.0% | 155 | 347 | **safety_cap 100** | — |

Both directions read 0%, and they fail in *opposite* ways. Narrow-trained
encoders in wide normalizers act **destructively** — a quarter to a third of
episodes end in collision, with a near-full fuel budget spent. Wide-trained
encoders in narrow normalizers are **inert** — 100% timeout, zero collisions,
never closer than 186 km.

The decisive line is the fourth. Giving e-E3 *perfect truth state* changes
nothing: 0.0%, 100% safety_cap, mean length 155 vs 148, dv 347 vs 341. The
collapse is not the estimator and not the task — E1's task (e_max 0.10) is
barely harder than X3's (0.05) and X3 is strictly easier than e-E3's home cell.
It is the encoder reading observations at a scale it has never seen.

**This is the largest single effect in the matrix.** For T-BO3, adding J2 — a
real change to the physics — costs 51 pp under truth (100 → 49). Changing the
normalizer family, on a *barely harder* task, costs 99 pp (100 → 1).

### Signature B — the day-warp is the mechanism, not a convenience

| cell | rate | mean len | dv_med | causes |
|---|---:|---:|---:|---|
| W1-driftwait → W1-plane (31-head, has row 30) | **97.0%** | **46** | 410 | success 97, safety_cap 2, stranded 1 |
| T-BO3D-TB5 → W1-plane (30-head, no row 30) | 0.0% | **840** | **73** | safety_cap 100 |
| T-BO3 → W1-plane (30-head) | 0.0% | 494 | 200 | safety_cap 100 |
| J2BO-nav → W1-plane (30-head) | 0.0% | 457 | 175 | safety_cap 100 |
| A3b-j2 → W1-plane (30-head) | 1.0% | 561 | **2** | success 1, safety_cap 99 |

Every 30-head checkpoint times out at ~100%, and the *shape* of the failure is
the point: 457–840 decisions spending a median of **2–200 m/s**. They are not
burning the budget and missing — they idle, because no action they own advances
time far enough for differential nodal precession to rotate the relative node.
A3b-j2 is the extreme case: 561 decisions at a median of **2 m/s**, which is a
policy sitting still for 366 hours. W1-driftwait closes the same 2–5°
node-dominant gap in **46 decisions**.

An 18× reduction in decisions at 2–6× the Δv is exactly the drift-and-wait
trade the row was appended for, and no 30-head policy can express it at all.

### Signature C — both dynamics barriers are one-directional

| direction | evidence |
|---|---|
| **tight → loose transfers** | T-BO3D-TB5 (5 km/1 m/s trained) scores **99.0%** at the 30 km loose box |
| **loose → tight does not** | T-BO3 → TB5 **1.0%**; J2BO-nav → TB5 **0.0%** (safety_cap 93, dv 437 — full budget, never converges) |
| **J2 → two-body transfers** | A3b-j2 (J2-trained) scores **91.0%** on two-body X3-loose |
| **two-body → J2 does not** | T-BO3 → J2X **34.0%**; T-BO3D-TB5 → J2X **15.0%** |

The only foreign checkpoint with real tight-box ability is A3b-j2 (59% at
TB5-3D) — and it is the one that was itself tight-box trained. The pattern is
consistent: a policy trained on the harder superset regime retains the easier
subset for free, and the reverse never holds.

---

## The read

**Which regime pairs conflict most.** The sharpest conflict is not a dynamics
pair at all — it is **{X3, TB5, J2X, J2-tight, W1} × {E1, E3}**, the narrow and
wide normalizer families, which are *mutually* near-zero in both directions (max
3.0% either way across ten ordered pairs) and stay near-zero under perfect
state. This is not a capability conflict that rehearsal can average out; it is
two encoders reading different units. Mixing them as-is would spend the whole
rehearsal budget teaching one network two scalings of the same physics. **The
E-ladder must be re-expressed in the narrow normalizers (or all lineages moved
to a single common scaling) before it enters any mixture** — that is a
prerequisite, not a mixture weight.

The genuine dynamics conflicts are milder and, crucially, **nested rather than
opposed**: tight ⊃ loose and J2 ⊃ two-body. Because both are one-directional,
the mixture does not need the subset regimes at all — training on {J2, tight}
already delivers {two-body, loose} at 91–99%. The one axis that is *not* nested
is the plane-gap/day-warp regime: W1-plane is unreachable without row 30, and
W1-driftwait is simultaneously the *weakest* generalist (7.8% off-diagonal, 12%
on plain X3-loose). Drift-and-wait is a genuinely disjoint skill and needs real
rehearsal weight, not free transfer.

**Strongest generalist-adjacent warm start: `extj2_A3b_j2_box5k1` (A3b-j2).**
41.8% off-diagonal against 19.5% for the runner-up, mode-matched, and it is the
only checkpoint scoring usefully in four of seven cells (X3 91, TB5 59, J2X 100,
J2-tight 98). It sits on the correct side of *both* nested barriers — J2-trained
and tight-box-trained — which is precisely why it dominates. Its two failures
are both structural rather than skill deficits: the E cells (wrong normalizer
family) and W1-plane (30-head, no day-warp row).

That makes the recommended rung concrete: start from A3b-j2, expand the head
30 → 31 with `scripts/orbital/extj2/expand_ckpt_30_to_31.py` so the day-warp
exists, restate the E-ladder in narrow normalizers, and weight rehearsal toward
the two regimes A3b-j2 cannot reach for free — eccentricity and drift-and-wait —
rather than toward the loose/two-body cells it already subsumes.

---

## Caveat found while building this

`Orbital.close()` (→ `binding.vec_close`) **segfaults after a multi-episode
rollout** — reproduced in truth, bearings-only/surrogate and bearings-only/real
alike. It is safe on a fresh env, after `reset()`, and after a single `step()`,
so the fault is state-dependent and consistent with a double-free tied to
episode reset or the trajectory log buffer, not with `close()` being broken
outright. No eval stage in `eval_relnav3d.py` calls `close()`, which is why
nothing has hit it before. `gen_matrix.py` therefore does not call it — each
cell is its own process. **Not fixed here** (this task was eval-only and the
fault is in C); flagged for whoever next holds the binding.

## Reproduce

```bash
python3 -c "import sys;sys.path.insert(0,'scripts/orbital/nav');import gen_matrix as G
[print(c,l) for c in G.CKPTS for l in G.CELLS]" | \
  xargs -P 12 -n 2 sh -c 'OMP_NUM_THREADS=1 python3 scripts/orbital/nav/gen_matrix.py \
    --ckpt "$0" --cell "$1" --episodes 100'
python3 scripts/orbital/nav/gen_matrix.py --report
```

Pin `OMP_NUM_THREADS=1`; unpinned parallel cells oversubscribe badly enough to
look like a hang.
