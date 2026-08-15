# MAJOR-10 — is the normal-axis fine burn guidance-critical?

Branch `ext-m10` off `origin/main` @ `420a7b1`. Worktree only, MAIN untouched,
**nothing launched**. Record: `scripts/orbital/nav/m10_gates.txt`.

**Every claim below names the blocked set: `fine_normal` = rows 20/21 = normal
±1 m/s.** MAJOR-10's whole complaint is that an ablation which does not say
which rows it blocked is uninterpretable.

---

## 1. What MAJOR-10 asks, and what was already done

The red-team item has two halves. The **naming** half was already implemented —
`nav_math.ACTION_SETS` carries `fine_inplane` / `fine_normal` / `fine_all` /
`normal_all` / `combined`, and `nav_block_set` selects among them. The
**experiment** half was not:

> *"A `T-BO−normal` arm therefore needs a truth control at the same box, or the
> plane leg's guidance cost will be misread as an information result."*

That control is what this campaign adds. The row indices are confirmed against
the shipped table: **20 = normal +1 m/s, 21 = normal −1 m/s** (22-25 are normal
±10/±25, which are `normal_all`, not the fine set).

---

## 2. The blocker: the existing interlock cannot produce the required control

The obvious tool was `nav_block_fine_below_m` + `nav_block_set`. It cannot do
this job, and the reason is invisible without reading `step()`:

```python
if self._block_below > 0.0 and self._nav_mode != 'truth':
    actions = self._apply_action_ablation(actions)
```

**The ablation is skipped entirely in truth mode.** Measured at TB5-3D,
blocking `fine_normal`, 12 episodes:

| nav mode | mechanism | intercept rate | success |
|---|---|---|---|
| bearings_only | `nav_ablate_rows` | 0.8008 | 2/12 |
| bearings_only | `nav_block_fine_below_m` | 0.8008 | 2/12 |
| **truth** | `nav_ablate_rows` | 0.8392 | **4/12** |
| **truth** | `nav_block_fine_below_m` | **n/a** | **12/12** |

`_d_block` is not even allocated in truth mode (`step()` returns before
`_nav_alloc`). Had the campaign used it, the truth control would have shown
**no deficit at all**, and MAJOR-10 would have been closed with precisely the
misreading it warns against, inverted — *"no truth effect ⇒ the normal axis is
an information result"*.

The interlock is a **navigation** interlock ("don't make fine burns when you
cannot see"), which is meaningless when you can always see. Right for NAV-F,
wrong for MAJOR-10.

**Added instead:** `nav_ablate_rows` (wrapper-side, default `''` = off). The
named rows coast, **unconditionally, in every nav mode, at every separation**.
In bearings_only it reproduces the old mechanism exactly (both 0.8008), so it
is a faithful generalisation rather than a different intervention.

---

## 3. Two masking mechanisms, and why both are needed

They answer different questions and neither substitutes for the other.

| | mechanism | used for | why |
|---|---|---|---|
| **eval-side** | `--mask-rows` — logits to −inf at sampling; the head is **never resized** (the j2wait lesson) | the **floor** | the policy was trained *with* the rows; the honest counterfactual is "do your best without them" |
| **training-side** | `--env.nav-ablate-rows` — the rows coast | the **treatment** | the question is what is *achievable* without the axis, not what this policy degrades to |

The gap is measured, not asserted. Used as a floor, the env ablation makes the
policy **spin on a dead row** — 80.8% of its decisions intercepted, 1064
decisions/episode against a 70 baseline — which conflates "lost the axis" with
"wasted the episode":

| floor mechanism | bearings_only | truth |
|---|---|---|
| sampling mask | 18/40 | 20/40 |
| env ablation | 13/40 | 13/40 |

Both are reported. For the **trained** arms the two must *converge* — a policy
trained under the ablation should have learned not to emit an inert row — and
that convergence is a gate: stages 2/3 report `nav_ablate_rate`, which should
fall toward 0.

Note the 0.80 intercept rate is **not** the baseline usage share. Measured
directly on the unablated policy, rows 20/21 are **8.7%** (truth) / **10.1%**
(bearings_only) of decisions — a real, used axis, but not a dominant one.

---

## 4. The measurement: cross-track decomposition

Success rate alone cannot separate "lost the plane axis" from "got worse", so
every eval now also decomposes the **in-box relative velocity** along the
target's ĥ — exactly what a normal burn controls and nothing else does —
versus in-plane. Zero-shot at TB5-3D, 40 eps:

| arm | success | cross-track \|v·ĥ\| | share |
|---|---|---|---|
| bearings_only baseline | 40/40 | **0.432 m/s** | 0.119 |
| bearings_only sampling mask | 18/40 | **3.053** | 0.447 |
| truth baseline | 40/40 | **0.352** | 0.094 |
| truth sampling mask | 20/40 | **2.677** | 0.468 |

Against a **1 m/s** box, removing normal ±1 leaves a **~2.7–3.1 m/s
cross-track residual — with perfect information.** The share of the relative
velocity that is cross-track roughly quintuples.

The truth-mode decomposition required a fix: `_nav_step` never runs in truth
mode, so `_prev_tgt`/`_prev_sat` are never populated and the in-box geometry
would have been **silently absent in exactly the arm where the
guidance-vs-information question is decided**. It now falls back to the C env's
own `get_state()`, which *is* truth by definition.

---

## 5. Gates — 8 PASS

| gate | result |
|---|---|
| **0 harness anchor**, 200 eps, TB5-3D | **PASS** — plain / truth / recon all 194/200, md5 `721def1b9d72`, the published value |
| 1 `nav_ablate_rows` off is bitwise-inert | PASS in **both** nav modes, md5-identical |
| 2 the ablation fires in truth mode | PASS — 0.8392 where the old interlock is `n/a` |
| 3 the intervention touches a real axis | PASS — 8.7% / 10.1% of baseline decisions |
| 4 the NAV-F pattern reproduces zero-shot | PASS — truth 100% → 50% |
| 5 the deficit tracks the velocity tolerance | PASS — much smaller at TB4's 2 m/s than TB5's 1 m/s |

**One regression this caught:** `stage_anchor` passes a plain `Orbital` to
`rollout`, and the original in-box block relied on short-circuit ordering
(`_prev_tgt is not None and env._dim3`) to never touch `_dim3` on it. My first
draft of the truth-mode fallback put `env._dim3` first and broke the anchor
with an `AttributeError`. Both new accesses now use `getattr` defaults — which
is why the anchor is a mandatory gate and not a formality.

---

## 6. Self-red-team

### (a) What NAV-F's `T-BO−act` taught, and whether it repeats

NAV-F blocked the **in-plane** fine burns below 10 km and found native 92.5%
and — decisively — **truth 94.0% against T-BO's 97.0%**. The deficit surviving
into truth mode is what proved fine burns *guidance-critical* rather than
*info-critical*.

The pre-registered expectation here is the same pattern on the normal axis, and
**this prep already measured it zero-shot**: bearings_only 40/40 → 18/40, truth
40/40 → 20/40. **The truth deficit is not smaller than the native one.**

If the trained arms reproduce that, the normal axis is guidance-critical and
N3D-B §3.3's "3D observability treatment" framing does **not** survive as an
*information* claim at this box. The campaign is built to be able to say the
opposite too: if the truth arm recovers to baseline after training while the
bearings-only arm does not, the deficit **is** informational. That is the point
of stage 3 being a required control rather than an extra.

### (b) Why masking during training *and* at eval are both needed

Covered in §3 with numbers. In short: eval-side masking measures **dependence**
of an existing policy; training-side ablation measures **achievability** without
the axis. Using only the first would confuse "this policy leaned on rows 20/21"
with "no policy can do without them". Using only the second would leave no
zero-shot floor to compare against, and — because the env ablation makes an
untrained policy spin on a dead row — would inflate the apparent deficit by
~5 pp (13/40 vs 18/40) for a reason that has nothing to do with the plane axis.

---

## 7. ETA and launch

Measured per-cell (bearings_only uses `--acq real`, which is the expensive
part; truth cells are seconds):

| cell | s/episode | 200 eps |
|---|---|---|
| bearings_only baseline @TB5 | 7.0 | ~23 min |
| bearings_only masked @TB5 | 4.5 | ~15 min |
| bearings_only ablated @TB5 | 5.5 | ~18 min |
| TB4 equivalents | ~3 | ~10 min each |
| truth (any) | <0.1 | <1 min |
| **50M nav training arm** | — | **~99 min** (nav lineage measured) |

| stage | content | ETA |
|---|---|---|
| 0 | anchors | ~5 min |
| 1 | 12 floor cells (6 bo + 6 truth), 2 boxes | ~1.5 h |
| 2 | train + 6 eval cells | ~2.5 h |
| 3 | train + 4 eval cells | ~2.1 h |
| **total `0,1,2,3`** | | **~6–7 h** |

Timings were taken while the machine was under load (the impl agent's sweep),
so they are upper bounds.

```bash
# from an INTERACTIVE session shell on MAIN, not from an agent
nohup caffeinate -is bash scripts/orbital/nav/m10_campaign.sh \
    > /tmp/m10_stdout.log 2>&1 &
tail -f /tmp/m10_progress.log

# knobs: M10_STAGES=0,1,2,3   M10_STEPS=50000000   M10_SEED=42
#        M10_SEED=7 suffixes data dirs / wandb groups / trained-row JSONs
#        stage 1 alone (M10_STAGES=1) is eval-only and answers the zero-shot
#        question in ~1.5 h if the training budget is not available yet
```

---

## 8. How to read the result

The matrix that decides MAJOR-10:

- **stage 3 (truth, trained ablated) stays well below the `F_base_tr`
  baseline** ⇒ the normal axis is **guidance-critical**; N3D-B §3.3's
  observability framing does not survive as an information claim at this box.
- **stage 3 recovers to baseline while stage 2 does not** ⇒ the deficit **is**
  informational, and the observability framing stands.

Report the cross-track decomposition alongside either way — success rate alone
cannot tell them apart, which is the methodological point MAJOR-10 was making
when it said *"the detection metric must be counterfactual information gain per
burn, not action-mix"*.
