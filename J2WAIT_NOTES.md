# ext-j2wait — drift-and-wait: trading time for fuel with J2

Branch `ext-j2wait` off `origin/main` @ `7148b21`. Worktree only, MAIN untouched,
**nothing launched**. Records: `scripts/orbital/extj2/j2wait_gates.txt`.

The experiment: when the **node-dominant plane gap** exceeds what the Δv budget
can buy directly, does the policy dip Δa, let differential nodal precession
rotate the relative node, and come back — instead of burning?

---

## 1. Design decision: a new action row, not a rebind

The prior was to rebind row 17 (τ 360 → 1440) under a flag, on the theory that
the 6 h warp is rarely used. **The measurement says the opposite.** Action usage
over 100 held-out episodes of the lineage this campaign warm-starts from:

| lineage | row 17 (τ=360) | rows 16+17 share of **sub-steps** |
|---|---|---|
| **A2, loose box** (the warm start) | **11.1% of decisions** (3rd most-used row) | **87.4%** |
| A3b, tight box | 2.3% of decisions | 36.7% |

Row 17 is the warm start's **primary time-advance tool**. Rebinding its τ would
quadruple the effect of 11% of the policy's decisions while leaving its logit
untouched — a silent semantic swap under a trained head, which is this project's
defect class #1. Rejected on evidence.

Rebinding a genuinely dead row (7, 8, 13, 18, 19 are 0% in both lineages) was
the fallback, but it rests on a property of *these checkpoints* rather than of
the env, and permanently misnames the row.

**Chosen: append row 30 (τ=1440), `NUM_ACTIONS` 30 → 31.** The cost the prior
was avoiding turns out to be near-zero here:

- rows 0-29 keep their indices and meanings **bit-for-bit** — this is a pure
  append, with none of the remapping `expand_ckpt_actions_7_to_9.py` needed;
- the head surgery is one appended decoder row
  (`scripts/orbital/extj2/expand_ckpt_30_to_31.py`), **verified**: the expanded
  checkpoint with row 30 masked reproduces the parent's action stream with an
  identical md5 (`4cb05cd630e4`, 725 decisions);
- **no new kwarg.** The exposed space already defaults to `Discrete(16)`; the
  day-warp is unreachable unless a lineage opts in with
  `legacy_action_space=31`. Same gating that introduced rows 16-19 and 20-29.
  An unreachable row is already off.

### 1.1 A third env change, required by the scenario spec

The brief said "nothing else", but the scenario it specifies —
*node-dominant* Δi_rel ~ U(2°,5°) — is not reachable with the shipped sampler,
which draws δ = di_max·√U (area-uniform, 16% of a 0-5° band below 2°) with the
rotation axis **uniform** in the target plane. Two knobs, both default-off and
RNG-stream-inert:

- `di_min_rad` — δ ~ U(di_min, di_max), uniform **in angle**;
- `di_phase_mode=1` — the rotation axis within 30° of the node axis.

The basis the sampler already builds makes this one line: û₁ = ẑ × ĥ_t **is** the
line of nodes, so rotating about û₁ (φ=0) tilts the plane about the node line —
a pure *inclination* error, which drift cannot touch — while rotating about
û₂ (φ=±90°) moves the node. Measured node fraction **p05 0.864, p50 0.968**
against the uniform-phase draw's E|cos φ| = 2/π = 0.637 (and ~0 sometimes).
Without it the arm would mix "the policy failed to drift" with "there was
nothing to drift for".

### 1.2 The appended row must be SEEDED, not zero-init — a blocker caught by measurement

The obvious init for an appended row is zero. **Measured, that would have made
the whole campaign vacuous.** The warm start is saturated (median argmax softmax
probability **0.986**), so a zero logit among trained ones gets:

| row 30 init | P(row 30) median | P(row 30) mean | expected samples per 100M decisions |
|---|---|---|---|
| **zero** | 3.9e−09 | — | **0** |
| **copied from row 17** | 5.1e−05 | 5.17e−02 | **5,172,304** |

With zero-init the day-warp would never be sampled, the arm would burn ~4 h of
compute, and **the mechanism under test would silently never have been
available** — a null result indistinguishable from a real one.

Row 30 is therefore seeded from **row 17** (the 6 h warp). That starts the two
as an exact tie at precisely the states where the policy already chooses to
warp, so PPO samples each ~50% of the time there and learns to separate them by
their τ. It is also the correct prior on the merits: the day-warp *is* "warp,
but longer", so inheriting the 6 h warp's learned *when-to-warp* representation
is the right starting point. Masked-row-30 equivalence to the parent is
**unaffected** (still md5-identical), and gate **A5g** now fails the campaign if
the row is ever unexplorable again. `--zero` is retained only so the measurement
reproduces.

---

## 2. Gates

| gate | result |
|---|---|
| a1-a4 full anchor battery (legacy / T3 / X3, + lvlh1, + sampler-off) | **18/18**, all md5s identical (`f8a2388f0992`, `68b267bed369`, `003105f29898`) |
| A5a exposed space: default 16, opt-in 31, 32 rejected | PASS |
| **A5b row 30 (τ=1440) ≡ 1440 coast sub-steps, BITWISE** | **PASS**, max \|Δ\| = 0.000e+00 over the 36-element state |
| A5c cross-check: 4 × row 17 ≡ 1 day of coasting, bitwise | PASS, 0.000e+00 |
| A5d do-nothing gain over a full 22000-step cap | **+0.665** = 7.2% of the discounted terminal (gate 25%) |
| A5e plane band + node dominance | realized 2.001-5.000°, node fraction p05 0.864 / p50 0.968 |
| A5f `di_min_rad` / `di_phase_mode` present-but-off are a no-op | PASS, 400 resets bit-identical |
| **A5g the day-warp row is EXPLORABLE under the warm start** | **PASS**, 5.17M expected samples/100M (zero-init measures 0) |
| head expansion 30 → 31, row 30 masked ≡ parent | PASS, md5 `4cb05cd630e4` both |

**25/25** across `a1,a2,a3,a4,a5`.

`NUM_ACTIONS` 30→31 and `MAX_STEPS` 12000→22000 both change the translation
unit under `-O2 -flto`, so the anchors were re-run after each — that is the
n3d_REDTEAM MAJOR-6 discipline, and it held.

### 2.1 Memory: the buffer stays virtual

`sizeof(TrajectoryRecord)` = 408 B, so `traj_log` goes 4.90 → **8.98 MB/env**
(+4.08 MB virtual). Measured with `log_enabled=0`, N=256 envs (≈2.3 GB of
virtual traj buffer):

```
baseline RSS                                    215.2 MB
after Orbital(N=256, cap 22000, log_enabled=0)  223.7 MB      (+8.5 MB)
after 200 steps                                 223.7 MB      (+0.0 MB)
```

**RSS delta ≈ 8.5 MB total, ~33 KB/env** — the non-`traj_log` part. The pages
are never faulted in because nothing writes them when logging is off.

---

## 3. The floor — measured, and richer than expected

The J2-trained A2 policy, head-expanded to 31, zero-shot at U(2,5) node-dominant
gaps, cap 22000, day-warp available, 60 episodes:

| quantity | value |
|---|---|
| success | **0/60** |
| causes | **59 safety_cap, 1 stranded** |
| median Δv spent | **50.0 m/s** of a 478 m/s budget |
| median episode | **366.7 h** (the full cap), **544 decisions**, **0 day-warps** |
| plane bought by **impulses** | **+0.154°** |
| plane bought by **precession** | **−1.533°** (it *opens*) |
| incidental dip | 62.3 km deep for 117.3 h |

**It caps out; it does not burn dry** — the question the brief asked. And the
last two rows are the experiment in miniature: the policy already dips (for
*phasing* reasons) and differential precession is already acting, but nothing
steers the node, so the mechanism runs backwards and costs it 1.5° of plane.
Training has to aim an effect that is already switched on.

The floor is unchanged by the row-17 seeding (0/60, 59 cap + 1 stranded), as
expected: at an exact logit tie `argmax` resolves to the lower index, so greedy
evaluation still never emits row 30. The seeding changes *exploration*, which is
a training-time property, not the greedy floor.

---

## 4. Self-red-team

### (a) Stall dominance at a 22000-step cap

With `cap_terminal_reward=0` and `shape_gamma=1`, a capped episode banks only
its shaping, and Φ's entire range is `W_λ + W_m = 1.8167`. A success pays
`10·γⁿ`. **Stall wins only if `10·γⁿ < 1.8167`, i.e. n > 340 decisions.**

Measured do-nothing gain over a *full* 22000-step cap, day-warps only, 64 envs
at this scenario distribution: **+0.665** — 95× the 6000-cap value (0.006).
That growth is not a leak: over 15.3 days precession really does close plane
error for free, which is the phenomenon under test. It is unfarmable because
`shape_gamma=1` telescopes (episode shaping total = Φ_T − Φ_0 regardless of
path, so looping buys nothing).

| play | decisions | γⁿ | terminal worth |
|---|---|---|---|
| day-warps (row 30) | 15.3 | 0.9263 | 9.26 |
| 6 h warps (row 17) | 61.1 | 0.7361 | 7.36 |
| 1 h warps (row 11) | 366.7 | 0.1591 | **1.59** |
| **the floor policy, measured** | **544** | **0.065** | **0.65** |

The last row is the finding: **without the day-warp, the floor already runs at
544 decisions/episode where the terminal is worth 0.65 — below the 1.8167 a
stalled episode can bank.** The day-warp is not a convenience; at a 22000-step
cap it is what keeps success worth more than stalling. The flatline tripwire
arms regardless (`--after-steps 40e6`).

### (b) Does Φ reward precession-bought closure identically?

**Yes, structurally.** Φ(mode 2)'s plane term is `Δv_pl = v_t·‖ĥ_s − ĥ_t‖`, a
pure function of **state**. It cannot see what caused the closure, so a degree
bought by waiting scores exactly as a degree bought by burning — while the Δv a
burn consumed is separately visible through the fuel-dependent obs. That
asymmetry *is* the gradient the experiment relies on: equal reward, unequal
cost. Gate A5d re-runs the J-A5-style do-nothing bound at the new horizon and
it stays bounded (§4a).

### (c) Credit horizon at γ = 0.995 **per decision**

A realistic drift-and-wait episode is ~12 day-warps + ~20 dip burns + ~20
endgame ≈ 50 decisions, **γ⁵⁰ = 0.78** — healthy. The same episode on hour-warps
is ~370 decisions and the terminal is worth 1.59, *under* the stall value, so
the credit horizon alone would kill it (see the table above).

**No γ bump is proposed.** The day-warp fixes the horizon at its source, and
raising `gamma_train` for one rung would make this arm's returns incomparable
with every other arm in the lineage — the compound-change failure mode every
T3-era collapse traces to.

---

## 5. Training budget: 100M, not 50M

PPO's budget is counted in **decisions**. The floor spends **544 decisions per
episode**, so 50M steps = **92k episodes** — against the ~2.8M episodes the
X3/A2 arms saw at ~18 decisions/episode. That is **30× fewer task instances**,
for a task with more to discover, not less. If training converges to day-warp
usage the count improves to ~50 decisions/episode (1M episodes at 50M), but the
early warm-start-shaped phase is exactly where the starvation bites.

Override with `J2WAIT_STEPS` if the window is tight.

---

## 6. Smoke, ETA and launch

**2M smoke at cap 22000, Discrete-31, warm-started:** trains cleanly and the
rolling perf moves off the 0.0% floor — **0.000 → 0.005 → 0.085 by 1.8M steps**.
A 10-episode greedy eval at that checkpoint is still 0/10 (consistent with a
~4.5% stochastic policy) but the **decomposition output is populated** on the
all-episodes row: precession −3.143°, impulses +0.455°, Δv 70.5 m/s, 540
decisions. The trainer **hung in teardown after writing its final checkpoint** —
the known behaviour the campaign's watchdog exists for.

**Throughput was measured under heavy contention.** Six `calib_triad.py`
processes were consuming ~1300% CPU (load average 67) for the whole smoke and
the trainer got ~35% of one core; the load cleared to 3.8 immediately after. At
**SPS ≈ 6.5K decisions/s** under that load:

| steps | contended ETA |
|---|---|
| 50M | ~2.1 h |
| **100M (default)** | **~4.3 h** |

These are **upper bounds measured under ~13 cores of competing load**, not clean
numbers. Uncontended it should be materially faster; I did not re-measure after
the load cleared rather than quote a figure I had not taken.

```bash
# from an INTERACTIVE session shell on MAIN, not from an agent
nohup caffeinate -is bash scripts/orbital/extj2/j2wait_campaign.sh \
    > /tmp/j2wait_stdout.log 2>&1 &
tail -f /tmp/j2wait_progress.log

# knobs: J2WAIT_STAGES=0,1,2   J2WAIT_STEPS=100000000   J2_SEED=42
#        J2_SEED=7 suffixes data dirs / wandb groups / trained-row JSONs
```

Stage 1 (floors) is eval-only and takes ~1 min. Stage 0 anchors ~1 min.

---

## 7. The measurement the campaign exists to produce

`j2wait_eval.py` decomposes every episode's plane closure **exactly**, with no
residual by construction: each decision either carries an impulse
(`ACTION_DV ≠ 0` and τ = 1) or does not, and Δi_rel is a pure function of state,
so

```
closure_impulse    = Σ over BURN decisions   of (di_rel_before − di_rel_after)
closure_precession = Σ over COAST/WARP decisions of (di_rel_before − di_rel_after)
```

The 60 s of precession inside a burn decision is misattributed to the impulse:
0.65 °/day × 60 s = 4.5e−4° against a 25 m/s burn's 0.19°, i.e. **0.2%**, and it
is reported rather than hidden.

Headline shape, emitted directly by the eval:

> *the policy buys **X°** of node-dominant plane alignment from precession at
> **Y m/s**, where direct burning the same X° costs **Z = 2·v_c·sin(X/2)** m/s*

Also per-episode: dip depth/duration, wall-time-in-episode, decisions and
day-warps used, and the `--no-daywarp` ablation of the *trained* policy, which
is the attribution arm — if performance survives without row 30 the day-warp
was a convenience; if it collapses, the day-warp is load-bearing.

**Every claim is in MEAN ELEMENTS and every plane number is a NODE-DOMINANT
plane gap**, never a general plane error — drift moves Ω and nothing else, and
a pure inclination difference is a hard floor no amount of waiting removes.
