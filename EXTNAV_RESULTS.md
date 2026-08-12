# ext-nav — T5 NB campaign results (bearings-only, filter-in-the-loop)

All cells: held-out seed 123, 200 episodes, greedy, T3 headline config
(`shaping_mode 1, shape_gamma 1.0, phase_gap_mode 1, phase_obs_mode 1,
episode_cap_steps 3000, cap_terminal_reward 0.0, valid_init_only 1`,
e≤0.05 both, ±180° physical phase gap, LEO 300–800 km). Success = terminal
cause == 1; gave-up inits excluded from the denominator.

**native** = `eval_relnav --stage bearings`, the REAL angles-only batch
acquisition (`bls_acquire_adaptive`, shipped grid knobs) handing off to a live
modified-polar recursive filter. **truth** = the same checkpoint on the plain
truth observation. Training arms are 50M steps of `puffer_orbital_nav` with
`nav_mode=bearings_only` (calibrated acquisition surrogate + the same live
recursive filter), 8 workers × 256 envs.

**Gates:** an arm passes if native ≥ 85% **and** truth-tax ≤ 2 pp.

## Reference row — canonical truth-trained checkpoint, zero-shot

`models/t3/seed42_L2_headline.pt` — never saw an estimate in training.

| metric | value |
|---|---|
| native (bearings-only, real acquisition) | **139/200 = 69.5%** — success=139, collision=38, safety_cap=1, stranded=22 |
| truth | **200/200 = 100.0%** |
| episodes never acquiring | 44/200 · blind decisions 30.7% · acquisition failures 0/200 |
| acquisition latency | median 109 min (p90 139) |
| acquisition EPOCH error vs truth | median 11.3 km, p90 86.9 km, **max 3,626.7 km** (REDTEAM MAJOR-2, live) |
| conditional success \| min separation < 200 km | 131/136 |
| σ_LOS/ρ (100 km–1 Mm bin) | 0.0010 |

### Blind-window behaviour — the mechanism the campaign is testing

| metric | canonical |
|---|---|
| Δv spent **before** acquisition | median **75.0 m/s**, p90 200.0, max 445.0 (of a 478 m/s budget) |
| fuel budget left **at** acquisition | median 0.832, p10 0.562 |
| burn rate, blind vs acquired | **0.822** vs 0.425 burns/decision — **1.94×** |
| Δv per decision, blind vs acquired | **19.32** vs 9.49 m/s — **2.04×** |
| stranded | 22/200 = 11.0%, of which blind at terminal 10 (5.0% of all episodes) |
| total Δv per episode | median 322.5 m/s |
| episodes that ever acquired | 78.0% |

Read: fed a ~9,000 km-wrong prior during the 45–185 min acquisition window, the
truth-trained policy burns **twice as hard** as it does once it has a real
estimate, and has already spent a median 75 m/s — 16% of its entire budget —
on nothing by the time the angles-only solver converges. That, not a bad
capture, is what the 30.5 pp gap is made of.

## Arms

| arm | native | truth | truth-tax | gate | ckpt |
|---|---|---|---|---|---|
| canonical (reference) | 139/200 = 69.5% | 200/200 = 100.0% | 0.0 pp | — | `models/t3/seed42_L2_headline.pt` |
| **NB1-warm** (seed 42, warm) | **196/200 = 98.0%** (+28.5 pp) | **200/200 = 100.0%** | **0.0 pp** | **PASS** | `models/t3/extnav_nb1_warm.pt` |
| **NB1-fresh-42** (seed 42, fresh) | **198/200 = 99.0%** (+29.5 pp) | **200/200 = 100.0%** | **0.0 pp** | **PASS** | `models/t3/extnav_nb1_fresh_42.pt` |
| **NB1-fresh-7** (seed 7, fresh) | **199/200 = 99.5%** (+30.0 pp) | **200/200 = 100.0%** | **0.0 pp** | **PASS** | `models/t3/extnav_nb1_fresh_7.pt` |
| **NB1-fresh-1337** (seed 1337, fresh) | **197/200 = 98.5%** (+29.0 pp) | **200/200 = 100.0%** | **0.0 pp** | **PASS** | `models/t3/extnav_nb1_fresh_1337.pt` |

**4/4 arms pass both gates** (native ≥ 85%, truth-tax ≤ 2 pp). Pooled across the
three fresh seeds: **594/600 = 99.0%**. Pooled across all four arms:
**790/800 = 98.75%**.

### NB1-warm — 50M steps, `nav_mode=bearings_only`, warm from `seed42_L2_headline.pt`

wandb group `t5-nav-nb1-warm`; final checkpoint epoch 382 (the epoch-200
checkpoint screens identically at 100/100 on the surrogate path, so this is not
a late-training spike). Training: 27.1K SPS, `perf` 0.998 throughout,
`nav_diverge_rate` 0.000, `nav_clip_rate` 0.000. `nav_acq_per_ep` rose
0.870 → 0.974 over training — visible in the training metrics before any eval.

| metric | canonical | NB1-warm | Δ |
|---|---|---|---|
| native success | 139/200 = 69.5% | **196/200 = 98.0%** | **+28.5 pp** |
| native causes | collision 38, stranded 22, cap 1 | **cap 4 only** | no collisions, no strandings |
| truth | 200/200 | 200/200 | **0.0 pp tax** |
| episodes never acquiring | 44/200 | **0/200** | −44 |
| blind decisions | 30.7% | **5.2%** | −25.5 pp |
| filter divergences | 56 | 8 | −48 |
| acquisition failures | 0/200 | 1/200 | +1 |
| conditional success \| min sep < 200 km | 131/136 | **181/183** | +2.6 pp |
| acquisition latency | median 109 min | median 105 min | ≈ unchanged |
| σ_LOS/ρ (100 km–1 Mm) | 0.0010 | 0.0009 | ≈ unchanged |

**Blind-window behaviour — the policy inverted it.**

| metric | canonical | NB1-warm |
|---|---|---|
| Δv spent **before** acquisition | median 75.0 m/s, p90 200, max 445 | **median 0.0 m/s**, p90 0.0, max 285 |
| fuel budget left **at** acquisition | median 0.832, p10 0.562 | **median 1.000**, p10 1.000 |
| burn rate blind ÷ acquired | **1.94×** (0.822 vs 0.425) | **0.26×** (0.127 vs 0.494) |
| Δv/decision blind ÷ acquired | **2.04×** (19.32 vs 9.49) | **0.25×** (2.67 vs 10.72) |
| stranded | 22/200 = 11.0% (10 blind at terminal) | **0/200** |
| total Δv/episode | median 322.5 m/s | median 285.0 m/s |
| episodes that ever acquired | 78.0% | **100.0%** |

The canonical policy burned **1.94× harder** while blind than while acquired.
NB1-warm burns **0.26× as hard** — a 7.5× swing in the ratio — and reaches
acquisition with a **full tank** (median Δv-before-acquisition 75 → 0 m/s).

It did **not** learn NAV-G's predicted 1 m/s observability burn. It learned the
complementary policy: *coast through the blind window, then maneuver.* Bearings
accrue during a coast regardless, so waiting costs only clock, and it removes
both failure modes at once — the 38 collisions and 22 strandings are gone, and
never-acquire goes 44 → 0 because the vehicle is still alive and fuelled when
the solver converges. Total Δv per episode also fell (322.5 → 285.0 m/s), so
this is not "spend later instead"; it is waste removed.

### NB1-fresh-42 — 50M steps, `nav_mode=bearings_only`, fresh nets, seed 42

wandb group `t5-nav-nb1-fresh-42`; final checkpoint epoch 382 (epoch-200 screens
99/100). Training: 27.3K SPS, `nav_diverge_rate` 0.000, `nav_clip_rate` 0.000,
`nav_acq_per_ep` 0.985. Rolling `perf` was 0.056 at 5M in the V5 smoke and
0.906 by 11.3M — a fresh bearings-only run does bootstrap, and takes off between
5M and 11M.

| metric | canonical | NB1-fresh-42 |
|---|---|---|
| native success | 139/200 = 69.5% | **198/200 = 99.0%** (+29.5 pp) |
| native causes | collision 38, stranded 22, cap 1 | cap 1, stranded 1 |
| truth | 200/200 | 200/200 (**0.0 pp tax**) |
| never acquiring | 44/200 | **0/200** |
| blind decisions | 30.7% | 5.2% |
| divergences | 56 | 6 |
| conditional \| min sep < 200 km | 131/136 | **185/186** |
| Δv before acquisition | median 75.0 m/s | **median 0.0 m/s** |
| fuel left at acquisition | 0.832 | **1.000** |
| burn rate blind ÷ acquired | **1.94×** | **0.41×** (0.189 vs 0.458) |
| Δv/decision blind ÷ acquired | **2.04×** | **0.40×** (4.04 vs 10.02) |
| stranded | 11.0% | 0.5% |
| total Δv/episode | 322.5 m/s | **257.5 m/s** |

### NB1-fresh-7 — 50M steps, `nav_mode=bearings_only`, fresh nets, seed 7

wandb group `t5-nav-nb1-fresh-7`; final checkpoint epoch 382 (epoch-200 screens
identically at 100/100). Training: 26.6K SPS, `perf` 0.997.

| metric | canonical | NB1-fresh-7 |
|---|---|---|
| native success | 139/200 = 69.5% | **199/200 = 99.5%** (+30.0 pp) |
| native causes | collision 38, stranded 22, cap 1 | cap 1 |
| truth | 200/200 | 200/200 (**0.0 pp tax**) |
| never acquiring | 44/200 | **0/200** |
| blind decisions | 30.7% | 5.5% |
| divergences | 56 | **4** |
| conditional \| min sep < 200 km | 131/136 | **190/190** |
| Δv before acquisition | median 75.0 m/s | **median 0.0 m/s** (p90 50, max 100) |
| fuel left at acquisition | 0.832 | **1.000** (p10 0.888) |
| burn rate blind ÷ acquired | **1.94×** | **0.52×** (0.241 vs 0.468) |
| Δv/decision blind ÷ acquired | **2.04×** | **0.60×** (6.03 vs 10.00) |
| stranded | 11.0% | **0.0%** |
| total Δv/episode | 322.5 m/s | 275.0 m/s |

### NB1-fresh-1337 — 50M steps, `nav_mode=bearings_only`, fresh nets, seed 1337

The seed that historically flatlined on fresh T3 Stage-1 training. wandb group
`t5-nav-nb1-fresh-1337`; final checkpoint epoch 382 (epoch-200 screens
identically at 100/100). Training: 25.6K SPS, `perf` 0.998,
`nav_acq_per_ep` 0.987, `nav_diverge_rate` 0.000.

| metric | canonical | NB1-fresh-1337 |
|---|---|---|
| native success | 139/200 = 69.5% | **197/200 = 98.5%** (+29.0 pp) |
| native causes | collision 38, stranded 22, cap 1 | cap 3 |
| truth | 200/200 | 200/200 (**0.0 pp tax**) |
| never acquiring | 44/200 | **0/200** |
| blind decisions | 30.7% | **3.8%** |
| conditional \| min sep < 200 km | 131/136 | **190/192** |
| Δv before acquisition | median 75.0 m/s | **median 0.0 m/s** |
| burn rate blind ÷ acquired | **1.94×** | **0.42×** (0.110 vs 0.264) |
| Δv/decision blind ÷ acquired | **2.04×** | **0.38×** (2.21 vs 5.80) |
| stranded | 11.0% | **0.0%** |
| total Δv/episode | 322.5 m/s | **235.0 m/s** (lowest of all arms) |

---

## Campaign summary — 4 arms

| arm | native | truth | tax | never-acq | stranded | burn ratio blind÷acq | Δv ratio blind÷acq | Δv before acq | total Δv/ep |
|---|---|---|---|---|---|---|---|---|---|
| canonical (zero-shot) | 69.5% | 100.0% | — | 44/200 | 22/200 | **1.94×** | **2.04×** | 75.0 m/s | 322.5 m/s |
| NB1-warm | 98.0% | 100.0% | 0.0 pp | 0/200 | 0/200 | 0.26× | 0.25× | 0.0 m/s | 285.0 m/s |
| NB1-fresh-42 | 99.0% | 100.0% | 0.0 pp | 0/200 | 1/200 | 0.41× | 0.40× | 0.0 m/s | 257.5 m/s |
| NB1-fresh-7 | 99.5% | 100.0% | 0.0 pp | 0/200 | 0/200 | 0.52× | 0.60× | 0.0 m/s | 275.0 m/s |
| NB1-fresh-1337 | 98.5% | 100.0% | 0.0 pp | 0/200 | 0/200 | 0.42× | 0.38× | 0.0 m/s | 235.0 m/s |

### Did estimate-training change blind-window behaviour?

**Yes, and it inverted the sign.** Every arm crossed from burning *harder* while
blind to burning *less*: the blind÷acquired burn-rate ratio goes 1.94× →
0.26–0.52×, a 3.7–7.5× swing, and all four arrive at acquisition with a **full
tank** (median Δv-before-acquisition 75 → 0 m/s). Total Δv per episode fell in
every arm (322.5 → 235–285 m/s), so this is waste removed, not spend deferred.

**It is not the maneuver NAV-G predicted.** NAV-G's headline was that training
on the estimate should teach the 1 m/s observability burn, since a drifting
chaser has a singular Fisher information for range below ~50 km. No arm learned
that. They learned the complementary policy: *coast through the blind window,
then maneuver.* That is the correct answer for this envelope — bearings accrue
during a coast regardless, so the information arrives anyway, and what was
actually killing the canonical was not a lack of information but the expenditure
of propellant on wrong information. Acquisition **latency is unchanged**
(105 min vs the canonical's 109), which is the direct evidence: the arms did not
acquire *faster*, they survived the wait.

**Warm vs fresh.** Both routes reach the same qualitative behaviour and the same
truth-tax (0.0 pp), but they are not identical:
- Fresh is slightly *better* on capability (99.0% mean over 3 seeds vs 98.0%
  warm) and clearly better on economy (235–275 vs 285 m/s).
- Warm is the *most* conservative (0.26× burn ratio) yet scores lowest of the
  four. Maximal timidity is not the optimum; the fresh arms keep a little blind
  maneuvering and do better with it.
- **The T3 fresh-seed fragility did not reproduce.** All 3 fresh seeds
  bootstrapped, including 1337, which historically flatlined at fresh T3
  Stage-1. NAV-H recommended fresh arms specifically because dropping range
  changes observability structure rather than noise scale; that concern was
  sound but the pessimism was not warranted — bootstrap-ability here is governed
  by the reward/curriculum context (the T3 flag set), not by the observation
  channel.

### What training did NOT fix

- **REDTEAM MAJOR-2 stands.** Acquisition epoch error against truth still peaks
  at 2,350–3,855 km across the arms with all three acceptance gates passing.
  Training made the closed loop *robust* to a confidently-wrong acquisition; it
  did not make the acquisition correct. The gates remain a self-consistency
  test, not an acceptance test.
- **Acquisition latency is unchanged** (~105 min). Nothing here accelerates
  initial orbit determination.
- Filter consistency is essentially unchanged (σ_LOS/ρ ≈ 0.001 in the
  100 km–1 Mm bin for every arm, canonical included) — the *filter* was never
  the problem; the *guidance response to it* was.
