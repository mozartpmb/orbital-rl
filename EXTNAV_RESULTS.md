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

_(filled in per arm as the campaign runs)_
