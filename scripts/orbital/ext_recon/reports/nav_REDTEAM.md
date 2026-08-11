All probes complete. Report below.

---

# RED-TEAM: ext-nav design (OrbitalNav + batched EKF + NAV-G bearings-only)

**Verdict: SHIP WITH CHANGES — except the NB (bearings-only) lineage, which is REDESIGN.**

Probes: `/Users/pete/space_training/scripts/orbital/ext_recon/redteamnav/ext_rtnav_{acq_cost,acq_batched,acq_cheap,decode_floor,obs_slots,blind_window,phi_leak}.py`; CSVs `web_data/results/ext_rtnav_*.csv`. Nothing existing modified, nothing committed.

**Two measurements reframe the whole design:**
1. **Mean episode = 20.9 decisions**, not the ~70–92 implied by cap 3000 ÷ mean τ 32.5 (`ext_rtnav_obs_slots.py`, canonical greedy, 40 eps, 836 decisions). Every per-episode cost amortizes over 21 decisions, not 70 — a 3–4× hit to NAV-H's projections.
2. **The blind window is free; close-range blindness is fatal.** The design spends all its complexity on the wrong end of the episode.

---

## BLOCKER-1 (A) — The NAV-G acquisition cannot run in the training loop. Not by 2×; by 17–90× at its *perfectly-vectorized lower bound*.

**Attack.** NAV-H's "SPS is not the binding constraint" was measured on the range+bearing EKF. NAV-G then substituted an algorithm with a different cost class and NAV-H's budget was never re-derived.

**Measurement — scalar (`ext_rtnav_acq_cost.py`, 6 geometries × 2 seeds):**

| geometry | wall | `propagate_cartesian` | `stm_numerical` | final window | gate |
|---|---|---|---|---|---|
| G6 5 km box, 1 m/s burns | 0.226 s | 23,400 | 2,640 | 45 (45 min) | PASS |
| G5 10 km burns | 0.217 s | 21,703 | 2,640 | 45 | PASS |
| G3 180° gap | 0.188 s | 19,255 | 2,144 | 45 | PASS |
| G4 wide e=0.30 | 0.806 s | 92,779 | 7,113 | 72 | PASS |
| G1 10 km drift | 1.86 s | 173,232 | 25,824 | 184 (184 min) | PASS |

median **0.226 s**, mean **0.550 s**, max **1.86 s** per acquisition. Unit costs: scalar `propagate_cartesian` 2.73 µs, `stm_numerical` 29.09 µs (= 10.7 props).

**Measurement — batched lower bound (`ext_rtnav_acq_batched.py`).** Batched primitives, OMP=1: `prop4` **89.3 ns/state** at B=262,144 (96.2 ns at 1024); `stm_fd4` 586–852 ns/state. Stage split of the scalar acquisition: grid-scoring 45–60%, GN-refine 40–55% — i.e. **roughly half the cost lives in the Levenberg loop, which is control-flow divergent** (per-node accept/reject `continue`, per-node admissibility `break`, per-env adaptive window growth, 1-or-2 intervals per env). Assuming *perfect* vectorization anyway — no LM divergence, no per-node break, all 1024 envs acquiring on the same window (none of which is achievable):

| case | per-1024-env acquisition | +ms/decision @20.9 dec/ep | SPS | 50M-step rung |
|---|---|---|---|---|
| **reference: nav60 range+bearing** | — | — | 61,000 | **13.6 min** |
| G5/G6 best case (w=45, 612 nodes) | 5,514 ms | +264 | 6,900 | 2.0 h |
| shipped config, mean over 6 geoms | 11,230 ms | +537 | 3,590 | **3.9 h** |
| G1 (3 window growths) | 28,429 ms | +1,360 | 1,470 | 9.5 h |
| **measured scalar (median / mean)** | 231,000 / 563,000 ms | +11,070 / +26,950 | 185 / 76 | **75 h / 183 h** |

**Cost-reduction knobs don't save it (`ext_rtnav_acq_cheap.py`, 6 configs × 6 geoms × 3 seeds).** Total dynamic range from `grid_ratio` 1.15→3.0, `n_bins` 8→4, `w0` 45→20, velocity lattice 9→1: **1.85×**, floor 2.2 h/rung. And the cheap configs break silently — see MAJOR-2.

**Implication for the run set.** NB1 fresh×3 + warm×3 + NB2×3 = 9 runs minimum. Lower bound 35 h; realistic (partially-vectorized) hundreds of hours; scalar ≈ 4 weeks.

### Recommendation for A (the gating decision): **calibrated acquisition surrogate in training, real acquisition at eval.**

Ranked against the four options you named, all costed:

| option | SPS cost | verdict |
|---|---|---|
| (i) simplified recursive BO filter, scripted coast during acquisition | ~0 | **Measured dead.** I ran the missing cell of NAV-G's own table (`--filters BO-MPC`, blind, 8 seeds, `redteamnav/ext_rtnav_blindmpc_run.txt`): BO-MPC blind-from-prior gives **89.8 km posRMSE / 101.9 m/s, NEES 1.4e5, in-bounds 0.04** at G1 and **597 km / 679 m/s, NEES 3.6e7** at G2 — confidently wrong, not merely imprecise — and then **crashes** with `ValueError: math domain error` at `msc_encode` (`ext_bo_filter.py:358`, unguarded `math.log(rho)`) at G3. There is no cheap recursive-only blind filter. |
| (ii) precompute acquisition offline and inject at t_acq | unchanged if you still *run* BLS | **Right shape, wrong mechanism** — fix below. |
| (iii) async acquisition every K decisions | unchanged | Smooths latency, identical total FLOPs. Reject. |
| (iv) smaller `num_envs` for NB rungs | wash | Cuts per-decision cost and agent-steps-per-decision proportionally. Reject except for memory. |

**Concrete spec.** At each episode reset, do **not** run BLS. Instead:
- Draw the acquisition outcome from the *measured* BO-BLS error distribution: `x̂_acq = x_true + N(0, Σ(ρ/r, arc, Δv_window))`, `P_acq = 4·Σ` (matching NAV-G's `cov_inflate=4`), with `Σ` read from the existing filter-independent CRLB map `web_data/results/ext_bo_observability.csv` (158 rows over separation × arc × Δv × e), scaled by the measured CRLB→BLS ratio per geometry from `ext_bo_filter.csv`. Cost: **one Gaussian draw per episode.**
- Hand it to the **live recursive `BearingMPC`** (modified-polar), which runs every nav tick for the rest of the episode. Per-tick cost ≈ 9 propagations + 4×4 algebra ≈ 1.5–2× the shipped v2 EKF tick → nav60 budget goes 3.89 → ~6–8 ms/decision, i.e. **within 1.15× of the range+bearing plan**. Guard `msc_encode`/`msc_decode` (`log(max(rho, 1.0))`, finite checks) before it goes near 50M steps.
- **Condition Σ on the realized Δv inside the acquisition arc** — this is load-bearing, not cosmetic: G5/G6 (with burns) acquire in 45 min at 10–20 m; G1 (drift) needs 115–184 min at 140–160 m, and the CRLB is *singular* at Δv=0 below 50 km. A Δv-independent surrogate would hand the policy free information it did not maneuver for and would destroy the headline observability claim.
- **At eval, run the real `bls_acquire_adaptive`.** 200 episodes × 0.55 s = 110 s. Free.

**What the trained policy then demonstrates, stated exactly** (this is the claim-honesty cost, and it must be in the write-up verbatim):
> *"The policy flies closed-loop on a live angles-only modified-polar recursive filter whose covariance responds to its own maneuvers. Initial orbit determination is performed by a batch angles-only IOD validated offline (NAV-G: 144 runs, 0 acquisition failures, 6 geometries); during training its outcome is replayed as a calibrated error draw conditioned on the realized geometry and Δv, because the batch solver is 17–90× the entire training step. Closed-loop acquisition — including acquisition failure and the policy's behavior during the blind window — is evaluated, not trained."*

That is a defensible GN&C statement (it is how flight nav is validated versus how guidance is tuned) and it preserves the one result worth chasing: the observability-driven 1 m/s burn is a property of the *recursive* filter's covariance, which is live.

---

## MAJOR-2 (A) — The acquisition gates certify 2,951 km errors. NAV-G's "0/144 failures" is conditional on grid density and does not generalize.

**Attack.** The χ² + ambiguity-margin + σ_LOS/ρ triple is presented as the acceptance test. If it is sound, it should catch a starved grid.

**Measurement (`ext_rtnav_acq_cheap.py`, 18 runs/config):**

| config | gate pass | err p50 | worst geometry |
|---|---|---|---|
| shipped (ratio 1.15, bins 8, w0 45, 9 vel) | 18/18 | 0.40 km | G4 27.4 km |
| ratio 2.0 | 18/18 | 0.40 km | G4 28.8 km |
| ratio 2 + w0=20 | 18/18 | 0.56 km | **G3 44.3 km** (from 6.6) |
| ratio 2 + lean velocity lattice | 15/18 | **3,533 km** | G1/G5 3,970 km |
| **minimal (ratio 3, bins 4, w0 20, 1 vel)** | **18/18** | 5.59 km | **G2 2,951 km, G4 1,619 km** |

`minimal` passes **every** gate while returning a 2,951 km epoch solution at G2. The gates test self-consistency of the chosen basin, not correctness of the basin choice — exactly the failure mode NAV-G says the ambiguity margin catches, and it doesn't when the grid is too coarse to have visited the true decade. **Minimal fix:** never tune the grid without re-measuring epoch error against truth; the gate is not a substitute. If the surrogate (BLOCKER-1) is adopted this is moot at training time but still governs the eval arm.

---

## MAJOR-3 (B) — "Hold coast/warp until acquired" is both unnecessary and PPO-corrupting. Delete it.

**Attack.** Two sub-claims: (a) is a hold needed, (b) does the hold corrupt PPO.

**(a) Not needed — measured.** `ext_rtnav_blind_window.py`, canonical `seed42_L2_headline.pt`, greedy, held-out seed 123, 100 eps/cell, T3 kwargs, target-derived slots rebuilt from NAV-G's own pre-acquisition state (`rho0 = sqrt(lo·hi)` on the analytic prior + circular velocity), truth afterwards:

| condition | success | causes |
|---|---|---|
| truth (control) | 100/100 | — |
| blind 1 / 2 / 4 decisions | 100 / 100 / 100 | — |
| blind 8 | 99 | stranded 1 |
| hold-warp1h 1/2/4/8 | 100 / 100 / 100 / 100 | — |
| hold-coast 2 / 8 | 100 / 100 | — |
| jump (single blind decision at t=0 / t=3) | 100 / 100 | — |

Median injected range error **9,276 km**; median max target-slot obs perturbation **1.74** (obs are Box(−2,2)); median |Δobs[7]| **0.105**. The acquisition window is 45–184 nav ticks = **1–4 decisions** at the policy's modal warp-1h — squarely inside the free zone. **Holding buys 0.0 pp.**

**(b) The hold corrupts the rollout — confirmed in code.** `pufferl.py:295` writes `self.actions[batch_rows, l] = action` — the *policy's sampled* action, before `vecenv.send`. An env-side override stores (obs, a_policy, logprob_policy) against a transition generated by a_forced: the PPO ratio is evaluated for an action never executed, and GAE attributes the resulting advantage to it. Fraction corrupted: at warp-1h hold, 1–4 of 20.9 decisions = **5–19% of every rollout**; at the naive coast reading, 45–184 forced decisions against a 21-decision baseline = **68–90%**. There is no masking escape: `pufferl.py:298` carries the comment *"We are not yet handling masks in this version"*, so `enable_action_mask` (which anyway only covers actions 0–9 and would push obs to 48-dim) is a dead route.

**Minimal fix:** no hold. Let the policy fly the unacquired estimate. This deletes the off-policy problem, the nav-valid-bit problem, and the layout problem simultaneously.

---

## MAJOR-4 (B/design) — The risk is inverted: blindness at episode *start* is free; blindness at *close range* is the failure mode, and that is exactly where the CRLB is singular and the policy warps.

**Measurement (same probe, injecting the unacquired estimate whenever true separation is below a threshold):**

| condition | success | causes |
|---|---|---|
| blind while sep < 50 km | 93/100 | stranded 7 |
| blind while sep < 200 km | **45/100** | stranded 55 |
| blind while sep < 1000 km | **18/100** | stranded 57, safety_cap 25 |
| blind for the whole episode | **1/100** | safety_cap 82, stranded 17 |
| blind for the first 16 / 32 decisions | 91 / 77 | stranded 9 / 21 + cap 2 |

Terminal-phase estimate quality is what the mission is made of; the failure mode is **fuel exhaustion from thrashing on a wrong estimate**, not a wrong early burn. NAV-G's own CRLB says drift-only angles-only range is *exactly singular* at ρ ≲ 50 km, and NAV-H measures the policy at 96.8% warp / 51% warp-1h. So the design's real hazard is: **a warp-heavy policy blinding itself inside 200 km, where a 55% stranded rate is the measured consequence.** That is also the *interesting* result — but it means the campaign's instrumentation and gating must be built around close-range covariance, not acquisition latency.

**Minimal fixes:** (1) make `nav_sigma_los_at_capture`, `Δv-spent-vs-trace(P)` correlation, and `sep<200 km` conditional success first-class gates, not "NAV-F diagnostics"; (2) add a `blindclose` ablation row to the eval matrix — it is the single most informative cell and it costs 100 episodes; (3) **`blindall` = 1/100 is the control that retires the "reward leaks truth" worry at inference time** (see MINOR-7): if truth leaked usefully, a permanently-blind policy would not collapse to 1%.

---

## MAJOR-5 (F) — The N-ladder as specified is cheap but *confounded*. Don't chain it; don't put it on NB1's critical path.

Wall clock is not the argument: 3 N-rungs at 13.6 min each is 41 min. The defect is attribution. NAV-H chains N1(30×)→N2(100×)→N3(300×), so the N3 row of eval matrix (d) is "trained on a noise curriculum", not "trained at 300×" — and the claim under test is precisely "training at noise level X buys robustness at noise level X". Chaining makes the off-diagonal uninterpretable.

**Minimal high-value run set:**
- **N0** truth-mode anchor (200/200 gate before anything trains). Keep.
- **N1/N2/N3 each warm-started independently from N0**, run in parallel, 1 seed. 3 × 13.6 min. Gives four clean rows of matrix (d).
- **NB1 warm from N0 directly** (not from N3) — NAV-H already recommends this; the chain was never on its path.
- **NB1 fresh × 3 seeds** — keep. This is the only rung that changes observability structure and the T3 seed-fragility finding (seed 1337 flatlined at Stage-1 fresh) is real.
- **NB2 only if NB1 clears.**
- Add **NB1-blindclose ablation** (MAJOR-4) as a 5th cell.

Zero-shot already gives 100/99.5/96.0/81.5/61.0% at 1/10/30/100/300×, so N1–N3 have ≤4/18.5/39 pp of headroom; run them for the matrix, not for capability.

---

## MINOR-6 (G) — Diagnostics/registration: five concrete gotchas, all verified in code.

1. **Info contract.** `pufferl.py:313-321` iterates `for i in info` and feeds each element to `unroll_nested_dict` — infos **must be a `list` of `dict`s** (asserted at `pufferlib.py:103,107`). `Orbital.step` (`orbital.py:237-256`) appends **at most one** dict, **only** when `self.tick % log_interval == 0` (=128) **and** `log["n"] > 0`. So `info` is frequently `[]`: the wrapper must **append its own dict**, never assume `info[0]`.
2. **Emit plain Python floats.** `pufferl.py:316-317` has a dead-store bug: an `np.ndarray` value is `.tolist()`-ed then falls through to `append` (not `extend`), landing a whole list in `stats[k]`. Reduction is an **unweighted** `np.mean` over emissions (`pufferl.py:534-543`), *not* weighted by `n`.
3. **Naming/plumbing.** Keys surface as `environment/<key>` in wandb (`pufferl.py:552`) and unprefixed on the dashboard, `%.3f`, **capped at 30 rows** (`pufferl.py:693-706`). Orbital already emits 11; ~19 free. No C change needed for `nav_*` — but note `g_shape_abs` is accumulated in `add_log` and *never exported* in `my_log`, the precedent for silently-missing metrics.
4. **Pipe cost.** `vector.py:213-221` sends infos only `if infos:` — at `log_interval=128` this is one small pickle per 128 steps per worker. Emitting a nav dict **every** step would put a pickle + syscall on the hot path. Match the 128 cadence.
5. **Seeding is not what the config says.** `vector.make()` (`vector.py:618, 708`) **never forwards `seed`** to the backend, so `[vec] seed = 42` in `default.ini` is silently dropped and `Multiprocessing.__init__(seed=0)` applies → `seed_i = 0 + i` (`vector.py:336`). Measurement noise derived from the constructor seed is therefore reproducible but **not user-controllable**; take the nav seed from an explicit `[env]` kwarg (`nav_seed`), not from `[vec] seed`. Pickling is a non-issue: the worker constructs the env itself (`vector.py:191`), so filter state is never serialized. `driver_env` (`vector.py:273`) still allocates a full 1024-env C env in the parent — keep `OrbitalNav.__init__` allocation lazy. Workers are spawned and inherit `os.environ`, so exporting `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` before `puffer train` is sufficient BLAS pinning.

---

## MINOR-7 (E) — Reward-on-truth is not a leak, but PBRS invariance is doubly broken. Bounded; state it, don't fix it.

**Code map.** `compute_phi` (`orbital.h:804-819`) reads four truth target scalars (`target.a/e/omega/M`); success (`orbital.h:960-967`) is **inertial relative position AND velocity** vs the propagated target (30 km / 50 m/s default), also truth. Nothing in reward or termination reads `env->observations`. Shaping is applied **after** `check_termination` (`orbital.h:1498-1514` vs `:1468`) — the ordering your memory flags — and terminal branches **assign** (`=`) rather than accumulate, so any same-step `collision_penalty_w` is silently erased.

**Measurement (`ext_rtnav_phi_leak.py`, exact Φ replica, 80 eps / 1,767 decisions):** per-decision shaping delta on truth has rms **0.0335**, p95|·| 0.0877. Fraction of that delta *unexplainable* from the estimate:

| estimate quality | corr(Δφ_truth, Δφ_est) | explained var | unexplained rms |
|---|---|---|---|
| EKF 1× (8 m) | +1.0000 | 1.0000 | 0.0% |
| EKF 100× (800 m) | +0.9998 | 0.9996 | 2.0% |
| BO settled (3 km) | +0.9969 | 0.9937 | **7.9%** |
| unacquired prior | **+0.0314** | −10.19 | **334.6%** |

Telescoped episode shaping return (`shape_gamma=1.0`, exact telescoping) = Φ_T − Φ_0 ∈ **[+0.09, +1.23]**, ≤ **24.7%** of the smallest success payout (`10·(0.5+0.5·fuel) ∈ [5,10]`).

**Honest assessment.** This is privileged-information-at-training-time (asymmetric actor-critic), not an exploitable leak — `blindall = 1/100` (MAJOR-4) shows the obs channel does essentially all the work at inference. But PBRS policy-invariance is broken twice over: (a) `shape_gamma=1.0 ≠ γ_GAE=0.995`, already documented as deliberate; (b) Φ is a function of the *hidden* state, and PBRS invariance requires Φ to be a function of the agent's information state. Under (b) the unexplained Δφ becomes advantage noise the value head cannot cancel — 7.9% at post-acquisition BO accuracy (negligible against a 5–10 terminal), 335% during the blind window (1–4 decisions, measured to cost 0–1 pp).

**Do not switch to estimate-based shaping.** It would restore invariance but makes Φ *spoofable*: the policy could maneuver to make the filter believe it is converging. Exploit ceiling is exactly +1.35 (Φ bounded on `orbital.h:817`) against a ≥5 success payout, so it wouldn't dominate — but it trades a 7.9%-noise problem for a reward-hacking problem, and your campaign's scar tissue ("never change the reward and the observation in the same run") forbids it anyway. **Write-up must state:** *"Reward and termination are computed on truth; the policy observes only the filter. Training therefore uses privileged information in the reward, which is standard for sim-trained GN&C and does not appear at deployment. Potential-based shaping is not policy-invariant under partial observability; the unexplained component is measured at 7.9% rms of the shaping delta at the shipped angles-only accuracy, against a shaping total bounded at 24.7% of the terminal reward."*

---

## NON-ISSUE-8 (D) — The float32 decode floor does not corrupt bearing synthesis. Your 0.1 mrad estimate is off by 40×, in your favour.

**Measurement (`ext_rtnav_decode_floor.py`, f64 truth → f32 obs → `recover_states_t3` → Cartesian, 400 trials/cell):**

| separation | target abs pos err p50/p95 | **relative** transverse err | synthesized bearing err p50/p95 |
|---|---|---|---|
| 1 km | 0.07 / 0.24 m | 0.01 / 0.03 m | **0.0091 / 0.0268 mrad** (2.7% of σ_β) |
| 5 km | 0.07 / 0.23 m | 0.01 / 0.02 m | 0.0018 / 0.0048 mrad |
| 10 km | 0.07 / 0.24 m | 0.01 / 0.03 m | 0.0008 / 0.0026 mrad |
| 200 km | 0.09 / 0.24 m | 0.01 / 0.04 m | 0.0001 mrad |
| 5,000 km | 0.20 / 0.99 m | 0.11 / 0.71 m | ~0 |

The naive estimate assumes ~1 m *independent* endpoint noise. It isn't: **obs[13,14] encode the relative mean longitude directly**, so the along-track decode errors of chaser and target cancel to first order — the *relative* transverse error is **0.01–0.03 m**, ~50× smaller than either absolute error. Worst case is 2.7% of σ_β at the 1 km closest approach, and it is 5× below even NAV-G's 0.1 mrad sensor rung (1.8 m at G6). **No C-side true-state accessor is needed.** Sub-propagating from the decoded state does *not* dodge the floor (it seeds from f32) — it doesn't need to.

Two caveats to carry: (1) at 13,000 km the p95 spikes to 2,897 m / 0.24 mrad in high-e draws — the θ→M inversion is stiff as e grows, so the wide/MEO lineages need the `reconcheck` gate exercised at e≥0.3, not just at LEO; (2) `obs[37] = √(µ/a_t³)` is an exact function of `obs[7]` — the filter must perturb both consistently or it opens a truth-leak channel worth watching.

---

## NON-ISSUE-9 (B) — "The 38-dim layout is frozen, so there's nowhere for a nav-valid bit." There are **twelve** free channels.

Every nav/T3 config *and the shipped training ini* run `num_debris_min = num_debris_max = 0` (`config/ocean/orbital.ini:9-10`; `eval_relnav.py` T3/WIDE/legacy kwargs). `fill_observations` hard-zeroes the three unused body blocks. Measured over 836 canonical decisions (`ext_rtnav_obs_slots.py`): **obs[21]–obs[32] are identically zero, min = max = 0.00000**, all 12.

But lighting one is not free — those weights received exactly zero gradient and sit at random init:

| obs[21] := | argmax flips | max\|Δlogit\| p50 / p95 |
|---|---|---|
| 0.1 | **0/836 (0.00%)** | 0.061 / 0.159 |
| 0.5 | 6/836 (0.72%) | 0.300 / 0.747 |
| 1.0 | 26/836 (**3.11%**) | 0.601 / 1.451 |
| 2.0 | 44/836 (5.26%) | 1.249 / 2.896 |

**Minimal fix if a nav-valid bit is ever wanted** (it isn't, given MAJOR-3): use obs[21] at magnitude **≤0.1**, or zero that column of the first-layer weight at warm-start load — a one-line surgery that makes the bit provably free at t=0 and learnable thereafter. Note the corollary hazard: any nav run that *re-enables* debris silently reclaims those 12 slots.

---

## NON-ISSUE-10 (C) — The LSTM tolerates estimate discontinuities. The prior for NB1 is "warm-start is fine".

The worry was that the canonical policy, trained on smooth truth, would break on the bimodal collapse at acquisition. Measured: `jump1` and `jump4` (a single isolated blind decision with median 9,276 km range error and 1.74 obs-units of target-slot perturbation, truth on both sides) score **100/100 and 100/100**. Sustained blindness for 8 decisions scores 99/100. The LSTM state is anyway zeroed at every epoch boundary for all envs (`pufferl.py:240-243`), so it never carries long-horizon history in training regardless.

One measured subtlety worth recording, because it is the sharpest version of the concern and it *passed*: over the canonical rollout, **obs[7], obs[8], obs[11], obs[12], obs[37] have exactly zero per-decision delta** — the target's a, e, ω are constants of Keplerian motion, so the policy has *never* seen those five slots move within an episode. An acquisition jump is the first time they ever would. It still costs nothing. Reference scale for those that do move: obs[13] p50 delta 0.009 / p99 0.250; obs[33] p50 0.005 / p99 0.365.

---

## Consolidated disposition

| # | severity | finding | change |
|---|---|---|---|
| 1 | **BLOCKER** | NAV-G acquisition is 17–90× the training step at its perfect-vectorization lower bound; 200–2000× scalar | Calibrated CRLB-conditioned acquisition surrogate in training; real `bls_acquire_adaptive` at eval; live `BearingMPC` recursive stage with guarded `log`/`exp` |
| 2 | MAJOR | Acquisition gates certify 2,951 km errors at reduced grid density | Never tune grid knobs without epoch-error-vs-truth; gate is not an acceptance test |
| 3 | MAJOR | "Hold until acquired" buys 0.0 pp and corrupts 5–90% of the PPO rollout | Delete the hold. No forced actions. |
| 4 | MAJOR | Risk is inverted — close-range blindness is the killer (45% @ sep<200 km, 18% @ <1000 km) | Promote close-range covariance + `blindclose` ablation to first-class gates |
| 5 | MAJOR | Chained N-ladder confounds the train×eval noise matrix | Run N1/N2/N3 as parallel independent warm-starts from N0; NB1 warm from N0 |
| 6 | MINOR | Info-list contract, float-only values, 128-tick cadence, 30-row cap, `[vec] seed` silently dropped | Follow the spec in MINOR-6 |
| 7 | MINOR | PBRS invariance broken twice; 7.9% unexplained shaping delta at BO accuracy | Keep truth-side shaping; state the deviation in the write-up |
| 8 | NON-ISSUE | float32 decode floor | ≤2.7% of σ_β at 1 km. No C accessor needed. Exercise `reconcheck` at high e. |
| 9 | NON-ISSUE | No free obs slot | 12 free (obs[21..32]); use ≤0.1 magnitude or zero the input weights if ever needed |
| 10 | NON-ISSUE | LSTM vs acquisition jump | 100/100 zero-shot; warm-start prior for NB1 is sound |

**Ship the range+bearing half (N-lineage) as designed, with fixes 5 and 6. The NB lineage does not ship until fix 1 is implemented and the surrogate's Δv-conditioning is validated against `ext_bo_filter.csv`.**