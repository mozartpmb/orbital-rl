# NAV-H — Filter-in-the-Loop TRAINING Architecture (recon memo)

All claims below are code-referenced or measured on this machine (Apple silicon, 14 physical cores, numpy 1.26.4, spawn start-method). Nothing existing was modified; new files are listed in §7.

---

## 1. Insertion point — decided: **PufferEnv subclass, worker-side, in-place obs write**

### 1.1 How `puffer train` actually drives the env (read, not assumed)

`pufferl.load_env` (`pufferl.py:1230-1233`) → `pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])`. `orbital.ini` has **no `[vec]` section**, so `default.ini` applies: `backend=Multiprocessing, num_envs=2, num_workers=auto→2, batch_size=auto→1, zero_copy=True`. `[env] num_envs=1024` is an *Orbital kwarg* (native C vectorization), so the shape is **2 worker processes × 1024 C envs = 2048 agents, `agents_per_batch`=1024**.

The load-bearing chain:

| step | code | consequence |
|---|---|---|
| worker allocates numpy views onto `RawArray` shm | `vector.py:176-187` | obs/rew/term buffers are shared with the driver |
| `if is_native and num_envs == 1: envs = env_creators[0](..., buf=buf, seed=seed)` | `vector.py:190` | **the worker constructs the env class itself** → any Python subclass state lives in the worker |
| `set_buffers` binds `env.observations = buf['observations']` | `pufferlib.py:38` | `self.observations` *is* the shm view |
| `binding.vec_init(self.observations, ...)` | `orbital.py:179-181` | C writes into that same memory |
| worker: `elif sem == STEP: envs.step(atn_arr)` then `semaphores[worker_idx] = MAIN` | `vector.py:213, 224` | the driver cannot read until `step()` **returns** — post-processing inside `step()` is race-free by construction |
| driver: `o = buf['observations'][w_slice].reshape(...)` | `vector.py:427` | driver sees whatever the worker last wrote |

**Therefore:** a subclass that mutates `self.observations[...]` in place *after* `binding.vec_step()` inside `step()` is (a) visible to the driver, (b) able to hold per-env filter state in the worker, (c) race-free.

### 1.2 Verified empirically — `ext_nav_wrapper_probe.py`

```
physical cores: 14  start method: spawn
[2] backend survival
  Serial (control)      sentinel unique = [3.25]
  Multiprocessing: 2 workers x 256 envs
    driver obs shape (512, 38), sentinel[17] unique = [7.5]
    after 5 steps: counter slot[18] min=5 max=5
    VERDICT worker-side in-place obs write visible to driver: PASS
    VERDICT worker-side per-env state persists across steps:   PASS
```

### 1.3 Options considered and rejected

| option | verdict |
|---|---|
| **A. `PufferEnv` subclass (chosen)** | Survives Multiprocessing (verified), Serial (verified), and native-`PufferEnv` backend (the vecenv *is* the env). Filter state co-located with the env it filters. Policy artifact is self-contained: the same class is used at eval. |
| B. Post-process `o` in `pufferl.evaluate` before the forward (`pufferl.py:248-292`) | **Reject.** `o` is a live shm view; filter state would have to be keyed by `env_id` in the driver; the added work lands on the *serial* driver thread instead of the parallel workers; and it requires editing `pufferl.py`, the file every regression anchor runs through. Worst: the resulting policy would not be evaluable without the trainer patch. |
| C. EKF in C inside `orbital.h` | **Reject for now.** The benchmark (§2) shows batched numpy is already ~compute-bound at the physics, not overhead-bound (1.30 µs/env asymptote at B=4096 vs 1.35 at B=1024), so a C port buys ~2-3×, not 100×, while touching the file that carries the bit-exact anchors and throwing away the already-validated `ekf.py`/`orbital_math.py` (NEES 0.934-0.947, symplectic 9.7e-7). Revisit only if §2's projection proves binding. |
| D. Serial backend + driver-side wrapper | Works but single-process; debugging only. |

### 1.4 Registration

`environment.py:168-178`: `env_creator('puffer_orbital_nav')` → `importlib.import_module('pufferlib.ocean.orbital_nav.orbital_nav')` + `MAKE_FUNCTIONS['orbital_nav']`. `load_config` (`pufferl.py:1291-1305`) globs `config/**/*.ini` and matches `[base] env_name`. Policy classes come from the shared `pufferlib.ocean.torch` (`pufferl.py:1238-1247`) — no new policy code.

**Recommended:** new package `pufferlib/ocean/orbital_nav/{__init__.py,orbital_nav.py}` with `class OrbitalNav(Orbital)` importing the *existing* `pufferlib.ocean.orbital.binding` (no C rebuild, no new extension in `setup.py`), one additive line in `MAKE_FUNCTIONS`, and its own `config/ocean/orbital_nav.ini` carrying sensor/cadence/curriculum defaults. This leaves `puffer_orbital` byte-identical, so every T3/T4 anchor keeps reproducing.

Rejected alternative: a `nav_mode=` kwarg inside `orbital.py`. Has precedent (all T3 flags) and is default-off-safe, but pollutes the file all anchors run through and forces nav defaults into `orbital.ini`.

### 1.5 What eval_relnav did, and what generalizes

`eval_relnav.py` is serial/single-env and **modifies nothing in C** — it inverts the truth observation (`recover_states_t3`, lines 172-200), synthesizes the measurement (`ekf.measure`), runs the EKF, and re-emits target-derived slots (`build_obs_t3`, lines 203-224). Generalizes wholesale, provided each scalar step becomes a batched numpy op:

- **Decode-from-obs, not a C getter.** Costs 0.263 ms/decision at B=1024 (§2) — 2% of env time — and it is *already regression-proven*: `recon` (rebuild path, zero estimation error) == `truth` at 200/200 in all three shipped configs (`t3_relnav_corrected.csv`). This means **NAV-H needs zero changes to `orbital.h`, `binding.c`, `pufferl.py`, or `vector.py`.** float32 obs quantization injects ≲1 m of pseudo-truth error, 50× below σ_ρ.
- **Truth propagation during warps** (`eval_relnav.py:489-510`): the harness propagates *both* truth states itself with `om.propagate_cartesian` (verified 0.29 m over 99 steps vs the live env). Safe because warps never burn; and for τ=1 no sub-stepping is needed at all, since the post-step obs *is* the measurement epoch. Batch this.
- Carry over the `--inject` group ablation, `reconcheck` stage, and the range-binned `err/(σ_β·ρ)` table verbatim as gates.
- `orbital_math.ACTION_TAU` is 16-long; the C env is Discrete-20 (`orbital.h:117-124`, `{...,180,360,1,1}`). **Must be extended** before the nav lineage touches actions 16-19.

---

## 2. Performance budget — measured, not estimated

### 2.1 Env baseline (B=1024, single process, T3 config)

```
tau=1  (coast) 2386 decisions/s  → 0.419 ms/decision   2.44M agent-SPS
tau=5  (warp5)  547 decisions/s  → 1.830 ms/decision   2.80M sub-step-SPS
tau=60 (warp1h) 47.8 decisions/s → 20.92 ms/decision   2.94M sub-step-SPS
fit: env_ms(B=1024) = 0.070 + 0.348·τ
```

Trainer profile from the real T3 L1 run (`/tmp/t3_L1_s31415_train.log`, final epoch): **SPS 139.4K**, Env 33%, Forward 19%, Learn 27%. Earlier epochs 71-72K.

### 2.2 Batched EKF microbenchmark (B=1024, float64 — float32 loses metres on 7e6 m positions)

`ext_nav_ekf_opt.py`, OMP_NUM_THREADS=1:

| variant | ms/tick | µs/env |
|---|---|---|
| v0 baseline (12-iter Newton, FD-STM, einsum, Joseph) | 4.00 | 3.90 |
| v1 4-iter Newton | 3.17 | 3.10 |
| **v2 `+ matmul` instead of einsum — SHIP THIS** | **1.353** | **1.32** |
| v3/v4 analytic-STM + short-form covariance (*FLOP proxy, lower bound only*) | 0.93-1.09 | 0.91-1.07 |

Components at 4 iters: `stm_fd` 0.651 ms (48%), `update matmul` 0.512 ms (38%), `prop` 0.100 ms, `FPF'` 0.145 ms. Obs decode+re-encode 0.263 ms.

Findings that matter:
- **Newton converges in 3 iterations at dt=60 s** — `max|Δpos| = 0.000e+00 m` vs a 20-iteration reference. The shipped `max_iter=40` is 4× wasted work.
- **`einsum` → `matmul` is a free 2.3×** on (N,4,4) chains.
- **BLAS threading is negative**: OMP=8 is *slower* than OMP=1 (1.587 vs 1.353 ms). Workers **must** export `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`, or 4 workers × N BLAS threads will thrash 14 cores.
- Scaling 256→4096 envs: 1.67→1.30 µs/env. Compute-bound, not overhead-bound → **the C-port lever is small; the tick-count lever is large.**

### 2.3 Ticks per decision — measured from the canonical policy

`ext_nav_tau_hist.py` on `models/t3/seed42_L2_headline.pt`, 38,400 greedy decisions:

```
action 11 (warp-1h, tau 60): 51.0%   action 3 (tau 1): 19.5%   action 6 (tau 1): 18.4%
action 10 (warp-30m, tau 30): 5.0%   MEAN TAU = 32.50 sub-steps/decision
mean ticks/decision: nav60 32.50 | cap12 7.15 | nav300 6.50 | cap6 3.80 | cap4 2.68
```

### 2.4 Projection (`ext_nav_sps_projection.csv`)

env = 11.38 ms/decision at mean τ=32.5, B=1024.

| nav cadence | ticks/dec | nav/env | SPS (2w, as shipped) | SPS (4w × 512 envs) | 50M-step wall 2w / 4w | status |
|---|---|---|---|---|---|---|
| perdec (1/decision) | 1.00 | 0.14 | 133K | 162K | 6.3 / 5.1 min | **rejected** — 50.5% closed-loop (T3) |
| cap-6 | 3.80 | 0.47 | 121K | 152K | 6.9 / 5.5 min | unvalidated |
| nav300 | 6.50 | 0.80 | 110K | 143K | 7.5 / 5.8 min | **validated** 100%@1×, 99.5%@10× |
| cap-12 | 7.15 | 0.87 | 108K | 141K | 7.7 / 5.9 min | ≡ nav300 at τ=60 |
| **nav60** | 32.50 | 3.89 | **61K** | **92K** | 13.6 / 9.1 min | **validated** 100%@1×, 96%@30× — headline |

**Conclusion: SPS is not the binding constraint.** Worst case (nav60, shipped 2-worker config) is a 2.28× slowdown that costs **~7 extra minutes per 50M-step rung**. Take nav60 — the most defensible sensor model and the one with the best degraded-noise curve — and do not pre-optimize. Two free levers if it ever matters: (i) rebalance to `--vec.num-envs 4 --vec.num-workers 4` with `--env.num-envs 512`, which keeps `agents_per_batch`=1024 and total agents 2048 *bit-identical to the trainer* while doubling worker parallelism (14 cores available, 2 in use) → 92K; (ii) a validated analytic STM → ~1.2× more.

**Discrete-20 hazard:** actions 16/17 are τ=180/360. At nav60 that is 360 filter ticks = 487 ms inside one `env.step()`. The MEO/nav lineage must either cap ticks-per-decision (cap-12 ⇒ 30-min nav gap at τ=360) or exclude 3h/6h warps. Flag before any M-lineage nav run.

---

## 3. Reset, seeding, curriculum

### 3.1 Episode boundaries — solved, no cross-process bookkeeping

The C env **autoresets inside `c_step`** (`orbital.h:1376, 1436, 1494`), and `c_reset` ends with `fill_observations` (`orbital.h:1353`). Verified in `ext_nav_wrapper_probe.py [1]`: at terminal steps the LVLH obs slots jump by median **1.356** (min 0.058) vs ~1e-4 for a same-episode step.

**⇒ the observation returned with `terminals[i]=1` is already the NEXT episode's first obs.** So the wrapper's rule is exactly:

```
for rows where terminals (or terminals|truncations): ekf.initialize(...)   # new episode
for all other rows:                                  ekf.predict/update    # this episode
```

`truncations` is never written by the orbital C env (grep: zero hits in `orbital.h`/`binding.c`) — all terminations flow through `terminals`. `pufferl.py:253` uses `d + t`, so either is safe; use `d | t`.

Note the deliberate asymmetry: the **filter resets per episode; the LSTM does not** (`pufferl.py:240-243` zeroes `lstm_h/c` once per `evaluate()` call, not per episode). That is correct — the filter is a physical device, the LSTM state is a BPTT artifact — but it means training-time and `eval_relnav`-time LSTM handling already differ, and that pre-existing mismatch should be stated in the write-up rather than "fixed".

### 3.2 Measurement-noise seeding

`vector.py:336` `seed_i = seed + i` is passed to the env constructor per worker; `vector.py:480` `send_pipes[i].send(seed+i)` re-seeds at `async_reset`. Store the constructor seed and derive `np.random.default_rng([seed_i, NAV_SALT])` — reproducible per worker, independent across workers, independent of the C RNG. Re-seed nothing on autoreset (the stream must not restart per episode, or noise correlates with episode index).

Wart to be aware of, not fix: `vector.py:273` constructs a **driver-side `driver_env`** with no buf/seed purely for space introspection — it allocates a full 1024-env C env (and would allocate the filter arrays) in the driver. Never stepped. Keep the wrapper's `__init__` allocation lazy or cheap.

### 3.3 Divergence / clipping guard — the one thing eval didn't need and training does

`single_observation_space` is `Box(-2, 2)` (`orbital.py:156-158`), but a diverged estimate produces unbounded slots (`a_est ≤ 0` ⇒ obs[7] garbage). `eval_relnav` only guards NaN (`np.nan_to_num`, line 224). Across 200-episode evals that is tolerable; across 50M training steps a diverged filter feeds poison to the value head for thousands of consecutive steps.

Measured divergence rates (`t3_relnav_corrected.csv`): **0.00% up to 100× noise at nav60**, 0.5% at 300×. So this is a safety net for the *bearings-only* arm, where weak observability makes divergence the expected failure mode. Required in the wrapper:
1. `np.clip(obs, -2, 2)` after re-encode, plus a clip-event counter;
2. explicit divergence test (`a_est ≤ 0`, or `trace(P) > threshold`) → **re-initialize from the current measurement**, count it;
3. surface `nav_diverge_rate`, `nav_clip_rate`, `nav_nees_med` through the existing `vec_log`/info path so they appear on the training dashboard next to `perf`.

### 3.4 Curriculum — fresh vs warm-start

**The T3 warm-start lesson does not transfer here, and the reason matters.** The memory entries (`phase4_r5_results`, `warmstart_continuation_fails`) are about **changing the reward on a committed policy** — R5's four reshape variants all collapsed to <10%, and plasticity (R3) was the prerequisite. NAV-H changes the **observation accuracy**, not the reward. Different perturbation class: the obs *layout* is byte-identical (the filter always emits the same 38 slots), only its noise increases — that is textbook observation-domain randomization, which anneals safely, and the L1→L2→L3→WL4→M5 ladder is 5-seed/4-seed evidence that warm-starting across rungs works *when the reward is held fixed*.

More decisive: **there is nothing to learn at 1×.** The truth-trained canonical already scores **100.0% on estimates** at nav60 1× and **99.5% at 10×** zero-shot. A from-scratch run at 1× can at best tie the parent. The regime where estimate-training must buy something is ≥30× (96.0%), 100× (81.5%), 300× (61.0%) — and bearings-only, which is strictly harder than any of those.

**Recommended ladder** (env flags frozen at the T3 §5 set — `shaping_mode 1, shape_gamma 1.0, phase_gap_mode 1, phase_obs_mode 1, episode_cap_steps 3000, cap_terminal_reward 0.0, valid_init_only 1` — so the *only* variable is the observation pipeline):

| rung | init | condition | gate |
|---|---|---|---|
| N0 anchor | — | `models/t3/seed42_L2_headline.pt`, truth-mode eval | must reproduce 200/200 before anything trains |
| N1 | warm from N0 | range+bearing, nav60, 30× noise | ≥ zero-shot 96.0% |
| N2 | warm from N1 | 100× | ≥ zero-shot 81.5% |
| N3 | warm from N2 | 300× | ≥ zero-shot 61.0% |
| **NB1** (headline) | **both** fresh **and** warm from N0 | **bearings-only** (range removed), nav60, 1× | run both — see below |
| NB2 | warm from NB1 | bearings-only, 3×/10× | multi-seed 3 seeds |

Fresh **is** required for the bearings-only arm, and only there: dropping range changes the *observability structure*, not just the noise scale — the target's range is unobservable from a single bearing and becomes observable only through dynamics plus chaser-induced parallax. The truth-trained parent has never had to maneuver for information, so a warm start may sit in a basin where the informative behavior is off-policy. Run fresh and warm as separately-ablated arms (the T3 precedent for imitation arms), and report both. Note the T3 seed-sensitivity finding: fresh Stage-1 runs are seed-fragile (seed 1337 historically flatlined), so **3 seeds minimum on any fresh arm**.

Two hard constraints, both from the campaign's own scar tissue:
- **Never change the reward and the observation in the same run.** Every T3 collapse traces to a compound change.
- **Keep the 38-dim layout fixed.** If the bearings-only filter wants to expose uncertainty to the policy (e.g. `trace(P)` as an observation — which is the *interesting* GN&C move, since it lets the agent trade Δv for information), that is a **separate, later, ablated arm** with its own obs-dim change and its own from-scratch requirement, not a free rider on NB1.

---

## 4. Eval protocol

Every cell: `eval_checkpoint.py` "Physical success" line, greedy, held-out seed 123, 200 episodes, `"0 gave-up inits excluded"` printed, cause histogram, Clopper–Pearson LB (0/200 ⇒ 98.2% LB; pool for family claims).

**(a) Native — on estimates.** The training distribution. Report at train-noise and at ±1 decade.

**(b) Truth — the estimate-training tax.** Same checkpoint, `mode="truth"` passthrough. This is the question the task asks and the one a Draper reviewer will ask: *did learning to fly on a filter cost you anything when the filter is perfect?* Report Δ(truth − native) with sign. Prediction to state in advance so it is falsifiable: ≤ 2pp, since the truth obs is a zero-noise draw from the training distribution.

**(c) `recon` control.** The zero-estimation-error rebuild path must equal truth *exactly*. This is the test that catches a wrong `obs_alt_scale_m`/`lvlh_scale_m` (`eval_relnav.py:595-633` — a 5× scale error is a 0.5 absolute residual on obs[7], 4e6× the float32 round-trip floor). Run `--stage reconcheck` as a pre-flight on every new config, report bit-identical-step fraction and policy-argmax mismatches.

**(d) Train-noise × eval-noise matrix.** 4×6 (train ∈ {truth, 1×, 30×, bearings-only} × eval ∈ {1,3,10,30,100,300}×). The claim "training on estimates buys robustness" is only supported if the off-diagonal beats the truth-trained row.

**(e) Filter consistency, per condition.** NEES/NIS in-bounds fraction with the χ² bounds, split into below/above (only *above* is a safety problem — reported covariance understating true error), settled pos/vel RMSE, divergence fraction, and the range-binned `err/(σ_β·ρ)` table. The T4 finding is the headline here: **NEES degrades a full decade of sensor noise before success does** (nav60: NEES-in-bounds 0.94→0.84→0.65 across 1×/10×/30× while success is 100/99.5/96.0%) — that is the argument for reporting NEES at all, and it is the single most Draper-legible number in the project.

**(f) NAV-F observability behavior.** Whatever NAV-F specifies, plus the metrics this architecture makes cheap: Δv spent before first covariance collapse, `trace(P)` trajectory vs burn events, correlation between burn epochs and ΔP, and the counterfactual "same policy, frozen-P filter". For the bearings-only arm these are the *result*, not diagnostics.

**(g) Regression anchors, unchanged.** Legacy ckpt 26/200 bit-exact; T3 canonical 200/200 at the headline; `puffer_orbital` (not `puffer_orbital_nav`) must remain byte-identical — assert by re-running the T3 canonical eval after the branch lands.

---

## 5. Residual risks

1. **Non-Markov obs in the PPO buffer.** The policy's input now depends on the filter's measurement history. Rollouts store the obs actually seen, so PPO is unaffected mechanically — but the env is now genuinely partially observed and the value function's `explained_variance` (0.999 in the T3 log) will drop. That is expected, not a bug; state it up front so it is not later read as a regression.
2. **float32 obs quantization in the decode path** injects ≲1 m of pseudo-truth error into measurement synthesis. 50× below σ_ρ=50 m; becomes relevant only if σ_ρ is driven below ~10 m. Gate: `reconcheck`.
3. **`ACTION_TAU` in `orbital_math.py` is 16-long** vs the env's 20 actions. Silent wrong-dt if a nav run ever enables actions 16-19. Fix before, not after.
4. **BLAS oversubscription** with >2 workers (§2.2). Must be pinned in the worker env.
5. **v3/v4 analytic-STM timings are FLOP-shape lower bounds only** — the implementation in `ext_nav_ekf_opt.py` is deliberately incomplete and documented as such. Do not lift it into a filter without validating against `stm_fd` at the `_stm_conditioning_test` symplecticity standard.

---

## 6. One-paragraph recommendation

Build `pufferlib/ocean/orbital_nav/orbital_nav.py` as `class OrbitalNav(Orbital)` overriding `reset()`/`step()`: decode truth from the post-step obs (batched, 0.26 ms), sub-propagate both truth states through warps, run a batched numpy EKF at **60 s nav cadence** (v2 path, 1.35 ms/tick, OMP pinned to 1), re-encode the target-derived slots, clip to ±2, re-initialize rows where `terminals|truncations`, and log `nav_*` diagnostics. Zero C changes, zero trainer changes, one additive line in `MAKE_FUNCTIONS`, one new `.ini`. Accept the measured 2.28× slowdown (139K→61K SPS, +7 min per 50M-step rung) rather than trading away the validated sensor model; if it ever binds, rebalance to 4 workers × 512 envs for 92K at identical trainer-facing batch shape. Train by warm-starting the noise ladder from `seed42_L2_headline.pt` with the reward frozen, and run the bearings-only rung **fresh and warm, 3 seeds each**, because that rung — and only that rung — changes observability rather than noise.

---

## 7. Artifacts

Scripts (all read-only probes, `/Users/pete/space_training/scripts/orbital/ext_recon/`):
- `ext_nav_wrapper_probe.py` — insertion-point + autoreset verification (both PASS)
- `ext_nav_ekf_bench.py` — batched EKF cost, first pass
- `ext_nav_ekf_opt.py` — optimization variants, component split, B-scaling, BLAS-thread sensitivity
- `ext_nav_tau_hist.py` — canonical-policy action/τ histogram (mean τ = 32.50)
- `ext_nav_sps_projection.py` — the §2.4 table from measured constants

CSVs (`/Users/pete/space_training/web_data/results/`): `ext_nav_ekf_bench.csv`, `ext_nav_sps_projection.csv`