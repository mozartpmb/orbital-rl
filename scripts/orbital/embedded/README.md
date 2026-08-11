# Embedded C port of the orbital-rendezvous RL policy

Flight-software-style port of the trained PyTorch guidance policy to a standalone,
single-precision, allocation-free C implementation — plus the verification stack that
proves the port is equivalent to the prototype and quantifies its real-time margin.

**Prototype**: PufferLib `Default` policy wrapped in `LSTMWrapper`, checkpoint
`models/t3/seed42_L2_headline.pt` (T3 canonical, sha256 `75fbee2e…b104c2`), 139,281
parameters. **Result**: 100.00% action agreement over 7,048 held-out observation vectors,
max |Δlogit| 5.2e-05; 200/200 closed-loop success with a **bit-identical action stream**;
p99 inference latency 62 µs against a 60 s decision period — a **9.7 × 10⁵×** real-time margin.

---

## 1. Contents

| File | Role |
|---|---|
| `export_weights.py` | Loads the checkpoint, verifies shapes/ties, emits `policy_weights.h` (C99 hex-float `static const float` arrays) + `policy_manifest.json` |
| `policy.h` / `policy.c` | The deliverable: encoder MLP + LSTM cell + action head, single precision, no allocation |
| `policy_weights.h` | Generated. 139,281 float32 constants, 557,124 B of `.rodata` |
| `policy_manifest.json` | Generated. Machine-readable layer order, shapes, activations, gate order, provenance hashes |
| `harvest_obs.py` | Runs the torch policy on the real C env at the T3 config; records obs / logits / value / action / LSTM (h,c) / episode boundaries |
| `test_parity.py` | Replays those sequences through the C library (ctypes) and reports logit/state deltas, action agreement, and error-path unit tests |
| `closed_loop.c` | Fully-C closed loop: `orbital.h` environment + `policy.c`, no Python in the loop |
| `bench_policy.c` | Single-core latency benchmark (median/p99 over 100k calls) |
| `Makefile` | All build variants |
| `results/` | Raw captured output of every number quoted below |

---

## 2. Build & run

```bash
cd scripts/orbital/embedded

# 0. regenerate the weight header from the checkpoint (already committed-ready)
python3 export_weights.py --ckpt ../../../models/t3/seed42_L2_headline.pt

# 1. build everything (shim + benchmark + fully-C closed loop)
make all              # libpolicy.dylib, bench_policy, closed_loop
make poly             # expf-only-libm variants
make variants         # all four shared-library variants for the parity sweep
make sanitize         # ASan + UBSan build of the closed loop

# 2. harvest torch reference data from real episodes
python3 harvest_obs.py --episodes 200 --seed 123 \
    --out /tmp/emb_harvest_s123.npz --trace /tmp/emb_torch_trace.csv \
    --obs-bin /tmp/emb_obs_s123.bin --record-state
python3 harvest_obs.py --episodes 100 --seed 777 \
    --out /tmp/emb_harvest_s777.npz --trace /tmp/emb_torch_trace_s777.csv --record-state

# 3. parity (all build variants + contract unit tests)
python3 test_parity.py --variants --verbose

# 4. closed loop, fully in C, 200 held-out episodes
./closed_loop --episodes 200 --seed 123 --bench --trace /tmp/emb_c_trace.csv
diff /tmp/emb_torch_trace.csv /tmp/emb_c_trace.csv    # must be empty

# 5. benchmark
./bench_policy --calls 100000 --obs /tmp/emb_obs_s123.bin

# 6. sanitizers
ASAN_OPTIONS=detect_leaks=0 ./closed_loop_asan --episodes 25 --seed 123
```

Toolchain used for the numbers below: Apple clang, `-std=c99 -O2`, Apple M3 Max, single
core, Darwin 25.3. Reference: PyTorch 2.11.0, NumPy 1.26.4, Python 3.12.1. The host was
lightly loaded (load average ≈ 3.5 of 14 cores) during the benchmark — see the tail
discussion in §6.

---

## 3. Architecture

Recovered from the checkpoint `state_dict` (printed by `export_weights.py`, never assumed)
and from `pufferlib/pufferlib/models.py`:

```
                  obs[38]  float32, already normalised BY THE ENVIRONMENT
                     |      (orbital.h::fill_observations; the policy applies
                     |       no normalisation of its own — Default.encode_
                     |       observations() is view(B,-1).float() -> Linear)
                     v
        +------------------------------+
        | encoder   Linear 38 -> 128   |   W (128,38) row-major + b (128)
        |           GELU (exact, erf)  |   torch nn.GELU(approximate='none')
        +------------------------------+
                     | enc[128]
                     v
        +--------------------------------------------------+
        | LSTMCell 128 -> 128                              |
        |   gates = W_ih·enc + b_ih + W_hh·h + b_hh        |   W_ih,W_hh (512,128)
        |   rows   0..127 -> i   (sigmoid)                 |   b_ih,b_hh (512)
        |        128..255 -> f   (sigmoid)                 |   BOTH biases are added
        |        256..383 -> g   (tanh)                    |   (torch convention)
        |        384..511 -> o   (sigmoid)                 |
        |   c' = f*c + i*g ;  h' = o*tanh(c')              |
        +--------------------------------------------------+
                 |  h'[128]                    ^  h,c carried across decision
                 |                             |  epochs; zeroed at episode start
     +-----------+-----------+-----------------+
     v                       v
+--------------+     +----------------+
| decoder      |     | value          |
| Linear 128x16|     | Linear 128x1   |   critic head; exported and parity-checked
| raw logits   |     | (not in the    |   but NOT on the control path
+--------------+     |  control path) |
     |               +----------------+
     v
  argmax  ->  action in [0,16)   (ties -> lowest index, matches torch.argmax)
```

Action semantics (`orbital.h::ACTION_DV` / `ACTION_TAU`): 0 coast; 1–8 impulsive burns
(prograde/retrograde ±{5,10,25} m/s, radial ±10 m/s); 9/10/11 time warps of 5 min /
30 min / 1 h; 12–15 sub-5 m/s fine prograde burns. One sub-step = 60 s of simulated time;
a warp action advances τ = 5/30/60 sub-steps under one decision.

Two subtleties the port had to get right, both verified rather than assumed:

* **Tied LSTM parameters.** The checkpoint stores `lstm.*` *and* `cell.*` (271,377 stored
  elements) because `LSTMWrapper.__init__` aliases the `LSTMCell` parameters onto the
  `nn.LSTM` ones. `export_weights.py` asserts bit-identity and exports one copy → 139,281
  unique parameters. A silent divergence there would have produced a plausible-looking but
  wrong policy.
* **`forward_eval` uses the `LSTMCell`, not `nn.LSTM`.** Inference and training take
  different code paths through the same weights; the C port follows the eval path, which is
  what `eval_checkpoint.py` scores.

---

## 4. Parity results

Reference data: 300 episodes at the T3 canonical configuration (`shaping_mode=1`,
`shape_gamma=1.0`, `phase_gap_mode=1`, `phase_obs_mode=1`, `episode_cap_steps=3000`,
`cap_terminal_reward=0.0`, `e_max` 0.05 both, `init_phase_gap_max=π`, `valid_init_only=1`,
no debris), held-out seeds 123 (200 eps) and 777 (100 eps) → **7,048 decision epochs**.
Sequences are replayed through the real C API including the episode-boundary
`policy_reset()`, so recurrent-state carry is under test, not bypassed. Action mix in the
reference set: 3,613 warp-1 h, 379 warp-30 min, 3,056 burns spread over all six
prograde/retrograde magnitudes (±5, ±10, ±25 m/s) — warp↔burn transitions are dense
(mean episode = 23.5 decisions containing both). The greedy policy never emits coast,
radial, warp-5 min or the ±1/±2 m/s fine burns at this configuration, so those 8 action
indices are not exercised as *outputs*; all 16 logits are nonetheless compared at every
epoch, so every output unit of the decoder is numerically verified.

| build | action agreement | max \|Δlogit\| | p99 \|Δlogit\| | mean \|Δlogit\| | max \|Δvalue\| | max \|Δh\| | max \|Δc\| |
|---|---|---|---|---|---|---|---|
| **GELU exact `erff`, float32 acc (default)** | **7048/7048 = 100.0000%** | 5.215e-05 | 1.574e-05 | 2.156e-06 | 9.537e-07 | 2.179e-05 | 2.345e-05 |
| GELU erf-poly (A&S 7.1.26, `expf` only) | 7048/7048 = 100.0000% | 5.198e-05 | 1.574e-05 | 2.119e-06 | 9.537e-07 | 1.240e-05 | 1.290e-05 |
| GELU tanh approximation | 7047/7048 = 99.9858% | 1.561e-02 | 4.886e-03 | 7.658e-04 | 1.280e-04 | 7.705e-03 | 8.313e-03 |
| GELU exact, double accumulation | 7048/7048 = 100.0000% | 4.721e-05 | 1.144e-05 | 1.386e-06 | 4.768e-07 | 1.496e-05 | 1.609e-05 |

Raw output: `results/parity.txt`.

**Is 5.2e-05 "float32 accumulation noise"?** Logits are O(10); one ULP at magnitude 10 is
9.5e-07, so the residual is ~55 ULP — the expected scale for a 128-term dot product summed
in a different order than torch's BLAS kernel, then fed back through the recurrence.
Two checks confirm the diagnosis rather than assuming it:

* Switching our accumulator to `double` moves the max only 5.215e-05 → 4.721e-05 (-9%).
  If our summation order were the dominant error source it would have collapsed. It does
  not, because torch's own float32 kernel carries comparable rounding — the two
  implementations straddle the exact answer.
* Error does **not** compound with recurrent depth: max |Δlogit| by within-episode step
  index is 2.8e-05 (step 0), 3.2e-05 (step 5), 4.1e-05 (step 20), 8.1e-06 (step 40). The
  LSTM's saturating gates contract the perturbation instead of amplifying it.

**Why 100% agreement is structurally safe, not lucky.** The decision margin (top-1 minus
top-2 logit) over the reference set has median 7.08, 1st percentile 0.118, minimum 0.0022.
The minimum margin is **42.6×** the maximum observed logit error, so no observed decision
was anywhere near a flip.

**The tanh-GELU variant is the counter-example that makes the above meaningful.** It costs
~300× the logit error and flips exactly one decision — seed 777, episode 84, step 26, at a
margin of 2.5e-03, warp-1 h → warp-30 min. That episode still succeeds, but it proves the
parity test has the resolution to detect a wrong activation choice. **`approximate='tanh'`
is not a free substitution for torch's default GELU.**

### 4.1 Contract / error-path unit tests (`results/parity.txt`)

All PASS: uninitialised state rejected (`POLICY_ERR_UNINIT`); NULL state and NULL obs
rejected (`POLICY_ERR_NULL`); NaN and Inf observations rejected (`POLICY_ERR_OBS_NONFINITE`);
**recurrent state and step counter bit-unchanged after a rejected frame**; repeated
inference from the same state is bit-identical; `policy_reset` zeroes h, c and the counter;
`policy_act` agrees with `policy_infer`'s argmax.

---

## 5. Closed-loop results

Fully-C loop: `closed_loop.c` compiles `orbital.h` (the same environment the policy was
trained in) and `policy.c` into one binary. `configure_t3_canonical()` reproduces
`binding.c::my_init()` field for field, and the seeding replicates
`env_binding.h::vec_reset` (`srand(i + seed*num_envs)`, i=0, num_envs=1) — so the episode
stream is identical to the Python harness at the same seed and the comparison is exact
rather than statistical.

| run | episodes | physical success | mean reward | decisions | action stream vs torch |
|---|---|---|---|---|---|
| seed 123, default build | 200 | **200/200 (100.0%)** | 7.420 | 4,742 | **bit-identical** (md5 `5f4aae71…`) |
| seed 123, `expf`-only build | 200 | **200/200 (100.0%)** | 7.420 | 4,742 | **bit-identical** (same md5) |
| seed 777, default build | 100 | **100/100 (100.0%)** | 7.595 | 2,306 | **bit-identical** (md5 `c75bbef5…`) |
| torch reference (`eval_checkpoint.py`, seed 123) | 200 | 200/200 (100.0%) | 7.42 | 4,742 | — |

Terminal causes: `success` only — no timeout, collision, stranding or escape in any run.
`policy faults: 0` (no rejected frames). Raw output: `results/closed_loop.txt`.

This is a stronger statement than matching success rates: over 7,048 decisions across two
seeds the C policy chose **the same action every single time**, so the two implementations
flew the same trajectories, not merely equally good ones.

---

## 6. Benchmark and real-time margin

`./bench_policy --calls 100000 --obs /tmp/emb_obs_s123.bin`, single core, `-O2`, warmup
5,000 calls excluded, real harvested observations, per-call timing via the raw mach
timebase (`clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`, ~42 ns tick).

| metric | default (`erff`) | `expf`-only (poly) |
|---|---|---|
| min | 45.13 µs | 43.42 µs |
| **p50** | **50.50 µs** | 48.46 µs |
| mean | 50.72 µs | 48.79 µs |
| p90 | 52.29 µs | 50.13 µs |
| **p99** | **62.04 µs** | 59.29 µs |
| p99.9 | 72.71 µs | 70.17 µs |
| max | 198 µs | 176 µs |
| throughput | 19,709 calls/s | 20,489 calls/s |

Repeatability across 5 independent 100k-call runs: p50 49.5–52.3 µs, p99 58.3–63.5 µs
(`results/bench.txt`). The p99.9/max tail is OS scheduler preemption on a loaded
non-real-time host — the algorithm itself has a fixed MAC count with no data-dependent
control flow, which is why min and p50 sit 1.1× apart.

### Real-time margin against the 60 s decision period

The guidance epoch is one environment sub-step, DT = 60 s (warp actions consume several
epochs under one decision, so 60 s is the *worst case* cadence).

| quantity | value |
|---|---|
| decision period | 60 s |
| inference p99 | 6.204e-05 s |
| **margin (period / p99)** | **967,102×  (9.7 × 10⁵)** |
| duty cycle at p99 | 1.034e-06 (0.000103 % of the epoch) |
| CPU per 24.3 h episode | 23.7 decisions × 50 µs ≈ **1.2 ms** |

Read plainly: the guidance computation consumes about one part in a million of its time
budget. A processor **~10⁵ times slower** than one M3 Max performance core would still
close the loop with 10× margin, which puts the workload comfortably inside rad-hard
class hardware (LEON4/GR740, RAD5545) without an accelerator, an FPU-heavy redesign, or
fixed-point quantisation. The binding real-time constraint for this GNC stack is not
the neural network.

### Fully-C environment + policy throughput

`./closed_loop --episodes 200 --seed 123 --bench` (200 episodes = 291,798 sub-steps =
202 days of simulated mission time, executed in 0.294 s):

| metric | value |
|---|---|
| decisions/s | 16,153 |
| environment sub-steps/s (whole loop) | 993,960 |
| policy share of loop time | 81.8 % (50.7 µs/decision) |
| environment share | 17.9 % (11.1 µs/decision, 0.180 µs/sub-step) |
| environment alone | 5.55 M sub-steps/s |

The policy, not the Keplerian propagator, dominates the closed loop — 61.5 sub-steps of
two-body propagation are cheaper than one forward pass.

---

## 7. Flight-software discipline notes

**Determinism.** No RNG, no time or environment dependence, no uninitialised reads, no
data-dependent branching that changes arithmetic order. The dot-product accumulation order
is fixed and documented (strictly ascending index). Identical `(state, obs)` yields
bit-identical logits — unit-tested in `test_parity.py`. The weight header uses C99
*hexadecimal* float literals so the compiled constants are bit-exact against the
checkpoint tensors regardless of the compiler's decimal conversion; `policy_manifest.json`
carries the checkpoint sha256 and a sha256 of the concatenated weight blob, and both are
compiled in and queryable at runtime (`policy_ckpt_sha256()`), so a flight image can be
traced back to the exact `.pt` that produced it.

**Memory.** No `malloc`/`free`/`alloca`/VLA anywhere in `policy.c` — verified by
inspection and by the measured static frame size.

| region | size | note |
|---|---|---|
| weights (read-only) | 557,124 B | 139,281 float32 constants, shareable across instances |
| code (`__TEXT,__text`) | 2,668 B | |
| whole `policy.o` image | 560,152 B | `size -m policy.o` |
| `policy_state_t` | 1,032 B | caller-owned; one per controlled vehicle |
| `policy_infer` stack frame | 3,808 B | `clang -O2 -fstack-usage`, static (bounded, non-recursive) |
| `policy_act` stack frame | 32 B | |
| **worst-case call-chain stack** | **3,840 B** | |

Total footprint for one controller: 560 KB read-only + 1,032 B RAM + a 3.9 KB stack
budget. No heap, so no fragmentation and no allocation-failure path to test.

**Error handling.** Every entry point validates its inputs and returns a typed status
(`policy_status_t`); nothing aborts, asserts, logs, or touches `errno`. The key property is
**fail-safe state commit**: encoder → LSTM → heads compute into stack temporaries, results
are checked for finiteness, and only then are `h`/`c` written. A single corrupt sensor frame
therefore cannot poison the recurrent memory — it is rejected, the filter state is preserved
bit-exactly, and the caller decides what to do. `closed_loop.c` demonstrates the intended
policy: a rejected frame commands **coast** (the physically safe no-op) and increments a
fault counter, rather than propagating an undefined action to the actuators.

**Reentrancy.** All mutable state is behind the caller's pointer. No file-scope mutable
data, so N vehicles or N Monte-Carlo threads can share one read-only weight image.

**Bounded execution.** Work per call is a fixed 138,112 multiply-accumulates (4,864 encoder
+ 65,536 W_ih + 65,536 W_hh + 2,048 decoder + 128 value) plus 128 GELU, 384 sigmoid and 256
tanh evaluations — independent of input values. There are no loops whose trip count depends
on data, so WCET analysis reduces to a straight-line count.

**Static analysis / sanitizers.** `policy.c` compiles clean under
`-std=c99 -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion -Wcast-align
-Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes` with zero diagnostics. The full
closed loop runs clean under AddressSanitizer + UndefinedBehaviorSanitizer
(`results/sanitize.txt`). Shape agreement between `policy.c` and the generated weight header
is enforced by compile-time static assertions, so a re-export with a different obs
dimension or action count fails the build instead of running.

**Libm dependency.** The default build calls `expf`, `tanhf` and `erff`. `erff` exists only
because torch's default GELU is the exact erf form; `make poly` /
`-DPOLICY_GELU_POLY` substitutes an Abramowitz & Stegun 7.1.26 erf using `expf` alone,
reducing the math-library surface to `expf`/`tanhf` at **measured-identical parity**
(100.0000% agreement, max |Δlogit| 5.198e-05 vs 5.215e-05) and identical closed-loop
behaviour (200/200, same action stream). That is the build to take to a target whose math
library is minimal or must be qualified.

---

## 8. Scope and honest limitations

* Parity is **numerical equivalence at float32 accumulation noise**, not bit-exactness.
  Bit-exactness against torch is not achievable without reimplementing its BLAS kernel's
  summation order, and would be the wrong target: the meaningful contract is that the
  *decisions* agree, which is verified exhaustively over 7,048 epochs with a measured 42.6×
  margin to the nearest possible flip.
* Verified on **one checkpoint** (T3 canonical, Discrete-16 head, 38-dim obs). The exporter
  reads dimensions from the `state_dict` and the C side static-asserts them, so a different
  head size fails the build rather than silently misbehaving — but the 20-action T4 and
  wide-envelope checkpoints have not been run through this pipeline.
* Verified on **one architecture** (arm64 / Apple clang -O2). The port has no arch-specific
  code, but the latency numbers are host-specific and a rad-hard target would need its own
  measurement; the margin is large enough that the conclusion is robust to five orders of
  magnitude.
* The **environment's** simplifications are unchanged and inherited: 2D coplanar, two-body,
  impulsive burns, no debris, 30 km / 50 m/s far-field success box. This deliverable is
  about the guidance software's implementation fidelity, not about widening the flight
  envelope.
* `closed_loop.c` duplicates the T3 kwargs as C literals. It is pinned to the Python
  defaults as of this writing; if `binding.c::my_init` gains a kwarg, that block must be
  updated. The bit-identical action-stream check is the regression test that catches it.
