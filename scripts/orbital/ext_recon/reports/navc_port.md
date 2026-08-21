# T14 — C port of the nav filter hot path

**Recommendation: MERGE, with `nav_filter_impl='py'` as the default and `'c'`
opt-in per campaign. 4.35x on W1xnav@K=0, 6.06x on the 7-cell mixture, every
gate green, and the Python implementation kept as a permanent oracle.**

## 1. Profile first — and it named a smaller target than "the filter"

cProfile self-time, batched numpy, no C written yet:

| category | W1 @ K=0 | mixture @ K=120 |
|---|---|---|
| **J2 propagation kernel (FD STM)** | dominant | **77.4%** |
| MSC6 chart + EKF update | — | 5.2% |
| CRLB surrogate | 0.1% | 0.2% |
| obs encode/decode | 0.0% | 0.0% |
| env C step (binding) | 1.2% | 0.4% |
| **filter total** | **70.9%** | **82.6%** |

Top frame in both: `cartesian_to_elements_3d` alone, called 241,956 times in a
12-decision W1 rollout. The reason is `stm_fd_j2`: it central-differences the
propagator, so ONE predict runs **13 full J2 propagations**, each of which does
a complete Cartesian -> elements -> Cartesian round trip.

So the port target is `stm_fd_j2` + its inner `propagate_cartesian_j2`, and
nothing else. The MSC6 chart, the EKF update, the surrogate, the batch IOD and
acquisition gating all stay in Python — together ~5% of runtime, and the place
the subtle linear algebra lives.

## 2. Speedup, end to end

| config | py | c | speedup |
|---|---|---|---|
| W1_driftwait @ nav_max_ticks=0 | 8.2 env-steps/s | 35.5 | **4.35x** |
| 7-cell mixture @ K=120 | 68.3 | 413.7 | **6.06x** |

The mixture speedup EXCEEDS naive Amdahl on 82.6% (max 5.7x). That is not an
error: the C path also deletes the kernel's numpy temporaries and its 530,730
`np.errstate`/`seterr` calls, which my profile had binned under "other python"
because they are overhead *caused by* the filter rather than *inside* it. The
profile therefore understated the addressable share.

## 3. The equivalence boundary, stated honestly

**Bitwise equality is not claimed and is not attainable** — numpy dispatches its
own SIMD transcendentals, so sin/cos/atan2 can differ from libm by an ulp.

The first version of this gate picked 1e-13 for the state and failed at 3.7e-9
in the extreme tail (dt = 86400 s in one step, e up to 0.95, where the shipped
`solve_kepler`'s FIVE FIXED Newton steps are nowhere near converged and the
function is genuinely ill-conditioned in its last digits). An absolute
tolerance was the wrong instrument.

**The gate is now self-calibrating**: perturb the input by exactly one ulp
(`np.nextafter`) and run PYTHON against ITSELF. That is how much the answer
moves for reasons unrelated to the port. Over 1,000,000 rows:

| regime | py-vs-c | py-vs-py (1 ulp) | ratio |
|---|---|---|---|
| leo_circ | 3.95e-10 | 8.88e-10 | 0.45 |
| eccentric | 5.58e-11 | 4.78e-10 | 0.12 |
| near_equatorial | 1.56e-11 | 3.16e-08 | 0.00 |
| high_e | 1.32e-09 | 5.20e-08 | 0.03 |
| diverged | 0.00 | 3.00e-15 | 0.00 |

**The port agrees with the oracle better than the oracle agrees with itself
under a 1-ulp input nudge, in every regime.** A gate of that shape cannot be
passed by loosening a constant.

Aggregate over 1e6 rows: state Y p99 **1.83e-14**, median exactly 0; STM Phi
p99 **1.16e-07**, median exactly 0, **90.9% of entries bitwise identical**.

Phi's MAX is deliberately not gated. Phi entries pass through zero (where a
relative metric is undefined), and the FD of the equatorial branch — which
switches on `hxy == 0.0` exactly — is a finite difference of a discontinuity.
What makes the tail safe is measured instead: propagating a realistic
covariance under both Phis moves the **trace by 7.0e-09 max, median exactly 0**,
about nine orders below the modelling error the 6-dof NEES band already
tolerates.

## 4. Gates

| gate | result |
|---|---|
| F1 ok-mask agrees exactly (branch, not number) | PASS, 1e6 rows |
| F2 state Y p99 <= 1e-13 | PASS (1.83e-14) |
| F2b port beats the oracle's own 1-ulp conditioning | PASS (worst ratio 0.45) |
| F3 STM Phi p99 <= 1e-6 | PASS (1.16e-07) |
| D1 propagated covariance trace moves < 1e-6 | PASS (7.04e-09) |
| P1 batch permutation permutes output BITWISE | PASS |
| P2 row-alone == same row in a batch | PASS |
| M mutation testing | PASS **5/5** |
| C1a instrument determinism (pristine vs pristine) | PASS |
| C1b `nav_filter_impl='py'` bit-identical to pre-port tree | PASS |

**Long-horizon divergence** (full 8-day W1 episode, 8 day-warps, K=0): the two
estimates differ by **7.4 mm median / 0.22 m max**, against the filter's own
88 m position error — three orders below its own accuracy.

**Published-eval reproduction** (t11_generalist_rungB, 100 eps, bearings-only):

| cell | impl | success | action md5 | causes |
|---|---|---|---|---|
| E0_j2 | py | 98/100 | f8da5e72e62b | success=98, collision=1, safety_cap=1 |
| E0_j2 | **c** | **98/100** | **f8da5e72e62b** | identical |
| TIGHT_5k1 | py | 0/100 | 529088efd02a | collision=1, safety_cap=99 |
| TIGHT_5k1 | **c** | **0/100** | **529088efd02a** | identical |

The **action stream md5 is identical** — the 1e-13 estimate differences never
flip an argmax across 100 episodes. Stronger than the rate-match the gate asked
for.

**W1-geometry NEES** (K=0, n=64): py and c agree to every printed digit —
24 h NEES 1.08 / 14.1% out-of-band / 485 m; 192 h NEES 0.85 / 14.1% / 88 m;
acquisition 100.0%, divergence 0.0493 both.

## 5. Two bugs the gates caught, both mine

**Mutation testing caught a hole in my own fuzz corpus.** The "one fewer Newton
step" mutant SURVIVED the first battery. Kepler converges quadratically, so at
e <= 0.3 the fifth step moves the answer ~1e-15 and no tolerance separates a
4-step solver from a 5-step one. Measured |E5 - E4|: 8.9e-16 at e=0.30,
3.2e-12 at 0.55, 5.3e-07 at 0.75, **1.4e-02 at 0.80** where the initial guess
switches from M to pi. My first "high_e" regime then turned out to produce
e median **0.087** — it did not exercise high eccentricity at all, because I
hand-built velocities instead of round-tripping through
`orbit_to_cartesian_3d`. Fixed; the mutant now shows Y max 4.67e-01 and 5/5 are
caught.

**C1b caught the MAJOR-17b trap, again.** I read the new flag with
`kwargs.get`, so `nav_filter_impl` stayed in the dict and reached
`Orbital.__init__`, which rejects unknown kwargs — a TypeError before
construction. `kwargs.pop` fixes it. This is the third time this exact shape has
appeared in this codebase; it is worth a lint rule, not another gate.

## 6. Merge recommendation

**Merge.** The speedup is 4.35-6.06x against a 2x bar, and it compounds: every
7/7 mixture run, every multi-seed confirm and every W1xnav rung at K=0 pays the
4.5x nav tax this removes. W1xnav at K=0 specifically goes from ~35 h for a 50M
rung to ~8 h.

**The maintenance cost is real and bounded.** ~330 lines of C that must track
`nav_math3d.py` if the propagator ever changes. Three things bound it:
1. `'py'` is the DEFAULT and is bit-identical to the pre-port tree (C1b), so
   any doubt is resolved by flipping one flag — no bisect, no rebuild.
2. The Python implementation is a PERMANENT oracle, not scaffolding to be
   retired. The fuzz + mutation battery is committed and re-runnable in minutes.
3. The port covers ONE function pair with a pure numerical contract and no
   state. It is the most portable kind of code there is.

**Conditions I would attach to a merge:**
- Any campaign that sets `nav_filter_impl='c'` runs `navc_fuzz.py --stage
  fuzz,perm,down,mut` in its stage 0, exactly as campaigns already run anchors.
- The `.so` stays gitignored and is built by `build_nav_kernel.sh`. A stale
  checked-in binary silently running instead of the source being read is a
  failure mode this project does not need.
- `-ffp-contract=off` is load-bearing (FMA rounds once instead of twice and
  numpy does not contract); it lives in the build script with that comment.
- The two-body MSC6 path is NOT ported. `nav_j2_mode=1` is required under dim3
  anyway, so the C path only ever runs where J2 is on; a two-body run silently
  gets Python, which is correct but worth knowing.
