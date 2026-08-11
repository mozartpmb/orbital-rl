"""RED-TEAM A2 — can the BO-BLS acquisition be BATCHED across 1024 envs?

Two measurements:
  (1) stage split of the scalar acquisition (grid scoring vs GN refine), so we
      know what fraction has uniform control flow and is therefore batchable;
  (2) throughput of the batched primitives (`prop`, `stm_fd` from
      ext_nav_ekf_opt) at acquisition-sized batches, giving an OPTIMISTIC
      lower-bound cost for a perfectly-vectorized acquisition (no Levenberg
      accept/reject divergence, no per-node admissibility break, no per-env
      adaptive window — i.e. strictly better than anything implementable).

The lower bound is the honest number to gate the design on: if the LOWER BOUND
is already unaffordable, no implementation effort rescues it.
"""
import math
import os
import sys
import time

import numpy as np

RECON = "/Users/pete/space_training/scripts/orbital/ext_recon"
NAV = "/Users/pete/space_training/scripts/orbital/nav"
sys.path.insert(0, RECON)
sys.path.insert(0, NAV)

import ext_bo_filter as bo            # noqa: E402
from ext_nav_ekf_opt import prop, stm_fd   # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_rtnav_acq_batched.csv"


def stage_split():
    """Instrument bls_acquire's internals by wrapping the two hot functions."""
    acc = {"cost": 0.0, "cost_n": 0, "pass": 0.0, "pass_n": 0}
    _bc, _bp = bo._bearing_cost, bo._bls_pass

    def bc(*a, **k):
        t = time.perf_counter()
        r = _bc(*a, **k)
        acc["cost"] += time.perf_counter() - t
        acc["cost_n"] += 1
        return r

    def bp(*a, **k):
        t = time.perf_counter()
        r = _bp(*a, **k)
        acc["pass"] += time.perf_counter() - t
        acc["pass_n"] += 1
        return r

    bo._bearing_cost, bo._bls_pass = bc, bp
    out = []
    for sc in bo.make_scenarios():
        T, S, G = bo.roll_truth(sc)
        rng = np.random.default_rng(0)
        beta = [bo.wrap_pi(bo.bearing_of(S[k], G[k]) +
                           rng.normal(0.0, bo.SIGMA_BETA)) for k in range(len(T))]
        iv = bo.range_prior_intervals(S[0], beta[0], sc['r_min'], sc['r_max'])
        for k in acc:
            acc[k] = 0.0
        t0 = time.perf_counter()
        bo.bls_acquire_adaptive(T, S, beta, iv, bo.SIGMA_BETA, sc['sigma_v_ecc'])
        tot = time.perf_counter() - t0
        out.append((sc['name'], tot, acc['cost'], acc['cost_n'],
                    acc['pass'], acc['pass_n']))
        print(f"{sc['name']:<24} total {tot:6.3f}s | grid-score "
              f"{acc['cost']:6.3f}s ({100*acc['cost']/tot:4.1f}%, "
              f"{acc['cost_n']:>5} calls) | GN-refine {acc['pass']:6.3f}s "
              f"({100*acc['pass']/tot:4.1f}%, {acc['pass_n']:>3} calls)")
    bo._bearing_cost, bo._bls_pass = _bc, _bp
    return out


def throughput():
    print("\n--- batched primitive throughput (OMP=1) ---")
    res = {}
    for B in (1024, 65536, 262144, 627712):
        X = np.tile(np.array([7.0e6, 0.0, 0.0, 7546.0]), (B, 1))
        X[:, 0] *= 1.0 + 0.001 * np.random.default_rng(0).standard_normal(B)
        for name, fn, iters in (("prop12", lambda x: prop(x, 60.0, 12), 12),
                                ("prop4", lambda x: prop(x, 60.0, 4), 4)):
            fn(X)
            t0 = time.perf_counter()
            R = 5 if B > 200000 else 20
            for _ in range(R):
                fn(X)
            dt = (time.perf_counter() - t0) / R
            res[(name, B)] = dt / B
            print(f"  {name:<7} B={B:>7,}: {dt*1e3:8.3f} ms  "
                  f"{dt/B*1e9:7.1f} ns/state")
        if B <= 65536:
            stm_fd(X, 60.0, 4)
            t0 = time.perf_counter()
            for _ in range(5):
                stm_fd(X, 60.0, 4)
            dt = (time.perf_counter() - t0) / 5
            res[("stm_fd4", B)] = dt / B
            print(f"  stm_fd4 B={B:>7,}: {dt*1e3:8.3f} ms  "
                  f"{dt/B*1e9:7.1f} ns/state")
    return res


def project(res):
    ns_prop = res[("prop4", 262144)] * 1e9        # ns per state-propagation
    ns_stm = res[("stm_fd4", 65536)] * 1e9        # ns per state-STM
    print(f"\nunit costs used: prop4 {ns_prop:.1f} ns/state, "
          f"stm_fd4 {ns_stm:.1f} ns/state")

    NE = 1024
    print("\n--- OPTIMISTIC batched acquisition lower bound, B=1024 envs ---")
    print("  (perfect vectorization: no LM divergence, no per-node break,")
    print("   all envs acquire on the SAME window - none of which is true)")
    rows = []
    for nodes, w, starts, coarse_it, fine_n, fine_it, label in (
            (612, 45, 11, 3, 3, 12, "G5/G6 best case (w=45, 612 nodes)"),
            (810, 72, 11, 3, 3, 12, "G4 wide (w=72, 810 nodes)"),
            (612, 45 + 72 + 115, 11, 3, 3, 12, "G1 (3 window growths, cum w)"),
    ):
        grid = NE * nodes * w * ns_prop
        gn_coarse = NE * starts * coarse_it * w * (ns_prop + ns_stm)
        gn_fine = NE * fine_n * fine_it * w * (ns_prop + ns_stm)
        tot_ms = (grid + gn_coarse + gn_fine) / 1e6
        rows.append((label, grid / 1e6, (gn_coarse + gn_fine) / 1e6, tot_ms))
        print(f"  {label:<38} grid {grid/1e6:8.1f} ms  GN {(gn_coarse+gn_fine)/1e6:7.1f} ms"
              f"  TOTAL {tot_ms:8.1f} ms")

    ENV_MS = 11.38     # NAV-H measured env ms/decision at mean tau=32.5, B=1024
    EKF_MS = 3.89      # NAV-H nav60 filter ms/decision
    print("\n--- amortized over an episode (all 1024 envs acquire once) ---")
    for label, _g, _n, tot_ms in rows:
        for dec_ep in (70, 92):
            add = tot_ms / dec_ep
            base = ENV_MS + EKF_MS
            sps = NE * 2 / ((base + add) / 1000.0)
            print(f"  {label:<38} {dec_ep} dec/ep: +{add:7.2f} ms/dec "
                  f"({(base+add)/base:5.2f}x) → SPS {sps:>7,.0f}, "
                  f"50M in {50e6/sps/60:6.1f} min")
    print(f"\n  reference: nav60 range+bearing projection = 61,000 SPS, "
          f"50M in 13.6 min")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("case,grid_ms,gn_ms,total_ms,ns_prop,ns_stm\n")
        for label, g, n, t in rows:
            f.write(f'"{label}",{g:.3f},{n:.3f},{t:.3f},{ns_prop:.2f},{ns_stm:.2f}\n')


if __name__ == "__main__":
    stage_split()
    project(throughput())
