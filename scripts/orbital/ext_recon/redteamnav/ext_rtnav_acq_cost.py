"""RED-TEAM A — measured cost of the NAV-G BO-BLS acquisition, and the
training-time budget it implies.

NAV-H benchmarked the *range+bearing* EKF (1.353 ms/tick batched at B=1024) and
concluded "SPS is not the binding constraint". NAV-G then recommended a
*bearings-only* filter whose recursive stage is comparable but whose ACQUISITION
is a dense multi-start batch least squares over the analytic range prior. This
probe measures the acquisition, not the recursive stage.

Measured per call:
  wall time, propagate_cartesian calls, stm_numerical calls, final window w,
  gate outcome, and the implied per-1024-env cost of doing this once per episode.

Read-only: imports ext_bo_filter and monkeypatches counters onto ITS module
namespace. Nothing in the repo is modified.
"""
import csv
import math
import os
import sys
import time

import numpy as np

RECON = "/Users/pete/space_training/scripts/orbital/ext_recon"
NAV = "/Users/pete/space_training/scripts/orbital/nav"
sys.path.insert(0, RECON)
sys.path.insert(0, NAV)

import ext_bo_filter as bo   # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_rtnav_acq_cost.csv"

CNT = {"prop": 0, "stm": 0}
_prop = bo.propagate_cartesian
_stm = bo.stm_numerical


def prop_c(x, dt):
    CNT["prop"] += 1
    return _prop(x, dt)


def stm_c(x, dt):
    CNT["stm"] += 1
    return _stm(x, dt)


def main():
    seeds = [int(s) for s in (sys.argv[1:] or ["0", "1", "2"])]
    scen = bo.make_scenarios()

    # raw propagate_cartesian cost, for the FLOP accounting
    x0 = np.array([7.0e6, 0.0, 0.0, 7546.0])
    t0 = time.perf_counter()
    for _ in range(20000):
        _prop(x0, 60.0)
    t_prop = (time.perf_counter() - t0) / 20000
    t0 = time.perf_counter()
    for _ in range(3000):
        _stm(x0, 60.0)
    t_stm = (time.perf_counter() - t0) / 3000
    print(f"scalar propagate_cartesian: {t_prop*1e6:.2f} us   "
          f"stm_numerical: {t_stm*1e6:.2f} us  ({t_stm/t_prop:.1f} props)")

    rows = []
    for sc in scen:
        T, S, G = bo.roll_truth(sc)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            n = len(T)
            beta = [bo.wrap_pi(bo.bearing_of(S[k], G[k]) +
                               rng.normal(0.0, bo.SIGMA_BETA))
                    for k in range(n)]
            iv = bo.range_prior_intervals(S[0], beta[0], sc['r_min'], sc['r_max'])
            lo, hi = bo.prior_span(iv)

            # count the grid the acquisition will build (window-independent)
            nodes = 0
            for a, b in iv:
                if b > a:
                    k = max(3, min(160, int(math.ceil(math.log(b / a) /
                                                      math.log(1.15)))))
                    nodes += k * 9      # 3x3 velocity lattice
            CNT["prop"] = CNT["stm"] = 0
            bo.propagate_cartesian = prop_c
            bo.stm_numerical = stm_c
            t0 = time.perf_counter()
            acq = bo.bls_acquire_adaptive(T, S, beta, iv, bo.SIGMA_BETA,
                                          sc['sigma_v_ecc'])
            wall = time.perf_counter() - t0
            bo.propagate_cartesian = _prop
            bo.stm_numerical = _stm
            props, stms = CNT["prop"], CNT["stm"]

            if acq is None:
                w, ok, err = -1, False, float('nan')
            else:
                x, P, c, w, ok = acq
                err = float(np.hypot(x[0] - G[0][0], x[1] - G[0][1]))
            lat_min = (w * sc['dt'] / 60.0) if w > 0 else float('nan')
            rows.append(dict(scenario=sc['name'], seed=seed,
                             n_obs=n, prior_lo_km=lo / 1e3, prior_hi_km=hi / 1e3,
                             n_intervals=len(iv), grid_nodes=nodes,
                             wall_s=wall, n_prop=props, n_stm=stms,
                             prop_equiv=props + stms * (t_stm / t_prop),
                             final_window=w, acq_min=lat_min,
                             gate_pass=int(bool(ok)), pos_err_m=err))
            print(f"{sc['name']:<24} seed{seed}  wall {wall:7.3f} s  "
                  f"prop {props:>9,}  stm {stms:>7,}  w={w:<4} "
                  f"({lat_min:.0f} min)  gate={'PASS' if ok else 'FAIL'}  "
                  f"err {err/1e3:8.2f} km  nodes {nodes}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    # ── projection ──────────────────────────────────────────────────────────
    walls = sorted(r['wall_s'] for r in rows)
    med = walls[len(walls) // 2]
    mean = sum(walls) / len(walls)
    print(f"\nacquisition wall time: median {med:.3f} s  mean {mean:.3f} s  "
          f"min {walls[0]:.3f}  max {walls[-1]:.3f}")

    print("\n--- training-time projection (scalar-Python acquisition) ---")
    # NAV-H measured constants
    ENV_MS_DEC = 11.38          # ms/decision, B=1024, mean tau=32.5
    for dec_per_ep in (50, 92):
        # 1024 envs each finish an episode every dec_per_ep decisions
        acq_ms_per_dec = med * 1000.0 * 1024 / dec_per_ep
        print(f"  {dec_per_ep} decisions/episode: acquisition adds "
              f"{acq_ms_per_dec:,.0f} ms/decision vs env {ENV_MS_DEC} ms "
              f"→ {acq_ms_per_dec/ENV_MS_DEC:,.0f}x slowdown")
        sps = 1024 * 2 / ((ENV_MS_DEC + acq_ms_per_dec) / 1000.0)
        print(f"      SPS {sps:,.0f} (vs 61K nav60 projection); "
              f"50M steps = {50e6/sps/3600:,.1f} hours "
              f"({50e6/sps/86400:,.1f} days)")
    n_ep = 50e6 / 70.0
    print(f"  total acquisitions for a 50M-decision rung @70 dec/ep: "
          f"{n_ep:,.0f}; serial cost {n_ep*med/3600:,.0f} core-hours")


if __name__ == "__main__":
    main()
