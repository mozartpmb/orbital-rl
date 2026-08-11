"""RED-TEAM A3 — is there a CHEAP acquisition that still works?

Probe A1/A2 showed the shipped BO-BLS acquisition costs 0.19-1.86 s scalar and
2.5-28 s batched-lower-bound per 1024 envs per episode — 5-28x the whole
training step. Cost is ~linear in (grid nodes x window) and the grid is set by
`grid_ratio=1.15` over a 4-decade range prior.

This sweeps the three knobs that set the cost and reports, per configuration:
FLOP proxy, wall time, gate-pass fraction, and epoch position error. The output
is the concrete spec for a training-time filter — or the evidence that no cheap
configuration works.
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

OUT = "/Users/pete/space_training/web_data/results/ext_rtnav_acq_cheap.csv"

CNT = {"prop": 0, "stm": 0}
_prop, _stm = bo.propagate_cartesian, bo.stm_numerical


def pc(x, dt):
    CNT["prop"] += 1
    return _prop(x, dt)


def sc_(x, dt):
    CNT["stm"] += 1
    return _stm(x, dt)


CONFIGS = [
    dict(tag="shipped",      grid_ratio=1.15, n_bins=8, w0=45, iters=12,
         v_tang=(0.8, 1.0, 1.2), v_rad=(-0.3, 0.0, 0.3)),
    dict(tag="ratio1.4",     grid_ratio=1.40, n_bins=8, w0=45, iters=12,
         v_tang=(0.8, 1.0, 1.2), v_rad=(-0.3, 0.0, 0.3)),
    dict(tag="ratio2.0",     grid_ratio=2.00, n_bins=8, w0=45, iters=12,
         v_tang=(0.8, 1.0, 1.2), v_rad=(-0.3, 0.0, 0.3)),
    dict(tag="ratio2+w20",   grid_ratio=2.00, n_bins=8, w0=20, iters=12,
         v_tang=(0.8, 1.0, 1.2), v_rad=(-0.3, 0.0, 0.3)),
    dict(tag="ratio2+vlean", grid_ratio=2.00, n_bins=6, w0=20, iters=8,
         v_tang=(0.9, 1.1), v_rad=(-0.3, 0.3)),
    dict(tag="minimal",      grid_ratio=3.00, n_bins=4, w0=20, iters=6,
         v_tang=(1.0,), v_rad=(0.0,)),
]


def main():
    seeds = [0, 1, 2]
    scen = bo.make_scenarios()
    rows = []
    for cfg in CONFIGS:
        agg = dict(wall=0.0, prop=0, stm=0, pass_n=0, n=0, err=[])
        per_scen = {}
        for sc in scen:
            T, S, G = bo.roll_truth(sc)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                beta = [bo.wrap_pi(bo.bearing_of(S[k], G[k]) +
                                   rng.normal(0.0, bo.SIGMA_BETA))
                        for k in range(len(T))]
                iv = bo.range_prior_intervals(S[0], beta[0], sc['r_min'],
                                              sc['r_max'])
                CNT["prop"] = CNT["stm"] = 0
                bo.propagate_cartesian, bo.stm_numerical = pc, sc_
                t0 = time.perf_counter()
                try:
                    acq = bo.bls_acquire_adaptive(
                        T, S, beta, iv, bo.SIGMA_BETA, sc['sigma_v_ecc'],
                        w0=cfg['w0'], grid_ratio=cfg['grid_ratio'],
                        n_bins=cfg['n_bins'], iters=cfg['iters'],
                        v_tang=cfg['v_tang'], v_rad=cfg['v_rad'])
                except Exception:
                    acq = None
                wall = time.perf_counter() - t0
                bo.propagate_cartesian, bo.stm_numerical = _prop, _stm
                ok = bool(acq and acq[4])
                err = (math.hypot(acq[0][0] - G[0][0], acq[0][1] - G[0][1])
                       if acq else float('inf'))
                agg['wall'] += wall
                agg['prop'] += CNT['prop']
                agg['stm'] += CNT['stm']
                agg['pass_n'] += int(ok)
                agg['n'] += 1
                agg['err'].append(err)
                d = per_scen.setdefault(sc['name'], [])
                d.append((ok, err, wall))
        n = agg['n']
        propeq = agg['prop'] + agg['stm'] * 10.7
        errs = np.array(agg['err'])
        rows.append(dict(tag=cfg['tag'], grid_ratio=cfg['grid_ratio'],
                         n_bins=cfg['n_bins'], w0=cfg['w0'],
                         iters=cfg['iters'],
                         n_vel=len(cfg['v_tang']) * len(cfg['v_rad']),
                         wall_mean_s=agg['wall'] / n,
                         prop_equiv_mean=propeq / n,
                         gate_pass=agg['pass_n'] / n,
                         err_p50_km=float(np.median(errs)) / 1e3,
                         err_p90_km=float(np.percentile(errs, 90)) / 1e3))
        print(f"{cfg['tag']:<13} wall {agg['wall']/n:7.3f} s  "
              f"prop-equiv {propeq/n:>10,.0f}  gate {agg['pass_n']}/{n}  "
              f"err p50 {np.median(errs)/1e3:8.2f} km  "
              f"p90 {np.percentile(errs,90)/1e3:9.2f} km")
        for name, d in per_scen.items():
            npass = sum(1 for o, _, _ in d if o)
            e = np.median([x[1] for x in d]) / 1e3
            print(f"     {name:<24} gate {npass}/{len(d)}  err p50 {e:9.2f} km")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    base = rows[0]['prop_equiv_mean']
    print("\n--- batched cost implication (89.3 ns/state-prop measured) ---")
    for r in rows:
        ms_1024 = r['prop_equiv_mean'] * 1024 * 89.3e-9 * 1e3
        # amortized over a MEASURED 20.9-decision episode; env+ekf = 15.27 ms/dec
        add = ms_1024 / 20.9
        sps = 2048 / ((15.27 + add) / 1000.0)
        print(f"  {r['tag']:<13} {r['prop_equiv_mean']/base:5.2f}x shipped  "
              f"→ {ms_1024:8.0f} ms/1024-env acquisition, +{add:7.1f} ms/dec, "
              f"SPS {sps:>7,.0f}, 50M in {50e6/sps/60:6.1f} min "
              f"(gate {100*r['gate_pass']:.0f}%)")


if __name__ == "__main__":
    main()
