#!/usr/bin/env python3
"""
T3: what does eccentricity actually COST, and is the 478 m/s budget enough?

c_reset samples omega ~ U(0, 2pi) INDEPENDENTLY for chaser and target whenever
e > 0. So "e_max = 0.05" is not a small perturbation: it randomizes the
eccentricity VECTOR over a disc of radius 0.05, and matching eccentricity
vectors is a v_c-scaled maneuver:

    dv_e  ~= v_c * |d e_vec| / 2      (two tangential impulses at apsides)
    |d e_vec| = |e_s*u(w_s) - e_t*u(w_t)|

At LEO v_c = 7673 m/s, so |d e_vec| = 0.05 already costs 192 m/s — 40% of the
whole budget — before any transfer or phasing.

Second question: the success test is a 30 km / 50 m/s BOX, not orbit matching.
Two ellipses with the same a but different e-vectors intersect; if the relative
velocity at an intersection is already under 50 m/s, the chaser can rendezvous
by TIMING alone and skip the e-matching burn entirely. We measure that.

Outputs: web_data/results/t3_ecc_dv_cost.csv , t3_ecc_box_relief.csv
"""
import csv
import math
import os

import numpy as np

MU, R_EARTH = 3.986004418e14, 6.371e6
DV_BUDGET = 300.0 * 9.80665 * math.log(1 / 0.85)
OUT = "/Users/pete/space_training/web_data/results"


def hohmann(a1, a2):
    return (np.abs(np.sqrt(MU / a1) * (np.sqrt(2 * a2 / (a1 + a2)) - 1.0)) +
            np.abs(np.sqrt(MU / a2) * (1.0 - np.sqrt(2 * a1 / (a1 + a2)))))


def mc_cost(alt_lo, alt_hi, e_max, n=200_000, seed=3, same_orbit=False):
    rng = np.random.default_rng(seed)
    a_s = R_EARTH + rng.uniform(alt_lo, alt_hi, n) * 1e3
    a_t = a_s.copy() if same_orbit else R_EARTH + rng.uniform(alt_lo, alt_hi, n) * 1e3
    e_s = rng.uniform(0, e_max, n)
    e_t = rng.uniform(0, e_max, n)
    w_s = rng.uniform(0, 2 * math.pi, n)
    w_t = rng.uniform(0, 2 * math.pi, n)
    de = np.hypot(e_s * np.cos(w_s) - e_t * np.cos(w_t),
                  e_s * np.sin(w_s) - e_t * np.sin(w_t))
    vc = np.sqrt(MU / (0.5 * (a_s + a_t)))
    dv_a = hohmann(a_s, a_t) if not same_orbit else np.zeros(n)
    dv_e = vc * de / 2.0
    tot = dv_a + dv_e
    return dict(
        alt_lo_km=alt_lo, alt_hi_km=alt_hi, e_max=e_max,
        same_orbit=int(same_orbit), n=n,
        v_circ_ms=float(vc.mean()),
        dv_a_mean=float(dv_a.mean()), dv_e_mean=float(dv_e.mean()),
        dv_total_mean=float(tot.mean()),
        dv_total_p50=float(np.percentile(tot, 50)),
        dv_total_p90=float(np.percentile(tot, 90)),
        dv_total_p99=float(np.percentile(tot, 99)),
        budget_ms=DV_BUDGET,
        frac_within_budget=float((tot <= DV_BUDGET).mean()),
        # leave 25% of budget for phasing + terminal trim
        frac_within_75pct_budget=float((tot <= 0.75 * DV_BUDGET).mean()),
        de_mean=float(de.mean()), de_p90=float(np.percentile(de, 90)),
    )


def min_relvel_at_crossing(a, e1, w1, e2, w2, ngrid=2048):
    """Minimum |v_rel| over points where the two coplanar ellipses cross.
    Scans true longitude; at a crossing r1 == r2 so the chaser can be
    co-located with the target there by timing alone."""
    L = np.linspace(0, 2 * math.pi, ngrid, endpoint=False)
    p1, p2 = a * (1 - e1**2), a * (1 - e2**2)
    r1 = p1 / (1 + e1 * np.cos(L - w1))
    r2 = p2 / (1 + e2 * np.cos(L - w2))
    d = r1 - r2
    sign = np.sign(d)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(idx) == 0:
        return np.inf, np.inf
    best_v, best_r = np.inf, np.inf
    for i in idx:
        Lc = L[i]
        nu1, nu2 = Lc - w1, Lc - w2
        h1, h2 = math.sqrt(MU * p1), math.sqrt(MU * p2)
        r = 0.5 * (r1[i] + r2[i])
        vr1, vt1 = MU / h1 * e1 * math.sin(nu1), h1 / r
        vr2, vt2 = MU / h2 * e2 * math.sin(nu2), h2 / r
        # rotate both into the common local frame at true longitude Lc
        v = math.hypot(vr1 - vr2, vt1 - vt2)
        if v < best_v:
            best_v, best_r = v, r
    return best_v, best_r


def mc_box_relief(e_max, n=4000, seed=5, a=6.771e6):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        e1, e2 = rng.uniform(0, e_max), rng.uniform(0, e_max)
        w1, w2 = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        v, _ = min_relvel_at_crossing(a, e1, w1, e2, w2)
        vals.append(v)
    v = np.array(vals)
    fin = v[np.isfinite(v)]
    return dict(
        e_max=e_max, n=n,
        frac_crossing_exists=float(np.isfinite(v).mean()),
        relvel_p10=float(np.percentile(fin, 10)),
        relvel_p50=float(np.percentile(fin, 50)),
        relvel_p90=float(np.percentile(fin, 90)),
        frac_relvel_under_50=float((fin < 50).mean()),
        frac_relvel_under_1=float((fin < 1).mean()),
        dv_e_full_match_p50=float(np.percentile(
            np.sqrt(MU / a) * np.abs(np.random.default_rng(seed).uniform(0, e_max, n)) / 2, 50)),
    )


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def main():
    print(f"budget = {DV_BUDGET:.1f} m/s\n")
    rows = []
    for lo, hi in ((300, 800), (300, 2000), (300, 8000), (300, 20200)):
        for em in (0.0, 0.025, 0.05, 0.084, 0.15, 0.3, 0.5):
            # geometric validity: perigee must clear R+200 km at the top of band
            if em > 1 - (R_EARTH + 200e3) / (R_EARTH + hi * 1e3):
                continue
            rows.append(mc_cost(lo, hi, em))
            rows.append(mc_cost(lo, hi, em, same_orbit=True))
    write(f"{OUT}/t3_ecc_dv_cost.csv", rows)

    print("band | e_max | same_orbit | dv_a | dv_e | dv_tot_p50 | dv_tot_p90 | %<budget | %<75%budget")
    for r in rows:
        print(f"{r['alt_lo_km']:.0f}-{r['alt_hi_km']:.0f} | {r['e_max']:.3f} | "
              f"{r['same_orbit']} | {r['dv_a_mean']:6.0f} | {r['dv_e_mean']:6.0f} | "
              f"{r['dv_total_p50']:6.0f} | {r['dv_total_p90']:6.0f} | "
              f"{100*r['frac_within_budget']:5.1f}% | {100*r['frac_within_75pct_budget']:5.1f}%")

    print()
    br = [mc_box_relief(em) for em in (0.01, 0.025, 0.05, 0.084)]
    write(f"{OUT}/t3_ecc_box_relief.csv", br)
    print("e_max | crossing exists | |v_rel| p10/p50/p90 at crossing | %<50 m/s")
    for r in br:
        print(f"{r['e_max']:.3f} | {100*r['frac_crossing_exists']:5.1f}% | "
              f"{r['relvel_p10']:7.1f}/{r['relvel_p50']:7.1f}/{r['relvel_p90']:7.1f} | "
              f"{100*r['frac_relvel_under_50']:5.1f}%")


if __name__ == "__main__":
    main()
