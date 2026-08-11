#!/usr/bin/env python3
"""
T3: joint feasibility of the HEADLINE eval distribution under corrected
dynamics — dv budget AND episode horizon at the same time.

Per sample:
  dv_transfer = Hohmann(a_s, a_t)
  dv_ecc      = v_c * |de_vec| / 2          (e-vector match; omega is uniform)
  phasing     = free if the natural drift closes the gap inside the horizon,
                else scan a phasing orbit and charge the cheapest surcharge
  time        = loiter/drift + transfer legs

Feasible iff sum(dv) <= 478.1 m/s AND total sim steps <= horizon.
Output: web_data/results/t3_joint_feasibility.csv
"""
import csv
import math
import os

import numpy as np

MU, R_EARTH, DT = 3.986004418e14, 6.371e6, 60.0
DV_BUDGET = 300.0 * 9.80665 * math.log(1 / 0.85)
OUT = "/Users/pete/space_training/web_data/results"


def hoh(a1, a2):
    at = 0.5 * (a1 + a2)
    dv = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0)) + \
         abs(math.sqrt(MU / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2))))
    return dv, math.pi * math.sqrt(at**3 / MU)


def n_of(a):
    return math.sqrt(MU / a**3)


def sample(n, alt_lo, alt_hi, e_max, horizon, seed, da_max_km=None,
           de_max=None, fuel_frac=0.15):
    """de_max: if set, the target's eccentricity VECTOR is drawn within de_max
    of the chaser's, instead of independently. This keeps 'both orbits are
    eccentric' while bounding the (expensive) e-vector matching maneuver.
    fuel_frac: propellant mass fraction -> dv budget via Tsiolkovsky."""
    budget = 300.0 * 9.80665 * math.log(1.0 / (1.0 - fuel_frac))
    rng = np.random.default_rng(seed)
    ok_dv = ok_t = ok_both = 0
    dvs, steps = [], []
    for _ in range(n):
        a_s = R_EARTH + rng.uniform(alt_lo, alt_hi) * 1e3
        if da_max_km is None:
            a_t = R_EARTH + rng.uniform(alt_lo, alt_hi) * 1e3
        else:
            a_t = a_s + rng.uniform(-da_max_km, da_max_km) * 1e3
            a_t = min(max(a_t, R_EARTH + alt_lo * 1e3), R_EARTH + alt_hi * 1e3)
        if abs(a_t - a_s) < 50e3:
            a_t = a_s + math.copysign(50e3, a_t - a_s or 1.0)
        e_s = rng.uniform(0, e_max)
        w_s = rng.uniform(0, 2 * math.pi)
        d0 = rng.uniform(-math.pi, math.pi)
        vc = math.sqrt(MU / (0.5 * (a_s + a_t)))
        if de_max is None:
            e_t = rng.uniform(0, e_max)
            w_t = rng.uniform(0, 2 * math.pi)
            de = math.hypot(e_s * math.cos(w_s) - e_t * math.cos(w_t),
                            e_s * math.sin(w_s) - e_t * math.sin(w_t))
        else:
            de = de_max * math.sqrt(rng.uniform(0, 1))   # uniform on the disc
        dv_e = vc * de / 2.0
        dv_a, th = hoh(a_s, a_t)

        # free-loiter phasing
        w = n_of(a_s) - n_of(a_t)
        num = (d0 + n_of(a_t) * th - math.pi) % (2 * math.pi)
        td = num / w if w >= 0 else (num - 2 * math.pi) / w
        tot_s = (td + th) / DT
        surcharge = 0.0
        if tot_s > horizon:
            best = math.inf
            best_t = math.inf
            for off in np.arange(-600, 620, 25.0) * 1e3:
                ap = a_t + off
                if ap < R_EARTH + 250e3 or abs(ap - a_t) < 25e3:
                    continue
                dvA, thA = hoh(a_s, ap)
                dvB, thB = hoh(ap, a_t)
                wp = n_of(ap) - n_of(a_t)
                if abs(wp) < 1e-13:
                    continue
                numB = (d0 + n_of(a_t) * (thA + thB) - 2 * math.pi) % (2 * math.pi)
                tdr = numB / wp if wp >= 0 else (numB - 2 * math.pi) / wp
                t2 = (thA + tdr + thB) / DT
                if t2 <= horizon and (dvA + dvB - dv_a) < best:
                    best, best_t = dvA + dvB - dv_a, t2
            if math.isfinite(best):
                surcharge, tot_s = best, best_t
            else:
                tot_s = math.inf

        dv_tot = dv_a + dv_e + surcharge
        dvs.append(dv_tot)
        steps.append(tot_s)
        a_ok = dv_tot <= budget
        t_ok = tot_s <= horizon
        ok_dv += a_ok
        ok_t += t_ok
        ok_both += (a_ok and t_ok)

    dvs = np.array(dvs)
    st = np.array(steps)
    fin = st[np.isfinite(st)]
    return dict(alt_lo_km=alt_lo, alt_hi_km=alt_hi, e_max=e_max,
                de_max=(de_max if de_max else -1), fuel_frac=fuel_frac,
                budget_ms=budget,
                da_max_km=(da_max_km if da_max_km else -1), horizon=horizon, n=n,
                frac_dv_ok=ok_dv / n, frac_time_ok=ok_t / n,
                frac_feasible=ok_both / n,
                dv_p50=float(np.percentile(dvs, 50)),
                dv_p90=float(np.percentile(dvs, 90)),
                steps_p50=float(np.percentile(fin, 50)) if len(fin) else math.nan,
                steps_p90=float(np.percentile(fin, 90)) if len(fin) else math.nan)


def main():
    cfgs = [
        # headline, as shipped
        dict(alt_lo=300, alt_hi=800, e_max=0.05, horizon=2000, da_max_km=None),
        dict(alt_lo=300, alt_hi=800, e_max=0.05, horizon=3000, da_max_km=None),
        dict(alt_lo=300, alt_hi=800, e_max=0.00, horizon=2000, da_max_km=None),
        dict(alt_lo=300, alt_hi=800, e_max=0.084, horizon=3000, da_max_km=None),
        # wide bands WITHOUT a da bound -> dv-starved
        dict(alt_lo=300, alt_hi=2000, e_max=0.15, horizon=3000, da_max_km=None),
        dict(alt_lo=300, alt_hi=8000, e_max=0.30, horizon=6000, da_max_km=None),
        # wide bands WITH a bounded |da| -> the proposed fix
        dict(alt_lo=300, alt_hi=2000, e_max=0.15, horizon=3000, da_max_km=400),
        dict(alt_lo=300, alt_hi=8000, e_max=0.30, horizon=6000, da_max_km=600),
        dict(alt_lo=300, alt_hi=20200, e_max=0.50, horizon=12000, da_max_km=1000),
        # proposed fix: bound the e-VECTOR mismatch, not e itself
        dict(alt_lo=300, alt_hi=2000, e_max=0.15, horizon=3000, da_max_km=400, de_max=0.06),
        dict(alt_lo=300, alt_hi=8000, e_max=0.30, horizon=6000, da_max_km=600, de_max=0.08),
        dict(alt_lo=300, alt_hi=20200, e_max=0.50, horizon=12000, da_max_km=1000, de_max=0.10),
        # or: keep independent e-vectors but raise the propellant fraction
        dict(alt_lo=300, alt_hi=2000, e_max=0.15, horizon=3000, da_max_km=400, fuel_frac=0.30),
        dict(alt_lo=300, alt_hi=8000, e_max=0.30, horizon=6000, da_max_km=600, fuel_frac=0.40),
    ]
    rows = [sample(n=3000, seed=11, **c) for c in cfgs]
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/t3_joint_feasibility.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("band | e_max | de_max | ff | da_max | horizon | %dv_ok | %FEASIBLE | dv_p50 | dv_p90 | steps_p50 | steps_p90")
    for r in rows:
        print(f"{r['alt_lo_km']:.0f}-{r['alt_hi_km']:.0f} | {r['e_max']:.3f} | "
              f"{r['de_max']:6.2f} | {r['fuel_frac']:.2f} | "
              f"{r['da_max_km']:>6.0f} | {r['horizon']:5d} | "
              f"{100*r['frac_dv_ok']:5.1f}% | {100*r['frac_time_ok']:5.1f}% | "
              f"{100*r['frac_feasible']:5.1f}% | {r['dv_p50']:6.0f} | {r['dv_p90']:6.0f} | "
              f"{r['steps_p50']:7.0f} | {r['steps_p90']:8.0f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
