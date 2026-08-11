#!/usr/bin/env python3
"""
ext-3d B1/B3/B4 — joint (Delta-v + horizon) feasibility of 3D rendezvous rungs.

Direct extension of scripts/orbital/t3/joint_feasibility.py: same in-plane cost
model (Hohmann + e-vector match + drift-phasing surcharge with a phasing-orbit
rescue scan + horizon check), plus a plane-change leg.

Plane model
-----------
The *only* geometric quantity that sets plane-change cost is the angle between
the two orbit planes, theta = |delta_i_vec| where

    delta_i_vec = ( di , dRAAN * sin(i) )        (small-angle; exact form used too)

so we sample delta_i_vec uniformly on a disc of radius di_max — one knob,
i-invariant, exactly analogous to de_max for the e-vector.

The relative node sits at argument of latitude u_node = atan2(di_y, di_x)
(and u_node + 180 deg).  The burn happens at a node; its local speed is
whatever the orbit is doing there, so for eccentric orbits the cost swings by
sqrt((1+e)/(1-e)) depending on where the node falls relative to periapsis.
We take the cheaper of the two nodes, on the cheaper of {chaser orbit before
the in-plane work, target orbit after it}.

Two accounting modes are reported:
  seq   : dv_total = dv_inplane + 2 v_node sin(theta/2)          (conservative)
  comb  : the single largest in-plane impulse is co-located with a node and
          combines vectorially:  dv = (dv_in - dv_big) + hypot(dv_big, dv_pl)
          (exact law: hypot(|v2-v1|, 2 sqrt(v1 v2) sin(theta/2)))

Writes web_data/results/ext_3d_dv_feasibility.csv (di_max sweep per band) and
       web_data/results/ext_3d_dv_dimax.csv      (solved di_max at 99% / 95%).
"""
import csv
import math
import os

import numpy as np

MU, R_EARTH, DT = 3.986004418e14, 6.371e6, 60.0
ISP, G0 = 300.0, 9.80665
OUT = "/Users/pete/space_training/web_data/results"


def hoh(a1, a2):
    at = 0.5 * (a1 + a2)
    dv1 = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0))
    dv2 = abs(math.sqrt(MU / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2))))
    return dv1 + dv2, math.pi * math.sqrt(at ** 3 / MU), max(dv1, dv2)


def n_of(a):
    return math.sqrt(MU / a ** 3)


def v_at(a, e, theta):
    r = a * (1 - e * e) / (1 + e * math.cos(theta))
    return math.sqrt(MU * (2.0 / r - 1.0 / a))


def sample(n, alt_lo, alt_hi, e_max, horizon, seed, di_max_deg,
           da_max_km=None, de_max=None, fuel_frac=0.15, same_orbit=False,
           i_lo_deg=0.0, i_hi_deg=90.0):
    budget = ISP * G0 * math.log(1.0 / (1.0 - fuel_frac))
    rng = np.random.default_rng(seed)
    di_max = math.radians(di_max_deg)
    ok = dict(dv_seq=0, dv_comb=0, t=0, both_seq=0, both_comb=0)
    dvs_seq, dvs_comb, dvs_pl, steps, thetas = [], [], [], [], []

    for _ in range(n):
        a_s = R_EARTH + rng.uniform(alt_lo, alt_hi) * 1e3
        if same_orbit:
            a_t = a_s
        elif da_max_km is None:
            a_t = R_EARTH + rng.uniform(alt_lo, alt_hi) * 1e3
        else:
            a_t = a_s + rng.uniform(-da_max_km, da_max_km) * 1e3
            a_t = min(max(a_t, R_EARTH + alt_lo * 1e3), R_EARTH + alt_hi * 1e3)
        if not same_orbit and abs(a_t - a_s) < 50e3:
            a_t = a_s + math.copysign(50e3, (a_t - a_s) or 1.0)

        e_s = 0.0 if same_orbit else rng.uniform(0, e_max)
        w_s = 0.0 if same_orbit else rng.uniform(0, 2 * math.pi)
        d0 = rng.uniform(-math.pi, math.pi)
        vc = math.sqrt(MU / (0.5 * (a_s + a_t)))
        if same_orbit:
            e_t, w_t, de = 0.0, 0.0, 0.0
        elif de_max is None:
            e_t = rng.uniform(0, e_max)
            w_t = rng.uniform(0, 2 * math.pi)
            de = math.hypot(e_s * math.cos(w_s) - e_t * math.cos(w_t),
                            e_s * math.sin(w_s) - e_t * math.sin(w_t))
        else:
            de = de_max * math.sqrt(rng.uniform(0, 1))
            ph = rng.uniform(0, 2 * math.pi)
            e_t = rng.uniform(0, e_max)
            w_t = rng.uniform(0, 2 * math.pi)
            e_sx = e_t * math.cos(w_t) + de * math.cos(ph)
            e_sy = e_t * math.sin(w_t) + de * math.sin(ph)
            e_s = math.hypot(e_sx, e_sy)
            w_s = math.atan2(e_sy, e_sx)
        dv_e = vc * de / 2.0
        dv_a, th, dv_big_a = hoh(a_s, a_t) if not same_orbit else (0.0, 0.0, 0.0)

        # ── in-plane: free-loiter drift phasing, else phasing-orbit rescue ──
        w = n_of(a_s) - n_of(a_t)
        surcharge = 0.0
        if abs(w) < 1e-14:
            tot_s = math.inf
        else:
            num = (d0 + n_of(a_t) * th - math.pi) % (2 * math.pi)
            td = num / w if w >= 0 else (num - 2 * math.pi) / w
            tot_s = (td + th) / DT
        if tot_s > horizon:
            best, best_t = math.inf, math.inf
            span = max(600.0, (da_max_km or 600.0))
            for off in np.arange(-span, span + 20.0, span / 24.0) * 1e3:
                ap = a_t + off
                if ap < R_EARTH + 250e3 or abs(ap - a_t) < 25e3:
                    continue
                dvA, thA, _ = hoh(a_s, ap)
                dvB, thB, _ = hoh(ap, a_t)
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
        dv_in = dv_a + dv_e + surcharge
        dv_big = max(dv_big_a, dv_e)          # largest single in-plane impulse

        # ── plane leg ──────────────────────────────────────────────────────
        if di_max > 0.0:
            r_di = di_max * math.sqrt(rng.uniform(0, 1))     # uniform on disc
            psi = rng.uniform(0, 2 * math.pi)
            theta = r_di
            u_node = psi
            cand = []
            for du in (0.0, math.pi):
                cand.append(v_at(a_s, e_s, u_node + du - w_s))
                cand.append(v_at(a_t, e_t, u_node + du - w_t))
            v_node = min(cand)
            dv_pl = 2.0 * v_node * math.sin(0.5 * theta)
            # Node wait: nodes recur twice per rev, and the plan already loiters
            # for many revs while drift-phasing, so the burn is scheduled at a
            # node crossing inside the existing loiter -> free. Only a plan
            # shorter than half a period can be forced to wait.
            P = 2 * math.pi * math.sqrt(max(a_s, a_t) ** 3 / MU)
            wait = rng.uniform(0, 0.5 * P)
            if tot_s * DT < 0.5 * P:
                tot_s = tot_s + wait / DT
        else:
            theta, dv_pl = 0.0, 0.0

        dv_seq = dv_in + dv_pl
        dv_comb = (dv_in - dv_big) + math.hypot(dv_big, dv_pl)
        dvs_seq.append(dv_seq); dvs_comb.append(dv_comb)
        dvs_pl.append(dv_pl); steps.append(tot_s); thetas.append(math.degrees(theta))
        a1 = dv_seq <= budget; a2 = dv_comb <= budget; t_ok = tot_s <= horizon
        ok['dv_seq'] += a1; ok['dv_comb'] += a2; ok['t'] += t_ok
        ok['both_seq'] += (a1 and t_ok); ok['both_comb'] += (a2 and t_ok)

    dvs_seq = np.array(dvs_seq); dvs_comb = np.array(dvs_comb)
    dvs_pl = np.array(dvs_pl); st = np.array(steps); fin = st[np.isfinite(st)]
    return dict(alt_lo_km=alt_lo, alt_hi_km=alt_hi, e_max=e_max,
                de_max=(de_max if de_max is not None else -1),
                da_max_km=(da_max_km if da_max_km is not None else -1),
                same_orbit=int(same_orbit), horizon=horizon,
                di_max_deg=di_max_deg, n=n, budget_ms=budget,
                frac_dv_ok_seq=ok['dv_seq'] / n, frac_dv_ok_comb=ok['dv_comb'] / n,
                frac_time_ok=ok['t'] / n,
                frac_feasible_seq=ok['both_seq'] / n,
                frac_feasible_comb=ok['both_comb'] / n,
                dv_seq_p50=float(np.percentile(dvs_seq, 50)),
                dv_seq_p90=float(np.percentile(dvs_seq, 90)),
                dv_seq_p99=float(np.percentile(dvs_seq, 99)),
                dv_comb_p90=float(np.percentile(dvs_comb, 90)),
                dv_plane_p50=float(np.percentile(dvs_pl, 50)),
                dv_plane_p90=float(np.percentile(dvs_pl, 90)),
                theta_p50=float(np.percentile(thetas, 50)),
                theta_p90=float(np.percentile(thetas, 90)),
                steps_p50=float(np.percentile(fin, 50)) if len(fin) else math.nan,
                steps_p90=float(np.percentile(fin, 90)) if len(fin) else math.nan)


BANDS = [
    # name, kwargs (mirroring the proven 2D rungs)
    ("L1/WL1 e=0 same-orbit 500-800", dict(alt_lo=500, alt_hi=800, e_max=0.0,
                                           horizon=3000, same_orbit=True)),
    ("L2 headline 300-800 e<=0.05", dict(alt_lo=300, alt_hi=800, e_max=0.05,
                                         horizon=3000)),
    ("WL3 300-2000 e<=0.15", dict(alt_lo=300, alt_hi=2000, e_max=0.15,
                                  horizon=3000, da_max_km=400, de_max=0.06)),
    ("WL4 300-8000 e<=0.30", dict(alt_lo=300, alt_hi=8000, e_max=0.30,
                                  horizon=6000, da_max_km=600, de_max=0.08)),
    ("M5 300-20200 e<=0.50", dict(alt_lo=300, alt_hi=20200, e_max=0.50,
                                  horizon=12000, da_max_km=1000, de_max=0.10)),
]
DI_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2, 1.5,
            1.75, 2.0, 2.5, 3.0]


def main():
    rows, solved = [], []
    N = 4000
    for name, kw in BANDS:
        band_rows = []
        for di in DI_SWEEP:
            r = sample(n=N, seed=2026, di_max_deg=di, **kw)
            r['band'] = name
            rows.append(r); band_rows.append(r)
        base = band_rows[0]
        b = base['frac_feasible_seq']
        # di_max criteria are stated RELATIVE to the band's own 2D baseline:
        # the plane leg may cost at most 1pp (or 5pp) of joint feasibility.
        d1 = d5 = d1c = 0.0
        for r in band_rows:
            if r['frac_feasible_seq'] >= b - 0.01:
                d1 = r['di_max_deg']
            if r['frac_feasible_seq'] >= b - 0.05:
                d5 = r['di_max_deg']
            if r['frac_feasible_comb'] >= b - 0.01:
                d1c = r['di_max_deg']
        solved.append(dict(band=name, alt_lo_km=kw['alt_lo'], alt_hi_km=kw['alt_hi'],
                           e_max=kw['e_max'], horizon=kw['horizon'],
                           feas_2d_baseline=b, dv_p90_2d=base['dv_seq_p90'],
                           di_max_deg_lose1pp_seq=d1,
                           di_max_deg_lose5pp_seq=d5,
                           di_max_deg_lose1pp_comb=d1c))
    os.makedirs(OUT, exist_ok=True)
    keys = ['band'] + [k for k in rows[0] if k != 'band']
    with open(f"{OUT}/ext_3d_dv_feasibility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(f"{OUT}/ext_3d_dv_dimax.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(solved[0].keys()))
        w.writeheader(); w.writerows(solved)

    print(f"{'band':34} {'di_max':>6} {'%dv_seq':>8} {'%dv_cmb':>8} {'%FEAS':>7} "
          f"{'dvp90':>7} {'dvpl_p90':>8} {'steps_p90':>9}")
    for r in rows:
        print(f"{r['band']:34} {r['di_max_deg']:6.2f} "
              f"{100*r['frac_dv_ok_seq']:7.1f}% {100*r['frac_dv_ok_comb']:7.1f}% "
              f"{100*r['frac_feasible_seq']:6.1f}% {r['dv_seq_p90']:7.0f} "
              f"{r['dv_plane_p90']:8.1f} {r['steps_p90']:9.0f}")
    print("\nSolved di_max (deg), relative to each band's own 2D baseline:")
    for s in solved:
        print(f"  {s['band']:34} 2D-baseline {100*s['feas_2d_baseline']:5.1f}%  "
              f"di_max(-1pp,seq) = {s['di_max_deg_lose1pp_seq']:.2f}   "
              f"(-5pp,seq) = {s['di_max_deg_lose5pp_seq']:.2f}   "
              f"(-1pp,comb) = {s['di_max_deg_lose1pp_comb']:.2f}")


if __name__ == "__main__":
    main()
