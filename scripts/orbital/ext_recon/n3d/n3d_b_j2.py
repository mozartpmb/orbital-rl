#!/usr/bin/env python3
"""N3D-B.3 — does J2 change the 3D angles-only observability picture?

Designed-for-later probe (the env's `j2_mode` obs slots are reserved but J2 is
not implemented). NAV-F/NAV-G both carry the same caveat: *in real LEO, J2 —
through differential nodal precession and the nonlinear mean->osculating map —
is the workhorse that breaks the angles-only scale ambiguity, and our two-body
sim has none, so our problem is HARDER than reality.* That caveat is stated
everywhere and measured nowhere. This measures it.

Mechanism under test: RAAN_dot = -1.5 n J2 (R_eq/p)^2 cos i and
argp_dot = 0.75 n J2 (R_eq/p)^2 (5cos^2 i - 1) both depend on (a, i), so a
difference in either produces a SECULAR relative drift — including an
out-of-plane one — that pure two-body does not. The sharp question for the
terminal box is whether J2 helps when the guidance objective has already nulled
da AND di, i.e. when the two orbits are identical and the differential rate is
therefore identically zero.

Both the truth roll and the 6x6 STM use the same RK4 flow (J2 on or off), so
the comparison is like-for-like and the J2-off arm doubles as an RK4-vs-oracle
control.

Writes web_data/results/j2_n3d_observability.csv
Run: python3 n3d_b_j2.py [--quick]
"""

import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import n3d_crlb3d as K                                          # noqa: E402

OUT = '/Users/pete/space_training/web_data/results/j2_n3d_observability.csv'
A_LEO = K.R_EARTH + 500e3
RHO_BOX_M = 5.0e3
RK4_H = 15.0

FIELDS = ['i_t_deg', 'da_m', 'di_deg', 'dv_ms', 'dv_axis', 'j2', 'orbits',
          'n_obs', 'rho0_km', 'rho_max_km', 'oop_amp_m', 'raan_rate_deg_day',
          'draan_deg_per_orbit', 'sigma_range_m', 'sigma_inplane_m',
          'sigma_plane_m', 'sigma_vrange_ms', 'sigma_vplane_ms',
          'range_frac_of_box', 'j2_gain', 'fim_cond']


def raan_rate(a, e, i_rad):
    p = a * (1.0 - e * e)
    n = math.sqrt(K.MU / a ** 3)
    return -1.5 * n * K.J2 * (K.R_EQ / p) ** 2 * math.cos(i_rad)


def prop_factory(j2):
    return lambda rv, dt: K.prop_j2(rv, dt, h=RK4_H, j2=j2)


def cell(i_t_deg, da, di_deg, dv, axis, j2, arcs, sep=RHO_BOX_M,
         burn_frac=0.20, nu_t=0.6):
    P = K.period(A_LEO)
    dur = max(arcs) * P
    pf = prop_factory(j2)
    rv_s, rv_t = K.make_pair(a_t=A_LEO, e=0.0, i_t=math.radians(i_t_deg),
                             nu_t=nu_t, sep_m=sep, da_m=da,
                             di_rad=math.radians(di_deg))
    burns = ((burn_frac * dur, dv, axis),) if dv > 0.0 else ()
    T, S, G = K.roll(rv_s, rv_t, dur, burns=burns, propagator=pf)
    idx = [min(int(round(o * P / K.DT_SENSOR)), len(T) - 1) for o in arcs]
    recs = K.crlb3d(T, S, G, cum_at=idx, propagator=pf)
    d = G[:, :3] - S[:, :3]
    rho = np.linalg.norm(d, axis=1)
    ht = np.cross(G[0, :3], G[0, 3:]); ht /= np.linalg.norm(ht)
    return recs, idx, rho, np.abs(d @ ht)


def main(quick=False):
    t0 = time.time()
    n = math.sqrt(K.MU / A_LEO ** 3)

    # RK4/J2-off control against the universal-variable oracle.
    rv = K.om3.coe2rv(A_LEO, 0.0, math.radians(51.6), 0.0, 0.0, 0.6)
    ctrl = float(np.linalg.norm(K.prop_j2(rv, 5545.0, h=RK4_H, j2=False)
                                - K.prop2b(rv, 5545.0))) / A_LEO
    F = K.stm_fd6(rv, 600.0, propagator=prop_factory(True))
    J = np.zeros((6, 6)); J[:3, 3:] = np.eye(3); J[3:, :3] = -np.eye(3)
    D = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])
    Fs = np.linalg.solve(D, F @ D)
    sym = float(np.abs(Fs.T @ J @ Fs - J).max())
    print(f"control: RK4(h={RK4_H:.0f} s, J2 off) vs oracle over 1 orbit = "
          f"{ctrl:.2e} rel   |   J2-STM symplecticity = {sym:.2e}\n")

    i_ts = [51.6, 97.4] if not quick else [51.6]
    v_tols = [0.0, 1.0, 50.0]
    das = [v / (1.5 * n) for v in v_tols]
    dis = [0.0, 0.04169392, 1.0] if not quick else [0.0, 1.0]
    burns = [(0.0, ''), (1.0, 'nor')] if not quick else [(0.0, '')]
    arcs = [1.0, 3.0] if not quick else [1.0]

    rows, base = [], {}
    print(f"{'i_t':>6} {'da_m':>9} {'dI':>8} {'dv':>5}{'ax':>5} {'J2':>4} "
          f"{'orb':>4} | {'sig_rng':>12} {'/box':>7} {'sig_plane':>10} "
          f"{'J2 gain':>10}")
    for i_t in i_ts:
        dO = math.degrees(raan_rate(A_LEO, 0.0, math.radians(i_t))) * 86400.0
        print(f"-- i_t = {i_t} deg, RAAN rate = {dO:+.3f} deg/day --")
        for v_tol, da in zip(v_tols, das):
            for di in dis:
                for dv, ax in burns:
                    for j2 in (False, True):
                        recs, idx, rho, oop = cell(i_t, da, di, dv, ax, j2,
                                                   arcs)
                        for o, r, kk in zip(arcs, recs, idx):
                            key = (i_t, round(da, 3), di, dv, ax, o)
                            if not j2:
                                base[key] = r['sigma_range_m']
                            b = base.get(key, float('nan'))
                            g = ((b / r['sigma_range_m']) ** 2
                                 if np.isfinite(b) and r['sigma_range_m'] > 0
                                 else float('inf'))
                            # differential nodal drift over one orbit, in deg
                            r1 = raan_rate(A_LEO, 0.0, math.radians(i_t))
                            r2 = raan_rate(A_LEO + da, 0.0,
                                           math.radians(i_t + di))
                            rows.append(dict(
                                i_t_deg=i_t, da_m=da, di_deg=di, dv_ms=dv,
                                dv_axis=ax or 'none', j2=int(j2), orbits=o,
                                n_obs=r['n_obs'], rho0_km=r['rho0_m'] / 1e3,
                                rho_max_km=float(rho[:kk + 1].max()) / 1e3,
                                oop_amp_m=float(oop[:kk + 1].max()),
                                raan_rate_deg_day=dO,
                                draan_deg_per_orbit=math.degrees(
                                    (r2 - r1) * K.period(A_LEO)),
                                sigma_range_m=r['sigma_range_m'],
                                sigma_inplane_m=r['sigma_inplane_m'],
                                sigma_plane_m=r['sigma_plane_m'],
                                sigma_vrange_ms=r['sigma_vrange_ms'],
                                sigma_vplane_ms=r['sigma_vplane_ms'],
                                range_frac_of_box=(r['sigma_range_m']
                                                   / RHO_BOX_M),
                                j2_gain=g, fim_cond=r['fim_cond']))
                            print(f"{i_t:6.1f} {da:9.1f} {di:8.4f} {dv:5.1f}"
                                  f"{(ax or '-'):>5} {int(j2):4d} {o:4.1f} | "
                                  f"{r['sigma_range_m']:12.2f} "
                                  f"{rows[-1]['range_frac_of_box']:7.3f} "
                                  f"{r['sigma_plane_m']:10.3f} "
                                  f"{g:10.3g}")
                    print()
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    print(f"wrote {OUT} ({len(rows)} rows) in {time.time()-t0:.1f} s")


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
