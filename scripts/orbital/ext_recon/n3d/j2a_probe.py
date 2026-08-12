#!/usr/bin/env python3
"""
J2-A probes. Read-only recon; writes web_data/results/j2_*.csv only.

  A  secular rate table + differential rates across the proven envelope
  B  warp-exactness of the proposed propagator ordering (additivity, anchor)
  C  shaping_mode-2 re-audit: 5-leg ledger under j2_mode 0 vs 1
  D  do-nothing leak: coast-only ΔΦ over the caps
  E  the differential-nodal-drift channel (free plane change per rad of phase)
  F  oracle protocol: Cowell-J2 orbit-averaged rates vs the secular formulas,
     with a mutation table (what each plausible sign/factor error scores)
  G  mean-element-truth error at the terminal box
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j2a_core import (MU, R_ENV, R_EQ, J2, DT, DV_BUDGET, Orb, propagate,
                      apply_impulse, secular_rates, phi_mode2, di_rel,
                      coe2rv, rv2coe, integrate_j2, nu_to_M, E_to_nu, kepler_E)

OUT = "/Users/pete/space_training/web_data/results"
D2R = math.radians
R2D = math.degrees


def dump(name, rows):
    p = f"{OUT}/{name}"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {p}  ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════ A ═════
def probe_A():
    print("\n=== A. secular rates + differential rates ===")
    rows = []
    for alt in (300.0, 400.0, 800.0, 2000.0, 8000.0, 20200.0, 35786.0):
        a = R_ENV + alt * 1e3
        vc = math.sqrt(MU / a)
        for i_deg in (0.0, 28.5, 45.0, 51.6, 63.4, 90.0, 97.4):
            i = D2R(i_deg)
            for e in (0.0, 0.30):
                n, Om, om, Md = secular_rates(a, e, i)
                dM = Md - n
                rows.append(dict(
                    alt_km=alt, i_deg=i_deg, e=e,
                    period_min=2 * math.pi / n / 60.0,
                    RAAN_dot_deg_day=R2D(Om) * 86400,
                    argp_dot_deg_day=R2D(om) * 86400,
                    Mdot_minus_n_deg_day=R2D(dM) * 86400,
                    Mdot_rel_corr=dM / n,
                    v_circ=vc,
                    dv_per_deg_plane=2 * vc * math.sin(0.5 * D2R(1.0)),
                ))
    dump("j2_rates.csv", rows)
    print(f"  {'alt':>7} {'i':>6} {'Om_dot':>10} {'w_dot':>10} {'dM/n':>11}")
    for r in rows:
        if r["e"] == 0.0 and r["i_deg"] in (28.5, 51.6, 97.4) and r["alt_km"] in (400.0, 8000.0, 20200.0):
            print(f"  {r['alt_km']:7.0f} {r['i_deg']:6.1f} {r['RAAN_dot_deg_day']:10.4f} "
                  f"{r['argp_dot_deg_day']:10.4f} {r['Mdot_rel_corr']:11.3e}")
    return rows


# ═══════════════════════════════════════════════════════════════════ B ═════
def probe_B():
    """Warp exactness: does one call with dt = tau*DT equal tau calls with DT?
    Also the j2_mode=0 bit-exact anchor."""
    print("\n=== B. warp exactness / anchor ===")
    rows = []
    base = dict(a=R_ENV + 500e3, e=0.05, i=D2R(51.6), raan=D2R(40.0),
                argp=D2R(70.0), M=D2R(200.0))
    for tau in (1, 5, 30, 60, 180, 360, 3000):
        for j2 in (0, 1):
            o1 = Orb(**base)
            for _ in range(tau):
                propagate(o1, DT, j2)
            o2 = Orb(**base)
            propagate(o2, tau * DT, j2)
            r1, v1 = o1.rv()
            r2, v2 = o2.rv()
            rows.append(dict(tau=tau, j2_mode=j2,
                             dM_rad=abs(o1.M - o2.M),
                             draan_rad=abs(o1.raan - o2.raan),
                             dargp_rad=abs(o1.argp - o2.argp),
                             dpos_m=float(np.linalg.norm(r1 - r2)),
                             dvel_ms=float(np.linalg.norm(v1 - v2))))
    # anchor: j2_mode=0 path vs a verbatim two-body reference
    worst_anchor = 0.0
    rng = np.random.default_rng(7)
    for _ in range(20000):
        a = R_ENV + rng.uniform(300e3, 8000e3)
        e = rng.uniform(0.0, 0.30)
        o = Orb(a, e, D2R(rng.uniform(0, 60)), rng.uniform(0, 2 * math.pi),
                rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi))
        ref_M = o.M + math.sqrt(MU / a ** 3) * DT
        ref_M = math.fmod(ref_M, 2 * math.pi)
        if ref_M < 0:
            ref_M += 2 * math.pi
        ref_raan, ref_argp = o.raan, o.argp
        propagate(o, DT, 0)
        worst_anchor = max(worst_anchor, abs(o.M - ref_M),
                           abs(o.raan - ref_raan), abs(o.argp - ref_argp))
    print(f"  j2_mode=0 anchor over 20000 draws: worst |Δ| = {worst_anchor:.3e} rad "
          f"({'BIT-EXACT' if worst_anchor == 0.0 else 'NOT bit-exact'})")
    dump("j2_warp_exact.csv", rows)
    for r in rows:
        if r["j2_mode"] == 1:
            print(f"  tau={r['tau']:5d}  j2=1  |Δpos| = {r['dpos_m']:.3e} m   "
                  f"|ΔM| = {r['dM_rad']:.3e} rad")
    return worst_anchor, rows


# ═══════════════════════════════════════════════════════════════════ C ═════
def scripted_ledger(j2, di0_deg=1.0, dlam0_deg=180.0, da_drift_km=-200.0,
                    e0=0.02, i_t_deg=51.6, raan_t_deg=40.0, alt_km=500.0,
                    crank_n=6, crank_dv=25.0, w_match=0.817, dv_ref=700.0):
    """3d_C §3.1 reference maneuver, re-scripted: coast-to-node → plane crank →
    drift-open → drift → drift-close. Returns per-leg ΔΦ ledger."""
    a_t = R_ENV + alt_km * 1e3
    tgt = Orb(a_t, e0, D2R(i_t_deg), D2R(raan_t_deg), D2R(20.0), D2R(10.0))
    # chaser: same a,e; plane rotated by di0 about a random-but-fixed axis in the
    # target plane; mean longitude offset by dlam0.
    ht = tgt.hhat()
    n_ax = np.array([math.cos(D2R(raan_t_deg)), math.sin(D2R(raan_t_deg)), 0.0])
    n_ax = n_ax - np.dot(n_ax, ht) * ht
    n_ax /= np.linalg.norm(n_ax)
    d = D2R(di0_deg)
    K = np.array([[0, -n_ax[2], n_ax[1]], [n_ax[2], 0, -n_ax[0]], [-n_ax[1], n_ax[0], 0]])
    R = np.eye(3) + math.sin(d) * K + (1 - math.cos(d)) * (K @ K)
    rt, vt = tgt.rv()
    a, e, i, raan, argp, nu = rv2coe(R @ rt, R @ vt)
    sat = Orb(a, e, i, raan, argp, nu_to_M(nu, e))
    sat.M = math.fmod(sat.M + D2R(dlam0_deg), 2 * math.pi)
    sat.nu = E_to_nu(kepler_E(sat.M, sat.e), sat.e)

    legs = []
    dv_tot = 0.0
    worst_adverse = 0.0
    phi0 = phi_mode2(sat, tgt, w_match=w_match, dv_ref=dv_ref)[0]
    phi = phi0

    def step_coast(nsteps):
        nonlocal phi, worst_adverse
        for _ in range(nsteps):
            propagate(sat, DT, j2)
            propagate(tgt, DT, j2)
            p = phi_mode2(sat, tgt, w_match=w_match, dv_ref=dv_ref)[0]
            worst_adverse = min(worst_adverse, p - phi)
            phi = p

    def step_burn(dv_pro, dv_rad, dv_nor):
        nonlocal sat, phi, dv_tot, worst_adverse
        sat, mag = apply_impulse(sat, dv_pro, dv_rad, dv_nor)
        dv_tot += mag
        propagate(sat, DT, j2)
        propagate(tgt, DT, j2)
        p = phi_mode2(sat, tgt, w_match=w_match, dv_ref=dv_ref)[0]
        worst_adverse = min(worst_adverse, p - phi)
        phi = p
        return mag

    # L1a coast to the relative node (search over one period)
    n_s = math.sqrt(MU / sat.a ** 3)
    per_steps = int(2 * math.pi / n_s / DT)
    best_k, best_c = 0, 1e9
    probe_s, probe_t = sat.copy(), tgt.copy()
    for k in range(per_steps + 1):
        hs = probe_s.hhat()
        ht2 = probe_t.hhat()
        c = np.cross(ht2, hs)
        nn = np.linalg.norm(c)
        node = c / nn if nn > 1e-15 else np.array([1.0, 0, 0])
        rr, _ = probe_s.rv()
        rr = rr / np.linalg.norm(rr)
        cost = abs(abs(float(np.dot(rr, node))) - 1.0)
        if cost < best_c:
            best_c, best_k = cost, k
        propagate(probe_s, DT, j2)
        propagate(probe_t, DT, j2)
    p_before = phi
    step_coast(best_k)
    legs.append(("L1a coast-to-node", phi - p_before, 0.0, best_k))

    # L1b plane crank
    p_before, dv_before = phi, dv_tot
    hs, ht2 = sat.hhat(), tgt.hhat()
    sgn = 1.0 if float(np.dot(np.cross(ht2, hs), sat.rv()[0])) < 0 else -1.0
    for _ in range(crank_n):
        step_burn(0.0, 0.0, sgn * crank_dv)
    legs.append(("L1b plane crank", phi - p_before, dv_tot - dv_before, crank_n))
    di_after_crank = R2D(di_rel(sat, tgt))

    # L2 drift-open: tangential burns to set δa = da_drift_km
    p_before, dv_before = phi, dv_tot
    v_c = math.sqrt(MU / sat.a)
    dv_needed = 0.5 * v_c * (da_drift_km * 1e3) / sat.a
    nburn = 5
    for _ in range(nburn):
        step_burn(dv_needed / nburn, 0.0, 0.0)
    legs.append(("L2 drift-open", phi - p_before, dv_tot - dv_before, nburn))

    # L3 drift until |Δλ| closes (capped)
    p_before = phi
    dlam_prev = abs(phi_mode2(sat, tgt, w_match=w_match, dv_ref=dv_ref)[3])
    nsteps = 0
    slice_rows = []
    while nsteps < 6000:
        step_coast(30)
        nsteps += 30
        dl = abs(phi_mode2(sat, tgt, w_match=w_match, dv_ref=dv_ref)[3])
        slice_rows.append((nsteps, R2D(dl), phi, R2D(di_rel(sat, tgt))))
        if dl > dlam_prev and dl < D2R(20.0):
            break
        if dl < D2R(2.0):
            break
        dlam_prev = dl
    legs.append(("L3 drift", phi - p_before, 0.0, nsteps))
    di_after_drift = R2D(di_rel(sat, tgt))

    # L4 drift-close
    p_before, dv_before = phi, dv_tot
    for _ in range(nburn):
        step_burn(-dv_needed / nburn, 0.0, 0.0)
    legs.append(("L4 drift-close", phi - p_before, dv_tot - dv_before, nburn))

    return dict(legs=legs, phi0=phi0, phiT=phi, total=phi - phi0, dv_tot=dv_tot,
                worst_adverse=worst_adverse, di_after_crank=di_after_crank,
                di_after_drift=di_after_drift, drift_steps=nsteps,
                slices=slice_rows)


def probe_C():
    print("\n=== C. shaping_mode 2 ledger, j2_mode 0 vs 1 ===")
    rows = []
    res = {}
    for j2 in (0, 1):
        r = scripted_ledger(j2)
        res[j2] = r
        for (name, dphi, dv, ns) in r["legs"]:
            rows.append(dict(j2_mode=j2, leg=name, dPhi=dphi, dv_ms=dv, steps=ns))
        rows.append(dict(j2_mode=j2, leg="TOTAL", dPhi=r["total"], dv_ms=r["dv_tot"],
                         steps=sum(l[3] for l in r["legs"])))
        rows.append(dict(j2_mode=j2, leg="worst_adverse_step", dPhi=r["worst_adverse"],
                         dv_ms=0.0, steps=1))
        rows.append(dict(j2_mode=j2, leg="di_rel_after_crank_deg", dPhi=r["di_after_crank"],
                         dv_ms=0.0, steps=0))
        rows.append(dict(j2_mode=j2, leg="di_rel_after_drift_deg", dPhi=r["di_after_drift"],
                         dv_ms=0.0, steps=0))
    dump("j2_shaping_ledger.csv", rows)
    print(f"  {'leg':<24} {'ΔΦ(j2=0)':>10} {'ΔΦ(j2=1)':>10} {'Δ':>10}")
    for k, (name, dphi0, _, _) in enumerate(res[0]["legs"]):
        dphi1 = res[1]["legs"][k][1]
        print(f"  {name:<24} {dphi0:10.4f} {dphi1:10.4f} {dphi1 - dphi0:10.4f}")
    for key in ("total", "worst_adverse", "di_after_crank", "di_after_drift",
                "dv_tot", "drift_steps"):
        print(f"  {key:<24} {res[0][key]:10.4f} {res[1][key]:10.4f} "
              f"{res[1][key] - res[0][key]:10.4f}")
    # drift-leg monotonicity under J2: per-slice ΔΦ signs
    for j2 in (0, 1):
        sl = res[j2]["slices"]
        d = [sl[k][2] - sl[k - 1][2] for k in range(1, len(sl))]
        print(f"  j2={j2} drift slices: n={len(d)} min={min(d):+.5f} max={max(d):+.5f} "
              f"n_negative={sum(1 for x in d if x < 0)}")
    return res


# ═══════════════════════════════════════════════════════════════════ D ═════
def probe_D():
    """Do-nothing (pure coast) ΔΦ under J2 — the T3 'shaping leak' gate."""
    print("\n=== D. do-nothing shaping leak (coast only) ===")
    rows = []
    for alt in (500.0, 8000.0, 20200.0):
        a_t = R_ENV + alt * 1e3
        for i_t_deg in (0.0, 28.5, 51.6, 97.4):
            for di_deg in (0.0, 0.05, 0.25, 1.0):
                for da_km in (0.0, 200.0):
                    for j2 in (0, 1):
                        tgt = Orb(a_t, 0.02, D2R(i_t_deg), D2R(40.0), D2R(20.0), D2R(10.0))
                        ht = tgt.hhat()
                        n_ax = np.array([1.0, 0.0, 0.0])
                        n_ax = n_ax - np.dot(n_ax, ht) * ht
                        n_ax /= np.linalg.norm(n_ax)
                        d = D2R(di_deg)
                        K = np.array([[0, -n_ax[2], n_ax[1]], [n_ax[2], 0, -n_ax[0]],
                                      [-n_ax[1], n_ax[0], 0]])
                        R = np.eye(3) + math.sin(d) * K + (1 - math.cos(d)) * (K @ K)
                        rt, vt = tgt.rv()
                        aa, ee, ii, rr, ww, nn = rv2coe(R @ rt, R @ vt)
                        sat = Orb(aa + da_km * 1e3, ee, ii, rr, ww, nu_to_M(nn, ee))
                        cap = 12000 if alt > 10000 else 6000
                        phi0, _, dvpl0, dl0 = phi_mode2(sat, tgt)
                        worst = 0.0
                        prev = phi0
                        for k in range(cap):
                            propagate(sat, DT, j2)
                            propagate(tgt, DT, j2)
                            if k % 50 == 0 or k == cap - 1:
                                p = phi_mode2(sat, tgt)[0]
                                worst = max(worst, p - phi0)
                                prev = p
                        phiT, _, dvplT, dlT = phi_mode2(sat, tgt)
                        rows.append(dict(alt_km=alt, i_t_deg=i_t_deg, di_deg=di_deg,
                                         da_km=da_km, j2_mode=j2, cap_steps=cap,
                                         dPhi_total=phiT - phi0,
                                         max_Phi_gain=worst,
                                         dvpl0=dvpl0, dvplT=dvplT,
                                         d_dvpl_ms=dvplT - dvpl0,
                                         dlam0_deg=R2D(dl0), dlamT_deg=R2D(dlT),
                                         di_rel_T_deg=R2D(di_rel(sat, tgt))))
    dump("j2_donothing.csv", rows)
    print(f"  {'alt':>6} {'i_t':>6} {'di':>5} {'da':>5} {'j2':>3} {'ΔΦ':>9} "
          f"{'maxgain':>9} {'Δdv_pl':>9} {'di_rel_T':>9}")
    for r in rows:
        if r["da_km"] in (0.0, 200.0) and r["di_deg"] in (0.0, 1.0) and r["alt_km"] == 500.0:
            print(f"  {r['alt_km']:6.0f} {r['i_t_deg']:6.1f} {r['di_deg']:5.2f} "
                  f"{r['da_km']:5.0f} {r['j2_mode']:3d} {r['dPhi_total']:9.4f} "
                  f"{r['max_Phi_gain']:9.4f} {r['d_dvpl_ms']:9.2f} {r['di_rel_T_deg']:9.4f}")
    return rows


# ═══════════════════════════════════════════════════════════════════ E ═════
def probe_E():
    """Differential-nodal-drift channel: free plane change per radian of phase
    closed, and what fits inside each cap."""
    print("\n=== E. differential nodal drift as a plane-change channel ===")
    rows = []
    for alt in (300.0, 500.0, 800.0, 2000.0, 8000.0, 20200.0):
        a = R_ENV + alt * 1e3
        vc = math.sqrt(MU / a)
        n = math.sqrt(MU / a ** 3)
        for i_deg in (0.0, 15.0, 28.5, 45.0, 51.6, 63.4, 75.0, 90.0, 97.4):
            i = D2R(i_deg)
            # analytic: dΩ/dλ = -3.5 J2 (R_EQ/p)^2 cos i  (independent of δa)
            kk = J2 * (R_EQ / a) ** 2
            dOm_per_rad = -3.5 * kk * math.cos(i)
            di_per_rad = abs(dOm_per_rad) * math.sin(i)   # small-angle Δi_rel ≈ ΔΩ sin i
            for dlam_deg in (180.0, 360.0):
                dOm = abs(dOm_per_rad) * D2R(dlam_deg)
                di_gain = 2 * math.asin(min(1.0, math.sin(i) * math.sin(0.5 * dOm)))
                dv_saved = 2 * vc * math.sin(0.5 * di_gain)
                for da_km in (100.0, 200.0, 400.0):
                    dn = 1.5 * n * (da_km * 1e3) / a
                    t_s = D2R(dlam_deg) / dn
                    rows.append(dict(alt_km=alt, i_deg=i_deg, dlam_deg=dlam_deg,
                                     da_km=da_km,
                                     dOm_per_rad_deg=R2D(dOm_per_rad),
                                     di_per_rad_deg=R2D(di_per_rad),
                                     dOm_deg=R2D(dOm),
                                     di_gain_deg=R2D(di_gain),
                                     dv_saved_ms=dv_saved,
                                     frac_budget=dv_saved / DV_BUDGET,
                                     drift_hours=t_s / 3600.0,
                                     drift_steps=t_s / DT,
                                     fits_6000=t_s / DT <= 6000,
                                     fits_12000=t_s / DT <= 12000))
    dump("j2_channel.csv", rows)
    print(f"  {'alt':>6} {'i':>6} {'dΩ/dλ(°/rad)':>13} {'Δi/λ(°/rad)':>12} "
          f"{'Δi@2π(°)':>10} {'Δv saved':>9} {'%budget':>8}")
    for r in rows:
        if r["da_km"] == 200.0 and r["dlam_deg"] == 360.0 and r["alt_km"] in (500.0, 8000.0, 20200.0):
            print(f"  {r['alt_km']:6.0f} {r['i_deg']:6.1f} {r['dOm_per_rad_deg']:13.4f} "
                  f"{r['di_per_rad_deg']:12.4f} {r['di_gain_deg']:10.4f} "
                  f"{r['dv_saved_ms']:9.1f} {100*r['frac_budget']:8.2f}")
    return rows


# ═══════════════════════════════════════════════════════════════════ F ═════
def fit_rate(t, ang):
    a = np.unwrap(ang)
    A = np.vstack([t, np.ones_like(t)]).T
    sol, *_ = np.linalg.lstsq(A, a, rcond=None)
    return sol[0]


def measured_rates(a, e, i, raan, argp, M, n_orbits=40, pts_per_orbit=64,
                   j2=J2, rtol=1e-12):
    n = math.sqrt(MU / a ** 3)
    T = 2 * math.pi / n
    nu = E_to_nu(kepler_E(M, e), e)
    r0, v0 = coe2rv(a, e, i, raan, argp, nu)
    ts, Y = integrate_j2(np.array(r0), np.array(v0), n_orbits * T,
                         n_orbits * pts_per_orbit + 1, rtol=rtol, j2=j2)
    Om, w, Mm = [], [], []
    for k in range(Y.shape[1]):
        aa, ee, ii, rr, ww, nn = rv2coe(Y[0:3, k], Y[3:6, k])
        Om.append(rr)
        w.append(ww)
        Mm.append(nu_to_M(nn, ee) % (2 * math.pi))
    return (fit_rate(ts, np.array(Om)), fit_rate(ts, np.array(w)),
            fit_rate(ts, np.array(Mm)))


def probe_F():
    print("\n=== F. oracle protocol: Cowell-J2 orbit-averaged rates vs secular ===")
    rows = []
    cases = [
        ("LEO-500 i=51.6 e=0.02", R_ENV + 500e3, 0.02, 51.6),
        ("LEO-500 i=28.5 e=0.00", R_ENV + 500e3, 1e-6, 28.5),
        ("LEO-800 i=97.4 e=0.05", R_ENV + 800e3, 0.05, 97.4),
        ("WIDE-8000 i=45 e=0.30", R_ENV + 8000e3, 0.30, 45.0),
        ("MEO-20200 i=55 e=0.01", R_ENV + 20200e3, 0.01, 55.0),
    ]
    for name, a, e, i_deg in cases:
        i = D2R(i_deg)
        n, Om_s, om_s, Md_s = secular_rates(a, e, i)
        for n_orb in (5, 20, 60):
            Om_m, om_m, Md_m = measured_rates(a, e, i, D2R(40.0), D2R(70.0), D2R(200.0),
                                              n_orbits=n_orb)
            rows.append(dict(case=name, n_orbits=n_orb,
                             Om_secular=Om_s, Om_measured=Om_m,
                             Om_relerr=abs(Om_m - Om_s) / abs(Om_s) if Om_s else float("nan"),
                             om_secular=om_s, om_measured=om_m,
                             om_relerr=abs(om_m - om_s) / abs(om_s) if om_s else float("nan"),
                             Mdot_secular=Md_s, Mdot_measured=Md_m,
                             Mdot_relerr=abs(Md_m - Md_s) / Md_s,
                             Mdot_corr_relerr=abs((Md_m - n) - (Md_s - n)) / abs(Md_s - n)))
        print(f"  {name}")
        for r in rows[-3:]:
            print(f"    N={r['n_orbits']:3d}  Ω̇ relerr {r['Om_relerr']:.3e}   "
                  f"ω̇ relerr {r['om_relerr']:.3e}   (Ṁ−n) relerr {r['Mdot_corr_relerr']:.3e}")

    # mutation table: what each plausible implementation error scores on Ω̇/ω̇
    print("\n  -- mutation discrimination (LEO-500 i=51.6 e=0.02, N=20 orbits) --")
    a, e, i = R_ENV + 500e3, 0.02, D2R(51.6)
    n, Om_s, om_s, Md_s = secular_rates(a, e, i)
    Om_m, om_m, Md_m = measured_rates(a, e, i, D2R(40.0), D2R(70.0), D2R(200.0), n_orbits=20)
    p = a * (1 - e * e)
    k = 1.5 * n * J2 * (R_EQ / p) ** 2
    muts = {
        "correct":                     (Om_s, om_s, Md_s),
        "Ω̇ sign flipped":              (+k * math.cos(i), om_s, Md_s),
        "Ω̇ factor 0.75 not 1.5":       (-0.5 * k * math.cos(i), om_s, Md_s),
        "Ω̇ uses sin i not cos i":      (-k * math.sin(i), om_s, Md_s),
        "Ω̇ uses R_ENV not R_EQ":       (-1.5 * n * J2 * (R_ENV / p) ** 2 * math.cos(i), om_s, Md_s),
        "Ω̇ uses a not p":              (-1.5 * n * J2 * (R_EQ / a) ** 2 * math.cos(i), om_s, Md_s),
        "ω̇ (2−2.5sin²i) [Ṁ form]":    (Om_s, 0.5 * k * (2 - 2.5 * math.sin(i) ** 2), Md_s),
        "ω̇ sign flipped":              (Om_s, -om_s, Md_s),
        "ω̇ factor 2×":                 (Om_s, 2 * om_s, Md_s),
        "Ṁ correction omitted":        (Om_s, om_s, n),
        "Ṁ correction sign flipped":   (Om_s, om_s, n - (Md_s - n)),
        "Ṁ missing √(1−e²)":           (Om_s, om_s, n + 0.5 * k * (2 - 3 * math.sin(i) ** 2)),
    }
    mrows = []
    for label, (O, W, MD) in muts.items():
        mrows.append(dict(mutation=label,
                          Om_relerr=abs(Om_m - O) / abs(Om_m),
                          om_relerr=abs(om_m - W) / abs(om_m),
                          Mdot_corr_relerr=abs((Md_m - n) - (MD - n)) / abs(Md_m - n)))
        print(f"    {label:<30} Ω̇ {mrows[-1]['Om_relerr']:9.3e}  "
              f"ω̇ {mrows[-1]['om_relerr']:9.3e}  (Ṁ−n) {mrows[-1]['Mdot_corr_relerr']:9.3e}")
    dump("j2_oracle.csv", rows)
    dump("j2_oracle_mutations.csv", mrows)
    return rows, mrows


# ═══════════════════════════════════════════════════════════════════ G ═════
def probe_G():
    """'Truth = mean elements': how far is the mean-element world from an
    osculating-J2 world, in the quantity the success classifier keys on
    (relative position / relative velocity)?"""
    print("\n=== G. mean-element truth vs osculating, at the box ===")
    rows = []
    a_t = R_ENV + 500e3
    tgt0 = Orb(a_t, 0.02, D2R(51.6), D2R(40.0), D2R(20.0), D2R(10.0))
    for sep_km in (5.0, 30.0, 200.0, 2000.0):
        # chaser offset purely along-track by sep (approximate: ΔM = sep/a)
        sat0 = Orb(a_t, 0.02, D2R(51.6), D2R(40.0), D2R(20.0),
                   D2R(10.0) + sep_km * 1e3 / a_t)
        n = math.sqrt(MU / a_t ** 3)
        T = 2 * math.pi / n
        for n_orb in (1, 3):
            # mean-element (env) path
            s, t = sat0.copy(), tgt0.copy()
            nsteps = int(n_orb * T / DT)
            for _ in range(nsteps):
                propagate(s, DT, 1)
                propagate(t, DT, 1)
            rs, vs = s.rv()
            rt, vt = t.rv()
            rel_mean = rs - rt
            vrel_mean = vs - vt
            # osculating path: same ICs read as osculating, full J2 Cowell
            r0s, v0s = sat0.rv()
            r0t, v0t = tgt0.rv()
            tend = nsteps * DT
            _, Ys = integrate_j2(np.array(r0s), np.array(v0s), tend, 2)
            _, Yt = integrate_j2(np.array(r0t), np.array(v0t), tend, 2)
            rel_osc = Ys[0:3, -1] - Yt[0:3, -1]
            vrel_osc = Ys[3:6, -1] - Yt[3:6, -1]
            # absolute (single-body) mean-vs-osculating for reference
            abs_err = float(np.linalg.norm(rt - Yt[0:3, -1]))
            rows.append(dict(sep_km=sep_km, n_orbits=n_orb, steps=nsteps,
                             rel_pos_err_m=float(np.linalg.norm(rel_mean - rel_osc)),
                             rel_vel_err_ms=float(np.linalg.norm(vrel_mean - vrel_osc)),
                             abs_pos_err_m=abs_err,
                             rel_range_mean_m=float(np.linalg.norm(rel_mean))))
    dump("j2_meantruth.csv", rows)
    print(f"  {'sep_km':>7} {'orbits':>7} {'|Δρ| err (m)':>14} {'|Δρ̇| err (m/s)':>16} "
          f"{'abs err (m)':>13}")
    for r in rows:
        print(f"  {r['sep_km']:7.0f} {r['n_orbits']:7d} {r['rel_pos_err_m']:14.1f} "
              f"{r['rel_vel_err_ms']:16.4f} {r['abs_pos_err_m']:13.1f}")
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "ABCDEFG"
    if "A" in which:
        probe_A()
    if "B" in which:
        probe_B()
    if "C" in which:
        probe_C()
    if "D" in which:
        probe_D()
    if "E" in which:
        probe_E()
    if "F" in which:
        probe_F()
    if "G" in which:
        probe_G()
