#!/usr/bin/env python3
"""
J2-A F2: the CORRECTED oracle protocol.

F1 (j2a_probe.py::probe_F) compared the Cowell-fitted secular rates against
secular_rates(a_IC, e_IC, i_IC) — i.e. it read the osculating initial condition
as if it were the mean element set. That injects a first-order-in-J2 bias:
  ⟨a_osc⟩ − a_IC ~ J2 (R/a)² a  ≈ 6 km at LEO  ⇒  δn/n ≈ −1.5·δa/a ≈ 1.4e-3
which is 3.5× amplified in Ω̇ (∝ a^-7/2) and ~12× LARGER than the whole J2
correction to Ṁ. Hence F1's Ω̇ floor stalls at 2.5e-3 and its Ṁ check is
useless (relerr 1.1–1000).

Fix: recover the MEAN elements from the integration by orbit-averaging the
osculating ones. Because M is linear in t to O(J2), uniform-in-time sampling is
uniform-in-M, and the Brouwer short-period terms in a, e, i average to zero over
M at first order — so ⟨a_osc⟩_t = a_mean + O(J2²).

Writes j2_oracle2.csv and j2_oracle2_mutations.csv.
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j2a_core import (MU, R_ENV, R_EQ, J2, Orb, secular_rates, coe2rv, rv2coe,
                      integrate_j2, nu_to_M, E_to_nu, kepler_E)

OUT = "/Users/pete/space_training/web_data/results"
D2R, R2D = math.radians, math.degrees


def fit_rate(t, ang):
    a = np.unwrap(ang)
    A = np.vstack([t, np.ones_like(t)]).T
    sol, *_ = np.linalg.lstsq(A, a, rcond=None)
    return sol[0]


def run_case(a, e, i, raan=D2R(40), argp=D2R(70), M=D2R(200),
             n_orbits=20, ppo=64, j2=J2):
    n = math.sqrt(MU / a ** 3)
    T = 2 * math.pi / n
    nu = E_to_nu(kepler_E(M, e), e)
    r0, v0 = coe2rv(a, e, i, raan, argp, nu)
    ts, Y = integrate_j2(np.array(r0), np.array(v0), n_orbits * T,
                         n_orbits * ppo + 1, j2=j2)
    A, E, I, Om, W, MM = [], [], [], [], [], []
    for k in range(Y.shape[1]):
        aa, ee, ii, rr, ww, nn = rv2coe(Y[0:3, k], Y[3:6, k])
        A.append(aa); E.append(ee); I.append(ii)
        Om.append(rr); W.append(ww); MM.append(nu_to_M(nn, ee) % (2 * math.pi))
    A, E, I = np.array(A), np.array(E), np.array(I)
    # orbit-average over an integer number of orbits, dropping the duplicate
    # endpoint so the average is over complete cycles of M
    a_m, e_m, i_m = A[:-1].mean(), E[:-1].mean(), I[:-1].mean()
    return (fit_rate(ts, np.array(Om)), fit_rate(ts, np.array(W)),
            fit_rate(ts, np.array(MM)), a_m, e_m, i_m, A, E, I)


CASES = [
    ("LEO-500 i=51.6 e=0.02", R_ENV + 500e3, 0.02, 51.6),
    ("LEO-500 i=28.5 e~0",    R_ENV + 500e3, 1e-6, 28.5),
    ("LEO-800 i=97.4 e=0.05", R_ENV + 800e3, 0.05, 97.4),
    ("WIDE-8000 i=45 e=0.30", R_ENV + 8000e3, 0.30, 45.0),
    ("MEO-20200 i=55 e=0.01", R_ENV + 20200e3, 0.01, 55.0),
]


def main():
    rows = []
    print("=== F2. corrected protocol: reference rates from ORBIT-AVERAGED mean elements ===")
    for name, a, e, i_deg in CASES:
        i = D2R(i_deg)
        for n_orb in (5, 20, 60):
            Om_m, om_m, Md_m, a_mean, e_mean, i_mean, A, E, I = run_case(
                a, e, i, n_orbits=n_orb)
            n_mean, Om_s, om_s, Md_s = secular_rates(a_mean, e_mean, i_mean)
            n_ic = math.sqrt(MU / a ** 3)
            rows.append(dict(
                case=name, n_orbits=n_orb,
                a_ic=a, a_mean=a_mean, da_mean_minus_ic_m=a_mean - a,
                da_rel=(a_mean - a) / a,
                a_sp_peak_to_peak_m=float(A.max() - A.min()),
                Om_secular=Om_s, Om_measured=Om_m,
                Om_relerr=abs(Om_m - Om_s) / abs(Om_s) if Om_s else float("nan"),
                om_secular=om_s, om_measured=om_m,
                om_relerr=abs(om_m - om_s) / abs(om_s),
                Mdot_corr_secular=Md_s - n_mean, Mdot_corr_measured=Md_m - n_mean,
                Mdot_corr_relerr=abs((Md_m - n_mean) - (Md_s - n_mean)) / abs(Md_s - n_mean),
            ))
        print(f"  {name}")
        for r in rows[-3:]:
            print(f"    N={r['n_orbits']:3d}  a_mean−a_IC = {r['da_mean_minus_ic_m']:9.1f} m "
                  f"({r['da_rel']:+.2e})   Ω̇ {r['Om_relerr']:.3e}  ω̇ {r['om_relerr']:.3e}  "
                  f"(Ṁ−n) {r['Mdot_corr_relerr']:.3e}")

    # mutation table at the two extreme cells
    mrows = []
    for name, a, e, i_deg, n_orb in [("LEO-500 i=51.6 e=0.02", R_ENV + 500e3, 0.02, 51.6, 20),
                                     ("WIDE-8000 i=45 e=0.30", R_ENV + 8000e3, 0.30, 45.0, 60)]:
        i = D2R(i_deg)
        Om_m, om_m, Md_m, a_mean, e_mean, i_mean, *_ = run_case(a, e, i, n_orbits=n_orb)
        n, Om_s, om_s, Md_s = secular_rates(a_mean, e_mean, i_mean)
        p = a_mean * (1 - e_mean ** 2)
        k = 1.5 * n * J2 * (R_EQ / p) ** 2
        si2 = math.sin(i_mean) ** 2
        muts = {
            "correct":                    (Om_s, om_s, Md_s),
            "Om_dot sign flipped":        (-Om_s, om_s, Md_s),
            "Om_dot 0.75 not 1.5":        (-0.5 * k * math.cos(i_mean), om_s, Md_s),
            "Om_dot sin i not cos i":     (-k * math.sin(i_mean), om_s, Md_s),
            "Om_dot R_ENV not R_EQ":      (-1.5 * n * J2 * (R_ENV / p) ** 2 * math.cos(i_mean), om_s, Md_s),
            "Om_dot a not p":             (-1.5 * n * J2 * (R_EQ / a_mean) ** 2 * math.cos(i_mean), om_s, Md_s),
            "argp_dot (2-2.5s2) [M form]": (Om_s, 0.5 * k * (2 - 2.5 * si2), Md_s),
            "argp_dot sign flipped":      (Om_s, -om_s, Md_s),
            "argp_dot 2x":                (Om_s, 2 * om_s, Md_s),
            "Mdot corr omitted":          (Om_s, om_s, n),
            "Mdot corr sign flipped":     (Om_s, om_s, n - (Md_s - n)),
            "Mdot corr (4-5s2) [w form]": (Om_s, om_s, n + 0.5 * k * math.sqrt(1 - e_mean ** 2) * (4 - 5 * si2)),
            "Mdot missing sqrt(1-e2)":    (Om_s, om_s, n + 0.5 * k * (2 - 3 * si2)),
        }
        print(f"\n  -- mutation discrimination, {name}, N={n_orb} --")
        for label, (O, W, MD) in muts.items():
            row = dict(case=name, n_orbits=n_orb, mutation=label,
                       Om_relerr=abs(Om_m - O) / abs(Om_m),
                       om_relerr=abs(om_m - W) / abs(om_m),
                       Mdot_corr_relerr=abs((Md_m - n) - (MD - n)) / abs(Md_m - n))
            mrows.append(row)
            print(f"    {label:<30} Om {row['Om_relerr']:9.3e}  "
                  f"argp {row['om_relerr']:9.3e}  Mcorr {row['Mdot_corr_relerr']:9.3e}")

    for nm, rs in (("j2_oracle2.csv", rows), ("j2_oracle2_mutations.csv", mrows)):
        with open(f"{OUT}/{nm}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader(); w.writerows(rs)
        print(f"  wrote {OUT}/{nm} ({len(rs)} rows)")


if __name__ == "__main__":
    main()
