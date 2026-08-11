"""RT3D-P7 — the phase_gap varpi fix under 3D sampling (mirrors the T3 red-team test).

The shipped C is `tgt_M += sat.omega - target.omega` (orbital.h:1291).  Both
design docs agree it must become `tgt_M += varpi_s - varpi_t`.  Tested here:
  (a) realized equinoctial Dlambda error vs the requested gap, and KS vs uniform;
  (b) the UNPATCHED version under 3D sampling (does it actually go inert?);
  (c) the same, measured in a frame-INVARIANT phase coordinate (target-plane
      gauge) -- because varpi = argp + RAAN is itself a broken-path angle.
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import (MU, R_EARTH, what_from_ie, rotate_about, coe2rv,
                         rv2coe, wrap_pi, pct)

OUT = "/Users/pete/space_training/web_data/results/ext_rt3d_phasegap.csv"


def ks_uniform(samples, lo, hi):
    s = sorted(samples)
    n = len(s)
    d = 0.0
    for k, x in enumerate(s):
        f = (x - lo) / (hi - lo)
        d = max(d, abs((k + 1) / n - f), abs(f - k / n))
    return d


def lam_target_gauge(r_s, v_s, r_t, v_t):
    a_t, e_t, i_t, O_t, w_t, nu_t = rv2coe(r_t, v_t)
    wh = what_from_ie(i_t, O_t)
    nx, ny = -wh[1], wh[0]
    nn = math.hypot(nx, ny)
    e1 = (nx / nn, ny / nn, 0.0) if nn > 1e-14 else (1.0, 0.0, 0.0)
    e3 = wh
    e2 = (e3[1] * e1[2] - e3[2] * e1[1], e3[2] * e1[0] - e3[0] * e1[2],
          e3[0] * e1[1] - e3[1] * e1[0])

    def to_t(v):
        return (sum(v[k] * e1[k] for k in range(3)),
                sum(v[k] * e2[k] for k in range(3)),
                sum(v[k] * e3[k] for k in range(3)))

    def lam(r, v):
        a, e, i, O, w, nu = rv2coe(r, v)
        if e < 1e-12:
            M = nu
        else:
            E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2),
                               math.sqrt(1 + e) * math.cos(nu / 2))
            M = E - e * math.sin(E)
        return M + w + O
    return wrap_pi(lam(to_t(r_s), to_t(v_s)) - lam(to_t(r_t), to_t(v_t)))


def main(n=20000):
    rows = []
    for di_deg in (0.0, 0.25, 1.0, 2.0):
        di = math.radians(di_deg)
        for patched in (True, False):
            rng = random.Random(8888)
            errs, errs_inv, realized = [], [], []
            for _ in range(n):
                a_t = R_EARTH + 550e3
                a_s = a_t + rng.uniform(-200e3, 200e3)
                e_t = rng.random() * 0.05
                e_s = rng.random() * 0.05
                # design gauge: target in the reference plane, chaser tilted
                i_t, O_t = 0.0, 0.0
                wh_t = what_from_ie(i_t, O_t)
                phi = rng.random() * 2 * math.pi
                wh_s = rotate_about(wh_t, (math.cos(phi), math.sin(phi), 0.0), di)
                i_s = math.atan2(math.hypot(wh_s[0], wh_s[1]), wh_s[2])
                O_s = math.atan2(wh_s[0], -wh_s[1]) % (2 * math.pi)
                w_t = rng.random() * 2 * math.pi
                w_s = rng.random() * 2 * math.pi
                M_s = rng.random() * 2 * math.pi
                gap = (rng.random() * 2 - 1) * math.pi
                # c_reset: tgt_M = sat.M + gap + <correction>
                corr = ((w_s + O_s) - (w_t + O_t)) if patched else (w_s - w_t)
                M_t = (M_s + gap + corr) % (2 * math.pi)

                def nu_of(M, e):
                    E = M
                    for _ in range(50):
                        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
                    return 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                                          math.sqrt(1 - e) * math.cos(E / 2))
                r_s, v_s = coe2rv(a_s, e_s, i_s, O_s, w_s, nu_of(M_s, e_s))
                r_t, v_t = coe2rv(a_t, e_t, i_t, O_t, w_t, nu_of(M_t, e_t))
                dl = wrap_pi((M_s + w_s + O_s) - (M_t + w_t + O_t))
                errs.append(abs(wrap_pi(dl - (-gap))))
                realized.append(dl)
                errs_inv.append(abs(wrap_pi(lam_target_gauge(r_s, v_s, r_t, v_t) - (-gap))))
            ks = ks_uniform(realized, -math.pi, math.pi)
            rows.append(dict(di_max_deg=di_deg, patched_to_varpi=patched,
                             realized_gap_err_p50_deg=round(math.degrees(pct(errs, .5)), 8),
                             realized_gap_err_max_deg=round(math.degrees(max(errs)), 8),
                             invariant_gauge_err_p50_deg=round(math.degrees(pct(errs_inv, .5)), 6),
                             invariant_gauge_err_p90_deg=round(math.degrees(pct(errs_inv, .9)), 6),
                             invariant_gauge_err_max_deg=round(math.degrees(max(errs_inv)), 6),
                             ks_vs_uniform=round(ks, 5),
                             verdict=("knob EXACT in lambda" if max(errs) < 1e-9
                                      else "KNOB INERT/BROKEN in lambda")))
            print(f"di={di_deg:5.2f} patched={str(patched):5s}: "
                  f"lambda-gap err p50={math.degrees(pct(errs,.5)):9.5f} "
                  f"max={math.degrees(max(errs)):10.5f} deg | KS={ks:.4f} | "
                  f"invariant-gauge err p50={math.degrees(pct(errs_inv,.5)):8.5f} "
                  f"p90={math.degrees(pct(errs_inv,.9)):8.5f} "
                  f"max={math.degrees(max(errs_inv)):9.5f} deg")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
