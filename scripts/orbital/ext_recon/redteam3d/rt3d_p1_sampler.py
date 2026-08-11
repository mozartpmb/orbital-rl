"""RT3D-P1 — attack the di_max relative-inclination sampler.

Two candidate samplers are on the table and the design docs disagree:

  S_A  3d_A section 4 (`c_reset` insertion, the code an implementer would paste):
       add an area-uniform disc of radius di_max to the *inclination vector*
       i_vec = (sin i cos O, sin i sin O), then i_s = asin(|i_vec|),
       O_s = atan2(iy, ix).

  S_C  3d_C section 4.4(a) (the prose fix): h_s = R(delta, n_hat) h_t with
       delta = di_max * sqrt(U) and n_hat uniform in the target plane.

Claim under attack (3d_C 4.4a): "p90 = 0.95*knob, max = knob exactly, 0.0% over".
Measured here against TRUE relative inclination = angle(h_s, h_t), across the
absolute target inclinations the ladder actually proposes (0-98 deg).

Also probes: (a) does S_A's asin() fold retrograde/high inclinations, (b) gauge
independence in RAAN_t, (c) implied plane-change dv vs the 478 m/s budget.
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import (MU, R_EARTH, BUDGET_DV, what_from_ie, ivec_from_ie,
                         plane_angle, rotate_about, pct)

OUT = "/Users/pete/space_training/web_data/results/ext_rt3d_sampler.csv"
N = 40000


def sampler_A(i_t, O_t, di_max, rng):
    """3d_A section 4 c_reset code, transcribed literally."""
    r_di = di_max * math.sqrt(rng.random())
    ph = rng.random() * 2.0 * math.pi
    ix = math.sin(i_t) * math.cos(O_t) + r_di * math.cos(ph)
    iy = math.sin(i_t) * math.sin(O_t) + r_di * math.sin(ph)
    si = math.hypot(ix, iy)
    if si > 1.0:
        si = 1.0
    inc = math.asin(si)
    raan = math.atan2(iy, ix) if si > 1e-12 else 0.0
    return inc, raan


def sampler_C(i_t, O_t, di_max, rng):
    """3d_C 4.4(a): exact rotation of h_t by delta about an axis in the plane."""
    w_t = what_from_ie(i_t, O_t)
    delta = di_max * math.sqrt(rng.random())
    phi = rng.random() * 2.0 * math.pi
    # any orthonormal pair spanning the target plane
    ref = (0.0, 0.0, 1.0) if abs(w_t[2]) < 0.9 else (1.0, 0.0, 0.0)
    ax = (w_t[1] * ref[2] - w_t[2] * ref[1],
          w_t[2] * ref[0] - w_t[0] * ref[2],
          w_t[0] * ref[1] - w_t[1] * ref[0])
    n = math.sqrt(sum(c * c for c in ax))
    u1 = tuple(c / n for c in ax)
    u2 = (w_t[1] * u1[2] - w_t[2] * u1[1],
          w_t[2] * u1[0] - w_t[0] * u1[2],
          w_t[0] * u1[1] - w_t[1] * u1[0])
    axis = tuple(u1[i] * math.cos(phi) + u2[i] * math.sin(phi) for i in range(3))
    w_s = rotate_about(w_t, axis, delta)
    return w_s


def main():
    rows = []
    v_leo = math.sqrt(MU / (R_EARTH + 550e3))
    for di_deg in (0.05, 0.25, 0.75, 1.0, 2.0):
        di = math.radians(di_deg)
        for i_t_deg in (0.0, 5.0, 28.5, 51.6, 80.0, 89.0, 90.0, 98.0):
            i_t = math.radians(i_t_deg)
            for name, fn in (("S_A_ivec_disc", sampler_A), ("S_C_rotation", sampler_C)):
                rng = random.Random(20260811)
                real, incs = [], []
                for _ in range(N):
                    O_t = rng.random() * 2.0 * math.pi   # gauge: RAAN_t random
                    if name == "S_A_ivec_disc":
                        i_s, O_s = fn(i_t, O_t, di, rng)
                        w_s = what_from_ie(i_s, O_s)
                    else:
                        w_s = fn(i_t, O_t, di, rng)
                        i_s = math.atan2(math.hypot(w_s[0], w_s[1]), w_s[2])
                    w_t = what_from_ie(i_t, O_t)
                    real.append(plane_angle(w_s, w_t))
                    incs.append(i_s)
                rd = [math.degrees(x) for x in real]
                over = sum(1 for x in rd if x > di_deg * (1 + 1e-9)) / float(N)
                dv90 = 2.0 * v_leo * math.sin(pct(real, 0.90) / 2.0)
                dvmax = 2.0 * v_leo * math.sin(max(real) / 2.0)
                rows.append(dict(
                    sampler=name, di_max_deg=di_deg, i_t_deg=i_t_deg,
                    p50_deg=round(pct(rd, .50), 6), p90_deg=round(pct(rd, .90), 6),
                    max_deg=round(max(rd), 6),
                    ratio_max_over_knob=round(max(rd) / di_deg, 4),
                    ratio_p90_over_knob=round(pct(rd, .90) / di_deg, 4),
                    frac_over_knob=round(over, 5),
                    inc_s_max_deg=round(math.degrees(max(incs)), 4),
                    dv_p90_ms=round(dv90, 2), dv_max_ms=round(dvmax, 2),
                    dv_max_frac_budget=round(dvmax / BUDGET_DV, 4)))
                print(f"{name:14s} di={di_deg:5.2f} i_t={i_t_deg:5.1f}  "
                      f"p50={pct(rd,.5):8.4f} p90={pct(rd,.9):8.4f} "
                      f"max={max(rd):9.4f} (x{max(rd)/di_deg:7.2f}) "
                      f"over={over:6.3%} dv_max={dvmax:8.1f} "
                      f"({dvmax/BUDGET_DV:5.2f}x budget)")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
