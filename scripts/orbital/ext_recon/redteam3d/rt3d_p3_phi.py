"""RT3D-P3 — attack the shaping_mode=2 potential.

Sub-attacks
  A1  min(1,.) saturation at each rung's REAL init distribution (design gate: <5%).
      Reproduces c_reset's sampling (a-band, da_max window, de_max disc,
      e_max_* draws) plus the S_C plane disc.
  A2  the "2.5% de-vector over-credit" claim.  The 3-vector |de| carries an
      O(e*di) out-of-plane part that the PLANE burn also removes -> Phi
      double-counts it.  Measured as (dv_in_3D - dv_in_after_plane_fix) / dv_pl
      across the ladder's real e range (the report measured e=0.05 only; the
      ladder goes to e<=0.50).
  A3  L1 vs hypot given the ACTUAL action table: every action in ACTION_DV is
      single-axis, so a combined-direction impulse is not executable.  Which
      combiner equals the achievable cost-to-go?
  A4  normal-sign dithering adversary (+n, -n, +n, ...) and the gamma<1
      telescoping leak bound.
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import (MU, R_EARTH, BUDGET_DV, what_from_ie, plane_angle,
                         rotate_about, coe2rv, rv2coe, evec_from_elements, pct)

OUT_SAT = "/Users/pete/space_training/web_data/results/ext_rt3d_phi_saturation.csv"
OUT_OC = "/Users/pete/space_training/web_data/results/ext_rt3d_phi_overcredit.csv"
OUT_ADV = "/Users/pete/space_training/web_data/results/ext_rt3d_phi_adversary.csv"

# rung name, a_min, a_max, e_max_t, e_max_s, de_max, da_max, same_orbit, di_deg, dv_ref
RUNGS = [
    ("X0_coplanar_anchor", 6.871e6, 7.171e6, 0.0, 0.0, -1, 0, 1, 0.00, 700.),
    ("X1_di0.05", 6.871e6, 7.171e6, 0.0, 0.0, -1, 0, 1, 0.05, 700.),
    ("X2_di0.25", 6.871e6, 7.171e6, 0.0, 0.0, -1, 0, 1, 0.25, 700.),
    ("X3_LEO_di1.0", 6.671e6, 7.171e6, 0.05, 0.05, -1, 0, 0, 1.00, 700.),
    ("X3b_LEO_di0.75_3dB", 6.671e6, 7.171e6, 0.05, 0.05, -1, 0, 0, 0.75, 700.),
    ("X4_wide_di1.0", 6.671e6, 14.371e6, 0.30, 0.30, 0.08, 600e3, 0, 1.00, 700.),
    ("X5_MEO_di2.0", 6.671e6, 26.571e6, 0.50, 0.50, 0.10, 1000e3, 0, 2.00, 900.),
    ("X5b_MEO_di2.0_dvref700", 6.671e6, 26.571e6, 0.50, 0.50, 0.10, 1000e3, 0, 2.00, 700.),
]


def sample_init(rng, cfg):
    _, amin, amax, emt, ems, demax, damax, same, di_deg, dvref = cfg
    for _ in range(256):                      # c_reset's valid_init_only loop
        a_s = amin + rng.random() * (amax - amin)
        if same:
            a_t = a_s
        else:
            lo, hi = amin, amax
            if damax > 0:
                da = max(damax, 200e3)
                lo2, hi2 = max(amin, a_s - da), min(amax, a_s + da)
                if hi2 - lo2 >= 150e3:
                    lo, hi = lo2, hi2
            for _ in range(200):
                a_t = lo + rng.random() * (hi - lo)
                if abs(a_t - a_s) >= 50e3:
                    break
        e_t = rng.random() * emt if emt > 0 else 0.0
        w_t = rng.random() * 2 * math.pi if e_t > 0 else 0.0
        if same:
            e_s, w_s = e_t, w_t
        elif demax >= 0:
            r_de = demax * math.sqrt(rng.random())
            ph = rng.random() * 2 * math.pi
            esx = e_t * math.cos(w_t) + r_de * math.cos(ph)
            esy = e_t * math.sin(w_t) + r_de * math.sin(ph)
            e_s = math.hypot(esx, esy)
            w_s = math.atan2(esy, esx)
        else:
            e_s = rng.random() * ems if ems > 0 else 0.0
            w_s = rng.random() * 2 * math.pi if e_s > 0 else 0.0
        if a_s * (1 - e_s) >= R_EARTH + 200e3 and a_t * (1 - e_t) >= R_EARTH + 200e3:
            return a_s, e_s, w_s, a_t, e_t, w_t
    return a_s, e_s, w_s, a_t, e_t, w_t


def plane_pair(rng, di_max):
    """Target plane at (i_t, O_t) random; chaser = S_C exact rotation by delta."""
    i_t = rng.random() * math.radians(98.0)
    O_t = rng.random() * 2 * math.pi
    w_t = what_from_ie(i_t, O_t)
    if di_max <= 0.0:
        return i_t, O_t, i_t, O_t, w_t, w_t, 0.0
    delta = di_max * math.sqrt(rng.random())
    phi = rng.random() * 2 * math.pi
    ref = (0., 0., 1.) if abs(w_t[2]) < 0.9 else (1., 0., 0.)
    ax = (w_t[1] * ref[2] - w_t[2] * ref[1], w_t[2] * ref[0] - w_t[0] * ref[2],
          w_t[0] * ref[1] - w_t[1] * ref[0])
    n = math.sqrt(sum(c * c for c in ax))
    u1 = tuple(c / n for c in ax)
    u2 = (w_t[1] * u1[2] - w_t[2] * u1[1], w_t[2] * u1[0] - w_t[0] * u1[2],
          w_t[0] * u1[1] - w_t[1] * u1[0])
    axis = tuple(u1[i] * math.cos(phi) + u2[i] * math.sin(phi) for i in range(3))
    w_s = rotate_about(w_t, axis, delta)
    i_s = math.atan2(math.hypot(w_s[0], w_s[1]), w_s[2])
    O_s = math.atan2(w_s[0], -w_s[1]) % (2 * math.pi)
    return i_t, O_t, i_s, O_s, w_t, w_s, delta


def run_saturation(n=20000):
    rows = []
    for cfg in RUNGS:
        name = cfg[0]
        di = math.radians(cfg[8])
        dvref = cfg[9]
        rng = random.Random(31415)
        l1, hy, dvpl, dvin = [], [], [], []
        for _ in range(n):
            a_s, e_s, w_s0, a_t, e_t, w_t0 = sample_init(rng, cfg)
            i_t, O_t, i_s, O_s, wh_t, wh_s, delta = plane_pair(rng, di)
            v_t = math.sqrt(MU / a_t)
            es3 = evec_from_elements(e_s, i_s, O_s, w_s0)
            et3 = evec_from_elements(e_t, i_t, O_t, w_t0)
            de = math.sqrt(sum((es3[k] - et3[k]) ** 2 for k in range(3)))
            da = (a_s - a_t) / a_t
            dv_in = 0.5 * v_t * math.hypot(da, de)
            dv_pl = v_t * math.sqrt(sum((wh_s[k] - wh_t[k]) ** 2 for k in range(3)))
            l1.append(dv_in + dv_pl)
            hy.append(math.hypot(dv_in, dv_pl))
            dvpl.append(dv_pl)
            dvin.append(dv_in)
        fl1 = sum(1 for x in l1 if x > dvref) / n
        fhy = sum(1 for x in hy if x > dvref) / n
        rows.append(dict(rung=name, di_max_deg=cfg[8], dv_ref=dvref,
                         dv_in_p50=round(pct(dvin, .5), 1), dv_in_p90=round(pct(dvin, .9), 1),
                         dv_pl_p50=round(pct(dvpl, .5), 1), dv_pl_p90=round(pct(dvpl, .9), 1),
                         dv3_L1_p50=round(pct(l1, .5), 1), dv3_L1_p90=round(pct(l1, .9), 1),
                         dv3_L1_max=round(max(l1), 1),
                         frac_sat_L1=round(fl1, 5), frac_sat_hypot=round(fhy, 5),
                         gate_lt_5pct=("PASS" if fl1 < 0.05 else "FAIL"),
                         dv3_over_budget_frac=round(
                             sum(1 for x in l1 if x > BUDGET_DV) / n, 5)))
        print(f"{name:24s} dvref={dvref:4.0f} dv_in p90={pct(dvin,.9):6.1f} "
              f"dv_pl p90={pct(dvpl,.9):6.1f}  L1 p90={pct(l1,.9):7.1f} "
              f"max={max(l1):8.1f}  sat_L1={fl1:7.3%} sat_hypot={fhy:7.3%} "
              f"{'FAIL' if fl1>=0.05 else 'pass'}")
    with open(OUT_SAT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_SAT, "\n")


def run_overcredit(n=8000):
    """dv_in computed on the 3-vector counts an O(e*di) term the plane burn removes."""
    rows = []
    for e in (0.0, 0.02, 0.05, 0.10, 0.15, 0.30, 0.50):
        for di_deg in (0.25, 0.75, 1.0, 2.0):
            di = math.radians(di_deg)
            rng = random.Random(777)
            ratios, abs_ms = [], []
            for _ in range(n):
                a = R_EARTH + 550e3
                v_t = math.sqrt(MU / a)
                i_t, O_t, i_s, O_s, wh_t, wh_s, delta = plane_pair(rng, di)
                w_t0 = rng.random() * 2 * math.pi
                # identical in-plane geometry: same e, same argument of periapsis
                # measured from each orbit's own node -> zero TRUE in-plane error
                et3 = evec_from_elements(e, i_t, O_t, w_t0)
                # chaser: same e, argp chosen so the apsidal line is the rotation
                # of the target's about the relative node -> in-plane-identical
                nx = wh_t[1] * wh_s[2] - wh_t[2] * wh_s[1]
                ny = wh_t[2] * wh_s[0] - wh_t[0] * wh_s[2]
                nz = wh_t[0] * wh_s[1] - wh_t[1] * wh_s[0]
                nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                if nn < 1e-14:
                    continue
                axis = (nx / nn, ny / nn, nz / nn)
                es3 = rotate_about(et3, axis, delta)
                de3 = math.sqrt(sum((es3[k] - et3[k]) ** 2 for k in range(3)))
                dv_in3 = 0.5 * v_t * de3        # da = 0 by construction
                dv_pl = v_t * math.sqrt(sum((wh_s[k] - wh_t[k]) ** 2 for k in range(3)))
                if dv_pl > 0:
                    ratios.append(dv_in3 / dv_pl)
                abs_ms.append(dv_in3)
            rows.append(dict(e=e, di_max_deg=di_deg,
                             overcredit_frac_p50=round(pct(ratios, .5), 5),
                             overcredit_frac_p90=round(pct(ratios, .9), 5),
                             overcredit_frac_max=round(max(ratios), 5),
                             phantom_dv_p90_ms=round(pct(abs_ms, .9), 2)))
            print(f"e={e:4.2f} di={di_deg:4.2f}  phantom in-plane cost = "
                  f"{pct(ratios,.5):6.2%} (p90 {pct(ratios,.9):6.2%}, "
                  f"max {max(ratios):6.2%}) of the plane leg; "
                  f"{pct(abs_ms,.9):6.2f} m/s p90")
    with open(OUT_OC, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_OC, "\n")


def run_adversary():
    """A3 combiner-vs-action-space + A4 dithering / gamma<1 leak."""
    rows = []
    v = math.sqrt(MU / (R_EARTH + 550e3))
    # --- A3: what does a SEQUENCE of single-axis impulses actually cost? ---
    print("A3 combiner vs the real (single-axis) action table:")
    for di_deg in (0.25, 0.75, 1.0, 2.0):
        P = 2 * v * math.sin(math.radians(di_deg) / 2.0)
        for D in (25.0, 50.0, 100.0, 200.0, 300.0):
            seq = D + P                              # two separate impulses
            comb = math.hypot(D, P)                  # ONE combined-direction impulse
            rows.append(dict(test="A3_combiner", di_max_deg=di_deg, in_plane_dv=D,
                             plane_dv=round(P, 2), cost_sequential=round(seq, 2),
                             cost_combined_1impulse=round(comb, 2),
                             L1_err_vs_achievable=round(seq / seq - 1, 6),
                             hypot_err_vs_achievable=round(comb / seq - 1, 6)))
        print(f"  di={di_deg:4.2f} P={P:6.1f}: at D=100 sequential={100+P:6.1f} "
              f"combined={math.hypot(100,P):6.1f} -> hypot under-states the "
              f"executable cost by {1-math.hypot(100,P)/(100+P):5.1%}")
    # --- A4: normal-sign dithering ---
    print("\nA4 normal-sign dithering (+q,-q,+q,... at LEO):")
    for q in (1.0, 10.0, 25.0):
        # each normal burn adds (q/v)^2 of relative a and e (second-order tax);
        # the plane term returns to ~0 after the pair only if the two burns are
        # at the same point -- they are one sub-step apart (3.9 deg at LEO).
        du = math.radians(3.90)
        # net plane rotation of +q then -q one sub-step later:
        w0 = (0., 0., 1.)
        # rotate by q/v about node axis, advance du, rotate back by -q/v
        a1 = (1., 0., 0.)
        w1 = rotate_about(w0, a1, q / v)
        a2 = (math.cos(du), math.sin(du), 0.)
        w2 = rotate_about(w1, a2, -q / v)
        residual = plane_angle(w2, w0)
        dv_pl_res = v * 2 * math.sin(residual / 2)
        tax_a = (q / v) ** 2                     # d(a)/a per burn, second order
        dv_in_tax = 2 * (0.5 * v * tax_a)        # two burns
        wm_over_ref = 0.35 / 300.0
        dPhi_cycle = -wm_over_ref * (dv_pl_res + dv_in_tax)
        rows.append(dict(test="A4_dither", di_max_deg="", in_plane_dv=2 * q,
                         plane_dv=round(dv_pl_res, 4),
                         cost_sequential=round(2 * q, 2),
                         cost_combined_1impulse=round(dv_in_tax, 4),
                         L1_err_vs_achievable=round(dPhi_cycle, 8),
                         hypot_err_vs_achievable=""))
        print(f"  q={q:5.1f} m/s: fuel spent {2*q:5.1f} m/s, residual plane "
              f"{math.degrees(residual)*3600:7.2f} arcsec ({dv_pl_res:.4f} m/s), "
              f"in-plane tax {dv_in_tax:.4f} m/s -> net dPhi per cycle "
              f"{dPhi_cycle:+.3e} (STRICTLY NEGATIVE = unfarmable)")
    # --- A4b: the gamma<1 leak on a gamma_shape=1 (non-PBRS) delta ---
    print("\nA4b gamma<1 leak bound on the shape_gamma=1 delta:")
    for g in (0.995, 0.99):
        for rng_phi in (1.817, 2.05):
            # sum_k g^k (Phi_{k+1}-Phi_k) = g^{N-1}Phi_N - Phi_0
            #                               + (1/g - 1) sum_{k=1}^{N-1} g^k Phi_k
            # Phi <= 0 so the correction term is <= 0: the deviation from exact
            # telescoping is a PENALTY for dawdling, never an income.
            bound_max = rng_phi          # = -Phi_0 at worst
            rows.append(dict(test="A4b_gamma_leak", di_max_deg="", in_plane_dv=g,
                             plane_dv=rng_phi, cost_sequential=round(bound_max, 4),
                             cost_combined_1impulse="", L1_err_vs_achievable="",
                             hypot_err_vs_achievable="sign of correction = NEGATIVE"))
            print(f"  gamma={g}: max discounted shaping return = -Phi_0 <= "
                  f"{bound_max:.3f}; correction term (1/g-1)*sum g^k Phi_k <= 0 "
                  f"(Phi<=0) -> no oscillation income. terminal/shape "
                  f"= {10.0/bound_max:.2f}:1")
    with open(OUT_ADV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_ADV)


if __name__ == "__main__":
    run_saturation()
    run_overcredit()
    run_adversary()
