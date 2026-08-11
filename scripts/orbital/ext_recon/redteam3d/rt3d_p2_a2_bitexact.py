"""RT3D-P2 — attack anchor A2: "Phi(mode 2) at di=0 == Phi(mode 1) bit-exactly".

mode 1 is the SHIPPED C code (orbital.h compute_phi, verbatim):
    de       = sqrt((e_sx-e_tx)^2 + (e_sy-e_ty)^2)        e_* = e*cos/sin(omega)
    da_rel   = (a_s - a_t)/a_t
    v_t      = sqrt(MU/a_t)
    dv_match = 0.5*v_t*sqrt(da_rel*da_rel + de*de)        <-- sqrt(x*x+y*y)
    Phi      = -(W_l*|dlam|/pi + W_m*min(1, dv_match/dv_ref))

mode 2 as written in 3d_C section 3:
    dv_in    = 0.5*v_t*hypot(da_rel, |de_vec3|)           <-- hypot(), and a
                                                              3-vector e built
                                                              from (v x h)/mu - r_hat
    dv_pl    = v_t*|h_s - h_t|
    Phi      = -(W_l*|dlam|/pi + W_m*min(1, (dv_in+dv_pl)/dv_ref))

Four independent candidate breaks are separated here:
  X1  hypot(x,y) vs sqrt(x*x+y*y)              (libm vs source arithmetic)
  X2  e-vector from Cartesian vs from elements  (different FP path, same math)
  X3  the weight rule W_m = 0.817 @ dv_ref 700 vs mode 1's 0.35 @ 300
      (0.817/700 = 1.167143e-3 vs 0.35/300 = 1.166667e-3)
  X4  float32 survival: rewards are float32, so "bit-exact" may still hold
      after the cast even if the doubles differ.
"""
import csv
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import MU, R_EARTH, coe2rv, evec_from_rv, evec_from_elements, wrap_pi

OUT = "/Users/pete/space_training/web_data/results/ext_rt3d_a2_bitexact.csv"
N = 200000


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def phi_mode1(a_s, e_s, w_s, M_s, a_t, e_t, w_t, M_t, W_l, W_m, dvref):
    dlam = wrap_pi((M_s + w_s) - (M_t + w_t))
    e_sx, e_sy = e_s * math.cos(w_s), e_s * math.sin(w_s)
    e_tx, e_ty = e_t * math.cos(w_t), e_t * math.sin(w_t)
    dex, dey = e_sx - e_tx, e_sy - e_ty
    de = math.sqrt(dex * dex + dey * dey)
    da_rel = (a_s - a_t) / a_t
    v_t = math.sqrt(MU / a_t)
    dv = 0.5 * v_t * math.sqrt(da_rel * da_rel + de * de)
    m = dv / dvref
    if m > 1.0:
        m = 1.0
    return -(W_l * abs(dlam) / math.pi + W_m * m)


def phi_mode2(a_s, e_s, w_s, M_s, a_t, e_t, w_t, M_t, W_l, W_m, dvref,
              use_hypot=True, evec_from_cart=True):
    # inclination = RAAN = 0 exactly on both bodies (the A2 condition)
    dlam = wrap_pi((M_s + w_s + 0.0) - (M_t + w_t + 0.0))
    if evec_from_cart:
        nu_s = _nu(M_s, e_s)
        nu_t = _nu(M_t, e_t)
        r_s, v_s = coe2rv(a_s, e_s, 0.0, 0.0, w_s, nu_s)
        r_t, v_t_ = coe2rv(a_t, e_t, 0.0, 0.0, w_t, nu_t)
        es3 = evec_from_rv(r_s, v_s)
        et3 = evec_from_rv(r_t, v_t_)
    else:
        es3 = evec_from_elements(e_s, 0.0, 0.0, w_s)
        et3 = evec_from_elements(e_t, 0.0, 0.0, w_t)
    dx, dy, dz = es3[0] - et3[0], es3[1] - et3[1], es3[2] - et3[2]
    de = math.sqrt(dx * dx + dy * dy + dz * dz)
    da_rel = (a_s - a_t) / a_t
    vt = math.sqrt(MU / a_t)
    if use_hypot:
        dv_in = 0.5 * vt * math.hypot(da_rel, de)
    else:
        dv_in = 0.5 * vt * math.sqrt(da_rel * da_rel + de * de)
    # h_hat identical (0,0,1) for both at i=0 -> dv_pl == 0.0 exactly
    dv_pl = vt * 0.0
    m = (dv_in + dv_pl) / dvref
    if m > 1.0:
        m = 1.0
    return -(W_l * abs(dlam) / math.pi + W_m * m)


_E_CACHE = {}


def _nu(M, e):
    E = M
    for _ in range(40):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    return 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                            math.sqrt(1 - e) * math.cos(E / 2))


def main():
    rng = random.Random(4242)
    variants = [
        ("V0_spec_hypot_cartesian_evec", dict(use_hypot=True, evec_from_cart=True)),
        ("V1_sqrt_cartesian_evec",       dict(use_hypot=False, evec_from_cart=True)),
        ("V2_hypot_element_evec",        dict(use_hypot=True, evec_from_cart=False)),
        ("V3_sqrt_element_evec_EXACT",   dict(use_hypot=False, evec_from_cart=False)),
    ]
    rows = []
    # ---- part 1: same weights both modes, isolate the arithmetic path ----
    draws = []
    for _ in range(N):
        a_s = R_EARTH + 300e3 + rng.random() * 500e3
        a_t = R_EARTH + 300e3 + rng.random() * 500e3
        e_s = rng.random() * 0.05
        e_t = rng.random() * 0.05
        draws.append((a_s, e_s, rng.random() * 2 * math.pi, rng.random() * 2 * math.pi,
                      a_t, e_t, rng.random() * 2 * math.pi, rng.random() * 2 * math.pi))
    for name, kw in variants:
        nd = nf = 0
        maxd = 0.0
        n_use = N if not kw["evec_from_cart"] else 20000   # cartesian path is slow
        for d in draws[:n_use]:
            p1 = phi_mode1(*d, 1.0, 0.35, 300.0)
            p2 = phi_mode2(*d, 1.0, 0.35, 300.0, **kw)
            if p1 != p2:
                nd += 1
                maxd = max(maxd, abs(p1 - p2))
            if f32(p1) != f32(p2):
                nf += 1
        rows.append(dict(test="A2_same_weights", variant=name, n=n_use,
                         frac_double_mismatch=round(nd / n_use, 6),
                         frac_float32_mismatch=round(nf / n_use, 6),
                         max_abs_dPhi=maxd,
                         verdict=("BIT-EXACT" if nd == 0 else
                                  ("f32-exact" if nf == 0 else "NOT EXACT"))))
        print(f"{name:32s} n={n_use:6d}  double-mismatch {nd/n_use:8.4%}  "
              f"f32-mismatch {nf/n_use:8.4%}  max|dPhi| {maxd:.3e}")

    # ---- part 1b: the OPERATIONALLY binding form — reward = f32(dPhi) ----
    # A one-step coast delta is what the trainer actually sees. Sweep the delta
    # magnitude down to the regime where a 7e-15 Phi error can survive the cast.
    for name, kw in variants:
        n_use = 20000 if kw["evec_from_cart"] else 60000
        nf = 0
        maxrel = 0.0
        for d in draws[:n_use]:
            a_s, e_s, w_s, M_s, a_t, e_t, w_t, M_t = d
            # advance the chaser by one 60 s sub-step -> a realistic coast delta
            n_s = math.sqrt(MU / a_s ** 3)
            n_t = math.sqrt(MU / a_t ** 3)
            d2 = (a_s, e_s, w_s, M_s + n_s * 60.0, a_t, e_t, w_t, M_t + n_t * 60.0)
            r1 = f32(phi_mode1(*d2, 1.0, 0.35, 300.) - phi_mode1(*d, 1.0, 0.35, 300.))
            r2 = f32(phi_mode2(*d2, 1.0, 0.35, 300., **kw)
                     - phi_mode2(*d, 1.0, 0.35, 300., **kw))
            if r1 != r2:
                nf += 1
                if r1 != 0.0:
                    maxrel = max(maxrel, abs(r1 - r2) / abs(r1))
        rows.append(dict(test="A2_reward_f32", variant=name, n=n_use,
                         frac_double_mismatch="", frac_float32_mismatch=round(nf / n_use, 6),
                         max_abs_dPhi=maxrel,
                         verdict=("reward f32-exact" if nf == 0 else "REWARD DIFFERS")))
        print(f"{name:32s} coast-step reward f32 mismatch {nf/n_use:8.4%} "
              f"max rel {maxrel:.3e}")

    # ---- part 2: the weight rule ----
    r_m1 = 0.35 / 300.0
    for wm, dvr in ((0.817, 700.0), (0.35 * 700.0 / 300.0, 700.0), (1.05, 900.0),
                    (0.35 * 900.0 / 300.0, 900.0)):
        r = wm / dvr
        rows.append(dict(test="weight_rule", variant=f"W_m={wm:.6f}@dv_ref={dvr:.0f}",
                         n=0, frac_double_mismatch="", frac_float32_mismatch="",
                         max_abs_dPhi=abs(r - r_m1) / r_m1,
                         verdict=f"per_ms={r:.9e} rel_err_vs_mode1={abs(r-r_m1)/r_m1:.3e} "
                                 f"Phi_range={1.0+wm:.4f} terminal_ratio={10.0/(1.0+wm):.3f}"))
        print(f"W_m={wm:9.6f} dv_ref={dvr:5.0f} -> {r:.9e} per m/s "
              f"(mode1 {r_m1:.9e}, rel err {abs(r-r_m1)/r_m1:.3e})  "
              f"Phi_range={1.0+wm:.4f}  terminal/range={10.0/(1.0+wm):.3f}"
              f"{'  <-- FAILS the >=5:1 gate' if 10.0/(1.0+wm) < 5.0 else ''}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
