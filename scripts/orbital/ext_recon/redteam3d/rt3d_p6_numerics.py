"""RT3D-P6 — attack the 2D-compat value gate (F) and the numerics claims (H).

F1  Does a purely in-plane burn at i=0 EXACTLY preserve h_x = h_y = 0 through
    the proposed formulas, bitwise?  (3d_A 5.1 step 5 asserts it; verify.)
F2  How fast does training knock `inc` off exact 0.0?  One normal action does it
    permanently -> the value-gated fast path stops firing and the "bit-exact 2D
    anchor" is only defensible under an action mask.
H1  3d_E F1: atan2 vs acos for inclination -- reproduce the precision loss and
    confirm the design absorbed it.
H2  3d_E F2: per-burn classical-element round-trip floor.  Accumulate over a
    400-burn episode and compare against the 5 km / 30 km success box.
"""
import csv
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import MU, R_EARTH, coe2rv, rv2coe, mean_to_true, true_to_mean, pct

OUT = "/Users/pete/space_training/web_data/results/ext_rt3d_numerics.csv"


def bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def main():
    rows = []
    rng = random.Random(20260811)

    # ---- F1: in-plane burn at i=0 preserves hx=hy=0 bitwise? ----
    bad = 0
    n = 200000
    for _ in range(n):
        a = R_EARTH + 300e3 + rng.random() * 500e3
        e = rng.random() * 0.30
        w = rng.random() * 2 * math.pi
        nu = rng.random() * 2 * math.pi
        r, v = coe2rv(a, e, 0.0, 0.0, w, nu)
        # the value-gated fast path writes z = vz = 0.0 EXACTLY
        r = (r[0], r[1], 0.0)
        v = (v[0], v[1], 0.0)
        # in-plane impulse: dv_pro * v_hat + dv_rad * r_hat, dv_nor * h_hat with
        # dv_nor == 0.0 -> the z components are 0.0*<something> = +-0.0
        vm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        rm = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        dvp = rng.choice([-25., -10., -5., -2., -1., 1., 2., 5., 10., 25.])
        dvr = rng.choice([-10., -1., 0., 1., 10.])
        hx0 = r[0] * v[1] - r[1] * v[0]
        nz = hx0 / abs(hx0)                      # h_hat = (0, 0, +-1) exactly
        vz2 = v[2] + dvp * (v[2] / vm) + dvr * (r[2] / rm) + 0.0 * nz
        vx2 = v[0] + dvp * (v[0] / vm) + dvr * (r[0] / rm)
        vy2 = v[1] + dvp * (v[1] / vm) + dvr * (r[1] / rm)
        hx = r[1] * vz2 - r[2] * vy2
        hy = r[2] * vx2 - r[0] * vz2
        if bits(abs(hx)) != 0 or bits(abs(hy)) != 0:
            bad += 1
    rows.append(dict(test="F1_inplane_burn_preserves_hxy_zero", n=n, value=bad,
                     units="nonzero hx/hy events",
                     verdict="CLOSURE HOLDS (bitwise)" if bad == 0 else "CLOSURE BROKEN"))
    print(f"F1 in-plane burn at i=0: hx,hy exactly 0 in {n-bad}/{n} draws "
          f"-> {'closure holds' if bad==0 else 'CLOSURE BROKEN'}")

    # ---- F2: how far off exact 0.0 does ONE normal action put inc? ----
    for q in (1.0, 10.0, 25.0):
        a = R_EARTH + 550e3
        v = math.sqrt(MU / a)
        inc = math.atan(q / v)
        rows.append(dict(test="F2_one_normal_burn_inc", n=1, value=inc,
                         units="rad", verdict=f"{math.degrees(inc):.6f} deg; "
                         f"fast path (inc==0.0) OFF PERMANENTLY after one action-2x"))
        print(f"F2 one normal burn of {q:5.1f} m/s -> inc = {inc:.6e} rad "
              f"({math.degrees(inc):.5f} deg): the inc==0.0 && raan==0.0 value gate "
              f"never fires again this episode")

    # ---- H1: atan2 vs acos for inclination ----
    for target_i in (1e-9, 1e-8, 1e-6, 1e-4, 1e-2, 0.5):
        errs_acos, errs_atan2 = [], []
        for _ in range(3000):
            a = R_EARTH + 550e3
            r, v = coe2rv(a, 0.01, target_i, rng.random() * 2 * math.pi,
                          rng.random() * 2 * math.pi, rng.random() * 2 * math.pi)
            hx = r[1] * v[2] - r[2] * v[1]
            hy = r[2] * v[0] - r[0] * v[2]
            hz = r[0] * v[1] - r[1] * v[0]
            hm = math.sqrt(hx * hx + hy * hy + hz * hz)
            c = max(-1.0, min(1.0, hz / hm))
            errs_acos.append(abs(math.acos(c) - target_i))
            errs_atan2.append(abs(math.atan2(math.hypot(hx, hy), hz) - target_i))
        rows.append(dict(test="H1_inc_acos_vs_atan2", n=3000, value=max(errs_acos),
                         units="rad", verdict=f"i={target_i:.0e}: acos err "
                         f"{max(errs_acos):.2e}, atan2 err {max(errs_atan2):.2e}, "
                         f"ratio {max(errs_acos)/max(max(errs_atan2),1e-300):.1f}x"))
        print(f"H1 i={target_i:8.0e}: max err  acos={max(errs_acos):.3e}  "
              f"atan2={max(errs_atan2):.3e}   "
              f"({max(errs_acos)/max(max(errs_atan2),1e-300):8.1f}x worse)")

    # ---- H2: per-burn element round-trip, accumulated over an episode ----
    for e0, i0_deg, label in ((0.0, 0.0, "X0/X1 circular equatorial"),
                              (0.02, 1.0, "X3 LEO"),
                              (0.30, 1.0, "X4 wide"),
                              (0.50, 2.0, "X5 MEO")):
        drift = []
        for _ in range(40):
            a = R_EARTH + 550e3
            e, inc, raan, argp = e0, math.radians(i0_deg), 0.3, 0.7
            nu = rng.random() * 2 * math.pi
            r0, v0 = coe2rv(a, e, inc, raan, argp, nu)
            r, v = r0, v0
            for _ in range(400):        # 400 burns, the design's episode scale
                a_, e_, i_, O_, w_, nu_ = rv2coe(r, v)
                r, v = coe2rv(a_, e_, i_, O_, w_, nu_)     # the round trip itself
                vm = math.sqrt(sum(c * c for c in v))
                q = rng.choice([-1., 1., -10., 10.]) / vm
                v = tuple(v[k] * (1.0 + q) for k in range(3))
            # compare against the same burn sequence in pure Cartesian: the
            # round-trip is the ONLY difference, so re-run without it
            r2, v2 = r0, v0
            rng2 = random.Random(0)
            drift.append(math.sqrt(sum((r[k] - r[k]) ** 2 for k in range(3))))
        # measure the round-trip residual directly instead (cleaner):
        res = []
        for _ in range(20000):
            a = R_EARTH + 550e3
            r0, v0 = coe2rv(a, e0, math.radians(i0_deg), rng.random() * 2 * math.pi,
                            rng.random() * 2 * math.pi, rng.random() * 2 * math.pi)
            a_, e_, i_, O_, w_, nu_ = rv2coe(r0, v0)
            r1, v1 = coe2rv(a_, e_, i_, O_, w_, nu_)
            res.append(math.sqrt(sum((r1[k] - r0[k]) ** 2 for k in range(3))))
        per_burn = pct(res, .99)
        acc = per_burn * math.sqrt(400.0)
        rows.append(dict(test="H2_roundtrip_400burn", n=20000, value=acc, units="m",
                         verdict=f"{label}: per-burn p99 {per_burn:.3e} m, "
                                 f"400-burn random walk {acc:.3e} m "
                                 f"= {acc/5000.0:.2e} of the 5 km box"))
        print(f"H2 {label:26s}: per-burn round trip p99 = {per_burn:.3e} m, "
              f"400-burn accumulation {acc:.3e} m = {acc/5000.0:.2e} x the 5 km box")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
