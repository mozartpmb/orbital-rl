"""RED-TEAM D — the float32 decode floor, measured in METRES and MILLIRADIANS.

NAV-H claims "float32 obs quantization injects <=1 m of pseudo-truth error,
50x below sigma_rho=50 m". True for the RANGE channel. But under bearings-only
the sensor is a BEARING synthesized FROM the decoded pseudo-truth, so the
relevant comparison is not 1 m vs 50 m, it is

    (decode error transverse to the LOS) / rho     vs     sigma_beta = 1 mrad

which blows up as rho shrinks. This probe measures the exact quantity, at every
separation the shipped envelopes contain, by round-tripping f64 truth ->
float32 observation -> `recover_states_t3` -> Cartesian.

No C env needed: build_obs/recover_states are pure functions of the elements.
"""
import csv
import math
import os
import sys

import numpy as np

NAV = "/Users/pete/space_training/scripts/orbital/nav"
sys.path.insert(0, NAV)

import orbital_math as om              # noqa: E402
import eval_relnav as ev               # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_rtnav_decode_floor.csv"
R_E = om.R_EARTH
ALT, LV = 1.6e6, 6.371e6


def elements(a, e, omega, theta):
    d = dict(a=a, e=e, omega=omega, theta=theta)
    d['M'] = om.mean_from_true(theta, e)
    return d


def truth_obs(sat, tgt, t_frac=0.5):
    """The 38-slot observation the C env would emit (f64, then cast to f32)."""
    o = np.zeros(38, dtype=np.float64)
    sx, sy, svx, svy = om.orbit_to_cartesian(sat)
    r = math.hypot(sx, sy)
    v_c = math.sqrt(om.MU / r)
    o[0] = (sat['a'] - R_E) / ALT
    o[1] = sat['e']
    o[2] = math.sin(sat['theta'])
    o[3] = math.cos(sat['theta'])
    o[4] = (sx * svx + sy * svy) / r / v_c
    o[5] = (sx * svy - sy * svx) / r / v_c
    o[6] = 1.0
    o[9] = math.sin(sat['omega'])
    o[10] = math.cos(sat['omega'])
    o[15] = t_frac
    # target-derived slots via the shipped encoder (exactly what the C env does)
    o = om.fill_target_obs(o, sat, tgt, ALT, LV)
    lam_s = om.mean_from_true(sat['theta'], sat['e']) + sat['omega']
    lam_t = om.mean_from_true(tgt['theta'], tgt['e']) + tgt['omega']
    o[13] = math.sin(lam_s - lam_t)
    o[14] = math.cos(lam_s - lam_t)
    o[15] = t_frac
    o[16] = math.cos(sat['omega'] - tgt['omega'])
    return o


def main():
    ev.PHASE_OBS_MODE = 1
    rng = np.random.default_rng(7)
    a0 = R_E + 500e3
    rows = []
    # separations spanning the envelopes: terminal box -> 180 deg phase gap
    seps = [1e3, 5e3, 10e3, 50e3, 200e3, 1e6, 5e6, 13e6]
    for sep in seps:
        dpos, dtrans, dbear, drho = [], [], [], []
        for trial in range(400):
            om_s = rng.uniform(0, 2 * math.pi)
            th_s = rng.uniform(0, 2 * math.pi)
            e_s = rng.uniform(0.0, 0.05)
            sat = elements(a0, e_s, om_s, th_s)
            sx, sy, svx, svy = om.orbit_to_cartesian(sat)
            # Build the TARGET by displacing the chaser's Cartesian state by the
            # requested separation in a random LVLH direction (with a matching
            # relative velocity of order n*sep) and re-deriving its elements.
            # This is the only construction that reaches the terminal box: at
            # 5 km an independently-drawn e_t already puts the orbits 300 km
            # apart radially.
            psi = rng.uniform(0, 2 * math.pi)
            r = math.hypot(sx, sy)
            ur = (sx / r, sy / r)
            ut = (-sy / r, sx / r)
            ox = sep * (math.cos(psi) * ur[0] + math.sin(psi) * ut[0])
            oy = sep * (math.cos(psi) * ur[1] + math.sin(psi) * ut[1])
            nmot = math.sqrt(om.MU / a0 ** 3)
            tgt_c = (sx + ox, sy + oy,
                     svx + rng.uniform(-1, 1) * nmot * sep,
                     svy + rng.uniform(-1, 1) * nmot * sep)
            tgt = om.cartesian_to_elements(*tgt_c)
            if not (0.0 <= tgt['e'] < 0.9):
                continue            # LVLH displacement went hyperbolic (far field)
            tgt['M'] = om.mean_from_true(tgt['theta'], tgt['e'])
            tx, ty, _, _ = om.orbit_to_cartesian(tgt)
            rho_true = math.hypot(tx - sx, ty - sy)
            if not (0.5 * sep < rho_true < 2.0 * sep):
                continue
            beta_true = math.atan2(ty - sy, tx - sx)

            o64 = truth_obs(sat, tgt)
            o32 = o64.astype(np.float32)              # what the policy buffer holds
            s_d, t_d = ev.recover_states_t3(o32, ALT)
            sxd, syd, _, _ = om.orbit_to_cartesian(s_d)
            txd, tyd, _, _ = om.orbit_to_cartesian(t_d)
            rho_d = math.hypot(txd - sxd, tyd - syd)
            beta_d = math.atan2(tyd - syd, txd - sxd)

            dpos.append(math.hypot(txd - tx, tyd - ty))
            # transverse (bearing-relevant) component of the RELATIVE error
            ex = (txd - sxd) - (tx - sx)
            ey = (tyd - syd) - (ty - sy)
            ub = (math.cos(beta_true), math.sin(beta_true))
            dtrans.append(abs(-ub[1] * ex + ub[0] * ey))
            drho.append(abs(rho_d - rho_true))
            dbear.append(abs(ev.wrap_pi(beta_d - beta_true)))
        if not dpos:
            continue
        f = lambda v: (float(np.median(v)), float(np.percentile(v, 95)))
        p50_t, p95_t = f(dtrans)
        p50_b, p95_b = f(np.array(dbear) * 1e3)     # mrad
        p50_r, p95_r = f(drho)
        p50_a, p95_a = f(dpos)
        rows.append(dict(sep_m=sep, n=len(dpos),
                         tgt_abs_pos_err_p50_m=p50_a, tgt_abs_pos_err_p95_m=p95_a,
                         rel_transverse_p50_m=p50_t, rel_transverse_p95_m=p95_t,
                         range_err_p50_m=p50_r, range_err_p95_m=p95_r,
                         bearing_err_p50_mrad=p50_b, bearing_err_p95_mrad=p95_b))
        print(f"sep {sep/1e3:9.1f} km | tgt abs pos err {p50_a:7.2f}/{p95_a:7.2f} m"
              f" | rel-transverse {p50_t:7.2f}/{p95_t:7.2f} m"
              f" | range err {p50_r:7.2f}/{p95_r:7.2f} m"
              f" | BEARING err {p50_b:8.4f}/{p95_b:8.4f} mrad"
              f"  ({100*p50_b/1.0:6.1f}% of sigma_beta)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nreference points:")
    print("  sigma_beta (shipped optical)      = 1.000 mrad")
    print("  sigma_rho  (shipped range)        = 50 m")
    print("  NAV-G CRLB, 5 km sep, 1 m/s burn  = 9.1 m epoch range")
    print("  NAV-G BO-BLS-MPC settled, G6      = 17.6 m / 0.019 m/s")
    print("  NAV-G 0.1 mrad sensor, G6         = 1.8 m  <-- decode floor lives here")


if __name__ == "__main__":
    main()
