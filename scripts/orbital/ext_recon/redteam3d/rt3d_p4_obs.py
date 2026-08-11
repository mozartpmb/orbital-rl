"""RT3D-P4 — attack the 38-slot obs relayout and the frame_randomize test.

E1  Is EVERY proposed slot SO(3)-invariant?  The design asserts "under two-body
    the env is SO(3)-invariant, so G is exactly zero-information", and proposes
    `frame_randomize` as a one-command invariance regression.  But slots 8/9/10
    carry Dlam = lam_s - lam_t with lam = M + argp + RAAN, and RAAN is measured
    from the INERTIAL x-axis.  Under a general rotation the two orbits' nodes
    move by DIFFERENT amounts whenever their planes differ -> Dlam is a
    frame-VARIANT quantity at di != 0.  Measured here.

E2  Slot saturation against the declared Box(-2, 2) at each rung's real init
    distribution (slots 16, 17, 19 = the dv ledger; slot 7 = delta-a).

E3  The terminal box vs the ladder's di values: at what relative inclination
    does the plane error alone fill the 30 km / 50 m/s box?  Rungs below that
    are solvable with ZERO normal burns.
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import (MU, R_EARTH, what_from_ie, plane_angle, rotate_about,
                         coe2rv, rv2coe, pct, wrap_pi)

OUT_INV = "/Users/pete/space_training/web_data/results/ext_rt3d_frame_invariance.csv"
OUT_SAT = "/Users/pete/space_training/web_data/results/ext_rt3d_obs_saturation.csv"
OUT_BOX = "/Users/pete/space_training/web_data/results/ext_rt3d_box_vs_ladder.csv"


def rand_rot(rng):
    """Uniform SO(3) via a random axis + angle (good enough for an invariance test)."""
    while True:
        x, y, z = (rng.gauss(0, 1) for _ in range(3))
        n = math.sqrt(x * x + y * y + z * z)
        if n > 1e-9:
            break
    return (x / n, y / n, z / n), rng.random() * 2 * math.pi


def lam_of(r, v):
    a, e, inc, raan, argp, nu = rv2coe(r, v)
    if e < 1e-12:
        # circular: argp := 0, nu := argument of latitude (rv2coe already does)
        M = nu
    else:
        E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2),
                             math.sqrt(1 + e) * math.cos(nu / 2))
        M = E - e * math.sin(E)
    return M + argp + raan


def lam_rel_target_gauge(r_s, v_s, r_t, v_t):
    """CANDIDATE FIX: mean longitude difference with the TARGET'S OWN orbit plane
    as the reference plane.  Both bodies are expressed in the target frame
    (n_hat_t, m_hat_t, h_hat_t), so a global SO(3) rotation cancels exactly."""
    a_t, e_t, i_t, O_t, w_t, nu_t = rv2coe(r_t, v_t)
    wh = what_from_ie(i_t, O_t)
    # target frame: e1 = target periapsis direction (or its node if circular)
    if e_t > 1e-12:
        a_, e_, i_, O_, w_, nu_ = rv2coe(r_t, v_t)
        # periapsis unit vector in inertial space
        from rt3d_common import evec_from_elements
        ev = evec_from_elements(1.0, i_t, O_t, w_t)
        e1 = ev
    else:
        nx = -wh[1]; ny = wh[0]; nn = math.hypot(nx, ny)
        e1 = (nx / nn, ny / nn, 0.0) if nn > 1e-14 else (1.0, 0.0, 0.0)
    e3 = wh
    e2 = (e3[1] * e1[2] - e3[2] * e1[1], e3[2] * e1[0] - e3[0] * e1[2],
          e3[0] * e1[1] - e3[1] * e1[0])
    def to_t(v):
        return (sum(v[k] * e1[k] for k in range(3)),
                sum(v[k] * e2[k] for k in range(3)),
                sum(v[k] * e3[k] for k in range(3)))
    rs, vs = to_t(r_s), to_t(v_s)
    rt, vt = to_t(r_t), to_t(v_t)
    return wrap_pi(lam_of(rs, vs) - lam_of(rt, vt))


def run_frame_invariance(n=4000):
    rows = []
    rng = random.Random(1234)
    for di_deg in (0.0, 0.05, 0.25, 1.0, 2.0, 5.0):
        di = math.radians(di_deg)
        errs_lam, errs_de, errs_dipl, errs_rho = [], [], [], []
        for _ in range(n):
            a_t = R_EARTH + 550e3
            a_s = a_t + rng.uniform(-200e3, 200e3)
            e_t = rng.random() * 0.05
            e_s = rng.random() * 0.05
            w_t = rng.random() * 2 * math.pi
            w_s = rng.random() * 2 * math.pi
            nu_t = rng.random() * 2 * math.pi
            nu_s = rng.random() * 2 * math.pi
            # design gauge: target plane = equator, all relative i in the chaser
            i_t, O_t = 0.0, 0.0
            wh_t = what_from_ie(i_t, O_t)
            phi = rng.random() * 2 * math.pi
            axis0 = (math.cos(phi), math.sin(phi), 0.0)
            wh_s = rotate_about(wh_t, axis0, di)
            i_s = math.atan2(math.hypot(wh_s[0], wh_s[1]), wh_s[2])
            O_s = math.atan2(wh_s[0], -wh_s[1]) % (2 * math.pi)
            rt, vt = coe2rv(a_t, e_t, i_t, O_t, w_t, nu_t)
            rs, vs = coe2rv(a_s, e_s, i_s, O_s, w_s, nu_s)
            lam0 = wrap_pi(lam_of(rs, vs) - lam_of(rt, vt))
            lamg0 = lam_rel_target_gauge(rs, vs, rt, vt)
            rel0 = math.sqrt(sum((rs[k] - rt[k]) ** 2 for k in range(3)))
            pl0 = plane_angle(wh_s, wh_t)
            ax, ang = rand_rot(rng)
            rt2, vt2 = rotate_about(rt, ax, ang), rotate_about(vt, ax, ang)
            rs2, vs2 = rotate_about(rs, ax, ang), rotate_about(vs, ax, ang)
            lam1 = wrap_pi(lam_of(rs2, vs2) - lam_of(rt2, vt2))
            lamg1 = lam_rel_target_gauge(rs2, vs2, rt2, vt2)
            errs_de.append(abs(wrap_pi(lamg1 - lamg0)))
            rel1 = math.sqrt(sum((rs2[k] - rt2[k]) ** 2 for k in range(3)))
            _, _, i2s, O2s, _, _ = rv2coe(rs2, vs2)
            _, _, i2t, O2t, _, _ = rv2coe(rt2, vt2)
            pl1 = plane_angle(what_from_ie(i2s, O2s), what_from_ie(i2t, O2t))
            errs_lam.append(abs(wrap_pi(lam1 - lam0)))
            errs_dipl.append(abs(pl1 - pl0))
            errs_rho.append(abs(rel1 - rel0))
        rows.append(dict(di_deg=di_deg,
                         dlambda_shift_p50_deg=round(math.degrees(pct(errs_lam, .5)), 8),
                         dlambda_shift_p90_deg=round(math.degrees(pct(errs_lam, .9)), 8),
                         dlambda_shift_max_deg=round(math.degrees(max(errs_lam)), 8),
                         dlambda_shift_max_phi_units=round(max(errs_lam) / math.pi, 8),
                         FIX_target_gauge_shift_max_deg=f"{math.degrees(max(errs_de)):.3e}",
                         plane_angle_shift_max_rad=f"{max(errs_dipl):.3e}",
                         lvlh_rho_shift_max_m=f"{max(errs_rho):.3e}",
                         verdict=("INVARIANT" if max(errs_lam) < 1e-9 else "FRAME-VARIANT")))
        print(f"di={di_deg:5.2f} deg: |Dlambda| shift under random SO(3)  "
              f"p50={math.degrees(pct(errs_lam,.5)):9.5f} deg  "
              f"p90={math.degrees(pct(errs_lam,.9)):9.5f}  "
              f"max={math.degrees(max(errs_lam)):9.5f} deg "
              f"(={max(errs_lam)/math.pi:.5f} Phi units)   "
              f"|| FIX(target-gauge) max={math.degrees(max(errs_de)):.3e} deg")
    with open(OUT_INV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_INV, "\n")


def run_obs_saturation(n=20000):
    """slots 7 (da), 16/17 (dv ledger), 19 (margin) vs Box(-2,2)."""
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    p3 = importlib.import_module("rt3d_p3_phi")
    rows = []
    for cfg in p3.RUNGS:
        name, amin, amax, emt, ems, demax, damax, same, di_deg, dvref = cfg
        rng = random.Random(999)
        s7, s16, s17, s19 = [], [], [], []
        obs_da_scale = 0.10
        for _ in range(n):
            a_s, e_s, w_s, a_t, e_t, w_t = p3.sample_init(rng, cfg)
            i_t, O_t, i_s, O_s, wh_t, wh_s, delta = p3.plane_pair(rng, math.radians(di_deg))
            v_t = math.sqrt(MU / a_t)
            from rt3d_common import evec_from_elements
            es3 = evec_from_elements(e_s, i_s, O_s, w_s)
            et3 = evec_from_elements(e_t, i_t, O_t, w_t)
            de = math.sqrt(sum((es3[k] - et3[k]) ** 2 for k in range(3)))
            da = (a_s - a_t) / a_t
            s7.append(da / obs_da_scale)
            in_ = 0.5 * v_t * math.hypot(da, de) / dvref
            pl_ = v_t * math.sqrt(sum((wh_s[k] - wh_t[k]) ** 2 for k in range(3))) / dvref
            rem = 478.13 / dvref
            s16.append(in_); s17.append(pl_); s19.append(rem - in_ - pl_)
        def clipfrac(v, lo=-2.0, hi=2.0):
            return sum(1 for x in v if x < lo or x > hi) / float(len(v))
        rows.append(dict(rung=name,
                         slot7_da_max=round(max(abs(x) for x in s7), 3),
                         slot7_clip_frac=round(clipfrac(s7), 5),
                         slot16_max=round(max(s16), 3), slot16_clip_frac=round(clipfrac(s16), 5),
                         slot17_max=round(max(s17), 3), slot17_clip_frac=round(clipfrac(s17), 5),
                         slot19_min=round(min(s19), 3), slot19_clip_frac=round(clipfrac(s19), 5)))
        print(f"{name:24s} slot7 |max|={max(abs(x) for x in s7):7.2f} "
              f"clip={clipfrac(s7):6.2%} | slot16 max={max(s16):6.2f} "
              f"clip={clipfrac(s16):6.2%} | slot19 min={min(s19):7.2f} "
              f"clip={clipfrac(s19):6.2%}")
    with open(OUT_SAT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_SAT, "\n")


def run_box_vs_ladder():
    """Plane error alone: at what di does it fill the success box?"""
    rows = []
    for alt_km, box_pos_km, box_vel in (
            (550, 30, 50), (550, 10, 10), (550, 5, 2), (550, 5, 1),
            (2000, 30, 50), (8000, 30, 50), (20200, 30, 50)):
        r = R_EARTH + alt_km * 1e3
        v = math.sqrt(MU / r)
        # For two same-size orbits separated by delta, at argument-of-latitude u
        # from the RELATIVE node:  |dr| = 2 r sin(d/2) |sin u|,
        #                          |dv| = 2 v sin(d/2) |cos u|.
        # Capture AT THE NODE (u=0) makes the position term exactly zero, so the
        # velocity tolerance alone decides how much plane error the box forgives.
        di_free = 2.0 * math.asin(min(1.0, box_vel / (2.0 * v)))
        di_pos_antinode = 2.0 * math.asin(min(1.0, box_pos_km * 1e3 / (2.0 * r)))
        row = dict(alt_km=alt_km, box_pos_km=box_pos_km, box_vel_ms=box_vel,
                   di_free_at_node_deg=round(math.degrees(di_free), 5),
                   di_pos_bound_at_antinode_deg=round(math.degrees(di_pos_antinode), 5),
                   binding="velocity_at_node")
        for dimax in (0.05, 0.25, 0.75, 1.0, 2.0):
            f = min(1.0, (math.degrees(di_free) / dimax) ** 2)   # area-uniform disc
            row[f"frac_no_plane_burn_needed_dimax{dimax}"] = round(f, 4)
        rows.append(row)
        print(f"alt={alt_km:6d} km box={box_pos_km:3d} km/{box_vel:3d} m/s -> plane "
              f"error FREE up to {math.degrees(di_free):7.4f} deg at the node; "
              f"frac of disc draws needing NO plane burn: "
              + "  ".join(f"di{d}={min(1.0,(math.degrees(di_free)/d)**2):5.1%}"
                          for d in (0.05, 0.25, 0.75, 1.0, 2.0)))
    with open(OUT_BOX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_BOX)


if __name__ == "__main__":
    run_frame_invariance()
    run_obs_saturation()
    run_box_vs_ladder()
