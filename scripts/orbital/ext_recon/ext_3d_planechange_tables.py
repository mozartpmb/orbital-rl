#!/usr/bin/env python3
"""
ext-3d B1/B2/B3 — analytic plane-change geometry tables.

Pure astrodynamics, no env, no training. Mirrors the units/constants of the
shipped C env (MU, R_EARTH, DT, 15% propellant fraction -> 478.13 m/s budget).

Writes:
  web_data/results/ext_3d_dv_planechange.csv  — dv = 2 v sin(di/2) vs altitude/di,
                                                incl. apoapsis-crank speeds for e>0
  web_data/results/ext_3d_dv_envelope.csv     — di_max affordable vs reserved in-plane budget
  web_data/results/ext_3d_dv_combined.csv     — combined (vector) vs sequential burn savings
  web_data/results/ext_3d_dv_threeburn.csv    — raise-crank-lower vs direct crank thresholds
  web_data/results/ext_3d_dv_raan.csv         — RAAN vs inclination decomposition cost
"""
import csv
import math
import os

MU = 3.986004418e14
R_EARTH = 6.371e6
ISP, G0, FF = 300.0, 9.80665, 0.15
DV_BUDGET = ISP * G0 * math.log(1.0 / (1.0 - FF))          # 478.12987798780637
OUT = "/Users/pete/space_training/web_data/results"

ALTS_KM = [400.0, 800.0, 2000.0, 8000.0, 20200.0]
DI_DEG = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
          7.5, 10.0, 15.0, 20.0, 28.5, 38.94, 45.0, 60.0, 90.0]


def v_circ(a):
    return math.sqrt(MU / a)


def v_at(a, e, theta):
    r = a * (1 - e * e) / (1 + e * math.cos(theta))
    return math.sqrt(MU * (2.0 / r - 1.0 / a))


def dv_plane(v, di_rad):
    return 2.0 * v * math.sin(0.5 * di_rad)


def di_for_dv(v, dv):
    """Inverse: largest plane angle purchasable with dv at local speed v."""
    s = dv / (2.0 * v)
    if s >= 1.0:
        return math.pi          # >=180 deg reachable (retro-flip)
    return 2.0 * math.asin(s)


def hohmann(a1, a2):
    dv = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0)) + \
         abs(math.sqrt(MU / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2))))
    return dv, math.pi * math.sqrt((0.5 * (a1 + a2)) ** 3 / MU)


# ── T1: dv = 2 v sin(di/2) across altitude, plus apoapsis crank for e>0 ─────
def table_planechange():
    rows = []
    for alt in ALTS_KM:
        a = R_EARTH + alt * 1e3
        vc = v_circ(a)
        for di in DI_DEG:
            r = math.radians(di)
            d_c = dv_plane(vc, r)
            row = dict(alt_km=alt, a_km=a / 1e3, v_circ_ms=vc, di_deg=di,
                       dv_circ_ms=d_c, frac_budget_circ=d_c / DV_BUDGET)
            # cranking at apoapsis of an eccentric orbit with the SAME a
            for e in (0.10, 0.30, 0.50):
                v_apo = v_at(a, e, math.pi)
                v_per = v_at(a, e, 0.0)
                row[f"dv_apo_e{int(e*100):02d}_ms"] = dv_plane(v_apo, r)
                row[f"dv_per_e{int(e*100):02d}_ms"] = dv_plane(v_per, r)
            rows.append(row)
    return rows


# ── T2: trainable envelope — di_max vs reserved in-plane budget ────────────
def table_envelope():
    """How much plane angle is left after the in-plane job is paid for.
    Reserves are the measured in-plane spend of the shipped policies:
      0     : whole budget on the plane change (upper bound, unrealistic)
      110   : e=0 LEO Hohmann-only median (t3_ecc_dv_cost dv_total_p50 @ e=0)
      150   : phasing-heavy circular case
      235   : LEO headline realized Delta-v p50 (t3_headline_characterization)
      270   : WL4 / M5 realized p50
      325   : LEO headline realized p90 (the number di_max must survive)
      386   : WL4 analytic p90 (t3_joint_feasibility, de_max row)
      439   : M5 analytic p90
    """
    reserves = [0.0, 110.0, 150.0, 235.0, 270.0, 325.0, 386.0, 439.0]
    rows = []
    for alt in ALTS_KM:
        a = R_EARTH + alt * 1e3
        vc = v_circ(a)
        for res in reserves:
            left = DV_BUDGET - res
            rows.append(dict(
                alt_km=alt, v_circ_ms=vc, inplane_reserve_ms=res,
                dv_left_ms=left,
                di_max_deg_circ=math.degrees(di_for_dv(vc, left)) if left > 0 else 0.0,
                di_max_deg_apo_e30=math.degrees(
                    di_for_dv(v_at(a, 0.30, math.pi), left)) if left > 0 else 0.0,
                di_max_deg_apo_e50=math.degrees(
                    di_for_dv(v_at(a, 0.50, math.pi), left)) if left > 0 else 0.0,
                di_max_deg_peri_e30=math.degrees(
                    di_for_dv(v_at(a, 0.30, 0.0), left)) if left > 0 else 0.0,
            ))
    return rows


# ── T3: combined (single vectorial impulse) vs sequential ─────────────────
def table_combined():
    """Exact identity used throughout:
         dv_comb = sqrt(v1^2 + v2^2 - 2 v1 v2 cos(theta))
                 = hypot(|v2-v1|, 2*sqrt(v1*v2)*sin(theta/2))
       i.e. the in-plane magnitude change and the plane rotation add IN
       QUADRATURE with the *geometric-mean* speed as the plane-change speed.
       Sequential = |v2-v1| + 2*v2*sin(theta/2).
    """
    rows = []
    for alt1 in ALTS_KM[:-1]:
        a1 = R_EARTH + alt1 * 1e3
        for dalt in (100.0, 400.0, 600.0, 1000.0, 2000.0):
            a2 = a1 + dalt * 1e3
            dv_h, _ = hohmann(a1, a2)
            at = 0.5 * (a1 + a2)
            v_ta = math.sqrt(MU * (2.0 / a2 - 1.0 / at))   # transfer speed at apoapsis
            v2 = v_circ(a2)
            dv1 = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0))
            for di in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
                th = math.radians(di)
                dv_pl_hi = dv_plane(v2, th)          # crank at the higher (slower) orbit
                seq = dv_h + dv_pl_hi
                # all plane change folded into burn 2 (the circularization burn)
                comb2 = dv1 + math.sqrt(v_ta**2 + v2**2 - 2 * v_ta * v2 * math.cos(th))
                # optimal split of theta between burn 1 and burn 2
                best, best_s = math.inf, 0.0
                for k in range(0, 101):
                    s = k / 100.0
                    t1, t2 = s * th, (1 - s) * th
                    v1c = math.sqrt(MU / a1)
                    v_tp = math.sqrt(MU * (2.0 / a1 - 1.0 / at))
                    b1 = math.sqrt(v1c**2 + v_tp**2 - 2 * v1c * v_tp * math.cos(t1))
                    b2 = math.sqrt(v_ta**2 + v2**2 - 2 * v_ta * v2 * math.cos(t2))
                    if b1 + b2 < best:
                        best, best_s = b1 + b2, s
                rows.append(dict(
                    alt1_km=alt1, alt2_km=alt1 + dalt, da_km=dalt, di_deg=di,
                    dv_hohmann_ms=dv_h, dv_plane_at_hi_ms=dv_pl_hi,
                    dv_sequential_ms=seq, dv_combined_burn2_ms=comb2,
                    dv_combined_optsplit_ms=best, opt_split_frac_burn1=best_s,
                    save_vs_seq_ms=seq - best, save_vs_seq_pct=100 * (seq - best) / seq))
    # theoretical ceiling on the saving: quadrature vs sum
    for ratio in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 10.0):
        rows.append(dict(alt1_km=-1, alt2_km=-1, da_km=-1, di_deg=-1,
                         dv_hohmann_ms=1.0, dv_plane_at_hi_ms=ratio,
                         dv_sequential_ms=1.0 + ratio,
                         dv_combined_burn2_ms=math.hypot(1.0, ratio),
                         dv_combined_optsplit_ms=math.hypot(1.0, ratio),
                         opt_split_frac_burn1=-1,
                         save_vs_seq_ms=(1.0 + ratio) - math.hypot(1.0, ratio),
                         save_vs_seq_pct=100 * (1 - math.hypot(1.0, ratio) / (1.0 + ratio))))
    return rows


# ── T4: raise-crank-lower vs direct crank ─────────────────────────────────
def three_burn_cost(r1, r2, di_rad):
    at = 0.5 * (r1 + r2)
    v1 = math.sqrt(MU / r1)
    v_p = math.sqrt(MU * (2.0 / r1 - 1.0 / at))
    v_a = math.sqrt(MU * (2.0 / r2 - 1.0 / at))
    dv_raise = abs(v_p - v1)
    dv_crank = 2.0 * v_a * math.sin(0.5 * di_rad)
    t = 2.0 * math.pi * math.sqrt(at**3 / MU)          # up and back = 1 full ellipse period
    return 2.0 * dv_raise + dv_crank, t


def table_threeburn():
    rows = []
    # base orbits + apoapsis ceilings (band ceilings of the proven rungs, plus free)
    cases = [
        ("LEO-headline", 400.0, [800.0]),
        ("WL3", 400.0, [2000.0]),
        ("WL4", 400.0, [8000.0]),
        ("M5", 400.0, [20200.0]),
        ("M5-hi", 8000.0, [20200.0]),
        # near-tangent raises: the infimum of the crossover over all r2 is the
        # classical marginal threshold (first-order condition at r2 -> r1)
        ("near-tangent", 400.0, [401.0, 410.0, 450.0, 500.0, 600.0]),
        ("free", 400.0, [2000.0, 8000.0, 20200.0, 50000.0, 100000.0, 380000.0]),
    ]
    for name, alt1, ceils in cases:
        r1 = R_EARTH + alt1 * 1e3
        v1 = math.sqrt(MU / r1)
        for ceil in ceils:
            r2 = R_EARTH + ceil * 1e3
            if r2 <= r1:
                continue
            # crossover di where 3-burn beats direct
            lo, hi = 1e-6, math.pi
            cross = None
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                d3, _t = three_burn_cost(r1, r2, mid)
                dd = 2.0 * v1 * math.sin(0.5 * mid)
                if d3 <= dd:
                    hi = mid
                else:
                    lo = mid
            cross = math.degrees(0.5 * (lo + hi))
            d3c, tc = three_burn_cost(r1, r2, math.radians(cross))
            # max di affordable inside the FULL budget, each strategy
            def max_di(fn):
                lo2, hi2 = 0.0, math.pi
                for _ in range(200):
                    m = 0.5 * (lo2 + hi2)
                    if fn(m) <= DV_BUDGET:
                        lo2 = m
                    else:
                        hi2 = m
                return math.degrees(lo2)
            di_direct = max_di(lambda x: 2.0 * v1 * math.sin(0.5 * x))
            di_three = max_di(lambda x: three_burn_cost(r1, r2, x)[0])
            _, t3 = three_burn_cost(r1, r2, math.radians(max(di_three, 1e-6)))
            rows.append(dict(
                case=name, alt1_km=alt1, apo_ceiling_km=ceil, r2_over_r1=r2 / r1,
                crossover_di_deg=cross, dv_at_crossover_ms=d3c,
                crossover_dv_over_budget=d3c / DV_BUDGET,
                di_max_direct_deg_fullbudget=di_direct,
                di_max_threeburn_deg_fullbudget=di_three,
                threeburn_gain_deg=di_three - di_direct,
                threeburn_roundtrip_hr=t3 / 3600.0,
                threeburn_roundtrip_steps60s=t3 / 60.0))
    return rows


# ── T5: RAAN geometry / decomposition ─────────────────────────────────────
def rel_plane_angle(i1, i2, dRAAN):
    """Exact angle between two orbit planes."""
    c = math.cos(i1) * math.cos(i2) + math.sin(i1) * math.sin(i2) * math.cos(dRAAN)
    return math.acos(max(-1.0, min(1.0, c)))


def table_raan():
    rows = []
    for alt in ALTS_KM:
        a = R_EARTH + alt * 1e3
        vc = v_circ(a)
        for i_deg in (0.0, 5.0, 28.5, 51.6, 63.4, 90.0, 98.0):
            i = math.radians(i_deg)
            for dOm_deg in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
                dOm = math.radians(dOm_deg)
                th_exact = rel_plane_angle(i, i, dOm)
                th_lin = abs(math.sin(i)) * dOm
                # relative-node location (argument of latitude from chaser's AN)
                # pure dRAAN -> node at u = +-90 deg
                rows.append(dict(
                    alt_km=alt, i_deg=i_deg, dRAAN_deg=dOm_deg,
                    theta_exact_deg=math.degrees(th_exact),
                    theta_smallangle_deg=math.degrees(th_lin),
                    lin_err_pct=(100 * (th_lin - th_exact) / th_exact) if th_exact > 0 else 0.0,
                    dv_ms=dv_plane(vc, th_exact),
                    dv_per_deg_RAAN=dv_plane(vc, th_exact) / dOm_deg,
                    dv_per_deg_incl=dv_plane(vc, math.radians(1.0)),
                    cost_ratio_RAAN_to_incl=(dv_plane(vc, th_exact) / dOm_deg) /
                                            dv_plane(vc, math.radians(1.0)),
                    node_u_deg=90.0))
    return rows


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{name}.csv"
    keys = list(rows[0].keys())
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", p, len(rows), "rows")


def main():
    print(f"DV_BUDGET = {DV_BUDGET:.2f} m/s")
    pc = table_planechange()
    write("ext_3d_dv_planechange", pc)
    print("\n-- dv (m/s) for a plane change at circular speed --")
    print(f"{'di_deg':>8} " + " ".join(f"{a:>9.0f}km" for a in ALTS_KM))
    for di in DI_DEG:
        vals = [r for r in pc if abs(r['di_deg'] - di) < 1e-9]
        print(f"{di:>8.2f} " + " ".join(f"{v['dv_circ_ms']:>11.1f}" for v in vals))

    env = table_envelope()
    write("ext_3d_dv_envelope", env)
    print("\n-- di_max (deg) affordable at circular speed vs in-plane reserve --")
    reserves = sorted({r['inplane_reserve_ms'] for r in env})
    print(f"{'reserve':>8} " + " ".join(f"{a:>9.0f}km" for a in ALTS_KM))
    for res in reserves:
        vals = [r for r in env if r['inplane_reserve_ms'] == res]
        vals.sort(key=lambda r: ALTS_KM.index(r['alt_km']))
        print(f"{res:>8.0f} " + " ".join(f"{v['di_max_deg_circ']:>11.3f}" for v in vals))

    cb = table_combined()
    write("ext_3d_dv_combined", cb)
    tb = table_threeburn()
    write("ext_3d_dv_threeburn", tb)
    print("\n-- raise-crank-lower crossover --")
    for r in tb:
        print(f"{r['case']:>13} r2/r1={r['r2_over_r1']:.2f}  crossover di="
              f"{r['crossover_di_deg']:7.2f} deg  (dv there = "
              f"{r['crossover_dv_over_budget']:6.1f}x budget)  "
              f"di_max direct={r['di_max_direct_deg_fullbudget']:.3f} vs 3burn="
              f"{r['di_max_threeburn_deg_fullbudget']:.3f}")
    ra = table_raan()
    write("ext_3d_dv_raan", ra)


if __name__ == "__main__":
    main()
