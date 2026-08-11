#!/usr/bin/env python3
"""
ext-3d B2/B3 — where to crank, what combining is worth, and how to bound the
plane knob (disc on |di_vec| vs a box on (di, dRAAN)).

Writes:
  web_data/results/ext_3d_dv_crankplacement.csv — apsis/node placement savings
  web_data/results/ext_3d_dv_marginal.csv       — marginal cost of the plane leg
                                                  when combined with an in-plane burn
  web_data/results/ext_3d_dv_knob.csv           — disc-knob vs box-knob comparison
"""
import csv
import math
import os

import numpy as np

MU, R_EARTH = 3.986004418e14, 6.371e6
DV_BUDGET = 300.0 * 9.80665 * math.log(1 / 0.85)
OUT = "/Users/pete/space_training/web_data/results"
ALTS_KM = [400.0, 800.0, 2000.0, 8000.0, 20200.0]


def v_at(a, e, th):
    r = a * (1 - e * e) / (1 + e * math.cos(th))
    return math.sqrt(MU * (2.0 / r - 1.0 / a))


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", p, len(rows), "rows")


# ── 1. crank placement: apsis choice and the two-node choice ───────────────
def crank_placement():
    rng = np.random.default_rng(7)
    rows = []
    for alt in ALTS_KM:
        a = R_EARTH + alt * 1e3
        for e in (0.0, 0.05, 0.15, 0.30, 0.50):
            v_ap, v_pe = v_at(a, e, math.pi), v_at(a, e, 0.0)
            vc = math.sqrt(MU / a)
            # node falls at a uniformly random argument of latitude; we may use
            # either of the two nodes (u, u+180). Sample the realized speed.
            u = rng.uniform(0, 2 * math.pi, 200000)
            w = rng.uniform(0, 2 * math.pi, 200000)
            th1, th2 = u - w, u + math.pi - w
            v1 = np.sqrt(MU * (2.0 / (a * (1 - e * e) / (1 + e * np.cos(th1))) - 1.0 / a))
            v2 = np.sqrt(MU * (2.0 / (a * (1 - e * e) / (1 + e * np.cos(th2))) - 1.0 / a))
            vmin = np.minimum(v1, v2)
            rows.append(dict(
                alt_km=alt, e=e, v_circ_ms=vc, v_apo_ms=v_ap, v_peri_ms=v_pe,
                apo_over_peri=v_ap / v_pe,
                dv_ratio_apo_vs_peri=v_ap / v_pe,
                save_apo_vs_peri_pct=100 * (1 - v_ap / v_pe),
                v_node_best_p50=float(np.percentile(vmin, 50)),
                v_node_best_p90=float(np.percentile(vmin, 90)),
                v_node_single_p50=float(np.percentile(v1, 50)),
                best_of_two_nodes_save_pct=100 * (1 - np.mean(vmin) / np.mean(v1)),
                worst_case_over_apo=float(np.percentile(vmin, 100)) / v_ap))
    return rows


# ── 2. marginal cost of the plane leg when combined with an in-plane burn ──
def marginal():
    """Exact combining law: a single impulse that changes speed v1 -> v2 while
    rotating the velocity by theta costs
        hypot(|v2-v1|, 2*sqrt(v1 v2)*sin(theta/2)).
    So the MARGINAL cost of adding the plane rotation to an in-plane impulse of
    size D is  hypot(D, P) - D,  where P = 2 sqrt(v1 v2) sin(theta/2).
    """
    rows = []
    for alt in ALTS_KM:
        a = R_EARTH + alt * 1e3
        vc = math.sqrt(MU / a)
        for di in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            P = 2 * vc * math.sin(0.5 * math.radians(di))
            for D in (0.0, 10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0):
                marg = math.hypot(D, P) - D
                rows.append(dict(alt_km=alt, di_deg=di, dv_plane_standalone_ms=P,
                                 inplane_impulse_ms=D, dv_marginal_ms=marg,
                                 save_vs_standalone_pct=100 * (1 - marg / P) if P > 0 else 0.0,
                                 marginal_frac_budget=marg / DV_BUDGET))
    return rows


# ── 3. knob design: disc on |di_vec| vs box on (di, dRAAN) ────────────────
def knob():
    """Compare two ways of bounding the plane mismatch, over i ~ U(i_lo,i_hi):
      DISC : |di_vec| <= K, uniform on the disc.  theta == |di_vec|, i-invariant.
      BOX  : |di| <= K and |dRAAN| <= K, independent uniforms.
             theta = exact angle between planes (depends on i through sin i).
    Reported at 400 km circular speed (the worst case).
    """
    rng = np.random.default_rng(11)
    n = 400000
    rows = []
    for alt in (400.0, 20200.0):
        vc = math.sqrt(MU / (R_EARTH + alt * 1e3))
        for K in (0.25, 0.5, 1.0, 2.0):
            for lab, i_lo, i_hi in (("i~U(0,90)", 0.0, 90.0),
                                    ("i~U(20,60)", 20.0, 60.0),
                                    ("i~U(80,100)", 80.0, 100.0)):
                i = np.radians(rng.uniform(i_lo, i_hi, n))
                # disc
                r = K * np.sqrt(rng.uniform(0, 1, n))
                th_disc = np.radians(r)
                # box
                di = np.radians(rng.uniform(-K, K, n))
                dOm = np.radians(rng.uniform(-K, K, n))
                i2 = i + di
                c = np.cos(i) * np.cos(i2) + np.sin(i) * np.sin(i2) * np.cos(dOm)
                th_box = np.arccos(np.clip(c, -1, 1))
                dv_d = 2 * vc * np.sin(0.5 * th_disc)
                dv_b = 2 * vc * np.sin(0.5 * th_box)
                rows.append(dict(
                    alt_km=alt, K_deg=K, i_range=lab,
                    theta_disc_p50=float(np.degrees(np.percentile(th_disc, 50))),
                    theta_disc_p99=float(np.degrees(np.percentile(th_disc, 99))),
                    theta_disc_max=float(np.degrees(th_disc.max())),
                    theta_box_p50=float(np.degrees(np.percentile(th_box, 50))),
                    theta_box_p99=float(np.degrees(np.percentile(th_box, 99))),
                    theta_box_max=float(np.degrees(th_box.max())),
                    dv_disc_p99=float(np.percentile(dv_d, 99)),
                    dv_disc_max=float(dv_d.max()),
                    dv_box_p99=float(np.percentile(dv_b, 99)),
                    dv_box_max=float(dv_b.max()),
                    box_over_disc_p99=float(np.percentile(dv_b, 99) / np.percentile(dv_d, 99)),
                    box_max_over_disc_max=float(dv_b.max() / dv_d.max())))
    # cost of 1 deg of RAAN vs 1 deg of inclination, by inclination
    for alt in ALTS_KM:
        vc = math.sqrt(MU / (R_EARTH + alt * 1e3))
        for i_deg in (0.0, 5.0, 28.5, 51.6, 63.4, 90.0, 98.0):
            s = abs(math.sin(math.radians(i_deg)))
            rows.append(dict(alt_km=alt, K_deg=-1.0, i_range=f"i={i_deg}",
                             theta_disc_p50=1.0, theta_disc_p99=1.0, theta_disc_max=1.0,
                             theta_box_p50=s, theta_box_p99=s, theta_box_max=s,
                             dv_disc_p99=2 * vc * math.sin(0.5 * math.radians(1.0)),
                             dv_disc_max=2 * vc * math.sin(0.5 * math.radians(1.0)),
                             dv_box_p99=2 * vc * math.sin(0.5 * math.radians(s)),
                             dv_box_max=2 * vc * math.sin(0.5 * math.radians(s)),
                             box_over_disc_p99=s, box_max_over_disc_max=s))
    return rows


def main():
    cp = crank_placement(); write("ext_3d_dv_crankplacement", cp)
    print("\n-- crank placement (fraction of the circular-speed cost) --")
    for r in cp:
        if r['alt_km'] in (400.0, 8000.0):
            print(f"  alt {r['alt_km']:>7.0f} km  e={r['e']:.2f}  "
                  f"v_apo/v_peri={r['apo_over_peri']:.3f}  "
                  f"best-node p50 v={r['v_node_best_p50']:7.1f}  "
                  f"(apo {r['v_apo_ms']:7.1f}, circ {r['v_circ_ms']:7.1f})  "
                  f"2-node choice saves {r['best_of_two_nodes_save_pct']:4.1f}%")
    mg = marginal(); write("ext_3d_dv_marginal", mg)
    print("\n-- marginal cost of the plane leg at 400 km (m/s) --")
    dis = sorted({r['di_deg'] for r in mg})
    Ds = sorted({r['inplane_impulse_ms'] for r in mg})
    print(f"{'di_deg':>7} {'standalone':>11} " + " ".join(f"D={d:<6.0f}" for d in Ds))
    for di in dis:
        sub = [r for r in mg if r['alt_km'] == 400.0 and r['di_deg'] == di]
        sub.sort(key=lambda r: r['inplane_impulse_ms'])
        print(f"{di:>7.2f} {sub[0]['dv_plane_standalone_ms']:>11.1f} " +
              " ".join(f"{r['dv_marginal_ms']:>8.1f}" for r in sub))
    kb = knob(); write("ext_3d_dv_knob", kb)
    print("\n-- knob: disc(|di_vec|<=K) vs box(|di|<=K, |dRAAN|<=K) at 400 km --")
    for r in kb:
        if r['alt_km'] == 400.0 and r['K_deg'] > 0:
            print(f"  K={r['K_deg']:.2f} deg  {r['i_range']:12}  "
                  f"theta max disc={r['theta_disc_max']:.2f} box={r['theta_box_max']:.2f}  "
                  f"dv_p99 disc={r['dv_disc_p99']:6.1f} box={r['dv_box_p99']:6.1f}  "
                  f"(box/disc p99 {r['box_over_disc_p99']:.2f}x)")


if __name__ == "__main__":
    main()
