#!/usr/bin/env python3
"""
ext-3d B4 — the proposed 3D curriculum ladder, with its analytic joint-feasibility
screen computed rung by rung (same MC as ext_3d_joint_feasibility.py).

Writes web_data/results/ext_3d_ladder.csv
"""
import csv
import importlib.util
import math
import os

spec = importlib.util.spec_from_file_location(
    "jf", "/Users/pete/space_training/scripts/orbital/ext_recon/ext_3d_joint_feasibility.py")
jf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jf)

OUT = "/Users/pete/space_training/web_data/results"

# lineage-constant observation scales (must not change inside a lineage)
I_SCALES = dict(obs_alt_scale_m="8e6", lvlh_scale_m="1.5e7", shape_dv_ref_ms="700")
MI_SCALES = dict(obs_alt_scale_m="2.1e7", lvlh_scale_m="4e7", shape_dv_ref_ms="700")

LADDER = [
    dict(rung="I0", lineage="regression", role="2D bit-exact anchor (no training)",
         i_lo=0.0, i_hi=0.0, di_max_deg=0.0, e_max=0.0, de_max=None, da_max_km=None,
         alt_lo=500, alt_hi=800, same_orbit=True, horizon=3000, scales=I_SCALES,
         warm="n/a", gate="legacy 2D anchors bit-exact (26/200 legacy ckpt; T3 canonical 100%)"),
    dict(rung="I1", lineage="I (wide scales)", role="3D frame, zero plane error — WL1 twin",
         i_lo=0.0, i_hi=98.0, di_max_deg=0.0, e_max=0.0, de_max=None, da_max_km=None,
         alt_lo=500, alt_hi=800, same_orbit=True, horizon=3000, scales=I_SCALES,
         warm="fresh", gate="200/200 (must equal WL1)"),
    dict(rung="I2", lineage="I (wide scales)", role="first plane error, circular",
         i_lo=0.0, i_hi=98.0, di_max_deg=0.5, e_max=0.0, de_max=None, da_max_km=None,
         alt_lo=500, alt_hi=800, same_orbit=True, horizon=3000, scales=I_SCALES,
         warm="I1", gate=">=190/200"),
    dict(rung="I3", lineage="I (wide scales)", role="3D HEADLINE (L2 twin + plane)",
         i_lo=0.0, i_hi=98.0, di_max_deg=0.75, e_max=0.05, de_max=None, da_max_km=None,
         alt_lo=300, alt_hi=800, same_orbit=False, horizon=3000, scales=I_SCALES,
         warm="I2", gate=">=190/200, 3 seeds"),
    dict(rung="I4", lineage="I (wide scales)", role="wide-e + plane (WL3 twin)",
         i_lo=0.0, i_hi=98.0, di_max_deg=0.75, e_max=0.15, de_max=0.06, da_max_km=400,
         alt_lo=300, alt_hi=2000, same_orbit=False, horizon=3000, scales=I_SCALES,
         warm="I3", gate=">=190/200"),
    dict(rung="I5", lineage="I (wide scales)", role="widest jointly-feasible (WL4 twin)",
         i_lo=0.0, i_hi=98.0, di_max_deg=0.75, e_max=0.30, de_max=0.08, da_max_km=600,
         alt_lo=300, alt_hi=8000, same_orbit=False, horizon=6000, scales=I_SCALES,
         warm="I4", gate=">=185/200, then 3-4 seeds"),
    dict(rung="MI5", lineage="MI (MEO scales)", role="stretch: MEO 3D (M5 twin)",
         i_lo=0.0, i_hi=98.0, di_max_deg=0.75, e_max=0.50, de_max=0.10, da_max_km=1000,
         alt_lo=300, alt_hi=20200, same_orbit=False, horizon=12000, scales=MI_SCALES,
         warm="fresh MI1-MI4 re-ladder", gate="report vs screen, not vs 100%"),
    # exploratory upper edge, for the record
    dict(rung="I5+", lineage="I (wide scales)", role="edge probe — NOT a gate rung",
         i_lo=0.0, i_hi=98.0, di_max_deg=1.5, e_max=0.30, de_max=0.08, da_max_km=600,
         alt_lo=300, alt_hi=8000, same_orbit=False, horizon=6000, scales=I_SCALES,
         warm="I5", gate="exploratory"),
]


def main():
    rows = []
    for r in LADDER:
        s = jf.sample(n=6000, seed=4242, di_max_deg=r['di_max_deg'],
                      alt_lo=r['alt_lo'], alt_hi=r['alt_hi'], e_max=r['e_max'],
                      de_max=r['de_max'], da_max_km=r['da_max_km'],
                      same_orbit=r['same_orbit'], horizon=r['horizon'])
        base = jf.sample(n=6000, seed=4242, di_max_deg=0.0,
                         alt_lo=r['alt_lo'], alt_hi=r['alt_hi'], e_max=r['e_max'],
                         de_max=r['de_max'], da_max_km=r['da_max_km'],
                         same_orbit=r['same_orbit'], horizon=r['horizon'])
        rows.append(dict(
            rung=r['rung'], lineage=r['lineage'], role=r['role'],
            i_range_deg=f"{r['i_lo']:.0f}-{r['i_hi']:.0f}",
            di_max_deg=r['di_max_deg'], e_max=r['e_max'],
            de_max=(r['de_max'] if r['de_max'] is not None else -1),
            da_max_km=(r['da_max_km'] if r['da_max_km'] is not None else -1),
            alt_lo_km=r['alt_lo'], alt_hi_km=r['alt_hi'],
            same_orbit_init=int(r['same_orbit']), cap_steps=r['horizon'],
            cap_hours=r['horizon'] * 60 / 3600.0,
            obs_alt_scale_m=r['scales']['obs_alt_scale_m'],
            lvlh_scale_m=r['scales']['lvlh_scale_m'],
            shape_dv_ref_ms=r['scales']['shape_dv_ref_ms'],
            warm_from=r['warm'], gate=r['gate'],
            feas_2d_screen=base['frac_feasible_seq'],
            feas_screen_seq=s['frac_feasible_seq'],
            feas_screen_comb=s['frac_feasible_comb'],
            plane_cost_pp=base['frac_feasible_seq'] - s['frac_feasible_seq'],
            dv_plane_p50=s['dv_plane_p50'], dv_plane_p90=s['dv_plane_p90'],
            dv_total_p50=s['dv_seq_p50'], dv_total_p90=s['dv_seq_p90'],
            theta_p50_deg=s['theta_p50'], theta_p90_deg=s['theta_p90'],
            steps_p50=s['steps_p50'], steps_p90=s['steps_p90']))
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/ext_3d_ladder.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", p)
    print(f"{'rung':5} {'di_max':>6} {'e_max':>6} {'band':>12} {'cap':>6} "
          f"{'2Dscr':>7} {'3Dscr':>7} {'comb':>7} {'cost':>6} {'dvpl_p90':>8} {'dv_p90':>7}")
    for r in rows:
        print(f"{r['rung']:5} {r['di_max_deg']:6.2f} {r['e_max']:6.2f} "
              f"{str(int(r['alt_lo_km']))+'-'+str(int(r['alt_hi_km'])):>12} "
              f"{r['cap_steps']:6d} {100*r['feas_2d_screen']:6.1f}% "
              f"{100*r['feas_screen_seq']:6.1f}% {100*r['feas_screen_comb']:6.1f}% "
              f"{100*r['plane_cost_pp']:5.1f}pp {r['dv_plane_p90']:8.1f} {r['dv_total_p90']:7.0f}")


if __name__ == "__main__":
    main()
