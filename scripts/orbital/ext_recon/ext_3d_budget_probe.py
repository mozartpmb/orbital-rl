#!/usr/bin/env python3
"""ext-3d recon: plane-change Δv economics and the relative-inclination curriculum.

Read-only. Answers:
  B1  What Δi does the 478 m/s budget actually buy, at each envelope the project
      has already solved (LEO headline, WL4 wide, M5 MEO)?
  B2  If absolute inclination is sampled with independent RAAN (the i_max_sat
      analogue of the retired e_max_sat), what does |Δī| — and hence the plane-
      change Δv — look like? (the de_max lesson, re-derived for inclination)
  B3  A di_max ladder: cost, burn count at each normal-Δv quantum, feasibility.
  B4  Apogee plane-change discount at the eccentric envelopes.

Writes web_data/results/ext_3d_budget.csv
"""
import math, os, csv
import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
BUDGET = None  # computed below from the env constants
ISP, G0, FUEL_FRAC = 300.0, 9.80665, 0.15
VE = ISP * G0
BUDGET = -VE * math.log(1.0 - FUEL_FRAC)          # 478.4 m/s
EXPERT_INPLANE_MED = 189.0                         # recon_expert_baseline.md

def vcirc(a): return math.sqrt(MU / a)
def dv_plane(v, di): return 2.0 * v * math.sin(0.5 * di)
def di_for_dv(v, dv): return 2.0 * math.asin(min(1.0, dv / (2.0 * v)))

ENVELOPES = [
    ("LEO headline  a=6.771e6 (400 km)", R_EARTH + 400e3, 0.025),
    ("LEO band top  a=7.171e6 (800 km)", R_EARTH + 800e3, 0.025),
    ("WL4 wide      a=1.037e7 (4000 km)", R_EARTH + 4000e3, 0.30),
    ("WL4 top       a=1.437e7 (8000 km)", R_EARTH + 8000e3, 0.30),
    ("M5 MEO top    a=2.657e7 (20200 km)", R_EARTH + 20200e3, 0.50),
]

rows = []
print("== B1: what the 478 m/s budget buys in PLANE CHANGE ==")
print(f"budget = -Ve*ln(1-0.15) = {BUDGET:.1f} m/s;  expert in-plane median {EXPERT_INPLANE_MED:.0f} m/s")
print(f"{'envelope':38s} {'v_c[m/s]':>9s} {'di@full':>9s} {'di@resid':>9s} {'dv@0.5deg':>10s} {'dv@1deg':>9s}")
for name, a, e in ENVELOPES:
    v = vcirc(a)
    di_full = math.degrees(di_for_dv(v, BUDGET))
    di_res = math.degrees(di_for_dv(v, BUDGET - EXPERT_INPLANE_MED))
    d05 = dv_plane(v, math.radians(0.5))
    d10 = dv_plane(v, math.radians(1.0))
    print(f"{name:38s} {v:9.0f} {di_full:8.3f}° {di_res:8.3f}° {d05:9.1f} {d10:8.1f}")
    rows.append(dict(probe="B1", cell=name, a_m=a, v_circ=v,
                     di_full_deg=di_full, di_residual_deg=di_res,
                     dv_0p5deg=d05, dv_1deg=d10))

print("\n== B2: independent-RAAN sampling (the i_max trap) ==")
print("i_s, i_t ~ U(0, i_max), RAAN_s, RAAN_t ~ U(0, 2pi) independent")
print(f"{'i_max':>8s} {'E|di_vec|':>10s} {'p50 dv':>9s} {'p90 dv':>9s} {'frac dv>budget':>15s} {'frac dv>resid':>14s}")
rng = np.random.default_rng(20260811)
N = 200_000
v_leo = vcirc(R_EARTH + 550e3)
for i_max_deg in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 51.6, 98.0]:
    im = math.radians(i_max_deg)
    i_s = rng.uniform(0, im, N); i_t = rng.uniform(0, im, N)
    O_s = rng.uniform(0, 2*math.pi, N); O_t = rng.uniform(0, 2*math.pi, N)
    # w-hat from (i, RAAN):  w = (sin i sin O, -sin i cos O, cos i)
    ws = np.stack([np.sin(i_s)*np.sin(O_s), -np.sin(i_s)*np.cos(O_s), np.cos(i_s)])
    wt = np.stack([np.sin(i_t)*np.sin(O_t), -np.sin(i_t)*np.cos(O_t), np.cos(i_t)])
    dw = np.linalg.norm(ws - wt, axis=0)          # = 2 sin(alpha/2) exactly
    dv = v_leo * dw
    frac_over = float((dv > BUDGET).mean())
    frac_over_r = float((dv > BUDGET - EXPERT_INPLANE_MED).mean())
    print(f"{i_max_deg:7.2f}° {dw.mean():10.5f} {np.percentile(dv,50):9.1f} "
          f"{np.percentile(dv,90):9.1f} {frac_over:15.3f} {frac_over_r:14.3f}")
    rows.append(dict(probe="B2", cell=f"i_max={i_max_deg}deg", i_max_deg=i_max_deg,
                     mean_dw=float(dw.mean()), dv_p50=float(np.percentile(dv,50)),
                     dv_p90=float(np.percentile(dv,90)),
                     frac_over_budget=frac_over, frac_over_residual=frac_over_r))

print("\n== B3: di_max ladder (bound the RELATIVE inclination vector, de_max style) ==")
print("di_vec ~ area-uniform disc of radius di_max around the target's i-vector")
print(f"{'di_max':>8s} {'E|di_vec|':>10s} {'p90|di|':>9s} {'p90 dv':>8s} {'%budget':>8s} "
      f"{'n@10m/s':>8s} {'n@25m/s':>8s} {'frac>resid':>11s}")
ladder = []
for di_max_deg in [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    dm = math.radians(di_max_deg)
    r = dm * np.sqrt(rng.uniform(0, 1, N))         # area-uniform, mirrors de_max
    dv = v_leo * r                                  # |dw| ~= |di_vec| for small rel. incl.
    p90 = float(np.percentile(dv, 90))
    n10 = math.ceil(p90 / 10.0); n25 = math.ceil(p90 / 25.0)
    frac = float((dv > BUDGET - EXPERT_INPLANE_MED).mean())
    print(f"{di_max_deg:7.2f}° {math.degrees(r.mean()):9.4f}° {math.degrees(np.percentile(r,90)):8.4f}° "
          f"{p90:8.1f} {100*p90/BUDGET:7.1f}% {n10:8d} {n25:8d} {frac:11.3f}")
    ladder.append((di_max_deg, p90, frac))
    rows.append(dict(probe="B3", cell=f"di_max={di_max_deg}deg", di_max_deg=di_max_deg,
                     dv_p90=p90, pct_budget=100*p90/BUDGET, n_burns_10=n10,
                     n_burns_25=n25, frac_over_residual=frac))

print("\n== B4: apogee plane-change discount ==")
print("burn at apoapsis: v_apo/v_circ = sqrt((1-e)/(1+e))")
print(f"{'e':>6s} {'v_apo/v_c':>10s} {'dv@1deg LEO':>12s} {'saving':>8s}")
for e in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
    ratio = math.sqrt((1 - e) / (1 + e))
    d = dv_plane(v_leo * ratio, math.radians(1.0))
    print(f"{e:6.2f} {ratio:10.4f} {d:12.1f} {100*(1-ratio):7.1f}%")
    rows.append(dict(probe="B4", cell=f"e={e}", ecc=e, v_ratio=ratio, dv_1deg=d))

print("\n== B5: relative-node timing budget ==")
T_leo = 2*math.pi*math.sqrt((R_EARTH+550e3)**3/MU)
print(f"LEO period {T_leo/60:.1f} min = {T_leo/60/1:.0f} sub-steps; relative-node crossings "
      f"every {T_leo/120:.1f} min = {T_leo/120:.0f} sub-steps")
print(f"3000-sub-step cap = {3000*60/T_leo:.1f} revs = {2*3000*60/T_leo:.0f} node crossings")

os.makedirs("/Users/pete/space_training/web_data/results", exist_ok=True)
keys = sorted({k for r in rows for k in r})
with open("/Users/pete/space_training/web_data/results/ext_3d_budget.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("\nwrote web_data/results/ext_3d_budget.csv")
