#!/usr/bin/env python3
"""Solve the e_max (and di_max) that restore >=97% joint feasibility at the wide
bands, under BOTH e-vector semantics.

Under the shipped sampler ('realized') the de_max knob is inert once di_max > 0,
so the realised inertial |d e_vec| is set by e_max alone. This script finds the
e_max that puts each band back over 97% combined-cost feasibility WITHOUT any
env change, and reports the di_max thresholds at 97% absolute for both costings.

Writes /tmp/w3d_emax_scan.csv
"""
import csv
import math
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training-ext3d/scripts/orbital/ext3d/diag")
from w3d_screen import screen, RUNGS, BUDGET   # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
E_SWEEP = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

BANDS = [
    dict(name="WIDE-e band 300-2000 (W3 band, da400, cap3000)",
         a_lo=6.671e6, a_hi=8.371e6, de_max=0.06, da_max_m=400e3, horizon=3000),
    dict(name="WIDE-a band 300-8000 (W4 band, da600, cap6000)",
         a_lo=6.671e6, a_hi=14.371e6, de_max=0.08, da_max_m=600e3, horizon=6000),
    dict(name="MID band 300-4000 (da500, cap4500)",
         a_lo=6.671e6, a_hi=10.371e6, de_max=0.07, da_max_m=500e3, horizon=4500),
]

rows = []
print(f"{'band':52} {'e_max':>6} {'di':>5} {'mode':>9} {'FEASseq':>8} {'FEAScomb':>9} "
      f"{'|de|p50':>8} {'dvcomb_p50':>10} {'p90':>6}")
for b in BANDS:
    for di in (1.0, 0.4):
        for e_max in E_SWEEP:
            cfg = dict(name=b["name"], a_lo=b["a_lo"], a_hi=b["a_hi"],
                       e_max_t=e_max, e_max_s=0.0, de_max=min(b["de_max"], e_max),
                       da_max_m=b["da_max_m"], horizon=b["horizon"],
                       same_orbit=False, valid_init_only=1)
            for mode in ("realized", "intended"):
                r = screen(cfg, di, n=N, evec_mode=mode)
                r["e_max"] = e_max
                r["band"] = b["name"]
                rows.append(r)
                print(f"{b['name'][:52]:52} {e_max:6.3f} {di:5.2f} {mode:>9} "
                      f"{100*r['frac_feasible_seq']:7.1f}% {100*r['frac_feasible_comb']:8.1f}% "
                      f"{r['de_p50']:8.4f} {r['dv_comb_p50']:10.0f} {r['dv_comb_p90']:6.0f}")

with open("/tmp/w3d_emax_scan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n=== max e_max at >=97% feasibility (no env change; 'realized' sampler) ===")
for b in BANDS:
    for di in (1.0, 0.4):
        for crit, key in (("comb", "frac_feasible_comb"), ("seq", "frac_feasible_seq")):
            best = None
            for r in sorted([x for x in rows if x["band"] == b["name"]
                             and x["di_max_deg"] == di
                             and x["evec_mode"] == "realized"],
                            key=lambda x: x["e_max"]):
                if r[key] >= 0.97:
                    best = r["e_max"]
            print(f"  {b['name'][:52]:52} di={di:.2f} {crit:>4}: e_max <= "
                  f"{best if best is not None else 'NONE'}")
print("\n=== same, if the sampler is fixed ('intended') ===")
for b in BANDS:
    for di in (1.0, 0.4):
        for crit, key in (("comb", "frac_feasible_comb"), ("seq", "frac_feasible_seq")):
            best = None
            for r in sorted([x for x in rows if x["band"] == b["name"]
                             and x["di_max_deg"] == di
                             and x["evec_mode"] == "intended"],
                            key=lambda x: x["e_max"]):
                if r[key] >= 0.97:
                    best = r["e_max"]
            print(f"  {b['name'][:52]:52} di={di:.2f} {crit:>4}: e_max <= "
                  f"{best if best is not None else 'NONE'}")
print("\nwrote /tmp/w3d_emax_scan.csv")
