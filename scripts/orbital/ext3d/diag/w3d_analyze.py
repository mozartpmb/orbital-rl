#!/usr/bin/env python3
"""ext-3d wide lineage: failure characterization tables from /tmp/w3d_episodes.csv."""
import csv
import math
import sys
from collections import Counter, defaultdict

import numpy as np

BUDGET = 478.13


def load():
    with open("/tmp/w3d_episodes.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("rung", "cause_name"):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def q(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=float), p)) if len(xs) else float("nan")


def sm(rows, key):
    xs = [r[key] for r in rows]
    if not xs:
        return "      n/a"
    return f"{np.mean(xs):7.2f} (p50 {q(xs,50):6.2f}, p90 {q(xs,90):6.2f})"


def bar(rows, title, keys):
    print(f"\n### {title}")
    hdr = f"{'group':>14} {'n':>4} {'succ%':>6} " + " ".join(f"{k:>10}" for k in keys)
    print(hdr)
    return hdr


def section(name):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)


def main():
    rows = load()
    for rung in ("W3", "W4"):
        R = [r for r in rows if r["rung"] == rung]
        S = [r for r in R if r["success"] == 1]
        F = [r for r in R if r["success"] == 0]
        section(f"{rung}  n={len(R)}  success={len(S)} ({100*len(S)/len(R):.1f}%)")

        print("\n-- cause histogram --")
        c = Counter(r["cause_name"] for r in R)
        for k, v in c.most_common():
            print(f"   {k:>12}  {v:4d}  {100*v/len(R):5.1f}%")

        print("\n-- successes vs failures --")
        keys = ["di_rel_deg", "e_tgt0", "e_sat0", "de0", "da0_km", "a_sat_km",
                "a_tgt_km", "band_pos", "rp_sat_km", "dlam0_deg",
                "dv_spent", "dv_frac_budget", "fuel_end", "exhausted",
                "n_decisions", "n_plane", "n_normal", "n_combined",
                "frac_plane", "frac_warp", "dv_normal_cmd",
                "need_seq", "need_comb", "margin_comb",
                "di_end_deg", "de_end", "da_end_km", "r_min_km", "steps"]
        print(f"{'field':>16} | {'SUCCESS mean (p50/p90)':>36} | {'FAIL mean (p50/p90)':>36}")
        for k in keys:
            a = [r[k] for r in S]
            b = [r[k] for r in F]
            fa = f"{np.mean(a):8.3f} ({q(a,50):7.3f}/{q(a,90):7.3f})" if a else "n/a"
            fb = f"{np.mean(b):8.3f} ({q(b,50):7.3f}/{q(b,90):7.3f})" if b else "n/a"
            print(f"{k:>16} | {fa:>36} | {fb:>36}")

        print("\n-- feasibility screen on the ACTUAL eval draws --")
        for lab, key in (("sequential", "feas_seq"), ("combined", "feas_comb")):
            fe = [r for r in R if r[key] == 1]
            inf = [r for r in R if r[key] == 0]
            sr_f = 100 * np.mean([r["success"] for r in fe]) if fe else float("nan")
            sr_i = 100 * np.mean([r["success"] for r in inf]) if inf else float("nan")
            print(f"   {lab:>10}: feasible {len(fe):3d}/{len(R)} ({100*len(fe)/len(R):5.1f}%)"
                  f"   succ|feasible {sr_f:5.1f}%   succ|infeasible {sr_i:5.1f}%")

        print("\n-- success rate by di_rel bin --")
        edges = [0, .2, .4, .6, .8, 1.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = [r for r in R if lo <= r["di_rel_deg"] < hi]
            if g:
                print(f"   di [{lo:.1f},{hi:.1f})°  n={len(g):3d}  succ {100*np.mean([x['success'] for x in g]):5.1f}%"
                      f"  need_comb p50 {q([x['need_comb'] for x in g],50):6.0f}"
                      f"  feas_comb {100*np.mean([x['feas_comb'] for x in g]):5.1f}%")

        print("\n-- success rate by e_target bin --")
        emax = 0.15 if rung == "W3" else 0.30
        edges = np.linspace(0, emax, 6)
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = [r for r in R if lo <= r["e_tgt0"] < hi]
            if g:
                print(f"   e_t [{lo:.3f},{hi:.3f})  n={len(g):3d}  succ {100*np.mean([x['success'] for x in g]):5.1f}%")

        print("\n-- success rate by altitude bin (sat) --")
        for lo, hi in ((300, 800), (800, 1400), (1400, 2000), (2000, 4000),
                       (4000, 6000), (6000, 8000)):
            g = [r for r in R if lo <= r["a_sat_km"] < hi]
            if g:
                print(f"   alt [{lo},{hi}) km  n={len(g):3d}  succ {100*np.mean([x['success'] for x in g]):5.1f}%"
                      f"  need_comb p50 {q([x['need_comb'] for x in g],50):6.0f}")

        print("\n-- success rate by |da0| bin --")
        dmax = 400 if rung == "W3" else 600
        edges = np.linspace(0, dmax, 5)
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = [r for r in R if lo <= r["da0_km"] < hi]
            if g:
                print(f"   |da| [{lo:.0f},{hi:.0f}) km  n={len(g):3d}  succ {100*np.mean([x['success'] for x in g]):5.1f}%")

        print("\n-- marginal-success surface: di x need_comb --")
        di_e = [0, 0.35, 0.6, 0.8, 1.0]
        nd_e = [0, 150, 250, 350, 450, 1e9]
        print(f"{'di\\need':>12} " + " ".join(f"{f'<{x:.0f}':>10}" for x in nd_e[1:]))
        for lo, hi in zip(di_e[:-1], di_e[1:]):
            cells = []
            for a, b in zip(nd_e[:-1], nd_e[1:]):
                g = [r for r in R if lo <= r["di_rel_deg"] < hi and a <= r["need_comb"] < b]
                cells.append(f"{100*np.mean([x['success'] for x in g]):5.0f}%/{len(g):<3d}" if g else "    -/0   ")
            print(f"{f'[{lo},{hi})':>12} " + " ".join(f"{c:>10}" for c in cells))

        print("\n-- plane-action usage --")
        for lab, sel in (("all", R), ("success", S), ("fail", F)):
            if not sel:
                continue
            print(f"   {lab:>8}: n_plane mean {np.mean([r['n_plane'] for r in sel]):7.2f} "
                  f"(normal {np.mean([r['n_normal'] for r in sel]):6.2f}, "
                  f"combined {np.mean([r['n_combined'] for r in sel]):6.2f}); "
                  f"episodes with >=1 plane action: "
                  f"{100*np.mean([r['n_plane'] > 0 for r in sel]):5.1f}%; "
                  f"di_end p50 {q([r['di_end_deg'] for r in sel],50):.4f}°")

        print("\n-- fuel --")
        exh = [r for r in R if r["exhausted"] == 1]
        print(f"   exhausted: {len(exh)}/{len(R)} ({100*len(exh)/len(R):.1f}%)  "
              f"succ|exhausted {100*np.mean([r['success'] for r in exh]) if exh else float('nan'):.1f}%")
        print(f"   dv_spent p50 {q([r['dv_spent'] for r in R],50):.1f} "
              f"p90 {q([r['dv_spent'] for r in R],90):.1f} max {max(r['dv_spent'] for r in R):.1f} "
              f"(budget {BUDGET:.1f})")
        cap = [r for r in R if r["cause_name"] == "safety_cap"]
        if cap:
            print(f"   cap-timeout episodes: dv_spent p50 {q([r['dv_spent'] for r in cap],50):.1f}, "
                  f"exhausted {100*np.mean([r['exhausted'] for r in cap]):.1f}%, "
                  f"r_min p50 {q([r['r_min_km'] for r in cap],50):.0f} km, "
                  f"di_end p50 {q([r['di_end_deg'] for r in cap],50):.4f}°, "
                  f"da_end p50 {q([r['da_end_km'] for r in cap],50):.1f} km, "
                  f"de_end p50 {q([r['de_end'] for r in cap],50):.4f}")

    # ── cross-rung: did W4 regress on W3-like (easy) draws? ──
    section("W4 REGRESSION TEST — success on draws inside W3's own envelope")
    W3 = [r for r in rows if r["rung"] == "W3"]
    W4 = [r for r in rows if r["rung"] == "W4"]

    def w3_like(r):
        return (r["e_tgt0"] <= 0.15 and r["e_sat0"] <= 0.21 and r["de0"] <= 0.06
                and r["da0_km"] <= 400 and r["a_sat_km"] <= 2000
                and r["a_tgt_km"] <= 2000)
    for lab, sub in (("W3 policy on W3 draws", W3), ("W4 policy on W4 draws", W4)):
        g = [r for r in sub if w3_like(r)]
        h = [r for r in sub if not w3_like(r)]
        print(f"  {lab:26}  W3-envelope subset n={len(g):3d} succ {100*np.mean([x['success'] for x in g]) if g else float('nan'):5.1f}%"
              f"   | outside n={len(h):3d} succ {100*np.mean([x['success'] for x in h]) if h else float('nan'):5.1f}%")

    print("\n  Strictly-easy subset (di<0.5°, e_t<0.08, |da|<250 km, alt<2000 km):")
    for lab, sub in (("W3", W3), ("W4", W4)):
        g = [r for r in sub if r["di_rel_deg"] < 0.5 and r["e_tgt0"] < 0.08
             and r["da0_km"] < 250 and r["a_sat_km"] < 2000]
        print(f"    {lab}: n={len(g):3d} succ {100*np.mean([x['success'] for x in g]) if g else float('nan'):5.1f}%"
              f"  feas_comb {100*np.mean([x['feas_comb'] for x in g]) if g else float('nan'):5.1f}%"
              f"  dv_spent p50 {q([x['dv_spent'] for x in g],50):6.1f}"
              f"  n_plane p50 {q([x['n_plane'] for x in g],50):5.0f}"
              f"  frac_warp p50 {q([x['frac_warp'] for x in g],50):5.2f}")

    print("\n  Feasible-only success (comb screen) by rung, and by feasible margin:")
    for lab, sub in (("W3", W3), ("W4", W4)):
        for mlo, mhi in ((0, 50), (50, 100), (100, 200), (200, 1e9)):
            g = [r for r in sub if r["feas_comb"] == 1 and mlo <= r["margin_comb"] < mhi]
            if g:
                print(f"    {lab} margin [{mlo},{mhi if mhi<1e8 else 'inf'}) m/s: n={len(g):3d}"
                      f" succ {100*np.mean([x['success'] for x in g]):5.1f}%")

    print("\n  Action-mix comparison (all episodes):")
    for lab, sub in (("W3", W3), ("W4", W4)):
        print(f"    {lab}: decisions p50 {q([r['n_decisions'] for r in sub],50):6.0f}"
              f"  frac_warp p50 {q([r['frac_warp'] for r in sub],50):.3f}"
              f"  n_plane p50 {q([r['n_plane'] for r in sub],50):5.0f}"
              f"  n_inplane p50 {q([r['n_inplane'] for r in sub],50):5.0f}"
              f"  dv_spent p50 {q([r['dv_spent'] for r in sub],50):6.1f}"
              f"  steps p50 {q([r['steps'] for r in sub],50):6.0f}")


if __name__ == "__main__":
    main()
