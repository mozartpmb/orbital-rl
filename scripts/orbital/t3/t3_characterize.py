#!/usr/bin/env python3
"""T3 headline characterization from eval_checkpoint.py trajectory npz files.

Per episode: initial physical mean-longitude gap, initial da, target e,
total delta-v, sim steps, residual |v_rel| and distance at capture, action mix.
Binned success + cost tables (red-team #6: report per-gap-bin success and
residual rel-vel at capture so ballistic intercept vs rendezvous is visible).

Usage: python3 t3_characterize.py <npz_dir> <out_csv>
"""
import glob, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nav'))
from orbital_math import true_to_mean  # correct inverse (post-fix)

MU = 3.986004418e14
R_E = 6.371e6

def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

def episode_row(f):
    d = np.load(f)
    steps = int(d['episode_steps'][0]) if 'episode_steps' in d else len(d['sat_a'])
    cause = int(d['terminal_cause'][0]) if 'terminal_cause' in d else -1
    n = len(d['sat_a'])
    # initial physical mean-longitude gap
    th_s, om_s, e_s, a_s = (float(d[k][0]) for k in ('sat_theta', 'sat_omega', 'sat_e', 'sat_a'))
    a_t, e_t, om_t = (float(d[k][0]) for k in ('target_a', 'target_e', 'target_omega'))
    tx, ty = float(d['target_x'][0]), float(d['target_y'][0])
    th_t = math.atan2(ty, tx) - om_t
    lam_s = true_to_mean(th_s, e_s) + om_s
    lam_t = true_to_mean(th_t, e_t) + om_t
    gap0 = math.degrees(wrap_pi(lam_s - lam_t))
    # terminal residuals (last record)
    i = n - 1
    dvx = float(d['sat_vx'][i] - d['target_vx'][i])
    dvy = float(d['sat_vy'][i] - d['target_vy'][i])
    rvel = math.hypot(dvx, dvy)
    dist = math.hypot(float(d['sat_x'][i] - d['target_x'][i]),
                      float(d['sat_y'][i] - d['target_y'][i]))
    dv_total = float(np.sum(d['delta_v']))
    n_burn_records = int(np.sum(d['delta_v'] > 0))
    acts = d['action'].astype(int)
    warp_share = float(np.mean((acts >= 9) & (acts <= 11)))
    # Hohmann between initial orbits (shape-only lower bound component)
    r1, r2 = a_s, a_t
    v1, v2 = math.sqrt(MU / r1), math.sqrt(MU / r2)
    at = 0.5 * (r1 + r2)
    hoh = abs(v1 * (math.sqrt(r2 / at) - 1)) + abs(v2 * (1 - math.sqrt(r1 / at)))
    evec = 0.5 * math.sqrt(MU / a_t) * math.hypot(
        e_s * math.cos(om_s) - e_t * math.cos(om_t),
        e_s * math.sin(om_s) - e_t * math.sin(om_t))
    return dict(file=os.path.basename(f), success=int(cause == 1), cause=cause,
                gap0_deg=round(gap0, 2), da0_km=round((a_s - a_t) / 1e3, 1),
                e_t=round(e_t, 4), e_s=round(e_s, 4), steps=steps,
                dv=round(dv_total, 1), hohmann=round(hoh, 1),
                shape_lb=round(max(hoh, evec), 1),
                burn_records=n_burn_records, warp_share=round(warp_share, 3),
                final_dist_km=round(dist / 1e3, 2), final_rvel=round(rvel, 2))

def q(v, p):
    return float(np.percentile(v, p))

def main():
    npz_dir, out_csv = sys.argv[1], sys.argv[2]
    rows = [episode_row(f) for f in sorted(glob.glob(os.path.join(npz_dir, '*.npz')))]
    import csv as _csv
    with open(out_csv, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"n={len(rows)}  success={sum(r['success'] for r in rows)}  -> {out_csv}")

    print("\nby |initial dlambda| bin:")
    for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
        b = [r for r in rows if lo <= abs(r['gap0_deg']) < hi]
        if not b: continue
        sr = sum(r['success'] for r in b) / len(b)
        dvs = [r['dv'] for r in b if r['success']]
        st = [r['steps'] for r in b if r['success']]
        print(f"  {lo:3d}-{hi:3d} deg  n={len(b):3d}  success={sr:6.1%}  "
              f"dv p50={np.median(dvs):5.0f}  steps p50={np.median(st):4.0f}")
    print("\nby e_target bin:")
    for lo, hi in ((0, .0125), (.0125, .025), (.025, .0375), (.0375, .051)):
        b = [r for r in rows if lo <= r['e_t'] < hi]
        if not b: continue
        sr = sum(r['success'] for r in b) / len(b)
        dvs = [r['dv'] for r in b if r['success']]
        print(f"  {lo:.4f}-{hi:.4f}  n={len(b):3d}  success={sr:6.1%}  dv p50={np.median(dvs):5.0f}")

    s = [r for r in rows if r['success']]
    dv = [r['dv'] for r in s]; rl = [r['final_rvel'] for r in s]
    fd = [r['final_dist_km'] for r in s]; ws = [r['warp_share'] for r in s]
    ratio = [r['dv'] / r['shape_lb'] for r in s if r['shape_lb'] > 5]
    print(f"\nsuccesses n={len(s)}")
    print(f"  dv:          p10={q(dv,10):.0f}  p50={q(dv,50):.0f}  p90={q(dv,90):.0f}  max={max(dv):.0f} m/s (budget 478)")
    print(f"  dv/shape_lb: p10={q(ratio,10):.2f} p50={q(ratio,50):.2f} p90={q(ratio,90):.2f}")
    print(f"  capture residual |v_rel|: p10={q(rl,10):.1f} p50={q(rl,50):.1f} p90={q(rl,90):.1f} (box 50)")
    print(f"  capture distance:         p10={q(fd,10):.1f} p50={q(fd,50):.1f} p90={q(fd,90):.1f} km (box 30)")
    print(f"  warp share of sim records: p50={q(ws,50):.2f}")

if __name__ == '__main__':
    main()
