#!/usr/bin/env python3
"""N3D-A — compact table renderer for the prototype/CRLB CSVs."""
import csv
import math
import sys

RES = "/Users/pete/space_training/web_data/results/"

# NAV-G 2D baselines (reports/nav_G_filter.md, BO-BLS-MPC column), keyed by the
# 3D coplanar control that reproduces the same geometry.
NAVG_2D = {
    'N0_10km_coplanar':       ('G1 10 km drift',   6.2, 1226.0, 0.78, 0.91),
    'N2b_5km_burns_coplanar': ('G6 5 km + 1 m/s',  5.7,   17.6, 0.57, 0.98),
    'N3b_180deg_coplanar':    ('G3 180 deg gap', 415.0, 3257.0, 0.59, 0.97),
    'N4b_300km_coplanar':     ('G2 300 km drift',  8.2, 1229.0, 0.60, 0.97),
}


def f(x, n=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return '-'
    if not math.isfinite(v):
        return 'inf'
    return f"{v:.{n}g}"


def filt():
    rows = list(csv.DictReader(open(RES + 'n3d_filter.csv')))
    print("### Prototype — settled (last quarter) medians over seeds\n")
    print("| geometry | di (deg) | rho0 | rho_mean | arm | pos RMSE | vel RMSE "
          "| NEES | in | plane err (deg) | acq (min) | fail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | {f(r.get('di_deg'),3)} | "
              f"{f(float(r['rho0_m'])/1e3,4)} km | "
              f"{f(float(r['rho_mean_m'])/1e3,4)} km | {r['arm']} | "
              f"{f(r.get('pos_rmse_m'))} m | {f(r.get('vel_rmse_ms'))} m/s | "
              f"{f(r.get('nees'),3)} | {f(r.get('nees_in'),2)} | "
              f"{f(r.get('plane_err_deg'))} | {f(r.get('acq_min'),3)} | "
              f"{r.get('fails')} |")
    print("\n### 2D cross-validation — coplanar controls vs NAV-G's own numbers\n")
    print("| 3D coplanar control | NAV-G 2D geometry | RB 2D | RB 3D | "
          "BO-BLS-MPC 2D | BO3-BLS-MSC 3D | ratio |")
    print("|---|---|---|---|---|---|---|")
    for key, (nm, rb2, bo2, _, _) in NAVG_2D.items():
        rb3 = bo3 = None
        for r in rows:
            if r['scenario'] == key and r['arm'] == 'RB3-EKF':
                rb3 = float(r['pos_rmse_m'])
            if r['scenario'] == key and r['arm'] == 'BO3-BLS-MSC':
                bo3 = float(r['pos_rmse_m'])
        if rb3 is None or bo3 is None:
            continue
        print(f"| {key} | {nm} | {rb2:.1f} m | {rb3:.1f} m | {bo2:.1f} m | "
              f"{bo3:.1f} m | {bo3/bo2:.2f}x |")
    print("\n### 3D vs coplanar, paired (same da, same burns, same arc)\n")
    pairs = [('N2b_5km_burns_coplanar', 'N2_5km_burns_di0p02'),
             ('N0_10km_coplanar', 'N1_10km_di0p04'),
             ('N3b_180deg_coplanar', 'N3_180deg_di0p75'),
             ('N4b_300km_coplanar', 'N4_300km_di1p0')]
    print("| geometry pair | arm | coplanar pos | 3D pos | gain | "
          "coplanar vel | 3D vel | gain |")
    print("|---|---|---|---|---|---|---|---|")
    for a, b in pairs:
        for arm in ('BO3-BLS-MSC', 'RB3-EKF'):
            ra = next((r for r in rows if r['scenario'] == a
                       and r['arm'] == arm and r.get('pos_rmse_m')), None)
            rb = next((r for r in rows if r['scenario'] == b
                       and r['arm'] == arm and r.get('pos_rmse_m')), None)
            if not ra or not rb:
                continue
            pa, pb = float(ra['pos_rmse_m']), float(rb['pos_rmse_m'])
            va, vb = float(ra['vel_rmse_ms']), float(rb['vel_rmse_ms'])
            print(f"| {a.split('_')[0]}->{b.split('_')[0]} | {arm} | "
                  f"{pa:.4g} m | {pb:.4g} m | {pa/pb:.2f}x | {va:.4g} | "
                  f"{vb:.4g} | {va/vb:.2f}x |")


def crlb():
    rows = list(csv.DictReader(open(RES + 'n3d_crlb.csv')))
    print("\n### CRLB — sigma_range / sigma_vel / sigma_plane vs relative "
          "inclination\n")
    print("| family | di (deg) | dv | sig_rho | sig_rho/rho | sig_v | "
          "sig_plane (deg) | RB sig_plane | corr(rho,plane) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['family']} | {f(r['di_deg'],4)} | {f(r['dv_ms'],2)} | "
              f"{f(r['bo_sig_range_m'])} m | "
              f"{float(r['bo_sig_range_frac'])*100:.3g}% | "
              f"{f(r['bo_sig_vel_ms'])} m/s | {f(r['bo_sig_plane_deg'])} | "
              f"{f(r['rb_sig_plane_deg'])} | "
              f"{f(r['bo_corr_range_plane'],3)} |")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'filter'):
        filt()
    if which in ('all', 'crlb'):
        crlb()
