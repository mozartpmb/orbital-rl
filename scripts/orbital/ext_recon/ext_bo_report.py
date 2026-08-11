"""Compact table from ext_bo_filter.csv — the comparison the memo quotes."""
import csv
import sys

CSV = "/Users/pete/space_training/web_data/results/ext_bo_filter.csv"
ORDER = ['RB-EKF', 'BO-EKF', 'BO-UKF', 'BO-EKF-o', 'BO-MPC-o', 'BO-RPB',
         'BO-BLS', 'BO-BLS-MPC']


def f(x, d='nan'):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(d)


def main():
    rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else CSV)))
    scen = []
    for r in rows:
        if r['scenario'] not in scen:
            scen.append(r['scenario'])
    for s in scen:
        rs = [r for r in rows if r['scenario'] == s]
        head = rs[0]
        print(f"\n## {s}   rho_mean {f(head['rho_km_mean']):.0f} km  "
              f"rho/r {f(head['rho_over_r']):.4f}  "
              f"CRLB sigLOS BO {f(head['crlb_los_m']):.1f} m / "
              f"RB {f(head['crlb_los_rb_m']):.1f} m")
        print(f"| {'filter':<11} | {'posRMSE m':>10} | {'velRMSE m/s':>11} | "
              f"{'relRange':>9} | {'NEES med':>9} | {'in-bnd':>6} | "
              f"{'acq min':>7} | {'div':>4} |")
        print(f"|{'-'*13}|{'-'*12}|{'-'*13}|{'-'*11}|{'-'*11}|{'-'*8}|"
              f"{'-'*9}|{'-'*6}|")
        for k in ORDER:
            m = [r for r in rs if r['filt'] == k]
            if not m:
                continue
            r = m[0]
            print(f"| {k:<11} | {f(r['pos_rmse_settled_m']):10.1f} | "
                  f"{f(r['vel_rmse_settled_ms']):11.4f} | "
                  f"{f(r['rel_range_err_settled']):9.2e} | "
                  f"{f(r['nees_med']):9.2f} | {f(r['nees_in_bounds']):6.2f} | "
                  f"{f(r.get('acq_min', 'nan')):7.1f} | "
                  f"{f(r['diverged_frac']):4.2f} |")


if __name__ == '__main__':
    main()
