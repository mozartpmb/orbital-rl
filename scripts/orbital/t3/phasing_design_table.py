#!/usr/bin/env python3
"""
Classical co-orbital phasing-maneuver design table, specialized to the orbital RL env.

Two-impulse phasing maneuver (Vallado Ch.6 / Curtis Ch.6 "phasing maneuvers"):
  chaser and target share a circular orbit of radius a; chaser trails/leads by
  phase angle dtheta.  The chaser burns once to enter a phasing ellipse whose
  period is T_ph = T * (1 -/+ dtheta/(2*pi*N)), coasts N revs on that ellipse,
  then burns again (equal and opposite magnitude) to re-circularize -- by which
  time the phase error has been fully absorbed.

  Both impulses occur at the same point (apse of the phasing ellipse == the
  original circular radius), so
      dv_total = 2 * | sqrt(mu*(2/a - 1/a_ph)) - sqrt(mu/a) |

Linearized drift rate (the quantity the RL shaping actually needs to encode):
      d(phase)/dt  =  n_chaser - n_target  ~=  -1.5 * n * (da / a)

Outputs CSV to web_data/results/t3_phasing_design_table.csv
"""
import csv
import math
import os

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0                 # sim seconds per sub-step
MAX_STEPS = 2000          # agent decisions
DV_BUDGET = 478.0         # m/s, 15% fuel fraction, Isp 300 s

OUT = "/Users/pete/space_training/web_data/results/t3_phasing_design_table.csv"


def period(a):
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU)


def phasing_maneuver(a, dtheta_rad, n_revs):
    """Return (a_ph, da, dv_total, t_total) for closing dtheta over n_revs.

    dtheta_rad > 0 means the chaser must GAIN phase (it is behind), so it drops
    to a faster (smaller) phasing orbit.
    """
    T = period(a)
    # time the chaser must "save" (or lose) relative to the target
    T_ph = T * (1.0 - dtheta_rad / (2.0 * math.pi * n_revs))
    if T_ph <= 0:
        return None
    a_ph = (MU * (T_ph / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
    # radius of periapsis/apoapsis at the burn point is a (the original circle)
    v_circ = math.sqrt(MU / a)
    arg = MU * (2.0 / a - 1.0 / a_ph)
    if arg <= 0:
        return None
    v_ph = math.sqrt(arg)
    dv = 2.0 * abs(v_ph - v_circ)
    return a_ph, a_ph - a, dv, n_revs * T_ph


def drift_rate_deg_per_hr(a, da):
    n = math.sqrt(MU / a ** 3)
    return math.degrees(-1.5 * n * (da / a)) * 3600.0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    alts_km = [300.0, 550.0, 800.0]
    gaps_deg = [15.0, 30.0, 60.0, 90.0, 120.0, 180.0]
    revs = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100]

    for alt in alts_km:
        a = R_EARTH + alt * 1e3
        T = period(a)
        for g in gaps_deg:
            for N in revs:
                r = phasing_maneuver(a, math.radians(g), N)
                if r is None:
                    continue
                a_ph, da, dv, t = r
                # perigee altitude of the phasing ellipse (burn point is apoapsis
                # when lowering): r_p = 2*a_ph - a
                r_p = 2.0 * a_ph - a
                peri_alt_km = (r_p - R_EARTH) / 1e3
                rows.append(dict(
                    alt_km=alt,
                    a_km=a / 1e3,
                    period_min=T / 60.0,
                    phase_gap_deg=g,
                    n_revs=N,
                    da_km=da / 1e3,
                    a_ph_km=a_ph / 1e3,
                    phasing_peri_alt_km=peri_alt_km,
                    dv_total_ms=dv,
                    dv_frac_of_budget=dv / DV_BUDGET,
                    t_total_hr=t / 3600.0,
                    drift_deg_per_hr=drift_rate_deg_per_hr(a, da),
                    decisions_if_60s=t / DT,
                    decisions_if_warp5min=t / (5 * DT),
                    decisions_if_warp1hr=t / 3600.0,
                    feasible_dv=dv <= DV_BUDGET,
                    feasible_60s_horizon=(t / DT) <= MAX_STEPS,
                    feasible_warp5_horizon=(t / (5 * DT)) <= MAX_STEPS,
                    reentry_risk=peri_alt_km < 200.0,
                ))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")

    # ------- console summary: cheapest feasible design per (alt, gap) -------
    print("\n== cheapest DV design that fits a 2000-decision episode ==")
    for horizon_key, label in (("decisions_if_60s", "coast-only (60 s steps)"),
                               ("decisions_if_warp5min", "warp-5min steps"),
                               ("decisions_if_warp1hr", "warp-1hr steps")):
        print(f"\n-- horizon model: {label} --")
        print(f"{'alt':>5} {'gap':>5} {'N':>4} {'da_km':>9} {'dv_m/s':>8} "
              f"{'t_hr':>8} {'periAlt':>8}")
        for alt in alts_km:
            for g in gaps_deg:
                cand = [r for r in rows
                        if r["alt_km"] == alt and r["phase_gap_deg"] == g
                        and r[horizon_key] <= MAX_STEPS
                        and not r["reentry_risk"]]
                if not cand:
                    print(f"{alt:5.0f} {g:5.0f}   -- infeasible --")
                    continue
                b = min(cand, key=lambda r: r["dv_total_ms"])
                print(f"{alt:5.0f} {g:5.0f} {b['n_revs']:4d} {b['da_km']:9.2f} "
                      f"{b['dv_total_ms']:8.2f} {b['t_total_hr']:8.2f} "
                      f"{b['phasing_peri_alt_km']:8.1f}")

    # ------- tolerance analysis: what does the 30 km / 50 m/s box mean -------
    a = R_EARTH + 550e3
    v = math.sqrt(MU / a)
    n = math.sqrt(MU / a ** 3)
    print("\n== success-box interpretation at 550 km ==")
    print(f"v_circ = {v:.1f} m/s, period = {period(a)/60:.1f} min, "
          f"n = {math.degrees(n)*3600:.1f} deg/hr")
    print(f"30 km along-track  = {math.degrees(30e3/a):.4f} deg of phase")
    print(f"da giving 50 m/s rel-vel (|dv|~0.5*v*da/a): "
          f"{2*a*50/v/1e3:.1f} km")
    print(f"de giving 50 m/s rel-vel (|dv|~e*v): {50/v:.5f}")
    print(f"de giving 30 km radial excursion (a*e): {30e3/a:.5f}")
    print(f"drift rate at da=1 km:   {drift_rate_deg_per_hr(a, 1e3):.4f} deg/hr")
    print(f"drift rate at da=10 km:  {drift_rate_deg_per_hr(a, 10e3):.3f} deg/hr")
    print(f"drift rate at da=100 km: {drift_rate_deg_per_hr(a, 100e3):.2f} deg/hr")
    print(f"drift rate at da=200 km: {drift_rate_deg_per_hr(a, 200e3):.2f} deg/hr")
    # dv to establish a given da (two-impulse round trip = 2x one-way Hohmann)
    for da_km in (10, 50, 100, 200, 340):
        da = da_km * 1e3
        a2 = a + da
        # one-way Hohmann a -> a2
        at = 0.5 * (a + a2)
        dv1 = abs(math.sqrt(MU * (2 / a - 1 / at)) - math.sqrt(MU / a))
        dv2 = abs(math.sqrt(MU / a2) - math.sqrt(MU * (2 / a2 - 1 / at)))
        print(f"da={da_km:4d} km: one-way Hohmann dv={dv1+dv2:6.2f} m/s, "
              f"round trip={2*(dv1+dv2):6.2f} m/s, "
              f"drift={drift_rate_deg_per_hr(a, da):6.2f} deg/hr, "
              f"180 deg closes in {180/abs(drift_rate_deg_per_hr(a, da)):7.2f} hr")


def capture_window():
    """How long is the terminal capture window as a function of drift depth?

    Success needs |dr| < 30 km, i.e. |phase error| < 30 km / a in radians.
    While drifting at rate w = 1.5*n*da/a, the chaser is inside that window for
    t_window = 2 * (30 km / a) / w.  Compare against the granularity of each
    action (60 s coast, 5 min warp, 30 min warp, 1 hr warp).
    """
    out = "/Users/pete/space_training/web_data/results/t3_phasing_capture_window.csv"
    a = R_EARTH + 550e3
    rows = []
    for da_km in (1, 2, 5, 10, 20, 23, 50, 100, 116, 200, 340):
        da = da_km * 1e3
        w_deg_hr = abs(drift_rate_deg_per_hr(a, da))
        win_deg = 2.0 * math.degrees(30e3 / a)
        t_win_s = win_deg / w_deg_hr * 3600.0
        rows.append(dict(
            da_km=da_km,
            drift_deg_per_hr=w_deg_hr,
            capture_window_s=t_win_s,
            capture_window_min=t_win_s / 60.0,
            n_60s_steps_in_window=t_win_s / 60.0,
            n_warp5_steps_in_window=t_win_s / 300.0,
            n_warp30_steps_in_window=t_win_s / 1800.0,
            n_warp60_steps_in_window=t_win_s / 3600.0,
            hours_to_close_180deg=180.0 / w_deg_hr,
            roundtrip_dv_ms=2 * hohmann_dv(a, a + da),
        ))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")
    print(f"{'da_km':>6} {'deg/hr':>8} {'window_min':>11} {'#60s':>7} "
          f"{'#w5':>7} {'#w30':>7} {'#w60':>7} {'hr_for_180':>11} {'rt_dv':>8}")
    for r in rows:
        print(f"{r['da_km']:6d} {r['drift_deg_per_hr']:8.3f} "
              f"{r['capture_window_min']:11.2f} {r['n_60s_steps_in_window']:7.2f} "
              f"{r['n_warp5_steps_in_window']:7.2f} "
              f"{r['n_warp30_steps_in_window']:7.3f} "
              f"{r['n_warp60_steps_in_window']:7.3f} "
              f"{r['hours_to_close_180deg']:11.2f} {r['roundtrip_dv_ms']:8.2f}")


def hohmann_dv(a1, a2):
    at = 0.5 * (a1 + a2)
    dv1 = abs(math.sqrt(MU * (2 / a1 - 1 / at)) - math.sqrt(MU / a1))
    dv2 = abs(math.sqrt(MU / a2) - math.sqrt(MU * (2 / a2 - 1 / at)))
    return dv1 + dv2


if __name__ == "__main__":
    main()
    capture_window()
