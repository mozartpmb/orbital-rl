"""Phase 5.5 P1 E1 — Lambert reachability at high altitudes.

Extends scripts/orbital/p5e_e1_lambert.py with an altitude axis. Per the
Phase 5.5 spec §2.1, we want to verify the Lambert solver works at higher
altitudes (MEO, GEO) where orbital periods are 2× to 16× longer than LEO and
velocities scale as a^(-1/2). The implementation should be altitude-agnostic
since it uses vis-viva and Stumpff functions in absolute units, but a probe
confirms it.

For each altitude band, sample (sat, target) pairs (with `valid_init_only`
filtering matching the env's c_reset behavior) and compute:
- doomed-by-R_EARTH rate (sub-surface perigee fraction)
- doomed-by-KEEPOUT rate (sub-keepout perigee fraction)
- Hohmann (circular surrogate) Δv vs fuel budget
- Lambert 2-impulse rendezvous Δv (random phasing) median + p10
- Fraction of cases fitting in fuel budget

Run:
    python3 scripts/orbital/p5_5_p1_e1_lambert_alt.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p5e_e1_lambert import (
    DV_BUDGET,
    EARTH_KEEPOUT,
    MU,
    R_EARTH,
    hohmann_dv,
    lambert_short_way,
    orbit_to_cartesian,
    solve_kepler,
    task_doomed_by_keepout,
    task_doomed_by_perigee,
    task_hohmann_dv,
    task_minimum_dv,
    true_from_eccentric,
    validate_lambert,
)


# Altitude bands per Phase 5.5 spec §2.1 — sat alt sampled from each band,
# target alt sampled independently from the same band (with the standard 50-km
# disjoint constraint mirrored from orbital.h c_reset).
#
# Each band is (label, alt_min_m, alt_max_m). For the LEO band we keep the
# legacy default 300-800 km for direct comparison to baseline E1 numbers.
ALT_BANDS = [
    ("LEO", 300e3, 800e3),
    ("MEO_low", 600e3, 1700e3),       # ~7000 km perigee floor; modest extension above LEO
    ("MEO", 5500e3, 6700e3),          # ~12000 km perigee floor
    ("MEO_high", 19500e3, 20700e3),   # ~26000 km perigee floor (GPS altitude)
    ("GEO", 35500e3, 36500e3),        # ~42000 km perigee (geostationary)
]


def sample_init_at_alt(e_max, alt_min, alt_max, valid_init_only=False,
                        max_attempts=4096, rng=None):
    """Sample (sat, target) like p5e_e1_lambert.sample_init but at an arbitrary
    altitude band. Matches the env's rejection-sampling cap from F1 (default
    4096) so cap-exhaust rates align with the env's behavior."""
    if rng is None:
        rng = np.random
    band_width = alt_max - alt_min
    for _ in range(max_attempts):
        # sat
        alt_init = alt_min + rng.random() * band_width
        a_sat = R_EARTH + alt_init
        e_sat = rng.random() * e_max
        omega_sat = rng.random() * 2 * math.pi
        M_sat = rng.random() * 2 * math.pi
        # target — distinct band (require ≥ 50 km Δa as in env)
        while True:
            alt_tgt = alt_min + rng.random() * band_width
            if abs(alt_tgt - alt_init) >= 50e3:
                break
        a_tgt = R_EARTH + alt_tgt
        e_tgt = rng.random() * e_max
        omega_tgt = rng.random() * 2 * math.pi
        M_tgt = rng.random() * 2 * math.pi
        if not valid_init_only:
            return (a_sat, e_sat, omega_sat, M_sat,
                    a_tgt, e_tgt, omega_tgt, M_tgt)
        if (a_sat * (1 - e_sat) >= EARTH_KEEPOUT and
            a_tgt * (1 - e_tgt) >= EARTH_KEEPOUT):
            return (a_sat, e_sat, omega_sat, M_sat,
                    a_tgt, e_tgt, omega_tgt, M_tgt)
    return None  # gave up


def task_minimum_dv_alt(task, T_grid_factor=1.0):
    """Compute minimum-Δv rendezvous, scaling the transfer-time grid with
    target orbital period. At GEO, the LEO-tuned T_grid of 300-10800 s is
    way too short — we need T values up to ~12 hr."""
    a_t = task[4]
    period = 2 * math.pi * math.sqrt(a_t ** 3 / MU)
    # Scale T_grid relative to LEO period (~5400 s) — at MEO/GEO use longer T
    T_grid = tuple(int(period * f) for f in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0])
    T_grid = tuple(max(t, 60) for t in T_grid)  # at least one timestep

    a_s, e_s, om_s, M_s, a_t, e_t, om_t, M_t = task
    th_s = true_from_eccentric(solve_kepler(M_s, e_s), e_s)
    x1, y1, vx1, vy1 = orbit_to_cartesian(a_s, e_s, om_s, th_s)
    best = float('inf')
    best_T = None
    n_t = math.sqrt(MU / a_t ** 3)
    for T in T_grid:
        M_t_T = (M_t + n_t * T) % (2 * math.pi)
        th_t_T = true_from_eccentric(solve_kepler(M_t_T, e_t), e_t)
        x2, y2, vx2_tgt, vy2_tgt = orbit_to_cartesian(a_t, e_t, om_t, th_t_T)
        sol = lambert_short_way((x1, y1), (x2, y2), T)
        if sol is None:
            continue
        v1_req, v2_req = sol
        dv1 = math.sqrt((v1_req[0] - vx1) ** 2 + (v1_req[1] - vy1) ** 2)
        dv2 = math.sqrt((v2_req[0] - vx2_tgt) ** 2 + (v2_req[1] - vy2_tgt) ** 2)
        total = dv1 + dv2
        if total < best:
            best = total
            best_T = T
    return best, best_T


def analyze_band(label, alt_min, alt_max, e_max, valid_init_only,
                 n_target=200, seed=42, lambert_subset=100):
    rng = np.random.RandomState(seed)
    tasks = []
    attempts = 0
    cap_exhaust = 0
    while len(tasks) < n_target and attempts < n_target * 50:
        t = sample_init_at_alt(e_max, alt_min, alt_max, valid_init_only,
                                max_attempts=4096, rng=rng)
        attempts += 1
        if t is None:
            cap_exhaust += 1
        else:
            tasks.append(t)
    if not tasks:
        return None
    doomed_r = sum(task_doomed_by_perigee(t) for t in tasks) / len(tasks) * 100
    doomed_ko = sum(task_doomed_by_keepout(t) for t in tasks) / len(tasks) * 100
    hohm_costs = [task_hohmann_dv(t) for t in tasks]
    hohm_over = sum(h > DV_BUDGET for h in hohm_costs) / len(tasks) * 100

    lamb_dvs = []
    for t in tasks[:lambert_subset]:
        dv, _ = task_minimum_dv_alt(t)
        if math.isfinite(dv):
            lamb_dvs.append(dv)
    if lamb_dvs:
        lamb_med = np.median(lamb_dvs)
        lamb_p10 = np.percentile(lamb_dvs, 10)
        lamb_under = sum(d < DV_BUDGET for d in lamb_dvs) / len(lamb_dvs) * 100
    else:
        lamb_med = lamb_p10 = lamb_under = float('nan')
    return dict(
        n=len(tasks), attempts=attempts, cap_exhaust=cap_exhaust,
        doomed_r=doomed_r, doomed_ko=doomed_ko,
        hohm_over=hohm_over, lamb_med=lamb_med,
        lamb_p10=lamb_p10, lamb_under=lamb_under,
    )


def main():
    print("=" * 78)
    print("Phase 5.5 P1 E1 — Lambert reachability at high altitudes")
    print("=" * 78)
    if not validate_lambert():
        print("ABORT: Lambert solver failed validation. Check implementation.")
        sys.exit(1)
    print(f"\nFuel-budget Δv: {DV_BUDGET:.1f} m/s\n")

    e_max_list = [0.05, 0.10, 0.20, 0.30, 0.50]
    print(f"{'band':<10} {'e_max':>6} {'mode':>16} {'n_eff':>6} {'attempts':>9} "
          f"{'doomed_R':>9} {'doomed_KO':>10} {'hohm>bud':>9} "
          f"{'lamb_med':>9} {'lamb_p10':>9} {'lamb<bud':>9}")
    print("-" * 130)
    for label, alt_min, alt_max in ALT_BANDS:
        for e_max in e_max_list:
            for valid_init in (False, True):
                mode = "valid_init" if valid_init else "raw"
                res = analyze_band(label, alt_min, alt_max, e_max, valid_init,
                                     n_target=200, seed=42)
                if res is None:
                    print(f"{label:<10} {e_max:>6.2f} {mode:>16} -- no tasks --")
                    continue
                print(
                    f"{label:<10} {e_max:>6.2f} {mode:>16} {res['n']:>6} {res['attempts']:>9} "
                    f"{res['doomed_r']:>8.1f}% {res['doomed_ko']:>9.1f}% "
                    f"{res['hohm_over']:>8.1f}% "
                    f"{res['lamb_med']:>9.0f} {res['lamb_p10']:>9.0f} "
                    f"{res['lamb_under']:>8.1f}%"
                )
        print()
    print("Notes:")
    print("  bands: LEO 300-800km, MEO_low 600-1700km, MEO 5.5-6.7Mm,")
    print("         MEO_high 19.5-20.7Mm (GPS), GEO 35.5-36.5Mm.")
    print("  attempts = total c_reset-style draws to fill n_target=200 tasks.")
    print("  cap_exhaust at 4096 attempts → None returned; counted in attempts.")
    print("  hohm>bud assumes circular orbits at (a_sat, a_tgt); upper bound for circular case.")
    print("  Lambert T_grid scaled with target orbital period (vs LEO-tuned in original E1).")


if __name__ == "__main__":
    main()
