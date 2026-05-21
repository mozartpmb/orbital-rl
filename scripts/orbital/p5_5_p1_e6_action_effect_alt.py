"""Phase 5.5 P1 E6 — Action effect at high altitude (doubles as P3).

Extends scripts/orbital/p5e_e6_action_effect.py with an altitude axis. For
each (altitude, action) pair, compute the Δa produced by one burn at
periapsis vs apoapsis using vis-viva. Compare to SUCCESS_TOL_A (rendezvous
tolerance, 10 km) and Φ_orbit gate threshold.

Action table from orbital.h:58-69:
    0 coast,    1 prograde 5,   2 prograde 10,   3 prograde 25,
    4 retrograde 5,   5 retrograde 10,   6 retrograde 25,
    7 radial out 10, 8 radial in 10,  9 warp

Per Phase 5.5 spec §2.3, this informs whether the Discrete(10) is adequate
across altitudes: at MEO/GEO does 5 m/s undershoot? Does 25 m/s overshoot?

Run:
    python3 scripts/orbital/p5_5_p1_e6_action_effect_alt.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from p5e_e6_action_effect import MU, R_EARTH, da_from_burn


SUCCESS_TOL_A = 10000.0   # m (rendezvous tolerance — Phase 4 setting)

ACTION_NAMES = [
    "coast",
    "pro+5",
    "pro+10",
    "pro+25",
    "retro-5",
    "retro-10",
    "retro-25",
    "rad+10",
    "rad-10",
    "warp5min",
]

# (dv_pro, dv_rad). Mirrors ACTION_DV from orbital.h:58-69.
ACTION_DV = [
    (0.0, 0.0),
    (5.0, 0.0),
    (10.0, 0.0),
    (25.0, 0.0),
    (-5.0, 0.0),
    (-10.0, 0.0),
    (-25.0, 0.0),
    (0.0, 10.0),
    (0.0, -10.0),
    (0.0, 0.0),  # warp
]

ALT_CELLS = [
    ("LEO",      R_EARTH + 500e3),
    ("MEO_low",  R_EARTH + 1100e3),
    ("MEO",      R_EARTH + 6000e3),
    ("MEO_high", R_EARTH + 20200e3),
    ("GEO",      R_EARTH + 35786e3),
]

ECC_LEVELS = [0.05, 0.20, 0.50]


def da_radial_burn(a, e, dv_rad, at_periapsis):
    """Δa from a radial burn (Δv perpendicular to velocity, along r̂).

    Radial burns change v² less efficiently than prograde, and the new v
    is sqrt(v_tan² + (v_rad + dv_rad)²) instead of v + dv_pro. At circular
    orbits (e=0), v_rad = 0, so v_new² = v² + dv_rad². For elliptical orbits,
    v_rad ≠ 0 except at peri/apo, but we evaluate at those extrema here
    where v_rad = 0 (which makes Δa_radial second-order in dv).
    """
    if at_periapsis:
        r = a * (1 - e)
    else:
        r = a * (1 + e)
    v_tan = math.sqrt(MU * (2.0/r - 1.0/a))
    v_new = math.sqrt(v_tan * v_tan + dv_rad * dv_rad)
    a_new = 1.0 / (2.0/r - v_new * v_new / MU)
    return a_new - a


def main():
    print("=" * 90)
    print("Phase 5.5 P1 E6/P3 — Action effect: Δa per Δv at altitude × eccentricity")
    print("=" * 90)
    print(f"SUCCESS_TOL_A = {SUCCESS_TOL_A/1000:.0f} km (rendezvous tolerance)")
    print()

    # Pass 1: Δa per action at e=0.0 (circular) for each altitude.
    # This is the simplest case — gives the "characteristic Δa per Δv" scale.
    print("--- Pass 1: Circular orbits (e=0), Δa per action (km) ---")
    print(f"{'band':<10} {'a (Mm)':>8} {'v_circ':>8}", end="")
    for ai in range(1, 9):  # skip coast (0) and warp (9)
        print(f" {ACTION_NAMES[ai]:>9}", end="")
    print()
    print("-" * 110)

    for label, a in ALT_CELLS:
        v_circ = math.sqrt(MU / a)
        print(f"{label:<10} {a/1e6:>8.2f} {v_circ:>8.0f}", end="")
        for ai in range(1, 9):
            dv_pro, dv_rad = ACTION_DV[ai]
            if dv_pro != 0:
                da = da_from_burn(a, 0.0, dv_pro, True)  # peri = apo for e=0
            else:
                da = da_radial_burn(a, 0.0, dv_rad, True)
            print(f" {da/1000:>9.1f}", end="")
        print()
    print()

    # Pass 2: peri-vs-apo asymmetry at high e for the bound-changing actions.
    print("--- Pass 2: peri vs apo asymmetry at e=0.50, prograde actions ---")
    print(f"{'band':<10} {'a (Mm)':>8} {'action':<10} "
          f"{'Δa @peri (km)':>14} {'Δa @apo (km)':>14} {'ratio':>7}")
    print("-" * 80)
    for label, a in ALT_CELLS:
        for ai in [1, 2, 3, 4, 5, 6]:  # all prograde + retrograde
            dv = ACTION_DV[ai][0]
            da_p = da_from_burn(a, 0.50, dv, True)
            da_a = da_from_burn(a, 0.50, dv, False)
            ratio = abs(da_p) / max(abs(da_a), 1.0)
            print(
                f"{label:<10} {a/1e6:>8.2f} {ACTION_NAMES[ai]:<10} "
                f"{da_p/1000:>14.1f} {da_a/1000:>14.1f} {ratio:>7.2f}"
            )
        print()

    # Pass 3: verdict — for each altitude band, identify if any action is
    # below SUCCESS_TOL_A (undershoot, lost in tolerance) or excessively
    # large (overshoot, can't fine-tune).
    print("--- Pass 3: Action adequacy verdict per altitude ---")
    print(f"{'band':<10} {'a (Mm)':>8} {'finest Δa (km)':>15} {'coarsest Δa (km)':>17} "
          f"{'undershoot':>11} {'overshoot':>11}")
    print("-" * 90)
    for label, a in ALT_CELLS:
        # Use prograde-5 (finest in-budget) and prograde-25 (coarsest)
        da_fine = da_from_burn(a, 0.0, 5.0, True)
        da_coarse = da_from_burn(a, 0.0, 25.0, True)
        undershoot = "YES" if abs(da_fine) < SUCCESS_TOL_A else "ok"
        # Overshoot: when single coarse burn moves more than 5× tolerance
        # = hard to fine-tune. Cutoff is heuristic.
        overshoot = "YES" if abs(da_coarse) > 50 * SUCCESS_TOL_A else "ok"
        print(
            f"{label:<10} {a/1e6:>8.2f} "
            f"{da_fine/1000:>15.1f} {da_coarse/1000:>17.1f} "
            f"{undershoot:>11} {overshoot:>11}"
        )
    print()
    print("undershoot YES = finest Δv (5 m/s) produces Δa < SUCCESS_TOL_A (10 km)")
    print("                 = burn effectively lost; agent has no fine-tune action")
    print("overshoot  YES = coarsest Δv (25 m/s) produces Δa > 50× SUCCESS_TOL_A (500 km)")
    print("                 = agent overshoots target by ≥50× tolerance per burn")
    print()
    print("Per spec §2.3: undershoot → add smaller actions; overshoot → smaller too")
    print("(no point adding LARGER actions if 25 m/s already overshoots).")


if __name__ == "__main__":
    main()
