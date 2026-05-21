"""Phase 5.5 P1 E2 — Kepler propagation precision at high altitude.

Extends scripts/orbital/p5e_e2_kepler_precision.py with an altitude axis.
Orbital periods scale as a^(3/2): LEO ~90 min, MEO ~6 hr, GEO ~24 hr.
Newton-Raphson convergence on Kepler's equation should be independent of a
(it operates on M and e only), but propagating for 100 periods at GEO means
100 days of sim time, so any floating-point error accumulation might show up.

Pass criterion: |Δa|, |Δe|, |Δω| < 1e-6 at each altitude × eccentricity cell.

Run:
    python3 scripts/orbital/p5_5_p1_e2_kepler_alt.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p5e_e2_kepler_precision import (
    DT,
    MU,
    R_EARTH,
    cartesian_to_elements,
    orbit_to_cartesian,
    propagate_orbit_py,
    solve_kepler,
    true_from_eccentric,
)


# (label, a_value_m). Center of each band, not a range — Kepler precision is
# evaluated at a single a per cell to keep the test focused.
ALT_CELLS = [
    ("LEO", R_EARTH + 500e3),       # ~6.87 Mm
    ("MEO_low", R_EARTH + 1100e3),
    ("MEO", R_EARTH + 6000e3),      # 12.37 Mm — near MEO low
    ("MEO_high", R_EARTH + 20200e3),  # 26.57 Mm — GPS altitude
    ("GEO", R_EARTH + 35786e3),     # 42.16 Mm — geostationary
]

ECC_LEVELS = [0.05, 0.10, 0.20, 0.50, 0.70]


def measure_drift(a0, e0, omega0=0.5, M0=0.3, n_periods=100):
    """Propagate for n_periods at the given (a, e), measuring round-trip
    drift in (a, e, ω)."""
    period = 2 * math.pi * math.sqrt(a0 * a0 * a0 / MU)
    n_steps = int(n_periods * period / DT)
    a, e, M, omega = a0, e0, M0, omega0
    max_da = max_de = max_dom = 0.0
    for k in range(n_steps):
        a, e, M, omega = propagate_orbit_py(a, e, M, omega, DT)
        E = solve_kepler(M, e)
        theta = true_from_eccentric(E, e)
        x, y, vx, vy = orbit_to_cartesian(a, e, omega, theta)
        a_rt, e_rt, theta_rt, omega_rt = cartesian_to_elements(x, y, vx, vy)
        max_da = max(max_da, abs(a - a_rt))
        max_de = max(max_de, abs(e - e_rt))
        dom = math.atan2(math.sin(omega - omega_rt), math.cos(omega - omega_rt))
        max_dom = max(max_dom, abs(dom))
    return n_steps, max_da, max_de, max_dom, period


def main():
    print("=" * 78)
    print("Phase 5.5 P1 E2 — Kepler precision at high altitude (100 periods)")
    print("=" * 78)
    print(f"{'band':<10} {'a (Mm)':>8} {'period (hr)':>12} {'e':>6} "
          f"{'n_steps':>9} {'max |Δa|':>11} {'max |Δe|':>11} {'max |Δω|':>11}")
    print("-" * 100)
    for label, a0 in ALT_CELLS:
        period = 2 * math.pi * math.sqrt(a0 ** 3 / MU)
        for e0 in ECC_LEVELS:
            n_steps, da, de, dom, _ = measure_drift(a0, e0, n_periods=100)
            print(
                f"{label:<10} {a0/1e6:>8.2f} {period/3600:>12.2f} "
                f"{e0:>6.2f} {n_steps:>9} "
                f"{da:>11.2e} {de:>11.2e} {dom:>11.2e}"
            )
        print()

    print("Pass criterion: |Δa|, |Δe|, |Δω| < 1e-6 at all cells")
    print("Note: at GEO, 100 periods = ~100 days of sim time; 144000 steps at DT=60s.")
    print("Note: at e=0.50, Kepler N-R converges in 2-4 iterations (validated at LEO).")
    print("      Higher altitude doesn't change the M/e dependence.")


if __name__ == "__main__":
    main()
