"""Phase 5.5 P1 E3 — Cartesian↔elements round-trip at high altitude.

Extends scripts/orbital/p5e_e3_round_trip.py with an altitude axis.
The conversion uses vis-viva (1/a = 2/r - v²/μ) which numerically has
precision concerns at high a (1/a is small, subtraction near zero).
This probe quantifies the round-trip error at MEO and GEO.

Run:
    python3 scripts/orbital/p5_5_p1_e3_roundtrip_alt.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p5e_e2_kepler_precision import (
    MU,
    R_EARTH,
    cartesian_to_elements,
    orbit_to_cartesian,
)


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


# Same altitude cells as E2, single-point per band
ALT_CELLS = [
    ("LEO", R_EARTH + 500e3),
    ("MEO_low", R_EARTH + 1100e3),
    ("MEO", R_EARTH + 6000e3),
    ("MEO_high", R_EARTH + 20200e3),
    ("GEO", R_EARTH + 35786e3),
]

ECC_LEVELS = [0.05, 0.10, 0.20, 0.50, 0.70]


def main():
    print("=" * 78)
    print("Phase 5.5 P1 E3 — Cartesian↔elements round-trip at high altitude")
    print("=" * 78)
    print(f"{'band':<10} {'a (Mm)':>8} {'e':>6} {'n':>5} "
          f"{'max |Δa|':>11} {'max |Δe|':>11} {'max |Δω|':>11} {'max |Δθ|':>11}")
    print("-" * 90)

    rng = np.random.RandomState(42)
    for label, a in ALT_CELLS:
        for e in ECC_LEVELS:
            max_da = max_de = max_dom = max_dth = 0.0
            for _ in range(100):
                omega = rng.random() * 2 * math.pi
                theta = rng.random() * 2 * math.pi
                x, y, vx, vy = orbit_to_cartesian(a, e, omega, theta)
                a2, e2, theta2, omega2 = cartesian_to_elements(x, y, vx, vy)
                max_da = max(max_da, abs(a - a2))
                max_de = max(max_de, abs(e - e2))
                max_dom = max(max_dom, angle_diff(omega, omega2))
                max_dth = max(max_dth, angle_diff(theta, theta2))
            print(
                f"{label:<10} {a/1e6:>8.2f} {e:>6.2f} {100:>5} "
                f"{max_da:>11.2e} {max_de:>11.2e} {max_dom:>11.2e} {max_dth:>11.2e}"
            )
        print()

    print("Pass criterion: all errors < 1e-3 (any larger means vis-viva loses precision)")
    print("Note: at GEO, a=42.16e6 m so 1/a ~ 2.4e-8; subtraction precision matters.")


if __name__ == "__main__":
    main()
