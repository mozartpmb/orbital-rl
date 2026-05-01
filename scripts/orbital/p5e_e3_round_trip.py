"""E3 — Cartesian↔elements round-trip at high e."""
import math, numpy as np
from p5e_e2_kepler_precision import (orbit_to_cartesian, cartesian_to_elements,
                                       solve_kepler, true_from_eccentric, MU, R_EARTH)


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def main():
    rng = np.random.RandomState(42)
    print(f"{'e':>6} {'n':>5} {'max |Δa|':>11} {'max |Δe|':>11} {'max |Δω|':>11} {'max |Δθ|':>11}")
    print("-" * 60)
    for e in [0.05, 0.10, 0.20, 0.50]:
        max_da = max_de = max_dom = max_dth = 0.0
        for _ in range(100):
            a = R_EARTH + 300e3 + rng.random() * 500e3
            omega = rng.random() * 2 * math.pi
            theta = rng.random() * 2 * math.pi
            x, y, vx, vy = orbit_to_cartesian(a, e, omega, theta)
            a2, e2, theta2, omega2 = cartesian_to_elements(x, y, vx, vy)
            max_da = max(max_da, abs(a - a2))
            max_de = max(max_de, abs(e - e2))
            max_dom = max(max_dom, angle_diff(omega, omega2))
            max_dth = max(max_dth, angle_diff(theta, theta2))
        print(f"{e:>6.2f} {100:>5} {max_da:>11.2e} {max_de:>11.2e} {max_dom:>11.2e} {max_dth:>11.2e}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    main()
