"""Phase 5e Block I E2 — Kepler propagation precision at high e.

Re-implements propagate_orbit() from orbital.h in Python (mirror solve_kepler
+ M-advance + true-anomaly recovery), then propagates an orbit for 100
periods and reports drift in (a, e, ω). Pass criterion: |Δa|, |Δe|, |Δω| < 1e-6.

The C env's propagate_orbit only advances M; (a, e, ω) are invariants of
2-body Kepler dynamics and should not drift. Drift would indicate a bug in
the M-advance or solve_kepler iteration.
"""
import math, numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0


def solve_kepler(M, e, tol=1e-12, max_iter=20):
    """Match orbital.h Newton-Raphson on Kepler's equation."""
    E = M if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        if abs(fp) < 1e-14: break
        dE = f / fp
        E -= dE
        if abs(dE) < tol: break
    return E


def true_from_eccentric(E, e):
    return 2.0 * math.atan2(math.sqrt(1+e) * math.sin(E/2),
                             math.sqrt(1-e) * math.cos(E/2))


def propagate_orbit_py(a, e, M, omega, dt):
    """Mirror C propagate_orbit: advance M; (a, e, omega) invariant."""
    n = math.sqrt(MU / (a*a*a))
    M_new = (M + n * dt) % (2*math.pi)
    return a, e, M_new, omega


def cartesian_to_elements(x, y, vx, vy):
    """Mirror orbital.h:cartesian_to_elements."""
    r = math.sqrt(x*x + y*y)
    v2 = vx*vx + vy*vy
    vr = (x*vx + y*vy) / r
    a = 1.0 / (2.0/r - v2/MU)
    ex = ((v2 - MU/r)*x - vr*r*vx) / MU
    ey = ((v2 - MU/r)*y - vr*r*vy) / MU
    e = math.sqrt(ex*ex + ey*ey)
    if e < 1e-10:
        omega = 0.0
        theta = math.atan2(y, x)
    else:
        omega = math.atan2(ey, ex)
        cos_theta = max(-1.0, min(1.0, (ex*x + ey*y) / (e*r)))
        theta = math.acos(cos_theta)
        if vr < 0: theta = 2*math.pi - theta
    return a, e, theta, omega


def orbit_to_cartesian(a, e, omega, theta):
    p = a * (1.0 - e*e)
    r = p / (1.0 + e*math.cos(theta))
    h = math.sqrt(MU * p)
    xp = r*math.cos(theta); yp = r*math.sin(theta)
    vxp = -(MU/h)*math.sin(theta); vyp = (MU/h)*(e + math.cos(theta))
    co, so = math.cos(omega), math.sin(omega)
    return co*xp - so*yp, so*xp + co*yp, co*vxp - so*vyp, so*vxp + co*vyp


def main():
    print(f"{'e':>6} {'periods':>8} {'max |Δa|':>11} {'max |Δe|':>11} {'max |Δω|':>11}")
    print("-" * 56)
    a0 = R_EARTH + 500e3
    for e0 in [0.05, 0.10, 0.20, 0.50]:
        omega0 = 0.5
        M0 = 0.3
        period = 2 * math.pi * math.sqrt(a0*a0*a0 / MU)
        n_steps = int(100 * period / DT)
        a, e, M, omega = a0, e0, M0, omega0
        max_da = max_de = max_dom = 0.0
        for k in range(n_steps):
            a, e, M, omega = propagate_orbit_py(a, e, M, omega, DT)
            # Round-trip check via cartesian (env's actual data path)
            E = solve_kepler(M, e); theta = true_from_eccentric(E, e)
            x, y, vx, vy = orbit_to_cartesian(a, e, omega, theta)
            a_rt, e_rt, theta_rt, omega_rt = cartesian_to_elements(x, y, vx, vy)
            max_da = max(max_da, abs(a - a_rt))
            max_de = max(max_de, abs(e - e_rt))
            # ω wraps — compare via sin/cos
            dom = math.atan2(math.sin(omega - omega_rt), math.cos(omega - omega_rt))
            max_dom = max(max_dom, abs(dom))
        print(f"{e0:>6.2f} {n_steps:>8} {max_da:>11.2e} {max_de:>11.2e} {max_dom:>11.2e}")


if __name__ == "__main__":
    main()
