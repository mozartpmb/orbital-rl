"""E6 — Action effect at periapsis vs apoapsis at high e.

For a chaser on an elliptic orbit (a, e), apply prograde Δv at peri/apo and
compute the resulting Δa via vis-viva: 1/a = 2/r - v²/μ.

If ratio of |Δa(peri)| / |Δa(apo)| is large for the same Δv, the discrete
action set may be ill-suited for high-e control.
"""
import math

MU = 3.986004418e14
R_EARTH = 6.371e6


def da_from_burn(a, e, dv_pro, at_periapsis):
    """Compute Δa from a prograde burn."""
    if at_periapsis:
        r = a * (1 - e)
    else:
        r = a * (1 + e)
    v = math.sqrt(MU * (2.0/r - 1.0/a))
    v_new = v + dv_pro
    a_new = 1.0 / (2.0/r - v_new*v_new/MU)
    return a_new - a


def main():
    print(f"{'config':<25} {'Δv':>5} {'Δa @peri':>10} {'Δa @apo':>10} {'ratio':>7}")
    print("-" * 60)
    for e in [0.05, 0.10, 0.20, 0.50]:
        a = R_EARTH + 500e3
        for dv in [5.0, 10.0, 25.0]:
            da_peri = da_from_burn(a, e, dv, True)
            da_apo  = da_from_burn(a, e, dv, False)
            ratio = abs(da_peri) / max(abs(da_apo), 1e-9)
            print(f"e={e:.2f} a=500km        {dv:>5.0f} {da_peri/1000:>9.1f}km "
                  f"{da_apo/1000:>9.1f}km {ratio:>7.2f}")
        print()


if __name__ == "__main__":
    main()
