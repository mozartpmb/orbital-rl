"""E5 — Φ_orbit calibration at high e.

Φ_orbit per orbital.h:compute_phi:
    Φ_orbit = |Δa|/SUCCESS_TOL_A + ||Δē||
where Δē = (e_s·cos ω_s, e_s·sin ω_s) − (e_t·cos ω_t, e_t·sin ω_t).

σ₂ activates when Φ_orbit < EPS_ORBIT = 2.0.
"""
import math
SUCCESS_TOL_A = 10000.0
EPS_ORBIT = 2.0


def phi_orbit(a_s, e_s, om_s, a_t, e_t, om_t):
    da = abs(a_s - a_t) / SUCCESS_TOL_A
    e_sx = e_s * math.cos(om_s); e_sy = e_s * math.sin(om_s)
    e_tx = e_t * math.cos(om_t); e_ty = e_t * math.sin(om_t)
    de = math.sqrt((e_sx-e_tx)**2 + (e_sy-e_ty)**2)
    return da, de, da + de


def main():
    R = 6.371e6
    print(f"{'config':<30} {'da':>8} {'de':>8} {'Φ':>8} {'σ₂ on?':>8}")
    print("-" * 64)
    cases = [
        ("almost-rendezvous e=0.05",  R+500e3, 0.05, 0.0,  R+500e3, 0.05, 0.0),
        ("almost-rendezvous e=0.20",  R+500e3, 0.20, 0.0,  R+500e3, 0.20, 0.0),
        ("Δa=10km e=0.05",            R+500e3, 0.05, 0.0,  R+490e3, 0.05, 0.0),
        ("Δa=10km e=0.20",            R+500e3, 0.20, 0.0,  R+490e3, 0.20, 0.0),
        ("perp ω e=0.05",             R+500e3, 0.05, 0.0,  R+500e3, 0.05, math.pi/2),
        ("perp ω e=0.20",             R+500e3, 0.20, 0.0,  R+500e3, 0.20, math.pi/2),
        ("opposite ω e=0.20",         R+500e3, 0.20, 0.0,  R+500e3, 0.20, math.pi),
        ("Δa=500km e=0.05",           R+800e3, 0.05, 0.0,  R+300e3, 0.05, 0.0),
        ("Δa=500km e=0.20",           R+800e3, 0.20, 0.0,  R+300e3, 0.20, 0.0),
        ("worst-case e=0.20",         R+800e3, 0.20, 0.0,  R+300e3, 0.20, math.pi),
    ]
    for name, a_s, e_s, om_s, a_t, e_t, om_t in cases:
        da, de, phi = phi_orbit(a_s, e_s, om_s, a_t, e_t, om_t)
        gate = "ON" if phi < EPS_ORBIT else "off"
        print(f"{name:<30} {da:>8.3f} {de:>8.3f} {phi:>8.3f} {gate:>8}")


if __name__ == "__main__":
    main()
