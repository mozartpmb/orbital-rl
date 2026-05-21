"""Phase 5.5 P1 E5 — Φ_orbit calibration at high altitude with K-variations.

Extends scripts/orbital/p5e_e5_phi_calib.py with two new axes:
1. Altitude band (governing Δa magnitudes between sat and target).
2. phi_orbit_scale_k (the F5/B.2 env-fix kwarg added 2026-05-12).

Φ_orbit per orbital.h:compute_phi after the B.2 update:
    phi_tol_eff = max(SUCCESS_TOL_A, K * obs_alt_scale_m)
    Φ_orbit = |Δa| / phi_tol_eff + ||Δē||

σ₂ gate activates when Φ_orbit < EPS_ORBIT = 2.0.

We want: for each altitude band, find a K that keeps worst-case Φ_orbit
in O(1-50) so the gate threshold remains operational.

Run:
    python3 scripts/orbital/p5_5_p1_e5_phi_calib_alt.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from p5e_e5_phi_calib import phi_orbit, EPS_ORBIT, SUCCESS_TOL_A


# (label, a_center_m, half_band_m) — used to derive worst-case Δa
ALT_BANDS = [
    ("LEO",      6671e3, 250e3),    # 300-800 km altitude, mid 550 km, half 250 km
    ("MEO_low",  7421e3, 550e3),    # 600-1700 km, mid 1150 km, half 550 km
    ("MEO",      12371e3, 600e3),   # 5500-6700 km alt, mid 6000 km, half 600 km
    ("MEO_high", 26571e3, 600e3),
    ("GEO",      42157e3, 500e3),
]

# (label, obs_alt_scale_m) — pairs with the new B.1 kwarg.
OBS_SCALES = [
    ("LEO_default", 1.6e6),
    ("GEO_default", 4.2e7),
]

# K values to scan
K_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]


def phi_orbit_scaled(a_s, e_s, om_s, a_t, e_t, om_t, K, obs_alt_scale_m):
    """Mirror compute_phi() after B.2 with configurable K + obs_alt_scale_m."""
    tol_eff = max(SUCCESS_TOL_A, K * obs_alt_scale_m)
    da_term = abs(a_s - a_t) / tol_eff
    e_sx = e_s * math.cos(om_s); e_sy = e_s * math.sin(om_s)
    e_tx = e_t * math.cos(om_t); e_ty = e_t * math.sin(om_t)
    de = math.sqrt((e_sx - e_tx) ** 2 + (e_sy - e_ty) ** 2)
    return da_term, de, da_term + de, tol_eff


def cell_worst_phi(a_center, half_band, e_max, K, obs_alt_scale_m):
    """Worst-case Φ_orbit for an (alt band, e_max) cell.

    Worst case: sat at a_center+half_band, target at a_center-half_band,
    perpendicular eccentricity vectors (max de). Returns scalar Φ."""
    a_s = a_center + half_band
    a_t = a_center - half_band
    # max de: e_s along +x, e_t along +y → |Δē| = sqrt(e_s² + e_t²) up to e_max·sqrt(2)
    _, _, phi, tol_eff = phi_orbit_scaled(
        a_s, e_max, 0.0,
        a_t, e_max, math.pi / 2,
        K, obs_alt_scale_m
    )
    return phi, tol_eff


def main():
    print("=" * 90)
    print("Phase 5.5 P1 E5 — Φ_orbit at altitude × K × obs_alt_scale_m")
    print("=" * 90)
    print(f"SUCCESS_TOL_A = {SUCCESS_TOL_A:.0f} m")
    print(f"EPS_ORBIT     = {EPS_ORBIT:.1f} (σ₂ gate threshold)")
    print()
    print("Worst-case Φ_orbit (max Δa within band × max Δē at e_max) per cell:")
    print(f"{'band':<10} {'half_Δa (km)':>13} {'e_max':>6} "
          f"{'obs_scale':<14} {'K':>7} {'tol_eff_km':>11} "
          f"{'da_term':>9} {'de_term':>9} {'Φ_orbit':>9} {'gate':>5}")
    print("-" * 130)

    for label, a_c, half in ALT_BANDS:
        for e_max in [0.05, 0.20, 0.50]:
            for scale_label, obs_scale in OBS_SCALES:
                for K in K_VALUES:
                    da_term = (2 * half) / max(SUCCESS_TOL_A, K * obs_scale)
                    de = e_max * math.sqrt(2)
                    phi = da_term + de
                    tol_eff = max(SUCCESS_TOL_A, K * obs_scale)
                    gate = "ON" if phi < EPS_ORBIT else "off"
                    print(
                        f"{label:<10} {half*2/1000:>13.0f} {e_max:>6.2f} "
                        f"{scale_label:<14} {K:>7.3f} {tol_eff/1000:>11.0f} "
                        f"{da_term:>9.2f} {de:>9.3f} {phi:>9.2f} {gate:>5}"
                    )
                print()
            print()

    print("Goal: pick K such that Φ_orbit at worst case stays O(1-50) so EPS_ORBIT=2.0 gate")
    print("still operates meaningfully (gate is closed when Φ > 2.0 = orbit far from target).")
    print()
    print("Backward compat: at LEO obs_alt_scale_m=1.6e6, K=0.001 → tol_eff = max(10, 1.6) km = 10 km")
    print("(identical to legacy Phase 5b/5e Φ_orbit behavior, by design).")
    print()
    print("Observations from the scan:")
    print("  - LEO band (Δa_max=500 km), K=0.001 → da_term=50 (matches legacy)")
    print("  - GEO band (Δa_max=1000 km if a∈[42-43Mm], but in actual GEO training")
    print("    you'd span LEO→GEO so Δa_max ~ 35000 km → da_term explodes without K)")
    print("  - K should scale with the obs_alt_scale_m used")
    print("  - When obs_alt_scale_m=4.2e7, K=0.01 gives tol_eff=420 km. For Δa=35000 km:")
    print("    da_term=83 — still high. Even K=0.05 gives tol_eff=2.1 Mm, da_term=17. Better.")


if __name__ == "__main__":
    main()
