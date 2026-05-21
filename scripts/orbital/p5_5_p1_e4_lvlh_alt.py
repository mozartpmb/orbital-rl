"""Phase 5.5 P1 E4 — LVLH frame at high altitude.

Fresh probe (no existing E4 standalone script). Mirrors the LVLH-frame
computation from pufferlib/pufferlib/ocean/orbital/orbital.h:560-590 in
Python, then evaluates LVLH obs magnitudes at altitude × eccentricity cells.

The LVLH frame in obs[33-37] uses:
  - obs[33-34]: (dx_l, dy_l) / R_EARTH  — spatial offset in target frame
  - obs[35-36]: (dvx_l, dvy_l) / v_circ_t  — relative velocity in target frame
  - obs[37]: n_tgt / 1e-3  — target mean motion scale (LEO ~1e-3)

At GEO, n_tgt drops to ~7e-5 (because period scales a^(3/2)), so obs[37] would
be ~0.07 vs LEO ~1.0. This is a known scaling concern.
R_EARTH-normalized spatial offsets at MEO/GEO could exceed [-2, 2] for
extreme cells. v_circ_t normalization is relative so should stay bounded.

Run:
    python3 scripts/orbital/p5_5_p1_e4_lvlh_alt.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p5e_e2_kepler_precision import (
    MU,
    R_EARTH,
    orbit_to_cartesian,
    solve_kepler,
    true_from_eccentric,
)


ALT_CELLS = [
    ("LEO", R_EARTH + 500e3),
    ("MEO_low", R_EARTH + 1100e3),
    ("MEO", R_EARTH + 6000e3),
    ("MEO_high", R_EARTH + 20200e3),
    ("GEO", R_EARTH + 35786e3),
]

ECC_LEVELS = [0.0, 0.05, 0.20, 0.50, 0.70]


def compute_lvlh(a_s, e_s, om_s, M_s, a_t, e_t, om_t, M_t):
    """Mirror orbital.h fill_observations LVLH block (lines 560-589)."""
    E_s = solve_kepler(M_s, e_s)
    th_s = true_from_eccentric(E_s, e_s)
    E_t = solve_kepler(M_t, e_t)
    th_t = true_from_eccentric(E_t, e_t)

    sx, sy, svx, svy = orbit_to_cartesian(a_s, e_s, om_s, th_s)
    tx, ty, tvx, tvy = orbit_to_cartesian(a_t, e_t, om_t, th_t)

    theta_t = th_t + om_t
    ct, st = math.cos(theta_t), math.sin(theta_t)

    dxi = sx - tx
    dyi = sy - ty
    dvxi = svx - tvx
    dvyi = svy - tvy

    dx_l = ct * dxi + st * dyi
    dy_l = -st * dxi + ct * dyi
    dvx_l = ct * dvxi + st * dvyi
    dvy_l = -st * dvxi + ct * dvyi

    n_tgt = math.sqrt(MU / (a_t ** 3))
    dvx_l += n_tgt * dy_l
    dvy_l -= n_tgt * dx_l

    v_circ_t = math.sqrt(MU / a_t)

    obs33 = dx_l / R_EARTH
    obs34 = dy_l / R_EARTH
    obs35 = dvx_l / v_circ_t
    obs36 = dvy_l / v_circ_t
    obs37 = n_tgt / 1e-3

    return obs33, obs34, obs35, obs36, obs37, n_tgt, v_circ_t


def main():
    print("=" * 90)
    print("Phase 5.5 P1 E4 — LVLH frame obs[33-37] magnitudes across altitude/eccentricity")
    print("=" * 90)
    print(f"{'band':<10} {'a (Mm)':>8} {'e':>6} {'n_tgt':>10} {'v_circ':>8} "
          f"{'|obs33|':>8} {'|obs34|':>8} {'|obs35|':>8} {'|obs36|':>8} {'obs37':>7}")
    print("-" * 100)

    rng = np.random.RandomState(42)
    # For each (alt, e) cell: 100 random (M_s, om_s, M_t, om_t) configurations
    # with sat & target at same a but different orbit angles. Mirrors the env's
    # rendezvous-at-init regime. Report worst-case |obs| per cell.
    for label, a in ALT_CELLS:
        for e in ECC_LEVELS:
            max_o33 = max_o34 = max_o35 = max_o36 = 0.0
            obs37_val = None
            n_tgt_val = None
            v_circ_val = None
            for _ in range(100):
                # sat & target at same a, same e — only orientation/phase differ.
                # This mirrors the curriculum's Stage 1 same_orbit_init regime.
                om_s = rng.random() * 2 * math.pi
                M_s = rng.random() * 2 * math.pi
                # target offset uniform in [-pi, pi]
                om_t = (om_s + (rng.random() - 0.5) * 2 * math.pi) % (2 * math.pi)
                M_t = (M_s + (rng.random() - 0.5) * 2 * math.pi) % (2 * math.pi)
                o33, o34, o35, o36, o37, n_t, vc = compute_lvlh(
                    a, e, om_s, M_s, a, e, om_t, M_t
                )
                max_o33 = max(max_o33, abs(o33))
                max_o34 = max(max_o34, abs(o34))
                max_o35 = max(max_o35, abs(o35))
                max_o36 = max(max_o36, abs(o36))
                obs37_val = o37
                n_tgt_val = n_t
                v_circ_val = vc
            print(
                f"{label:<10} {a/1e6:>8.2f} {e:>6.2f} {n_tgt_val:>10.2e} {v_circ_val:>8.0f} "
                f"{max_o33:>8.3f} {max_o34:>8.3f} {max_o35:>8.3f} {max_o36:>8.3f} {obs37_val:>7.3f}"
            )
        print()

    print("Pass criterion: |obs| ≤ ~2.0 (gymnasium Box bound). Saturation = OOD for policy.")
    print("Notes:")
    print("  obs[33-34] spatial offset / R_EARTH: bounded by orbit geometry (≈ 2a/R_EARTH worst case)")
    print("  obs[35-36] velocity / v_circ_t: bounded by 2 (relative velocity ≤ 2× circular)")
    print("  obs[37] n_tgt/1e-3: LEO=1, GEO≈0.07 — informational only, not in normalization")
    print("  Spatial offset normalization is R_EARTH-based (~6.37 Mm), so at GEO (a~42Mm)")
    print("  the relative position can naturally reach |obs| ~ 12 if sat & target are on")
    print("  opposite sides of the orbit. This is a known scaling concern for high-altitude")
    print("  rendezvous — NOT addressed by obs_alt_scale_m which only rescales obs[0,7,17-32].")


if __name__ == "__main__":
    main()
