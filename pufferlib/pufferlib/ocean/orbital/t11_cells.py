#!/usr/bin/env python3
"""T11 — the shipped generalist cell mixture, in ONE place.

Imported by both the gates and the campaign so they cannot drift apart on the
table, which is the thing every number in the rung is conditioned on.

Column order matches `orbital.h`'s CF_* defines and `Orbital.CELL_FIELDS`.

── THE NORMALIZER PAIRING ───────────────────────────────────────────────────
    obs_alt_scale_m = 8.0e6    (WIDE)   — bound by the eccentricity requirement:
        e and altitude are coupled by the perigee keepout, and narrow (1.6e6)
        caps realized e at p90 0.112 against E3's 0.257 (t11 family probe).
    lvlh_scale_m    = 6.371e6  (NARROW) — bound by the tight box: a 5 km box is
        7.8e-4 obs units at 6.371e6 but only 3.3e-4 at 1.5e7.
The two are independent kwargs; the "family" is two knobs, not one. Any lineage
reaches this pairing exactly via rescale_ckpt_normalizers.py.

── WEIGHTS ──────────────────────────────────────────────────────────────────
From GEN_MATRIX's nesting result: tight ⊃ loose (99%) and J2 ⊃ two-body (91%)
are ONE-DIRECTIONAL, so the subset regimes are free and get weight ZERO. The
one axis that is not nested is drift-and-wait — no 30-row head can express it
at all (Signature B) and its own specialist is the weakest generalist in the
matrix (7.8% off-diagonal) — so it gets the largest single weight.

── FUEL ─────────────────────────────────────────────────────────────────────
U(0.113, 0.20) -> 353-656 m/s. The floor is measured, not chosen: at 245 m/s
(f=0.08, the original proposal) 46-59% of ordinary cells are Dv-INFEASIBLE, and
`valid_init_only` cannot catch it because it rejects on perigee only — fuel
never enters the rejection sampler, so a lean draw yields feasible-LOOKING but
unsolvable episodes that nothing counts.
"""

import math

# obs normalizer pairing — see the header.
OBS_ALT_SCALE = 8.0e6
LVLH_SCALE = 6.371e6

FUEL_MIN, FUEL_MAX = 0.113, 0.20
NAV_MAX_TICKS = 120   # see base_env_kwargs; gate G6 pins it to 60..240

_I_MIN, _I_MAX = math.radians(30.0), math.radians(60.0)

FIELDS = ('weight', 'cap', 'box_r', 'box_v', 'a_min', 'a_max',
          'e_max_target', 'e_max_sat', 'de_max', 'da_max',
          'di_max', 'di_min', 'di_phase', 'j2',
          'i_t_min', 'i_t_max', 'fuel_min', 'fuel_max')


def _cell(name, weight, cap, box_r, box_v, a_min, a_max, e_max_t,
          e_max_s=-1.0, de_max=-1.0, da_max=-1.0, di_max=0.017453,
          di_min=-1.0, di_phase=0, j2=1,
          i_min=_I_MIN, i_max=_I_MAX, fuel_min=FUEL_MIN, fuel_max=FUEL_MAX):
    return name, dict(
        weight=weight, cap=cap, box_r=box_r, box_v=box_v,
        a_min=a_min, a_max=a_max, e_max_target=e_max_t,
        e_max_sat=(e_max_t if e_max_s < 0 else e_max_s),
        de_max=de_max, da_max=da_max, di_max=di_max, di_min=di_min,
        di_phase=di_phase, j2=j2, i_t_min=i_min, i_t_max=i_max,
        fuel_min=fuel_min, fuel_max=fuel_max)


# ── the mixture ──────────────────────────────────────────────────────────────
CELLS = [
    # 1-4: the eccentricity ladder, restated with J2 + inclined targets.
    _cell('E0_j2', 0.15, 3000, 30000.0, 50.0, 6.671e6, 7.171e6, 0.05),
    _cell('E1_j2', 0.15, 3000, 30000.0, 50.0, 6.671e6, 7.871e6, 0.10,
          de_max=0.05, da_max=300e3),
    _cell('E2_j2', 0.15, 4500, 30000.0, 50.0, 6.671e6, 9.871e6, 0.20,
          de_max=0.065, da_max=450e3, di_max=0.013090),
    _cell('E3_j2', 0.15, 6000, 30000.0, 50.0, 6.671e6, 14.371e6, 0.30,
          de_max=0.08, da_max=600e3, di_max=0.013090),
    # 5: drift-and-wait. The disjoint skill — largest single weight.
    #    Needs the 31-row head (row 30 = the 1440-substep day-warp) and the
    #    22000 cap; at cap 3000 a single day-warp is 48% of the episode.
    _cell('W1_driftwait', 0.20, 22000, 30000.0, 50.0, 6.671e6, 7.171e6, 0.05,
          di_max=math.radians(5.0), di_min=math.radians(2.0), di_phase=1),
    # 6: the tight box. Unproven in this pairing -> low weight, gated on a
    #    zero-shot transplanted-TB5 check before the rung is trusted.
    _cell('TIGHT_5k1', 0.10, 3000, 5000.0, 1.0, 6.671e6, 7.171e6, 0.05),
    # 7: LONG-RANGE transfer. NOT "LEO->MEO": that was in the design doc and is
    #    NOT AFFORDABLE. A Hohmann from 6.671e6 to 14.371e6 costs ~1304 m/s
    #    against a 656 m/s ceiling. The affordable long-range cell is bounded
    #    by the budget instead: Dv ~ (v_c/2)(da/a) = 656 m/s at da ~ 1.18e6 m,
    #    so da_max is set to 1.0e6 with margin. Verified by the feasibility
    #    gate rather than assumed.
    #    TUNED AGAINST THE FLOOR, not guessed:
    #      da_max 1.0e6 -> 32.3% infeasible at 353 m/s
    #             7.5e5 -> 16.4%
    #             5.0e5 ->  2.3%   <- shipped
    #    5.0e5 m is still a 500 km transfer over a 12000-substep (200 h) cap,
    #    which is the long-range phasing character the cell exists for.
    _cell('LONGRANGE', 0.10, 12000, 30000.0, 50.0, 6.671e6, 9.871e6, 0.12,
          de_max=0.06, da_max=5.0e5, di_max=0.013090),
]

NAMES = [n for n, _ in CELLS]
TABLE = [c for _, c in CELLS]


def as_array():
    import numpy as np
    return np.array([[c[f] for f in FIELDS] for c in TABLE], dtype=np.float64)


def base_env_kwargs(**over):
    """Kwargs shared by every cell. The per-cell fields are overwritten at reset
    by the C sampler, so the values here are only the pre-mixture defaults."""
    kw = dict(
        num_debris_min=0, num_debris_max=0, same_orbit_init=0,
        init_phase_gap_max=3.14159, valid_init_only=1,
        gave_up_action='terminate', max_valid_init_attempts=4096,
        obs_alt_scale_m=OBS_ALT_SCALE, lvlh_scale_m=LVLH_SCALE,
        shaping_mode=2, shape_w_lambda=1.0,
        # 0.8166667/700 and 0.35/300 have IDENTICAL gradients (1.167e-3 per
        # m/s) and differ only in where min(1,.) saturates. Measured required
        # Dv has p90 302-810 m/s across these cells, so dv_ref=300 would
        # saturate on essentially every episode and leave no gradient at all.
        shape_w_match=0.8166667, shape_dv_ref_ms=700.0, shape_gamma=1.0,
        phase_gap_mode=1, phase_obs_mode=1, cap_terminal_reward=0.0,
        dim3_mode=1, j2_mode=1, lvlh_frame_mode=1,
        raan_target_sample=0,
        i_target_min_rad=_I_MIN, i_target_max_rad=_I_MAX,
        legacy_action_space=31,
        episode_cap_steps=3000,
        rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
        a_min_override=6.671e6, a_max_override=7.171e6,
        e_max_target=0.05, e_max_sat=0.05,
        cell_mixture_mode=1, fuel_frac_min=FUEL_MIN, fuel_frac_max=FUEL_MAX,
    )
    kw.update(over)
    return kw


def nav_env_kwargs(**over):
    """base_env_kwargs plus the navigation-only knobs.

    `nav_max_ticks` lives HERE and not in `base_env_kwargs` because the plain
    `Orbital` constructor REJECTS it -- it is a wrapper kwarg, so an unknown-
    kwarg TypeError, not a silent no-op. An earlier revision of this file put it
    in the shared dict on the reasoning that truth-mode callers "never consult
    it"; they do not, but they do refuse to be constructed with it, and every
    truth-mode gate died on `Orbital.__init__() got an unexpected keyword
    argument`. Kept separate so that cannot recur.

    Row 30's tau=1440 costs 1440 EKF sub-ticks per decision at nav_max_ticks=0,
    a 4.6x throughput loss on this mixture that puts rung B at 5.4 days.
    MAJOR-7's fixed-count/adaptive-interval form makes a cap safe; 120 is the
    knee (divergence 0.003, tying uncapped, against 0.109 at K=30) and leaves 28
    of 31 rows bit-identical.
    """
    kw = base_env_kwargs()
    kw['nav_max_ticks'] = NAV_MAX_TICKS
    kw.update(over)
    return kw


if __name__ == '__main__':
    a = as_array()
    print(f'{len(CELLS)} cells x {len(FIELDS)} fields; weights sum '
          f'{sum(c["weight"] for c in TABLE):.3f}')
    for n, c in CELLS:
        print(f'  {n:14s} w {c["weight"]:.2f} cap {int(c["cap"]):6d} '
              f'box {c["box_r"]/1e3:5.1f}km/{c["box_v"]:.0f} '
              f'a {c["a_min"]/1e6:.3f}-{c["a_max"]/1e6:.3f}e6 '
              f'e {c["e_max_target"]:.2f} j2 {int(c["j2"])}')
