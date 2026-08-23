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


# ── the tight-box LADDER (T11-tight campaign) — NOT part of the mixture ──────
# SINGLE-CELL TRAINING specs for the 6/7 attempt, kept out of `CELLS` on
# purpose: `CELLS` is the shipped mixture that G1/G2, every recorded floor and
# every published lineage are defined against, and stage 4 re-evaluates that
# mixture unchanged. A row added here would silently move rung B's own weights.
#
# fuel_min is 0.133, NOT the mixture's 0.113, and that is a MEASURED change.
# The tight box needs a terminal fine-burn train (~27 x 1 m/s at TB5) that the
# transfer-only feasibility estimate does not price. Infeasible mass for
# transfer + 27 m/s, 1024 draws:
#
#     352.8 m/s (fuel 0.1130) -> 15.8%   <- ABOVE G4's 12% gate
#     418.9 m/s (fuel 0.1327) ->  2.1%   <- shipped; matches LONGRANGE's 2.3%
#     499.0 m/s (fuel 0.1560) ->  0.1%   (range too narrow for fuel-awareness)
#
# At the mixture floor one TB5 episode in six is unwinnable, which is bootstrap
# poison for a rung whose whole purpose is to bootstrap the skill. 0.133 keeps a
# 419-656 m/s spread (1.57x) so budget-awareness still has signal. Stage 4
# evaluates the tight cell BOTH ways — at this training range, and at the
# mixture's shipped 0.113 with its ~84% feasibility ceiling stated — so the gain
# number stays comparable to the existing T11 lineage.
_TIGHT_FUEL_MIN = 0.133

TIGHT_CELLS = [
    # The intermediate rung. 10 km / 10 m/s: loose enough that the rung-B child
    # has a gradient to follow, tight enough that it is not merely E0 again.
    _cell('TIGHT_10k10', 0.0, 3000, 10000.0, 10.0, 6.671e6, 7.171e6, 0.05,
          fuel_min=_TIGHT_FUEL_MIN),
    # The target rung. Same box as the mixture's TIGHT_5k1, own fuel floor.
    # cap stays 3000 — the setting proven at TB5 in the narrow pairing, and
    # 3000 substeps is 50 h (~30 orbits) for a same-band transfer, so the cap is
    # not the binding constraint. Kept unless evidence says otherwise.
    _cell('TIGHT_5k1_T', 0.0, 3000, 5000.0, 1.0, 6.671e6, 7.171e6, 0.05,
          fuel_min=_TIGHT_FUEL_MIN),
]

# One lookup for the evaluator; `CELLS` stays the mixture and only the mixture.
ALL_CELLS = dict(CELLS)
ALL_CELLS.update(dict(TIGHT_CELLS))


# ── the CONSOLIDATION mixture (t11_mixture=2) ────────────────────────────────
# Same seven cells, same physics, DIFFERENT WEIGHTS AND ONE FUEL FLOOR. Derived
# from `CELLS` programmatically rather than restated, so the cell physics cannot
# drift from the shipped mixture: this experiment's whole point is that weights
# + LR + root are the ONLY things that change.
#
# WHY THESE WEIGHTS. The T11-tight seesaw measured a bidirectional failure at
# the full acquisition LR (1e-2): the forward swing collapsed the wide cells,
# the back swing re-zeroed the tight box at 0.10 rehearsal weight. Both
# mechanisms are addressed here — LR drops to 1e-3 (maintenance, not
# acquisition; set on the command line) and the minority skill stops being a
# minority. TIGHT goes 0.10 -> 0.25 because it is the skill that has twice
# proven hardest to rebuild; the wide cells rebuild readily under majority
# gradient, which is the asymmetry both swings showed.
#
# W1_driftwait is EXCLUDED (weight 0.0), stated rather than quietly dropped: it
# scored 0.0 in all three seesaw states — dead gradient in both directions — and
# its 22000-step cap costs disproportionate wall-clock per unit of learning.
# The target here is the SIX-skill consolidation. W1 is still EVALUATED, so the
# zero stays on the record honestly instead of vanishing from the battery.
#
# A zero weight is safe in the C picker because W1 is not the last row: the
# cumulative walk can only land on a zero-weight cell via the trailing
# `pick = c` fallthrough, which is reachable only for the FINAL index when
# rounding puts u past the total. G9 asserts the zero empirically so a future
# reordering of `CELLS` cannot turn this into a silent 22000-step cell.
_CONSOL_WEIGHTS = {
    'TIGHT_5k1':    0.25,
    'E3_j2':        0.20,
    'E2_j2':        0.15,
    'LONGRANGE':    0.15,
    'E0_j2':        0.125,
    'E1_j2':        0.125,
    'W1_driftwait': 0.0,
}
# Per-cell fuel floors, exactly as the shipped mixture already does it: the
# tight cell carries the MEASURED floor (0.133 = 419 m/s) because at 0.113 the
# terminal fine-burn train is unaffordable in 15.8% of tight episodes, and
# training a consolidation run on unwinnable episodes would look like
# interference while actually being infeasibility. Every other cell keeps the
# shipped 0.113 — their floors were already measured against their own bands.
_CONSOL_FUEL_MIN = {'TIGHT_5k1': _TIGHT_FUEL_MIN}

CONSOL_CELLS = []
for _n, _c in CELLS:
    _d = dict(_c)
    _d['weight'] = _CONSOL_WEIGHTS[_n]
    if _n in _CONSOL_FUEL_MIN:
        _d['fuel_min'] = _CONSOL_FUEL_MIN[_n]
    CONSOL_CELLS.append((_n, _d))
CONSOL_TABLE = [c for _, c in CONSOL_CELLS]
assert abs(sum(c['weight'] for c in CONSOL_TABLE) - 1.0) < 1e-12
assert len(CONSOL_TABLE) == len(TABLE)


# ── the T15 7/7 mixture (t11_mixture=3) ─────────────────────────────────────
# All seven cells, W1 back in, weights set on MEASURED STEP SHARE rather than
# episode share — which turned out to matter in the opposite direction from the
# usual warning. Decisions per episode, measured with the t13b root at K=0:
#
#     TIGHT_5k1 623   W1_driftwait 536   E3 44   E2 35   LONGRANGE 29   E0/E1 28
#
# The long-CAP cells (W1 22000, LONGRANGE 12000) are NOT the step-share
# drivers: they cover their transfers with big warps and finish in ~30 decisions.
# TIGHT is, because its terminal fine-burn train spends hundreds of small
# actions. So the naive "22000-step cells outweigh everything" correction would
# have been applied to the wrong cell.
#
# Resulting step share at the weights below:
#     W1 61.4%   TIGHT 28.6%   E3 3.2%   E2 2.1%   LONGRANGE 1.6%   E0 1.5%   E1 1.5%
#
# W1 takes the majority because it is the ACQUISITION target and the only cell
# at exactly 0.0 from this root. TIGHT keeps a large share because it is the
# cell that has collapsed before, and it also carries a defense anchor. The
# wides sit at 10.0% combined, which is the level T13b PROVED sufficient to
# maintain them (~8% there, all six held 93.5-100).
_T15_WEIGHTS = {
    'W1_driftwait': 0.25,
    'E3_j2':        0.16,
    'E2_j2':        0.13,
    'E0_j2':        0.12,
    'E1_j2':        0.12,
    'LONGRANGE':    0.12,
    'TIGHT_5k1':    0.10,
}
# Per-cell fuel floors exactly as shipped: the tight cell carries its MEASURED
# floor (0.133 = 419 m/s) because at 0.113 the terminal fine-burn train is
# unaffordable in 15.8% of tight episodes, and training on unwinnable episodes
# would read as interference while being infeasibility.
_T15_FUEL_MIN = {'TIGHT_5k1': _TIGHT_FUEL_MIN}

T15_CELLS = []
for _n, _c in CELLS:
    _d = dict(_c)
    _d['weight'] = _T15_WEIGHTS[_n]
    if _n in _T15_FUEL_MIN:
        _d['fuel_min'] = _T15_FUEL_MIN[_n]
    T15_CELLS.append((_n, _d))
T15_TABLE = [c for _, c in T15_CELLS]
assert abs(sum(c['weight'] for c in T15_TABLE) - 1.0) < 1e-12
assert all(c['weight'] > 0.0 for c in T15_TABLE), 'T15 trains all SEVEN cells'


# ── the T15b DAgger-refresh mixture (t11_mixture=4) ─────────────────────────
# Same seven cells; weights RE-SOLVED because the step-share arithmetic moved
# under us. W1's decisions/episode collapsed 536 -> 177 once it became
# competent (successful W1 episodes are 46 decisions, failed ones 359), so the
# T15 weights that bought W1 61.4% of gradient now buy it 34.5%.
#
# THAT IS A BUILT-IN BRAKE, and it is a candidate explanation for the 31.5
# plateau alongside the lambda wean: a fixed episode weight delivers a
# DECREASING gradient share exactly as the skill starts working. Projected at
# these weights, W1's share runs 65.6% (32% success) -> 47.7% (75%) -> 25.2%
# (100%) — graceful, but it does mean acquisition pressure fades on success.
#
# Step share at the CURRENT root:
#     W1 44.4%   TIGHT 34.8%   E3 6.1%   E2 5.2%   E0 4.2%   E1 3.1%   LR 2.2%
#
# TIGHT keeps 34.8% (up from 28.6% intended in T15) because its defense anchor
# is MEASURED SATURATED — CE(teacher||policy) on TIGHT states is 0.035-0.042,
# so per-step supervision has almost no headroom left and only reward share can
# recover it from 85.5.
_T15B_WEIGHTS = {
    'W1_driftwait': 0.27,
    'E2_j2':        0.16,
    'E0_j2':        0.16,
    'E3_j2':        0.15,
    'E1_j2':        0.12,
    'LONGRANGE':    0.08,
    'TIGHT_5k1':    0.06,
}
T15B_CELLS = []
for _n, _c in CELLS:
    _d = dict(_c)
    _d['weight'] = _T15B_WEIGHTS[_n]
    if _n in _T15_FUEL_MIN:
        _d['fuel_min'] = _T15_FUEL_MIN[_n]
    T15B_CELLS.append((_n, _d))
T15B_TABLE = [c for _, c in T15B_CELLS]
assert abs(sum(c['weight'] for c in T15B_TABLE) - 1.0) < 1e-12
assert all(c['weight'] > 0.0 for c in T15B_TABLE)


# ── T15c: iteration-2 weights (t11_mixture=5) ───────────────────────────────
# Re-solved because W1's decisions/episode moved AGAIN, and non-monotonically:
# 536 (@0%) -> 177 (@31.5%) -> 307 (@45%). It fell as successes appeared (they
# are ~40 decisions) and then ROSE as the failures got longer (359 -> 622), so
# the self-attenuation is not a simple decay — it tracks the success/failure
# LENGTH MIX. At the T15b weights that drift had pushed TIGHT's share from an
# intended 34.8% down to 26.2%, and TIGHT's recovery depends entirely on reward
# share because its anchor is measured saturated (CE 0.035-0.042).
# Restored: W1 45.9%, TIGHT 34.3%, wides 19.7%.
_T15C_WEIGHTS = {
    'E0_j2': 0.19, 'E2_j2': 0.19, 'W1_driftwait': 0.19, 'E3_j2': 0.18,
    'E1_j2': 0.09, 'LONGRANGE': 0.09, 'TIGHT_5k1': 0.07,
}
T15C_CELLS = []
for _n, _c in CELLS:
    _d = dict(_c)
    _d['weight'] = _T15C_WEIGHTS[_n]
    if _n in _T15_FUEL_MIN:
        _d['fuel_min'] = _T15_FUEL_MIN[_n]
    T15C_CELLS.append((_n, _d))
T15C_TABLE = [c for _, c in T15C_CELLS]
assert abs(sum(c['weight'] for c in T15C_TABLE) - 1.0) < 1e-12
assert all(c['weight'] > 0.0 for c in T15C_TABLE)


def t15c_as_array():
    import numpy as np
    return np.array([[c[f] for f in FIELDS] for c in T15C_TABLE], dtype=np.float64)


def t15b_as_array():
    import numpy as np
    return np.array([[c[f] for f in FIELDS] for c in T15B_TABLE], dtype=np.float64)


def t15_as_array():
    import numpy as np
    return np.array([[c[f] for f in FIELDS] for c in T15_TABLE], dtype=np.float64)


def consol_as_array():
    import numpy as np
    return np.array([[c[f] for f in FIELDS] for c in CONSOL_TABLE],
                    dtype=np.float64)


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
