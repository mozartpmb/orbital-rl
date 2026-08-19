#!/usr/bin/env python3
"""Per-episode delta-v against the optimal-transfer baseline, as a DISTRIBUTION.

The project's standing fuel number is "median 2.49x Hohmann", which is a ratio
to a circular-Hohmann surrogate that ignores phasing, eccentricity, and the
fact that rendezvous must match the target's position AND velocity. T1 replaced
that with a real two-impulse Lambert optimum solved per episode
(`scripts/orbital/t1_lambert_baseline.py`, validated against the headline task
at 1.14x time-matched). This extends T1 across the current flagship lineages.

WHAT IS AND IS NOT COMPARABLE. The Lambert comparator is a 2D coplanar
two-impulse transfer. It is the right optimum for the T3 canonical lineage and
is quoted as a ratio there. For the 3D lineages the mission also contains a
PLANE change and, under J2, a secular nodal drift, neither of which a coplanar
Lambert can price — so quoting a single ratio there would be comparing a policy
against a baseline that is not solving its problem. Those lineages get a
DECOMPOSITION instead (tangential / plane / fine-terminal), plus an
in-plane-projected Lambert reference for the transfer part alone, clearly
labelled as covering only that part.

    python3 scripts/orbital/nav/fuel_audit.py --episodes 200
    python3 scripts/orbital/nav/fuel_audit.py --lineages T3-canonical --episodes 20
    python3 scripts/orbital/nav/fuel_audit.py --analyze-only

Outputs: web_data/results/fuel_audit.csv, FUEL_AUDIT.md
"""

import argparse
import csv
import glob
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'orbital'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'orbital', 'extj2'))

from pufferlib.ocean.orbital.orbital import Orbital                # noqa: E402
from pufferlib.models import Default, LSTMWrapper                  # noqa: E402
import t1_lambert_baseline as T1                                   # noqa: E402
import eval_relnav3d as ER3                                        # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav      # noqa: E402

MU = 3.986004418e14
DT = 60.0
TWO_PI = 2.0 * math.pi
TRAJ_ROOT = '/tmp/fuel_audit_traj'
CSV_OUT = os.path.join(ROOT, 'web_data', 'results', 'fuel_audit.csv')
MD_OUT = os.path.join(ROOT, 'FUEL_AUDIT.md')

# orbital.h ACTION_DV — (prograde, radial, normal) per action, m/s.
ACTION_DV = np.array([
    [0, 0, 0], [5, 0, 0], [10, 0, 0], [25, 0, 0], [-5, 0, 0], [-10, 0, 0],
    [-25, 0, 0], [0, 10, 0], [0, -10, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1], [0, 0, 10], [0, 0, -10], [0, 0, 25], [0, 0, -25],
    [25, 0, 25], [25, 0, -25], [-25, 0, 25], [-25, 0, -25],
], dtype=np.float64)
FINE_MAG = 2.0 + 1e-9      # |dv| <= 2 m/s is a fine / terminal burn

# ── lineages, each at the config its published number was produced under ────
_T3_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
)
_J2_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
    shape_dv_ref_ms=700.0, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
    dim3_mode=1, di_max_rad=0.017453, legacy_action_space=30,
    j2_mode=1, lvlh_frame_mode=1, raan_target_sample=0,
    i_target_min_rad=math.radians(30.0), i_target_max_rad=math.radians(60.0),
    rendezvous_radius_m=5000.0, rel_vel_tol_ms=1.0,
)

LINEAGES = {
    # 2D coplanar: Lambert is the right optimum and is quoted as a ratio.
    'T3-canonical': dict(
        ckpt='models/t3/seed42_L2_headline.pt', mode='guidance',
        kw=_T3_KW, lambert='full',
        note='T3 headline, 2D, Discrete-16, box 30 km / 50 m/s'),
    # 3D + bearings-only nav at the tight box.
    'TB5-3D': dict(
        ckpt='models/t3/n3dnav_T-BO3D-TB5.pt', mode='nav', rung='X3',
        box='TB5-3D', shape_w_match=0.35, nav_mode='bearings_only', acq='real',
        lambert='inplane',
        note='3D + bearings-only (real IOD), box 5 km / 1 m/s'),
    # 3D + bearings-only nav up the eccentricity axis, wide normalizers.
    'E3': dict(
        ckpt='models/t3/n3dnav_e_E3.pt', mode='nav', rung='E3',
        nav_mode='bearings_only', acq='real', lambert='inplane',
        note='3D + bearings-only, e_max 0.30 (realized e_t 0.126), box 30 km / 50 m/s'),
    # 3D + J2 secular drift, truth-state guidance.
    'J2-A3b': dict(
        ckpt='models/t3/extj2_A3b_j2_box5k1.pt', mode='guidance',
        kw=_J2_KW, lambert='inplane',
        note='3D + J2 (mean elements), i_t 30-60 deg, box 5 km / 1 m/s'),
}

# ── the T11 generalist, one row per GREEN mixture cell ──────────────────────
# Added so the generalist is measured against the SAME comparator as the four
# specialists above rather than against T11's own internal 1.23-1.29x figure,
# which is a different reference (the cell's linearised direct-burn estimate)
# and is therefore not comparable to a box-credited Lambert ratio.
#
# Each cell is built by t11_eval.env_kwargs -- the generalist's OWN eval path,
# including its acquisition configuration -- because the quantity being audited
# is the fuel the shipped policy actually spends, and swapping in a different
# acquisition front-end would change the trajectory before the comparator ever
# ran. The two red cells (W1_driftwait, TIGHT_5k1) are omitted: both score 0.0%,
# so there are no successful episodes to audit.
for _c in ('E0_j2', 'E1_j2', 'E2_j2', 'E3_j2', 'LONGRANGE'):
    LINEAGES[f'T11-{_c}'] = dict(
        ckpt='models/t3/t11_generalist_rungB.pt', mode='t11', cell=_c,
        lambert='inplane',
        note=f'T11 generalist (200M, 7-cell mixture) at cell {_c}')


# ── collection ──────────────────────────────────────────────────────────────
def build_env(name, spec, traj_dir):
    """The lineage's own eval env, plus per-episode trajectory logging."""
    traj = dict(traj_log_dir=traj_dir, traj_log_every=1)
    if spec['mode'] == 'guidance':
        return Orbital(num_envs=1, **dict(spec['kw'], **traj))
    if spec['mode'] == 't11':
        import t11_eval as T11E
        kw = dict(T11E.env_kwargs(spec['cell'], 'bearings_only', None), **traj)
        env = OrbitalNav(num_envs=1, nav_mode='bearings_only', **kw)
        env._acq_real = True            # exactly what t11_eval.run() sets
        return env
    over = {}
    if spec.get('box'):
        over.update(ER3.box_kw(spec['box']))
    if spec.get('shape_w_match') is not None:
        over['shape_w_match'] = spec['shape_w_match']
    over.update(traj)
    env = ER3.make_env(spec['rung'], spec['nav_mode'], acq=spec.get('acq', 'surrogate'),
                       **over)
    if spec.get('acq') == 'real':
        ER3._install_real_acq(env)
    return env


def collect(name, spec, episodes, seed):
    traj_dir = os.path.join(TRAJ_ROOT, name)
    shutil.rmtree(traj_dir, ignore_errors=True)
    os.makedirs(traj_dir, exist_ok=True)
    env = build_env(name, spec, traj_dir)
    ck = os.path.join(ROOT, spec['ckpt'])
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
    policy.eval()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    obs, _ = env.reset(seed=seed)
    st = {'lstm_h': torch.zeros(1, policy.hidden_size),
          'lstm_c': torch.zeros(1, policy.hidden_size)}
    n_done, t0 = 0, time.time()
    while n_done < episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), st)
            a = int(torch.argmax(logits, dim=-1).item())
        obs, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        if term[0]:
            n_done += 1
            st = {'lstm_h': torch.zeros(1, policy.hidden_size),
                  'lstm_c': torch.zeros(1, policy.hidden_size)}
            if n_done % 50 == 0:
                print(f'    [{name}] {n_done}/{episodes} '
                      f'({time.time() - t0:.0f}s)', flush=True)
    env.close()
    return sorted(glob.glob(os.path.join(traj_dir, 'ep_*.npz')))


# ── per-episode analysis ────────────────────────────────────────────────────
def decompose(d):
    """Split the episode's applied delta-v by axis and by size.

    Keyed on `delta_v > 0`, which is also what skips traj_log[0]: orbital.h
    writes row 0 BEFORE the first action is read, so its `action` column is
    stale (the diag tooling's documented off-by-one). Row 0 carries no burn, so
    keying on the applied magnitude drops it for free — and, unlike slicing
    from index 1, it stays correct if that ever changes.

    `delta_v` is the magnitude the env ACTUALLY applied (apply_impulse clamps
    the assembled vector's norm when fuel-limited, preserving direction), so
    splitting it by the action's unit direction is exact rather than nominal.
    """
    dv = np.asarray(d['delta_v'], dtype=np.float64)
    act = np.asarray(d['action'], dtype=np.int64)
    m = dv > 0.0
    if not m.any():
        return dict(dv_total=0.0, dv_tangential=0.0, dv_plane=0.0,
                    dv_radial=0.0, dv_fine=0.0, dv_coarse=0.0, n_burns=0)
    a = np.clip(act[m], 0, len(ACTION_DV) - 1)
    mag = dv[m]
    nominal = ACTION_DV[a]
    nrm = np.linalg.norm(nominal, axis=1)
    nrm = np.where(nrm > 0, nrm, 1.0)
    applied = nominal * (mag / nrm)[:, None]        # exact: direction preserved
    fine = mag <= FINE_MAG
    return dict(
        dv_total=float(mag.sum()),
        dv_tangential=float(np.abs(applied[:, 0]).sum()),
        dv_radial=float(np.abs(applied[:, 1]).sum()),
        dv_plane=float(np.abs(applied[:, 2]).sum()),
        dv_fine=float(mag[fine].sum()),
        dv_coarse=float(mag[~fine].sum()),
        n_burns=int(m.sum()))


def teleport_abs_deg(d):
    """Per-episode unphysical along-track jump, FRAME-CORRECT in 3D.

    T1's `phase_teleport` reads the in-plane anomaly as `atan2(y, x) - omega`,
    which is only the true anomaly for an EQUATORIAL orbit. Applied to an
    inclined lineage it measures the frame mismatch, not a physics violation:
    on the TB5-3D checkpoint it reports 125 deg/episode of "teleport" where the
    dynamics are in fact clean. So the anomaly is taken in the orbit plane
    here, which reduces to T1's expression exactly at inc = raan = 0.

    The quantity itself is the check T1 built to catch the inverted
    `true_to_mean` half-angle map, which handed the agent ~24 deg of free
    along-track phase per episode and made its sub-1.0 delta-v ratios
    inadmissible. It is carried per episode so this audit STATES that the
    corrected dynamics hold rather than assuming it.
    """
    x = d['sat_x'].astype(np.float64)
    y = d['sat_y'].astype(np.float64)
    z = (d['sat_z'].astype(np.float64) if 'sat_z' in d
         else np.zeros_like(x))
    a1 = d['sat_a'][1:].astype(np.float64)
    e1 = d['sat_e'][1:].astype(np.float64)
    om1 = d['sat_omega'][1:].astype(np.float64)
    inc = (d['sat_inc'][1:].astype(np.float64) if 'sat_inc' in d
           else np.zeros_like(a1))
    ra = (d['sat_raan'][1:].astype(np.float64) if 'sat_raan' in d
          else np.zeros_like(a1))
    ci, si = np.cos(inc), np.sin(inc)
    cO, sO = np.cos(ra), np.sin(ra)
    # node direction and its in-plane perpendicular
    n_x, n_y, n_z = cO, sO, np.zeros_like(cO)
    h_x, h_y, h_z = si * sO, -si * cO, ci
    m_x = h_y * n_z - h_z * n_y
    m_y = h_z * n_x - h_x * n_z
    m_z = h_x * n_y - h_y * n_x

    def anomaly(px, py, pz):
        u = np.arctan2(px * m_x + py * m_y + pz * m_z,
                       px * n_x + py * n_y + pz * n_z)
        return u - om1

    th_burn = anomaly(x[:-1], y[:-1], z[:-1])
    th_end = anomaly(x[1:], y[1:], z[1:])
    n = np.sqrt(MU / a1 ** 3)
    delta = (T1.true_to_mean(th_end, e1) - T1.true_to_mean(th_burn, e1)
             - n * DT)
    delta = np.mod(delta + math.pi, TWO_PI) - math.pi
    return math.degrees(float(np.abs(delta).sum()))


def _inplane_state(d, i, body):
    """Rotate a logged 3D state into the TARGET's orbit plane and drop z.

    For the 3D lineages the RELATIVE geometry is near-coplanar by construction
    (di_rel <= 1 deg), so the transfer part of the mission is well approximated
    in the target's plane. This is a rotation, not a projection of a tilted
    orbit onto the equator, so the in-plane radius and speed are preserved to
    O(di_rel^2); the dropped out-of-plane component is reported alongside as
    `oop_frac` so the approximation is visible rather than assumed.
    """
    inc = float(d['target_inc'][i]); raan = float(d['target_raan'][i])
    ci, si = math.cos(inc), math.sin(inc)
    cO, sO = math.cos(raan), math.sin(raan)
    # rows of R = the target-plane basis (node, in-plane-perp, h)
    e1 = np.array([cO, sO, 0.0])
    e3 = np.array([si * sO, -si * cO, ci])
    e2 = np.cross(e3, e1)
    R = np.stack([e1, e2, e3])
    r = np.array([float(d[f'{body}_x'][i]), float(d[f'{body}_y'][i]),
                  float(d[f'{body}_z'][i])])
    v = np.array([float(d[f'{body}_vx'][i]), float(d[f'{body}_vy'][i]),
                  float(d[f'{body}_vz'][i])])
    rr, vv = R @ r, R @ v
    oop = (abs(rr[2]) / max(np.linalg.norm(rr), 1.0),
           abs(vv[2]) / max(np.linalg.norm(vv), 1.0))
    return rr[:2], vv[:2], max(oop)


def lambert_split(rc0, vc0, rt0, vt0, wait, tof, revs, branch):
    """(dv1, dv2) of the optimal schedule — `baseline_search` returns only the
    sum, and the split is what lets the box be priced."""
    if not (np.isfinite(wait) and np.isfinite(tof) and tof > 0):
        return float('nan'), float('nan')
    rc, vc = T1.propagate_rv(rc0, vc0, wait)
    r2, v2t = T1.propagate_rv(rt0, vt0, wait + tof)
    v1, v2, ok = T1.lambert_2d(np.asarray(rc), np.asarray(r2), tof,
                               revs=int(max(revs, 0)), branch=int(branch))
    if not np.all(ok):
        return float('nan'), float('nan')
    return (float(np.linalg.norm(v1 - vc)), float(np.linalg.norm(v2t - v2)))


def episode_row(path, lam_mode, n_wait=90, n_tof=110, box_vel=None):
    d = np.load(path)
    steps = int(d['episode_steps'][0])
    cause = int(d['terminal_cause'][0])
    dec = decompose(d)
    a_t = float(d['target_a'][0]); a_s = float(d['sat_a'][0])
    P_tgt = TWO_PI * math.sqrt(a_t ** 3 / MU)
    T_ep = steps * DT

    oop = 0.0
    if lam_mode == 'inplane' and 'sat_z' in d:
        rc0, vc0, o1 = _inplane_state(d, 0, 'sat')
        rt0, vt0, o2 = _inplane_state(d, 0, 'target')
        oop = max(o1, o2)
    else:
        rc0 = np.array([float(d['sat_x'][0]), float(d['sat_y'][0])])
        vc0 = np.array([float(d['sat_vx'][0]), float(d['sat_vy'][0])])
        rt0 = np.array([float(d['target_x'][0]), float(d['target_y'][0])])
        vt0 = np.array([float(d['target_vx'][0]), float(d['target_vy'][0])])

    # TIME-MATCHED Lambert: the same mission clock the policy actually spent.
    R = min(20, int(T_ep / P_tgt) + 1)
    dv_m, wait_m, tof_m, rev_m, br_m = T1.baseline_search(
        rc0, vc0, rt0, vt0, wait_max=T_ep, tof_min=600.0, tof_max=T_ep,
        max_revs=R, n_wait=n_wait, n_tof=n_tof, budget=T_ep)

    # ── the comparator is not symmetric, and the asymmetry is priceable ────
    # Lambert is held to EXACT rendezvous (0 m, 0 m/s). The policy only has to
    # reach the success box, and the box grants |v_rel| < rel_vel_tol at
    # arrival — so a schedule whose second impulse stops short by up to that
    # tolerance still satisfies the env's own test. Crediting it is exact and
    # tight for the velocity half. T1 named this bias but could not remove it;
    # at a 50 m/s box it is most of the second impulse, which is why the raw
    # ratio can sit below 1.0 without anything being wrong.
    #
    # The 30 km POSITION slack is NOT credited — it is worth far less (a few
    # seconds of along-track drift) and pricing it needs a different solve, so
    # the credited baseline stays a conservative one.
    dv1, dv2 = lambert_split(rc0, vc0, rt0, vt0, wait_m, tof_m, rev_m, br_m)
    dv_credited = float('nan')
    if np.isfinite(dv1) and box_vel is not None:
        dv_credited = dv1 + max(0.0, dv2 - float(box_vel))

    # Burn-path teleport (T1's diagnostic). Pre-fix this was a median 24 deg of
    # FREE along-track phase per episode, which is what made T1's sub-1.0
    # ratios inadmissible. Carried per episode so this audit states rather than
    # assumes that the corrected dynamics hold.
    tele_abs = teleport_abs_deg(d)

    # The policy's IN-PLANE delta-v is what a coplanar Lambert prices: the
    # tangential + radial burns. The plane budget is excluded because the
    # baseline cannot buy a plane change at all.
    dv_inplane = dec['dv_tangential'] + dec['dv_radial']
    row = dict(
        episode=os.path.basename(path), terminal_cause=cause,
        success=int(cause == 1), steps=steps, episode_orbits=T_ep / P_tgt,
        a_sat_km=a_s / 1e3, a_tgt_km=a_t / 1e3,
        hohmann_dv=T1.hohmann_dv(a_s, a_t),
        lambert_dv_timematched=dv_m,
        lambert_dv1=dv1, lambert_dv2=dv2,
        lambert_dv_boxcredited=dv_credited,
        lambert_wait_s=wait_m, lambert_tof_s=tof_m, lambert_revs=rev_m,
        inplane_oop_frac=oop, teleport_abs_deg=tele_abs, **dec)
    row['ratio_total'] = dec['dv_total'] / dv_m if dv_m > 0 else float('nan')
    row['ratio_inplane'] = dv_inplane / dv_m if dv_m > 0 else float('nan')
    row['ratio_boxcredited'] = (dv_inplane / dv_credited
                                if np.isfinite(dv_credited) and dv_credited > 0
                                else float('nan'))
    row['dv_inplane'] = dv_inplane
    return row


def _box_vel(spec):
    """The rel-velocity tolerance the lineage's own success test uses."""
    if spec['mode'] == 'guidance':
        return float(spec['kw'].get('rel_vel_tol_ms', 50.0))
    if spec['mode'] == 't11':
        import t11_eval as T11E
        return float(T11E.env_kwargs(spec['cell'], 'truth', None)['rel_vel_tol_ms'])
    if spec.get('box'):
        return float(ER3.BOXES[spec['box']][1])
    return float(ER3.RUNGS[spec['rung']]['rel_vel_tol_ms'])


def q(x, p):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(x, p)) if x.size else float('nan')


def summarise(name, spec, rows):
    ok = [r for r in rows if r['success']]
    src = ok if ok else rows
    return dict(
        lineage=name, note=spec['note'], lambert=spec['lambert'],
        n=len(rows), n_success=len(ok),
        success_rate=len(ok) / max(len(rows), 1),
        dv_total_med=q([r['dv_total'] for r in src], 50),
        dv_total_p90=q([r['dv_total'] for r in src], 90),
        ratio_med=q([r['ratio_total'] for r in src], 50),
        ratio_p75=q([r['ratio_total'] for r in src], 75),
        ratio_p90=q([r['ratio_total'] for r in src], 90),
        ratio_inplane_med=q([r['ratio_inplane'] for r in src], 50),
        ratio_inplane_p90=q([r['ratio_inplane'] for r in src], 90),
        ratio_boxcredited_med=q([r['ratio_boxcredited'] for r in src], 50),
        ratio_boxcredited_p75=q([r['ratio_boxcredited'] for r in src], 75),
        ratio_boxcredited_p90=q([r['ratio_boxcredited'] for r in src], 90),
        di_rel_med=q([r.get('di_rel_deg', float('nan')) for r in src], 50),
        plane_opt_med=q([r.get('plane_opt_dv', float('nan')) for r in src], 50),
        plane_ratio_med=q([r.get('plane_ratio', float('nan')) for r in src], 50),
        plane_ratio_p90=q([r.get('plane_ratio', float('nan')) for r in src], 90),
        teleport_abs_med=q([r['teleport_abs_deg'] for r in src], 50),
        teleport_abs_max=max([r['teleport_abs_deg'] for r in src] or [0.0]),
        dv_tan_med=q([r['dv_tangential'] for r in src], 50),
        dv_plane_med=q([r['dv_plane'] for r in src], 50),
        dv_radial_med=q([r['dv_radial'] for r in src], 50),
        dv_fine_med=q([r['dv_fine'] for r in src], 50),
        dv_coarse_med=q([r['dv_coarse'] for r in src], 50),
        n_burns_med=q([r['n_burns'] for r in src], 50),
        lambert_med=q([r['lambert_dv_timematched'] for r in src], 50),
        oop_max=max([r['inplane_oop_frac'] for r in src] or [0.0]),
        orbits_med=q([r['episode_orbits'] for r in src], 50))


def plane_floor(d):
    """(di_rel at t0 in deg, the impulsive plane-change optimum in m/s).

    A pure inclination change of di at circular speed v_c costs
    2 v_c sin(di/2), and that is the floor no impulsive schedule beats — so it
    is what says whether a lineage's plane budget has headroom or is already
    spent optimally. Measured per episode against the REALIZED di_rel, because
    di is drawn as di_max*sqrt(U): quoting the floor at di_max would understate
    the ratio by ~1.4x on the median episode.
    """
    def hhat(inc, raan):
        return np.array([math.sin(inc) * math.sin(raan),
                         -math.sin(inc) * math.cos(raan), math.cos(inc)])
    if 'sat_inc' not in d:
        return 0.0, 0.0
    hs = hhat(float(d['sat_inc'][0]), float(d['sat_raan'][0]))
    ht = hhat(float(d['target_inc'][0]), float(d['target_raan'][0]))
    di = math.atan2(float(np.linalg.norm(np.cross(ht, hs))), float(hs @ ht))
    v_c = math.sqrt(MU / max(float(d['target_a'][0]), 1.0))
    return math.degrees(di), 2.0 * v_c * math.sin(0.5 * di)


def retele():
    """Recompute the cheap (non-Lambert) columns in place.

    Kept as its own pass because these fixes are independent of the Lambert
    columns, which cost minutes per lineage to regenerate.
    """
    rows = list(csv.DictReader(open(CSV_OUT)))
    for r in rows:
        p = os.path.join(TRAJ_ROOT, r['lineage'], r['episode'])
        if os.path.exists(p):
            d = np.load(p)
            r['teleport_abs_deg'] = teleport_abs_deg(d)
            di, opt = plane_floor(d)
            r['di_rel_deg'] = di
            r['plane_opt_dv'] = opt
            r['plane_ratio'] = (float(r['dv_plane']) / opt
                                if opt > 1e-9 else float('nan'))
    with open(CSV_OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'refreshed teleport_abs_deg in {CSV_OUT} ({len(rows)} rows)')


def report():
    """Build FUEL_AUDIT.md from the CSV."""
    rows = [dict(r) for r in csv.DictReader(open(CSV_OUT))]
    for r in rows:
        for k, v in list(r.items()):
            if k not in ('lineage', 'episode'):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float('nan')
    out = []
    for name in LINEAGES:
        rs = [r for r in rows if r['lineage'] == name]
        if not rs:
            continue
        s = summarise(name, LINEAGES[name], rs)
        s['box_vel_ms'] = _box_vel(LINEAGES[name])
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--lineages', default=','.join(LINEAGES))
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--retele', action='store_true',
                    help='recompute only the teleport column, in place')
    ap.add_argument('--report', action='store_true',
                    help='print per-lineage summaries from the CSV')
    ap.add_argument('--n-wait', type=int, default=90)
    ap.add_argument('--n-tof', type=int, default=110)
    a = ap.parse_args()

    if a.retele:
        retele(); return 0
    if a.report:
        for s_ in report():
            print(s_)
        return 0
    names = [n for n in a.lineages.split(',') if n in LINEAGES]
    all_rows, summaries = [], []
    for name in names:
        spec = LINEAGES[name]
        traj_dir = os.path.join(TRAJ_ROOT, name)
        if a.analyze_only:
            paths = sorted(glob.glob(os.path.join(traj_dir, 'ep_*.npz')))
        else:
            print(f'== collecting {name} ({spec["note"]})', flush=True)
            paths = collect(name, spec, a.episodes, a.seed)
        box_vel = _box_vel(spec)
        print(f'== analysing {name}: {len(paths)} episodes '
              f'(box vel tol {box_vel} m/s)', flush=True)
        t0 = time.time()
        rows = []
        for i, p in enumerate(paths):
            r = episode_row(p, spec['lambert'], a.n_wait, a.n_tof,
                            box_vel=box_vel)
            r['lineage'] = name
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f'    {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)',
                      flush=True)
        all_rows += rows
        s = summarise(name, spec, rows)
        s['box_vel_ms'] = box_vel
        summaries.append(s)
        print(f"   {name}: success {s['n_success']}/{s['n']}  "
              f"dv med {s['dv_total_med']:.0f} m/s  raw ratio med "
              f"{s['ratio_med']:.2f}  box-credited med "
              f"{s['ratio_boxcredited_med']:.2f}  teleport {s['teleport_abs_med']:.3f} deg",
              flush=True)

    if all_rows:
        # MERGE, don't clobber. This used to truncate the file to whatever
        # --lineages named, so auditing one lineage silently destroyed the
        # other three -- a 4-episode smoke run deleted 800 rows of measured
        # data that only existed because it happened to be committed. Rows for
        # the lineages just measured are replaced; every other lineage in the
        # file is carried through untouched.
        os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
        prev = []
        if os.path.exists(CSV_OUT):
            done = {r['lineage'] for r in all_rows}
            prev = [r for r in csv.DictReader(open(CSV_OUT))
                    if r['lineage'] not in done]
        merged = prev + all_rows
        keys = sorted({k for r in merged for k in r})
        with open(CSV_OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, restval='')
            w.writeheader()
            w.writerows(merged)
        print(f'wrote {CSV_OUT} ({len(merged)} rows: {len(all_rows)} new, '
              f'{len(prev)} carried)')
    for s in summaries:
        print(s)
    return 0


if __name__ == '__main__':
    sys.exit(main())
