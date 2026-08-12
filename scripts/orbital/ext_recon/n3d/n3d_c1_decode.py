"""N3D-C probe 1 — is the dim3 observation vector invertible to the full 3D truth?

READ-ONLY. Instantiates `puffer_orbital` at the shipped X3 / V-ladder configs with
`dim3_mode=1`, pulls the C-side TRUTH out of the (already existing) trajectory
accessor, and measures, per obs channel group:

  A. what the ext-3d block DOES decode exactly       (i_rel, u_s, rho_N, rhodot_N)
  B. what the 2D-lineage decoder gets WRONG under 3D (target position error)
  C. the conditioning of the only obs-only route to the missing d.o.f. Omega_s
     (the in-plane LVLH slots 33-36), as a function of e_target

Everything is measured against the running C environment, not a reimplementation.

Output: web_data/results/n3d_decode_completeness.csv
        web_data/results/n3d_omega_conditioning.csv
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/Users/pete/space_training/pufferlib')
from pufferlib.ocean.orbital.orbital import Orbital, TRAJ_FLOATS, TRAJ_COLS  # noqa: E402
from pufferlib.ocean.orbital import binding  # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math as nm  # noqa: E402

MU = 3.986004418e14
R_E = 6.371e6
OUT = '/Users/pete/space_training/web_data/results'
COL = {c: i for i, c in enumerate(TRAJ_COLS)}

X3 = dict(dim3_mode=1, legacy_action_space=30, e_max_target=0.05, e_max_sat=0.05,
          di_max_rad=0.017453, a_min_override=6.871e6, a_max_override=7.171e6,
          valid_init_only=1, phase_obs_mode=1, phase_gap_mode=1, shaping_mode=2,
          shape_gamma=1.0, shape_dv_ref_ms=700.0, episode_cap_steps=3000,
          cap_terminal_reward=0.0, init_phase_gap_max=3.14159)

V5 = dict(X3, e_max_target=0.30, e_max_sat=0.05, de_max=0.08, da_max_m=600e3,
          a_min_override=6.671e6, a_max_override=14.371e6, di_max_rad=0.013090,
          episode_cap_steps=6000, obs_alt_scale_m=8e6, lvlh_scale_m=1.5e7)


def truth_rows(env, n):
    """(n, TRAJ_FLOATS) — trajectory record row 0 = the post-reset epoch.

    NOTE (measured): `vec_get_trajectory` returns `env->last_traj_records`,
    which c_reset sets to 1. Mid-episode the count is therefore stale and only
    row 0 is copied — so the probe reads the state at reset, where obs and the
    trajectory record are the SAME epoch by construction. Stepping first and
    taking the last copied row reads a state one decision (~4 deg of arc at
    LEO) behind the observation; that is what the first run of this probe did,
    and it is why every "error" came out at ~3.8 deg / 4.5e5 m.
    """
    buf = np.zeros((12000, TRAJ_FLOATS), dtype=np.float32)
    out = np.zeros((n, TRAJ_FLOATS), dtype=np.float64)
    for i in range(n):
        binding.vec_get_trajectory(env.c_envs, i, buf)
        out[i] = buf[0]
    return out


def coe2rv(a, e, th, om, inc, raan):
    """Vectorised 3-1-3 elements -> (r, v). Identical convention to orbital.h."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(th))
    h = np.sqrt(MU * p)
    xp, yp = r * np.cos(th), r * np.sin(th)
    vxp, vyp = -(MU / h) * np.sin(th), (MU / h) * (e + np.cos(th))
    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(om), np.sin(om)
    ci, si = np.cos(inc), np.sin(inc)
    R11 = cO * cw - sO * sw * ci; R12 = -cO * sw - sO * cw * ci
    R21 = sO * cw + cO * sw * ci; R22 = -sO * sw + cO * cw * ci
    R31 = sw * si;                R32 = cw * si
    rr = np.stack([R11 * xp + R12 * yp, R21 * xp + R22 * yp, R31 * xp + R32 * yp], -1)
    vv = np.stack([R11 * vxp + R12 * vyp, R21 * vxp + R22 * vyp, R31 * vxp + R32 * vyp], -1)
    return rr, vv


def lvlh_inplane(rs, vs, rt, vt, a_t, lvlh_scale):
    """Exact reproduction of orbital.h obs[33-36] (in-plane projection only)."""
    theta_t = np.arctan2(rt[:, 1], rt[:, 0])   # == omega_t + theta_t at i_t = 0
    ct, st = np.cos(theta_t), np.sin(theta_t)
    dxi, dyi = rs[:, 0] - rt[:, 0], rs[:, 1] - rt[:, 1]
    dvxi, dvyi = vs[:, 0] - vt[:, 0], vs[:, 1] - vt[:, 1]
    dx = ct * dxi + st * dyi
    dy = -st * dxi + ct * dyi
    dvx = ct * dvxi + st * dvyi
    dvy = -st * dvxi + ct * dvyi
    n_t = np.sqrt(MU / a_t ** 3)
    dvx = dvx + n_t * dy
    dvy = dvy - n_t * dx
    vc = np.sqrt(MU / a_t)
    return np.stack([dx / lvlh_scale, dy / lvlh_scale, dvx / vc, dvy / vc], -1)


def run(name, cfg, n_env=256, n_step=0, seed=11):
    kw = dict(cfg)
    alt_scale = kw.get('obs_alt_scale_m', 1.6e6)
    lvlh_scale = kw.get('lvlh_scale_m', R_E)
    di_scale = max(kw['di_max_rad'], 0.25 * np.pi / 180.0)
    tmp = f'/tmp/n3d_traj_{name}'
    env = Orbital(num_envs=n_env, traj_log_dir=tmp, traj_log_every=10 ** 9, **kw)
    env.reset(seed=seed)
    # The observation of the RESET epoch...
    o = np.asarray(env.observations, dtype=np.float64).copy()
    # ...and one coast step, purely to make c_step flush `traj_log[0]`, which
    # orbital.h writes at the START of the first step from the pre-step (i.e.
    # reset-epoch) state. Nothing about the recorded row depends on the action.
    env.step(np.zeros(n_env, dtype=np.int32))
    T = truth_rows(env, n_env)

    sat_a = T[:, COL['sat_a']]; sat_e = T[:, COL['sat_e']]
    sat_th = T[:, COL['sat_theta']]; sat_om = T[:, COL['sat_omega']]
    sat_i = T[:, COL['sat_inc']]; sat_O = T[:, COL['sat_raan']]
    tgt_a = T[:, COL['target_a']]; tgt_e = T[:, COL['target_e']]
    tgt_om = T[:, COL['target_omega']]
    tgt_i = T[:, COL['target_inc']]; tgt_O = T[:, COL['target_raan']]
    rs_t = np.stack([T[:, COL['sat_x']], T[:, COL['sat_y']], T[:, COL['sat_z']]], -1)
    rt_t = np.stack([T[:, COL['target_x']], T[:, COL['target_y']], T[:, COL['target_z']]], -1)

    # keep only rows where the traj record is the current epoch (float32 match)
    ok = np.abs(sat_a - (o[:, 0] * alt_scale + R_E)) < 200.0
    ok &= np.abs(sat_e - o[:, 1]) < 1e-4
    rows = []

    def rec(metric, v, unit, note):
        v = np.asarray(v, dtype=np.float64)[ok]
        v = v[np.isfinite(v)]
        rows.append(dict(config=name, n=int(v.size), metric=metric, unit=unit,
                         median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                         max=float(np.max(v)), note=note))

    rec('epoch_match_frac', np.full(n_env, ok.mean()), 'frac', 'traj row == obs epoch')

    # ── A. what the ext-3d block decodes exactly ────────────────────────────
    di_hat = di_scale * np.hypot(o[:, 21], o[:, 22])
    rec('di_rel_err_deg', np.degrees(np.abs(di_hat - sat_i)), 'deg',
        'obs[21,22] -> |di_rel| vs truth inc (target gauge i_t=0)')
    u_hat = np.arctan2(-o[:, 22], o[:, 21])
    u_true = sat_om + sat_th
    rec('u_arglat_err_deg', np.degrees(np.abs(nm.wrap_pi(u_hat - u_true))), 'deg',
        'obs[21,22] -> argument of latitude u = omega_s + theta_s')
    rec('rho_N_err_m', np.abs(o[:, 24] * lvlh_scale - (rs_t[:, 2] - rt_t[:, 2])), 'm',
        'obs[24] -> cross-track separation')

    # ── B. the 2D-lineage decoder applied to a dim3 observation ─────────────
    sat2, tgt2 = nm.recover_states_t3(o, alt_scale)
    r2, v2 = coe2rv(tgt2['a'], tgt2['e'], tgt2['theta'], tgt2['omega'],
                    np.zeros(n_env), np.zeros(n_env))
    rec('t3decode_tgt_pos_err_m', np.linalg.norm(r2 - rt_t, axis=1), 'm',
        '2D decoder target position error under dim3 (Omega_s absorbed into M_t)')
    rec('t3decode_sat_theta_err_deg',
        np.degrees(np.abs(nm.wrap_pi(sat2['theta'] - sat_th))), 'deg',
        '2D decoder chaser true anomaly (unaffected: obs[2,3] are node-referenced)')

    # the same decode with the ONE missing d.o.f. supplied (candidate C getter)
    lam_s = nm.mean_from_true(sat2['theta'], sat2['e']) + sat2['omega'] + sat_O
    dlam = np.arctan2(o[:, 13], o[:, 14])
    Mt3 = np.mod(lam_s - dlam - tgt_om, 2 * np.pi)
    th3 = nm.eccentric_to_true(nm.solve_kepler(Mt3, tgt2['e']), tgt2['e'])
    r3, v3 = coe2rv(tgt2['a'], tgt2['e'], th3, tgt_om, np.zeros(n_env), np.zeros(n_env))
    rec('getter_tgt_pos_err_m', np.linalg.norm(r3 - rt_t, axis=1), 'm',
        'decode WITH chaser Omega_s from a C getter: residual = float32 obs quantisation')

    i_hat = di_hat
    rs3, vs3 = coe2rv(sat2['a'], sat2['e'], sat2['theta'], sat2['omega'], i_hat, sat_O)
    rec('getter_sat_pos_err_m', np.linalg.norm(rs3 - rs_t, axis=1), 'm',
        'chaser rebuilt from obs + (i from obs[21,22], Omega from getter)')

    # ── B2. does either reconstruction REPRODUCE the C env's own obs? ───────
    # Recon A = the existing 2D decoder verbatim + i from obs[21,22], Omega_s
    # set to 0 (the obs-only route). Recon B = same, with Omega_s supplied.
    def recon(alpha_raan, Mt_extra):
        om_s = sat2['omega']
        rs, vs = coe2rv(sat2['a'], sat2['e'], sat2['theta'], om_s, i_hat, alpha_raan)
        Mt = np.mod(nm.mean_from_true(sat2['theta'], sat2['e']) + om_s + Mt_extra
                    - dlam - tgt_om, 2 * np.pi)
        tht = nm.eccentric_to_true(nm.solve_kepler(Mt, tgt2['e']), tgt2['e'])
        rt, vt = coe2rv(tgt2['a'], tgt2['e'], tht, tgt_om,
                        np.zeros(n_env), np.zeros(n_env))
        return rs, vs, rt, vt

    dlam = np.arctan2(o[:, 13], o[:, 14])
    zero = np.zeros(n_env)
    for tag, (al, mx) in (('obsonly_raan0', (zero, zero)),
                          ('getter_raan', (sat_O, sat_O))):
        rs, vs, rt, vt = recon(al, mx)
        lv = lvlh_inplane(rs, vs, rt, vt, tgt2['a'], lvlh_scale)
        rec(f'{tag}_lvlh_pos_err_m',
            np.linalg.norm(lv[:, :2] - o[:, 33:35], axis=1) * lvlh_scale, 'm',
            'rebuilt obs[33,34] vs the C env\'s own obs[33,34]')
        rec(f'{tag}_lvlh_vel_err_ms',
            np.linalg.norm(lv[:, 2:] - o[:, 35:37], axis=1) * np.sqrt(MU / tgt2['a']),
            'm/s', 'rebuilt obs[35,36] vs the C env\'s own obs[35,36]')

    # ── C. conditioning of the obs-only route to Omega_s ────────────────────
    # The observable constraint is dlam = (M_s + w_s + Omega_s) - (M_t + w_t)
    # with w_t pinned by obs[11,12]; so the unobserved family is exactly
    #     Omega_s -> Omega_s + alpha,  M_t -> M_t + alpha,  w_t fixed.
    # At e_t = 0 that is a rigid rotation of BOTH bodies about h_t and every
    # relative channel is invariant. At e_t > 0 the target's mean-anomaly
    # advance is not a rotation and obs[33-36] move by O(a_t e_t).
    dalpha = 1e-4
    obs_l = []
    for s in (-1.0, +1.0):
        al = s * dalpha
        rs, vs, rt, vt = recon(sat_O + al, sat_O + al)
        obs_l.append(lvlh_inplane(rs, vs, rt, vt, tgt2['a'], lvlh_scale))
    d_obs = (obs_l[1] - obs_l[0]) / (2 * dalpha)
    rec('dLVLHpos_dOmega_m_per_rad',
        np.linalg.norm(d_obs[:, :2], axis=1) * lvlh_scale, 'm/rad',
        'sensitivity of obs[33,34] to the Omega_s<->M_t trade (the obs-only solve)')
    rec('dLVLHvel_dOmega_ms_per_rad',
        np.linalg.norm(d_obs[:, 2:], axis=1) * np.sqrt(MU / tgt2['a']), 'm/s/rad',
        'sensitivity of obs[35,36] to the same trade')
    # float32 resolution floor -> smallest resolvable alpha
    q_pos = np.float32(np.abs(o[:, 33]) + 1e-3) * np.finfo(np.float32).eps * lvlh_scale
    rec('omega_resolution_rad',
        q_pos / np.maximum(np.linalg.norm(d_obs[:, :2], axis=1) * lvlh_scale, 1e-30),
        'rad', 'obs float32 quantisation / sensitivity = best-case Omega_s solve accuracy')
    rec('realized_e_target', tgt_e, '-', 'realized target eccentricity')
    rec('realized_di_deg', np.degrees(sat_i), 'deg', 'realized relative inclination')
    rec('target_inc_raan_max', np.abs(tgt_i) + np.abs(tgt_O), 'rad',
        'target plane gauge (must be exactly 0)')
    env.close()
    return rows


if __name__ == '__main__':
    all_rows = []
    all_rows += run('X3_leo_e0.05', X3)
    all_rows += run('V5_wide_e0.30', V5)
    # e_target = 0 control: the decode ambiguity must vanish
    all_rows += run('X3_e0_control', dict(X3, e_max_target=0.0, e_max_sat=0.0))
    # e_target sweep: the conditioning of the obs-only Omega_s solve
    for et in (0.0, 0.005, 0.02, 0.05, 0.15, 0.30):
        all_rows += run(f'sweep_et{et:g}',
                        dict(X3, e_max_target=max(et, 1e-9), e_target_fixed=et,
                             e_max_sat=0.02, a_min_override=7.371e6,
                             a_max_override=7.871e6),
                        n_env=128)
    os.makedirs(OUT, exist_ok=True)
    import csv
    p = os.path.join(OUT, 'n3d_decode_completeness.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    for r in all_rows:
        print(f"{r['config']:16s} {r['metric']:30s} med={r['median']:12.4g} "
              f"p95={r['p95']:12.4g} max={r['max']:12.4g} {r['unit']}")
    print(f"\nwrote {p}")
