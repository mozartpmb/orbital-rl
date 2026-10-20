#!/usr/bin/env python3
"""Deck figures from the T15e 3-D "seven-skill generalist" policy.

READ-ONLY with respect to the repo: the main checkout at REPO is imported (its
compiled `binding.cpython-312-darwin.so` is the only build that exists) and the
C env's own per-sub-step trajectory log is dumped into a scratch directory.
Nothing under REPO is written.

What it produces
----------------
    figs/traj3d_rendezvous.pdf   Figure A — one successful bearings-only 3-D
                                 rendezvous: inertial 3-D view, target-relative
                                 panels, range-to-target with the success box.
    figs/traj3d_overlay.pdf      Figure A' — N-episode 3-D overlay.
    figs/nav3d_estimate.pdf      Figure B — the EKF/MSC target-position error
                                 with its 3-sigma envelope, acquisition latency,
                                 and the true range, for the SAME episode.

Provenance
----------
The rollout replicates `scripts/orbital/extj2/t11_eval.py`'s loop exactly
(same `env_kwargs`, argmax over `LSTMWrapper.forward_eval`, hidden state zeroed
at each episode boundary, `torch.set_num_threads(1)`, `filter_impl='py'`), and
imports that module rather than re-typing its kwargs. At `--seed 123` the
episodes are therefore the first N of the published 200-episode cell results in
`web_data/results/t15e_dagger/fin_<cell>.json`.

All plotted truth is MEAN ELEMENTS under `j2_mode=1` (the evaluator's own
disclaimer): the C env propagates mean elements with secular J2 rates, so the
short-period oscillation is absent by construction, not by filtering.

Two state sources, deliberately:
  * the dense path  — the C trajectory log, one record per 60 s SUB-STEP, so
    warps (tau up to 1440) leave no gaps in the drawn orbit;
  * the nav record  — one sample per DECISION, read from the filter before the
    step, because that is the estimate the policy actually acted on.
"""

import argparse
import math
import os
import sys

import numpy as np

REPO = '/Users/pete/space_training'
sys.path.insert(0, os.path.join(REPO, 'pufferlib'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'orbital', 'extj2'))

import torch                                                        # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                             # noqa: F401,E402

import t11_eval as TE                                               # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav       # noqa: E402
from pufferlib.ocean.orbital.orbital import TRAJ_COLS                # noqa: E402
from pufferlib.models import Default, LSTMWrapper                    # noqa: E402
from pufferlib.ocean.orbital import t11_cells as T                   # noqa: E402

R_EARTH_KM = 6371.0
VE = TE.VE
DRY = TE.DRY
CAUSES = TE.CAUSES

# ── palette (dataviz reference instance, light surface; validated trio) ──────
C_CHASER = '#e34948'   # series red
C_TARGET = '#2a78d6'   # series blue
C_BURN = '#eb6834'     # series orange
C_EARTH = '#9ec5f4'    # sequential blue step 200
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
SURF = '#fcfcfb'


def style():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'savefig.facecolor': SURF,
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'grid.color': GRID, 'grid.linewidth': 0.5,
        'legend.frameon': False, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.2,
    })


# ─────────────────────────────── rollout ────────────────────────────────────
def rollout(ckpt, cell, episodes, seed, scratch, nav_mode='bearings_only'):
    """Replicate t11_eval.run()'s loop, recording the nav state per decision.

    Returns (episodes_list, traj_dir). Each episode dict carries its summary
    plus the per-decision nav record; the dense sub-step path lives in the
    .npz the C env dumped at `npz`.
    """
    traj_dir = os.path.join(scratch, f'traj_{cell}')
    os.makedirs(traj_dir, exist_ok=True)
    for f in os.listdir(traj_dir):                      # a stale dump would
        os.remove(os.path.join(traj_dir, f))            # mis-number episodes

    kw = TE.env_kwargs(cell, nav_mode)
    kw.update(traj_log_dir=traj_dir, traj_log_every=1)
    env = OrbitalNav(num_envs=1, nav_mode=nav_mode, **kw)
    env._acq_real = (nav_mode == 'bearings_only')
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(ckpt, map_location='cpu',
                                      weights_only=True))
    policy.eval()
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    idx0 = np.array([0])
    obs, _ = env.reset(seed=seed)
    hid = {'lstm_h': torch.zeros(1, policy.hidden_size),
           'lstm_c': torch.zeros(1, policy.hidden_size)}

    def new_ep():
        ec = env.episode_cells()
        return dict(cap=float(ec[0, 1]), fuel=float(ec[0, 2]),
                    ref=TE.direct_ref(env.get_state()[0]), nav=[], acts=[],
                    _lat0=len(getattr(getattr(env, '_acq', None),
                                      'acq_latency_s', ())))

    cur = new_ep()
    out = []
    while len(out) < episodes:
        st = env.get_state()[0].copy()           # NB: get_state() hands back a
        # shared buffer that `_decode()` overwrites inside `_nav_step` — copy.
        rec = dict(step=float(st[29]), fuel=float(st[14]),
                   sat_r=st[8:11].copy(), tgt_r=st[23:26].copy(),
                   sat_h=st[5:8].copy(), tgt_h=st[20:23].copy())
        if nav_mode != 'truth':
            x, P = env._filt.mean_cov(idx0)
            rec['est_r'] = x[0, :3].copy()
            rec['sig_pos'] = float(np.sqrt(max(np.trace(P[0, :3, :3]), 0.0)))
            rec['acq'] = bool(env._acq.acquired[0]) if hasattr(env, '_acq') else True
        cur['nav'].append(rec)

        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), hid)
            a = int(torch.argmax(logits, dim=-1).item())
        cur['acts'].append(a)
        rec['action'] = a
        f_before = float(st[14])

        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))
        # The surrogate stamps the true acquisition instant in SIM SECONDS the
        # tick it fires; `_nav_info` drains the list every `log_interval` ticks,
        # so snapshot it here rather than inferring the instant from which
        # DECISION first saw acquired=True (a tau=60 warp would quantise that
        # to the whole hour).
        if (hasattr(env, '_acq') and 'acq_s' not in cur
                and len(env._acq.acq_latency_s) > cur['_lat0']):
            cur['acq_s'] = float(env._acq.acq_latency_s[-1])
        if term[0]:
            steps, cause = env.last_episode_result(0)
            m_tot = DRY / max(1.0 - f_before, 1e-12)
            fuel0 = DRY * cur['fuel'] / (1.0 - cur['fuel'])
            cur.update(cause=int(cause), steps=int(steps),
                       dv_used=VE * math.log((DRY + fuel0)
                                             / max(m_tot, DRY + 1e-9)),
                       budget=-VE * math.log(1.0 - cur['fuel']),
                       ep_index=len(out) + 1)
            cur['npz'] = os.path.join(traj_dir, f"ep_{len(out) + 1:07d}.npz")
            out.append(cur)
            cur = new_ep()
            hid = {'lstm_h': torch.zeros(1, policy.hidden_size),
                   'lstm_c': torch.zeros(1, policy.hidden_size)}
    return out, traj_dir


# ───────────────────────── geometry from the dense log ──────────────────────
def load_dense(npz_path):
    d = np.load(npz_path)
    g = {c: d[c] for c in TRAJ_COLS}
    rs = np.stack([g['sat_x'], g['sat_y'], g['sat_z']], 1)
    rt = np.stack([g['target_x'], g['target_y'], g['target_z']], 1)
    vt = np.stack([g['target_vx'], g['target_vy'], g['target_vz']], 1)
    # Target orbit normal from its own r x v — the log's inc/raan columns would
    # need the env's h-hat convention re-derived; r x v cannot disagree with it.
    h = np.cross(rt, vt)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
    rt_n = np.maximum(np.linalg.norm(rt, axis=1), 1.0)
    r_hat = rt / rt_n[:, None]
    t_hat = np.cross(h, r_hat)
    dr = rs - rt
    phase = np.degrees(np.arctan2(np.einsum('ni,ni->n', rs, t_hat),
                                  np.einsum('ni,ni->n', rs, r_hat)))
    # continuous phase: unwrap the +-180 seam so the loop is one curve
    phase_cut = np.degrees(np.unwrap(np.radians(phase)))
    # anchor so the final (rendezvous) point sits at the target's 0 deg
    phase_cut -= 360.0 * np.round(phase_cut[-1] / 360.0)
    return dict(
        t_h=g['sim_time'] / 3600.0,
        rs_km=rs / 1e3, rt_km=rt / 1e3,
        dv=g['delta_v'], action=g['action'], fuel=g['fuel'],
        rng_km=np.linalg.norm(dr, axis=1) / 1e3,
        # signed along-track angle about the target's own normal
        phase_deg=phase, phase_cut=phase_cut,
        radial_km=(np.linalg.norm(rs, axis=1) - rt_n) / 1e3,
        cross_km=np.einsum('ni,ni->n', dr, h) / 1e3,
        a_sat_km=g['sat_a'] / 1e3, a_tgt_km=g['target_a'] / 1e3)


def ep_stats(ep):
    n0 = ep['nav'][0]
    rs, rt = n0['sat_r'], n0['tgt_r']
    gap = math.degrees(math.acos(float(np.clip(
        np.dot(rs, rt) / (np.linalg.norm(rs) * np.linalg.norm(rt)), -1, 1))))
    plane = math.degrees(math.acos(float(np.clip(
        np.dot(n0['sat_h'], n0['tgt_h']), -1, 1))))
    return dict(gap_deg=gap, plane_deg=plane,
                success=(ep['cause'] == 1), cause=CAUSES[ep['cause']],
                dv=ep['dv_used'], budget=ep['budget'],
                hours=ep['steps'] / 60.0, decisions=len(ep['acts']),
                range0_km=float(np.linalg.norm(rs - rt)) / 1e3)


# ──────────────────────────────── Figure A ──────────────────────────────────
def earth(ax, n=36):
    u, v = np.mgrid[0:2 * np.pi:n * 1j, 0:np.pi:(n // 2) * 1j]
    ax.plot_surface(R_EARTH_KM * np.cos(u) * np.sin(v),
                    R_EARTH_KM * np.sin(u) * np.sin(v),
                    R_EARTH_KM * np.cos(v),
                    color=C_EARTH, alpha=0.30, linewidth=0,
                    shade=True, zorder=0)


def equal3d(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, float((hi - lo).max()) / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def fig_a(ep, stats, cell, box_r_km, out):
    D = load_dense(ep['npz'])
    bi = np.flatnonzero(D['dv'] > 1e-6)
    dec = max(1, len(D['t_h']) // 4000)
    s, t = D['rs_km'][::dec], D['rt_km'][::dec]

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1.0],
                          height_ratios=[1.35, 0.8, 1.0],
                          hspace=0.58, wspace=0.20,
                          left=0.02, right=0.97, top=0.86, bottom=0.08)

    # (a) inertial 3-D
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    earth(ax)
    ax.plot(*t.T, color=C_TARGET, lw=0.7, alpha=0.85, label='target orbit')
    ax.plot(*s.T, color=C_CHASER, lw=1.1, alpha=0.95, label='chaser path')
    if len(bi):
        ax.scatter(*D['rs_km'][bi].T, marker='^', s=22, color=C_BURN,
                   depthshade=False, label=f'burns ({len(bi)})', zorder=5)
    ax.scatter(*D['rs_km'][0], marker='o', s=34, facecolor='none',
               edgecolor=C_CHASER, lw=1.4, label='chaser t=0')
    ax.scatter(*D['rt_km'][0], marker='o', s=34, facecolor='none',
               edgecolor=C_TARGET, lw=1.4, label='target t=0')
    ax.scatter(*D['rs_km'][-1], marker='*', s=150, color='#006300',
               depthshade=False, label='rendezvous', zorder=6)
    equal3d(ax, np.vstack([s, t, [[-R_EARTH_KM] * 3, [R_EARTH_KM] * 3]]))
    ax.set_xlabel('x (km)', labelpad=-4)
    ax.set_ylabel('y (km)', labelpad=-4)
    ax.set_zlabel('z (km)', labelpad=-4)
    ax.tick_params(labelsize=6, pad=-1)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.set_facecolor(SURF)
        p.pane.set_edgecolor(GRID)
    ax.set_title('(a) inertial frame — ECI, mean elements', pad=-2, loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.04, 0.98), labelspacing=0.3)

    # (b) target-relative: along-track phase vs radial offset
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(D['phase_cut'], D['radial_km'], color=C_CHASER, lw=0.9)
    if len(bi):
        ax2.scatter(D['phase_cut'][bi], D['radial_km'][bi], marker='^', s=18,
                    color=C_BURN, zorder=4, label='burn')
    ax2.scatter([D['phase_cut'][0]], [D['radial_km'][0]], marker='o', s=30,
                facecolor='none', edgecolor=C_CHASER, lw=1.3, label='t=0')
    ax2.scatter([0], [0], marker='*', s=130, color=C_TARGET, zorder=5,
                label='target')
    ax2.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax2.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax2.set_xlabel('along-track phase relative to target (deg, continuous)')
    ax2.set_ylabel('radial offset (km)')
    ax2.set_title('(b) target-relative: phasing loop',
                  loc='left')
    ax2.grid(True, alpha=0.6, lw=0.4)
    ax2.legend(loc='best', ncols=2)

    # (b2) cross-track offset vs time — where the plane difference lives
    axc = fig.add_subplot(gs[1, 1])
    axc.plot(D['t_h'], D['cross_km'], color=C_TARGET, lw=0.8)
    axc.axhline(0, color=GRID, lw=0.8, zorder=0)
    if len(bi):
        axc.scatter(D['t_h'][bi], D['cross_km'][bi], marker='^', s=14,
                    color=C_BURN, zorder=4)
    axc.set_ylabel('cross-track (km)')
    axc.set_xlabel('time (h)', labelpad=1)
    axc.set_title(f"(b2) out-of-plane offset — relative plane "
                  f"{stats['plane_deg']:.2f}°", loc='left')
    axc.grid(True, alpha=0.6, lw=0.4)

    # (c) range to target
    ax3 = fig.add_subplot(gs[2, 1], sharex=axc)
    ax3.plot(D['t_h'], D['rng_km'], color=C_CHASER, lw=0.9,
             label='range to target')
    ax3.axhline(box_r_km, color='#006300', lw=1.0, ls='--',
                label=f'success box {box_r_km:.0f} km')
    if len(bi):
        ax3.scatter(D['t_h'][bi], D['rng_km'][bi], marker='^', s=16,
                    color=C_BURN, zorder=4, label='burn')
    ax3.set_yscale('log')
    ax3.set_xlabel('time (h)')
    ax3.set_ylabel('range (km, log)')
    ax3.set_ylim(bottom=box_r_km / 6.0)
    ax3.set_title('(c) closure to the success box', loc='left')
    ax3.grid(True, which='both', alpha=0.5, lw=0.4)
    ax3.legend(loc='lower left', ncols=3, fontsize=6.5)

    fig.suptitle(
        f"3-D rendezvous under J2, bearings-only navigation — cell {cell}, "
        f"T15e generalist (31-action head)\n"
        f"initial phase gap {stats['gap_deg']:.0f}°  ·  initial range "
        f"{stats['range0_km']:,.0f} km  ·  relative plane "
        f"{stats['plane_deg']:.2f}°  ·  "
        f"Δv {stats['dv']:.0f} / {stats['budget']:.0f} m/s  ·  "
        f"{stats['hours']:.1f} h  ·  {stats['decisions']} decisions  ·  "
        f"{stats['cause'].upper()}",
        fontsize=9, y=0.985, ha='center')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return D, bi


def fig_overlay(eps, cell, out, nmax=20):
    fig = plt.figure(figsize=(6.4, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    earth(ax)
    pts = [np.array([[-R_EARTH_KM] * 3, [R_EARTH_KM] * 3])]
    nw = nf = 0
    for ep in eps[:nmax]:
        if not os.path.exists(ep['npz']):
            continue
        D = load_dense(ep['npz'])
        dec = max(1, len(D['t_h']) // 2500)
        s = D['rs_km'][::dec]
        ok = ep['cause'] == 1
        nw, nf = nw + ok, nf + (not ok)
        ax.plot(*s.T, color=(C_CHASER if ok else MUTED),
                lw=0.55, alpha=0.55 if ok else 0.8)
        ax.plot(*D['rt_km'][::dec].T, color=C_TARGET, lw=0.3, alpha=0.22)
        pts.append(s)
    equal3d(ax, np.vstack(pts))
    ax.set_xlabel('x (km)', labelpad=-4)
    ax.set_ylabel('y (km)', labelpad=-4)
    ax.set_zlabel('z (km)', labelpad=-4)
    ax.tick_params(labelsize=6, pad=-1)
    # Look near-edge-on down the common node line (raan_target_sample=0 pins
    # every target's node at 0), which is the view that fans out the sampled
    # inclination spread instead of stacking it.
    ax.view_init(elev=22, azim=-14)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.set_facecolor(SURF)
        p.pane.set_edgecolor(GRID)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_CHASER, lw=1.2,
                label=f'chaser, success ({nw})')]
    if nf:
        h.append(Line2D([], [], color=MUTED, lw=1.2,
                        label=f'chaser, failure ({nf})'))
    h.append(Line2D([], [], color=C_TARGET, lw=1.0, alpha=0.5,
                    label='target orbits'))
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(-0.06, 0.99))
    ax.set_title(f'{nw + nf} independent episodes — cell {cell}, bearings-only\n'
                 f'(ECI, mean elements under J2; i_target 30-60°, '
                 f'common node by construction)', fontsize=9, loc='left')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return nw, nf


# ──────────────────────────────── Figure B ──────────────────────────────────
def fig_b(ep, stats, cell, out, acq_min_s=2700.0):
    nav = ep['nav']
    t = np.array([r['step'] for r in nav]) * 60.0 / 3600.0
    err = np.array([np.linalg.norm(r['est_r'] - r['tgt_r']) for r in nav]) / 1e3
    sig = np.array([r['sig_pos'] for r in nav]) / 1e3
    rng = np.array([np.linalg.norm(r['sat_r'] - r['tgt_r']) for r in nav]) / 1e3
    acq = np.array([r['acq'] for r in nav])
    dv_dec = np.array([r['fuel'] for r in nav])
    burns = np.flatnonzero(np.diff(dv_dec) < -1e-12) + 1   # fuel fell => burn

    # The true-range line gets the TERMINAL truth appended: the nav samples are
    # pre-decision, so without it the trace stops one decision short of the box.
    D = load_dense(ep['npz'])
    t_rng = np.append(t, D['t_h'][-1])
    v_rng = np.append(rng, D['rng_km'][-1])

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    t_acq = ep.get('acq_s')
    t_acq = (t_acq / 3600.0 if t_acq is not None
             else (t[np.flatnonzero(acq)[0]] if acq.any() else t[-1]))
    if (~acq).any():
        ax.axvspan(t[0], t_acq, color=GRID, alpha=0.75, zorder=0)
        ax.axvline(t_acq, color=INK2, lw=0.9, ls=':', zorder=1)
        ax.annotate(f'batch IOD acquires at {t_acq * 60:.0f} min '
                    f'(floor {acq_min_s / 60:.0f} min)',
                    xy=(t_acq, 0.985), xycoords=('data', 'axes fraction'),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=7, color=INK2, va='top')
        ax.annotate('shaded: pre-acquisition. The MSC filter is already\n'
                    'running on bearings, but range is weakly observable and\n'
                    'it is seeded from an analytic shell prior — so its '
                    '$3\\sigma$\nUNDERSTATES the error here. The batch IOD '
                    'overwrites the\nstate at the gate; consistency is claimed '
                    'only after it.',
                    xy=(t_acq, 0.90), xycoords=('data', 'axes fraction'),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=6.3, color=INK2, ha='left', va='top')

    # Split at the acquisition index, OVERLAPPING by one sample so the two
    # segments join; a strict `t <= t_acq` mask leaves the single pre-handoff
    # sample as an invisible one-point line.
    ia = int(np.searchsorted(t, t_acq))
    pre, post = slice(0, ia + 1), slice(ia, None)
    ax.plot(t[pre], (3 * sig)[pre], color=C_TARGET, lw=1.0, ls=':', alpha=0.8,
            marker='.', ms=4)
    ax.plot(t[pre], err[pre], color=C_CHASER, lw=1.0, ls=':', alpha=0.8,
            marker='.', ms=4)
    ax.plot(t[post], (3 * sig)[post], color=C_TARGET, lw=1.2, ls='--',
            label='filter $3\\sigma$ position bound')
    ax.plot(t[post], err[post], color=C_CHASER, lw=1.4,
            label='|target position estimate $-$ truth|')
    ax.plot(t_rng, v_rng, color=MUTED, lw=0.9,
            label='true range chaser$\\to$target')
    if len(burns):
        ax.scatter(t[burns], err[burns], marker='^', s=18, color=C_BURN,
                   zorder=5, label=f'burn decisions ({len(burns)})')
    ax.set_yscale('log')
    ax.set_xlabel('time (h)')
    ax.set_ylabel('km (log)')
    ax.set_xlim(t_rng[0], t_rng[-1])
    ax.grid(True, which='both', alpha=0.5, lw=0.4)
    ax.legend(loc='lower left', ncols=2)
    ax.set_title(
        f'Bearings-only navigation inside the control loop — cell {cell}, '
        f'same episode as Figure A\n'
        f'angles-only MSC filter, 1 mrad / 60 s sensor; acquisition floor '
        f'{acq_min_s / 60:.0f} min; error and $3\\sigma$ are of the TARGET '
        f'inertial position', fontsize=8.5, loc='left')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return len(burns)


# ──────────────────────────────── cell table ────────────────────────────────
def cell_table():
    rows = []
    # NO WEIGHT COLUMN on purpose. Sampling weights are a property of a
    # TRAINING mixture, not of a cell: `t15e_dagger_campaign.sh` runs stages at
    # t11_mixture 3, 4 and 5 (T15 / T15B / T15C weights), and which one emitted
    # `t15e_dagger_final.pt` is not resolvable from the campaign script alone.
    # Evaluation pins ONE cell with the mixture off, so no weight enters any
    # number on this slide.
    for name, c in T.CELLS:
        rows.append(dict(
            cell=name, weight=c['weight'], cap=int(c['cap']),
            cap_h=int(c['cap']) / 60.0,
            box=f"{c['box_r'] / 1e3:.0f} km / {c['box_v']:.0f} m/s",
            alt=f"{(c['a_min'] - 6.371e6) / 1e3:.0f}-"
                f"{(c['a_max'] - 6.371e6) / 1e3:.0f} km",
            e_max=c['e_max_target'],
            di=f"{math.degrees(c['di_max']):.2f}°"
               + (f" (min {math.degrees(c['di_min']):.1f}°, node-dominant)"
                  if c['di_min'] > 0 else ''),
            inc=f"{math.degrees(c['i_t_min']):.0f}-"
                f"{math.degrees(c['i_t_max']):.0f}°",
            j2=int(c['j2']),
            fuel=f"{c['fuel_min']:.3f}-{c['fuel_max']:.2f} "
                 f"({-VE * math.log(1 - c['fuel_min']):.0f}-"
                 f"{-VE * math.log(1 - c['fuel_max']):.0f} m/s)"))
    return rows


def print_cell_table():
    print('\n── the seven cells (pufferlib/ocean/orbital/t11_cells.py, CELLS) ──')
    hdr = ('cell', 'cap (sub-steps / h)', 'box r/v', 'altitude band',
           'e_max', 'rel-plane di_max', 'i_target', 'J2', 'fuel frac (dv budget)')
    print(' | '.join(hdr))
    for r in cell_table():
        print(' | '.join([r['cell'],
                          f"{r['cap']} / {r['cap_h']:.0f} h", r['box'],
                          r['alt'], f"{r['e_max']:.2f}", r['di'], r['inc'],
                          'on' if r['j2'] else 'off', r['fuel']]))
    print('All seven are evaluated bearings-only (angles-only MSC/EKF, '
          '1 mrad, 60 s, acq gate 0.20, acq floor 2700 s, nav_max_ticks 120).')


# ──────────────────────────────────── main ──────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=os.path.join(REPO, 'models/t3/t15e_dagger_final.pt'))
    p.add_argument('--cell', default='E1_j2')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--scratch', default='/Users/pete/.claude/jobs/fd4ad701/tmp')
    p.add_argument('--figs', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs'))
    p.add_argument('--episode', type=int, default=0,
                   help='1-based episode to feature; 0 = auto-pick')
    a = p.parse_args()

    style()
    os.makedirs(a.figs, exist_ok=True)
    box_r_km = T.ALL_CELLS[a.cell]['box_r'] / 1e3

    print(f'rollout: cell={a.cell} episodes={a.episodes} seed={a.seed} '
          f'nav=bearings_only ckpt={os.path.basename(a.ckpt)}')
    eps, traj_dir = rollout(a.ckpt, a.cell, a.episodes, a.seed, a.scratch)
    S = [ep_stats(e) for e in eps]
    nsucc = sum(s['success'] for s in S)
    # Provenance, CHECKED rather than asserted: t11_eval.py hashes the whole
    # action stream. Matching it proves that enabling the trajectory log and
    # reading the filter per decision changed neither the env nor the policy,
    # and that these episodes are the first N of the published 200-episode run.
    import hashlib
    md5 = hashlib.md5(np.concatenate([e['acts'] for e in eps])
                      .astype(np.int32).tobytes()).hexdigest()
    print(f'  {nsucc}/{len(eps)} success; action md5 {md5[:12]} '
          f'(compare: t11_eval.py --cell {a.cell} --episodes {a.episodes} '
          f'--seed {a.seed}); dumps in {traj_dir}')
    print('  ep  outcome      gap     plane    range0        dv/budget      h   dec')
    for i, s in enumerate(S, 1):
        print(f"  {i:3d}  {s['cause']:11s} {s['gap_deg']:6.1f}° "
              f"{s['plane_deg']:6.2f}° {s['range0_km']:9,.0f} km "
              f"{s['dv']:6.0f}/{s['budget']:.0f} m/s {s['hours']:6.1f} "
              f"{s['decisions']:5d}")

    if a.episode:
        pick = a.episode - 1
    else:
        # a success that is non-trivial on BOTH axes the figure has to show:
        # a large initial phase gap and a plane difference near the cell's
        # di_max (both normalised, so neither axis can win on its own).
        di_max = math.degrees(T.ALL_CELLS[a.cell]['di_max'])
        cand = [i for i, s in enumerate(S) if s['success'] and s['gap_deg'] > 30]
        if not cand:
            cand = [i for i, s in enumerate(S) if s['success']]
        if not cand:
            raise SystemExit('no successful episode in this rollout')
        pick = max(cand, key=lambda i: (S[i]['plane_deg'] / di_max
                                        + S[i]['gap_deg'] / 180.0))
    ep, st = eps[pick], S[pick]
    print(f"\nfeatured episode {pick + 1}: gap {st['gap_deg']:.1f}°, plane "
          f"{st['plane_deg']:.2f}°, initial range {st['range0_km']:,.0f} km, "
          f"Δv {st['dv']:.1f}/{st['budget']:.0f} m/s, "
          f"{st['hours']:.1f} h ({ep['steps']} sub-steps), "
          f"{st['decisions']} decisions, {st['cause']}")

    pa = os.path.join(a.figs, 'traj3d_rendezvous.pdf')
    D, bi = fig_a(ep, st, a.cell, box_r_km, pa)
    print(f'  wrote {pa}  ({len(D["t_h"])} sub-step records, {len(bi)} burn '
          f'sub-steps, final range {D["rng_km"][-1]:.1f} km)')

    pb = os.path.join(a.figs, 'nav3d_estimate.pdf')
    nb = fig_b(ep, st, a.cell, pb)
    print(f'  wrote {pb}  ({len(ep["nav"])} per-decision nav samples, '
          f'{nb} burn decisions)')

    po = os.path.join(a.figs, 'traj3d_overlay.pdf')
    nw, nf = fig_overlay(eps, a.cell, po)
    print(f'  wrote {po}  ({nw} success / {nf} failure paths)')

    print_cell_table()
    return 0


if __name__ == '__main__':
    sys.exit(main())
