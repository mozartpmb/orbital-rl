"""plot_trajectory.py — visualize orbital RL episode trajectories.

Usage:
    python scripts/orbital/plot_trajectory.py logs/orbital/ep_0000500.npz
    python scripts/orbital/plot_trajectory.py logs/orbital/  # plot all in dir
    python scripts/orbital/plot_trajectory.py logs/orbital/ --overlay  # overlay multiple

Each .npz contains arrays saved by orbital.py's _save_trajectory():
    sim_time, sat_x, sat_y, sat_vx, sat_vy, sat_a, sat_e, sat_theta,
    fuel, action, reward, delta_v, min_conj_dist,
    target_a, target_e, target_x, target_y, num_bodies,
    body_x_0..15, body_y_0..15, body_hard_r_0..15, body_keepout_r_0..15,
    episode_id, episode_reward, col_names
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

R_EARTH = 6.371e6   # m
AU_KM   = 1e3       # display in km


def load_episode(path):
    """Load one .npz file, return dict of arrays."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def draw_orbit_ellipse(ax, a_m, e, color, lw=1.0, ls='--', label=None, alpha=0.6,
                       omega=0.0):
    """Draw an orbit ellipse given semi-major axis, eccentricity, and ω.

    ω (radians) rotates the ellipse about the focus (Earth). When ω=0, periapsis
    lies along +x, matching Phase 1's assumed orientation.
    """
    a = a_m / AU_KM
    b = a * np.sqrt(1 - e**2)
    c = a * e   # distance from center to focus
    # Earth is at one focus; ellipse center sits c units opposite periapsis.
    cx = -c * np.cos(omega)
    cy = -c * np.sin(omega)
    ellipse = patches.Ellipse(
        xy=(cx, cy), width=2*a, height=2*b,
        angle=np.degrees(omega),
        edgecolor=color, facecolor='none',
        linestyle=ls, linewidth=lw, alpha=alpha, label=label,
        transform=ax.transData
    )
    ax.add_patch(ellipse)


def plot_episode(ep, ax=None, title=None, show=True, save_path=None,
                 color_by='time', alpha_traj=0.8):
    """
    Plot a single episode trajectory.

    Parameters
    ----------
    ep : dict  from load_episode()
    color_by : 'time' | 'fuel' | 'action'
    """
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        ax_orbit = axes[0]
        ax_stats = axes[1]
        fig.patch.set_facecolor('#0d1117')
        for a in axes:
            a.set_facecolor('#0d1117')
    else:
        ax_orbit = ax
        ax_stats = None
        fig = ax.figure

    sat_x  = ep['sat_x'] / AU_KM
    sat_y  = ep['sat_y'] / AU_KM
    steps  = len(sat_x)
    t      = np.arange(steps)

    # ── Trajectory line colored by time (green → red) ─────────────────────
    points = np.array([sat_x, sat_y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)

    if color_by == 'time':
        cvals = t[:-1] / max(t[-1], 1)
        cmap  = 'RdYlGn_r'
        clabel = 'Time (step fraction)'
    elif color_by == 'fuel':
        cvals = ep['fuel'][:-1]
        cmap  = 'plasma'
        clabel = 'Fuel fraction'
    elif color_by == 'action':
        cvals = ep['action'][:-1] / 8.0
        cmap  = 'tab10'
        clabel = 'Action'
    else:
        cvals = np.ones(len(segs))
        cmap  = 'Blues'
        clabel = ''

    lc = LineCollection(segs, cmap=cmap, norm=Normalize(0, 1),
                        linewidth=1.2, alpha=alpha_traj, zorder=3)
    lc.set_array(cvals)
    ax_orbit.add_collection(lc)

    # ── Earth ────────────────────────────────────────────────────────────
    earth_r = R_EARTH / AU_KM
    earth_circle = plt.Circle((0, 0), earth_r,
                               color='royalblue', zorder=5, label='Earth')
    ax_orbit.add_patch(earth_circle)
    # Atmosphere keepout ring (200 km altitude)
    atm_r = (R_EARTH + 200e3) / AU_KM
    atm_circle = plt.Circle((0, 0), atm_r,
                             color='steelblue', fill=False,
                             linestyle=':', linewidth=0.8, alpha=0.5)
    ax_orbit.add_patch(atm_circle)

    # ── Find first valid step (step 0 may have zeros from logging bug) ──
    valid_step = 0
    for si in range(len(ep['sat_a'])):
        if ep['sat_a'][si] > 0:
            valid_step = si
            break

    # ── Target orbit ─────────────────────────────────────────────────────
    target_a = float(ep['target_a'][valid_step])
    target_e = float(ep['target_e'][valid_step])
    # target_omega is Phase 2+; Phase 1 files don't have it (default 0)
    target_omega = float(ep['target_omega'][valid_step]) if 'target_omega' in ep else 0.0
    draw_orbit_ellipse(ax_orbit, target_a, target_e,
                       color='deepskyblue', lw=1.5, ls='--', label='Target orbit',
                       omega=target_omega)

    # ── Initial orbit (first valid point) ─────────────────────────────────
    init_a = float(ep['sat_a'][valid_step])
    init_e = float(ep['sat_e'][valid_step])
    init_omega = float(ep['sat_omega'][valid_step]) if 'sat_omega' in ep else 0.0
    draw_orbit_ellipse(ax_orbit, init_a, init_e,
                       color='lime', lw=1.0, ls=':', label='Initial orbit', alpha=0.4,
                       omega=init_omega)

    # ── Compute max range early (needed for debris visual scaling) ────────
    valid_pos = ep['sat_x']**2 + ep['sat_y']**2
    max_r = max(target_a, init_a, np.max(np.sqrt(valid_pos[valid_pos > 0]))) / AU_KM

    # ── Debris bodies ─────────────────────────────────────────────────────
    num_bodies = int(ep['num_bodies'][valid_step])
    debris_drawn = False
    # Minimum visual radius for debris markers (% of plot range)
    min_vis_r = max_r * 0.015  # 1.5% of plot range — visible but not huge
    for i in range(1, num_bodies):   # skip 0 = Earth
        bx = ep[f'body_x_{i}'][valid_step] / AU_KM
        by = ep[f'body_y_{i}'][valid_step] / AU_KM
        kr = ep[f'body_keepout_r_{i}'][valid_step] / AU_KM
        hr = ep[f'body_hard_r_{i}'][valid_step] / AU_KM
        lbl = 'Debris (keepout)' if not debris_drawn else None
        vis_kr = max(kr, min_vis_r)  # scale up for visibility
        vis_hr = max(hr, min_vis_r * 0.4)
        kc = plt.Circle((bx, by), vis_kr, color='orange', fill=False,
                         linestyle='--', linewidth=0.7, alpha=0.5, label=lbl)
        hc = plt.Circle((bx, by), vis_hr,
                         color='orangered', fill=True, alpha=0.8)
        ax_orbit.add_patch(kc)
        ax_orbit.add_patch(hc)
        debris_drawn = True

    # ── Burn markers ──────────────────────────────────────────────────────
    burns = ep['delta_v'] > 0.5
    if np.any(burns):
        bx_pts = sat_x[burns]
        by_pts = sat_y[burns]
        ax_orbit.scatter(bx_pts, by_pts, marker='^', s=20,
                         color='yellow', zorder=6, alpha=0.9, label='Burn')

    # ── Start / End markers ───────────────────────────────────────────────
    ax_orbit.scatter([sat_x[0]], [sat_y[0]], marker='o', s=60,
                     color='lime', zorder=7, label='Start')
    ax_orbit.scatter([sat_x[-1]], [sat_y[-1]], marker='*', s=100,
                     color='red', zorder=7, label='End')

    # ── Colorbar for trajectory ───────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_orbit, fraction=0.03, pad=0.02)
    cbar.set_label(clabel, color='white', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # ── Axes formatting ───────────────────────────────────────────────────
    lim = max_r * 1.2
    ax_orbit.set_xlim(-lim, lim)
    ax_orbit.set_ylim(-lim, lim)
    ax_orbit.set_aspect('equal')
    ax_orbit.set_xlabel('x (km)', color='white')
    ax_orbit.set_ylabel('y (km)', color='white')
    ax_orbit.tick_params(colors='white')
    for spine in ax_orbit.spines.values():
        spine.set_edgecolor('#333')
    ax_orbit.legend(loc='upper right', fontsize=7,
                    facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

    ep_rew  = float(ep['episode_reward'][0]) if 'episode_reward' in ep else float(ep['reward'][-1])
    ep_id   = int(ep['episode_id'][0])       if 'episode_id' in ep   else 0
    total_dv = float(np.sum(ep['delta_v']))
    result  = 'SUCCESS' if ep_rew > 0 else 'FAILURE'
    color   = 'lime' if ep_rew > 0 else 'tomato'

    t_str = title or f"Episode {ep_id}  |  {result}  |  Δv={total_dv:.1f} m/s  |  steps={steps}"
    ax_orbit.set_title(t_str, color=color, fontsize=10, pad=8)

    # ── Stats panel ───────────────────────────────────────────────────────
    if ax_stats is not None:
        ax_stats.set_facecolor('#0d1117')
        step_arr = np.arange(steps)

        # Semi-major axis over time
        ax2 = ax_stats
        a_km = ep['sat_a'] / AU_KM
        t_a  = target_a / AU_KM
        ax2.plot(step_arr, a_km, color='cyan', lw=1.2, label='sat a (km)')
        ax2.axhline(t_a, color='deepskyblue', lw=1, ls='--', alpha=0.8, label=f'target a={t_a:.0f} km')
        ax2.set_ylabel('Semi-major axis (km)', color='white')
        ax2.tick_params(colors='white')

        ax3 = ax2.twinx()
        ax3.plot(step_arr, ep['fuel'], color='orange', lw=1.2, alpha=0.8, label='fuel')
        ax3.set_ylabel('Fuel fraction', color='orange')
        ax3.tick_params(colors='orange')
        ax3.set_ylim(0, 0.2)

        # Burn events as vertical lines
        for s in np.where(burns)[0]:
            ax2.axvline(s, color='yellow', alpha=0.3, lw=0.5)

        ax2.set_xlabel('Step', color='white')
        ax2.set_title('Orbital elements & fuel', color='white', fontsize=9)
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=7,
                   facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
        print(f"Saved: {save_path}")
    if show:
        plt.show()


def plot_overlay(paths, max_episodes=20, save_path=None):
    """Plot multiple episode trajectories overlaid on one axes."""
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    earth_circle = plt.Circle((0, 0), R_EARTH / AU_KM,
                               color='royalblue', zorder=5, label='Earth')
    ax.add_patch(earth_circle)

    for i, path in enumerate(paths[:max_episodes]):
        ep = load_episode(path)
        sat_x = ep['sat_x'] / AU_KM
        sat_y = ep['sat_y'] / AU_KM
        rew   = float(ep['episode_reward'][0]) if 'episode_reward' in ep else 0
        col   = 'lime' if rew > 0 else 'tomato'
        ax.plot(sat_x, sat_y, color=col, alpha=0.3, lw=0.8)

    # Draw target orbit from last episode
    ep = load_episode(paths[-1])
    tomega = float(ep['target_omega'][0]) if 'target_omega' in ep else 0.0
    draw_orbit_ellipse(ax, float(ep['target_a'][0]), float(ep['target_e'][0]),
                       'deepskyblue', lw=1.5, ls='--', label='Target orbit',
                       omega=tomega)

    max_r = float(ep['target_a'][0]) * 1.3 / AU_KM
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect('equal')
    ax.set_xlabel('x (km)', color='white')
    ax.set_ylabel('y (km)', color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'Overlay: {len(paths[:max_episodes])} episodes  '
                 f'(green=success, red=failure)', color='white')
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot orbital RL trajectories')
    parser.add_argument('path', help='.npz file or directory of .npz files')
    parser.add_argument('--overlay', action='store_true',
                        help='Overlay all trajectories on one plot')
    parser.add_argument('--color-by', choices=['time', 'fuel', 'action'],
                        default='time', help='Trajectory color mapping')
    parser.add_argument('--save-dir', default=None,
                        help='Save plots to this directory instead of showing')
    parser.add_argument('--max', type=int, default=20,
                        help='Max episodes to plot (overlay mode)')
    args = parser.parse_args()

    matplotlib.rcParams['axes.facecolor'] = '#0d1117'
    matplotlib.rcParams['figure.facecolor'] = '#0d1117'
    matplotlib.rcParams['text.color'] = 'white'

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, '*.npz')))
        if not paths:
            print(f"No .npz files found in {args.path}")
            sys.exit(1)
        print(f"Found {len(paths)} episodes")
    else:
        paths = [args.path]

    if args.overlay:
        save_path = os.path.join(args.save_dir, 'overlay.png') if args.save_dir else None
        plot_overlay(paths, max_episodes=args.max, save_path=save_path)
    else:
        for path in paths:
            ep = load_episode(path)
            ep_id = int(ep['episode_id'][0]) if 'episode_id' in ep else 0
            save_path = (os.path.join(args.save_dir, f'ep_{ep_id:07d}.png')
                         if args.save_dir else None)
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
            plot_episode(ep, color_by=args.color_by,
                         show=(args.save_dir is None),
                         save_path=save_path)


if __name__ == '__main__':
    main()
