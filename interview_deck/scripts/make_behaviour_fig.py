#!/usr/bin/env python3
"""Behaviour profile of the T15e seven-skill generalist, across five cells.

ONE figure, `figs/generalist_behaviour.pdf`:
    (a) per cell, a 100% stacked horizontal bar of DECISION shares by action
        family (coast/warp duration ladder + burn axes);
    (b) per cell, a compact table: median decisions, burns, Dv, duration, and
        success n/20.

READ-ONLY with respect to the main checkout at REPO; scratch output only.

── WHERE THE ACTION FAMILIES COME FROM (not guessed) ────────────────────────
`ACTION_DV` and `ACTION_TAU` are PARSED OUT OF `orbital.h` at run time by
`parse_action_table()`, and then cross-checked, row for row, against
`nav_math.ACTION_TAU` / `nav_math.ACTION_DV_MAG` — the repo's own verbatim
mirrors. Both must agree or the script aborts. So the index -> family map is
derived from the C table twice over, by two independent routes.

The one judgement call is the PRECEDENCE of "fine burn" over the axis
families, and it is taken from the repo rather than invented: `nav_math.py`
names `FINE_INPLANE = [12,13,14,15,18,19]` the "correct set of *in-plane* fine
burns" and deliberately holds `FINE_NORMAL = [20,21]` (normal +/-1 m/s) OUT of
it, because N3D-B measures those two as the 3-D observability treatment rather
than as fine control. This script follows that split exactly: rows 20/21 are
counted as out-of-plane, NOT as fine. The raw per-row counts are printed so
any other split can be recomputed from the output without re-running.

── WHAT A "DECISION" IS ─────────────────────────────────────────────────────
One policy forward pass = one decision = one row of the Discrete(31) head.
A decision advances the C env by `ACTION_TAU[row]` sub-steps of 60 s, so
decisions and sub-steps are NOT interchangeable: the duration column is
sub-steps/60 (hours) and the decision column is the forward-pass count. This
is the same collapse the 3-D figure's "decisions" number uses.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

REPO = '/Users/pete/space_training'
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'pufferlib'))
sys.path.insert(0, _HERE)

import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Patch                                 # noqa: E402

CELLS = ['E1_j2', 'E3_j2', 'LONGRANGE', 'TIGHT_5k1', 'W1_driftwait']
ORBITAL_H = os.path.join(REPO, 'pufferlib/pufferlib/ocean/orbital/orbital.h')

# ── palette ─────────────────────────────────────────────────────────────────
# COMPOSITE encoding on purpose. The four coast families differ only in HOW
# MUCH TIME they skip, which is a magnitude, so they get four steps of the one
# sequential blue ramp (light -> dark = short -> long) and read as one ladder.
# The four burn families are identities, not magnitudes, so they get
# categorical hues. Eight arbitrary categorical hues would have thrown away
# the ladder, which is the single most legible thing in panel (a).
FAMILIES = [
    ('coast ≤5 min',       '#cde2fb'),   # seq 100
    ('coast 30 min–1 h',   '#9ec5f4'),   # seq 200
    ('coast 3–6 h',        '#5598e7'),   # seq 350
    ('coast 24 h',              '#1c5cab'),   # seq 550
    ('along-track burn',        '#e34a33'),   # warm: red
    # aqua, NOT the palette's yellow: orange<->yellow scores dE 13.7 on the
    # validator's normal-vision floor (hard FAIL, <15) and these two segments
    # are ADJACENT in the stack. orange<->aqua is 27.6. Checked, not eyeballed:
    #   validate_palette.js "#1c5cab,#eb6834,#1baf7a,#4a3aa7,#e87ba4" --mode light
    #   -> all adjacent pairs PASS (worst CVD dE 9.2 deutan, normal 27.6).
    ('radial burn',             '#fdae61'),   # warm: orange
    ('out-of-plane burn',       '#99000d'),   # warm: dark red
    ('fine burn ≤2 m/s',   '#fcbba1'),   # warm: salmon
]
FNAMES = [f for f, _ in FAMILIES]
LIGHT_TEXT = {3, 6}          # the two dark fills that need light numerals
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#898781', '#fcfcfb'


# ── the action table, parsed from the C header ──────────────────────────────
def parse_action_table(path=ORBITAL_H):
    """(dv[31,3], tau[31], NUM_ACTIONS) read out of orbital.h."""
    src = open(path).read()

    m = re.search(r'ACTION_DV\[NUM_ACTIONS\]\[3\]\s*=\s*\{(.*?)\n\};', src, re.S)
    if not m:
        raise SystemExit('could not locate ACTION_DV in ' + path)
    dv = [[float(x) for x in row]
          for row in re.findall(r'\{\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,'
                                r'\s*(-?[\d.]+)\s*\}', m.group(1))]

    m = re.search(r'ACTION_TAU\[NUM_ACTIONS\]\s*=\s*\{(.*?)\n\};', src, re.S)
    if not m:
        raise SystemExit('could not locate ACTION_TAU in ' + path)
    body = re.sub(r'/\*.*?\*/', '', m.group(1), flags=re.S)
    tau = [int(x) for x in re.findall(r'-?\d+', body)]

    n = int(re.search(r'#define\s+NUM_ACTIONS\s+(\d+)', src).group(1))
    if not (len(dv) == len(tau) == n):
        raise SystemExit(f'table/NUM_ACTIONS mismatch: dv {len(dv)}, '
                         f'tau {len(tau)}, NUM_ACTIONS {n}')
    return np.array(dv), np.array(tau), n


def family_map():
    """row index -> family index, derived from the parsed C table.

    Cross-checked against the repo's own mirrors; the script refuses to draw a
    chart off a table it could not confirm.
    """
    dv, tau, n = parse_action_table()
    from pufferlib.ocean.orbital_nav import nav_math as nm
    if n != nm.NUM_ACTIONS:
        raise SystemExit('NUM_ACTIONS disagrees with nav_math')
    if not np.array_equal(tau, nm.ACTION_TAU):
        raise SystemExit('parsed ACTION_TAU != nav_math.ACTION_TAU')
    mag = np.linalg.norm(dv, axis=1)
    if not np.allclose(mag, nm.ACTION_DV_MAG, atol=1e-9):
        raise SystemExit('parsed |ACTION_DV| != nav_math.ACTION_DV_MAG')

    fine = set(nm.FINE_INPLANE.tolist())     # repo's own split; 20/21 excluded
    fam = np.full(n, -1, dtype=int)
    for i in range(n):
        if mag[i] == 0.0:                    # coast or warp: bucket by tau
            secs = tau[i] * 60.0
            fam[i] = (0 if secs <= 300 else 1 if secs <= 3600
                      else 2 if secs <= 6 * 3600 else 3)
        elif i in fine:
            fam[i] = 7
        elif dv[i, 2] != 0.0:                # normal component => out-of-plane
            fam[i] = 6                       # (includes the combined rows)
        elif dv[i, 1] != 0.0:
            fam[i] = 5
        else:
            fam[i] = 4
    if (fam < 0).any():
        raise SystemExit('unmapped action rows: ' + str(np.flatnonzero(fam < 0)))
    return fam, dv, tau, mag


# ── rollout (reuses the 3-D figure's harness verbatim) ──────────────────────
def cell_result(cell, ckpt, episodes, seed, scratch):
    """Cached per-cell rollout summary. `make_3d_figs.rollout` is reused
    UNCHANGED so the action stream stays md5-identical to t11_eval.py's."""
    cache = os.path.join(scratch, f'behaviour_{cell}_{episodes}_{seed}.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    import hashlib
    import make_3d_figs as M
    eps, _ = M.rollout(ckpt, cell, episodes, seed, scratch)
    res = dict(
        cell=cell, episodes=episodes, seed=seed,
        md5=hashlib.md5(np.concatenate([e['acts'] for e in eps])
                        .astype(np.int32).tobytes()).hexdigest(),
        eps=[dict(acts=[int(x) for x in e['acts']], cause=int(e['cause']),
                  steps=int(e['steps']), dv=float(e['dv_used']),
                  budget=float(e['budget'])) for e in eps])
    json.dump(res, open(cache, 'w'))
    return res


def _worker(args):
    cell, ckpt, episodes, seed, scratch = args
    sys.path.insert(0, _HERE)
    return cell_result(cell, ckpt, episodes, seed, scratch)


# ── summarise ───────────────────────────────────────────────────────────────
def summarise(res, fam, n_actions):
    """BOTH panels are computed over SUCCESSFUL episodes.

    Not a detail. Pooling panel (a) over all valid episodes while panel (b)
    medians over successes would put two different populations side by side,
    and on W1_driftwait the mismatch dominates: t11_cells measures successful
    W1 episodes at ~46 decisions against 359-622 for failures, so at ~74%
    success the all-valid bar is mostly FAILURE behaviour sitting next to a
    table describing successes. The slide is about the skill, so successes it
    is — and the all-valid shares are printed alongside so nothing is hidden.
    """
    eps = [e for e in res['eps'] if e['cause'] != 7]       # drop `gave_up`
    win = [e for e in eps if e['cause'] == 1]

    def _share(sel):
        acts = (np.concatenate([e['acts'] for e in sel]) if sel
                else np.zeros(0, dtype=int))
        rows = np.bincount(acts, minlength=n_actions)
        sh = np.zeros(len(FNAMES))
        for i, c in enumerate(rows):
            sh[fam[i]] += c
        return sh / max(sh.sum(), 1), rows

    share, rows = _share(win)
    share_valid, rows_valid = _share(eps)

    burn_rows = np.flatnonzero(fam >= 4)
    burns = [int(np.isin(e['acts'], burn_rows).sum()) for e in win]

    def med(v):
        return float(np.median(v)) if len(v) else float('nan')

    return dict(
        cell=res['cell'], share=share, rows=rows,
        share_valid=share_valid, rows_valid=rows_valid,
        n_valid=len(eps), n_total=len(res['eps']),
        gave_up=len(res['eps']) - len(eps),
        success=sum(e['cause'] == 1 for e in eps),
        dec=med([len(e['acts']) for e in win]),
        burns=med(burns), dv=med([e['dv'] for e in win]),
        hours=med([e['steps'] / 60.0 for e in win]),
        md5=res['md5'])


# ── figure ──────────────────────────────────────────────────────────────────
def draw(S, out):
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'savefig.facecolor': SURF, 'font.size': 9,
        'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 9,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'legend.frameon': False, 'legend.fontsize': 7.2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False,
    })
    fig = plt.figure(figsize=(6.3, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.86], wspace=0.05,
                          left=0.155, right=0.985, top=0.775, bottom=0.295)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(S))[::-1]            # first cell at the top

    left = np.zeros(len(S))
    for k, (name, col) in enumerate(FAMILIES):
        w = np.array([s['share'][k] for s in S])
        ax.barh(y, w, left=left, height=0.6, color=col,
                edgecolor=SURF, linewidth=0.9)          # surface spacer
        for yi, wi, li in zip(y, w, left):
            if wi >= 0.10:                              # selective labels only
                ax.text(li + wi / 2, yi, f'{wi * 100:.0f}',
                        ha='center', va='center', fontsize=6.4,
                        color=(SURF if k in LIGHT_TEXT else INK))
        left += w

    ax.set_yticks(y)
    ax.set_yticklabels([s['cell'] for s in S], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '25', '50', '75', '100%'])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('share of decisions (%)')
    ax.set_title('(a) decision mix', loc='left', pad=13)   # pad matches (b)
    # Legend on the FIGURE, spanning both panels: at 6.3 in there is no room
    # for eight swatches inside the bars' axes without colliding with (b).
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in FAMILIES],
               loc='upper left', bbox_to_anchor=(0.012, 0.998), ncols=4,
               handlelength=1.0, handleheight=0.85, columnspacing=1.0,
               labelspacing=0.28, handletextpad=0.45, borderpad=0)

    # (b) the table, on its own axes sharing the bars' row positions
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    ax2.axis('off')
    cols = [('decisions', 'dec', '{:.0f}'), ('burns', 'burns', '{:.0f}'),
            ('Δv m/s', 'dv', '{:.0f}'), ('hours', 'hours', '{:.0f}'),
            ('success', None, None)]
    xs = np.linspace(0.09, 0.90, len(cols))
    for (hd, key, fmt), x in zip(cols, xs):
        ax2.text(x, y.max() + 0.40, hd, ha='center', va='bottom',
                 fontsize=7.2, color=INK2)
        for s, yi in zip(S, y):
            v = (f"{s['success']}/{s['n_valid']}" if key is None
                 else fmt.format(s[key]))
            ax2.text(x, yi, v, ha='center', va='center', fontsize=7.8,
                     color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_title('(b) per-episode medians', loc='left', pad=13)

    fig.text(0.155, 0.025,
             'T15e generalist, Discrete(31), bearings-only nav, seed 123, '
             '20 episodes/cell.  Both panels over SUCCESSFUL episodes.\n'
             'One decision = one forward pass, advancing the sim by '
             'τ = 1–1440 sub-steps of 60 s — decisions are not hours.',
             fontsize=5.9, color=MUTED, ha='left', va='bottom', linespacing=1.6)
    # No bbox_inches='tight': the 6.3 x 3.2 in canvas is the spec, and a tight
    # box would silently resize it to whatever the artists happen to span.
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',
                   default=os.path.join(REPO, 'models/t3/t15e_dagger_final.pt'))
    p.add_argument('--cells', default=','.join(CELLS))
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--scratch', default='/Users/pete/.claude/jobs/fd4ad701/tmp')
    p.add_argument('--figs', default=os.path.join(os.path.dirname(_HERE), 'figs'))
    p.add_argument('--jobs', type=int, default=5)
    a = p.parse_args()
    cells = a.cells.split(',')
    os.makedirs(a.figs, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)

    fam, dv, tau, mag = family_map()
    print('action table: parsed from orbital.h and cross-checked against '
          'nav_math.ACTION_TAU / ACTION_DV_MAG — OK')
    for k, name in enumerate(FNAMES):
        r = np.flatnonzero(fam == k)
        print(f'  {name:20s} rows {list(map(int, r))}')

    todo = [(c, a.ckpt, a.episodes, a.seed, a.scratch) for c in cells]
    if a.jobs > 1 and len(todo) > 1:
        # One process per cell: W1_driftwait alone is ~2.5 min at 20 episodes
        # (cap 22000 sub-steps), so sequential would blow the time budget.
        # Each worker builds its own env + policy and is single-threaded.
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(min(a.jobs, len(todo))) as pool:
            out = pool.map(_worker, todo)
    else:
        out = [_worker(t) for t in todo]

    S = [summarise(r, fam, len(tau)) for r in out]

    for key, lab in (('share', 'SUCCESSFUL episodes — this is panel (a)'),
                     ('share_valid', 'all valid episodes, for comparison')):
        print(f'\n── decision shares (%), {lab} ──')
        print(f"{'cell':14s} " + ' '.join(f'{n[:14]:>14s}' for n in FNAMES))
        for s in S:
            print(f"{s['cell']:14s} "
                  + ' '.join(f'{v * 100:14.1f}' for v in s[key]))
    print('\n── medians over successful episodes ──')
    print(f"{'cell':14s} {'dec':>6s} {'burns':>6s} {'dv m/s':>8s} "
          f"{'hours':>7s} {'success':>9s}  action md5")
    for s in S:
        print(f"{s['cell']:14s} {s['dec']:6.0f} {s['burns']:6.0f} "
              f"{s['dv']:8.1f} {s['hours']:7.1f} "
              f"{s['success']:4d}/{s['n_valid']:<4d} {s['md5'][:12]}")
    print('\n── raw per-row decision counts, successful episodes (any other '
          'family split is recomputable from these) ──')
    for s in S:
        nz = {int(i): int(c) for i, c in enumerate(s['rows']) if c}
        print(f"  {s['cell']:14s} {nz}")
    gu = [s for s in S if s['gave_up']]
    if gu:
        print('\n  gave_up (cause 7) episodes excluded from n_valid, as '
              't11_eval.py does: '
              + ', '.join(f"{s['cell']}={s['gave_up']}" for s in gu))

    out_pdf = os.path.join(a.figs, 'generalist_behaviour.pdf')
    draw(S, out_pdf)
    print(f'\nwrote {out_pdf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
