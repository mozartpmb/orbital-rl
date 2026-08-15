#!/usr/bin/env python3
"""The generalization matrix: every flagship checkpoint x every regime cell.

PURPOSE. This is the design input for a generalist-mixture rung. Six lineages
were each trained to solve one regime, and each is published only on its own
diagonal cell. What nobody has measured is the OFF-diagonal: what a tight-box
policy does in an eccentric world, what a J2 policy does without J2, what the
drift-and-wait policy does when there is nothing to wait for. The rehearsal mix
for a generalist has to be chosen from those numbers, because the pairs that
conflict are the pairs that need co-training, and the pairs that transfer for
free are the pairs that would waste rehearsal budget.

PROTOCOL. `eval_relnav3d.rollout` verbatim — num_envs=1, greedy argmax, LSTM
zeroed per episode, gave-up inits excluded from the denominator, held-out seed
123. Reusing it rather than batching means the diagonal cells are directly
comparable to the published numbers for each lineage, which is the only check
available that the harness itself is not the story.

ACTION HEADS ARE NEVER RESIZED. `j2wait_W1_driftwait` carries a 31-row head
(row 30 = the 1440-substep day-warp); the other five carry 30. A 31-head
checkpoint cannot load into a 30-head model and vice versa, so the env's action
space is set from the CHECKPOINT, not the cell:

  - 30-head checkpoints in the wide-plane-gap cell run Discrete-30. The
    day-warp does not exist for them. That is reported, not engineered around:
    forcing a row they never had would measure a policy that was never trained.
  - The 31-head checkpoint runs Discrete-31 everywhere. Masking row 30 outside
    its home cell was considered and rejected — masking measures a crippled
    policy, and the question here is what each checkpoint actually does when
    dropped into a foreign regime.

TRUTH VS NAV. Two of the six lineages (extj2_A3b, j2wait_W1) are truth-state
control policies with no navigation training, and two of the seven cells
(J2-tight, W1-plane) are truth-only cells. Every (row, cell) pair is labelled
with the mode it actually ran in; a nav checkpoint in a truth cell is measuring
control skill with the estimator removed, which is a real and different
quantity from its headline number.

OOD CELLS ARE RUN ANYWAY. A narrow-normalizer encoder in a wide-normalizer cell
is out of distribution by construction, and so is a two-body policy under J2.
Those cells are run and labelled with the reason rather than left blank: an
honest low number with a mechanism attached is a design input, and a blank is
not.

    python3 scripts/orbital/nav/gen_matrix.py --ckpt X --cell Y   # one pair
    python3 scripts/orbital/nav/gen_matrix.py --report            # both tables
"""

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

import eval_relnav3d as ER3                                        # noqa: E402

OUT = os.path.join(ROOT, 'web_data', 'results', 'gen_matrix')
M = os.path.join(ROOT, 'models', 't3')
R = math.radians

# ── rows ────────────────────────────────────────────────────────────────────
# `norm` = observation normalizer family the encoder was trained under:
#   narrow = obs_alt_scale_m 1.6e6 / lvlh_scale_m 6.371e6  (X3, J2, j2wait)
#   wide   = obs_alt_scale_m 8.0e6 / lvlh_scale_m 1.5e7    (E-ladder)
# `j2` = trained with J2 in the dynamics. `head` = action rows.
CKPTS = {
    'T-BO3': dict(
        f='n3dnav_T-BO3.pt', head=30, norm='narrow', j2=False, nav=True,
        home='X3-loose', note='rung-1 3D bearings-only nav'),
    'T-BO3D-TB5': dict(
        f='n3dnav_T-BO3D-TB5.pt', head=30, norm='narrow', j2=False, nav=True,
        home='TB5-3D', note='tight-box nav, 5 km / 1 m/s'),
    'e-E3': dict(
        f='n3dnav_e_E3.pt', head=30, norm='wide', j2=False, nav=True,
        home='E3', note='eccentric nav, e_max 0.30'),
    'J2BO-nav': dict(
        f='j2nav_T-J2BO-nav_final.pt', head=30, norm='narrow', j2=True,
        nav=True, home='J2X-loose', note='J2 x angles-only nav'),
    'A3b-j2': dict(
        f='extj2_A3b_j2_box5k1.pt', head=30, norm='narrow', j2=True,
        nav=False, home='J2-tight', note='J2 tight-box, truth state'),
    'W1-driftwait': dict(
        f='j2wait_W1_driftwait.pt', head=31, norm='narrow', j2=True,
        nav=False, home='W1-plane', note='drift-and-wait, day-warp row 30'),
}

# ── columns ─────────────────────────────────────────────────────────────────
# Every cell is composed from the rung config that VALIDATED it, so no cell is
# a config this project has never scored a checkpoint under.
_J2_BAND = dict(i_target_min_rad=R(30.0), i_target_max_rad=R(60.0),
                raan_target_sample=0, lvlh_frame_mode=1)
CELLS = {
    'X3-loose':  dict(base='X3', nav=True, over={},
                      note='30 km / 50 m/s, two-body, di<=1 deg'),
    'TB5-3D':    dict(base='X3', nav=True,
                      over=dict(rendezvous_radius_m=5000.0, rel_vel_tol_ms=1.0),
                      note='5 km / 1 m/s, two-body'),
    'E1':        dict(base='E1', nav=True, over={},
                      note='e_max 0.10, WIDE normalizers'),
    'E3':        dict(base='E3', nav=True, over={},
                      note='e_max 0.30, WIDE normalizers'),
    'J2X-loose': dict(base='J2X', nav=True, over={},
                      note='J2 on, i_t U(30,60) deg, 30 km / 50 m/s'),
    'J2-tight':  dict(base='J2X', nav=False,
                      over=dict(rendezvous_radius_m=5000.0, rel_vel_tol_ms=1.0),
                      note='J2 on, 5 km / 1 m/s — TRUTH-ONLY cell'),
    'W1-plane':  dict(base='X3', nav=False,
                      over=dict(_J2_BAND, j2_mode=1, episode_cap_steps=22000,
                                di_min_rad=R(2.0), di_max_rad=R(5.0),
                                di_phase_mode=1),
                      note='node-dominant plane gap U(2,5) deg, 22000 cap '
                           '— TRUTH-ONLY cell; day-warp only for 31-head'),
}


def cell_kwargs(cell, head):
    """Cell config, with the action space taken from the CHECKPOINT's head."""
    spec = CELLS[cell]
    kw = dict(ER3.RUNGS[spec['base']])
    kw.update(spec['over'])
    kw['legacy_action_space'] = head
    kw.pop('nav_j2_mode', None) if not spec['nav'] else None
    return kw


def ood_reasons(ck, cell):
    """Why this pair is out of distribution, as a list of short reasons."""
    c, s = CKPTS[ck], CELLS[cell]
    kw = cell_kwargs(cell, c['head'])
    out = []
    wide = float(kw.get('obs_alt_scale_m', 1.6e6)) > 4e6
    if wide and c['norm'] == 'narrow':
        out.append('narrow-trained encoder in WIDE normalizers')
    if not wide and c['norm'] == 'wide':
        out.append('wide-trained encoder in NARROW normalizers')
    if kw.get('j2_mode', 0) and not c['j2']:
        out.append('two-body policy under J2')
    if not kw.get('j2_mode', 0) and c['j2']:
        out.append('J2 policy without J2')
    if float(kw.get('di_max_rad', 0.0)) > R(1.5):
        out.append('plane gap %.0f-%.0f deg vs <=1 deg trained'
                   % (math.degrees(kw.get('di_min_rad', 0.0)),
                      math.degrees(kw['di_max_rad'])))
        if c['head'] < 31:
            out.append('no day-warp (30-head)')
    if float(kw.get('rendezvous_radius_m', 3e4)) <= 5000.0 \
            and c['home'] not in ('TB5-3D', 'J2-tight'):
        out.append('tight box vs loose-box training')
    return out


def run_one(ck, cell, episodes, seed, force_truth=False):
    """One (checkpoint, cell) pair.

    `force_truth` runs the TRUTH-MODE CONTROL. It exists because the row means
    of the main matrix are not mode-matched: the two truth lineages (A3b-j2,
    W1-driftwait) run truth in all seven cells, while the four nav lineages
    carry real estimation error in five of them. Without this control, "which
    checkpoint is the best generalist" would be partly answering "which
    checkpoint was allowed to see the target".
    """
    c, s = CKPTS[ck], CELLS[cell]
    # A truth-only cell forces truth for every row; a truth-lineage checkpoint
    # runs truth in every cell, because it has no navigation training at all.
    mode = 'bearings_only' if (s['nav'] and c['nav']) else 'truth'
    if force_truth:
        mode = 'truth'
    acq = 'real' if mode == 'bearings_only' else 'surrogate'
    kw = cell_kwargs(cell, c['head'])

    base = s['base']
    ER3.RUNGS['_GM'] = kw                      # register so make_env composes it
    env = ER3.make_env('_GM', mode, acq=acq)
    if acq == 'real':
        ER3._install_real_acq(env)
    res = ER3.rollout(env, os.path.join(M, c['f']), episodes, seed,
                      label=f'{ck}/{cell}', verbose=True)
    # NOT env.close(). Orbital.close() -> binding.vec_close segfaults after a
    # multi-episode rollout (verified in truth, bearings_only/surrogate and
    # bearings_only/real alike; safe on a fresh env, after reset, and after a
    # single step). No eval stage in eval_relnav3d calls close(), which is why
    # nothing has hit it before. Each cell here is its own process, so the OS
    # reclaims; flagged rather than worked around silently.

    res.update(ckpt=ck, cell=cell, mode=mode, acq=acq, head=c['head'],
               base_rung=base, home=(c['home'] == cell),
               ood=ood_reasons(ck, cell), episodes=episodes, seed=seed,
               daywarp=(c['head'] == 31))
    res['truth_control'] = bool(force_truth)
    os.makedirs(OUT, exist_ok=True)
    sfx = '__T' if force_truth else ''
    with open(os.path.join(OUT, f'{ck}__{cell}{sfx}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'{ck:14} {cell:10} {mode:14} {res["success"]}/{res["n_valid"]} '
          f'= {100 * res["rate"]:.1f}%  gave_up {res["gave_up"]}  '
          f'{res["wall_s"]:.0f}s', flush=True)
    return res


def load_all(control=False):
    out = {}
    if not os.path.isdir(OUT):
        return out
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith('.json'):
            continue
        r = json.load(open(os.path.join(OUT, fn)))
        if bool(r.get('truth_control')) != control:
            continue
        out[(r['ckpt'], r['cell'])] = r
    return out


def report():
    res = load_all()
    print('\n== SUCCESS MATRIX (100 eps, held-out seed 123) ==')
    hdr = f'{"checkpoint":14}' + ''.join(f'{c:>12}' for c in CELLS)
    print(hdr)
    print('-' * len(hdr))
    for ck in CKPTS:
        line = f'{ck:14}'
        for cell in CELLS:
            r = res.get((ck, cell))
            if r is None:
                line += f'{"--":>12}'
                continue
            tag = '*' if r['home'] else ('!' if r['ood'] else ' ')
            line += f'{100 * r["rate"]:10.1f}{tag}'
        print(line)
    print('  * = home cell (diagonal)   ! = OOD (reason in JSON)')

    print('\n== MODE / OOD KEY ==')
    for ck in CKPTS:
        for cell in CELLS:
            r = res.get((ck, cell))
            if r and r['ood']:
                print(f'  {ck:14} {cell:10} {r["mode"]:14} '
                      f'{"; ".join(r["ood"])}')

    print('\n== FAILURE SIGNATURES: worst off-diagonal cells ==')
    off = [r for r in res.values() if not r['home']]
    off.sort(key=lambda r: (r['rate'], -r['n_valid']))
    print(f'{"checkpoint":14}{"cell":11}{"mode":14}{"rate":>7}  '
          f'{"len":>6} {"dv_med":>7}  causes')
    for r in off[:12]:
        cz = ', '.join(f'{k} {v}' for k, v in sorted(
            r['causes'].items(), key=lambda kv: -kv[1]))
        print(f'{r["ckpt"]:14}{r["cell"]:11}{r["mode"]:14}'
              f'{100 * r["rate"]:6.1f}%  {r["mean_len"]:6.0f} '
              f'{r["dv_med_ms"]:7.0f}  {cz}')

    ctl = load_all(control=True)
    print('\n== ROW MEANS (generalist-adjacency) ==')
    print(f'{"checkpoint":14}{"all":>7}{"off-diag":>10}   '
          f'{"all(truth)":>11}{"off(truth)":>11}   mode-matched?')
    for ck in CKPTS:
        rs = [res[(ck, c)]['rate'] for c in CELLS if (ck, c) in res]
        off = [res[(ck, c)]['rate'] for c in CELLS
               if (ck, c) in res and not res[(ck, c)]['home']]
        # truth-matched view: the control where one exists, else the main cell
        # (which already ran truth for the truth-only cells and truth rows)
        tr, tro = [], []
        for c in CELLS:
            r = ctl.get((ck, c)) or res.get((ck, c))
            if r is None or r['mode'] != 'truth':
                continue
            tr.append(r['rate'])
            if not r['home']:
                tro.append(r['rate'])
        if rs:
            print(f'  {ck:14}{100 * sum(rs) / len(rs):6.1f}%'
                  f'{100 * sum(off) / max(len(off), 1):9.1f}%   '
                  f'{100 * sum(tr) / max(len(tr), 1):10.1f}%'
                  f'{100 * sum(tro) / max(len(tro), 1):10.1f}%   '
                  f'n={len(tr)}/7')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', choices=sorted(CKPTS))
    ap.add_argument('--cell', choices=sorted(CELLS))
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--truth-control', action='store_true',
                    help='force truth mode: the mode-matched control arm')
    a = ap.parse_args()
    if a.report:
        report()
        return 0
    if not (a.ckpt and a.cell):
        ap.error('need --ckpt and --cell, or --report')
    run_one(a.ckpt, a.cell, a.episodes, a.seed, force_truth=a.truth_control)
    return 0


if __name__ == '__main__':
    sys.exit(main())
