#!/usr/bin/env python3
"""NOTE-22: the (range, in-plane, plane) calibration triad for the BO stack.

n3d_REDTEAM NOTE-22, verbatim intent: "Require the (range, in-plane, plane)
triad over >=6 geometry classes x >=24 noise seeds, including normal-burn
cells... A scalar RMSE makes the new channel unmeasurable, which is the exact
failure this campaign exists to avoid."

WHAT IS BEING GATED. Not code correctness — the filter's own gates already pass.
This gates the CALIBRATION CLAIM: that the covariance the stack reports is the
covariance it actually delivers, PER CHANNEL. A filter can be globally
consistent (in-loop NEES ~1) while being optimistic on one axis and
conservative on another, because NEES averages over the ellipsoid and the two
errors cancel inside the average. That cancellation is exactly what NOTE-22
exists to break open, and it is why the 13th-bug fix restoring global NEES does
NOT by itself license a per-channel claim.

THE FRAME. Errors are resolved in the target-relative triad at each sample:
    e_range   = LOS unit vector, chaser -> target
    e_plane   = the target's orbit normal, orthogonalised against e_range
    e_inplane = e_plane x e_range   (completes a right-handed set)
Per component c: z = (x_hat - x_true).e_c / sqrt(e_c^T P e_c), for the position
block and the velocity block separately. The triad is built from TRUTH
geometry, never from the estimate, so a biased estimate cannot rotate the frame
into agreement with itself.

VERDICTS. z is N(0,1) iff the covariance is honest.
    std(z) ~ 1          calibrated
    std(z) < 1          conservative by 1/std      (sigma too large)
    std(z) > 1          OPTIMISTIC by std          (sigma too small)
Optimistic is the load-bearing direction: it is the estimator claiming
precision it does not have, and the acquisition gate, the handoff and every
success box downstream believe it. Coverage at 1 sigma (68.3% nominal) and 2
sigma (95.4%) is reported alongside, because a heavy tail hides comfortably
inside a respectable std and the tail is what a gate actually trips on.

WHY POLICIES, NOT SCRIPTED ACTIONS. Each class is flown by the checkpoint that
ships for it. Calibration is a property of the filter EVALUATED ON THE
GEOMETRY IT ACTUALLY SEES, and a converged policy visits a narrow, particular
tube of geometries — closing range, shrinking relative velocity — that random
actions never enter. Scripting the actions would have measured a filter nobody
flies. It would also have collapsed the TB5-3D class onto X3-loose, since those
two rungs differ ONLY in the success box: without a policy driving to the tight
box, the box never binds and the two cells are the same experiment run twice.

NORMAL-BURN CELLS ARE NOT OPTIONAL. 3d_E section 5's coverage ablation showed
that dropping normal-burn policies made two seeded bug classes undetectable.
The NORMBURN class therefore forces the plane axis directly with a scripted
plane-heavy action mix — the one place a scripted mix is the right instrument,
because the point is to excite a channel the trained policies deliberately
avoid.

    python3 scripts/orbital/nav/calib_triad.py --seeds 24 --envs 16
    python3 scripts/orbital/nav/calib_triad.py --report
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_T0 = time.time()

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

import eval_relnav3d as ER3                                    # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav  # noqa: E402
from pufferlib.models import Default, LSTMWrapper               # noqa: E402

CSV_OUT = os.path.join(ROOT, 'web_data', 'results', 'calib_triad.csv')
MD_OUT = os.path.join(ROOT, 'CALIB_TRIAD.md')
M = os.path.join(ROOT, 'models', 't3')
COMPONENTS = ('range', 'inplane', 'plane')

# Plane-heavy discrete-30 mix for the NOTE-22 normal-burn cell.
MIX_NORMAL = np.array([20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25,
                       26, 27, 28, 29, 0, 9, 11])

CLASSES = {
    'X3-loose': dict(
        rung='X3', ckpt=os.path.join(M, 'n3dnav_T-BO3.pt'),
        note='30 km / 50 m/s box, two-body, near-equatorial — rung-1 flagship'),
    'TB5-3D': dict(
        rung='X3', ckpt=os.path.join(M, 'n3dnav_T-BO3D-TB5.pt'), box='TB5-3D',
        note='5 km / 1 m/s box — the cell where sigma_vel binds'),
    'E1-ecc': dict(
        rung='E1', ckpt=os.path.join(M, 'n3dnav_e_E1.pt'),
        note='e_max 0.10, wide normalizers'),
    'E3-ecc': dict(
        rung='E3', ckpt=os.path.join(M, 'n3dnav_e_E3.pt'),
        note='e_max 0.30, wide normalizers — the eccentric edge we ship'),
    'J2X-incl': dict(
        rung='J2X', ckpt=os.path.join(M, 'j2nav_T-J2BO-nav.pt'),
        note='J2 + i_t U(30,60) deg, nav_j2_mode=1, post-13th-bug harness'),
    'NORMBURN': dict(
        rung='X3', ckpt=None, mix=MIX_NORMAL,
        note='scripted plane-heavy burns — NOTE-22 mandated coverage cell'),
}


def triad(sat, tgt):
    """(e_range, e_inplane, e_plane) per row, from TRUTH geometry."""
    d = tgt[:, :3] - sat[:, :3]
    e_r = d / np.maximum(np.linalg.norm(d, axis=1), 1.0)[:, None]
    h = np.cross(tgt[:, :3], tgt[:, 3:6])
    e_p = h / np.maximum(np.linalg.norm(h, axis=1), 1e-9)[:, None]
    e_p = e_p - (np.einsum('ni,ni->n', e_p, e_r))[:, None] * e_r
    nrm = np.linalg.norm(e_p, axis=1)
    ok = nrm > 1e-6                       # LOS parallel to the normal: no basis
    e_p = np.where(ok[:, None], e_p / np.maximum(nrm, 1e-12)[:, None], 0.0)
    return e_r, np.cross(e_p, e_r), e_p, ok


def z_of(err, P, e):
    """Whitened error along a unit direction: e.err / sqrt(e^T P e)."""
    num = np.einsum('ni,ni->n', err, e)
    var = np.einsum('ni,nij,nj->n', e, P, e)
    return num / np.sqrt(np.maximum(var, 1e-300))


def run_class(name, spec, seeds, envs, steps, sample_every, scen_seed,
              out=None):
    """Yields nothing; writes one CSV batch PER SEED.

    Per-seed rather than per-class because a calibration sweep runs against a
    wall clock on a shared machine: a class that is killed at 80% should leave
    80% of its evidence on disk, not none of it.
    """
    kw = dict(ER3.RUNGS[spec['rung']])
    if spec.get('box'):
        kw.update(ER3.box_kw(spec['box']))
    total, rng_act = 0, np.random.default_rng(1234)
    torch.set_num_threads(1)

    for si, nav_seed in enumerate(seeds):
        rows = []
        env = OrbitalNav(num_envs=envs, nav_mode='bearings_only',
                         nav_seed=int(nav_seed), log_interval=10 ** 9, **kw)
        policy = None
        if spec.get('ckpt'):
            policy = LSTMWrapper(env, Default(env))
            policy.load_state_dict(torch.load(spec['ckpt'], map_location='cpu',
                                              weights_only=True))
            policy.eval()
        obs, _ = env.reset(seed=scen_seed + si)
        hid = policy.hidden_size if policy else 1
        st = {'lstm_h': torch.zeros(envs, hid),
              'lstm_c': torch.zeros(envs, hid)}
        idx = np.arange(envs)

        for s in range(steps):
            if policy is not None:
                with torch.no_grad():
                    logits, _ = policy.forward_eval(
                        torch.from_numpy(np.asarray(obs)).float(), st)
                    act = torch.argmax(logits, dim=-1).numpy().astype(np.int32)
            else:
                act = rng_act.choice(spec['mix'], envs).astype(np.int32)
            obs, _, term, trunc, _ = env.step(act)
            done = np.asarray(term, bool) | np.asarray(trunc, bool)
            if policy is not None and done.any():
                # the env auto-resets the row, so its recurrent state must go
                # with it or the policy carries a dead episode's memory forward
                d = torch.from_numpy(done)
                st['lstm_h'][d] = 0.0
                st['lstm_c'][d] = 0.0

            if s < sample_every or s % sample_every:
                continue
            acq = env._acq.acquired
            if not acq.any():
                continue
            x, P = env._filt.mean_cov(idx)
            tgt, sat = env._prev_tgt, env._prev_sat
            e_r, e_ip, e_p, okt = triad(sat, tgt)
            good = (acq & okt & ~done & np.isfinite(x).all(axis=1)
                    & np.isfinite(P).all(axis=(1, 2)))
            if not good.any():
                continue
            err_p, err_v = x[:, :3] - tgt[:, :3], x[:, 3:6] - tgt[:, 3:6]
            hot = np.flatnonzero(good)
            rho = np.linalg.norm(tgt[:, :3] - sat[:, :3], axis=1)
            for cn, e in (('range', e_r), ('inplane', e_ip), ('plane', e_p)):
                zp, zv = z_of(err_p, P[:, :3, :3], e), z_of(err_v, P[:, 3:, 3:], e)
                for k in hot:
                    if np.isfinite(zp[k]):
                        rows.append((name, int(nav_seed), cn, 'pos',
                                     float(zp[k]), float(rho[k])))
                    if np.isfinite(zv[k]):
                        rows.append((name, int(nav_seed), cn, 'vel',
                                     float(zv[k]), float(rho[k])))
        env.close()
        if rows:
            write_csv(rows, out)
        total += len(rows)
        print(f'  {name:10} seed {si + 1:2d}/{len(seeds)}  +{len(rows):5d} z  '
              f'({time.time() - _T0:.0f}s)', flush=True)
    return total


def verdict(p68, cov2, p_gt3):
    """Verdict from the ROBUST scale, with the tail called out separately.

    std(z) is the obvious statistic and the wrong one here. The first sweep
    returned std = 1294 on a cell whose 1-sigma and 2-sigma coverage were 65%
    and 92% — i.e. essentially nominal. A std that large with nominal coverage
    is not a mis-scaled covariance, it is a well-scaled covariance with a few
    catastrophic samples, and calling it "optimistic 1294x" would have
    misdescribed the defect and pointed any fix in the wrong direction.

    So the scale verdict is taken from p68(|z|), which is 1.0 for N(0,1) and is
    immune to the tail, and the tail is reported as its own fact: P(|z|>3),
    nominal 0.27%. A cell can be CALIBRATED in the core and still carry a tail
    that trips a 3-sigma gate two orders of magnitude too often, and an
    operator needs both numbers to know which risk they are holding.
    """
    tag = 'calibrated' if 0.80 <= p68 <= 1.25 else (
        f'conservative {1.0 / p68:.1f}x' if p68 < 0.80
        else f'OPTIMISTIC {p68:.1f}x')
    if p_gt3 > 0.027:                     # 10x the nominal 0.27%
        tag += f' +HEAVY TAIL ({100 * p_gt3:.1f}% >3sig)'
    elif p_gt3 > 0.0081:                  # 3x nominal
        tag += f' +tail ({100 * p_gt3:.1f}% >3sig)'
    return tag


def summarise(rows):
    out = []
    for cls in dict.fromkeys(r[0] for r in rows):
        nseed = len({r[1] for r in rows if r[0] == cls})
        for block in ('pos', 'vel'):
            for comp in COMPONENTS:
                z = np.array([r[4] for r in rows if r[0] == cls
                              and r[2] == comp and r[3] == block])
                if z.size < 20:
                    continue
                az = np.abs(z)
                p68 = float(np.percentile(az, 68.27))
                cov1 = float(np.mean(az <= 1.0))
                cov2 = float(np.mean(az <= 2.0))
                pg3 = float(np.mean(az > 3.0))
                out.append(dict(cls=cls, block=block, comp=comp, n=int(z.size),
                                seeds=nseed, med=float(np.median(z)), p68=p68,
                                cov1=cov1, cov2=cov2, pg3=pg3,
                                p99=float(np.percentile(az, 99)),
                                mx=float(az.max()), std=float(z.std()),
                                verdict=verdict(p68, cov2, pg3)))
    return out


def load_rows(path=None):
    """Read one CSV, or every shard beside it if given the canonical path.

    Classes are run as parallel processes (J2X costs 4.45x the two-body filter
    and would otherwise set the wall clock alone), each writing its own shard,
    so that concurrent appends cannot interleave a half-written row.
    """
    path = path or CSV_OUT
    shards = [path]
    if path == CSV_OUT:
        d, base = os.path.dirname(path), os.path.basename(path)[:-4]
        shards = sorted(os.path.join(d, f) for f in os.listdir(d)
                        if f.startswith(base) and f.endswith('.csv'))
    rows = []
    for s in shards:
        rows += [(r['cls'], int(r['nav_seed']), r['component'], r['block'],
                  float(r['z']), float(r['rho']))
                 for r in csv.DictReader(open(s))]
    return rows


def write_csv(rows, path=None):
    path = path or CSV_OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['cls', 'nav_seed', 'component', 'block', 'z', 'rho'])
        w.writerows(rows)


def print_table(rows):
    hdr = (f'{"class":10} {"blk":4} {"comp":9} {"n":>6} {"sd":>3} {"med":>7} '
           f'{"p68|z|":>7} {"1sig":>6} {"2sig":>6} {">3sig":>7} {"p99":>8} '
           f'{"max":>9}  verdict')
    print(hdr)
    print('-' * len(hdr))
    for r in summarise(rows):
        print(f'{r["cls"]:10} {r["block"]:4} {r["comp"]:9} {r["n"]:6d} '
              f'{r["seeds"]:3d} {r["med"]:7.3f} {r["p68"]:7.3f} '
              f'{100 * r["cov1"]:5.1f}% {100 * r["cov2"]:5.1f}% '
              f'{100 * r["pg3"]:6.2f}% {r["p99"]:8.2f} {r["mx"]:9.1f}  '
              f'{r["verdict"]}')


def print_by_range(rows, cls, block='pos'):
    """z(std) bucketed by true range. An optimistic cell is only actionable
    once you can name the operating condition it binds, and for a bearings-only
    filter range is the axis that condition almost always lies along."""
    edges = [0, 25e3, 100e3, 400e3, 1e9]
    lbl = ['<25 km', '25-100 km', '100-400 km', '>400 km']
    print(f'\n{cls} [{block}]  p68|z| (nominal 1.00) / %|z|>3 (nominal 0.27%) '
          f'by true range')
    print(f'{"bucket":12} {"n":>6} ' + ' '.join(f'{c:>16}' for c in COMPONENTS))
    for i in range(4):
        n0, cells = 0, []
        for comp in COMPONENTS:
            z = np.abs(np.array(
                [r[4] for r in rows if r[0] == cls and r[2] == comp
                 and r[3] == block and edges[i] <= r[5] < edges[i + 1]]))
            n0 = max(n0, z.size)
            cells.append(f'{np.percentile(z, 68.27):7.2f} /{100 * np.mean(z > 3):6.2f}%'
                         if z.size >= 20 else f'{"-":>16}')
        print(f'{lbl[i]:12} {n0:6d} ' + ' '.join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--envs', type=int, default=16)
    ap.add_argument('--steps', type=int, default=110)
    ap.add_argument('--sample-every', type=int, default=12)
    ap.add_argument('--scen-seed', type=int, default=4100)
    ap.add_argument('--classes', default=','.join(CLASSES))
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--out', default=None, help='shard CSV for parallel runs')
    ap.add_argument('--by-range', default=None, help='class to bucket by range')
    a = ap.parse_args()

    if a.report or a.by_range:
        rows = load_rows()
        if a.report:
            print_table(rows)
        if a.by_range:
            for b in ('pos', 'vel'):
                print_by_range(rows, a.by_range, b)
        return 0
    out = a.out or CSV_OUT
    if a.fresh and os.path.exists(out):
        os.remove(out)

    seeds = [20260815 + 7 * i for i in range(a.seeds)]
    for name in a.classes.split(','):
        if name not in CLASSES:
            continue
        t0 = time.time()
        n = run_class(name, CLASSES[name], seeds, a.envs, a.steps,
                      a.sample_every, a.scen_seed, out)
        print(f'{name:10} {n:7d} z  {a.seeds} seeds  '
              f'({time.time() - t0:.0f}s)', flush=True)
    print()
    print_table(load_rows(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
