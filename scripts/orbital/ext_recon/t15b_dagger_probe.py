#!/usr/bin/env python3
"""T15b: is a DAgger refresh actually well-posed here?

The pre-registered escalation says "rebuild the W1 anchor from the student's
visited distribution, labelled by the teacher". That is only worth doing if the
TEACHER'S LABELS ARE MEANINGFUL THERE. If the student wanders somewhere the
teacher has also never been, the teacher's output is not supervision — it is
noise with a confident-looking shape, and training on it would be the
bug-#13 pattern in a new costume.

So this measures, before any dataset is committed:

  1. decisions/episode at the NEW root (W1 competence changed; the step-share
     arithmetic that set the T15 weights was measured at 0.0% W1)
  2. teacher ENTROPY on student-visited states vs on its own states — a lost
     teacher goes flat toward ln(31)=3.43
  3. teacher MAX-PROB (how committed it is)
  4. CE(teacher || student) — how much correction is on offer
  all split by EPISODE OUTCOME, because red-team (i) proposes anchoring only on
  FAILED-episode states, and that only makes sense if the teacher is reliable
  precisely there.
"""
import argparse
import os
import sys

import numpy as np
import torch

WT = '/Users/pete/space_training'
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))

from pufferlib.models import Default, LSTMWrapper                # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav   # noqa: E402
import t11_cells as T                                            # noqa: E402

LN31 = float(np.log(31))


def cell_env(cell, n, seed):
    c = T.ALL_CELLS[cell]
    kw = T.nav_env_kwargs(
        num_envs=n, nav_mode='bearings_only', cell_mixture_mode=0,
        j2_mode=int(c['j2']), nav_j2_mode=int(c['j2']),
        episode_cap_steps=int(c['cap']), rendezvous_radius_m=c['box_r'],
        rel_vel_tol_ms=c['box_v'], a_min_override=c['a_min'],
        a_max_override=c['a_max'], e_max_target=c['e_max_target'],
        e_max_sat=c['e_max_sat'], de_max=c['de_max'], da_max_m=c['da_max'],
        di_max_rad=c['di_max'], di_min_rad=c['di_min'],
        di_phase_mode=int(c['di_phase']),
        i_target_min_rad=c['i_t_min'], i_target_max_rad=c['i_t_max'],
        fuel_frac_min=c['fuel_min'], fuel_frac_max=c['fuel_max'])
    kw['nav_max_ticks'] = 0
    kw['nav_filter_impl'] = 'c'
    e = OrbitalNav(**kw)
    e.reset(seed=seed)
    return e


def load(env, path):
    p = LSTMWrapper(env, Default(env))
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval()
    return p


def collect_student_states(cell, student_ck, n=32, horizon=64,
                           want_windows=600, seed=17):
    """Roll the STUDENT; return windows tagged by their episode's outcome."""
    e = cell_env(cell, n, seed)
    pol = load(e, student_ck)
    obs, _ = e.reset(seed=seed)
    h = torch.zeros(n, pol.hidden_size)
    c = torch.zeros(n, pol.hidden_size)
    dim = e.single_observation_space.shape[0]
    buf = np.zeros((n, horizon, dim), np.float32)
    fill = np.zeros(n, int)
    pend = [[] for _ in range(n)]        # windows of the in-flight episode
    out_w, out_lab = [], []
    dec = np.zeros(n, int)
    ep_len, ep_ok = [], []
    guard = 0
    while len(out_w) < want_windows and guard < 40000:
        guard += 1
        with torch.no_grad():
            lg, _ = pol.forward_eval(
                torch.from_numpy(np.asarray(obs)).float(), {'lstm_h': h, 'lstm_c': c})
        a = torch.argmax(lg, -1).numpy().astype(np.int32)
        for i in range(n):
            buf[i, fill[i]] = np.asarray(obs)[i]
            fill[i] += 1
            if fill[i] == horizon:
                pend[i].append(buf[i].copy())
                fill[i] = 0
        obs, _, term, _, _ = e.step(a)
        dec += 1
        for i in np.flatnonzero(term):
            steps, cause = e.last_episode_result(i)
            ok = int(cause) == 1                     # 1 == success
            ep_len.append(dec[i]); ep_ok.append(ok)
            for w in pend[i]:
                out_w.append(w); out_lab.append(ok)
            pend[i] = []; dec[i] = 0; fill[i] = 0
            h[i] = 0; c[i] = 0
    e.close()
    return (np.asarray(out_w[:want_windows]),
            np.asarray(out_lab[:want_windows], bool),
            np.asarray(ep_len), np.asarray(ep_ok, bool))


def stats(teacher, student, obs, tag):
    """Teacher entropy / max-prob / CE(teacher||student) on a state block."""
    with torch.no_grad():
        lt, _ = teacher(torch.as_tensor(obs).float(),
                        dict(action=None, lstm_h=None, lstm_c=None))
        ls, _ = student(torch.as_tensor(obs).float(),
                        dict(action=None, lstm_h=None, lstm_c=None))
    if isinstance(lt, (list, tuple)):
        lt, ls = lt[0], ls[0]
    tlp = torch.log_softmax(lt.float(), -1)
    slp = torch.log_softmax(ls.float(), -1)
    tp = tlp.exp()
    ent = (-(tp * tlp).sum(-1)).flatten().numpy()
    mx = tp.max(-1).values.flatten().numpy()
    ce = ((tp * (tlp - slp)).sum(-1)).flatten().numpy()
    print(f'  {tag:34s} n={ent.size:7d}  teacher H {np.median(ent):5.3f} '
          f'(p90 {np.percentile(ent,90):5.3f}, ln31={LN31:.3f})  '
          f'maxp {np.median(mx):5.3f}  CE(t||s) {np.median(ce):7.3f}')
    return ent, mx, ce


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', default=f'{WT}/models/t3/t15_remix_final.pt')
    ap.add_argument('--teacher', default=f'{WT}/models/t3/w1nav_child.pt')
    ap.add_argument('--cell', default='W1_driftwait')
    ap.add_argument('--windows', type=int, default=600)
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--teacher-set',
                    default=f'{WT}/models/t15/anchor_w1_k0.pt')
    a = ap.parse_args()

    print(f'== rolling the STUDENT ({os.path.basename(a.student)}) on {a.cell} ==')
    W, lab, ep_len, ep_ok = collect_student_states(
        a.cell, a.student, n=a.n, want_windows=a.windows)
    print(f'  episodes {ep_ok.size}: success {ep_ok.mean():.1%}  '
          f'decisions/ep median {np.median(ep_len):.0f} '
          f'(success {np.median(ep_len[ep_ok]) if ep_ok.any() else -1:.0f} / '
          f'fail {np.median(ep_len[~ep_ok]) if (~ep_ok).any() else -1:.0f})')
    print(f'  windows {W.shape[0]}: from SUCCESS episodes {lab.mean():.1%}, '
          f'from FAILED {1-lab.mean():.1%}')

    env = cell_env(a.cell, 2, 3)
    teach = load(env, a.teacher)
    stud = load(env, a.student)
    env.close()

    print('\n== teacher label quality (is the teacher lost where the student goes?) ==')
    ref = torch.load(a.teacher_set, map_location='cpu', weights_only=False)['obs']
    e0, m0, c0 = stats(teach, stud, ref[:200].numpy(), 'TEACHER-visited (T15 set)')
    e1, m1, c1 = stats(teach, stud, W[:200], 'STUDENT-visited (all)')
    if lab.any():
        stats(teach, stud, W[lab][:200], 'STUDENT-visited (SUCCESS eps)')
    if (~lab).any():
        stats(teach, stud, W[~lab][:200], 'STUDENT-visited (FAILED eps)')

    print('\n== reading ==')
    dh = np.median(e1) - np.median(e0)
    print(f'  teacher entropy shift on student states: {dh:+.3f} nats '
          f'({np.median(e1):.3f} vs {np.median(e0):.3f}; flat would be {LN31:.3f})')
    if np.median(e1) > 0.75 * LN31:
        print('  -> TEACHER IS LOST on student states: labels are noise, DAgger '
              'refresh is NOT well-posed as specified.')
    elif dh > 0.5:
        print('  -> teacher is MEASURABLY less certain off its own manifold; '
              'refresh is usable but should be CE-weighted or failure-scoped.')
    else:
        print('  -> teacher remains confident on student states: labels are '
              'meaningful and a straight refresh is well-posed.')
