#!/usr/bin/env python3
"""W1xnav gates — the two that would have caught this, and did not exist.

W1 was trained inside T11 at `nav_max_ticks=120` and scored 0.0/200. The
post-mortem recorded "incompetent root" and flagged the surrogate as unvalidated
beyond 6 h. It is now measured, and the surrogate was 50-67% OPTIMISTIC at that
cadence: the policy was told it had acquired the target on roughly two episodes
in three where the real batch IOD would not have converged. That is bug-#13's
class — a fictional information signal — and no shipped diagnostic saw it,
because every nav_* metric measures the FILTER and this is the SURROGATE.

W1A  SURROGATE AGREES WITH THE REAL BATCH IOD AT THE TRAINING CADENCE.
     Not "at some cadence" — at the one the campaign actually trains with,
     because the disagreement is a function of tick spacing and nothing else:

         day-warp dt    w0 obs    24 h arc optimism
             60 s          45        +0.0%      <- nav_max_ticks 0
            180 s          15       +20.8%      <- nav_max_ticks 480
            720 s           4       +66.7%      <- nav_max_ticks 120 (T11)

W1B  NEES THROUGH A DAY-WARP STAYS IN BAND at the horizon being trained.
     The estimate is not the fragile part — position error IMPROVES across
     warps (8298 -> 1675 m through one) and is ~730 m at 192 h. The covariance
     is: median NEES runs 1.2 (24 h) -> 1.7 (48 h) -> 5.3 (96 h) -> 13 (192 h)
     with 81% of rows outside the 6-dof band by W1's own ~8-day horizon. That
     is NOT currently a training-signal bug — NAV-F dropped the sigma channel,
     so no covariance reaches the policy (obs[29-32] are hard zero) — but it
     WOULD become one the moment anyone re-enables it, and it already shows up
     as ~5% filter divergence. Gated so the reason is on the record.
"""
import argparse
import os
import subprocess
import sys

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))

G_PASS, G_FAIL = [], []


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


def _run(script, args):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       capture_output=True, text=True, env=env)
    return r.stdout


def gate_w1a(args):
    print('\n== W1A  surrogate vs real batch IOD at the TRAINING cadence ==')
    out = _run('w1nav_surrogate_vs_real.py',
               ['--n', str(args.n), '--seeds'] + [str(s) for s in args.seeds]
               + ['--hours', str(args.hours), '--max-ticks', str(args.max_ticks)])
    row = [l for l in out.splitlines() if 'h ' in l and '%' in l]
    worst, txt = 0.0, ''
    for l in row:
        try:
            dis = float(l.split('%')[3].split()[-1])
        except Exception:
            continue
        worst = max(worst, abs(dis)); txt = l.strip()
    check(f'W1A surrogate/real disagreement <= {args.tol}% at nav_max_ticks='
          f'{args.max_ticks}', worst <= args.tol,
          f'{txt or out.strip()[-200:]}  (worst |disagree| {worst:.1f}%)')


def gate_w1b(args):
    print('\n== W1B  NEES through the day-warp at the trained horizon ==')
    out = _run('w1nav_warp_probe.py',
               ['--stage', 'arc', '--n', str(args.n), '--settle', '45',
                '--max-ticks', str(args.max_ticks), '--seed', str(args.seeds[0]),
                '--hours', str(args.hours)])
    line = [l for l in out.splitlines() if 'h ' in l and 'm' in l]
    ok, txt = False, ''
    for l in line:
        p = l.split()
        try:
            med, oob = float(p[2]), float(p[4].rstrip('%'))
        except Exception:
            continue
        txt = l.strip()
        ok = (med <= 2.408) and (oob <= args.oob_tol)
    check(f'W1B NEES median in band and out-of-band <= {args.oob_tol}%', ok,
          f'{txt}  (6-dof band top 2.408)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='w1a,w1b')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seeds', type=int, nargs='+', default=[11, 4242])
    ap.add_argument('--hours', type=float, default=24.0)
    ap.add_argument('--max-ticks', type=int, default=0)
    ap.add_argument('--tol', type=float, default=5.0)
    ap.add_argument('--oob-tol', type=float, default=35.0)
    a = ap.parse_args()
    for s in a.stage.split(','):
        {'w1a': gate_w1a, 'w1b': gate_w1b}[s.strip()](a)
    print(f'\n=== {len(G_PASS)}/{len(G_PASS) + len(G_FAIL)} W1xnav gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    sys.exit(1 if G_FAIL else 0)
