#!/usr/bin/env python3
"""Bootstrap tripwire for the ext-j2 ladder arms.

WHY THIS EXISTS. The tight-box arms run with `cap_terminal_reward = 0.0` and
`shape_gamma = 1`, so an episode that runs out the clock pays NO terminal
penalty and still collects the full telescoped Phi_T - Phi_0. If the warm start
never samples a success, the +10 is never seen and PPO has nothing to climb —
the T3 red-team #1 flatline mechanism (a -10 timeout under flat per-decision
gamma prices warps as a -7.8 bet; setting it to 0 fixes that but removes the
pressure to ever finish). At the 5 km / 1 m/s box the J2-blind policy fails
200/200 on safety_cap, so this is a live risk, not a hypothetical.

This parses the trainer's own rolling `perf` against its `Steps` counter and
says FLATLINE in the progress log if the arm never got more than `--margin`
above the measured zero-shot floor after `--after-steps`. A flatline named in
the log is a finding; one discovered in the eval three stages later is a waste.

The trainer redraws a dashboard, so both fields appear once per frame in the
order `Steps ... perf ...`. Each perf sample is paired with the most recent
Steps value. Verified against a real 384-frame run log.

    python3 j2_flatline_check.py --log /tmp/x.log --floor 0.565 \
        --after-steps 20e6 --margin 0.05
"""

import argparse
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
TOKEN = re.compile(r'(Steps|perf)\s+([0-9.]+)\s*([KMG]?)')
SCALE = {'': 1.0, 'K': 1e3, 'M': 1e6, 'G': 1e9}


def parse(path):
    """Return [(steps, perf), ...] in log order."""
    with open(path, 'r', errors='replace') as f:
        text = ANSI.sub('', f.read())
    out, cur_steps = [], None
    for m in TOKEN.finditer(text):
        field, val, suf = m.group(1), float(m.group(2)), m.group(3)
        if field == 'Steps':
            cur_steps = val * SCALE[suf]
        elif cur_steps is not None:
            # 'perf' is a rate in [0,1]; a suffix on it would mean we matched
            # something else on the same line, so ignore those.
            if suf == '':
                out.append((cur_steps, val))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--floor', type=float, required=True,
                    help='measured zero-shot floor for THIS box, as a rate in [0,1]')
    ap.add_argument('--after-steps', type=float, default=20e6)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--label', default='arm')
    args = ap.parse_args()

    try:
        samples = parse(args.log)
    except OSError as e:
        print(f'FLATLINE-CHECK {args.label}: UNREADABLE ({e})')
        return 2

    if not samples:
        print(f'FLATLINE-CHECK {args.label}: NO PERF SAMPLES in {args.log} '
              f'(trainer produced no dashboard frames?)')
        return 2

    target = args.floor + args.margin
    late = [(s, p) for s, p in samples if s >= args.after_steps]
    peak_all = max(p for _, p in samples)
    final = samples[-1][1]
    last_steps = samples[-1][0]

    if not late:
        print(f'FLATLINE-CHECK {args.label}: INCOMPLETE — run reached only '
              f'{last_steps/1e6:.1f}M steps, never passed the '
              f'{args.after_steps/1e6:.0f}M mark. peak perf {peak_all:.3f}, '
              f'floor {args.floor:.3f}, target {target:.3f}')
        return 2

    peak_late = max(p for _, p in late)
    verdict = 'OK' if peak_late > target else 'FLATLINE'
    first_cross = next((s for s, p in samples if p > target), None)
    cross_txt = (f'first crossed at {first_cross/1e6:.1f}M'
                 if first_cross is not None else 'never crossed')
    print(f'FLATLINE-CHECK {args.label}: {verdict} — peak perf after '
          f'{args.after_steps/1e6:.0f}M = {peak_late:.3f} vs target {target:.3f} '
          f'(floor {args.floor:.3f} + {args.margin:.2f}); peak overall '
          f'{peak_all:.3f}, final {final:.3f}, {cross_txt}, '
          f'{len(samples)} samples to {last_steps/1e6:.1f}M steps')
    # NOT a hard failure: a flatline is a FINDING and the campaign must
    # continue and report it (exit 0 either way; the verdict is in the text).
    return 0


if __name__ == '__main__':
    sys.exit(main())
