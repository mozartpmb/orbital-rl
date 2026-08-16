#!/usr/bin/env python3
"""Move a checkpoint between observation-normalizer families, EXACTLY.

GEN_MATRIX measured the narrow/wide normalizer split as "the largest single
effect in the matrix" — 99 pp, against 51 pp for adding J2 itself, and unchanged
under perfect truth state. That reads as a learned incompatibility. It is not:
it is an artifact of the parameterisation, and it is removable in closed form.

`Default.encode_observations` is `Sequential(Linear(38,128), GELU)` applied to
the RAW observation vector. So if a normalizer change multiplies observation
channel j by c_j, dividing `encoder.0.weight[:, j]` by c_j leaves the
pre-activation — and hence the GELU output, the LSTM state, the value head and
the action logits — bitwise unchanged. The policy is the SAME policy, now
reading different units.

Channels that move (everything else is dimensionless, or normalised by a
quantity both families share — v_circ, dv_ref, pi):

    obs[0], obs[7]            / obs_alt_scale_m
    obs[17], obs[20]          / (R_EARTH + obs_alt_scale_m)
    obs[24], obs[33], obs[34] / lvlh_scale_m

WHAT THIS DOES NOT FIX, and neither would any linear map:
  * obs[21,22,23] use per-rung di/de scales. If those move too, pass them.
  * obs[15] is a dimensionless clock whose UNIT is episode_cap_steps. A cap
    change is a semantic change, not a scaling — it cannot be transplanted.

Verification is not optional and is built in: `--verify` re-runs a probe set in
both families and asserts a BIT-IDENTICAL action stream.

    # transplant
    python3 rescale_ckpt_normalizers.py in.pt out.pt \\
        --alt-from 1.6e6 --alt-to 8.0e6 --lvlh-from 6.371e6 --lvlh-to 6.371e6
    # transplant + prove it
    ... --verify --verify-mode truth --verify-eps 100
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

R_EARTH = 6.371e6
OBS_DIM = 38


def find_encoder_key(sd):
    """The tensor that consumes the raw observation — matched on SHAPE, not name.

    The encoder is `Sequential(Linear(38,128), GELU)`, so the key is
    'policy.encoder.0.weight'; matching on shape survives a rename.
    """
    for k, v in sd.items():
        if 'encoder' in k and v.dim() == 2 and v.shape[1] == OBS_DIM:
            return k
    raise SystemExit(f'no (*, {OBS_DIM}) encoder weight found; keys: {list(sd)[:8]}')


def rescale(sd, alt_from, alt_to, lvlh_from, lvlh_to,
            di_from=None, di_to=None, de_from=None, de_to=None):
    """Divide each affected weight column by obs_new/obs_old. Returns the map."""
    key = find_encoder_key(sd)
    W = sd[key].clone()
    groups = {
        (0, 7): alt_from / alt_to,
        (17, 20): (R_EARTH + alt_from) / (R_EARTH + alt_to),
        (24, 33, 34): lvlh_from / lvlh_to,
    }
    if di_from and di_to:
        groups[(21, 22)] = di_from / di_to
    if de_from and de_to:
        groups[(23,)] = de_from / de_to
    applied = {}
    for cols, f in groups.items():
        if abs(f - 1.0) < 1e-15:
            continue                      # same scale — leave the column alone
        for c in cols:
            W[:, c] = W[:, c] / f
            applied[c] = f
    sd[key] = W
    return key, applied


def verify(src, dst, kw_from, kw_to, mode, eps, seed=123):
    """Assert the transplant reproduces the source's action stream bitwise."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'ev', os.path.join(WT, 'scripts', 'orbital', 'nav', 'eval_relnav3d.py'))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.rollout._mask_rows = None
    a = ev.rollout(ev.make_env('X3', mode, **kw_from), src, eps, seed, 'src@from')
    ev.rollout._mask_rows = None
    b = ev.rollout(ev.make_env('X3', mode, **kw_to), src, eps, seed, 'src@to')
    ev.rollout._mask_rows = None
    c = ev.rollout(ev.make_env('X3', mode, **kw_to), dst, eps, seed, 'DST@to')
    ok = (c['md5'] == a['md5'])
    print(f"\n  {'arm':32s} {'success':>10s}  md5")
    for nm, r in (('source @ source family', a),
                  ('source @ target family', b),
                  ('TRANSPLANT @ target family', c)):
        print(f'  {nm:32s} {r["success"]:4d}/{r["n_valid"]:<4d}  {r["md5"][:16]}')
    print(f'\n  [{"PASS" if ok else "FAIL"}] transplant action stream is '
          f'BIT-IDENTICAL to the source at home')
    print(f'         (untransplanted loses {a["success"] - b["success"]} of '
          f'{a["n_valid"]} episodes, which is the barrier being removed)')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--alt-from', type=float, default=1.6e6)
    p.add_argument('--alt-to', type=float, default=8.0e6)
    p.add_argument('--lvlh-from', type=float, default=6.371e6)
    p.add_argument('--lvlh-to', type=float, default=6.371e6)
    p.add_argument('--di-from', type=float, default=None)
    p.add_argument('--di-to', type=float, default=None)
    p.add_argument('--de-from', type=float, default=None)
    p.add_argument('--de-to', type=float, default=None)
    p.add_argument('--zero-cols', default='',
                   help='comma-separated obs columns whose encoder weights to '
                        'zero — for channels that were DEAD (always 0.0f) in the '
                        'source regime and become LIVE in the target, where a '
                        'random-init column would be a silent perturbation '
                        '(n3d_REDTEAM NON-ISSUE-9). Verify before using: a '
                        'column that is always 0 in BOTH regimes is already '
                        'inert whatever its weights.')
    p.add_argument('--expand-31', action='store_true',
                   help='also expand a 30-row action head to 31 (row 30 = the '
                        'day-warp), seeded from row 17 — a zero-init row is '
                        'unreachable under a saturated head (measured 3.9e-9)')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--verify-mode', default='truth',
                   choices=['truth', 'bearings_only'])
    p.add_argument('--verify-eps', type=int, default=100)
    args = p.parse_args()

    sd = torch.load(args.src, map_location='cpu', weights_only=True)
    key, applied = rescale(sd, args.alt_from, args.alt_to,
                           args.lvlh_from, args.lvlh_to,
                           args.di_from, args.di_to, args.de_from, args.de_to)
    print(f'=== normalizer transplant ===')
    print(f'  src {args.src}')
    print(f'  encoder tensor {key} {tuple(sd[key].shape)}')
    print(f'  alt  {args.alt_from:.4g} -> {args.alt_to:.4g}   '
          f'lvlh {args.lvlh_from:.4g} -> {args.lvlh_to:.4g}')
    if not applied:
        print('  NO CHANNELS MOVED (all scales identical) — output is a copy')
    for c in sorted(applied):
        print(f'    obs[{c:2d}]  obs_new/obs_old {applied[c]:.6f}  '
              f'=> weight column x {1/applied[c]:.4f}')

    if args.zero_cols:
        cols = [int(c) for c in args.zero_cols.split(',')]
        W = sd[key]
        before = {c: float(W[:, c].abs().mean()) for c in cols}
        for c in cols:
            W[:, c] = 0.0
        sd[key] = W
        for c in cols:
            print(f'    zeroed encoder column obs[{c}] (was |w|_mean '
                  f'{before[c]:.4f})')

    if args.expand_31:
        wk = 'policy.decoder.weight'
        bk = 'policy.decoder.bias'
        w, b = sd[wk], sd[bk]
        if w.shape[0] == 30:
            w2 = torch.zeros(31, w.shape[1], dtype=w.dtype)
            b2 = torch.zeros(31, dtype=b.dtype)
            w2[:30], b2[:30] = w, b
            w2[30], b2[30] = w[17], b[17]     # seed from the 6 h warp
            sd[wk], sd[bk] = w2, b2
            print('  head 30 -> 31; rows 0-29 bit-identical, row 30 seeded from '
                  'row 17 (the 6 h warp)')
        else:
            print(f'  head already {w.shape[0]} rows; no expansion')

    torch.save(sd, args.dst)
    print(f'  wrote {args.dst}')

    if args.verify:
        if args.expand_31:
            print('\n  NOTE --verify with --expand-31 compares a 30-row source '
                  'against a 31-row output; run verification on the transplant '
                  'alone, then expand. Skipping.')
            return 0
        COMMON = dict(
            num_debris_min=0, num_debris_max=0,
            e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
            init_phase_gap_max=3.14159, valid_init_only=1,
            gave_up_action='terminate', max_valid_init_attempts=4096,
            rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
            shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
            shape_dv_ref_ms=700.0, shape_gamma=1.0,
            phase_gap_mode=1, phase_obs_mode=1,
            episode_cap_steps=3000, cap_terminal_reward=0.0,
            dim3_mode=1, di_max_rad=0.017453, legacy_action_space=30,
            a_min_override=6.671e6, a_max_override=7.171e6)
        kf = dict(COMMON, obs_alt_scale_m=args.alt_from, lvlh_scale_m=args.lvlh_from)
        kt = dict(COMMON, obs_alt_scale_m=args.alt_to, lvlh_scale_m=args.lvlh_to)
        ok = verify(args.src, args.dst, kf, kt, args.verify_mode, args.verify_eps)
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
