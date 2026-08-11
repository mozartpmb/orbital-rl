"""ATTACK E — is phase_gap_mode=1 actually correct?

Derivation.  c_reset sets  M_t = M_s + gap  (+ omega_s - omega_t in mode 1),
so with lambda = M + omega:

  mode 0:  lambda_t - lambda_s = gap + (omega_t - omega_s)   <- inert at e>0
  mode 1:  lambda_t - lambda_s = gap                          <- exact

i.e. the observed  dlam = wrap(lambda_s - lambda_t) = -gap.  NOTE THE SIGN:
`init_phase_gap` puts the TARGET AHEAD by gap.

Verified empirically over 200 resets per cell, for e in {0, 0.05}, both
same_orbit_init values, both a_sat==a_tgt and a_sat!=a_tgt, and with
valid_init_only on/off at an eccentricity where the rejection sampler is
actually active.

Outputs web_data/results/t3_redteam_phase_gap.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_common import ObsView, T3_KW, make_env  # noqa

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_phase_gap.csv'


def sample(n, **kw):
    """n resets; returns arrays of realised dlam, domega, and init attempts."""
    dl, dw = [], []
    env = make_env(seed=kw.pop('seed', 0), **kw)
    for i in range(n):
        env.reset(seed=1000 + i)
        o = ObsView(env.observations[0],
                    phase_obs_mode=kw.get('phase_obs_mode', 1))
        dl.append(o.dlam)
        dw.append((o.omega_s - o.omega_t + math.pi) % (2 * math.pi) - math.pi)
    env.close()
    return np.array(dl), np.array(dw)


def main():
    rows = []
    N = 200
    print('=== FIXED gap: realised |dlam| must equal the requested gap ===')
    print(f'{"cfg":42s} {"mode":>4s} {"req":>6s} {"mean dlam":>10s} '
          f'{"max err":>10s} {"mean|dw|":>9s}')
    cells = [
        ('e=0    soi=1 (a_s==a_t)', dict(e_max_target=0.0, e_max_sat=0.0, same_orbit_init=1)),
        ('e=0    soi=0 (a_s!=a_t)', dict(e_max_target=0.0, e_max_sat=0.0, same_orbit_init=0)),
        ('e=0.05 soi=1', dict(e_target_fixed=0.05, e_sat_fixed=0.05, same_orbit_init=1)),
        ('e=0.05 soi=0 (headline)', dict(e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0)),
        ('e=0.20 soi=0 valid_init_only=1', dict(e_max_target=0.20, e_max_sat=0.20,
                                                same_orbit_init=0, valid_init_only=1)),
        ('e=0.20 soi=0 valid_init_only=0', dict(e_max_target=0.20, e_max_sat=0.20,
                                                same_orbit_init=0, valid_init_only=0)),
    ]
    for name, kw in cells:
        for mode in (0, 1):
            for req in (30.0, 90.0, 180.0):
                kws = dict(T3_KW)
                kws.update(kw)
                kws['phase_gap_mode'] = mode
                kws['phase_gap_fixed'] = math.radians(req)
                dl, dw = sample(N, **kws)
                err = np.abs(np.degrees(dl) + req)
                err = np.minimum(err, 360 - err)
                rows.append(dict(cfg=name, mode=mode, requested_deg=req,
                                 mean_dlam_deg=float(np.degrees(dl).mean()),
                                 max_err_deg=float(err.max()),
                                 mean_abs_domega_deg=float(np.abs(np.degrees(dw)).mean())))
                if req == 90.0:
                    print(f'{name:42s} {mode:4d} {req:6.0f} '
                          f'{np.degrees(dl).mean():10.3f} {err.max():10.4f} '
                          f'{np.abs(np.degrees(dw)).mean():9.2f}')

    print('\n=== UNIFORM gap (init_phase_gap_max): distribution shape ===')
    print(f'{"cfg":42s} {"mode":>4s} {"max_deg":>8s} {"mean|dlam|":>11s} '
          f'{"p99|dlam|":>10s} {"KS-vs-uniform":>14s}')
    for name, kw in cells:
        for mode in (0, 1):
            for gmax in (30.0, 180.0):
                kws = dict(T3_KW)
                kws.update(kw)
                kws['phase_gap_mode'] = mode
                kws['init_phase_gap_max'] = math.radians(gmax)
                dl, dw = sample(N, **kws)
                ad = np.abs(np.degrees(dl))
                # KS distance of |dlam| against U(0, gmax)
                x = np.sort(ad) / gmax
                ks = float(np.max(np.abs(np.arange(1, len(x) + 1) / len(x) - x)))
                rows.append(dict(cfg=name + ' [uniform]', mode=mode,
                                 requested_deg=gmax,
                                 mean_dlam_deg=float(ad.mean()),
                                 max_err_deg=float(ad.max()),
                                 mean_abs_domega_deg=ks))
                if gmax == 30.0:
                    print(f'{name:42s} {mode:4d} {gmax:8.0f} {ad.mean():11.3f} '
                          f'{np.percentile(ad, 99):10.3f} {ks:14.3f}')

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {OUT}')


if __name__ == '__main__':
    main()
