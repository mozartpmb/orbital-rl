#!/usr/bin/env python3
"""Backfill wandb summary runs for T3/T4 rungs that trained before wandb
logging was enabled (2026-08-11). One run per rung: exact config, held-out
result, checkpoint path, repo commit. Tagged 'backfill' — these are summary
entries, NOT live training curves (those were only in terminal dashboards).
Natively-logged chains (t4-wide-s1337, t4-wide-s20260423, t4-meo-s42) are
excluded. Idempotence: run ids are deterministic (bf-<group>-<name>), so
re-running overwrites rather than duplicates.
"""
import subprocess, sys
import wandb

COMMIT = subprocess.check_output(
    ['git', '-C', '/Users/pete/space_training', 'rev-parse', '--short', 'HEAD']).decode().strip()

HEAD_CFG = dict(shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
                episode_cap_steps=3000, cap_terminal_reward=0.0, valid_init_only=1,
                init_phase_gap_max=3.14159, eval_seed=123, eval_episodes=200)
L1   = dict(HEAD_CFG, same_orbit_init=1, e_max=0.0, a_band='500-800km')
L2   = dict(HEAD_CFG, e_max=0.05, a_band='300-800km')
L3   = dict(HEAD_CFG, e_max_target=0.15, de_max=0.06, da_max_m=400e3, a_band='300-2000km', shape_dv_ref_ms=700)
WIDE = dict(obs_alt_scale_m=8e6, lvlh_scale_m=1.5e7, shape_dv_ref_ms=700)
W1, W2 = dict(L1, **WIDE), dict(L2, **WIDE)
W3 = dict(L3, **WIDE)
W4 = dict(HEAD_CFG, **WIDE, e_max_target=0.30, de_max=0.08, da_max_m=600e3,
          a_band='300-8000km', episode_cap_steps=6000)
D20 = dict(legacy_action_space=20)
TBH = dict(L2, **D20)

# (group, name, config, success_n, notes, ckpt)
RUNS = [
  ('t3-leo-s42',  'L1',  L1, 200, 'first corrected-dynamics RL success; fresh 50M', 'puffer_orbital_178642016215'),
  ('t3-leo-s42',  'L2-headline', L2, 200, 'CANONICAL: +200/200 stochastic, +200/200 cap2000, +500/500 seed777', 'puffer_orbital_178642097817 = models/t3/seed42_L2_headline.pt'),
  ('t3-leo-s42',  'L3-widee', L3, 200, 'also retains 200/200 at headline', 'puffer_orbital_178642670436 = models/t3/seed42_L3_widee.pt'),
  ('t3-leo-s7',   'L1', L1, 200, 'batch1; attribution by launch-stamp (campaign doc §6)', 'puffer_orbital_178642176815'),
  ('t3-leo-s7',   'L2', L2, 200, 'batch1; warm-start source ambiguous between two 100% L1s', 'puffer_orbital_178642258109'),
  ('t3-leo-s1337','L1', L1, 200, 'historical 0%-flatline seed', 'puffer_orbital_178642177125'),
  ('t3-leo-s1337','L2', L2, 200, 'historical 0%-flatline seed', 'puffer_orbital_178642259027'),
  ('t3-leo-s20260423', 'L1', L1, 200, 'batch2, clean attribution', 'puffer_orbital_178642360522'),
  ('t3-leo-s20260423', 'L2', L2, 200, '', 'puffer_orbital_178642431536'),
  ('t3-leo-s31415', 'L1', L1, 200, '', 'puffer_orbital_178642505036'),
  ('t3-leo-s31415', 'L2', L2, 200, '', 'puffer_orbital_178642576174'),
  ('t3-wide-s42', 'WL1', W1, 200, '', 'puffer_orbital_178642770764'),
  ('t3-wide-s42', 'WL2', W2, 200, '', 'puffer_orbital_178642839460'),
  ('t3-wide-s42', 'WL3', W3, 200, '', 'puffer_orbital_178642912348'),
  ('t3-wide-s42', 'WL4', W4, 200, 'realized e_t p50 0.140/p90 0.261; models/t3/seed42_WL4_wide.pt', 'puffer_orbital_178642988768'),
  ('t4-wide-s7',  'WL1', W1, 200, '', 't4_wide_s7/puffer_orbital_178645234552'),
  ('t4-wide-s7',  'WL2', W2, 199, 'sole wide-family miss: 1 safety_cap timeout (reproduced)', 't4_wide_s7/puffer_orbital_178645303879'),
  ('t4-wide-s7',  'WL3', W3, 200, '', 't4_wide_s7/puffer_orbital_178645379168'),
  ('t4-wide-s7',  'WL4', W4, 200, '', 't4_wide_s7/puffer_orbital_178645455683'),
  ('t4-tb-s42', 'TB1', dict(L1, **D20), 200, 'fresh Discrete-20 bootstraps cleanly', 't4_tb_s42/puffer_orbital_178645234552'),
  ('t4-tb-s42', 'TB2', TBH, 200, '', 't4_tb_s42/puffer_orbital_178645375267'),
  ('t4-tb-s42', 'TB3', dict(TBH, rendezvous_radius_m=10000, rel_vel_tol_ms=10), 200, 'box 10km/10m/s = models/t3/seed42_TB3_box10k10.pt', 't4_tb_s42/puffer_orbital_178645491908'),
  ('t4-tb-s42', 'TB4', dict(TBH, rendezvous_radius_m=5000, rel_vel_tol_ms=2), 200, 'box 5km/2m/s', 't4_tb_s42/puffer_orbital_178645596566'),
  ('t4-tb-s42', 'TB5', dict(TBH, rendezvous_radius_m=5000, rel_vel_tol_ms=1), 191, 'box 5km/1m/s; 9 cap timeouts; capture |v_rel| p50 0.91 m/s vs 1.0 tol = models/t3/seed42_TB5_box5k1.pt', 't4_tb_s42/puffer_orbital_178645681906'),
]

def main():
    for group, name, cfg, n_ok, notes, ckpt in RUNS:
        seed = int(group.rsplit('-s', 1)[1])
        run = wandb.init(
            project='orbital-rl', group=group, name=f'{name}-s{seed}',
            id=f'bf-{group}-{name}'.replace('.', '-'), resume='allow',
            tags=['backfill', 'summary-only', 'corrected-dynamics'],
            config=dict(cfg, train_seed=seed, total_timesteps=50_000_000,
                        experiment_dir=ckpt, repo_commit=COMMIT),
            notes=(f'Backfilled summary (no live curves). Held-out greedy eval, '
                   f'Physical-success classifier. {notes} '
                   f'Record: T3_RECOVERY_CAMPAIGN.md / MODELS.md.'),
            settings=wandb.Settings(console='off'),
        )
        run.summary['heldout_success'] = n_ok / 200.0
        run.summary['heldout_n'] = 200
        run.summary['heldout_frac'] = f'{n_ok}/200'
        run.finish(quiet=True)
        print(f'{group}/{name}: {n_ok}/200 -> {run.url}')

if __name__ == '__main__':
    main()
