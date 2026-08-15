"""Evaluate a trained orbital model and save trajectory logs.

Usage:
    python scripts/orbital/eval_checkpoint.py experiments/puffer_orbital_177610309074/model_puffer_orbital_000153.pt
    python scripts/orbital/eval_checkpoint.py experiments/puffer_orbital_177610323688/model_puffer_orbital_000306.pt --debris
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add pufferlib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pufferlib.ocean.orbital.orbital import Orbital
from pufferlib.models import Default, LSTMWrapper


def evaluate(checkpoint_path, num_episodes=50, debris=False, out_dir=None, seed=42,
             e_max_target=0.0, init_phase_gap_max=0.524, e_max_sat=0.0, same_orbit_init=0,
             enable_action_mask=0, valid_init_only=0,
             e_target_fixed=-1.0, e_sat_fixed=-1.0, phase_gap_fixed=-1.0,
             omega_offset_fixed=-99.0, a_min_override=-1.0, a_max_override=-1.0,
             log_validation_debug=0,
             max_valid_init_attempts=4096, gave_up_action="terminate",
             obs_alt_scale_m=1.6e6, phi_orbit_scale_k=0.001,
             lvlh_scale_m=6.371e6, legacy_action_space=None,
             stochastic=False,
             rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
             shaping_mode=0, shape_w_lambda=1.0, shape_w_match=0.35,
             shape_dv_ref_ms=300.0, shape_gamma=0.995,
             phase_gap_mode=0, phase_obs_mode=0, episode_cap_steps=2000,
             cap_terminal_reward=-10.0, de_max=-1.0, da_max_m=-1.0,
             dim3_mode=0, di_max_rad=-1.0, i_target_rad=0.0, raan_target_rad=0.0,
             obs_di_scale_rad=-1.0, obs_de_scale=-1.0, shape_match_squash=0,
             j2_mode=0, i_target_min_rad=-1.0, i_target_max_rad=-1.0,
             raan_target_sample=0, lvlh_frame_mode=0,
             di_min_rad=-1.0, di_phase_mode=0):
    if out_dir is None:
        tag = "debris" if debris else "no_debris"
        ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        out_dir = f"logs/orbital/eval_{tag}_{ckpt_name}"

    os.makedirs(out_dir, exist_ok=True)

    # Create environment (single env for clean trajectory collection)
    num_debris_min = 4 if debris else 0
    num_debris_max = 8 if debris else 0
    env = Orbital(
        num_envs=1,
        num_debris_min=num_debris_min,
        num_debris_max=num_debris_max,
        e_max_target=e_max_target,
        init_phase_gap_max=init_phase_gap_max,
        e_max_sat=e_max_sat,
        same_orbit_init=same_orbit_init,
        enable_action_mask=enable_action_mask,
        valid_init_only=valid_init_only,
        e_target_fixed=e_target_fixed,
        e_sat_fixed=e_sat_fixed,
        phase_gap_fixed=phase_gap_fixed,
        omega_offset_fixed=omega_offset_fixed,
        a_min_override=a_min_override,
        a_max_override=a_max_override,
        log_validation_debug=log_validation_debug,
        max_valid_init_attempts=max_valid_init_attempts,
        gave_up_action=gave_up_action,
        obs_alt_scale_m=obs_alt_scale_m,
        phi_orbit_scale_k=phi_orbit_scale_k,
        lvlh_scale_m=lvlh_scale_m,
        rendezvous_radius_m=rendezvous_radius_m,
        rel_vel_tol_ms=rel_vel_tol_ms,
        shaping_mode=shaping_mode,
        shape_w_lambda=shape_w_lambda,
        shape_w_match=shape_w_match,
        shape_dv_ref_ms=shape_dv_ref_ms,
        shape_gamma=shape_gamma,
        phase_gap_mode=phase_gap_mode,
        phase_obs_mode=phase_obs_mode,
        episode_cap_steps=episode_cap_steps,
        cap_terminal_reward=cap_terminal_reward,
        de_max=de_max,
        da_max_m=da_max_m,
        dim3_mode=dim3_mode,
        di_max_rad=di_max_rad,
        i_target_rad=i_target_rad,
        raan_target_rad=raan_target_rad,
        obs_di_scale_rad=obs_di_scale_rad,
        obs_de_scale=obs_de_scale,
        shape_match_squash=shape_match_squash,
        j2_mode=j2_mode,
        i_target_min_rad=i_target_min_rad,
        i_target_max_rad=i_target_max_rad,
        raan_target_sample=raan_target_sample,
        di_min_rad=di_min_rad,
        di_phase_mode=di_phase_mode,
        lvlh_frame_mode=lvlh_frame_mode,
        traj_log_dir=out_dir,
        traj_log_every=1,  # save every episode
    )

    # M2/M3 (phase5-5-env-mods): Phase 5b/5e ckpts have a 10-dim logits head.
    # The env exposes Discrete(16) by default; coerce env.single_action_space so
    # policy construction produces a head matching the ckpt's state_dict shape.
    # Works both ways: 10-head Phase-5b/5e ckpts shrink the view, 20-head T4
    # ckpts (fine radial + 3h/6h warps) enlarge it. c_step accepts any int in
    # [0, NUM_ACTIONS=20); an N-logit argmax only ever emits [0, N).
    if legacy_action_space is not None and legacy_action_space != env.single_action_space.n:
        import gymnasium
        full_n = env.single_action_space.n
        env.single_action_space = gymnasium.spaces.Discrete(legacy_action_space)
        print(f"[legacy-action-space] policy head sized to {legacy_action_space}; "
              f"env exposed Discrete({full_n}), C env accepts ints in [0, 20).")

    # Load model — PufferLib wraps Default in LSTMWrapper
    device = torch.device("cpu")
    base_policy = Default(env)
    policy = LSTMWrapper(env, base_policy)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Debris: {debris} (min={num_debris_min}, max={num_debris_max})")
    print(f"Saving trajectories to: {out_dir}")

    torch.manual_seed(seed)  # reproducible stochastic sampling
    obs, _ = env.reset(seed=seed)
    episodes_done = 0
    episode_rewards = []
    episode_lengths = []
    successes = 0           # legacy classifier: terminal reward > 0 (Φ-clamp-leak prone)
    physical_successes = 0  # corrected classifier: terminal branch == success
    cause_names = ['none', 'success', 'collision', 'escape', 'safety_cap',
                   'stranded', 'hyperbolic', 'gave_up']
    cause_counts = {name: 0 for name in cause_names}
    step_count = 0

    # LSTM hidden state (dict form for LSTMWrapper.forward_eval)
    state = {
        'lstm_h': torch.zeros(1, policy.hidden_size),
        'lstm_c': torch.zeros(1, policy.hidden_size),
    }

    while episodes_done < num_episodes:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = policy.forward_eval(obs_tensor, state)
            if stochastic:
                # Sample from softmax distribution (matches PPO training-time policy)
                probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0).numpy()
            else:
                action = torch.argmax(action_logits, dim=-1).numpy()

        obs, rewards, terminals, truncations, infos = env.step(action)
        step_count += 1

        if terminals[0]:
            episodes_done += 1
            rew = float(rewards[0])
            episode_rewards.append(rew)
            episode_lengths.append(step_count)
            if rew > 0:
                successes += 1
            # Corrected classifier: key on the terminal branch that fired, not
            # the reward sign — the Φ-clamp can flip failure terminals positive
            # at wide altitude bands (Φ-clamp leak, 9th metric-vs-implementation
            # instance). Gave-up inits are counted separately and excluded from
            # the physical-success denominator.
            sim_steps, cause = env.last_episode_result(0)
            cause_counts[cause_names[cause]] += 1
            if cause == 1:
                physical_successes += 1
            step_count = 0
            # Reset LSTM state for new episode
            state['lstm_h'] = torch.zeros(1, policy.hidden_size)
            state['lstm_c'] = torch.zeros(1, policy.hidden_size)

            if episodes_done % 10 == 0:
                print(f"  Episode {episodes_done}/{num_episodes}: "
                      f"reward={rew:.1f}, success_rate={successes/episodes_done:.1%}")

    env.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Success rate:    {successes}/{num_episodes} ({successes/num_episodes:.1%})")
    n_gave_up = cause_counts['gave_up']
    n_valid = num_episodes - n_gave_up
    phys_rate = physical_successes / n_valid if n_valid > 0 else 0.0
    print(f"Physical success: {physical_successes}/{n_valid} ({phys_rate:.1%})"
          f"  [terminal branch == success; {n_gave_up} gave-up inits excluded]")
    causes_str = ", ".join(f"{k}={v}" for k, v in cause_counts.items() if v > 0)
    print(f"Terminal causes: {causes_str}")
    print(f"Mean reward:     {np.mean(episode_rewards):.2f}")
    print(f"Mean ep length:  {np.mean(episode_lengths):.0f} steps")

    # Check saved files
    npz_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.npz')])
    print(f"Saved {len(npz_files)} trajectory files to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate orbital RL checkpoint')
    parser.add_argument('checkpoint', help='Path to .pt checkpoint file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of eval episodes')
    parser.add_argument('--debris', action='store_true', help='Enable debris (4-8)')
    parser.add_argument('--out-dir', default=None, help='Output directory for trajectories')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--e-max-target', type=float, default=0.0,
                        help='Max target eccentricity (default 0.0 = circular target, matches orbital.ini)')
    parser.add_argument('--init-phase-gap-max', type=float, default=0.524,
                        help='Max initial phase gap in radians (default 0.524 = 30°)')
    parser.add_argument('--e-max-sat', type=float, default=0.0,
                        help='Max satellite eccentricity at init (Phase 5b; default 0.0 = circular chaser)')
    parser.add_argument('--same-orbit-init', type=int, default=0,
                        help='Phase 5b Stage 1: 1 = sat.{a,e,ω} = target.{a,e,ω}, only θ differs')
    parser.add_argument('--enable-action-mask', type=int, default=0,
                        help='Phase 5d I2: 1 = 48-dim obs with action validity mask')
    parser.add_argument('--valid-init-only', type=int, default=0,
                        help='Phase 5d: 1 = reject inits whose sat/target perigee < EARTH_KEEPOUT')
    parser.add_argument('--e-target-fixed', type=float, default=-1.0,
                        help='W1: exact target.e (>=0) overrides uniform-up-to-e_max_target')
    parser.add_argument('--e-sat-fixed', type=float, default=-1.0,
                        help='W1: exact sat.e (>=0) overrides uniform-up-to-e_max_sat')
    parser.add_argument('--phase-gap-fixed', type=float, default=-1.0,
                        help='W1: exact phase gap (rad, >=0) overrides uniform [-init_phase_gap_max,+]')
    parser.add_argument('--omega-offset-fixed', type=float, default=-99.0,
                        help='W1: target.omega - sat.omega offset (rad); >-10 enables')
    parser.add_argument('--a-min-override', type=float, default=-1.0,
                        help='Override semi-major axis floor (m). >= R_EARTH=6.371e6 to enable.')
    parser.add_argument('--a-max-override', type=float, default=-1.0,
                        help='Override semi-major axis ceiling (m). Must be > a_min_override.')
    parser.add_argument('--log-validation-debug', type=int, default=0,
                        help='Phase 5 verification I1: 1 = emit per-reset debug line to stderr')
    parser.add_argument('--max-valid-init-attempts', type=int, default=4096,
                        help='Phase 5 env-fix F1: cap on rejection-sampling attempts (was hardcoded 256)')
    parser.add_argument('--gave-up-action', choices=['accept', 'terminate'], default='terminate',
                        help='Phase 5 env-fix F3: behavior when cap exhausts with doomed init '
                             '(default "terminate" — cleaner eval semantics; "accept" = legacy)')
    parser.add_argument('--obs-alt-scale-m', type=float, default=1.6e6,
                        help='Phase 5.5: altitude obs normalization scale (m). Default 1.6e6 = '
                             'LEO ALT_MAX (backward compat). Use 4.2e7 for GEO-inclusive envelope.')
    parser.add_argument('--phi-orbit-scale-k', type=float, default=0.001,
                        help='Phase 5.5: Φ_orbit scale gain. Effective tolerance = '
                             'max(SUCCESS_TOL_A, K * obs_alt_scale_m). Default 0.001 → LEO compat.')
    parser.add_argument('--lvlh-scale-m', type=float, default=6.371e6,
                        help='M1 (phase5-5-env-mods): LVLH spatial obs normalizer. '
                             'Default 6.371e6 (R_EARTH) preserves Phase 5b/5e LEO behavior. '
                             'Set ~4.2e7 for GEO training.')
    parser.add_argument('--legacy-action-space', type=int, default=None,
                        help='M2/M3 backward compat: pass 10 to eval Phase 5b/5e ckpts whose '
                             'policy head has 10 logits. Coerces env.single_action_space for '
                             'policy construction; env still accepts ints in [0, NUM_ACTIONS).')
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions from softmax(logits) instead of argmax. Matches '
                             'PPO training-time policy. Default: greedy (argmax).')
    parser.add_argument('--rendezvous-radius-m', type=float, default=30000.0,
                        help='Success-box position tolerance (m). Default 30000 = historical '
                             '30 km far-field criterion. T1 tightening: try 5000 / 1000.')
    parser.add_argument('--rel-vel-tol-ms', type=float, default=50.0,
                        help='Success-box relative-velocity tolerance (m/s). Default 50 = '
                             'historical criterion. T1 tightening: try 1.0 / 0.5.')
    # T3 corrected-dynamics recovery kwargs (defaults = legacy; see T3_RECOVERY_CAMPAIGN.md §5)
    parser.add_argument('--shaping-mode', type=int, default=0,
                        help='T3: 0 = legacy gated Φ; 1 = S-R3 mean-longitude phase-time potential. '
                             'Shaping only affects training signal, but set it to match the ckpt '
                             'lineage when comparing episode returns.')
    parser.add_argument('--shape-w-lambda', type=float, default=1.0,
                        help='T3 shaping_mode 1: weight on |Δλ|/π term.')
    parser.add_argument('--shape-w-match', type=float, default=0.35,
                        help='T3 shaping_mode 1: weight on orbit-match Δv term.')
    parser.add_argument('--shape-dv-ref-ms', type=float, default=300.0,
                        help='T3 shaping_mode 1: Δv_match normalizer (m/s).')
    parser.add_argument('--shape-gamma', type=float, default=0.995,
                        help='T3: shaping discount base; >= 1.0 → γ_shape = 1 exactly.')
    parser.add_argument('--phase-gap-mode', type=int, default=0,
                        help='T3: 1 = init_phase_gap controls PHYSICAL mean-longitude gap '
                             '(legacy M-offset is inert at e>0).')
    parser.add_argument('--phase-obs-mode', type=int, default=0,
                        help='T3: 1 = obs[13-16] use mean longitude (BREAKING for pre-T3 ckpts).')
    parser.add_argument('--episode-cap-steps', type=int, default=2000,
                        help='T3: runtime safety cap in 60 s sim sub-steps (max 3000). '
                             'Legacy 2000 = 33.3 h wall clock.')
    parser.add_argument('--cap-terminal-reward', type=float, default=-10.0,
                        help='T3: reward at the safety-cap terminal. Legacy -10; T3 runs 0.0 '
                             '(red-team #1: -10 under per-decision gamma suppresses warps).')
    parser.add_argument('--de-max', type=float, default=-1.0,
                        help='T3 L3+: sat e-vector within de_max of target e-vector '
                             '(bounds |delta-e-vec|, not e). <0 = off (legacy sampling).')
    parser.add_argument('--da-max-m', type=float, default=-1.0,
                        help='T3 L3+: |a_target - a_sat| <= da_max_m (min 200 km). <0 = off.')
    # ── ext-3d (defaults = legacy 2D, bit-exact) ────────────────────────────
    parser.add_argument('--dim3-mode', type=int, default=0,
                        help='ext-3d: 1 = 3D planes, 3D obs block (slots 21-32), 3D phase-gap '
                             'knob. Requires num_debris = 0. Default 0 = 2D lineage.')
    parser.add_argument('--di-max-rad', type=float, default=-1.0,
                        help='ext-3d: relative-inclination knob (rad). >=0 samples '
                             'h_sat = R(delta, n)·h_target with delta = di_max_rad*sqrt(U). '
                             '~133 m/s per degree at LEO on a 478 m/s budget.')
    parser.add_argument('--i-target-rad', type=float, default=0.0,
                        help='ext-3d: absolute target inclination (rad). Pure gauge under '
                             'two-body; a test hook for the sampler/frame gates.')
    parser.add_argument('--raan-target-rad', type=float, default=0.0,
                        help='ext-3d: absolute target RAAN (rad). Gauge; test hook.')
    parser.add_argument('--obs-di-scale-rad', type=float, default=-1.0,
                        help='ext-3d: obs[21,22] normalizer. <=0 -> max(di_max_rad, 0.25 deg).')
    parser.add_argument('--obs-de-scale', type=float, default=-1.0,
                        help='ext-3d: obs[23] normalizer. <=0 -> max(de_max, 0.05).')
    parser.add_argument('--shape-match-squash', type=int, default=0,
                        help='ext-3d: Phi match squash. 0 = min(1, x) (legacy, A2 anchor); '
                             '1 = x/(1+x) (no dead zone).')
    # ── ext-j2 (default = legacy propagator, bit-exact) ─────────────────────
    parser.add_argument('--j2-mode', type=int, default=0,
                        help='ext-j2: 1 = secular mean-element J2 (Omega-dot, omega-dot, '
                             'M-dot corrections; exact under warps). Requires dim3-mode 1 '
                             'and num_debris 0. Writes obs[29]=cos i_sat, obs[30]=cos i_tgt. '
                             'Default 0 = verbatim legacy propagator.')
    parser.add_argument('--i-target-min-rad', type=float, default=-1.0,
                        help='ext-j2: sample target inclination U(min, max) per episode. '
                             'Both >= 0 and max > min to enable. THIS is what makes J2 '
                             'non-inert (channel goes as sin 2i, zero at 0 and 90 deg). '
                             'Rung band 0.5236..1.0472 = 30..60 deg.')
    parser.add_argument('--i-target-max-rad', type=float, default=-1.0,
                        help='ext-j2: see --i-target-min-rad.')
    parser.add_argument('--raan-target-sample', type=int, default=0,
                        help='ext-j2: 1 = Omega_t = raan_target_rad + U(0, 2pi) per episode. '
                             'GAUGE under J2 (axisymmetric about z), so this is the '
                             'SO(2)-about-z leak detector, not task variation.')
    parser.add_argument('--lvlh-frame-mode', type=int, default=0,
                        help='ext-j2: obs[33-36] frame. 0 = legacy (equatorial projection '
                             'rotated by omega+theta; correct ONLY at i_t = Omega_t = 0). '
                             '1 = true target orbital frame. Default 0 keeps every trained '
                             "checkpoint's primary rendezvous channel bit-identical.")
    parser.add_argument('--di-min-rad', type=float, default=-1.0,
                        help='ext-j2wait: delta ~ U(di_min, di_max) uniform in angle. '
                             '<0 = legacy area-uniform sqrt(U) draw.')
    parser.add_argument('--di-phase-mode', type=int, default=0,
                        help='ext-j2wait: 1 = node-dominant plane error (axis within '
                             '30 deg of the node axis, >=86.6%% drift-correctable).')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.debris, args.out_dir, args.seed,
             args.e_max_target, args.init_phase_gap_max, args.e_max_sat, args.same_orbit_init,
             args.enable_action_mask, args.valid_init_only,
             args.e_target_fixed, args.e_sat_fixed, args.phase_gap_fixed, args.omega_offset_fixed,
             args.a_min_override, args.a_max_override, args.log_validation_debug,
             args.max_valid_init_attempts, args.gave_up_action,
             args.obs_alt_scale_m, args.phi_orbit_scale_k,
             args.lvlh_scale_m, args.legacy_action_space, args.stochastic,
             args.rendezvous_radius_m, args.rel_vel_tol_ms,
             args.shaping_mode, args.shape_w_lambda, args.shape_w_match,
             args.shape_dv_ref_ms, args.shape_gamma,
             args.phase_gap_mode, args.phase_obs_mode, args.episode_cap_steps,
             args.cap_terminal_reward, args.de_max, args.da_max_m,
             args.dim3_mode, args.di_max_rad, args.i_target_rad, args.raan_target_rad,
             args.obs_di_scale_rad, args.obs_de_scale, args.shape_match_squash,
             args.j2_mode, args.i_target_min_rad, args.i_target_max_rad,
             args.raan_target_sample, args.lvlh_frame_mode,
             args.di_min_rad, args.di_phase_mode)


if __name__ == '__main__':
    main()
