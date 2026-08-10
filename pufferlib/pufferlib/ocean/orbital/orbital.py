"""Orbital maneuver RL environment — PufferLib Ocean wrapper.

Single satellite performs a fuel-optimal rendezvous with a propagated
target body, avoiding debris.

Observation space: Box(float32, 38) — normalised to roughly [-1, 1] (33 legacy + 5 LVLH)
Action space:      Discrete(16)     — Phase 4-5b legacy 0-9 (coast + ±{5,10,25} m/s prograde
                                       + ±10 m/s radial + warp 5min) plus M2 warps (10:30min, 11:1hr)
                                       and M3 fine burns (12-15: ±{1,2} m/s prograde).
"""

import os
import time
import numpy as np
import gymnasium
import pufferlib

from pufferlib.ocean.orbital import binding

# Trajectory column names — matches fill_traj_row() in binding.c
TRAJ_COLS = (
    ['sim_time', 'sat_x', 'sat_y', 'sat_vx', 'sat_vy',
     'sat_a', 'sat_e', 'sat_theta', 'sat_omega',
     'fuel', 'action', 'reward', 'delta_v', 'min_conj_dist',
     'target_a', 'target_e', 'target_omega',
     'target_x', 'target_y', 'target_vx', 'target_vy',
     'num_bodies']
    + [f'body_x_{i}' for i in range(16)]
    + [f'body_y_{i}' for i in range(16)]
    + [f'body_hard_r_{i}' for i in range(16)]
    + [f'body_keepout_r_{i}' for i in range(16)]
)
TRAJ_FLOATS = len(TRAJ_COLS)  # 86


class Orbital(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        # Debris config: set both to 0 for no-debris curriculum
        num_debris_min=4,
        num_debris_max=8,
        # Target eccentricity curriculum (0.0 = circular target)
        e_max_target=0.0,
        # Phase-3 rendezvous phase-gap curriculum (rad). 0.0 = target co-phased with sat.
        init_phase_gap_max=0.0,
        # Phase 5b: Satellite eccentricity (0.0 = circular chaser, Phase 4 default)
        e_max_sat=0.0,
        # Phase 5b Stage 1: force sat.{a,e,ω} = target.{a,e,ω}, only θ differs
        same_orbit_init=0,
        # Phase 5c B3: mixed-distribution training. With probability e_mix_easy_frac,
        # sample target.e (and sat.e if random) from [0, e_mix_easy_max] instead of
        # the full curriculum bound. Default 0.0 = off.
        e_mix_easy_frac=0.0,
        e_mix_easy_max=0.05,
        # Phase 5d I4: soft collision-prevention penalty. Subtracts this weight from
        # step reward when a burn places sat on reentry trajectory (perigee < EARTH_KEEPOUT).
        collision_penalty_w=0.0,
        # Phase 5d I2: hard action masking. When 1, obs is 48-dim (last 10 = action mask).
        # Coast & warp always valid; burns masked when post-burn perigee < EARTH_KEEPOUT.
        # NOTE post phase5-5-env-mods (M2/M3): the mask is still 10-dim covering only
        # legacy actions 0-9. New actions 10-15 (warps + sub-5 m/s burns) are not masked.
        # Extend fill_observations() if action masking with the new actions is needed.
        enable_action_mask=0,
        # Phase 5d: rejection-sample sat & target init until both perigees >= EARTH_KEEPOUT.
        # Without this, ~64% of e_max=0.20 inits are physically doomed (sub-surface perigee).
        valid_init_only=0,
        # Phase 5 wrap-up W1: fixed-value sampling for per-condition surface eval.
        # Each defaults to a sentinel that preserves uniform sampling.
        e_target_fixed=-1.0,
        e_sat_fixed=-1.0,
        phase_gap_fixed=-1.0,
        omega_offset_fixed=-99.0,
        a_min_override=-1.0,   # < R_EARTH → use 300km altitude floor
        a_max_override=-1.0,   # if a_min_override is set, must be > a_min_override
        # Phase 5 verification I1: when 1, c_reset emits a per-reset debug line on stderr.
        log_validation_debug=0,
        # Phase 5 env-fix F1: configurable cap for rejection-sampling loop (was hardcoded 256).
        # At 4096 with e_max=0.70 LEO (~12% per-attempt acceptance), exhaust prob ≈ 0.88^4096 ≈ 0.
        max_valid_init_attempts=4096,
        # Phase 5 env-fix F3: behavior when the cap exhausts and accepted init is still doomed.
        # "accept" (default, legacy) — keep the doomed init, episode plays out and likely fails.
        # "terminate" — emit a single terminal step with reward 0, no learning signal from doomed inits.
        gave_up_action="accept",
        # Phase 5.5 altitude expansion: observation altitude normalization scale (meters above R_EARTH).
        # Default 1.6e6 = ALT_MAX preserves Phase 5b/5e checkpoint compatibility.
        # Set to ~4.2e7 (GEO altitude) for training on full Earth-orbit envelope.
        obs_alt_scale_m=1.6e6,
        # Phase 5.5: Φ_orbit scale gain. Effective orbit-match tolerance is
        # max(SUCCESS_TOL_A, phi_orbit_scale_k * obs_alt_scale_m). With K=0.001
        # at LEO obs_alt_scale_m=1.6e6, max(10km, 1.6km)=10km → backward compat.
        # At GEO obs_alt_scale_m=4.2e7, max(10km, 42km)=42km → keeps Φ in O(1-10).
        phi_orbit_scale_k=0.001,
        # M1 (phase5-5-env-mods): LVLH spatial obs normalizer (obs[33-34]).
        # Default R_EARTH = 6.371e6 preserves Phase 5b/5e LEO behavior exactly.
        # Set ~4.2e7 for GEO training; otherwise obs[33] saturates at ~20 at GEO.
        lvlh_scale_m=6.371e6,
        # Phase 5.5 Stage 5.5.1: coerce single_action_space to a smaller Discrete(N) so
        # a Phase 5b/5e 10-head ckpt can warm-start a Discrete(16) env without surgery.
        # c_step still accepts any int in [0, 16). Default None = no coercion (full Discrete(16)).
        legacy_action_space=None,
        # Trajectory logging
        traj_log_dir=None,   # if set, save .npz files here
        traj_log_every=500,  # save trajectory every N episodes (per env 0)
        buf=None,
        seed=0,
    ):
        # Encode gave_up_action enum (unpack() in C only accepts int/float).
        _gave_up_action_int = {"accept": 0, "terminate": 1}.get(gave_up_action, None)
        if _gave_up_action_int is None:
            raise ValueError(f"gave_up_action must be 'accept' or 'terminate', got {gave_up_action!r}")
        obs_dim = 48 if enable_action_mask else 38
        self.action_mask_dim = 10 if enable_action_mask else 0
        self.single_observation_space = gymnasium.spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        # M2/M3 (phase5-5-env-mods): extended action space. Phase 5b/5e ckpts have
        # a 10-dim policy head; eval_checkpoint.py uses --legacy-action-space 10 to
        # coerce the policy's view of the action space for backward compat.
        _las = -1 if legacy_action_space is None else int(legacy_action_space)
        _act_n = 16 if _las <= 0 else _las
        if not (1 <= _act_n <= 16):
            raise ValueError(f"legacy_action_space must be in [1, 16] or sentinel <=0, got {_las}")
        self.single_action_space = gymnasium.spaces.Discrete(_act_n)
        self.render_mode  = render_mode
        self.num_agents   = num_envs
        self.log_interval = log_interval

        self.traj_log_dir   = traj_log_dir
        self.traj_log_every = traj_log_every
        self._episode_count = 0  # global episode counter (env 0 only)

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            num_envs, seed,
            num_debris_min=num_debris_min,
            num_debris_max=num_debris_max,
            e_max_target=e_max_target,
            init_phase_gap_max=init_phase_gap_max,
            e_max_sat=e_max_sat,
            same_orbit_init=same_orbit_init,
            e_mix_easy_frac=e_mix_easy_frac,
            e_mix_easy_max=e_mix_easy_max,
            collision_penalty_w=collision_penalty_w,
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
            gave_up_action=_gave_up_action_int,
            obs_alt_scale_m=obs_alt_scale_m,
            phi_orbit_scale_k=phi_orbit_scale_k,
            lvlh_scale_m=lvlh_scale_m,
        )

        # Pre-allocated trajectory buffer (reused every call)
        self._traj_buf = np.zeros((2000, TRAJ_FLOATS), dtype=np.float32)

        if traj_log_dir:
            os.makedirs(traj_log_dir, exist_ok=True)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log.get("n", 0) > 0:
                info.append(log)

        # Trajectory logging: when env 0 reaches a terminal, save its episode.
        # Must be called before the next step (which would overwrite traj_log).
        if self.traj_log_dir and self.terminals[0]:
            self._episode_count += 1
            if self._episode_count % self.traj_log_every == 0:
                self._save_trajectory(env_idx=0, episode_reward=float(self.rewards[0]))

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def _save_trajectory(self, env_idx=0, episode_reward=0.0):
        """Copy traj_log from C, save to .npz."""
        steps = binding.vec_get_trajectory(self.c_envs, env_idx, self._traj_buf)
        if steps <= 0:
            return

        data = self._traj_buf[:steps]  # slice to actual episode length
        ep_id = int(self._episode_count)
        reward = episode_reward
        path = os.path.join(self.traj_log_dir, f"ep_{ep_id:07d}.npz")

        # Save each column separately for easy numpy access
        arrays = {col: data[:, i] for i, col in enumerate(TRAJ_COLS)}
        arrays['episode_id']     = np.array([ep_id])
        arrays['episode_reward'] = np.array([reward])
        arrays['col_names']      = np.array(TRAJ_COLS)

        # Phase 5 env-fix F2: realized-init outcome metadata (per-episode scalars).
        # `last_init_attempts` = how many rejection-sampling iterations c_reset took.
        # `last_init_gave_up`  = 1 if cap exhausted with sub-keepout perigee accepted.
        # Realized perigees are computed downstream from sat_a[0]/sat_e[0] etc.
        attempts, gave_up = binding.vec_get_episode_init_info(self.c_envs, env_idx)
        arrays['last_init_attempts'] = np.array([int(attempts)])
        arrays['last_init_gave_up']  = np.array([int(gave_up)])

        np.savez_compressed(path, **arrays)

    def enable_logging(self, env_idx=None):
        """Enable C-side trajectory logging (call before training/eval runs)."""
        binding.env_put(
            binding.vectorize(*[
                binding.env_init.__self__ if False else h
                for h in []
            ])
        )
        # Simpler: just set via vec_put if available, or accept it's always on
        # for now. In practice, log_enabled defaults to 0 and we control
        # saving from the Python terminal check above.

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == "__main__":
    """Quick SPS benchmark — run with: python orbital.py"""
    N = 1024
    env = Orbital(num_envs=N)
    env.reset()

    CACHE = 256
    actions = np.random.randint(0, 7, (CACHE, N))

    steps = 0
    i = 0
    start = time.time()
    while time.time() - start < 5.0:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    sps = int(steps / (time.time() - start))
    print(f"Orbital SPS: {sps:,}  ({N} envs, 5s)")
    env.close()
