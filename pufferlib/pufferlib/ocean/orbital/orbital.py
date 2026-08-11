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
    # ── ext-3d columns, APPENDED so every pre-existing index is unchanged ──
    + ['sat_z', 'sat_vz', 'sat_inc', 'sat_raan',
       'target_z', 'target_vz', 'target_inc', 'target_raan',
       # post-impulse / pre-propagation chaser state of the burn sub-step
       # (zeros on coast/warp rows) — what the 3D invariant battery needs to
       # test I7/I8/I9/I11/I15 against the env rather than a reconstruction.
       'burn_post_x', 'burn_post_y', 'burn_post_z',
       'burn_post_vx', 'burn_post_vy', 'burn_post_vz']
)
TRAJ_FLOATS = len(TRAJ_COLS)  # 100


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
        # T1 terminal-criterion tightening: runtime success box. Defaults preserve the
        # historical 30 km / 50 m/s far-field criterion. Affects only the termination
        # check; shaping normalizers keep the historical constants.
        rendezvous_radius_m=30000.0,
        rel_vel_tol_ms=50.0,
        # ── T3 corrected-dynamics recovery kwargs (2026-08-11) ─────────────────
        # Defaults preserve legacy behavior bit-exactly. See T3_RECOVERY_CAMPAIGN.md §5.
        # shaping_mode 1 = S-R3 "phase-time" potential: Φ = −[w_λ·|Δλ|/π +
        # w_m·min(1, Δv_match/dv_ref)] on MEAN longitude λ = M+ω, no gates.
        shaping_mode=0,
        shape_w_lambda=1.0,
        shape_w_match=0.35,
        shape_dv_ref_ms=300.0,
        # Shaping discount base; >= 1.0 → γ_shape = 1 exactly (kills the
        # (1−γ^τ)|Φ| do-nothing income, measured +1.78/ep under legacy γ^τ).
        shape_gamma=0.995,
        # 1 → init_phase_gap_max / phase_gap_fixed control the PHYSICAL
        # mean-longitude gap (legacy M-offset is inert at e>0 — recon ANOM-4).
        phase_gap_mode=0,
        # 1 → obs[13-16] use mean longitude (sign-correct, burn-continuous)
        # instead of true anomaly. BREAKING for pre-T3 checkpoints.
        phase_obs_mode=0,
        # Runtime safety cap in 60 s sim sub-steps (wall clock = cap minutes).
        # Legacy 2000 = 33.3 h. T3 recovery 3000. Wide-envelope 6000. MEO 12000 (max).
        episode_cap_steps=2000,
        # Reward at the safety-cap terminal. Legacy −10. T3 recovery uses 0.0:
        # under flat per-decision γ a −10 timeout prices warp-heavy play as a
        # −7.8 bet vs −0.0 for coast-to-cap, so PPO suppresses warps before the
        # first success is sampled (red-team #1, the flatline mechanism).
        cap_terminal_reward=-10.0,
        # ── T3 wide-eccentricity ladder (L3+) sampler kwargs ────────────────────
        # de_max >= 0: sat e-vector = target e-vector + area-uniform disc(de_max).
        # Bounds the e-vector MISMATCH instead of e itself — with independent ω,
        # matching e-vectors at e_max=0.15 alone averages 396 m/s of the 478 m/s
        # budget; bounding |Δē| keeps wide-e affordable (recon feasibility §3.4).
        # Overrides e_max_sat sampling (not e_sat_fixed / same_orbit_init).
        de_max=-1.0,
        # da_max_m > 0: |a_target − a_sat| ≤ da_max_m (∩ altitude band; min 200 km).
        # Required when widening the band to unlock e, else Δv_a is unaffordable.
        da_max_m=-1.0,
        # ── ext-3d (2026-08-11). Defaults preserve the 2D lineage BIT-EXACTLY.
        # Binding spec: scripts/orbital/ext_recon/reports/3d_{A,B,C,E}*.md, as
        # amended by 3d_REDTEAM.md (amendments win on every conflict).
        # dim3_mode 1 enables: 3D element/plane state, the 3D obs block in the
        # dead body slots 21-32, shaping_mode 2, the relative-plane sampler and
        # the 3D form of the phase-gap knob. Requires num_debris = 0 (the 3D obs
        # block writes over body slots 1-3).
        dim3_mode=0,
        # Relative-inclination knob (rad). >= 0 turns the sampler on:
        #   ĥ_sat = R(δ, n̂)·ĥ_target,  δ = di_max_rad·√U,  n̂ uniform in the
        # target plane — the *relative* plane bound, the de_max pattern lifted
        # to the plane. Realized Δi_rel max/knob = 1.000, 0.0% over, at every
        # target inclination. Δv cost ≈ 133 m/s per degree at LEO, so on the
        # 478 m/s budget the whole envelope is a few degrees.
        di_max_rad=-1.0,
        # Absolute target plane. Pure GAUGE under two-body dynamics (the env is
        # SO(3)-invariant), so 0 by default; these exist as test hooks for the
        # sampler and frame-invariance gates.
        i_target_rad=0.0,
        raan_target_rad=0.0,
        # obs normalizers for the 3D block. <= 0 → auto:
        #   obs_di_scale_rad → max(di_max_rad, 0.25°),  obs_de_scale → max(de_max, 0.05)
        obs_di_scale_rad=-1.0,
        obs_de_scale=-1.0,
        # Φ match-term squash: 0 = min(1, x) (legacy; required for the A2
        # bit-exact anchor), 1 = x/(1+x) (bounded, strictly monotone, no dead zone).
        shape_match_squash=0,
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
        # M2/M3 (phase5-5-env-mods) + T4 (Discrete-20): the C env accepts actions
        # in [0, 20) — 16-17 are 3h/6h warps, 18-19 radial ±1 m/s. The EXPOSED
        # space defaults to Discrete(16) so every pre-T4 checkpoint and command
        # is untouched; new lineages opt in with legacy_action_space=20. Phase
        # 5b/5e ckpts (10-dim heads) still use legacy_action_space=10.
        # ext-3d raises the C env to Discrete(30): 20-25 = normal ±{1,10,25} m/s,
        # 26-29 = combined tangential+normal {±25, 0, ±25}. The EXPOSED default
        # stays Discrete(16) so both 2D anchors and every existing command are
        # untouched; 3D lineages opt in with legacy_action_space=30.
        _las = -1 if legacy_action_space is None else int(legacy_action_space)
        _act_n = 16 if _las <= 0 else _las
        if not (1 <= _act_n <= 30):
            raise ValueError(f"legacy_action_space must be in [1, 30] or sentinel <=0, got {_las}")
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
            # Red-team #4: per-sub-step trajectory recording (352 B/record,
            # ~1 MB/env buffer) only when trajectories are actually saved.
            log_enabled=1 if traj_log_dir else 0,
        )

        # Pre-allocated trajectory buffer (reused every call).
        # Row count MUST equal MAX_STEPS in orbital.h (MEO lineage: 12000) —
        # vec_get_trajectory can fill up to that many records.
        self._traj_buf = np.zeros((12000, TRAJ_FLOATS), dtype=np.float32)

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
        # vec_get_trajectory returns the number of valid records, which now
        # includes the terminal record (steps+1, capped at MAX_STEPS). The old
        # count dropped the terminal record, so the last row of every exported
        # trajectory was the state 60 s before the episode actually ended.
        records = binding.vec_get_trajectory(self.c_envs, env_idx, self._traj_buf)
        if records <= 0:
            return

        data = self._traj_buf[:records]
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

        # Outcome metadata: sim-step count and terminal-cause code, so
        # downstream analysis can classify success without relying on the
        # terminal reward's sign (Φ-clamp leak). Cause codes: 0 none,
        # 1 success, 2 collision, 3 escape, 4 safety_cap, 5 stranded,
        # 6 hyperbolic, 7 gave_up_init.
        sim_steps, cause = binding.vec_get_episode_result(self.c_envs, env_idx)
        arrays['episode_steps']  = np.array([int(sim_steps)])
        arrays['terminal_cause'] = np.array([int(cause)])

        np.savez_compressed(path, **arrays)

    def last_episode_result(self, env_idx=0):
        """(sim_steps, terminal_cause) of env_idx's most recent completed episode.

        Cause codes: 0 none, 1 success, 2 collision, 3 escape, 4 safety_cap,
        5 stranded, 6 hyperbolic, 7 gave_up_init. Use cause == 1 as the success
        classifier — the terminal reward's sign is unreliable at wide altitude
        bands (Φ-clamp leak).
        """
        return binding.vec_get_episode_result(self.c_envs, env_idx)

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
