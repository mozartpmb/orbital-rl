"""OrbitalNav — filter-in-the-loop training for the orbital rendezvous agent.

`puffer_orbital` is guidance: the policy sees the target's true state. This
subclass makes it GN&C. Between the C environment and the policy it inserts a
measurement model and a live recursive filter, so the policy flies on an
*estimate* whose covariance responds to its own maneuvers.

Nothing in C changes. `puffer_orbital` stays byte-identical; this is a separate
package, a separate `.ini`, and one additive line in `MAKE_FUNCTIONS`.

── Where the work happens (NAV-H §1, verified empirically) ──────────────────

`vector.py:190` has the *worker* construct the env class, `pufferlib.py:38`
binds `self.observations` to the shared-memory view the C env writes into, and
`vector.py:213-224` does not release the worker's semaphore until `step()`
RETURNS. So a subclass that mutates `self.observations[...]` in place after
`binding.vec_step()` inside `step()` is (a) visible to the driver, (b) able to
hold per-env filter state in the worker, and (c) race-free by construction.
Verified: `ext_nav_wrapper_probe.py` — worker-side in-place obs write visible
to driver PASS, worker-side per-env state persists PASS.

── Modes ────────────────────────────────────────────────────────────────────

  nav_mode='truth'          passthrough. MUST be byte-identical to Orbital.
  nav_mode='recon'          rebuild the target obs slots from the DECODED truth
                            (zero estimation error). Isolates the encode/decode
                            layer from the filter; must reproduce truth exactly.
  nav_mode='rb_ekf'         range + bearing EKF (the T2/T4 sensor suite),
                            `nav_noise_mult` x the nominal 50 m / 1 mrad.
  nav_mode='bearings_only'  angles only: calibrated acquisition surrogate
                            (nav_surrogate) handing off to a live recursive
                            modified-polar filter.

── Design decisions that are red-team dispositions, not preferences ─────────

* **No blind-window hold, no forced actions, no nav-valid bit** (MAJOR-3). A
  hold buys 0.0 pp (100/100 at every blind depth to 4 decisions) and corrupts
  5-90% of the PPO rollout, because `pufferl.py:295` stores the policy's
  *sampled* action, not the one the env executed. The policy always flies the
  current estimate, acquired or not.
* **Close-range covariance is the gate, not acquisition latency** (MAJOR-4).
  Blindness at episode start is free; blindness inside 200 km scores 45/100 and
  inside 1000 km scores 18/100, all by fuel exhaustion. `nav_sigma_los_over_rho`
  is therefore a first-class training diagnostic.
* **Navigation runs at a fixed 60 s cadence, decoupled from the decision
  cadence.** The T4 finding is that welding the sensor to the decision rate
  (the policy warps 1 h 51% of the time) costs 100% -> 50.5% closed loop. Warps
  are sub-propagated: they apply no impulse, so both truth states simply coast
  under the same exact Kepler map the env sub-steps with.
* **Reward and termination stay on truth.** That is privileged information at
  training time (asymmetric actor-critic), standard for sim-trained GN&C, and
  absent at deployment. It is also not an exploitable leak: a permanently blind
  policy scores 1/100. Potential-based shaping is *not* policy-invariant under
  partial observability; the unexplained component measures 7.9% rms of the
  shaping delta at angles-only accuracy, against a shaping total bounded at
  24.7% of the terminal reward (MINOR-7).

── Diagnostics ──────────────────────────────────────────────────────────────

Appended as the wrapper's OWN info dict every `log_interval` steps, never
assuming `info[0]` exists (`Orbital.step` emits at most one dict and only when
`log["n"] > 0`), and always as plain Python floats (`pufferl.py:316-317` has a
dead-store bug that mangles ndarray values).
"""

import numpy as np

from pufferlib.ocean.orbital.orbital import Orbital
from . import nav_math as nm
from . import nav_math3d as n3
from . import nav_encode3d as ne
from .nav_surrogate import AcquisitionSurrogate, ACQ_MIN_SEC

NAV_MODES = ('truth', 'recon', 'rb_ekf', 'bearings_only')

# Divergence thresholds. `a_est <= 0` is the hyperbolic estimate; the trace
# bound is 4 decades above the widest legitimate prior (a 13,000 km separation
# with a 20% range gate is a 2.6e6 m sigma => 7e12 m^2 of trace).
DIVERGE_TRACE_M2 = 1.0e16
RHO_MIN_M, RHO_MAX_M = 1.0, 1.0e10


class OrbitalNav(Orbital):
    """Orbital with a live navigation filter between the env and the policy."""

    def __init__(self,
                 nav_mode='rb_ekf',
                 # Measurement noise multiplier on the nominal suite
                 # (sigma_rho = 50 m, sigma_beta = 1 mrad).
                 nav_noise_mult=1.0,
                 # Explicit nav RNG seed. NOT `[vec] seed`: `vector.make()`
                 # never forwards it (vector.py:618,708), so the configured
                 # value is silently dropped and every worker would share a
                 # stream derived from 0.
                 nav_seed=20260811,
                 # Sensor sampling period (s). Fixed at 60 s = the env's own
                 # sub-step; exposed so the sensor-per-decision regression can
                 # be reproduced, not because it should be changed.
                 nav_sensor_dt=60.0,
                 # Process-noise PSD (covariance floor; the dynamics are exact).
                 nav_q_a=-1.0,          # <0 = mode default
                 nav_sigma_v0=nm.SIGMA_V0,
                 # Bearings-only acquisition surrogate.
                 nav_acq_mode='crlb_online',   # or 'table'
                 nav_acq_gate=0.20,
                 # n3d_REDTEAM BLOCKER-2. The acquisition floor is SIM TIME,
                 # never an observation count: every cost lever divides
                 # observations, so a tick floor and a cadence knob are the
                 # same knob, and composing them silently multiplies the
                 # campaign's headline metric. `nav_acq_min_ticks` survives
                 # only as a tripwire — setting it > 0 RAISES rather than
                 # silently re-introducing the unit.
                 nav_acq_min_sec=ACQ_MIN_SEC,       # 2700 s = 45 min
                 nav_acq_min_ticks=-1,
                 nav_cov_inflate=4.0,
                 # Altitude annulus for the analytic blind range prior. <0
                 # derives from the env's own sampler bounds.
                 nav_r_min_m=-1.0,
                 nav_r_max_m=-1.0,
                 # Saturation bound on the rebuilt slots. NAV-H §3.3 asks for
                 # +/-2 (the declared Box) so a diverged estimate cannot feed
                 # unbounded values to the value head for thousands of
                 # consecutive steps. Measured objection: the C env's OWN
                 # obs[33] (LVLH downrange) reaches 2.34 under the T3 LEO
                 # config — the env already exceeds its declared Box — so a
                 # +/-2 clip would cut legitimate truth and break the `recon`
                 # == `truth` gate that catches scale-constant errors. 4.0 is
                 # 1.7x the measured physical envelope and still 3 decades
                 # below where a diverged LVLH slot lands.
                 nav_clip=4.0,
                 # NEES is 8 encodes + a 4x4 inverse per row; sample it.
                 nav_nees_every=8,
                 # Hard cap on filter ticks inside one env.step(). 0 = uncapped.
                 # NAV-H §2.4: actions 16/17 are tau=180/360, i.e. 360 filter
                 # ticks in a single step() for the MEO lineage. Discrete(16)
                 # tops out at 60, so this is inert for every shipped config.
                 nav_max_ticks=0,
                 # ── NAV-F dual-control campaign ─────────────────────────────
                 # T-BO+Sigma: expose the filter's OWN range uncertainty to the
                 # policy on obs[21]. NAV-F §3.4 calls this the arm that makes a
                 # null interpretable: the 38-dim layout has no uncertainty
                 # channel, so without it the agent must infer nav uncertainty
                 # from estimate jitter through the LSTM, and a null cannot be
                 # separated from "it never saw the signal".
                 #
                 # obs[21..32] are identically zero in every nav/T3 config
                 # (num_debris = 0 hard-zeroes three body blocks; measured
                 # min = max = 0.00000 over 836 canonical decisions). The slot
                 # is free, but NOT free of consequence: those weights got zero
                 # gradient and sit at random init, so red-team NON-ISSUE-9
                 # measured 0/836 argmax flips at magnitude 0.1 rising to
                 # 26/836 (3.11%) at 1.0. Hence the 0.1 ceiling here AND zeroing
                 # the encoder's column 21 at warm-start load
                 # (scripts/orbital/nav/zero_obs_column.py), which makes the
                 # channel provably inert at t=0 and learnable thereafter.
                 nav_sigma_channel=0,
                 nav_sigma_ref=1e-6,      # sigma_LOS/rho mapping to 0 (log floor)
                 nav_sigma_gain=0.1,      # output magnitude ceiling
                 # T-BO-act necessity ablation: inside this separation the fine
                 # burns (12-15 = +/-1, +/-2 m/s prograde; 18-19 = radial
                 # +/-1 m/s) execute as COAST.
                 #
                 # This is an env-variant DYNAMICS change, not a forced-action
                 # substitution. PPO stores the action the policy sampled
                 # (`pufferl.py:295` writes into its own experience buffer and
                 # never reads back the env's shm), and the environment simply
                 # makes those actions no-ops in that region — a wall in the
                 # MDP, which is a perfectly well-defined transition function.
                 # Red-team MAJOR-3's objection was to overriding the agent's
                 # choice with a DIFFERENT action while storing the original;
                 # here the stored action is the executed action, and only the
                 # transition it induces has changed.
                 nav_block_fine_below_m=0.0,
                 # MAJOR-10: WHICH action set the ablation blocks. `_FINE` was
                 # the in-plane fine burns only, which under Discrete-30 leaves
                 # actions 20/21 (normal +/-1) — the 3D observability treatment
                 # — wide open, making a `T-BO-act` arm uninterpretable. Every
                 # arm must now state its set by name.
                 nav_block_set='fine_inplane',
                 # ── n3d_REDTEAM MAJOR-10: the CAPABILITY ablation. ──────────
                 # `nav_block_fine_below_m` is a NAVIGATION interlock: it gates
                 # on separation AND is skipped entirely in truth mode
                 # (`step()`: `and self._nav_mode != 'truth'`), because "don't
                 # make fine burns when you cannot see" is meaningless when you
                 # can always see. That makes it the wrong tool for MAJOR-10,
                 # which requires the SAME rows removed in bearings-only AND in
                 # truth — the truth control is the entire point of the fix
                 # ("a T-BO-normal arm therefore needs a truth control at the
                 # same box, or the plane leg's guidance cost will be misread
                 # as an information result").
                 #
                 # `nav_ablate_rows` is therefore separate and unconditional:
                 # the named ACTION_SETS rows coast in EVERY nav mode, at every
                 # separation. '' = off, and off is bitwise-inert.
                 nav_ablate_rows='',
                 # ── ext-j2: secular J2 in the FILTER ────────────────────────
                 # Default OFF and bit-inert when off: the filter class is only
                 # swapped when this is 1, so every published nav number keeps
                 # its action stream. Turning it on costs 217% of a baseline
                 # nav step (predict 0.603 -> 3.072 ms/tick, 5.09x) and is the
                 # only arm measured consistent under a J2 truth — see
                 # nav_math3d.BatchedBearingMSC6J2 for the head-to-head.
                 #
                 # This tracks the ENV's j2_mode; it does not imply it. Running
                 # nav_j2_mode=1 against a two-body env models a perturbation
                 # that is not there, and nav_j2_mode=0 against a j2_mode=1 env
                 # is the FIXED arm the head-to-head rejected (NEES 11.4,
                 # 2.63x position at 24 h). The constructor refuses the second.
                 nav_j2_mode=0,
                 **kwargs):
        if nav_mode not in NAV_MODES:
            raise ValueError(f'nav_mode must be one of {NAV_MODES}, got {nav_mode!r}')
        self._nav_mode = nav_mode
        self._nav_noise = float(nav_noise_mult)
        self._nav_seed = int(nav_seed)
        self._nav_dt = float(nav_sensor_dt)
        self._nav_q_a = float(nav_q_a)
        self._nav_sigma_v0 = float(nav_sigma_v0)
        self._acq_mode = nav_acq_mode
        self._acq_gate = float(nav_acq_gate)
        if int(nav_acq_min_ticks) > 0:
            raise ValueError(
                'nav_acq_min_ticks is retired (n3d_REDTEAM BLOCKER-2): the '
                'acquisition floor is sim TIME, because every cost lever '
                'divides ticks and the two are the same knob. Set '
                f'nav_acq_min_sec instead (currently {float(nav_acq_min_sec)} s).')
        self._acq_min_sec = float(nav_acq_min_sec)
        self._cov_inflate = float(nav_cov_inflate)
        self._nav_clip = float(nav_clip)
        self._sig_ch = int(nav_sigma_channel)
        self._sig_ref = float(nav_sigma_ref)
        self._sig_gain = float(nav_sigma_gain)
        self._block_below = float(nav_block_fine_below_m)
        if nav_block_set not in nm.ACTION_SETS:
            raise ValueError(f'nav_block_set must be one of '
                             f'{sorted(nm.ACTION_SETS)}, got {nav_block_set!r}')
        self._block_set_name = nav_block_set
        self._block_set = nm.ACTION_SETS[nav_block_set]
        # MAJOR-10 capability ablation, unconditional and nav-mode-independent.
        self._ablate_name = str(nav_ablate_rows or '')
        if self._ablate_name:
            if self._ablate_name not in nm.ACTION_SETS:
                raise ValueError(
                    f'nav_ablate_rows must be one of {sorted(nm.ACTION_SETS)} '
                    f'or "" for off, got {nav_ablate_rows!r}')
            self._ablate_rows = nm.ACTION_SETS[self._ablate_name]
        else:
            self._ablate_rows = None
        self._d_ablate = 0
        self._d_ablate_n = 0
        self._nees_every = max(1, int(nav_nees_every))
        self._nav_max_ticks = int(nav_max_ticks)

        self._alt_scale = float(kwargs.get('obs_alt_scale_m', 1.6e6))
        self._lvlh_scale = float(kwargs.get('lvlh_scale_m', 6.371e6))
        self._nav_j2 = int(nav_j2_mode)
        self._dim3 = int(kwargs.get('dim3_mode', 0))
        _env_j2 = int(kwargs.get('j2_mode', 0))
        if _env_j2 and not self._nav_j2 and nav_mode in ('rb_ekf', 'bearings_only'):
            raise ValueError(
                'j2_mode=1 with nav_j2_mode=0 is the FIXED arm the ext-j2 '
                'head-to-head rejected: a two-body covariance under a J2 truth '
                'leaves NEES 11.37 and pays 2.63x in position at 24 h, because '
                'the overconfident P shrinks the Kalman gain. Set '
                'nav_j2_mode=1, or run the env two-body.')
        if self._nav_j2 and not self._dim3:
            raise ValueError('nav_j2_mode=1 requires dim3_mode=1 '
                             '(the J2 filter is the 6-state MSC path).')
        self._di_max = float(kwargs.get('di_max_rad', -1.0))
        self._kw3 = {k: kwargs.get(k) for k in
                     ('obs_di_scale_rad', 'de_max', 'obs_de_scale',
                      'shape_dv_ref_ms') if kwargs.get(k) is not None}
        if self._dim3 and nav_mode != 'truth':
            # NOTE-20, asserted here rather than discovered in a wide rung.
            # `omega_offset_fixed > -10` sets target.omega = sat.omega + offset
            # EVEN AT e_t = 0, at which point obs[11,12,16] carry a value no
            # estimate can recover; `raan_target_rad != 0` makes target.omega
            # node-referenced while the estimate's is inertial. Both are
            # eval-only hooks today, and both break the recon gate for reasons
            # that have nothing to do with the port.
            if float(kwargs.get('omega_offset_fixed', -99.0)) > -10.0:
                raise ValueError(
                    'omega_offset_fixed is incompatible with dim3 nav '
                    '(n3d_REDTEAM NOTE-20): it sets target.omega from the '
                    'chaser even at e_t = 0, which no estimate can recover.')
            if float(kwargs.get('raan_target_rad', 0.0)) != 0.0:
                raise ValueError(
                    'raan_target_rad != 0 is incompatible with dim3 nav '
                    '(n3d_REDTEAM NOTE-20): target.omega becomes '
                    'node-referenced while the estimate-side varpi is '
                    'inertial.')

        a_min = kwargs.get('a_min_override', -1.0)
        a_max = kwargs.get('a_max_override', -1.0)
        a_min = a_min if a_min >= nm.R_EARTH else nm.R_EARTH + 300e3
        a_max = a_max if a_max > a_min else nm.R_EARTH + 800e3
        e_max = max(float(kwargs.get('e_max_target', 0.0)), 0.0)
        # The blind range prior's inner shell.
        #
        # KNOWN LOOSE, deliberately not fixed by default. `a_min*(1-e_max)` is
        # the smallest perigee the (a, e) box could produce, but the ENV
        # guarantees more: c_reset rejection-samples until both orbits have
        # perigee >= EARTH_KEEPOUT (6.571e6 m), so this prior admits target
        # radii the sampler has already ruled out — and at e_max = 0.30 with
        # a_min = 6.671e6 it admits 4.67e6 m, i.e. INSIDE THE EARTH. Flooring
        # it at the keepout narrows the log-range prior by 25% there for free.
        #
        # It is not floored by default because the default is not free: the
        # floor binds at e_max = 0.05 too (6.337e6 -> 6.571e6), so it would
        # change the blind seed, hence the observation, hence the action
        # stream, of the ALREADY-PUBLISHED rung-1 and tight-box bearings-only
        # results. Campaigns that want the tighter prior — the eccentricity
        # ladder does, and it is the only place it materially matters — set
        # `nav_r_min_m` explicitly. Worth promoting to the default at the next
        # deliberate re-baseline of those numbers, not silently.
        self._r_min = nav_r_min_m if nav_r_min_m > 0 else a_min * (1.0 - e_max)
        self._r_max = nav_r_max_m if nav_r_max_m > 0 else a_max * (1.0 + e_max)
        # Eccentricity-driven velocity-guess error (T4 §8.2: the circular-guess
        # error is ~v_c * e, not a function of range).
        self._sigma_v_ecc = max(7.7e3 * max(e_max, 0.02), 100.0)

        self._nav_ready = False
        super().__init__(**kwargs)
        self._ctor_seed = int(kwargs.get('seed', 0))

    # ── lazy allocation (vector.py:273 builds an unstepped driver_env) ───────
    def _nav_alloc(self):
        if self._nav_ready or self._nav_mode in ('truth',):
            self._nav_ready = True
            return
        n = self.num_agents
        d = 6 if self._dim3 else 4
        self._cdim = d
        self._rng = np.random.default_rng([self._nav_seed, self._ctor_seed])
        self._s_rho = nm.SIGMA_RHO_M * self._nav_noise
        self._s_beta = nm.SIGMA_BETA_RAD * self._nav_noise
        self._prev_sat = np.zeros((n, d))
        self._prev_tgt = np.zeros((n, d))
        self._slots = np.array(ne.TARGET_SLOTS_3D if self._dim3
                               else nm.TARGET_SLOTS_T3)
        self._scratch = np.zeros((n, nm.OBS_DIM), dtype=np.float64)
        self._pair = np.zeros((2 * n, d), dtype=np.float64)
        self._state_buf = np.zeros((n, self.STATE_FLOATS), dtype=np.float64)
        self._enc3 = None
        if self._dim3:
            self._enc3 = ne.Encoder3D(
                n, obs_alt_scale_m=self._alt_scale,
                lvlh_scale_m=self._lvlh_scale,
                di_max_rad=self._di_max,
                obs_di_scale_rad=float(self._kw3.get('obs_di_scale_rad', -1.0)),
                de_max=float(self._kw3.get('de_max', -1.0)),
                obs_de_scale=float(self._kw3.get('obs_de_scale', -1.0)),
                shape_dv_ref_ms=float(self._kw3.get('shape_dv_ref_ms', 300.0)),
                clip=self._nav_clip)

        if self._nav_mode == 'rb_ekf':
            q = self._nav_q_a if self._nav_q_a > 0 else nm.Q_ACCEL_PSD_RB
            if self._dim3:
                # N1-rb3d, the control arm. Same sensor as the bearings-only
                # arm plus one row, so the two differ in exactly one
                # measurement channel.
                self._filt = n3.BatchedRangeBearingEKF3D(
                    n, sigma_rho=self._s_rho, sigma_beta=self._s_beta,
                    q_a=q, sigma_v0=self._nav_sigma_v0)
            else:
                self._filt = nm.BatchedRangeBearingEKF(
                    n, sigma_rho=self._s_rho, sigma_beta=self._s_beta,
                    q_a=q, sigma_v0=self._nav_sigma_v0)
        elif self._nav_mode == 'bearings_only':
            q = self._nav_q_a if self._nav_q_a > 0 else nm.Q_ACCEL_PSD_BO
            if self._dim3 and self._nav_j2:
                self._filt = n3.BatchedBearingMSC6J2(
                    n, sigma_beta=self._s_beta, q_a=q, stm_j2='fd')
            elif self._dim3:
                self._filt = n3.BatchedBearingMSC6(
                    n, sigma_beta=self._s_beta, q_a=q, stm='analytic')
            else:
                self._filt = nm.BatchedBearingMPC(
                    n, sigma_beta=self._s_beta, q_a=q)
            self._acq = AcquisitionSurrogate(
                n, sigma_beta=self._s_beta, gate=self._acq_gate,
                min_sec=self._acq_min_sec, cov_inflate=self._cov_inflate,
                mode=self._acq_mode, dim=d)
        else:
            self._filt = None

        self._d_reset()
        self._nav_ready = True

    def _d_reset(self):
        self._d_n = 0
        self._d_sq = 0.0
        self._d_nees = []
        self._d_clip = 0
        self._d_clip_n = 0
        self._d_div = 0
        self._d_slr = []
        self._d_blind = 0
        self._d_sig = []
        self._d_block = 0
        self._d_block_n = 0
        self._d_feas_neg = 0            # NOTE-21: obs[28] < 0 while unacquired
        self._d_feas_n = 0

    # ── decode / re-encode ───────────────────────────────────────────────────
    def _decode(self):
        """(chaser elements, chaser Cartesian, target elements, target Cartesian).

        In 2D this is the obs-only decode, unchanged. Under dim3 it is the
        read-only C getter: the ext-3d observation block is SO(3)-invariant by
        construction, so the chaser's plane is ABSENT from the observation and
        no decoder can recover it (MAJOR-5). The getter also hands over each
        body's Cartesian state and element-route e-vector on the env's own FP
        path, which is what makes the recon gate exact.
        """
        if not self._dim3:
            o = self.observations
            sat, tgt = nm.recover_states_t3(o, self._alt_scale)
            sat_c = nm.orbit_to_cartesian(sat['a'], sat['e'], sat['theta'],
                                          sat['omega'])
            tgt_c = nm.orbit_to_cartesian(tgt['a'], tgt['e'], tgt['theta'],
                                          tgt['omega'])
            return sat, sat_c, tgt, tgt_c
        st = self.get_state(self._state_buf)
        sat = ne.Encoder3D.chaser(st)
        tgt = ne.Encoder3D.target_truth(st)
        return sat, sat['cart'], tgt, tgt['cart']

    def _di_rel_truth(self, sat, tgt):
        """MAJOR-17: realized Delta_i_rel from TRUTH, captured at the TOP of
        `_nav_step`.

        Conditioning the surrogate's noise process on truth is legitimate — it
        is exactly what the shipped 2D surrogate already does with realized
        separation and realized delta-v. The defect the red-team caught is the
        READ SITE: `_encode` overwrites obs[21,22] in place, so reading them
        afterwards conditions the error model on the estimate it just produced,
        a self-referential loop that raises nothing and produces a
        plausible-looking number.
        """
        if not self._dim3:
            return None
        hs, ht = sat['hhat'], tgt['hhat']
        c = np.linalg.norm(np.cross(ht, hs), axis=1)
        return np.arctan2(c, np.einsum('ni,ni->n', ht, hs))

    def _encode(self, sat, sat_c, est_x):
        """Rebuild the target-derived obs slots from the estimate, in place."""
        if self._dim3:
            n_clip, n_tot = self._enc3.write(self.observations,
                                             self._state_buf, est_x)
            self._d_clip += n_clip
            self._d_clip_n += n_tot
            if self._nav_mode == 'bearings_only':
                unacq = ~self._acq.acquired
                self._d_feas_neg += int(np.count_nonzero(
                    self._enc3.last_feas_neg & unacq))
                self._d_feas_n += int(np.count_nonzero(unacq))
            if self._sig_ch:
                self._write_sigma_channel(est_x, sat_c)
            return
        a, e, om, th = nm.cartesian_to_elements(est_x)
        # `fill_target_obs_t3` only WRITES the target slots, so a persistent
        # scratch buffer is enough — no need to copy the 38-wide observation
        # every step just to overwrite 12 of its columns.
        out = self._scratch
        nm.fill_target_obs_t3(out, sat, (a, e, om, th), est_x, sat_c,
                              self._alt_scale, self._lvlh_scale)
        if self._sig_ch:
            self._write_sigma_channel(est_x, sat_c)
        c = self._nav_clip
        v = out[:, self._slots]
        v = np.nan_to_num(v, nan=0.0, posinf=c, neginf=-c)
        self._d_clip += int(np.count_nonzero(np.abs(v) > c))
        self._d_clip_n += v.size
        np.clip(v, -c, c, out=v)
        self.observations[:, self._slots] = v.astype(np.float32)

    # ── T-BO+Sigma: the uncertainty channel ──────────────────────────────────
    def _sigma_los_over_rho(self, est_x, sat_c):  # noqa: D401
        """The filter's OWN 1-sigma range uncertainty, normalised by its OWN
        estimated separation. Everything here is estimator-side — no truth — so
        it is a legitimate observation."""
        n = self.num_agents
        idx = np.arange(n)
        p = self._filt.POS_DIM
        _, P = self._filt.mean_cov(idx)
        d = est_x[:, :p] - sat_c[:, :p]
        rho = np.maximum(np.sqrt(np.einsum('ni,ni->n', d, d)), 1.0)
        u = d / rho[:, None]
        with np.errstate(all='ignore'):
            s2 = np.einsum('ni,nij,nj->n', u, P[:, :p, :p], u)
            s = np.sqrt(np.maximum(s2, 0.0)) / rho
        return np.nan_to_num(s, nan=1.0, posinf=1.0, neginf=0.0)

    def _write_sigma_channel(self, est_x, sat_c):
        """obs[21] = gain * normalised log10(sigma_LOS / rho), bounded [0, gain].

        LOG, not linear. The quantity spans four decades over the regime that
        matters: NAV-F §2.6 measures sigma_range/rho ~ 5e-4 for a well-acquired
        filter and 0.52 for a drift-only filter at the TB5 box — and the whole
        dual-control tension lives in between, as the policy nulls delta-a and
        observability collapses. A linear map with any single reference either
        saturates across that band or resolves nothing at the bottom of it.
        """
        s = self._sigma_los_over_rho(est_x, sat_c)
        lo, hi = self._sig_ref, 1.0
        with np.errstate(all='ignore'):
            t = np.log10(np.maximum(s, 1e-12) / lo) / np.log10(hi / lo)
        v = self._sig_gain * np.clip(np.nan_to_num(t, nan=1.0), 0.0, 1.0)
        self._d_sig.append(float(np.median(s)))
        # MAJOR-14: under dim3 obs[21] is the ext-3d plane channel, not a free
        # slot. The Sigma channel is DROPPED from the 3D layout (NAV-F measured
        # T-BO+Sigma statistically identical to T-BO on every metric, so it has
        # no claim on a slot); if an arm wants it back it goes on obs[29],
        # which orbital.h hard-zeroes, and zero_obs_column.py retargets there.
        self.observations[:, 29 if self._dim3 else 21] = v.astype(np.float32)

    # ── T-BO-act: fine burns are no-ops inside the ablation radius ───────────
    def _apply_action_ablation(self, actions):
        """Env-variant dynamics: fine burns coast inside `nav_block_fine_below_m`.

        Uses the TRUE separation at the decision epoch (the state the action is
        taken in), which is env-side information, exactly as a real actuator
        interlock would be.
        """
        a = np.asarray(actions, dtype=np.int32).reshape(-1).copy()
        if not self._nav_ready or self._prev_tgt is None:
            return a
        # MAJOR-9: the separation is 3-D under dim3. A 2-component rho
        # under-reports it by the cross-track term (1.62 km at the 5 km box)
        # and fires the interlock in the wrong states.
        p = self._cdim // 2
        d = self._prev_tgt[:, :p] - self._prev_sat[:, :p]
        rho = np.sqrt(np.einsum('ni,ni->n', d, d))
        blocked = np.isin(a, self._block_set) & (rho < self._block_below)
        if blocked.any():
            a[blocked] = 0
            self._d_block += int(blocked.sum())
        self._d_block_n += a.size
        return a

    def _apply_row_ablation(self, actions):
        """MAJOR-10: named rows are UNAVAILABLE — they coast instead.

        Unconditional by design. No separation gate (this is a capability
        question, not a navigation interlock) and no nav_mode branch (the truth
        control must see the identical dynamics, or it is not a control).

        Rows 20/21 (normal +/-1 m/s) are tau=1 burns and coast is tau=1, so the
        substitution changes no timing and no sub-step bookkeeping.
        """
        a = np.asarray(actions, dtype=np.int32).reshape(-1).copy()
        hit = np.isin(a, self._ablate_rows)
        if hit.any():
            a[hit] = 0
            self._d_ablate += int(hit.sum())
        self._d_ablate_n += a.size
        return a

    # ── filter mean, cheaply ─────────────────────────────────────────────────
    def _mean(self, idx=None):
        """MAJOR-9: no `nav_mode` branch and no positional slice — the filter
        knows its own parametrisation and returns an inertial Cartesian mean of
        width `filt.CART_DIM` (4 in 2D, 6 in 3D)."""
        if idx is None:
            idx = np.arange(self.num_agents)
        return self._filt.mean_cart(idx)

    # ── episode boundaries ───────────────────────────────────────────────────
    def _init_rows(self, idx, sat_c, tgt_c, sat_el):
        """(Re)initialise the filter on `idx`. `sat_c`/`tgt_c` are subsets."""
        if idx.size == 0:
            return
        if self._dim3:
            return self._init_rows3(idx, sat_c, tgt_c, sat_el)
        d = tgt_c[:, :2] - sat_c[:, :2]
        rho_t = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1.0)
        beta_t = np.arctan2(d[:, 1], d[:, 0])
        beta = nm.wrap_pi(beta_t + self._rng.normal(0.0, self._s_beta, idx.size))

        if self._nav_mode == 'rb_ekf':
            rho = rho_t + self._rng.normal(0.0, self._s_rho, idx.size)
            self._filt.initialize(idx, sat_c, np.maximum(rho, 1.0), beta)
            return

        # Bearings-only: blind seed from the analytic range prior. The feasible
        # range set is the LOS ray intersected with the altitude annulus; its
        # geometric mean is the least-committal point of a set that spans up to
        # 4 decades (NAV-G §3.1: 0.1-4055 km at LEO). Seeded in MODIFIED POLAR
        # so the 4-decade ignorance lives entirely in ln rho, where it is
        # bounded, instead of being pushed through the encoding Jacobian.
        lo, hi = nm.range_prior_intervals(sat_c, beta, self._r_min, self._r_max)
        rho0 = np.sqrt(lo * hi)
        x0, _ = nm.ray_init(sat_c, beta, rho0, 0.3 * rho0, self._s_beta,
                            self._sigma_v_ecc)
        y0 = nm.msc_encode(sat_c, x0)
        sig_lnr = np.clip(np.log(hi / np.maximum(lo, 1.0)) / np.sqrt(12.0),
                          0.5, 4.0)
        # A circular-orbit velocity guess is wrong by ~v_c*e; at an unknown
        # range that is also an unknown angular rate, so both rate components
        # get sigma_v / rho0.
        sig_rate = np.clip(self._sigma_v_ecc / np.maximum(rho0, 1.0),
                           1e-5, 1e-1)
        Py0 = np.zeros((idx.size, 4, 4))
        Py0[:, 0, 0] = self._s_beta ** 2
        Py0[:, 1, 1] = sig_rate ** 2
        Py0[:, 2, 2] = sig_rate ** 2
        Py0[:, 3, 3] = sig_lnr ** 2
        self._filt.set_polar(idx, y0, Py0, sat_c)
        period = 2.0 * np.pi * np.sqrt(sat_el['a'][idx] ** 3 / nm.MU)
        self._acq.reset_rows(idx, sat_c, tgt_c, period)
        self._acq.accumulate(idx, sat_c, tgt_c, tgt_c, self._nav_dt, first=True)

    # ── 3D blind seed ────────────────────────────────────────────────────────
    def _init_rows3(self, idx, sat_c, tgt_c, sat_el):
        """Blind seed for the 6-state modified-spherical filter.

        N3D-A 3c: the 3D acquisition is the 2D acquisition plus two
        locally-convergent degrees of freedom — the analytic range prior lifts
        verbatim (the annulus becomes a spherical shell, the LOS stays a ray),
        and the out-of-plane POSITION comes free and exactly from the measured
        elevation because the seed sits on the LOS ray.
        """
        a_s = sat_el['a'][idx]
        self._filt.set_pole(idx, sat_c)
        Rp = self._filt.Rp[idx]
        d = tgt_c[:, :3] - sat_c[:, :3]
        rho_t = np.maximum(np.linalg.norm(d, axis=1), 1.0)
        u_t = np.einsum('nij,nj->ni', Rp, d / rho_t[:, None])
        el_t = np.arcsin(np.clip(u_t[:, 2], -1.0, 1.0))
        az_t = np.arctan2(u_t[:, 1], u_t[:, 0])
        ce = np.maximum(np.cos(el_t), 1e-8)
        az = nm.wrap_pi(az_t + self._rng.normal(0.0, 1.0, idx.size)
                        * (self._s_beta / ce))
        el = el_t + self._rng.normal(0.0, self._s_beta, idx.size)

        if self._nav_mode == 'rb_ekf':
            rho = np.maximum(
                rho_t + self._rng.normal(0.0, self._s_rho, idx.size), 1.0)
            self._filt.initialize(idx, sat_c, rho, az, el)
            return

        # The range prior, in the pole frame's own coordinates.
        u = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                      np.sin(el)], axis=1)
        u_in = np.einsum('nji,nj->ni', Rp, u)          # back to inertial
        rc2 = np.einsum('ni,ni->n', sat_c[:, :3], sat_c[:, :3])
        b0 = np.einsum('ni,ni->n', sat_c[:, :3], u_in)
        disc_out = b0 * b0 - rc2 + self._r_max ** 2
        rho_out = np.where(disc_out > 0.0,
                           -b0 + np.sqrt(np.maximum(disc_out, 0.0)), 1e3)
        rho_out = np.maximum(rho_out, 200.0)
        disc_in = b0 * b0 - rc2 + self._r_min ** 2
        sq = np.sqrt(np.maximum(disc_in, 0.0))
        lo_in = -b0 - sq
        lo = np.where((disc_in > 0.0) & (lo_in <= 100.0),
                      np.minimum(np.maximum(-b0 + sq, 100.0), 0.5 * rho_out),
                      100.0)
        hi = np.maximum(rho_out, lo * 1.01)
        rho0 = np.sqrt(lo * hi)

        y0 = np.zeros((idx.size, 6))
        y0[:, n3.IDX_AZ] = az
        y0[:, n3.IDX_EL] = el
        y0[:, n3.IDX_LNRHO] = np.log(np.maximum(rho0, 1.0))
        sig_lnr = np.clip(np.log(hi / np.maximum(lo, 1.0)) / np.sqrt(12.0),
                          0.5, 4.0)
        v_c = np.sqrt(nm.MU / np.maximum(a_s, 1.0))
        # In-plane rate ignorance: a circular-orbit velocity guess is wrong by
        # ~v_c*e, and at an unknown range that is also an unknown angular rate.
        sig_rate_ip = np.clip(self._sigma_v_ecc / np.maximum(rho0, 1.0),
                              1e-5, 1e-1)
        # MAJOR-16: the OUT-OF-PLANE rate is seeded from its OWN ignorance,
        # v_c sin(di_max)/rho0, decoupled from the eccentricity term. The
        # shipped 2D seed reused _sigma_v_ecc for both rate components, which
        # at di_max = 1 deg gives 154 m/s against 134 m/s of real ignorance —
        # 1.15x of margin — and at di_max = 2 deg it is over-confident by ~3x
        # on a channel with no measurement of its own at epoch, which is the
        # classic bearings-only divergence trigger.
        di = self._di_max if self._di_max > 0.0 else 0.0
        v_oop = v_c * np.sin(di)
        sig_rate_oop = np.clip(
            np.maximum(v_oop, self._sigma_v_ecc) / np.maximum(rho0, 1.0),
            1e-5, 1e-1)
        assert np.all(sig_rate_oop * rho0 >= v_oop - 1e-9), \
            'MAJOR-16: out-of-plane rate seed is below its own ignorance'
        Py0 = np.zeros((idx.size, 6, 6))
        Py0[:, n3.IDX_AZ, n3.IDX_AZ] = (self._s_beta / np.maximum(
            np.cos(el), 1e-8)) ** 2
        Py0[:, n3.IDX_EL, n3.IDX_EL] = self._s_beta ** 2
        Py0[:, n3.IDX_WA, n3.IDX_WA] = sig_rate_ip ** 2
        Py0[:, n3.IDX_WE, n3.IDX_WE] = sig_rate_oop ** 2
        Py0[:, n3.IDX_RDOT, n3.IDX_RDOT] = sig_rate_ip ** 2
        Py0[:, n3.IDX_LNRHO, n3.IDX_LNRHO] = sig_lnr ** 2
        self._filt.set_polar(idx, y0, Py0, sat_c)
        period = 2.0 * np.pi * np.sqrt(a_s ** 3 / nm.MU)
        self._acq.reset_rows(idx, sat_c, tgt_c, period, a_ref=a_s)
        self._acq.accumulate(idx, sat_c, tgt_c, tgt_c, self._nav_dt, first=True)

    # ── one navigation tick on a subset ──────────────────────────────────────
    def _tick(self, idx, sat_from, sat_to, tgt_to, tgt_prev, dt=None):
        dt = self._nav_dt if dt is None else dt
        if self._dim3:
            self._tick3(idx, sat_from, sat_to, tgt_to, tgt_prev, dt)
            return
        d = tgt_to[:, :2] - sat_to[:, :2]
        rho_t = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1.0)
        beta = nm.wrap_pi(np.arctan2(d[:, 1], d[:, 0])
                          + self._rng.normal(0.0, self._s_beta, idx.size))

        if self._nav_mode == 'rb_ekf':
            self._filt.predict(idx, dt)
            rho = np.maximum(rho_t + self._rng.normal(0.0, self._s_rho, idx.size), 1.0)
            self._filt.update(idx, sat_to, rho, beta)
            return

        self._filt.predict(idx, dt, sat_from, sat_to)
        self._filt.update(idx, sat_to, beta)

        # Acquisition surrogate: information accrues on the realized geometry.
        self._acq.accumulate(idx, sat_to, tgt_to, tgt_prev, dt)
        rows, sig = self._acq.ready(idx, dt)
        if rows.size:
            sel = np.searchsorted(idx, rows)
            x_hat, P_acq, _ = self._acq.draw(rows, sig, tgt_to[sel], sat_to[sel],
                                             self._rng)
            self._filt.set_cart(rows, x_hat, P_acq, sat_to[sel])

    def _tick3(self, idx, sat_from, sat_to, tgt_to, tgt_prev, dt):
        """One 3D nav tick: two focal-plane angles in the epoch-frozen frame.

        The measurement noise is isotropic ON THE SPHERE, which in az/el is
        `sigma_az = sigma_beta / cos el` and `sigma_el = sigma_beta`
        (MAJOR-12). Drawing both at sigma_beta would be a different — and
        easier — sensor than the filter's R claims.
        """
        self._filt.predict(idx, dt, sat_from, sat_to)
        Rp = self._filt.Rp[idx]
        d = tgt_to[:, :3] - sat_to[:, :3]
        rho_t = np.maximum(np.linalg.norm(d, axis=1), 1.0)
        u = np.einsum('nij,nj->ni', Rp, d / rho_t[:, None])
        el_t = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
        az_t = np.arctan2(u[:, 1], u[:, 0])
        ce = np.maximum(np.cos(el_t), 1e-8)
        az = nm.wrap_pi(az_t + self._rng.normal(0.0, 1.0, idx.size)
                        * (self._s_beta / ce))
        el = el_t + self._rng.normal(0.0, self._s_beta, idx.size)

        if self._nav_mode == 'rb_ekf':
            rho = np.maximum(
                rho_t + self._rng.normal(0.0, self._s_rho, idx.size), 1.0)
            self._filt.update(idx, sat_to, rho, az, el)
            return

        self._filt.update(idx, sat_to, az, el)
        self._filt.repole(idx)

        # Hook for an acquisition backend that needs the REALISED measurement
        # arc rather than a re-draw of it — the eval-side real batch IOD. The
        # training surrogate does not define it and pays nothing. It matters
        # that this is the same az/el the filter was fed: a re-draw would give
        # the batch solver an independent noise realisation and the two would
        # disagree for a reason that is not the estimator.
        rec = getattr(self._acq, 'record_meas', None)
        if rec is not None:
            rec(idx, Rp, sat_to, az, el)

        self._acq.accumulate(idx, sat_to, tgt_to, tgt_prev, dt)
        rows, sig = self._acq.ready(idx, dt)
        if rows.size:
            sel = np.searchsorted(idx, rows)
            x_hat, P_acq, _ = self._acq.draw(rows, sig, tgt_to[sel],
                                             sat_to[sel], self._rng)
            self._filt.set_cart(rows, x_hat, P_acq, sat_to[sel])

    # ── divergence guard ─────────────────────────────────────────────────────
    def _guard(self, sat_c, tgt_c, sat_el):
        """MAJOR-9: dimension-generic. `rho` comes from the filter's OWN
        `IDX_LNRHO` (3 under the 4-state, 5 under the 6-state); reading index 3
        unconditionally reads `w_e` in 3D, whose exp() is ~1 m, which trips
        RHO_MIN_M on every row of every step and silently reinitialises the
        filter forever."""
        x = self._mean()
        p = self._filt.POS_DIM
        r2 = np.sum(x[:, :p] ** 2, axis=1)
        v2 = np.sum(x[:, p:2 * p] ** 2, axis=1)
        with np.errstate(all='ignore'):
            inv_a = 2.0 / np.sqrt(np.maximum(r2, 1.0)) - v2 / nm.MU
        bad = ~np.isfinite(x).all(axis=1) | ~(inv_a > 0.0)
        tr = self._filt.trace()
        if self._filt.IDX_LNRHO is not None:
            rho = self._filt.rho()
            bad |= ~np.isfinite(self._filt.y).all(axis=1)
            bad |= (rho < RHO_MIN_M) | (rho > RHO_MAX_M)
        bad |= ~np.isfinite(tr)
        if self._nav_mode == 'rb_ekf':
            bad |= tr > DIVERGE_TRACE_M2
        idx = np.flatnonzero(bad)
        if idx.size:
            self._d_div += int(idx.size)
            self._init_rows(idx, sat_c[idx], tgt_c[idx], sat_el)
        return idx.size

    # ── the wrapper's own step ───────────────────────────────────────────────
    def _nav_step(self, actions, done):
        sat_el, sat_c, _tgt_el, tgt_c = self._decode()

        if self._nav_mode == 'recon':
            self._encode(sat_el, sat_c, tgt_c)
            self._d_n += self.num_agents
            return

        # MAJOR-17: capture the realized plane geometry from TRUTH here, at the
        # top, BEFORE `_encode` overwrites obs[21,22] in place.
        self._di_rel_now = self._di_rel_truth(sat_el, _tgt_el)

        tau = nm.ACTION_TAU[np.asarray(actions, dtype=np.int64).reshape(-1)]
        # MAJOR-7: `nav_max_ticks` used to CAP tau and then tick that many
        # times at dt = 60 s while the C env advanced the full tau, pinning the
        # last sub-tick to the env's current epoch — so at tau = 360, K = 60
        # the filter predicted 60 minutes and was then handed a bearing of the
        # true geometry five hours later, presented as one 60 s innovation. The
        # filter diverges, `_guard` reinitialises it, and the only symptom is
        # an elevated nav_diverge_rate. NAV-H called this inert; it was, at
        # Discrete-16, and it is NOT at Discrete-30 where tau = 180/360 is
        # 26.0% of decisions.
        #
        # The fix is a fixed measurement COUNT with an adaptive INTERVAL, and
        # it is only safe because BLOCKER-2 made the acquisition floor sim
        # time: n_ticks = min(tau, K), dt_tick = tau * 60 / n_ticks. Filter and
        # truth stay on the same clock at every K.
        n_tick_row = tau.copy()
        dt_row = np.full(tau.shape, self._nav_dt)
        if self._nav_max_ticks > 0:
            n_tick_row = np.minimum(tau, self._nav_max_ticks)
            dt_row = (tau * self._nav_dt) / np.maximum(n_tick_row, 1)
        reset = np.asarray(done, dtype=bool).reshape(-1)
        live = ~reset

        if live.any() and self._nav_dt <= 0.0:
            # V6 cross-check path, OFF by design. nav_sensor_dt=0 welds the
            # navigation rate to the guidance rate: exactly one measurement per
            # decision, the filter propagated by the whole tau*60 s in one step.
            # That is the T3/T4 "perdec" condition, measured at 50.5% closed
            # loop against 100% at a 60 s cadence, because the canonical policy
            # spends 51% of its decisions on the 1-hour warp and the filter goes
            # blind for the duration. Kept runnable so the finding is
            # reproducible inside the wrapper rather than only in the serial
            # harness; never used for training.
            for tv in np.unique(tau[live]):
                idx = np.flatnonzero(live & (tau == tv))
                if idx.size:
                    self._tick(idx, self._prev_sat[idx], sat_c[idx],
                               tgt_c[idx], self._prev_tgt[idx],
                               dt=float(tv) * nm.DT)
        elif live.any():
            sc = self._prev_sat.copy()
            tc = self._prev_tgt.copy()
            # The wrapper sub-propagates BOTH TRUTH states between decision
            # epochs at the nav cadence, and the filter is fed measurements
            # built from those states. Under j2_mode that propagation must be
            # J2-aware or the "truth" handed to the filter is a two-body
            # extrapolation: tau reaches 360 sub-steps (6 h), where two-body
            # drifts ~132 km from the env's own J2 truth (measured). The filter
            # then tracks a fiction, which is what produced a secular azimuth
            # innovation ramp to +38 sigma and in-loop NEES 25520 while the
            # filter's OWN propagation, its STM and its chart were all exact.
            if self._nav_j2:
                prop = n3.propagate_cartesian_j2
            elif self._dim3:
                prop = n3.propagate_cartesian_nd
            else:
                prop = nm.propagate_cartesian
            n_tick = int(n_tick_row[live].max())
            for i in range(n_tick):
                idx = np.flatnonzero(live & (n_tick_row > i))
                if idx.size == 0:
                    break
                sat_from = sc[idx]
                tgt_prev = tc[idx]
                # One batched call for both truth states: at B=1024 these are
                # numpy-call-overhead bound, not FLOP bound.
                m = idx.size
                pair = self._pair[:2 * m]
                pair[:m] = sat_from
                pair[m:] = tgt_prev
                step_dt = dt_row[idx]
                pn, _ = prop(pair, np.concatenate([step_dt, step_dt]))
                s_new, t_new = pn[:m], pn[m:]
                sc[idx] = s_new
                tc[idx] = t_new
                # The last sub-tick of a decision IS the env's own epoch: pin
                # truth to the observation instead of the sub-propagation, so
                # no integration drift accumulates across a whole episode.
                last = idx[n_tick_row[idx] == i + 1]
                if last.size:
                    sc[last] = sat_c[last]
                    tc[last] = tgt_c[last]
                for dv in np.unique(step_dt):
                    sub = idx[step_dt == dv]
                    if sub.size:
                        j = np.searchsorted(idx, sub)
                        self._tick(sub, sat_from[j], sc[sub], tc[sub],
                                   tgt_prev[j], dt=float(dv))

        if reset.any():
            idx = np.flatnonzero(reset)
            self._init_rows(idx, sat_c[idx], tgt_c[idx], sat_el)

        if self._nav_mode == 'bearings_only' and live.any():
            dv = nm.ACTION_DV_MAG[np.asarray(actions, dtype=np.int64).reshape(-1)]
            li = np.flatnonzero(live)
            self._acq.add_dv(li, dv[li])

        self._guard(sat_c, tgt_c, sat_el)

        est = self._mean()
        self._encode(sat_el, sat_c, est)
        self._prev_sat = sat_c
        self._prev_tgt = tgt_c
        self._diag(est, tgt_c, sat_c)

    # ── diagnostics ──────────────────────────────────────────────────────────
    def _diag(self, est, tgt_c, sat_c):
        n = self.num_agents
        p = self._filt.POS_DIM
        d = est[:, :p] - tgt_c[:, :p]
        self._d_sq += float(np.sum(d * d))
        self._d_n += n
        if self._nav_mode == 'bearings_only':
            self._d_blind += int(np.count_nonzero(~self._acq.acquired))
        if self.tick % self._nees_every:
            return
        all_idx = np.arange(n)
        x, P = self._filt.mean_cov(all_idx)
        with np.errstate(all='ignore'):
            v = (n3.nees_nd(x, P, tgt_c) if self._dim3
                 else nm.nees(x, P, tgt_c))
        v = v[np.isfinite(v)]
        if v.size:
            self._d_nees.append(float(np.median(v)))
        # sigma along the LOS, normalised by separation — the close-range gate.
        # MAJOR-11 corollary: the RANGE channel only. Adding a plane term
        # certifies garbage at di_rel = 0 (where the plane decouples completely
        # and is measurement-limited irrespective of the range error — 0.73 m
        # of plane against 682 km of range at the 5 km box) and double-counts
        # at di_rel > 0 (where corr -> 1.000), which would also make the 3D
        # gate non-comparable to the 2D gate NB1 was measured against.
        dl = tgt_c[:, :p] - sat_c[:, :p]
        rho = np.maximum(np.sqrt(np.einsum('ni,ni->n', dl, dl)), 1.0)
        u = dl / rho[:, None]
        with np.errstate(all='ignore'):
            s2 = np.einsum('ni,nij,nj->n', u, P[:, :p, :p], u)
        s = np.sqrt(np.maximum(s2, 0.0)) / rho
        s = s[np.isfinite(s)]
        if s.size:
            self._d_slr.append(float(np.median(s)))

    def _nav_info(self):
        n = max(self._d_n, 1)
        info = {
            'nav_pos_rmse': float(np.sqrt(self._d_sq / n)),
            'nav_nees_med': float(np.median(self._d_nees)) if self._d_nees else 0.0,
            'nav_diverge_rate': float(self._d_div / n),
            'nav_clip_rate': float(self._d_clip / max(self._d_clip_n, 1)),
            'nav_sigma_los_over_rho': float(np.median(self._d_slr)) if self._d_slr else 0.0,
        }
        if self._nav_mode == 'bearings_only':
            info['nav_blind_frac'] = float(self._d_blind / n)
            info['nav_acq_per_ep'] = float(self._acq.n_acq / max(self._acq.n_reset, 1))
            # BLOCKER-2: latency in SIM SECONDS. Never decisions (the 3D policy
            # packs 2.45x more sim-time into each decision than the 2D one, so
            # a decision-counted latency shows 3D winning by 2.45x from the tau
            # mix alone) and never ticks (they are the cost knob).
            lat = self._acq.acq_latency_s
            info['nav_acq_latency_s'] = float(np.median(lat)) if lat else 0.0
            info['nav_acq_latency_n'] = float(len(lat))
            self._acq.acq_latency_s = []
            # MAJOR-11: the Cholesky diagonal fallback silently converts an
            # anisotropic scale-family error into a near-isotropic one. The
            # campaign gates on this being zero.
            info['nav_chol_fallback'] = float(self._acq.n_chol_fallback)
        if self._dim3:
            # BLOCKER-1: re-poles per episode, surfaced.
            info['nav_repole_per_ep'] = float(
                np.mean(self._filt.n_repole)) if hasattr(
                    self._filt, 'n_repole') else 0.0
            # NOTE-21: obs[28] is the only channel where a diverged estimate
            # produces a CONFIDENTLY WRONG feasibility margin rather than an
            # obviously broken number. Nothing else covers it.
            info['nav_obs28_neg_unacq'] = float(
                self._d_feas_neg / max(self._d_feas_n, 1))
        if self._sig_ch:
            info['nav_sigma_ch'] = (float(np.median(self._d_sig))
                                    if self._d_sig else 0.0)
        if self._block_below > 0.0:
            info['nav_block_rate'] = float(self._d_block / max(self._d_block_n, 1))
        if self._ablate_rows is not None:
            # MAJOR-10: the fraction of decisions the ablation actually
            # intercepted. A trained ablated arm should drive this toward 0 as
            # it learns the rows are inert; a zero-shot floor should show it
            # NONZERO, which is the proof the rows were being used at all.
            info['nav_ablate_rate'] = float(self._d_ablate / max(self._d_ablate_n, 1))
            self._d_ablate = 0
            self._d_ablate_n = 0
        self._d_reset()
        return info

    # ── PufferEnv surface ────────────────────────────────────────────────────
    def reset(self, seed=0):
        obs, info = super().reset(seed)
        if self._nav_mode == 'truth':
            return obs, info
        self._nav_alloc()
        sat_el, sat_c, _, tgt_c = self._decode()
        if self._nav_mode != 'recon':
            self._init_rows(np.arange(self.num_agents), sat_c, tgt_c, sat_el)
            est = self._mean()
        else:
            est = tgt_c
        self._encode(sat_el, sat_c, est)
        self._prev_sat = sat_c
        self._prev_tgt = tgt_c
        return obs, info

    def step(self, actions):
        # MAJOR-10: the capability ablation runs FIRST and is not gated on
        # nav_mode or on separation. Everything downstream — the C env, the
        # filter's tau, the diagnostics — sees the EXECUTED action set.
        if self._ablate_rows is not None:
            actions = self._apply_row_ablation(actions)
        if self._block_below > 0.0 and self._nav_mode != 'truth':
            actions = self._apply_action_ablation(actions)
        obs, rew, term, trunc, info = super().step(actions)
        if self._nav_mode == 'truth':
            return obs, rew, term, trunc, info
        if not self._nav_ready:
            self._nav_alloc()
        # `actions` is the EXECUTED action set, so tau below is the tau the C
        # env actually applied — required for the filter's propagation interval.
        self._nav_step(actions, np.logical_or(term, trunc))
        if self.tick % self.log_interval == 0:
            info = list(info)
            info.append(self._nav_info())
        return obs, rew, term, trunc, info
