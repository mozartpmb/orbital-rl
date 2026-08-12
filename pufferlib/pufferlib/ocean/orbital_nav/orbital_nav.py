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
from .nav_surrogate import AcquisitionSurrogate

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
                 nav_acq_min_ticks=45,
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
        self._acq_min_ticks = int(nav_acq_min_ticks)
        self._cov_inflate = float(nav_cov_inflate)
        self._nav_clip = float(nav_clip)
        self._sig_ch = int(nav_sigma_channel)
        self._sig_ref = float(nav_sigma_ref)
        self._sig_gain = float(nav_sigma_gain)
        self._block_below = float(nav_block_fine_below_m)
        self._nees_every = max(1, int(nav_nees_every))
        self._nav_max_ticks = int(nav_max_ticks)

        self._alt_scale = float(kwargs.get('obs_alt_scale_m', 1.6e6))
        self._lvlh_scale = float(kwargs.get('lvlh_scale_m', 6.371e6))

        a_min = kwargs.get('a_min_override', -1.0)
        a_max = kwargs.get('a_max_override', -1.0)
        a_min = a_min if a_min >= nm.R_EARTH else nm.R_EARTH + 300e3
        a_max = a_max if a_max > a_min else nm.R_EARTH + 800e3
        e_max = max(float(kwargs.get('e_max_target', 0.0)), 0.0)
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
        self._rng = np.random.default_rng([self._nav_seed, self._ctor_seed])
        self._s_rho = nm.SIGMA_RHO_M * self._nav_noise
        self._s_beta = nm.SIGMA_BETA_RAD * self._nav_noise
        self._prev_sat = np.zeros((n, 4))
        self._prev_tgt = np.zeros((n, 4))
        self._slots = np.array(nm.TARGET_SLOTS_T3)
        self._scratch = np.zeros((n, nm.OBS_DIM), dtype=np.float64)
        self._pair = np.zeros((2 * n, 4), dtype=np.float64)

        if self._nav_mode == 'rb_ekf':
            q = self._nav_q_a if self._nav_q_a > 0 else nm.Q_ACCEL_PSD_RB
            self._filt = nm.BatchedRangeBearingEKF(
                n, sigma_rho=self._s_rho, sigma_beta=self._s_beta,
                q_a=q, sigma_v0=self._nav_sigma_v0)
        elif self._nav_mode == 'bearings_only':
            q = self._nav_q_a if self._nav_q_a > 0 else nm.Q_ACCEL_PSD_BO
            self._filt = nm.BatchedBearingMPC(n, sigma_beta=self._s_beta, q_a=q)
            self._acq = AcquisitionSurrogate(
                n, sigma_beta=self._s_beta, gate=self._acq_gate,
                min_ticks=self._acq_min_ticks, cov_inflate=self._cov_inflate,
                mode=self._acq_mode)
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

    # ── decode / re-encode ───────────────────────────────────────────────────
    def _decode(self):
        o = self.observations
        sat, tgt = nm.recover_states_t3(o, self._alt_scale)
        sat_c = nm.orbit_to_cartesian(sat['a'], sat['e'], sat['theta'], sat['omega'])
        tgt_c = nm.orbit_to_cartesian(tgt['a'], tgt['e'], tgt['theta'], tgt['omega'])
        return sat, sat_c, tgt, tgt_c

    def _encode(self, sat, sat_c, est_x):
        """Rebuild the target-derived obs slots from the estimate, in place."""
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
    def _sigma_los_over_rho(self, est_x, sat_c):
        """The filter's OWN 1-sigma range uncertainty, normalised by its OWN
        estimated separation. Everything here is estimator-side — no truth — so
        it is a legitimate observation."""
        n = self.num_agents
        idx = np.arange(n)
        _, P = self._filt.mean_cov(idx)
        d = est_x[:, :2] - sat_c[:, :2]
        rho = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1.0)
        u = d / rho[:, None]
        with np.errstate(all='ignore'):
            s2 = np.einsum('ni,nij,nj->n', u, P[:, :2, :2], u)
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
        self.observations[:, 21] = v.astype(np.float32)

    # ── T-BO-act: fine burns are no-ops inside the ablation radius ───────────
    _FINE = np.array([12, 13, 14, 15, 18, 19])

    def _apply_action_ablation(self, actions):
        """Env-variant dynamics: fine burns coast inside `nav_block_fine_below_m`.

        Uses the TRUE separation at the decision epoch (the state the action is
        taken in), which is env-side information, exactly as a real actuator
        interlock would be.
        """
        a = np.asarray(actions, dtype=np.int32).reshape(-1).copy()
        if not self._nav_ready or self._prev_tgt is None:
            return a
        d = self._prev_tgt[:, :2] - self._prev_sat[:, :2]
        rho = np.hypot(d[:, 0], d[:, 1])
        blocked = np.isin(a, self._FINE) & (rho < self._block_below)
        if blocked.any():
            a[blocked] = 0
            self._d_block += int(blocked.sum())
        self._d_block_n += a.size
        return a

    # ── filter mean, cheaply ─────────────────────────────────────────────────
    def _mean(self, idx=None):
        if idx is None:
            idx = np.arange(self.num_agents)
        if self._nav_mode == 'rb_ekf':
            return self._filt.x[idx]
        return nm.msc_decode(self._filt.y[idx], self._filt.sat[idx])

    # ── episode boundaries ───────────────────────────────────────────────────
    def _init_rows(self, idx, sat_c, tgt_c, sat_el):
        """(Re)initialise the filter on `idx`. `sat_c`/`tgt_c` are subsets."""
        if idx.size == 0:
            return
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

    # ── one navigation tick on a subset ──────────────────────────────────────
    def _tick(self, idx, sat_from, sat_to, tgt_to, tgt_prev, dt=None):
        dt = self._nav_dt if dt is None else dt
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

    # ── divergence guard ─────────────────────────────────────────────────────
    def _guard(self, sat_c, tgt_c, sat_el):
        x = self._mean()
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2
        v2 = x[:, 2] ** 2 + x[:, 3] ** 2
        with np.errstate(all='ignore'):
            inv_a = 2.0 / np.sqrt(np.maximum(r2, 1.0)) - v2 / nm.MU
        bad = ~np.isfinite(x).all(axis=1) | ~(inv_a > 0.0)
        if self._nav_mode == 'rb_ekf':
            tr = np.trace(self._filt.P, axis1=1, axis2=2)
        else:
            tr = np.trace(self._filt.Py, axis1=1, axis2=2)
            rho = np.exp(np.minimum(self._filt.y[:, 3], 25.0))
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
        sat_el, sat_c, _, tgt_c = self._decode()

        if self._nav_mode == 'recon':
            self._encode(sat_el, sat_c, tgt_c)
            self._d_n += self.num_agents
            return

        tau = nm.ACTION_TAU[np.asarray(actions, dtype=np.int64).reshape(-1)]
        if self._nav_max_ticks > 0:
            tau = np.minimum(tau, self._nav_max_ticks)
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
            n_tick = int(tau[live].max())
            for i in range(n_tick):
                idx = np.flatnonzero(live & (tau > i))
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
                pn, _ = nm.propagate_cartesian(pair, self._nav_dt)
                s_new, t_new = pn[:m], pn[m:]
                sc[idx] = s_new
                tc[idx] = t_new
                # The last sub-tick of a decision IS the env's own epoch: pin
                # truth to the observation instead of the sub-propagation, so
                # no integration drift accumulates across a whole episode.
                last = idx[tau[idx] == i + 1]
                if last.size:
                    sc[last] = sat_c[last]
                    tc[last] = tgt_c[last]
                self._tick(idx, sat_from, sc[idx], tc[idx], tgt_prev)

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
        d = est[:, :2] - tgt_c[:, :2]
        self._d_sq += float(np.sum(d[:, 0] ** 2 + d[:, 1] ** 2))
        self._d_n += n
        if self._nav_mode == 'bearings_only':
            self._d_blind += int(np.count_nonzero(~self._acq.acquired))
        if self.tick % self._nees_every:
            return
        all_idx = np.arange(n)
        if self._nav_mode == 'rb_ekf':
            x, P = self._filt.mean_cov(all_idx)
        else:
            x, P = self._filt.mean_cov(all_idx)
        with np.errstate(all='ignore'):
            v = nm.nees(x, P, tgt_c)
        v = v[np.isfinite(v)]
        if v.size:
            self._d_nees.append(float(np.median(v)))
        # sigma along the LOS, normalised by separation — the close-range gate.
        dl = tgt_c[:, :2] - sat_c[:, :2]
        rho = np.maximum(np.hypot(dl[:, 0], dl[:, 1]), 1.0)
        u = dl / rho[:, None]
        with np.errstate(all='ignore'):
            s2 = np.einsum('ni,nij,nj->n', u, P[:, :2, :2], u)
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
        if self._sig_ch:
            info['nav_sigma_ch'] = (float(np.median(self._d_sig))
                                    if self._d_sig else 0.0)
        if self._block_below > 0.0:
            info['nav_block_rate'] = float(self._d_block / max(self._d_block_n, 1))
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
