"""Shared helpers for the T3 red-team probes.

Everything here drives the REAL C env (pufferlib.ocean.orbital.Orbital) and
re-derives the proposed S-R3 potential in Python from the env's own observation
vector, so every number is measured, not assumed.

Run from /Users/pete/space_training/pufferlib.
"""
import os
import sys
import math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PUF = '/Users/pete/space_training/pufferlib'
if _PUF not in sys.path:
    sys.path.insert(0, _PUF)

from pufferlib.ocean.orbital.orbital import Orbital  # noqa: E402

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
VE = 300.0 * 9.80665
FUEL_FRAC = 0.15
DRY_MASS = 850.0
DV_BUDGET = -VE * math.log(1.0 - FUEL_FRAC)   # ~478 m/s

CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

# Action table mirrored from orbital.h (dv_prograde, dv_radial), tau
ACTION_DV = [(0, 0), (5, 0), (10, 0), (25, 0), (-5, 0), (-10, 0), (-25, 0),
             (0, 10), (0, -10), (0, 0), (0, 0), (0, 0),
             (1, 0), (-1, 0), (2, 0), (-2, 0)]
ACTION_TAU = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1]

# T3 proposed design (as implemented in the working tree)
T3_KW = dict(shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1,
             phase_obs_mode=1, episode_cap_steps=3000)
W_LAMBDA = 1.0
W_MATCH = 0.35
DV_REF = 300.0


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def true_to_mean(theta, e):
    """Exact inverse Kepler map used by the C env (post-fix)."""
    x = math.sqrt(1.0 + e) * math.cos(theta / 2.0)
    y = math.sqrt(1.0 - e) * math.sin(theta / 2.0)
    E = 2.0 * math.atan2(y, x)
    return E - e * math.sin(E)


class ObsView:
    """Decode the 38-dim observation back into physical state.

    Valid for both phase_obs_mode 0 and 1: a, e, theta, omega for the chaser are
    always in obs[0..3, 9,10]; the target's a, e, omega in obs[7,8,11,12]. The
    target's anomaly comes from obs[15,16] (theta_t in mode 0, lambda_t in mode 1).
    """

    def __init__(self, obs, obs_alt_scale_m=1.6e6, phase_obs_mode=1):
        o = np.asarray(obs, dtype=np.float64)
        self.raw = o
        self.a_s = o[0] * obs_alt_scale_m + R_EARTH
        self.e_s = o[1]
        self.theta_s = math.atan2(o[2], o[3])
        self.vr_over_vc = o[4]
        self.vt_over_vc = o[5]
        self.fuel_frac = o[6]
        self.a_t = o[7] * obs_alt_scale_m + R_EARTH
        self.e_t = o[8]
        self.omega_s = math.atan2(o[9], o[10])
        self.omega_t = math.atan2(o[11], o[12])
        self.M_s = true_to_mean(self.theta_s, self.e_s)
        self.lam_s = self.M_s + self.omega_s
        if phase_obs_mode == 1:
            self.lam_t = math.atan2(o[15], o[16])
            self.M_t = self.lam_t - self.omega_t
            self.dlam_obs = math.atan2(o[13], o[14])
        else:
            self.theta_t = math.atan2(o[15], o[16])
            self.M_t = true_to_mean(self.theta_t, self.e_t)
            self.lam_t = self.M_t + self.omega_t
            self.dlam_obs = None
        self.dlam = wrap_pi(self.lam_s - self.lam_t)
        # LVLH block
        self.lvlh = o[33:38].copy()

    @property
    def n_s(self):
        return math.sqrt(MU / self.a_s ** 3)

    @property
    def da(self):
        return self.a_s - self.a_t

    def dv_match(self):
        e_sx = self.e_s * math.cos(self.omega_s)
        e_sy = self.e_s * math.sin(self.omega_s)
        e_tx = self.e_t * math.cos(self.omega_t)
        e_ty = self.e_t * math.sin(self.omega_t)
        de = math.hypot(e_sx - e_tx, e_sy - e_ty)
        da_rel = (self.a_s - self.a_t) / self.a_t
        v_t = math.sqrt(MU / self.a_t)
        return 0.5 * v_t * math.hypot(da_rel, de)

    def phi_sr3(self, w_lambda=W_LAMBDA, w_match=W_MATCH, dv_ref=DV_REF):
        m = min(1.0, self.dv_match() / dv_ref)
        return -(w_lambda * abs(self.dlam) / math.pi + w_match * m)


def make_env(seed=0, **kw):
    cfg = dict(num_envs=1, num_debris_min=0, num_debris_max=0,
               e_max_target=0.0, e_max_sat=0.0, same_orbit_init=0,
               init_phase_gap_max=0.0, valid_init_only=1)
    cfg.update(kw)
    return Orbital(seed=seed, **cfg)


def rollout(env, policy, max_dec=100000, obs0=None):
    """Run ONE episode. Returns dict of per-decision arrays + terminal info."""
    obs = env.observations
    recs = []
    k = 0
    cause = -1
    while k < max_dec:
        a = policy(k, obs[0])
        env.actions[0] = a
        prev_obs = obs[0].copy()
        _, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        recs.append((k, a, float(rew[0]), prev_obs, obs[0].copy()))
        k += 1
        if term[0] or trunc[0]:
            steps, cause = env.last_episode_result(0)
            break
    return dict(recs=recs, decisions=k, cause=cause,
                sim_steps=(steps if cause >= 0 else None))


def disc_sum(rewards, gamma=0.995):
    g = 1.0
    s = 0.0
    for r in rewards:
        s += g * r
        g *= gamma
    return s
