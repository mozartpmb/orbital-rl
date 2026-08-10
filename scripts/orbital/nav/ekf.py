"""T2 relative navigation — EKF over the target's absolute Cartesian state.

Architecture (chaser-nav-known, target-tracked):
  The chaser knows its own state (GPS/INS). The target is tracked from noisy
  range + inertial line-of-sight bearing. The filter state is the target's
  absolute inertial state x = [x, y, vx, vy]; the chaser position enters only
  through the measurement model, as a known quantity.

Dynamics
  Mean: exact two-body Kepler propagation (orbital_math.propagate_cartesian),
  which is the identical map the C env applies to the target. There is
  therefore *zero* dynamic model mismatch; process noise exists only to keep
  the covariance from collapsing to a numerically singular value and to absorb
  EKF linearization error. Q is the standard continuous white-noise-
  acceleration discretization with PSD q_a (m^2/s^3):
      Q = q_a * [[dt^3/3 I2, dt^2/2 I2], [dt^2/2 I2, dt I2]]
  STM (orbital_math.stm_numerical): central differences of the exact
  propagator, h = 1 m in position / 1e-3 m/s in velocity. Symplectic to 9.7e-7
  and linear to 1.2e-7 over a 100 m probe — far below anything Q or R sees.
  The propagator must be the Cartesian f&g one, not an element round trip: on
  near-circular orbits the latter loses ~1e-7 rad of angle precision, which is
  70% error on a 1 m perturbation and collapses P to ~57 m sigma while the true
  error grows past 30 km. That failure is what motivated the f&g rewrite.

Measurements (one per env.step(), so dt = ACTION_TAU[action] * 60 s)
  rho  = |r_t - r_c|                       sigma_rho = 50 m   (nominal)
  beta = atan2(dy, dx) in the INERTIAL frame, i.e. the line-of-sight azimuth an
         optical sensor registered against a star tracker delivers. Chaser
         attitude is assumed known, so an LVLH-referenced bearing carries the
         same information; inertial is used because it keeps H free of chaser
         velocity terms.                    sigma_beta = 1 mrad (nominal)
  Update uses the Joseph form and wraps the bearing innovation to [-pi, pi].

Initialization (single measurement, no acquisition phase)
  Position: r_c + rho * [cos beta, sin beta]  -- exactly invertible, so P0's
  position block is J R J^T with J = d(pos)/d(rho, beta).
  Velocity: circular-orbit guess at the measured position, prograde
  (all orbits in this env are counter-clockwise): v = sqrt(mu/r) * (-y, x)/r.
  P0 velocity block = sigma_v0^2 I, with sigma_v0 set to the measured RMS of
  that guess's error under the eval init distribution (calibrate_sigma_v0;
  107.6 m/s -> 100). NEES is insensitive to it beyond the first few steps.
"""

import math
import numpy as np

from orbital_math import MU, propagate_cartesian, stm_numerical

# Nominal sensor suite.
SIGMA_RHO_M = 50.0        # range 1-sigma (m)
SIGMA_BETA_RAD = 1.0e-3   # bearing 1-sigma (rad)

# Defaults, both set empirically (see T2_RELNAV_FINDINGS.md):
#   q_a    selected on NEES in-bounds fraction over a 6-point sweep; peaks at
#          1e-11 with balanced tails (3.8% below / 4.3% above the chi2 bounds).
#          Dynamics are exact, so q_a has no model error to absorb -- it is
#          purely a covariance floor.
#   sigma_v0  = measured RMS of the circular-velocity init guess error
#          (107.6 m/s over the eval init distribution), rounded to 100.
Q_ACCEL_PSD = 1.0e-11     # m^2/s^3
SIGMA_V0 = 100.0          # m/s, initial velocity-guess uncertainty



def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def measure(sat_cart, tgt_cart, rng, sigma_rho=SIGMA_RHO_M,
            sigma_beta=SIGMA_BETA_RAD):
    """Draw a noisy (range, inertial bearing) measurement of the target."""
    dx = tgt_cart[0] - sat_cart[0]
    dy = tgt_cart[1] - sat_cart[1]
    rho = math.hypot(dx, dy) + rng.normal(0.0, sigma_rho)
    beta = wrap_pi(math.atan2(dy, dx) + rng.normal(0.0, sigma_beta))
    return rho, beta


def process_noise(dt, q_a):
    """Continuous white-noise-acceleration Q for a 2D double integrator."""
    I2 = np.eye(2)
    Q = np.zeros((4, 4))
    Q[:2, :2] = (dt ** 3 / 3.0) * I2
    Q[:2, 2:] = (dt ** 2 / 2.0) * I2
    Q[2:, :2] = (dt ** 2 / 2.0) * I2
    Q[2:, 2:] = dt * I2
    return q_a * Q


class TargetEKF:
    """EKF on the target's absolute inertial Cartesian state."""

    def __init__(self, sigma_rho=SIGMA_RHO_M, sigma_beta=SIGMA_BETA_RAD,
                 q_a=Q_ACCEL_PSD, sigma_v0=SIGMA_V0):
        self.sigma_rho = sigma_rho
        self.sigma_beta = sigma_beta
        self.q_a = q_a
        self.sigma_v0 = sigma_v0
        self.R = np.diag([sigma_rho ** 2, sigma_beta ** 2])
        self.x = None
        self.P = None
        self.n_updates = 0

    # ── init ────────────────────────────────────────────────────────────────
    def initialize(self, sat_cart, rho, beta):
        """Single-measurement init: inverted measurement + circular velocity."""
        px = sat_cart[0] + rho * math.cos(beta)
        py = sat_cart[1] + rho * math.sin(beta)
        r = math.hypot(px, py)
        v_c = math.sqrt(MU / r)
        vx = -v_c * py / r     # counter-clockwise: v_hat = z_hat x r_hat
        vy = v_c * px / r

        # Position block: exact Jacobian of (rho, beta) -> (px, py).
        cb, sb = math.cos(beta), math.sin(beta)
        J = np.array([[cb, -rho * sb], [sb, rho * cb]])
        P = np.zeros((4, 4))
        P[:2, :2] = J @ self.R @ J.T
        P[2, 2] = P[3, 3] = self.sigma_v0 ** 2

        self.x = np.array([px, py, vx, vy], dtype=float)
        self.P = P
        self.n_updates = 0
        return self.x.copy()

    # ── predict / update ────────────────────────────────────────────────────
    def predict(self, dt):
        F = stm_numerical(self.x, dt)
        self.x = np.asarray(propagate_cartesian(self.x, dt), dtype=float)
        self.P = F @ self.P @ F.T + process_noise(dt, self.q_a)
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def update(self, sat_cart, rho, beta):
        """Returns (innovation, S) so the caller can accumulate NIS."""
        dx = self.x[0] - sat_cart[0]
        dy = self.x[1] - sat_cart[1]
        rho_hat = math.hypot(dx, dy)
        if rho_hat < 1e-6:
            return np.zeros(2), np.eye(2)
        beta_hat = math.atan2(dy, dx)

        H = np.array([
            [dx / rho_hat, dy / rho_hat, 0.0, 0.0],
            [-dy / rho_hat ** 2, dx / rho_hat ** 2, 0.0, 0.0],
        ])
        nu = np.array([rho - rho_hat, wrap_pi(beta - beta_hat)])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ nu
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1
        return nu, S

    # ── diagnostics ─────────────────────────────────────────────────────────
    def nees(self, truth_cart):
        """Normalized estimation error squared, normalized by state dim (4)."""
        err = self.x - np.asarray(truth_cart, dtype=float)
        try:
            v = float(err @ np.linalg.solve(self.P, err))
        except np.linalg.LinAlgError:
            return float('nan')
        return v / 4.0

    def sigmas(self):
        d = np.clip(np.diag(self.P), 0.0, None)
        return np.sqrt(d)


# 95% two-sided chi-square bounds, normalized by dof.
NEES_LO, NEES_HI = 0.4844 / 4.0, 11.1433 / 4.0     # chi2(4)
NIS_LO, NIS_HI = 0.0506 / 2.0, 7.3778 / 2.0        # chi2(2)


def calibrate_sigma_v0(errors):
    """RMS of the circular-guess velocity error, per component."""
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))
