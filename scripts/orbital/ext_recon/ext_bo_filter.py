"""NAV-G — bearings-only (angles-only) relative navigation prototype.

Go/no-go datum for the ext-nav campaign: can the T2/T4 range+bearing EKF be
replaced by an angles-only filter without losing the state estimate the policy
flies on?

Architecture is inherited unchanged from `scripts/orbital/nav/ekf.py`: the
chaser knows its own absolute state (GPS/INS), the filter tracks the *target's
absolute inertial Cartesian state* x = [x, y, vx, vy] under exact two-body
dynamics (orbital_math.propagate_cartesian + stm_numerical). Only the
measurement changes: the range row is deleted, leaving the scalar inertial
line-of-sight bearing.

Why that architecture matters (and why the classic bearings-only-tracking
pathologies do not transfer verbatim):
  * The textbook BOT unobservability theorem (Nardone & Aidala 1981) is a
    statement about a *constant-velocity* target seen from a *constant-velocity*
    observer: the family x_rel -> k*x_rel is dynamically closed, so the whole
    bearing history is invariant and range is structurally unobservable.
  * The rendezvous-specific version (Woffinden & Geller 2007; Grzymisch &
    Fichter 2014) is the same statement for *linearized* relative motion (CW /
    YA): those dynamics are linear and homogeneous, so scaling is again a
    symmetry, and range is unobservable absent a maneuver.
  * Here neither applies as stated. The filter state is the target's ABSOLUTE
    state on a Keplerian orbit about a known center, observed from a KNOWN
    absolute chaser position. Scaling the relative vector moves the target to a
    different absolute radius, hence a different mean motion (n ~ a^-3/2), hence
    a different bearing history. The scale symmetry is broken by the
    nonlinearity of gravity over the separation, at strength O(rho/r).
So the question is never "is it observable" — it is "how well conditioned", and
the answer is a function of rho/r, arc length, and whether the chaser burns.
This script measures that, and compares four filters against the shipped
range+bearing baseline on identical truth trajectories.

Filters compared (a '-o' suffix means the epoch range is handed to the filter as
an oracle with 30% sigma, which isolates STEADY-STATE behaviour from the
acquisition problem — the only fair way to test a claim about consistency)
  RB-EKF      range + bearing EKF        (baseline; ekf.TargetEKF verbatim)
  BO-EKF      bearings-only single EKF, initialized at the geometric-mean range
              of the analytic feasible interval   (the naive / pathology arm)
  BO-UKF      bearings-only UKF, same init  (does sigma-point propagation of the
              exact flow rescue a decades-wide range prior? no.)
  BO-EKF-o    bearings-only EKF, oracle epoch range
  BO-MPC-o    bearings-only EKF in modified-polar coordinates,
              y = [beta, betadot, rhodot/rho, ln rho] (Aidala & Hammel 1983):
              the measurement becomes LINEAR (H = e_1) and the weakly observable
              direction is isolated in one component — the classic cure for
              premature covariance collapse. Oracle epoch range.
  BO-RPB      range-parameterized EKF bank (Peach 1995; Ristic et al.):
              log-spaced range hypotheses, Gaussian-mixture bearing likelihood
              weights. Fully blind. The recursive-only fallback.
  BO-BLS      RECOMMENDED. Angles-only batch acquisition (dense range grid +
              binned multi-start Gauss-Newton, arc grown until a chi-square, an
              ambiguity-margin and a covariance gate all pass) handed off to the
              recursive bearings-only EKF. Fully blind.
  BO-BLS-MPC  same acquisition, modified-polar recursive stage.

Range prior without a range measurement
  The scenario distribution bounds the target's radius to an annulus
  [r_min, r_max] (altitude band x eccentricity). A bearing from the known
  chaser position is a ray; intersecting it with the annulus gives the exact
  feasible range set analytically — and that set is BIMODAL whenever the ray
  cuts the inner circle (near-side and far-side branches), which is the
  structural reason a single-hypothesis filter is the wrong object here.
  See `range_prior_intervals`.

Run:  python3 ext_bo_filter.py            # full: 5 geometries x 24 seeds
      python3 ext_bo_filter.py --quick    # 5 geometries x 6 seeds
No environment import, no C build, no file in the repo is modified.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

NAV_DIR = "/Users/pete/space_training/scripts/orbital/nav"
sys.path.insert(0, NAV_DIR)

import orbital_math as om                                       # noqa: E402
from orbital_math import MU, R_EARTH, propagate_cartesian, stm_numerical  # noqa: E402
from ekf import (TargetEKF, wrap_pi, process_noise, measure,     # noqa: E402
                 NEES_LO, NEES_HI, SIGMA_RHO_M, SIGMA_BETA_RAD)

OUT_CSV = "/Users/pete/space_training/web_data/results/ext_bo_filter.csv"
OUT_CSV_STEP = "/Users/pete/space_training/web_data/results/ext_bo_convergence.csv"

SIGMA_BETA = SIGMA_BETA_RAD      # 1 mrad, the shipped optical bearing sigma
Q_ACCEL_PSD_BO = 1.0e-13         # covariance floor; wide-envelope tuned value
SETTLE_FRAC = 0.75               # "settled" = last quarter of the arc


# ── truth generation (oracle) ────────────────────────────────────────────────
def elements(a, e, omega, theta):
    el = {'a': a, 'e': e, 'omega': omega, 'theta': theta}
    el['M'] = om.mean_from_true(theta, e)
    return el


def period(a):
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU)


def roll_truth(sc):
    """Propagate chaser and target with the oracle. Returns (t, sat[], tgt[])."""
    dt = sc['dt']
    n = int(round(sc['duration'] / dt))
    s = om.orbit_to_cartesian(sc['sat'])
    g = om.orbit_to_cartesian(sc['tgt'])
    burns = {int(round(bt / dt)): bdv for bt, bdv in sc.get('burns', ())}
    S, G, T = [s], [g], [0.0]
    for k in range(1, n + 1):
        s = propagate_cartesian(s, dt)
        g = propagate_cartesian(g, dt)
        if k in burns:
            dv = burns[k]
            vn = math.hypot(s[2], s[3])
            s = (s[0], s[1], s[2] + dv * s[2] / vn, s[3] + dv * s[3] / vn)
        S.append(s)
        G.append(g)
        T.append(k * dt)
    return T, S, G


def bearing_of(sat, tgt):
    return math.atan2(tgt[1] - sat[1], tgt[0] - sat[0])


def range_of(sat, tgt):
    return math.hypot(tgt[0] - sat[0], tgt[1] - sat[1])


# ── range prior from scenario geometry (no range measurement needed) ─────────
def range_prior_intervals(sat_cart, beta, r_min, r_max, rho_floor=100.0):
    """Feasible range set for a target constrained to the annulus [r_min, r_max].

    The LOS is the ray p(rho) = r_c + rho*u, u = (cos b, sin b).  |p|^2 = R^2 is
    a quadratic in rho with roots -b0 +/- sqrt(b0^2 - r_c^2 + R^2), b0 = r_c.u.
    The outer circle always admits exactly one positive root (r_c <= r_max), so
    the feasible set is (0, rho_out] minus the open chord inside the inner
    circle. When the ray pierces the inner circle the result is TWO disjoint
    intervals — the multi-hypothesis case that no single Gaussian can represent.
    """
    ux, uy = math.cos(beta), math.sin(beta)
    rc2 = sat_cart[0] ** 2 + sat_cart[1] ** 2
    b0 = sat_cart[0] * ux + sat_cart[1] * uy

    disc_out = b0 * b0 - rc2 + r_max * r_max
    if disc_out <= 0.0:
        return [(rho_floor, rho_floor * 10.0)]          # degenerate, unreachable
    rho_out = -b0 + math.sqrt(disc_out)
    if rho_out <= rho_floor:
        return [(rho_floor, max(2.0 * rho_floor, rho_out))]

    disc_in = b0 * b0 - rc2 + r_min * r_min
    if disc_in <= 0.0:
        return [(rho_floor, rho_out)]
    sq = math.sqrt(disc_in)
    lo_in, hi_in = -b0 - sq, -b0 + sq                    # chord inside r_min
    out = []
    if lo_in > rho_floor:
        out.append((rho_floor, min(lo_in, rho_out)))
    if hi_in < rho_out:
        out.append((max(hi_in, rho_floor), rho_out))
    return out or [(rho_floor, rho_out)]


def prior_span(intervals):
    lo = min(a for a, _ in intervals)
    hi = max(b for _, b in intervals)
    return lo, hi


# ── shared init helpers ──────────────────────────────────────────────────────
def circular_state(px, py):
    """Prograde circular-orbit velocity guess at (px, py)."""
    r = math.hypot(px, py)
    vc = math.sqrt(MU / r)
    return -vc * py / r, vc * px / r


def ray_init(sat_cart, beta, rho0, sig_rho, sigma_beta, sigma_v_ecc):
    """(x, P) for a target hypothesized at range rho0 along the LOS.

    Position covariance is anisotropic by construction: sig_rho^2 along the LOS
    and (rho0*sigma_beta)^2 across it. Velocity is the prograde circular guess,
    whose error has two independent parts: the target's actual eccentricity
    (~v_c*e, the measured driver per T4 §8.2) and the radius uncertainty
    propagated through v_c ~ r^-1/2, i.e. 0.5*v_c*sig_r/r. Both are folded into
    an isotropic velocity block; the cross terms are deliberately dropped, which
    is conservative (larger P0).
    """
    px = sat_cart[0] + rho0 * math.cos(beta)
    py = sat_cart[1] + rho0 * math.sin(beta)
    r = math.hypot(px, py)
    if r <= R_EARTH * 0.5:
        return None
    vx, vy = circular_state(px, py)
    vc = math.sqrt(MU / r)

    u = np.array([math.cos(beta), math.sin(beta)])
    nvec = np.array([-math.sin(beta), math.cos(beta)])
    Ppos = (sig_rho ** 2) * np.outer(u, u) + ((rho0 * sigma_beta) ** 2) * np.outer(nvec, nvec)
    # radius uncertainty implied by the LOS range uncertainty
    sig_r = abs(sig_rho * (px * u[0] + py * u[1]) / r)
    sig_v = math.hypot(sigma_v_ecc, 0.5 * vc * sig_r / r)

    P = np.zeros((4, 4))
    P[:2, :2] = Ppos
    P[2, 2] = P[3, 3] = sig_v ** 2
    return np.array([px, py, vx, vy], dtype=float), P


# ── BO-EKF: bearings-only, Cartesian state ───────────────────────────────────
class BearingEKF:
    """Single-hypothesis bearings-only EKF on the target's absolute state."""

    name = "BO-EKF"

    def __init__(self, sigma_beta=SIGMA_BETA, q_a=Q_ACCEL_PSD_BO):
        self.sigma_beta = sigma_beta
        self.q_a = q_a
        self.x = None
        self.P = None
        self.alive = True
        self.logw = 0.0

    def set(self, x, P):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(P, dtype=float).copy()
        return self

    def predict(self, dt):
        if not self.alive:
            return
        F = stm_numerical(self.x, dt)
        self.x = np.asarray(propagate_cartesian(self.x, dt), dtype=float)
        self.P = F @ self.P @ F.T + process_noise(dt, self.q_a)
        self.P = 0.5 * (self.P + self.P.T)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.P)):
            self.alive = False

    def update(self, sat_cart, beta):
        """Scalar bearing update. Returns (nu, S, loglik)."""
        if not self.alive:
            return 0.0, 1.0, -1e9
        dx = self.x[0] - sat_cart[0]
        dy = self.x[1] - sat_cart[1]
        r2 = dx * dx + dy * dy
        if r2 < 1e-6:
            return 0.0, 1.0, -1e9
        H = np.array([[-dy / r2, dx / r2, 0.0, 0.0]])
        nu = wrap_pi(beta - math.atan2(dy, dx))
        S = float((H @ self.P @ H.T).item()) + self.sigma_beta ** 2
        K = (self.P @ H.T) / S
        self.x = self.x + (K.flatten() * nu)
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + (K @ K.T) * self.sigma_beta ** 2
        self.P = 0.5 * (self.P + self.P.T)
        ll = -0.5 * (nu * nu / S + math.log(2.0 * math.pi * S))
        if not np.all(np.isfinite(self.x)):
            self.alive = False
        return nu, S, ll

    def mean_cov(self):
        return self.x, self.P


# ── BO-UKF: bearings-only unscented ──────────────────────────────────────────
class BearingUKF:
    name = "BO-UKF"

    def __init__(self, sigma_beta=SIGMA_BETA, q_a=Q_ACCEL_PSD_BO,
                 alpha=1e-3, beta_ut=2.0, kappa=0.0):
        self.sigma_beta = sigma_beta
        self.q_a = q_a
        self.n = 4
        self.lam = alpha ** 2 * (self.n + kappa) - self.n
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lam)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lam / (self.n + self.lam)
        self.wc[0] = self.wm[0] + (1.0 - alpha ** 2 + beta_ut)
        self.x = None
        self.P = None
        self.alive = True

    def set(self, x, P):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(P, dtype=float).copy()
        return self

    def _sigmas(self):
        try:
            A = np.linalg.cholesky((self.n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh((self.n + self.lam) * self.P)
            A = V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
        pts = [self.x]
        for j in range(self.n):
            pts.append(self.x + A[:, j])
            pts.append(self.x - A[:, j])
        return np.array(pts)

    def predict(self, dt):
        if not self.alive:
            return
        X = self._sigmas()
        Y = np.array([propagate_cartesian(list(p), dt) for p in X])
        if not np.all(np.isfinite(Y)):
            self.alive = False
            return
        self.x = self.wm @ Y
        d = Y - self.x
        self.P = (d.T * self.wc) @ d + process_noise(dt, self.q_a)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, sat_cart, beta):
        if not self.alive:
            return 0.0, 1.0, -1e9
        X = self._sigmas()
        b0 = math.atan2(self.x[1] - sat_cart[1], self.x[0] - sat_cart[0])
        Z = np.array([b0 + wrap_pi(math.atan2(p[1] - sat_cart[1],
                                              p[0] - sat_cart[0]) - b0)
                      for p in X])
        zbar = float(self.wm @ Z)
        dz = Z - zbar
        S = float((dz * self.wc) @ dz) + self.sigma_beta ** 2
        dx = X - self.x
        Pxz = (dx.T * self.wc) @ dz
        K = Pxz / S
        nu = wrap_pi(beta - zbar)
        self.x = self.x + K * nu
        self.P = self.P - S * np.outer(K, K)
        self.P = 0.5 * (self.P + self.P.T)
        ll = -0.5 * (nu * nu / S + math.log(2.0 * math.pi * S))
        if not np.all(np.isfinite(self.x)):
            self.alive = False
        return nu, S, ll

    def mean_cov(self):
        return self.x, self.P


# ── BO-MPC: modified polar coordinates ───────────────────────────────────────
RHO_FLOOR_M = 1.0     # 1 m: 4 decades below the tightest shipped success box


def msc_encode(sat, tgt):
    """[beta, betadot, rhodot/rho, ln rho] from absolute Cartesian states.

    GUARDED (red-team BLOCKER-1). The original divided by rho2 and took
    math.log(rho) unguarded, so a finite-difference iterate that lands on the
    chaser raises `ValueError: math domain error` (or emits nan through the
    rho2 divides) and kills the run. Reproduced live in the ext-nav
    bearings-only closed loop on the very first episode. Flooring rho at 1 m
    cannot fire for a physical geometry and turns a crash into a bounded
    number.
    """
    dx, dy = tgt[0] - sat[0], tgt[1] - sat[1]
    dvx, dvy = tgt[2] - sat[2], tgt[3] - sat[3]
    rho2 = dx * dx + dy * dy
    if not (rho2 > RHO_FLOOR_M ** 2) or not math.isfinite(rho2):
        rho2 = RHO_FLOOR_M ** 2
        if not (math.isfinite(dx) and math.isfinite(dy)):
            dx, dy = RHO_FLOOR_M, 0.0
        if not (math.isfinite(dvx) and math.isfinite(dvy)):
            dvx = dvy = 0.0
    return np.array([math.atan2(dy, dx),
                     (dx * dvy - dy * dvx) / rho2,
                     (dx * dvx + dy * dvy) / rho2,
                     0.5 * math.log(rho2)])


def msc_decode(y, sat):
    beta, bdot, rr, lr = y
    if not math.isfinite(lr):
        lr = math.log(RHO_FLOOR_M)
    rho = max(math.exp(min(lr, 25.0)), RHO_FLOOR_M)
    c, s = math.cos(beta), math.sin(beta)
    dx, dy = rho * c, rho * s
    rdot = rr * rho
    dvx = rdot * c - rho * bdot * s
    dvy = rdot * s + rho * bdot * c
    return (sat[0] + dx, sat[1] + dy, sat[2] + dvx, sat[3] + dvy)


def _msc_dy(a, b):
    d = a - b
    d[0] = wrap_pi(d[0])
    return d


class BearingMPC:
    """Bearings-only EKF in modified polar coordinates.

    The measurement is y[0] exactly, so H = [1,0,0,0] and the update is linear —
    no linearization error at all in the update, which is what removes the
    classic premature-covariance-collapse failure. All the nonlinearity is
    pushed into the transition, which is evaluated exactly (encode o exact
    two-body flow o decode) with a numerically differenced Jacobian whose step
    sizes are chosen to induce ~1 m / ~1e-3 m/s perturbations, matching
    orbital_math.stm_numerical's calibration.
    """

    name = "BO-MPC"

    def __init__(self, sigma_beta=SIGMA_BETA, q_a=Q_ACCEL_PSD_BO):
        self.sigma_beta = sigma_beta
        self.q_a = q_a
        self.y = None
        self.Py = None
        self.sat = None
        self.alive = True

    def set_cart(self, x, P, sat_cart):
        self.sat = tuple(sat_cart)
        self.y = msc_encode(sat_cart, x)
        G = self._enc_jac(sat_cart, x)
        self.Py = G @ P @ G.T
        self.Py = 0.5 * (self.Py + self.Py.T)
        return self

    @staticmethod
    def _enc_jac(sat, xt):
        G = np.empty((4, 4))
        h = (1.0, 1.0, 1e-3, 1e-3)
        for j in range(4):
            xp, xm = list(xt), list(xt)
            xp[j] += h[j]
            xm[j] -= h[j]
            G[:, j] = _msc_dy(msc_encode(sat, xp), msc_encode(sat, xm)) / (2.0 * h[j])
        return G

    def _trans(self, y, sat_from, sat_to, dt):
        xt = msc_decode(y, sat_from)
        xt2 = propagate_cartesian(xt, dt)
        return msc_encode(sat_to, xt2)

    def predict(self, dt, sat_from, sat_to):
        if not self.alive:
            return
        # Same guard as msc_encode/msc_decode (red-team BLOCKER-1): ln rho can
        # underflow to exp(-inf) = 0 once a trained bearings-only policy drives
        # the filter somewhere the prototype's own scenarios never went, and an
        # unguarded 1/rho then kills the run.
        rho = max(math.exp(min(self.y[3], 25.0)), RHO_FLOOR_M)
        h = (1.0 / rho, 1e-3 / rho, 1e-3 / rho, 1.0 / rho)
        y0 = self._trans(self.y, sat_from, sat_to, dt)
        F = np.empty((4, 4))
        for j in range(4):
            yp, ym = self.y.copy(), self.y.copy()
            yp[j] += h[j]
            ym[j] -= h[j]
            F[:, j] = _msc_dy(self._trans(yp, sat_from, sat_to, dt),
                              self._trans(ym, sat_from, sat_to, dt)) / (2.0 * h[j])
        # process noise: Cartesian PSD mapped through the encoding Jacobian
        xt = msc_decode(y0, sat_to)
        G = self._enc_jac(sat_to, xt)
        Qy = G @ process_noise(dt, self.q_a) @ G.T
        self.y = y0
        self.Py = F @ self.Py @ F.T + Qy
        self.Py = 0.5 * (self.Py + self.Py.T)
        self.sat = tuple(sat_to)
        if not np.all(np.isfinite(self.y)) or not np.all(np.isfinite(self.Py)):
            self.alive = False

    def update(self, sat_cart, beta):
        if not self.alive:
            return 0.0, 1.0, -1e9
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        nu = wrap_pi(beta - self.y[0])
        S = float(self.Py[0, 0]) + self.sigma_beta ** 2
        K = (self.Py @ H.T) / S
        self.y = self.y + K.flatten() * nu
        self.y[0] = wrap_pi(self.y[0])
        IKH = np.eye(4) - K @ H
        self.Py = IKH @ self.Py @ IKH.T + (K @ K.T) * self.sigma_beta ** 2
        self.Py = 0.5 * (self.Py + self.Py.T)
        ll = -0.5 * (nu * nu / S + math.log(2.0 * math.pi * S))
        return nu, S, ll

    def mean_cov(self):
        """Back to Cartesian for a like-for-like NEES against the other filters."""
        x = np.asarray(msc_decode(self.y, self.sat), dtype=float)
        G = self._enc_jac(self.sat, x)
        try:
            Ginv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return x, np.eye(4) * 1e12
        P = Ginv @ self.Py @ Ginv.T
        return x, 0.5 * (P + P.T)


# ── BO-RPB: range-parameterized bank ─────────────────────────────────────────
class RangeParameterizedBank:
    """Gaussian-mixture bank of bearings-only EKFs over log-spaced range.

    Peach (1995) / Ristic-Arulampalam-Gordon: partition the feasible range set
    geometrically so every component carries the SAME relative range uncertainty
    sigma_rho/rho = (w-1)/((w+1)*sqrt(12)) with w the geometric ratio. Constant
    relative uncertainty is the whole point: it is what keeps each component's
    linearization valid regardless of where in the decades-wide prior it sits.
    Weights are the recursive bearing likelihood; components are pruned at
    log-weight 25 below the best (weight ratio 1.4e-11).
    """

    name = "BO-RPB"

    def __init__(self, sigma_beta=SIGMA_BETA, q_a=Q_ACCEL_PSD_BO, K=0,
                 prune_ll=25.0, ratio=1.5, k_max=48):
        self.sigma_beta = sigma_beta
        self.q_a = q_a
        self.K = K                # 0 = size the bank from the prior span
        self.ratio = ratio        # target geometric width per component
        self.k_max = k_max
        self.prune_ll = prune_ll
        self.comps = []
        self.logw = None

    def initialize(self, sat_cart, beta, intervals, sigma_v_ecc):
        comps, logw = [], []
        # distribute components across intervals proportional to log-measure
        spans = [(lo, hi, math.log(hi / lo)) for lo, hi in intervals if hi > lo]
        tot = sum(s[2] for s in spans) or 1.0
        # Bank size is a property of the prior, not a hand-tuned constant: hold
        # the per-component geometric width fixed at `ratio` so the relative
        # range sigma (and hence the linearization validity) is the same for a
        # 10x prior and a 40000x one.
        K = self.K or min(self.k_max,
                          max(4, int(math.ceil(tot / math.log(self.ratio)))))
        for lo, hi, sp in spans:
            k = max(1, int(round(K * sp / tot)))
            w = (hi / lo) ** (1.0 / k)
            rel = (w - 1.0) / ((w + 1.0) * math.sqrt(12.0))
            for i in range(k):
                rho_lo = lo * w ** i
                rho_k = rho_lo * (1.0 + w) / 2.0
                init = ray_init(sat_cart, beta, rho_k, rel * rho_k,
                                self.sigma_beta, sigma_v_ecc)
                if init is None:
                    continue
                f = BearingEKF(self.sigma_beta, self.q_a).set(*init)
                comps.append(f)
                logw.append(math.log(sp / tot / max(k, 1)))
        if not comps:
            init = ray_init(sat_cart, beta, 1e5, 1e5, self.sigma_beta, sigma_v_ecc)
            comps = [BearingEKF(self.sigma_beta, self.q_a).set(*init)]
            logw = [0.0]
        self.comps = comps
        self.logw = np.array(logw, dtype=float)
        self.logw -= self._lse(self.logw)
        return self

    @staticmethod
    def _lse(v):
        m = np.max(v)
        return m + math.log(float(np.sum(np.exp(v - m))))

    def predict(self, dt):
        for f in self.comps:
            f.predict(dt)

    def update(self, sat_cart, beta):
        lls = np.empty(len(self.comps))
        for i, f in enumerate(self.comps):
            _, _, ll = f.update(sat_cart, beta)
            lls[i] = ll if f.alive else -1e9
        self.logw = self.logw + lls
        self.logw -= self._lse(self.logw)
        keep = self.logw > (self.logw.max() - self.prune_ll)
        if keep.sum() >= 1:
            self.comps = [c for c, k in zip(self.comps, keep) if k]
            self.logw = self.logw[keep]
            self.logw -= self._lse(self.logw)
        return 0.0, 1.0, 0.0

    def weights(self):
        return np.exp(self.logw)

    def n_eff(self):
        w = self.weights()
        return 1.0 / float(np.sum(w * w))

    def mean_cov(self):
        w = self.weights()
        X = np.array([c.x for c in self.comps])
        xb = w @ X
        P = np.zeros((4, 4))
        for wi, c in zip(w, self.comps):
            d = (c.x - xb).reshape(4, 1)
            P += wi * (c.P + d @ d.T)
        return xb, 0.5 * (P + P.T)


# ── BO-BLS: range-grid batch least squares (angles-only IOD, 2D) ─────────────
R_LO, R_HI = 0.9 * R_EARTH, 60.0 * R_EARTH     # physically admissible radii


def _bearing_cost(times, sats, beta_obs, x0, sigma_beta):
    """Sum of squared normalized bearing residuals for a candidate epoch state.

    Propagation only — no STM — so this is ~9x cheaper than a Gauss-Newton
    iteration and can be evaluated on a dense range grid. Returns +inf for any
    candidate whose trajectory leaves the admissible radius band, which is what
    keeps the grid search from scoring numerical garbage.
    """
    xt = np.asarray(x0, dtype=float)
    c = 0.0
    for k in range(len(times)):
        if k > 0:
            xt = np.asarray(propagate_cartesian(xt, times[k] - times[k - 1]),
                            dtype=float)
        if not np.all(np.isfinite(xt)):
            return float('inf')
        rr = math.hypot(xt[0], xt[1])
        if rr < R_LO or rr > R_HI:
            return float('inf')
        dx, dy = xt[0] - sats[k][0], xt[1] - sats[k][1]
        if dx * dx + dy * dy < 1.0:
            return float('inf')
        res = wrap_pi(beta_obs[k] - math.atan2(dy, dx))
        c += (res / sigma_beta) ** 2
    return c


def _bls_pass(times, sats, beta_obs, x0, sigma_beta, iters, lam0=1e-2):
    """Levenberg-damped Gauss-Newton on the target state at t0 from bearings.

    Residual r_k = wrap(beta_k - beta(x(t_k))) with x(t) the exact two-body flow
    of the iterate; the Jacobian row is (d beta / d r_k) * Phi(t_k, t_0), the
    same STM chain the filters use. Returns (x0, cost, information matrix).

    In 2D the target state has 4 dof and each observation supplies 1 scalar, so
    4 well-separated bearings are the algebraic minimum for an angles-only orbit
    — the planar analogue of Gauss's 3-observation (3 x 2 angles) method. In
    practice the arc must also be long enough for the curvature signature that
    breaks the scale symmetry, which is what the CRLB column quantifies.
    """
    x = np.array(x0, dtype=float)
    lam = lam0
    best = (x.copy(), float('inf'), np.eye(4))
    for _ in range(iters):
        Phi = np.eye(4)
        xt = x.copy()
        rows, res = [], []
        ok = True
        for k in range(len(times)):
            if k > 0:
                dt = times[k] - times[k - 1]
                Phi = stm_numerical(xt, dt) @ Phi
                xt = np.asarray(propagate_cartesian(xt, dt), dtype=float)
            rr = math.hypot(xt[0], xt[1]) if np.all(np.isfinite(xt)) else 0.0
            dx = xt[0] - sats[k][0]
            dy = xt[1] - sats[k][1]
            r2 = dx * dx + dy * dy
            if not np.isfinite(r2) or r2 < 1.0 or rr < R_LO or rr > R_HI:
                ok = False
                break
            rows.append((np.array([[-dy / r2, dx / r2, 0.0, 0.0]]) @ Phi)[0])
            res.append(wrap_pi(beta_obs[k] - math.atan2(dy, dx)))
        if not ok:
            # inadmissible iterate: retreat toward the best-so-far and re-damp
            lam *= 100.0
            x = 0.5 * (x + best[0])
            if lam > 1e12:
                break
            continue
        A = np.array(rows) / sigma_beta
        b = np.array(res) / sigma_beta
        cost = float(b @ b)
        N = A.T @ A
        if cost < best[1]:
            best = (x.copy(), cost, N)
            lam = max(lam * 0.3, 1e-8)
        else:
            lam = min(lam * 10.0, 1e12)
            x = best[0].copy()          # reject the step, retry with more damping
            continue
        try:
            step = np.linalg.solve(N + lam * np.diag(np.diag(N)) +
                                   1e-9 * np.eye(4), A.T @ b)
        except np.linalg.LinAlgError:
            break
        # Trust region on the RANGE direction, which is the one bad direction:
        # never let a single step move the epoch position by more than half the
        # current chaser-target separation.
        rho_now = math.hypot(x[0] - sats[0][0], x[1] - sats[0][1])
        sp = math.hypot(step[0], step[1])
        if sp > 0.5 * rho_now:
            step = step * (0.5 * rho_now / sp)
        # and on velocity: half the local circular speed, so one step can never
        # throw an iterate onto a hyperbolic or retrograde orbit
        vc = math.sqrt(MU / max(math.hypot(x[0], x[1]), R_LO))
        sv = math.hypot(step[2], step[3])
        if sv > 0.5 * vc:
            step = step.copy()
            step[2:] *= 0.5 * vc / sv
        x = x + step
    return best


def bls_acquire(times, sats, beta_obs, intervals, sigma_beta, sigma_v_ecc,
                window=90, grid_ratio=1.15, grid_max=160,
                iters=12, n_bins=8, cov_inflate=4.0,
                v_tang=(0.80, 1.00, 1.20), v_rad=(-0.30, 0.0, 0.30),
                extra_starts=()):
    """Angles-only acquisition: dense range grid + multi-start Gauss-Newton.

    The bearings-only cost is close to quadratic in every direction EXCEPT the
    range at epoch — that one direction carries the near-symmetry the classic
    theorems describe, and it is where all the local minima live. So:
      1. score a dense log-spaced range grid over the analytic feasible set by
         forward propagation alone (cheap, no STM, no linearization);
      2. take the n_refine lowest LOCAL minima of that 1-D cost curve;
      3. Levenberg-damped Gauss-Newton from each, keeping the best iterate;
      4. return the lowest-cost solution and its information-matrix covariance.
    Step 2 is what makes this robust where a single forward EKF pass is not: the
    range hypothesis is chosen by global cost, not by a linearization made
    before any range information exists. Step 3 re-linearizes, which no
    recursive one-pass filter can do.

    `cov_inflate` scales the returned covariance: inv(N) is the Cramer-Rao
    value, which is optimistic for a nonlinear problem at finite noise, and the
    handoff filter should not start overconfident.
    """
    w = min(window, len(times))
    t_w, s_w, b_w = times[:w], sats[:w], beta_obs[:w]
    spans = [(lo, hi, math.log(hi / lo)) for lo, hi in intervals if hi > lo]
    tot = sum(s[2] for s in spans) or 1.0

    grid = []
    for lo, hi, sp in spans:
        k = max(3, min(grid_max, int(math.ceil(sp / math.log(grid_ratio)))))
        for i in range(k):
            rho = lo * (hi / lo) ** ((i + 0.5) / k)
            init = ray_init(s_w[0], b_w[0], rho, 0.3 * rho, sigma_beta,
                            sigma_v_ecc)
            if init is None:
                continue
            # Velocity hypotheses. The circular guess is wrong by ~v_c*e, which
            # at the wide envelope (e -> 0.5) is a 50% velocity error — outside
            # any Gauss-Newton basin. Bracketing the two eccentricity degrees of
            # freedom (radial rate, tangential speed) with a coarse 3x3 lattice
            # costs one propagation sweep each and is what makes the acquisition
            # work at e >= 0.2. At e ~ 0 the extra nodes simply lose.
            px, py = init[0][0], init[0][1]
            rr = math.hypot(px, py)
            vc = math.sqrt(MU / rr)
            ur = (px / rr, py / rr)
            ut = (-py / rr, px / rr)
            for ft in v_tang:
                for fr in v_rad:
                    if abs(ft - 1.0) < 1e-9 and abs(fr) < 1e-9:
                        x0 = init[0]
                    else:
                        x0 = np.array([px, py,
                                       ft * vc * ut[0] + fr * vc * ur[0],
                                       ft * vc * ut[1] + fr * vc * ur[1]])
                    grid.append((rho, x0,
                                 _bearing_cost(t_w, s_w, b_w, x0, sigma_beta)))
    if not grid:
        return None
    # Binned multi-start: split the prior into n_refine equal LOG-range bins and
    # refine the cheapest node in each. Taking the globally cheapest nodes
    # instead systematically misses the true basin — a wrong-range node with a
    # lucky velocity often out-scores the true range with a circular velocity
    # guess, and then a "diverse" top-k selection never visits the true decade
    # at all. That failure is what produced the confidently-wrong 3800 km
    # solutions at 10 km true separation.
    lo_all = min(g[0] for g in grid)
    hi_all = max(g[0] for g in grid)
    span_all = math.log(max(hi_all / max(lo_all, 1e-9), 1.0 + 1e-9))
    nb = max(2, int(n_bins))
    bins = [None] * nb
    for g in grid:
        if not math.isfinite(g[2]):
            continue
        b = min(nb - 1, int(nb * math.log(g[0] / lo_all) / span_all))
        if bins[b] is None or g[2] < bins[b][2]:
            bins[b] = g
    starts = [b[1] for b in bins if b is not None]
    starts += [np.asarray(s, dtype=float) for s in extra_starts]
    if not starts:
        return None

    # coarse pass on every start, fine pass on the two most promising
    coarse = []
    for x0 in starts:
        coarse.append(_bls_pass(t_w, s_w, b_w, x0, sigma_beta, 3))
    coarse.sort(key=lambda r: r[1])
    sols = [_bls_pass(t_w, s_w, b_w, r[0], sigma_beta, iters)
            for r in coarse[:3] if np.isfinite(r[1])]
    sols += [r for r in coarse if np.isfinite(r[1])]
    if not sols:
        return None
    sols.sort(key=lambda r: r[1])
    x, c, N = sols[0]

    # Ambiguity margin: cost of the best solution at a MATERIALLY different
    # range. Short angles-only arcs admit several well-separated orbits that fit
    # the bearings comparably; a covariance test cannot see that (each basin is
    # locally tight), so the acquisition must report the likelihood-ratio gap.
    rho_1 = math.hypot(x[0] - s_w[0][0], x[1] - s_w[0][1])
    c2 = float('inf')
    for xs, cs, _ in sols[1:]:
        rs = math.hypot(xs[0] - s_w[0][0], xs[1] - s_w[0][1])
        if max(rs / max(rho_1, 1.0), rho_1 / max(rs, 1.0)) > 1.2:
            c2 = min(c2, cs)
    try:
        P = cov_inflate * np.linalg.inv(N + 1e-9 * np.eye(4))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
        return None
    return x, 0.5 * (P + P.T), c, w, c2 - c


def bls_acquire_adaptive(times, sats, beta_obs, intervals, sigma_beta,
                         sigma_v_ecc, w0=45, growth=1.6, gate=0.20,
                         amb_margin=16.0, **kw):
    """Grow the batch arc until the range is actually determined, then hand off.

    A fixed acquisition window is the wrong object: the arc length needed to
    break the scale near-symmetry is a function of the geometry (rho/r, whether
    the chaser burned), spanning 45 min to several orbits across this envelope.
    So batch at w0 observations, test the resulting 1-sigma LOS range
    uncertainty against `gate` (fraction of range), and if it fails, re-batch on
    a longer arc. This makes acquisition LATENCY the reported quantity instead
    of silently reporting a diverged filter — which is exactly how a flight
    angles-only system is specified.

    Two gates, both necessary and neither sufficient alone:
      chi-square  cost <= dof + 3*sqrt(2*dof), dof = w - 4. Catches convergence
                  to the WRONG range basin, which a covariance test cannot: a
                  wrong basin can be locally tight and report a small sigma.
      covariance  1-sigma LOS range uncertainty <= `gate` * range. Catches the
                  right basin found too early to be useful.
    Each longer window is additionally warm-started from the previous solution,
    so once the right basin is found it is never lost when the arc grows (the
    grid nodes themselves become useless starts on a long arc — their residuals
    are enormous and Gauss-Newton cannot descend from them).
    """
    n = len(times)
    w = min(int(w0), n)
    prev = None
    last = None
    while True:
        acq = bls_acquire(times, sats, beta_obs, intervals, sigma_beta,
                          sigma_v_ecc, window=w,
                          extra_starts=() if prev is None else (prev,), **kw)
        if acq is not None:
            x, P, c, ww, margin = acq
            last = acq[:4]
            prev = x
            dof = max(1, ww - 4)
            chi_ok = c <= dof + 3.0 * math.sqrt(2.0 * dof)
            rho_v = np.array([x[0] - sats[0][0], x[1] - sats[0][1]])
            rho = max(float(np.linalg.norm(rho_v)), 1.0)
            u = rho_v / rho
            sig = math.sqrt(max(float(u @ P[:2, :2] @ u), 0.0))
            sep_ok = margin >= amb_margin
            if chi_ok and sep_ok and sig / rho <= gate:
                return acq[:4] + (True,)
        if w >= n:
            return (last + (False,)) if last is not None else None
        w = min(n, max(w + 1, int(round(w * growth))))


# ── metrics ──────────────────────────────────────────────────────────────────
def nees_of(x, P, truth):
    err = np.asarray(x, dtype=float) - np.asarray(truth, dtype=float)
    try:
        v = float(err @ np.linalg.solve(P, err))
    except np.linalg.LinAlgError:
        return float('nan')
    return v / 4.0


def crlb_range_sigma(times, sats, tgts, sigma_beta, with_range=False,
                     sigma_rho=SIGMA_RHO_M):
    """Cramer-Rao floor on the target state estimated from the whole arc.

    FIM = sum_k H_k^T R^-1 H_k with H_k = (dz/dr_k) Phi(t_k, t_0). Phi is
    accumulated from the same numerically differenced STM the filters use, so
    this is the exact information content of the geometry — filter-independent.
    Returns (sigma along the initial LOS in m, sigma of total initial position,
    FIM condition number).
    """
    n = len(times)
    Phi = np.eye(4)
    F = np.zeros((4, 4))
    for k in range(n):
        if k > 0:
            dt = times[k] - times[k - 1]
            Phi = stm_numerical(tgts[k - 1], dt) @ Phi
        dx = tgts[k][0] - sats[k][0]
        dy = tgts[k][1] - sats[k][1]
        r2 = dx * dx + dy * dy
        Hb = np.array([[-dy / r2, dx / r2, 0.0, 0.0]]) @ Phi
        F += (Hb.T @ Hb) / sigma_beta ** 2
        if with_range:
            rho = math.sqrt(r2)
            Hr = np.array([[dx / rho, dy / rho, 0.0, 0.0]]) @ Phi
            F += (Hr.T @ Hr) / sigma_rho ** 2
    w = np.linalg.eigvalsh(F)
    cond = float(abs(w).max() / max(abs(w).min(), 1e-300))
    # A numerically singular FIM means the arc carries no range information at
    # all (the exactly degenerate co-orbital drift case). Inverting it returns a
    # meaningless tiny sigma, so report it honestly as unbounded.
    if not np.all(np.isfinite(w)) or cond > 1e16 or abs(w).min() <= 0.0:
        return float('inf'), float('inf'), cond
    try:
        C = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return float('inf'), float('inf'), cond
    u = np.array([tgts[0][0] - sats[0][0], tgts[0][1] - sats[0][1]])
    u = u / max(np.linalg.norm(u), 1e-12)
    sig_los = math.sqrt(max(float(u @ C[:2, :2] @ u), 0.0))
    sig_pos = math.sqrt(max(float(np.trace(C[:2, :2])), 0.0))
    return sig_los, sig_pos, cond


# ── scenarios ────────────────────────────────────────────────────────────────
def make_scenarios():
    """Five representative geometries spanning the shipped envelopes.

    G1/G5 are the hard corner (rho/r ~ 1e-3, the near-linear regime where the
    classic theorems bite); G3 is the headline 180-degree phase gap; G4 is the
    WL4/M5 wide-eccentric envelope. G5 = G1 plus two chaser burns, which is the
    Woffinden-Geller observability maneuver, and is what the policy actually
    does.
    """
    a_leo = R_EARTH + 400e3
    Pleo = period(a_leo)
    sc = []

    # G1 close co-orbital drift, 10 km separation, 3 orbits
    at = a_leo
    tgt = elements(at, 0.001, 0.3, 0.0)
    dtheta = 10e3 / at                                # ~10 km along-track
    sat = elements(at + 2.0e3, 0.001, 0.3, -dtheta)
    sc.append(dict(name='G1_leo_10km_drift', sat=sat, tgt=tgt, dt=60.0,
                   duration=3.0 * Pleo, r_min=R_EARTH + 300e3,
                   r_max=R_EARTH + 800e3, sigma_v_ecc=100.0, burns=()))

    # G2 mid-field, 300 km along-track + 60 km da, 2 orbits
    tgt2 = elements(at, 0.01, 1.1, 0.0)
    sat2 = elements(at + 60e3, 0.012, 1.1, -300e3 / at)
    sc.append(dict(name='G2_leo_300km', sat=sat2, tgt=tgt2, dt=60.0,
                   duration=2.0 * Pleo, r_min=R_EARTH + 300e3,
                   r_max=R_EARTH + 800e3, sigma_v_ecc=100.0, burns=()))

    # G3 headline 180-degree phase gap: separation ~ 2r ~ 13 500 km
    tgt3 = elements(at, 0.02, 0.7, 0.0)
    sat3 = elements(at + 30e3, 0.02, 0.7, math.pi)
    sc.append(dict(name='G3_leo_180deg', sat=sat3, tgt=tgt3, dt=60.0,
                   duration=1.5 * Pleo, r_min=R_EARTH + 300e3,
                   r_max=R_EARTH + 800e3, sigma_v_ecc=200.0, burns=()))

    # G4 WL4/M5 wide-eccentric: a = 12 000 km, e_t = 0.30, 90-degree gap
    a_w = 12.0e6
    tgt4 = elements(a_w, 0.30, 2.0, 0.0)
    sat4 = elements(a_w + 300e3, 0.26, 2.0, math.pi / 2.0)
    sc.append(dict(name='G4_wide_e30_90deg', sat=sat4, tgt=tgt4, dt=60.0,
                   duration=1.5 * period(a_w), r_min=R_EARTH + 300e3,
                   r_max=1.868e7, sigma_v_ecc=1500.0, burns=()))

    # G6 terminal phase at the TIGHT success box (5 km / 1 m/s), with only the
    # smallest burns in the action set (1 m/s). The question this answers is
    # whether the actuation the policy actually uses at closure is enough to keep
    # range observable, or whether angles-only nav needs a dedicated maneuver.
    tgt6 = elements(at, 0.001, 1.7, 0.0)
    sat6 = elements(at + 400.0, 0.001, 1.7, -5e3 / at)
    sc.append(dict(name='G6_leo_5km_1ms_burns', sat=sat6, tgt=tgt6, dt=60.0,
                   duration=2.0 * Pleo, r_min=R_EARTH + 300e3,
                   r_max=R_EARTH + 800e3, sigma_v_ecc=100.0,
                   burns=((1200.0, 1.0), (3600.0, -1.0))))

    # G5 = G1 plus two chaser burns (the observability maneuver the policy flies)
    sc.append(dict(name='G5_leo_10km_burns', sat=elements(at + 2.0e3, 0.001, 0.3,
                                                          -dtheta),
                   tgt=elements(at, 0.001, 0.3, 0.0), dt=60.0,
                   duration=3.0 * Pleo, r_min=R_EARTH + 300e3,
                   r_max=R_EARTH + 800e3, sigma_v_ecc=100.0,
                   burns=((1800.0, 5.0), (5400.0, -5.0))))
    return sc


# ── one filter, one geometry, one noise seed ─────────────────────────────────
def run_filter(kind, times, sats, tgts, rng, sigma_beta, sigma_v_ecc,
               r_min, r_max, K=10):
    n = len(times)
    beta_obs = [wrap_pi(bearing_of(sats[k], tgts[k]) + rng.normal(0.0, sigma_beta))
                for k in range(n)]
    rho_obs = [range_of(sats[k], tgts[k]) + rng.normal(0.0, SIGMA_RHO_M)
               for k in range(n)]

    intervals = range_prior_intervals(sats[0], beta_obs[0], r_min, r_max)
    lo, hi = prior_span(intervals)
    rho0 = math.sqrt(lo * hi)
    sig0 = (hi - lo) / math.sqrt(12.0)

    # '-o' suffix: oracle range at epoch (true rho, 10% sigma). Isolates the
    # STEADY-STATE behaviour of a filter from the acquisition problem, which is
    # the only fair way to test the modified-polar claim (its advantage is about
    # covariance consistency, not initialization).
    oracle = kind.endswith('-o')
    base = kind[:-2] if oracle else kind
    if oracle:
        rho0 = range_of(sats[0], tgts[0])
        sig0 = 0.30 * rho0

    k0 = 0            # first step at which an estimate is available at all
    acq_ok = True     # BLS arms: did the acquisition gate ever pass?
    if base == 'RB-EKF':
        f = TargetEKF(sigma_rho=SIGMA_RHO_M, sigma_beta=sigma_beta,
                      q_a=Q_ACCEL_PSD_BO, sigma_v0=sigma_v_ecc)
        f.initialize(sats[0], rho_obs[0], beta_obs[0])
        f.mean_cov = lambda ff=f: (ff.x, ff.P)
    elif base == 'BO-EKF':
        init = ray_init(sats[0], beta_obs[0], rho0, sig0, sigma_beta, sigma_v_ecc)
        f = BearingEKF(sigma_beta, Q_ACCEL_PSD_BO).set(*init)
    elif base == 'BO-UKF':
        init = ray_init(sats[0], beta_obs[0], rho0, sig0, sigma_beta, sigma_v_ecc)
        f = BearingUKF(sigma_beta, Q_ACCEL_PSD_BO).set(*init)
    elif base == 'BO-MPC':
        init = ray_init(sats[0], beta_obs[0], rho0, sig0, sigma_beta, sigma_v_ecc)
        f = BearingMPC(sigma_beta, Q_ACCEL_PSD_BO).set_cart(init[0], init[1], sats[0])
    elif base == 'BO-RPB':
        f = RangeParameterizedBank(sigma_beta, Q_ACCEL_PSD_BO, K=K)
        f.initialize(sats[0], beta_obs[0], intervals, sigma_v_ecc)
    elif base == 'BO-BLS':
        acq = bls_acquire_adaptive(times, sats, beta_obs, intervals,
                                   sigma_beta, sigma_v_ecc)
        if acq is None:
            return None
        x_b, P_b, _, w, acq_ok = acq
        # Advance the batch solution to the end of the acquisition window, then
        # hand off to the recursive bearings-only EKF. Steps < w are the batch
        # SMOOTHED solution (available only at t_w), which is why every headline
        # metric here is taken over the second half of the arc, always > w.
        f = BearingEKF(sigma_beta, Q_ACCEL_PSD_BO).set(x_b, P_b)
        for j in range(1, w):
            f.predict(times[j] - times[j - 1])
        k0 = w - 1
    elif base == 'BO-BLS-MPC':
        # The recommended composite: batch range-grid acquisition (robust to the
        # decades-wide prior and to bimodality, and re-linearizing) handed to a
        # modified-polar recursive filter (linear update, best consistency).
        acq = bls_acquire_adaptive(times, sats, beta_obs, intervals,
                                   sigma_beta, sigma_v_ecc)
        if acq is None:
            return None
        x_b, P_b, _, w, acq_ok = acq
        tmp = BearingEKF(sigma_beta, Q_ACCEL_PSD_BO).set(x_b, P_b)
        for j in range(1, w):
            tmp.predict(times[j] - times[j - 1])
        f = BearingMPC(sigma_beta, Q_ACCEL_PSD_BO).set_cart(tmp.x, tmp.P,
                                                            sats[w - 1])
        k0 = w - 1
    else:
        raise ValueError(kind)

    pe, ve, ne, rel_rho, neff = [], [], [], [], []
    diverged = False
    for k in range(k0, n):
        if k > k0:
            dt = times[k] - times[k - 1]
            if isinstance(f, BearingMPC):
                f.predict(dt, sats[k - 1], sats[k])
            else:
                f.predict(dt)
            if base == 'RB-EKF':
                f.update(sats[k], rho_obs[k], beta_obs[k])
            else:
                f.update(sats[k], beta_obs[k])
        x, P = f.mean_cov()
        if not np.all(np.isfinite(x)):
            diverged = True
            break
        err = x - np.asarray(tgts[k], dtype=float)
        pe.append(math.hypot(err[0], err[1]))
        ve.append(math.hypot(err[2], err[3]))
        ne.append(nees_of(x, P, tgts[k]))
        rt = range_of(sats[k], tgts[k])
        rh = math.hypot(x[0] - sats[k][0], x[1] - sats[k][1])
        rel_rho.append(abs(rh - rt) / max(rt, 1.0))
        neff.append(f.n_eff() if base == 'BO-RPB' else 1.0)

    m = len(pe)
    if m == 0:
        return None
    s0 = max(0, int(len(times) * SETTLE_FRAC) - k0)
    rr = np.array(rel_rho)
    conv = next((i for i in range(m) if np.all(rr[i:] < 0.05)), -1)
    nn = np.array(ne)
    finite = nn[np.isfinite(nn)]
    return dict(
        diverged=diverged or (np.array(pe)[-1] > 10.0 * range_of(sats[m - 1], tgts[m - 1])),
        pos_rmse_settled=float(np.sqrt(np.mean(np.square(pe[s0:])))),
        vel_rmse_settled=float(np.sqrt(np.mean(np.square(ve[s0:])))),
        pos_err_final=float(pe[-1]),
        rel_rho_settled=float(np.median(rr[s0:])),
        rel_rho_final=float(rr[-1]),
        nees_med=float(np.median(finite)) if finite.size else float('nan'),
        nees_in=float(np.mean((finite >= NEES_LO) & (finite <= NEES_HI)))
        if finite.size else float('nan'),
        conv_min=(times[k0 + conv] / 60.0) if conv >= 0 else float('nan'),
        acq_min=times[k0] / 60.0, acq_fail=(not acq_ok),
        n_eff_final=float(neff[-1]),
        steps=m, k0=k0,
        pe=pe, rr=list(rr),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=0)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--filters',
                    default='RB-EKF,BO-EKF,BO-UKF,BO-EKF-o,BO-MPC-o,'
                            'BO-RPB,BO-BLS,BO-BLS-MPC')
    ap.add_argument('--scenarios', default='')
    ap.add_argument('--sigma-scale', type=float, default=1.0,
                    help='multiplier on the 1 mrad bearing sigma')
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    global SIGMA_BETA, OUT_CSV, OUT_CSV_STEP
    SIGMA_BETA = SIGMA_BETA_RAD * args.sigma_scale
    if args.out_tag:
        OUT_CSV = OUT_CSV.replace('.csv', f'_{args.out_tag}.csv')
        OUT_CSV_STEP = OUT_CSV_STEP.replace('.csv', f'_{args.out_tag}.csv')

    n_seeds = args.seeds or (6 if args.quick else 24)
    kinds = args.filters.split(',')
    scs = make_scenarios()
    if args.scenarios:
        want = set(args.scenarios.split(','))
        scs = [s for s in scs if s['name'].split('_')[0] in want or s['name'] in want]

    rows, steprows = [], []
    t_start = time.time()
    for sc in scs:
        times, sats, tgts = roll_truth(sc)
        rho_t = [range_of(a, b) for a, b in zip(sats, tgts)]
        r_t = [math.hypot(b[0], b[1]) for b in tgts]
        ratio = float(np.mean([p / q for p, q in zip(rho_t, r_t)]))
        sl_b, sp_b, cond_b = crlb_range_sigma(times, sats, tgts, SIGMA_BETA, False)
        sl_r, sp_r, cond_r = crlb_range_sigma(times, sats, tgts, SIGMA_BETA, True)
        beta0 = bearing_of(sats[0], tgts[0])
        iv = range_prior_intervals(sats[0], beta0, sc['r_min'], sc['r_max'])
        lo, hi = prior_span(iv)
        print(f"\n=== {sc['name']} :: {len(times)} obs @ {sc['dt']:.0f}s, "
              f"{(times[-1]/3600.0):.2f} h ===")
        print(f"    separation rho: {min(rho_t)/1e3:.1f}-{max(rho_t)/1e3:.1f} km, "
              f"mean rho/r = {ratio:.4f}")
        print(f"    range prior from annulus: {len(iv)} interval(s), "
              f"{lo/1e3:.1f}-{hi/1e3:.1f} km  (span {hi/max(lo,1.0):.0f}x)")
        print(f"    CRLB sigma_LOS  bearings-only {sl_b:9.2f} m   "
              f"range+bearing {sl_r:8.2f} m   (ratio {sl_b/max(sl_r,1e-9):8.1f}x)")
        print(f"    FIM cond        bearings-only {cond_b:.3e}   "
              f"range+bearing {cond_r:.3e}")
        for kind in kinds:
            accs = []
            for s in range(n_seeds):
                rng = np.random.default_rng(10000 + 97 * s)
                r = run_filter(kind, times, sats, tgts, rng, SIGMA_BETA,
                               sc['sigma_v_ecc'], sc['r_min'], sc['r_max'],
                               K=args.K)
                if r is not None:
                    accs.append(r)
            if not accs:
                continue
            agg = lambda k: float(np.nanmedian([a[k] for a in accs]))
            div = float(np.mean([a['diverged'] for a in accs]))
            convs = [a['conv_min'] for a in accs]
            conv_frac = float(np.mean([not math.isnan(c) for c in convs]))
            row = dict(scenario=sc['name'], filt=kind, seeds=len(accs),
                       sigma_beta_mrad=SIGMA_BETA * 1e3,
                       rho_km_mean=float(np.mean(rho_t)) / 1e3,
                       rho_over_r=ratio,
                       crlb_los_m=sl_b, crlb_los_rb_m=sl_r,
                       pos_rmse_settled_m=agg('pos_rmse_settled'),
                       vel_rmse_settled_ms=agg('vel_rmse_settled'),
                       pos_err_final_m=agg('pos_err_final'),
                       rel_range_err_settled=agg('rel_rho_settled'),
                       rel_range_err_final=agg('rel_rho_final'),
                       nees_med=agg('nees_med'), nees_in_bounds=agg('nees_in'),
                       conv5pct_frac=conv_frac,
                       conv5pct_min=float(np.nanmedian(convs)) if conv_frac else float('nan'),
                       acq_min=agg('acq_min'),
                       acq_fail_frac=float(np.mean([a['acq_fail'] for a in accs])),
                       diverged_frac=div,
                       n_eff_final=agg('n_eff_final'))
            rows.append(row)
            print(f"  {kind:8s} posRMSE {row['pos_rmse_settled_m']:11.1f} m | "
                  f"velRMSE {row['vel_rmse_settled_ms']:8.4f} m/s | "
                  f"relRng {row['rel_range_err_settled']:.3e} | "
                  f"NEES med {row['nees_med']:8.3f} in {row['nees_in_bounds']:.2f} | "
                  f"conv {row['conv5pct_frac']*100:5.1f}% @ {row['conv5pct_min']:7.1f} min | "
                  f"acq {row['acq_min']:6.1f} min (fail {row['acq_fail_frac']:.2f}) | "
                  f"div {div:.2f}")
            # convergence trace, median seed
            med = accs[len(accs) // 2]
            for i in range(0, med['steps'], max(1, med['steps'] // 60)):
                steprows.append(dict(scenario=sc['name'], filt=kind,
                                     t_min=times[med['k0'] + i] / 60.0,
                                     pos_err_m=med['pe'][i],
                                     rel_range_err=med['rr'][i]))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_CSV_STEP, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(steprows[0].keys()))
        w.writeheader()
        w.writerows(steprows)
    print(f"\nwrote {OUT_CSV}  ({len(rows)} rows)")
    print(f"wrote {OUT_CSV_STEP}  ({len(steprows)} rows)")
    print(f"elapsed {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
