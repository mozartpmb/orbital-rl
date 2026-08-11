"""Calibrated angles-only ACQUISITION SURROGATE for filter-in-the-loop training.

Why this exists (red-team BLOCKER-1, measured). NAV-G's shipped acquisition is
a dense-range-grid batch least-squares IOD with an adaptive arc. It costs
0.23 s (median) / 1.86 s (worst) per acquisition *scalar*, and 5.5-28 s per
1024-env batch at its perfectly-vectorised lower bound — 17-90x the entire
training step, i.e. 2-9.5 h per 50M-step rung against 13.6 min for the
range+bearing reference. Vectorising it is not merely unimplemented, it is
control-flow divergent (per-node Levenberg accept/reject, per-node
admissibility break, per-env adaptive window growth, 1-or-2 prior intervals).

So training does not run BLS. It replays its *outcome*:

  1. Every episode starts UNACQUIRED. The filter is seeded blind from the
     analytic range prior (the LOS ray x altitude annulus) and the policy flies
     that estimate — no hold, no forced action, no nav-valid bit (red-team
     MAJOR-3: a hold buys 0.0 pp and corrupts 5-90% of the PPO rollout, because
     `pufferl.py:295` stores the policy's sampled action, not the executed one).
  2. Each 60 s nav tick the surrogate asks a filter-independent Cramer-Rao
     bound whether an angles-only batch solver would have converged YET, given
     the arc elapsed so far and what the chaser has ACTUALLY done inside it.
     The gate is NAV-G's own: sigma_LOS/rho <= 0.20, with a 45-tick floor equal
     to its w0=45 initial window.
  3. When it passes, the estimate is drawn as
     x_hat = x_true + N(0, Sigma),   P = 4 * Sigma
     with Sigma's LOS scale = (measured BO-BLS-MPC / CRLB ratio) x sigma_CRLB,
     the ratio read per geometry from `web_data/results/ext_bo_filter.csv`, and
     handed to the LIVE recursive modified-polar filter for the rest of the
     episode. Cost: one Gaussian draw per episode.

**The Δv conditioning is load-bearing, not cosmetic.** The Fisher information
for range is exactly singular for a *co-orbital* drifting chaser below ~50 km
separation; 1 m/s removes it and is worth 4-6 orders of magnitude. A
Δv-independent surrogate would hand the policy information it did not maneuver
for and would destroy the entire observability claim.

── Two acquisition modes, and why the default is not the table ───────────────

`acq_mode='crlb_online'` (DEFAULT). Accumulate the exact Fisher information of
the realized bearing arc, tick by tick:
    Phi_k = STM(t_k <- t_k-1) Phi_k-1,   H_k = (dbeta/dr) Phi_k,
    F    += H_k^T H_k / sigma_beta^2,    sigma_LOS = sqrt(u0^T inv(F)[:2,:2] u0)
This is *literally* `ext_bo_filter.crlb_range_sigma`, the same estimator that
generated the design map, evaluated on the geometry that actually happened.

`acq_mode='table'`. Trilinear interpolation of the shipped CRLB design map
(`ext_bo_observability.csv`, 158 rows over separation x arc x Δv x e).

The table is NOT the default, and the reason is measured. Evaluated at each
NAV-G geometry's full scenario arc, the map reproduces that scenario's own
measured CRLB to within 1.0-1.9x wherever the geometry matches the map's
parametrization — G6 5.9 m vs 6.0 m, G5 10.7 vs 11.0, G4 1801 vs 2291, G3 1684
vs 900 — and fails where it does not:

  | geometry            | map sigma_LOS | scenario's own CRLB | error   |
  |---------------------|---------------|---------------------|---------|
  | G6 5 km, 1 m/s burns| 5.9 m         | 6.0 m               | 0.98x   |
  | G5 10 km, 5 m/s     | 10.7 m        | 11.0 m              | 0.97x   |
  | G4 wide e=0.30      | 1 801 m       | 2 291 m             | 0.79x   |
  | G3 180 deg gap      | 1 684 m       | 900 m               | 1.87x   |
  | G2 300 km drift     | 6 515 m       | 711 m               | 9.2x    |
  | G1 10 km drift      | SINGULAR      | 223 m               | inf     |

The map's Δv=0 column is the CO-ORBITAL drift case: chaser and target share a
semi-major axis, the relative motion is a closed ellipse, and the FIM for range
is exactly degenerate. Our chaser is almost never co-orbital — that is the
task — so differential mean motion supplies parallax the table cannot see.
NAV-G's own G1 (10 km initial, drifting to 38 km) acquired in 114 min with a
223 m CRLB where the table says the range is never observable at all.

Using the table would therefore make the bearings-only arm catastrophic for a
reason that is an artifact of the parametrization, not of the physics, and
would measure nothing. The online CRLB keeps every property the red-team
called load-bearing — a burn enters the information matrix through exactly the
same geometry it does in the table's construction — and adds the differential
drift the table omits. It costs 8 extra propagations per tick, charged only
while a row is unacquired (~7% of ticks at the canonical policy's 680-tick
mean episode). Both modes are implemented; both are reported.
"""

import csv
import math
import os

import numpy as np

from . import nav_math as nm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))

OBS_CSV = os.path.join(_REPO, 'web_data', 'results', 'ext_bo_observability.csv')
FILT_CSV = os.path.join(_REPO, 'web_data', 'results', 'ext_bo_filter.csv')
_ALT = '/Users/pete/space_training/web_data/results'
if not os.path.exists(OBS_CSV):
    OBS_CSV = os.path.join(_ALT, 'ext_bo_observability.csv')
if not os.path.exists(FILT_CSV):
    FILT_CSV = os.path.join(_ALT, 'ext_bo_filter.csv')

MU = nm.MU

# NAV-G's shipped acquisition gate (`bls_acquire_adaptive(gate=0.20)`) and its
# initial window w0=45 observations at 60 s = 45 min. Reproducing the floor is
# what makes the surrogate's best-case latency 44-45 min, as measured.
ACQ_GATE = 0.20
ACQ_MIN_TICKS = 45

# rel_sigma placeholder for a numerically singular FIM.
_INF_REL = 1.0e3

# Settled velocity error / settled position error over all six NAV-G
# geometries: 1.13, 1.16, 0.92, 0.38, 1.11, 1.10 (x 1e-3 1/s). Five of six sit
# within 20% of the LEO mean motion n = 1.13e-3 rad/s — the settled velocity
# error is n x the settled position error, which is what a Keplerian constraint
# on the estimate implies. The velocity block is built from that rather than
# carrying its own table.


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


class ObservabilityMap:
    """Trilinear log-space interpolator over the measured CRLB design map.

    Axes: log10(separation_km) x log10(arc_orbits) x log10(1 + dv_ms).
    Values: log10(rel_sigma_bo) = log10(sigma_LOS / rho), filter-independent.

    Only the LEO (a=6771 km, e=0) block is used: it is the sole complete grid
    (7 separations x 5 arcs x 4 dv = 140 rows). The three wide/eccentric blocks
    carry 6 rows each, and NAV-G §2.3 measured that eccentricity *helps*
    far-field and is irrelevant close-in once a burn exists, so falling back to
    the circular-LEO map is the conservative direction.
    """

    def __init__(self, path=OBS_CSV, a_km=6771.0, e=0.0):
        rows = [r for r in csv.DictReader(open(path))
                if _f(r['a_km']) == a_km and _f(r['e']) == e]
        if not rows:
            raise RuntimeError(f'no observability rows for a={a_km} e={e} in {path}')
        self.sep = np.array(sorted({_f(r['sep_km']) for r in rows}))
        self.orb = np.array(sorted({_f(r['orbits']) for r in rows}))
        self.dv = np.array(sorted({_f(r['dv_ms']) for r in rows}))
        G = np.full((self.sep.size, self.orb.size, self.dv.size), np.nan)
        for r in rows:
            i = int(np.searchsorted(self.sep, _f(r['sep_km'])))
            j = int(np.searchsorted(self.orb, _f(r['orbits'])))
            k = int(np.searchsorted(self.dv, _f(r['dv_ms'])))
            v = _f(r['rel_sigma_bo'])
            if not np.isfinite(v) or v <= 0.0:
                v = _INF_REL
            G[i, j, k] = v
        G = np.where(np.isnan(G), _INF_REL, G)
        self.logG = np.log10(G)
        self.x = np.log10(self.sep)
        self.y = np.log10(self.orb)
        self.z = np.log10(1.0 + self.dv)
        self.n_rows = len(rows)

    @staticmethod
    def _frac(grid, v):
        i = np.clip(np.searchsorted(grid, v) - 1, 0, grid.size - 2)
        t = (v - grid[i]) / (grid[i + 1] - grid[i])
        return i, np.clip(t, 0.0, 1.0)

    def rel_sigma(self, sep_km, orbits, dv_ms):
        """sigma_LOS / rho an angles-only batch solver would achieve."""
        xv = np.log10(np.clip(np.asarray(sep_km, dtype=np.float64), 1e-3, None))
        yv = np.log10(np.clip(np.asarray(orbits, dtype=np.float64), 1e-4, None))
        zv = np.log10(1.0 + np.clip(np.asarray(dv_ms, dtype=np.float64), 0.0, None))
        i, tx = self._frac(self.x, xv)
        j, ty = self._frac(self.y, yv)
        k, tz = self._frac(self.z, zv)
        g = self.logG
        out = 0.0
        for di, wx in ((0, 1.0 - tx), (1, tx)):
            for dj, wy in ((0, 1.0 - ty), (1, ty)):
                for dk, wz in ((0, 1.0 - tz), (1, tz)):
                    out = out + wx * wy * wz * g[i + di, j + dj, k + dk]
        return 10.0 ** out


class BLSRatioTable:
    """Measured BO-BLS-MPC settled error / CRLB LOS sigma, per geometry.

    The CRLB is the information floor; the shipped batch-plus-recursive filter
    sits above it by a factor that depends on geometry. Interpolating that
    factor in log10(rho/r), split by whether the chaser burned during the arc,
    converts a CRLB into a realistic angles-only error.

    Measured (`web_data/results/ext_bo_filter.csv`, 24 noise seeds/cell):
      drift: G1 rho/r 0.0057 -> 5.50   G2 0.127 -> 1.73
             G4 0.841 -> 1.97          G3 2.004 -> 3.62
      burn:  G6 0.0022 -> 2.95         G5 0.0118 -> 5.16
    """

    _BURN_SCEN = ('G5_leo_10km_burns', 'G6_leo_5km_1ms_burns')

    def __init__(self, path=FILT_CSV, filt='BO-BLS-MPC'):
        drift, burn = [], []
        self.rows = {}
        for r in csv.DictReader(open(path)):
            if r['filt'] != filt:
                continue
            rr = _f(r['rho_over_r'])
            ratio = _f(r['pos_rmse_settled_m']) / max(_f(r['crlb_los_m']), 1e-9)
            self.rows[r['scenario']] = dict(
                rho_over_r=rr, ratio=ratio, crlb=_f(r['crlb_los_m']),
                pos=_f(r['pos_rmse_settled_m']), vel=_f(r['vel_rmse_settled_ms']),
                nees_ib=_f(r['nees_in_bounds']), acq_min=_f(r['acq_min']),
                rho_km=_f(r['rho_km_mean']),
                burn=r['scenario'] in self._BURN_SCEN)
            (burn if r['scenario'] in self._BURN_SCEN else drift).append((rr, ratio))
        if not drift or not burn:
            raise RuntimeError(f'incomplete ratio table in {path}')
        drift.sort(); burn.sort()
        self.drift = (np.array([d[0] for d in drift]),
                      np.array([d[1] for d in drift]))
        self.burn = (np.array([b[0] for b in burn]),
                     np.array([b[1] for b in burn]))

    def ratio(self, rho_over_r, has_burn):
        lr = np.log10(np.clip(np.asarray(rho_over_r, dtype=np.float64), 1e-6, None))
        rd = np.interp(lr, np.log10(self.drift[0]), self.drift[1])
        rb = np.interp(lr, np.log10(self.burn[0]), self.burn[1])
        return np.where(np.asarray(has_burn, dtype=bool), rb, rd)


class AcquisitionSurrogate:
    """Per-row acquisition state machine + calibrated error draw."""

    def __init__(self, n, sigma_beta, gate=ACQ_GATE, min_ticks=ACQ_MIN_TICKS,
                 cov_inflate=4.0, mode='crlb_online',
                 obs_csv=OBS_CSV, filt_csv=FILT_CSV):
        if mode not in ('crlb_online', 'table'):
            raise ValueError(f"acq mode must be crlb_online|table, got {mode!r}")
        self.n = n
        self.mode = mode
        self.sigma_beta = float(sigma_beta)
        self.gate = float(gate)
        self.min_ticks = int(min_ticks)
        self.cov_inflate = float(cov_inflate)
        self.ratios = BLSRatioTable(filt_csv)
        self.map = ObservabilityMap(obs_csv) if mode == 'table' else None

        self.acquired = np.zeros(n, dtype=bool)
        self.ticks = np.zeros(n, dtype=np.int64)
        self.dv = np.zeros(n)                # commanded |dv| inside the arc
        self.sep0 = np.ones(n)               # epoch separation (m)
        self.u0 = np.zeros((n, 2))           # epoch LOS unit vector
        self.period = np.full(n, 5545.0)     # chaser orbital period (s)
        self.Phi = np.tile(np.eye(4), (n, 1, 1))
        self.FIM = np.zeros((n, 4, 4))
        self.n_acq = 0
        self.n_reset = 0
        self.last_sigma_los = np.full(n, np.inf)
        # Set to a list to record (sigma_LOS, predicted sigma_pos, realised
        # |dpos|, rho, n_obs) at every handoff. None in training (the list
        # would grow without bound over 50M steps).
        self.log_draws = None

    # -- episode boundaries ---------------------------------------------------
    def reset_rows(self, idx, sat_cart, tgt_cart, period_s):
        if idx.size == 0:
            return
        d = tgt_cart[:, :2] - sat_cart[:, :2]
        sep = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1.0)
        self.acquired[idx] = False
        self.ticks[idx] = 0
        self.dv[idx] = 0.0
        self.sep0[idx] = sep
        self.u0[idx] = d / sep[:, None]
        self.period[idx] = np.maximum(period_s, 60.0)
        self.Phi[idx] = np.eye(4)
        self.FIM[idx] = 0.0
        self.last_sigma_los[idx] = np.inf
        self.n_reset += int(idx.size)

    def add_dv(self, idx, dv):
        """Accumulate the Δv the chaser commanded inside the acquisition arc."""
        if idx.size == 0:
            return
        m = ~self.acquired[idx]
        if m.any():
            self.dv[idx[m]] += dv[m]

    # -- information accumulation --------------------------------------------
    def accumulate(self, idx, sat_now, tgt_now, tgt_prev, dt, first=False):
        """One 60 s tick of bearing information for the unacquired rows.

        `sat_now`/`tgt_now`/`tgt_prev` are TRUTH states already subset to `idx`;
        `first` marks the epoch observation (no state transition yet). Mirrors
        `ext_bo_filter.crlb_range_sigma` exactly, on the realized geometry.
        """
        if idx.size == 0:
            return
        sel = ~self.acquired[idx]
        if not sel.any():
            return
        pend = idx[sel]
        self.ticks[pend] += 1
        if self.mode != 'crlb_online':
            return
        if not first:
            F, _ = nm.stm_fd(tgt_prev[sel], dt)
            self.Phi[pend] = F @ self.Phi[pend]
        sat, tgt = sat_now[sel], tgt_now[sel]
        dx = tgt[:, 0] - sat[:, 0]
        dy = tgt[:, 1] - sat[:, 1]
        r2 = np.maximum(dx * dx + dy * dy, 1.0)
        Hr = np.zeros((pend.size, 1, 4))
        Hr[:, 0, 0] = -dy / r2
        Hr[:, 0, 1] = dx / r2
        H = Hr @ self.Phi[pend]
        self.FIM[pend] += (np.swapaxes(H, 1, 2) @ H) / self.sigma_beta ** 2

    # -- gate -----------------------------------------------------------------
    def _sigma_los_online(self, pend):
        F = self.FIM[pend]
        w = np.linalg.eigvalsh(F)
        wmin = np.abs(w).min(axis=1)
        wmax = np.abs(w).max(axis=1)
        bad = (wmin <= 0.0) | (wmax / np.maximum(wmin, 1e-300) > 1e16) \
            | ~np.isfinite(w).all(axis=1)
        Fs = np.where(bad[:, None, None], np.eye(4), F)
        C = np.linalg.inv(Fs + 1e-30 * np.eye(4))
        u = self.u0[pend]
        v = np.einsum('ni,nij,nj->n', u, C[:, :2, :2], u)
        sig = np.sqrt(np.maximum(v, 0.0))
        return np.where(bad, np.inf, sig)

    def ready(self, idx, dt=60.0):
        """Rows whose angles-only batch solver would have converged by now.

        Returns (rows, sigma_LOS at those rows).
        """
        if idx.size == 0:
            return idx[:0], np.zeros(0)
        pend = idx[~self.acquired[idx]]
        pend = pend[self.ticks[pend] >= self.min_ticks]
        if pend.size == 0:
            return pend, np.zeros(0)
        if self.mode == 'crlb_online':
            sig = self._sigma_los_online(pend)
        else:
            orbits = (self.ticks[pend] * dt) / self.period[pend]
            rel = self.map.rel_sigma(self.sep0[pend] / 1e3, orbits, self.dv[pend])
            sig = rel * self.sep0[pend]
        self.last_sigma_los[pend] = sig
        ok = sig <= self.gate * self.sep0[pend]
        return pend[ok], sig[ok]

    # -- handoff --------------------------------------------------------------
    def draw(self, idx, sigma_crlb, x_true, sat_cart, rng):
        """x_hat ~ N(x_true, Sigma), P = cov_inflate * Sigma.

        Sigma is anisotropic in position: the LOS component carries the batch
        solver's range error (CRLB x the measured BLS ratio); the transverse
        component is measurement-limited, but at the ARC-AVERAGED bearing
        precision rho*sigma_beta/sqrt(N), not the single-sample rho*sigma_beta.
        The velocity block is n_target x sigma_LOS.

        The sqrt(N) matters and is calibrated, not assumed. At the 180-degree
        phase gap rho = 13,566 km, so a single bearing sample is 13.6 km of
        transverse position — four times the measured BO-BLS-MPC settled error
        of 3,257 m, which is impossible for an estimator that saw 45+ bearings
        (T4 §8.2 independently measured the filter 795x better than a single
        bearing sample at Mm separations). With the sqrt(N) term the predicted
        total sigma reproduces all six measured geometries within 1.0-1.2x.
        """
        m = idx.size
        dx = x_true[:, 0] - sat_cart[:, 0]
        dy = x_true[:, 1] - sat_cart[:, 1]
        rho_now = np.maximum(np.hypot(dx, dy), 1.0)
        r_t = np.maximum(np.hypot(x_true[:, 0], x_true[:, 1]), 1.0)

        has_burn = self.dv[idx] > 0.0
        ratio = self.ratios.ratio(self.sep0[idx] / r_t, has_burn)
        sig_los = np.maximum(ratio * sigma_crlb, 1e-3)

        if self.mode == 'crlb_online':
            # Sigma = ratio^2 * Phi inv(F) Phi^T. inv(F) IS the batch solver's
            # covariance (NAV-G hands off cov_inflate * inv(N), the same
            # object), so this carries the full position-velocity and
            # LOS-transverse correlation structure of the realized arc instead
            # of an assumed block form; Phi maps it from the epoch, where the
            # information was accumulated, to now, where it is handed over.
            Sig = self._sigma_from_fim(idx, ratio)
        else:
            n_obs = np.maximum(self.ticks[idx], 1)
            sig_tr = np.maximum(rho_now * self.sigma_beta / np.sqrt(n_obs), 1e-3)
            ux, uy = dx / rho_now, dy / rho_now
            nx, ny = -uy, ux
            Sig = np.zeros((m, 4, 4))
            Sig[:, 0, 0] = sig_los ** 2 * ux * ux + sig_tr ** 2 * nx * nx
            Sig[:, 0, 1] = sig_los ** 2 * ux * uy + sig_tr ** 2 * nx * ny
            Sig[:, 1, 0] = Sig[:, 0, 1]
            Sig[:, 1, 1] = sig_los ** 2 * uy * uy + sig_tr ** 2 * ny * ny
            n_t = np.sqrt(MU / r_t ** 3)
            sig_v = n_t * sig_los
            Sig[:, 2, 2] = sig_v ** 2
            Sig[:, 3, 3] = sig_v ** 2

        # Correlated draw via Cholesky, with a jitter ladder: inv(F) on a short
        # arc can be numerically indefinite.
        z = rng.standard_normal((m, 4))
        L = self._chol(Sig)
        x_hat = x_true + np.einsum('nij,nj->ni', L, z)

        # Handoff telemetry: predicted 1-sigma position vs the error actually
        # realised, so the draw itself is falsifiable rather than assumed.
        if self.log_draws is not None:
            pred = np.sqrt(np.maximum(Sig[:, 0, 0] + Sig[:, 1, 1], 0.0))
            real = np.hypot(x_hat[:, 0] - x_true[:, 0], x_hat[:, 1] - x_true[:, 1])
            for a, b, c, d, e in zip(sig_los, pred, real, rho_now,
                                     self.ticks[idx]):
                self.log_draws.append((float(a), float(b), float(c),
                                       float(d), int(e)))

        self.acquired[idx] = True
        self.n_acq += m
        return x_hat, self.cov_inflate * Sig, sig_los

    def _sigma_from_fim(self, idx, ratio):
        F = self.FIM[idx]
        w, V = np.linalg.eigh(0.5 * (F + np.swapaxes(F, 1, 2)))
        # Floor the spectrum at 1e-12 of the largest eigenvalue: a direction
        # the arc carries no information about is bounded by the prior, not by
        # infinity.
        wmax = np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        w = np.maximum(w, 1e-12 * wmax)
        C = (V * (1.0 / w)[:, None, :]) @ np.swapaxes(V, 1, 2)
        Phi = self.Phi[idx]
        S = Phi @ C @ np.swapaxes(Phi, 1, 2)
        S = 0.5 * (S + np.swapaxes(S, 1, 2)) * (ratio ** 2)[:, None, None]
        return S

    @staticmethod
    def _chol(S):
        n = S.shape[0]
        d = np.einsum('nii->ni', S)
        scale = np.maximum(d.max(axis=1), 1e-30)
        for jit in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
            try:
                return np.linalg.cholesky(
                    S + (jit * scale)[:, None, None] * np.eye(4))
            except np.linalg.LinAlgError:
                continue
        # Per-row fallback: diagonal only.
        L = np.zeros_like(S)
        idx = np.arange(4)
        L[:, idx, idx] = np.sqrt(np.maximum(d, 0.0))
        return L


# ── calibration report (V4 deliverable) ──────────────────────────────────────
def map_vs_scenario_crlb():                               # pragma: no cover
    """Table quoted in this module's docstring: the design map evaluated at each
    NAV-G scenario's full arc, against that scenario's own measured CRLB."""
    m = ObservabilityMap()
    t = BLSRatioTable()
    geoms = [('G6_leo_5km_1ms_burns', 5.0, 3.0, 1.0),
             ('G5_leo_10km_burns', 10.0, 3.0, 5.0),
             ('G4_wide_e30_90deg', 9771.0, 3.0, 0.0),
             ('G3_leo_180deg', 13000.0, 3.0, 0.0),
             ('G2_leo_300km', 300.0, 3.0, 0.0),
             ('G1_leo_10km_drift', 10.0, 3.0, 0.0)]
    print(f"{'geometry':24s} {'map sigma_LOS':>15s} {'scenario CRLB':>15s} "
          f"{'map/CRLB':>10s} {'BLS/CRLB':>10s}")
    for name, sep, orb, dv in geoms:
        r = t.rows[name]
        sig = float(m.rel_sigma(sep, orb, dv)) * sep * 1e3
        tag = 'SINGULAR' if sig > 1e6 else f'{sig:13.1f} m'
        rat = 'inf' if sig > 1e6 else f'{sig / r["crlb"]:10.2f}'
        print(f"{name:24s} {tag:>15s} {r['crlb']:13.1f} m {rat:>10s} "
              f"{r['ratio']:10.2f}")


if __name__ == '__main__':                                # pragma: no cover
    map_vs_scenario_crlb()
