"""The 19-slot estimated-state observation encode for `dim3_mode=1` + nav.

n3d_REDTEAM BLOCKER-4. Under `dim3_mode=1` nineteen observation slots are
target-derived. The shipped 2D wrapper rebuilds twelve. The seven it does not
rebuild are precisely the 3D task's state, and every one of them would hand the
policy TRUTH while the write-up said angles-only:

    21, 22  the true relative-inclination vector and relative-node phase
            — the burn-timing signal
    23      true (e_s - e_t) . h_t
    24, 25  true cross-track LVLH position and velocity
    26      the true PLANE-CHANGE DELTA-V-TO-GO
    28      true feasibility margin

Not one shipped `nav_*` diagnostic would notice: `nav_pos_rmse`,
`nav_nees_med`, `nav_clip_rate`, `nav_diverge_rate` and
`nav_sigma_los_over_rho` all measure the FILTER, not the OBSERVATION, so they
stay green while the policy flies the plane leg on truth. The gate that closes
this is `leakcheck` (verify_n3dnav stage c1): garbage the estimate, assert all
19 slots move and the 19-slot complement does not.

Layout (MAJOR-14, definitive):

    0-6, 9, 10, 15, 17-20, 27    chaser-truth / Earth-conjunction / clock.
                                 NEVER written by the wrapper.
    7, 8, 11-14, 16, 33-37       target-derived, the existing 12
    21-28 (minus 27)             the C's ext-3d block, rebuilt, clamped at +/-2
    29-32                        stay 0.0f. Reserved.

The Sigma-channel is DROPPED from the layout: NAV-F measured `T-BO+Sigma`
statistically identical to `T-BO` on every metric, so it has no claim on a
slot, and deleting it deletes the obs[21] collision outright instead of
relocating it. If a later arm wants it back it goes on obs[29] (see
`OrbitalNav._write_sigma_channel`).
"""

import numpy as np

from . import nav_math as nm
from . import nav_math3d as n3

# The 19 target-derived slots under dim3_mode=1, derived from
# orbital.h::fill_observations.
TARGET_SLOTS_3D = (7, 8, 11, 12, 13, 14, 16,
                   21, 22, 23, 24, 25, 26, 28,
                   33, 34, 35, 36, 37)

# Slots the wrapper must NEVER touch. (b) of the leakcheck gate: an over-broad
# rebuild is as damaging as a missed one — overwriting obs[15] hands the policy
# a fake mission deadline, overwriting obs[27] corrupts the chaser-only
# delta-v ledger.
COMPLEMENT_SLOTS_3D = (0, 1, 2, 3, 4, 5, 6, 9, 10, 15,
                       17, 18, 19, 20, 27, 29, 30, 31, 32)

# The subset the C clamps at +/-2 via obs_clamp2 (orbital.h:907). The rest are
# written raw, so a single uniform nav_clip breaks `recon == truth` whenever
# the C clamp fires (measured 3.24 mid-episode at the widest rung).
CLAMP2_SLOTS = (21, 22, 23, 24, 25, 26, 28)

assert len(TARGET_SLOTS_3D) == 19
assert len(COMPLEMENT_SLOTS_3D) == 19
assert set(TARGET_SLOTS_3D) | set(COMPLEMENT_SLOTS_3D) == set(range(38))
assert not (set(TARGET_SLOTS_3D) & set(COMPLEMENT_SLOTS_3D))


def obs_clamp2(v):
    """orbital.h::obs_clamp2, including its NaN trap, exactly.

        if (!(v > -2.0)) return (v > 0.0) ? 2.0f : -2.0f;   // also traps NaN
        if (v > 2.0) return 2.0f;
        return (float)v;

    NaN takes the first branch (`NaN > -2.0` is false) and then `NaN > 0.0` is
    false, so NaN maps to -2.0 — not to +2.0 and not to 0.0.
    """
    v = np.asarray(v, dtype=np.float64)
    lo = ~(v > -2.0)
    return np.where(lo, np.where(v > 0.0, 2.0, -2.0), np.minimum(v, 2.0))


class Encoder3D:
    """Rebuild the 19 target-derived observation slots from a target ESTIMATE.

    One function, used identically by `recon` (estimate := truth) and by the
    live filter modes. That is deliberate: an encode that special-cases recon
    is an encode whose exactness gate does not test the shipped path.
    """

    def __init__(self, n, obs_alt_scale_m, lvlh_scale_m,
                 di_max_rad=-1.0, obs_di_scale_rad=-1.0,
                 de_max=-1.0, obs_de_scale=-1.0,
                 shape_dv_ref_ms=300.0, clip=4.0):
        self.n = int(n)
        self.alt_scale = float(obs_alt_scale_m)
        self.lvlh_scale = float(lvlh_scale_m)
        # orbital.h:1105-1111, verbatim.
        self.di_scale = (float(obs_di_scale_rad) if obs_di_scale_rad > 0.0
                         else max(di_max_rad if di_max_rad > 0.0 else 0.0,
                                  0.25 * np.pi / 180.0))
        self.de_scale = (float(obs_de_scale) if obs_de_scale > 0.0
                         else max(de_max if de_max > 0.0 else 0.0, 0.05))
        self.dv_ref = float(shape_dv_ref_ms) if shape_dv_ref_ms > 0.0 else 300.0
        self.clip = float(clip)
        self.slots = np.array(TARGET_SLOTS_3D)
        self.clamp2 = np.array(CLAMP2_SLOTS)
        self._is_clamp2 = np.isin(self.slots, self.clamp2)
        self.scratch = np.zeros((self.n, nm.OBS_DIM), dtype=np.float64)
        # Diagnostics (NOTE-21): fraction of decisions with obs[28] < 0.
        self.last_feas_neg = None

    # -- the truth side, read off vec_get_state -------------------------------
    @staticmethod
    def chaser(st):
        """Views into the (N,36) getter block. CHASER ONLY — self-knowledge."""
        return dict(a=st[:, 0], e=st[:, 1], M=st[:, 2], theta=st[:, 3],
                    omega=st[:, 4], hhat=st[:, 5:8], r=st[:, 8:11],
                    v=st[:, 11:14], fuel_frac=st[:, 14], evec=st[:, 30:33],
                    cart=np.concatenate([st[:, 8:11], st[:, 11:14]], axis=1))

    @staticmethod
    def target_truth(st):
        """Views into the target half. Legitimate consumers are exactly two:
        `nav_mode='recon'` (estimate := truth by construction) and the
        surrogate's NOISE-PROCESS conditioning (MAJOR-17). Never the encode."""
        return dict(a=st[:, 15], e=st[:, 16], M=st[:, 17], theta=st[:, 18],
                    omega=st[:, 19], hhat=st[:, 20:23], r=st[:, 23:26],
                    v=st[:, 26:29], evec=st[:, 33:36],
                    cart=np.concatenate([st[:, 23:26], st[:, 26:29]], axis=1))

    # -- the encode -----------------------------------------------------------
    def values(self, st, xt):
        """Compute the 19 slot values. Returns the (N,38) scratch buffer.

        `st` is the (N,36) truth block; ONLY its chaser half is read.
        `xt` is the (N,6) target-estimate inertial Cartesian state.
        """
        out = self.scratch
        s = self.chaser(st)
        rt, vt = xt[:, :3], xt[:, 3:6]

        with np.errstate(all='ignore'):
            r_t = np.maximum(n3._norm(rt), 1.0)
            v2 = np.einsum('ni,ni->n', vt, vt)
            inv_a = 2.0 / r_t - v2 / n3.MU
            a_t = 1.0 / np.where(np.abs(inv_a) < 1e-300, 1e-300, inv_a)
            hvec = n3.cross(rt, vt)
            h_t = n3.unit(hvec)
            e_vec = n3.evec_from_cartesian(xt)
            e_t = n3._norm(e_vec)
            # obs[11,12,16]: reproduce the VALUE the C writes, which is
            # `target.omega` under i_t = RAAN_t = 0 — i.e. the INERTIAL
            # periapsis longitude. Never compute it from the node: an estimate
            # carries ~1e-6 rad of inclination noise, so its node direction is
            # random and a node-referenced omega-hat is garbage. C has the same
            # e -> 0 degeneracy, so match it (omega := 0 below 1e-9).
            varpi_t = np.where(e_t < 1e-9, 0.0,
                               np.arctan2(e_vec[:, 1], e_vec[:, 0]))
            a_eff = np.where(a_t > 0.0, a_t, r_t)
            a_eff = np.maximum(a_eff, 1.0)
            v_c_t = np.sqrt(n3.MU / a_eff)
            n_t = np.sqrt(n3.MU / (a_eff * a_eff * a_eff))   # C: a*a*a, not a**3

            out[:, 7] = (a_t - n3.R_EARTH) / self.alt_scale
            out[:, 8] = e_t
            out[:, 11] = np.sin(varpi_t)
            out[:, 12] = np.cos(varpi_t)
            out[:, 16] = np.cos(s['omega'] - varpi_t)

            # obs[13,14]: mean-longitude gap in the gauge of the ESTIMATED
            # target plane. NON-ISSUE-18: the gauge substitution is safe, gain
            # tan(di_rel/2) sin(psi - Omega), which vanishes exactly where
            # guidance drives.
            el_t = n3.cartesian_to_elements_3d(xt)
            g = n3.PlaneGauge(el_t['inc'], el_t['raan'])
            lam_s = n3.lambda_gauge(g, s['cart'], s['M'])
            lam_t = n3.lambda_gauge(g, xt, el_t['M'])
            dlam = lam_s - lam_t
            out[:, 13] = np.sin(dlam)
            out[:, 14] = np.cos(dlam)

            # ── ext-3d block, orbital.h:1076-1149 ───────────────────────────
            h_s = s['hhat']
            c = n3.cross(h_t, h_s)
            cn = n3._norm(c)
            hdot = np.einsum('ni,ni->n', h_t, h_s)
            di_rel = np.arctan2(cn, hdot)
            di_vec = np.where((cn > 1e-300)[:, None],
                              (di_rel / np.where(cn > 1e-300, cn, 1.0))[:, None] * c,
                              0.0)
            Rh = n3.unit(s['r'])
            Th = n3.cross(h_s, Rh)
            out[:, 21] = np.einsum('ni,ni->n', di_vec, Rh) / self.di_scale
            out[:, 22] = np.einsum('ni,ni->n', di_vec, Th) / self.di_scale

            de_vec = s['evec'] - e_vec
            out[:, 23] = np.einsum('ni,ni->n', de_vec, h_t) / self.de_scale

            rho = s['r'] - rt
            rhod = s['v'] - vt
            out[:, 24] = np.einsum('ni,ni->n', rho, h_t) / self.lvlh_scale
            out[:, 25] = np.einsum('ni,ni->n', rhod, h_t) / v_c_t

            dh = h_s - h_t
            dv_pl = v_c_t * n3._norm(dh)
            da_rel = (s['a'] - a_t) / a_t
            de3 = n3._norm(de_vec)
            dv_in = 0.5 * v_c_t * np.sqrt(da_rel * da_rel + de3 * de3)
            # dv_rem = VE ln(m/m_dry); m/m_dry = 1/(1 - fuel_frac).
            dv_rem = -n3.VE * np.log(np.maximum(1.0 - s['fuel_frac'], 1e-300))
            out[:, 26] = dv_pl / self.dv_ref
            out[:, 28] = (dv_rem - dv_pl - dv_in) / self.dv_ref
            self.last_feas_neg = out[:, 28] < 0.0

            # ── in-plane LVLH block, orbital.h:1152-1196 ────────────────────
            # theta_t via atan2 on the ESTIMATE's Cartesian position, not
            # (theta + omega): the C's element route equals the Cartesian one
            # only because i_t = RAAN_t = 0, and a tilted estimate diverges.
            theta_t = np.arctan2(rt[:, 1], rt[:, 0])
            ct, stt = np.cos(theta_t), np.sin(theta_t)
            dxi, dyi = rho[:, 0], rho[:, 1]
            dvxi, dvyi = rhod[:, 0], rhod[:, 1]
            dx_l = ct * dxi + stt * dyi
            dy_l = -stt * dxi + ct * dyi
            dvx_l = ct * dvxi + stt * dvyi
            dvy_l = -stt * dvxi + ct * dvyi
            dvx_l = dvx_l + n_t * dy_l          # C: dvx_l += n_tgt * dy_l
            dvy_l = dvy_l - n_t * dx_l
            out[:, 33] = dx_l / self.lvlh_scale
            out[:, 34] = dy_l / self.lvlh_scale
            out[:, 35] = dvx_l / v_c_t
            out[:, 36] = dvy_l / v_c_t
            out[:, 37] = n_t / 1e-3
        return out

    def write(self, obs, st, xt):
        """Encode and write into `obs` (N,38 float32) in place.

        Returns (n_clipped, n_total) for the `nav_clip_rate` diagnostic.
        """
        out = self.values(st, xt)
        v = out[:, self.slots].copy()
        c = self.clip
        v = np.nan_to_num(v, nan=0.0, posinf=c, neginf=-c)
        n_clip = int(np.count_nonzero(np.abs(v) > c))
        np.clip(v, -c, c, out=v)
        # MAJOR-14: the C applies obs_clamp2 (+/-2) to 21-28 ONLY. A uniform
        # nav_clip of 4.0 makes recon != truth the moment the C clamp fires.
        raw = out[:, self.slots]
        v[:, self._is_clamp2] = obs_clamp2(raw[:, self._is_clamp2])
        obs[:, self.slots] = v.astype(np.float32)
        return n_clip, v.size
