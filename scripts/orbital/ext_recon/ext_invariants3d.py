#!/usr/bin/env python3
"""ext-3d — the 3D invariant battery, plus a MUTATION TEST that proves it works.

Two things live here:

1. `INVARIANTS` — the checks a 3D fuzz harness runs on a trajectory log.  Each
   is a pure function of (oracle, trajectory) so `fuzz_dynamics3d.py` can call
   the identical code against the real C env once it exists.

2. A mutation test.  An invariant list is worthless until you show it FIRES.
   `Model3D` is a deliberately naive reference "3D env" — classical elements as
   state, mean-anomaly propagation, cartesian_to_elements after each burn,
   i.e. the exact architecture the current 2D `orbital.h` would grow into.  Its
   internals can be swapped for known-wrong variants drawn from a catalogue of
   3D bug classes (`BUGS`).  Running every invariant against every mutant
   produces a detection matrix: which check catches which bug, and how loudly.
   Any bug with an empty row is a hole in the battery, and any invariant with
   an empty column is dead weight.

Run:
    python3 /Users/pete/space_training/scripts/orbital/ext_recon/ext_invariants3d.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbital_math3d as o3                                       # noqa: E402
from orbital_math3d import MU, R_EARTH, unit, wrap_pi, wrap_2pi   # noqa: E402

DT = 60.0
ISP, G0 = 300.0, 9.80665
VE = ISP * G0


# ═══════════════════════════════════════════════════════════════════════════
# A naive reference 3D environment, mutable
# ═══════════════════════════════════════════════════════════════════════════
class Model3D:
    """Classical-element 3D env.  `bug=None` is the correct implementation."""

    def __init__(self, el, bug=None, dry_mass=850.0, fuel_frac=0.15):
        self.el = dict(el)
        self.bug = bug
        self.dry = dry_mass
        self.fuel = dry_mass * fuel_frac / (1.0 - fuel_frac)

    # ── elements -> Cartesian ────────────────────────────────────────────
    def coe2rv(self, el):
        i, O, w = el['i'], el['raan'], el['argp']
        if self.bug == 'B6_R1_sign':
            i = -i                       # R1(+i) instead of R1(-i): mirrored plane
        if self.bug == 'B7_rot_order':
            O, w = w, O                  # 3-1-3 sequence applied backwards
        return o3.coe2rv(el['a'], el['e'], i, O, w, el['nu'])

    # ── Cartesian -> elements ────────────────────────────────────────────
    def rv2coe(self, rv):
        rvec, vvec = np.asarray(rv[:3]), np.asarray(rv[3:])
        r = float(np.linalg.norm(rvec))
        v = float(np.linalg.norm(vvec))
        hvec = np.cross(rvec, vvec)
        if self.bug == 'B8_h_reversed':
            hvec = np.cross(vvec, rvec)  # v x r: plane normal points the wrong way
        h = float(np.linalg.norm(hvec))
        evec = ((v * v - MU / r) * rvec - float(np.dot(rvec, vvec)) * vvec) / MU
        e = float(np.linalg.norm(evec))
        a = 1.0 / (2.0 / r - v * v / MU)
        i = math.acos(max(-1.0, min(1.0, hvec[2] / h)))
        nvec = np.cross([0.0, 0.0, 1.0], hvec)
        nn = float(np.linalg.norm(nvec))

        if nn < 1e-9 * h:
            raan, nhat = 0.0, np.array([1.0, 0.0, 0.0])
        elif self.bug == 'B2_raan_acos':
            raan = math.acos(max(-1.0, min(1.0, nvec[0] / nn)))   # no n_y sign fix
            nhat = nvec / nn
        else:
            raan = wrap_2pi(math.atan2(nvec[1], nvec[0]))
            nhat = nvec / nn

        if e < 1e-11:
            argp, ehat = 0.0, nhat
        elif self.bug == 'B3_argp_acos':
            ehat = evec / e
            argp = math.acos(max(-1.0, min(1.0, float(np.dot(nhat, ehat)))))
            # no e_z sign fix
        else:
            ehat = evec / e
            argp = wrap_2pi(o3.signed_angle(nhat, ehat, hvec))

        if self.bug == 'B4_nu_acos':
            nu = math.acos(max(-1.0, min(1.0,
                                         float(np.dot(ehat, rvec)) / r)))
            # no r.v sign fix
        else:
            nu = wrap_2pi(o3.signed_angle(ehat, rvec, hvec))

        out = dict(a=a, e=e, i=i, raan=raan, argp=argp, nu=nu)
        if self.bug == 'B10_stale_plane':
            out['i'] = self.el['i']      # plane elements never updated by a burn
            out['raan'] = self.el['raan']
        return out

    # ── anomaly conversions ──────────────────────────────────────────────
    def nu_to_M(self, nu, e):
        if self.bug == 'B5_half_angle':
            # the 2026-08-10 bug's 3D twin: forward map used as its own inverse
            E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * nu),
                                 math.sqrt(1.0 + e) * math.cos(0.5 * nu))
            E = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * E),
                                 math.sqrt(1.0 - e) * math.cos(0.5 * E))
            return E - e * math.sin(E)
        return o3.nu_to_M(nu, e)

    # ── burn frame ───────────────────────────────────────────────────────
    def frame(self, rv):
        pro = unit(rv[3:])
        if self.bug == 'B13_radial_inertial':
            # "2D code lifted to 3D": radial taken as the in-plane-looking
            # x-y projection of the position, normalised in 3D.  Identical to
            # r_hat for an equatorial orbit, out-of-plane for anything else.
            rad = unit(np.array([rv[0], rv[1], 0.0]))
        else:
            rad = unit(rv[:3])
        if self.bug == 'B1_normal_is_z':
            nor = np.array([0.0, 0.0, 1.0])          # z_hat instead of h_hat
        else:
            nor = unit(np.cross(rv[:3], rv[3:]))
            if self.bug == 'B12_normal_north':
                # "make the normal axis always point north" -- a convenience
                # that silently reverses the sense of every plane-change burn
                # on a RETROGRADE orbit, and is a perfect no-op on prograde
                # ones.  Only the retrograde scenarios can expose it.
                if nor[2] < 0.0:
                    nor = -nor
        return pro, rad, nor

    # ── one sub-step: optional impulse, then DT of coast ─────────────────
    def step(self, dv_pro=0.0, dv_rad=0.0, dv_nor=0.0, dt=DT):
        rv_pre = self.coe2rv(self.el)
        pro, rad, nor = self.frame(rv_pre)
        dv_mag = 0.0
        if abs(dv_pro) + abs(dv_rad) + abs(dv_nor) > 0.0:
            dvv = dv_pro * pro + dv_rad * rad + dv_nor * nor
            dv_mag = float(np.linalg.norm(dvv))
            # Tsiolkovsky with the same fuel clamp as orbital.h apply_impulse
            m = self.dry + self.fuel
            need = m * (1.0 - math.exp(-dv_mag / VE))
            if need > self.fuel:
                actual = -VE * math.log(1.0 - self.fuel / m)
                if actual < 1e-6:
                    dvv = np.zeros(3)
                    dv_mag = 0.0
                else:
                    dvv = dvv * (actual / dv_mag)
                    dv_mag = actual
                self.fuel = 0.0
            else:
                self.fuel -= need
            if self.bug == 'B9_dv_mag_sum':
                dv_mag = abs(dv_pro) + abs(dv_rad) + abs(dv_nor)
            if dv_mag > 0.0 or float(np.linalg.norm(dvv)) > 0.0:
                raw = np.concatenate([rv_pre[:3], rv_pre[3:] + dvv])
                self.el = self.rv2coe(raw)

        # The logged post-burn state is what the ENV would log: reconstructed
        # from the elements it just stored, not the raw velocity update.  That
        # is what makes the element round trip observable to the battery.
        rv_post = self.coe2rv(self.el)

        # propagate through the mean anomaly
        el = dict(self.el)
        M = self.nu_to_M(el['nu'], el['e']) + math.sqrt(MU / el['a'] ** 3) * dt
        el['nu'] = wrap_2pi(o3.M_to_nu(M, el['e']))
        if self.bug == 'B14_raan_unwrapped':
            el['raan'] = wrap_pi(el['raan'])   # left on (-pi, pi] branch
        if self.bug == 'B15_coast_energy_drift':
            # the energy twin of B11: a perturbation (drag, a mis-scaled J2
            # secular term) left inside what is documented as a two-body
            # propagator.  Invisible to every angle check.
            el['a'] *= (1.0 - 1e-11)
        if self.bug == 'B11_coast_plane_drift':
            # a plane rotation applied during COAST -- e.g. a mis-scaled J2
            # nodal-regression term left in a two-body propagator.  Invisible
            # to |h| and to energy; visible only to the h-VECTOR.
            el['raan'] = wrap_2pi(el['raan']
                                  + 1e-7 * math.sqrt(MU / el['a'] ** 3) * dt)
        self.el = el
        rv_end = self.coe2rv(el)
        return dict(rv_pre=np.asarray(rv_pre), rv_post=np.asarray(rv_post),
                    rv_end=np.asarray(rv_end), el=dict(el), dt=dt,
                    dv_pro=dv_pro, dv_rad=dv_rad, dv_nor=dv_nor, dv_mag=dv_mag,
                    fuel=self.fuel)


BUGS = [
    (None, 'correct implementation'),
    ('B1_normal_is_z', 'normal burn along z_hat instead of h_hat'),
    ('B2_raan_acos', 'RAAN from acos, no n_y quadrant fix'),
    ('B3_argp_acos', 'argp from acos, no e_z quadrant fix'),
    ('B4_nu_acos', 'true anomaly from acos, no r.v quadrant fix'),
    ('B5_half_angle', 'nu->M half-angle inverted (2026-08-10 bug, 3D twin)'),
    ('B6_R1_sign', 'perifocal rotation uses R1(+i): mirrored plane'),
    ('B7_rot_order', '3-1-3 rotation sequence applied in reverse order'),
    ('B8_h_reversed', 'h computed as v x r: plane normal reversed'),
    ('B9_dv_mag_sum', '|dv| taken as the sum of frame components'),
    ('B10_stale_plane', 'i / RAAN not refreshed after a burn'),
    ('B11_coast_plane_drift', 'orbital plane rotates during COAST (stray nodal rate)'),
    ('B12_normal_north', 'normal axis forced to point north (breaks retrograde only)'),
    ('B13_radial_inertial', 'radial axis taken as the x-y projection of r'),
    ('B14_raan_unwrapped', 'RAAN left in (-pi, pi] instead of [0, 2pi)'),
    ('B15_coast_energy_drift', 'semi-major axis decays during COAST (stray perturbation)'),
]


# ═══════════════════════════════════════════════════════════════════════════
# The invariant battery
# ═══════════════════════════════════════════════════════════════════════════
def _coast_rows(traj):
    return [r for r in traj if r['dv_mag'] == 0.0]


def _burn_rows(traj):
    return [r for r in traj if r['dv_mag'] > 0.0]


def I1_energy_coast(traj):
    """Specific energy is invariant across a coast, scaled by mu/r."""
    w = 0.0
    for r in _coast_rows(traj):
        a, b = r['rv_pre'], r['rv_end']
        ea = 0.5 * np.dot(a[3:], a[3:]) - MU / np.linalg.norm(a[:3])
        eb = 0.5 * np.dot(b[3:], b[3:]) - MU / np.linalg.norm(b[:3])
        w = max(w, abs(eb - ea) / (MU / np.linalg.norm(a[:3])))
    return w


def I2_hvec_coast(traj):
    """|Δh| / |h| across a coast — the three-component check.  A scalar |h|
    check passes for any bug that ROTATES the orbital plane during coast; this
    one does not.  Nothing in the 2D battery could ever see this."""
    w = 0.0
    for r in _coast_rows(traj):
        h0 = np.cross(r['rv_pre'][:3], r['rv_pre'][3:])
        h1 = np.cross(r['rv_end'][:3], r['rv_end'][3:])
        w = max(w, float(np.linalg.norm(h1 - h0) / np.linalg.norm(h0)))
    return w


def I3_hhat_coast(traj):
    """Plane DIRECTION across a coast, in radians.  Separated from I2 so a
    magnitude bug and an orientation bug can be told apart in the report."""
    w = 0.0
    for r in _coast_rows(traj):
        h0 = unit(np.cross(r['rv_pre'][:3], r['rv_pre'][3:]))
        h1 = unit(np.cross(r['rv_end'][:3], r['rv_end'][3:]))
        w = max(w, math.asin(min(1.0, float(np.linalg.norm(np.cross(h0, h1))))))
    return w


def I4_node_coast(traj):
    """RAAN (node direction) across a coast, in radians.  Redundant with I3 in
    exact arithmetic, but it is the quantity a reader recognises, and it is the
    one that appears in the observation vector."""
    w = 0.0
    for r in _coast_rows(traj):
        c0, c1 = o3.rv2coe(r['rv_pre']), o3.rv2coe(r['rv_end'])
        if math.sin(c0['i']) < 1e-6:
            continue
        w = max(w, abs(wrap_pi(c1['raan'] - c0['raan'])))
    return w


def I5_evec_coast(traj):
    """Eccentricity VECTOR across a coast (apsidal line must not precess in
    two-body motion).  Catches argp bugs that |e| alone cannot see."""
    w = 0.0
    for r in _coast_rows(traj):
        c0, c1 = o3.rv2coe(r['rv_pre']), o3.rv2coe(r['rv_end'])
        w = max(w, float(np.linalg.norm(c1['e_vec'] - c0['e_vec'])))
    return w


def I6_oracle_resim(traj):
    """Step-local oracle re-simulation: apply the logged Δv in the ORACLE's
    frame to the logged pre-state, coast dt with universal variables, compare
    to the logged end-state.  Relative position error.

    The Δv DIRECTION comes from the commanded frame components and the
    MAGNITUDE from the logged |Δv|, exactly as the 2D harness does — otherwise
    every fuel-clamped burn reads as a dynamics error."""
    w = 0.0
    for r in traj:
        rv = np.asarray(r['rv_pre'], dtype=np.float64).copy()
        if r['dv_mag'] > 0.0:
            pro, rad, nor = o3.frame_ntw(rv)[0], unit(rv[:3]), o3.frame_rsw(rv)[2]
            d = r['dv_pro'] * pro + r['dv_rad'] * rad + r['dv_nor'] * nor
            dn = float(np.linalg.norm(d))
            if dn > 0.0:
                rv[3:] += d * (r['dv_mag'] / dn)
        got = o3.propagate_universal(rv, r['dt'])
        w = max(w, float(np.linalg.norm(got[:3] - r['rv_end'][:3])
                         / np.linalg.norm(r['rv_end'][:3])))
    return w


def I7_impulse_position(traj):
    """An impulse cannot move the spacecraft.  |r_post − r_pre| in metres."""
    w = 0.0
    for r in _burn_rows(traj):
        w = max(w, float(np.linalg.norm(r['rv_post'][:3] - r['rv_pre'][:3])))
    return w


def I8_inplane_burn_plane(traj):
    """THE killer regression: a burn with no normal component must not move the
    orbital plane.  Reported as the angle between h_hat before and after, in
    radians.  Any 2D-lifted-to-3D frame bug lands here."""
    w = 0.0
    for r in _burn_rows(traj):
        if r['dv_nor'] != 0.0:
            continue
        h0 = unit(np.cross(r['rv_pre'][:3], r['rv_pre'][3:]))
        h1 = unit(np.cross(r['rv_post'][:3], r['rv_post'][3:]))
        w = max(w, math.asin(min(1.0, float(np.linalg.norm(np.cross(h0, h1))))))
    return w


def I8b_inplane_burn_i_raan(traj):
    """Same event, expressed in the elements the observation exposes:
    max(|Δi|, |ΔRAAN|) across an in-plane burn, radians."""
    w = 0.0
    for r in _burn_rows(traj):
        if r['dv_nor'] != 0.0:
            continue
        c0, c1 = o3.rv2coe(r['rv_pre']), o3.rv2coe(r['rv_post'])
        w = max(w, abs(c1['i'] - c0['i']))
        if math.sin(c0['i']) > 1e-6:
            w = max(w, abs(wrap_pi(c1['raan'] - c0['raan'])))
    return w


def I9_burn_energy_h(traj):
    """Burn bookkeeping in 3D: ΔE = v·Δv + |Δv|²/2 and Δh_vec = r × Δv (all
    three components).  Returns max of the two residuals, each normalised."""
    w = 0.0
    for r in _burn_rows(traj):
        rv0, rv1 = r['rv_pre'], r['rv_post']
        dvv = rv1[3:] - rv0[3:]
        dE = (0.5 * np.dot(rv1[3:], rv1[3:]) - 0.5 * np.dot(rv0[3:], rv0[3:]))
        dE_pred = float(np.dot(rv0[3:], dvv)) + 0.5 * float(np.dot(dvv, dvv))
        # Normalise by |v||dv|, NOT by |v.dv|: a pure normal burn has v.dv = 0
        # by construction, and dividing by it turns fp noise into a false alarm
        # on exactly the manoeuvre class this extension is about.
        dvn = float(np.linalg.norm(dvv))
        scale_E = float(np.linalg.norm(rv0[3:])) * dvn + 0.5 * dvn * dvn + 1e-12
        w = max(w, abs(dE - dE_pred) / scale_E)
        dh = np.cross(rv1[:3], rv1[3:]) - np.cross(rv0[:3], rv0[3:])
        dh_pred = np.cross(rv0[:3], dvv)
        # Normalise by |r||dv| (the largest |r x dv| can be), not by |dh_pred|:
        # a purely radial burn has r x dv = 0 exactly, and normalising by it
        # turns a zero into a division by noise.
        scale_h = float(np.linalg.norm(rv0[:3])) * dvn + 1e-12
        w = max(w, float(np.linalg.norm(dh - dh_pred)) / scale_h)
    return w


def I15_burn_direction(traj):
    """Commanded-frame fidelity: the angle between the REALISED Δv and the
    direction the commanded (prograde, radial, normal) triple names in the
    oracle's frame.  Radians.

    The sharpest single check for the 3D extension's new machinery, because a
    burn-frame bug is a pure DIRECTION error: it can leave |Δv|, the fuel
    ledger, the energy budget and every element magnitude looking plausible
    while pointing the thrust somewhere else.  Small burns are excluded (below
    0.1 m/s the element round trip dominates the angle)."""
    w = 0.0
    for r in _burn_rows(traj):
        if r['dv_mag'] < 0.1:
            continue
        rv = r['rv_pre']
        pro = unit(rv[3:])
        rad = unit(rv[:3])
        nor = unit(np.cross(rv[:3], rv[3:]))
        want = r['dv_pro'] * pro + r['dv_rad'] * rad + r['dv_nor'] * nor
        got = r['rv_post'][3:] - rv[3:]
        if float(np.linalg.norm(want)) < 1e-12 or float(np.linalg.norm(got)) < 1e-12:
            continue
        a, b = unit(want), unit(got)
        w = max(w, math.atan2(float(np.linalg.norm(np.cross(a, b))),
                              float(np.dot(a, b))))
    return w


def I10_arglat_continuity(traj):
    """Argument-of-latitude continuity across an in-plane impulse — the direct
    3D lift of the 2D battery's (theta + omega) invariant.  Because the plane is
    fixed for an in-plane burn, u = argp + nu is the position angle and cannot
    jump; argp and nu individually can and do (violently, at small e)."""
    w = 0.0
    for r in _burn_rows(traj):
        if r['dv_nor'] != 0.0:
            continue
        c0, c1 = o3.rv2coe(r['rv_pre']), o3.rv2coe(r['rv_post'])
        w = max(w, abs(wrap_pi((c1['argp'] + c1['nu'])
                               - (c0['argp'] + c0['nu']))))
    return w


def I10b_position_reconstruction(traj):
    """Bind the LOGGED angle set to the LOGGED geometry: rebuild r_hat from the
    env's own (i, RAAN, argp+nu) with the oracle's 3-1-3 rotation and compare
    against the env's own position vector.  Radians.

    This is the check that says "your reported orbital plane and your reported
    position agree", which no scalar element comparison can express, and which
    is convention-free (it never needs argp and nu separately)."""
    w = 0.0
    for r in traj:
        el = r['el']
        R = o3.rot_pf_to_eci(el['i'], el['raan'], el['argp'] + el['nu'])
        rhat = R @ np.array([1.0, 0.0, 0.0])
        meas = unit(r['rv_end'][:3])
        c = float(np.dot(rhat, meas))
        s = float(np.linalg.norm(np.cross(rhat, meas)))
        w = max(w, math.atan2(s, c))
    return w


def I11_dv_magnitude(traj):
    """Logged |Δv| must equal the realised |v_post − v_pre|.  The prograde and
    radial axes are NOT orthogonal (v_hat vs r_hat), so any env that reports
    |Δv| as a quadrature or a sum of components is wrong — and it is wrong in
    the fuel ledger, not just the log."""
    w = 0.0
    for r in _burn_rows(traj):
        real = float(np.linalg.norm(r['rv_post'][3:] - r['rv_pre'][3:]))
        w = max(w, abs(real - r['dv_mag']))
    return w


def I12_elements_selfconsistent(traj):
    """Logged elements vs the logged Cartesian state.

    Compared: a, e, i (all three convention-free) and the full state rebuilt
    from the logged elements through the ORACLE's coe2rv.  Deliberately NOT a
    per-angle comparison of RAAN/argp/nu: those depend on the degeneracy
    convention the env happens to pick, and a convention difference is a
    reporting choice, not a physics bug.  Rebuilding the state tests the only
    thing that matters — that the reported element set names the reported
    orbit."""
    w = 0.0
    for r in traj:
        c = o3.rv2coe(r['rv_end'])
        el = r['el']
        w = max(w, abs(c['a'] - el['a']) / abs(el['a']))
        w = max(w, abs(c['e'] - el['e']))
        w = max(w, abs(wrap_pi(c['i'] - el['i'])))
        rebuilt = o3.coe2rv(el['a'], el['e'], el['i'], el['raan'], el['argp'],
                            el['nu'])
        w = max(w, float(np.linalg.norm(rebuilt[:3] - r['rv_end'][:3])
                         / np.linalg.norm(r['rv_end'][:3])))
    return w


def I13_planar_2d_anchor(traj):
    """2D-compatibility anchor: for a trajectory confined to i = 0 with no
    normal burns, every state must have z = vz = 0 exactly and the in-plane
    motion must match the validated 2D oracle.  Returns max(|z|/r, |vz|/v,
    planar propagation disagreement)."""
    sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/nav')
    import orbital_math as om2                                    # noqa: E402
    w = 0.0
    for r in traj:
        for s in (r['rv_pre'], r['rv_post'], r['rv_end']):
            w = max(w, abs(s[2]) / float(np.linalg.norm(s[:3])))
            w = max(w, abs(s[5]) / float(np.linalg.norm(s[3:])))
        if r['dv_mag'] == 0.0:
            p = om2.propagate_cartesian((r['rv_pre'][0], r['rv_pre'][1],
                                         r['rv_pre'][3], r['rv_pre'][4]), r['dt'])
            w = max(w, math.hypot(p[0] - r['rv_end'][0], p[1] - r['rv_end'][1])
                    / float(np.linalg.norm(r['rv_end'][:3])))
    return w


def I14_angle_ranges(traj):
    """Domain hygiene: i in [0, pi], RAAN/argp/nu in [0, 2pi), a > 0, e in
    [0, 1), nothing non-finite.  Returns 1.0 on any violation.

    Cheap and catches a bug class nothing else can: an angle left on the wrong
    branch is numerically harmless everywhere sin/cos is applied (so every
    dynamics check stays quiet) and silently wrong everywhere an angle is
    normalised into an observation or compared against a threshold."""
    for r in traj:
        el = r['el']
        vals = [el['a'], el['e'], el['i'], el['raan'], el['argp'], el['nu']]
        if not all(math.isfinite(v) for v in vals):
            return 1.0
        if not (0.0 <= el['i'] <= math.pi + 1e-12):
            return 1.0
        for k in ('raan', 'argp', 'nu'):
            if not (-1e-12 <= el[k] < 2.0 * math.pi + 1e-12):
                return 1.0
        if el['a'] <= 0.0 or not (0.0 <= el['e'] < 1.0):
            return 1.0
        if not np.all(np.isfinite(r['rv_end'])):
            return 1.0
    return 0.0


# (label, fn, threshold_double, applies(scenario, policy)).
#
# Thresholds are for a DOUBLE-PRECISION trajectory log.  The floor is set by
# the classical-element degeneracy seam (~1e-11 relative position, see
# orbital_math3d.v9_element_conditioning), not by the propagator, which is good
# to ~1e-15.  For a float32 log like the current C env's, scale every length /
# relative threshold to >= 8 x the float32 quantisation noise instead — see
# FLOAT32_THRESHOLDS below.
_ALL = lambda s, p: True                                          # noqa: E731
_PLANAR = lambda s, p: s == 'C0_circ_eq' and p in (               # noqa: E731
    'inplane_only', 'coast_heavy')

INVARIANTS = [
    ('I1  energy, coast (rel to mu/r)', I1_energy_coast, 1e-12, _ALL),
    ('I2  h-VECTOR, coast (rel)', I2_hvec_coast, 1e-12, _ALL),
    ('I3  h_hat direction, coast (rad)', I3_hhat_coast, 1e-12, _ALL),
    ('I4  RAAN, coast (rad)', I4_node_coast, 1e-10, _ALL),
    ('I5  e-VECTOR, coast (abs)', I5_evec_coast, 1e-11, _ALL),
    ('I6  oracle re-sim, step-local (rel)', I6_oracle_resim, 1e-9, _ALL),
    ('I7  impulse moves position (m)', I7_impulse_position, 1e-3, _ALL),
    ('I8  in-plane burn tilts plane (rad)', I8_inplane_burn_plane, 1e-9, _ALL),
    ('I8b in-plane burn moves i/RAAN (rad)', I8b_inplane_burn_i_raan, 1e-9, _ALL),
    # I9's floor is set by the element round trip (I7), not by the propagator:
    # r_post differs from r_pre by ~4e-5 m at the degeneracy seam, which enters
    # dh directly.  1e-6 keeps 2 decades of margin over that floor.
    ('I9  burn dE / dh_VECTOR residual (rel)', I9_burn_energy_h, 1e-6, _ALL),
    ('I15 realised vs commanded dv direction (rad)', I15_burn_direction,
     1e-4, _ALL),
    ('I10 arglat continuity at burn (rad)', I10_arglat_continuity, 1e-9, _ALL),
    ('I10b r_hat from logged (i,RAAN,u) (rad)', I10b_position_reconstruction,
     1e-9, _ALL),
    ('I11 |dv| logged vs realised (m/s)', I11_dv_magnitude, 1e-5, _ALL),
    ('I12 logged elements vs state (rel/rad)', I12_elements_selfconsistent,
     1e-9, _ALL),
    ('I13 planar 2D anchor (rel)', I13_planar_2d_anchor, 1e-9, _PLANAR),
    ('I14 angle domains / finiteness [1=bad]', I14_angle_ranges, 0.0, _ALL),
]

# Recommended thresholds when the harness reads the C env's float32 trajectory
# log (0.5 ulp = 5.96e-8 relative), at >= 8x the modelled noise floor.  Mirrors
# the 2D harness's convention so the two reports are directly comparable.
FLOAT32_THRESHOLDS = {
    'I1': 1e-6, 'I2': 1e-6, 'I3': 1e-6, 'I4': 1e-5, 'I5': 1e-6,
    'I6': 10.0,          # metres, as in the 2D harness's A_pos
    'I7': 5.0,           # metres
    'I8': 1e-6, 'I8b': 1e-6, 'I9': 1e-4, 'I10': 1e-5, 'I10b': 1e-5,
    'I11': 1e-3, 'I12': 1e-5, 'I13': 1e-6, 'I14': 0.0, 'I15': 1e-3,
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario / policy matrix
# ═══════════════════════════════════════════════════════════════════════════
SCENARIOS = {
    'C0_circ_eq':   dict(e=0.0, i=0.0),
    'C1_circ_28':   dict(e=0.0, i=math.radians(28.5)),
    'C2_ell_51':    dict(e=0.05, i=math.radians(51.6)),
    'C3_polar':     dict(e=0.02, i=math.radians(90.0)),
    'C4_retro':     dict(e=0.10, i=math.radians(116.0)),
    'C5_molniya':   dict(e=0.70, i=math.radians(63.4), a=26554e3),
    'C6_near_eq':   dict(e=0.05, i=math.radians(0.001)),
    'C7_near_pole': dict(e=0.05, i=math.radians(89.999)),
}

# (name, description).  A "node" policy fires only near the ascending node, an
# "antinode" policy only 90 deg from it — the pair that separates a plane-change
# implementation that works at the node from one that works anywhere.
POLICIES = ('uniform', 'inplane_only', 'normal_only', 'normal_at_node',
            'normal_at_antinode', 'burn_burst', 'coast_heavy')

DV_INPLANE = (5.0, 10.0, 25.0, -5.0, -10.0, -25.0)
DV_RADIAL = (1.0, 10.0, -1.0, -10.0)
DV_NORMAL = (5.0, 25.0, 50.0, -5.0, -25.0, -50.0)


def _arglat(rv):
    c = o3.rv2coe(rv)
    return wrap_2pi(c['argp'] + c['nu'])


def run_model(scenario, policy, bug, n_steps=140, seed=0):
    rng = np.random.default_rng(seed)
    sc = SCENARIOS[scenario]
    el = dict(a=sc.get('a', R_EARTH + 700e3), e=sc['e'], i=sc['i'],
              raan=float(rng.uniform(0, 2 * math.pi)),
              argp=float(rng.uniform(0, 2 * math.pi)),
              nu=float(rng.uniform(0, 2 * math.pi)))
    m = Model3D(el, bug=bug)
    traj = []
    for _ in range(n_steps):
        dv_pro = dv_rad = dv_nor = 0.0
        u = _arglat(m.coe2rv(m.el))
        if policy == 'uniform':
            k = int(rng.integers(0, 4))
            if k == 1:
                dv_pro = float(rng.choice(DV_INPLANE))
            elif k == 2:
                dv_rad = float(rng.choice(DV_RADIAL))
            elif k == 3:
                dv_nor = float(rng.choice(DV_NORMAL))
        elif policy == 'inplane_only':
            if rng.random() < 0.6:
                dv_pro = float(rng.choice(DV_INPLANE))
            if rng.random() < 0.4:
                dv_rad = float(rng.choice(DV_RADIAL))
        elif policy == 'normal_only':
            if rng.random() < 0.5:
                dv_nor = float(rng.choice(DV_NORMAL))
        elif policy == 'normal_at_node':
            if min(u, 2 * math.pi - u, abs(u - math.pi)) < 0.15:
                dv_nor = float(rng.choice(DV_NORMAL))
        elif policy == 'normal_at_antinode':
            if abs(abs(wrap_pi(u)) - math.pi / 2) < 0.15:
                dv_nor = float(rng.choice(DV_NORMAL))
        elif policy == 'burn_burst':
            if rng.random() < 0.85:
                dv_pro = float(rng.choice(DV_INPLANE))
                dv_nor = float(rng.choice(DV_NORMAL)) if rng.random() < 0.5 else 0.0
        else:  # coast_heavy
            if rng.random() < 0.08:
                dv_pro = float(rng.choice(DV_INPLANE))
        try:
            traj.append(m.step(dv_pro, dv_rad, dv_nor))
        except Exception:
            break
    return traj


def evaluate(bug, scenarios=None, policies=None, n_steps=140, seed=0):
    """Max residual per invariant over the whole scenario x policy matrix."""
    scenarios = scenarios or list(SCENARIOS)
    policies = policies or list(POLICIES)
    res = {name: 0.0 for name, _, _, _ in INVARIANTS}
    for si, sc in enumerate(scenarios):
        for pi, pol in enumerate(policies):
            traj = run_model(sc, pol, bug, n_steps, seed + 101 * si + 7 * pi)
            if not traj:
                for name, _, _, _ in INVARIANTS:
                    res[name] = max(res[name], 1e30)
                continue
            for name, fn, _, applies in INVARIANTS:
                if not applies(sc, pol):
                    continue
                try:
                    v = fn(traj)
                except Exception:
                    v = 1e30
                if not math.isfinite(v):
                    v = 1e30
                res[name] = max(res[name], v)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--out', default='/Users/pete/space_training/web_data/results')
    args = ap.parse_args()

    thresh = {name: t for name, _, t, _ in INVARIANTS}
    names = [n for n, _, _, _ in INVARIANTS]

    print('== clean model: every invariant must be quiet ==')
    clean = evaluate(None, n_steps=args.steps)
    nfail = 0
    for n in names:
        ok = clean[n] <= thresh[n]
        nfail += not ok
        print(f'   {n:42s} {clean[n]:11.3e}  thresh {thresh[n]:9.2e}  '
              f'{"PASS" if ok else "FAIL"}')
    print(f'   {len(names) - nfail}/{len(names)} quiet on the correct model')

    print('\n== mutation matrix: residual per (bug, invariant); '
          '"." = quiet, else log10(residual/threshold) ==')
    rows = []
    hdr = '   ' + ' '.join(f'{n.split()[0]:>5s}' for n in names)
    print(f'   {"bug":24s}' + hdr)
    for bug, desc in BUGS:
        if bug is None:
            continue
        r = evaluate(bug, n_steps=args.steps)
        cells = []
        for n in names:
            t = thresh[n]
            if r[n] <= t:
                cells.append('    .')
            else:
                ratio = r[n] / t if t > 0 else float('inf')
                cells.append(f'{min(99.0, math.log10(ratio)):5.1f}'
                             if math.isfinite(ratio) else '  inf')
        caught = sum(1 for n in names if r[n] > thresh[n])
        print(f'   {bug:24s}   ' + ' '.join(cells) + f'   [{caught} checks fire]')
        rows.append(dict(bug=bug, description=desc, n_checks_firing=caught,
                         **{n.split()[0]: f'{r[n]:.3e}' for n in names}))

    print('\n== per-bug detail ==')
    undetected = []
    for bug, desc in BUGS:
        if bug is None:
            continue
        row = next(x for x in rows if x['bug'] == bug)
        if row['n_checks_firing'] == 0:
            undetected.append(bug)
        first = [n.split()[0] for n in names
                 if float(row[n.split()[0]]) > thresh[n]]
        print(f'   {bug:24s} {desc}')
        print(f'   {"":24s}   caught by: {", ".join(first) if first else "NOTHING"}')

    print('\n== invariant coverage (bugs each check catches) ==')
    dead = []
    for n in names:
        k = n.split()[0]
        c = [r['bug'] for r in rows if float(r[k]) > thresh[n]]
        if not c:
            dead.append(n)
        print(f'   {n:42s} {len(c):2d}  {", ".join(c) if c else "-- catches nothing --"}')

    print()
    if undetected:
        print(f'HOLE: bugs no invariant catches: {undetected}')
    else:
        print('No holes: every seeded bug is caught by at least one invariant.')
    if dead:
        print(f'Redundant-in-this-matrix checks: {dead}')

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, 'ext_3d_invariant_mutation_matrix.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bug', 'description', 'n_checks_firing']
                           + [n.split()[0] for n in names])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {p}')
    return 1 if (nfail or undetected) else 0


if __name__ == '__main__':
    sys.exit(main())
