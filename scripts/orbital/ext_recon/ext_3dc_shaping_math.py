#!/usr/bin/env python3
"""
ext-3d task C: analytic/numeric verification of the proposed 3D observation
coordinates and 3D shaping potential.

READ-ONLY w.r.t. the env: this script contains its own standalone 3D Keplerian
element<->cartesian machinery (independent of pufferlib/ocean/orbital/orbital.h)
so that every claim in the design memo is checked against an independent
implementation, in the spirit of the T3 f&g-oracle cross-checks.

Outputs (stdout tables + CSVs in web_data/results/ with ext_ prefix):
  A. Gauss control levers  -> the shaping coefficients (0.5v, 0.5v, 1.0v)
  B. Longitude-coordinate burn-continuity audit (equinoctial lambda vs ROE dlambda)
  C. Node-crank efficiency: dPhi/dv vs off-node angle psi
  D. Scripted 5-leg 3D maneuver: per-leg Phi ledger (monotonicity)
  E. Plane-change affordability vs altitude -> di_max recommendation
  F. Terminal-box -> required plane-match tolerance
"""
import csv
import math
import os

import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
VE = 300.0 * 9.80665
FUEL_FRAC = 0.15
DV_BUDGET = VE * math.log(1.0 / (1.0 - FUEL_FRAC))  # 478.1 m/s

OUT = "/Users/pete/space_training/web_data/results"


# ---------------------------------------------------------------- kepler 3D
def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def solve_kepler(M, e):
    M = math.fmod(M, 2 * math.pi)
    if M < 0:
        M += 2 * math.pi
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-15:
            break
    return E


def true_from_E(E, e):
    return 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                            math.sqrt(1 - e) * math.cos(E / 2))


def mean_from_true(th, e):
    E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(th / 2),
                         math.sqrt(1 + e) * math.cos(th / 2))
    return E - e * math.sin(E)


class Orb:
    """Classical elements (a,e,i,raan,argp,M). theta derived."""

    __slots__ = ("a", "e", "i", "raan", "argp", "M")

    def __init__(self, a, e, i, raan, argp, M):
        self.a, self.e, self.i, self.raan, self.argp, self.M = a, e, i, raan, argp, M

    def copy(self):
        return Orb(self.a, self.e, self.i, self.raan, self.argp, self.M)

    @property
    def theta(self):
        return true_from_E(solve_kepler(self.M, self.e), self.e)

    @property
    def n(self):
        return math.sqrt(MU / self.a ** 3)

    @property
    def lam_eq(self):
        """equinoctial mean longitude lambda = M + argp + raan (nonsingular e->0, i->0)"""
        return self.M + self.argp + self.raan

    @property
    def u_mean(self):
        """mean argument of latitude M + argp (ROE convention)"""
        return self.M + self.argp

    def propagate(self, dt):
        o = self.copy()
        o.M = math.fmod(o.M + o.n * dt, 2 * math.pi)
        return o

    def rv(self):
        th = self.theta
        p = self.a * (1 - self.e ** 2)
        r = p / (1 + self.e * math.cos(th))
        h = math.sqrt(MU * p)
        r_p = np.array([r * math.cos(th), r * math.sin(th), 0.0])
        v_p = np.array([-(MU / h) * math.sin(th), (MU / h) * (self.e + math.cos(th)), 0.0])
        R = rot_z(self.raan) @ rot_x(self.i) @ rot_z(self.argp)
        return R @ r_p, R @ v_p

    def hhat(self):
        r, v = self.rv()
        h = np.cross(r, v)
        return h / np.linalg.norm(h)

    def evec(self):
        r, v = self.rv()
        h = np.cross(r, v)
        return np.cross(v, h) / MU - r / np.linalg.norm(r)


def elements_from_rv(r, v):
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    a = 1.0 / (2.0 / rn - vn ** 2 / MU)
    ev = np.cross(v, h) / MU - r / rn
    e = np.linalg.norm(ev)
    i = math.atan2(math.hypot(h[0], h[1]), h[2])
    nvec = np.array([-h[1], h[0], 0.0])
    nn = np.linalg.norm(nvec)
    if nn < 1e-12:                       # equatorial: raan undefined -> 0 (standard)
        raan = 0.0
        nvec = np.array([1.0, 0.0, 0.0])
        nn = 1.0
    else:
        raan = math.atan2(nvec[1], nvec[0]) % (2 * math.pi)
    if e < 1e-12:                        # circular: argp undefined -> 0 (standard)
        argp = 0.0
        th = math.atan2(np.dot(np.cross(nvec / nn, r), h / hn), np.dot(nvec / nn, r))
    else:
        argp = math.atan2(np.dot(np.cross(nvec / nn, ev), h / hn), np.dot(nvec / nn, ev)) % (2 * math.pi)
        th = math.atan2(np.dot(np.cross(ev / e, r), h / hn), np.dot(ev / e, r))
    M = mean_from_true(th, e) % (2 * math.pi)
    return Orb(a, e, i, raan, argp, M)


def rtn_basis(o):
    r, v = o.rv()
    Rh = r / np.linalg.norm(r)
    Nh = np.cross(r, v)
    Nh = Nh / np.linalg.norm(Nh)
    Th = np.cross(Nh, Rh)
    return Rh, Th, Nh


def burn(o, dv_pro=0.0, dv_rad=0.0, dv_nor=0.0):
    """env-convention burn: prograde = vhat, radial = rhat, normal = hhat."""
    r, v = o.rv()
    vh = v / np.linalg.norm(v)
    rh = r / np.linalg.norm(r)
    nh = np.cross(r, v)
    nh = nh / np.linalg.norm(nh)
    dv = dv_pro * vh + dv_rad * rh + dv_nor * nh
    return elements_from_rv(r, v + dv), float(np.linalg.norm(dv))


def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------- relative quantities
def rel_state(s, t):
    """Everything the proposed obs/Phi needs, computed geometrically."""
    hs, ht = s.hhat(), t.hhat()
    cross = np.cross(ht, hs)                       # points at relative ASCENDING node
    di_rel = math.atan2(np.linalg.norm(cross), float(np.dot(ht, hs)))
    chord = float(np.linalg.norm(hs - ht))         # = 2 sin(di_rel/2)
    nhat = cross / np.linalg.norm(cross) if np.linalg.norm(cross) > 1e-15 else np.zeros(3)
    de_vec = s.evec() - t.evec()
    da_rel = (s.a - t.a) / t.a
    dlam_eq = wrap_pi(s.lam_eq - t.lam_eq)
    dlam_roe = wrap_pi((s.u_mean - t.u_mean) + (s.raan - t.raan) * math.cos(t.i))
    return dict(di_rel=di_rel, chord=chord, nhat=nhat, ivec=di_rel * nhat,
                de=float(np.linalg.norm(de_vec)), de_vec=de_vec,
                da_rel=da_rel, dlam_eq=dlam_eq, dlam_roe=dlam_roe)


W_LAM, W_M = 1.0, 0.817
DV_REF = 700.0


def phi(s, t, w_lam=W_LAM, w_m=W_M, dv_ref=DV_REF, form="L1"):
    rs = rel_state(s, t)
    v_t = math.sqrt(MU / t.a)
    dv_in = 0.5 * v_t * math.hypot(rs["da_rel"], rs["de"])
    dv_pl = v_t * rs["chord"]
    if form == "L1":
        dv3 = dv_in + dv_pl
    elif form == "hypot3":
        dv3 = 0.5 * v_t * math.sqrt(rs["da_rel"] ** 2 + rs["de"] ** 2 + (2 * rs["chord"]) ** 2)
    else:
        raise ValueError(form)
    return -(w_lam * abs(rs["dlam_eq"]) / math.pi + w_m * min(1.0, dv3 / dv_ref)), dv_in, dv_pl, rs


# ============================================================== A. levers
def sec_A():
    rows = []
    print("\n=== A. Gauss control levers (near-circular LEO, a=6771 km) ===")
    a0 = R_EARTH + 400e3
    v_c = math.sqrt(MU / a0)
    for e0 in (0.0, 0.05):
        for u in (0.0, math.pi / 2):
            base = Orb(a0, e0, math.radians(0.0), 0.0, 0.0, u)
            for lbl, kw in (("tangential", dict(dv_pro=1.0)),
                            ("radial", dict(dv_rad=1.0)),
                            ("normal", dict(dv_nor=1.0))):
                nb, dvm = burn(base, **kw)
                d_a_rel = (nb.a - base.a) / base.a
                d_e = float(np.linalg.norm(nb.evec() - base.evec()))
                d_i = math.atan2(np.linalg.norm(np.cross(base.hhat(), nb.hhat())),
                                 float(np.dot(base.hhat(), nb.hhat())))
                rows.append(dict(e=e0, u_deg=round(math.degrees(u), 1), axis=lbl,
                                 lever_a_over_v=d_a_rel * v_c / dvm,
                                 lever_e_over_v=d_e * v_c / dvm,
                                 lever_i_over_v=d_i * v_c / dvm))
                print(f"  e={e0:.2f} u={math.degrees(u):5.1f}deg {lbl:>10s}  "
                      f"d(da/a)*v/dv={d_a_rel*v_c/dvm:+7.4f}  "
                      f"d|de|*v/dv={d_e*v_c/dvm:7.4f}  "
                      f"d(di)*v/dv={d_i*v_c/dvm:7.4f}")
    print("  -> tangential lever on da/a and |de| = 2/v  => coefficient 0.5*v")
    print("  -> normal     lever on di           = 1/v  => coefficient 1.0*v  (does NOT halve)")
    return rows


# ==================================== B. longitude burn-continuity audit
def sec_B():
    rows = []
    print("\n=== B. Longitude coordinate: burn-continuity audit ===")
    print("    jump = |lambda(after burn) - lambda(before burn)| for the SAME physical instant")
    a0 = R_EARTH + 500e3
    tgt = Orb(a0, 0.0, math.radians(0.0), 0.0, 0.0, 0.0)
    for i_t_deg in (0.0, 28.5, 51.6, 97.4):
        tgt.i = math.radians(i_t_deg)
        for e0 in (0.0, 0.02, 0.05, 0.15):
            for dv_n in (1.0, 10.0, 25.0):
                worst_eq = 0.0
                worst_roe = 0.0
                worst_eq_inplane = 0.0
                for k in range(72):
                    u = 2 * math.pi * k / 72
                    s = Orb(a0 + 50e3, e0, math.radians(i_t_deg + 0.3), 0.0, 0.0, u)
                    b_eq, b_roe = s.lam_eq, s.u_mean + s.raan * math.cos(tgt.i)
                    nb, _ = burn(s, dv_nor=dv_n)
                    worst_eq = max(worst_eq, abs(wrap_pi(nb.lam_eq - b_eq)))
                    a_roe = nb.u_mean + nb.raan * math.cos(tgt.i)
                    worst_roe = max(worst_roe, abs(wrap_pi(a_roe - b_roe)))
                    nb2, _ = burn(s, dv_pro=dv_n)
                    worst_eq_inplane = max(worst_eq_inplane, abs(wrap_pi(nb2.lam_eq - b_eq)))
                rows.append(dict(i_t_deg=i_t_deg, e=e0, dv_ms=dv_n,
                                 eq_normal_jump_deg=math.degrees(worst_eq),
                                 roe_normal_jump_deg=math.degrees(worst_roe),
                                 eq_tangential_jump_deg=math.degrees(worst_eq_inplane)))
                if dv_n == 10.0:
                    print(f"  i_t={i_t_deg:5.1f}deg e={e0:.2f} dv=10 m/s : "
                          f"equinoctial lam jump(normal)={math.degrees(worst_eq):7.4f}deg  "
                          f"ROE dlam jump(normal)={math.degrees(worst_roe):8.3f}deg  "
                          f"equinoctial jump(tangential)={math.degrees(worst_eq_inplane):7.4f}deg")
    print("  -> ROE dlambda = du + dRAAN*cos(i_t) TELEPORTS on normal burns "
          "(jump ~ u*(1-cos i_t)); equinoctial lambda = M+argp+RAAN does not.")
    return rows


# ==================================== C. node-crank efficiency
def sec_C():
    rows = []
    print("\n=== C. Plane-change credit vs off-node burn angle psi ===")
    a0 = R_EARTH + 500e3
    v_c = math.sqrt(MU / a0)
    tgt = Orb(a0, 0.0, math.radians(30.0), math.radians(20.0), 0.0, 0.0)
    for di0_deg in (1.0, 0.25, 0.05):
        # build chaser with a pure relative inclination di0 about a chosen node
        base = Orb(a0, 0.0, math.radians(30.0 + di0_deg), math.radians(20.0), 0.0, 0.0)
        r0 = rel_state(base, tgt)
        print(f"  di_rel={di0_deg:5.2f}deg  (chord={r0['chord']:.6f}, "
              f"dv_plane={math.sqrt(MU/tgt.a)*r0['chord']:7.2f} m/s)")
        for psi_deg in (0, 15, 30, 45, 60, 90, 135, 180):
            # place chaser at argument-of-latitude offset psi from the relative
            # ascending node, burn normal(-) i.e. toward closing
            nhat = r0["nhat"]
            Rh, Th, Nh = rtn_basis(base)
            u_node = math.atan2(float(np.dot(nhat, Th)), float(np.dot(nhat, Rh)))
            s = base.copy()
            s.M = (base.M + u_node + math.radians(psi_deg)) % (2 * math.pi)
            best = None
            for sgn in (+1.0, -1.0):
                nb, dvm = burn(s, dv_nor=sgn * 10.0)
                rr = rel_state(nb, tgt)
                d_dv_plane = math.sqrt(MU / tgt.a) * (r0["chord"] - rr["chord"])
                if best is None or d_dv_plane > best[0]:
                    best = (d_dv_plane, sgn, dvm)
            eff = best[0] / best[2]
            rows.append(dict(di0_deg=di0_deg, psi_deg=psi_deg, credit_per_dv=eff,
                             cos_psi=math.cos(math.radians(psi_deg))))
            print(f"      psi={psi_deg:4d}deg  best credit = {best[0]:+7.3f} m/s "
                  f"per {best[2]:.1f} m/s spent  -> ratio {eff:+6.3f}  (cos psi = "
                  f"{math.cos(math.radians(psi_deg)):+6.3f})")
    print("  -> credit/dv = |cos psi| exactly: max at the node, zero 90deg off-node,")
    print("     and the residual is a bounded 2nd-order PENALTY (never a cliff).")
    return rows


# ==================================== D. scripted maneuver ledger
def sec_D(form="L1"):
    rows = []
    print(f"\n=== D. Scripted 5-leg 3D maneuver, per-leg Phi ledger (form={form}) ===")
    a_t = R_EARTH + 500e3
    tgt = Orb(a_t, 0.02, math.radians(51.6), math.radians(40.0), math.radians(70.0), 0.0)
    v_t = math.sqrt(MU / a_t)

    # chaser: same orbit shape, +1.0 deg relative inclination, 180 deg mean-longitude gap
    s = Orb(a_t, 0.02, math.radians(52.6), math.radians(40.0), math.radians(70.0), 0.0)
    # set chaser lambda so that dlam_eq = +180 deg (target ahead => dlam = -gap; use +pi)
    s.M = (tgt.lam_eq + math.pi - s.argp - s.raan) % (2 * math.pi)

    ledger = []
    p0, dvin0, dvpl0, rs0 = phi(s, tgt, form=form)
    print(f"  init: dlam={math.degrees(rs0['dlam_eq']):+7.2f}deg  di_rel="
          f"{math.degrees(rs0['di_rel']):.4f}deg  dv_plane={dvpl0:.1f}  dv_inplane={dvin0:.1f}"
          f"  Phi0={p0:+.4f}")

    dv_spent = 0.0
    cur, tt = s, tgt

    def leg(name, new_s, new_t, dv):
        nonlocal cur, tt, dv_spent
        pa, _, _, _ = phi(cur, tt, form=form)
        pb, dvin, dvpl, rr = phi(new_s, new_t, form=form)
        dv_spent += dv
        ledger.append(dict(leg=name, dPhi=pb - pa, dv_ms=dv,
                           dlam_deg=math.degrees(rr["dlam_eq"]),
                           di_deg=math.degrees(rr["di_rel"]),
                           dv_plane=dvpl, dv_inplane=dvin, phi=pb))
        print(f"  {name:<34s} dPhi={pb-pa:+8.4f}  dv={dv:6.1f} m/s  "
              f"dlam={math.degrees(rr['dlam_eq']):+8.2f}deg  di={math.degrees(rr['di_rel']):.4f}deg"
              f"  Phi={pb:+.4f}")
        cur, tt = new_s, new_t

    # L1: coast to the relative node, then plane-change in 25 m/s quanta
    rr = rel_state(cur, tt)
    Rh, Th, Nh = rtn_basis(cur)
    u_node = math.atan2(float(np.dot(rr["nhat"], Th)), float(np.dot(rr["nhat"], Rh)))
    dt_to_node = (u_node % (2 * math.pi)) / cur.n
    ns, nt = cur.propagate(dt_to_node), tt.propagate(dt_to_node)
    leg("L1a coast to relative node", ns, nt, 0.0)
    dv_pl_needed = v_t * rel_state(cur, tt)["chord"]
    n_burn = int(math.ceil(dv_pl_needed / 25.0))
    for k in range(n_burn):
        q = min(25.0, dv_pl_needed - 25.0 * k)
        best = None
        for sgn in (+1, -1):
            nb, dvm = burn(cur, dv_nor=sgn * q)
            c = rel_state(nb, tt)["chord"]
            if best is None or c < best[0]:
                best = (c, nb, dvm)
        leg(f"L1b plane burn {k+1}/{n_burn} ({q:.0f} m/s)", best[1], tt, best[2])

    # L2: open the drift orbit (da = -200 km  => chaser drifts ahead faster)
    da_drift = -200e3
    dv_open = abs(0.5 * math.sqrt(MU / cur.a) * (da_drift / cur.a))
    n_b = int(math.ceil(dv_open / 25.0))
    for k in range(n_b):
        q = min(25.0, dv_open - 25.0 * k) * (1 if da_drift > 0 else -1)
        nb, dvm = burn(cur, dv_pro=q)
        leg(f"L2 drift-open burn {k+1}/{n_b} ({abs(q):.0f} m/s)", nb, tt, dvm)

    # L3: drift until |dlam| is closed
    rr = rel_state(cur, tt)
    dlam = rr["dlam_eq"]
    ddot = cur.n - tt.n
    t_drift = -dlam / ddot
    if t_drift < 0:
        t_drift += 2 * math.pi / abs(ddot)
    # report in 6 slices to show monotonicity of the drift leg
    for k in range(6):
        ns, nt = cur.propagate(t_drift / 6), tt.propagate(t_drift / 6)
        leg(f"L3 drift slice {k+1}/6 ({t_drift/6/3600:.2f} h)", ns, nt, 0.0)

    # L4: close the drift orbit
    dv_close = dv_open
    n_b = int(math.ceil(dv_close / 25.0))
    for k in range(n_b):
        q = min(25.0, dv_close - 25.0 * k) * (1 if da_drift < 0 else -1)
        nb, dvm = burn(cur, dv_pro=q)
        leg(f"L4 drift-close burn {k+1}/{n_b} ({abs(q):.0f} m/s)", nb, tt, dvm)

    pT, _, _, rrT = phi(cur, tt, form=form)
    tot = sum(l["dPhi"] for l in ledger)
    adverse = [l for l in ledger if l["dPhi"] < 0]
    print(f"  ---- total dPhi = {tot:+.4f}   (telescoping check: Phi_T - Phi_0 = {pT-p0:+.4f})")
    print(f"       total dv   = {dv_spent:.1f} m/s of {DV_BUDGET:.0f} budget "
          f"({100*dv_spent/DV_BUDGET:.0f}%)")
    print(f"       worst adverse step = {min([l['dPhi'] for l in ledger]):+.4f} "
          f"({len(adverse)} adverse legs of {len(ledger)})")
    print(f"       drift-leg share    = "
          f"{sum(l['dPhi'] for l in ledger if l['leg'].startswith('L3')):+.4f}")
    print(f"       plane-leg share    = "
          f"{sum(l['dPhi'] for l in ledger if l['leg'].startswith('L1b')):+.4f}")
    print(f"       residual dlam={math.degrees(rrT['dlam_eq']):+.3f}deg  "
          f"di={math.degrees(rrT['di_rel']):.5f}deg")
    for l in ledger:
        l["form"] = form
    rows.extend(ledger)
    return rows


# ============================== E. plane-change affordability vs altitude
def sec_E():
    rows = []
    print("\n=== E. Plane-change affordability (dv = 2*v*sin(di/2)); budget "
          f"{DV_BUDGET:.1f} m/s ===")
    print(f"  {'alt/a':>16s} {'v_c m/s':>9s} {'dv per deg':>11s} "
          f"{'di @100% budget':>16s} {'di @28% budget':>15s}")
    for label, a in (("LEO 300 km", R_EARTH + 300e3),
                     ("LEO 800 km", R_EARTH + 800e3),
                     ("2000 km", R_EARTH + 2000e3),
                     ("8000 km", R_EARTH + 8000e3),
                     ("MEO 20200 km", R_EARTH + 20200e3),
                     ("GEO 35786 km", R_EARTH + 35786e3)):
        v = math.sqrt(MU / a)
        per_deg = 2 * v * math.sin(math.radians(1.0) / 2)
        di_full = math.degrees(2 * math.asin(min(1.0, DV_BUDGET / (2 * v))))
        di_28 = math.degrees(2 * math.asin(min(1.0, 0.28 * DV_BUDGET / (2 * v))))
        rows.append(dict(regime=label, a_m=a, v_c=v, dv_per_deg=per_deg,
                         di_max_full_budget_deg=di_full, di_max_28pct_deg=di_28))
        print(f"  {label:>16s} {v:9.1f} {per_deg:11.1f} {di_full:16.3f} {di_28:15.3f}")
    print("  -> at LEO the 478 m/s budget affords ~3.5 deg of PURE plane change and")
    print("     ~1.0 deg if the plane leg is capped at 28% of budget. di_max must be")
    print("     set from this table per rung (exactly the de_max precedent).")
    return rows


# ==================================== F. terminal box -> plane tolerance
def sec_F():
    rows = []
    print("\n=== F. Terminal success box -> required relative-plane tolerance ===")
    print(f"  {'regime':>14s} {'box':>16s} {'di from pos box':>16s} {'di from vel box':>16s}")
    for label, a in (("LEO 500 km", R_EARTH + 500e3),
                     ("MEO 20200 km", R_EARTH + 20200e3)):
        v = math.sqrt(MU / a)
        for rbox, vbox in ((30e3, 50.0), (10e3, 10.0), (5e3, 1.0)):
            di_pos = math.degrees(math.asin(min(1.0, rbox / a)))
            di_vel = math.degrees(2 * math.asin(min(1.0, vbox / (2 * v))))
            rows.append(dict(regime=label, r_box_m=rbox, v_box_ms=vbox,
                             di_pos_deg=di_pos, di_vel_deg=di_vel))
            print(f"  {label:>14s} {rbox/1e3:5.0f} km/{vbox:5.1f} m/s "
                  f"{di_pos:16.4f} {di_vel:16.5f}")
    print("  -> the VELOCITY box is the binding plane constraint. At LEO the 5 km/1 m/s")
    print("     tight box demands di_rel < 0.0075 deg = 27 arcsec, which is exactly one")
    print("     1 m/s normal quantum -> the 3D tight box lands on the actuation floor,")
    print("     the same signature as the 2D radial-quantum finding (T3 sec 8.3).")
    return rows


# ==================================== G. saturation / weight calibration
def sec_G():
    print("\n=== G. min(1, dv3/dv_ref) saturation check for candidate 3D rungs ===")
    rows = []
    for label, a_lo, a_hi, e_max, de_max, di_max_deg, dv_ref in (
            ("X1 LEO coplanar anchor", 500e3, 800e3, 0.0, 0.0, 0.0, 300.0),
            ("X2 LEO di<=0.25deg", 300e3, 800e3, 0.05, 0.05, 0.25, 300.0),
            ("X3 LEO di<=1.0deg", 300e3, 800e3, 0.05, 0.05, 1.00, 700.0),
            ("X4 wide di<=1.0deg", 300e3, 8000e3, 0.30, 0.08, 1.00, 700.0),
            ("X5 MEO di<=2.0deg", 300e3, 20200e3, 0.50, 0.10, 2.00, 900.0)):
        a_t = R_EARTH + (a_lo + a_hi) / 2
        v_t = math.sqrt(MU / a_t)
        da_rel = min(600e3, (a_hi - a_lo)) / a_t
        dv_in = 0.5 * v_t * math.hypot(da_rel, de_max)
        dv_pl = 2 * v_t * math.sin(math.radians(di_max_deg) / 2)
        dv3 = dv_in + dv_pl
        rows.append(dict(rung=label, dv_inplane=dv_in, dv_plane=dv_pl, dv3=dv3,
                         dv_ref=dv_ref, ratio=dv3 / dv_ref,
                         w_match=1.167e-3 * dv_ref))
        print(f"  {label:<24s} dv_in={dv_in:6.1f} dv_pl={dv_pl:6.1f} dv3={dv3:6.1f} "
              f"/ dv_ref={dv_ref:5.0f} = {dv3/dv_ref:5.2f}  "
              f"{'SATURATES' if dv3 > dv_ref else 'ok':>9s}   "
              f"W_match(iso-rate)={1.167e-3*dv_ref:.3f}")
    print("  -> design rule: hold W_match/dv_ref = 0.35/300 = 1.167e-3 per (m/s) so the")
    print("     'one potential unit per m/s of dv-to-go removed' calibration is invariant;")
    print("     move dv_ref only to control saturation.")
    return rows


def dump(name, rows):
    if not rows:
        return
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"    [wrote {p}]")


if __name__ == "__main__":
    dump("ext_3d_levers.csv", sec_A())
    dump("ext_3d_longitude_continuity.csv", sec_B())
    dump("ext_3d_node_crank.csv", sec_C())
    d1 = sec_D("L1")
    d2 = sec_D("hypot3")
    dump("ext_3d_maneuver_ledger.csv", d1 + d2)
    dump("ext_3d_plane_affordability.csv", sec_E())
    dump("ext_3d_box_plane_tol.csv", sec_F())
    dump("ext_3d_saturation.csv", sec_G())
