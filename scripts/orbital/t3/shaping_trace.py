#!/usr/bin/env python3
"""T3 forensic audit of the orbital env's reward shaping under CORRECT dynamics.

Q1  Trace Phi (legacy + 4 candidate redesigns) along a hand-scripted, physically
    correct phasing trajectory (open drift orbit -> drift -> close -> fine null),
    plus null policies (coast-forever, warp-forever, park-far-and-warp).
Q2  The omega question: how omega_sat / omega_target are initialised, what
    init_phase_gap_max actually parameterises, whether obs[13-16] and
    compute_phi()'s Phi_phase use true anomaly or true longitude, and how far
    the shaping's phase target sits from the real rendezvous point.
Q3  (report only) recommendation.

Everything is recomputed in python from the env's own observation vector, which
carries a_sat, e_sat, theta_sat, omega_sat, a_tgt, e_tgt, theta_tgt, omega_tgt
exactly (float32).  The python Phi is cross-validated against the realised C
reward on every non-terminal step, so the reimplementation is provably faithful.

Run from /Users/pete/space_training/pufferlib with python3.
Outputs CSVs into /Users/pete/space_training/web_data/results/.
"""

import math
import os
import sys

import numpy as np

REPO = "/Users/pete/space_training"
PUFFER = os.path.join(REPO, "pufferlib")
OUT = os.path.join(REPO, "web_data", "results")
sys.path.insert(0, PUFFER)

from pufferlib.ocean.orbital.orbital import Orbital  # noqa: E402

# ── constants mirrored from orbital.h ────────────────────────────────────────
MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
MAX_STEPS = 2000
SUCCESS_TOL_A = 10000.0
REL_VEL_TOL = 50.0
BETA_SHAPE = 1.0
W_ORBIT = W_PHASE = W_VEL = 0.01
EPS_ORBIT, EPS_PHASE = 2.0, 0.3
TAU_ORBIT, TAU_PHASE = 0.1 * EPS_ORBIT, 0.1 * EPS_PHASE
GAMMA = 0.995
OBS_ALT_SCALE = 1.6e6
PHI_ORBIT_SCALE_K = 0.001

ACTION_DV = [
    (0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (25.0, 0.0), (-5.0, 0.0), (-10.0, 0.0),
    (-25.0, 0.0), (0.0, 10.0), (0.0, -10.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (1.0, 0.0), (-1.0, 0.0), (2.0, 0.0), (-2.0, 0.0),
]
ACTION_TAU = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1]
COAST, WARP5, WARP30, WARP60 = 0, 9, 10, 11
PRO = {25: 3, 10: 2, 5: 1, 2: 14, 1: 12}
RETRO = {25: 6, 10: 5, 5: 4, 2: 15, 1: 13}

# ── design-side constants (candidate redesigns) ──────────────────────────────
DA_MAX = 300e3                 # m, phasing-orbit semi-major-axis budget
A_MAX_RATE = 25.0 / 60.0       # m/s of dv per second of sim time (one 25 m/s burn/step)
T_CAP = MAX_STEPS * DT         # 120 000 s
K_DASTAR = DA_MAX / math.pi    # S-C: pi rad of gap -> 300 km


def wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def sig(x):
    if x > 60.0:
        return 1.0
    if x < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ── state extraction ─────────────────────────────────────────────────────────
def state_from_obs(obs):
    o = np.asarray(obs, dtype=np.float64).ravel()
    s = {}
    s["a_s"] = o[0] * OBS_ALT_SCALE + R_EARTH
    s["e_s"] = o[1]
    s["th_s"] = math.atan2(o[2], o[3])
    s["fuel"] = o[6]
    s["a_t"] = o[7] * OBS_ALT_SCALE + R_EARTH
    s["e_t"] = o[8]
    s["w_s"] = math.atan2(o[9], o[10])
    s["w_t"] = math.atan2(o[11], o[12])
    s["th_t"] = math.atan2(o[15], o[16])
    return s


def elem_to_cart(a, e, th, w):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(th))
    h = math.sqrt(MU * p)
    xp, yp = r * math.cos(th), r * math.sin(th)
    vxp, vyp = -(MU / h) * math.sin(th), (MU / h) * (e + math.cos(th))
    co, so = math.cos(w), math.sin(w)
    return (co * xp - so * yp, so * xp + co * yp,
            co * vxp - so * vyp, so * vxp + co * vyp)


def lvlh(s):
    sx, sy, svx, svy = elem_to_cart(s["a_s"], s["e_s"], s["th_s"], s["w_s"])
    tx, ty, tvx, tvy = elem_to_cart(s["a_t"], s["e_t"], s["th_t"], s["w_t"])
    ang = s["th_t"] + s["w_t"]
    ct, st = math.cos(ang), math.sin(ang)
    dxi, dyi = sx - tx, sy - ty
    dvxi, dvyi = svx - tvx, svy - tvy
    dx_l = ct * dxi + st * dyi
    dy_l = -st * dxi + ct * dyi
    dvx_l = ct * dvxi + st * dvyi
    dvy_l = -st * dvxi + ct * dvyi
    n_t = math.sqrt(MU / s["a_t"] ** 3)
    dvx_l += n_t * dy_l
    dvy_l -= n_t * dx_l
    return dx_l, dy_l, dvx_l, dvy_l, math.hypot(dxi, dyi), math.hypot(dvxi, dvyi)


def true_to_mean(th, e):
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(th / 2.0),
                         math.sqrt(1.0 + e) * math.cos(th / 2.0))
    return E - e * math.sin(E)


# ── potentials ───────────────────────────────────────────────────────────────
def phi_parts(s):
    """Common scalar ingredients + the exact legacy components."""
    da = s["a_s"] - s["a_t"]
    esx, esy = s["e_s"] * math.cos(s["w_s"]), s["e_s"] * math.sin(s["w_s"])
    etx, ety = s["e_t"] * math.cos(s["w_t"]), s["e_t"] * math.sin(s["w_t"])
    de = math.hypot(esx - etx, esy - ety)
    tol = max(SUCCESS_TOL_A, PHI_ORBIT_SCALE_K * OBS_ALT_SCALE)
    phi_orbit = abs(da) / tol + de
    dth = s["th_s"] - s["th_t"]                     # <- raw TRUE ANOMALY (as in C)
    phi_phase = 1.0 - math.cos(dth)
    dx_l, dy_l, dvx_l, dvy_l, dist, vrel_i = lvlh(s)
    vrel_l = math.hypot(dvx_l, dvy_l)
    phi_vel = vrel_l / REL_VEL_TOL
    s2 = sig((EPS_ORBIT - phi_orbit) / TAU_ORBIT)
    s3 = s2 * sig((EPS_PHASE - phi_phase) / TAU_PHASE)
    # longitudes
    u_s, u_t = s["th_s"] + s["w_s"], s["th_t"] + s["w_t"]
    lam_s = true_to_mean(s["th_s"], s["e_s"]) + s["w_s"]
    lam_t = true_to_mean(s["th_t"], s["e_t"]) + s["w_t"]
    return dict(
        da=da, de=de, phi_orbit=phi_orbit, dtheta=wrap(dth), phi_phase=phi_phase,
        phi_vel=phi_vel, sigma2=s2, sigma3=s3, vrel_lvlh=vrel_l, dist=dist,
        vrel_inertial=vrel_i, dx_l=dx_l, dy_l=dy_l,
        g_true=wrap(u_s - u_t), g_mean=wrap(lam_s - lam_t),
        n_s=math.sqrt(MU / s["a_s"] ** 3), n_t=math.sqrt(MU / s["a_t"] ** 3),
    )


def phi_legacy(p):
    return -(W_ORBIT * p["phi_orbit"]
             + W_PHASE * p["phi_phase"] * p["sigma2"]
             + W_VEL * p["phi_vel"] * p["sigma3"])


def phi_SA(p):
    """S-A phase-first gates: orbit term gated on phase already being closed."""
    sA = sig((EPS_PHASE - p["phi_phase"]) / TAU_PHASE)
    s3 = sA * sig((EPS_ORBIT - p["phi_orbit"]) / TAU_ORBIT)
    return -(W_PHASE * p["phi_phase"]
             + W_ORBIT * p["phi_orbit"] * sA
             + W_VEL * p["phi_vel"] * s3)


def phi_SC(p):
    """S-C guidance target: da tracks a phase-proportional corridor, no sigma2."""
    da_star = max(-DA_MAX, min(DA_MAX, K_DASTAR * p["dtheta"]))
    tol = max(SUCCESS_TOL_A, PHI_ORBIT_SCALE_K * OBS_ALT_SCALE)
    phi_orbit_c = abs(p["da"] - da_star) / tol + p["de"]
    s3 = (sig((EPS_ORBIT - phi_orbit_c) / TAU_ORBIT)
          * sig((EPS_PHASE - p["phi_phase"]) / TAU_PHASE))
    return -(W_ORBIT * phi_orbit_c + W_PHASE * p["phi_phase"] + W_VEL * p["phi_vel"] * s3)


def _gdot_max(a_t):
    n_t = math.sqrt(MU / a_t ** 3)
    return 1.5 * (DA_MAX / a_t) * n_t


def tgo_realised(p, a_t):
    """S-B: time-to-go using the CURRENT drift rate (direction-aware)."""
    g, gd = p["g_mean"], p["n_s"] - p["n_t"]
    if gd > 0:
        travel = (-g) % (2.0 * math.pi)
    elif gd < 0:
        travel = g % (2.0 * math.pi)
    else:
        travel = math.inf
    t_phase = travel / abs(gd) if abs(gd) > 1e-12 else math.inf
    t_v = p["vrel_lvlh"] / A_MAX_RATE
    return min(T_CAP, t_phase + t_v)


def phi_SB(p, a_t, w=1.0):
    return -w * tgo_realised(p, a_t) / T_CAP


DV_REF = 300.0     # m/s — orbit-match dv normaliser for S_R3
W_PHASE_R3 = 1.0
W_ORBIT_R3 = 0.35


def dv_match(p, s):
    """Linearised two-impulse dv to move the chaser onto the target's orbit
    (Edelbaum / Gauss small-element form).  NOTE: adding |da|/1855 and de*v
    naively double-counts, because the single tangential burn that removes the
    phasing orbit's da also removes the e it created."""
    v_t = math.sqrt(MU / s["a_t"])
    return 0.5 * v_t * math.hypot(p["da"] / s["a_t"], p["de"])


def phi_SR3(p, s):
    lam = abs(p["g_mean"]) / math.pi
    orb = min(1.0, dv_match(p, s) / DV_REF)
    return -(W_PHASE_R3 * lam + W_ORBIT_R3 * orb)


def tgo_admissible(p, a_t, use_true_longitude=True):
    """S-R: admissible min-time heuristic. |gap| / max-achievable-drift-rate,
    plus the time to burn off the current LVLH relative velocity."""
    g = p["g_true"] if use_true_longitude else p["g_mean"]
    return abs(g) / _gdot_max(a_t) + p["vrel_lvlh"] / A_MAX_RATE


def phi_SR(p, a_t, use_true_longitude=True):
    t_ref = math.pi / _gdot_max(a_t)
    return -tgo_admissible(p, a_t, use_true_longitude) / t_ref


DESIGNS = ["legacy", "S_A", "S_B", "S_C", "S_R", "S_R3"]


def all_phis(s):
    p = phi_parts(s)
    return p, {
        "legacy": phi_legacy(p),
        "S_A": phi_SA(p),
        "S_B": phi_SB(p, s["a_t"]),
        "S_C": phi_SC(p),
        "S_R": phi_SR(p, s["a_t"], True),
        "S_R3": phi_SR3(p, s),
    }


# ── env driver ───────────────────────────────────────────────────────────────
def make_env(**kw):
    base = dict(num_envs=1, num_debris_min=0, num_debris_max=0,
                e_max_target=0.0, e_max_sat=0.0, same_orbit_init=1,
                init_phase_gap_max=0.0, valid_init_only=1,
                max_valid_init_attempts=4096, gave_up_action="terminate",
                phase_gap_fixed=-1.0, e_target_fixed=-1.0, e_sat_fixed=-1.0,
                omega_offset_fixed=-99.0, a_min_override=-1.0, a_max_override=-1.0,
                obs_alt_scale_m=OBS_ALT_SCALE, phi_orbit_scale_k=PHI_ORBIT_SCALE_K,
                lvlh_scale_m=R_EARTH, rendezvous_radius_m=30000.0,
                rel_vel_tol_ms=50.0, log_interval=1 << 30)
    base.update(kw)
    return Orbital(**base)


def dv_actions(dv_target, retro):
    """Greedy decomposition of |dv_target| into available burn magnitudes."""
    table = RETRO if retro else PRO
    out, rem = [], abs(dv_target)
    for m in (25, 10, 5, 2, 1):
        while rem >= m - 1e-9:
            out.append(table[m])
            rem -= m
    return out


def run_policy(env, policy, max_decisions=4000, tag=""):
    obs, _ = env.reset(seed=0)
    s = state_from_obs(obs[0])
    p, phis = all_phis(s)
    rows = []
    prev = dict(s=s, p=p, phis=phis)
    d = 0
    while d < max_decisions:
        act, leg = policy(prev["s"], prev["p"], d)
        if act is None:
            break
        obs, rew, term, trunc, _ = env.step(np.array([act], dtype=np.int32))
        d += 1
        tau = ACTION_TAU[act]
        cause = 0
        if term[0]:
            # the env autoresets inside c_step, so obs now describes the NEXT
            # episode.  At a state-caused terminal PBRS uses Phi(s')=0 (the NHR
            # clamp); the safety-cap branch deliberately pays nothing.
            _, cause = env.last_episode_result(0)
            s2 = prev["s"]
            p2 = prev["p"]
            phis2 = {k: 0.0 for k in DESIGNS}
        else:
            s2 = state_from_obs(obs[0])
            p2, phis2 = all_phis(s2)
        row = dict(tag=tag, decision=d, leg=leg, action=act, tau=tau,
                   reward=float(rew[0]), terminal=int(term[0]), cause=cause)
        for k in ("a_s", "a_t", "e_s", "e_t", "th_s", "th_t", "w_s", "w_t", "fuel"):
            row[k] = prev["s"][k]
        for k in ("da", "de", "phi_orbit", "dtheta", "phi_phase", "phi_vel",
                  "sigma2", "sigma3", "vrel_lvlh", "dist", "g_true", "g_mean",
                  "dx_l", "dy_l"):
            row[k] = prev["p"][k]
        cap = (cause == 4)
        for dn in DESIGNS:
            row["phi_" + dn] = prev["phis"][dn]
            row["phi_next_" + dn] = phis2[dn]
            gt = GAMMA ** tau
            if term[0] and cap:
                r_gt = r_g1 = r_gd = 0.0        # no clamp at the safety cap
            elif term[0]:
                r_gt = r_g1 = r_gd = BETA_SHAPE * (0.0 - prev["phis"][dn])
            else:
                r_gt = BETA_SHAPE * (gt * phis2[dn] - prev["phis"][dn])
                r_g1 = BETA_SHAPE * (phis2[dn] - prev["phis"][dn])
                r_gd = BETA_SHAPE * (GAMMA * phis2[dn] - prev["phis"][dn])
            row["r_" + dn + "_gtau"] = r_gt
            row["r_" + dn + "_g1"] = r_g1
            row["r_" + dn + "_gdec"] = r_gd
        if term[0]:
            row["leg"] = "TERMINAL"
        rows.append(row)
        if term[0]:
            break
        prev = dict(s=s2, p=p2, phis=phis2)
    return rows


# ── scripted expert ──────────────────────────────────────────────────────────
EARTH_KEEPOUT = R_EARTH + 200e3


def dv_for_da(a, da):
    """First-order tangential dv needed to change semi-major axis by da."""
    v = math.sqrt(MU / a)
    return da * MU / (2.0 * a * a * v)


def plan_phasing(a_s, a_t, g0, t_budget, da_cap=DA_MAX, alt_margin=25e3,
                 n_max=30):
    """Least-dv N-revolution phasing orbit that nulls the mean-longitude gap g0
    (chaser - target) within t_budget, respecting the 200 km altitude floor.

    Chaser BEHIND (g0 < 0) can either LOWER (close |g0|, k=0) or RAISE
    (close 2pi-|g0|, k=+1).  Lowering is fuel-cheap but perigee-limited: a
    tangential burn at r = a_s drops the perigee by 2|da|.
    """
    T0 = 2.0 * math.pi * math.sqrt(a_t ** 3 / MU)
    best = None
    for k in (0, 1, -1, 2):
        for N in range(1, n_max + 1):
            T1 = T0 * (g0 / (2.0 * math.pi) + N + k) / N
            if T1 <= 0.3 * T0:
                continue
            a1 = a_t * (T1 / T0) ** (2.0 / 3.0)
            r_other = 2.0 * a1 - a_s
            rp, ra = min(a_s, r_other), max(a_s, r_other)
            if rp < EARTH_KEEPOUT + alt_margin:
                continue
            if ra > R_EARTH + 2.5e6:
                continue
            if abs(a1 - a_s) > da_cap:
                continue
            if N * T1 > t_budget:
                continue
            v1 = math.sqrt(MU * max(1e-9, 2.0 / a_s - 1.0 / a1))
            dv = v1 - math.sqrt(MU / a_s)
            if 2.0 * abs(dv) > 460.0:          # fuel budget ~478 m/s
                continue
            cand = dict(dv=dv, N=N, a1=a1, T1=T1, T_tot=N * T1, k=k,
                        da=a1 - a_s, rp=rp, ra=ra)
            if best is None or abs(dv) < abs(best["dv"]):
                best = cand
    return best


def _warp_for(t_wait):
    if t_wait > 3900:
        return WARP60
    if t_wait > 2100:
        return WARP30
    if t_wait > 400:
        return WARP5
    return COAST


class Expert:
    """Physically correct phasing, closed-loop:
      A_open  tangential burns onto an N-rev phasing orbit (perigee-safe)
      B_drift warp/coast to the apsis passage whose predicted gap is smallest
      C_close tangential burns back onto the target semi-major axis + da trim
      F_*     fine phasing rounds (1-25 m/s) to shrink the residual gap
      D_null  trim residual LVLH relative velocity, then coast
    """

    def __init__(self, verbose=False, fine_rounds_max=3):
        self.stage = "plan"
        self.q = []
        self.sim = 0
        self.plan = {}
        self.verbose = verbose
        self.trims = 0
        self.fine_rounds = 0
        self.fine_rounds_max = fine_rounds_max
        self.fine = {}
        self.dsteps = 0
        self.infeasible = False
        self.xfer = {}

    def __call__(self, s, p, d):
        act, leg = self._decide(s, p, d)
        if act is not None:
            self.sim += ACTION_TAU[act]
        return act, leg

    # --------------------------------------------------------------- planning
    def _decide(self, s, p, d):
        if self.stage == "plan":
            if abs(s["a_s"] - s["a_t"]) > 20e3:
                r1, r2 = s["a_s"], s["a_t"]
                a_tr = 0.5 * (r1 + r2)
                dv1 = (math.sqrt(MU * (2.0 / r1 - 1.0 / a_tr))
                       - math.sqrt(MU / r1))
                dv2 = (math.sqrt(MU / r2)
                       - math.sqrt(MU * (2.0 / r2 - 1.0 / a_tr)))
                self.xfer = dict(dv1=dv1, dv2=dv2, a_tr=a_tr, up=(r2 > r1))
                self.q = dv_actions(dv1, retro=(dv1 < 0))
                self.stage = "T1"
                if self.verbose:
                    print(f"  [xfer] da0={(r2-r1)/1e3:+7.1f} km  dv1={dv1:+6.2f} "
                          f"dv2={dv2:+6.2f} m/s")
            else:
                self.stage = "plan2"

        if self.stage == "T1":
            if self.q:
                return self.q.pop(0), "T_xfer"
            self.stage = "T2"

        if self.stage == "T2":
            # coast/warp to the far apsis of the transfer ellipse
            th_ap = math.pi if self.xfer["up"] else 0.0
            n1 = p["n_s"]
            dM = (true_to_mean(th_ap, s["e_s"])
                  - true_to_mean(s["th_s"], s["e_s"])) % (2.0 * math.pi)
            nq = len(dv_actions(self.xfer["dv2"], retro=True))
            t_wait = dM / n1 - 0.5 * nq * DT
            if t_wait > 45.0:
                return _warp_for(t_wait), "T_xfer"
            self.stage = "T3"
            self.q = dv_actions(self.xfer["dv2"], retro=(self.xfer["dv2"] < 0))

        if self.stage == "T3":
            if self.q:
                return self.q.pop(0), "T_xfer"
            dv_need = abs(dv_for_da(s["a_t"], p["da"]))
            if dv_need >= 1.0 and self.trims < 25:
                self.trims += 1
                mag = next(m for m in (25, 10, 5, 2, 1) if dv_need >= m)
                return (PRO[mag] if p["da"] < 0 else RETRO[mag]), "T_xfer"
            self.trims = 0
            self.stage = "plan2"

        if self.stage == "plan2":
            b = plan_phasing(s["a_s"], s["a_t"], p["g_mean"],
                             t_budget=max(300.0, (MAX_STEPS - self.sim - 220) * DT))
            if b is None:
                self.infeasible = True
                self.plan = dict(dv=0.0, N=0, da=0.0, T_tot=0.0, k=0,
                                 a1=s["a_s"], rp=s["a_s"], ra=s["a_s"], T1=0.0)
                self.stage = "F"
            else:
                self.plan = b
                self.q = dv_actions(b["dv"], retro=(b["dv"] < 0))
                self.stage = "A"
                if self.verbose:
                    print(f"  [plan] g0={math.degrees(p['g_mean']):+7.2f}deg "
                          f"k={b['k']} N={b['N']} da={b['da']/1e3:+7.1f}km "
                          f"dv={b['dv']:+7.2f}m/s x2  rp={(b['rp']-R_EARTH)/1e3:.0f}km alt "
                          f"drift={b['T_tot']/DT:.0f} steps  burns={len(self.q)}")

        if self.stage == "A":
            if self.q:
                return self.q.pop(0), "A_open"
            self.stage = "B"

        if self.stage == "B":
            gd = p["n_s"] - p["n_t"]
            if abs(gd) < 1e-11:
                self.stage = "C"
            else:
                # the phasing orbit's apsis at r = a_s is theta = pi when we
                # lowered (burn point becomes apoapsis) and theta = 0 when we
                # raised.  Wait for the apsis passage with the smallest |gap|.
                th_ap = math.pi if self.plan["dv"] < 0 else 0.0
                n1 = p["n_s"]
                T1 = 2.0 * math.pi / n1
                dM = (true_to_mean(th_ap, s["e_s"])
                      - true_to_mean(s["th_s"], s["e_s"])) % (2.0 * math.pi)
                t_ap = dM / n1
                nq = len(dv_actions(self.plan["dv"], retro=True))
                lead = 0.5 * nq * DT
                cands = [(abs(wrap(p["g_mean"] + gd * (t_ap + kk * T1))), kk)
                         for kk in range(0, self.plan["N"] + 4)]
                _, kbest = min(cands)
                t_wait = t_ap + kbest * T1 - lead
                if t_wait > 45.0:
                    return _warp_for(t_wait), "B_drift"
                self.stage = "C"
                self.q = dv_actions(self.plan["dv"], retro=(self.plan["dv"] > 0))

        if self.stage == "C":
            if self.q:
                return self.q.pop(0), "C_close"
            dv_need = abs(dv_for_da(s["a_t"], p["da"]))
            if dv_need >= 1.0 and self.trims < 40:
                self.trims += 1
                mag = next(m for m in (25, 10, 5, 2, 1) if dv_need >= m)
                return (PRO[mag] if p["da"] < 0 else RETRO[mag]), "C_close"
            self.stage = "F"

        if self.stage == "F":
            g = p["g_true"]
            miss = abs(g) * s["a_t"]
            t_avail = (MAX_STEPS - self.sim - 90) * DT
            if (miss < 20e3 or self.fine_rounds >= self.fine_rounds_max
                    or t_avail < 900.0):
                self.stage = "D"
            else:
                self.fine_rounds += 1
                n_t = p["n_t"]
                da_f = (2.0 / 3.0) * s["a_t"] * g / (n_t * 0.75 * t_avail)
                dv_f = dv_for_da(s["a_t"], da_f)
                if abs(dv_f) < 1.0:
                    dv_f = math.copysign(1.0, dv_f)
                if abs(dv_f) > 25.0:
                    dv_f = math.copysign(25.0, dv_f)
                self.fine = dict(dv=dv_f)
                self.q = dv_actions(dv_f, retro=(dv_f < 0))
                self.stage = "F_open"

        if self.stage == "F_open":
            if self.q:
                return self.q.pop(0), "F_open"
            self.stage = "F_drift"

        if self.stage == "F_drift":
            gd = p["n_s"] - p["n_t"]
            g = p["g_true"]
            nq = len(dv_actions(self.fine["dv"], retro=True))
            t_left = (-g / gd - 0.5 * nq * DT) if abs(gd) > 1e-13 else -1.0
            if self.sim > MAX_STEPS - 40:
                self.stage = "D"
            elif t_left > 45.0:
                return _warp_for(t_left), "F_drift"
            else:
                self.stage = "F_close"
                self.q = dv_actions(self.fine["dv"], retro=(self.fine["dv"] > 0))

        if self.stage == "F_close":
            if self.q:
                return self.q.pop(0), "F_close"
            dv_need = abs(dv_for_da(s["a_t"], p["da"]))
            if dv_need >= 1.0 and self.trims < 60:
                self.trims += 1
                mag = next(m for m in (25, 10, 5, 2, 1) if dv_need >= m)
                return (PRO[mag] if p["da"] < 0 else RETRO[mag]), "F_close"
            self.stage = "F"
            return self._decide(s, p, d)

        if self.stage == "D":
            self.dsteps += 1
            if self.dsteps > 500 or self.sim >= MAX_STEPS - 2:
                return None, "D_null"
            dx, dy, dvx, dvy, _, _ = lvlh(s)
            if abs(dvy) >= 1.0 and abs(dvy) >= abs(dvx):
                mag = next(m for m in (25, 10, 5, 2, 1) if abs(dvy) >= m)
                return (PRO[mag] if dvy > 0 else RETRO[mag]), "D_null"
            if abs(dvx) >= 10.0:
                return (7 if dvx > 0 else 8), "D_null"
            return COAST, "D_null"
        return COAST, "D_null"


def coast_policy(s, p, d):
    return COAST, "null"


def warp_policy(s, p, d):
    return WARP60, "null"


class ParkFarm:
    """Adversary: burn away from the target orbit, then warp forever."""

    def __init__(self, n_burns=6):
        self.n = n_burns

    def __call__(self, s, p, d):
        if d < self.n:
            return 3, "farm_open"     # prograde 25
        return WARP60, "farm_warp"


# ── analysis helpers ─────────────────────────────────────────────────────────
def leg_table(rows, design, key):
    legs, order = {}, []
    for r in rows:
        L = r["leg"]
        if L not in legs:
            legs[L] = dict(n=0, ret=0.0, worst=0.0, steps=0)
            order.append(L)
        legs[L]["n"] += 1
        legs[L]["steps"] += r["tau"]
        legs[L]["ret"] += r[key.format(design)]
        legs[L]["worst"] = min(legs[L]["worst"], r[key.format(design)])
    return order, legs


def write_csv(path, rows):
    if not rows:
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x, n=4):
    return f"{x:+.{n}f}"


# ═════════════════════════════════════════════════════════════════════════════
# Q1 — the drift-orbit valley
# ═════════════════════════════════════════════════════════════════════════════
ALT_LO = R_EARTH + 750e3
ALT_HI = R_EARTH + 800e3


def q1(gaps_deg=(90.0, 150.0, 180.0), e_fix=0.0, verbose=True,
       alt_lo=ALT_LO, alt_hi=ALT_HI):
    all_rows, summary = [], []
    for gd in gaps_deg:
        gap = math.radians(gd)
        # phase_gap_fixed sets tgt_M = sat_M + gap  ->  chaser is BEHIND by `gap`
        env = make_env(same_orbit_init=1, e_target_fixed=e_fix, e_sat_fixed=e_fix,
                       phase_gap_fixed=gap, init_phase_gap_max=gap,
                       a_min_override=alt_lo, a_max_override=alt_hi)
        ex = Expert(verbose=verbose)
        if verbose:
            print(f"\n=== expert @ gap {gd:.0f} deg, e={e_fix} ===")
        rows = run_policy(env, ex, tag=f"expert_g{int(gd)}_e{e_fix}")
        env.close()
        last = rows[-1]
        if verbose:
            print(f"  [done] {len(rows)} decisions / {sum(r['tau'] for r in rows)} "
                  f"sim steps  terminal={last['terminal']} r={last['reward']:+.3f} "
                  f"miss={last['dist']/1e3:.1f} km vrel={last['vrel_lvlh']:.2f} m/s "
                  f"fuel_left={last['fuel']:.4f} fine_rounds={ex.fine_rounds}")
        all_rows += rows
        summary.append((f"expert_g{int(gd)}", rows, ex.plan))
    # headline-shaped scenario: different orbits (transfer) AND a phase gap
    env = make_env(same_orbit_init=0, e_target_fixed=e_fix, e_sat_fixed=e_fix,
                   phase_gap_fixed=math.radians(120.0),
                   init_phase_gap_max=math.radians(120.0),
                   a_min_override=R_EARTH + 500e3, a_max_override=R_EARTH + 800e3)
    ex = Expert(verbose=verbose)
    if verbose:
        print(f"\n=== expert transfer+phasing (da0 != 0, gap 120 deg) ===")
    rows = run_policy(env, ex, tag="expert_xfer_g120")
    env.close()
    if verbose:
        last = rows[-1]
        print(f"  [done] {len(rows)} decisions / {sum(r['tau'] for r in rows)} "
              f"sim steps  cause={last['cause']} r={last['reward']:+.3f} "
              f"fuel_left={last['fuel']:.4f}")
    all_rows += rows
    summary.append(("expert_xfer_g120", rows, ex.plan))

    # null policies at the hardest gap
    for name, pol in (("coast", coast_policy), ("warp60", warp_policy),
                      ("parkfarm", ParkFarm(6))):
        env = make_env(same_orbit_init=1, e_target_fixed=e_fix, e_sat_fixed=e_fix,
                       phase_gap_fixed=math.radians(180.0),
                       init_phase_gap_max=math.radians(180.0),
                       a_min_override=alt_lo, a_max_override=alt_hi)
        rows = run_policy(env, pol, tag=f"{name}_e{e_fix}")
        env.close()
        all_rows += rows
        summary.append((name, rows, {}))
    return all_rows, summary


def feasibility_scan():
    """For each (initial altitude, gap): is a 2-burn N-rev phasing maneuver
    feasible inside 2000 steps / 478 m/s with a 200 km perigee floor?"""
    rows = []
    for alt_km in (300, 400, 500, 600, 700, 800):
        a0 = R_EARTH + alt_km * 1e3
        for gdeg in (-180, -135, -90, -45, -15, 15, 45, 90, 135, 180):
            b = plan_phasing(a0, a0, math.radians(gdeg), t_budget=1450 * DT)
            rows.append(dict(alt_km=alt_km, gap_deg=gdeg,
                             feasible=int(b is not None),
                             dv_one_way=(b["dv"] if b else float("nan")),
                             dv_total=(2 * abs(b["dv"]) if b else float("nan")),
                             da_km=(b["da"] / 1e3 if b else float("nan")),
                             N=(b["N"] if b else -1),
                             direction=("lower" if b and b["dv"] < 0
                                        else ("raise" if b else "none")),
                             drift_steps=(b["T_tot"] / DT if b else float("nan"))))
    return rows


def q1_report(summary, gamma_key):
    """gamma_key in {'r_{}_gtau','r_{}_g1','r_{}_gdec'}"""
    lines = []
    for name, rows, plan in summary:
        lines.append(f"\n### {name}   ({len(rows)} decisions, "
                     f"{sum(r['tau'] for r in rows)} sim steps, "
                     f"terminal_reward={rows[-1]['reward']:+.3f})")
        if plan:
            lines.append(f"    plan: N={plan['N']} da*={plan['da']/1e3:+.1f} km "
                         f"dv={plan['dv']:+.1f} m/s (x2)")
        order, _ = leg_table(rows, "legacy", gamma_key)
        hdr = f"    {'leg':<10}{'n':>5}{'steps':>7}" + "".join(f"{d:>12}" for d in DESIGNS)
        lines.append(hdr)
        for L in order:
            cells = []
            for d in DESIGNS:
                _, lt = leg_table(rows, d, gamma_key)
                cells.append(f"{lt[L]['ret']:>+12.4f}")
            n = sum(1 for r in rows if r["leg"] == L)
            st = sum(r["tau"] for r in rows if r["leg"] == L)
            lines.append(f"    {L:<10}{n:>5}{st:>7}" + "".join(cells))
        tot, worst = [], []
        for d in DESIGNS:
            tot.append(sum(r[gamma_key.format(d)] for r in rows))
            worst.append(min([r[gamma_key.format(d)] for r in rows] or [0.0]))
        lines.append(f"    {'TOTAL':<10}{len(rows):>5}{sum(r['tau'] for r in rows):>7}"
                     + "".join(f"{t:>+12.4f}" for t in tot))
        lines.append(f"    {'worst step':<10}{'':>5}{'':>7}"
                     + "".join(f"{w:>+12.4f}" for w in worst))
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Q2 — the omega question
# ═════════════════════════════════════════════════════════════════════════════
def q2_reset_stats(n=4000):
    """Headline conditions: e_max_target = e_max_sat = 0.05, gap +-180, soi=0."""
    out = {}
    cfgs = {
        "headline(e=0.05,soi=0)": dict(e_max_target=0.05, e_max_sat=0.05,
                                       same_orbit_init=0,
                                       init_phase_gap_max=math.pi),
        "phase4(e=0,soi=0)": dict(e_max_target=0.0, e_max_sat=0.0,
                                  same_orbit_init=0, init_phase_gap_max=math.pi),
        "stage1(e=0.05,soi=1)": dict(e_max_target=0.05, e_max_sat=0.05,
                                     same_orbit_init=1, init_phase_gap_max=math.pi),
    }
    for name, kw in cfgs.items():
        env = make_env(valid_init_only=1, **kw)
        dw, gtrue, gmean, gth, dphi_dist, phase_target_miss = [], [], [], [], [], []
        for i in range(n):
            obs, _ = env.reset(seed=1000 + i)
            s = state_from_obs(obs[0])
            p = phi_parts(s)
            dw.append(wrap(s["w_s"] - s["w_t"]))
            gtrue.append(p["g_true"])
            gmean.append(p["g_mean"])
            gth.append(p["dtheta"])
            dphi_dist.append(p["dist"])
            # where does Phi_phase = 0 put us?  theta_s = theta_t  ->  the chaser
            # sits at true longitude theta_t + w_s, target at theta_t + w_t.
            phase_target_miss.append(abs(wrap(s["w_s"] - s["w_t"])))
        env.close()
        out[name] = dict(
            dw=np.array(dw), g_true=np.array(gtrue), g_mean=np.array(gmean),
            dtheta=np.array(gth), dist=np.array(dphi_dist),
            miss=np.array(phase_target_miss))
    return out


def q2_omega_drift(gap_deg=180.0, e_fix=0.0):
    """omega_sat / omega_target across a scripted episode (same_orbit_init=1)."""
    gap = math.radians(gap_deg)
    env = make_env(same_orbit_init=1, e_target_fixed=e_fix, e_sat_fixed=e_fix,
                   phase_gap_fixed=gap, init_phase_gap_max=gap)
    ex = Expert()
    rows = run_policy(env, ex, tag=f"omegadrift_g{int(gap_deg)}_e{e_fix}")
    env.close()
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# drift-direction sign check (empirical, per the brief: do NOT trust derivation)
# ═════════════════════════════════════════════════════════════════════════════
def sign_check():
    res = []
    for sgn, name, act in ((-1, "retrograde-25 (lower a)", 6),
                           (+1, "prograde-25 (raise a)", 3)):
        env = make_env(same_orbit_init=1, e_target_fixed=0.0, e_sat_fixed=0.0,
                       phase_gap_fixed=math.radians(90.0),
                       init_phase_gap_max=math.radians(90.0),
                       a_min_override=R_EARTH + 750e3, a_max_override=R_EARTH + 800e3)
        obs, _ = env.reset(seed=7)
        s0 = state_from_obs(obs[0]); p0 = phi_parts(s0)
        for _ in range(4):
            obs, *_ = env.step(np.array([act], dtype=np.int32))
        s1 = state_from_obs(obs[0]); p1 = phi_parts(s1)
        killed = 0
        for _ in range(6):
            obs, r, t, tr, _ = env.step(np.array([WARP60], dtype=np.int32))
            if t[0]:
                killed = 1
                break
            s2 = state_from_obs(obs[0]); p2 = phi_parts(s2)
        s2 = state_from_obs(obs[0]); p2 = phi_parts(s2)
        env.close()
        res.append(dict(name=name, da_km=(s1["a_s"] - s1["a_t"]) / 1e3,
                        g0_deg=math.degrees(p0["g_mean"]),
                        g1_deg=math.degrees(p1["g_mean"]),
                        g2_deg=math.degrees(p2["g_mean"]),
                        gdot_deg_hr=math.degrees(p1["n_s"] - p1["n_t"]) * 3600.0,
                        killed=killed))
    return res


# ═════════════════════════════════════════════════════════════════════════════
def validate_python_phi(rows):
    """Cross-check: realised C reward on non-terminal steps must equal
    beta*(gamma^tau*Phi'-Phi) with the python legacy Phi."""
    errs = [abs(r["reward"] - r["r_legacy_gtau"]) for r in rows if not r["terminal"]]
    return (max(errs) if errs else 0.0), len(errs)


def main():
    os.makedirs(OUT, exist_ok=True)
    np.set_printoptions(suppress=True)

    print("=" * 78)
    print("SIGN CHECK — which burn direction closes a 'chaser behind' gap")
    print("=" * 78)
    for r in sign_check():
        print(f"  {r['name']:<26} da={r['da_km']:+8.1f} km  "
              f"gdot={r['gdot_deg_hr']:+7.3f} deg/hr  "
              f"g: {r['g0_deg']:+7.2f} -> {r['g1_deg']:+7.2f} -> {r['g2_deg']:+7.2f} deg"
              f"  (6h later)")

    print()
    print("=" * 78)
    print("Q1 — scripted-expert Phi trace")
    print("=" * 78)
    rows, summary = q1()
    err, n = validate_python_phi(rows)
    print(f"\n[validation] max |realised C reward - python legacy shaping| over "
          f"{n} non-terminal steps = {err:.3e}")

    write_csv(os.path.join(OUT, "t3_shaping_trace.csv"), rows)
    print(f"[csv] {os.path.join(OUT, 't3_shaping_trace.csv')}  ({len(rows)} rows)")

    for gk, label in (("r_{}_gtau", "AS IMPLEMENTED: gamma^tau per sub-step"),
                      ("r_{}_gdec", "gamma^1 per agent decision (PPO-consistent)"),
                      ("r_{}_g1", "gamma_shape = 1 (pure telescoping)")):
        print("\n" + "-" * 78)
        print(f"PER-LEG SHAPING RETURN — {label}")
        print("-" * 78)
        print(q1_report(summary, gk))

    # terminal clamp accounting for the legacy design
    print("\n" + "-" * 78)
    print("TERMINAL ACCOUNTING (legacy)")
    print("-" * 78)
    for name, rws, _ in summary:
        last = rws[-1]
        print(f"  {name:<16} terminal={last['terminal']}  reward_last={last['reward']:+.4f} "
              f"Phi_prev={last['phi_legacy']:+.5f}  dist={last['dist']/1e3:9.1f} km "
              f"vrel={last['vrel_lvlh']:8.2f} m/s  fuel_left={last['fuel']:.4f}")

    comp = []
    for name, rws, _ in summary:
        for dn in DESIGNS:
            for gk, gl in (("r_{}_gtau", "gamma^tau"), ("r_{}_gdec", "gamma^1/decision"),
                           ("r_{}_g1", "gamma_shape=1")):
                vals = [r[gk.format(dn)] for r in rws]
                comp.append(dict(scenario=name, design=dn, gamma_form=gl,
                                 decisions=len(rws), total=sum(vals),
                                 worst_step=min(vals), best_step=max(vals),
                                 open_leg=sum(r[gk.format(dn)] for r in rws
                                              if r["leg"] in ("A_open", "F_open")),
                                 drift_leg=sum(r[gk.format(dn)] for r in rws
                                               if r["leg"] in ("B_drift", "F_drift")),
                                 close_leg=sum(r[gk.format(dn)] for r in rws
                                               if r["leg"] in ("C_close", "F_close")),
                                 xfer_leg=sum(r[gk.format(dn)] for r in rws
                                              if r["leg"] == "T_xfer"),
                                 terminal=sum(r[gk.format(dn)] for r in rws
                                              if r["leg"] == "TERMINAL")))
    write_csv(os.path.join(OUT, "t3_shaping_design_comparison.csv"), comp)
    print(f"[csv] {os.path.join(OUT, 't3_shaping_design_comparison.csv')}")

    print("\n" + "-" * 78)
    print("GATE ACTIVATION + gamma^tau LEAK (legacy design, expert paths)")
    print("-" * 78)
    for name, rws, _ in summary:
        n_open = sum(1 for r in rws if r["sigma2"] > 0.01)
        leak = sum(r["r_legacy_gtau"] - r["r_legacy_g1"] for r in rws)
        leak_sr = sum(r["r_S_R_gtau"] - r["r_S_R_g1"] for r in rws)
        warpsteps = sum(r["tau"] for r in rws if r["tau"] > 1)
        print(f"  {name:<18} sigma2>0.01 on {n_open:>4}/{len(rws):<4} decisions "
              f"({100.0*n_open/len(rws):5.1f}%)   gamma^tau leak: legacy "
              f"{leak:+7.4f}   S_R {leak_sr:+7.4f}   warp sim-steps {warpsteps}")

    print("\n" + "-" * 78)
    print("FEASIBILITY OF THE 2-BURN PHASING MANEUVER (200 km perigee floor, "
          "1450 steps, 460 m/s)")
    print("-" * 78)
    fs = feasibility_scan()
    write_csv(os.path.join(OUT, "t3_phasing_feasibility.csv"), fs)
    alts = sorted({r["alt_km"] for r in fs})
    gaps = sorted({r["gap_deg"] for r in fs})
    print("    alt\\gap " + "".join(f"{g:>9}" for g in gaps))
    for a in alts:
        cells = []
        for g in gaps:
            r = next(x for x in fs if x["alt_km"] == a and x["gap_deg"] == g)
            cells.append("     ----" if not r["feasible"]
                         else f"{r['dv_total']:>7.0f}{'d' if r['direction']=='lower' else 'u'}")
        print(f"    {a:>7} " + "".join(cells))
    print("    (cell = total two-burn dv in m/s; d = lower/catch-up, "
          "u = raise/lap-around; ---- = infeasible)")

    print()
    print("=" * 78)
    print("Q2 — the omega question")
    print("=" * 78)
    stats = q2_reset_stats(n=3000)
    q2rows = []
    for name, d in stats.items():
        dw = np.abs(d["dw"])
        print(f"\n  {name}")
        print(f"    |omega_s - omega_t| (rad): mean {dw.mean():.4f}  median "
              f"{np.median(dw):.4f}  p90 {np.percentile(dw,90):.4f}  max {dw.max():.4f}")
        print(f"    |omega_s - omega_t| (deg): mean {math.degrees(dw.mean()):.2f}  "
              f"median {math.degrees(np.median(dw)):.2f}  "
              f"max {math.degrees(dw.max()):.2f}")
        for key, lbl in (("dtheta", "theta_s - theta_t (what Phi_phase uses)"),
                         ("g_true", "true longitude gap (physical)"),
                         ("g_mean", "mean  longitude gap (drift-relevant)")):
            v = np.abs(d[key])
            print(f"    |{lbl:<40}| mean {math.degrees(v.mean()):7.2f} deg  "
                  f"sd {math.degrees(v.std()):6.2f}  max {math.degrees(v.max()):7.2f}")
        # miss distance if the agent satisfies Phi_phase = 0 exactly
        miss_ang = d["miss"]
        a_typ = 7.0e6
        print(f"    if Phi_phase driven to 0 (theta_s = theta_t): residual true-"
              f"longitude error = |dw| -> chord miss ~ 2a sin(|dw|/2):")
        chord = 2 * a_typ * np.sin(miss_ang / 2)
        print(f"        mean {chord.mean()/1e3:8.1f} km   median "
              f"{np.median(chord)/1e3:8.1f} km   min {chord.min()/1e3:8.1f} km "
              f"(success box = 30 km)")
        frac = float((chord < 30e3).mean())
        print(f"        fraction of episodes where Phi_phase=0 is within the "
              f"30 km box: {frac*100:.2f}%")
        q2rows.append(dict(config=name, n=len(dw),
                           dw_mean_deg=math.degrees(dw.mean()),
                           dw_med_deg=math.degrees(np.median(dw)),
                           dw_max_deg=math.degrees(dw.max()),
                           dtheta_sd_deg=math.degrees(np.std(np.abs(d["dtheta"]))),
                           gtrue_mean_deg=math.degrees(np.abs(d["g_true"]).mean()),
                           gmean_mean_deg=math.degrees(np.abs(d["g_mean"]).mean()),
                           chord_miss_mean_km=chord.mean() / 1e3,
                           chord_miss_med_km=float(np.median(chord)) / 1e3,
                           frac_in_30km=frac))
    write_csv(os.path.join(OUT, "t3_omega_audit.csv"), q2rows)

    # does init_phase_gap_max still control the physical gap when e>0?
    print("\n  init_phase_gap_max sweep (e_max=0.05, soi=0): does it control the "
          "PHYSICAL gap?")
    sweep = []
    for gmax_deg in (10.0, 30.0, 90.0, 180.0):
        env = make_env(e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
                       init_phase_gap_max=math.radians(gmax_deg), valid_init_only=1)
        gt, gm, dth = [], [], []
        for i in range(1500):
            obs, _ = env.reset(seed=5000 + i)
            p = phi_parts(state_from_obs(obs[0]))
            gt.append(abs(math.degrees(p["g_true"])))
            gm.append(abs(math.degrees(p["g_mean"])))
            dth.append(abs(math.degrees(p["dtheta"])))
        env.close()
        print(f"    gap_max={gmax_deg:6.1f} deg -> |true-lon gap| mean "
              f"{np.mean(gt):7.2f} max {np.max(gt):7.2f} | "
              f"|mean-lon gap| mean {np.mean(gm):7.2f} | "
              f"|dtheta| (Phi_phase) mean {np.mean(dth):7.2f}")
        sweep.append(dict(gap_max_deg=gmax_deg, true_lon_gap_mean_deg=float(np.mean(gt)),
                          true_lon_gap_max_deg=float(np.max(gt)),
                          mean_lon_gap_mean_deg=float(np.mean(gm)),
                          dtheta_mean_deg=float(np.mean(dth))))
        # same at e=0 for contrast
    for gmax_deg in (30.0, 180.0):
        env = make_env(e_max_target=0.0, e_max_sat=0.0, same_orbit_init=0,
                       init_phase_gap_max=math.radians(gmax_deg), valid_init_only=1)
        gt = []
        for i in range(800):
            obs, _ = env.reset(seed=9000 + i)
            gt.append(abs(math.degrees(phi_parts(state_from_obs(obs[0]))["g_true"])))
        env.close()
        print(f"    [e=0]    gap_max={gmax_deg:6.1f} deg -> |true-lon gap| mean "
              f"{np.mean(gt):7.2f} max {np.max(gt):7.2f}")
        sweep.append(dict(gap_max_deg=gmax_deg, true_lon_gap_mean_deg=float(np.mean(gt)),
                          true_lon_gap_max_deg=float(np.max(gt)),
                          mean_lon_gap_mean_deg=float("nan"),
                          dtheta_mean_deg=float("nan")))
    write_csv(os.path.join(OUT, "t3_phase_gap_sweep.csv"), sweep)

    print("\n  omega drift across a scripted episode (same_orbit_init=1, e=0):")
    for efix in (0.0, 0.03):
        dr = q2_omega_drift(180.0, efix)
        dws = [math.degrees(wrap(r["w_s"] - r["w_t"])) for r in dr]
        jumps = []
        for i in range(1, len(dr)):
            jumps.append((abs(math.degrees(wrap(dr[i]["w_s"] - dr[i - 1]["w_s"]))),
                          dr[i - 1]["action"], i))
        jumps.sort(reverse=True)
        th_jumps = [abs(math.degrees(wrap(dr[i]["th_s"] - dr[i - 1]["th_s"]
                                          - (dr[i]["th_t"] - dr[i - 1]["th_t"]))))
                    for i in range(1, len(dr)) if ACTION_TAU[dr[i - 1]["action"]] == 1
                    and dr[i - 1]["action"] != 0]
        print(f"    e_init={efix}:  omega_s-omega_t  start {dws[0]:+8.3f} deg -> "
              f"end {dws[-1]:+8.3f} deg | range [{min(dws):+.2f},{max(dws):+.2f}] | "
              f"largest single-decision d(omega_s) = {jumps[0][0]:.2f} deg "
              f"(after action {jumps[0][1]})")
        if th_jumps:
            print(f"              burn-induced jump in (theta_s - theta_t): "
                  f"median {np.median(th_jumps):.2f} deg  max {max(th_jumps):.2f} deg")

    print("\n  BURN-INDUCED DISCONTINUITY IN (theta_s - theta_t)  [the quantity "
          "Phi_phase and obs[13-14] are built from]")
    probe = []
    for act in (3, 6, 2, 5, 12, 13, 7, 8):
        for e0 in (0.0, 0.02, 0.05):
            env = make_env(same_orbit_init=1, e_target_fixed=e0, e_sat_fixed=e0,
                           phase_gap_fixed=math.radians(90.0),
                           init_phase_gap_max=math.radians(90.0),
                           a_min_override=R_EARTH + 700e3,
                           a_max_override=R_EARTH + 800e3)
            djs, dphis, dphys, dws = [], [], [], []
            for k in range(40):
                obs, _ = env.reset(seed=20000 + k)
                for _ in range(k % 20):          # spread the burn over one orbit
                    obs, *_ = env.step(np.array([COAST], dtype=np.int32))
                s0 = state_from_obs(obs[0]); p0 = phi_parts(s0)
                obs, *_ = env.step(np.array([act], dtype=np.int32))
                s1 = state_from_obs(obs[0]); p1 = phi_parts(s1)
                # what a 60 s coast alone would have done to dtheta
                obs2, _ = env.reset(seed=20000 + k)
                for _ in range(k % 20):
                    obs2, *_ = env.step(np.array([COAST], dtype=np.int32))
                obs2, *_ = env.step(np.array([COAST], dtype=np.int32))
                sc = state_from_obs(obs2[0]); pc = phi_parts(sc)
                djs.append(abs(math.degrees(wrap(p1["dtheta"] - pc["dtheta"]))))
                dphis.append(abs(p1["phi_phase"] - pc["phi_phase"]))
                dws.append(abs(math.degrees(wrap(s1["w_s"] - s0["w_s"]))))
                dphys.append(abs(p1["dist"] - pc["dist"]))
            env.close()
            probe.append(dict(action=act, dv=ACTION_DV[act][0] or ACTION_DV[act][1],
                              e_init=e0,
                              dtheta_jump_med_deg=float(np.median(djs)),
                              dtheta_jump_max_deg=float(np.max(djs)),
                              phi_phase_jump_med=float(np.median(dphis)),
                              phi_phase_jump_max=float(np.max(dphis)),
                              omega_jump_med_deg=float(np.median(dws)),
                              phys_disp_med_km=float(np.median(dphys)) / 1e3))
    write_csv(os.path.join(OUT, "t3_burn_phase_discontinuity.csv"), probe)
    print(f"    {'act':>4}{'dv':>6}{'e0':>7}{'|d(dtheta)| med':>16}{'max':>8}"
          f"{'|dPhi_ph| med':>14}{'max':>8}{'|d omega| med':>14}"
          f"{'phys disp med km':>18}")
    for r in probe:
        print(f"    {r['action']:>4}{r['dv']:>6.0f}{r['e_init']:>7.2f}"
              f"{r['dtheta_jump_med_deg']:>16.2f}{r['dtheta_jump_max_deg']:>8.2f}"
              f"{r['phi_phase_jump_med']:>14.4f}{r['phi_phase_jump_max']:>8.4f}"
              f"{r['omega_jump_med_deg']:>14.2f}{r['phys_disp_med_km']:>18.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
