#!/usr/bin/env python3
"""ext-3d V5 — Φ(shaping_mode = 2) traced along a scripted 3D maneuver.

Method follows the T3 discipline: a Python replica of Φ is rebuilt from the
env's own trajectory log, cross-validated against the realised C reward on every
non-terminal step (shape_gamma = 1 ⇒ reward IS the Φ delta, exactly), and only
then used to decompose the potential into its λ / in-plane / plane parts. A
replica that does not match the C reward proves nothing, so the match is
reported first and everything else is conditional on it.

Legs, per 3d_C §3.1:
  L0  do-nothing        — coast + warp-5min only. The plane term must be
                          EXACTLY constant: ĥ is a constant of Keplerian motion
                          and v_t = √(μ/a_t) is a per-episode constant, so
                          holding a drift orbit while phasing must earn zero
                          false plane gradient.
  L1  node crank        — one 10 m/s normal burn at a controlled angle ψ from
                          the relative node (read off obs[21], obs[22]).
                          Plane credit per m/s must track |cos ψ|, so
                          "burn at the node" falls out of the geometry with no
                          node detector and no gate.
  L2  full maneuver     — coast-to-node (warp-5min) → crank → drift-open →
                          drift → drift-close. Per-leg ΔΦ and the worst single
                          adverse step.

Run (from the pufferlib dir):
    python3 ../scripts/orbital/ext3d/v5_shaping_trace3d.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'orbital', 'ext_recon'))

import orbital_math3d as o3                                       # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital, TRAJ_COLS    # noqa: E402
from pufferlib.ocean.orbital import binding                       # noqa: E402

MU = 3.986004418e14
R_EARTH = 6.371e6
D = math.pi / 180.0
COL = {c: i for i, c in enumerate(TRAJ_COLS)}

W_LAM, W_MATCH, DV_REF = 1.0, 0.8166667, 700.0
COAST, WARP5 = 0, 9
PRO25, RETRO25, PRO10, RETRO10 = 3, 6, 2, 5
NOR_P10, NOR_M10 = 22, 23

KW = dict(
    num_debris_min=0, num_debris_max=0, valid_init_only=1,
    gave_up_action="terminate",
    e_target_fixed=0.02, e_sat_fixed=0.02, omega_offset_fixed=0.0,
    a_min_override=R_EARTH + 500e3, a_max_override=R_EARTH + 800e3,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6, phi_orbit_scale_k=0.001,
    rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
    shaping_mode=2, shape_w_lambda=W_LAM, shape_w_match=W_MATCH,
    shape_dv_ref_ms=DV_REF, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000,
    cap_terminal_reward=0.0, legacy_action_space=30,
    dim3_mode=1, i_target_rad=0.0, raan_target_rad=0.0,
    init_phase_gap_max=math.pi, phase_gap_fixed=math.pi,
    di_max_rad=1.0 * D, traj_log_dir='/tmp/ext3d_v5', traj_log_every=10 ** 9,
)


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def hhat(inc, raan):
    si, ci = math.sin(inc), math.cos(inc)
    return np.array([si * math.sin(raan), -si * math.cos(raan), ci])


def evec(e, inc, raan, argp):
    cO, sO = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    return np.array([e * (cO * cw - sO * sw * ci),
                     e * (sO * cw + cO * sw * ci),
                     e * (sw * si)])


def phi_parts(row):
    """Replica of orbital.h compute_phi(shaping_mode = 2), from the log."""
    g = lambda c: float(row[COL[c]])                              # noqa: E731
    a_s, e_s = g('sat_a'), g('sat_e')
    i_s, O_s, w_s, nu_s = g('sat_inc'), g('sat_raan'), g('sat_omega'), g('sat_theta')
    a_t, e_t = g('target_a'), g('target_e')
    i_t, O_t, w_t = g('target_inc'), g('target_raan'), g('target_omega')
    # target true anomaly from its logged position (the target plane is the
    # gauge, i_t = Ω_t = 0, so the inertial angle is ω_t + ν_t exactly)
    nu_t = wrap_pi(math.atan2(g('target_y'), g('target_x')) - w_t)

    M_s = o3.nu_to_M(nu_s, e_s)
    M_t = o3.nu_to_M(nu_t, e_t)
    dlam = wrap_pi((M_s + w_s + O_s) - (M_t + w_t + O_t))

    de = float(np.linalg.norm(evec(e_s, i_s, O_s, w_s) - evec(e_t, i_t, O_t, w_t)))
    da_rel = (a_s - a_t) / a_t
    v_t = math.sqrt(MU / a_t)
    dv_in = 0.5 * v_t * math.sqrt(da_rel * da_rel + de * de)
    dv_pl = v_t * float(np.linalg.norm(hhat(i_s, O_s) - hhat(i_t, O_t)))
    match = min(1.0, (dv_in + dv_pl) / DV_REF)
    phi = -(W_LAM * abs(dlam) / math.pi + W_MATCH * match)
    return dict(phi=phi, lam=-W_LAM * abs(dlam) / math.pi,
                match=-W_MATCH * match, dv_in=dv_in, dv_pl=dv_pl,
                dlam=dlam, saturated=(dv_in + dv_pl) > DV_REF)


def run(seed, script, kw_extra=None):
    """Drive one scripted episode. Returns (traj rows, per-decision records).

    Each decision record is (action, sub_step_lo, sub_step_hi, C reward). Only
    decisions whose end sub-step actually exists in the exported log are kept:
    mid-episode, `vec_get_trajectory` reports `env->step` rows while the log
    holds `env->step + 1` (row 0 is the reset state), so the final sub-step is
    off the end of the export. Comparing Φ across a boundary that is one
    sub-step short is exactly the kind of silent off-by-one that would be read
    as a 6e-4 shaping discrepancy.
    """
    kw = dict(KW)
    if kw_extra:
        kw.update(kw_extra)
    env = Orbital(num_envs=1, seed=seed, **kw)
    env.reset(seed=seed)
    buf = np.zeros((12000, len(TRAJ_COLS)), dtype=np.float32)
    recs = []
    cum = 0
    done = False
    for a in script(env):
        _, r, t, _, _ = env.step(np.array([a], dtype=np.int32))
        tau = 5 if a == WARP5 else 1
        recs.append((a, cum, cum + tau, float(r[0])))
        cum += tau
        if t[0]:
            done = True
            break
    n = binding.vec_get_trajectory(env.c_envs, 0, buf)
    env.close()
    arr = buf[:n].copy()
    recs = [rc for rc in recs if rc[2] <= n - 1]
    return arr, recs, done


def node_psi(obs):
    """Angle from the relative node, straight off the 3D obs block."""
    return math.atan2(float(obs[22]), float(obs[21]))


# ────────────────────────────────────────────────────────────── L0 ──
def leg0():
    print("== L0  do-nothing (coast + warp-5min only) ==")

    def script(env):
        for i in range(300):
            yield (WARP5 if i % 4 else COAST)

    arr, recs, done = run(11, script)
    P = [phi_parts(arr[k]) for k in range(arr.shape[0])]
    dpl = [abs(P[k + 1]['dv_pl'] - P[k]['dv_pl']) for k in range(len(P) - 1)]
    dphi_match = [abs(P[k + 1]['match'] - P[k]['match']) for k in range(len(P) - 1)]
    per_dec = max(abs((P[hi]['phi'] - P[lo]['phi']) - r) for _, lo, hi, r in recs)
    tot_c = sum(r for _, _, _, r in recs)
    tot_r = P[recs[-1][2]]['phi'] - P[0]['phi']
    print(f"   sub-steps                             : {len(P) - 1}  (terminal={done})")
    print(f"   plane term Δv_pl, max |Δ| per sub-step : {max(dpl):.3e} m/s   "
          f"[required EXACTLY 0]")
    print(f"   match term Φ_m,  max |Δ| per sub-step  : {max(dphi_match):.3e}")
    print(f"   Σ C rewards = {tot_c:+.6f}   replica ΔΦ = {tot_r:+.6f}   "
          f"|diff| = {abs(tot_c - tot_r):.2e}")
    print(f"   replica vs C, worst PER-DECISION      : {per_dec:.2e}")
    ok = max(dpl) == 0.0 and per_dec < 1e-5
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}\n")
    return ok, per_dec


# ────────────────────────────────────────────────────────────── L1 ──
def _crank_once(psi_target, action):
    """Coast to angle psi_target from the relative node, fire one normal burn.
    Returns (psi realised at the burn point, |Δv|, plane credit per m/s)."""
    def script(env):
        obs = env.observations
        for _ in range(600):
            if abs(wrap_pi(abs(node_psi(obs[0])) - psi_target)) < 1.5 * D:
                break
            yield COAST
        yield action
        for _ in range(3):
            yield COAST

    arr, recs, done = run(21, script)
    P = [phi_parts(arr[k]) for k in range(arr.shape[0])]
    dv = arr[:, COL['delta_v']]
    kb = int(np.argmax(dv))
    if dv[kb] <= 0.0 or kb == 0:
        return None
    pre = arr[kb - 1]
    hs = hhat(float(pre[COL['sat_inc']]), float(pre[COL['sat_raan']]))
    ht = hhat(float(pre[COL['target_inc']]), float(pre[COL['target_raan']]))
    n = np.cross(ht, hs)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return None
    n /= nn
    rvec = np.array([float(pre[COL['sat_x']]), float(pre[COL['sat_y']]),
                     float(pre[COL['sat_z']])])
    rh = rvec / np.linalg.norm(rvec)
    th = np.cross(hs, rh)
    psi = math.atan2(float(np.dot(n, th)), float(np.dot(n, rh)))
    credit = (P[kb - 1]['dv_pl'] - P[kb]['dv_pl']) / float(dv[kb])
    return psi, float(dv[kb]), credit


def leg1():
    print("== L1  node crank: plane credit per m/s vs |cos ψ| ==")
    print("   (both normal signs are tried at each ψ; the CORRECT-sign burn is")
    print("    the one the geometry rewards — at ψ = 0 that is one sign, at")
    print("    ψ = 180° the other, which is what 'burn at a node' means.)")
    print(f"   {'ψ req':>7s} {'ψ real':>8s} {'Δv':>6s} {'credit+':>9s} "
          f"{'credit−':>9s} {'best':>8s} {'|cos ψ|':>9s} {'err':>8s}")
    rows = []
    for psi_deg in (0, 15, 30, 45, 60, 90, 120, 180):
        rp = _crank_once(psi_deg * D, NOR_P10)
        rm = _crank_once(psi_deg * D, NOR_M10)
        if rp is None or rm is None:
            continue
        psi = rp[0]
        best = max(rp[2], rm[2])
        ref = abs(math.cos(psi))
        rows.append((psi_deg, psi, rp[2], rm[2], best, ref))
        print(f"   {psi_deg:7d} {math.degrees(psi):8.1f} {rp[1]:6.1f} "
              f"{rp[2]:+9.4f} {rm[2]:+9.4f} {best:+8.4f} {ref:9.4f} "
              f"{best - ref:+8.4f}")
    near = [r for r in rows if r[0] <= 60 or r[0] >= 120]
    ok_pos = all(r[4] > 0.0 for r in near)
    ok_track = all(abs(r[4] - r[5]) <= 0.10 for r in near)
    # No cliff: the norm is 1-Lipschitz in Δv, so |credit| can never exceed 1,
    # and the off-node case is a bounded second-order penalty, not a gate.
    ok_cap = all(abs(r[2]) <= 1.02 and abs(r[3]) <= 1.02 for r in rows)
    q = [r for r in rows if r[0] == 90]
    ok_90 = (not q) or abs(q[0][4]) <= 0.15
    ok = ok_pos and ok_track and ok_cap and ok_90
    print(f"   at-node credit/Δv = {rows[0][4]:+.4f}  "
          f"(1.0 = one m/s of thrust buys one m/s of plane cost-to-go)")
    print(f"   quadrature (ψ=90°) credit = {q[0][4] if q else float('nan'):+.4f}  "
          f"— bounded second-order penalty, no cliff  [|·| ≤ 0.15]")
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}"
          f"  [sign {ok_pos}, tracking {ok_track}, Lipschitz cap {ok_cap}, "
          f"ψ90 {ok_90}]\n")
    return ok, rows


# ────────────────────────────────────────────────────────────── L2 ──
def leg2():
    print("== L2  full scripted maneuver ==")
    marks = []

    def script(env):
        obs = env.observations
        for _ in range(400):                       # L1a coast to relative node
            if abs(node_psi(obs[0])) < 6.0 * D:
                break
            yield WARP5
        # The node has two crossings and the crank sign is opposite at each.
        # δı⃗·R̂_s > 0 is the ascending relative node, where −normal is the
        # correcting direction; the descending node takes +normal. This is the
        # whole content of "burn at the node", read straight off obs[21] with
        # no explicit node detector in the env.
        nor = NOR_M10 if float(obs[0][21]) > 0.0 else NOR_P10
        for _ in range(6):                         # L1b plane crank
            yield nor
        for _ in range(4):                         # L2 drift-open
            yield RETRO25
        for _ in range(120):                       # L3 drift
            yield WARP5
        for _ in range(4):                         # L4 drift-close
            yield PRO25
        for _ in range(3):
            yield COAST

    arr, recs, done = run(31, script)
    P = [phi_parts(arr[k]) for k in range(arr.shape[0])]
    # rebuild leg boundaries from the action stream (first warp block = the
    # coast-to-node leg, a later one = the phasing drift)
    LEGS = {}
    bounds, cur, seen_burn = [], None, False
    for a, lo, hi, r in recs:
        if a == WARP5:
            lab = 'L3  drift (warp-5min)' if seen_burn else 'L1a coast-to-node (warp5)'
        elif a in (NOR_P10, NOR_M10):
            lab, seen_burn = 'L1b plane crank (normal 10)', True
        elif a == RETRO25:
            lab, seen_burn = 'L2  drift-open (retro 25)', True
        elif a == PRO25:
            lab, seen_burn = 'L4  drift-close (pro 25)', True
        else:
            lab = 'coast'
        if lab != cur:
            bounds.append([lab, lo, hi])
            cur = lab
        bounds[-1][2] = hi
    del LEGS, marks
    print(f"   {'leg':32s} {'sub-steps':>10s} {'ΔΦ':>10s} {'Δv (m/s)':>9s}")
    dvcol = arr[:, COL['delta_v']]
    total = 0.0
    for lab, lo, hi in bounds:
        d = P[hi]['phi'] - P[lo]['phi']
        total += d
        print(f"   {lab:32s} {hi - lo:10d} {d:+10.4f} "
              f"{float(dvcol[lo + 1:hi + 1].sum()):9.1f}")
    last = recs[-1][2]
    tele = P[last]['phi'] - P[0]['phi']
    tot_c = sum(r for _, _, _, r in recs)
    per_dec = max(abs((P[hi]['phi'] - P[lo]['phi']) - r) for _, lo, hi, r in recs)
    worst_dec = min(r for _, _, _, r in recs)
    drift = [b for b in bounds if b[0].startswith('L3')]
    plane = [b for b in bounds if b[0].startswith('L1b')]
    d_drift = sum(P[b[2]]['phi'] - P[b[1]]['phi'] for b in drift)
    # plane cost REMOVED = start − end (positive is progress)
    d_plane = sum(P[b[1]]['dv_pl'] - P[b[2]]['dv_pl'] for b in plane)
    d_plane_phi = sum(P[b[2]]['phi'] - P[b[1]]['phi'] for b in plane)
    dv_plane = sum(float(dvcol[b[1] + 1:b[2] + 1].sum()) for b in plane)
    print(f"   total ΔΦ (legs)             = {total:+.4f}")
    print(f"   telescoping Φ_T − Φ_0       = {tele:+.4f}   "
          f"Σ C rewards = {tot_c:+.4f}   |diff| = {abs(tele - tot_c):.2e}")
    print(f"   replica vs C, worst decision= {per_dec:.2e}")
    print(f"   drift leg ΔΦ                = {d_drift:+.4f}   [must be > 0]")
    print(f"   plane leg ΔΦ                = {d_plane_phi:+.4f}   [must be > 0]")
    print(f"   plane leg Δv_pl removed     = {d_plane:+.1f} m/s for {dv_plane:.0f} m/s "
          f"spent (credit/Δv = {d_plane / max(dv_plane, 1e-9):.3f})")
    print(f"   worst adverse DECISION      = {worst_dec:+.4f}   [gate ≤ 0.05]")
    print(f"   terminal reached            = {done}")
    ok = (abs(worst_dec) <= 0.05 and per_dec < 1e-5
          and d_drift > 0.0 and d_plane > 0.0 and d_plane_phi > 0.0)
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}\n")
    return ok, worst_dec, per_dec


def main():
    print("=== ext-3d V5: Φ(mode 2) shaping trace, 3D scripted maneuver ===")
    print(f"    W_λ={W_LAM} W_m={W_MATCH} dv_ref={DV_REF}  "
          f"Φ range = {W_LAM + W_MATCH:.4f}  "
          f"terminal dominance = {10.0 / (W_LAM + W_MATCH):.2f} : 1  [gate ≥ 5:1]\n")
    ok0, rep0 = leg0()
    ok1, rows = leg1()
    ok2, worst, rep2 = leg2()
    print(f"replica-vs-C agreement: {max(rep0, rep2):.2e} over the whole trace")
    ok = ok0 and ok1 and ok2
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
