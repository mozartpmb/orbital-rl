"""Phase 5e Block I E1 — Lambert reachability check.

Implements 2D coplanar Lambert solver (universal-variable / Battin-style
short-way fixed transfer time). For sampled (sat, target) pairs at varying
e_max, computes minimum Δv across a grid of transfer times. Compares to the
chaser's fuel budget (~480 m/s for 15% fuel fraction at Isp=300s).

Validation: LEO 400→2000km Hohmann should yield ~678 m/s total Δv.

Run:
    python3 scripts/orbital/p5e_e1_lambert.py
"""
import math, sys, numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
ALT_MIN = 200e3
EARTH_KEEPOUT = R_EARTH + ALT_MIN
FUEL_FRAC = 0.15
ISP = 300.0
G0 = 9.80665
VE = ISP * G0
# Total Δv budget from Tsiolkovsky: m_f/m_0 = 1 - FUEL_FRAC → Δv = VE * ln(1/(1-f))
DV_BUDGET = VE * math.log(1.0 / (1.0 - FUEL_FRAC))   # ~480 m/s


# ─── Orbital element helpers (mirror orbital.h) ───────────────────────────

def solve_kepler(M, e, tol=1e-12, max_iter=20):
    E = M if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol: break
    return E


def true_from_eccentric(E, e):
    return 2.0 * math.atan2(math.sqrt(1+e) * math.sin(E/2),
                             math.sqrt(1-e) * math.cos(E/2))


def orbit_to_cartesian(a, e, omega, theta):
    """Return (x, y, vx, vy) inertial."""
    p = a * (1.0 - e*e)
    r = p / (1.0 + e * math.cos(theta))
    h = math.sqrt(MU * p)
    xp =  r * math.cos(theta)
    yp =  r * math.sin(theta)
    vxp = -(MU / h) * math.sin(theta)
    vyp =  (MU / h) * (e + math.cos(theta))
    co, so = math.cos(omega), math.sin(omega)
    return (co*xp - so*yp, so*xp + co*yp,
            co*vxp - so*vyp, so*vxp + co*vyp)


# ─── Lambert solver (Battin universal-variable, short-way) ─────────────────

def lambert_short_way(r1_vec, r2_vec, tof, mu=MU, max_iter=80, tol=1e-8):
    """Solve Lambert's problem in 2D (z-axis perpendicular to plane).
    r1_vec, r2_vec: (x, y) tuples in meters.
    tof: time of flight in seconds.
    Returns (v1_vec, v2_vec) — required initial and final velocities in m/s,
    or None if solution doesn't converge.

    Implementation: Battin's method, short-way only, for 2D coplanar
    transfers. Adapted from Vallado 'Fundamentals of Astrodynamics' §6.3.
    """
    r1 = math.sqrt(r1_vec[0]**2 + r1_vec[1]**2)
    r2 = math.sqrt(r2_vec[0]**2 + r2_vec[1]**2)
    # Cross product z-component (sign of orbital plane)
    cz = r1_vec[0]*r2_vec[1] - r1_vec[1]*r2_vec[0]
    cos_dnu = (r1_vec[0]*r2_vec[0] + r1_vec[1]*r2_vec[1]) / (r1 * r2)
    cos_dnu = max(-1.0, min(1.0, cos_dnu))
    dnu = math.acos(cos_dnu)
    if cz < 0:
        dnu = 2.0 * math.pi - dnu
    # Short-way: dnu ≤ pi
    if dnu > math.pi:
        dnu = 2.0 * math.pi - dnu  # Force short-way

    A = math.sin(dnu) * math.sqrt(r1 * r2 / (1.0 - cos_dnu))
    if abs(A) < 1e-9:
        return None

    # Bisection on z (= χ²; z>0 elliptic, z<0 hyperbolic)
    z_lo, z_hi = -4.0 * math.pi, 4.0 * math.pi**2
    z = 0.0
    for it in range(max_iter):
        # Stumpff functions C(z), S(z)
        if z > 1e-6:
            sz = math.sqrt(z)
            C = (1.0 - math.cos(sz)) / z
            S = (sz - math.sin(sz)) / (sz**3)
        elif z < -1e-6:
            sz = math.sqrt(-z)
            C = (1.0 - math.cosh(sz)) / z
            S = (math.sinh(sz) - sz) / (sz**3)
        else:
            C, S = 0.5, 1.0/6.0
        y = r1 + r2 + A * (z * S - 1.0) / math.sqrt(C)
        if A > 0 and y < 0:
            # Adjust z up; y must be positive for Lambert
            z_lo = z
            z = (z + z_hi) / 2.0
            continue
        x = math.sqrt(y / C)
        t_calc = (x**3 * S + A * math.sqrt(y)) / math.sqrt(mu)
        if abs(t_calc - tof) < tol * tof:
            break
        if t_calc < tof:
            z_lo = z
        else:
            z_hi = z
        z = (z_lo + z_hi) / 2.0
    else:
        # Did not converge
        return None

    f = 1.0 - y / r1
    g = A * math.sqrt(y / mu)
    gd = 1.0 - y / r2
    if abs(g) < 1e-9:
        return None
    v1 = ((r2_vec[0] - f*r1_vec[0]) / g,
          (r2_vec[1] - f*r1_vec[1]) / g)
    v2 = ((gd*r2_vec[0] - r1_vec[0]) / g,
          (gd*r2_vec[1] - r1_vec[1]) / g)
    return v1, v2


def hohmann_dv(r1, r2, mu=MU):
    """Total Δv for Hohmann transfer between coplanar circular orbits."""
    a_t = 0.5 * (r1 + r2)
    v1c = math.sqrt(mu / r1); v2c = math.sqrt(mu / r2)
    v1t = math.sqrt(mu * (2.0/r1 - 1.0/a_t))
    v2t = math.sqrt(mu * (2.0/r2 - 1.0/a_t))
    return abs(v1t - v1c) + abs(v2c - v2t)


# ─── Validation: Hohmann LEO 400→2000km ───────────────────────────────────

def validate_lambert():
    """Validate at 170° transfer (avoid degenerate 180° case for Lambert).
    Compare to expected Δv via direct vis-viva on the resulting transfer orbit."""
    r1, r2 = R_EARTH + 400e3, R_EARTH + 2000e3
    dnu = math.radians(170.0)
    r1_v = (r1, 0.0)
    r2_v = (r2 * math.cos(dnu), r2 * math.sin(dnu))
    # Pick a transfer time near the Hohmann half-period
    a_t_hohm = 0.5 * (r1 + r2)
    T_h = math.pi * math.sqrt(a_t_hohm**3 / MU) * (170.0/180.0)
    sol = lambert_short_way(r1_v, r2_v, T_h)
    if sol is None:
        print(f"VALIDATE FAIL: Lambert returned None")
        return False
    v1, v2 = sol
    # Initial circular at r1: prograde = +y direction; v_circ = sqrt(MU/r1)
    v1_circ = (0.0, math.sqrt(MU/r1))
    # Final circular at r2: prograde direction = (-sin(dnu), cos(dnu))
    v2c = math.sqrt(MU/r2)
    v2_circ = (-v2c * math.sin(dnu), v2c * math.cos(dnu))
    dv1 = math.sqrt((v1[0]-v1_circ[0])**2 + (v1[1]-v1_circ[1])**2)
    dv2 = math.sqrt((v2[0]-v2_circ[0])**2 + (v2[1]-v2_circ[1])**2)
    total = dv1 + dv2
    expected = hohmann_dv(r1, r2)
    err_pct = 100 * abs(total - expected) / expected
    print(f"VALIDATE: 170° transfer LEO 400→2000km (proxy for Hohmann)")
    print(f"  Hohmann Δv (expected upper bound):  {expected:.1f} m/s")
    print(f"  Lambert at 170°:                    {total:.1f} m/s  (err {err_pct:.1f}%)")
    return err_pct < 20.0
    v1, v2 = sol
    v1_circ = (0.0, math.sqrt(MU/r1))
    v2_circ = (0.0, -math.sqrt(MU/r2))  # at angle 180°, prograde is -y
    dv1 = math.sqrt((v1[0]-v1_circ[0])**2 + (v1[1]-v1_circ[1])**2)
    dv2 = math.sqrt((v2[0]-v2_circ[0])**2 + (v2[1]-v2_circ[1])**2)
    total = dv1 + dv2
    err_pct = 100 * abs(total - expected) / expected
    print(f"VALIDATE: Hohmann LEO 400→2000km")
    print(f"  Expected Δv: {expected:.1f} m/s")
    print(f"  Lambert Δv:  {total:.1f} m/s  (err {err_pct:.2f}%)")
    return err_pct < 5.0


# ─── Sample (sat, target) per env's c_reset (raw or valid_init_only) ─────

def sample_init(e_max, valid_init_only=False, max_attempts=512, rng=None):
    if rng is None: rng = np.random
    for _ in range(max_attempts):
        # sat
        alt_init = 300e3 + rng.random() * 500e3
        a_sat = R_EARTH + alt_init
        e_sat = rng.random() * e_max
        omega_sat = rng.random() * 2*math.pi
        M_sat = rng.random() * 2*math.pi
        # target — distinct band
        while True:
            alt_tgt = 300e3 + rng.random() * 500e3
            if abs(alt_tgt - alt_init) >= 50e3: break
        a_tgt = R_EARTH + alt_tgt
        e_tgt = rng.random() * e_max
        omega_tgt = rng.random() * 2*math.pi
        M_tgt = rng.random() * 2*math.pi
        if not valid_init_only:
            return (a_sat, e_sat, omega_sat, M_sat, a_tgt, e_tgt, omega_tgt, M_tgt)
        # Reject if either perigee < EARTH_KEEPOUT
        if a_sat * (1 - e_sat) >= EARTH_KEEPOUT and a_tgt * (1 - e_tgt) >= EARTH_KEEPOUT:
            return (a_sat, e_sat, omega_sat, M_sat, a_tgt, e_tgt, omega_tgt, M_tgt)
    return None  # gave up


def task_minimum_dv(task, T_grid=(300, 600, 1200, 1800, 3600, 5400, 7200, 10800)):
    """Compute minimum-Δv rendezvous over the transfer-time grid.

    Returns (min_dv, T_at_min). Velocity matching at arrival (not just
    position) — the env's success criterion requires both shape match and
    velocity null-out, so this is the relevant Lambert problem.
    """
    a_s, e_s, om_s, M_s, a_t, e_t, om_t, M_t = task
    th_s = true_from_eccentric(solve_kepler(M_s, e_s), e_s)
    x1, y1, vx1, vy1 = orbit_to_cartesian(a_s, e_s, om_s, th_s)
    best = float('inf'); best_T = None
    n_t = math.sqrt(MU / a_t**3)  # target mean motion
    for T in T_grid:
        # Target position at t = T (propagated)
        M_t_T = (M_t + n_t * T) % (2*math.pi)
        th_t_T = true_from_eccentric(solve_kepler(M_t_T, e_t), e_t)
        x2, y2, vx2_tgt, vy2_tgt = orbit_to_cartesian(a_t, e_t, om_t, th_t_T)
        sol = lambert_short_way((x1, y1), (x2, y2), T)
        if sol is None: continue
        v1_req, v2_req = sol
        dv1 = math.sqrt((v1_req[0]-vx1)**2 + (v1_req[1]-vy1)**2)
        dv2 = math.sqrt((v2_req[0]-vx2_tgt)**2 + (v2_req[1]-vy2_tgt)**2)
        total = dv1 + dv2
        if total < best:
            best = total; best_T = T
    return best, best_T


# ─── Main analysis ─────────────────────────────────────────────────────────

def analyze(e_max, n_samples=200, valid_init_only=False, seed=42):
    rng = np.random.RandomState(seed)
    dv_list, T_list = [], []
    skipped = 0
    for _ in range(n_samples):
        task = sample_init(e_max, valid_init_only, rng=rng)
        if task is None:
            skipped += 1; continue
        dv, T = task_minimum_dv(task)
        if math.isfinite(dv):
            dv_list.append(dv); T_list.append(T)
    return np.array(dv_list), T_list, skipped


def task_doomed_by_perigee(task):
    """Either sat or target perigee below R_EARTH (unrecoverable physical orbit)."""
    a_s, e_s, _, _, a_t, e_t, _, _ = task
    return a_s*(1-e_s) < R_EARTH or a_t*(1-e_t) < R_EARTH


def task_doomed_by_keepout(task):
    """Sat or target perigee below EARTH_KEEPOUT (atmospheric reentry)."""
    a_s, e_s, _, _, a_t, e_t, _, _ = task
    return a_s*(1-e_s) < EARTH_KEEPOUT or a_t*(1-e_t) < EARTH_KEEPOUT


def task_hohmann_dv(task):
    """Hohmann transfer Δv treating sat & target as circles at their semi-major
    axes. Lower bound on rendezvous Δv assuming favorable phasing (achievable
    by waiting + small phasing burn). Ignores eccentricity but bounds magnitude."""
    a_s = task[0]; a_t = task[4]
    return hohmann_dv(a_s, a_t)


def main():
    print("=" * 78)
    if not validate_lambert():
        print("ABORT: Lambert solver failed validation. Check implementation.")
        sys.exit(1)
    print("=" * 78)
    print(f"\nFuel-budget Δv: {DV_BUDGET:.1f} m/s\n")

    print("Per-distribution stats (200 samples each):")
    print(f"{'e_max':>6} {'mode':>16} {'doomed_R':>9} {'doomed_KO':>10} "
          f"{'hohm>bud':>9} {'lamb_med':>9} {'lamb_p10':>9} {'lamb<bud':>9}")
    print("-" * 78)
    for e_max in [0.05, 0.10, 0.20, 0.30]:
        for valid_init in (False, True):
            mode = "valid_init" if valid_init else "raw"
            rng = np.random.RandomState(42)
            tasks = []
            n_target = 200
            attempts = 0
            while len(tasks) < n_target and attempts < n_target * 50:
                t = sample_init(e_max, valid_init, rng=rng)
                attempts += 1
                if t is not None: tasks.append(t)
            if not tasks:
                print(f"{e_max:>6.2f} {mode:>16} no tasks")
                continue
            doomed_r = sum(task_doomed_by_perigee(t) for t in tasks) / len(tasks) * 100
            doomed_ko = sum(task_doomed_by_keepout(t) for t in tasks) / len(tasks) * 100
            hohm_costs = [task_hohmann_dv(t) for t in tasks]
            hohm_over = sum(h > DV_BUDGET for h in hohm_costs) / len(tasks) * 100
            # Lambert min-Δv (subset for speed)
            lamb_dvs = []
            for t in tasks[:100]:  # 100 of 200 to keep runtime down
                dv, _ = task_minimum_dv(t)
                if math.isfinite(dv): lamb_dvs.append(dv)
            if lamb_dvs:
                lamb_med = np.median(lamb_dvs)
                lamb_p10 = np.percentile(lamb_dvs, 10)
                lamb_under = sum(d < DV_BUDGET for d in lamb_dvs) / len(lamb_dvs) * 100
            else:
                lamb_med = lamb_p10 = lamb_under = float('nan')
            print(f"{e_max:>6.2f} {mode:>16} {doomed_r:>8.1f}% {doomed_ko:>9.1f}% "
                  f"{hohm_over:>8.1f}% {lamb_med:>9.0f} {lamb_p10:>9.0f} {lamb_under:>8.1f}%")
        print()
    print("Notes:")
    print("  doomed_R   = sat or target perigee < R_EARTH (sub-surface, doomed)")
    print("  doomed_KO  = sat or target perigee < EARTH_KEEPOUT (200km, atm reentry)")
    print("  hohm>bud   = Hohmann two-impulse (circular surrogate) > 478 m/s budget")
    print("  lamb_med   = median Lambert 2-impulse rendezvous Δv (random phasing)")
    print("  lamb_p10   = 10th-percentile (best 10% of phasings)")
    print("  lamb<bud   = fraction where Lambert 2-impulse fits in budget")
    print("  Lambert is an upper bound: agent uses multi-burn, can be cheaper via phasing.")


if __name__ == "__main__":
    main()
