"""T2 relative navigation — Python port of the C environment's orbital math.

Exact port of `pufferlib/pufferlib/ocean/orbital/orbital.h`:
  solve_kepler / eccentric_to_true / true_to_mean / propagate_orbit
  orbit_to_cartesian / cartesian_to_elements
  the LVLH block and the target-derived slots of fill_observations()

Everything here is double precision. The environment writes float32 observations,
so a truth observation quantizes the state at ~1e-7 relative; `recover_states()`
inverts that mapping and `build_obs()` re-emits the 38-dim vector. Round-tripping
obs -> states -> obs is the correctness proof that the port is exact (see
`validate()` at the bottom, run this file directly).

No env modification: this module never imports the C extension except in the
self-test, where it is used purely as an oracle.
"""

import math

# ── Constants (orbital.h lines 19-38) ────────────────────────────────────────
MU = 3.986004418e14       # Earth gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6         # Earth radius (m)
DT = 60.0                 # sim timestep (s)
OBS_DIM = 38

# Per-action sub-step count (ACTION_TAU in orbital.h). The filter's propagation
# interval for one env.step() is ACTION_TAU[a] * DT seconds.
ACTION_TAU = (1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1)

# Default observation normalizers (LEO / Phase 5b-5e compatible).
OBS_ALT_SCALE_M = 1.6e6
LVLH_SCALE_M = 6.371e6

# Observation slots that depend on the target's state. Everything else is
# chaser-only and stays truth in the relative-nav harness.
TARGET_OBS_IDX = (7, 8, 11, 12, 13, 14, 15, 16, 33, 34, 35, 36, 37)


# ── Kepler machinery ─────────────────────────────────────────────────────────
def solve_kepler(M, e):
    """M = E - e sin E, Newton-Raphson, 5 iterations (matches C exactly)."""
    M = math.fmod(M, 2.0 * math.pi)
    if M < 0.0:
        M += 2.0 * math.pi
    E = M if e < 0.8 else math.pi
    for _ in range(5):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def eccentric_to_true(E, e):
    x = math.sqrt(1.0 - e) * math.cos(0.5 * E)
    y = math.sqrt(1.0 + e) * math.sin(0.5 * E)
    return 2.0 * math.atan2(y, x)


def true_to_mean(theta, e):
    """Exact port of the C true_to_mean — INCLUDING its inverted half-angle ratio.

    The C version applies tan(E/2) = sqrt((1+e)/(1-e)) tan(theta/2), which is
    the E -> theta map, not its inverse (correct is sqrt((1-e)/(1+e))). It is
    reproduced verbatim because this module must mirror the environment, not
    correct it.

    Inert for this harness: true_to_mean is only reached from
    cartesian_to_elements, whose M output is never used here (observations need
    a, e, omega, theta only), and the *target* — the sole body the filter
    propagates — never has its elements recomputed from Cartesian by the env.
    Its M is sampled directly at reset and only ever advanced by n*dt, so the
    target's motion is exact two-body motion and matches propagate_cartesian.
    See _env_propagation_test() for the end-to-end confirmation.
    """
    x = math.sqrt(1.0 - e) * math.cos(0.5 * theta)
    y = math.sqrt(1.0 + e) * math.sin(0.5 * theta)
    E = 2.0 * math.atan2(y, x)
    return E - e * math.sin(E)


def mean_from_true(theta, e):
    """Correct theta -> M, for building self-test references."""
    x = math.sqrt(1.0 + e) * math.cos(0.5 * theta)
    y = math.sqrt(1.0 - e) * math.sin(0.5 * theta)
    E = 2.0 * math.atan2(y, x)
    return E - e * math.sin(E)


def propagate_elements(el, dt):
    """Advance (M, theta) by dt seconds. Returns a new element dict.

    Analytically exact for any dt — this is what makes warp actions and the
    filter's variable propagation interval free of integration error.
    """
    a, e = el['a'], el['e']
    n = math.sqrt(MU / (a * a * a))
    M = math.fmod(el['M'] + n * dt, 2.0 * math.pi)
    if M < 0.0:
        M += 2.0 * math.pi
    E = solve_kepler(M, e)
    return {'a': a, 'e': e, 'M': M, 'theta': eccentric_to_true(E, e),
            'omega': el['omega']}


def orbit_to_cartesian(el):
    """(a, e, theta, omega) -> (x, y, vx, vy) via perifocal -> inertial."""
    a, e, theta, omega = el['a'], el['e'], el['theta'], el['omega']
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(theta))
    h = math.sqrt(MU * p)

    xp = r * math.cos(theta)
    yp = r * math.sin(theta)
    vxp = -(MU / h) * math.sin(theta)
    vyp = (MU / h) * (e + math.cos(theta))

    co, so = math.cos(omega), math.sin(omega)
    return (co * xp - so * yp,
            so * xp + co * yp,
            co * vxp - so * vyp,
            so * vxp + co * vyp)


def cartesian_to_elements(x, y, vx, vy):
    """(x, y, vx, vy) -> (a, e, omega, theta, M). Exact port, incl. e<1e-10 branch."""
    r = math.sqrt(x * x + y * y)
    v2 = vx * vx + vy * vy
    vr = (x * vx + y * vy) / r

    a = 1.0 / (2.0 / r - v2 / MU)

    ex = ((v2 - MU / r) * x - vr * r * vx) / MU
    ey = ((v2 - MU / r) * y - vr * r * vy) / MU
    e = math.sqrt(ex * ex + ey * ey)

    omega = 0.0 if e < 1e-10 else math.atan2(ey, ex)

    if e < 1e-10:
        theta = math.atan2(y, x)
    else:
        c = max(-1.0, min(1.0, (ex * x + ey * y) / (e * r)))
        theta = math.acos(c)
        if vr < 0.0:
            theta = 2.0 * math.pi - theta

    # e can exceed 1 for a badly degraded *estimate* (never for a truth state);
    # clamp only inside the M computation so the returned e stays honest.
    M = true_to_mean(theta, min(e, 1.0 - 1e-12))
    if M < 0.0:
        M += 2.0 * math.pi
    return {'a': a, 'e': e, 'omega': omega, 'theta': theta, 'M': M}


def _rk4_two_body(state, dt, h_max=5.0):
    """Fallback propagator for non-elliptical states (a <= 0).

    Only reachable when the filter's *estimate* has been pushed hyperbolic by
    extreme measurement noise; the truth state never is (the env terminates on
    E >= 0). Fixed-step RK4 on the two-body ODE, which is conic-agnostic.
    """
    def deriv(s):
        x, y, vx, vy = s
        r3 = (x * x + y * y) ** 1.5
        return (vx, vy, -MU * x / r3, -MU * y / r3)

    n = max(1, int(math.ceil(abs(dt) / h_max)))
    h = dt / n
    s = tuple(state)
    for _ in range(n):
        k1 = deriv(s)
        k2 = deriv(tuple(si + 0.5 * h * ki for si, ki in zip(s, k1)))
        k3 = deriv(tuple(si + 0.5 * h * ki for si, ki in zip(s, k2)))
        k4 = deriv(tuple(si + h * ki for si, ki in zip(s, k3)))
        s = tuple(si + (h / 6.0) * (a_ + 2 * b_ + 2 * c_ + d_)
                  for si, a_, b_, c_, d_ in zip(s, k1, k2, k3, k4))
    return s


def propagate_cartesian(state, dt, tol=1e-13, max_iter=40):
    """Exact two-body propagation of a Cartesian state by dt, via Lagrange f&g.

    Solves Kepler's equation in the *eccentric-anomaly difference* dE and
    applies the f & g coefficients directly in Cartesian coordinates (BMW
    §4.4). This is analytically identical to the element-wise propagation the
    C env applies to the target, but unlike a round trip through classical
    elements it stays well conditioned as e -> 0: it never forms the
    eccentricity vector, so omega and theta — which individually become
    ill-determined for a circular orbit while their sum stays exact — never
    appear.

    That conditioning is load-bearing: the EKF's state-transition matrix is a
    numerical derivative of this map, and the element-route version loses
    ~1e-7 rad of angle precision on near-circular orbits, which is 70% error
    on a 1 m position perturbation and collapses the filter covariance.

    F'(dE) = sqrt(a) * r(dE) > 0, so Newton from dE0 = n*dt is monotone-safe.
    """
    x0, y0, vx0, vy0 = state
    r0 = math.sqrt(x0 * x0 + y0 * y0)
    v2 = vx0 * vx0 + vy0 * vy0
    a = 1.0 / (2.0 / r0 - v2 / MU)
    if a <= 0.0 or r0 < 1.0:
        return _rk4_two_body(state, dt)

    sqmu = math.sqrt(MU)
    sqa = math.sqrt(a)
    sigma0 = (x0 * vx0 + y0 * vy0) / sqmu       # r . v / sqrt(mu)
    n = sqmu / (a * sqa)

    target = sqmu * dt
    dE = n * dt
    for _ in range(max_iter):
        s, c = math.sin(dE), math.cos(dE)
        F = (a * sqa * (dE - s) + sigma0 * a * (1.0 - c) + r0 * sqa * s) - target
        dF = a * sqa * (1.0 - c) + sigma0 * a * s + r0 * sqa * c   # = sqrt(a)*r
        step = F / dF
        dE -= step
        if abs(step) < tol:
            break

    s, c = math.sin(dE), math.cos(dE)
    f = 1.0 - (a / r0) * (1.0 - c)
    g = dt - (a * sqa / sqmu) * (dE - s)
    x = f * x0 + g * vx0
    y = f * y0 + g * vy0
    r = math.sqrt(x * x + y * y)
    fdot = -(sqmu * sqa / (r * r0)) * s
    gdot = 1.0 - (a / r) * (1.0 - c)
    return (x, y, fdot * x0 + gdot * vx0, fdot * y0 + gdot * vy0)


# ── LVLH block (orbital.h lines 628-666) ─────────────────────────────────────
def lvlh_relative(sat_cart, tgt_cart, tgt_a, theta_t):
    """Chaser state relative to the target's rotating LVLH frame.

    theta_t is the target's inertial angle omega + theta (equivalently
    atan2(y_t, x_t)). Returns (dx_l, dy_l, dvx_l, dvy_l, n_t).
    """
    sx, sy, svx, svy = sat_cart
    tx, ty, tvx, tvy = tgt_cart
    ct, st = math.cos(theta_t), math.sin(theta_t)

    dxi, dyi = sx - tx, sy - ty
    dvxi, dvyi = svx - tvx, svy - tvy

    dx_l = ct * dxi + st * dyi
    dy_l = -st * dxi + ct * dyi
    dvx_l = ct * dvxi + st * dvyi
    dvy_l = -st * dvxi + ct * dvyi

    n_t = math.sqrt(MU / (tgt_a * tgt_a * tgt_a))
    dvx_l += n_t * dy_l
    dvy_l -= n_t * dx_l
    return dx_l, dy_l, dvx_l, dvy_l, n_t


# ── Observation reconstruction ───────────────────────────────────────────────
def recover_states(obs, obs_alt_scale_m=OBS_ALT_SCALE_M):
    """Invert fill_observations() for both absolute states.

    Chaser: a from [0], e from [1], theta from [2,3], omega from [9,10].
    Target: a from [7], e from [8], theta from [15,16], omega from [11,12].
    Both are fully determined; quantization is the float32 obs itself
    (~1 m in position, ~1e-3 m/s in velocity at LEO).
    """
    sat = {
        'a': float(obs[0]) * obs_alt_scale_m + R_EARTH,
        'e': float(obs[1]),
        'theta': math.atan2(float(obs[2]), float(obs[3])),
        'omega': math.atan2(float(obs[9]), float(obs[10])),
    }
    sat['M'] = true_to_mean(sat['theta'], sat['e'])
    tgt = {
        'a': float(obs[7]) * obs_alt_scale_m + R_EARTH,
        'e': float(obs[8]),
        'theta': math.atan2(float(obs[15]), float(obs[16])),
        'omega': math.atan2(float(obs[11]), float(obs[12])),
    }
    tgt['M'] = true_to_mean(tgt['theta'], tgt['e'])
    return sat, tgt


def fill_target_obs(obs_out, sat_el, tgt_el, obs_alt_scale_m=OBS_ALT_SCALE_M,
                    lvlh_scale_m=LVLH_SCALE_M, tgt_cart=None):
    """Overwrite the target-derived slots of obs_out in place.

    obs_out must already hold the chaser-only slots (a truth observation is the
    natural carrier). sat_el is the chaser's elements, tgt_el the target's —
    truth for validation, EKF estimate in the closed loop. Pass tgt_cart to use
    a Cartesian state directly (the filter mean) instead of re-deriving it from
    tgt_el; the two agree to 1e-9 relative for elliptical states, but only the
    direct path stays well defined if extreme sensor noise drives the estimate
    hyperbolic.
    """
    sat_cart = orbit_to_cartesian(sat_el)
    if tgt_cart is None:
        tgt_cart = orbit_to_cartesian(tgt_el)

    obs_out[7] = (tgt_el['a'] - R_EARTH) / obs_alt_scale_m
    obs_out[8] = tgt_el['e']
    obs_out[11] = math.sin(tgt_el['omega'])
    obs_out[12] = math.cos(tgt_el['omega'])

    dphase = sat_el['theta'] - tgt_el['theta']
    obs_out[13] = math.sin(dphase)
    obs_out[14] = math.cos(dphase)
    obs_out[15] = math.sin(tgt_el['theta'])
    obs_out[16] = math.cos(tgt_el['theta'])

    # theta + omega is the target's inertial angle; take it from the Cartesian
    # state so the LVLH frame stays exact even when the (omega, theta) split is
    # ill-conditioned (near-circular) or undefined (hyperbolic estimate).
    theta_t = math.atan2(tgt_cart[1], tgt_cart[0])
    # a <= 0 is only reachable from a degraded estimate; fall back to the
    # instantaneous radius so the normalizers stay finite and monotone.
    a_eff = tgt_el['a'] if tgt_el['a'] > 0.0 else math.hypot(tgt_cart[0],
                                                             tgt_cart[1])
    dx_l, dy_l, dvx_l, dvy_l, n_t = lvlh_relative(sat_cart, tgt_cart,
                                                  a_eff, theta_t)
    v_circ_t = math.sqrt(MU / a_eff)
    obs_out[33] = dx_l / lvlh_scale_m
    obs_out[34] = dy_l / lvlh_scale_m
    obs_out[35] = dvx_l / v_circ_t
    obs_out[36] = dvy_l / v_circ_t
    obs_out[37] = n_t / 1e-3
    return obs_out


def build_obs(obs_truth, sat_el, tgt_el, obs_alt_scale_m=OBS_ALT_SCALE_M,
              lvlh_scale_m=LVLH_SCALE_M, tgt_cart=None):
    """Copy of obs_truth with every target-derived slot recomputed from tgt_el."""
    import numpy as np
    out = np.array(obs_truth, dtype=np.float64, copy=True)
    fill_target_obs(out, sat_el, tgt_el, obs_alt_scale_m, lvlh_scale_m, tgt_cart)
    # Overflow guard only: a diverged estimate can push LVLH slots past float32
    # range, and inf/NaN would poison the LSTM rather than just degrade it. The
    # cap is 3 orders of magnitude outside the declared Box(-2, 2), so it never
    # touches a converged filter (verified: `recon` reproduces `truth` exactly).
    np.clip(np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4),
            -1e4, 1e4, out=out)
    return out.astype(np.float32)


# ── Self-test ────────────────────────────────────────────────────────────────
def _roundtrip_test(n=2000, seed=7):
    """elements -> cartesian -> elements, in double. Reports max relative error."""
    import random
    rng = random.Random(seed)
    worst = {'a': 0.0, 'e': 0.0, 'omega': 0.0, 'theta': 0.0}
    for _ in range(n):
        el = {
            'a': R_EARTH + rng.uniform(300e3, 800e3),
            'e': rng.uniform(1e-6, 0.6),
            'theta': rng.uniform(0.0, 2.0 * math.pi),
            'omega': rng.uniform(0.0, 2.0 * math.pi),
        }
        el['M'] = mean_from_true(el['theta'], el['e'])
        back = cartesian_to_elements(*orbit_to_cartesian(el))
        worst['a'] = max(worst['a'], abs(back['a'] - el['a']) / el['a'])
        worst['e'] = max(worst['e'], abs(back['e'] - el['e']) / el['e'])
        for k in ('omega', 'theta'):
            d = (back[k] - el[k] + math.pi) % (2.0 * math.pi) - math.pi
            worst[k] = max(worst[k], abs(d) / (2.0 * math.pi))
    return worst


def _propagation_test(n=400, seed=11):
    """f&g Cartesian propagation vs the env's element-wise Kepler propagation.

    Draws include e down to 1e-9 (the regime that broke the element route).
    Returns (max rel pos err vs elements, max rel pos err 1x3600s vs 60x60s).
    """
    import random
    rng = random.Random(seed)
    worst_vs_el, worst_split = 0.0, 0.0
    for i in range(n):
        e = 10.0 ** rng.uniform(-9, math.log10(0.3)) if i % 2 else 0.0
        el = {
            'a': R_EARTH + rng.uniform(300e3, 800e3),
            'e': e,
            'theta': rng.uniform(0.0, 2.0 * math.pi),
            'omega': rng.uniform(0.0, 2.0 * math.pi),
        }
        el['M'] = mean_from_true(el['theta'], el['e'])
        st = orbit_to_cartesian(el)
        for dt in (60.0, 1800.0, 3600.0):
            ref = orbit_to_cartesian(propagate_elements(el, dt))
            got = propagate_cartesian(st, dt)
            r = math.hypot(ref[0], ref[1])
            worst_vs_el = max(worst_vs_el,
                              math.hypot(got[0] - ref[0], got[1] - ref[1]) / r)
        one = propagate_cartesian(st, 3600.0)
        many = st
        for _ in range(60):
            many = propagate_cartesian(many, 60.0)
        r = math.hypot(one[0], one[1])
        worst_split = max(worst_split,
                          math.hypot(one[0] - many[0], one[1] - many[1]) / r)
    return worst_vs_el, worst_split


H_POS = 1.0        # STM central-difference step, position (m)
H_VEL = 1.0e-3     # STM central-difference step, velocity (m/s)


def stm_numerical(state, dt, h_pos=H_POS, h_vel=H_VEL):
    """d(propagate_cartesian(x, dt))/dx by central differences.

    Truncation error is O((h/L)^2) ~ 2e-14; double-precision cancellation is
    ~1e-9 relative. Both are far below anything Q or R sees, and the result is
    symplectic to ~1e-9 (see _stm_conditioning_test).
    """
    import numpy as np
    F = np.empty((4, 4))
    h = (h_pos, h_pos, h_vel, h_vel)
    for j in range(4):
        xp, xm = list(state), list(state)
        xp[j] += h[j]
        xm[j] -= h[j]
        F[:, j] = (np.asarray(propagate_cartesian(xp, dt)) -
                   np.asarray(propagate_cartesian(xm, dt))) / (2.0 * h[j])
    return F


def _stm_conditioning_test(n=200, seed=13):
    """STM quality on near-circular orbits, by two parameter-free measures.

    1. Symplecticity. Two-body flow is Hamiltonian in (q, p) = (pos, vel), so
       the exact STM satisfies F^T J F = J. max|F^T J F - J| is a pure measure
       of differencing error, needing no reference STM.
    2. Second-order consistency. F @ d must reproduce
       propagate(x + d) - propagate(x) to O(|d|^2) for a finite perturbation
       (100 m, 0.1 m/s here).

    Draws span e in [1e-12, 1e-3] — the regime where the element-route
    propagator loses ~1e-7 rad of angle precision and collapses the filter.
    """
    import random
    import numpy as np
    rng = random.Random(seed)
    J = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
                  [-1., 0., 0., 0.], [0., -1., 0., 0.]])
    worst_sym, worst_lin = 0.0, 0.0
    for _ in range(n):
        el = {'a': R_EARTH + rng.uniform(300e3, 800e3),
              'e': 10.0 ** rng.uniform(-12, -3),
              'theta': rng.uniform(0.0, 2.0 * math.pi),
              'omega': rng.uniform(0.0, 2.0 * math.pi)}
        el['M'] = mean_from_true(el['theta'], el['e'])
        st = orbit_to_cartesian(el)
        F = stm_numerical(st, 60.0)
        worst_sym = max(worst_sym, float(np.abs(F.T @ J @ F - J).max()))

        d = np.array([rng.gauss(0, 1) for _ in range(4)])
        d[:2] *= 100.0 / max(np.linalg.norm(d[:2]), 1e-12)
        d[2:] *= 0.1 / max(np.linalg.norm(d[2:]), 1e-12)
        exact = (np.asarray(propagate_cartesian(list(np.asarray(st) + d), 60.0)) -
                 np.asarray(propagate_cartesian(st, 60.0)))
        pred = F @ d
        worst_lin = max(worst_lin, float(np.abs(exact - pred).max() /
                                         max(np.abs(exact).max(), 1e-12)))
    return worst_sym, worst_lin


def _env_propagation_test(env, obs, n_steps, rng):
    """propagate_cartesian vs the C env's own target motion, step by step.

    The definitive check that (a) the propagator matches the environment's
    dynamics and (b) the harness's dt convention -- ACTION_TAU[a] * 60 s per
    env.step() -- is right. Returns (max abs position error in m, max abs
    velocity error in m/s, n compared).
    """
    import numpy as np
    worst_p, worst_v, n_cmp = 0.0, 0.0, 0
    for _ in range(n_steps):
        o = np.array(obs[0], dtype=np.float32, copy=True)
        _, tgt_el = recover_states(o)
        tgt_cart = orbit_to_cartesian(tgt_el)
        a = int(rng.integers(0, 12))
        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))
        if term[0]:
            continue                      # obs now belongs to the next episode
        o2 = np.array(obs[0], dtype=np.float32, copy=True)
        _, tgt_el2 = recover_states(o2)
        truth = orbit_to_cartesian(tgt_el2)
        pred = propagate_cartesian(tgt_cart, ACTION_TAU[a] * DT)
        worst_p = max(worst_p, math.hypot(pred[0] - truth[0], pred[1] - truth[1]))
        worst_v = max(worst_v, math.hypot(pred[2] - truth[2], pred[3] - truth[3]))
        n_cmp += 1
    return worst_p, worst_v, n_cmp, obs


def validate(n_steps=100, seed=42, verbose=True):
    """Reconstruct the env's own 38-dim obs from the states recovered out of it.

    Runs the real C env, takes each truth observation, recovers both absolute
    states from it, and re-emits all 38 dims. Any error in the port shows up
    immediately and hugely; float32 obs quantization alone bounds the residual
    at ~1e-7.
    """
    import numpy as np
    from pufferlib.ocean.orbital.orbital import Orbital

    rt = _roundtrip_test()
    prop_vs_el, prop_split = _propagation_test()
    stm_sym, stm_lin = _stm_conditioning_test()

    env = Orbital(num_envs=1, num_debris_min=0, num_debris_max=0,
                  e_max_target=0.05, e_max_sat=0.05,
                  init_phase_gap_max=3.14159, valid_init_only=1,
                  gave_up_action="terminate", legacy_action_space=10)
    obs, _ = env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    max_abs = np.zeros(OBS_DIM)
    for _ in range(n_steps):
        o = np.asarray(obs[0], dtype=np.float32)
        sat_el, tgt_el = recover_states(o)
        rebuilt = build_obs(o, sat_el, tgt_el)
        max_abs = np.maximum(max_abs, np.abs(rebuilt.astype(np.float64) -
                                             o.astype(np.float64)))
        a = int(rng.integers(0, 10))
        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))

    envp, envv, envn, obs = _env_propagation_test(env, obs, n_steps, rng)
    env.close()

    if verbose:
        print("── orbital_math validation ──────────────────────────────────")
        print(f"elements->cart->elements round trip (2000 draws, double):")
        print(f"  max rel err   a {rt['a']:.3e}   e {rt['e']:.3e}   "
              f"omega {rt['omega']:.3e}   theta {rt['theta']:.3e}")
        print(f"f&g Cartesian propagation vs env element propagation "
              f"(e in [0, 0.3], dt in {{60,1800,3600}} s):")
        print(f"  max rel pos err           {prop_vs_el:.3e}")
        print(f"  1x3600s vs 60x60s         {prop_split:.3e}")
        print(f"STM quality on near-circular orbits (e in [1e-12, 1e-3], dt=60 s):")
        print(f"  max |F^T J F - J|         {stm_sym:.3e}   (symplecticity)")
        print(f"  max rel linearization err {stm_lin:.3e}   (100 m / 0.1 m/s probe)")
        print(f"obs reconstruction over {n_steps} env steps "
              f"(38 dims, target dims marked *):")
        print(f"  max |err| all dims      {max_abs.max():.3e}")
        tgt_max = max(max_abs[i] for i in TARGET_OBS_IDX)
        print(f"  max |err| target dims*  {tgt_max:.3e}")
        worst = int(np.argmax(max_abs))
        print(f"  worst dim               [{worst}] {max_abs[worst]:.3e}")
        print(f"target propagation vs live env ({envn} steps, actions 0-11 incl. warps):")
        print(f"  max |pos err|             {envp:.3e} m")
        print(f"  max |vel err|             {envv:.3e} m/s")
    return rt, (prop_vs_el, prop_split, stm_sym, stm_lin), max_abs, (envp, envv, envn)


if __name__ == '__main__':
    validate()
