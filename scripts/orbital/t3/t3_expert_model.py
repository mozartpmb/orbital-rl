"""Exact Python replica of the orbital.h dynamics + obs decoding.

Mirrors pufferlib/pufferlib/ocean/orbital/orbital.h at HEAD (post-f55d9cb,
true_to_mean() corrected).  Used by expert_controller.py to predict the effect
of each discrete action without touching the C env.

Everything here is derived from the observation vector only (see decode_obs);
no ground-truth accessor is used.
"""
import math

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
ISP = 300.0
G0 = 9.80665
VE = ISP * G0                 # 2941.995 m/s
FUEL_FRAC = 0.15
DRY_MASS = 850.0
MAX_STEPS = 2000          # legacy default episode cap (env kwarg episode_cap_steps)
DEFAULT_PHASE_OBS_MODE = 0  # set to 1 when driving a phase_obs_mode=1 env
EARTH_KEEPOUT = R_EARTH + 200e3

TWO_PI = 2.0 * math.pi

# Action table — must match ACTION_DV / ACTION_TAU in orbital.h
ACTION_DV = [
    (0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (25.0, 0.0),
    (-5.0, 0.0), (-10.0, 0.0), (-25.0, 0.0),
    (0.0, 10.0), (0.0, -10.0),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (1.0, 0.0), (-1.0, 0.0), (2.0, 0.0), (-2.0, 0.0),
]
ACTION_TAU = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1]

COAST = 0
WARP5, WARP30, WARP60 = 9, 10, 11
# prograde burn actions ordered by magnitude
PRO = {1.0: 12, 2.0: 14, 5.0: 1, 10.0: 2, 25.0: 3}
RETRO = {1.0: 13, 2.0: 15, 5.0: 4, 10.0: 5, 25.0: 6}
BURN_QUANTA = (25.0, 10.0, 5.0, 2.0, 1.0)
RAD_OUT, RAD_IN = 7, 8


# ── Kepler ───────────────────────────────────────────────────────────────────
def solve_kepler(M, e):
    M = math.fmod(M, TWO_PI)
    if M < 0.0:
        M += TWO_PI
    E = M if e < 0.8 else math.pi
    for _ in range(5):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def eccentric_to_true(E, e):
    x = math.sqrt(1.0 - e) * math.cos(E / 2.0)
    y = math.sqrt(1.0 + e) * math.sin(E / 2.0)
    return 2.0 * math.atan2(y, x)


def true_to_mean(theta, e):
    """Corrected inverse (matches orbital.h post-fix)."""
    x = math.sqrt(1.0 + e) * math.cos(theta / 2.0)
    y = math.sqrt(1.0 - e) * math.sin(theta / 2.0)
    E = 2.0 * math.atan2(y, x)
    return E - e * math.sin(E)


def propagate(el, dt):
    """Return a new element dict advanced by dt seconds."""
    a, e = el['a'], el['e']
    n = math.sqrt(MU / (a * a * a))
    M = math.fmod(el['M'] + n * dt, TWO_PI)
    if M < 0.0:
        M += TWO_PI
    E = solve_kepler(M, e)
    return {'a': a, 'e': e, 'omega': el['omega'], 'M': M,
            'theta': eccentric_to_true(E, e)}


def orbit_to_cartesian(el):
    a, e, th, om = el['a'], el['e'], el['theta'], el['omega']
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(th))
    h = math.sqrt(MU * p)
    xp, yp = r * math.cos(th), r * math.sin(th)
    vxp = -(MU / h) * math.sin(th)
    vyp = (MU / h) * (e + math.cos(th))
    co, so = math.cos(om), math.sin(om)
    return (co * xp - so * yp, so * xp + co * yp,
            co * vxp - so * vyp, so * vxp + co * vyp)


def cartesian_to_elements(x, y, vx, vy):
    r = math.hypot(x, y)
    v2 = vx * vx + vy * vy
    vr = (x * vx + y * vy) / r
    a = 1.0 / (2.0 / r - v2 / MU)
    ex = ((v2 - MU / r) * x - vr * r * vx) / MU
    ey = ((v2 - MU / r) * y - vr * r * vy) / MU
    e = math.hypot(ex, ey)
    omega = 0.0 if e < 1e-10 else math.atan2(ey, ex)
    if e < 1e-10:
        theta = math.atan2(y, x)
    else:
        c = max(-1.0, min(1.0, (ex * x + ey * y) / (e * r)))
        theta = math.acos(c)
        if vr < 0.0:
            theta = TWO_PI - theta
    M = true_to_mean(theta, e)
    if M < 0.0:
        M += TWO_PI
    return {'a': a, 'e': e, 'omega': omega, 'theta': theta, 'M': M}


def apply_impulse(el, dv_pro, dv_rad):
    """Impulse in the local orbital frame; returns new elements (no fuel model)."""
    x, y, vx, vy = orbit_to_cartesian(el)
    vmag = math.hypot(vx, vy)
    rmag = math.hypot(x, y)
    dvx = dv_pro * vx / vmag + dv_rad * x / rmag
    dvy = dv_pro * vy / vmag + dv_rad * y / rmag
    return cartesian_to_elements(x, y, vx + dvx, vy + dvy)


# ── Observation decoding (obs-only; mirrors fill_observations) ────────────────
def decode_obs(obs, obs_alt_scale_m=1.6e6, phase_obs_mode=None):
    """Invert fill_observations. phase_obs_mode MUST match the env kwarg:
    mode 0 (legacy): obs[15,16] = sin/cos(theta_target)
    mode 1 (T3):     obs[13,14] = sin/cos(lambda_s - lambda_t),
                     obs[15]    = remaining_steps / episode_cap_steps (clock),
                     obs[16]    = cos(omega_s - omega_t)
    Under mode 1 the target anomaly is reconstructed via
    lambda_t = lambda_s - dlambda, M_t = lambda_t - omega_t (red-team #3)."""
    if phase_obs_mode is None:
        phase_obs_mode = DEFAULT_PHASE_OBS_MODE
    sat = {
        'a': float(obs[0]) * obs_alt_scale_m + R_EARTH,
        'e': float(obs[1]),
        'theta': math.atan2(float(obs[2]), float(obs[3])),
        'omega': math.atan2(float(obs[9]), float(obs[10])),
    }
    sat['M'] = true_to_mean(sat['theta'], sat['e'])
    if sat['M'] < 0.0:
        sat['M'] += TWO_PI
    tgt = {
        'a': float(obs[7]) * obs_alt_scale_m + R_EARTH,
        'e': float(obs[8]),
        'omega': math.atan2(float(obs[11]), float(obs[12])),
    }
    if phase_obs_mode == 1:
        dlam = math.atan2(float(obs[13]), float(obs[14]))   # lambda_s - lambda_t
        lam_s = sat['M'] + sat['omega']
        tgt['M'] = (lam_s - dlam) - tgt['omega']
        tgt['M'] %= TWO_PI
        E_t = solve_kepler(tgt['M'], tgt['e'])
        tgt['theta'] = eccentric_to_true(E_t, tgt['e'])
    else:
        tgt['theta'] = math.atan2(float(obs[15]), float(obs[16]))
        tgt['M'] = true_to_mean(tgt['theta'], tgt['e'])
        if tgt['M'] < 0.0:
            tgt['M'] += TWO_PI
    fuel_frac = float(obs[6])          # fuel_mass / (dry + fuel)
    return sat, tgt, fuel_frac


def dv_remaining(fuel_frac):
    """Δv still available given the observed fuel mass fraction."""
    if fuel_frac <= 0.0:
        return 0.0
    return VE * math.log(1.0 / (1.0 - fuel_frac))


def dv_spent(fuel_frac):
    return VE * math.log((1.0 - fuel_frac) / (1.0 - FUEL_FRAC))


def fuel_after(fuel_frac, dv):
    """Fuel mass fraction after burning dv m/s."""
    m = DRY_MASS / (1.0 - fuel_frac)          # total mass now
    fm = m * fuel_frac
    used = m * (1.0 - math.exp(-dv / VE))
    fm = max(0.0, fm - used)
    return fm / (DRY_MASS + fm)


def wrap_pi(x):
    return x - TWO_PI * math.floor((x + math.pi) / TWO_PI)


def rel_state(sat, tgt):
    sx, sy, svx, svy = orbit_to_cartesian(sat)
    tx, ty, tvx, tvy = orbit_to_cartesian(tgt)
    d = math.hypot(sx - tx, sy - ty)
    dv = math.hypot(svx - tvx, svy - tvy)
    return d, dv
