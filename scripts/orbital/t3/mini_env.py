"""mini_env.py — the orbital rendezvous environment, rebuilt in plain Python.

A faithful 2D re-implementation of pufferlib/ocean/orbital/orbital.h in the
T3 configuration (no debris, phase_obs_mode 1, shaping_mode 1, cap 3000).
Every function names the C function it mirrors. Runs standalone:

    python3 scripts/orbital/t3/mini_env.py          # random rollout + summary
    python3 scripts/orbital/t3/mini_env.py --check  # compare against the C env

What is left out on purpose: 3D (inc, raan), J2, debris bodies, the LVLH
observation block (obs[33:38]), the cell mixture, trajectory logging.
"""
import math, random, sys

# ── 1. Constants (orbital.h lines 20–61) ─────────────────────────────────────
MU       = 3.986004418e14   # Earth GM, m^3/s^2
R_EARTH  = 6.371e6          # m
ALT_MIN  = 200e3            # survivable altitude floor
DT       = 60.0             # one sub-step, s
ISP, G0  = 300.0, 9.80665
VE       = ISP * G0         # exhaust velocity ~2942 m/s
FUEL_FRAC = 0.15            # fuel is 15% of initial total mass
DRY_MASS  = 850.0           # kg
EARTH_KEEPOUT = R_EARTH + ALT_MIN
OBS_ALT_SCALE = 1.6e6       # obs altitude normaliser
TWO_PI = 2.0 * math.pi

# ── 2. The action table (orbital.h ACTION_DV / ACTION_TAU) ───────────────────
# (dv_prograde, dv_radial) in m/s, and tau = how many 60 s sub-steps it takes.
ACTION_DV  = [(0,0),(5,0),(10,0),(25,0),(-5,0),(-10,0),(-25,0),(0,10),(0,-10),
              (0,0),(0,0),(0,0),(1,0),(-1,0),(2,0),(-2,0)]
ACTION_TAU = [1,1,1,1,1,1,1,1,1, 5,30,60, 1,1,1,1]
NUM_ACTIONS = 16

# ── 3. Termination causes (orbital.h TERM_*) ─────────────────────────────────
TERM = ['none','success','collision','escape','safety_cap','stranded','hyperbolic']


# ═════════════════════════════════════════════════════════════════════════════
# 4. THE STATE. In C these are three structs; here, plain dicts.
#
#   Orbit      {a, e, M, theta, omega}     one orbit, 5 doubles
#   Satellite  {orbit, dry_mass, fuel_mass}
#   Env        {sat, target, step, cap, phi_prev, ...}
#
# a, e, omega describe the ellipse. M is the mean anomaly (steady clock angle).
# theta is the true anomaly (real angle), always derived from M, never stored
# independently — it is recomputed after every propagate and every burn.
# ═════════════════════════════════════════════════════════════════════════════

def orbit(a, e, M, omega):
    o = {'a': a, 'e': e, 'M': M % TWO_PI, 'omega': omega}
    o['theta'] = eccentric_to_true(solve_kepler(o['M'], e), e)
    return o


# ── 5. Kepler's equation (orbital.h solve_kepler, eccentric_to_true, true_to_mean)
def solve_kepler(M, e):
    """M = E - e sin E, solved for E. Newton, 5 iterations, exactly as the C."""
    M = M % TWO_PI
    E = M if e < 0.8 else math.pi
    for _ in range(5):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E

def eccentric_to_true(E, e):
    """E -> theta, half-angle form (no quadrant ambiguity)."""
    return 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                            math.sqrt(1 - e) * math.cos(E / 2))

def true_to_mean(theta, e):
    """theta -> E -> M. The sqrt factors are SWAPPED relative to the forward
    map; this is the function whose bug (2026-08-10) teleported phase on burns."""
    E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(theta / 2),
                         math.sqrt(1 + e) * math.cos(theta / 2))
    return E - e * math.sin(E)


# ── 6. Coasting (orbital.h propagate_orbit) ──────────────────────────────────
def propagate(o, dt):
    """Advance the clock angle by n*dt, then re-derive theta. Nothing else moves."""
    n = math.sqrt(MU / o['a'] ** 3)
    o['M'] = (o['M'] + n * dt) % TWO_PI
    o['theta'] = eccentric_to_true(solve_kepler(o['M'], o['e']), o['e'])


# ── 7. Elements <-> Cartesian (orbital.h orbit_to_cartesian / cartesian_to_elements)
def to_cartesian(o):
    """Perifocal frame (periapsis on +x), then rotate by omega."""
    p = o['a'] * (1 - o['e'] ** 2)              # semi-latus rectum
    r = p / (1 + o['e'] * math.cos(o['theta']))
    h = math.sqrt(MU * p)                       # angular momentum
    xp, yp   = r * math.cos(o['theta']), r * math.sin(o['theta'])
    vxp, vyp = -(MU / h) * math.sin(o['theta']), (MU / h) * (o['e'] + math.cos(o['theta']))
    co, so = math.cos(o['omega']), math.sin(o['omega'])
    return (co*xp - so*yp, so*xp + co*yp, co*vxp - so*vyp, so*vxp + co*vyp)

def from_cartesian(x, y, vx, vy):
    """Position+velocity -> elements. Mirrors the C equatorial branch exactly."""
    r  = math.hypot(x, y)
    v2 = vx*vx + vy*vy
    vr = (x*vx + y*vy) / r                       # radial speed
    a  = 1.0 / (2.0/r - v2/MU)                   # vis-viva
    ex = ((v2 - MU/r)*x - vr*r*vx) / MU          # eccentricity vector
    ey = ((v2 - MU/r)*y - vr*r*vy) / MU
    e  = math.hypot(ex, ey)
    if e < 1e-10:
        omega, theta = 0.0, math.atan2(y, x)
    else:
        omega = math.atan2(ey, ex)
        c = max(-1.0, min(1.0, (ex*x + ey*y) / (e*r)))
        theta = math.acos(c)
        if vr < 0.0:
            theta = TWO_PI - theta
    M = true_to_mean(theta, e)
    if M < 0.0:
        M += TWO_PI
    return {'a': a, 'e': e, 'M': M, 'theta': theta, 'omega': omega}


# ── 8. Burns and fuel (orbital.h apply_impulse) ──────────────────────────────
def apply_impulse(sat, dv_pro, dv_rad):
    """Returns |dv| actually applied. Mutates sat['orbit'] and sat['fuel_mass']."""
    x, y, vx, vy = to_cartesian(sat['orbit'])
    v, r = math.hypot(vx, vy), math.hypot(x, y)
    # local frame: prograde = v-hat, radial = r-hat (not orthogonal when e > 0)
    dvx = dv_pro * vx / v + dv_rad * x / r
    dvy = dv_pro * vy / v + dv_rad * y / r
    dv  = math.hypot(dvx, dvy)
    if dv < 1e-10:
        return 0.0
    # Tsiolkovsky: fuel burned = m_total * (1 - exp(-dv / VE))
    m_total = DRY_MASS + sat['fuel_mass']
    need = m_total * (1.0 - math.exp(-dv / VE))
    if need > sat['fuel_mass']:                  # tank runs dry mid-burn
        actual = -VE * math.log(1.0 - sat['fuel_mass'] / m_total)
        if actual < 1e-6:
            sat['fuel_mass'] = 0.0
            return 0.0
        dvx *= actual / dv; dvy *= actual / dv; dv = actual
        sat['fuel_mass'] = 0.0
    else:
        sat['fuel_mass'] -= need
    sat['orbit'] = from_cartesian(x, y, vx + dvx, vy + dvy)
    return dv


# ── 9. Reset: sample an episode (orbital.h c_reset, T3 defaults) ─────────────
def reset(env, rng):
    """T3 headline config: LEO 300-800 km, e <= 0.05 both, gap +-180 deg,
    perigee validity, physical mean-longitude gap (phase_gap_mode 1)."""
    e_max, gap_max, cap = env['e_max'], env['gap_max'], env['cap']
    for _attempt in range(4096):
        a_init   = R_EARTH + 300e3 + rng.random() * 500e3
        while True:
            a_target = R_EARTH + 300e3 + rng.random() * 500e3
            if abs(a_target - a_init) >= 50e3:      # meaningful transfer
                break
        e_t = rng.random() * e_max
        w_t = rng.random() * TWO_PI if e_t > 0 else 0.0
        e_s = rng.random() * e_max
        w_s = rng.random() * TWO_PI
        M_s = rng.random() * TWO_PI
        # reject inits whose perigee is below the keep-out altitude
        if a_init*(1-e_s) >= EARTH_KEEPOUT and a_target*(1-e_t) >= EARTH_KEEPOUT:
            break
    env['sat'] = {'orbit': orbit(a_init, e_s, M_s, w_s),
                  'dry_mass': DRY_MASS,
                  'fuel_mass': DRY_MASS * FUEL_FRAC / (1 - FUEL_FRAC)}   # 150 kg
    # Phase gap in MEAN LONGITUDE (lambda = M + omega), so it is physical:
    gap = (2*rng.random() - 1) * gap_max
    M_t = M_s + gap + (w_s - w_t)                 # lambda_t - lambda_s == gap
    env['target'] = orbit(a_target, e_t, M_t, w_t)
    env['step'] = 0
    env['phi_prev'] = compute_phi(env)
    env['cause'] = 0
    return observe(env)


# ── 10. Termination (orbital.h check_termination) ────────────────────────────
def check_termination(env):
    """Returns (terminal?, reward). Order matters: hyperbolic, collision,
    escape, cap, stranded, success."""
    sat, tgt = env['sat'], env['target']
    if sat['orbit']['a'] <= 0.0:
        env['cause'] = 6; return True, -10.0
    sx, sy, svx, svy = to_cartesian(sat['orbit'])
    r = math.hypot(sx, sy)
    if r < R_EARTH:                                # Earth is bodies[0]
        env['cause'] = 2; return True, -10.0
    if 0.5*(svx*svx + svy*svy) - MU/r >= 0.0:      # specific energy >= 0
        env['cause'] = 3; return True, -10.0
    if env['step'] >= env['cap']:
        env['cause'] = 4; return True, env['cap_reward']
    tx, ty, tvx, tvy = to_cartesian(tgt)
    d  = math.hypot(sx - tx, sy - ty)
    rv = math.hypot(svx - tvx, svy - tvy)
    at_target = d < env['box_r'] and rv < env['box_v']
    if sat['fuel_mass'] <= 0.0 and not at_target:
        env['cause'] = 5; return True, -10.0
    if at_target:
        initial_fuel = DRY_MASS * FUEL_FRAC / (1 - FUEL_FRAC)
        frac = max(0.0, min(1.0, sat['fuel_mass'] / initial_fuel))
        env['cause'] = 1; return True, 10.0 * (0.5 + 0.5 * frac)
    return False, 0.0


# ── 11. The shaping potential (orbital.h compute_phi, shaping_mode 1) ────────
def wrap_pi(x):
    return x - TWO_PI * math.floor((x + math.pi) / TWO_PI)

def compute_phi(env):
    """Phi = -[ W_l * |dlambda|/pi + W_m * min(1, dv_match / DV_REF) ] <= 0."""
    s, t = env['sat']['orbit'], env['target']
    dlam = wrap_pi((s['M'] + s['omega']) - (t['M'] + t['omega']))
    de = math.hypot(s['e']*math.cos(s['omega']) - t['e']*math.cos(t['omega']),
                    s['e']*math.sin(s['omega']) - t['e']*math.sin(t['omega']))
    da_rel = (s['a'] - t['a']) / t['a']
    v_t = math.sqrt(MU / t['a'])
    dv_match = 0.5 * v_t * math.hypot(da_rel, de)   # linearised 2-impulse cost
    match = min(1.0, dv_match / env['dv_ref'])
    return -(env['w_lambda'] * abs(dlam) / math.pi + env['w_match'] * match)


# ── 12. The observation (orbital.h fill_observations, slots 0-16) ────────────
def observe(env):
    s, t = env['sat']['orbit'], env['target']
    sx, sy, svx, svy = to_cartesian(s)
    r = math.hypot(sx, sy)
    vr = (sx*svx + sy*svy) / r
    vt = (sx*svy - sy*svx) / r
    v_circ = math.sqrt(MU / r)
    fuel_frac = env['sat']['fuel_mass'] / (DRY_MASS + env['sat']['fuel_mass'])
    dlam = (s['M'] + s['omega']) - (t['M'] + t['omega'])
    t_frac = max(0.0, (env['cap'] - env['step']) / env['cap'])
    obs = [0.0] * 17
    obs[0] = (s['a'] - R_EARTH) / OBS_ALT_SCALE        # chaser altitude, ~[0,1]
    obs[1] = s['e']
    obs[2], obs[3] = math.sin(s['theta']), math.cos(s['theta'])
    obs[4], obs[5] = vr / v_circ, vt / v_circ            # radial, tangential speed
    obs[6] = fuel_frac
    obs[7] = (t['a'] - R_EARTH) / OBS_ALT_SCALE        # target altitude
    obs[8] = t['e']
    obs[9], obs[10]  = math.sin(s['omega']), math.cos(s['omega'])
    obs[11], obs[12] = math.sin(t['omega']), math.cos(t['omega'])
    obs[13], obs[14] = math.sin(dlam), math.cos(dlam)   # phase gap
    obs[15] = t_frac                                    # episode clock
    obs[16] = math.cos(s['omega'] - t['omega'])         # apsidal alignment
    return obs


# ── 13. One step (orbital.h c_step) ──────────────────────────────────────────
def step(env, action):
    """Returns (obs, reward, terminal, dv_applied)."""
    reward, dv = 0.0, 0.0
    tau = ACTION_TAU[action]
    # 13a. burn (only tau == 1 non-coast actions burn; warps never do)
    if action != 0 and tau == 1 and env['sat']['fuel_mass'] > 0.0:
        dv = apply_impulse(env['sat'], *ACTION_DV[action])
    if env['sat']['orbit']['a'] <= 0.0:              # burned onto a hyperbola
        env['cause'] = 6
        return observe(env), -10.0, True, dv
    # 13b. propagate tau sub-steps, checking termination after each
    for _ in range(tau):
        propagate(env['target'], DT)
        propagate(env['sat']['orbit'], DT)
        env['step'] += 1
        term, r_term = check_termination(env)
        if term:
            return observe(env), r_term, True, dv
    # 13c. non-terminal: potential-based shaping, gamma_shape = 1 (telescoping)
    phi = compute_phi(env)
    reward += phi - env['phi_prev']
    env['phi_prev'] = phi
    return observe(env), reward, False, dv


def make_env(cap=3000, e_max=0.05, gap_max=math.pi, box_r=30e3, box_v=50.0):
    return dict(cap=cap, e_max=e_max, gap_max=gap_max, box_r=box_r, box_v=box_v,
                cap_reward=0.0, w_lambda=1.0, w_match=0.35, dv_ref=300.0)


# ═════════════════════════════════════════════════════════════════════════════
def random_rollout(seed=0):
    rng = random.Random(seed)
    env = make_env()
    obs = reset(env, rng)
    total, n = 0.0, 0
    print(f"init: chaser {(env['sat']['orbit']['a']-R_EARTH)/1e3:.0f} km e={env['sat']['orbit']['e']:.3f}, "
          f"target {(env['target']['a']-R_EARTH)/1e3:.0f} km e={env['target']['e']:.3f}, "
          f"gap {math.degrees(wrap_pi(math.atan2(obs[13], obs[14]))):+.1f} deg")
    while True:
        a = rng.randrange(NUM_ACTIONS)
        obs, r, term, dv = step(env, a)
        total += r; n += 1
        if term:
            print(f"terminal={TERM[env['cause']]} after {n} decisions / {env['step']} sub-steps, "
                  f"return={total:+.3f}, fuel left={env['sat']['fuel_mass']:.1f} kg")
            return


def check_against_c(episodes=3, decisions=400, seed=42):
    """Drive the real C env and this one with the SAME actions from the SAME
    initial state; report the max observation mismatch."""
    import numpy as np
    sys.path.insert(0, '/Users/pete/space_training/pufferlib')
    from pufferlib.ocean.orbital.orbital import Orbital
    cenv = Orbital(num_envs=1, num_debris_min=0, num_debris_max=0, e_max_target=0.05,
                   e_max_sat=0.05, init_phase_gap_max=math.pi, valid_init_only=1,
                   legacy_action_space=16, shaping_mode=1, shape_gamma=1.0,
                   phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000,
                   cap_terminal_reward=0.0)
    o, _ = cenv.reset(seed=seed); o = o[0]
    rng = random.Random(seed)
    worst = 0.0
    for ep in range(episodes):
        # copy the C env's initial state into ours
        st = cenv.get_state()[0]
        env = make_env()
        env['sat'] = {'orbit': from_cartesian(*st[8:10], *st[11:13]), 'dry_mass': DRY_MASS,
                      'fuel_mass': DRY_MASS * FUEL_FRAC / (1 - FUEL_FRAC)}
        env['target'] = from_cartesian(*st[23:25], *st[26:28])
        env['step'] = 0; env['phi_prev'] = compute_phi(env); env['cause'] = 0
        for k in range(decisions):
            a = rng.choice([0, 1, 2, 4, 5, 9, 10, 11, 12, 13])
            obs_c, r_c, term_c, _, _ = cenv.step(np.array([a], dtype=np.int32))
            obs_p, r_p, term_p, _ = step(env, a)
            if not (term_c[0] or term_p):      # C auto-resets on terminal: obs is the NEXT episode
                err = max(abs(float(obs_c[0][i]) - obs_p[i]) for i in range(17))
                err = max(err, abs(float(r_c[0]) - r_p))
                worst = max(worst, err)
            if term_c[0] or term_p:
                print(f"  ep{ep}: terminated at decision {k}: C={bool(term_c[0])} py={term_p} "
                      f"cause_py={TERM[env['cause']]} r_C={float(r_c[0]):+.2f} r_py={r_p:+.2f}")
                o = obs_c[0]
                break
        else:
            print(f"  ep{ep}: {decisions} decisions, no terminal; step={env['step']}")
            cenv.reset(seed=seed + ep + 1)
    print(f"max |obs_C - obs_py| (slots 0-16) and |r_C - r_py| over all non-terminal steps: {worst:.2e}")


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_against_c()
    else:
        for s in range(3):
            random_rollout(s)
