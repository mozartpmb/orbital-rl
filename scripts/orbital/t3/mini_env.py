"""mini_env.py — the orbital rendezvous environment, rebuilt in plain Python.

Full-scope re-implementation of pufferlib/ocean/orbital/orbital.h:
3D orbits (i, RAAN), secular J2, debris bodies, all 31 actions (prograde,
radial, normal, combined, warps), the target-plane gauge, both LVLH frame
modes, the 3D observation block, shaping modes 1 and 2, Tsiolkovsky fuel.
Every function names the C function it mirrors.

    python3 scripts/orbital/t3/mini_env.py          # random rollouts (2D and 3D+J2)
    python3 scripts/orbital/t3/mini_env.py --check  # lockstep against the C env

Left out on purpose: the T11 cell mixture and fuel sampling (config plumbing),
trajectory logging, the action-validity mask (obs 38-47), the legacy
shaping_mode 0, and the gave-up terminal.
"""
import math, random, sys

# ── 1. Constants (orbital.h lines 20–75) ─────────────────────────────────────
MU        = 3.986004418e14   # Earth GM, m^3/s^2
R_EARTH   = 6.371e6          # m (collision radius)
ALT_MIN   = 200e3            # survivable altitude floor
DT        = 60.0             # one sub-step, s
ISP, G0   = 300.0, 9.80665
VE        = ISP * G0         # exhaust velocity ~2942 m/s
FUEL_FRAC = 0.15             # fuel is 15% of initial total mass
DRY_MASS  = 850.0            # kg
EARTH_KEEPOUT = R_EARTH + ALT_MIN
DEBRIS_HARD_R, DEBRIS_KEEPOUT = 1.0, 5000.0
J2_COEF   = 1.08262668e-3    # WGS-84 J2
J2_R_EQ   = 6.378137e6       # equatorial radius used by J2 (NOT R_EARTH)
OBS_ALT_SCALE = 1.6e6
OBS_DIM   = 38
TWO_PI    = 2.0 * math.pi

# ── 2. The action table (orbital.h ACTION_DV / ACTION_TAU), 31 rows ──────────
# (dv_prograde, dv_radial, dv_normal) in m/s; tau = 60 s sub-steps consumed.
ACTION_DV = [
    (0,0,0),(5,0,0),(10,0,0),(25,0,0),(-5,0,0),(-10,0,0),(-25,0,0),(0,10,0),(0,-10,0),
    (0,0,0),(0,0,0),(0,0,0),                         # 9-11: warp 5 / 30 / 60 min
    (1,0,0),(-1,0,0),(2,0,0),(-2,0,0),               # 12-15: fine prograde/retro
    (0,0,0),(0,0,0),(0,1,0),(0,-1,0),                # 16-17: warp 3 h / 6 h; 18-19: radial ±1
    (0,0,1),(0,0,-1),(0,0,10),(0,0,-10),(0,0,25),(0,0,-25),   # 20-25: normal (out of plane)
    (25,0,25),(25,0,-25),(-25,0,25),(-25,0,-25),     # 26-29: combined prograde+normal
    (0,0,0),                                         # 30: warp 24 h
]
ACTION_TAU = [1]*9 + [5,30,60] + [1]*4 + [180,360] + [1]*2 + [1]*10 + [1440]
NUM_ACTIONS = 31
TERM = ['none','success','collision','escape','safety_cap','stranded','hyperbolic']


# ═════════════════════════════════════════════════════════════════════════════
# 3. THE STATE (orbital.h Orbit / Satellite / Body / Orbital structs)
#
#   Orbit  {a, e, M, theta, omega, inc, raan}   seven doubles
#   Sat    {orbit, dry_mass, fuel_mass}
#   Body   {orbit, hard_radius, keepout_radius, is_static}   Earth = bodies[0]
#   Env    {sat, target, bodies, step, cap, phi_prev, config...}
#
# omega is measured FROM THE ASCENDING NODE; when raan == 0 it is the legacy 2D
# argument of periapsis. theta is always derived from M.
# ═════════════════════════════════════════════════════════════════════════════

def orbit(a, e, M, omega, inc=0.0, raan=0.0):
    o = {'a': a, 'e': e, 'M': M % TWO_PI, 'omega': omega, 'inc': inc, 'raan': raan}
    o['theta'] = eccentric_to_true(solve_kepler(o['M'], e), e)
    return o


# ── 4. Kepler's equation (solve_kepler, eccentric_to_true, true_to_mean) ─────
def solve_kepler(M, e):
    """M = E - e sin E. Newton from E0 = M (or pi for e >= 0.8), 5 iterations."""
    M = M % TWO_PI
    E = M if e < 0.8 else math.pi
    for _ in range(5):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E

def eccentric_to_true(E, e):
    return 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                            math.sqrt(1 - e) * math.cos(E / 2))

def true_to_mean(theta, e):
    """Inverse of the above (sqrt factors swapped). Site of the 2026-08-10 bug."""
    E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(theta / 2),
                         math.sqrt(1 + e) * math.cos(theta / 2))
    return E - e * math.sin(E)

def wrap_2pi(x):
    x = math.fmod(x, TWO_PI)
    return x + TWO_PI if x < 0.0 else x

def wrap_pi(x):
    return x - TWO_PI * math.floor((x + math.pi) / TWO_PI)


# ── 5. Coasting: two-body and secular J2 (propagate_orbit, propagate_orbit_j2)
def propagate(o, dt, j2_mode=0):
    """Advance the clock angle; under J2 also precess the node and periapsis."""
    n = math.sqrt(MU / o['a'] ** 3)
    if not j2_mode:
        o['M'] = wrap_2pi(o['M'] + n * dt)
    else:
        # Secular J2 rates: functions of (a, e, i) only, all three constant, so
        # the map is exact at ANY dt. R_EQ is the equatorial radius, not R_EARTH.
        p   = o['a'] * (1 - o['e'] ** 2)
        k   = 1.5 * n * J2_COEF * (J2_R_EQ / p) ** 2
        si2 = math.sin(o['inc']) ** 2
        Om  = -k * math.cos(o['inc'])                        # node regression
        om  =  0.5 * k * (4.0 - 5.0 * si2)                   # apsidal precession
        Md  =  n + 0.5 * k * math.sqrt(1 - o['e']**2) * (2.0 - 3.0 * si2)
        if o['inc'] == 0.0:
            # Equatorial: RAAN is a gauge angle with a MAXIMAL rate. Fold both
            # rates into omega and keep raan exactly 0.0 so the identity gauge
            # fast paths never disengage mid-episode.
            o['omega'] = wrap_2pi(o['omega'] + (om + Om) * dt)
        else:
            o['raan']  = wrap_2pi(o['raan']  + Om * dt)
            o['omega'] = wrap_2pi(o['omega'] + om * dt)
        o['M'] = wrap_2pi(o['M'] + Md * dt)
    o['theta'] = eccentric_to_true(solve_kepler(o['M'], o['e']), o['e'])


# ── 6. Element combinations consumers are allowed to read (orb_hhat, orb_evec)
def hhat(o):
    """Unit angular momentum, 3-1-3 convention: (sin i sin O, -sin i cos O, cos i)."""
    si, ci = math.sin(o['inc']), math.cos(o['inc'])
    return (si * math.sin(o['raan']), -si * math.cos(o['raan']), ci)

def evec(o):
    """Inertial eccentricity 3-vector from ELEMENTS (reduces to (e cos w, e sin w, 0) at i = 0)."""
    cO, sO = math.cos(o['raan']), math.sin(o['raan'])
    cw, sw = math.cos(o['omega']), math.sin(o['omega'])
    ci, si = math.cos(o['inc']), math.sin(o['inc'])
    e = o['e']
    return (e * (cO*cw - sO*sw*ci), e * (sO*cw + cO*sw*ci), e * (sw*si))


# ── 7. Elements <-> Cartesian (orbit_to_cartesian / cartesian_to_elements) ──
def to_cartesian(o):
    """Perifocal (periapsis on +x) -> inertial via the 3-1-3 rotation
    R3(-Omega) R1(-i) R3(-omega). Value-gated 2D fast path when i = Omega = 0."""
    p = o['a'] * (1 - o['e'] ** 2)
    r = p / (1 + o['e'] * math.cos(o['theta']))
    h = math.sqrt(MU * p)
    xp, yp   = r * math.cos(o['theta']), r * math.sin(o['theta'])
    vxp, vyp = -(MU / h) * math.sin(o['theta']), (MU / h) * (o['e'] + math.cos(o['theta']))
    if o['inc'] == 0.0 and o['raan'] == 0.0:
        co, so = math.cos(o['omega']), math.sin(o['omega'])
        return (co*xp - so*yp, so*xp + co*yp, 0.0, co*vxp - so*vyp, so*vxp + co*vyp, 0.0)
    cO, sO = math.cos(o['raan']), math.sin(o['raan'])
    cw, sw = math.cos(o['omega']), math.sin(o['omega'])
    ci, si = math.cos(o['inc']), math.sin(o['inc'])
    R11, R12 =  cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci
    R21, R22 =  sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci
    R31, R32 =  sw*si,             cw*si
    return (R11*xp + R12*yp, R21*xp + R22*yp, R31*xp + R32*yp,
            R11*vxp + R12*vyp, R21*vxp + R22*vyp, R31*vxp + R32*vyp)

def from_cartesian(x, y, z, vx, vy, vz):
    """Position+velocity -> elements. Exact hxy == 0 test picks the 2D branch."""
    hx, hy, hz = y*vz - z*vy, z*vx - x*vz, x*vy - y*vx
    hxy = math.sqrt(hx*hx + hy*hy)
    if hxy == 0.0:                                        # ── equatorial branch
        r  = math.sqrt(x*x + y*y); v2 = vx*vx + vy*vy
        vr = (x*vx + y*vy) / r
        a  = 1.0 / (2.0/r - v2/MU)
        ex = ((v2 - MU/r)*x - vr*r*vx) / MU
        ey = ((v2 - MU/r)*y - vr*r*vy) / MU
        e  = math.sqrt(ex*ex + ey*ey)
        if e < 1e-10:
            omega, theta = 0.0, math.atan2(y, x)
        else:
            omega = math.atan2(ey, ex)
            c = max(-1.0, min(1.0, (ex*x + ey*y) / (e*r)))
            theta = math.acos(c)
            if vr < 0.0: theta = TWO_PI - theta
        inc, raan = 0.0, 0.0
    else:                                                 # ── inclined branch
        r  = math.sqrt(x*x + y*y + z*z); v2 = vx*vx + vy*vy + vz*vz
        vr = (x*vx + y*vy + z*vz) / r
        a  = 1.0 / (2.0/r - v2/MU)
        hmag = math.sqrt(hx*hx + hy*hy + hz*hz)
        inc  = math.atan2(hxy, hz)                        # atan2, never acos(hz/h)
        raan = wrap_2pi(math.atan2(hx, -hy))              # node vector n = z x h
        ex = ((v2 - MU/r)*x - vr*r*vx) / MU
        ey = ((v2 - MU/r)*y - vr*r*vy) / MU
        ez = ((v2 - MU/r)*z - vr*r*vz) / MU
        e  = math.sqrt(ex*ex + ey*ey + ez*ez)
        nx, ny = -hy/hxy, hx/hxy                          # n-hat (node direction)
        wx, wy, wz = hx/hmag, hy/hmag, hz/hmag            # h-hat
        mx, my, mz = -wz*ny, wz*nx, wx*ny - wy*nx         # m-hat = h x n
        if e < 1e-10:
            omega = 0.0
            theta = wrap_2pi(math.atan2(x*mx + y*my + z*mz, x*nx + y*ny))
        else:
            omega = wrap_2pi(math.atan2(ex*mx + ey*my + ez*mz, ex*nx + ey*ny))
            eux, euy, euz = ex/e, ey/e, ez/e
            qx, qy, qz = wy*euz - wz*euy, wz*eux - wx*euz, wx*euy - wy*eux   # q = h x e
            theta = wrap_2pi(math.atan2(x*qx + y*qy + z*qz, x*eux + y*euy + z*euz))
    M = true_to_mean(theta, e)
    if M < 0.0: M += TWO_PI
    return {'a': a, 'e': e, 'M': M, 'theta': theta, 'omega': omega, 'inc': inc, 'raan': raan}


# ── 8. The target-plane gauge (PlaneGauge, gauge_from_orbit, orb_lambda_gauge)
def gauge_from_orbit(t):
    """Orthonormal frame of the target's orbit plane: e1 along its node line,
    e3 = h-hat, e2 = e3 x e1. Identity when the target is exactly equatorial."""
    if t['inc'] == 0.0 and t['raan'] == 0.0:
        return None
    wx, wy, wz = hhat(t)
    nx, ny = -wy, wx
    nn = math.sqrt(nx*nx + ny*ny)
    e1 = (nx/nn, ny/nn, 0.0) if nn > 1e-14 else (1.0, 0.0, 0.0)
    e3 = (wx, wy, wz)
    e2 = (e3[1]*e1[2] - e3[2]*e1[1], e3[2]*e1[0] - e3[0]*e1[2], e3[0]*e1[1] - e3[1]*e1[0])
    return (e1, e2, e3)

def varpi_gauge(o, g):
    """Longitude of periapsis (omega + Omega) measured in the target's plane frame."""
    if g is None:
        return o['omega'] + o['raan']
    x, y, z, vx, vy, vz = to_cartesian(o)
    e1, e2, e3 = g
    dot = lambda v, e: v[0]*e[0] + v[1]*e[1] + v[2]*e[2]
    t = from_cartesian(dot((x,y,z),e1), dot((x,y,z),e2), dot((x,y,z),e3),
                       dot((vx,vy,vz),e1), dot((vx,vy,vz),e2), dot((vx,vy,vz),e3))
    return t['omega'] + t['raan']

def lambda_gauge(o, g):
    """Mean longitude M + varpi in the target-plane gauge (M is frame-invariant)."""
    return o['M'] + varpi_gauge(o, g)


# ── 9. Burns and fuel (apply_impulse) ────────────────────────────────────────
def apply_impulse(sat, dv_pro, dv_rad, dv_nor):
    """Impulse in the local frame: prograde = v-hat, radial = r-hat, normal = h-hat.
    Returns |dv| applied; mutates sat['orbit'] and sat['fuel_mass']."""
    x, y, z, vx, vy, vz = to_cartesian(sat['orbit'])
    v = math.sqrt(vx*vx + vy*vy + vz*vz); r = math.sqrt(x*x + y*y + z*z)
    hx, hy, hz = y*vz - z*vy, z*vx - x*vz, x*vy - y*vx
    h = math.sqrt(hx*hx + hy*hy + hz*hz)
    # in-plane terms FIRST, normal LAST (so dv_nor == 0 is bitwise the 2D value)
    dvx = dv_pro * vx/v + dv_rad * x/r + dv_nor * hx/h
    dvy = dv_pro * vy/v + dv_rad * y/r + dv_nor * hy/h
    dvz = dv_pro * vz/v + dv_rad * z/r + dv_nor * hz/h
    dv  = math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)          # norm of the ASSEMBLED vector
    if dv < 1e-10:
        return 0.0
    m_total = sat['dry_mass'] + sat['fuel_mass']
    need = m_total * (1.0 - math.exp(-dv / VE))           # Tsiolkovsky
    if need > sat['fuel_mass']:
        actual = -VE * math.log(1.0 - sat['fuel_mass'] / m_total)
        if actual < 1e-6:
            sat['fuel_mass'] = 0.0
            return 0.0
        s = actual / dv; dvx *= s; dvy *= s; dvz *= s; dv = actual
        sat['fuel_mass'] = 0.0
    else:
        sat['fuel_mass'] -= need
    sat['orbit'] = from_cartesian(x, y, z, vx + dvx, vy + dvy, vz + dvz)
    return dv


# ── 10. Bodies (body_position) ───────────────────────────────────────────────
def body_position(b):
    if b['is_static']:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return to_cartesian(b['orbit'])


# ── 11. Reset (c_reset) ──────────────────────────────────────────────────────
def reset(env, rng):
    cfg = env['cfg']
    for _ in range(4096):                                  # perigee rejection sampling
        a_init = R_EARTH + 300e3 + rng.random() * 500e3
        while True:
            a_target = R_EARTH + 300e3 + rng.random() * 500e3
            if abs(a_target - a_init) >= 50e3: break
        e_t = rng.random() * cfg['e_max']
        w_t = rng.random() * TWO_PI if e_t > 0 else 0.0
        e_s = rng.random() * cfg['e_max']
        w_s = rng.random() * TWO_PI
        M_s = rng.random() * TWO_PI
        if a_init*(1-e_s) >= EARTH_KEEPOUT and a_target*(1-e_t) >= EARTH_KEEPOUT: break
    # ── planes. Target plane defaults to i = Omega = 0 (it is pure gauge under
    # two-body dynamics). Chaser plane = target plane rotated by delta about a
    # random axis IN the target plane (Rodrigues), delta = di_max * sqrt(U).
    i_t, O_t = 0.0, 0.0
    if cfg['dim3']:
        i_t = cfg['i_t_min'] + rng.random() * (cfg['i_t_max'] - cfg['i_t_min']) if cfg['i_t_max'] > cfg['i_t_min'] else cfg['i_t']
        O_t = cfg['raan_t']
    tgt = orbit(a_target, e_t, 0.0, w_t, i_t, O_t)
    inc_s, raan_s = i_t, O_t
    if cfg['dim3'] and cfg['di_max'] >= 0.0:
        delta = cfg['di_max'] * math.sqrt(rng.random())
        ph = rng.random() * TWO_PI
        wx, wy, wz = hhat(tgt)
        u1 = (-wy, wx, 0.0); n1 = math.hypot(u1[0], u1[1])
        u1 = (u1[0]/n1, u1[1]/n1, 0.0) if n1 > 1e-14 else (1.0, 0.0, 0.0)
        u2 = (wy*u1[2] - wz*u1[1], wz*u1[0] - wx*u1[2], wx*u1[1] - wy*u1[0])
        nx, ny, nz = (u1[0]*math.cos(ph) + u2[0]*math.sin(ph),
                      u1[1]*math.cos(ph) + u2[1]*math.sin(ph),
                      u1[2]*math.cos(ph) + u2[2]*math.sin(ph))
        cd, sd = math.cos(delta), math.sin(delta)
        crx, cry, crz = ny*wz - nz*wy, nz*wx - nx*wz, nx*wy - ny*wx
        hs = (wx*cd + crx*sd, wy*cd + cry*sd, wz*cd + crz*sd)
        hxy = math.hypot(hs[0], hs[1])
        if hxy > 1e-14:
            inc_s, raan_s = math.atan2(hxy, hs[2]), wrap_2pi(math.atan2(hs[0], -hs[1]))
        else:
            inc_s, raan_s = (0.0 if hs[2] >= 0 else math.pi), 0.0
    sat_orbit = orbit(a_init, e_s, M_s, w_s, inc_s, raan_s)
    env['sat'] = {'orbit': sat_orbit, 'dry_mass': DRY_MASS,
                  'fuel_mass': DRY_MASS * FUEL_FRAC / (1 - FUEL_FRAC)}
    # ── phase gap set in MEAN LONGITUDE, in the target-plane gauge
    gap = (2*rng.random() - 1) * cfg['gap_max']
    g = gauge_from_orbit(tgt)
    tgt['M'] = wrap_2pi(M_s + gap + varpi_gauge(sat_orbit, g) - varpi_gauge(tgt, g))
    tgt['theta'] = eccentric_to_true(solve_kepler(tgt['M'], tgt['e']), tgt['e'])
    env['target'] = tgt
    # ── bodies: Earth + optional debris on random low-e LEO orbits
    env['bodies'] = [{'orbit': orbit(0,0,0,0), 'hard_radius': R_EARTH,
                      'keepout_radius': EARTH_KEEPOUT, 'is_static': 1}]
    nd = cfg['debris_min'] + (rng.randrange(cfg['debris_max'] - cfg['debris_min'] + 1)
                              if cfg['debris_max'] > cfg['debris_min'] else 0)
    for _ in range(nd):
        d_a = R_EARTH + 300e3 + rng.random() * 500e3
        d_e = rng.random() * 0.05
        d_M = rng.random() * TWO_PI
        d_w = rng.random() * TWO_PI
        env['bodies'].append({'orbit': orbit(d_a, d_e, d_M, d_w), 'hard_radius': DEBRIS_HARD_R,
                              'keepout_radius': DEBRIS_KEEPOUT, 'is_static': 0})
    env['step'] = 0; env['cause'] = 0
    env['phi_prev'] = compute_phi(env)
    return observe(env)


# ── 12. Termination (check_termination) ──────────────────────────────────────
def check_termination(env):
    sat, tgt, cfg = env['sat'], env['target'], env['cfg']
    if sat['orbit']['a'] <= 0.0:
        env['cause'] = 6; return True, -10.0
    sx, sy, sz, svx, svy, svz = to_cartesian(sat['orbit'])
    r = math.sqrt(sx*sx + sy*sy + sz*sz)
    for b in env['bodies']:                                 # every distance includes z
        bx, by, bz, *_ = body_position(b)
        if math.sqrt((sx-bx)**2 + (sy-by)**2 + (sz-bz)**2) < b['hard_radius']:
            env['cause'] = 2; return True, -10.0
    if 0.5*(svx*svx + svy*svy + svz*svz) - MU/r >= 0.0:
        env['cause'] = 3; return True, -10.0
    if env['step'] >= cfg['cap']:
        env['cause'] = 4; return True, cfg['cap_reward']
    tx, ty, tz, tvx, tvy, tvz = to_cartesian(tgt)
    d  = math.sqrt((sx-tx)**2 + (sy-ty)**2 + (sz-tz)**2)
    rv = math.sqrt((svx-tvx)**2 + (svy-tvy)**2 + (svz-tvz)**2)
    at_target = d < cfg['box_r'] and rv < cfg['box_v']
    if sat['fuel_mass'] <= 0.0 and not at_target:
        env['cause'] = 5; return True, -10.0
    if at_target:
        frac = max(0.0, min(1.0, sat['fuel_mass'] / (DRY_MASS * FUEL_FRAC / (1 - FUEL_FRAC))))
        env['cause'] = 1; return True, 10.0 * (0.5 + 0.5 * frac)
    return False, 0.0


# ── 13. The shaping potential (compute_phi, modes 1 and 2) ───────────────────
def compute_phi(env):
    s, t, cfg = env['sat']['orbit'], env['target'], env['cfg']
    v_t = math.sqrt(MU / t['a'])
    da_rel = (s['a'] - t['a']) / t['a']
    if cfg['shaping_mode'] == 1:                            # 2D: mean-longitude gap + in-plane match
        dlam = wrap_pi((s['M'] + s['omega']) - (t['M'] + t['omega']))
        de = math.hypot(s['e']*math.cos(s['omega']) - t['e']*math.cos(t['omega']),
                        s['e']*math.sin(s['omega']) - t['e']*math.sin(t['omega']))
        dv_match = 0.5 * v_t * math.sqrt(da_rel*da_rel + de*de)
        match = min(1.0, dv_match / cfg['dv_ref'])
    else:                                                   # mode 2: the 3D lift
        g = gauge_from_orbit(t)
        dlam = wrap_pi(lambda_gauge(s, g) - lambda_gauge(t, g))
        es, et = evec(s), evec(t)
        de = math.sqrt(sum((es[i]-et[i])**2 for i in range(3)))
        dv_in = 0.5 * v_t * math.sqrt(da_rel*da_rel + de*de)         # in-plane, lever 2
        hs, ht = hhat(s), hhat(t)
        dv_pl = 1.0 * v_t * math.sqrt(sum((hs[i]-ht[i])**2 for i in range(3)))  # plane, lever 1
        match = (dv_in + dv_pl) / cfg['dv_ref']                        # L1, not hypot
        match = match / (1 + match) if cfg['squash'] else min(1.0, match)
    return -(cfg['w_lambda'] * abs(dlam) / math.pi + cfg['w_match'] * match)


# ── 14. The observation (fill_observations), all 38 slots ────────────────────
def clamp2(v):
    if not (v > -2.0): return 2.0 if v > 0.0 else -2.0    # also traps NaN
    return 2.0 if v > 2.0 else v

def observe(env):
    s, t, cfg = env['sat']['orbit'], env['target'], env['cfg']
    sat = env['sat']
    sx, sy, sz, svx, svy, svz = to_cartesian(s)
    r = math.hypot(sx, sy)                                  # in-plane radius, as the C
    vr = (sx*svx + sy*svy) / r
    vt = (sx*svy - sy*svx) / r
    v_circ = math.sqrt(MU / r)
    scale_dist = R_EARTH + OBS_ALT_SCALE
    obs = [0.0] * OBS_DIM
    # [0-6] chaser
    obs[0] = (s['a'] - R_EARTH) / OBS_ALT_SCALE; obs[1] = s['e']
    obs[2], obs[3] = math.sin(s['theta']), math.cos(s['theta'])
    obs[4], obs[5] = vr / v_circ, vt / v_circ
    obs[6] = sat['fuel_mass'] / (sat['dry_mass'] + sat['fuel_mass'])
    # [7-12] target size/shape, both orientations
    obs[7] = (t['a'] - R_EARTH) / OBS_ALT_SCALE; obs[8] = t['e']
    obs[9], obs[10]  = math.sin(s['omega']), math.cos(s['omega'])
    obs[11], obs[12] = math.sin(t['omega']), math.cos(t['omega'])
    # [13-16] phase gap (gauge-corrected in 3D), clock, apsidal alignment
    if cfg['dim3']:
        g = gauge_from_orbit(t); dlam = lambda_gauge(s, g) - lambda_gauge(t, g)
    else:
        dlam = (s['M'] + s['omega']) - (t['M'] + t['omega'])
    obs[13], obs[14] = math.sin(dlam), math.cos(dlam)
    obs[15] = max(0.0, (cfg['cap'] - env['step']) / cfg['cap'])
    obs[16] = math.cos(s['omega'] - t['omega'])
    # [17-32] four nearest bodies: distance, bearing, closing rate, keep-out radius
    dists = []
    for b in env['bodies']:
        bx, by, bz, bvx, bvy, bvz = body_position(b)
        dists.append((math.sqrt((sx-bx)**2 + (sy-by)**2 + (sz-bz)**2), b, (bx,by,bz,bvx,bvy,bvz)))
    dists.sort(key=lambda q: q[0])
    for k in range(4):
        base = 17 + 4*k
        if k >= len(dists): continue
        dr, b, (bx,by,bz,bvx,bvy,bvz) = dists[k]
        dth = wrap_pi(math.atan2(sy, sx) - math.atan2(by, bx))
        closing = ((sx-bx)*(svx-bvx) + (sy-by)*(svy-bvy) + (sz-bz)*(svz-bvz)) / dr
        obs[base], obs[base+1] = dr / scale_dist, dth / math.pi
        obs[base+2], obs[base+3] = closing / v_circ, b['keepout_radius'] / scale_dist
    # [21-30] 3D block overwrites body slots 1-3 (requires no debris)
    tx, ty, tz, tvx, tvy, tvz = to_cartesian(t)
    v_c_t = math.sqrt(MU / t['a'])
    if cfg['dim3']:
        hs, ht = hhat(s), hhat(t)
        cx, cy, cz = ht[1]*hs[2] - ht[2]*hs[1], ht[2]*hs[0] - ht[0]*hs[2], ht[0]*hs[1] - ht[1]*hs[0]
        cn = math.sqrt(cx*cx + cy*cy + cz*cz)
        di_rel = math.atan2(cn, ht[0]*hs[0] + ht[1]*hs[1] + ht[2]*hs[2])
        di = (di_rel*cx/cn, di_rel*cy/cn, di_rel*cz/cn) if cn > 1e-300 else (0.0, 0.0, 0.0)
        r3 = math.sqrt(sx*sx + sy*sy + sz*sz)
        R = (sx/r3, sy/r3, sz/r3)                                       # chaser radial
        T = (hs[1]*R[2] - hs[2]*R[1], hs[2]*R[0] - hs[0]*R[2], hs[0]*R[1] - hs[1]*R[0])  # along-track
        di_scale = max(cfg['di_max'] if cfg['di_max'] > 0 else 0.0, math.radians(0.25))
        de_scale = 0.05
        obs[21] = clamp2(sum(di[i]*R[i] for i in range(3)) / di_scale)  # rel. inclination vector,
        obs[22] = clamp2(sum(di[i]*T[i] for i in range(3)) / di_scale)  # in chaser RTN
        es, et = evec(s), evec(t); de3 = tuple(es[i]-et[i] for i in range(3))
        obs[23] = clamp2(sum(de3[i]*ht[i] for i in range(3)) / de_scale)
        rho_N  = sum((p-q)*w for p, q, w in zip((sx,sy,sz), (tx,ty,tz), ht))
        rhod_N = sum((p-q)*w for p, q, w in zip((svx,svy,svz), (tvx,tvy,tvz), ht))
        obs[24] = clamp2(rho_N / cfg['lvlh_scale']); obs[25] = clamp2(rhod_N / v_c_t)
        dv_pl = v_c_t * math.sqrt(sum((hs[i]-ht[i])**2 for i in range(3)))
        dv_in = 0.5 * v_c_t * math.sqrt(((s['a']-t['a'])/t['a'])**2 + sum(x*x for x in de3))
        dv_rem = VE * math.log((sat['dry_mass'] + sat['fuel_mass']) / sat['dry_mass'])
        obs[26] = clamp2(dv_pl / cfg['dv_ref']); obs[27] = clamp2(dv_rem / cfg['dv_ref'])
        obs[28] = clamp2((dv_rem - dv_pl - dv_in) / cfg['dv_ref'])    # feasibility margin
        if cfg['j2']:
            obs[29], obs[30] = clamp2(math.cos(s['inc'])), clamp2(math.cos(t['inc']))
    # [33-37] LVLH relative state in the target's rotating frame
    dxi, dyi, dzi = sx - tx, sy - ty, sz - tz
    dvxi, dvyi, dvzi = svx - tvx, svy - tvy, svz - tvz
    if cfg['lvlh_frame_mode'] == 1:                         # true orbital frame: r-hat, h x r-hat
        rn = math.sqrt(tx*tx + ty*ty + tz*tz); rt = (tx/rn, ty/rn, tz/rn)
        ht = hhat(t); tt = (ht[1]*rt[2] - ht[2]*rt[1], ht[2]*rt[0] - ht[0]*rt[2], ht[0]*rt[1] - ht[1]*rt[0])
        dx_l  = dxi*rt[0] + dyi*rt[1] + dzi*rt[2];  dy_l  = dxi*tt[0] + dyi*tt[1] + dzi*tt[2]
        dvx_l = dvxi*rt[0] + dvyi*rt[1] + dvzi*rt[2]; dvy_l = dvxi*tt[0] + dvyi*tt[1] + dvzi*tt[2]
    else:                                                   # legacy: equatorial projection by u = w + theta
        u = t['theta'] + t['omega']; cu, su = math.cos(u), math.sin(u)
        dx_l, dy_l   =  cu*dxi + su*dyi,  -su*dxi + cu*dyi
        dvx_l, dvy_l =  cu*dvxi + su*dvyi, -su*dvxi + cu*dvyi
    n_t = math.sqrt(MU / t['a']**3)
    dvx_l += n_t * dy_l; dvy_l -= n_t * dx_l                # rotating-frame correction
    obs[33], obs[34] = dx_l / cfg['lvlh_scale'], dy_l / cfg['lvlh_scale']
    obs[35], obs[36] = dvx_l / v_c_t, dvy_l / v_c_t
    obs[37] = n_t / 1e-3
    return obs


# ── 15. One step (c_step) ────────────────────────────────────────────────────
def step(env, action):
    cfg = env['cfg']; dv = 0.0
    tau = ACTION_TAU[action]
    if action != 0 and tau == 1 and env['sat']['fuel_mass'] > 0.0:
        dv = apply_impulse(env['sat'], *ACTION_DV[action])
    if env['sat']['orbit']['a'] <= 0.0:
        env['cause'] = 6; return observe(env), -10.0, True, dv
    for _ in range(tau):                                    # sub-step; check every 60 s
        for b in env['bodies'][1:]:
            propagate(b['orbit'], DT, cfg['j2'])
        propagate(env['target'], DT, cfg['j2'])
        propagate(env['sat']['orbit'], DT, cfg['j2'])
        env['step'] += 1
        term, r_term = check_termination(env)
        if term:
            return observe(env), r_term, True, dv
    phi = compute_phi(env)                                  # PBRS, gamma_shape = 1
    reward = phi - env['phi_prev']; env['phi_prev'] = phi
    return observe(env), reward, False, dv


def make_env(dim3=0, j2=0, debris=(0, 0), shaping_mode=1, dv_ref=300.0, w_match=0.35,
             di_max=-1.0, i_t=0.0, i_t_band=(-1.0, -1.0), raan_t=0.0, lvlh_frame_mode=0,
             cap=3000, e_max=0.05, gap_max=math.pi, box_r=30e3, box_v=50.0, squash=0):
    return {'cfg': dict(dim3=dim3, j2=j2, debris_min=debris[0], debris_max=debris[1],
                        shaping_mode=shaping_mode, dv_ref=dv_ref, w_lambda=1.0, w_match=w_match,
                        di_max=di_max, i_t=i_t, i_t_min=i_t_band[0], i_t_max=i_t_band[1],
                        raan_t=raan_t, lvlh_frame_mode=lvlh_frame_mode, lvlh_scale=R_EARTH,
                        cap=cap, cap_reward=0.0, e_max=e_max, gap_max=gap_max,
                        box_r=box_r, box_v=box_v, squash=squash)}


# ═════════════════════════════════════════════════════════════════════════════
CONFIGS = {
    '2D-T3':  dict(),
    '3D-J2':  dict(dim3=1, j2=1, di_max=math.radians(1.0), i_t_band=(math.radians(20), math.radians(60)),
                   shaping_mode=2, dv_ref=700.0, w_match=0.8166667, lvlh_frame_mode=1),
    'debris': dict(debris=(4, 8)),
}

def random_rollout(name, seed=0):
    rng = random.Random(seed); env = make_env(**CONFIGS[name]); obs = reset(env, rng)
    s, t = env['sat']['orbit'], env['target']
    print(f"[{name}] chaser {(s['a']-R_EARTH)/1e3:.0f} km e={s['e']:.3f} i={math.degrees(s['inc']):.2f} deg | "
          f"target {(t['a']-R_EARTH)/1e3:.0f} km e={t['e']:.3f} i={math.degrees(t['inc']):.2f} deg | "
          f"gap {math.degrees(wrap_pi(math.atan2(obs[13], obs[14]))):+.1f} deg | bodies {len(env['bodies'])}")
    total, n = 0.0, 0
    acts = list(range(30)) if env['cfg']['dim3'] else list(range(16))
    while True:
        obs, r, term, dv = step(env, rng.choice(acts)); total += r; n += 1
        if term:
            print(f"    -> {TERM[env['cause']]} after {n} decisions / {env['step']} sub-steps, return {total:+.3f}")
            return

def check_against_c(episodes=2, decisions=300, seed=42):
    """Feed the C env and this one identical actions from identical states."""
    import numpy as np
    sys.path.insert(0, '/Users/pete/space_training/pufferlib')
    from pufferlib.ocean.orbital.orbital import Orbital
    base = dict(num_envs=1, num_debris_min=0, num_debris_max=0, e_max_target=0.05, e_max_sat=0.05,
                init_phase_gap_max=math.pi, valid_init_only=1, shape_gamma=1.0,
                phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000, cap_terminal_reward=0.0)
    runs = {
        '2D-T3': (dict(base, legacy_action_space=16, shaping_mode=1), CONFIGS['2D-T3'],
                  [0,1,2,4,5,9,10,11,12,13], 17),
        '3D-J2': (dict(base, legacy_action_space=30, shaping_mode=2, dim3_mode=1, j2_mode=1,
                       di_max_rad=math.radians(1.0), i_target_min_rad=math.radians(20),
                       i_target_max_rad=math.radians(60), lvlh_frame_mode=1, shape_dv_ref_ms=700.0,
                       shape_w_match=0.8166667), CONFIGS['3D-J2'],
                  [0,1,2,4,5,9,10,11,12,13,20,21,22,23,26,27,28,29], 38),
    }
    for name, (ckw, pkw, acts, nslots) in runs.items():
        cenv = Orbital(**ckw); cenv.reset(seed=seed)
        rng = random.Random(seed); worst = 0.0
        for ep in range(episodes):
            st = cenv.get_state()[0]
            def orb_from_state(a, e, M, omega, hx, hy, hz):
                hxy = math.hypot(hx, hy)
                inc = math.atan2(hxy, hz) if hxy != 0.0 else 0.0
                raan = wrap_2pi(math.atan2(hx, -hy)) if hxy != 0.0 else 0.0
                return orbit(a, e, M, omega, inc, raan)
            env = make_env(**pkw)
            env['sat'] = {'orbit': orb_from_state(*st[0:3], st[4], *st[5:8]), 'dry_mass': DRY_MASS,
                          'fuel_mass': DRY_MASS * FUEL_FRAC / (1 - FUEL_FRAC)}
            env['target'] = orb_from_state(*st[15:18], st[19], *st[20:23])
            env['bodies'] = [{'orbit': orbit(0,0,0,0), 'hard_radius': R_EARTH,
                              'keepout_radius': EARTH_KEEPOUT, 'is_static': 1}]
            env['step'] = 0; env['cause'] = 0; env['phi_prev'] = compute_phi(env)
            for k in range(decisions):
                a = rng.choice(acts)
                oc, rc, tc, _, _ = cenv.step(np.array([a], dtype=np.int32))
                op, rp, tp, _ = step(env, a)
                if not (tc[0] or tp):
                    err = max(abs(float(oc[0][i]) - op[i]) for i in range(nslots))
                    worst = max(worst, err, abs(float(rc[0]) - rp))
                if tc[0] or tp:
                    print(f"  [{name}] ep{ep}: terminal at decision {k}: C={bool(tc[0])} py={tp} "
                          f"cause_py={TERM[env['cause']]} r_C={float(rc[0]):+.2f} r_py={rp:+.2f}")
                    break
            else:
                print(f"  [{name}] ep{ep}: {decisions} decisions, no terminal (step {env['step']})")
                cenv.reset(seed=seed + ep + 1)
        print(f"  [{name}] max |obs_C - obs_py| over {nslots} slots and |r_C - r_py|, non-terminal steps: {worst:.2e}")


if __name__ == '__main__':
    if '--check' in sys.argv:
        check_against_c()
    else:
        for name in CONFIGS:
            for s in range(2):
                random_rollout(name, s)
