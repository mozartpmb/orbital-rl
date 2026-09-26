"""mini_expert.py — a teaching-sized version of the scripted GNC expert.

Same six rules as scripts/orbital/t3/expert_controller.py, stripped of
red-team plumbing so the whole thing fits on two screens. 2D, two-body,
impulsive burns. Runs standalone:

    python3 scripts/orbital/t3/mini_expert.py

Differences from the real controller, on purpose:
  * reads truth state directly (the real one decodes the 38-dim observation)
  * no fuel-mass model (real one inverts Tsiolkovsky on the fuel fraction)
  * fewer guard rails, simpler warp menu, no deadband-relative term
"""
import math

MU = 3.986004418e14          # Earth GM, m^3/s^2
R_E = 6.371e6                # Earth radius, m
DT = 60.0                    # one sim step, s
TWO_PI = 2.0 * math.pi

# ── 1. An orbit and how to move it ──────────────────────────────────────────
# a = size (semi-major axis, m). e = shape (0 circle, 0.05 slight oval).
# omega = which way the oval points. M = "clock angle" around the orbit,
# which advances at a perfectly steady rate n = sqrt(MU/a^3).

def make_orbit(a, e, omega, M):
    return {'a': a, 'e': e, 'omega': omega, 'M': M % TWO_PI}

def mean_motion(a):
    return math.sqrt(MU / a ** 3)

def propagate(o, dt):
    """Coast for dt seconds. Only the clock angle changes."""
    return make_orbit(o['a'], o['e'], o['omega'], o['M'] + mean_motion(o['a']) * dt)

def solve_kepler(M, e):
    """Clock angle M -> geometric angle E, by Newton's method."""
    E = M
    for _ in range(8):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    return E

def to_cartesian(o):
    """Orbit -> (x, y, vx, vy) in metres and m/s."""
    E = solve_kepler(o['M'], o['e'])
    theta = 2.0 * math.atan2(math.sqrt(1 + o['e']) * math.sin(E / 2),
                             math.sqrt(1 - o['e']) * math.cos(E / 2))
    p = o['a'] * (1 - o['e'] ** 2)
    r = p / (1 + o['e'] * math.cos(theta))
    h = math.sqrt(MU * p)
    xp, yp = r * math.cos(theta), r * math.sin(theta)
    vxp, vyp = -(MU / h) * math.sin(theta), (MU / h) * (o['e'] + math.cos(theta))
    c, s = math.cos(o['omega']), math.sin(o['omega'])
    return (c * xp - s * yp, s * xp + c * yp, c * vxp - s * vyp, s * vxp + c * vyp)

def from_cartesian(x, y, vx, vy):
    """(x, y, vx, vy) -> orbit. The inverse of to_cartesian."""
    r = math.hypot(x, y); v2 = vx * vx + vy * vy; rv = x * vx + y * vy
    a = 1.0 / (2.0 / r - v2 / MU)
    ex = ((v2 - MU / r) * x - rv * vx) / MU
    ey = ((v2 - MU / r) * y - rv * vy) / MU
    e = math.hypot(ex, ey)
    omega = math.atan2(ey, ex) if e > 1e-12 else 0.0
    theta = math.atan2(y, x) - omega
    E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(theta / 2),
                         math.sqrt(1 + e) * math.cos(theta / 2))
    return make_orbit(a, e, omega, E - e * math.sin(E))

def apply_burn(o, dv_prograde):
    """Instant push along (+) or against (-) the direction of travel."""
    x, y, vx, vy = to_cartesian(o)
    v = math.hypot(vx, vy)
    return from_cartesian(x, y, vx + dv_prograde * vx / v, vy + dv_prograde * vy / v)

def separation(sat, tgt):
    """Straight-line distance (m) and relative speed (m/s)."""
    sx, sy, svx, svy = to_cartesian(sat); tx, ty, tvx, tvy = to_cartesian(tgt)
    return math.hypot(sx - tx, sy - ty), math.hypot(svx - tvx, svy - tvy)

def wrap_pi(x):
    return (x + math.pi) % TWO_PI - math.pi

def longitude(o):
    """Where you are around the Earth, as one angle: pointing + clock."""
    return o['omega'] + o['M']

# ── 2. The action menu (a subset of the env's) ──────────────────────────────
BURNS = [1, 2, 5, 10, 25, -1, -2, -5, -10, -25]      # m/s prograde (+) / retro (-)
WARPS = [60, 30, 5]                                  # minutes of fast-forward

# ── 3. Tuning constants, same names as the real controller ─────────────────
PERIGEE_FLOOR = R_E + 200e3    # never dip below 200 km altitude
BURN_SLACK = 0.12              # a burn must recover >= 88% of its cost
TRACK_K = 0.65                 # plan the closure over 65% of time left
T_CTRL_MIN = 5400.0            # ...but never faster than 90 minutes
DA_FLOOR, DA_MAX = 3e3, 340e3  # commanded offset limits, m
A_DEADBAND = 6e3               # ignore goal changes smaller than this, m
SAFE_R, SAFE_V = 26e3, 42.0    # aim inside the 30 km / 50 m/s box
BOX_R, BOX_V = 30e3, 50.0      # the actual success box


class MiniExpert:
    def __init__(self, cap_steps=3000):
        self.cap_steps = cap_steps
        self.steps = 0
        self.mode = 'PLAN'
        self.dirn = -1
        self.a_hold = None
        self.hold_t = 0.0

    # ── 4. The error metric: "how much fuel to reach the goal orbit?" ──────
    @staticmethod
    def cost_to_go(o, a_goal, e_goal):
        """V = max(size error, shape error), both in m/s.
        A tangential burn dv changes a by 2dv/n and the e-vector by 2dv/v,
        so dividing by those turns both errors into m/s. An antipodal burn
        pair can null both at once for exactly max(|A|, |E|) of fuel."""
        n, v = mean_motion(a_goal), math.sqrt(MU / a_goal)
        A = n * (o['a'] - a_goal) / 2
        ex, ey = o['e'] * math.cos(o['omega']), o['e'] * math.sin(o['omega'])
        E = v * math.hypot(ex - e_goal[0], ey - e_goal[1]) / 2
        return max(abs(A), E)

    # ── 5. PLAN: pick the drift direction once ─────────────────────────────
    def plan(self, sat, tgt, t_rem):
        dl = wrap_pi(longitude(sat) - longitude(tgt))
        n_t = mean_motion(tgt['a'])
        T_ctrl = max(T_CTRL_MIN, TRACK_K * t_rem)
        best = None
        for dirn in (+1, -1):
            # angle still to sweep if we drive the gap up (+1) or down (-1)
            psi = (-dl) % TWO_PI if dirn > 0 else dl % TWO_PI
            da = -math.copysign(psi, dirn) * tgt['a'] / (1.5 * n_t * T_ctrl)
            da = math.copysign(min(abs(da), DA_MAX), da)
            a_park = max(tgt['a'] + da, PERIGEE_FLOOR / (1 - tgt['e']))
            eta = psi / (1.5 * n_t * abs(a_park - tgt['a']) / tgt['a'])
            cost = n_t * (abs(a_park - sat['a']) + abs(a_park - tgt['a'])) / 2
            score = cost + (400.0 if eta > 0.85 * t_rem else 0.0)
            if best is None or score < best[0]:
                best = (score, dirn)
        self.dirn = best[1]

    # ── 6. PHASING LAW: which orbit should I be parked on right now? ───────
    def track_goal(self, sat, tgt, t_rem):
        dl = wrap_pi(longitude(sat) - longitude(tgt))
        n_t = mean_motion(tgt['a'])
        if abs(dl) < math.radians(25):
            delta = -dl                                   # close: take the short way
        else:
            delta = (-dl) % TWO_PI if self.dirn > 0 else -(dl % TWO_PI)
        T_ctrl = max(T_CTRL_MIN, TRACK_K * t_rem)
        # Two orbits drift apart at 1.5 * n_t * da / a_t rad/s. Solve for the
        # da that closes `delta` in T_ctrl seconds.
        da = -delta * tgt['a'] / (1.5 * n_t * T_ctrl)
        if abs(da) < DA_FLOOR:                            # never park exactly on it
            da = math.copysign(DA_FLOOR, da if da else -1.0)
        da = math.copysign(min(abs(da), DA_MAX), da)
        a_goal = max(tgt['a'] + da, PERIGEE_FLOOR / (1 - tgt['e']))
        # Deadband: keep the old goal unless the new one moved meaningfully.
        if self.a_hold is None or abs(a_goal - self.a_hold) > A_DEADBAND:
            self.a_hold = a_goal
        return self.a_hold

    # ── 7. BURN LAW: is any single burn worth it right now? ────────────────
    def choose_burn(self, sat, a_goal, e_goal):
        V0 = self.cost_to_go(sat, a_goal, e_goal)
        if V0 < 0.4:
            return None
        best_act, best_gain = None, 0.0
        for dv in BURNS:
            after = apply_burn(sat, dv)                   # simulate it
            if after['a'] <= 0 or after['a'] * (1 - after['e']) < PERIGEE_FLOOR:
                continue                                  # escape or too low
            V1 = self.cost_to_go(after, a_goal, e_goal)
            gain = (V0 - V1) - abs(dv) * (1 - BURN_SLACK)  # saved minus 88% of cost
            if gain > best_gain:
                best_gain, best_act = gain, dv
        return best_act

    # ── 8. HOLD SCAN: will coasting carry me through the box? ──────────────
    def scan(self, sat, tgt, t_rem):
        s, t = sat, tgt
        for k in range(int(min(t_rem, 20 * 3600) / DT)):
            d, dv = separation(s, t)
            if d < SAFE_R and dv < SAFE_V:
                return True, k * DT
            s, t = propagate(s, DT), propagate(t, DT)
        return False, 0.0

    # ── 9. Put it together: one decision ───────────────────────────────────
    def act(self, sat, tgt):
        """Returns ('burn', dv) | ('warp', minutes) | ('coast', 1)."""
        t_rem = (self.cap_steps - self.steps) * DT
        if self.mode == 'PLAN':
            self.plan(sat, tgt, t_rem)
            self.mode = 'TRACK'

        if self.mode == 'HOLD':
            if self.hold_t <= 0:
                hit, t_hit = self.scan(sat, tgt, t_rem)
                if not hit:
                    self.mode = 'TRACK'
                self.hold_t = t_hit
            if self.mode == 'HOLD':
                return self.warp(self.hold_t)

        a_goal = self.track_goal(sat, tgt, t_rem)
        e_goal = (tgt['e'] * math.cos(tgt['omega']), tgt['e'] * math.sin(tgt['omega']))
        dv = self.choose_burn(sat, a_goal, e_goal)
        if dv is not None:
            return ('burn', dv)

        gap = abs(wrap_pi(longitude(sat) - longitude(tgt)))
        if gap < 0.09 and self.cost_to_go(sat, a_goal, e_goal) < 12.0:
            hit, t_hit = self.scan(sat, tgt, t_rem)
            if hit:
                self.mode, self.hold_t = 'HOLD', t_hit
                return self.warp(t_hit)
        return self.warp(300.0 if gap < 0.25 else 1800.0)

    def warp(self, budget_s):
        for minutes in WARPS:
            if minutes * 60 <= budget_s:
                self.hold_t -= minutes * 60
                return ('warp', minutes)
        self.hold_t -= DT
        return ('coast', 1)


# ── 10. Run one episode and narrate it ──────────────────────────────────────
def run(sat, tgt, cap_steps=3000, verbose=True):
    ctl = MiniExpert(cap_steps)
    dv_total, step, last_mode = 0.0, 0, None
    while step < cap_steps:
        mode = ctl.mode
        kind, val = ctl.act(sat, tgt)
        if kind == 'burn':
            sat = apply_burn(sat, val); dv_total += abs(val); tau = 1
        else:
            tau = val if kind == 'warp' else 1
        for _ in range(tau):                    # sub-step so the box check is exact
            sat, tgt = propagate(sat, DT), propagate(tgt, DT)
            step += 1; ctl.steps = step
            d, rv = separation(sat, tgt)
            if d < BOX_R and rv < BOX_V:
                if verbose:
                    print(f"  SUCCESS at step {step} ({step/60:.1f} h): d={d/1e3:.1f} km, "
                          f"vrel={rv:.1f} m/s, total dv={dv_total:.0f} m/s")
                return True, dv_total, step
        if verbose and (kind == 'burn' or mode != last_mode):
            gap = math.degrees(wrap_pi(longitude(sat) - longitude(tgt)))
            print(f"  step {step:5d} {mode:<5} alt={(sat['a']-R_E)/1e3:6.1f} km "
                  f"e={sat['e']:.4f} gap={gap:+7.2f} deg  -> {kind} {val}")
        last_mode = mode
    return False, dv_total, step


if __name__ == '__main__':
    # Roughly the seed-42 episode: chaser 556 km, e=0.045, 91 deg ahead;
    # target 429 km, e=0.005.
    tgt = make_orbit(R_E + 428.6e3, 0.005, 0.3, 1.0)
    sat = make_orbit(R_E + 556.2e3, 0.045, 1.1, 1.0 + 0.3 - 1.1 + math.radians(90.9))
    print("chaser ahead by", round(math.degrees(wrap_pi(longitude(sat) - longitude(tgt))), 1), "deg")
    run(sat, tgt)
