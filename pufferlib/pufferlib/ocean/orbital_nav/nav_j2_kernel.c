/* nav_j2_kernel.c — C port of the ONE hot path the profile named.
 *
 * Profiled share of a nav rollout (cProfile self-time, batched numpy):
 *     W1_driftwait @ nav_max_ticks=0 ... filter 70.9%
 *     7-cell mixture @ K=120 ......... J2 propagation kernel 77.4%,
 *                                      MSC6 chart + EKF update 5.2%
 * so this file ports `stm_fd_j2` and its inner `propagate_cartesian_j2`, and
 * NOTHING else. The MSC6 chart, the EKF update, the CRLB surrogate, the batch
 * IOD and the acquisition gating all stay in Python: together they are ~5% of
 * runtime and they are where the subtle linear algebra lives.
 *
 * EXACTNESS CONTRACT. Every constant, every branch and every order of
 * operations mirrors nav_math3d.py / nav_math.py line for line, because the
 * differential fuzz harness holds the Python side as a permanent oracle:
 *
 *   - MU, J2_COEF, J2_R_EQ, H_POS, H_VEL taken from the Python constants.
 *   - solve_kepler: mod into [0,2pi), E0 = (e < 0.8 ? M : pi), EXACTLY five
 *     Newton steps — not "until converged". A sixth step is a different
 *     function at 1e-16 and the harness would (correctly) fail it.
 *   - the equatorial test is `hxy == 0.0` EXACTLY, never a tolerance. With
 *     z = vz = 0 the cross products are differences of exact zeros, so the
 *     branch fires bit-deterministically and must fire the same way here.
 *   - `orbit_to_cartesian_3d`'s value-gated 2D fast path on
 *     (inc == 0.0 && raan == 0.0) is reproduced, including that it ZEROES
 *     out-of-plane components rather than computing them.
 *   - np.mod is Python-modulo (non-negative for positive divisor); C fmod is
 *     not. `pmod` below is the difference, and getting it wrong would be a
 *     silent 2pi error in exactly the rows nearest theta = 0.
 *
 * NOT CLAIMED: bitwise equality with numpy. numpy dispatches its own SIMD
 * transcendentals, so sin/cos/atan2 can differ from libm by an ulp. The gate
 * is a derived relative tolerance, not bit-identity — see navc_fuzz.py.
 */
#include <math.h>
#include <string.h>

#define MU       3.986004418e14
#define J2_COEF  1.08262668e-3
#define J2_R_EQ  6.378137e6
#define H_POS    1.0
#define H_VEL    1.0e-3
#define TWO_PI   6.283185307179586476925286766559

/* np.mod semantics: result carries the divisor's sign, so always in [0,2pi). */
static inline double pmod(double x, double m) {
    double r = fmod(x, m);
    if (r < 0.0) r += m;
    return r;
}

static inline double clip(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* nav_math.solve_kepler — five Newton steps, no early exit. */
static inline double solve_kepler(double M, double e) {
    M = pmod(M, TWO_PI);
    double E = (e < 0.8) ? M : M_PI;
    for (int i = 0; i < 5; i++) {
        double dE = (M - E + e * sin(E)) / (1.0 - e * cos(E));
        E = E + dE;
    }
    return E;
}

/* nav_math.eccentric_to_true */
static inline double ecc_to_true(double E, double e) {
    double x = sqrt(fmax(1.0 - e, 0.0)) * cos(0.5 * E);
    double y = sqrt(1.0 + e) * sin(0.5 * E);
    return 2.0 * atan2(y, x);
}

/* nav_math.mean_from_true — the CORRECTED map (not the C env's true_to_mean) */
static inline double mean_from_true(double theta, double e) {
    double ec = clip(e, 0.0, 1.0 - 1e-12);
    double x = sqrt(1.0 + ec) * cos(0.5 * theta);
    double y = sqrt(1.0 - ec) * sin(0.5 * theta);
    double E = 2.0 * atan2(y, x);
    return E - ec * sin(E);
}

typedef struct {
    double a, e, omega, theta, M, inc, raan;
    int equatorial;
} El;

/* nav_math3d.cartesian_to_elements_3d, both branches, exact hxy == 0.0 test */
static void cart_to_el(const double* X, El* o) {
    double x = X[0], y = X[1], z = X[2], vx = X[3], vy = X[4], vz = X[5];
    double hx = y * vz - z * vy;
    double hy = z * vx - x * vz;
    double hz = x * vy - y * vx;
    double hxy = sqrt(hx * hx + hy * hy);
    int eq = (hxy == 0.0);
    o->equatorial = eq;

    if (eq) {
        double r2 = sqrt(x * x + y * y);
        double v2q = vx * vx + vy * vy;
        double vrq = (x * vx + y * vy) / r2;
        double a_q = 1.0 / (2.0 / r2 - v2q / MU);
        double exq = ((v2q - MU / r2) * x - vrq * r2 * vx) / MU;
        double eyq = ((v2q - MU / r2) * y - vrq * r2 * vy) / MU;
        double e_q = sqrt(exq * exq + eyq * eyq);
        int circ = (e_q < 1e-10);
        double om_q = circ ? 0.0 : atan2(eyq, exq);
        double den = circ ? 1.0 : (e_q * r2);
        double cth = clip((exq * x + eyq * y) / den, -1.0, 1.0);
        double thq = acos(cth);
        if (vrq < 0.0) thq = 2.0 * M_PI - thq;
        double th_q = circ ? atan2(y, x) : thq;
        o->a = a_q; o->e = e_q; o->omega = om_q; o->theta = th_q;
        o->inc = 0.0; o->raan = 0.0;
    } else {
        double r3 = sqrt(x * x + y * y + z * z);
        double v23 = vx * vx + vy * vy + vz * vz;
        double rv = x * vx + y * vy + z * vz;
        double vr3 = rv / r3;
        double a_i = 1.0 / (2.0 / r3 - v23 / MU);
        double hmag = sqrt(hx * hx + hy * hy + hz * hz);
        double inc_i = atan2(hxy, hz);
        double raan_i = pmod(atan2(hx, -hy), TWO_PI);

        double exi = ((v23 - MU / r3) * x - vr3 * r3 * vx) / MU;
        double eyi = ((v23 - MU / r3) * y - vr3 * r3 * vy) / MU;
        double ezi = ((v23 - MU / r3) * z - vr3 * r3 * vz) / MU;
        double e_i = sqrt(exi * exi + eyi * eyi + ezi * ezi);

        double hmag_s = (hmag > 0.0) ? hmag : 1.0;
        double nx = -hy / hxy, ny = hx / hxy;
        double wx = hx / hmag_s, wy = hy / hmag_s, wz = hz / hmag_s;
        double mx = -wz * ny, my = wz * nx, mz = wx * ny - wy * nx;

        int circ = (e_i < 1e-10);
        double th_circ = pmod(atan2(x * mx + y * my + z * mz, x * nx + y * ny),
                              TWO_PI);
        double w_i = pmod(atan2(exi * mx + eyi * my + ezi * mz,
                                exi * nx + eyi * ny), TWO_PI);
        double e_s = circ ? 1.0 : e_i;
        double eux = exi / e_s, euy = eyi / e_s, euz = ezi / e_s;
        double qx = wy * euz - wz * euy;
        double qy = wz * eux - wx * euz;
        double qz = wx * euy - wy * eux;
        double th_ecc = pmod(atan2(x * qx + y * qy + z * qz,
                                   x * eux + y * euy + z * euz), TWO_PI);
        o->a = a_i; o->e = e_i;
        o->omega = circ ? 0.0 : w_i;
        o->theta = circ ? th_circ : th_ecc;
        o->inc = inc_i; o->raan = raan_i;
    }
    o->M = pmod(mean_from_true(o->theta, o->e), TWO_PI);
}

/* nav_math3d.orbit_to_cartesian_3d, including the value-gated 2D fast path */
static void el_to_cart(double a, double e, double theta, double omega,
                       double inc, double raan, double* out) {
    double p = a * (1.0 - e * e);
    double r = p / (1.0 + e * cos(theta));
    double h = sqrt(MU * fmax(p, 1e-9));
    double xp = r * cos(theta);
    double yp = r * sin(theta);
    double vxp = -(MU / h) * sin(theta);
    double vyp = (MU / h) * (e + cos(theta));

    if (inc == 0.0 && raan == 0.0) {
        double co = cos(omega), so = sin(omega);
        out[0] = co * xp - so * yp;
        out[1] = so * xp + co * yp;
        out[2] = 0.0;
        out[3] = co * vxp - so * vyp;
        out[4] = so * vxp + co * vyp;
        out[5] = 0.0;
        return;
    }
    double cO = cos(raan), sO = sin(raan);
    double cw = cos(omega), sw = sin(omega);
    double ci = cos(inc), si = sin(inc);
    double R11 = cO * cw - sO * sw * ci;
    double R12 = -cO * sw - sO * cw * ci;
    double R21 = sO * cw + cO * sw * ci;
    double R22 = -sO * sw + cO * cw * ci;
    double R31 = sw * si;
    double R32 = cw * si;
    out[0] = R11 * xp + R12 * yp;
    out[1] = R21 * xp + R22 * yp;
    out[2] = R31 * xp + R32 * yp;
    out[3] = R11 * vxp + R12 * vyp;
    out[4] = R21 * vxp + R22 * vyp;
    out[5] = R31 * vxp + R32 * vyp;
}

/* nav_math3d.propagate_cartesian_j2 for ONE row.
 * Returns ok; writes Y (which is X unchanged when !ok, as numpy does). */
static int prop_j2_row(const double* X, double dt, double* Y) {
    El el;
    cart_to_el(X, &el);
    double a = el.a, e = el.e, inc = el.inc;

    /* bad-row discipline, identical to the numpy: a diverged estimate is
     * routinely hyperbolic and sqrt(MU/a^3) would poison the whole chain. */
    int bad = (!isfinite(a)) || (a <= 1.0) || (!isfinite(e)) || (e >= 1.0);
    double a_s = bad ? 7.0e6 : a;
    double e_s = bad ? 0.0 : e;
    double inc_s = bad ? 0.0 : inc;

    double n = sqrt(MU / (a_s * a_s * a_s));
    double p = a_s * (1.0 - e_s * e_s);
    double rp = J2_R_EQ / p;
    double k = 1.5 * n * J2_COEF * rp * rp;
    double si = sin(inc_s);
    double si2 = si * si;
    double Om = -k * cos(inc_s);
    double om = 0.5 * k * (4.0 - 5.0 * si2);
    double Md = n + 0.5 * k * sqrt(1.0 - e_s * e_s) * (2.0 - 3.0 * si2);

    int eqb = (si == 0.0);
    double omega = eqb ? (el.omega + (om + Om) * dt) : (el.omega + om * dt);
    double raan = eqb ? el.raan : (el.raan + Om * dt);
    double M = el.M + Md * dt;
    double th = ecc_to_true(solve_kepler(M, e_s), e_s);

    double tmp[6];
    el_to_cart(a_s, e_s, th, omega, inc_s, raan, tmp);

    int fin = 1;
    for (int j = 0; j < 6; j++) if (!isfinite(tmp[j])) { fin = 0; break; }
    int ok = fin && !bad;
    for (int j = 0; j < 6; j++) Y[j] = ok ? tmp[j] : X[j];
    return ok;
}

/* Batched propagate — exposed for the fuzz harness to test the inner kernel
 * directly rather than only through the STM. */
void propagate_cartesian_j2_batch(const double* X, int nrow, double dt,
                                  double* Y, unsigned char* ok) {
    for (int i = 0; i < nrow; i++) {
        ok[i] = (unsigned char)prop_j2_row(X + 6 * i, dt, Y + 6 * i);
    }
}

/* nav_math3d.stm_fd_j2 — central difference, 12 perturbed propagations.
 * Phi is (nrow, 6, 6) row-major: Phi[i*36 + k*6 + j] = dY_k / dX_j. */
void stm_fd_j2_batch(const double* X, int nrow, double dt,
                     double* Phi, unsigned char* ok, double* Y) {
    const double h[6] = {H_POS, H_POS, H_POS, H_VEL, H_VEL, H_VEL};
    for (int i = 0; i < nrow; i++) {
        const double* x = X + 6 * i;
        double* y = Y + 6 * i;
        double* P = Phi + 36 * i;
        int good = prop_j2_row(x, dt, y);

        double xp[6], xm[6], yp[6], ym[6];
        for (int j = 0; j < 6; j++) {
            memcpy(xp, x, 6 * sizeof(double));
            memcpy(xm, x, 6 * sizeof(double));
            xp[j] += h[j];
            xm[j] -= h[j];
            int okp = prop_j2_row(xp, dt, yp);
            int okm = prop_j2_row(xm, dt, ym);
            good = good && okp && okm;
            /* DIVIDE, do not multiply by a precomputed reciprocal: numpy
             * writes (Yp - Ym) / (2.0 * h[j]), and for h = 1e-3 the quantity
             * 2h is not a binary fraction, so 1/(2h) rounds and the two differ
             * by an ulp on every velocity column. Exactly the "boring reason"
             * a fuzz harness fails at 1e-12. */
            double den = 2.0 * h[j];
            for (int k = 0; k < 6; k++) {
                P[k * 6 + j] = (yp[k] - ym[k]) / den;
            }
        }
        ok[i] = (unsigned char)good;
    }
}
