/* ext-3d anchor A2 — Φ(shaping_mode=2) at Δi = 0 must equal Φ(shaping_mode=1)
 * BIT-EXACTLY, in double precision.
 *
 * Why this is a blocker rather than a nicety (3d_REDTEAM BLOCKER-2): the
 * literal 3d_C spelling of mode 2 (hypot + Cartesian-derived ē) mismatches mode
 * 1 on 87.7% of draws, and the break survives the float32 cast — 1 reward in
 * 10^4 differs, so a 3000-step episode diverges with ~0.3 expected mismatches
 * and a 200-episode anchor with ~60. The shipped spelling is variant V3:
 * sqrt(x*x + y*y) (NOT hypot) with ELEMENT-derived 3D e-vectors.
 *
 * Build & run:
 *   cc -O2 -flto -lm -I pufferlib/pufferlib/ocean/orbital \
 *      scripts/orbital/ext3d/a2_bitexact.c -o /tmp/a2 && /tmp/a2
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "orbital.h"

static double urand(void) { return rand() / (double)RAND_MAX; }

static int bitcmp(double a, double b) {
    uint64_t ua, ub;
    memcpy(&ua, &a, 8);
    memcpy(&ub, &b, 8);
    return ua != ub;
}

/* One cell of the sweep: N draws at a given weight set and eccentricity band. */
static void cell(const char* name, long n, double e_max, double dv_ref,
                 double w_lambda, double w_match, double a_lo, double a_hi,
                 unsigned seed) {
    Orbital env;
    memset(&env, 0, sizeof(env));
    env.shape_w_lambda  = w_lambda;
    env.shape_w_match   = w_match;
    env.shape_dv_ref_ms = dv_ref;
    env.shape_match_squash = 0;

    srand(seed);
    long mism = 0, sat_hit = 0;
    double worst = 0.0;
    for (long k = 0; k < n; k++) {
        double a_t = a_lo + urand() * (a_hi - a_lo);
        double a_s = a_lo + urand() * (a_hi - a_lo);
        env.target.a         = a_t;
        env.target.e         = urand() * e_max;
        env.target.omega     = urand() * 2.0 * M_PI;
        env.target.M         = urand() * 2.0 * M_PI;
        env.target.theta     = eccentric_to_true(solve_kepler(env.target.M, env.target.e),
                                                 env.target.e);
        env.target.inc = 0.0; env.target.raan = 0.0;

        env.sat.orbit.a      = a_s;
        env.sat.orbit.e      = urand() * e_max;
        env.sat.orbit.omega  = urand() * 2.0 * M_PI;
        env.sat.orbit.M      = urand() * 2.0 * M_PI;
        env.sat.orbit.theta  = eccentric_to_true(solve_kepler(env.sat.orbit.M, env.sat.orbit.e),
                                                 env.sat.orbit.e);
        env.sat.orbit.inc = 0.0; env.sat.orbit.raan = 0.0;

        env.shaping_mode = 1;  double p1 = compute_phi(&env);
        env.shaping_mode = 2;  double p2 = compute_phi(&env);

        if (bitcmp(p1, p2)) {
            mism++;
            double d = fabs(p1 - p2);
            if (d > worst) worst = d;
        }
        /* saturation bookkeeping: is min(1, ·) active at this draw? */
        double da_rel = (a_s - a_t) / a_t;
        double v_t = sqrt(MU / a_t);
        double e_sx = env.sat.orbit.e * cos(env.sat.orbit.omega);
        double e_sy = env.sat.orbit.e * sin(env.sat.orbit.omega);
        double e_tx = env.target.e * cos(env.target.omega);
        double e_ty = env.target.e * sin(env.target.omega);
        double de = sqrt((e_sx-e_tx)*(e_sx-e_tx) + (e_sy-e_ty)*(e_sy-e_ty));
        if (0.5 * v_t * sqrt(da_rel*da_rel + de*de) > dv_ref) sat_hit++;
    }
    printf("%-38s n=%ld  double mismatches=%ld (%.4f%%)  max|ΔΦ|=%.3e  "
           "min(1,·) saturated=%.3f%%\n",
           name, n, mism, 100.0 * mism / (double)n, worst,
           100.0 * sat_hit / (double)n);
}

int main(void) {
    printf("=== ext-3d anchor A2: Φ(mode 2) == Φ(mode 1) at Δi = 0, bitwise ===\n");
    /* mode 1's OWN weights (3d_REDTEAM MAJOR-6: A2 must be run at them) */
    cell("T3 weights  w_m=0.35 dv_ref=300",  200000, 0.05, 300.0, 1.0, 0.35,
         R_EARTH + 300e3, R_EARTH + 800e3, 20260811u);
    /* the ext-3d rung weights */
    cell("3D weights  w_m=0.8166667 ref=700", 200000, 0.05, 700.0, 1.0, 0.8166667,
         R_EARTH + 300e3, R_EARTH + 800e3, 20260812u);
    /* wide-e / wide-band stress (X4/X5-like), where the squash is live */
    cell("wide  e<=0.30, 300-8000 km, ref=700", 200000, 0.30, 700.0, 1.0, 0.8166667,
         R_EARTH + 300e3, R_EARTH + 8000e3, 20260813u);
    /* e = 0 exactly: the ω degeneracy corner */
    cell("e = 0 exactly (ω degeneracy)",      200000, 0.0,  700.0, 1.0, 0.8166667,
         R_EARTH + 500e3, R_EARTH + 800e3, 20260814u);
    return 0;
}
