/* ext_3d_conv_probe.c — recon probe for the ext-3d element-set decision.
 *
 * READ-ONLY w.r.t. the repo: this file duplicates the legacy 2D conversion
 * functions verbatim from pufferlib/pufferlib/ocean/orbital/orbital.h and puts
 * candidate 3D implementations beside them. Nothing here is imported by the env.
 *
 * Probes
 *   P1  bit-exactness of a GENERIC 313 forward rotation at i=raan=0 vs the
 *       legacy 2-line rotation (does the compiler contract differently?)
 *   P1b bit-exactness of the 3D inverse (equatorial branch) vs legacy inverse,
 *       for two spellings of the eccentricity vector (operation-order trap)
 *   P2  conditioning sweep in i: round-trip error in raan, omega, varpi=raan+omega
 *       and lambda = M+omega+raan as i -> 0
 *   P3  normal-burn geometry: node lands at the burn point; varpi preserved
 *   P4  chained anchor: 400 x {propagate 60 s; impulse} legacy-2D vs 3D-at-i=0
 *   P5  theta accuracy: legacy acos+vr-sign vs atan2(r.qhat, r.ehat)
 *
 * Build: cc -O2 -flto -lm ext_3d_conv_probe.c -o /tmp/ext3d && /tmp/ext3d
 *        (also run with -ffp-contract=off to isolate FMA effects)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MU 3.986004418e14
#define R_EARTH 6.371e6
#define TWO_PI (2.0 * M_PI)

/* ══ legacy scalar helpers (verbatim, orbital.h:371-412) ══════════════════ */
static inline double solve_kepler(double M, double e) {
    M = fmod(M, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;
    double E = (e < 0.8) ? M : M_PI;
    for (int i = 0; i < 5; i++) {
        double dE = (M - E + e * sin(E)) / (1.0 - e * cos(E));
        E += dE;
        if (fabs(dE) < 1e-12) break;
    }
    return E;
}
static inline double eccentric_to_true(double E, double e) {
    double x = sqrt(1.0 - e) * cos(E / 2.0);
    double y = sqrt(1.0 + e) * sin(E / 2.0);
    return 2.0 * atan2(y, x);
}
static inline double true_to_mean(double theta, double e) {
    double x = sqrt(1.0 + e) * cos(theta / 2.0);
    double y = sqrt(1.0 - e) * sin(theta / 2.0);
    double E = 2.0 * atan2(y, x);
    return E - e * sin(E);
}

/* ══ legacy 2D orbit ══════════════════════════════════════════════════════ */
typedef struct { double a, e, M, theta, omega; } O2;

static inline void p2_propagate(O2* o, double dt) {
    double n = sqrt(MU / (o->a * o->a * o->a));
    o->M += n * dt;
    o->M = fmod(o->M, 2.0 * M_PI);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
    double E = solve_kepler(o->M, o->e);
    o->theta = eccentric_to_true(E, o->e);
}
static inline void o2_cart(const O2* o, double* x, double* y, double* vx, double* vy) {
    double p = o->a * (1.0 - o->e * o->e);
    double r = p / (1.0 + o->e * cos(o->theta));
    double h = sqrt(MU * p);
    double xp  =  r * cos(o->theta);
    double yp  =  r * sin(o->theta);
    double vxp = -(MU / h) * sin(o->theta);
    double vyp =  (MU / h) * (o->e + cos(o->theta));
    double co = cos(o->omega), so = sin(o->omega);
    *x  = co * xp - so * yp;
    *y  = so * xp + co * yp;
    *vx = co * vxp - so * vyp;
    *vy = so * vxp + co * vyp;
}
static inline void o2_elem(double x, double y, double vx, double vy, O2* o) {
    double r  = sqrt(x*x + y*y);
    double v2 = vx*vx + vy*vy;
    double vr = (x*vx + y*vy) / r;
    o->a = 1.0 / (2.0/r - v2/MU);
    double ex = ((v2 - MU/r)*x - vr*r*vx) / MU;
    double ey = ((v2 - MU/r)*y - vr*r*vy) / MU;
    o->e = sqrt(ex*ex + ey*ey);
    if (o->e < 1e-10) o->omega = 0.0; else o->omega = atan2(ey, ex);
    if (o->e < 1e-10) {
        o->theta = atan2(y, x);
    } else {
        double ct = (ex*x + ey*y) / (o->e * r);
        ct = fmax(-1.0, fmin(1.0, ct));
        o->theta = acos(ct);
        if (vr < 0.0) o->theta = 2.0 * M_PI - o->theta;
    }
    o->M = true_to_mean(o->theta, o->e);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
}
/* legacy impulse (fuel bookkeeping stripped) */
static inline void o2_impulse(O2* o, double dv_pro, double dv_rad) {
    double x, y, vx, vy;
    o2_cart(o, &x, &y, &vx, &vy);
    double v_mag = sqrt(vx*vx + vy*vy);
    double pro_x = vx / v_mag, pro_y = vy / v_mag;
    double rad_x = x / sqrt(x*x + y*y), rad_y = y / sqrt(x*x + y*y);
    double dvx = dv_pro * pro_x + dv_rad * rad_x;
    double dvy = dv_pro * pro_y + dv_rad * rad_y;
    vx += dvx; vy += dvy;
    o2_elem(x, y, vx, vy, o);
}

/* ══ candidate 3D orbit ═══════════════════════════════════════════════════ */
typedef struct { double a, e, M, theta, omega, inc, raan; } O3;

/* forward, GENERIC 313 (no fast path) */
static inline void o3_cart_generic(const O3* o, double* x, double* y, double* z,
                                   double* vx, double* vy, double* vz) {
    double p = o->a * (1.0 - o->e * o->e);
    double r = p / (1.0 + o->e * cos(o->theta));
    double h = sqrt(MU * p);
    double xp  =  r * cos(o->theta);
    double yp  =  r * sin(o->theta);
    double vxp = -(MU / h) * sin(o->theta);
    double vyp =  (MU / h) * (o->e + cos(o->theta));
    double co = cos(o->omega), so = sin(o->omega);
    double cO = cos(o->raan),  sO = sin(o->raan);
    double ci = cos(o->inc),   si = sin(o->inc);
    double R11 =  cO*co - sO*so*ci, R12 = -cO*so - sO*co*ci;
    double R21 =  sO*co + cO*so*ci, R22 = -sO*so + cO*co*ci;
    double R31 =  so*si,            R32 =  co*si;
    *x  = R11*xp  + R12*yp;   *y  = R21*xp  + R22*yp;   *z  = R31*xp  + R32*yp;
    *vx = R11*vxp + R12*vyp;  *vy = R21*vxp + R22*vyp;  *vz = R31*vxp + R32*vyp;
}
/* forward, value-gated fast path: legacy statements verbatim when equatorial */
static inline void o3_cart_gated(const O3* o, double* x, double* y, double* z,
                                 double* vx, double* vy, double* vz) {
    if (o->inc == 0.0 && o->raan == 0.0) {
        double p = o->a * (1.0 - o->e * o->e);
        double r = p / (1.0 + o->e * cos(o->theta));
        double h = sqrt(MU * p);
        double xp  =  r * cos(o->theta);
        double yp  =  r * sin(o->theta);
        double vxp = -(MU / h) * sin(o->theta);
        double vyp =  (MU / h) * (o->e + cos(o->theta));
        double co = cos(o->omega), so = sin(o->omega);
        *x  = co * xp - so * yp;
        *y  = so * xp + co * yp;
        *vx = co * vxp - so * vyp;
        *vy = so * vxp + co * vyp;
        *z = 0.0; *vz = 0.0;
        return;
    }
    o3_cart_generic(o, x, y, z, vx, vy, vz);
}

/* inverse, branch table. evec_legacy_order=1 uses the legacy (vr*r*v) spelling. */
static inline void o3_elem(double x, double y, double z,
                           double vx, double vy, double vz,
                           O3* o, int evec_legacy_order, int theta_atan2) {
    double r  = sqrt(x*x + y*y + z*z);
    double v2 = vx*vx + vy*vy + vz*vz;
    double rv = x*vx + y*vy + z*vz;
    double vr = rv / r;
    o->a = 1.0 / (2.0/r - v2/MU);

    double hx = y*vz - z*vy, hy = z*vx - x*vz, hz = x*vy - y*vx;
    double hxy = sqrt(hx*hx + hy*hy);
    double hmag = sqrt(hx*hx + hy*hy + hz*hz);
    o->inc = atan2(hxy, hz);

    double ex, ey, ez;
    if (evec_legacy_order) {
        ex = ((v2 - MU/r)*x - vr*r*vx) / MU;
        ey = ((v2 - MU/r)*y - vr*r*vy) / MU;
        ez = ((v2 - MU/r)*z - vr*r*vz) / MU;
    } else {
        ex = ((v2 - MU/r)*x - rv*vx) / MU;
        ey = ((v2 - MU/r)*y - rv*vy) / MU;
        ez = ((v2 - MU/r)*z - rv*vz) / MU;
    }
    o->e = sqrt(ex*ex + ey*ey + ez*ez);
    int circular = (o->e < 1e-10);

    if (hxy == 0.0) {                       /* equatorial: legacy statements */
        o->raan = 0.0;
        if (circular) {
            o->omega = 0.0;
            o->theta = atan2(y, x);
        } else {
            o->omega = atan2(ey, ex);
            double ct = (ex*x + ey*y) / (o->e * r);
            ct = fmax(-1.0, fmin(1.0, ct));
            o->theta = acos(ct);
            if (vr < 0.0) o->theta = 2.0 * M_PI - o->theta;
        }
    } else {
        o->raan = atan2(hx, -hy);           /* n = zhat x h = (-hy, hx, 0) */
        if (o->raan < 0.0) o->raan += TWO_PI;
        double nx = -hy/hxy, ny = hx/hxy;   /* nz = 0 */
        double wx = hx/hmag, wy = hy/hmag, wz = hz/hmag;
        /* m = what x nhat  (in-plane, 90 deg ahead of the node) */
        double mx = wy*0.0 - wz*ny, my = wz*nx - wx*0.0, mz = wx*ny - wy*nx;
        if (circular) {
            o->omega = 0.0;
            o->theta = atan2(x*mx + y*my + z*mz, x*nx + y*ny);
        } else {
            o->omega = atan2(ex*mx + ey*my + ez*mz, ex*nx + ey*ny);
            if (o->omega < 0.0) o->omega += TWO_PI;
            if (theta_atan2) {
                /* qhat = what x ehat, 90 deg ahead of periapsis */
                double eux = ex/o->e, euy = ey/o->e, euz = ez/o->e;
                double qx = wy*euz - wz*euy, qy = wz*eux - wx*euz, qz = wx*euy - wy*eux;
                o->theta = atan2(x*qx + y*qy + z*qz, x*eux + y*euy + z*euz);
                if (o->theta < 0.0) o->theta += TWO_PI;
            } else {
                double ct = (ex*x + ey*y + ez*z) / (o->e * r);
                ct = fmax(-1.0, fmin(1.0, ct));
                o->theta = acos(ct);
                if (vr < 0.0) o->theta = 2.0 * M_PI - o->theta;
            }
        }
    }
    o->M = true_to_mean(o->theta, o->e);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
}

static inline void o3_impulse(O3* o, double dv_pro, double dv_rad, double dv_nor,
                              int gated) {
    double x, y, z, vx, vy, vz;
    if (gated) o3_cart_gated(o, &x, &y, &z, &vx, &vy, &vz);
    else       o3_cart_generic(o, &x, &y, &z, &vx, &vy, &vz);
    double v_mag = sqrt(vx*vx + vy*vy + vz*vz);
    double r_mag = sqrt(x*x + y*y + z*z);
    double pro_x = vx/v_mag, pro_y = vy/v_mag, pro_z = vz/v_mag;
    double rad_x = x/r_mag,  rad_y = y/r_mag,  rad_z = z/r_mag;
    double hx = y*vz - z*vy, hy = z*vx - x*vz, hz = x*vy - y*vx;
    double hm = sqrt(hx*hx + hy*hy + hz*hz);
    double nor_x = hx/hm, nor_y = hy/hm, nor_z = hz/hm;
    double dvx = dv_pro*pro_x + dv_rad*rad_x + dv_nor*nor_x;
    double dvy = dv_pro*pro_y + dv_rad*rad_y + dv_nor*nor_y;
    double dvz = dv_pro*pro_z + dv_rad*rad_z + dv_nor*nor_z;
    vx += dvx; vy += dvy; vz += dvz;
    o3_elem(x, y, z, vx, vy, vz, o, 1, 0);
}
/* legacy-order-of-operations 3D impulse: in-plane terms summed first,
 * normal term added last, so at dv_nor=0 the sums are the legacy sums. */
static inline void o3_impulse_gated_legacyorder(O3* o, double dv_pro, double dv_rad,
                                                double dv_nor) {
    double x, y, z, vx, vy, vz;
    o3_cart_gated(o, &x, &y, &z, &vx, &vy, &vz);
    if (o->inc == 0.0 && o->raan == 0.0 && dv_nor == 0.0) {
        double v_mag = sqrt(vx*vx + vy*vy);
        double pro_x = vx / v_mag, pro_y = vy / v_mag;
        double rad_x = x / sqrt(x*x + y*y), rad_y = y / sqrt(x*x + y*y);
        double dvx = dv_pro * pro_x + dv_rad * rad_x;
        double dvy = dv_pro * pro_y + dv_rad * rad_y;
        vx += dvx; vy += dvy;
        o3_elem(x, y, 0.0, vx, vy, 0.0, o, 1, 0);
        return;
    }
    o3_impulse(o, dv_pro, dv_rad, dv_nor, 1);
}

/* ══ helpers ══════════════════════════════════════════════════════════════ */
static double urand(void) { return rand() / (double)RAND_MAX; }
static int bitsame(double a, double b) {
    uint64_t ua, ub; memcpy(&ua, &a, 8); memcpy(&ub, &b, 8);
    return ua == ub;
}
static double wrap_pi(double x) {
    x = fmod(x + M_PI, TWO_PI);
    if (x < 0.0) x += TWO_PI;
    return x - M_PI;
}

int main(void) {
    srand(20260811);
    const int N = 200000;

    /* ── P1: forward rotation bit-exactness at i = raan = 0 ─────────────── */
    long bad_gen = 0, bad_gate = 0;
    double worst_gen = 0.0;
    for (int k = 0; k < N; k++) {
        O2 o2; O3 o3;
        o2.a = R_EARTH + 3e5 + urand()*2e7;
        o2.e = urand()*0.5;
        o2.theta = (urand()*2.0 - 1.0)*M_PI;
        o2.omega = urand()*TWO_PI;
        o2.M = true_to_mean(o2.theta, o2.e);
        o3.a=o2.a; o3.e=o2.e; o3.theta=o2.theta; o3.omega=o2.omega; o3.M=o2.M;
        o3.inc=0.0; o3.raan=0.0;
        double x2,y2,vx2,vy2; o2_cart(&o2,&x2,&y2,&vx2,&vy2);
        double xg,yg,zg,vxg,vyg,vzg; o3_cart_generic(&o3,&xg,&yg,&zg,&vxg,&vyg,&vzg);
        double xt,yt,zt,vxt,vyt,vzt; o3_cart_gated(&o3,&xt,&yt,&zt,&vxt,&vyt,&vzt);
        if (!(bitsame(x2,xg)&&bitsame(y2,yg)&&bitsame(vx2,vxg)&&bitsame(vy2,vyg)
              &&zg==0.0&&vzg==0.0)) {
            bad_gen++;
            double e1 = fabs(x2-xg)+fabs(y2-yg), rr = fabs(x2)+fabs(y2);
            if (e1/rr > worst_gen) worst_gen = e1/rr;
        }
        if (!(bitsame(x2,xt)&&bitsame(y2,yt)&&bitsame(vx2,vxt)&&bitsame(vy2,vyt)
              &&zt==0.0&&vzt==0.0)) bad_gate++;
    }
    printf("P1  forward at i=raan=0, N=%d\n", N);
    printf("    generic 313 : %ld/%d NOT bit-exact vs legacy (worst rel %.3e)\n",
           bad_gen, N, worst_gen);
    printf("    value-gated : %ld/%d NOT bit-exact vs legacy\n\n", bad_gate, N);

    /* ── P1b: inverse bit-exactness (equatorial branch), evec spelling ──── */
    long bad_inv_leg = 0, bad_inv_new = 0;
    double worst_M_new = 0.0;
    for (int k = 0; k < N; k++) {
        O2 seed; seed.a = R_EARTH + 3e5 + urand()*2e7; seed.e = urand()*0.5;
        seed.theta = (urand()*2.0-1.0)*M_PI; seed.omega = urand()*TWO_PI;
        double x,y,vx,vy; o2_cart(&seed,&x,&y,&vx,&vy);
        vx += (urand()*2.0-1.0)*25.0; vy += (urand()*2.0-1.0)*25.0;   /* post-burn */
        O2 r2; o2_elem(x,y,vx,vy,&r2);
        O3 rl, rn;
        o3_elem(x,y,0.0,vx,vy,0.0,&rl,1,0);
        o3_elem(x,y,0.0,vx,vy,0.0,&rn,0,0);
        if (!(bitsame(r2.a,rl.a)&&bitsame(r2.e,rl.e)&&bitsame(r2.omega,rl.omega)
              &&bitsame(r2.theta,rl.theta)&&bitsame(r2.M,rl.M)&&rl.inc==0.0&&rl.raan==0.0))
            bad_inv_leg++;
        if (!(bitsame(r2.a,rn.a)&&bitsame(r2.e,rn.e)&&bitsame(r2.omega,rn.omega)
              &&bitsame(r2.theta,rn.theta)&&bitsame(r2.M,rn.M))) {
            bad_inv_new++;
            double d = fabs(wrap_pi(r2.M - rn.M));
            if (d > worst_M_new) worst_M_new = d;
        }
    }
    printf("P1b inverse at z=vz=0, N=%d\n", N);
    printf("    evec legacy spelling ((v2-mu/r)r - vr*r*v): %ld/%d NOT bit-exact\n",
           bad_inv_leg, N);
    printf("    evec rewritten       ((v2-mu/r)r - (r.v)v): %ld/%d NOT bit-exact"
           " (worst |dM| %.3e rad = %.3e m along-track @7000km)\n\n",
           bad_inv_new, N, worst_M_new, worst_M_new*7.0e6);

    /* ── P2: conditioning sweep in inclination ──────────────────────────── */
    printf("P2  round-trip elements->cart->elements, 20k draws per cell\n");
    printf("    %-10s %12s %12s %12s %12s %12s\n",
           "inc[rad]", "max|dRAAN|", "max|dom|", "max|dvarpi|", "max|dlam|", "max|di|");
    double incs[] = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 0.0};
    for (int c = 0; c < 10; c++) {
        double inc = incs[c];
        double mO=0, mw=0, mvp=0, mlam=0, mi=0;
        for (int k = 0; k < 20000; k++) {
            O3 o; o.a = R_EARTH + 3e5 + urand()*2e6; o.e = 1e-3 + urand()*0.3;
            o.theta = (urand()*2.0-1.0)*M_PI; o.omega = urand()*TWO_PI;
            o.raan = urand()*TWO_PI; o.inc = inc;
            if (inc == 0.0) o.raan = 0.0;
            o.M = true_to_mean(o.theta, o.e);
            double x,y,z,vx,vy,vz; o3_cart_generic(&o,&x,&y,&z,&vx,&vy,&vz);
            O3 r; o3_elem(x,y,z,vx,vy,vz,&r,1,0);
            double dO  = fabs(wrap_pi(r.raan - o.raan));
            double dw  = fabs(wrap_pi(r.omega - o.omega));
            double dvp = fabs(wrap_pi((r.raan+r.omega) - (o.raan+o.omega)));
            double dl  = fabs(wrap_pi((r.raan+r.omega+r.M) - (o.raan+o.omega+o.M)));
            double di  = fabs(r.inc - o.inc);
            if (dO>mO) mO=dO; if (dw>mw) mw=dw; if (dvp>mvp) mvp=dvp;
            if (dl>mlam) mlam=dl; if (di>mi) mi=di;
        }
        printf("    %-10.0e %12.3e %12.3e %12.3e %12.3e %12.3e\n",
               inc, mO, mw, mvp, mlam, mi);
    }
    printf("\n");

    /* ── P3: normal-burn geometry ───────────────────────────────────────── */
    {
        double worst_node = 0, worst_varpi = 0, worst_inc = 0;
        for (int k = 0; k < 20000; k++) {
            O3 o; o.a = R_EARTH + 4e5; o.e = urand()*0.05;
            o.theta = (urand()*2.0-1.0)*M_PI; o.omega = urand()*TWO_PI;
            o.inc = 0.0; o.raan = 0.0; o.M = true_to_mean(o.theta, o.e);
            double x,y,z,vx,vy,vz; o3_cart_gated(&o,&x,&y,&z,&vx,&vy,&vz);
            double ell = atan2(y, x);                  /* inertial longitude of burn */
            double varpi0 = o.raan + o.omega;
            double vt = sqrt(vx*vx+vy*vy);
            double dvn = 10.0;
            O3 b = o; o3_impulse(&b, 0.0, 0.0, dvn, 1);
            double dnode = fabs(wrap_pi(b.raan - ell));
            double dvarpi = fabs(wrap_pi((b.raan + b.omega) - varpi0));
            double inc_pred = atan(dvn / vt);          /* exact for circular */
            double dinc = fabs(b.inc - inc_pred);
            if (dnode>worst_node) worst_node=dnode;
            if (dvarpi>worst_varpi) worst_varpi=dvarpi;
            if (dinc>worst_inc) worst_inc=dinc;
        }
        printf("P3  +10 m/s normal burn from an equatorial orbit, 20k draws\n");
        printf("    max |RAAN_new - longitude_of_burn| = %.3e rad\n", worst_node);
        printf("    max |varpi_new - varpi_old|        = %.3e rad  (invariance)\n", worst_varpi);
        printf("    max |i_new - atan(dv/v_t)|         = %.3e rad  (e<=0.05 model err)\n\n",
               worst_inc);
    }

    /* ── P4: chained anchor ─────────────────────────────────────────────── */
    {
        long fail_bit = 0; double worst_rel = 0.0;
        for (int ep = 0; ep < 200; ep++) {
            O2 a2; O3 a3;
            a2.a = R_EARTH + 3e5 + urand()*5e5; a2.e = urand()*0.05;
            a2.M = urand()*TWO_PI;
            a2.theta = eccentric_to_true(solve_kepler(a2.M, a2.e), a2.e);
            a2.omega = urand()*TWO_PI;
            a3.a=a2.a; a3.e=a2.e; a3.M=a2.M; a3.theta=a2.theta; a3.omega=a2.omega;
            a3.inc=0.0; a3.raan=0.0;
            for (int s = 0; s < 400; s++) {
                int act = rand() % 6;
                double dvp = 0, dvr = 0;
                if (act==1) dvp=5.0; else if (act==2) dvp=-5.0;
                else if (act==3) dvr=10.0; else if (act==4) dvr=-10.0;
                if (dvp!=0.0 || dvr!=0.0) {
                    o2_impulse(&a2, dvp, dvr);
                    o3_impulse_gated_legacyorder(&a3, dvp, dvr, 0.0);
                }
                p2_propagate(&a2, 60.0);
                { double n = sqrt(MU/(a3.a*a3.a*a3.a));
                  a3.M += n*60.0; a3.M = fmod(a3.M, TWO_PI);
                  if (a3.M < 0.0) a3.M += TWO_PI;
                  a3.theta = eccentric_to_true(solve_kepler(a3.M, a3.e), a3.e); }
                if (!(bitsame(a2.a,a3.a)&&bitsame(a2.e,a3.e)&&bitsame(a2.M,a3.M)
                      &&bitsame(a2.theta,a3.theta)&&bitsame(a2.omega,a3.omega))) {
                    fail_bit++;
                    double rel = fabs(a2.a-a3.a)/a2.a + fabs(a2.e-a3.e);
                    if (rel>worst_rel) worst_rel=rel;
                    break;
                }
            }
        }
        printf("P4  chained anchor 200 eps x 400 steps (burn+propagate), 2D vs 3D@i=0\n");
        printf("    episodes diverging from bit-exact: %ld/200  (worst rel dev %.3e)\n\n",
               fail_bit, worst_rel);
    }

    /* ── P5: theta accuracy, acos+sign vs atan2 ─────────────────────────── */
    {
        double worst_acos = 0, worst_atan = 0;
        for (int k = 0; k < 200000; k++) {
            O3 o; o.a = R_EARTH + 4e5 + urand()*4e5; o.e = 1e-4 + urand()*0.3;
            /* concentrate near apsides where acos is worst */
            double u = urand();
            o.theta = (u < 0.5) ? (urand()-0.5)*2e-2 : M_PI + (urand()-0.5)*2e-2;
            o.omega = urand()*TWO_PI; o.inc = 0.3 + urand()*0.5; o.raan = urand()*TWO_PI;
            o.M = true_to_mean(o.theta, o.e);
            double x,y,z,vx,vy,vz; o3_cart_generic(&o,&x,&y,&z,&vx,&vy,&vz);
            O3 ra, rb;
            o3_elem(x,y,z,vx,vy,vz,&ra,1,0);   /* acos + vr sign */
            o3_elem(x,y,z,vx,vy,vz,&rb,1,1);   /* atan2 */
            double ea = fabs(wrap_pi(ra.theta - o.theta));
            double eb = fabs(wrap_pi(rb.theta - o.theta));
            if (ea>worst_acos) worst_acos=ea;
            if (eb>worst_atan) worst_atan=eb;
        }
        printf("P5  true anomaly recovery near apsides (|theta| or |theta-pi| < 0.01 rad)\n");
        printf("    legacy acos + vr-sign : max err %.3e rad  (= %.3f m along-track @7000 km)\n",
               worst_acos, worst_acos*7.0e6);
        printf("    atan2(r.qhat, r.ehat) : max err %.3e rad  (= %.3e m)\n\n",
               worst_atan, worst_atan*7.0e6);
    }

    /* ── P6: conditioning under ABSOLUTE state noise (the real pipeline) ── */
    printf("P6  element error induced by a 1 m / 1 mm/s Cartesian perturbation\n");
    printf("    (median over 20k draws; e ~ U[0.01,0.30]; this is the number that\n");
    printf("     matters for float32 logs, EKF estimates and obs quantization)\n");
    printf("    %-10s %12s %12s %12s %12s %12s\n",
           "inc[rad]", "med|dRAAN|", "med|dom|", "med|dvarpi|", "med|dlam|", "med|dwhat|");
    {
        double incs2[] = {5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
        for (int c = 0; c < 7; c++) {
            double inc = incs2[c];
            static double sO[20000], sw[20000], sv[20000], sl[20000], sh[20000];
            for (int k = 0; k < 20000; k++) {
                O3 o; o.a = R_EARTH + 4e5 + urand()*4e5; o.e = 0.01 + urand()*0.29;
                o.theta = (urand()*2.0-1.0)*M_PI; o.omega = urand()*TWO_PI;
                o.raan = urand()*TWO_PI; o.inc = inc;
                o.M = true_to_mean(o.theta, o.e);
                double x,y,z,vx,vy,vz; o3_cart_generic(&o,&x,&y,&z,&vx,&vy,&vz);
                O3 r0; o3_elem(x,y,z,vx,vy,vz,&r0,1,1);
                /* isotropic 1 m position / 1 mm/s velocity perturbation */
                double px=(urand()*2-1), py=(urand()*2-1), pz=(urand()*2-1);
                double pn=sqrt(px*px+py*py+pz*pz)+1e-30; px/=pn; py/=pn; pz/=pn;
                double qx=(urand()*2-1), qy=(urand()*2-1), qz=(urand()*2-1);
                double qn=sqrt(qx*qx+qy*qy+qz*qz)+1e-30; qx/=qn; qy/=qn; qz/=qn;
                O3 r1; o3_elem(x+px, y+py, z+pz,
                               vx+1e-3*qx, vy+1e-3*qy, vz+1e-3*qz, &r1, 1, 1);
                sO[k] = fabs(wrap_pi(r1.raan - r0.raan));
                sw[k] = fabs(wrap_pi(r1.omega - r0.omega));
                sv[k] = fabs(wrap_pi((r1.raan+r1.omega) - (r0.raan+r0.omega)));
                sl[k] = fabs(wrap_pi((r1.raan+r1.omega+r1.M) - (r0.raan+r0.omega+r0.M)));
                /* unit angular-momentum direction change */
                double w0x=sin(r0.inc)*sin(r0.raan), w0y=-sin(r0.inc)*cos(r0.raan), w0z=cos(r0.inc);
                double w1x=sin(r1.inc)*sin(r1.raan), w1y=-sin(r1.inc)*cos(r1.raan), w1z=cos(r1.inc);
                sh[k] = sqrt((w1x-w0x)*(w1x-w0x)+(w1y-w0y)*(w1y-w0y)+(w1z-w0z)*(w1z-w0z));
            }
            #define MEDIAN(arr) ({ static double t[20000]; memcpy(t,arr,sizeof(t)); \
                for (int i=1;i<20000;i++){double v=t[i];int j=i-1; \
                    while(j>=0&&t[j]>v){t[j+1]=t[j];j--;} t[j+1]=v;} t[10000]; })
            printf("    %-10.0e %12.3e %12.3e %12.3e %12.3e %12.3e\n",
                   inc, MEDIAN(sO), MEDIAN(sw), MEDIAN(sv), MEDIAN(sl), MEDIAN(sh));
            #undef MEDIAN
        }
    }
    printf("\n");

    /* ── P7: continuity of lambda / varpi through a NORMAL burn ─────────── */
    printf("P7  normal burn continuity (e ~ U[0.01,0.30], i0 ~ U[0,0.1], dv_n=10 m/s)\n");
    {
        static double dl[20000], dv_[20000], du[20000];
        for (int k = 0; k < 20000; k++) {
            O3 o; o.a = R_EARTH + 4e5; o.e = 0.01 + urand()*0.29;
            o.theta = (urand()*2.0-1.0)*M_PI; o.omega = urand()*TWO_PI;
            o.inc = urand()*0.1; o.raan = urand()*TWO_PI;
            o.M = true_to_mean(o.theta, o.e);
            double lam0 = o.raan + o.omega + o.M;
            double u0   = o.raan + o.omega + o.theta;
            double vp0  = o.raan + o.omega;
            O3 b = o; o3_impulse(&b, 0.0, 0.0, 10.0, 0);
            dl[k]  = fabs(wrap_pi((b.raan+b.omega+b.M) - lam0));
            dv_[k] = fabs(wrap_pi((b.raan+b.omega) - vp0));
            du[k]  = fabs(wrap_pi((b.raan+b.omega+b.theta) - u0));
        }
        #define MED(arr) ({ static double t[20000]; memcpy(t,arr,sizeof(t)); \
            for (int i=1;i<20000;i++){double v=t[i];int j=i-1; \
                while(j>=0&&t[j]>v){t[j+1]=t[j];j--;} t[j+1]=v;} t[10000]; })
        #define P99(arr) ({ static double t[20000]; memcpy(t,arr,sizeof(t)); \
            for (int i=1;i<20000;i++){double v=t[i];int j=i-1; \
                while(j>=0&&t[j]>v){t[j+1]=t[j];j--;} t[j+1]=v;} t[19800]; })
        printf("    |d lambda| (M+om+RAAN) med %.3e  p99 %.3e rad\n", MED(dl), P99(dl));
        printf("    |d varpi|  (om+RAAN)   med %.3e  p99 %.3e rad\n", MED(dv_), P99(dv_));
        printf("    |d u|      (th+om+RAAN)med %.3e  p99 %.3e rad\n", MED(du), P99(du));
        #undef MED
        #undef P99
    }
    printf("\n");
    return 0;
}
