"""N3D-C probe 3 — how much does obs[13,14] (the phase channel) move when the
TARGET PLANE used to build the gauge is an ESTIMATE rather than the truth?

Under `dim3_mode=1`, orbital.h builds `PlaneGauge g = gauge_from_orbit(target)`
and writes obs[13,14] = sin/cos(orb_lambda_gauge(sat,g) - orb_lambda_gauge(tgt,g)).
A 3D-nav wrapper must rebuild those two slots from the target ESTIMATE, so the
gauge is built from an estimated angular-momentum direction. Bearings-only is
weakest exactly out of plane, so this probe measures the transfer function

        d(Delta lambda) / d(gauge tilt)

as a function of the chaser's own inclination (the ext-3d envelope is
di_rel <= 1 deg, where lambda's conditioning was the whole reason for the
gauge in the first place).

Also measured: the same transfer for the OTHER target-plane-dependent slots
21-28, so the encode table can be ordered by fragility.

Output: web_data/results/n3d_gauge_sensitivity.csv
"""
import csv
import os

import numpy as np

MU = 3.986004418e14
OUT = '/Users/pete/space_training/web_data/results'


def hhat(inc, raan):
    return np.stack([np.sin(inc) * np.sin(raan),
                     -np.sin(inc) * np.cos(raan),
                     np.cos(inc)], -1)


def evec(e, om, inc, raan):
    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(om), np.sin(om)
    ci, si = np.cos(inc), np.sin(inc)
    return np.stack([e * (cO * cw - sO * sw * ci),
                     e * (sO * cw + cO * sw * ci),
                     e * (sw * si)], -1)


def coe2rv(a, e, th, om, inc, raan):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(th))
    h = np.sqrt(MU * p)
    xp, yp = r * np.cos(th), r * np.sin(th)
    vxp, vyp = -(MU / h) * np.sin(th), (MU / h) * (e + np.cos(th))
    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(om), np.sin(om)
    ci, si = np.cos(inc), np.sin(inc)
    R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci],
                  [sw * si, cw * si]])
    rr = np.einsum('ij...,j...->i...', R, np.stack([xp, yp]))
    vv = np.einsum('ij...,j...->i...', R, np.stack([vxp, vyp]))
    return np.moveaxis(rr, 0, -1), np.moveaxis(vv, 0, -1)


def rv2elem(r, v):
    """(a, e, om, inc, raan, M) — the orbital.h inclined branch."""
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, -1)
    a = 1.0 / (2.0 / rn - v2 / MU)
    ev = (np.cross(v, h) / MU) - r / rn[..., None]
    e = np.linalg.norm(ev, axis=-1)
    inc = np.arctan2(np.linalg.norm(h[..., :2], axis=-1), h[..., 2])
    nvec = np.stack([-h[..., 1], h[..., 0], np.zeros_like(h[..., 0])], -1)
    nn = np.linalg.norm(nvec, axis=-1)
    raan = np.where(nn > 1e-14, np.arctan2(nvec[..., 1], nvec[..., 0]), 0.0)
    nu = np.where((nn > 1e-14)[..., None], nvec / np.maximum(nn, 1e-300)[..., None],
                  np.array([1.0, 0.0, 0.0]))
    om = np.arctan2(np.sum(np.cross(nu, ev) * (h / hn[..., None]), -1),
                    np.sum(nu * ev, -1))
    # true anomaly
    th = np.arctan2(np.sum(np.cross(ev, r) * (h / hn[..., None]), -1),
                    np.sum(ev * r, -1))
    ec = np.clip(e, 0, 1 - 1e-12)
    E = 2.0 * np.arctan2(np.sqrt(1 - ec) * np.sin(0.5 * th),
                         np.sqrt(1 + ec) * np.cos(0.5 * th))
    M = E - ec * np.sin(E)
    return a, e, om, inc, raan, M


def lambda_gauge(a, e, th, om, inc, raan, M, g):
    """orb_lambda_gauge: M + (omega + Omega) recomputed in the gauge frame."""
    r, v = coe2rv(a, e, th, om, inc, raan)
    R = np.stack(g, -2)                       # (..., 3, 3) rows e1,e2,e3
    rg = np.einsum('...ij,...j->...i', R, r)
    vg = np.einsum('...ij,...j->...i', R, v)
    _, _, omg, _, rag, _ = rv2elem(rg, vg)
    return M + omg + rag


def gauge_from_h(h):
    e3 = h / np.linalg.norm(h, axis=-1, keepdims=True)
    nx, ny = -e3[..., 1], e3[..., 0]
    nn = np.hypot(nx, ny)
    e1 = np.where(nn[..., None] > 1e-14,
                  np.stack([nx / np.maximum(nn, 1e-300),
                            ny / np.maximum(nn, 1e-300),
                            np.zeros_like(nx)], -1),
                  np.array([1.0, 0.0, 0.0]))
    e2 = np.cross(e3, e1)
    return (e1, e2, e3)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def run(n=400, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for i_s_deg in (0.02, 0.1, 0.25, 0.5, 1.0):
        a_s = rng.uniform(6.871e6, 7.171e6, n)
        a_t = rng.uniform(6.871e6, 7.171e6, n)
        e_s = rng.uniform(0.0, 0.05, n)
        e_t = rng.uniform(0.0, 0.05, n)
        om_s = rng.uniform(0, 2 * np.pi, n)
        om_t = rng.uniform(0, 2 * np.pi, n)
        M_s = rng.uniform(0, 2 * np.pi, n)
        M_t = rng.uniform(0, 2 * np.pi, n)
        inc_s = np.full(n, np.radians(i_s_deg))
        raan_s = rng.uniform(0, 2 * np.pi, n)
        inc_t = np.zeros(n); raan_t = np.zeros(n)

        def th_of(M, e):
            E = M.copy()
            for _ in range(12):
                E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(0.5 * E),
                                  np.sqrt(1 - e) * np.cos(0.5 * E))
        th_s, th_t = th_of(M_s, e_s), th_of(M_t, e_t)

        h_t = hhat(inc_t, raan_t)
        g0 = gauge_from_h(h_t)
        dl0 = wrap(lambda_gauge(a_s, e_s, th_s, om_s, inc_s, raan_s, M_s, g0)
                   - lambda_gauge(a_t, e_t, th_t, om_t, inc_t, raan_t, M_t, g0))

        for tilt_deg in (0.001, 0.01, 0.1, 0.5):
            ang = rng.uniform(0, 2 * np.pi, n)
            eps = np.radians(tilt_deg)
            # tilt h_t by eps about a random in-plane axis
            ax = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], -1)
            h_p = (h_t * np.cos(eps) + np.cross(ax, h_t) * np.sin(eps)
                   + ax * np.sum(ax * h_t, -1)[..., None] * (1 - np.cos(eps)))
            gp = gauge_from_h(h_p)
            dl1 = wrap(lambda_gauge(a_s, e_s, th_s, om_s, inc_s, raan_s, M_s, gp)
                       - lambda_gauge(a_t, e_t, th_t, om_t, inc_t, raan_t, M_t, gp))
            d = np.degrees(np.abs(wrap(dl1 - dl0)))
            rows.append(dict(i_chaser_deg=i_s_deg, gauge_tilt_deg=tilt_deg, n=n,
                             dlam_err_deg_med=float(np.median(d)),
                             dlam_err_deg_p95=float(np.percentile(d, 95)),
                             dlam_err_deg_max=float(d.max()),
                             gain_deg_per_deg=float(np.median(d) / tilt_deg)))
            print(f"i_s={i_s_deg:5.2f} deg  tilt={tilt_deg:6.3f} deg  "
                  f"|d(dlam)| med {np.median(d):.5f} deg  p95 "
                  f"{np.percentile(d, 95):.5f}  max {d.max():.5f}  "
                  f"gain {np.median(d)/tilt_deg:.3f} deg/deg")

        # the plane channels themselves, for comparison
        for tilt_deg in (0.01, 0.1):
            eps = np.radians(tilt_deg)
            ang = rng.uniform(0, 2 * np.pi, n)
            ax = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], -1)
            h_p = (h_t * np.cos(eps) + np.cross(ax, h_t) * np.sin(eps)
                   + ax * np.sum(ax * h_t, -1)[..., None] * (1 - np.cos(eps)))
            h_s = hhat(inc_s, raan_s)

            def di_vec(ht):
                c = np.cross(ht, h_s)
                cn = np.linalg.norm(c, axis=-1)
                dot = np.sum(ht * h_s, -1)
                di = np.arctan2(cn, dot)
                return di[..., None] * c / np.maximum(cn, 1e-300)[..., None], di
            _, di0 = di_vec(h_t)
            _, di1 = di_vec(h_p)
            d = np.degrees(np.abs(di1 - di0))
            rows.append(dict(i_chaser_deg=i_s_deg, gauge_tilt_deg=tilt_deg, n=n,
                             dlam_err_deg_med=float(np.median(d)),
                             dlam_err_deg_p95=float(np.percentile(d, 95)),
                             dlam_err_deg_max=float(d.max()),
                             gain_deg_per_deg=float(np.median(d) / tilt_deg)))
    return rows


if __name__ == '__main__':
    rows = run()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'n3d_gauge_sensitivity.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p}")
