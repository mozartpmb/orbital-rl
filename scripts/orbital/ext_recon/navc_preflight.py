#!/usr/bin/env python3
"""Two preflight checks for any campaign that runs the C filter path.

L1  THE kwargs.get LEAK LINT.
    `OrbitalNav.__init__` forwards **kwargs to `Orbital.__init__`, which takes
    EXPLICIT parameters and therefore REJECTS anything it does not know. So a
    wrapper-only option read with `kwargs.get(...)` — which leaves the key in
    the dict — is a TypeError before construction, while the same option read
    with `kwargs.pop(...)` is fine. The distinction is invisible at the call
    site and the failure is total.

    This shape has now fired three times:
        MAJOR-17b  nav_max_ticks landed in the shared base_env_kwargs
        T13b       (same class, caught by the anchor gate)
        T14        nav_filter_impl read with .get, caught by gate C1b

    Each time it was found by an expensive end-to-end gate. It is decidable
    statically: collect every key read via `kwargs.get(` in the wrapper, drop
    the ones `Orbital.__init__` actually accepts, drop the ones also popped,
    and anything left is a leak. That is this check, and it runs in
    milliseconds instead of a two-epoch training run.

L2  THE C PATH IS ACTUALLY REACHABLE.
    The port only replaces `stm_fd_j2`, which only `BatchedBearingMSC6J2` calls.
    A two-body run therefore gets Python NO MATTER WHAT `nav_filter_impl` says —
    correct behaviour, but it means a campaign can request 'c', pass every gate,
    run at Python speed, and report itself as having run on C. That is an
    attribution falsehood and a ~4.5x wall-clock surprise (a 50M W1 rung is ~8 h
    on C and ~35 h without).

    ABORT, not warn. A warning in a nohup log at 23:45 is a warning nobody
    reads until the run is a day late, and the fix is one flag. The cost of
    being wrong in each direction is not symmetric.
"""
import argparse
import ast
import inspect
import os
import sys

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))

G_PASS, G_FAIL = [], []


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


WRAPPER = os.path.join(WT, 'pufferlib', 'pufferlib', 'ocean', 'orbital_nav',
                       'orbital_nav.py')


def lint_kwarg_leak():
    print('\n== L1  wrapper-only kwargs must be POPPED, not GET (MAJOR-17b class) ==')
    from pufferlib.ocean.orbital.orbital import Orbital
    sig = inspect.signature(Orbital.__init__)
    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    known = set(sig.parameters)

    tree = ast.parse(open(WRAPPER).read())
    got, popped = {}, set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'kwargs'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        key = node.args[0].value
        if node.func.attr == 'get':
            got.setdefault(key, node.lineno)
        elif node.func.attr == 'pop':
            popped.add(key)

    if accepts_var_kw:
        check('L1 Orbital takes **kwargs — leak is not possible', True,
              'the lint is vacuous for this signature; kept so it fails loudly '
              'if the signature ever tightens')
        return
    leaks = {k: ln for k, ln in got.items()
             if k not in known and k not in popped}
    check('L1 no wrapper-only kwarg is read with .get and forwarded',
          not leaks,
          (f'LEAKS: ' + ', '.join(f'{k!r} (line {ln})' for k, ln in sorted(
              leaks.items(), key=lambda kv: kv[1]))
           if leaks else
           f'{len(got)} kwargs.get keys checked against '
           f'{len(known)} Orbital parameters; {len(popped)} popped. '
           f'A key that is neither accepted by Orbital nor popped is a '
           f'TypeError before construction.'))


def check_c_reachable(require_impl):
    print(f'\n== L2  the C path is reachable for the W1 cell (impl={require_impl!r}) ==')
    if require_impl != 'c':
        check('L2 skipped — campaign is not requesting the C path', True,
              f'nav_filter_impl={require_impl!r}')
        return
    import numpy as np                                            # noqa: F401
    import pufferlib.ocean.orbital_nav.nav_math3d as n3
    from pufferlib.ocean.orbital_nav import nav_c
    import t11_cells as T
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav

    check('L2a the kernel is built and loadable', nav_c.available(),
          nav_c.why_unavailable() if not nav_c.available()
          else os.path.join(os.path.dirname(nav_c.__file__), 'nav_j2_kernel.so'))
    if not nav_c.available():
        return

    c = dict(T.CELLS)['W1_driftwait']
    kw = T.nav_env_kwargs(
        num_envs=4, nav_mode='bearings_only', cell_mixture_mode=0,
        j2_mode=int(c['j2']), nav_j2_mode=int(c['j2']),
        episode_cap_steps=int(c['cap']), rendezvous_radius_m=c['box_r'],
        rel_vel_tol_ms=c['box_v'], a_min_override=c['a_min'],
        a_max_override=c['a_max'], e_max_target=c['e_max_target'],
        e_max_sat=c['e_max_sat'], de_max=c['de_max'], da_max_m=c['da_max'],
        di_max_rad=c['di_max'], di_min_rad=c['di_min'],
        di_phase_mode=int(c['di_phase']),
        i_target_min_rad=c['i_t_min'], i_target_max_rad=c['i_t_max'],
        fuel_frac_min=c['fuel_min'], fuel_frac_max=c['fuel_max'],
        nav_filter_impl='c')
    kw['nav_max_ticks'] = 0
    e = OrbitalNav(**kw)
    e.reset(seed=3)
    impl = n3.filter_impl()
    filt = type(e._filt).__name__
    e.close()
    n3.set_filter_impl('py')

    check('L2b the active implementation is "c" after construction',
          impl == 'c', f'nav_math3d.filter_impl() = {impl!r}')
    # The port only replaces stm_fd_j2, which ONLY the J2 filter calls.
    check('L2c the W1 cell instantiates the J2 filter, so the port is on the '
          'hot path', 'J2' in filt,
          f'filter class = {filt}. Without J2 the C kernel is never entered and '
          f'the run silently costs ~4.5x more wall clock than reported '
          f'(50M W1 rung: ~8 h on C vs ~35 h).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', default='c')
    a = ap.parse_args()
    lint_kwarg_leak()
    check_c_reachable(a.impl)
    print(f'\n=== {len(G_PASS)}/{len(G_PASS)+len(G_FAIL)} preflight checks pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    sys.exit(1 if G_FAIL else 0)
