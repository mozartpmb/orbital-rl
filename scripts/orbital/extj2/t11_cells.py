#!/usr/bin/env python3
"""T11 cell table — re-export of the canonical definition.

The table lives inside the env package (`pufferlib.ocean.orbital.t11_cells`)
because it IS task configuration: the trainer constructs the env and must be
able to install the mixture without reaching into scripts/. This module exists
so the gates and the campaign import the same object the trainer does — a
mixture whose table differs between training and evaluation is exactly the
class of silent divergence T11's gates are built to catch.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'pufferlib'))

from pufferlib.ocean.orbital.t11_cells import (           # noqa: F401,E402
    CELLS, TABLE, NAMES, FIELDS, as_array, base_env_kwargs, nav_env_kwargs,
    OBS_ALT_SCALE, LVLH_SCALE, FUEL_MIN, FUEL_MAX, NAV_MAX_TICKS,
    TIGHT_CELLS, ALL_CELLS, CONSOL_CELLS, CONSOL_TABLE, consol_as_array,
    T15_CELLS, T15_TABLE, t15_as_array,
    T15B_CELLS, T15B_TABLE, t15b_as_array,
    T15C_CELLS, T15C_TABLE, t15c_as_array,
)

if __name__ == '__main__':
    print(f'{len(CELLS)} cells; weights sum {sum(c["weight"] for c in TABLE):.3f}')
    for n, c in CELLS:
        print(f'  {n:14s} w {c["weight"]:.2f} cap {int(c["cap"]):6d} '
              f'box {c["box_r"]/1e3:5.1f}km/{c["box_v"]:.0f} '
              f'a {c["a_min"]/1e6:.3f}-{c["a_max"]/1e6:.3f}e6 '
              f'e {c["e_max_target"]:.2f}')
