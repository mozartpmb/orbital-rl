"""Filter-in-the-loop navigation wrapper around the Orbital C environment.

No new C extension: `OrbitalNav` subclasses `pufferlib.ocean.orbital.Orbital`
and reuses its compiled binding unchanged, so `puffer_orbital` stays
byte-identical and every T3/T4 regression anchor keeps reproducing.
"""

from .orbital_nav import OrbitalNav

__all__ = ['OrbitalNav']
