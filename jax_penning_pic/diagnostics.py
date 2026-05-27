from __future__ import annotations

import jax.numpy as jnp

from .geometry import inside_hot_cathode
from .state import CounterState, GeometryState, ParticlePool


def count_alive(pool: ParticlePool) -> int:
    return int(jnp.sum(pool.alive))


def count_hot(pool: ParticlePool, geometry: GeometryState) -> int:
    return int(jnp.sum(pool.alive & inside_hot_cathode(pool.x, pool.y, geometry)))


def counters_row(step: int, electrons: ParticlePool, ions: ParticlePool, geometry: GeometryState, counters: CounterState):
    return {
        "step": int(step),
        "ne": count_alive(electrons),
        "ni": count_alive(ions),
        "ne_hot": count_hot(electrons, geometry),
        "ni_hot": count_hot(ions, geometry),
        "Ntot_ionized": int(counters.ntot_ionized),
        "Ntot_cold_cathode_leave": int(counters.ntot_cold_cathode_leave),
        "Ntot_anode_leave": int(counters.ntot_anode_leave),
        "Ntot_hot_cathode_emission": int(counters.ntot_hot_cathode_emission),
    }
