import jax.numpy as jnp

from jax_penning_pic.particles import write_particles
from jax_penning_pic.state import empty_particle_pool


def test_write_particles_drops_invalid_updates_without_touching_slot_zero():
    pool = empty_particle_pool(capacity=4, magnetic_field=5.0)
    pool = pool._replace(alive=jnp.array([False, True, True, False]))

    indices = jnp.array([0, 3, -1, -1])
    valid = jnp.array([True, False, False, False])
    values = jnp.array([10.0, 20.0, 30.0, 40.0])
    zeros = jnp.zeros_like(values)

    updated = write_particles(pool, indices, valid, values, values, zeros, zeros, zeros)

    assert bool(updated.alive[0])
    assert not bool(updated.alive[3])
    assert float(updated.x[0]) == 10.0
    assert float(updated.x[3]) == 0.0
