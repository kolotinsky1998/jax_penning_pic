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
    assert int(updated.generation[0]) == 1
    assert int(updated.generation[3]) == 0


def test_reusing_a_slot_increments_its_generation():
    pool = empty_particle_pool(capacity=1, magnetic_field=0.0)
    index = jnp.array([0])
    valid = jnp.array([True])
    values = jnp.array([1.0])
    zeros = jnp.zeros_like(values)

    first = write_particles(pool, index, valid, values, values, zeros, zeros, zeros)
    dead = first._replace(alive=jnp.array([False]))
    second = write_particles(dead, index, valid, values, values, zeros, zeros, zeros)

    assert int(first.generation[0]) == 1
    assert int(second.generation[0]) == 2
