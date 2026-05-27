from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import inside_anode, inside_cold_cathode_ring
from .particles import all_free_slots, deactivate, first_free_slots, write_particles
from .state import CounterState, GeometryState, ParticlePool, RuntimeState


def _sample_ring(key, center_x, center_y, rmin, rmax, count):
    key_r, key_theta = jax.random.split(key)
    radius = rmin + (rmax - rmin) * jnp.sqrt(jax.random.uniform(key_r, (count,), dtype=jnp.float32))
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (count,), dtype=jnp.float32)
    x = radius * jnp.cos(theta) + center_x
    y = radius * jnp.sin(theta) + center_y
    return x, y


def inject_initial_disk(pool: ParticlePool, key, geometry: GeometryState, count: int, speed_std: float):
    key_r, key_theta, key_v = jax.random.split(key, 3)
    radius = geometry.radius_injection * jnp.sqrt(jax.random.uniform(key_r, (count,), dtype=jnp.float32))
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (count,), dtype=jnp.float32)
    x = radius * jnp.cos(theta) + geometry.x_center
    y = radius * jnp.sin(theta) + geometry.y_center
    vel = jax.random.normal(key_v, (count, 3), dtype=jnp.float32) * speed_std
    indices, valid = first_free_slots(pool.alive, count)
    return write_particles(pool, indices, valid, x, y, vel[:, 0], vel[:, 1], vel[:, 2])


def sample_reference_initial_disk(key, geometry: GeometryState, count: int):
    key_pos, key_vel = jax.random.split(key)
    key_r, key_theta = jax.random.split(key_pos)
    radius = geometry.radius_injection * jnp.sqrt(jax.random.uniform(key_r, (count,), dtype=jnp.float32))
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (count,), dtype=jnp.float32)
    x = radius * jnp.cos(theta) + geometry.x_center
    y = radius * jnp.sin(theta) + geometry.y_center

    chunks = []
    accepted = 0
    batch_size = max(64, count * 2)
    while accepted < count:
        key_vel, batch_key = jax.random.split(key_vel)
        candidates = np.asarray(jax.random.normal(batch_key, (batch_size, 3), dtype=jnp.float32))
        candidates = candidates[np.linalg.norm(candidates, axis=1) < 3.0]
        if candidates.size == 0:
            continue
        take = min(count - accepted, candidates.shape[0])
        chunks.append(candidates[:take])
        accepted += take

    velocity_factors = jnp.asarray(np.concatenate(chunks, axis=0), dtype=jnp.float32)
    return x, y, velocity_factors


def inject_initial_disk_samples(pool: ParticlePool, x, y, velocity_factors, speed_std: float):
    vel = velocity_factors * speed_std
    indices, valid = first_free_slots(pool.alive, x.shape[0])
    return write_particles(pool, indices, valid, x, y, vel[:, 0], vel[:, 1], vel[:, 2])


def hot_cathode_emission(pool: ParticlePool, key, geometry: GeometryState, runtime: RuntimeState, particle_mass: float):
    size = pool.x.shape[0]
    count_mask = jnp.arange(size) < runtime.hot_emit_count
    x, y = _sample_ring(key, geometry.x_center, geometry.y_center, 0.0, geometry.ion_radius_leave_min, size)
    vz = jnp.full((size,), jnp.sqrt(2.0 * 100.0 * 1.6021766208e-19 / particle_mass), dtype=jnp.float32)
    zeros = jnp.zeros((size,), dtype=jnp.float32)
    indices, free_valid = all_free_slots(pool.alive)
    valid = count_mask & free_valid
    pool = write_particles(pool, indices, valid, x, y, zeros, zeros, vz)
    emitted = valid.astype(jnp.int32).sum()
    return pool, emitted


def cold_secondary_emission(pool: ParticlePool, key, geometry: GeometryState, runtime: RuntimeState, particle_mass: float):
    size = pool.x.shape[0]
    count_mask = jnp.arange(size) < runtime.cold_emit_count
    x, y = _sample_ring(
        key,
        geometry.x_center,
        geometry.y_center,
        geometry.ion_radius_leave_min,
        geometry.ion_radius_leave_max,
        size,
    )
    vz = jnp.full((size,), jnp.sqrt(2.0 * 100.0 * 1.6021766208e-19 / particle_mass), dtype=jnp.float32)
    zeros = jnp.zeros((size,), dtype=jnp.float32)
    indices, free_valid = all_free_slots(pool.alive)
    valid = count_mask & free_valid
    pool = write_particles(pool, indices, valid, x, y, zeros, zeros, vz)
    return pool, valid.astype(jnp.int32).sum()


def remove_on_anode(pool: ParticlePool, geometry: GeometryState):
    outside = pool.alive & (~inside_anode(pool.x, pool.y, geometry))
    removed = outside.astype(jnp.int32).sum()
    return deactivate(pool, outside), removed


def remove_some_ions_on_cold_cathode(pool: ParticlePool, key, geometry: GeometryState, runtime: RuntimeState):
    candidates = pool.alive & inside_cold_cathode_ring(pool.x, pool.y, geometry)
    perm = jax.random.permutation(key, pool.alive.shape[0])
    shuffled_candidates = candidates[perm]
    chosen_in_perm = shuffled_candidates & (jnp.cumsum(shuffled_candidates.astype(jnp.int32)) <= runtime.ion_leave_count)
    kill_mask = jnp.zeros_like(pool.alive)
    kill_mask = kill_mask.at[perm].set(chosen_in_perm)
    removed = chosen_in_perm.astype(jnp.int32).sum()
    return deactivate(pool, kill_mask), removed


def update_counter(counters: CounterState, **kwargs):
    values = counters._asdict()
    values.update(kwargs)
    return CounterState(**values)
