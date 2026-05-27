from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import ParticlePool


def boris_update_velocity(pool: ParticlePool, dt: float, charge: float, mass: float) -> ParticlePool:
    q_over_m = charge / mass
    tx = q_over_m * pool.bx * 0.5 * dt
    ty = q_over_m * pool.by * 0.5 * dt
    tz = q_over_m * pool.bz * 0.5 * dt
    t_mag2 = tx * tx + ty * ty + tz * tz
    sx = 2.0 * tx / (1.0 + t_mag2)
    sy = 2.0 * ty / (1.0 + t_mag2)
    sz = 2.0 * tz / (1.0 + t_mag2)

    v_minus_x = pool.vx + q_over_m * pool.ex * 0.5 * dt
    v_minus_y = pool.vy + q_over_m * pool.ey * 0.5 * dt
    v_minus_z = pool.vz

    v_prime_x = v_minus_x + (v_minus_y * tz - v_minus_z * ty)
    v_prime_y = v_minus_y + (v_minus_z * tx - v_minus_x * tz)
    v_prime_z = v_minus_z + (v_minus_x * ty - v_minus_y * tx)

    v_plus_x = v_minus_x + (v_prime_y * sz - v_prime_z * sy)
    v_plus_y = v_minus_y + (v_prime_z * sx - v_prime_x * sz)
    v_plus_z = v_minus_z + (v_prime_x * sy - v_prime_y * sx)

    vx = v_plus_x + q_over_m * pool.ex * 0.5 * dt
    vy = v_plus_y + q_over_m * pool.ey * 0.5 * dt
    vz = v_plus_z
    vx = jnp.where(pool.alive, vx, pool.vx)
    vy = jnp.where(pool.alive, vy, pool.vy)
    vz = jnp.where(pool.alive, vz, pool.vz)
    return pool._replace(vx=vx, vy=vy, vz=vz, vx_c=vx, vy_c=vy, vz_c=vz)


def boris_push(pool: ParticlePool, dt: float, charge: float, mass: float) -> ParticlePool:
    pool = boris_update_velocity(pool, dt, charge, mass)
    x = pool.x + pool.vx * dt
    y = pool.y + pool.vy * dt
    x = jnp.where(pool.alive, x, pool.x)
    y = jnp.where(pool.alive, y, pool.y)
    return pool._replace(x=x, y=y)


def first_free_slots(alive: jax.Array, count: int):
    indices = jnp.where(~alive, size=alive.shape[0], fill_value=-1)[0]
    chosen = indices[:count]
    valid = chosen >= 0
    return chosen, valid


def all_free_slots(alive: jax.Array):
    indices = jnp.where(~alive, size=alive.shape[0], fill_value=-1)[0]
    valid = indices >= 0
    return indices, valid


def write_particles(pool: ParticlePool, indices, valid, x, y, vx, vy, vz):
    drop_idx = pool.x.shape[0]
    safe_idx = jnp.where(valid, indices, drop_idx)
    x_old, y_old = pool.x, pool.y
    vx_old, vy_old, vz_old = pool.vx, pool.vy, pool.vz
    alive_old = pool.alive
    x_new = x_old.at[safe_idx].set(x, mode="drop")
    y_new = y_old.at[safe_idx].set(y, mode="drop")
    vx_new = vx_old.at[safe_idx].set(vx, mode="drop")
    vy_new = vy_old.at[safe_idx].set(vy, mode="drop")
    vz_new = vz_old.at[safe_idx].set(vz, mode="drop")
    ex_new = pool.ex
    ey_new = pool.ey
    bx_new = pool.bx
    by_new = pool.by
    bz_new = pool.bz
    vx_c_new = pool.vx_c.at[safe_idx].set(vx, mode="drop")
    vy_c_new = pool.vy_c.at[safe_idx].set(vy, mode="drop")
    vz_c_new = pool.vz_c.at[safe_idx].set(vz, mode="drop")
    alive_new = alive_old.at[safe_idx].set(True, mode="drop")
    return pool._replace(
        x=x_new,
        y=y_new,
        vx=vx_new,
        vy=vy_new,
        vz=vz_new,
        ex=ex_new,
        ey=ey_new,
        bx=bx_new,
        by=by_new,
        bz=bz_new,
        vx_c=vx_c_new,
        vy_c=vy_c_new,
        vz_c=vz_c_new,
        alive=alive_new,
    )


def deactivate(pool: ParticlePool, mask: jax.Array):
    return pool._replace(alive=pool.alive & ~mask)
