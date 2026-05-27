from __future__ import annotations

import jax.numpy as jnp

from .state import ParticlePool


def gyro_update_velocity(pool: ParticlePool, dt: float, charge: float, mass: float) -> ParticlePool:
    b_norm2 = pool.bx * pool.bx + pool.by * pool.by + pool.bz * pool.bz
    b_norm2 = jnp.where(b_norm2 > 0.0, b_norm2, 1.0)
    exb_x = pool.ey * pool.bz / b_norm2
    exb_y = -pool.ex * pool.bz / b_norm2
    exb_z = (pool.ex * pool.by - pool.ey * pool.bx) / b_norm2

    vx = pool.vx - exb_x
    vy = pool.vy - exb_y
    vz = pool.vz - exb_z

    v_dot_b = vx * pool.bx + vy * pool.by + vz * pool.bz
    e_dot_b = pool.ex * pool.bx + pool.ey * pool.by

    v_par_x = pool.bx * v_dot_b / b_norm2
    v_par_y = pool.by * v_dot_b / b_norm2
    v_par_z = pool.bz * v_dot_b / b_norm2

    e_par_x = pool.bx * e_dot_b / b_norm2
    e_par_y = pool.by * e_dot_b / b_norm2
    e_par_z = pool.bz * e_dot_b / b_norm2

    q_over_m = charge / mass
    v_par_x = v_par_x + q_over_m * e_par_x * dt
    v_par_y = v_par_y + q_over_m * e_par_y * dt
    v_par_z = v_par_z + q_over_m * e_par_z * dt

    v_perp_x = vx - v_par_x
    v_perp_y = vy - v_par_y
    v_perp_z = vz - v_par_z
    v_perp_norm = jnp.sqrt(v_perp_x * v_perp_x + v_perp_y * v_perp_y + v_perp_z * v_perp_z + 1.0e-30)
    phi = jnp.arccos(jnp.clip(v_perp_x / v_perp_norm, -1.0, 1.0)) + jnp.abs(q_over_m) * jnp.sqrt(b_norm2) * dt
    v_perp_x = v_perp_norm * jnp.cos(phi)
    v_perp_y = v_perp_norm * jnp.sin(phi)
    v_perp_z = jnp.zeros_like(v_perp_x)

    vx_c = v_par_x + exb_x
    vy_c = v_par_y + exb_y
    vz_c = v_par_z + exb_z
    vx = vx_c + v_perp_x
    vy = vy_c + v_perp_y
    vz = vz_c + v_perp_z

    vx = jnp.where(pool.alive, vx, pool.vx)
    vy = jnp.where(pool.alive, vy, pool.vy)
    vz = jnp.where(pool.alive, vz, pool.vz)
    vx_c = jnp.where(pool.alive, vx_c, pool.vx_c)
    vy_c = jnp.where(pool.alive, vy_c, pool.vy_c)
    vz_c = jnp.where(pool.alive, vz_c, pool.vz_c)
    return pool._replace(vx=vx, vy=vy, vz=vz, vx_c=vx_c, vy_c=vy_c, vz_c=vz_c)


def gyro_push(pool: ParticlePool, dt: float, charge: float, mass: float) -> ParticlePool:
    pool = gyro_update_velocity(pool, dt, charge, mass)
    x = jnp.where(pool.alive, pool.x + pool.vx_c * dt, pool.x)
    y = jnp.where(pool.alive, pool.y + pool.vy_c * dt, pool.y)
    return pool._replace(x=x, y=y)
