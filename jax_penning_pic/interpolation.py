from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import GeometryState, ParticlePool


def _particle_cells(pool: ParticlePool, geometry: GeometryState, nx: int, ny: int):
    cell_x = jnp.floor(pool.x / geometry.dx).astype(jnp.int32)
    cell_y = jnp.floor(pool.y / geometry.dy).astype(jnp.int32)
    cell_x = jnp.clip(cell_x, 0, nx - 2)
    cell_y = jnp.clip(cell_y, 0, ny - 2)
    hx = (pool.x - cell_x * geometry.dx) / geometry.dx
    hy = (pool.y - cell_y * geometry.dy) / geometry.dy
    return cell_x, cell_y, hx, hy


def linear_charge_deposition(pool: ParticlePool, geometry: GeometryState, particle_charge: float, rho_template: jax.Array):
    rho = jnp.zeros_like(rho_template, dtype=jnp.float32)
    nx, ny = rho.shape
    cell_x, cell_y, hx, hy = _particle_cells(pool, geometry, nx, ny)
    alive = pool.alive.astype(jnp.float32)
    scale = particle_charge / (geometry.dx * geometry.dy)
    w00 = alive * (1.0 - hx) * (1.0 - hy) * scale
    w10 = alive * hx * (1.0 - hy) * scale
    w11 = alive * hx * hy * scale
    w01 = alive * (1.0 - hx) * hy * scale
    rho = rho.at[cell_x, cell_y].add(w00)
    rho = rho.at[cell_x + 1, cell_y].add(w10)
    rho = rho.at[cell_x + 1, cell_y + 1].add(w11)
    rho = rho.at[cell_x, cell_y + 1].add(w01)
    return rho


def linear_field_gather(pool: ParticlePool, ex_grid, ey_grid, geometry: GeometryState):
    nx, ny = ex_grid.shape
    cell_x, cell_y, hx, hy = _particle_cells(pool, geometry, nx, ny)
    ex = ex_grid[cell_x, cell_y] * (1.0 - hx) * (1.0 - hy)
    ex = ex + ex_grid[cell_x + 1, cell_y] * hx * (1.0 - hy)
    ex = ex + ex_grid[cell_x + 1, cell_y + 1] * hx * hy
    ex = ex + ex_grid[cell_x, cell_y + 1] * (1.0 - hx) * hy
    ey = ey_grid[cell_x, cell_y] * (1.0 - hx) * (1.0 - hy)
    ey = ey + ey_grid[cell_x + 1, cell_y] * hx * (1.0 - hy)
    ey = ey + ey_grid[cell_x + 1, cell_y + 1] * hx * hy
    ey = ey + ey_grid[cell_x, cell_y + 1] * (1.0 - hx) * hy
    ex = jnp.where(pool.alive, ex, 0.0)
    ey = jnp.where(pool.alive, ey, 0.0)
    return ex.astype(jnp.float32), ey.astype(jnp.float32)


def rho_filter_new(rho: jax.Array, nr_anode: int):
    nx, ny = rho.shape
    ii = jnp.arange(nx)[:, None]
    jj = jnp.arange(ny)[None, :]
    mask = (ii - nr_anode) ** 2 + (jj - nr_anode) ** 2 < nr_anode**2

    center = 4.0 * rho
    axis = 2.0 * (
        jnp.roll(rho, 1, axis=0)
        + jnp.roll(rho, -1, axis=0)
        + jnp.roll(rho, 1, axis=1)
        + jnp.roll(rho, -1, axis=1)
    )
    diag = (
        jnp.roll(jnp.roll(rho, 1, axis=0), 1, axis=1)
        + jnp.roll(jnp.roll(rho, 1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(rho, -1, axis=0), 1, axis=1)
        + jnp.roll(jnp.roll(rho, -1, axis=0), -1, axis=1)
    )
    rho_f1 = (center + axis + diag) / 16.0

    center2 = 20.0 * rho_f1
    axis2 = -1.0 * (
        jnp.roll(rho_f1, 1, axis=0)
        + jnp.roll(rho_f1, -1, axis=0)
        + jnp.roll(rho_f1, 1, axis=1)
        + jnp.roll(rho_f1, -1, axis=1)
    )
    diag2 = -1.0 * (
        jnp.roll(jnp.roll(rho_f1, 1, axis=0), 1, axis=1)
        + jnp.roll(jnp.roll(rho_f1, 1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(rho_f1, -1, axis=0), 1, axis=1)
        + jnp.roll(jnp.roll(rho_f1, -1, axis=0), -1, axis=1)
    )
    rho_f2 = (center2 + axis2 + diag2) / 12.0
    return jnp.where(mask, rho_f2, rho)
