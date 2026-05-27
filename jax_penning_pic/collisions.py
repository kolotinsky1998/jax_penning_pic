from __future__ import annotations

from pathlib import Path
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import EV
from .particles import all_free_slots, first_free_slots, write_particles
from .state import ParticlePool

K_B = 1.380649e-23
C = 299792458.0
E_M = 9.10938356e-31


class CrossSectionTable(NamedTuple):
    energy_ev: jax.Array
    sigma_m2: jax.Array


def load_cross_section(path: str | Path) -> CrossSectionTable:
    data = np.loadtxt(path)
    return CrossSectionTable(
        energy_ev=jnp.asarray(data[:, 0], dtype=jnp.float32),
        sigma_m2=jnp.asarray(data[:, 1], dtype=jnp.float32),
    )


def cross_section_value(table: CrossSectionTable, energy_joule):
    energy_ev = energy_joule / EV
    return jnp.interp(energy_ev, table.energy_ev, table.sigma_m2, left=table.sigma_m2[0], right=table.sigma_m2[-1])


def velocity_norm(vx, vy, vz):
    return jnp.sqrt(vx * vx + vy * vy + vz * vz + 1.0e-30)


def electron_elastic_probability(pool: ParticlePool, table: CrossSectionTable, gas_density: float, dt: float, particle_mass: float):
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    energy_ev = energy / EV
    p = jnp.sqrt(2.0 * energy * E_M) / (E_M * C)
    z_argon = 18.0
    nu = 1.7e-5 * (0.556 - 0.0825 * jnp.log(jnp.maximum(energy / (E_M * C * C), 1.0e-30))) * (
        z_argon ** (2.0 / 3.0)
    ) / jnp.maximum(p * p, 1.0e-30)
    classical_electron_radius = 2.817940326727e-15
    beta = jnp.sqrt(jnp.maximum(2.0 * energy / E_M, 0.0)) / C
    sigma_theory = (
        z_argon
        * z_argon
        * jnp.pi
        * classical_electron_radius
        * classical_electron_radius
        / jnp.maximum(beta * beta * p * p * nu * (nu + 1.0), 1.0e-30)
    )
    sigma = jnp.where(energy_ev > 100.0, sigma_theory, cross_section_value(table, energy))
    return jnp.where(pool.alive, sigma * gas_density * speed * dt, 0.0)


def ion_elastic_probability(pool: ParticlePool, table: CrossSectionTable, gas_density: float, dt: float, particle_mass: float):
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    sigma = cross_section_value(table, energy)
    return jnp.where(pool.alive, sigma * gas_density * dt, 0.0)


def ionization_probability(pool: ParticlePool, table: CrossSectionTable, gas_density: float, dt: float, particle_mass: float):
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    sigma = cross_section_value(table, energy)
    return jnp.where(pool.alive, sigma * gas_density * speed * dt, 0.0)


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _scatter_like_reference(vx, vy, vz, new_speed, theta, phi, axis_x, axis_y, axis_z):
    speed = velocity_norm(vx, vy, vz)
    ux = vx / speed
    uy = vy / speed
    uz = vz / speed
    dot = jnp.clip(ux * axis_x + uy * axis_y + uz * axis_z, -1.0, 1.0)
    sin_alpha = jnp.sqrt(jnp.maximum(1.0 - dot * dot, 0.0))
    safe = sin_alpha > 1.0e-12

    vel1_x = ux * jnp.cos(theta)
    vel1_y = uy * jnp.cos(theta)
    vel1_z = uz * jnp.cos(theta)

    cross_ui_x, cross_ui_y, cross_ui_z = _cross(ux, uy, uz, axis_x, axis_y, axis_z)
    cross_iu_x, cross_iu_y, cross_iu_z = _cross(axis_x, axis_y, axis_z, ux, uy, uz)
    cross_u_i_u_x, cross_u_i_u_y, cross_u_i_u_z = _cross(ux, uy, uz, cross_iu_x, cross_iu_y, cross_iu_z)

    sin_alpha_safe = jnp.where(safe, sin_alpha, 1.0)
    vel2_scale = jnp.where(safe, jnp.sin(theta) * jnp.sin(phi) / sin_alpha_safe, 0.0)
    vel3_scale = jnp.where(safe, jnp.sin(theta) * jnp.cos(phi) / sin_alpha_safe, 0.0)

    out_x = (vel1_x + cross_ui_x * vel2_scale + cross_u_i_u_x * vel3_scale) * new_speed
    out_y = (vel1_y + cross_ui_y * vel2_scale + cross_u_i_u_y * vel3_scale) * new_speed
    out_z = (vel1_z + cross_ui_z * vel2_scale + cross_u_i_u_z * vel3_scale) * new_speed
    return out_x, out_y, out_z


def apply_electron_elastic(pool: ParticlePool, collision_mask: jax.Array, key, gas_mass: float, particle_mass: float):
    key_phi, key_theta_low, key_theta_high = jax.random.split(key, 3)
    phi = 2.0 * jnp.pi * jax.random.uniform(key_phi, pool.vx.shape, dtype=jnp.float32)
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    energy_ev = energy / EV
    rand_low = jax.random.uniform(key_theta_low, pool.vx.shape, dtype=jnp.float32)
    theta_low = jnp.where(
        energy_ev > 0.0,
        jnp.arccos(
            jnp.clip(
                (2.0 + energy_ev - 2.0 * jnp.power(1.0 + energy_ev, rand_low)) / jnp.maximum(energy_ev, 1.0e-6),
                -1.0,
                1.0,
            )
        ),
        0.0,
    )
    rand_high = jax.random.uniform(key_theta_high, pool.vx.shape, dtype=jnp.float32)
    p = jnp.sqrt(2.0 * energy * E_M) / (E_M * C)
    z_argon = 18.0
    nu = 1.7e-5 * (0.556 - 0.0825 * jnp.log(jnp.maximum(energy / (E_M * C * C), 1.0e-30))) * (
        z_argon ** (2.0 / 3.0)
    ) / jnp.maximum(p * p, 1.0e-30)
    theta_high = jnp.arccos(jnp.clip(1.0 - 2.0 * nu * rand_high / (1.0 + nu - rand_high), -1.0, 1.0))
    high_energy = energy_ev > 100.0
    theta = jnp.where(high_energy, theta_high, theta_low)
    factor = jnp.sqrt(jnp.maximum(1.0 - 2.0 * particle_mass / gas_mass * (1.0 - jnp.cos(theta)), 0.0))
    new_speed = speed * factor
    low_vx, low_vy, low_vz = _scatter_like_reference(pool.vx, pool.vy, pool.vz, new_speed, theta, phi, 1.0, 0.0, 0.0)
    high_vx, high_vy, high_vz = _scatter_like_reference(pool.vx, pool.vy, pool.vz, new_speed, theta, phi, 0.0, 0.0, 1.0)
    new_vx = jnp.where(high_energy, high_vx, low_vx)
    new_vy = jnp.where(high_energy, high_vy, low_vy)
    new_vz = jnp.where(high_energy, high_vz, low_vz)
    vx = jnp.where(collision_mask, new_vx, pool.vx)
    vy = jnp.where(collision_mask, new_vy, pool.vy)
    vz = jnp.where(collision_mask, new_vz, pool.vz)
    vx_c = jnp.where(collision_mask, new_vx, pool.vx_c)
    vy_c = jnp.where(collision_mask, new_vy, pool.vy_c)
    vz_c = jnp.where(collision_mask, new_vz, pool.vz_c)
    return pool._replace(vx=vx, vy=vy, vz=vz, vx_c=vx_c, vy_c=vy_c, vz_c=vz_c)


def apply_ion_elastic_or_cx(pool: ParticlePool, collision_mask: jax.Array, key, gas_mass: float, gas_temperature: float, charge_exchange: bool = True):
    key_branch, key_vel = jax.random.split(key)
    sigma = jnp.sqrt(K_B * gas_temperature / gas_mass)
    gas_vel = jax.random.normal(key_vel, (pool.vx.shape[0], 3), dtype=jnp.float32) * sigma
    branch = jax.random.uniform(key_branch, pool.vx.shape, dtype=jnp.float32)
    elastic_mask = collision_mask & ((branch <= 0.5) if charge_exchange else jnp.ones_like(branch, dtype=bool))
    cx_mask = collision_mask & (~elastic_mask)
    new_vx_el = (pool.vx * gas_mass + gas_vel[:, 0] * gas_mass + gas_mass * (gas_vel[:, 0] - pool.vx)) / (2.0 * gas_mass)
    new_vy_el = (pool.vy * gas_mass + gas_vel[:, 1] * gas_mass + gas_mass * (gas_vel[:, 1] - pool.vy)) / (2.0 * gas_mass)
    new_vz_el = (pool.vz * gas_mass + gas_vel[:, 2] * gas_mass + gas_mass * (gas_vel[:, 2] - pool.vz)) / (2.0 * gas_mass)
    vx = jnp.where(elastic_mask, new_vx_el, pool.vx)
    vy = jnp.where(elastic_mask, new_vy_el, pool.vy)
    vz = jnp.where(elastic_mask, new_vz_el, pool.vz)
    vx = jnp.where(cx_mask, gas_vel[:, 0], vx)
    vy = jnp.where(cx_mask, gas_vel[:, 1], vy)
    vz = jnp.where(cx_mask, gas_vel[:, 2], vz)
    return pool._replace(vx=vx, vy=vy, vz=vz)


def apply_ionization(electrons: ParticlePool, ions: ParticlePool, ionization_mask: jax.Array, particle_mass_e: float):
    speed = velocity_norm(electrons.vx, electrons.vy, electrons.vz)
    energy = 0.5 * particle_mass_e * speed * speed
    ion_threshold = 16.0 * EV
    new_energy = energy - ion_threshold
    valid = ionization_mask & (new_energy > 0.0)
    new_speed = jnp.sqrt(jnp.maximum(2.0 * new_energy / particle_mass_e, 0.0))
    scale = new_speed / speed
    vx = jnp.where(valid, electrons.vx * scale, electrons.vx)
    vy = jnp.where(valid, electrons.vy * scale, electrons.vy)
    vz = jnp.where(valid, electrons.vz * scale, electrons.vz)
    vx_c = jnp.where(valid, vx, electrons.vx_c)
    vy_c = jnp.where(valid, vy, electrons.vy_c)
    vz_c = jnp.where(valid, vz, electrons.vz_c)
    electrons = electrons._replace(vx=vx, vy=vy, vz=vz, vx_c=vx_c, vy_c=vy_c, vz_c=vz_c)
    source_idx = jnp.where(valid, size=electrons.alive.shape[0], fill_value=-1)[0]
    source_valid = source_idx >= 0
    ex_idx, ex_valid = all_free_slots(electrons.alive)
    ion_idx, ion_valid = all_free_slots(ions.alive)
    src_safe = jnp.where(source_valid, source_idx, 0)
    pair_valid = source_valid & ex_valid & ion_valid
    x_new = electrons.x[src_safe]
    y_new = electrons.y[src_safe]
    zeros = jnp.zeros_like(x_new, dtype=jnp.float32)
    electrons = write_particles(electrons, ex_idx, pair_valid, x_new, y_new, zeros, zeros, zeros)
    ions = write_particles(ions, ion_idx, pair_valid, x_new, y_new, zeros, zeros, zeros)
    created = pair_valid.astype(jnp.int32).sum()
    return electrons, ions, created
