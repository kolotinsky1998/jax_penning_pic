from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import EV, E_M, K_B, SimulationConfig, debye_radius


class ParticlePool(NamedTuple):
    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array
    ex: jax.Array
    ey: jax.Array
    bx: jax.Array
    by: jax.Array
    bz: jax.Array
    vx_c: jax.Array
    vy_c: jax.Array
    vz_c: jax.Array
    alive: jax.Array


class FieldState(NamedTuple):
    rho_e: jax.Array
    rho_i: jax.Array
    rho: jax.Array
    phi: jax.Array
    ex: jax.Array
    ey: jax.Array


class CounterState(NamedTuple):
    ntot_ionized: jax.Array
    ntot_cold_cathode_leave: jax.Array
    ntot_anode_leave: jax.Array
    ntot_hot_cathode_emission: jax.Array


class GeometryState(NamedTuple):
    nx: int
    ny: int
    dx: float
    dy: float
    x_center: float
    y_center: float
    radius_anode: float
    nr_anode: int
    radius_injection: float
    ion_radius_leave_min: float
    ion_radius_leave_max: float
    inside_mask: jax.Array


class RuntimeState(NamedTuple):
    dt: float
    dt_e: float
    dt_i: float
    ion_step: int
    n_gas_scaled: float
    ptcls_per_macro: float
    hot_emit_rate: float
    cold_emit_rate: float
    ion_leave_rate: float


class SimState(NamedTuple):
    electrons: ParticlePool
    ions: ParticlePool
    fields: FieldState
    geometry: GeometryState
    runtime: RuntimeState
    counters: CounterState
    hot_emit_remainder: jax.Array
    cold_emit_remainder: jax.Array
    ion_leave_remainder: jax.Array
    step: jax.Array
    rng_key: jax.Array


def empty_particle_pool(capacity: int, magnetic_field: float) -> ParticlePool:
    zeros = jnp.zeros((capacity,), dtype=jnp.float32)
    alive = jnp.zeros((capacity,), dtype=bool)
    bz = jnp.full((capacity,), magnetic_field, dtype=jnp.float32)
    return ParticlePool(
        x=zeros,
        y=zeros,
        vx=zeros,
        vy=zeros,
        vz=zeros,
        ex=zeros,
        ey=zeros,
        bx=zeros,
        by=zeros,
        bz=bz,
        vx_c=zeros,
        vy_c=zeros,
        vz_c=zeros,
        alive=alive,
    )


def build_runtime(config: SimulationConfig) -> tuple[GeometryState, RuntimeState]:
    scale = config.scale
    b_scaled = config.b / scale
    n_gas = config.p_pa / (K_B * config.t_gas)
    n_gas_scaled = n_gas / scale
    n_e_scaled = config.n_e / scale
    n_i_scaled = config.n_i / scale
    r_scaled = config.r * scale
    dt_e = 1.0 / (EV * b_scaled / E_M) / 10.0 * config.gyro_coeff
    dt_i = 1.0 / (EV * b_scaled / config.m_ion) / 10.0
    dt = dt_e
    ion_step = max(1, round(dt_i / dt_e))
    r_d = debye_radius(n_e_scaled, n_i_scaled, config.t_e_kelvin, config.t_i_kelvin) * 2.0
    if config.grid_size is not None and config.grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    nx = config.grid_size if config.grid_size is not None else int((r_scaled * 2.0) / r_d)
    ny = nx
    dx = (r_scaled * 2.0) / nx if config.grid_size is not None else r_d
    dy = dx
    radius_injection = 0.3 * nx * dx
    x_center = 0.5 * nx * dx
    y_center = 0.5 * ny * dy
    init_dens = n_e_scaled / 10.0
    ntot_seed = max(1, int(jnp.pi * (radius_injection / dx) ** 2 * config.ptcls_per_cell))
    ptcls_per_macro = float(init_dens * jnp.pi * radius_injection**2 / ntot_seed)
    radius_anode = (nx - 1) * dx / 2.0
    nr_anode = (nx - 1) // 2
    ion_radius_leave_min = 0.01 * scale
    ion_radius_leave_max = 0.15 * scale
    i_hot_scaled = config.i_hot * scale
    i_cold_scaled = config.i_cold * scale
    i_ion_leave = i_cold_scaled / (1.0 + config.gamma)
    ion_leave_rate = i_ion_leave * config.ion_leave_step * dt / EV / ptcls_per_macro
    cold_emit_rate = config.gamma * i_ion_leave * config.electron_emission_cold_step * dt / EV / ptcls_per_macro
    hot_emit_rate = i_hot_scaled * config.electron_emission_hot_step * dt / EV / ptcls_per_macro
    geometry = GeometryState(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        x_center=x_center,
        y_center=y_center,
        radius_anode=radius_anode,
        nr_anode=nr_anode,
        radius_injection=radius_injection,
        ion_radius_leave_min=ion_radius_leave_min,
        ion_radius_leave_max=ion_radius_leave_max,
        inside_mask=((jnp.arange(nx, dtype=jnp.float32)[:, None] * dx - x_center) ** 2
                     + (jnp.arange(ny, dtype=jnp.float32)[None, :] * dy - y_center) ** 2)
        < radius_anode**2,
    )
    runtime = RuntimeState(
        dt=dt,
        dt_e=dt_e,
        dt_i=dt_i,
        ion_step=ion_step,
        n_gas_scaled=n_gas_scaled,
        ptcls_per_macro=ptcls_per_macro,
        hot_emit_rate=hot_emit_rate,
        cold_emit_rate=cold_emit_rate,
        ion_leave_rate=ion_leave_rate,
    )
    return geometry, runtime


def create_initial_state(config: SimulationConfig) -> SimState:
    geometry, runtime = build_runtime(config)
    electrons = empty_particle_pool(config.max_electrons, config.b / config.scale)
    ions = empty_particle_pool(config.max_ions, config.b / config.scale)
    fields = FieldState(
        rho_e=jnp.zeros((geometry.nx, geometry.ny), dtype=jnp.float32),
        rho_i=jnp.zeros((geometry.nx, geometry.ny), dtype=jnp.float32),
        rho=jnp.zeros((geometry.nx, geometry.ny), dtype=jnp.float32),
        phi=jnp.zeros((geometry.nx, geometry.ny), dtype=jnp.float32),
        ex=jnp.zeros((geometry.nx, geometry.ny), dtype=jnp.float32),
        ey=jnp.zeros((geometry.nx, geometry.ny), dtype=jnp.float32),
    )
    counters = CounterState(
        ntot_ionized=jnp.array(0, dtype=jnp.int32),
        ntot_cold_cathode_leave=jnp.array(0, dtype=jnp.int32),
        ntot_anode_leave=jnp.array(0, dtype=jnp.int32),
        ntot_hot_cathode_emission=jnp.array(0, dtype=jnp.int32),
    )
    return SimState(
        electrons=electrons,
        ions=ions,
        fields=fields,
        geometry=geometry,
        runtime=runtime,
        counters=counters,
        hot_emit_remainder=jnp.array(0.0, dtype=jnp.float32),
        cold_emit_remainder=jnp.array(0.0, dtype=jnp.float32),
        ion_leave_remainder=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=jax.random.PRNGKey(config.seed),
    )
