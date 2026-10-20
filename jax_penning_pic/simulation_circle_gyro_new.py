from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np

from .collisions import (
    apply_electron_elastic,
    apply_ion_elastic_or_cx,
    apply_ionization,
    electron_elastic_probability,
    ion_elastic_probability,
    ionization_probability,
    load_cross_section,
)
from .config import E_M, EV, default_config, SimulationConfig
from .diagnostics import counters_row
from .interpolation import linear_charge_deposition, linear_field_gather, rho_filter_new
from .io import append_csv_row, prepare_output_dir, save_matrix_txt, write_json, write_metadata, write_triangular_vtk
from .gyro import gyro_push, gyro_update_velocity
from .particles import boris_push, boris_update_velocity
from .poisson import (
    build_direct_poisson_data,
    build_fft_capacitance_data,
    compute_electric_field,
    solve_poisson_cg,
    solve_poisson_direct,
    solve_poisson_fft_capacitance,
    solve_poisson_weighted_jacobi,
)
from .sources_sinks import (
    cold_secondary_emission,
    hot_cathode_emission,
    inject_initial_disk_samples,
    remove_on_anode,
    remove_some_ions_on_cold_cathode,
    sample_reference_initial_disk,
)
from .state import create_initial_state, create_triangular_initial_state
from .triangular_fem import (
    cell_field_to_nodes,
    remove_outside_triangular_mesh,
    solve_triangular_poisson,
    triangular_charge_deposition,
    triangular_electric_field,
    triangular_field_gather,
    triangular_mesh_contains,
    triangular_particle_location,
)


def _electron_charge(state):
    return -EV * state.runtime.ptcls_per_macro


def _ion_charge(state):
    return EV * state.runtime.ptcls_per_macro


def initialize_particles(state, config: SimulationConfig):
    speed_std_e = jnp.sqrt(2.0 * config.init_energy_ev * EV / (3.0 * E_M))
    speed_std_i = jnp.sqrt(2.0 * config.init_energy_ev * EV / (3.0 * config.m_ion))
    ntot_seed = max(
        1,
        int(jnp.pi * (state.geometry.radius_injection / state.geometry.dx) ** 2 * config.ptcls_per_cell),
    )
    key_init, new_key = jax.random.split(state.rng_key)
    x, y, velocity_factors = sample_reference_initial_disk(key_init, state.geometry, ntot_seed)
    electrons = inject_initial_disk_samples(state.electrons, x, y, velocity_factors, speed_std_e)
    ions = inject_initial_disk_samples(state.ions, x, y, velocity_factors, speed_std_i)
    electrons = gyro_update_velocity(electrons, -0.5 * state.runtime.dt, -EV, E_M)
    ions = boris_update_velocity(ions, -0.5 * state.runtime.dt, EV, config.m_ion)
    return state._replace(electrons=electrons, ions=ions, rng_key=new_key)


def _block_particles(pool):
    jax.block_until_ready(pool.x)
    jax.block_until_ready(pool.y)
    jax.block_until_ready(pool.vx)
    jax.block_until_ready(pool.vy)
    jax.block_until_ready(pool.vz)
    jax.block_until_ready(pool.alive)


def _block_fields(fields):
    jax.block_until_ready(fields.rho_e)
    jax.block_until_ready(fields.rho_i)
    jax.block_until_ready(fields.rho)
    jax.block_until_ready(fields.phi)
    jax.block_until_ready(fields.ex)
    jax.block_until_ready(fields.ey)


def _timed_stage(profile_row: dict, name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    if isinstance(result, tuple):
        for item in result:
            if hasattr(item, "block_until_ready"):
                jax.block_until_ready(item)
    elif hasattr(result, "block_until_ready"):
        jax.block_until_ready(result)
    profile_row[name] = time.perf_counter() - t0
    return result


def _advance_event_count(rate: float, remainder):
    target = jnp.asarray(rate, dtype=jnp.float32) + remainder
    count = jnp.floor(target).astype(jnp.int32)
    return count, target - count.astype(jnp.float32)


def _event_count(do_event, rate: float, remainder):
    return jax.lax.cond(
        do_event,
        lambda rem: _advance_event_count(rate, rem),
        lambda rem: (jnp.array(0, dtype=jnp.int32), rem),
        remainder,
    )


def _profile_event_count(rate: float, remainder):
    target = float(remainder) + rate
    count = int(math.floor(target))
    return count, jnp.array(target - count, dtype=jnp.float32)


def _electron_collisions_kernel(electrons, ions, counters, rng_key, runtime, tables, config: SimulationConfig):
    key1, key2, new_key = jax.random.split(rng_key, 3)
    prob_el = electron_elastic_probability(
        electrons,
        tables["electron_elastic"],
        runtime.n_gas_scaled,
        config.collision_step_electron * runtime.dt,
        E_M,
    )
    prob_ion = ionization_probability(
        electrons,
        tables["electron_ionization"],
        runtime.n_gas_scaled,
        config.collision_step_electron * runtime.dt,
        E_M,
    )
    rand = jax.random.uniform(key1, prob_el.shape, dtype=jnp.float32)
    elastic_mask = rand < prob_el
    ionization_mask = (rand >= prob_el) & (rand < prob_el + prob_ion)
    electrons = apply_electron_elastic(electrons, elastic_mask, key2, config.m_ion, E_M)
    electrons, ions, created = apply_ionization(electrons, ions, ionization_mask, E_M)
    counters = counters._replace(ntot_ionized=counters.ntot_ionized + created)
    return electrons, ions, counters, new_key


def _ion_collisions_kernel(ions, rng_key, runtime, tables, config: SimulationConfig):
    key, new_key = jax.random.split(rng_key)
    prob_i = ion_elastic_probability(
        ions,
        tables["ion_elastic"],
        runtime.n_gas_scaled,
        config.collision_step_ion * runtime.dt,
        config.m_ion,
    )
    rand_i = jax.random.uniform(key, prob_i.shape, dtype=jnp.float32)
    ions = apply_ion_elastic_or_cx(ions, rand_i < prob_i, key, config.m_ion, config.t_gas)
    return ions, new_key


def _cold_sink_kernel(ions, counters, rng_key, geometry, count):
    ions, removed_i = remove_some_ions_on_cold_cathode(ions, rng_key, geometry, count)
    counters = counters._replace(ntot_cold_cathode_leave=counters.ntot_cold_cathode_leave + removed_i)
    return ions, counters


def _hot_emission_kernel(electrons, counters, rng_key, geometry, count):
    electrons, emitted_hot = hot_cathode_emission(electrons, rng_key, geometry, count, E_M)
    counters = counters._replace(ntot_hot_cathode_emission=counters.ntot_hot_cathode_emission + emitted_hot)
    return electrons, counters


def _cold_secondary_kernel(electrons, counters, rng_key, geometry, count):
    electrons, emitted_cold = cold_secondary_emission(electrons, rng_key, geometry, count, E_M)
    counters = counters._replace(
        ntot_cold_cathode_emission=counters.ntot_cold_cathode_emission + emitted_cold
    )
    return electrons, counters


def _solve_poisson(rho, state, tables, config: SimulationConfig):
    if config.mesh_mode == "triangular_fem":
        phi, info = solve_triangular_poisson(
            state.fields.phi, rho, state.geometry.mesh, config.poisson_iterations, config.poisson_tol
        )
        return phi, (info.iterations, info.relative_residual, info.converged)
    if config.poisson_solver == "direct_inverse":
        phi = solve_poisson_direct(rho, tables["poisson_direct"])
        return phi, (jnp.array(0, dtype=jnp.int32), jnp.array(0.0), jnp.array(True))
    if config.poisson_solver == "fft_capacitance":
        phi = solve_poisson_fft_capacitance(rho, tables["poisson_fft_capacitance"])
        return phi, (jnp.array(0, dtype=jnp.int32), jnp.array(0.0), jnp.array(True))
    if config.poisson_solver == "cg":
        phi = solve_poisson_cg(state.fields.phi, rho, state.geometry, config.poisson_iterations, config.poisson_tol)
        return phi, (jnp.array(0, dtype=jnp.int32), jnp.array(0.0), jnp.array(True))
    phi = solve_poisson_weighted_jacobi(
        state.fields.phi, rho, state.geometry, config.poisson_omega, config.poisson_iterations,
    )
    return phi, (jnp.array(0, dtype=jnp.int32), jnp.array(0.0), jnp.array(True))


def _step_kernel_impl(state, tables, config: SimulationConfig):
    if config.mesh_mode == "triangular_fem":
        mesh = state.geometry.mesh
        with jax.named_scope("deposit"):
            rho_e = triangular_charge_deposition(state.electrons, mesh, _electron_charge(state))
            rho_i = triangular_charge_deposition(state.ions, mesh, _ion_charge(state))
        rho = rho_e + rho_i
        with jax.named_scope("poisson"):
            phi, poisson_info = _solve_poisson(rho, state, tables, config)
        with jax.named_scope("field_from_phi"):
            field_cells = triangular_electric_field(phi, mesh)
            field_nodes = cell_field_to_nodes(field_cells, mesh)
            ex_grid, ey_grid = field_nodes[:, 0], field_nodes[:, 1]
        with jax.named_scope("gather"):
            ex_e, ey_e = triangular_field_gather(state.electrons, field_cells, mesh)
            ex_i, ey_i = triangular_field_gather(state.ions, field_cells, mesh)
    else:
        with jax.named_scope("deposit"):
            rho_e = linear_charge_deposition(state.electrons, state.geometry, _electron_charge(state), state.fields.rho_e)
            rho_i = linear_charge_deposition(state.ions, state.geometry, _ion_charge(state), state.fields.rho_i)
        with jax.named_scope("rho_filter"):
            rho = rho_e + rho_i
            if config.use_rho_filter:
                rho = rho_filter_new(rho, state.geometry.nr_anode)
        with jax.named_scope("poisson"):
            phi, poisson_info = _solve_poisson(rho, state, tables, config)
        with jax.named_scope("field_from_phi"):
            ex_grid, ey_grid = compute_electric_field(phi, state.geometry)
        with jax.named_scope("gather"):
            ex_e, ey_e = linear_field_gather(state.electrons, ex_grid, ey_grid, state.geometry)
            ex_i, ey_i = linear_field_gather(state.ions, ex_grid, ey_grid, state.geometry)
    electrons = state.electrons._replace(ex=ex_e, ey=ey_e)
    ions = state.ions._replace(ex=ex_i, ey=ey_i)

    with jax.named_scope("push_electrons"):
        electrons = gyro_push(electrons, state.runtime.dt, -EV, E_M)
    with jax.named_scope("push_ions"):
        ions = boris_push(ions, state.runtime.dt, EV, config.m_ion)

    counters = state.counters
    rng_key = state.rng_key
    hot_emit_remainder = state.hot_emit_remainder
    cold_emit_remainder = state.cold_emit_remainder
    ion_leave_remainder = state.ion_leave_remainder

    # The P1 polygon is the FEM computational domain and its absorbing anode.
    # Absorb immediately after the push so escaped particles cannot collide or
    # ionize while carrying zero gathered field outside the mesh.
    if config.mesh_mode == "triangular_fem":
        with jax.named_scope("mesh_boundary_loss"):
            electrons, removed_mesh_e = remove_outside_triangular_mesh(electrons, state.geometry.mesh)
            ions, removed_mesh_i = remove_outside_triangular_mesh(ions, state.geometry.mesh)
            counters = counters._replace(
                ntot_anode_leave=counters.ntot_anode_leave + removed_mesh_e,
                ntot_mesh_boundary_leave_e=counters.ntot_mesh_boundary_leave_e + removed_mesh_e,
                ntot_mesh_boundary_leave_i=counters.ntot_mesh_boundary_leave_i + removed_mesh_i,
            )

    do_electron_collisions = jnp.equal(jnp.mod(state.step, config.collision_step_electron), 0)
    with jax.named_scope("electron_collisions"):
        electrons, ions, counters, rng_key = jax.lax.cond(
            do_electron_collisions,
            lambda args: _electron_collisions_kernel(args[0], args[1], args[2], args[3], state.runtime, tables, config),
            lambda args: args,
            (electrons, ions, counters, rng_key),
        )

    do_ion_collisions = jnp.equal(jnp.mod(state.step, config.collision_step_ion), 0)
    with jax.named_scope("ion_collisions"):
        ions, rng_key = jax.lax.cond(
            do_ion_collisions,
            lambda args: _ion_collisions_kernel(args[0], args[1], state.runtime, tables, config),
            lambda args: args,
            (ions, rng_key),
        )

    if config.mesh_mode == "uniform_cartesian":
        with jax.named_scope("anode_loss"):
            electrons, removed_e = remove_on_anode(electrons, state.geometry)
            counters = counters._replace(ntot_anode_leave=counters.ntot_anode_leave + removed_e)
            ions, _ = remove_on_anode(ions, state.geometry)

    do_cold_sink = jnp.equal(jnp.mod(state.step, config.ion_leave_step), 0)
    ion_leave_count, ion_leave_remainder = _event_count(
        do_cold_sink,
        state.runtime.ion_leave_rate,
        ion_leave_remainder,
    )
    rng_key, sink_key = jax.random.split(rng_key)
    with jax.named_scope("cold_cathode_sink"):
        ions, counters = jax.lax.cond(
            do_cold_sink,
            lambda args: _cold_sink_kernel(*args),
            lambda args: args[:2],
            (ions, counters, sink_key, state.geometry, ion_leave_count),
        )

    do_hot_emit = jnp.equal(jnp.mod(state.step, config.electron_emission_hot_step), 0)
    hot_emit_count, hot_emit_remainder = _event_count(
        do_hot_emit,
        state.runtime.hot_emit_rate,
        hot_emit_remainder,
    )
    rng_key, hot_key = jax.random.split(rng_key)
    with jax.named_scope("hot_cathode_emission"):
        electrons, counters = jax.lax.cond(
            do_hot_emit,
            lambda args: _hot_emission_kernel(*args),
            lambda args: args[:2],
            (electrons, counters, hot_key, state.geometry, hot_emit_count),
        )

    do_cold_emit = jnp.equal(jnp.mod(state.step, config.electron_emission_cold_step), 0)
    cold_emit_count, cold_emit_remainder = _event_count(
        do_cold_emit,
        state.runtime.cold_emit_rate,
        cold_emit_remainder,
    )
    rng_key, cold_key = jax.random.split(rng_key)
    with jax.named_scope("cold_secondary_emission"):
        electrons, counters = jax.lax.cond(
            do_cold_emit,
            lambda args: _cold_secondary_kernel(*args),
            lambda args: args[:2],
            (electrons, counters, cold_key, state.geometry, cold_emit_count),
        )

    return state._replace(
        electrons=electrons,
        ions=ions,
        fields=state.fields._replace(
            rho_e=rho_e, rho_i=rho_i, rho=rho, phi=phi, ex=ex_grid, ey=ey_grid,
            poisson_iterations=poisson_info[0], poisson_relative_residual=poisson_info[1],
            poisson_converged=poisson_info[2],
        ),
        counters=counters,
        hot_emit_remainder=hot_emit_remainder,
        cold_emit_remainder=cold_emit_remainder,
        ion_leave_remainder=ion_leave_remainder,
        step=state.step + 1,
        rng_key=rng_key,
    )


step_once_jitted = jax.jit(_step_kernel_impl, static_argnames=("config",))


def _run_steps_block(state, tables, config: SimulationConfig, num_steps: int):
    def body(_, carry):
        return step_once_jitted(carry, tables, config)

    return jax.lax.fori_loop(0, num_steps, body, state)


run_steps_block_jitted = jax.jit(_run_steps_block, static_argnames=("config", "num_steps"))


def step_once(state, tables, config: SimulationConfig):
    return step_once_jitted(state, tables, config)


def step_once_profiled(state, tables, config: SimulationConfig):
    profile_row = {"step": int(state.step)}

    # The FEM backend has one fused JAX kernel.  Keep profiling functional
    # without duplicating the full branch-specific PIC implementation.
    if config.mesh_mode == "triangular_fem":
        t0 = time.perf_counter()
        state = step_once_jitted(state, tables, config)
        _block_particles(state.electrons)
        _block_particles(state.ions)
        _block_fields(state.fields)
        profile_row["step_total_s"] = time.perf_counter() - t0
        return state, profile_row

    rho_e = _timed_stage(
        profile_row,
        "deposit_electrons_s",
        lambda: linear_charge_deposition(state.electrons, state.geometry, _electron_charge(state), state.fields.rho_e),
    )
    rho_i = _timed_stage(
        profile_row,
        "deposit_ions_s",
        lambda: linear_charge_deposition(state.ions, state.geometry, _ion_charge(state), state.fields.rho_i),
    )
    rho = rho_e + rho_i
    if config.use_rho_filter:
        rho = _timed_stage(profile_row, "rho_filter_s", lambda: rho_filter_new(rho, state.geometry.nr_anode))
    else:
        profile_row["rho_filter_s"] = 0.0
    phi, poisson_info = _timed_stage(
        profile_row,
        "poisson_s",
        lambda: _solve_poisson(rho, state, tables, config),
    )
    ex_grid, ey_grid = _timed_stage(profile_row, "field_from_phi_s", lambda: compute_electric_field(phi, state.geometry))
    ex_e, ey_e = _timed_stage(
        profile_row,
        "gather_electrons_s",
        lambda: linear_field_gather(state.electrons, ex_grid, ey_grid, state.geometry),
    )
    ex_i, ey_i = _timed_stage(
        profile_row,
        "gather_ions_s",
        lambda: linear_field_gather(state.ions, ex_grid, ey_grid, state.geometry),
    )
    electrons = state.electrons._replace(ex=ex_e, ey=ey_e)
    ions = state.ions._replace(ex=ex_i, ey=ey_i)

    electrons = _timed_stage(profile_row, "push_electrons_s", lambda: gyro_push(electrons, state.runtime.dt, -EV, E_M))
    ions = _timed_stage(
        profile_row,
        "push_ions_s",
        lambda: boris_push(ions, state.runtime.dt, EV, config.m_ion),
    )

    counters = state.counters
    rng_key = state.rng_key
    hot_emit_remainder = state.hot_emit_remainder
    cold_emit_remainder = state.cold_emit_remainder
    ion_leave_remainder = state.ion_leave_remainder
    if int(state.step) % config.collision_step_electron == 0:
        key1, key2, new_key = jax.random.split(rng_key, 3)
        prob_el = _timed_stage(
            profile_row,
            "electron_collision_probability_s",
            lambda: electron_elastic_probability(
                electrons, tables["electron_elastic"], state.runtime.n_gas_scaled, config.collision_step_electron * state.runtime.dt, E_M
            ),
        )
        prob_ion = _timed_stage(
            profile_row,
            "ionization_probability_s",
            lambda: ionization_probability(
                electrons, tables["electron_ionization"], state.runtime.n_gas_scaled, config.collision_step_electron * state.runtime.dt, E_M
            ),
        )
        rand = _timed_stage(profile_row, "electron_collision_rng_s", lambda: jax.random.uniform(key1, prob_el.shape, dtype=jnp.float32))
        elastic_mask = rand < prob_el
        ionization_mask = (rand >= prob_el) & (rand < prob_el + prob_ion)
        electrons = _timed_stage(
            profile_row,
            "electron_elastic_collision_s",
            lambda: apply_electron_elastic(electrons, elastic_mask, key2, config.m_ion, E_M),
        )
        electrons, ions, created = _timed_stage(
            profile_row,
            "electron_ionization_s",
            lambda: apply_ionization(electrons, ions, ionization_mask, E_M),
        )
        counters = counters._replace(ntot_ionized=counters.ntot_ionized + created)
        rng_key = new_key
    else:
        profile_row["electron_collision_probability_s"] = 0.0
        profile_row["ionization_probability_s"] = 0.0
        profile_row["electron_collision_rng_s"] = 0.0
        profile_row["electron_elastic_collision_s"] = 0.0
        profile_row["electron_ionization_s"] = 0.0

    if int(state.step) % config.collision_step_ion == 0:
        key4, rng_key = jax.random.split(rng_key)
        prob_i = _timed_stage(
            profile_row,
            "ion_collision_probability_s",
            lambda: ion_elastic_probability(
                ions, tables["ion_elastic"], state.runtime.n_gas_scaled, config.collision_step_ion * state.runtime.dt, config.m_ion
            ),
        )
        rand_i = _timed_stage(profile_row, "ion_collision_rng_s", lambda: jax.random.uniform(key4, prob_i.shape, dtype=jnp.float32))
        ions = _timed_stage(
            profile_row,
            "ion_collision_apply_s",
            lambda: apply_ion_elastic_or_cx(ions, rand_i < prob_i, key4, config.m_ion, config.t_gas),
        )
    else:
        profile_row["ion_collision_probability_s"] = 0.0
        profile_row["ion_collision_rng_s"] = 0.0
        profile_row["ion_collision_apply_s"] = 0.0

    electrons, removed_e = _timed_stage(profile_row, "anode_loss_s", lambda: remove_on_anode(electrons, state.geometry))
    counters = counters._replace(ntot_anode_leave=counters.ntot_anode_leave + removed_e)
    ions, _ = remove_on_anode(ions, state.geometry)

    if int(state.step) % config.ion_leave_step == 0:
        key5, rng_key = jax.random.split(rng_key)
        ion_leave_count, ion_leave_remainder = _profile_event_count(
            state.runtime.ion_leave_rate,
            ion_leave_remainder,
        )
        ions, removed_i = _timed_stage(
            profile_row,
            "cold_cathode_ion_sink_s",
            lambda: remove_some_ions_on_cold_cathode(ions, key5, state.geometry, ion_leave_count),
        )
        counters = counters._replace(ntot_cold_cathode_leave=counters.ntot_cold_cathode_leave + removed_i)
    else:
        profile_row["cold_cathode_ion_sink_s"] = 0.0

    if int(state.step) % config.electron_emission_hot_step == 0:
        key6, rng_key = jax.random.split(rng_key)
        hot_emit_count, hot_emit_remainder = _profile_event_count(
            state.runtime.hot_emit_rate,
            hot_emit_remainder,
        )
        electrons, emitted_hot = _timed_stage(
            profile_row,
            "hot_cathode_emission_s",
            lambda: hot_cathode_emission(electrons, key6, state.geometry, hot_emit_count, E_M),
        )
        counters = counters._replace(ntot_hot_cathode_emission=counters.ntot_hot_cathode_emission + emitted_hot)
    else:
        profile_row["hot_cathode_emission_s"] = 0.0

    if int(state.step) % config.electron_emission_cold_step == 0:
        key7, rng_key = jax.random.split(rng_key)
        cold_emit_count, cold_emit_remainder = _profile_event_count(
            state.runtime.cold_emit_rate,
            cold_emit_remainder,
        )
        electrons, emitted_cold = _timed_stage(
            profile_row,
            "cold_secondary_emission_s",
            lambda: cold_secondary_emission(electrons, key7, state.geometry, cold_emit_count, E_M),
        )
        counters = counters._replace(
            ntot_cold_cathode_emission=counters.ntot_cold_cathode_emission + emitted_cold
        )
    else:
        profile_row["cold_secondary_emission_s"] = 0.0

    state = state._replace(
        electrons=electrons,
        ions=ions,
        fields=state.fields._replace(
            rho_e=rho_e, rho_i=rho_i, rho=rho, phi=phi, ex=ex_grid, ey=ey_grid,
            poisson_iterations=poisson_info[0], poisson_relative_residual=poisson_info[1],
            poisson_converged=poisson_info[2],
        ),
        counters=counters,
        hot_emit_remainder=hot_emit_remainder,
        cold_emit_remainder=cold_emit_remainder,
        ion_leave_remainder=ion_leave_remainder,
        step=state.step + 1,
        rng_key=rng_key,
    )
    _block_particles(state.electrons)
    _block_particles(state.ions)
    _block_fields(state.fields)
    profile_row["step_total_s"] = sum(
        value for key, value in profile_row.items() if key.endswith("_s") and key != "step_total_s"
    )
    return state, profile_row


def _compute_block_size(config: SimulationConfig) -> int:
    intervals = [config.log_interval, config.field_dump_interval]
    intervals = [int(v) for v in intervals if int(v) > 0]
    if not intervals:
        return max(1, config.it_num)
    block = intervals[0]
    for interval in intervals[1:]:
        block = math.gcd(block, interval)
    return max(1, block)


def _mesh_derived_metadata(state) -> dict:
    mesh = state.geometry.mesh
    points = np.asarray(mesh.points)
    cells = np.asarray(mesh.cells)
    areas = np.asarray(mesh.areas)
    edges = np.concatenate((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    boundary_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = boundary_edges[counts == 1]
    boundary_lengths = np.linalg.norm(
        points[boundary_edges[:, 1]] - points[boundary_edges[:, 0]], axis=1
    )
    area = float(areas.sum())
    return {
        "n_nodes": int(points.shape[0]),
        "n_triangles": int(cells.shape[0]),
        "area": area,
        "area_min": float(areas.min()),
        "area_max": float(areas.max()),
        "effective_radius": math.sqrt(area / math.pi),
        "max_boundary_edge_length": float(boundary_lengths.max()),
    }


def _derived_metadata(state, config: SimulationConfig) -> dict:
    derived = {
        "dt": float(state.runtime.dt),
        "macro_particle_weight": float(state.runtime.ptcls_per_macro),
    }
    if config.mesh_mode == "triangular_fem":
        derived["mesh"] = _mesh_derived_metadata(state)
    return derived


def _validate_triangular_mesh_geometry(state, config: SimulationConfig):
    if config.mesh_mode != "triangular_fem":
        return
    mesh = state.geometry.mesh
    points = np.asarray(mesh.points)
    boundary = np.asarray(mesh.boundary_mask, dtype=bool)
    if not boundary.any():
        raise ValueError("triangular FEM mesh has no boundary nodes")
    mesh_info = _mesh_derived_metadata(state)
    center = np.array([state.geometry.x_center, state.geometry.y_center], dtype=np.float64)
    boundary_radius = np.linalg.norm(points[boundary] - center, axis=1)
    expected_radius = float(state.geometry.radius_anode)
    tolerance = max(1.0e-7, 1.1 * mesh_info["max_boundary_edge_length"])
    if np.max(np.abs(boundary_radius - expected_radius)) > tolerance:
        raise ValueError(
            "triangular mesh boundary is incompatible with the configured anode: "
            f"expected radius {expected_radius:.8g}, observed "
            f"[{boundary_radius.min():.8g}, {boundary_radius.max():.8g}]"
        )

    # Emission positions must be inside the computational domain by
    # construction; otherwise they would violate the alive/located invariant
    # before the next field deposition.
    angles = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
    for label, radius in (
        ("hot cathode", state.geometry.ion_radius_leave_min),
        ("cold cathode", state.geometry.ion_radius_leave_max),
    ):
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        _, _, located = triangular_particle_location(jnp.asarray(x), jnp.asarray(y), mesh)
        if not bool(jnp.all(located)):
            raise ValueError(f"{label} emission region is not fully contained in the triangular mesh")


def _validate_triangular_particle_domain(state, config: SimulationConfig):
    if config.mesh_mode != "triangular_fem" or not config.validate_particle_domain:
        return
    mesh = state.geometry.mesh
    electrons_located = triangular_mesh_contains(state.electrons, mesh)
    ions_located = triangular_mesh_contains(state.ions, mesh)
    invalid_e = int(jnp.sum(state.electrons.alive & ~electrons_located))
    invalid_i = int(jnp.sum(state.ions.alive & ~ions_located))
    if invalid_e or invalid_i:
        raise RuntimeError(
            f"FEM particle-domain invariant failed at step {int(state.step)}: "
            f"{invalid_e} electrons and {invalid_i} ions are alive outside the mesh"
        )


def _dump_fields(out_dir: Path, state, config: SimulationConfig):
    step = int(state.step)
    if config.mesh_mode == "triangular_fem":
        mesh = state.geometry.mesh
        write_triangular_vtk(
            out_dir / f"fields_{step}.vtu",
            mesh.points,
            mesh.cells,
            rho_e=state.fields.rho_e,
            rho_i=state.fields.rho_i,
            rho=state.fields.rho,
            phi=state.fields.phi,
            ex=state.fields.ex,
            ey=state.fields.ey,
            elementary_charge=EV,
        )
        return
    save_matrix_txt(out_dir / f"rho_e_{step}.txt", state.fields.rho_e)
    save_matrix_txt(out_dir / f"rho_i_{step}.txt", state.fields.rho_i)
    save_matrix_txt(out_dir / f"phi_{step}.txt", state.fields.phi)


def _record_poisson_status(out_dir: Path, state, config: SimulationConfig):
    if config.mesh_mode != "triangular_fem":
        return
    append_csv_row(
        out_dir / "poisson.csv",
        {
            "step": int(state.step),
            "iterations": int(state.fields.poisson_iterations),
            "relative_residual": float(state.fields.poisson_relative_residual),
            "converged": bool(state.fields.poisson_converged),
        },
    )
    if not bool(state.fields.poisson_converged):
        raise RuntimeError(
            "triangular FEM CG did not converge; increase poisson_iterations or adjust poisson_tol"
        )


def run_simulation(
    config: SimulationConfig | None = None,
    profile: bool = False,
    profile_summary_only: bool = False,
    trace_dir: str | None = None,
    trace_start_step: int = 0,
    trace_num_steps: int = 0,
    overwrite_output: bool = False,
):
    config = config or default_config()
    out_dir = prepare_output_dir(config.output_dir, overwrite=overwrite_output)
    if config.mesh_mode not in {"uniform_cartesian", "triangular_fem"}:
        raise ValueError("mesh_mode must be 'uniform_cartesian' or 'triangular_fem'")
    state = create_triangular_initial_state(config) if config.mesh_mode == "triangular_fem" else create_initial_state(config)
    if config.mesh_mode == "uniform_cartesian":
        state = initialize_particles(state, config)
    _validate_triangular_mesh_geometry(state, config)
    write_metadata(
        out_dir / "metadata.json",
        {
            "output_schema_version": 2,
            "config": asdict(config),
            "derived": _derived_metadata(state, config),
            "profiling_enabled": profile,
            "trace_dir": trace_dir,
            "trace_start_step": trace_start_step,
            "trace_num_steps": trace_num_steps,
            "execution_mode": "profiled_python_loop" if profile else "jit_block_loop",
        },
    )
    tables = {
        "electron_elastic": load_cross_section(config.cross_sections.electron_elastic),
        "electron_ionization": load_cross_section(config.cross_sections.electron_ionization),
        "ion_elastic": load_cross_section(config.cross_sections.ion_elastic),
    }
    # Triangular FEM always uses its matrix-free PCG implementation and must
    # not initialize data structures for the legacy Cartesian solvers.
    if config.mesh_mode == "uniform_cartesian":
        if config.poisson_solver == "direct_inverse":
            tables["poisson_direct"] = build_direct_poisson_data(state.geometry)
        elif config.poisson_solver == "fft_capacitance":
            tables["poisson_fft_capacitance"] = build_fft_capacitance_data(state.geometry)

    append_csv_row(out_dir / "counters.csv", counters_row(0, state.electrons, state.ions, state.geometry, state.counters))
    _dump_fields(out_dir, state, config)
    _record_poisson_status(out_dir, state, config)
    _validate_triangular_particle_domain(state, config)

    if profile:
        profile_rows = []
        for it in range(config.it_num):
            io_t0 = time.perf_counter()
            state, profile_row = step_once_profiled(state, tables, config)
            io_before = time.perf_counter()
            if (it + 1) % config.log_interval == 0:
                _validate_triangular_particle_domain(state, config)
                append_csv_row(
                    out_dir / "counters.csv",
                    counters_row(int(state.step), state.electrons, state.ions, state.geometry, state.counters),
                )
            if (it + 1) % config.field_dump_interval == 0:
                _dump_fields(out_dir, state, config)
                _record_poisson_status(out_dir, state, config)
            io_after = time.perf_counter()
            profile_row["io_s"] = io_after - io_before
            profile_row["wall_total_with_io_s"] = io_after - io_t0
            if not profile_summary_only:
                profile_rows.append(profile_row)

        profile_path = out_dir / "profiling.csv"
        summary_path = out_dir / "profiling_summary.json"
        if profile_rows:
            for row in profile_rows:
                append_csv_row(profile_path, row)
            keys = [key for key in profile_rows[0].keys() if key.endswith("_s")]
            summary = {
                key: {
                    "mean_s": sum(row[key] for row in profile_rows) / len(profile_rows),
                    "max_s": max(row[key] for row in profile_rows),
                    "min_s": min(row[key] for row in profile_rows),
                    "sum_s": sum(row[key] for row in profile_rows),
                }
                for key in keys
            }
            write_json(summary_path, summary)
        else:
            write_json(summary_path, {"note": "Profiling enabled, but per-step rows were not stored."})
        return state

    block_size = _compute_block_size(config)
    steps_done = 0
    trace_path = Path(trace_dir) if trace_dir else None
    trace_active = False
    trace_end_step = trace_start_step + trace_num_steps if trace_num_steps > 0 else None
    while steps_done < config.it_num:
        if trace_path is not None and not trace_active and steps_done >= trace_start_step:
            trace_path.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(trace_path))
            trace_active = True
        num_steps = min(block_size, config.it_num - steps_done)
        if trace_active:
            with jax.profiler.StepTraceAnnotation("sim_block", step_num=steps_done):
                state = run_steps_block_jitted(state, tables, config, num_steps)
        else:
            state = run_steps_block_jitted(state, tables, config, num_steps)
        _block_particles(state.electrons)
        _block_particles(state.ions)
        _block_fields(state.fields)
        steps_done += num_steps
        if trace_active and trace_end_step is not None and steps_done >= trace_end_step:
            jax.profiler.stop_trace()
            trace_active = False
            trace_path = None
        current_step = int(state.step)
        if current_step % config.log_interval == 0:
            _validate_triangular_particle_domain(state, config)
            append_csv_row(out_dir / "counters.csv", counters_row(current_step, state.electrons, state.ions, state.geometry, state.counters))
        if current_step % config.field_dump_interval == 0:
            _dump_fields(out_dir, state, config)
            _record_poisson_status(out_dir, state, config)

    if trace_active:
        jax.profiler.stop_trace()

    return state
