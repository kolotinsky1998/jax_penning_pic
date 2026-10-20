from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_penning_pic.collisions import CrossSectionTable, ionization_probability
from jax_penning_pic.config import default_config
from jax_penning_pic.simulation_circle_gyro_new import step_once
from jax_penning_pic.state import (
    CounterState,
    FieldState,
    RuntimeState,
    SimState,
    empty_particle_pool,
)
from jax_penning_pic.triangular_fem import (
    EPSILON_0,
    TriMeshState,
    TriangularGeometryState,
    remove_outside_triangular_mesh,
    solve_triangular_poisson,
    triangular_charge_deposition,
    triangular_electric_field,
    triangular_mesh_contains,
    triangular_particle_location,
)


def _unit_square_mesh():
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    cells = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    areas = jnp.array([0.5, 0.5], dtype=jnp.float32)
    gradients = jnp.array(
        [[[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]], [[0.0, -1.0], [1.0, 0.0], [-1.0, 1.0]]],
        dtype=jnp.float32,
    )
    mass = jnp.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0], dtype=jnp.float32)
    stiffness_diag = jnp.zeros((4,), dtype=jnp.float32)
    return TriMeshState(
        points=points, cells=cells, areas=areas, grad_shape=gradients, lumped_mass=mass,
        boundary_mask=jnp.array([True, True, True, True]), stiffness_diag=stiffness_diag,
        lookup_origin=jnp.array([0.0, 0.0], dtype=jnp.float32), lookup_bin_size=jnp.array(1.0),
        lookup_candidates=jnp.array([[[0, 1, -1]]], dtype=jnp.int32),
    )


def _diamond_mesh():
    points = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=jnp.float32,
    )
    cells = jnp.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=jnp.int32)
    return TriMeshState(
        points=points,
        cells=cells,
        areas=jnp.full((4,), 0.5, dtype=jnp.float32),
        grad_shape=jnp.zeros((4, 3, 2), dtype=jnp.float32),
        lumped_mass=jnp.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        boundary_mask=jnp.array([False, True, True, True, True]),
        stiffness_diag=jnp.ones((5,), dtype=jnp.float32),
        lookup_origin=jnp.array([-1.0, -1.0], dtype=jnp.float32),
        lookup_bin_size=jnp.array(2.0, dtype=jnp.float32),
        lookup_candidates=jnp.array([[[0, 1, 2, 3, -1]]], dtype=jnp.int32),
    )


def test_p1_deposition_conserves_particle_charge():
    mesh = _unit_square_mesh()
    pool = empty_particle_pool(1, magnetic_field=0.0)
    pool = pool._replace(x=jnp.array([0.25]), y=jnp.array([0.25]), alive=jnp.array([True]))

    rho = triangular_charge_deposition(pool, mesh, particle_charge=2.5)

    assert float(jnp.sum(rho * mesh.lumped_mass)) == pytest.approx(2.5)


def test_p1_field_is_exact_for_linear_potential():
    mesh = _unit_square_mesh()
    # phi = 2 x - 3 y
    phi = 2.0 * mesh.points[:, 0] - 3.0 * mesh.points[:, 1]

    electric_field = triangular_electric_field(phi, mesh)

    assert bool(jnp.allclose(electric_field[:, 0], jnp.array([-2.0, -2.0])))
    assert bool(jnp.allclose(electric_field[:, 1], jnp.array([3.0, 3.0])))


@pytest.mark.parametrize("x,y", [(0.25, 0.25), (0.5, 0.5), (0.0, 0.0)])
def test_locator_accepts_triangle_interiors_shared_edges_and_vertices(x, y):
    mesh = _unit_square_mesh()
    _, bary, located = triangular_particle_location(jnp.array([x]), jnp.array([y]), mesh)

    assert bool(located[0])
    assert float(jnp.sum(bary[0])) == pytest.approx(1.0, abs=1.0e-5)


def test_lookup_upper_boundary_is_absorbing():
    mesh = _unit_square_mesh()
    _, _, located = triangular_particle_location(jnp.array([1.0]), jnp.array([1.0]), mesh)

    assert not bool(located[0])


def test_mesh_boundary_absorbs_particle_inside_analytic_circle_but_outside_polygon():
    mesh = _diamond_mesh()
    pool = empty_particle_pool(2, magnetic_field=0.0)._replace(
        x=jnp.array([0.2, 0.6]),
        y=jnp.array([0.2, 0.6]),
        alive=jnp.array([True, True]),
    )
    assert 0.6**2 + 0.6**2 < 1.0

    inside = triangular_mesh_contains(pool, mesh)
    updated, removed = remove_outside_triangular_mesh(pool, mesh)

    assert bool(inside[0])
    assert not bool(inside[1])
    assert list(map(bool, updated.alive)) == [True, False]
    assert int(removed) == 1


def test_absorbed_particle_no_longer_deposits_charge():
    mesh = _diamond_mesh()
    pool = empty_particle_pool(1, magnetic_field=0.0)._replace(
        x=jnp.array([0.6]), y=jnp.array([0.6]), alive=jnp.array([True])
    )

    pool, _ = remove_outside_triangular_mesh(pool, mesh)
    rho = triangular_charge_deposition(pool, mesh, particle_charge=2.5)

    assert float(jnp.sum(rho * mesh.lumped_mass)) == pytest.approx(0.0)


def test_absorbed_particle_cannot_ionize():
    mesh = _diamond_mesh()
    pool = empty_particle_pool(1, magnetic_field=0.0)._replace(
        x=jnp.array([0.6]),
        y=jnp.array([0.6]),
        vx=jnp.array([1.0e7]),
        alive=jnp.array([True]),
    )
    pool, _ = remove_outside_triangular_mesh(pool, mesh)
    table = CrossSectionTable(
        energy_ev=jnp.array([0.0, 1.0e6]), sigma_m2=jnp.array([1.0, 1.0])
    )

    probability = ionization_probability(pool, table, gas_density=1.0, dt=1.0, particle_mass=1.0)

    assert float(probability[0]) == pytest.approx(0.0)


def test_full_fem_step_absorbs_before_ionization():
    mesh = _unit_square_mesh()
    geometry = TriangularGeometryState(mesh, 0.5, 0.5, 0.5, 0.2, 0.05, 0.1)
    runtime = RuntimeState(1.0e-9, 1.0e30, 1.0e10, 0.0, 0.0, 0.0)
    electrons = empty_particle_pool(2, magnetic_field=1.0)._replace(
        x=jnp.array([1.1, 0.0]),
        y=jnp.array([0.5, 0.0]),
        vx=jnp.array([1.0e7, 0.0]),
        alive=jnp.array([True, False]),
    )
    ions = empty_particle_pool(2, magnetic_field=1.0)
    zeros = jnp.zeros((4,), dtype=jnp.float32)
    fields = FieldState(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(True),
    )
    counters = CounterState(*([jnp.array(0, dtype=jnp.int32)] * 7))
    state = SimState(
        electrons,
        ions,
        fields,
        geometry,
        runtime,
        counters,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jax.random.PRNGKey(1),
    )
    table = CrossSectionTable(
        jnp.array([0.0, 1.0e20], dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    config = replace(
        default_config(),
        mesh_mode="triangular_fem",
        mesh_path="unused",
        max_electrons=2,
        max_ions=2,
        poisson_iterations=5,
    )

    result = step_once(
        state,
        {"electron_elastic": table, "electron_ionization": table, "ion_elastic": table},
        config,
    )

    assert int(result.electrons.alive.sum()) == 0
    assert int(result.counters.ntot_mesh_boundary_leave_e) == 1
    assert int(result.counters.ntot_anode_leave) == 1
    assert int(result.counters.ntot_ionized) == 0


def _radial_disk_mesh(n_radial: int, n_theta: int) -> TriMeshState:
    points = [[0.0, 0.0]]
    for ring in range(1, n_radial + 1):
        radius = ring / n_radial
        for angle_index in range(n_theta):
            angle = 2.0 * np.pi * angle_index / n_theta
            points.append([radius * np.cos(angle), radius * np.sin(angle)])

    def node(ring, angle_index):
        return 1 + (ring - 1) * n_theta + angle_index % n_theta

    cells = []
    for angle_index in range(n_theta):
        cells.append([0, node(1, angle_index), node(1, angle_index + 1)])
    for ring in range(1, n_radial):
        for angle_index in range(n_theta):
            inner0 = node(ring, angle_index)
            inner1 = node(ring, angle_index + 1)
            outer0 = node(ring + 1, angle_index)
            outer1 = node(ring + 1, angle_index + 1)
            cells.extend(([inner0, outer0, outer1], [inner0, outer1, inner1]))

    points_np = np.asarray(points, dtype=np.float64)
    cells_np = np.asarray(cells, dtype=np.int32)
    vertices = points_np[cells_np]
    matrices = np.stack(
        (
            vertices[:, :, 0],
            vertices[:, :, 1],
            np.ones((vertices.shape[0], 3)),
        ),
        axis=1,
    )
    inverse = np.linalg.inv(matrices)
    gradients = inverse[:, :, :2]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    twice_area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    areas = 0.5 * twice_area
    mass = np.zeros((points_np.shape[0],), dtype=np.float64)
    stiffness_diag = np.zeros_like(mass)
    np.add.at(mass, cells_np, areas[:, None] / 3.0)
    np.add.at(stiffness_diag, cells_np, areas[:, None] * np.sum(gradients**2, axis=2))
    boundary = np.zeros_like(mass, dtype=bool)
    boundary[1 + (n_radial - 1) * n_theta :] = True
    return TriMeshState(
        points=jnp.asarray(points_np, dtype=jnp.float32),
        cells=jnp.asarray(cells_np, dtype=jnp.int32),
        areas=jnp.asarray(areas, dtype=jnp.float32),
        grad_shape=jnp.asarray(gradients, dtype=jnp.float32),
        lumped_mass=jnp.asarray(mass, dtype=jnp.float32),
        boundary_mask=jnp.asarray(boundary),
        stiffness_diag=jnp.asarray(stiffness_diag, dtype=jnp.float32),
        lookup_origin=jnp.array([-1.0, -1.0], dtype=jnp.float32),
        lookup_bin_size=jnp.array(2.0, dtype=jnp.float32),
        lookup_candidates=jnp.array([[[0]]], dtype=jnp.int32),
    )


def _manufactured_disk_errors(n_radial):
    mesh = _radial_disk_mesh(n_radial, 8 * n_radial)
    rho_value = 1.0e-12
    rho = jnp.full(mesh.points.shape[0], rho_value, dtype=jnp.float32)
    phi, info = solve_triangular_poisson(
        jnp.zeros_like(rho), rho, mesh, n_iter=2000, tol=1.0e-6
    )
    exact_phi = rho_value * (1.0 - jnp.sum(mesh.points**2, axis=1)) / (4.0 * EPSILON_0)
    interior = ~mesh.boundary_mask
    phi_error = jnp.linalg.norm((phi - exact_phi)[interior]) / jnp.linalg.norm(exact_phi[interior])
    field = triangular_electric_field(phi, mesh)
    centroids = jnp.mean(mesh.points[mesh.cells], axis=1)
    exact_field = rho_value * centroids / (2.0 * EPSILON_0)
    field_error = jnp.linalg.norm(field - exact_field) / jnp.linalg.norm(exact_field)
    assert bool(info.converged)
    return float(phi_error), float(field_error)


def test_circular_poisson_manufactured_solution_converges_under_refinement():
    coarse_phi, coarse_field = _manufactured_disk_errors(3)
    fine_phi, fine_field = _manufactured_disk_errors(6)

    assert fine_phi < coarse_phi
    assert fine_field < coarse_field
