import jax.numpy as jnp
import pytest

from jax_penning_pic.state import empty_particle_pool
from jax_penning_pic.triangular_fem import (
    TriMeshState,
    triangular_charge_deposition,
    triangular_electric_field,
)


def _unit_square_mesh():
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    cells = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    areas = jnp.array([0.5, 0.5], dtype=jnp.float32)
    gradients = jnp.array(
        [[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], [[0.0, -1.0], [1.0, 0.0], [-1.0, 1.0]]],
        dtype=jnp.float32,
    )
    mass = jnp.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0], dtype=jnp.float32)
    stiffness_diag = jnp.zeros((4,), dtype=jnp.float32)
    return TriMeshState(
        points=points, cells=cells, areas=areas, grad_shape=gradients, lumped_mass=mass,
        boundary_mask=jnp.array([True, True, True, True]), stiffness_diag=stiffness_diag,
        lookup_origin=jnp.array([0.0, 0.0], dtype=jnp.float32), lookup_bin_size=jnp.array(1.0),
        lookup_candidates=jnp.array([[[0, 1]]], dtype=jnp.int32),
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
