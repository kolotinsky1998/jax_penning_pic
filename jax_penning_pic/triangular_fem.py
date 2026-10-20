"""JAX kernels and immutable data for a P1 triangular FEM mesh.

The mesh is deliberately prepared outside of JAX by
``scripts/generate_triangular_mesh.py``.  This module contains no Gmsh or
meshio dependency and is safe to import in the simulation process.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


EPSILON_0 = 8.854187817620389e-12


class TriMeshState(NamedTuple):
    points: jax.Array                 # (n_nodes, 2)
    cells: jax.Array                  # (n_cells, 3)
    areas: jax.Array                  # (n_cells,)
    grad_shape: jax.Array             # (n_cells, 3, 2)
    lumped_mass: jax.Array            # (n_nodes,)
    boundary_mask: jax.Array          # (n_nodes,)
    stiffness_diag: jax.Array         # (n_nodes,)
    lookup_origin: jax.Array          # (2,)
    lookup_bin_size: jax.Array        # scalar
    lookup_candidates: jax.Array      # (nx, ny, max_candidates), -1 padded


class TriangularGeometryState(NamedTuple):
    """Physical geometry plus the precomputed finite-element mesh."""

    mesh: TriMeshState
    x_center: float
    y_center: float
    radius_anode: float
    radius_injection: float
    ion_radius_leave_min: float
    ion_radius_leave_max: float


class PoissonInfo(NamedTuple):
    iterations: jax.Array
    relative_residual: jax.Array
    converged: jax.Array


def load_tri_mesh(path: str | Path) -> TriMeshState:
    """Load the compact, runtime-only representation produced by the generator."""

    data = np.load(Path(path), allow_pickle=False)
    required = {
        "points", "cells", "areas", "grad_shape", "lumped_mass",
        "boundary_mask", "stiffness_diag", "lookup_origin",
        "lookup_bin_size", "lookup_candidates",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"mesh archive is missing arrays: {sorted(missing)}")
    return TriMeshState(
        points=jnp.asarray(data["points"], dtype=jnp.float32),
        cells=jnp.asarray(data["cells"], dtype=jnp.int32),
        areas=jnp.asarray(data["areas"], dtype=jnp.float32),
        grad_shape=jnp.asarray(data["grad_shape"], dtype=jnp.float32),
        lumped_mass=jnp.asarray(data["lumped_mass"], dtype=jnp.float32),
        boundary_mask=jnp.asarray(data["boundary_mask"], dtype=bool),
        stiffness_diag=jnp.asarray(data["stiffness_diag"], dtype=jnp.float32),
        lookup_origin=jnp.asarray(data["lookup_origin"], dtype=jnp.float32),
        lookup_bin_size=jnp.asarray(data["lookup_bin_size"], dtype=jnp.float32),
        lookup_candidates=jnp.asarray(data["lookup_candidates"], dtype=jnp.int32),
    )


def triangular_particle_location(x, y, mesh: TriMeshState):
    """Return containing cells and P1 barycentric weights for particle positions."""

    # Array dimensions are static under JIT; do not rely on the metadata tuple
    # here, since it is a PyTree leaf when carried in simulation state.
    nx, ny = mesh.lookup_candidates.shape[:2]
    ix = jnp.floor((x - mesh.lookup_origin[0]) / mesh.lookup_bin_size).astype(jnp.int32)
    iy = jnp.floor((y - mesh.lookup_origin[1]) / mesh.lookup_bin_size).astype(jnp.int32)
    in_table = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix_safe = jnp.clip(ix, 0, nx - 1)
    iy_safe = jnp.clip(iy, 0, ny - 1)
    candidates = mesh.lookup_candidates[ix_safe, iy_safe]
    safe_candidates = jnp.maximum(candidates, 0)

    vertices = mesh.points[mesh.cells[safe_candidates]]  # (p, k, 3, 2)
    v0 = vertices[:, :, 0, :]
    e1 = vertices[:, :, 1, :] - v0
    e2 = vertices[:, :, 2, :] - v0
    det = e1[:, :, 0] * e2[:, :, 1] - e1[:, :, 1] * e2[:, :, 0]
    rel = jnp.stack((x, y), axis=-1)[:, None, :] - v0
    l1 = (rel[:, :, 0] * e2[:, :, 1] - rel[:, :, 1] * e2[:, :, 0]) / det
    l2 = (e1[:, :, 0] * rel[:, :, 1] - e1[:, :, 1] * rel[:, :, 0]) / det
    weights = jnp.stack((1.0 - l1 - l2, l1, l2), axis=-1)
    contains = (candidates >= 0) & jnp.all(weights >= -2.0e-5, axis=-1)
    contains = contains & jnp.all(weights <= 1.0 + 2.0e-5, axis=-1) & in_table[:, None]
    found = jnp.any(contains, axis=1)
    first = jnp.argmax(contains.astype(jnp.int32), axis=1)
    cell = jnp.take_along_axis(safe_candidates, first[:, None], axis=1)[:, 0]
    bary = jnp.take_along_axis(weights, first[:, None, None], axis=1)[:, 0, :]
    return cell, bary, found


def triangular_mesh_contains(pool, mesh: TriMeshState):
    """Return whether each particle position belongs to the FEM domain."""

    _, _, located = triangular_particle_location(pool.x, pool.y, mesh)
    return located


def remove_outside_triangular_mesh(pool, mesh: TriMeshState):
    """Absorb live particles that have crossed the actual P1 mesh boundary."""

    located = triangular_mesh_contains(pool, mesh)
    outside = pool.alive & ~located
    removed = outside.astype(jnp.int32).sum()
    return pool._replace(alive=pool.alive & located), removed


def triangular_charge_deposition(pool, mesh: TriMeshState, particle_charge: float):
    cells, bary, located = triangular_particle_location(pool.x, pool.y, mesh)
    valid = pool.alive & located
    contributions = valid[:, None].astype(jnp.float32) * bary * particle_charge
    nodes = mesh.cells[cells]
    charge = jnp.zeros_like(mesh.lumped_mass)
    charge = charge.at[nodes].add(contributions)
    return charge / mesh.lumped_mass


def triangular_electric_field(phi, mesh: TriMeshState):
    """P1 FEM electric field; it is constant in each triangle."""

    return -jnp.einsum("ti,tij->tj", phi[mesh.cells], mesh.grad_shape)


def triangular_field_gather(pool, field_cells, mesh: TriMeshState):
    cells, _, located = triangular_particle_location(pool.x, pool.y, mesh)
    field = field_cells[cells]
    valid = pool.alive & located
    ex = jnp.where(valid, field[:, 0], 0.0)
    ey = jnp.where(valid, field[:, 1], 0.0)
    return ex.astype(jnp.float32), ey.astype(jnp.float32)


def cell_field_to_nodes(field_cells, mesh: TriMeshState):
    """Mass-lumped projection used only for convenient nodal state/output."""

    weighted = field_cells * (mesh.areas / 3.0)[:, None]
    result = jnp.zeros((mesh.points.shape[0], 2), dtype=jnp.float32)
    result = result.at[mesh.cells].add(jnp.broadcast_to(weighted[:, None, :], (mesh.cells.shape[0], 3, 2)))
    return result / mesh.lumped_mass[:, None]


def _fem_matvec(x, mesh: TriMeshState):
    local = x[mesh.cells]
    local_out = EPSILON_0 * mesh.areas[:, None] * jnp.einsum(
        "tij,tj->ti", jnp.einsum("tik,tjk->tij", mesh.grad_shape, mesh.grad_shape), local
    )
    out = jnp.zeros_like(x).at[mesh.cells].add(local_out)
    return jnp.where(mesh.boundary_mask, x, out)


def solve_triangular_poisson(phi0, rho, mesh: TriMeshState, n_iter: int, tol: float):
    """Jacobi-preconditioned, matrix-free CG with homogeneous Dirichlet BC."""

    b = jnp.where(mesh.boundary_mask, 0.0, rho * mesh.lumped_mass)
    phi = jnp.where(mesh.boundary_mask, 0.0, phi0)
    diagonal = jnp.where(mesh.boundary_mask, 1.0, EPSILON_0 * mesh.stiffness_diag)
    r0 = b - _fem_matvec(phi, mesh)
    z0 = r0 / diagonal
    p0 = z0
    rz0 = jnp.vdot(r0, z0)
    b_norm = jnp.sqrt(jnp.vdot(b, b) + 1.0e-30)

    def cond(carry):
        _, r, _, _, iteration = carry
        relative = jnp.sqrt(jnp.vdot(r, r)) / b_norm
        return (iteration < n_iter) & (relative > tol)

    def body(carry):
        x, r, p, rz, iteration = carry
        ap = _fem_matvec(p, mesh)
        denom = jnp.vdot(p, ap)
        alpha = jnp.where(jnp.abs(denom) > 1.0e-30, rz / denom, 0.0)
        x_next = jnp.where(mesh.boundary_mask, 0.0, x + alpha * p)
        r_next = r - alpha * ap
        z_next = r_next / diagonal
        rz_next = jnp.vdot(r_next, z_next)
        beta = jnp.where(jnp.abs(rz) > 1.0e-30, rz_next / rz, 0.0)
        return x_next, r_next, z_next + beta * p, rz_next, iteration + 1

    phi, residual, _, _, iterations = jax.lax.while_loop(
        cond, body, (phi, r0, p0, rz0, jnp.array(0, dtype=jnp.int32))
    )
    relative = jnp.sqrt(jnp.vdot(residual, residual)) / b_norm
    return phi.astype(jnp.float32), PoissonInfo(iterations, relative, relative <= tol)
