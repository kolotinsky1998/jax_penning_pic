from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import circular_domain_mask
from .state import GeometryState

EPSILON_0 = 8.854187817620389e-12


class DirectPoissonData(NamedTuple):
    interior_indices: jax.Array
    inverse_operator: jax.Array


class FFTCapacitanceData(NamedTuple):
    green_hat: jax.Array
    boundary_indices: jax.Array
    correction_operator: jax.Array


def _mask_inside(arr, inside_mask):
    return jnp.where(inside_mask, arr, 0.0)


def _poisson_operator(phi, inside_mask, dx, dy):
    phi = _mask_inside(phi, inside_mask)
    dx2 = dx * dx
    dy2 = dy * dy
    center = (2.0 / dx2 + 2.0 / dy2) * phi
    neighbors = (
        (jnp.roll(phi, 1, axis=0) + jnp.roll(phi, -1, axis=0)) / dx2
        + (jnp.roll(phi, 1, axis=1) + jnp.roll(phi, -1, axis=1)) / dy2
    )
    return _mask_inside(center - neighbors, inside_mask)


def _masked_dot(a, b, inside_mask):
    return jnp.sum(jnp.where(inside_mask, a * b, 0.0))


def weighted_jacobi_step(phi, rho, inside_mask, dx, dy, omega):
    dx2 = dx * dx
    dy2 = dy * dy
    laplace_rhs = (
        rho / EPSILON_0
        + (jnp.roll(phi, 1, axis=0) + jnp.roll(phi, -1, axis=0)) / dx2
        + (jnp.roll(phi, 1, axis=1) + jnp.roll(phi, -1, axis=1)) / dy2
    ) / (2.0 / dx2 + 2.0 / dy2)
    updated = (1.0 - omega) * phi + omega * laplace_rhs
    return jnp.where(inside_mask, updated, 0.0)


def solve_poisson_weighted_jacobi(phi0, rho, geometry: GeometryState, omega: float, n_iter: int):
    inside_mask = circular_domain_mask(geometry)

    def body(_, phi):
        return weighted_jacobi_step(phi, rho, inside_mask, geometry.dx, geometry.dy, omega)

    return jax.lax.fori_loop(0, n_iter, body, phi0)


def solve_poisson_cg(phi0, rho, geometry: GeometryState, n_iter: int, tol: float):
    inside_mask = circular_domain_mask(geometry)
    phi = _mask_inside(phi0, inside_mask)
    b = _mask_inside(rho / EPSILON_0, inside_mask)
    diag = jnp.where(
        inside_mask,
        2.0 / (geometry.dx * geometry.dx) + 2.0 / (geometry.dy * geometry.dy),
        1.0,
    )

    r0 = b - _poisson_operator(phi, inside_mask, geometry.dx, geometry.dy)
    z0 = _mask_inside(r0 / diag, inside_mask)
    p0 = z0
    rz0 = _masked_dot(r0, z0, inside_mask)
    b_norm = jnp.sqrt(_masked_dot(b, b, inside_mask) + 1.0e-30)

    def cond_fn(carry):
        _, r, _, _, k = carry
        residual = jnp.sqrt(_masked_dot(r, r, inside_mask)) / b_norm
        return (k < n_iter) & (residual > tol)

    def body_fn(carry):
        phi_k, r_k, p_k, rz_k, k = carry
        ap_k = _poisson_operator(p_k, inside_mask, geometry.dx, geometry.dy)
        denom = _masked_dot(p_k, ap_k, inside_mask)
        alpha = jnp.where(jnp.abs(denom) > 1.0e-30, rz_k / denom, 0.0)
        phi_next = _mask_inside(phi_k + alpha * p_k, inside_mask)
        r_next = _mask_inside(r_k - alpha * ap_k, inside_mask)
        z_next = _mask_inside(r_next / diag, inside_mask)
        rz_next = _masked_dot(r_next, z_next, inside_mask)
        beta = jnp.where(jnp.abs(rz_k) > 1.0e-30, rz_next / rz_k, 0.0)
        p_next = _mask_inside(z_next + beta * p_k, inside_mask)
        return phi_next, r_next, p_next, rz_next, k + 1

    phi, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, (phi, r0, p0, rz0, jnp.array(0, dtype=jnp.int32)))
    return _mask_inside(phi, inside_mask)


def build_direct_poisson_data(geometry: GeometryState) -> DirectPoissonData:
    inside_mask = np.asarray(circular_domain_mask(geometry))
    nx, ny = inside_mask.shape
    interior_indices = np.flatnonzero(inside_mask.reshape(-1))
    n_unknowns = interior_indices.size
    flat_to_local = -np.ones(nx * ny, dtype=np.int32)
    flat_to_local[interior_indices] = np.arange(n_unknowns, dtype=np.int32)

    dx2 = geometry.dx * geometry.dx
    dy2 = geometry.dy * geometry.dy
    center = 2.0 / dx2 + 2.0 / dy2
    x_coeff = -1.0 / dx2
    y_coeff = -1.0 / dy2

    operator = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
    for local_idx, flat_idx in enumerate(interior_indices):
        i = flat_idx // ny
        j = flat_idx % ny
        operator[local_idx, local_idx] = center
        for ni, nj, coeff in ((i - 1, j, x_coeff), (i + 1, j, x_coeff), (i, j - 1, y_coeff), (i, j + 1, y_coeff)):
            if 0 <= ni < nx and 0 <= nj < ny and inside_mask[ni, nj]:
                neighbor_flat = ni * ny + nj
                operator[local_idx, flat_to_local[neighbor_flat]] = coeff

    inverse_operator = np.linalg.inv(operator).astype(np.float32)
    return DirectPoissonData(
        interior_indices=jnp.asarray(interior_indices, dtype=jnp.int32),
        inverse_operator=jnp.asarray(inverse_operator, dtype=jnp.float32),
    )


def _build_boundary_mask(inside_mask: np.ndarray) -> np.ndarray:
    boundary_mask = np.zeros_like(inside_mask, dtype=bool)
    nx, ny = inside_mask.shape
    for i in range(nx):
        for j in range(ny):
            if not inside_mask[i, j]:
                continue
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or not inside_mask[ni, nj]:
                    boundary_mask[i, j] = True
                    break
    return boundary_mask


def _build_fft_green_hat(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    lam = (
        (2.0 - 2.0 * np.cos(kx))[:, None] / (dx * dx)
        + (2.0 - 2.0 * np.cos(ky))[None, :] / (dy * dy)
    )
    green_hat = np.zeros((nx, ny), dtype=np.complex64)
    nonzero = lam > 0.0
    green_hat[nonzero] = (1.0 / lam[nonzero]).astype(np.float32)
    green_hat[0, 0] = 0.0
    return green_hat


def build_fft_capacitance_data(geometry: GeometryState) -> FFTCapacitanceData:
    inside_mask = np.asarray(circular_domain_mask(geometry))
    nx, ny = inside_mask.shape
    boundary_mask = _build_boundary_mask(inside_mask)
    boundary_indices = np.flatnonzero(boundary_mask.reshape(-1))
    n_boundary = boundary_indices.size

    green_hat = _build_fft_green_hat(nx, ny, geometry.dx, geometry.dy)
    response = np.zeros((n_boundary, n_boundary), dtype=np.float64)

    for col, flat_idx in enumerate(boundary_indices):
        delta = np.zeros((nx, ny), dtype=np.float32)
        delta.reshape(-1)[flat_idx] = 1.0
        phi = np.fft.ifft2(np.fft.fft2(delta / EPSILON_0) * green_hat).real.astype(np.float32)
        response[:, col] = phi.reshape(-1)[boundary_indices]

    reg = 1.0e-6 * max(1.0, float(np.max(np.abs(np.diag(response)))))
    correction_operator = (-np.linalg.inv(response + reg * np.eye(n_boundary, dtype=np.float64))).astype(np.float32)
    return FFTCapacitanceData(
        green_hat=jnp.asarray(green_hat, dtype=jnp.complex64),
        boundary_indices=jnp.asarray(boundary_indices, dtype=jnp.int32),
        correction_operator=jnp.asarray(correction_operator, dtype=jnp.float32),
    )


def solve_poisson_direct(rho: jax.Array, direct_data: DirectPoissonData):
    rhs = jnp.take((rho / EPSILON_0).reshape(-1), direct_data.interior_indices)
    phi_interior = direct_data.inverse_operator @ rhs
    phi_flat = jnp.zeros((rho.size,), dtype=jnp.float32)
    phi_flat = phi_flat.at[direct_data.interior_indices].set(phi_interior)
    return phi_flat.reshape(rho.shape)


def solve_poisson_fft_capacitance(rho: jax.Array, fft_data: FFTCapacitanceData):
    rhs = rho / EPSILON_0
    phi_raw = jnp.fft.ifft2(jnp.fft.fft2(rhs) * fft_data.green_hat).real.astype(jnp.float32)
    b_corr = jnp.take(phi_raw.reshape(-1), fft_data.boundary_indices)
    f_corr = fft_data.correction_operator @ b_corr
    rho_corr_flat = jnp.zeros((rho.size,), dtype=jnp.float32)
    rho_corr_flat = rho_corr_flat.at[fft_data.boundary_indices].set(f_corr)
    rho_final = rho + rho_corr_flat.reshape(rho.shape)
    return jnp.fft.ifft2(jnp.fft.fft2(rho_final / EPSILON_0) * fft_data.green_hat).real.astype(jnp.float32)


def compute_electric_field(phi, geometry: GeometryState):
    ex = (jnp.roll(phi, 1, axis=0) - jnp.roll(phi, -1, axis=0)) / (2.0 * geometry.dx)
    ey = (jnp.roll(phi, 1, axis=1) - jnp.roll(phi, -1, axis=1)) / (2.0 * geometry.dy)
    ex = ex.at[0, :].set((phi[0, :] - phi[1, :]) / geometry.dx)
    ex = ex.at[-1, :].set((phi[-2, :] - phi[-1, :]) / geometry.dx)
    ey = ey.at[:, 0].set((phi[:, 0] - phi[:, 1]) / geometry.dy)
    ey = ey.at[:, -1].set((phi[:, -2] - phi[:, -1]) / geometry.dy)
    return ex.astype(jnp.float32), ey.astype(jnp.float32)
