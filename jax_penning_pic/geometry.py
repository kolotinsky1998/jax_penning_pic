from __future__ import annotations

import jax.numpy as jnp

from .state import GeometryState


def grid_coordinates(geometry: GeometryState):
    x = jnp.arange(geometry.nx, dtype=jnp.float32) * geometry.dx
    y = jnp.arange(geometry.ny, dtype=jnp.float32) * geometry.dy
    return jnp.meshgrid(x, y, indexing="ij")


def circular_domain_mask(geometry: GeometryState) -> jax.Array:
    return geometry.inside_mask


def radius_squared(x, y, geometry: GeometryState):
    return (x - geometry.x_center) ** 2 + (y - geometry.y_center) ** 2


def inside_anode(x, y, geometry: GeometryState):
    return radius_squared(x, y, geometry) < geometry.radius_anode**2


def inside_hot_cathode(x, y, geometry: GeometryState):
    return radius_squared(x, y, geometry) < geometry.ion_radius_leave_min**2


def inside_cold_cathode_ring(x, y, geometry: GeometryState):
    r2 = radius_squared(x, y, geometry)
    return (r2 >= geometry.ion_radius_leave_min**2) & (r2 < geometry.ion_radius_leave_max**2)
