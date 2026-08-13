from dataclasses import replace

import pytest

from jax_penning_pic.config import default_config
from jax_penning_pic.state import build_runtime


def test_runtime_geometry_is_positive():
    geometry, runtime = build_runtime(default_config())
    assert geometry.nx > 0
    assert geometry.ny > 0
    assert geometry.dx > 0
    assert runtime.dt > 0


def test_grid_size_override_sets_square_grid_and_spacing():
    config = replace(default_config(), grid_size=41)
    geometry, _ = build_runtime(config)

    assert (geometry.nx, geometry.ny) == (41, 41)
    assert geometry.dx == pytest.approx(2.0 * config.r * config.scale / 41)
    assert geometry.dy == geometry.dx


def test_grid_size_must_be_at_least_three():
    with pytest.raises(ValueError, match="grid_size must be at least 3"):
        build_runtime(replace(default_config(), grid_size=2))
