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


@pytest.mark.parametrize(
    ("model_radius", "expected_physical_radius"),
    [(0.01, 0.0002), (0.04, 0.0008), (0.08, 0.0016)],
)
def test_hot_cathode_radius_is_scaled_once(model_radius, expected_physical_radius):
    config = replace(default_config(), hot_cathode_radius=model_radius)
    geometry, _ = build_runtime(config)

    assert geometry.ion_radius_leave_min == pytest.approx(expected_physical_radius)


@pytest.mark.parametrize("model_radius", [0.0, -0.01, 0.15, 0.2])
def test_hot_cathode_radius_must_fit_inside_cold_cathode(model_radius):
    config = replace(default_config(), hot_cathode_radius=model_radius)
    with pytest.raises(ValueError, match="hot_cathode_radius"):
        build_runtime(config)


@pytest.mark.parametrize(
    ("grid_size", "model_radius", "expected_cells"),
    [
        (70, 0.04, 5.3846153846),
        (140, 0.04, 10.7692307692),
        (280, 0.04, 21.5384615385),
        (70, 0.08, 10.7692307692),
        (140, 0.08, 21.5384615385),
        (280, 0.08, 43.0769230769),
    ],
)
def test_hot_cathode_resolution_in_cells(grid_size, model_radius, expected_cells):
    config = replace(default_config(), grid_size=grid_size, hot_cathode_radius=model_radius)
    geometry, _ = build_runtime(config)

    assert geometry.ion_radius_leave_min / geometry.dx == pytest.approx(expected_cells)


def test_hot_cathode_radius_does_not_change_prescribed_current_rates():
    _, baseline = build_runtime(default_config())

    for model_radius in (0.04, 0.08):
        _, runtime = build_runtime(replace(default_config(), hot_cathode_radius=model_radius))
        assert runtime.hot_emit_rate == pytest.approx(baseline.hot_emit_rate)
        assert runtime.cold_emit_rate == pytest.approx(baseline.cold_emit_rate)
        assert runtime.ion_leave_rate == pytest.approx(baseline.ion_leave_rate)
