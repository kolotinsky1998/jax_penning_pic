from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "analysis" / "animate_hot_cathode_fields.py"
SPEC = importlib.util.spec_from_file_location("animate_hot_cathode_fields", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
animation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = animation
SPEC.loader.exec_module(animation)


def geometry(size: int = 5, spacing: float = 0.1, hot_radius: float = 0.21):
    return animation.Geometry(
        size=size,
        dx=spacing,
        dy=spacing,
        x_center=0.5 * size * spacing,
        y_center=0.5 * size * spacing,
        hot_radius=hot_radius,
        dt=1.0e-9,
    )


def test_quantities_are_concentrations_and_total_charge_density():
    macro_weight = 123.0
    rho_e = np.full((5, 5), -animation.ELEMENTARY_CHARGE * macro_weight)
    rho_i = np.full((5, 5), animation.ELEMENTARY_CHARGE * 2.0 * macro_weight)
    phi = np.zeros((5, 5))

    quantities = animation.calculate_quantities(rho_e, rho_i, phi, geometry())

    np.testing.assert_allclose(quantities["n_e"], macro_weight)
    np.testing.assert_allclose(quantities["n_i"], 2.0 * macro_weight)
    np.testing.assert_allclose(quantities["rho"], rho_e + rho_i)


def test_electric_field_is_negative_potential_gradient():
    grid = geometry(size=7, spacing=0.02)
    x = np.arange(grid.size)[:, None] * grid.dx
    y = np.arange(grid.size)[None, :] * grid.dy
    phi = 3.0 * x - 4.0 * y
    zeros = np.zeros_like(phi)

    quantities = animation.calculate_quantities(zeros, zeros, phi, grid)

    np.testing.assert_allclose(quantities["e_abs"], 5.0, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("size", [10, 11])
def test_hot_region_and_centered_profile_support_even_and_odd_grids(size):
    grid = geometry(size=size, spacing=0.01, hot_radius=0.035)
    x_indices, y_indices, mask = animation.hot_region(grid)
    values = np.arange(size, dtype=float)[:, None] * np.ones((1, size))

    coordinate, profile = animation.centered_profile(values, grid, "x")

    assert mask.shape == (x_indices.size, y_indices.size)
    assert np.any(mask)
    assert np.all(np.abs(coordinate) <= grid.hot_radius)
    np.testing.assert_allclose(profile, values[:, 0][np.abs(np.arange(size) * grid.dx - grid.x_center) <= grid.hot_radius])


def test_reconstruct_geometry_for_explicit_grid_size():
    metadata = {
        "config": {"scale": 0.02, "r": 0.26, "grid_size": 560},
        "derived": {"dt": 2.5e-11},
    }

    grid = animation.reconstruct_geometry(metadata, 560)

    assert grid.dx == pytest.approx(2.0 * 0.26 * 0.02 / 560)
    assert grid.hot_radius == pytest.approx(0.01 * 0.02)
    assert grid.dt == pytest.approx(2.5e-11)


def test_reconstruct_geometry_for_automatic_grid_size():
    config = {
        "scale": 0.02,
        "r": 0.26,
        "grid_size": None,
        "n_e": 1.0e15,
        "n_i": 1.0e15,
        "t_e_ev": 10.0,
        "t_i_ev": 10.0,
    }
    spacing = animation.debye_grid_spacing(config)
    expected_size = int(2.0 * config["r"] * config["scale"] / spacing)

    grid = animation.reconstruct_geometry({"config": config, "derived": {"dt": 1.0e-9}}, expected_size)

    assert grid.size == expected_size
    assert grid.dx == pytest.approx(spacing)


def test_discover_frames_rejects_unsynchronized_steps(tmp_path: Path):
    for name in ("rho_e_0.txt", "rho_i_0.txt", "phi_0.txt", "rho_e_1.txt"):
        (tmp_path / name).write_text("0 0\n0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="не синхронизированы"):
        animation.discover_frames(tmp_path)


def test_load_metadata_rejects_missing_and_invalid_files(tmp_path: Path):
    with pytest.raises(ValueError, match="Не найден"):
        animation.load_metadata(tmp_path)

    (tmp_path / "metadata.json").write_text(json.dumps({"derived": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        animation.load_metadata(tmp_path)


def test_load_matrix_rejects_nonfinite_and_wrong_shape(tmp_path: Path):
    nonsquare = tmp_path / "nonsquare.txt"
    nonsquare.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="квадратная"):
        animation.load_matrix(nonsquare)

    nonfinite = tmp_path / "nonfinite.txt"
    nonfinite.write_text("1 nan\n3 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        animation.load_matrix(nonfinite)
