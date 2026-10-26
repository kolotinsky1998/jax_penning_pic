import argparse

import pytest

from jax_penning_pic import cli
from jax_penning_pic.cli import non_negative_int, positive_float


def test_positive_float_accepts_ion_mass_ratio():
    assert positive_float("1836") == 1836.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_rejects_nonpositive_ion_mass_ratio(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite number greater than zero"):
        positive_float(value)


def test_non_negative_int_validation():
    assert non_negative_int("0") == 0
    assert non_negative_int("5") == 5
    with pytest.raises(argparse.ArgumentTypeError, match="greater than or equal to zero"):
        non_negative_int("-1")


def test_output_options_are_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "jax-penning-run",
            "--macro-particle-weight", "123",
            "--particle-sample-stride", "7",
            "--overwrite-output",
        ],
    )
    monkeypatch.setattr(cli, "run_simulation", lambda config, **kwargs: captured.update(config=config, kwargs=kwargs))

    cli.main()

    assert captured["config"].macro_particle_weight == 123.0
    assert captured["config"].particle_sample_stride == 7
    assert captured["kwargs"]["overwrite_output"] is True


@pytest.mark.parametrize(("args", "expected"), [([], True), (["--disable-rho-filter"], False)])
def test_rho_filter_flag(monkeypatch, args, expected):
    captured = {}
    monkeypatch.setattr("sys.argv", ["jax-penning-run", *args])
    monkeypatch.setattr(cli, "run_simulation", lambda config, **kwargs: captured.setdefault("config", config))

    cli.main()

    assert captured["config"].use_rho_filter is expected
