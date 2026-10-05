import argparse

import pytest

from jax_penning_pic import cli
from jax_penning_pic.cli import positive_float


def test_positive_float_accepts_ion_mass_ratio():
    assert positive_float("1836") == 1836.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_rejects_nonpositive_ion_mass_ratio(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite number greater than zero"):
        positive_float(value)


@pytest.mark.parametrize(("args", "expected"), [([], True), (["--disable-rho-filter"], False)])
def test_rho_filter_flag(monkeypatch, args, expected):
    captured = {}
    monkeypatch.setattr("sys.argv", ["jax-penning-run", *args])
    monkeypatch.setattr(cli, "run_simulation", lambda config, **kwargs: captured.setdefault("config", config))

    cli.main()

    assert captured["config"].use_rho_filter is expected
