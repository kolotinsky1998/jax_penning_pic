import argparse

import pytest

from jax_penning_pic.cli import positive_float


def test_positive_float_accepts_ion_mass_ratio():
    assert positive_float("1836") == 1836.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_rejects_nonpositive_ion_mass_ratio(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite number greater than zero"):
        positive_float(value)
