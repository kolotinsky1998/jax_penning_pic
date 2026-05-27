from jax_penning_pic.config import default_config
from jax_penning_pic.state import build_runtime


def test_runtime_geometry_is_positive():
    geometry, runtime = build_runtime(default_config())
    assert geometry.nx > 0
    assert geometry.ny > 0
    assert geometry.dx > 0
    assert runtime.dt > 0
