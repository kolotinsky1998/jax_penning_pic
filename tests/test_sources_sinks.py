import jax
import jax.numpy as jnp

from jax_penning_pic.config import default_config
from jax_penning_pic.sources_sinks import (
    inject_initial_disk_samples,
    sample_reference_initial_disk,
)
from jax_penning_pic.state import build_runtime, empty_particle_pool


def test_reference_initial_samples_reuse_positions_and_velocity_factors():
    config = default_config()
    geometry, _ = build_runtime(config)
    count = 16
    x, y, velocity_factors = sample_reference_initial_disk(
        jax.random.PRNGKey(config.seed),
        geometry,
        count,
    )
    speed_std_e = 3.0
    speed_std_i = 0.5
    electrons = inject_initial_disk_samples(
        empty_particle_pool(count, config.b / config.scale),
        x,
        y,
        velocity_factors,
        speed_std_e,
    )
    ions = inject_initial_disk_samples(
        empty_particle_pool(count, config.b / config.scale),
        x,
        y,
        velocity_factors,
        speed_std_i,
    )

    assert jnp.allclose(electrons.x, ions.x)
    assert jnp.allclose(electrons.y, ions.y)
    assert jnp.allclose(electrons.vx / speed_std_e, ions.vx / speed_std_i)
    assert jnp.all(jnp.linalg.norm(velocity_factors, axis=1) < 3.0)
