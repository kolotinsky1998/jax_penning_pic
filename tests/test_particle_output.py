from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np

from jax_penning_pic.config import default_config
from jax_penning_pic.io import write_particle_vtp
from jax_penning_pic.particles import write_particles
from jax_penning_pic.simulation_circle_gyro_new import _dump_particles, _sample_particle_pool
from jax_penning_pic.state import empty_particle_pool


def _write_into_slots(pool, indices, values):
    indices = jnp.asarray(indices)
    valid = jnp.ones(indices.shape, dtype=bool)
    values = jnp.asarray(values, dtype=jnp.float32)
    zeros = jnp.zeros_like(values)
    return write_particles(pool, indices, valid, values, values + 0.5, zeros, zeros, zeros)


def test_sample_uses_stable_unique_lifetime_ids():
    pool = _write_into_slots(empty_particle_pool(3, 0.0), [0, 1], [10.0, 20.0])

    first = _sample_particle_pool(
        pool, species=0, species_offset=0, total_capacity=5, stride=1
    )
    moved = pool._replace(x=pool.x.at[0].set(11.0))
    second = _sample_particle_pool(
        moved, species=0, species_offset=0, total_capacity=5, stride=1
    )

    np.testing.assert_array_equal(first["particle_id"], [0, 1])
    np.testing.assert_array_equal(second["particle_id"], first["particle_id"])
    assert second["x"][0] == 11.0

    dead = moved._replace(alive=moved.alive.at[0].set(False))
    reused = _write_into_slots(dead, [0], [30.0])
    third = _sample_particle_pool(
        reused, species=0, species_offset=0, total_capacity=5, stride=1
    )
    assert third["particle_id"][0] == 5
    assert third["particle_id"][0] not in first["particle_id"]


def test_sampling_filters_dead_particles_and_applies_stride():
    pool = _write_into_slots(empty_particle_pool(5, 0.0), [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0])
    pool = pool._replace(alive=pool.alive.at[2].set(False))

    sampled = _sample_particle_pool(
        pool, species=0, species_offset=0, total_capacity=7, stride=2
    )

    np.testing.assert_array_equal(sampled["particle_id"], [0])
    np.testing.assert_array_equal(sampled["slot_id"], [0])


def test_electron_and_ion_ids_do_not_overlap():
    electrons = _write_into_slots(empty_particle_pool(3, 0.0), [0, 1], [0.0, 1.0])
    ions = _write_into_slots(empty_particle_pool(2, 0.0), [0, 1], [0.0, 1.0])

    electron_data = _sample_particle_pool(
        electrons, species=0, species_offset=0, total_capacity=5, stride=1
    )
    ion_data = _sample_particle_pool(
        ions, species=1, species_offset=3, total_capacity=5, stride=1
    )

    assert set(electron_data["particle_id"]).isdisjoint(ion_data["particle_id"])


def test_vtp_and_pvd_support_empty_frames_and_numeric_step_order(tmp_path: Path):
    empty = {
        "x": np.array([], dtype=np.float32),
        "y": np.array([], dtype=np.float32),
        "particle_id": np.array([], dtype=np.int64),
        "species": np.array([], dtype=np.int8),
        "slot_id": np.array([], dtype=np.int32),
        "generation": np.array([], dtype=np.uint32),
    }
    write_particle_vtp(tmp_path / "particles_10.vtp", **empty)
    write_particle_vtp(tmp_path / "particles_2.vtp", **empty)

    root = ET.parse(tmp_path / "particles_2.vtp").getroot()
    piece = root.find("./PolyData/Piece")
    point_data = root.find("./PolyData/Piece/PointData")
    assert piece is not None and piece.attrib["NumberOfPoints"] == "0"
    assert point_data is not None and point_data.attrib["GlobalIds"] == "particle_id"
    assert {item.attrib.get("Name") for item in point_data.findall("DataArray")} == {
        "particle_id", "species", "slot_id", "generation"
    }

    collection = ET.parse(tmp_path / "particles.pvd").getroot()
    assert [item.attrib["timestep"] for item in collection.findall("./Collection/DataSet")] == ["2", "10"]


def test_dump_particles_combines_species_and_writes_current_step(tmp_path: Path):
    electrons = _write_into_slots(empty_particle_pool(2, 0.0), [0], [1.0])
    ions = _write_into_slots(empty_particle_pool(2, 0.0), [0], [2.0])

    class State:
        step = 7

    state = State()
    state.electrons = electrons
    state.ions = ions
    config = replace(default_config(output_dir=str(tmp_path)), particle_sample_stride=1)

    _dump_particles(tmp_path, state, config)

    root = ET.parse(tmp_path / "particles_7.vtp").getroot()
    piece = root.find("./PolyData/Piece")
    assert piece is not None and piece.attrib["NumberOfPoints"] == "2"
