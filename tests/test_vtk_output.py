from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from jax_penning_pic.config import default_config
from jax_penning_pic.io import write_cartesian_vtu, write_particle_vtp
from jax_penning_pic.particles import write_particles
from jax_penning_pic.simulation_circle_gyro_new import _sample_particle_pool
from jax_penning_pic.state import create_initial_state, empty_particle_pool


def _write_into_slots(pool, indices, values):
    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32)
    valid = np.ones(indices.size, dtype=bool)
    zeros = np.zeros(indices.size, dtype=np.float32)
    return write_particles(pool, indices, valid, values, values + 0.5, zeros, zeros, zeros)


def test_initial_state_has_empty_pools_and_explicit_macro_weight():
    state = create_initial_state(default_config())

    assert not bool(state.electrons.alive.any())
    assert not bool(state.ions.alive.any())
    assert state.runtime.ptcls_per_macro == 1.0e7


def test_sample_ids_are_stable_unique_and_do_not_overlap_between_species():
    electrons = _write_into_slots(empty_particle_pool(3, 0.0), [0, 1], [10.0, 20.0])
    ions = _write_into_slots(empty_particle_pool(2, 0.0), [0, 1], [30.0, 40.0])
    first = _sample_particle_pool(electrons, species=0, species_offset=0, total_capacity=5, stride=1)
    second = _sample_particle_pool(electrons, species=0, species_offset=0, total_capacity=5, stride=1)
    ion_data = _sample_particle_pool(ions, species=1, species_offset=3, total_capacity=5, stride=1)

    np.testing.assert_array_equal(first["particle_id"], second["particle_id"])
    assert set(first["particle_id"]).isdisjoint(ion_data["particle_id"])

    electrons = electrons._replace(alive=electrons.alive.at[0].set(False))
    electrons = _write_into_slots(electrons, [0], [50.0])
    reused = _sample_particle_pool(electrons, species=0, species_offset=0, total_capacity=5, stride=1)
    assert reused["particle_id"][0] == 5
    assert reused["particle_id"][0] not in first["particle_id"]


def test_particle_vtp_and_pvd_support_empty_frames_and_numeric_order(tmp_path: Path):
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

    point_data = ET.parse(tmp_path / "particles_2.vtp").getroot().find(".//PointData")
    assert point_data is not None and point_data.attrib["GlobalIds"] == "particle_id"
    assert [item.attrib["Name"] for item in point_data.findall("DataArray")] == [
        "particle_id", "species", "slot_id", "generation"
    ]
    datasets = ET.parse(tmp_path / "particles.pvd").getroot().findall(".//DataSet")
    assert [item.attrib["timestep"] for item in datasets] == ["2", "10"]


def test_cartesian_vtu_contains_quad_grid_fields_and_pvd(tmp_path: Path):
    shape = (3, 4)
    base = np.arange(12, dtype=np.float32).reshape(shape)
    write_cartesian_vtu(
        tmp_path / "fields_7.vtu",
        dx=0.25,
        dy=0.5,
        rho_e=-base,
        rho_i=base,
        rho=np.zeros(shape, dtype=np.float32),
        phi=base,
        ex=np.ones(shape, dtype=np.float32),
        ey=np.ones(shape, dtype=np.float32) * 2,
        elementary_charge=1.0,
    )

    root = ET.parse(tmp_path / "fields_7.vtu").getroot()
    piece = root.find(".//Piece")
    assert piece is not None
    assert piece.attrib == {"NumberOfPoints": "12", "NumberOfCells": "6"}
    assert set(item.attrib.get("Name") for item in root.findall(".//PointData/DataArray")) == {
        "rho_e", "rho_i", "rho", "n_e", "n_i", "phi", "E"
    }
    cell_types = root.find(".//Cells/DataArray[@Name='types']")
    assert cell_types is not None and set(cell_types.text.split()) == {"9"}
    datasets = ET.parse(tmp_path / "fields.pvd").getroot().findall(".//DataSet")
    assert datasets[0].attrib["file"] == "fields_7.vtu"
