from pathlib import Path

import pytest

from jax_penning_pic.io import prepare_output_dir


def test_existing_simulation_output_is_rejected(tmp_path: Path):
    (tmp_path / "fields_10.vtu").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite-output"):
        prepare_output_dir(tmp_path)


def test_overwrite_removes_only_known_simulation_artifacts(tmp_path: Path):
    known = [
        tmp_path / "metadata.json",
        tmp_path / "poisson.csv",
        tmp_path / "fields.pvd",
        tmp_path / "fields_10.vtu",
        tmp_path / "particles.pvd",
        tmp_path / "particles_100.vtp",
        tmp_path / "rho_e_10.txt",
    ]
    for path in known:
        path.write_text("old", encoding="utf-8")
    keep = tmp_path / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    result = prepare_output_dir(tmp_path, overwrite=True)

    assert result == tmp_path
    assert all(not path.exists() for path in known)
    assert keep.read_text(encoding="utf-8") == "keep"
