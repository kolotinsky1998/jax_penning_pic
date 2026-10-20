from pathlib import Path

import pytest

from jax_penning_pic.io import prepare_output_dir


def test_existing_simulation_output_is_rejected(tmp_path: Path):
    (tmp_path / "counters.csv").write_text("step,ne\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite-output"):
        prepare_output_dir(tmp_path)


def test_overwrite_removes_only_known_simulation_artifacts(tmp_path: Path):
    generated = [
        tmp_path / "metadata.json",
        tmp_path / "counters.csv",
        tmp_path / "fields_100.vtu",
        tmp_path / "rho_e_100.txt",
    ]
    for path in generated:
        path.write_text("generated", encoding="utf-8")
    user_file = tmp_path / "notes.txt"
    user_file.write_text("keep", encoding="utf-8")

    result = prepare_output_dir(tmp_path, overwrite=True)

    assert result == tmp_path
    assert all(not path.exists() for path in generated)
    assert user_file.read_text(encoding="utf-8") == "keep"
