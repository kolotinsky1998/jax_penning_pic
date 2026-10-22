from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


OUTPUT_EXACT_FILES = {
    "metadata.json",
    "counters.csv",
    "poisson.csv",
    "profiling.csv",
    "profiling_summary.json",
    "fields.pvd",
    "particles.pvd",
}
OUTPUT_GLOBS = (
    "fields_*.vtu",
    "particles_*.vtp",
    "rho_e_*.txt",
    "rho_i_*.txt",
    "phi_*.txt",
)


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_output_dir(path: str | Path, overwrite: bool = False) -> Path:
    """Create an output directory without silently mixing simulation runs."""

    out = ensure_output_dir(path)
    artifacts = [out / name for name in OUTPUT_EXACT_FILES if (out / name).exists()]
    for pattern in OUTPUT_GLOBS:
        artifacts.extend(out.glob(pattern))
    artifacts = sorted(set(artifacts))
    if artifacts and not overwrite:
        names = ", ".join(item.name for item in artifacts[:5])
        if len(artifacts) > 5:
            names += f", ... ({len(artifacts)} files total)"
        raise FileExistsError(
            f"output directory already contains simulation artifacts: {names}; "
            "choose a new --output-dir or pass --overwrite-output"
        )
    if overwrite:
        for artifact in artifacts:
            artifact.unlink()
    return out


def write_metadata(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv_row(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_matrix_txt(path: Path, array):
    np.savetxt(path, np.asarray(array))


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _vtk_ascii(values) -> str:
    array = np.asarray(values)
    if array.size == 0:
        return ""
    if np.issubdtype(array.dtype, np.floating):
        return " ".join(f"{value:.9g}" for value in array.reshape(-1))
    return " ".join(str(int(value)) for value in array.reshape(-1))


def write_particle_vtp(
    path: Path,
    *,
    x,
    y,
    particle_id,
    species,
    slot_id,
    generation,
):
    """Write a dependency-free VTK PolyData point cloud for ParaView."""

    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    particle_id_np = np.asarray(particle_id, dtype=np.int64)
    species_np = np.asarray(species, dtype=np.int8)
    slot_id_np = np.asarray(slot_id, dtype=np.int32)
    generation_np = np.asarray(generation, dtype=np.uint32)
    size = x_np.size
    arrays = (y_np, particle_id_np, species_np, slot_id_np, generation_np)
    if any(array.ndim != 1 or array.size != size for array in arrays) or x_np.ndim != 1:
        raise ValueError("particle VTP arrays must be one-dimensional and have equal lengths")

    points = np.column_stack((x_np, y_np, np.zeros(size, dtype=np.float32)))
    connectivity = np.arange(size, dtype=np.int64)
    offsets = np.arange(1, size + 1, dtype=np.int64)
    path.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <PolyData>\n"
        f"    <Piece NumberOfPoints=\"{size}\" NumberOfVerts=\"{size}\" "
        "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n"
        "      <PointData GlobalIds=\"particle_id\">\n"
        f"        <DataArray type=\"Int64\" Name=\"particle_id\" format=\"ascii\">{_vtk_ascii(particle_id_np)}</DataArray>\n"
        f"        <DataArray type=\"Int8\" Name=\"species\" format=\"ascii\">{_vtk_ascii(species_np)}</DataArray>\n"
        f"        <DataArray type=\"Int32\" Name=\"slot_id\" format=\"ascii\">{_vtk_ascii(slot_id_np)}</DataArray>\n"
        f"        <DataArray type=\"UInt32\" Name=\"generation\" format=\"ascii\">{_vtk_ascii(generation_np)}</DataArray>\n"
        "      </PointData>\n"
        "      <CellData/>\n"
        "      <Points>\n"
        f"        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">{_vtk_ascii(points)}</DataArray>\n"
        "      </Points>\n"
        "      <Verts>\n"
        f"        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">{_vtk_ascii(connectivity)}</DataArray>\n"
        f"        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">{_vtk_ascii(offsets)}</DataArray>\n"
        "      </Verts>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )

    files = sorted(path.parent.glob("particles_*.vtp"), key=lambda item: int(item.stem.split("_")[-1]))
    entries = "\n".join(
        f'    <DataSet timestep="{item.stem.split("_")[-1]}" group="" part="0" file="{item.name}"/>'
        for item in files
    )
    (path.parent / "particles.pvd").write_text(
        "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        f"  <Collection>\n{entries}\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )


def write_triangular_vtk(
    path: Path,
    points,
    cells,
    *,
    rho_e,
    rho_i,
    rho,
    phi,
    ex,
    ey,
    elementary_charge: float,
):
    """Write a ParaView-readable FEM field dump outside JAX tracing."""

    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - package dependency guard
        raise RuntimeError("VTK output requires meshio; install the FEM dependencies") from exc
    points_np = np.asarray(points)
    ex_np = np.asarray(ex)
    ey_np = np.asarray(ey)
    meshio.write_points_cells(
        path,
        points_np,
        [("triangle", np.asarray(cells))],
        point_data={
            "rho_e": np.asarray(rho_e),
            "rho_i": np.asarray(rho_i),
            "rho": np.asarray(rho),
            "n_e": -np.asarray(rho_e) / elementary_charge,
            "n_i": np.asarray(rho_i) / elementary_charge,
            "phi": np.asarray(phi),
            "E": np.column_stack((ex_np, ey_np, np.zeros_like(ex_np))),
        },
    )
    # A compact PVD collection lets ParaView open every field snapshot at once.
    files = sorted(path.parent.glob("fields_*.vtu"), key=lambda p: int(p.stem.split("_")[-1]))
    entries = "\n".join(
        f'    <DataSet timestep="{item.stem.split("_")[-1]}" group="" part="0" file="{item.name}"/>'
        for item in files
    )
    (path.parent / "fields.pvd").write_text(
        "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        f"  <Collection>\n{entries}\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
