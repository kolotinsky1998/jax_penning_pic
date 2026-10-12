#!/usr/bin/env python3
"""Generate a graded circular triangular mesh for the FEM PIC backend.

The script is intentionally separate from simulation execution.  Inspect the
resulting ``mesh.vtu`` in ParaView before passing ``mesh.npz`` to a run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import meshio
import numpy as np
import pygmsh


def _boundary_nodes(cells: np.ndarray) -> np.ndarray:
    edges = np.concatenate((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]))
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1].ravel())


def _precompute(points: np.ndarray, cells: np.ndarray):
    vertices = points[cells]
    twice_area = (
        (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 1, 1] - vertices[:, 0, 1]) * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )
    if np.any(twice_area <= 0.0):
        # Gmsh normally guarantees orientation, but normalize the archive.
        bad = twice_area < 0.0
        cells = cells.copy()
        cells[bad, 1], cells[bad, 2] = cells[bad, 2], cells[bad, 1].copy()
        vertices = points[cells]
        twice_area = (
            (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 0, 1])
            - (vertices[:, 1, 1] - vertices[:, 0, 1]) * (vertices[:, 2, 0] - vertices[:, 0, 0])
        )
    areas = 0.5 * twice_area
    inv_m = np.empty((cells.shape[0], 3, 3), dtype=np.float64)
    for index, triangle in enumerate(vertices):
        matrix = np.array(
            [[triangle[0, 0], triangle[1, 0], triangle[2, 0]],
             [triangle[0, 1], triangle[1, 1], triangle[2, 1]],
             [1.0, 1.0, 1.0]],
            dtype=np.float64,
        )
        inv_m[index] = np.linalg.inv(matrix)
    grad_shape = np.transpose(inv_m[:, :, :2], (0, 1, 2))
    lumped_mass = np.zeros(points.shape[0], dtype=np.float64)
    stiffness_diag = np.zeros(points.shape[0], dtype=np.float64)
    local_diag = areas[:, None] * np.sum(grad_shape * grad_shape, axis=2)
    np.add.at(lumped_mass, cells, areas[:, None] / 3.0)
    np.add.at(stiffness_diag, cells, local_diag)
    return cells, areas, grad_shape, lumped_mass, stiffness_diag


def _build_lookup(points, cells, origin, radius, bin_size):
    n_bins = int(math.ceil(2.0 * radius / bin_size))
    bins: list[list[list[int]]] = [[[] for _ in range(n_bins)] for _ in range(n_bins)]
    triangles = points[cells]
    for cell_id, triangle in enumerate(triangles):
        lo = np.floor((triangle.min(axis=0) - origin) / bin_size).astype(int)
        hi = np.floor((triangle.max(axis=0) - origin) / bin_size).astype(int)
        for i in range(max(0, lo[0]), min(n_bins - 1, hi[0]) + 1):
            for j in range(max(0, lo[1]), min(n_bins - 1, hi[1]) + 1):
                bins[i][j].append(cell_id)
    max_candidates = max(1, max(len(item) for row in bins for item in row))
    lookup = np.full((n_bins, n_bins, max_candidates), -1, dtype=np.int32)
    for i, row in enumerate(bins):
        for j, candidates in enumerate(row):
            lookup[i, j, :len(candidates)] = candidates
    return lookup


def generate(args):
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise SystemExit("gmsh is required; install the project FEM extra") from exc

    if not (args.h_hot > 0 and args.h_bulk > 0 and args.transition_width > 0):
        raise ValueError("h_hot, h_bulk and transition_width must be positive")
    if args.hot_radius <= 0 or args.radius <= args.hot_radius:
        raise ValueError("require 0 < hot_radius < radius")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Pygmsh owns the geometry lifecycle; Gmsh's field API is used because it
    # exposes MathEval background fields directly.
    with pygmsh.occ.Geometry() as geom:
        geom.add_disk([args.center_x, args.center_y, 0.0], args.radius)
        geom.synchronize()
        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) != 1:
            raise RuntimeError("expected exactly one disk surface")
        boundary = gmsh.model.getBoundary(surfaces, oriented=False, recursive=False)
        gmsh.model.addPhysicalGroup(1, [tag for _, tag in boundary], 1)
        gmsh.model.setPhysicalName(1, 1, "anode")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(args.h_hot, args.h_bulk))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(args.h_hot, args.h_bulk))
        field = gmsh.model.mesh.field.add("MathEval")
        radius_expr = f"sqrt((x-({args.center_x}))^2+(y-({args.center_y}))^2)"
        expression = (
            f"({args.h_hot}) + (({args.h_bulk})-({args.h_hot}))"
            f"*(0.5+0.5*tanh(({radius_expr}-({args.hot_radius}))/({args.transition_width})))"
        )
        gmsh.model.mesh.field.setString(field, "F", expression)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        # The runtime solver is P1 FEM, therefore retain only corner vertices
        # in every element irrespective of a user's global Gmsh defaults.
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out / "mesh.msh"))

    mesh = meshio.read(out / "mesh.msh")
    triangle_block = next((block for block in mesh.cells if block.type.startswith("triangle")), None)
    if triangle_block is None:
        found = ", ".join(f"{block.type}{block.data.shape}" for block in mesh.cells)
        raise RuntimeError(f"Gmsh did not create triangular cells; meshio found: {found}")
    points = np.asarray(mesh.points[:, :2], dtype=np.float64)
    # meshio may expose an existing second-order Gmsh file as ``triangle6``.
    # P1 FEM uses its first three corner-node entries.
    cells = np.asarray(triangle_block.data[:, :3], dtype=np.int32)
    cells, areas, grad_shape, lumped_mass, stiffness_diag = _precompute(points, cells)
    boundary_nodes = _boundary_nodes(cells)
    boundary_mask = np.zeros(points.shape[0], dtype=bool)
    boundary_mask[boundary_nodes] = True
    origin = np.array([args.center_x - args.radius, args.center_y - args.radius], dtype=np.float64)
    lookup_bin_size = min(args.h_hot, args.h_bulk) * 0.5
    lookup = _build_lookup(points, cells, origin, args.radius, lookup_bin_size)

    np.savez_compressed(
        out / "mesh.npz",
        points=points.astype(np.float32), cells=cells, areas=areas.astype(np.float32),
        grad_shape=grad_shape.astype(np.float32), lumped_mass=lumped_mass.astype(np.float32),
        boundary_mask=boundary_mask, stiffness_diag=stiffness_diag.astype(np.float32),
        lookup_origin=origin.astype(np.float32), lookup_bin_size=np.float32(lookup_bin_size),
        lookup_shape=np.asarray(lookup.shape[:2], dtype=np.int32), lookup_candidates=lookup,
    )
    meshio.write_points_cells(out / "mesh.vtu", points, [("triangle", cells)])
    metadata = {
        "center": [args.center_x, args.center_y], "radius": args.radius,
        "hot_radius": args.hot_radius, "h_hot": args.h_hot, "h_bulk": args.h_bulk,
        "transition_width": args.transition_width, "n_nodes": int(points.shape[0]),
        "n_triangles": int(cells.shape[0]), "n_boundary_nodes": int(boundary_nodes.size),
        "area_min": float(areas.min()), "area_max": float(areas.max()),
        "lookup_shape": list(lookup.shape), "lookup_bin_size": lookup_bin_size,
    }
    (out / "mesh_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--hot-radius", type=float, required=True)
    parser.add_argument("--h-hot", type=float, required=True)
    parser.add_argument("--h-bulk", type=float, required=True)
    parser.add_argument("--transition-width", type=float, required=True)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
