from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
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
