#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    print("Point this script to JAX counters.csv and the C++ reference logs for custom comparison.")


if __name__ == "__main__":
    main()
