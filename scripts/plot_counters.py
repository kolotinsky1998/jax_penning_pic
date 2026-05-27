#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="outputs/simulation_circle_gyro_new/counters.csv")
    args = parser.parse_args()
    path = Path(args.csv_path)
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    ne = [int(r["ne"]) for r in rows]
    ni = [int(r["ni"]) for r in rows]
    plt.plot(steps, ne, label="Ne")
    plt.plot(steps, ni, label="Ni")
    plt.xlabel("step")
    plt.ylabel("count")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
