#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DT_RE = re.compile(r"^\s*dt_e\s*=\s*([0-9eE+\-.]+)")
ITER_RE = re.compile(r"^\s*iter:\s*(\d+)\s+(Ntot_[A-Za-z0-9_]+):\s*([0-9eE+\-.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Penning output and plot particle counters vs time."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="output_penning_500me_ion_mass",
        help="Path to the simulation output file. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="penning_counters_vs_time.png",
        help="Output plot path. Default: %(default)s",
    )
    parser.add_argument(
        "--csv-output",
        default="penning_counters_vs_time.csv",
        help="Output CSV path with parsed counters. Default: %(default)s",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    parser.add_argument(
        "--keep-negative",
        action="store_true",
        help="Keep original signs for leave counters instead of plotting absorbed counts as positive values.",
    )
    return parser.parse_args()


def parse_output(path: Path) -> tuple[float, dict[int, dict[str, float]]]:
    dt_e = None
    by_iter: dict[int, dict[str, float]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if dt_e is None:
                match_dt = DT_RE.match(line)
                if match_dt:
                    dt_e = float(match_dt.group(1))

            match_iter = ITER_RE.match(line)
            if not match_iter:
                continue

            iteration = int(match_iter.group(1))
            key = match_iter.group(2)
            value = float(match_iter.group(3))

            if iteration not in by_iter:
                by_iter[iteration] = {}
            by_iter[iteration][key] = value

    if dt_e is None:
        raise ValueError(f"Could not find dt_e in {path}")
    if not by_iter:
        raise ValueError(f"Could not find counter lines in {path}")

    return dt_e, by_iter


def build_arrays(
    dt_e: float, by_iter: dict[int, dict[str, float]], keep_negative: bool
) -> dict[str, np.ndarray]:
    iterations = np.array(sorted(by_iter), dtype=float)
    time_us = iterations * dt_e * 1.0e6

    def values(name: str) -> np.ndarray:
        return np.array([by_iter[it].get(name, np.nan) for it in sorted(by_iter)], dtype=float)

    ionized = values("Ntot_ionized")
    cold_leave = values("Ntot_cold_cathode_leave")
    anode_leave = values("Ntot_anode_leave")
    hot_emit = values("Ntot_hot_cathode_emission")

    if not keep_negative:
        cold_leave = np.abs(cold_leave)
        anode_leave = np.abs(anode_leave)

    return {
        "iteration": iterations,
        "time_us": time_us,
        "Ntot_ionized": ionized,
        "Ntot_cold_cathode_leave": cold_leave,
        "Ntot_anode_leave": anode_leave,
        "Ntot_hot_cathode_emission": hot_emit,
    }


def style_plot() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f7fb",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#f6f7fb",
            "axes.edgecolor": "#d8deeb",
            "grid.color": "#dfe5f0",
            "grid.alpha": 0.9,
            "axes.titleweight": "bold",
            "font.size": 13,
        }
    )


def save_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    headers = [
        "iteration",
        "time_us",
        "Ntot_ionized",
        "Ntot_cold_cathode_leave",
        "Ntot_anode_leave",
        "Ntot_hot_cathode_emission",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip(*(data[key] for key in headers)):
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    dt_e, by_iter = parse_output(input_path)
    data = build_arrays(dt_e, by_iter, keep_negative=args.keep_negative)
    style_plot()

    fig, ax = plt.subplots(figsize=(11.5, 7.6), constrained_layout=True)

    colors = {
        "Ntot_ionized": "#d62728",
        "Ntot_cold_cathode_leave": "#1f77b4",
        "Ntot_anode_leave": "#2ca02c",
        "Ntot_hot_cathode_emission": "#9467bd",
    }

    labels = {
        "Ntot_ionized": "Ionized electrons",
        "Ntot_cold_cathode_leave": "Ions absorbed on cold cathode",
        "Ntot_anode_leave": "Electrons absorbed on anode",
        "Ntot_hot_cathode_emission": "Electrons emitted from hot cathode",
    }

    order = [
        "Ntot_ionized",
        "Ntot_cold_cathode_leave",
        "Ntot_anode_leave",
        "Ntot_hot_cathode_emission",
    ]

    for key in order:
        ax.plot(
            data["time_us"],
            data[key],
            lw=3.0 if key == "Ntot_ionized" else 2.6,
            color=colors[key],
            label=labels[key],
            alpha=0.96,
        )

    ax.set_title("Penning Discharge Counters vs Time", fontsize=20, pad=14)
    ax.set_xlabel("Time, µs", fontsize=15)
    ax.set_ylabel("Count of macroparticles", fontsize=15)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=12)

    info = (
        f"dt_e = {dt_e:.5e} s\n"
        f"points = {len(data['time_us'])}\n"
        f"t_max = {data['time_us'][-1]:.3f} µs"
    )
    ax.text(
        0.985,
        0.03,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "#c9d3e6",
            "alpha": 0.95,
            "boxstyle": "round,pad=0.35",
        },
    )

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    csv_output_path = Path(args.csv_output)
    save_csv(csv_output_path, data)
    print(f"Saved figure to {output_path.resolve()}")
    print(f"Saved CSV to {csv_output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
