#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# USER SETTINGS
# =========================
# Set these paths/values and run the file.
INPUT_FILE = "data/cross_sections/Ar+-Ar_elastic.txt"
OUTPUT_FILE = "arplus_resampled.txt"

ENERGY_MIN_EV = 0.001
ENERGY_MAX_EV = 400.0
NUM_POINTS = 501

# Multiply the resampled cross section by this factor.
# Example: 0.5 to divide all cross sections by 2.
SIGMA_SCALE = 1.0

# If True, save the plot to file. In Colab it is usually enough to display it inline.
SAVE_PLOT_TO_FILE = False
PLOT_FILE = "arplus_resampled.png"


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    energies = np.array([row[0] for row in rows], dtype=float)
    sigmas = np.array([row[1] for row in rows], dtype=float)
    return energies, sigmas


def resample_uniform(
    energies: np.ndarray,
    sigmas: np.ndarray,
    emin: float,
    emax: float,
    npoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    if npoints < 2:
        raise ValueError("NUM_POINTS must be at least 2")
    if emax <= emin:
        raise ValueError("ENERGY_MAX_EV must be greater than ENERGY_MIN_EV")

    order = np.argsort(energies)
    energies = energies[order]
    sigmas = sigmas[order]

    new_energies = np.linspace(emin, emax, npoints)
    new_sigmas = np.interp(
        new_energies,
        energies,
        sigmas,
        left=sigmas[0],
        right=sigmas[-1],
    )
    return new_energies, new_sigmas


def write_table(path: Path, energies: np.ndarray, sigmas: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for energy, sigma in zip(energies, sigmas):
            f.write(f"{energy:.15g}\t{sigma:.15g}\n")


def make_plot(
    src_e: np.ndarray,
    src_s: np.ndarray,
    dst_e: np.ndarray,
    dst_s: np.ndarray,
    input_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)

    ax.plot(
        src_e,
        src_s,
        color="#d62728",
        lw=2.2,
        marker="o",
        ms=4.0,
        alpha=0.9,
        label="Original data",
    )
    ax.plot(
        dst_e,
        dst_s,
        color="#1f77b4",
        lw=2.6,
        alpha=0.95,
        label="Uniform-grid data",
    )

    ax.set_title(f"Collision Table Resampling\n{input_path.name}", fontsize=17, weight="bold")
    ax.set_xlabel("Energy, eV", fontsize=13)
    ax.set_ylabel(r"Cross section, m$^2$", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=12)

    info = (
        f"original points = {len(src_e)}\n"
        f"new points = {len(dst_e)}\n"
        f"E range = [{dst_e[0]:.6g}, {dst_e[-1]:.6g}] eV\n"
        f"sigma scale = {SIGMA_SCALE:g}"
    )
    ax.text(
        0.98,
        0.03,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
            "boxstyle": "round,pad=0.35",
        },
    )

    if SAVE_PLOT_TO_FILE:
        plot_path = Path(PLOT_FILE)
        fig.savefig(plot_path, dpi=220, bbox_inches="tight")
        print(f"Saved plot to {plot_path.resolve()}")

    plt.show()


def main() -> None:
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    energies, sigmas = load_table(input_path)
    new_energies, new_sigmas = resample_uniform(
        energies,
        sigmas,
        ENERGY_MIN_EV,
        ENERGY_MAX_EV,
        NUM_POINTS,
    )
    new_sigmas = new_sigmas * SIGMA_SCALE

    write_table(output_path, new_energies, new_sigmas)
    print(f"Saved resampled table to {output_path.resolve()}")

    make_plot(
        energies,
        sigmas,
        new_energies,
        new_sigmas,
        input_path,
    )


if __name__ == "__main__":
    main()
