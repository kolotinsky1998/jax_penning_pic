#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot particle trajectories from ions_traj_particles.txt-like files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="ions_traj_particles.txt",
        help="Path to trajectory file. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="ions_traj_particles.png",
        help="Output image path. Default: %(default)s",
    )
    parser.add_argument(
        "--title",
        default="Ion Trajectories",
        help="Plot title. Default: %(default)s",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window after saving.",
    )
    return parser.parse_args()


def load_trajectories(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty trajectory file: {path}")

    blocks: list[tuple[int, list[list[float]]]] = []
    idx = 0
    n_particles = None

    while idx < len(lines):
        step = int(lines[idx])
        idx += 1

        coords: list[list[float]] = []
        while idx < len(lines):
            parts = lines[idx].split()
            if len(parts) == 1:
                break
            if len(parts) != 2:
                raise ValueError(f"Malformed coordinate line: {lines[idx]}")
            coords.append([float(parts[0]), float(parts[1])])
            idx += 1

        if n_particles is None:
            n_particles = len(coords)
        elif len(coords) != n_particles:
            raise ValueError(
                f"Inconsistent particle count at step {step}: {len(coords)} vs {n_particles}"
            )

        blocks.append((step, coords))

    if n_particles is None or n_particles == 0:
        raise ValueError(f"No particle coordinates found in {path}")

    steps = np.array([step for step, _ in blocks], dtype=int)
    trajectories = [np.empty((len(blocks), 2), dtype=float) for _ in range(n_particles)]

    for frame_idx, (_, coords) in enumerate(blocks):
        for particle_idx, coord in enumerate(coords):
            trajectories[particle_idx][frame_idx] = coord

    return steps, trajectories


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    steps, trajectories = load_trajectories(input_path)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)

    colors = ["#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b", "#b197fc", "#ffa94d"]

    for idx, traj in enumerate(trajectories):
        color = colors[idx % len(colors)]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            lw=2.6,
            color=color,
            alpha=0.95,
            label=f"particle {idx}",
        )
        ax.scatter(traj[0, 0], traj[0, 1], s=70, color=color, marker="o", edgecolors="black", zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=90, color=color, marker="*", edgecolors="black", zorder=4)
        ax.annotate(
            f"{idx}",
            (traj[-1, 0], traj[-1, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            color=color,
            weight="bold",
        )

    ax.set_title(args.title, fontsize=18, weight="bold", pad=12)
    ax.set_xlabel("x, m", fontsize=13)
    ax.set_ylabel("y, m", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)

    step_text = (
        f"steps: {steps[0]}..{steps[-1]}\n"
        f"saved frames: {len(steps)}\n"
        f"tracked particles: {len(trajectories)}"
    )
    ax.text(
        0.02,
        0.98,
        step_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
