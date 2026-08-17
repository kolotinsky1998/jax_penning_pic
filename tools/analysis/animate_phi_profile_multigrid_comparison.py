#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


GRID_NAMES = ("35", "70", "140")
COLORS = ("#ef4444", "#3b82f6", "#22c55e")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate potential profiles from 35x35, 70x70, and 140x140 grids. "
            "Profiles are interpolated onto one normalized spatial coordinate."
        )
    )
    for grid in GRID_NAMES:
        parser.add_argument(
            f"--dir-{grid}",
            required=True,
            help=f"Directory containing phi dumps for the {grid}x{grid} grid.",
        )
    parser.add_argument(
        "--pattern",
        default="phi_[0-9]*.txt",
        help="Glob pattern used in every input directory. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="phi_profile_multigrid_comparison.mp4",
        help="Output animation path (.mp4 or .gif). Default: %(default)s",
    )
    parser.add_argument("--fps", type=int, default=8, help="Frames per second. Default: %(default)s")
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI. Default: %(default)s")
    parser.add_argument(
        "--profile-axis",
        choices=("x", "y"),
        default="y",
        help="Axis along which the profile is drawn. Default: %(default)s",
    )
    parser.add_argument(
        "--fixed-fraction",
        type=float,
        default=0.5,
        help=(
            "Relative position on the orthogonal axis, from 0 to 1. "
            "The default 0.5 selects the center line on every grid."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of points in the common interpolated profile. Default: largest input grid.",
    )
    parser.add_argument("--show", action="store_true", help="Show the animation window after saving.")
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_frame(path: Path) -> np.ndarray | None:
    if path.stat().st_size == 0:
        print(f"Skipping empty file: {path}")
        return None
    frame = np.loadtxt(path)
    if frame.ndim != 2:
        print(f"Skipping non-2D file: {path}")
        return None
    return frame


def load_frame_map(directory: Path, pattern: str) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob(pattern), key=extract_step):
        step = extract_step(path)
        if step < 0:
            continue
        frame = load_frame(path)
        if frame is not None:
            frames[step] = frame
    return frames


def extract_centered_profile(frame: np.ndarray, profile_axis: str, fixed_fraction: float) -> np.ndarray:
    if profile_axis == "y":
        fixed_index = round(fixed_fraction * (frame.shape[0] - 1))
        return frame[fixed_index, :]
    fixed_index = round(fixed_fraction * (frame.shape[1] - 1))
    return frame[:, fixed_index]


def interpolate_profile(profile: np.ndarray, common_coordinate: np.ndarray) -> np.ndarray:
    native_coordinate = np.linspace(-1.0, 1.0, profile.size)
    return np.interp(common_coordinate, native_coordinate, profile)


def add_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "text.color": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "font.size": 13,
        }
    )


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
        return output_path
    if suffix == ".mp4":
        if FFMpegWriter.isAvailable():
            anim.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
            return output_path
        fallback_path = output_path.with_suffix(".gif")
        print(f"ffmpeg is unavailable, saving GIF instead: {fallback_path.resolve()}")
        anim.save(fallback_path, writer=PillowWriter(fps=fps), dpi=dpi)
        return fallback_path
    raise SystemExit(f"Unsupported output extension: {suffix}. Use .mp4 or .gif.")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.fixed_fraction <= 1.0:
        raise SystemExit("--fixed-fraction must be between 0 and 1.")
    if args.samples is not None and args.samples < 2:
        raise SystemExit("--samples must be at least 2.")

    directories = {
        "35x35": Path(args.dir_35),
        "70x70": Path(args.dir_70),
        "140x140": Path(args.dir_140),
    }
    frame_maps = {label: load_frame_map(directory, args.pattern) for label, directory in directories.items()}
    for label, frames in frame_maps.items():
        if not frames:
            raise SystemExit(f"No valid frames matched {args.pattern} in {directories[label]} ({label}).")

    common_steps = sorted(set.intersection(*(set(frames) for frames in frame_maps.values())))
    if not common_steps:
        raise SystemExit("No common iteration numbers were found across all three directories.")

    first_profiles = {
        label: extract_centered_profile(frames[common_steps[0]], args.profile_axis, args.fixed_fraction)
        for label, frames in frame_maps.items()
    }
    sample_count = args.samples or max(profile.size for profile in first_profiles.values())
    coordinate = np.linspace(-1.0, 1.0, sample_count)

    profiles: dict[str, list[np.ndarray]] = {}
    for label, frames in frame_maps.items():
        profiles[label] = [
            interpolate_profile(
                extract_centered_profile(frames[step], args.profile_axis, args.fixed_fraction),
                coordinate,
            )
            for step in common_steps
        ]

    all_values = np.concatenate([profile for grid_profiles in profiles.values() for profile in grid_profiles])
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if np.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.01, 1.0e-12)
    else:
        padding = 0.05 * (y_max - y_min)

    add_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    lines = {}
    for (label, grid_profiles), color in zip(profiles.items(), COLORS):
        (lines[label],) = ax.plot(coordinate, grid_profiles[0], color=color, lw=2.4, label=label)

    axis_name = args.profile_axis
    ax.set_xlabel(f"normalized {axis_name} coordinate", fontsize=15)
    ax.set_ylabel(r"$\phi$ (V)", fontsize=15)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.grid(True, color="#d0d0d0", alpha=0.8, linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", framealpha=0.9, facecolor="white", edgecolor="#555555")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "#60a5fa", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )
    stat_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )

    def update(frame_index: int):
        for label, line in lines.items():
            line.set_data(coordinate, profiles[label][frame_index])
        reference = profiles["70x70"][frame_index]
        diff_35 = profiles["35x35"][frame_index] - reference
        diff_140 = profiles["140x140"][frame_index] - reference
        time_text.set_text(f"iteration = {common_steps[frame_index]}")
        stat_text.set_text(
            f"RMS(35−70) = {np.sqrt(np.mean(diff_35 * diff_35)):.3e} V\n"
            f"RMS(140−70) = {np.sqrt(np.mean(diff_140 * diff_140)):.3e} V"
        )
        return (*lines.values(), time_text, stat_text)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(common_steps),
        interval=1000 / max(args.fps, 1),
        blit=False,
        repeat=True,
    )
    output_path = save_animation(animation, Path(args.output), args.fps, args.dpi)
    print(f"Compared {len(common_steps)} common frames.")
    print(f"Saved animation to {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
