#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate line profiles of electric potential from phi*.txt files."
    )
    parser.add_argument(
        "--pattern",
        default="phi[0-9]*.txt",
        help="Glob pattern for input files. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="phi_profile_evolution.mp4",
        help="Output animation path (.mp4 or .gif). Default: %(default)s",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second. Default: %(default)s",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI. Default: %(default)s",
    )
    parser.add_argument(
        "--profile-axis",
        choices=("x", "y"),
        default="y",
        help="Axis along which the profile is drawn. Default: %(default)s",
    )
    parser.add_argument(
        "--fixed-index",
        type=int,
        default=None,
        help="Fixed grid index for the orthogonal axis. Default: center line.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the animation window after saving.",
    )
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    shape = None
    for path in paths:
        if path.stat().st_size == 0:
            print(f"Skipping empty file: {path}")
            continue
        frame = np.loadtxt(path)
        if frame.ndim != 2:
            print(f"Skipping non-2D file: {path}")
            continue
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(
                f"Inconsistent frame shape: {path} has {frame.shape}, expected {shape}."
            )
        frames.append(frame)
    if not frames:
        raise ValueError("No valid 2D potential frames were loaded.")
    return frames


def add_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#0b1020",
            "savefig.facecolor": "#0b1020",
            "axes.edgecolor": "#d0d7ff",
            "axes.labelcolor": "#f4f7ff",
            "text.color": "#f4f7ff",
            "xtick.color": "#d7deff",
            "ytick.color": "#d7deff",
            "font.size": 13,
            "axes.titleweight": "bold",
        }
    )


def extract_profile(frame: np.ndarray, profile_axis: str, fixed_index: int) -> np.ndarray:
    if profile_axis == "y":
        return frame[fixed_index, :]
    return frame[:, fixed_index]


def main() -> None:
    args = parse_args()
    add_style()

    paths = sorted(Path(".").glob(args.pattern), key=extract_step)
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    steps = [extract_step(path) for path in paths]
    frames = load_frames(paths)
    nx, ny = frames[0].shape

    if args.profile_axis == "y":
        max_fixed = nx - 1
        default_fixed = nx // 2
        x_values = np.arange(ny)
        x_label = "y cell index"
        fixed_label = "x"
    else:
        max_fixed = ny - 1
        default_fixed = ny // 2
        x_values = np.arange(nx)
        x_label = "x cell index"
        fixed_label = "y"

    fixed_index = default_fixed if args.fixed_index is None else args.fixed_index
    if fixed_index < 0 or fixed_index > max_fixed:
        raise SystemExit(
            f"Invalid --fixed-index={fixed_index}. Allowed range: 0..{max_fixed}."
        )

    profiles = [extract_profile(frame, args.profile_axis, fixed_index) for frame in frames]
    all_values = np.concatenate(profiles)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.01, 1.0e-12)
        y_min -= pad
        y_max += pad
    else:
        pad = 0.05 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    fig, ax = plt.subplots(figsize=(10.0, 6.6), constrained_layout=True)
    (line,) = ax.plot(x_values, profiles[0], color="#8bd3ff", lw=2.6)

    ax.set_title("Electric Potential Profile Evolution", fontsize=22, pad=14)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(r"$\phi$ (V)", fontsize=15)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.22, linestyle="--")

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
        bbox={"facecolor": "#111933", "edgecolor": "#8bd3ff", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )
    stat_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox={"facecolor": "#111933", "edgecolor": "#fef08a", "alpha": 0.8, "boxstyle": "round,pad=0.35"},
    )
    slice_text = ax.text(
        0.5,
        1.02,
        f"profile along {args.profile_axis} at fixed {fixed_label} = {fixed_index}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#d7deff",
    )

    def update(frame_idx: int):
        profile = profiles[frame_idx]
        line.set_data(x_values, profile)
        time_text.set_text(f"iteration = {steps[frame_idx]}")
        stat_text.set_text(
            f"min phi = {profile.min():.3e}\nmax phi = {profile.max():.3e}\nmean phi = {profile.mean():.3e}"
        )
        return line, time_text, stat_text, slice_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(profiles),
        interval=1000 / max(args.fps, 1),
        blit=False,
        repeat=True,
    )

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        anim.save(output_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    elif suffix == ".mp4":
        if FFMpegWriter.isAvailable():
            anim.save(
                output_path,
                writer=FFMpegWriter(fps=args.fps, bitrate=2400),
                dpi=args.dpi,
            )
        else:
            fallback_path = output_path.with_suffix(".gif")
            print(
                "ffmpeg is unavailable, saving GIF instead: "
                f"{fallback_path.resolve()}"
            )
            anim.save(fallback_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
            output_path = fallback_path
    else:
        raise SystemExit(
            f"Unsupported output extension: {suffix}. Use .mp4 or .gif."
        )

    print(f"Saved animation to {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
