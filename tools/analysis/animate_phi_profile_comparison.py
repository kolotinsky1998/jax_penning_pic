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
        description="Animate and compare electric-potential line profiles from two folders."
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory with Penning reference phi files, e.g. phi1660000.txt",
    )
    parser.add_argument(
        "--jax-dir",
        required=True,
        help="Directory with JAX phi files, e.g. phi_1660000.txt",
    )
    parser.add_argument(
        "--reference-pattern",
        default="phi[0-9]*.txt",
        help="Glob pattern for Penning reference files. Default: %(default)s",
    )
    parser.add_argument(
        "--jax-pattern",
        default="phi_[0-9]*.txt",
        help="Glob pattern for JAX files. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="phi_profile_comparison.mp4",
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
        "--reference-label",
        default="Penning reference",
        help="Legend label for the Penning reference profile. Default: %(default)s",
    )
    parser.add_argument(
        "--jax-label",
        default="JAX",
        help="Legend label for the JAX profile. Default: %(default)s",
    )
    parser.add_argument(
        "--reference-color",
        default="#f59e0b",
        help="Line color for the Penning reference profile. Default: %(default)s",
    )
    parser.add_argument(
        "--jax-color",
        default="#60a5fa",
        help="Line color for the JAX profile. Default: %(default)s",
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
    frame_map: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob(pattern), key=extract_step):
        step = extract_step(path)
        if step < 0:
            continue
        frame = load_frame(path)
        if frame is not None:
            frame_map[step] = frame
    return frame_map


def extract_profile(frame: np.ndarray, profile_axis: str, fixed_index: int) -> np.ndarray:
    if profile_axis == "y":
        return frame[fixed_index, :]
    return frame[:, fixed_index]


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


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int) -> Path:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
        return output_path
    if suffix == ".mp4":
        if FFMpegWriter.isAvailable():
            anim.save(
                output_path,
                writer=FFMpegWriter(fps=fps, bitrate=2400),
                dpi=dpi,
            )
            return output_path
        fallback_path = output_path.with_suffix(".gif")
        print(f"ffmpeg is unavailable, saving GIF instead: {fallback_path.resolve()}")
        anim.save(fallback_path, writer=PillowWriter(fps=fps), dpi=dpi)
        return fallback_path
    raise SystemExit(f"Unsupported output extension: {suffix}. Use .mp4 or .gif.")


def main() -> None:
    args = parse_args()
    add_style()

    reference_dir = Path(args.reference_dir)
    jax_dir = Path(args.jax_dir)
    reference_frames = load_frame_map(reference_dir, args.reference_pattern)
    jax_frames = load_frame_map(jax_dir, args.jax_pattern)

    if not reference_frames:
        raise SystemExit(f"No valid Penning reference frames matched {args.reference_pattern} in {reference_dir}")
    if not jax_frames:
        raise SystemExit(f"No valid JAX frames matched {args.jax_pattern} in {jax_dir}")

    common_steps = sorted(set(reference_frames) & set(jax_frames))
    if not common_steps:
        raise SystemExit("No common iteration numbers were found between Penning reference and JAX folders.")

    reference_shape = next(iter(reference_frames.values())).shape
    jax_shape = next(iter(jax_frames.values())).shape
    if reference_shape != jax_shape:
        raise SystemExit(
            f"Shape mismatch: Penning reference frames have {reference_shape}, JAX frames have {jax_shape}."
        )

    nx, ny = reference_shape
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
        raise SystemExit(f"Invalid --fixed-index={fixed_index}. Allowed range: 0..{max_fixed}.")

    reference_profiles = [extract_profile(reference_frames[step], args.profile_axis, fixed_index) for step in common_steps]
    jax_profiles = [extract_profile(jax_frames[step], args.profile_axis, fixed_index) for step in common_steps]

    all_values = np.concatenate(reference_profiles + jax_profiles)
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

    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    (reference_line,) = ax.plot(x_values, reference_profiles[0], color=args.reference_color, lw=2.6, label=args.reference_label)
    (jax_line,) = ax.plot(x_values, jax_profiles[0], color=args.jax_color, lw=2.6, label=args.jax_label)

    ax.set_title("Electric Potential Profile Comparison", fontsize=22, pad=14)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(r"$\phi$ (V)", fontsize=15)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.85)

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
        reference_profile = reference_profiles[frame_idx]
        jax_profile = jax_profiles[frame_idx]
        reference_line.set_data(x_values, reference_profile)
        jax_line.set_data(x_values, jax_profile)
        step = common_steps[frame_idx]
        time_text.set_text(f"iteration = {step}")
        diff = jax_profile - reference_profile
        stat_text.set_text(
            f"max |Δphi| = {np.max(np.abs(diff)):.3e}\n"
            f"mean Δphi = {np.mean(diff):.3e}\n"
            f"RMS Δphi = {np.sqrt(np.mean(diff * diff)):.3e}"
        )
        return reference_line, jax_line, time_text, stat_text, slice_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(common_steps),
        interval=1000 / max(args.fps, 1),
        blit=False,
        repeat=True,
    )

    output_path = save_animation(anim, Path(args.output), args.fps, args.dpi)
    print(f"Saved animation to {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
