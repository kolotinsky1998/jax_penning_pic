#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#ca8a04")


@dataclass(frozen=True)
class Case:
    mass_ratio: float
    directory: Path
    ion_step: int

    @property
    def label(self) -> str:
        return rf"$m_i/m_e={self.mass_ratio:g}$"


def case_argument(value: str) -> tuple[float, Path]:
    try:
        mass_text, directory_text = value.split("=", 1)
        mass_ratio = float(mass_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MASS=DIR, for example 500=outputs/ion_mass_scan/mi_500_me") from exc
    if not math.isfinite(mass_ratio) or mass_ratio <= 0.0 or not directory_text:
        raise argparse.ArgumentTypeError("MASS must be positive and DIR must not be empty")
    return mass_ratio, Path(directory_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate potential profiles for selected ion-mass scan cases. "
            "Repeat --case to choose which mass results are included."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=case_argument,
        required=True,
        metavar="MASS=DIR",
        help="Ion/electron mass ratio and its output directory. May be repeated.",
    )
    parser.add_argument(
        "--pattern",
        default="phi_[0-9]*.txt",
        help="Glob pattern used in every selected directory. Default: %(default)s",
    )
    parser.add_argument(
        "--time-axis",
        choices=("ion-updates", "step"),
        default="ion-updates",
        help="Synchronize files by step/ion_step or by raw simulation step. Default: %(default)s",
    )
    parser.add_argument(
        "--gyro-coeff",
        type=float,
        default=100.0,
        help="gyro_coeff used to derive ion_step=round(MASS/gyro_coeff). Default: %(default)s",
    )
    parser.add_argument("--start-time", type=float, default=None, help="First synchronized time value to include.")
    parser.add_argument("--end-time", type=float, default=None, help="Last synchronized time value to include.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every N-th common file after time filtering. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="phi_profile_ion_mass_scan.mp4",
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
        help="Relative position on the orthogonal axis, from 0 to 1. Default: center line.",
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


def synchronized_time(raw_step: int, case: Case, time_axis: str) -> float:
    if time_axis == "step":
        return float(raw_step)
    return raw_step / case.ion_step


def load_frames(case: Case, pattern: str, time_axis: str) -> dict[float, np.ndarray]:
    frames: dict[float, np.ndarray] = {}
    for path in sorted(case.directory.glob(pattern), key=extract_step):
        raw_step = extract_step(path)
        if raw_step < 0 or path.stat().st_size == 0:
            continue
        frame = np.loadtxt(path)
        if frame.ndim != 2:
            print(f"Skipping non-2D file: {path}")
            continue
        time_value = synchronized_time(raw_step, case, time_axis)
        if time_value in frames:
            raise SystemExit(f"Duplicate synchronized time {time_value:g} in {case.directory}")
        frames[time_value] = frame
    return frames


def extract_profile(frame: np.ndarray, profile_axis: str, fixed_fraction: float) -> np.ndarray:
    if profile_axis == "y":
        fixed_index = round(fixed_fraction * (frame.shape[0] - 1))
        return frame[fixed_index, :]
    fixed_index = round(fixed_fraction * (frame.shape[1] - 1))
    return frame[:, fixed_index]


def interpolate_profile(profile: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    native_coordinate = np.linspace(-1.0, 1.0, profile.size)
    return np.interp(coordinate, native_coordinate, profile)


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


def save_animation(animation: FuncAnimation, output_path: Path, fps: int, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
        return output_path
    if output_path.suffix.lower() == ".mp4":
        if FFMpegWriter.isAvailable():
            animation.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=dpi)
            return output_path
        fallback = output_path.with_suffix(".gif")
        print(f"ffmpeg is unavailable, saving GIF instead: {fallback.resolve()}")
        animation.save(fallback, writer=PillowWriter(fps=fps), dpi=dpi)
        return fallback
    raise SystemExit("Unsupported output extension. Use .mp4 or .gif.")


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.gyro_coeff) or args.gyro_coeff <= 0.0:
        raise SystemExit("--gyro-coeff must be positive.")
    if not 0.0 <= args.fixed_fraction <= 1.0:
        raise SystemExit("--fixed-fraction must be between 0 and 1.")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1.")
    if args.samples is not None and args.samples < 2:
        raise SystemExit("--samples must be at least 2.")

    mass_ratios = [mass for mass, _ in args.case]
    if len(set(mass_ratios)) != len(mass_ratios):
        raise SystemExit("Each MASS value may be specified only once.")
    cases = [
        Case(mass, directory, max(1, round(mass / args.gyro_coeff)))
        for mass, directory in args.case
    ]
    frame_maps = {case: load_frames(case, args.pattern, args.time_axis) for case in cases}
    for case, frames in frame_maps.items():
        if not frames:
            raise SystemExit(f"No valid files matched {args.pattern} in {case.directory}")

    common_times = sorted(set.intersection(*(set(frames) for frames in frame_maps.values())))
    if args.start_time is not None:
        common_times = [value for value in common_times if value >= args.start_time]
    if args.end_time is not None:
        common_times = [value for value in common_times if value <= args.end_time]
    common_times = common_times[:: args.frame_stride]
    if not common_times:
        raise SystemExit("No common files remain after synchronization and time filtering.")

    first_profiles = {
        case: extract_profile(frame_maps[case][common_times[0]], args.profile_axis, args.fixed_fraction)
        for case in cases
    }
    sample_count = args.samples or max(profile.size for profile in first_profiles.values())
    coordinate = np.linspace(-1.0, 1.0, sample_count)
    profiles = {
        case: [
            interpolate_profile(
                extract_profile(frame_maps[case][time_value], args.profile_axis, args.fixed_fraction),
                coordinate,
            )
            for time_value in common_times
        ]
        for case in cases
    }

    all_values = np.concatenate([profile for values in profiles.values() for profile in values])
    y_min, y_max = float(np.min(all_values)), float(np.max(all_values))
    padding = max(0.05 * (y_max - y_min), abs(y_min) * 0.01, 1.0e-12)

    add_style()
    fig, ax = plt.subplots(figsize=(11.2, 7.0), constrained_layout=True)
    lines = {}
    for index, case in enumerate(cases):
        (lines[case],) = ax.plot(
            coordinate,
            profiles[case][0],
            color=COLORS[index % len(COLORS)],
            lw=2.2,
            label=case.label,
        )
    ax.set_xlabel(f"normalized {args.profile_axis} coordinate", fontsize=15)
    ax.set_ylabel(r"$\phi$ (V)", fontsize=15)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.grid(True, color="#d0d0d0", alpha=0.8, linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", framealpha=0.9, ncols=2 if len(cases) > 4 else 1)

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

    def update(frame_index: int):
        for case, line in lines.items():
            line.set_data(coordinate, profiles[case][frame_index])
        value = common_times[frame_index]
        label = "ion updates" if args.time_axis == "ion-updates" else "step"
        time_text.set_text(f"{label} = {value:g}")
        return (*lines.values(), time_text)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(common_times),
        interval=1000 / max(args.fps, 1),
        blit=False,
        repeat=True,
    )
    output_path = save_animation(animation, Path(args.output), args.fps, args.dpi)
    print(f"Selected mass ratios: {', '.join(f'{case.mass_ratio:g}' for case in cases)}")
    print(f"Common frames used: {len(common_times)}")
    print(f"Saved animation to {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
