#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate electric-potential maps from phi*.txt files."
    )
    parser.add_argument(
        "--pattern",
        default="phi[0-9]*.txt",
        help="Glob pattern for input files. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="phi_evolution.mp4",
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
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap. Default: %(default)s",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Robust color-scale percentile. Default: %(default)s",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use a symmetric color scale around zero.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the animation window after saving.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Optional circle radius in cell units to draw on top of the map.",
    )
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames = []
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


def compute_norm(frames: list[np.ndarray], percentile: float, symmetric: bool):
    values = np.concatenate([frame.ravel() for frame in frames])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0)

    percentile = float(np.clip(percentile, 50.0, 100.0))
    if symmetric:
        limit = np.percentile(np.abs(finite), percentile)
        limit = max(float(limit), 1.0e-12)
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    vmin = float(np.percentile(finite, 100.0 - percentile))
    vmax = float(np.percentile(finite, percentile))
    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.01, 1.0e-12)
        vmin -= pad
        vmax += pad
    return Normalize(vmin=vmin, vmax=vmax)


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


def main() -> None:
    args = parse_args()
    add_style()

    paths = sorted(Path(".").glob(args.pattern), key=extract_step)
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    steps = [extract_step(path) for path in paths]
    frames = load_frames(paths)
    ny, nx = frames[0].shape
    norm = compute_norm(frames, args.percentile, args.symmetric)

    fig, ax = plt.subplots(figsize=(9.5, 8.4), constrained_layout=True)
    image = ax.imshow(
        frames[0],
        origin="lower",
        cmap=args.cmap,
        norm=norm,
        interpolation="bicubic",
    )

    ax.set_title("Electric Potential Evolution", fontsize=22, pad=14)
    ax.set_xlabel("x cell index", fontsize=15)
    ax.set_ylabel("z cell index", fontsize=15)
    ax.set_aspect("equal")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if args.radius is not None:
        ax.add_patch(
            Circle(
                ((nx - 1) / 2.0, (ny - 1) / 2.0),
                radius=args.radius,
                fill=False,
                lw=1.5,
                ls="--",
                ec="#fef08a",
                alpha=0.85,
            )
        )

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
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

    cbar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.94)
    cbar.set_label(r"$\phi$ (V)", fontsize=14)

    def update(frame_idx: int):
        frame = frames[frame_idx]
        image.set_data(frame)
        time_text.set_text(f"iteration = {steps[frame_idx]}")
        stat_text.set_text(
            f"min phi = {frame.min():.3e}\nmax phi = {frame.max():.3e}\nmean phi = {frame.mean():.3e}"
        )
        return image, time_text, stat_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
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
