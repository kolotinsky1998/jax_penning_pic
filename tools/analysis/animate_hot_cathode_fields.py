#!/usr/bin/env python3
"""Animate plasma quantities in the hot-cathode region of a Penning run."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm


ELEMENTARY_CHARGE = 1.6021766208e-19
ELECTRON_MASS = 9.10938356e-31
EV_TO_KELVIN = 11604.52500617
EFFECTIVE_DEPTH_M = 1.0
FIELD_FILE_RE = re.compile(r"^(rho_e|rho_i|phi)_(\d+)\.txt$")
QUANTITY_KEYS = ("n_e", "n_i", "rho", "e_abs", "phi")
TITLES = {
    "n_e": r"Концентрация электронов $n_e$",
    "n_i": r"Концентрация ионов $n_i$",
    "rho": r"Общая зарядовая плотность $\rho$",
    "e_abs": r"Модуль электрического поля $|E|$",
    "phi": r"Электрический потенциал $\varphi$",
}
UNITS = {
    "n_e": r"м$^{-3}$",
    "n_i": r"м$^{-3}$",
    "rho": r"Кл/м$^3$",
    "e_abs": "В/м",
    "phi": "В",
}
CMAPS = {
    "n_e": "magma",
    "n_i": "viridis",
    "rho": "coolwarm",
    "e_abs": "plasma",
    "phi": "coolwarm",
}


@dataclass(frozen=True)
class Geometry:
    size: int
    dx: float
    dy: float
    x_center: float
    y_center: float
    hot_radius: float
    dt: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return parsed


def percentile_value(value: str) -> float:
    parsed = float(value)
    if not 50.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("значение должно находиться в диапазоне 50..100")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Построить сводные анимации концентраций и полей в области "
            "горячего катода. По умолчанию данные читаются из текущей папки."
        )
    )
    parser.add_argument("--fps", type=positive_int, default=8, help="Частота кадров. По умолчанию: %(default)s")
    parser.add_argument("--dpi", type=positive_int, default=160, help="Разрешение вывода. По умолчанию: %(default)s")
    parser.add_argument(
        "--percentile",
        type=percentile_value,
        default=99.5,
        help="Процентиль устойчивых шкал, 50..100. По умолчанию: %(default)s",
    )
    parser.add_argument(
        "--output-prefix",
        default="hot_cathode",
        help="Префикс выходных файлов. По умолчанию: %(default)s",
    )
    parser.add_argument(
        "--profile-axis",
        choices=("x", "y"),
        default="x",
        help="Ось профиля через центр катода. По умолчанию: %(default)s",
    )
    parser.add_argument("--show", action="store_true", help="Показать окно после сохранения.")
    return parser.parse_args()


def load_metadata(directory: Path) -> dict:
    path = directory / "metadata.json"
    if not path.is_file():
        raise ValueError(f"Не найден файл метаданных: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc
    if not isinstance(metadata.get("config"), dict):
        raise ValueError(f"В {path} отсутствует объект config")
    return metadata


def discover_frames(directory: Path) -> tuple[list[int], dict[str, dict[int, Path]]]:
    files: dict[str, dict[int, Path]] = {name: {} for name in ("rho_e", "rho_i", "phi")}
    for path in directory.iterdir():
        match = FIELD_FILE_RE.match(path.name)
        if match:
            files[match.group(1)][int(match.group(2))] = path

    missing_types = [name for name, paths in files.items() if not paths]
    if missing_types:
        raise ValueError("Не найдены кадры: " + ", ".join(f"{name}_<step>.txt" for name in missing_types))

    step_sets = {name: set(paths) for name, paths in files.items()}
    all_steps = set.union(*step_sets.values())
    incomplete = {
        name: sorted(all_steps - steps)
        for name, steps in step_sets.items()
        if steps != all_steps
    }
    if incomplete:
        details = "; ".join(f"{name}: нет {values[:8]}" for name, values in incomplete.items())
        raise ValueError(f"Наборы кадров не синхронизированы ({details})")
    return sorted(all_steps), files


def load_matrix(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    if path.stat().st_size == 0:
        raise ValueError(f"Пустой файл: {path}")
    try:
        matrix = np.loadtxt(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{path} не является двумерной матрицей")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Ожидалась квадратная сетка, получена {matrix.shape} в {path}")
    if expected_shape is not None and matrix.shape != expected_shape:
        raise ValueError(f"Форма {path} равна {matrix.shape}, ожидалась {expected_shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path} содержит NaN или бесконечные значения")
    return matrix


def debye_grid_spacing(config: dict) -> float:
    scale = float(config["scale"])
    n_e_scaled = float(config["n_e"]) / scale
    n_i_scaled = float(config["n_i"]) / scale
    t_e_kelvin = float(config["t_e_ev"]) * EV_TO_KELVIN
    t_i_kelvin = float(config["t_i_ev"]) * EV_TO_KELVIN
    k_cgs = 1.38e-16
    q_cgs = 4.8e-10
    denominator = (
        4.0 * math.pi * q_cgs**2 * (n_i_scaled / 1.0e6) / (k_cgs * t_i_kelvin)
        + 4.0 * math.pi * q_cgs**2 * (n_e_scaled / 1.0e6) / (k_cgs * t_e_kelvin)
    )
    return 2.0 * denominator ** -0.5 / 100.0


def time_step(metadata: dict) -> float:
    derived = metadata.get("derived", {})
    if "dt" in derived:
        dt = float(derived["dt"])
    else:
        config = metadata["config"]
        scale = float(config["scale"])
        magnetic_field = float(config["b"]) / scale
        dt = 1.0 / (ELEMENTARY_CHARGE * magnetic_field / ELECTRON_MASS) / 10.0
        dt *= int(config["gyro_coeff"])
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Некорректный шаг времени dt={dt}")
    return dt


def reconstruct_geometry(metadata: dict, actual_size: int) -> Geometry:
    config = metadata["config"]
    try:
        scale = float(config["scale"])
        model_radius = float(config["r"]) * scale
        hot_radius = float(config.get("hot_cathode_radius", 0.01)) * scale
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "В metadata.json отсутствуют корректные config.scale, config.r или config.hot_cathode_radius"
        ) from exc
    if scale <= 0.0 or model_radius <= 0.0 or hot_radius <= 0.0:
        raise ValueError("config.scale, config.r и config.hot_cathode_radius должны быть положительными")

    configured_size = config.get("grid_size")
    if configured_size is not None:
        configured_size = int(configured_size)
        if configured_size != actual_size:
            raise ValueError(
                f"metadata.json задаёт grid_size={configured_size}, "
                f"но файлы содержат сетку {actual_size}x{actual_size}"
            )
        dx = 2.0 * model_radius / configured_size
    else:
        try:
            dx = debye_grid_spacing(config)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("Не удалось восстановить автоматически рассчитанный шаг сетки") from exc
        reconstructed_size = int((2.0 * model_radius) / dx)
        if reconstructed_size != actual_size:
            raise ValueError(
                "Автоматически восстановленный размер сетки "
                f"{reconstructed_size} не совпадает с данными {actual_size}"
            )

    return Geometry(
        size=actual_size,
        dx=dx,
        dy=dx,
        x_center=0.5 * actual_size * dx,
        y_center=0.5 * actual_size * dx,
        hot_radius=hot_radius,
        dt=time_step(metadata),
    )


def calculate_quantities(rho_e: np.ndarray, rho_i: np.ndarray, phi: np.ndarray, geometry: Geometry) -> dict[str, np.ndarray]:
    edge_order = 2 if geometry.size >= 3 else 1
    dphi_dx, dphi_dy = np.gradient(phi, geometry.dx, geometry.dy, edge_order=edge_order)
    ex = -dphi_dx
    ey = -dphi_dy
    return {
        "n_e": -rho_e / (ELEMENTARY_CHARGE * EFFECTIVE_DEPTH_M),
        "n_i": rho_i / (ELEMENTARY_CHARGE * EFFECTIVE_DEPTH_M),
        "rho": (rho_e + rho_i) / EFFECTIVE_DEPTH_M,
        "e_abs": np.hypot(ex, ey),
        "phi": phi,
    }


def hot_region(geometry: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_relative = np.arange(geometry.size, dtype=float) * geometry.dx - geometry.x_center
    y_relative = np.arange(geometry.size, dtype=float) * geometry.dy - geometry.y_center
    x_indices = np.flatnonzero(np.abs(x_relative) <= geometry.hot_radius)
    y_indices = np.flatnonzero(np.abs(y_relative) <= geometry.hot_radius)
    if x_indices.size < 2 or y_indices.size < 2:
        raise ValueError(
            "Горячий катод недостаточно разрешён сеткой: внутри его диаметра "
            "должно находиться не менее двух узлов по каждой оси"
        )
    xx, yy = np.meshgrid(x_relative[x_indices], y_relative[y_indices], indexing="ij")
    mask = xx**2 + yy**2 <= geometry.hot_radius**2
    if not np.any(mask):
        raise ValueError("В области горячего катода нет узлов сетки")
    return x_indices, y_indices, mask


def centered_profile(values: np.ndarray, geometry: Geometry, axis: str) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(geometry.size, dtype=float) * geometry.dx
    center = geometry.y_center if axis == "x" else geometry.x_center
    upper = int(np.searchsorted(coordinates, center, side="left"))
    upper = min(max(upper, 0), geometry.size - 1)
    lower = max(upper - 1, 0)
    if upper == lower or coordinates[upper] == coordinates[lower]:
        alpha = 0.0
    else:
        alpha = (center - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
    if axis == "x":
        profile = (1.0 - alpha) * values[:, lower] + alpha * values[:, upper]
        relative = coordinates - geometry.x_center
    else:
        profile = (1.0 - alpha) * values[lower, :] + alpha * values[upper, :]
        relative = coordinates - geometry.y_center
    inside = np.abs(relative) <= geometry.hot_radius
    return relative[inside], profile[inside]


def crop_and_mask(values: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    cropped = values[np.ix_(x_indices, y_indices)]
    return np.ma.array(cropped, mask=~mask)


def safe_limit(values: np.ndarray, percentile: float, *, absolute: bool = False) -> float:
    finite = values[np.isfinite(values)]
    if absolute:
        finite = np.abs(finite)
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, percentile))
    return max(limit, np.finfo(float).tiny)


def build_norms(samples: dict[str, list[np.ndarray]], percentile: float) -> dict[str, Normalize]:
    combined = {key: np.concatenate(values) for key, values in samples.items()}
    norms: dict[str, Normalize] = {}
    lower_percentile = max(0.0, 100.0 - percentile)
    for key in ("n_e", "n_i"):
        positive = combined[key][combined[key] > 0.0]
        if positive.size == 0:
            norms[key] = LogNorm(vmin=1.0, vmax=10.0)
        else:
            vmin = max(float(np.percentile(positive, lower_percentile)), np.finfo(float).tiny)
            vmax = max(float(np.percentile(positive, percentile)), vmin * 10.0)
            norms[key] = LogNorm(vmin=vmin, vmax=vmax)

    norms["rho"] = Normalize(
        vmin=-safe_limit(combined["rho"], percentile, absolute=True),
        vmax=safe_limit(combined["rho"], percentile, absolute=True),
    )
    norms["e_abs"] = Normalize(vmin=0.0, vmax=safe_limit(combined["e_abs"], percentile))
    phi = combined["phi"]
    phi_min = float(np.percentile(phi, lower_percentile))
    phi_max = float(np.percentile(phi, percentile))
    if np.isclose(phi_min, phi_max):
        pad = max(abs(phi_min) * 0.01, 1.0e-12)
        phi_min -= pad
        phi_max += pad
    if phi_min < 0.0 < phi_max:
        norms["phi"] = TwoSlopeNorm(vmin=phi_min, vcenter=0.0, vmax=phi_max)
    else:
        norms["phi"] = Normalize(vmin=phi_min, vmax=phi_max)
    return norms


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )


def save_animation(animation: FuncAnimation, output: Path, fps: int, dpi: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if FFMpegWriter.isAvailable():
        animation.save(output, writer=FFMpegWriter(fps=fps, bitrate=3200), dpi=dpi)
        return output
    fallback = output.with_suffix(".gif")
    print(f"ffmpeg недоступен; сохраняется GIF: {fallback.resolve()}")
    animation.save(fallback, writer=PillowWriter(fps=fps), dpi=dpi)
    return fallback


def main() -> None:
    args = parse_args()
    if not args.show:
        # Saving is commonly run on a headless compute node.  The native macOS
        # backend can terminate before rendering when no GUI session is present.
        plt.switch_backend("Agg")
    directory = Path.cwd()
    try:
        metadata = load_metadata(directory)
        steps, frame_paths = discover_frames(directory)
        first = load_matrix(frame_paths["phi"][steps[0]])
        geometry = reconstruct_geometry(metadata, first.shape[0])
        expected_shape = first.shape
        x_indices, y_indices, hot_mask = hot_region(geometry)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    x_relative = np.arange(geometry.size) * geometry.dx - geometry.x_center
    y_relative = np.arange(geometry.size) * geometry.dy - geometry.y_center
    extent_mm = (
        (x_relative[x_indices[0]] - 0.5 * geometry.dx) * 1.0e3,
        (x_relative[x_indices[-1]] + 0.5 * geometry.dx) * 1.0e3,
        (y_relative[y_indices[0]] - 0.5 * geometry.dy) * 1.0e3,
        (y_relative[y_indices[-1]] + 0.5 * geometry.dy) * 1.0e3,
    )

    @lru_cache(maxsize=2)
    def quantities_for_step(step: int) -> dict[str, np.ndarray]:
        rho_e = load_matrix(frame_paths["rho_e"][step], expected_shape)
        rho_i = load_matrix(frame_paths["rho_i"][step], expected_shape)
        phi = load_matrix(frame_paths["phi"][step], expected_shape)
        return calculate_quantities(rho_e, rho_i, phi, geometry)

    print(f"Найдено кадров: {len(steps)}; сетка: {geometry.size}x{geometry.size}; dx={geometry.dx:.6e} м")
    samples: dict[str, list[np.ndarray]] = {key: [] for key in QUANTITY_KEYS}
    try:
        for step in steps:
            quantities = quantities_for_step(step)
            for key in QUANTITY_KEYS:
                cropped = quantities[key][np.ix_(x_indices, y_indices)]
                samples[key].append(cropped[hot_mask])
        norms = build_norms(samples, args.percentile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    configure_style()
    first_quantities = quantities_for_step(steps[0])
    prefix = Path(args.output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")

    heatmap_fig, heatmap_axes = plt.subplots(2, 3, figsize=(16.5, 10.2), constrained_layout=True)
    heatmap_axes_flat = heatmap_axes.ravel()
    images = {}
    for axis, key in zip(heatmap_axes_flat, QUANTITY_KEYS):
        displayed = crop_and_mask(first_quantities[key], x_indices, y_indices, hot_mask)
        images[key] = axis.imshow(
            displayed.T,
            origin="lower",
            extent=extent_mm,
            cmap=CMAPS[key],
            norm=norms[key],
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(TITLES[key])
        axis.set_xlabel("x − x₀, мм")
        axis.set_ylabel("y − y₀, мм")
        heatmap_fig.colorbar(images[key], ax=axis, shrink=0.86, label=UNITS[key])
    heatmap_axes_flat[-1].set_visible(False)
    heatmap_title = heatmap_fig.suptitle("")

    def update_heatmaps(frame_index: int):
        step = steps[frame_index]
        quantities = quantities_for_step(step)
        for key in QUANTITY_KEYS:
            images[key].set_data(crop_and_mask(quantities[key], x_indices, y_indices, hot_mask).T)
        heatmap_title.set_text(f"Горячий катод: шаг {step},  t = {step * geometry.dt:.6e} с")
        return (*images.values(), heatmap_title)

    heatmap_animation = FuncAnimation(
        heatmap_fig,
        update_heatmaps,
        frames=len(steps),
        interval=1000.0 / args.fps,
        blit=False,
        repeat=True,
    )
    heatmap_output = prefix.parent / f"{prefix.name}_heatmaps.mp4"
    saved_heatmap = save_animation(heatmap_animation, heatmap_output, args.fps, args.dpi)
    print(f"Сохранено: {saved_heatmap.resolve()}")

    profile_fig, profile_axes = plt.subplots(2, 3, figsize=(16.5, 10.2), constrained_layout=True)
    profile_axes_flat = profile_axes.ravel()
    lines = {}
    profile_coordinate, _ = centered_profile(first_quantities["phi"], geometry, args.profile_axis)
    profile_coordinate_mm = profile_coordinate * 1.0e3
    for axis, key in zip(profile_axes_flat, QUANTITY_KEYS):
        _, profile = centered_profile(first_quantities[key], geometry, args.profile_axis)
        plotted = np.where(profile > 0.0, profile, np.nan) if key in ("n_e", "n_i") else profile
        (lines[key],) = axis.plot(profile_coordinate_mm, plotted, lw=2.0)
        axis.set_title(TITLES[key])
        axis.set_xlabel(f"{args.profile_axis} − {args.profile_axis}₀, мм")
        axis.set_ylabel(UNITS[key])
        axis.set_xlim(profile_coordinate_mm[0], profile_coordinate_mm[-1])
        axis.grid(True, alpha=0.3, linestyle="--")
        if key in ("n_e", "n_i"):
            axis.set_yscale("log")
        axis.set_ylim(norms[key].vmin, norms[key].vmax)
    profile_axes_flat[-1].set_visible(False)
    profile_title = profile_fig.suptitle("")

    def update_profiles(frame_index: int):
        step = steps[frame_index]
        quantities = quantities_for_step(step)
        for key in QUANTITY_KEYS:
            _, profile = centered_profile(quantities[key], geometry, args.profile_axis)
            if key in ("n_e", "n_i"):
                profile = np.where(profile > 0.0, profile, np.nan)
            lines[key].set_ydata(profile)
        profile_title.set_text(
            f"Профиль вдоль оси {args.profile_axis}: шаг {step},  t = {step * geometry.dt:.6e} с"
        )
        return (*lines.values(), profile_title)

    profile_animation = FuncAnimation(
        profile_fig,
        update_profiles,
        frames=len(steps),
        interval=1000.0 / args.fps,
        blit=False,
        repeat=True,
    )
    profile_output = prefix.parent / f"{prefix.name}_profiles.mp4"
    saved_profile = save_animation(profile_animation, profile_output, args.fps, args.dpi)
    print(f"Сохранено: {saved_profile.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(heatmap_fig)
        plt.close(profile_fig)


if __name__ == "__main__":
    main()
