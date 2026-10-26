from __future__ import annotations

import argparse
from dataclasses import replace
import math

from .config import E_M, default_config
from .simulation_circle_gyro_new import run_simulation


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JAX Penning PIC simulation.")
    parser.add_argument(
        "--poisson-solver",
        choices=("direct_inverse", "fft_capacitance", "cg", "jacobi"),
        help="Poisson solver backend to use.",
    )
    parser.add_argument(
        "--disable-rho-filter",
        action="store_true",
        help="Disable smoothing of the deposited total charge density.",
    )
    parser.add_argument("--output-dir", help="Directory where counters and field dumps will be written.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Remove known simulation artifacts from an existing output directory before running.",
    )
    parser.add_argument("--cross-section-dir", help="Directory with collision cross-section .txt files.")
    parser.add_argument("--it-num", type=int, help="Number of simulation steps.")
    parser.add_argument("--grid-size", type=int, help="Poisson grid size in each dimension (N x N, minimum 3).")
    parser.add_argument(
        "--macro-particle-weight",
        type=positive_float,
        help="Number of physical particles represented by one macroparticle.",
    )
    parser.add_argument(
        "--ion-mass-ratio",
        type=positive_float,
        help="Ion-to-electron mass ratio m_ion / m_e (default: 500).",
    )
    parser.add_argument(
        "--ptcls-per-cell",
        type=float,
        help="Legacy compatibility option; ignored because particle pools start empty.",
    )
    parser.add_argument("--log-interval", type=int, help="Particle counter logging interval.")
    parser.add_argument("--field-dump-interval", type=int, help="Field dump interval for rho_e/rho_i/phi txt files.")
    parser.add_argument(
        "--particle-sample-stride",
        type=non_negative_int,
        help="Write roughly one in every N live particles at field dump checkpoints (0 disables).",
    )
    parser.add_argument("--max-electrons", type=int, help="Fixed electron pool capacity.")
    parser.add_argument("--max-ions", type=int, help="Fixed ion pool capacity.")
    parser.add_argument("--trace-dir", help="Directory for JAX trace output.")
    parser.add_argument("--trace-start-step", type=int, default=0, help="Step to start JAX tracing from.")
    parser.add_argument("--trace-num-steps", type=int, default=0, help="How many steps to capture in the JAX trace.")
    parser.add_argument("--profile", action="store_true", help="Enable detailed per-step profiling.")
    parser.add_argument(
        "--profile-summary-only",
        action="store_true",
        help="Collect timing summary without writing per-step profiling rows.",
    )
    args = parser.parse_args()

    config = default_config(
        output_dir=args.output_dir,
        cross_section_dir=args.cross_section_dir,
    )
    replacements = {
        "poisson_solver": args.poisson_solver,
        "it_num": args.it_num,
        "grid_size": args.grid_size,
        "macro_particle_weight": args.macro_particle_weight,
        "m_ion": args.ion_mass_ratio * E_M if args.ion_mass_ratio is not None else None,
        "ptcls_per_cell": args.ptcls_per_cell,
        "log_interval": args.log_interval,
        "field_dump_interval": args.field_dump_interval,
        "particle_sample_stride": args.particle_sample_stride,
        "max_electrons": args.max_electrons,
        "max_ions": args.max_ions,
        "use_rho_filter": False if args.disable_rho_filter else None,
    }
    replacements = {key: value for key, value in replacements.items() if value is not None}
    if replacements:
        config = replace(config, **replacements)

    run_simulation(
        config,
        profile=args.profile,
        profile_summary_only=args.profile_summary_only,
        trace_dir=args.trace_dir,
        trace_start_step=args.trace_start_step,
        trace_num_steps=args.trace_num_steps,
        overwrite_output=args.overwrite_output,
    )


if __name__ == "__main__":
    main()
