from __future__ import annotations

import argparse
from dataclasses import replace

from .config import default_config
from .simulation_circle_gyro_new import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JAX Penning PIC simulation.")
    parser.add_argument(
        "--poisson-solver",
        choices=("direct_inverse", "fft_capacitance", "cg", "jacobi"),
        help="Poisson solver backend to use.",
    )
    parser.add_argument("--output-dir", help="Directory where counters and field dumps will be written.")
    parser.add_argument("--cross-section-dir", help="Directory with collision cross-section .txt files.")
    parser.add_argument("--it-num", type=int, help="Number of simulation steps.")
    parser.add_argument("--ptcls-per-cell", type=float, help="Initial macroparticles per grid cell.")
    parser.add_argument("--log-interval", type=int, help="Particle counter logging interval.")
    parser.add_argument("--field-dump-interval", type=int, help="Field dump interval for rho_e/rho_i/phi txt files.")
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
        "ptcls_per_cell": args.ptcls_per_cell,
        "log_interval": args.log_interval,
        "field_dump_interval": args.field_dump_interval,
        "max_electrons": args.max_electrons,
        "max_ions": args.max_ions,
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
    )


if __name__ == "__main__":
    main()
