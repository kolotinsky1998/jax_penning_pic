#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "analysis" / "animate_rho_i_evolution.py"
    output_dir = root / "outputs" / "simulation_circle_gyro_new"
    subprocess.run([sys.executable, str(script), "--pattern", str(output_dir / "rho_i_*.txt")], check=False)


if __name__ == "__main__":
    main()
