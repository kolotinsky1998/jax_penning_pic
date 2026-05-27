# jax-penning-pic

Standalone JAX implementation of the Penning-discharge PIC scenario
(`SimulationCircleGyroNEW`), packaged so it can be moved into its own Git
repository without the full WarpX or C++ reference tree.

## What Is Included

- `jax_penning_pic/` - importable Python package with the simulation kernels.
- `jax_penning_pic/data/cross_sections/` - collision cross-section tables used
  by the default configuration.
- `data/cross_sections/` - the same tables in a convenient top-level location
  for manual inspection and resampling tools.
- `scripts/` - command-line entry points for running and plotting results.
- `tools/analysis/` - animation and comparison utilities for `phi`, `rho_e`,
  `rho_i`, and particle counters.
- `tools/collisions/` - collision-table resampling helper.
- `tests/` - focused numerical and regression tests.
- `examples/` - example configuration snapshot.

Generated outputs are written under `outputs/` by default and are ignored by
Git.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For GPU runs, install the JAX wheel that matches your CUDA setup if the default
`jax`/`jaxlib` packages are not appropriate for your machine.

## Run A Short Smoke Simulation

```bash
python scripts/run_simulation_circle_gyro_new.py \
  --it-num 1000 \
  --field-dump-interval 1000 \
  --log-interval 100
```

Equivalent installed console command:

```bash
jax-penning-run --it-num 1000 --field-dump-interval 1000
```

Default outputs:

```text
outputs/simulation_circle_gyro_new/
  counters.csv
  metadata.json
  rho_e_*.txt
  rho_i_*.txt
  phi_*.txt
```

## Useful Commands

Plot particle counts:

```bash
python scripts/plot_counters.py outputs/simulation_circle_gyro_new/counters.csv
```

Animate electron density:

```bash
python scripts/animate_rho_e.py
```

Animate ion density:

```bash
python scripts/animate_rho_i.py
```

Compare potential profiles from two folders:

```bash
python tools/analysis/animate_phi_profile_comparison.py \
  --reference-dir /path/to/reference/phi_hist \
  --jax-dir outputs/simulation_circle_gyro_new \
  --jax-pattern "phi_[0-9]*.txt" \
  --output phi_profile_comparison.gif
```

## Configuration

The default configuration lives in `jax_penning_pic/config.py`.

Important CLI overrides:

- `--output-dir`
- `--cross-section-dir`
- `--it-num`
- `--ptcls-per-cell`
- `--log-interval`
- `--field-dump-interval`
- `--poisson-solver`
- `--max-electrons`
- `--max-ions`

By default, cross sections are loaded from:

```text
jax_penning_pic/data/cross_sections/
```

## Tests

```bash
python -m pytest tests
```

The current tests cover deposition, Poisson helpers, geometry, collisions,
particle writing, source/sink behavior, and gyro-push behavior.
