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

For the optional graded triangular FEM backend, also install:

```bash
python -m pip install -e ".[fem]"
```

## Run A Short Smoke Simulation

```bash
python scripts/run_simulation_circle_gyro_new.py \
  --it-num 1000 \
  --field-dump-interval 1000 \
  --particle-sample-stride 1000 \
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
  particles_*.vtp
  particles.pvd
```

Particle output is disabled by default.  Set `--particle-sample-stride N` to
write a stable deterministic sample of approximately one in every `N` live
macroparticles whenever fields are dumped.  Open `particles.pvd` in ParaView
to animate the point cloud.  Electron and ion points are combined; point data
contains a persistent `particle_id` (`GlobalIds`), `species` (`0` electron,
`1` ion), `slot_id`, and `generation`.  When a dead particle's pool slot is
reused, the replacement receives a different ID.

## Optional triangular FEM mesh

The original uniform Cartesian finite-difference solver remains the default.
To use FEM, generate and inspect a static mesh first.  The values below use
the simulation's scaled length units; with default physics the anode radius is
`r * scale` and the hot-cathode radius is `0.01 * scale`.

```bash
python scripts/generate_triangular_mesh.py \
  --output-dir outputs/meshes/default \
  --radius 0.0052 --hot-radius 0.0002 \
  --h-hot 0.00001 --h-bulk 0.00008 --transition-width 0.00005
```

This produces `mesh.msh`, `mesh.vtu`, `mesh.npz`, and `mesh_metadata.json`.
Open `mesh.vtu` in ParaView before running:

```bash
jax-penning-run \
  --mesh-mode triangular_fem \
  --mesh-path outputs/meshes/default/mesh.npz \
  --macro-particle-weight 1e7 \
  --poisson-solver cg
```

FEM runs start with empty particle pools and write `fields_*.vtu`, `fields.pvd`
and `poisson.csv`.  `field_dump_interval` controls the VTK cadence.
The triangular mesh boundary is the absorbing anode: particles are removed as
soon as a push takes them outside the FEM domain, before collision handling.

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
- `--disable-rho-filter` disables smoothing of the deposited total charge density
- `--grid-size` (square Poisson grid, `N x N`; minimum `3`)
- `--mesh-mode` (`uniform_cartesian`, default, or `triangular_fem`)
- `--mesh-path` (generated `mesh.npz` used by `triangular_fem`)
- `--macro-particle-weight` (physical particles represented by one macro-particle)
- `--ion-mass-ratio` (ion mass as a multiple of the electron mass; default `500`)
- `--ptcls-per-cell`
- `--log-interval`
- `--field-dump-interval`
- `--particle-sample-stride` (`0` disables particle snapshots; `N > 0` writes approximately `1/N`)
- `--poisson-solver`
- `--max-electrons`
- `--max-ions`
- `--validate-particle-domain` checks at logging checkpoints that every live FEM particle is inside the mesh
- `--overwrite-output` removes only known simulation artifacts before reusing an output directory

Existing simulation output is never appended to silently.  Use a fresh
`--output-dir` for each run, or opt in explicitly with `--overwrite-output`.

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
