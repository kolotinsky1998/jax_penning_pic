# Tools

- `analysis/` contains plotting, animation, and comparison scripts for field
  dumps and particle counters.
- `collisions/` contains helper scripts for inspecting or resampling collision
  cross-section tables.

These scripts are intentionally kept outside the importable package so the core
simulation code stays small and easy to reuse.

## Hot-cathode field animations

Run `analysis/animate_hot_cathode_fields.py` from a simulation output directory
containing `metadata.json` and matching `rho_e_*`, `rho_i_*`, and `phi_*` dumps:

```bash
python /path/to/jax_penning_pic/tools/analysis/animate_hot_cathode_fields.py
```

The script writes `hot_cathode_heatmaps.mp4` and
`hot_cathode_profiles.mp4`. Use `--help` to configure the frame rate, output
prefix, robust color percentile, or profile axis. Electron and ion dumps are
converted to number concentrations; their signed sum is shown separately as
the total charge density. The 2-D model is interpreted with an effective depth
of one metre when labelling volume-density units.
