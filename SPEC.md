# Specification

This project targets the physical scenario implemented in C++ reference
`CPP_2D_PIC_GYRO-main/Tests/SimulationCircleGyroNEW.h` reference code.

## Success criterion

The JAX implementation does not need bitwise equivalence with the C++ code.
It does need to reproduce the same physical scenario and deliver comparable:

- `Ne(t)`
- `Ni(t)`
- `Ntot_ionized(t)`
- `Ntot_cold_cathode_leave(t)`
- `Ntot_anode_leave(t)`
- `Ntot_hot_cathode_emission(t)`
- maps of `rho_e`, `rho_i`, `phi`

## Included in the first target scenario

- gyrokinetic electrons
- ions
- circular anode geometry
- linear charge deposition
- linear field gather
- `rho_filter_new`
- Poisson solve on the circular domain
- hot cathode source
- cold cathode ion sink
- cold secondary emission
- all collision channels from Penning reference:
  - electron-neutral elastic
  - ion-neutral elastic / charge exchange
  - electron impact ionization

## Explicitly excluded for now

- Ag/Pb dynamics
- restart support
- multi-GPU support

## Design constraints

- single GPU, NVIDIA + CUDA
- code should remain compact and readable
- performance target: 70% speed, 30% simplicity
- dynamic particle life cycle must be expressed with fixed-capacity particle
  pools and `alive` masks
