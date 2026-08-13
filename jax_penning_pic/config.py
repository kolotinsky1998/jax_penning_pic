from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


E_M = 9.10938356e-31
EV = 1.6021766208e-19
K_B = 1.380649e-23
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROSS_SECTION_DIR = Path(__file__).resolve().parent / "data" / "cross_sections"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "simulation_circle_gyro_new"


def debye_radius(n_e: float, n_i: float, t_e_kelvin: float, t_i_kelvin: float) -> float:
    k_cgs = 1.38e-16
    q_cgs = 4.8e-10
    n_e_cgs = n_e / 1e6
    n_i_cgs = n_i / 1e6
    denom = (
        4.0 * math.pi * q_cgs * q_cgs * n_i_cgs / (k_cgs * t_i_kelvin)
        + 4.0 * math.pi * q_cgs * q_cgs * n_e_cgs / (k_cgs * t_e_kelvin)
    )
    return denom ** (-0.5) / 100.0


@dataclass(frozen=True)
class CrossSectionPaths:
    electron_elastic: str
    electron_ionization: str
    ion_elastic: str


@dataclass(frozen=True)
class SimulationConfig:
    scale: float
    gyro_coeff: int
    it_num: int
    ptcls_per_cell: float
    r: float
    b: float
    n_e: float
    n_i: float
    t_e_ev: float
    t_i_ev: float
    i_hot: float
    i_cold: float
    p_pa: float
    t_gas: float
    m_ion: float
    seed: int
    init_energy_ev: float
    gamma: float
    energy_emission_cold_ev: float
    energy_emission_hot_ev: float
    ion_leave_step: int
    electron_emission_cold_step: int
    electron_emission_hot_step: int
    collision_step_ion: int
    collision_step_electron: int
    poisson_omega: float
    poisson_iterations: int
    poisson_tol: float
    poisson_solver: str
    log_interval: int
    field_dump_interval: int
    max_electrons: int
    max_ions: int
    cross_sections: CrossSectionPaths
    output_dir: str
    grid_size: int | None = None

    @property
    def t_e_kelvin(self) -> float:
        return self.t_e_ev * 11604.52500617

    @property
    def t_i_kelvin(self) -> float:
        return self.t_i_ev * 11604.52500617


def default_config(
    output_dir: str | None = None,
    cross_section_dir: str | Path | None = None,
) -> SimulationConfig:
    cross_section_dir = Path(cross_section_dir) if cross_section_dir is not None else DEFAULT_CROSS_SECTION_DIR
    output_dir = output_dir or str(DEFAULT_OUTPUT_DIR)
    return SimulationConfig(
        scale=0.02,
        gyro_coeff=100,
        grid_size=None,
        it_num=int(400),
        ptcls_per_cell=1.0,
        r=0.26,
        b=0.1,
        n_e=1e15,
        n_i=1e15,
        t_e_ev=10.0,
        t_i_ev=10.0,
        i_hot=6.0,
        i_cold=4.0,
        p_pa=4e-3 * 133.0,
        t_gas=500.0,
        m_ion=500.0 * E_M,
        seed=1,
        init_energy_ev=0.1,
        gamma=0.1,
        energy_emission_cold_ev=100.0,
        energy_emission_hot_ev=100.0,
        ion_leave_step=500,
        electron_emission_cold_step=500,
        electron_emission_hot_step=500,
        collision_step_ion=25,
        collision_step_electron=5,
        poisson_omega=0.85,
        poisson_iterations=250,
        poisson_tol=1.0e-4,
        poisson_solver="direct_inverse",
        log_interval=100,
        field_dump_interval=10_000,
        max_electrons=300_000,
        max_ions=300_000,
        cross_sections=CrossSectionPaths(
            electron_elastic=str(cross_section_dir / "e-Ar_elastic.txt"),
            electron_ionization=str(cross_section_dir / "e-Ar_ionization.txt"),
            ion_elastic=str(cross_section_dir / "Ar+-Ar_elastic.txt"),
        ),
        output_dir=output_dir,
    )
