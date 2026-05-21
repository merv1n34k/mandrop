"""Parameters for mandrop simulations.

Two parameter groups, intentionally separated by purpose:

- `PhysicalParams`: the chip operating point and the fluid + surfactant
  chemistry. These are what you adjust to study a new buffer, surfactant,
  flow-rate change, or chip variant. Units are SI.

- `SimulationParams`: numerical knobs (resolution, Mach target, interface
  width, mobility). Change only when validating numerics or switching
  resolution.

`Params` bundles both and exposes `.lattice` — a `LatticeParams` snapshot of
all the *_lu values needed by the engine/generator. `derive(phys, sim)` is
the pure function behind it.
"""

from dataclasses import dataclass, field


@dataclass
class PhysicalParams:
    # Fluid properties (default: water + HFE-7500 treated as ν≈ν_water for LBM)
    nu_phys: float  = 1e-6     # m²/s kinematic viscosity (continuous phase)
    rho_phys: float = 1000.0   # kg/m³ density reference

    # Chip operating point — flow rates expressed as inlet velocities (m/s).
    # Defaults match the real chip: 300 µL/min oil (2 inlets, 77.5×100 µm),
    # 80 µL/min water (40 central + 20+20 sides).
    u_oil_phys:  float = 0.32   # m/s per oil inlet
    u_top_phys:  float = 0.053  # m/s central water inlet
    u_side_phys: float = 0.043  # m/s per side water inlet

    # Surfactant chemistry — primary knobs for buffer/surfactant studies.
    sigma_eq_phys:    float = 5e-3   # N/m IFT with full surfactant coverage
    sigma_clean_phys: float = 20e-3  # N/m bare-interface IFT (fresh, no surfactant)
    tau_ads_s:        float = 1e-4   # s adsorption timescale (Pluronic ≈ 0.1-1 s)


@dataclass
class SimulationParams:
    resolution_um:   float = 2.5     # µm per lattice unit
    outlet_extra_mm: float = 0.3575  # extend outlet channel (mm)
    W:               float = 4.0     # phase-field interface width (lu)
    M_ch:            float = 0.05    # Cahn-Hilliard mobility
    D_gamma_lu:      float = 0.001   # Γ-field bulk diffusion (lu)
    mach_target:     float = 0.025   # inlet u_lu/cs cap; throat focusing ~3.6× → ~0.09


@dataclass
class LatticeParams:
    """All values the engine/generator need, in lattice units."""
    dx: float
    dt: float
    nu_lu:   float
    tau_f:   float
    u_oil_in_lu:        float
    u_top_in_lu:        float
    u_water_side_in_lu: float
    sigma_eq:    float
    sigma_clean: float
    tau_ads_lu:  float
    W:           float
    M_ch:        float
    D_gamma:     float
    rho0:    float = 1.0
    rho_out: float = 0.9995

    # Convenience conversions
    @property
    def p_lu_to_pa(self) -> float:
        # 1 unit of ρ_lu corresponds to ρ_phys·(dx/dt)² Pa
        # (callers usually divide by 3 to get Δp from Δρ)
        return 1000.0 * (self.dx / self.dt) ** 2

    @property
    def sigma_lu_to_Nm(self) -> float:
        return 1000.0 * self.dx ** 3 / self.dt ** 2


def derive(phys: PhysicalParams, sim: SimulationParams) -> LatticeParams:
    """Compute lattice values that satisfy Mach + viscosity constraints."""
    dx = sim.resolution_um * 1e-6
    # ν_lu chosen so that inlet u_lu sits at mach_target × cs
    nu_lu = (sim.mach_target / (3 ** 0.5)) / phys.u_oil_phys * phys.nu_phys / dx
    nu_lu = max(min(nu_lu, 0.5), 0.02)  # τ ∈ [0.56, 2.0]
    dt = nu_lu * dx ** 2 / phys.nu_phys
    u_lu_per_mps   = dt / dx
    sigma_lu_to_Nm = phys.rho_phys * dx ** 3 / dt ** 2

    return LatticeParams(
        dx=dx, dt=dt, nu_lu=nu_lu, tau_f=3.0 * nu_lu + 0.5,
        u_oil_in_lu        = phys.u_oil_phys  * u_lu_per_mps,
        u_top_in_lu        = phys.u_top_phys  * u_lu_per_mps,
        u_water_side_in_lu = phys.u_side_phys * u_lu_per_mps,
        sigma_eq    = phys.sigma_eq_phys    / sigma_lu_to_Nm,
        sigma_clean = phys.sigma_clean_phys / sigma_lu_to_Nm,
        tau_ads_lu  = phys.tau_ads_s        / dt,
        W=sim.W, M_ch=sim.M_ch, D_gamma=sim.D_gamma_lu,
    )


@dataclass
class Params:
    """Top-level config: physical chemistry + simulation knobs."""
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    sim:      SimulationParams = field(default_factory=SimulationParams)

    @property
    def lattice(self) -> LatticeParams:
        return derive(self.physical, self.sim)

    def summary(self) -> str:
        lat = self.lattice
        return (
            f"dx={lat.dx*1e6:.2f} µm  dt={lat.dt*1e9:.1f} ns  ν_lu={lat.nu_lu:.4f}  τ={lat.tau_f:.3f}\n"
            f"u_oil_lu={lat.u_oil_in_lu:.4f}  Mach={lat.u_oil_in_lu*3**0.5:.3f}\n"
            f"σ_eq={lat.sigma_eq:.4f}  σ_clean={lat.sigma_clean:.4f}  τ_ads={lat.tau_ads_lu:.0f} lu_ts\n"
            f"1 lu of ρ ≈ {lat.p_lu_to_pa/3:.0f} Pa  1 s = {int(1/lat.dt):,} lu_ts"
        )
