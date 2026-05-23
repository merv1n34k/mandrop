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
    # Fluid kinematic viscosities (m²/s). Continuous = oil (HFE-7500),
    # dispersed = aqueous mix. NOTE: HFE has higher *dynamic* μ but lower
    # *kinematic* ν than water because it's 60% denser; LBM cares about ν.
    nu_c_phys:  float = 0.77e-6   # HFE-7500: μ=1.24 mPa·s / ρ=1614
    nu_d_phys:  float = 1.0e-6    # water-side
    # Single reference density (Cahn-Hilliard scheme limitation — see report).
    # Use an average; horizontal flow-focusing → Bo negligible anyway. When we
    # add density-ratio support, this will be replaced by rho_c_phys / rho_d_phys.
    rho_phys:   float = 1300.0    # kg/m³ — average of HFE (1614) and water (1000)

    # Chip operating point — flow rates expressed as inlet velocities (m/s).
    # Defaults: 300 µL/min oil (2 inlets, 77.5×100 µm), 80 µL/min water
    # (40 central + 20+20 sides).
    u_oil_phys:  float = 0.32   # m/s per oil inlet
    u_top_phys:  float = 0.053  # m/s central water inlet
    u_side_phys: float = 0.043  # m/s per side water inlet

    # Effective interfacial tension for buffer/surfactant studies.
    # Single σ representing the operating-point effective IFT at the chip's
    # pinch-off timescale (per Mu et al. 2021, constant σ is sufficient for
    # droplet-size prediction at fixed operating conditions). Adjust per
    # measured/back-fitted value for each buffer formulation.
    sigma_phys: float = 20e-3   # N/m effective IFT (capped below CH-LBM stability ceiling)


@dataclass
class SimulationParams:
    resolution_um:   float = 1.0     # µm per lattice unit
    outlet_extra_mm: float = 0.3575  # extend outlet channel (mm)
    W:               float = 4.0     # phase-field interface width (lu)
    M_ch:            float = 0.05    # Cahn-Hilliard mobility
    mach_target:     float = 0.025   # inlet u_lu/cs cap; throat focusing ~3.6× → ~0.09


@dataclass
class LatticeParams:
    """All values the engine/generator need, in lattice units."""
    dx: float
    dt: float
    nu_c_lu:  float
    nu_d_lu:  float
    tau_c:    float
    tau_d:    float
    u_oil_in_lu:        float
    u_top_in_lu:        float
    u_water_side_in_lu: float
    sigma:    float
    W:        float
    M_ch:     float
    rho_phys: float
    rho0:    float = 1.0
    rho_out: float = 0.9995

    @property
    def p_lu_to_pa(self) -> float:
        # 1 unit of ρ_lu corresponds to ρ_phys·(dx/dt)² Pa
        return self.rho_phys * (self.dx / self.dt) ** 2

    @property
    def sigma_lu_to_Nm(self) -> float:
        return self.rho_phys * self.dx ** 3 / self.dt ** 2


def derive(phys: PhysicalParams, sim: SimulationParams) -> LatticeParams:
    """Compute lattice values that satisfy Mach + viscosity constraints.

    ν_c (continuous, oil) sets the time scale via Mach cap on inlet velocity.
    ν_d (dispersed, water) is mapped at the same dt using its own kinematic ν.
    The engine interpolates τ(φ) per cell so each phase carries its own viscous
    response — preserves the kinematic viscosity ratio λ_lu = ν_d/ν_c.
    """
    dx = sim.resolution_um * 1e-6

    nu_c_lu = (sim.mach_target / (3 ** 0.5)) / phys.u_oil_phys * phys.nu_c_phys / dx
    nu_c_lu = max(min(nu_c_lu, 0.5), 0.02)
    dt = nu_c_lu * dx ** 2 / phys.nu_c_phys
    nu_d_lu = phys.nu_d_phys * dt / dx ** 2

    u_lu_per_mps   = dt / dx
    sigma_lu_to_Nm = phys.rho_phys * dx ** 3 / dt ** 2

    return LatticeParams(
        dx=dx, dt=dt,
        nu_c_lu=nu_c_lu, nu_d_lu=nu_d_lu,
        tau_c=3.0 * nu_c_lu + 0.5,
        tau_d=3.0 * nu_d_lu + 0.5,
        u_oil_in_lu        = phys.u_oil_phys  * u_lu_per_mps,
        u_top_in_lu        = phys.u_top_phys  * u_lu_per_mps,
        u_water_side_in_lu = phys.u_side_phys * u_lu_per_mps,
        sigma  = phys.sigma_phys / sigma_lu_to_Nm,
        W=sim.W, M_ch=sim.M_ch,
        rho_phys=phys.rho_phys,
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
            f"dx={lat.dx*1e6:.2f} µm  dt={lat.dt*1e9:.1f} ns\n"
            f"ν_c={lat.nu_c_lu:.4f} (τ={lat.tau_c:.3f})  ν_d={lat.nu_d_lu:.4f} (τ={lat.tau_d:.3f})  λ=ν_d/ν_c={lat.nu_d_lu/lat.nu_c_lu:.3f}\n"
            f"u_oil_lu={lat.u_oil_in_lu:.4f}  Mach={lat.u_oil_in_lu*3**0.5:.3f}\n"
            f"σ={lat.sigma:.4f}  κ={6*lat.sigma*lat.W:.2f}\n"
            f"1 lu of ρ ≈ {lat.p_lu_to_pa/3:.0f} Pa  1 s = {int(1/lat.dt):,} lu_ts"
        )
