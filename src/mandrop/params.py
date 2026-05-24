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

    # Surfactant chemistry — primary knobs for buffer/surfactant studies.
    sigma_eq_phys:    float = 5e-3   # N/m IFT at full surfactant coverage (HFE + 2% PicoSurf)
    sigma_clean_phys: float = 50e-3  # N/m bare HFE/water IFT (no surfactant)
    tau_ads_s:        float = 50e-3  # s adsorption timescale (PicoSurf ~50 ms)


@dataclass
class SimulationParams:
    resolution_um:   float = 1.0     # µm per lattice unit (1.0 needed for σ_clean=50mN/m)
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
    nu_c_lu:  float    # continuous (oil) lattice viscosity
    nu_d_lu:  float    # dispersed (water) lattice viscosity
    tau_c:    float    # continuous τ (φ=1)
    tau_d:    float    # dispersed τ (φ=0)
    u_oil_in_lu:        float
    u_top_in_lu:        float
    u_water_side_in_lu: float
    sigma_eq:    float
    sigma_clean: float
    tau_ads_lu:  float
    W:           float
    M_ch:        float
    D_gamma:     float
    rho_phys:    float        # reference physical density used in unit conversions
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
    response — preserves the real-chip viscosity ratio λ = μ_d/μ_c.
    """
    dx = sim.resolution_um * 1e-6

    # Continuous-phase τ sets the timestep (oil drives the flow).
    nu_c_lu = (sim.mach_target / (3 ** 0.5)) / phys.u_oil_phys * phys.nu_c_phys / dx
    nu_c_lu = max(min(nu_c_lu, 0.5), 0.02)
    dt = nu_c_lu * dx ** 2 / phys.nu_c_phys

    # Dispersed-phase ν at the same dt: nu_d_lu = nu_d_phys * dt/dx²
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
        sigma_eq    = phys.sigma_eq_phys    / sigma_lu_to_Nm,
        sigma_clean = phys.sigma_clean_phys / sigma_lu_to_Nm,
        tau_ads_lu  = phys.tau_ads_s        / dt,
        W=sim.W, M_ch=sim.M_ch, D_gamma=sim.D_gamma_lu,
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
            f"σ_eq={lat.sigma_eq:.4f}  σ_clean={lat.sigma_clean:.4f}  κ_clean={6*lat.sigma_clean*lat.W:.2f}  τ_ads={lat.tau_ads_lu:.0f} lu_ts\n"
            f"1 lu of ρ ≈ {lat.p_lu_to_pa/3:.0f} Pa  1 s = {int(1/lat.dt):,} lu_ts"
        )
