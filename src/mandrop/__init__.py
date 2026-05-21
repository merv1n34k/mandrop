"""mandrop — phase-field LBM droplet generator with dynamic IFT.

High-level API for both real-time simulation (`__main__.py`) and
verification work (`main.ipynb`):

    from mandrop import Params, build, run, plot_fields, droplet_stats

    params = Params()                       # defaults match the real chip
    geo, step, state0, lat = build(params)  # geometry + step fn + initial state
    f, phi, Gamma, total = run(step, *state0, geo['interior'], geo['params'])
    plot_fields(f, phi, Gamma)
    stats = droplet_stats(phi, geo['params'])

Adjust chemistry by overriding `params.physical.*`; adjust numerics via
`params.sim.*`. The runner publishes a probe-φ history through `on_chunk`
so `droplet_stats(..., probe_history, dt_phys)` can report frequency.
"""

from mandrop.params import (
    PhysicalParams, SimulationParams, LatticeParams, Params, derive,
)
from mandrop.generator import setup, boundary_stats
from mandrop.engine import make_step, init_state, compute_macros
from mandrop.run import run
from mandrop.stats import plot_fields, droplet_stats


def build(params=None):
    """One-shot setup: build geometry, step function, and initial state from Params.

    Returns:
        geo:    dict from generator.setup (wall, fluid, interior, BCs, params, ...)
        step:   JIT-compiled step function with all closures bound
        state0: (f0, phi0, Gamma0) initial-state tuple
        lat:    LatticeParams snapshot (for unit conversions in plots/printing)
    """
    p = params or Params()
    lat = p.lattice

    geo = setup(
        resolution_um      = p.sim.resolution_um,
        outlet_extra_mm    = p.sim.outlet_extra_mm,
        u_top_in_lu        = lat.u_top_in_lu,
        u_water_side_in_lu = lat.u_water_side_in_lu,
        u_oil_in_lu        = lat.u_oil_in_lu,
        rho_out            = lat.rho_out,
    )

    step = make_step(
        geo["wall"], geo["fluid"], geo["interior"], geo["opp_jnp"],
        lat.tau_f, lat.sigma_clean, lat.sigma_eq, lat.W,
        lat.tau_ads_lu, lat.D_gamma, lat.M_ch,
        geo["apply_f_bcs"], geo["apply_phi_bcs"], geo["apply_gamma_bcs"], geo["boundary_mask"],
    )

    Nx, Ny = geo["params"]["Nx"], geo["params"]["Ny"]
    f0, phi0, Gamma0 = init_state(
        Nx, Ny, lat.rho0,
        geo["apply_phi_bcs"], geo["apply_gamma_bcs"],
        lat.sigma_eq, lat.W, lat.M_ch, geo["water_prefill"],
    )
    return geo, step, (f0, phi0, Gamma0), lat


__all__ = [
    "PhysicalParams", "SimulationParams", "LatticeParams", "Params", "derive",
    "setup", "boundary_stats",
    "make_step", "init_state", "compute_macros",
    "run",
    "plot_fields", "droplet_stats",
    "build",
]
