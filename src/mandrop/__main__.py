"""LBM flow-focusing simulation for water-in-oil droplet generation."""

import signal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mandrop.engine import make_step, compute_macros, init_state
from mandrop.generator import setup
from mandrop.run import run

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RESOLUTION_UM     = 1.0      # µm per lattice unit
OUTLET_EXTRA_MM   = 0.3575   # extend outlet by this much below the DXF default

# ── Physical chip operating point (real units) ──────────────────────────────
NU_PHYS      = 1e-6       # m²/s (water kinematic viscosity)
RHO_PHYS     = 1000.0     # kg/m³
U_OIL_PHYS   = 0.32       # m/s per oil inlet (150 µL/min ÷ 77.5×100 µm)
U_TOP_PHYS   = 0.053      # m/s central water (40 µL/min ÷ 125×100 µm)
U_SIDE_PHYS  = 0.043      # m/s per side water (20 µL/min ÷ 77.5×100 µm)
SIGMA_EQ_PHYS    = 5e-3   # N/m equilibrium IFT (HFE-7500 + 2% PicoSurf)
SIGMA_CLEAN_PHYS = 5e-3   # N/m bare IFT (set equal to σ_eq for now; raise to ~50e-3 to enable Stage 3)

# ── Numerical knob: Mach target sets dt via ν_lu ─────────────────────────────
MACH_TARGET = 0.025       # u_lu/cs cap at inlet; throat focusing ~3.6× → throat Mach ≈ 0.09

# Derived lattice values (so changing RESOLUTION_UM keeps Re and Ca correct)
dx        = RESOLUTION_UM * 1e-6
nu_lu     = (MACH_TARGET / (3**0.5)) / U_OIL_PHYS * NU_PHYS / dx  # solves for ν_lu given Mach cap on u_oil
nu_lu     = max(min(nu_lu, 0.5), 0.02)                            # clamp to safe τ range (τ ∈ [0.56, 2.0])
dt        = nu_lu * dx**2 / NU_PHYS
u_lu_per_mps = dt / dx
p_lu_to_pa   = RHO_PHYS * (dx/dt)**2
sigma_lu_to_Nm = RHO_PHYS * dx**3 / dt**2

W        = 4.0
SIGMA_EQ    = SIGMA_EQ_PHYS    / sigma_lu_to_Nm
SIGMA_CLEAN = SIGMA_CLEAN_PHYS / sigma_lu_to_Nm
TAU_ADS_LU  = 2000.0
D_GAMMA     = 0.001
rho0  = 1.0
nu    = nu_lu
tau_f = 3.0 * nu + 0.5
M_ch  = 0.05

U_OIL_IN_LU        = U_OIL_PHYS  * u_lu_per_mps
U_TOP_IN_LU        = U_TOP_PHYS  * u_lu_per_mps
U_WATER_SIDE_IN_LU = U_SIDE_PHYS * u_lu_per_mps

drho  = 0.001
F_OUT = -1.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    geo = setup(
        resolution_um    =RESOLUTION_UM,
        outlet_extra_mm  =OUTLET_EXTRA_MM,
        u_top_in_lu       =U_TOP_IN_LU,
        u_water_side_in_lu=U_WATER_SIDE_IN_LU,
        u_oil_in_lu       =U_OIL_IN_LU,
        rho_out           =rho0 + F_OUT * drho / 2.0,
    )
    p = geo["params"]
    Nx, Ny = p["Nx"], p["Ny"]

    step = make_step(
        geo["wall"], geo["fluid"], geo["interior"], geo["opp_jnp"],
        tau_f, SIGMA_CLEAN, SIGMA_EQ, W, TAU_ADS_LU, D_GAMMA, M_ch,
        geo["apply_f_bcs"], geo["apply_phi_bcs"], geo["apply_gamma_bcs"], geo["boundary_mask"],
    )

    print("mandrop — flow-focusing droplet generation")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Resolution: {p['resolution_um']} µm/lu  Domain: {Nx}×{Ny}")
    print(f"Channel x∈[{p['gxL']},{p['gxR']}]  Throat x∈[{p['gxTL']},{p['gxTR']}]")
    print(f"Upper slots (water) y∈[{p['Y_USLOT_BOT']},{p['Y_USLOT_TOP']})  Lower slots (oil) y∈[{p['Y_LSLOT_BOT']},{p['Y_LSLOT_TOP']})")
    print(f"σ_clean={SIGMA_CLEAN}  σ_eq={SIGMA_EQ}  τ_ads={TAU_ADS_LU}  Δρ={drho}, tau={tau_f}")

    f0, phi0, Gamma0 = init_state(
        Nx, Ny, rho0, geo["apply_phi_bcs"], geo["apply_gamma_bcs"],
        SIGMA_EQ, W, M_ch, geo["water_prefill"],
    )
    interior = geo["interior"]
    chunk_size = 200

    # Graceful shutdown
    running = [True]

    def on_signal(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Matplotlib interactive setup
    plt.ion()
    fig, axes = plt.subplots(1, 5, figsize=(20, 12))
    for ax in axes:
        ax.set_aspect("equal")

    im0 = axes[0].imshow(phi0.T, origin="lower", cmap="RdBu", vmin=0, vmax=1)
    plt.colorbar(im0, ax=axes[0], shrink=0.5)
    axes[0].set_title("φ (oil=1, water=0)")

    im1 = axes[1].imshow(jnp.ones((Nx, Ny)).T, origin="lower", cmap="viridis")
    plt.colorbar(im1, ax=axes[1], shrink=0.5)
    axes[1].set_title("ρ (pressure)")

    im2 = axes[2].imshow(jnp.zeros((Nx, Ny)).T, origin="lower", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    plt.colorbar(im2, ax=axes[2], shrink=0.5)
    axes[2].set_title("u_y (flow direction)")

    im3 = axes[3].imshow(jnp.zeros((Nx, Ny)).T, origin="lower", cmap="hot", vmin=0, vmax=0.01)
    plt.colorbar(im3, ax=axes[3], shrink=0.5)
    axes[3].set_title("|u|")

    im4 = axes[4].imshow(Gamma0.T, origin="lower", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im4, ax=axes[4], shrink=0.5)
    axes[4].set_title("Γ (surfactant coverage)")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    def on_key(event):
        if event.key == "escape":
            running[0] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"\nRunning... Press Escape or Ctrl+C to stop.\n")
    print(f"{'step':>8} | {'MLUPS':>8} | {'max|u|':>10} | {'phi_min':>10} {'phi_max':>10} | {'water_px':>8} | Γ_iface")
    print("-" * 90)

    def update_plots(f_c, phi_c, Gamma_c, step_num, dt):
        mlups = Nx * Ny * chunk_size / dt / 1e6

        rho_c, ux_c, uy_c = compute_macros(f_c)
        vel_mag = jnp.sqrt(ux_c ** 2 + uy_c ** 2)
        max_vel = float(vel_mag.max())
        n_water = float(((phi_c < 0.5).astype(jnp.float64) * interior).sum())
        gamma_mean_iface = float(jnp.where((phi_c > 0.05) & (phi_c < 0.95), Gamma_c, 0.0).sum() /
                                  jnp.maximum(((phi_c > 0.05) & (phi_c < 0.95)).sum(), 1))

        print(f"{step_num:8d} | {mlups:8.2f} | {max_vel:10.2e} | {float(phi_c.min()):10.6f} {float(phi_c.max()):10.6f} | {n_water:8.0f} | Γ_iface={gamma_mean_iface:.3f}")

        im0.set_data(phi_c.T)
        im1.set_data(rho_c.T)
        im1.set_clim(float(rho_c.min()), float(rho_c.max()))
        im2.set_data(uy_c.T)
        vm = max(max_vel, 1e-6)
        im2.set_clim(-vm, vm)
        im3.set_data(vel_mag.T)
        im3.set_clim(0, vm)
        axes[3].set_title(f"|u| (max={max_vel:.2e})")
        im4.set_data(Gamma_c.T)
        axes[4].set_title(f"Γ (⟨Γ⟩_iface={gamma_mean_iface:.2f})")

        fig.suptitle(f"Step {step_num}  |  {mlups:.1f} MLUPS", fontsize=12)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if not running[0] or not plt.fignum_exists(fig.number):
            return False

    _, _, _, total_steps = run(
        step, f0, phi0, Gamma0, interior, geo["params"],
        chunk_size=chunk_size, n_chunks=999_999,
        on_chunk=update_plots, verbose=False,
    )

    elapsed_msg = f"\nStopped at step {total_steps}."
    print(elapsed_msg)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
