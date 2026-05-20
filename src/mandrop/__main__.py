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
Nx, Ny = 200, 400

W = 3.0
sigma = 0.01
beta = 3.0 * sigma / W
kappa = 6.0 * sigma * W
rho0 = 1.0
nu = 1.0 / 6.0
tau_f = 3.0 * nu + 0.5
M_ch = 0.01
drho = 0.001


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    geo = setup(
        Nx=Nx, Ny=Ny,
        rho_in_water=rho0 + drho / 2.0,
        rho_in_oil=rho0 + drho / 2.0,
        rho_out=rho0 - drho / 2.0,
    )
    p = geo["params"]

    step = make_step(
        geo["wall"], geo["fluid"], geo["interior"], geo["opp_jnp"],
        tau_f, beta, kappa, M_ch,
        geo["apply_f_bcs"], geo["apply_phi_bcs"], geo["boundary_mask"],
    )

    print("mandrop — flow-focusing droplet generation")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Domain: {Nx}×{Ny}, channel={p['w_channel']}, side={p['w_side']}")
    print(f"Junction y={p['junction_y']} [{p['jy_bot']},{p['jy_top']}]")
    print(f"Δρ={drho}, tau={tau_f}")

    f0, phi0 = init_state(Nx, Ny, rho0, geo["apply_phi_bcs"], geo["water_prefill"])
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
    fig, axes = plt.subplots(1, 4, figsize=(16, 12))
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

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    def on_key(event):
        if event.key == "escape":
            running[0] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"\nRunning... Press Escape or Ctrl+C to stop.\n")
    print(f"{'step':>8} | {'MLUPS':>8} | {'max|u|':>10} | {'phi_min':>10} {'phi_max':>10} | {'water_px':>8}")
    print("-" * 75)

    def update_plots(f_c, phi_c, step_num, dt):
        mlups = Nx * Ny * chunk_size / dt / 1e6

        rho_c, ux_c, uy_c = compute_macros(f_c)
        vel_mag = jnp.sqrt(ux_c ** 2 + uy_c ** 2)
        max_vel = float(vel_mag.max())
        n_water = float(((phi_c < 0.5).astype(jnp.float64) * interior).sum())

        print(f"{step_num:8d} | {mlups:8.2f} | {max_vel:10.2e} | {float(phi_c.min()):10.6f} {float(phi_c.max()):10.6f} | {n_water:8.0f}")

        im0.set_data(phi_c.T)
        im1.set_data(rho_c.T)
        im1.set_clim(float(rho_c.min()), float(rho_c.max()))
        im2.set_data(uy_c.T)
        vm = max(max_vel, 1e-6)
        im2.set_clim(-vm, vm)
        im3.set_data(vel_mag.T)
        im3.set_clim(0, vm)
        axes[3].set_title(f"|u| (max={max_vel:.2e})")

        fig.suptitle(f"Step {step_num}  |  {mlups:.1f} MLUPS", fontsize=12)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if not running[0] or not plt.fignum_exists(fig.number):
            return False

    _, _, total_steps = run(
        step, f0, phi0, interior, geo["params"],
        chunk_size=chunk_size, n_chunks=999_999,
        on_chunk=update_plots, verbose=False,
    )

    elapsed_msg = f"\nStopped at step {total_steps}."
    print(elapsed_msg)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
