"""LBM simulation for water-in-oil droplet generation with real-time visualization."""

import signal
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt

from mandrop.engine import (
    make_step, feq_fn, ex_jnp, ey_jnp, opp,
    zou_he_top, zou_he_bottom,
)
from jax import jit

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Nx, Ny = 200, 600
R = 50.0
W = 3.0
sigma = 0.01
beta = 3.0 * sigma / W
kappa = 6.0 * sigma * W
rho0 = 1.0
nu = 1.0 / 6.0
tau_f = 3.0 * nu + 0.5
M_ch = 0.01
drho = 0.001
rho_in = rho0 + drho / 2.0
rho_out = rho0 - drho / 2.0
droplet_centers = [150.0, 300.0, 450.0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    # Phi initialization: 3 droplets
    x = jnp.arange(Nx, dtype=jnp.float64)
    y = jnp.arange(Ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    xc = Nx / 2.0

    phi0 = jnp.ones((Nx, Ny))
    for yc_d in droplet_centers:
        r = jnp.sqrt((X - xc) ** 2 + (Y - yc_d) ** 2)
        phi0 = jnp.minimum(phi0, 0.5 * (1.0 + jnp.tanh((r - R) / (2.0 * W))))

    # Walls and boundary conditions
    wall = jnp.zeros((Nx, Ny), dtype=bool)
    wall = wall.at[0, :].set(True)
    wall = wall.at[-1, :].set(True)

    fluid = ~wall
    interior = fluid & (jnp.arange(Ny)[None, :] > 0) & (jnp.arange(Ny)[None, :] < Ny - 1)
    opp_jnp = jnp.array(opp)

    inlet_width = int(0.4 * Nx)
    inlet_x0 = Nx // 2 - inlet_width // 2
    inlet_x1 = Nx // 2 + inlet_width // 2
    inlet_water = jnp.zeros(Nx, dtype=bool).at[inlet_x0:inlet_x1].set(True)
    phi_inlet = jnp.where(inlet_water, 0.0, 1.0)

    @jit
    def apply_f_bcs(f):
        f = zou_he_top(f, 1, Nx - 1, rho_in)
        f = zou_he_bottom(f, 1, Nx - 1, rho_out)
        return f

    @jit
    def apply_phi_bcs(phi):
        phi = jnp.where(wall, 1.0, phi)
        phi = phi.at[:, -1].set(phi_inlet)
        phi = phi.at[:, 0].set(phi[:, 1])
        return phi

    boundary_mask = (jnp.arange(Ny)[None, :] == 0) | (jnp.arange(Ny)[None, :] == Ny - 1)

    step = make_step(
        wall, fluid, interior, opp_jnp,
        tau_f, beta, kappa, M_ch,
        apply_f_bcs, apply_phi_bcs, boundary_mask,
    )

    print(f"mandrop — LBM droplet simulation")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Domain: {Nx}×{Ny}, 3 droplets R={R:.0f}, inlet: center 40% water")
    print(f"Δρ={drho}, tau={tau_f}")

    # Init
    rho_init = jnp.ones((Nx, Ny)) * rho0
    ux0 = jnp.zeros((Nx, Ny))
    uy0 = jnp.zeros((Nx, Ny))
    f0 = feq_fn(rho_init, ux0, uy0)
    phi0_box = apply_phi_bcs(phi0)

    # JIT warmup
    print("Compiling (JIT warmup)...", end=" ", flush=True)
    state = (f0, phi0_box)
    state = step(state)
    state[0].block_until_ready()
    print("done.")

    # Reset
    state = (f0, phi0_box)

    # Scan body for chunked stepping
    def scan_body(state, _):
        return step(state), None

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

    im0 = axes[0].imshow(phi0_box.T, origin="lower", cmap="RdBu", vmin=0, vmax=1)
    cb0 = plt.colorbar(im0, ax=axes[0], shrink=0.5)
    axes[0].set_title("φ (oil=1, droplet=0)")

    im1 = axes[1].imshow(jnp.ones((Nx, Ny)).T, origin="lower", cmap="viridis")
    cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.5)
    axes[1].set_title("ρ (pressure)")

    im2 = axes[2].imshow(jnp.zeros((Nx, Ny)).T, origin="lower", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    cb2 = plt.colorbar(im2, ax=axes[2], shrink=0.5)
    axes[2].set_title("u_y (flow direction)")

    im3 = axes[3].imshow(jnp.zeros((Nx, Ny)).T, origin="lower", cmap="hot", vmin=0, vmax=0.01)
    cb3 = plt.colorbar(im3, ax=axes[3], shrink=0.5)
    axes[3].set_title("|u|")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    def on_key(event):
        if event.key == "escape":
            running[0] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    total_steps = 0
    t_start = time.time()
    t_chunk = t_start

    print(f"\nRunning... Press Escape or Ctrl+C to stop.\n")
    print(f"{'step':>8} | {'MLUPS':>8} | {'max|u|':>10} | {'phi_min':>10} {'phi_max':>10} | {'droplets':>8}")
    print("-" * 75)

    try:
        while running[0]:
            state, _ = lax.scan(scan_body, state, None, length=chunk_size)
            f_c, phi_c = state
            f_c.block_until_ready()

            total_steps += chunk_size
            t_now = time.time()
            dt = t_now - t_chunk
            t_chunk = t_now
            mlups = Nx * Ny * chunk_size / dt / 1e6

            rho_c = jnp.sum(f_c, axis=-1)
            ux_c = jnp.sum(f_c * ex_jnp, axis=-1) / rho_c
            uy_c = jnp.sum(f_c * ey_jnp, axis=-1) / rho_c
            vel_mag = jnp.sqrt(ux_c ** 2 + uy_c ** 2)
            max_vel = float(vel_mag.max())
            n_drop = float(((phi_c < 0.5).astype(jnp.float64) * interior).sum())

            print(f"{total_steps:8d} | {mlups:8.2f} | {max_vel:10.2e} | {float(phi_c.min()):10.6f} {float(phi_c.max()):10.6f} | {n_drop:8.0f}")

            if jnp.isnan(phi_c).any():
                print("NaN detected, stopping.")
                break

            # Update plots
            im0.set_data(phi_c.T)
            im1.set_data(rho_c.T)
            im1.set_clim(float(rho_c.min()), float(rho_c.max()))
            im2.set_data(uy_c.T)
            vm = max(max_vel, 1e-6)
            im2.set_clim(-vm, vm)
            im3.set_data(vel_mag.T)
            im3.set_clim(0, vm)
            axes[3].set_title(f"|u| (max={max_vel:.2e})")

            fig.suptitle(f"Step {total_steps}  |  {mlups:.1f} MLUPS", fontsize=12)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    avg_mlups = Nx * Ny * total_steps / elapsed / 1e6
    print(f"\nStopped at step {total_steps}. Elapsed: {elapsed:.1f}s, avg {avg_mlups:.1f} MLUPS")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
