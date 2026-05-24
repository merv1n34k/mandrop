"""Real-time LBM flow-focusing simulation with live droplet stats.

Run with:  uv run python -m mandrop

To study chemistry: edit `params.physical.*` (sigma, surfactant kinetics,
flow rates). To switch resolution: edit `params.sim.resolution_um`.
"""

import signal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mandrop import (
    Params, PhysicalParams, SimulationParams,
    build, run, droplet_stats, compute_macros,
)


def main():
    params = Params(
        physical=PhysicalParams(),                   # default = real chip op point
        sim     =SimulationParams(resolution_um=1.0),  # 1 µm/lu for production
    )
    geo, step, (f0, phi0, Gamma0), lat = build(params)
    p_geo = geo["params"]
    Nx, Ny = p_geo["Nx"], p_geo["Ny"]
    interior = geo["interior"]

    print("mandrop — flow-focusing droplet generation")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Domain: {Nx}×{Ny}  Channel x∈[{p_geo['gxL']},{p_geo['gxR']}]  Throat x∈[{p_geo['gxTL']},{p_geo['gxTR']}]")
    print(params.summary())

    chunk_size = 200

    # Graceful shutdown
    running = [True]
    def on_signal(sig, frame): running[0] = False
    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Matplotlib interactive setup
    plt.ion()
    fig, axes = plt.subplots(1, 5, figsize=(20, 12))
    for ax in axes: ax.set_aspect("equal")

    im0 = axes[0].imshow(phi0.T, origin="lower", cmap="RdBu", vmin=0, vmax=1)
    plt.colorbar(im0, ax=axes[0], shrink=0.5); axes[0].set_title("φ (oil=1, water=0)")

    im1 = axes[1].imshow(jnp.ones((Nx, Ny)).T, origin="lower", cmap="viridis")
    plt.colorbar(im1, ax=axes[1], shrink=0.5); axes[1].set_title("ρ (pressure)")

    im2 = axes[2].imshow(jnp.zeros((Nx, Ny)).T, origin="lower", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    plt.colorbar(im2, ax=axes[2], shrink=0.5); axes[2].set_title("u_y (flow direction)")

    im3 = axes[3].imshow(jnp.zeros((Nx, Ny)).T, origin="lower", cmap="hot", vmin=0, vmax=0.01)
    plt.colorbar(im3, ax=axes[3], shrink=0.5); axes[3].set_title("|u|")

    im4 = axes[4].imshow(Gamma0.T, origin="lower", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im4, ax=axes[4], shrink=0.5); axes[4].set_title("Γ (surfactant coverage)")

    fig.tight_layout(); fig.canvas.draw(); fig.canvas.flush_events()

    def on_key(event):
        if event.key == "escape": running[0] = False
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"\nRunning... Press Escape or Ctrl+C to stop.\n")
    print(f"{'step':>8} | {'MLUPS':>6} | {'max|u| (m/s)':>12} | {'Δp (Pa)':>9} | {'drops':>5} {'d_mean (µm)':>11} {'CV':>6} | {'f (Hz)':>8} | Γ_iface")
    print("-" * 110)

    def update_plots(f_c, phi_c, Gamma_c, probe_history, step_num, wall_dt):
        mlups = Nx * Ny * chunk_size / wall_dt / 1e6
        rho_c, ux_c, uy_c = compute_macros(f_c)
        vel_mag = jnp.sqrt(ux_c ** 2 + uy_c ** 2)
        max_vel_lu = float(vel_mag.max())
        max_vel_phys = max_vel_lu * lat.dx / lat.dt
        delta_p_pa = float(rho_c.max() - rho_c.min()) / 3.0 * lat.p_lu_to_pa

        drops = droplet_stats(phi_c, p_geo, probe_history=probe_history, dt_phys=lat.dt)
        iface_mask = (phi_c > 0.05) & (phi_c < 0.95)
        gamma_mean = float(jnp.where(iface_mask, Gamma_c, 0.0).sum() /
                           jnp.maximum(iface_mask.sum(), 1))

        print(f"{step_num:8d} | {mlups:6.2f} | {max_vel_phys:12.4f} | {delta_p_pa:9.0f} | "
              f"{drops['n_drops']:5d} {drops['d_mean_um']:11.1f} {drops['d_cv']:6.2%} | "
              f"{drops['freq_Hz']:8.0f} | {gamma_mean:.3f}")

        im0.set_data(phi_c.T)
        im1.set_data(rho_c.T); im1.set_clim(float(rho_c.min()), float(rho_c.max()))
        im2.set_data(uy_c.T)
        vm = max(max_vel_lu, 1e-6)
        im2.set_clim(-vm, vm)
        im3.set_data(vel_mag.T); im3.set_clim(0, vm)
        axes[3].set_title(f"|u| (max={max_vel_phys:.3f} m/s)")
        im4.set_data(Gamma_c.T)
        axes[4].set_title(f"Γ (⟨Γ⟩_iface={gamma_mean:.2f})")

        fig.suptitle(f"step {step_num}  |  {mlups:.1f} MLUPS  |  drops {drops['n_drops']} @ {drops['d_mean_um']:.0f} µm  |  f={drops['freq_Hz']:.0f} Hz", fontsize=12)
        fig.canvas.draw_idle(); fig.canvas.flush_events()

        if not running[0] or not plt.fignum_exists(fig.number):
            return False

    _, _, _, total_steps = run(
        step, f0, phi0, Gamma0, interior, p_geo,
        chunk_size=chunk_size, n_chunks=999_999,
        on_chunk=update_plots, verbose=False,
        warmup_steps=5000,   # ramp water inlets over first 5k steps to dodge jet lock-in
    )

    print(f"\nStopped at step {total_steps}.")
    plt.ioff(); plt.show()


if __name__ == "__main__":
    main()
