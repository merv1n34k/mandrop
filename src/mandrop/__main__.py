"""LBM simulation for water-in-oil droplet generation with real-time visualization."""

import signal
import sys
import time

import jax
import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# D2Q9 lattice
# ---------------------------------------------------------------------------
w = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
ex = [0, 1, 0, -1, 0, 1, -1, -1, 1]
ey = [0, 0, 1, 0, -1, 1, 1, -1, -1]
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
cs2 = 1.0 / 3.0
ex_jnp = jnp.array(ex, dtype=jnp.float64)
ey_jnp = jnp.array(ey, dtype=jnp.float64)

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

# ---------------------------------------------------------------------------
# Phi initialization: 3 droplets
# ---------------------------------------------------------------------------
x = jnp.arange(Nx, dtype=jnp.float64)
y = jnp.arange(Ny, dtype=jnp.float64)
X, Y = jnp.meshgrid(x, y, indexing="ij")
xc = Nx / 2.0
droplet_centers = [150.0, 300.0, 450.0]

phi0 = jnp.ones((Nx, Ny))
for yc_d in droplet_centers:
    r = jnp.sqrt((X - xc) ** 2 + (Y - yc_d) ** 2)
    phi0 = jnp.minimum(phi0, 0.5 * (1.0 + jnp.tanh((r - R) / (2.0 * W))))

# ---------------------------------------------------------------------------
# Walls and boundary conditions
# ---------------------------------------------------------------------------
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
def apply_bounce_back(f):
    return jnp.where(wall[..., None], f[:, :, opp_jnp], f)


@jit
def apply_phi_walls(phi):
    phi = jnp.where(wall, 1.0, phi)
    phi = phi.at[:, -1].set(phi_inlet)
    phi = phi.at[:, 0].set(phi[:, 1])
    return phi


@jit
def zou_he_inlet(f, rho_target):
    rho_t = rho_target
    f_top = f[1:-1, -1, :]
    uy_t = -1.0 + (f_top[:, 0] + f_top[:, 1] + f_top[:, 3]
                    + 2.0 * (f_top[:, 2] + f_top[:, 5] + f_top[:, 6])) / rho_t
    f = f.at[1:-1, -1, 4].set(f_top[:, 2] - (2.0 / 3.0) * rho_t * uy_t)
    f = f.at[1:-1, -1, 7].set(f_top[:, 5] + 0.5 * (f_top[:, 1] - f_top[:, 3]) - (1.0 / 6.0) * rho_t * uy_t)
    f = f.at[1:-1, -1, 8].set(f_top[:, 6] - 0.5 * (f_top[:, 1] - f_top[:, 3]) - (1.0 / 6.0) * rho_t * uy_t)
    return f


@jit
def zou_he_outlet(f, rho_target):
    rho_t = rho_target
    f_bot = f[1:-1, 0, :]
    uy_t = 1.0 - (f_bot[:, 0] + f_bot[:, 1] + f_bot[:, 3]
                   + 2.0 * (f_bot[:, 4] + f_bot[:, 7] + f_bot[:, 8])) / rho_t
    f = f.at[1:-1, 0, 2].set(f_bot[:, 4] + (2.0 / 3.0) * rho_t * uy_t)
    f = f.at[1:-1, 0, 5].set(f_bot[:, 7] - 0.5 * (f_bot[:, 1] - f_bot[:, 3]) + (1.0 / 6.0) * rho_t * uy_t)
    f = f.at[1:-1, 0, 6].set(f_bot[:, 8] + 0.5 * (f_bot[:, 1] - f_bot[:, 3]) + (1.0 / 6.0) * rho_t * uy_t)
    return f


# ---------------------------------------------------------------------------
# Core operators
# ---------------------------------------------------------------------------
@jit
def compute_laplacian(field):
    lap = jnp.zeros_like(field)
    for i in range(1, 9):
        lap += w[i] * jnp.roll(jnp.roll(field, -ex[i], axis=0), -ey[i], axis=1)
    return (2.0 / cs2) * (lap - (1.0 - w[0]) * field)


@jit
def compute_gradient(field):
    gx = jnp.zeros_like(field)
    gy = jnp.zeros_like(field)
    for i in range(1, 9):
        shifted = jnp.roll(jnp.roll(field, -ex[i], axis=0), -ey[i], axis=1)
        gx += w[i] * ex[i] * shifted
        gy += w[i] * ey[i] * shifted
    return gx / cs2, gy / cs2


@jit
def compute_divergence(Fx, Fy):
    div = jnp.zeros_like(Fx)
    for i in range(1, 9):
        sx = jnp.roll(jnp.roll(Fx, -ex[i], axis=0), -ey[i], axis=1)
        sy = jnp.roll(jnp.roll(Fy, -ex[i], axis=0), -ey[i], axis=1)
        div += w[i] * (ex[i] * sx + ey[i] * sy)
    return div / cs2


@jit
def chem_potential(phi, lap_phi):
    return 2.0 * beta * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - kappa * lap_phi


@jit
def feq_fn(rho, ux, uy):
    eu = ux[..., None] * ex_jnp + uy[..., None] * ey_jnp
    usq = ux ** 2 + uy ** 2
    return w * rho[..., None] * (1.0 + eu / cs2 + eu ** 2 / (2.0 * cs2 ** 2) - usq[..., None] / (2.0 * cs2))


@jit
def forcing_guo(ux, uy, Fx, Fy):
    eu = ux[..., None] * ex_jnp + uy[..., None] * ey_jnp
    return (1.0 - 0.5 / tau_f) * w * (
        (ex_jnp - ux[..., None]) * Fx[..., None] / cs2
        + (ey_jnp - uy[..., None]) * Fy[..., None] / cs2
        + eu / cs2 ** 2 * (ex_jnp * Fx[..., None] + ey_jnp * Fy[..., None])
    )


@jit
def stream(f):
    return jnp.stack(
        [jnp.roll(jnp.roll(f[..., i], ex[i], axis=0), ey[i], axis=1) for i in range(9)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------
@jit
def step(state):
    f, phi = state

    phi = apply_phi_walls(phi)
    phi = jnp.clip(phi, 0.0, 1.0)

    lap_phi = compute_laplacian(phi)
    gx, gy = compute_gradient(phi)
    mu = chem_potential(phi, lap_phi)

    Fx = mu * gx
    Fy = mu * gy
    Fx = Fx.at[:, 0].set(0.0).at[:, -1].set(0.0)
    Fy = Fy.at[:, 0].set(0.0).at[:, -1].set(0.0)

    rho = jnp.sum(f, axis=-1)
    ux = (jnp.sum(f * ex_jnp, axis=-1) + 0.5 * Fx) / rho
    uy = (jnp.sum(f * ey_jnp, axis=-1) + 0.5 * Fy) / rho
    ux = jnp.where(wall, 0.0, ux)
    uy = jnp.where(wall, 0.0, uy)

    feq = feq_fn(rho, ux, uy)
    Fi = forcing_guo(ux, uy, Fx, Fy)
    f_collided = f - (f - feq) / tau_f + Fi
    f = jnp.where(fluid[..., None], f_collided, f)

    f = stream(f)
    f = apply_bounce_back(f)
    f = zou_he_inlet(f, rho_in)
    f = zou_he_outlet(f, rho_out)

    lap_mu = compute_laplacian(mu)
    div_flux = compute_divergence(phi * ux, phi * uy)
    ch_update = -div_flux + M_ch * lap_mu
    phi = phi + jnp.where(interior, ch_update, 0.0)

    phi = apply_phi_walls(phi)
    phi = jnp.clip(phi, 0.0, 1.0)
    return (f, phi)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print(f"mandrop — LBM droplet simulation")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Domain: {Nx}×{Ny}, 3 droplets R={R:.0f}, inlet: center 40% water")
    print(f"Δρ={drho}, tau={tau_f}")

    # Init
    rho_init = jnp.ones((Nx, Ny)) * rho0
    ux0 = jnp.zeros((Nx, Ny))
    uy0 = jnp.zeros((Nx, Ny))
    f0 = feq_fn(rho_init, ux0, uy0)
    phi0_box = apply_phi_walls(phi0)

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
