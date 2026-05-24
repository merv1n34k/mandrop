"""LBM phase-field engine for droplet simulation (D2Q9 + Cahn-Hilliard)."""

import jax
import jax.numpy as jnp
from jax import jit

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
# JIT-compiled operators (lattice-only, no geometry dependence)
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
def feq_fn(rho, ux, uy):
    eu = ux[..., None] * ex_jnp + uy[..., None] * ey_jnp
    usq = ux ** 2 + uy ** 2
    return w * rho[..., None] * (1.0 + eu / cs2 + eu ** 2 / (2.0 * cs2 ** 2) - usq[..., None] / (2.0 * cs2))


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


def zou_he_top(f, x_start, x_end, rho_target):
    """Pressure BC at y=Ny-1, x in [x_start, x_end). Flow enters in -y."""
    rho_t = rho_target
    f_s = f[x_start:x_end, -1, :]
    uy_t = -1.0 + (f_s[:, 0] + f_s[:, 1] + f_s[:, 3]
                    + 2.0 * (f_s[:, 2] + f_s[:, 5] + f_s[:, 6])) / rho_t
    f = f.at[x_start:x_end, -1, 4].set(f_s[:, 2] - (2.0 / 3.0) * rho_t * uy_t)
    f = f.at[x_start:x_end, -1, 7].set(f_s[:, 5] + 0.5 * (f_s[:, 1] - f_s[:, 3]) - (1.0 / 6.0) * rho_t * uy_t)
    f = f.at[x_start:x_end, -1, 8].set(f_s[:, 6] - 0.5 * (f_s[:, 1] - f_s[:, 3]) - (1.0 / 6.0) * rho_t * uy_t)
    return f


def zou_he_bottom(f, x_start, x_end, rho_target):
    """Pressure BC at y=0, x in [x_start, x_end). Flow exits in -y."""
    rho_t = rho_target
    f_s = f[x_start:x_end, 0, :]
    uy_t = 1.0 - (f_s[:, 0] + f_s[:, 1] + f_s[:, 3]
                   + 2.0 * (f_s[:, 4] + f_s[:, 7] + f_s[:, 8])) / rho_t
    f = f.at[x_start:x_end, 0, 2].set(f_s[:, 4] + (2.0 / 3.0) * rho_t * uy_t)
    f = f.at[x_start:x_end, 0, 5].set(f_s[:, 7] - 0.5 * (f_s[:, 1] - f_s[:, 3]) + (1.0 / 6.0) * rho_t * uy_t)
    f = f.at[x_start:x_end, 0, 6].set(f_s[:, 8] + 0.5 * (f_s[:, 1] - f_s[:, 3]) + (1.0 / 6.0) * rho_t * uy_t)
    return f


def zou_he_left(f, y_start, y_end, rho_target):
    """Pressure BC at x=0, y in [y_start, y_end). Flow enters in +x."""
    rho_t = rho_target
    f_s = f[0, y_start:y_end, :]
    ux_t = 1.0 - (f_s[:, 0] + f_s[:, 2] + f_s[:, 4]
                   + 2.0 * (f_s[:, 3] + f_s[:, 6] + f_s[:, 7])) / rho_t
    f = f.at[0, y_start:y_end, 1].set(f_s[:, 3] + (2.0 / 3.0) * rho_t * ux_t)
    f = f.at[0, y_start:y_end, 5].set(f_s[:, 7] - 0.5 * (f_s[:, 2] - f_s[:, 4]) + (1.0 / 6.0) * rho_t * ux_t)
    f = f.at[0, y_start:y_end, 8].set(f_s[:, 6] + 0.5 * (f_s[:, 2] - f_s[:, 4]) + (1.0 / 6.0) * rho_t * ux_t)
    return f


def zou_he_right(f, y_start, y_end, rho_target):
    """Pressure BC at x=Nx-1, y in [y_start, y_end). Flow enters in -x."""
    rho_t = rho_target
    f_s = f[-1, y_start:y_end, :]
    ux_t = -1.0 + (f_s[:, 0] + f_s[:, 2] + f_s[:, 4]
                    + 2.0 * (f_s[:, 1] + f_s[:, 5] + f_s[:, 8])) / rho_t
    f = f.at[-1, y_start:y_end, 3].set(f_s[:, 1] - (2.0 / 3.0) * rho_t * ux_t)
    f = f.at[-1, y_start:y_end, 6].set(f_s[:, 8] - 0.5 * (f_s[:, 2] - f_s[:, 4]) - (1.0 / 6.0) * rho_t * ux_t)
    f = f.at[-1, y_start:y_end, 7].set(f_s[:, 5] + 0.5 * (f_s[:, 2] - f_s[:, 4]) - (1.0 / 6.0) * rho_t * ux_t)
    return f


def zou_he_top_u(f, x_start, x_end, uy_target):
    """Velocity BC at y=Ny-1, x in [x_start, x_end). uy<0 = inflow."""
    uy_t = uy_target
    f_s = f[x_start:x_end, -1, :]
    rho_t = (f_s[:, 0] + f_s[:, 1] + f_s[:, 3]
              + 2.0 * (f_s[:, 2] + f_s[:, 5] + f_s[:, 6])) / (1.0 + uy_t)
    f = f.at[x_start:x_end, -1, 4].set(f_s[:, 2] - (2.0 / 3.0) * rho_t * uy_t)
    f = f.at[x_start:x_end, -1, 7].set(f_s[:, 5] + 0.5 * (f_s[:, 1] - f_s[:, 3]) - (1.0 / 6.0) * rho_t * uy_t)
    f = f.at[x_start:x_end, -1, 8].set(f_s[:, 6] - 0.5 * (f_s[:, 1] - f_s[:, 3]) - (1.0 / 6.0) * rho_t * uy_t)
    return f


def zou_he_left_u(f, y_start, y_end, ux_target):
    """Velocity BC at x=0, y in [y_start, y_end). ux>0 = inflow."""
    ux_t = ux_target
    f_s = f[0, y_start:y_end, :]
    rho_t = (f_s[:, 0] + f_s[:, 2] + f_s[:, 4]
              + 2.0 * (f_s[:, 3] + f_s[:, 6] + f_s[:, 7])) / (1.0 - ux_t)
    f = f.at[0, y_start:y_end, 1].set(f_s[:, 3] + (2.0 / 3.0) * rho_t * ux_t)
    f = f.at[0, y_start:y_end, 5].set(f_s[:, 7] - 0.5 * (f_s[:, 2] - f_s[:, 4]) + (1.0 / 6.0) * rho_t * ux_t)
    f = f.at[0, y_start:y_end, 8].set(f_s[:, 6] + 0.5 * (f_s[:, 2] - f_s[:, 4]) + (1.0 / 6.0) * rho_t * ux_t)
    return f


def zou_he_right_u(f, y_start, y_end, ux_target):
    """Velocity BC at x=Nx-1, y in [y_start, y_end). ux<0 = inflow."""
    ux_t = ux_target
    f_s = f[-1, y_start:y_end, :]
    rho_t = (f_s[:, 0] + f_s[:, 2] + f_s[:, 4]
              + 2.0 * (f_s[:, 1] + f_s[:, 5] + f_s[:, 8])) / (1.0 + ux_t)
    f = f.at[-1, y_start:y_end, 3].set(f_s[:, 1] - (2.0 / 3.0) * rho_t * ux_t)
    f = f.at[-1, y_start:y_end, 6].set(f_s[:, 8] - 0.5 * (f_s[:, 2] - f_s[:, 4]) - (1.0 / 6.0) * rho_t * ux_t)
    f = f.at[-1, y_start:y_end, 7].set(f_s[:, 5] + 0.5 * (f_s[:, 2] - f_s[:, 4]) - (1.0 / 6.0) * rho_t * ux_t)
    return f


@jit
def compute_macros(f):
    rho = jnp.sum(f, axis=-1)
    ux = jnp.sum(f * ex_jnp, axis=-1) / rho
    uy = jnp.sum(f * ey_jnp, axis=-1) / rho
    return rho, ux, uy


def relax_phi(phi, apply_phi_bcs, sigma_eq, W, M_ch, n_steps=50):
    """Cahn-Hilliard-only relaxation at zero velocity. Spreads a step-function
    phi to a tanh interface of width W. Uses uniform sigma_eq (stable regime)
    so the bare-interface κ_clean stiffness is bypassed during warmup."""
    beta = 3.0 * sigma_eq / W
    kappa = 6.0 * sigma_eq * W
    for _ in range(n_steps):
        phi = apply_phi_bcs(phi)
        phi = jnp.clip(phi, 0.0, 1.0)
        lap_phi = compute_laplacian(phi)
        mu = 2.0 * beta * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - kappa * lap_phi
        lap_mu = compute_laplacian(mu)
        phi = phi + M_ch * lap_mu
    return jnp.clip(phi, 0.0, 1.0)


def init_state(Nx, Ny, rho0, apply_phi_bcs, apply_gamma_bcs,
               sigma_eq, W, M_ch, water_prefill=None, n_relax=50):
    f0 = feq_fn(jnp.ones((Nx, Ny)) * rho0, jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)))
    phi0 = jnp.ones((Nx, Ny))
    if water_prefill is not None:
        phi0 = jnp.where(water_prefill, 0.0, phi0)
    phi0 = apply_phi_bcs(phi0)
    # Pre-relax phi to natural tanh profile so a real interface band exists at t=0.
    phi0 = relax_phi(phi0, apply_phi_bcs, sigma_eq, W, M_ch, n_steps=n_relax)
    # Existing interface (post-relaxation) is aged (Γ=1); bulk Γ=0 so any new
    # interface formed during the run inherits bare coverage.
    interface_t0 = (phi0 > 0.05) & (phi0 < 0.95)
    Gamma0 = jnp.where(interface_t0, 1.0, 0.0)
    Gamma0 = apply_gamma_bcs(Gamma0)
    return f0, phi0, Gamma0


@jit
def stream(f):
    return jnp.stack(
        [jnp.roll(jnp.roll(f[..., i], ex[i], axis=0), ey[i], axis=1) for i in range(9)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Step factory — closes over geometry + physical parameters
# ---------------------------------------------------------------------------
def make_step(wall, fluid, interior, opp_jnp,
              tau_c, tau_d, sigma_clean, sigma_eq, W, tau_ads, D_gamma, M_ch,
              apply_f_bcs, apply_phi_bcs, apply_gamma_bcs, boundary_mask):
    """Build the JIT step function.

    State is a 3-tuple (f, phi, Gamma):
      - f: D2Q9 distributions
      - phi: phase field (0=water, 1=oil)
      - Gamma: surfactant coverage fraction [0,1] at the interface

    Local interfacial tension: sigma(x) = sigma_clean + (sigma_eq - sigma_clean)·Gamma
    Local relaxation time:     tau(x)   = (1-phi)·tau_d + phi·tau_c
       (phi=0 = water/dispersed, phi=1 = oil/continuous)
    """
    inv_tau_ads = 1.0 / tau_ads

    @jit
    def apply_bounce_back(f):
        return jnp.where(wall[..., None], f[:, :, opp_jnp], f)

    @jit
    def chem_potential(phi, lap_phi, beta_local, kappa_local):
        return 2.0 * beta_local * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - kappa_local * lap_phi

    @jit
    def forcing_guo(ux, uy, Fx, Fy, tau_local):
        eu = ux[..., None] * ex_jnp + uy[..., None] * ey_jnp
        factor = (1.0 - 0.5 / tau_local)[..., None]
        return factor * w * (
            (ex_jnp - ux[..., None]) * Fx[..., None] / cs2
            + (ey_jnp - uy[..., None]) * Fy[..., None] / cs2
            + eu / cs2 ** 2 * (ex_jnp * Fx[..., None] + ey_jnp * Fy[..., None])
        )

    @jit
    def step(state, water_scale=1.0):
        f, phi, Gamma = state

        phi = apply_phi_bcs(phi)
        phi = jnp.clip(phi, 0.0, 1.0)
        Gamma = apply_gamma_bcs(Gamma)

        sigma_local = sigma_clean + (sigma_eq - sigma_clean) * Gamma
        beta_local  = 3.0 * sigma_local / W
        kappa_local = 6.0 * sigma_local * W
        tau_local   = (1.0 - phi) * tau_d + phi * tau_c

        lap_phi = compute_laplacian(phi)
        gx, gy  = compute_gradient(phi)
        mu = chem_potential(phi, lap_phi, beta_local, kappa_local)

        Fx = mu * gx
        Fy = mu * gy
        Fx = jnp.where(boundary_mask, 0.0, Fx)
        Fy = jnp.where(boundary_mask, 0.0, Fy)

        rho = jnp.sum(f, axis=-1)
        ux = (jnp.sum(f * ex_jnp, axis=-1) + 0.5 * Fx) / rho
        uy = (jnp.sum(f * ey_jnp, axis=-1) + 0.5 * Fy) / rho
        ux = jnp.where(wall, 0.0, ux)
        uy = jnp.where(wall, 0.0, uy)

        feq = feq_fn(rho, ux, uy)
        Fi  = forcing_guo(ux, uy, Fx, Fy, tau_local)
        f_collided = f - (f - feq) / tau_local[..., None] + Fi
        f = jnp.where(fluid[..., None], f_collided, f)

        f = stream(f)
        f = apply_bounce_back(f)
        f = apply_f_bcs(f, water_scale)

        lap_mu = compute_laplacian(mu)
        div_flux = compute_divergence(phi * ux, phi * uy)
        ch_update = -div_flux + M_ch * lap_mu
        phi = phi + jnp.where(interior, ch_update, 0.0)

        phi = apply_phi_bcs(phi)
        phi = jnp.clip(phi, 0.0, 1.0)

        # Gamma evolution: advection + small diffusion + interfacial adsorption source
        interface_mask = (phi > 0.05) & (phi < 0.95)
        div_gamma_flux = compute_divergence(Gamma * ux, Gamma * uy)
        lap_gamma = compute_laplacian(Gamma)
        gamma_source = jnp.where(interface_mask, (1.0 - Gamma) * inv_tau_ads, 0.0)
        gamma_update = -div_gamma_flux + D_gamma * lap_gamma + gamma_source
        Gamma = Gamma + jnp.where(interior, gamma_update, 0.0)
        Gamma = apply_gamma_bcs(Gamma)
        Gamma = jnp.clip(Gamma, 0.0, 1.0)

        return (f, phi, Gamma)

    return step
