"""Cross-junction flow-focusing geometry."""

import jax.numpy as jnp
from jax import jit

from mandrop.engine import (
    opp, zou_he_top, zou_he_bottom, zou_he_left, zou_he_right,
)


def setup(
    Nx=200,
    Ny=600,
    w_channel=80,
    w_side=60,
    junction_y=450,
    rho_in_water=1.0005,
    rho_in_oil=1.0005,
    rho_out=0.9995,
):
    """Build wall mask, BC functions, and masks for a cross-junction device.

    The channel has uniform width (w_channel) from top to bottom.
    Oil enters from both sides at the junction zone.

    Returns dict with: wall, fluid, interior, opp_jnp,
                        apply_f_bcs, apply_phi_bcs, boundary_mask, params
    """
    x_L = Nx // 2 - w_channel // 2
    x_R = Nx // 2 + w_channel // 2

    jy_bot = junction_y - w_side // 2
    jy_top = junction_y + w_side // 2

    # --- Wall mask: start solid, carve out ---
    wall = jnp.ones((Nx, Ny), dtype=bool)

    # Main channel above junction
    wall = wall.at[x_L + 1:x_R, jy_top:].set(False)

    # Junction zone: full width open
    wall = wall.at[:, jy_bot:jy_top].set(False)

    # Channel below junction (same width as main)
    wall = wall.at[x_L + 1:x_R, :jy_bot].set(False)

    fluid = ~wall
    at_edge = (
        (jnp.arange(Nx)[:, None] == 0) |
        (jnp.arange(Nx)[:, None] == Nx - 1) |
        (jnp.arange(Ny)[None, :] == 0) |
        (jnp.arange(Ny)[None, :] == Ny - 1)
    )
    interior = fluid & ~at_edge
    opp_jnp = jnp.array(opp)

    @jit
    def apply_f_bcs(f):
        f = zou_he_top(f, x_L + 1, x_R, rho_in_water)
        f = zou_he_left(f, jy_bot, jy_top, rho_in_oil)
        f = zou_he_right(f, jy_bot, jy_top, rho_in_oil)
        f = zou_he_bottom(f, x_L + 1, x_R, rho_out)
        return f

    @jit
    def apply_phi_bcs(phi):
        phi = jnp.where(wall, 1.0, phi)
        phi = phi.at[x_L + 1:x_R, -1].set(0.0)
        phi = phi.at[0, jy_bot:jy_top].set(1.0)
        phi = phi.at[-1, jy_bot:jy_top].set(1.0)
        phi = phi.at[x_L + 1:x_R, 0].set(phi[x_L + 1:x_R, 1])
        return phi

    boundary_mask = (
        (jnp.arange(Nx)[:, None] == 0) |
        (jnp.arange(Nx)[:, None] == Nx - 1) |
        (jnp.arange(Ny)[None, :] == 0) |
        (jnp.arange(Ny)[None, :] == Ny - 1)
    )

    params = dict(
        Nx=Nx, Ny=Ny, w_channel=w_channel, w_side=w_side,
        junction_y=junction_y, x_L=x_L, x_R=x_R,
        jy_bot=jy_bot, jy_top=jy_top,
        rho_in_water=rho_in_water, rho_in_oil=rho_in_oil, rho_out=rho_out,
    )

    return dict(
        wall=wall, fluid=fluid, interior=interior, opp_jnp=opp_jnp,
        apply_f_bcs=apply_f_bcs, apply_phi_bcs=apply_phi_bcs,
        boundary_mask=boundary_mask, params=params,
    )
