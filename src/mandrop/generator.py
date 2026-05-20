"""Flow-focusing geometry rasterized from cropped.dxf.

Geometry is stored in physical mm and rasterized at a chosen lattice resolution.
5 inlets + 1 outlet:
  top central     (water, y=Ny-1, x in main channel)
  upper L+R slots (water, x=0 and x=Nx-1, mid y)
  lower L+R slots (oil,   x=0 and x=Nx-1, lower y)
  outlet          (y=0, x in main channel)
"""

import jax.numpy as jnp
from jax import jit

from mandrop.engine import (
    opp, zou_he_top, zou_he_bottom, zou_he_left, zou_he_right,
    zou_he_top_u, zou_he_left_u, zou_he_right_u,
    compute_macros,
)


# Physical geometry (mm), measured from cropped.dxf
# x relative to channel midline; y from outlet bottom
X_OUTER_L_MM  = -0.1450
X_CHAN_L_MM   = -0.0625
X_THROAT_L_MM = -0.0475
X_THROAT_R_MM =  0.0475
X_CHAN_R_MM   =  0.0625
X_OUTER_R_MM  =  0.1600

Y_BOT_MM        = 0.0000
Y_OUTLET_TOP_MM = 0.3575
Y_THROAT_BOT_MM = 0.3875
Y_THROAT_TOP_MM = 0.4500
Y_LSLOT_BOT_MM  = 0.4825
Y_LSLOT_TOP_MM  = 0.5600
Y_USLOT_BOT_MM  = 0.6375
Y_USLOT_TOP_MM  = 0.7150
Y_TOP_MM        = 0.8900


def setup(
    resolution_um=2.5,
    outlet_extra_mm=0.0,
    n_seed_droplets=0,
    droplet_diameter_mm=0.075,
    u_top_in_lu=0.005,
    u_water_side_in_lu=0.010,
    rho_in_oil=1.0025,
    rho_out=0.9995,
):
    """Build wall mask, BC closures, and masks for the DXF geometry.

    Hybrid BCs (matches a Fluigent pressure + flow-rate controller):
      - velocity BC on water inlets (top central + upper L/R slots) — no backflow
      - pressure BC on oil inlets and outlet

    Args:
        resolution_um: physical size of one lattice unit, in µm.
                       Default 2.5 → 50-node channel, 123×357 domain.
                       1.0 → 125-node channel, ~308×892 domain (4× finer).
        outlet_extra_mm: extra outlet channel length below the throat (mm).
                       0.0 keeps DXF default (0.3575 mm). 0.3575 doubles it.
        n_seed_droplets: number of water droplets to seed in the outlet channel
                       (evenly spaced along y, centered on the channel midline).
        droplet_diameter_mm: diameter of seed droplets (mm).
        u_top_in_lu:        downward inflow speed at top central water inlet (lu/ts).
        u_water_side_in_lu: inflow speed at upper L+R water slots (lu/ts).
        rho_in_oil:         pressure BC at the lower L+R oil slots.
        rho_out:            pressure BC at the outlet (bottom).
    """
    dx_mm = resolution_um / 1000.0
    nodes_per_mm = 1.0 / dx_mm

    def x_node(x_mm):
        return int(round(x_mm * nodes_per_mm))

    def y_node(y_mm):
        return int(round(y_mm * nodes_per_mm))

    X_OUTER_L  = x_node(X_OUTER_L_MM)
    X_CHAN_L   = x_node(X_CHAN_L_MM)
    X_THROAT_L = x_node(X_THROAT_L_MM)
    X_THROAT_R = x_node(X_THROAT_R_MM)
    X_CHAN_R   = x_node(X_CHAN_R_MM)
    X_OUTER_R  = x_node(X_OUTER_R_MM)

    Y_OUTLET_TOP = y_node(Y_OUTLET_TOP_MM + outlet_extra_mm)
    Y_THROAT_BOT = y_node(Y_THROAT_BOT_MM + outlet_extra_mm)
    Y_THROAT_TOP = y_node(Y_THROAT_TOP_MM + outlet_extra_mm)
    Y_LSLOT_BOT  = y_node(Y_LSLOT_BOT_MM  + outlet_extra_mm)
    Y_LSLOT_TOP  = y_node(Y_LSLOT_TOP_MM  + outlet_extra_mm)
    Y_USLOT_BOT  = y_node(Y_USLOT_BOT_MM  + outlet_extra_mm)
    Y_USLOT_TOP  = y_node(Y_USLOT_TOP_MM  + outlet_extra_mm)
    Y_TOP        = y_node(Y_TOP_MM        + outlet_extra_mm)

    x_off = -X_OUTER_L
    Nx = X_OUTER_R - X_OUTER_L + 1
    Ny = Y_TOP + 1

    def gx(xn):
        return xn + x_off

    gxL  = gx(X_CHAN_L)
    gxR  = gx(X_CHAN_R)
    gxTL = gx(X_THROAT_L)
    gxTR = gx(X_THROAT_R)

    wall = jnp.ones((Nx, Ny), dtype=bool)

    wall = wall.at[gxL+1:gxR, Y_USLOT_TOP:Y_TOP+1].set(False)
    wall = wall.at[:, Y_USLOT_BOT:Y_USLOT_TOP].set(False)
    wall = wall.at[gxL+1:gxR, Y_LSLOT_TOP:Y_USLOT_BOT].set(False)
    wall = wall.at[:, Y_LSLOT_BOT:Y_LSLOT_TOP].set(False)

    for yn in range(Y_THROAT_TOP, Y_LSLOT_BOT):
        frac = (yn - Y_THROAT_TOP) / (Y_LSLOT_BOT - Y_THROAT_TOP)
        xL_n = int(round(X_THROAT_L + frac * (X_CHAN_L - X_THROAT_L)))
        xR_n = int(round(X_THROAT_R + frac * (X_CHAN_R - X_THROAT_R)))
        wall = wall.at[gx(xL_n)+1:gx(xR_n), yn].set(False)

    wall = wall.at[gxTL+1:gxTR, Y_THROAT_BOT:Y_THROAT_TOP].set(False)

    for yn in range(Y_OUTLET_TOP, Y_THROAT_BOT):
        frac = (yn - Y_OUTLET_TOP) / (Y_THROAT_BOT - Y_OUTLET_TOP)
        xL_n = int(round(X_CHAN_L + frac * (X_THROAT_L - X_CHAN_L)))
        xR_n = int(round(X_CHAN_R + frac * (X_THROAT_R - X_CHAN_R)))
        wall = wall.at[gx(xL_n)+1:gx(xR_n), yn].set(False)

    wall = wall.at[gxL+1:gxR, 0:Y_OUTLET_TOP].set(False)

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
        f = zou_he_top_u(f,   gxL+1, gxR,                 -u_top_in_lu)
        f = zou_he_left_u(f,  Y_USLOT_BOT, Y_USLOT_TOP,   +u_water_side_in_lu)
        f = zou_he_right_u(f, Y_USLOT_BOT, Y_USLOT_TOP,   -u_water_side_in_lu)
        f = zou_he_left(f,    Y_LSLOT_BOT, Y_LSLOT_TOP,   rho_in_oil)
        f = zou_he_right(f,   Y_LSLOT_BOT, Y_LSLOT_TOP,   rho_in_oil)
        f = zou_he_bottom(f,  gxL+1, gxR,                 rho_out)
        return f

    @jit
    def apply_phi_bcs(phi):
        phi = jnp.where(wall, 1.0, phi)
        phi = phi.at[gxL+1:gxR, -1].set(0.0)
        phi = phi.at[0,  Y_USLOT_BOT:Y_USLOT_TOP].set(0.0)
        phi = phi.at[-1, Y_USLOT_BOT:Y_USLOT_TOP].set(0.0)
        phi = phi.at[0,  Y_LSLOT_BOT:Y_LSLOT_TOP].set(1.0)
        phi = phi.at[-1, Y_LSLOT_BOT:Y_LSLOT_TOP].set(1.0)
        phi = phi.at[gxL+1:gxR, 0].set(phi[gxL+1:gxR, 1])
        return phi

    boundary_mask = (
        (jnp.arange(Nx)[:, None] == 0) |
        (jnp.arange(Nx)[:, None] == Nx - 1) |
        (jnp.arange(Ny)[None, :] == 0) |
        (jnp.arange(Ny)[None, :] == Ny - 1)
    )

    water_prefill = jnp.zeros((Nx, Ny), dtype=bool)
    water_prefill = water_prefill.at[gxL+1:gxR, Y_USLOT_TOP:].set(True)
    water_prefill = water_prefill.at[:, Y_USLOT_BOT:Y_USLOT_TOP].set(True)
    water_prefill = water_prefill.at[gxL+1:gxR, Y_LSLOT_TOP:Y_USLOT_BOT].set(True)

    if n_seed_droplets > 0:
        r_lu = max(1, int(round(0.5 * droplet_diameter_mm * nodes_per_mm)))
        spacing = int(round(2.5 * r_lu))   # center-to-center = 2.5 × radius
        cx = x_off
        y0 = Y_OUTLET_TOP // 2 - (n_seed_droplets - 1) * spacing // 2
        xx = jnp.arange(Nx)[:, None]
        yy = jnp.arange(Ny)[None, :]
        for i in range(n_seed_droplets):
            cy = y0 + i * spacing
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_lu ** 2
            water_prefill = water_prefill | disk

    water_prefill = water_prefill & fluid

    params = dict(
        Nx=Nx, Ny=Ny, resolution_um=resolution_um, dx_mm=dx_mm,
        gxL=gxL, gxR=gxR, gxTL=gxTL, gxTR=gxTR,
        Y_USLOT_BOT=Y_USLOT_BOT, Y_USLOT_TOP=Y_USLOT_TOP,
        Y_LSLOT_BOT=Y_LSLOT_BOT, Y_LSLOT_TOP=Y_LSLOT_TOP,
        u_top_in_lu=u_top_in_lu, u_water_side_in_lu=u_water_side_in_lu,
        rho_in_oil=rho_in_oil, rho_out=rho_out,
    )

    return dict(
        wall=wall, fluid=fluid, interior=interior, opp_jnp=opp_jnp,
        apply_f_bcs=apply_f_bcs, apply_phi_bcs=apply_phi_bcs,
        boundary_mask=boundary_mask, water_prefill=water_prefill, params=params,
    )


def boundary_stats(f, phi, params):
    """Per-port diagnostics: top water, upper L/R water, lower L/R oil, outlet."""
    rho, ux, uy = compute_macros(f)
    xL, xR = params["gxL"]+1, params["gxR"]
    yub, yut = params["Y_USLOT_BOT"], params["Y_USLOT_TOP"]
    ylb, ylt = params["Y_LSLOT_BOT"], params["Y_LSLOT_TOP"]

    stats = {}
    for name, rho_s, u_s, phi_s, u_label in [
        ("top", rho[xL:xR, -1],   uy[xL:xR, -1],   phi[xL:xR, -1],   "uy"),
        ("ul",  rho[0,  yub:yut], ux[0,  yub:yut], phi[0,  yub:yut], "ux"),
        ("ur",  rho[-1, yub:yut], ux[-1, yub:yut], phi[-1, yub:yut], "ux"),
        ("ll",  rho[0,  ylb:ylt], ux[0,  ylb:ylt], phi[0,  ylb:ylt], "ux"),
        ("lr",  rho[-1, ylb:ylt], ux[-1, ylb:ylt], phi[-1, ylb:ylt], "ux"),
        ("bot", rho[xL:xR, 0],    uy[xL:xR, 0],    phi[xL:xR, 0],    "uy"),
    ]:
        stats[name] = dict(
            rho_min=float(rho_s.min()), rho_max=float(rho_s.max()),
            u_min=float(u_s.min()),     u_max=float(u_s.max()), u_label=u_label,
            phi_min=float(phi_s.min()), phi_max=float(phi_s.max()),
        )
    return rho, ux, uy, stats
