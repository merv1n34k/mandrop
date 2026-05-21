"""Diagnostic measurements and visualisation for mandrop simulations.

Two responsibilities:
- `droplet_stats(phi, geo_params, probe_history, dt_phys)` — segments water
  in the outlet channel into individual droplets, returns diameter / CV
  (monodispersity) and droplet-generation frequency (Hz).
- `plot_fields(f, phi, Gamma=None, ...)` — 4- or 5-panel summary figure.
"""

import numpy as np
from scipy import ndimage
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mandrop.engine import compute_macros


def droplet_stats(phi, geo_params, probe_history=None, dt_phys=None):
    """Measure droplets in the outlet channel.

    Connected-component segmentation of (phi < 0.5) restricted to
    y ∈ [0, Y_OUTLET_TOP) gives per-droplet pixel counts → equivalent
    diameter. If a per-step phi history at the probe point plus the lattice
    timestep are supplied, also returns the generation frequency from
    threshold crossings.

    Returns:
        n_drops:    number of distinct droplets in the outlet
        d_mean_um:  mean equivalent diameter (µm)
        d_cv:       coefficient of variation of diameter (0..1)
        freq_Hz:    droplet generation frequency from pulse counter (Hz)
        sizes_um:   ndarray of per-droplet diameters (µm)
    """
    Y_outlet_top = geo_params["Y_OUTLET_TOP"]
    um_per_lu    = geo_params["dx_mm"] * 1000.0

    water = np.asarray(phi) < 0.5
    band  = np.zeros_like(water)
    band[:, :Y_outlet_top] = True
    water_outlet = water & band

    labeled, n_drops = ndimage.label(water_outlet)
    if n_drops == 0:
        sizes_um  = np.array([])
        d_mean_um = 0.0
        d_cv      = 0.0
    else:
        sizes_lu     = np.bincount(labeled.ravel())[1:]  # drop background label 0
        diameters_lu = 2.0 * np.sqrt(sizes_lu / np.pi)    # 2D equivalent
        sizes_um     = diameters_lu * um_per_lu
        d_mean_um    = float(sizes_um.mean())
        d_cv         = float(sizes_um.std() / d_mean_um) if d_mean_um > 0 else 0.0

    freq_Hz = 0.0
    if probe_history is not None and dt_phys is not None:
        hist = np.asarray(probe_history)
        in_water = hist < 0.5
        n_entries = int(np.sum(np.diff(in_water.astype(np.int8)) == 1))
        chunk_time = len(hist) * dt_phys
        if chunk_time > 0:
            freq_Hz = n_entries / chunk_time

    return dict(
        n_drops   = int(n_drops),
        d_mean_um = d_mean_um,
        d_cv      = d_cv,
        freq_Hz   = freq_Hz,
        sizes_um  = sizes_um,
    )


def plot_fields(f, phi, Gamma=None, interior=None, title=None):
    """4- or 5-panel summary plot: φ, ρ, u_y, |u|, optional Γ."""
    rho, ux, uy = compute_macros(f)
    vel_mag = jnp.sqrt(ux ** 2 + uy ** 2)
    vm = max(float(vel_mag.max()), 1e-6)

    n_panels = 5 if Gamma is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 12))

    im0 = axes[0].imshow(phi.T, origin="lower", cmap="RdBu", vmin=0, vmax=1)
    axes[0].set_title("φ (oil=1, water=0)")
    plt.colorbar(im0, ax=axes[0], shrink=0.5)

    im1 = axes[1].imshow(rho.T, origin="lower", cmap="viridis")
    axes[1].set_title("ρ (pressure)")
    plt.colorbar(im1, ax=axes[1], shrink=0.5)

    im2 = axes[2].imshow(uy.T, origin="lower", cmap="coolwarm", vmin=-vm, vmax=vm)
    axes[2].set_title("u_y (flow direction)")
    plt.colorbar(im2, ax=axes[2], shrink=0.5)

    im3 = axes[3].imshow(vel_mag.T, origin="lower", cmap="hot", vmin=0, vmax=vm)
    axes[3].set_title(f"|u| (max={vm:.2e})")
    plt.colorbar(im3, ax=axes[3], shrink=0.5)

    if Gamma is not None:
        im4 = axes[4].imshow(Gamma.T, origin="lower", cmap="magma", vmin=0, vmax=1)
        axes[4].set_title("Γ (surfactant coverage)")
        plt.colorbar(im4, ax=axes[4], shrink=0.5)

    for ax in axes:
        ax.set_aspect("equal")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig, axes
