"""Visualization helpers for LBM simulation results."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from mandrop.engine import compute_macros


def plot_fields(f, phi, Gamma=None, interior=None, title=None):
    """5-panel plot: phi, rho, u_y, |u|, Γ (if provided)."""
    rho, ux, uy = compute_macros(f)
    vel_mag = jnp.sqrt(ux**2 + uy**2)
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
