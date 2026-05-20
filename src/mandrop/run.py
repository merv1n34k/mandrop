"""Simulation runner for LBM flow-focusing."""

import time

import jax.numpy as jnp
from jax import lax

from mandrop.engine import compute_macros
from mandrop.generator import boundary_stats


def run(step, f0, phi0, Gamma0, interior, params,
        chunk_size=500, n_chunks=60, on_chunk=None, verbose=True):
    """Run simulation in chunks with JIT warmup and diagnostics.

    on_chunk(f, phi, Gamma, step_num, dt) is called after each chunk.
    Return False from on_chunk to stop early.

    Returns (f_final, phi_final, Gamma_final, total_steps).
    """
    # JIT warmup
    state = (f0, phi0, Gamma0)
    state = step(state)
    state[0].block_until_ready()
    if verbose:
        print("JIT compiled.")

    state = (f0, phi0, Gamma0)

    def scan_body(state, _):
        return step(state), None

    t0 = time.time()
    t_chunk = t0
    chunk = 0

    while chunk < n_chunks:
        state, _ = lax.scan(scan_body, state, None, length=chunk_size)
        f_c, phi_c, Gamma_c = state
        f_c.block_until_ready()

        chunk += 1
        step_num = chunk * chunk_size
        t_now = time.time()
        dt = t_now - t_chunk
        t_chunk = t_now

        if jnp.isnan(phi_c).any() or jnp.isnan(f_c).any() or jnp.isnan(Gamma_c).any():
            print(f"  *** NaN detected at step {step_num} ***")
            break

        if verbose:
            rho_c, ux_c, uy_c, stats = boundary_stats(f_c, phi_c, params)
            max_vel = jnp.max(jnp.sqrt(ux_c**2 + uy_c**2))
            n_water = ((phi_c < 0.5).astype(jnp.float64) * interior).sum()

            print(f"\n=== Step {step_num} === max|u|={float(max_vel):.2e}  water_px={float(n_water):.0f}  rho=[{float(rho_c.min()):.4f},{float(rho_c.max()):.4f}]  Gamma=[{float(Gamma_c.min()):.3f},{float(Gamma_c.max()):.3f}]")
            for name, label in [("top", "TOP    (water in)"), ("ul", "UL slot(water in)"),
                                ("ur",  "UR slot(water in)"), ("ll", "LL slot(oil in) "),
                                ("lr",  "LR slot(oil in) "), ("bot","BOT    (outlet) ")]:
                s = stats[name]
                print(f"  {label:18s} rho=[{s['rho_min']:.4f},{s['rho_max']:.4f}]  {s['u_label']}=[{s['u_min']:.4e},{s['u_max']:.4e}]  phi=[{s['phi_min']:.3f},{s['phi_max']:.3f}]")

        if on_chunk is not None:
            if on_chunk(f_c, phi_c, Gamma_c, step_num, dt) is False:
                break

    total_steps = chunk * chunk_size
    f_final, phi_final, Gamma_final = state

    if verbose:
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")

    return f_final, phi_final, Gamma_final, total_steps
