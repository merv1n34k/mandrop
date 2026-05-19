# mandrop

Lattice Boltzmann Method (LBM) simulation for water-in-oil droplet generation using JAX.

Phase-field Cahn-Hilliard model coupled with D2Q9 LBM. Bounce-back walls, Zou-He pressure inlet/outlet, Guo forcing for surface tension.

## Quick start

```
uv sync
uv run mandrop
```

Press **Escape** or **Ctrl+C** to stop.

## Physical system

- Water-in-HFE-7500 oil
- 200×600 μm channel (1 μm/lu)
- 3 initial droplets (100 μm diameter)
- Center 40% water inlet, oil on sides
- Pressure-driven Poiseuille flow
