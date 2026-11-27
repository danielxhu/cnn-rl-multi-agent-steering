"""JAX port of ``sim/`` for GPU-batched training (DESIGN_FRONTIER.md §2).

- ``bank``: scenes -> fixed-size numpy arrays (CPU, numpy only; no JAX).
- ``sim``: pure-JAX reset / observe / step on one packed level; ``jax.vmap``
  over levels for batches.
- ``port_check``: step-by-step comparison against ``sim/physics.py``.
"""
