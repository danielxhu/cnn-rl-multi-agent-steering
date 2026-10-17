"""Instance targets for the optional heat/heading head, and world<->pixel helpers.

Targets are rendered from the ground-truth ``Scene`` **at the working
resolution**, never downsampled, so the Gaussian peak sits at the exact
sub-pixel agent centre at 128 as much as at 512.

Pixel convention: pixel ``(row, col)`` has continuous coordinates ``(col, row)``,
the same convention PIL uses when ``render.py`` draws at ``_to_px(...)``.  The
one y-flip mirrors ``render._to_px``: ``py = (world - y) * scale``.

Invariants:
- ``heat`` is the max over agents of ``exp(-d² / 2σ²)`` with σ = agent radius
  in px; the pixel nearest each centre is set to exactly 1 so the focal loss
  has a well-defined positive set (CenterNet convention). Sketch v5
  (DESIGN_SKETCH §9.2): σ = ``heat_sigma`` × the agent's *drawn* radius when
  the scene records one (``Agent.radius``), else ``heat_sigma`` × the nominal
  radius; ``heat_sigma = 1`` and no drawn radii is exactly the E5 target.
- ``dir`` holds ``(cos θ, sin θ)`` in the *world* heading convention
  (0 = +x, counter-clockwise), constant over each agent's disk; the network
  regresses it and the extractor reads it back with ``atan2``.
- ``mask`` is the union of agent disks of the nominal radius; agents are
  ≥ 2r apart in the world so disks never overlap and ``dir`` is single-valued
  (drawn centres of touching v5 agents may come a little closer: the later
  agent's direction wins on the overlap).
"""
from __future__ import annotations

import math

import numpy as np

import _paths  # noqa: F401


def world_to_px(world_size: float, size: int, x, y):
    """World (x, y up) -> image (px, py down), the one y-flip."""
    scale = size / world_size
    return x * scale, (world_size - y) * scale


def px_to_world(world_size: float, size: int, px, py):
    scale = size / world_size
    return px / scale, world_size - py / scale


def letter_targets(scene, world: dict, size: int):
    """(heat (2,S,S) float32, mask float32 scalar): S / G letter heat (sketch v6,
    DESIGN_SKETCH §10.3). Per letter a Gaussian at its centre with σ = 0.3 ×
    its height (px), the pixel nearest the centre exactly 1, in channel 0 (S) or
    1 (G). A scene without ``letters`` gets an all-zero map and mask 0: its
    letter loss is masked out, not trained towards "no letter"."""
    heat = np.zeros((2, size, size), np.float32)
    letters = getattr(scene, "letters", None)
    if letters is None:
        return heat, np.float32(0.0)
    W = float(world["size"])
    scale = size / W
    for L in letters:
        ch = 0 if L["kind"] == "S" else 1
        cx, cy = world_to_px(W, size, L["pos"][0], L["pos"][1])
        sigma = max(0.3 * float(L["height"]) * scale, 0.5)
        reach = int(math.ceil(4.0 * sigma)) + 1
        x0, x1 = max(0, int(cx) - reach), min(size, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(size, int(cy) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        heat[ch, y0:y1, x0:x1] = np.maximum(heat[ch, y0:y1, x0:x1], g)
        ic, jc = int(round(cy)), int(round(cx))
        if 0 <= ic < size and 0 <= jc < size:
            heat[ch, ic, jc] = 1.0
    return heat, np.float32(1.0)


def instance_targets(scene, world: dict, size: int, heat_sigma: float = 1.0):
    """(heat (1,S,S), dir (2,S,S), mask (1,S,S)) float32 at resolution `size`.

    σ of each heat peak = ``heat_sigma`` × the agent's drawn radius if known,
    else × the nominal radius (both in px)."""
    W = float(world["size"])
    scale = size / W
    r_px = float(world["agent_radius"]) * scale
    heat = np.zeros((size, size), np.float32)
    dirs = np.zeros((2, size, size), np.float32)
    mask = np.zeros((size, size), np.float32)

    for a in scene.agents:
        drawn = getattr(a, "radius", None)
        sigma = heat_sigma * (float(drawn) * scale if drawn else r_px)
        # beyond 4σ the Gaussian is < 3e-4; never smaller than the direction disk
        reach = max(int(math.ceil(4.0 * sigma)), int(math.ceil(r_px))) + 1
        cx, cy = world_to_px(W, size, a.pos[0], a.pos[1])
        x0, x1 = max(0, int(cx) - reach), min(size, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(size, int(cy) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        g = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g)
        ic, jc = int(round(cy)), int(round(cx))
        if 0 <= ic < size and 0 <= jc < size:
            heat[ic, jc] = 1.0
        disk = d2 <= r_px * r_px
        mask[y0:y1, x0:x1][disk] = 1.0
        dirs[0, y0:y1, x0:x1][disk] = math.cos(a.heading)
        dirs[1, y0:y1, x0:x1][disk] = math.sin(a.heading)
    return heat[None], dirs, mask[None]
