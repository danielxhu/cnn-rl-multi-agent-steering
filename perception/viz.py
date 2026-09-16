"""Pictures a human can check: label colouring, side-by-side panels, parsed overlays.

PIL only (matplotlib figures live in evaluate.py / aggregate.py).  ``colorize``
is re-exported from ``sim/render.py`` so parsed and generated label maps use
one palette.  Parsed scenes are drawn here and never through ``render.render``
(their walls are zero-thickness surface edges, DESIGN.md §1.2).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

import _paths  # noqa: F401
from render import PALETTE, colorize  # noqa: F401  (re-exported)
from targets import world_to_px

COL_AGENT = (220, 30, 200)
COL_OBST = (30, 170, 60)
COL_WALL = (230, 40, 40)
COL_START = (232, 151, 58)
COL_GOAL = (43, 92, 230)
COL_GT = (0, 200, 220)


def to_uint8_image(x) -> np.ndarray:
    """(3,S,S) float tensor/array in [0,1] or (S,S,3) uint8 -> (S,S,3) uint8."""
    a = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def draw_parsed(image_uint8, parsed, gt_scene=None, width=2) -> Image.Image:
    """The parsed scene drawn over the input image (resized to the parse resolution).

    Optional `gt_scene`: ground-truth agent centres as cyan crosses for comparison.
    """
    S = parsed.size
    W = float(parsed.world["size"])
    img = Image.fromarray(to_uint8_image(image_uint8)).convert("RGB")
    if img.size != (S, S):
        img = img.resize((S, S), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    sc = parsed.scene

    def px(x, y):
        return world_to_px(W, S, x, y)

    for (ax, ay), (bx, by) in sc.walls:
        d.line([px(ax, ay), px(bx, by)], fill=COL_WALL, width=width)
    for start, goal in sc.groups:
        for reg, col in ((start, COL_START), (goal, COL_GOAL)):
            x0, y0 = px(reg.x0, reg.y1)
            x1, y1 = px(reg.x1, reg.y0)
            d.rectangle([x0, y0, x1, y1], outline=col, width=width)
    for o in sc.obstacles:
        cx, cy = px(*o.c)
        r = o.r * S / W
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=COL_OBST, width=width)
    r = float(parsed.world["agent_radius"]) * S / W
    for a in sc.agents:
        cx, cy = px(*a.pos)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=COL_AGENT, width=width)
        hx, hy = cx + 2.2 * r * math.cos(a.heading), cy - 2.2 * r * math.sin(a.heading)
        d.line([cx, cy, hx, hy], fill=COL_AGENT, width=width)
    if gt_scene is not None:
        k = max(2, int(r / 2))
        for a in gt_scene.agents:
            cx, cy = px(*a.pos)
            d.line([cx - k, cy, cx + k, cy], fill=COL_GT, width=1)
            d.line([cx, cy - k, cx, cy + k], fill=COL_GT, width=1)
    return img


def panel(images, pad=4, bg=(255, 255, 255)) -> Image.Image:
    """Side-by-side strip of PIL images / uint8 arrays, resized to a common height."""
    ims = [im if isinstance(im, Image.Image) else Image.fromarray(to_uint8_image(im))
           for im in images]
    h = max(im.height for im in ims)
    ims = [im.convert("RGB").resize((int(im.width * h / im.height), h)) if im.height != h
           else im.convert("RGB") for im in ims]
    w = sum(im.width for im in ims) + pad * (len(ims) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + pad
    return out


def grid(rows, pad=4, bg=(255, 255, 255)) -> Image.Image:
    """Stack panels vertically."""
    strips = [panel(r, pad, bg) for r in rows]
    w = max(s.width for s in strips)
    h = sum(s.height for s in strips) + pad * (len(strips) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for s in strips:
        out.paste(s, (0, y))
        y += s.height + pad
    return out


def preview_row(image_uint8, gt_labels, pred_labels, parsed, gt_scene=None) -> list:
    """input / GT labels / predicted labels / parsed overlay (DESIGN.md §3.4 previews)."""
    img = Image.fromarray(to_uint8_image(image_uint8))
    over = draw_parsed(image_uint8, parsed, gt_scene) if parsed is not None else \
        Image.fromarray(np.full_like(to_uint8_image(image_uint8), 128))
    return [img, colorize(np.asarray(gt_labels, np.uint8)),
            colorize(np.asarray(pred_labels, np.uint8)), over]
