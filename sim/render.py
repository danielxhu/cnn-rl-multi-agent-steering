"""Scene rendering: RGB image plus a 6-class label map, drawn from the same geometry.

World y increases upward; image row 0 is the top. `_to_px` is the only place
that flips (pinned by tests/run_tests.py::test_world_y_up_maps_to_image_y_down).

Draw order (later overwrites earlier):
    background -> unreachable fill -> regions -> walls -> obstacles -> agents
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import C_AGENT, C_BG, C_GOAL, C_OBST, C_START, C_WALL, Config
from reachability import unreachable_free


def _to_px(cfg, x, y):
    return (x * cfg.scale, (cfg.world - y) * cfg.scale)


def _rect_px(cfg, region):
    x0, y0 = _to_px(cfg, region.x0, region.y1)      # top-left in image space
    x1, y1 = _to_px(cfg, region.x1, region.y0)
    return [x0, y0, x1, y1]


def _jitter_color(c, amount, rng):
    if amount <= 0 or rng is None:
        return tuple(int(v) for v in c)
    off = rng.uniform(-amount, amount, size=3)
    return tuple(int(np.clip(v + o, 0, 255)) for v, o in zip(c, off))


def _dashed_rect(draw, box, color, width, dash):
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
             ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (ax, ay), (bx, by) in edges:
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-6:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        t = 0.0
        while t < length:
            e = min(t + dash, length)
            draw.line([ax + ux * t, ay + uy * t, ax + ux * e, ay + uy * e],
                      fill=color, width=int(width))
            t += 2 * dash


def render(scene, cfg=None, rng=None, fill_unreachable=True, clear=None):
    """Return (RGB PIL image, uint8 label array of shape (img, img))."""
    cfg = cfg or Config()
    st = cfg.style
    n = cfg.img_size
    lw = st.agent_line_width
    if st.jitter_line_width > 0 and rng is not None:
        lw = max(1.0, lw + rng.uniform(-st.jitter_line_width, st.jitter_line_width))

    bg = list(st.background)
    if st.jitter_background > 0 and rng is not None:
        bg = [int(np.clip(v + rng.uniform(-st.jitter_background, st.jitter_background), 0, 255))
              for v in bg]
    wall_c = _jitter_color(st.wall_color, st.jitter_color, rng)
    obst_c = _jitter_color(st.obstacle_color, st.jitter_color, rng)
    agent_c = _jitter_color(st.agent_color, st.jitter_color, rng)

    rgb = np.zeros((n, n, 3), np.uint8)
    rgb[:] = bg
    lab = np.full((n, n), C_BG, np.uint8)

    # --- dead space beyond the walls, painted as wall -----------------------
    if fill_unreachable:
        mask = unreachable_free(cfg, scene, clear=clear)
        if mask.any():
            big = cv2.resize(mask.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST)
            big = np.flipud(big)                     # grid row 0 is the world floor
            # dilate slightly to close the upsampling seam; the wall is drawn over it
            grow = max(1, int(cfg.wall_thickness * cfg.scale / 4))
            k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
            big = cv2.dilate(big, k).astype(bool)
            rgb[big] = wall_c
            lab[big] = C_WALL

    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    lab_img = Image.fromarray(lab, "L")
    dl = ImageDraw.Draw(lab_img)

    # --- regions: translucent so whatever sits on them stays readable --------
    overlay = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    do = ImageDraw.Draw(overlay)
    pairs = scene.groups or [(scene.start_region, scene.goal_region)]
    for start, goal in pairs:
        if start is not None:
            box = _rect_px(cfg, start)
            do.rectangle(box, fill=(*st.start_color, st.start_alpha))
            _dashed_rect(do, box, (*st.start_color, 200), st.region_border_width, st.region_dash)
            dl.rectangle(box, fill=C_START)
        if goal is not None:
            box = _rect_px(cfg, goal)
            do.rectangle(box, fill=(*st.goal_color, st.goal_alpha))
            do.rectangle(box, outline=(*st.goal_color, 220), width=int(st.region_border_width))
            dl.rectangle(box, fill=C_GOAL)
    img = Image.alpha_composite(img, overlay)
    d = ImageDraw.Draw(img)

    # --- walls: capsules, so render and physics agree on where the surface is
    wpx = cfg.wall_thickness * cfg.scale
    for (ax, ay), (bx, by) in scene.walls:
        p0, p1 = _to_px(cfg, ax, ay), _to_px(cfg, bx, by)
        d.line([p0, p1], fill=wall_c, width=int(round(wpx)))
        dl.line([p0, p1], fill=C_WALL, width=int(round(wpx)))
        for px, py in (p0, p1):                  # round the joints
            cap = [px - wpx / 2, py - wpx / 2, px + wpx / 2, py + wpx / 2]
            d.ellipse(cap, fill=wall_c)
            dl.ellipse(cap, fill=C_WALL)

    for o in scene.obstacles:
        cx, cy = _to_px(cfg, *o.c)
        rp = o.r * cfg.scale
        box = [cx - rp, cy - rp, cx + rp, cy + rp]
        d.ellipse(box, fill=obst_c)
        dl.ellipse(box, fill=C_OBST)

    # --- agents: hollow circle + heading line, no FOV wedge -----------------
    rp = cfg.agent_radius * cfg.scale
    hlen = st.heading_len_mult * cfg.agent_radius * cfg.scale
    for a in scene.agents:
        cx, cy = _to_px(cfg, *a.pos)
        box = [cx - rp, cy - rp, cx + rp, cy + rp]
        hx, hy = cx + hlen * math.cos(a.heading), cy - hlen * math.sin(a.heading)
        d.ellipse(box, outline=agent_c, width=int(round(lw)))
        d.line([cx, cy, hx, hy], fill=agent_c, width=int(round(lw)))
        # label drawn fatter than the ink so it survives downsampling
        dl.ellipse(box, outline=C_AGENT, width=int(round(lw)) + 3)
        dl.line([cx, cy, hx, hy], fill=C_AGENT, width=int(round(lw)) + 3)

    img = img.convert("RGB")

    # --- domain randomisation ----------------------------------------------
    if st.blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(st.blur_sigma))
    if st.noise_std > 0 and rng is not None:
        arr = np.asarray(img, np.float32) + rng.normal(0, st.noise_std, (n, n, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    return img, np.asarray(lab_img, np.uint8)


PALETTE = np.array([
    [255, 255, 255],   # background
    [40, 40, 40],      # wall
    [140, 140, 140],   # obstacle
    [232, 151, 58],    # start region
    [43, 92, 230],     # goal region
    [92, 45, 145],     # agent
], np.uint8)


def colorize(labels: np.ndarray) -> Image.Image:
    """Label map -> a picture a human can check at a glance."""
    return Image.fromarray(PALETTE[labels], "RGB")
