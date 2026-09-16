"""Geometry extraction: a 6-class label map (+ optional heat / dir) -> ``Parsed``.

Imports numpy and cv2 only (no torch), so the pipeline code that later runs
the policy can parse without a GPU stack.  All thresholds live in
``runconfig.ExtractConfig``.

Pixel ↔ world: ``x = px / scale``, ``y = world - py / scale``,
``scale = S / world`` (the one y-flip, mirroring ``render._to_px``).  Pixel
``(row, col)`` has continuous coordinates ``(col, row)``.

Measured rendering facts the detectors rely on (512 px, from ``sim/render.py``):
- The agent label ring is drawn with PIL's ``ellipse(outline, width=lw)``,
  which draws *inward* from the bounding circle of radius ``r_px``: ring pixels
  span radii ``[r_px - lw_px, r_px + 0.2 lw_px]``.  The template annulus uses
  exactly those bounds.
- The heading line runs from the centre to ``heading_len_mult · r_px``.
- Obstacle contour points lie at radius ``r_px`` (no half-pixel bias).
- A region drawn from world edge ``e`` covers pixels ``floor(e · scale)``
  inclusive on both sides, so a pixel bound ``b`` maps back to ``(b + 0.5) / scale``.

Invariants:
- ``extract`` is a pure function of its inputs (no RNG, no global state).
- Parsed walls are *surface* polygon edges with ``wall_thickness = 0``;
  build the simulator with ``sim_config(base_cfg)``, never the generator's config.
- ``extract`` returns ``None`` (parse failure) only when there is no start–goal
  pair or no agent; everything else comes back with ``diagnostics``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import _paths  # noqa: F401
from config import C_AGENT, C_GOAL, C_OBST, C_START, C_WALL, Config, StyleConfig
from runconfig import ExtractConfig
from scene import Agent, Obstacle, Region, Scene
from targets import px_to_world, world_to_px

_STYLE = StyleConfig()
REF_SIZE = 512                              # resolution the label line width is defined at


@dataclass
class Parsed:
    scene: Scene
    size: int
    world: dict
    diagnostics: dict = field(default_factory=dict)


# ------------------------------------------------------------------ geometry

def pixel_geometry(world: dict, size: int) -> dict:
    """Agent ring radius, label line width and heading length in px at `size`."""
    W = float(world["size"])
    scale = size / W
    lw512 = int(round(float(world.get("agent_line_width", _STYLE.agent_line_width)))) + 3
    return {
        "scale": scale,
        "r_px": float(world["agent_radius"]) * scale,
        "lw_px": lw512 * size / REF_SIZE,
        "hlen_px": float(world.get("heading_len_mult", _STYLE.heading_len_mult))
        * float(world["agent_radius"]) * scale,
    }


def _kernel(size_px: int) -> np.ndarray:
    return np.ones((size_px, size_px), np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Set every background pocket not connected to the image border."""
    h, w = mask.shape
    inv = (mask == 0).astype(np.uint8)
    n, cc = cv2.connectedComponents(inv, connectivity=4)
    border = np.zeros(n, bool)
    border[np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))] = True
    border[0] = True
    holes = ~border[cc]
    out = mask.copy()
    out[holes] = 1
    return out


def _subpixel(resp: np.ndarray, iy: int, ix: int):
    """3×3 separable quadratic peak refinement; offsets clipped to ±1."""
    h, w = resp.shape
    dx = dy = 0.0
    if 0 < ix < w - 1:
        l, c, r = float(resp[iy, ix - 1]), float(resp[iy, ix]), float(resp[iy, ix + 1])
        den = l - 2 * c + r
        if den < -1e-9:
            dx = float(np.clip(0.5 * (l - r) / den, -1.0, 1.0))
    if 0 < iy < h - 1:
        u, c, d = float(resp[iy - 1, ix]), float(resp[iy, ix]), float(resp[iy + 1, ix])
        den = u - 2 * c + d
        if den < -1e-9:
            dy = float(np.clip(0.5 * (u - d) / den, -1.0, 1.0))
    return ix + dx, iy + dy


def _peaks(resp: np.ndarray, thresh: float, min_dist: float) -> list:
    """Local maxima ≥ thresh, greedy NMS by distance, sub-pixel refined: [(cx, cy, score)]."""
    k = max(3, 2 * int(math.ceil(min_dist / 2)) + 1)
    dil = cv2.dilate(resp, _kernel(k))
    cand = np.argwhere((resp >= dil) & (resp >= thresh))
    if len(cand) == 0:
        return []
    scores = resp[cand[:, 0], cand[:, 1]]
    order = np.argsort(-scores, kind="stable")
    kept = []
    for o in order:
        iy, ix = int(cand[o, 0]), int(cand[o, 1])
        if any((ix - kx) ** 2 + (iy - ky) ** 2 < min_dist * min_dist for kx, ky, _ in kept):
            continue
        kept.append((ix, iy, float(scores[o])))
    out = []
    for ix, iy, s in kept:
        cx, cy = _subpixel(resp, iy, ix)
        out.append((cx, cy, s))
    return out


# ------------------------------------------------------------------ agents

def ring_template(r_px: float, lw_px: float) -> np.ndarray:
    """Annulus matching the drawn ring, normalised so a perfect ring sums to ≈ 1."""
    inner = max(0.0, r_px - lw_px)
    outer = r_px + 0.2 * lw_px
    outer = max(outer, inner + 1.0)             # never thinner than one pixel
    R = int(math.ceil(outer)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(xx, yy)
    ring = ((d >= inner - 0.5) & (d <= outer + 0.5)).astype(np.float32)
    # normalise by the pixel count of the drawn band, not of the (wider) template,
    # so a perfect isolated ring scores ≈ 1 at every resolution
    expected = math.pi * (outer ** 2 - inner ** 2)
    return ring / max(1.0, min(expected, float(ring.sum())))


def ring_response(agent_mask: np.ndarray, r_px: float, lw_px: float) -> np.ndarray:
    """Hough accumulator for a known radius, computed exactly by cross-correlation."""
    k = ring_template(r_px, lw_px)
    m = agent_mask.astype(np.float32)
    return cv2.filter2D(m, -1, k, borderType=cv2.BORDER_CONSTANT)


def detect_agents_ring(agent_mask, r_px, lw_px, ecfg=ExtractConfig()) -> list:
    if not agent_mask.any():
        return []
    resp = ring_response(agent_mask, r_px, lw_px)
    return _peaks(resp, ecfg.ring_thresh, ecfg.nms_factor * r_px)


def detect_agents_heat(heat, r_px, ecfg=ExtractConfig()) -> list:
    return _peaks(np.asarray(heat, np.float32), ecfg.heat_thresh, ecfg.nms_factor * r_px)


def ring_outer_px(r_px: float, lw_px: float) -> float:
    """Outer radius of the drawn label ring (PIL draws the outline inward)."""
    return r_px + 0.2 * lw_px


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ring_band(r_px: float, lw_px: float):
    """(inner, outer) radii of the drawn ring band, with a one-pixel margin each side."""
    return max(0.0, r_px - lw_px) - 1.0, ring_outer_px(r_px, lw_px) + 1.0


def refine_centers(agent_mask, centers, r_px, lw_px, iters=4) -> list:
    """Re-centre each detection with a geometric circle fit to its ring-band pixels.

    The ring-vote peak is flat over ~2 px (the template annulus is wide), so
    the quadratic sub-pixel fit is only good to about a pixel.  A Gauss-Newton
    fit of a circle (free radius) to the pixels in the ring band recovers the
    centre to a fraction of a pixel and, unlike a centroid, is not biased when
    part of the ring is missing (a neighbour's band overlaps it).  Band pixels
    shared with a neighbour's band go to the ring whose nominal radius they
    match better; the heading line's pixels inside the band lie at ring radius
    anyway.
    """
    ys, xs = np.nonzero(agent_mask)
    C = np.array([(c[0], c[1]) for c in centers], float).reshape(-1, 2)
    if len(C) == 0 or len(xs) == 0:
        return [tuple(c) for c in C]
    inner, outer = ring_band(r_px, lw_px)
    mid = 0.5 * (inner + outer)
    dist = np.sqrt((xs[:, None] - C[None, :, 0]) ** 2 + (ys[:, None] - C[None, :, 1]) ** 2)
    in_band = (dist >= inner) & (dist <= outer)
    best = np.abs(dist - mid).argmin(1)                           # ring this pixel fits best
    out = []
    for i, c in enumerate(C):
        sel = in_band[:, i] & (best == i)
        if sel.sum() < 8:
            out.append((float(c[0]), float(c[1])))
            continue
        px, py = xs[sel].astype(float), ys[sel].astype(float)
        cx, cy = float(c[0]), float(c[1])
        for _ in range(iters):
            dx, dy = px - cx, py - cy
            d = np.hypot(dx, dy)
            d[d < 1e-9] = 1e-9
            ux, uy = dx / d, dy / d
            r = d - d.mean()                                       # residuals for the mean radius
            A = np.array([[(ux * ux).sum(), (ux * uy).sum()], [(ux * uy).sum(), (uy * uy).sum()]])
            b = np.array([(r * ux).sum(), (r * uy).sum()])
            if abs(np.linalg.det(A)) < 1e-9:
                break
            step = np.linalg.solve(A, b)
            step = np.clip(step, -2.0, 2.0)                        # never jump to a neighbour
            cx, cy = cx + step[0], cy + step[1]
        out.append((cx, cy))
    return out


def _angular_clusters(ang: np.ndarray, n_bins: int = 16) -> list:
    """Peaks of a circular histogram: [(centre_angle, weight)], strongest first."""
    h, _ = np.histogram(ang, bins=n_bins, range=(-math.pi, math.pi))
    hs = h + np.roll(h, 1) + np.roll(h, -1)
    peaks = []
    for b in np.argsort(-hs, kind="stable"):
        if hs[b] < max(1.0, 0.2 * hs.max()):
            break
        if all(min(abs(b - q), n_bins - abs(b - q)) >= 2 for q, _ in peaks):
            peaks.append((int(b), float(hs[b])))
    width = 2 * math.pi / n_bins
    return [(-math.pi + (b + 0.5) * width, w) for b, w in peaks]


def _line_direction(along, perp, lo, hi, lw_px, binw=2.0):
    """Angle correction from the line's cross-sections, robust to one-sided cuts.

    The drawn line has a known width, so every 2-px slice across it is a
    segment of length lw_px centred on the true axis.  A complete slice gives
    its centre directly; a slice with one side cut away (a neighbour's ring
    band overlapping the line) gives the centre from the intact edge; a slice
    without a usable edge is skipped.  The centres are regressed through the
    origin (the ring centre) so the correction is atan(slope).  None if no
    slice is usable.
    """
    hw = 0.5 * lw_px
    full = lw_px * binw
    a_list, c_list = [], []
    b = lo
    while b < hi:
        m = (along >= b) & (along < b + binw)
        n = int(m.sum())
        if n >= 2:
            q = perp[m]
            if n >= 0.7 * full:
                c = 0.5 * (q.min() + q.max())
            else:
                top_ok, bot_ok = q.max() >= hw - 1.0, q.min() <= -hw + 1.0
                c = q.max() - hw if top_ok and not bot_ok else (
                    q.min() + hw if bot_ok and not top_ok else None)
            if c is not None:
                a_list.append(b + 0.5 * binw)
                c_list.append(c)
        b += binw
    if not a_list:
        return None
    a = np.asarray(a_list)
    c = np.asarray(c_list)
    return math.atan(float((a * c).sum() / (a * a).sum()))


def _refine_angle(ang, ox, oy, theta, lo, hi, lw_px, windows=(0.6, 0.35)):
    """Mean offset within tighter angular windows, then along the line only.

    The middle passes keep the pixels within half a line width of the ray
    (rejecting a neighbour's pixels that share the angle but not the line)
    and take the direction of their mean offset from the centre.  The final
    passes re-centre each cross-section from its intact edge, which removes
    the bias of a strip cut obliquely by a neighbour's ring band.
    """
    for w in windows:
        sel = np.abs(_wrap(ang - theta)) <= w
        if not sel.any():
            return theta
        theta = math.atan2(oy[sel].mean(), ox[sel].mean())
    for it in range(5):
        ux, uy = math.cos(theta), math.sin(theta)
        along = ox * ux + oy * uy
        perp = -ox * uy + oy * ux
        sel = (along > lo) & (along < hi) & (np.abs(perp) <= 0.5 * lw_px + 1.5)
        if sel.sum() < 2:
            break
        if it < 2:
            theta = math.atan2(oy[sel].mean(), ox[sel].mean())
            continue
        d = _line_direction(along[sel], perp[sel], lo, hi, lw_px)
        if d is None or abs(d) < 1e-4:
            break
        theta = _wrap(theta + d)
    return theta


def estimate_headings(agent_mask, centers, r_px, lw_px, hlen_px):
    """θ per centre from the heading line; (thetas, uncertain_flags).

    For centre i, candidate pixels are agent pixels at distance in
    (r_px + 0.5 lw_px, 1.3 hlen_px) -- outside its own ring, a little beyond
    the line tip -- that are not inside any *other* detection's ring disk.
    The line is a tight angular cluster of those pixels; θ is the mean offset
    of the pixels around the chosen cluster.  A neighbour's line pointing at
    this agent forms a second, equally large cluster, and the two cannot be
    told apart from the annulus alone, so the cluster is chosen by the line
    *root*: the pixels inside the ring's inner radius belong to this agent's
    line only (any neighbour's tip stops at its own ring at normal spacing).
    Without a usable root (128 px, or a filled ring) the centroid of the whole
    blob decides (the ring is symmetric, so the offset points along the line);
    without either cue the largest cluster wins.

    Fewer than two candidates: the blob centroid direction, flagged uncertain.
    """
    ys, xs = np.nonzero(agent_mask)
    if len(centers) == 0:
        return [], []
    C = np.array([(c[0], c[1]) for c in centers], float)
    if len(xs) == 0:
        return [0.0] * len(C), [True] * len(C)
    dist = np.sqrt((xs[:, None] - C[None, :, 0]) ** 2 + (ys[:, None] - C[None, :, 1]) ** 2)
    inner, outer = ring_band(r_px, lw_px)
    inside = (dist >= inner) & (dist <= outer)                        # (P, N): in ring i's band
    n_inside = inside.sum(1)
    lo, hi = r_px + 0.5 * lw_px, 1.3 * hlen_px
    root_r = r_px - lw_px + 0.3                      # inside the ring's hole
    thetas, flags = [], []
    for i, (cx, cy) in enumerate(C):
        di = dist[:, i]
        free = (n_inside - inside[:, i]) == 0                        # not in another ring's band
        cand = free & (di > lo) & (di < hi)
        if cand.sum() >= 2:
            ox, oy = xs[cand] - cx, -(ys[cand] - cy)                 # world-oriented offsets
            ang = np.arctan2(oy, ox)
            clusters = _angular_clusters(ang)
            theta = clusters[0][0]
            if len(clusters) > 1:
                cue = _root_direction(xs, ys, di, root_r, cx, cy)
                if cue is None:
                    cue = _blob_direction(xs, ys, free & (di < hi), cx, cy)
                if cue is not None:
                    theta = min(clusters, key=lambda c: abs(_wrap(c[0] - cue)))[0]
            thetas.append(_refine_angle(ang, ox, oy, theta, lo, hi, lw_px))
            flags.append(False)
            continue
        cue = _blob_direction(xs, ys, free & (di < hi), cx, cy)
        thetas.append(cue if cue is not None else 0.0)
        flags.append(True)
    return thetas, flags


def _root_direction(xs, ys, di, root_r, cx, cy):
    """Direction of the line root inside the ring's hole, or None if unusable."""
    root = di < root_r
    if root.sum() < 3:
        return None
    rx, ry = (xs[root] - cx).mean(), -(ys[root] - cy).mean()
    if math.hypot(rx, ry) < 0.25:                                  # a symmetric root: no cue
        return None
    return math.atan2(ry, rx)


def _blob_direction(xs, ys, sel, cx, cy):
    """Direction of the centroid of the whole blob (ring + line): the ring is
    symmetric about the centre, so the offset points along the line."""
    if sel.sum() < 3:
        return None
    bx, by = (xs[sel] - cx).mean(), -(ys[sel] - cy).mean()
    if math.hypot(bx, by) < 0.15:
        return None
    return math.atan2(by, bx)


def headings_from_dir(dir_map, centers, r_px):
    """θ = atan2(mean sin, mean cos) over `dir` within 0.8 r_px of each peak."""
    S = dir_map.shape[-1]
    yy, xx = np.mgrid[0:S, 0:S]
    out = []
    for cx, cy in centers:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.8 * r_px) ** 2
        if not m.any():
            iy, ix = min(max(int(round(cy)), 0), S - 1), min(max(int(round(cx)), 0), S - 1)
            m[iy, ix] = True
        out.append(math.atan2(float(dir_map[1][m].mean()), float(dir_map[0][m].mean())))
    return out


# --------------------------------------------------------------- obstacles

def fit_circle_kasa(pts: np.ndarray):
    """Algebraic least-squares circle (Kåsa): (cx, cy, r) or None if degenerate."""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


def extract_obstacles(obst_mask, wall_mask, scale, world, ecfg=ExtractConfig()) -> list:
    m = obst_mask.astype(np.uint8)
    if not m.any():
        return []
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(3))
    m = _fill_holes(m)
    r_min = float(world.get("obstacle_r_min", Config().obstacle_r_min))
    min_area = math.pi * (0.5 * r_min * scale) ** 2
    wall_near = cv2.dilate(wall_mask.astype(np.uint8), _kernel(3)).astype(bool)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    W = float(world["size"])
    S = obst_mask.shape[0]
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (cc == k).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        pts = max(cs, key=len).reshape(-1, 2)
        keep = ~wall_near[pts[:, 1], pts[:, 0]]
        # a disk clipped by the image edge has a straight run along the border
        # that is not part of the circle
        keep &= (pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < S - 1) & (pts[:, 1] < S - 1)
        fit = fit_circle_kasa(pts[keep]) if keep.sum() >= 6 else None
        if fit is None:
            (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32).reshape(-1, 1, 2))
            fit = (float(cx), float(cy), float(r))
        cx, cy, r = fit
        wx, wy = px_to_world(W, S, cx, cy)
        out.append(Obstacle((float(wx), float(wy)), float(r / scale)))
    return out


# ------------------------------------------------------------------- walls

def extract_walls(wall_mask, scale, world, ecfg=ExtractConfig()) -> list:
    m = wall_mask.astype(np.uint8)
    if not m.any():
        return []
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(3))
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    W = float(world["size"])
    S = wall_mask.shape[0]
    segs = []
    for c in cs:
        if len(c) < 3 or cv2.contourArea(c) < ecfg.min_wall_area_px:
            continue
        poly = cv2.approxPolyDP(c, ecfg.poly_eps_px, True).reshape(-1, 2)
        if len(poly) < 2:
            continue
        for i in range(len(poly)):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
            if ax == bx and ay == by:
                continue
            a = px_to_world(W, S, float(ax), float(ay))
            b = px_to_world(W, S, float(bx), float(by))
            segs.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return segs


# ----------------------------------------------------------------- regions

def _boxes(mask, scale, world, min_area, seal_px=0.0) -> list:
    """Bounding boxes of the mask's components (world units), smallest area dropped.

    `seal_px`: close gaps up to this width first.  Agents are drawn over the
    regions, and several rings side by side can cut a region mask in two; a
    closing just wider than the label line width seals those cuts while a wall
    (twice as thick) still splits a region.
    """
    m = mask.astype(np.uint8)
    if seal_px > 0:
        k = 2 * int(math.ceil(seal_px / 2 + 0.5)) + 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(k))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    W = float(world["size"])
    S = mask.shape[0]
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (int(stats[k, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        x0, y1 = px_to_world(W, S, x + 0.5, y + 0.5)
        x1, y0 = px_to_world(W, S, x + w - 0.5, y + h - 0.5)
        out.append((Region(float(x0), float(y0), float(x1), float(y1)), int(stats[k, cv2.CC_STAT_AREA])))
    return out


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def pair_score(start: Region, goal: Region) -> float:
    """How well two regions face each other: shared extent on one axis, as a fraction."""
    sy = _overlap(start.y0, start.y1, goal.y0, goal.y1) / max(1e-9, min(start.height, goal.height))
    sx = _overlap(start.x0, start.x1, goal.x0, goal.x1) / max(1e-9, min(start.width, goal.width))
    return max(sx, sy)


def extract_regions(start_mask, goal_mask, scale, world, ecfg=ExtractConfig()):
    """[(start, goal)] pairs sorted by start (x0, y0); plus the number of unpaired regions."""
    seal = pixel_geometry(world, start_mask.shape[0])["lw_px"]
    starts = [r for r, _ in _boxes(start_mask, scale, world, ecfg.min_region_area_px, seal)]
    goals = [r for r, _ in _boxes(goal_mask, scale, world, ecfg.min_region_area_px, seal)]
    pairs = []
    if len(starts) == 1 and len(goals) == 1:
        pairs = [(starts[0], goals[0])]
    else:
        cands = sorted(((pair_score(s, g), i, j) for i, s in enumerate(starts)
                        for j, g in enumerate(goals)), reverse=True)
        used_s, used_g = set(), set()
        for score, i, j in cands:
            if score < 0.5 or i in used_s or j in used_g:
                continue
            used_s.add(i)
            used_g.add(j)
            pairs.append((starts[i], goals[j]))
    pairs.sort(key=lambda p: (p[0].x0, p[0].y0))
    unpaired = len(starts) + len(goals) - 2 * len(pairs)
    return pairs, unpaired


# ----------------------------------------------------------------- driver

def extract(labels, world, heat=None, dir=None, ecfg=ExtractConfig()):
    labels = np.asarray(labels)
    S = labels.shape[0]
    geo = pixel_geometry(world, S)
    scale, r_px, lw_px, hlen_px = geo["scale"], geo["r_px"], geo["lw_px"], geo["hlen_px"]
    W = float(world["size"])
    diag = {"n_dropped_components": 0, "unpaired_regions": 0, "heading_uncertain": [],
            "agent_scores": [], "agent_method": "ring", "heat_fallback": False}

    agent_mask = labels == C_AGENT
    wall_mask = labels == C_WALL
    obst_mask = labels == C_OBST

    # --- regions -----------------------------------------------------------
    pairs, unpaired = extract_regions(labels == C_START, labels == C_GOAL, scale, world, ecfg)
    diag["unpaired_regions"] = unpaired
    if not pairs:
        return None

    # --- agents ------------------------------------------------------------
    dets = []
    if heat is not None:
        diag["agent_method"] = "heat"
        dets = detect_agents_heat(heat, r_px, ecfg)
        if not dets and agent_mask.any():
            diag["heat_fallback"] = True
            diag["agent_method"] = "ring"
            dets = detect_agents_ring(agent_mask, r_px, lw_px, ecfg)
    else:
        dets = detect_agents_ring(agent_mask, r_px, lw_px, ecfg)
    if not dets:
        return None
    centers = [(cx, cy) for cx, cy, _ in dets]
    if diag["agent_method"] == "ring":
        centers = refine_centers(agent_mask, centers, r_px, lw_px)
    if diag["agent_method"] == "heat" and dir is not None:
        thetas = headings_from_dir(np.asarray(dir), centers, r_px)
        flags = [False] * len(thetas)
    else:
        thetas, flags = estimate_headings(agent_mask, centers, r_px, lw_px, hlen_px)
    diag["agent_scores"] = [float(s) for _, _, s in dets]
    diag["heading_uncertain"] = [bool(f) for f in flags]

    # --- static geometry --------------------------------------------------
    obstacles = extract_obstacles(obst_mask, wall_mask, scale, world, ecfg)
    walls = extract_walls(wall_mask, scale, world, ecfg)

    # --- assemble -------------------------------------------------------
    agents = []
    for (cx, cy), th in zip(centers, thetas):
        wx, wy = px_to_world(W, S, cx, cy)
        gi = _group_of((wx, wy), pairs)
        agents.append(Agent(pos=(float(wx), float(wy)), heading=float(th),
                            goal_region=pairs[gi][1] if gi > 0 else None))
    scene = Scene(walls=walls, obstacles=obstacles, agents=agents,
                  start_region=pairs[0][0], goal_region=pairs[0][1],
                  layout="parsed", groups=list(pairs))
    pw = dict(world)
    pw["img_size"] = S
    pw["wall_thickness"] = 0.0
    return Parsed(scene=scene, size=S, world=pw, diagnostics=diag)


def _group_of(p, pairs) -> int:
    for i, (s, _) in enumerate(pairs):
        if s.contains(p):
            return i
    return int(np.argmin([s.distance_to(p) for s, _ in pairs]))


# ------------------------------------------------------------------ config

def sim_config(base_cfg: Config) -> Config:
    """Simulator config for parsed scenes: surface walls have no thickness."""
    return base_cfg.merged(wall_thickness=0.0)


def parsed_to_json(parsed: Parsed, sid=None) -> dict:
    d = parsed.scene.as_dict(None, sid)
    d["layout"] = None
    d["bottleneck"] = None
    d["parsed"] = True
    w = parsed.world
    d["world"] = {"size": w["size"], "img_size": parsed.size, "agent_radius": w["agent_radius"],
                  "wall_thickness": 0.0, "fov_deg": w.get("fov_deg", Config().fov_deg),
                  "fov_range": w.get("fov_range", Config().fov_range)}
    d["diagnostics"] = parsed.diagnostics
    return d


def parsed_from_json(d: dict) -> Parsed:
    scene = Scene.from_dict(d)
    scene.layout = "parsed"
    return Parsed(scene=scene, size=int(d["world"]["img_size"]), world=dict(d["world"]),
                  diagnostics=dict(d.get("diagnostics", {})))
