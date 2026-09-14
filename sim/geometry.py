"""Analytic geometry: ray casting and point-to-primitive distances.

Distance helpers are numpy-broadcast friendly (used to rasterise occupancy
grids); ray helpers are scalar (used per sector, per agent, per step).
"""
from __future__ import annotations

import math

import numpy as np

INF = math.inf


# ------------------------------------------------------------------ distances

def point_seg_dist(px, py, seg):
    """Distance from point(s) to a segment.  px/py may be arrays."""
    (ax, ay), (bx, by) = seg
    ex, ey = bx - ax, by - ay
    ll = ex * ex + ey * ey
    if ll < 1e-12:
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * ex + (py - ay) * ey) / ll
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * ex), py - (ay + t * ey))


def point_seg_dist_scalar(px, py, seg) -> float:
    (ax, ay), (bx, by) = seg
    ex, ey = bx - ax, by - ay
    ll = ex * ex + ey * ey
    if ll < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * ex + (py - ay) * ey) / ll
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + t * ex), py - (ay + t * ey))


# ------------------------------------------------------------------ ray casts

def ray_circle(px, py, dx, dy, cx, cy, r) -> float:
    """Nearest positive t with |p + t*d - c| = r, or inf.  d must be unit."""
    fx, fy = px - cx, py - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0.0:
        return INF
    s = math.sqrt(disc)
    t1 = -b - s
    if t1 > 1e-9:
        return t1
    t2 = -b + s
    return t2 if t2 > 1e-9 else INF


def ray_segment(px, py, dx, dy, seg) -> float:
    """Nearest positive t where the ray crosses the segment, or inf.

    Solves  P + t*D = A + u*E  with E = B - A, keeping t > 0 and u in [0, 1].
    """
    (ax, ay), (bx, by) = seg
    ex, ey = bx - ax, by - ay
    det = ex * dy - ey * dx
    if abs(det) < 1e-12:
        return INF                      # parallel (grazing hits are ignored)
    wx, wy = ax - px, ay - py
    t = (ex * wy - ey * wx) / det
    if t <= 1e-9:
        return INF
    u = (dx * wy - dy * wx) / det
    if u < 0.0 or u > 1.0:
        return INF
    return t


def ray_capsule(px, py, dx, dy, seg, half) -> float:
    """Ray vs a segment thickened by `half` (a capsule): nearest surface hit."""
    if half <= 1e-9:
        return ray_segment(px, py, dx, dy, seg)
    (ax, ay), (bx, by) = seg
    ex, ey = bx - ax, by - ay
    ll = math.hypot(ex, ey)
    if ll < 1e-12:
        return ray_circle(px, py, dx, dy, ax, ay, half)
    nx, ny = -ey / ll, ex / ll          # unit normal
    best = INF
    for s in (half, -half):             # the two flat sides
        off = ((ax + s * nx, ay + s * ny), (bx + s * nx, by + s * ny))
        t = ray_segment(px, py, dx, dy, off)
        if t < best:
            best = t
    for cx, cy in ((ax, ay), (bx, by)):  # the two round caps
        t = ray_circle(px, py, dx, dy, cx, cy, half)
        if t < best:
            best = t
    return best
