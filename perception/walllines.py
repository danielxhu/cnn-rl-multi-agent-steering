"""Wall label mask -> straight wall centre lines, in the generator's form.

``extract(..., ecfg.wall_mode="lines")`` uses this instead of the surface
polygons of ``extract.extract_walls``: a hand-drawn wobbly line, or the edge of
a hatched dead-space block, comes back as a few straight centre-line segments
with the generator's wall thickness, so the parsed scene reads like a
generated one (``render.render`` draws it, ``Simulator`` runs it with the
generator's ``wall_thickness``).

Steps (all thresholds in world units, in ``ExtractConfig``):

1. **Rim.** A wall is a pen line: its centre line is half a stroke width from
   the free space. The stroke width ``w`` is measured on the free-standing
   strokes (twice the median distance-transform ridge), else it is the world's
   wall thickness. Wall pixels farther than ``w`` from free space are the
   inside of a filled dead-space block and are dropped, so a block
   contributes the line along its edge only. Obstacle pixels count as neither
   wall nor free space.
2. **Skeleton.** Zhang–Suen thinning of the rim; skeleton pixels within
   ``line_border`` of the image edge (the printed / generated boundary band)
   are dropped, the four boundary walls are added exactly at the end.
3. **Pieces.** The skeleton is split at junctions into branches, each branch
   is ordered and cut by Douglas–Peucker at ``line_tol``, and every piece is
   replaced by its least-squares line.
4. **Clean-up.** Near-axis segments are made axis-parallel (every layout wall
   is); collinear pieces are merged across junction gaps, and across gaps an
   obstacle was drawn over; ends near another wall are moved onto the
   intersection (corners, T junctions, the overshoot of a hand-drawn corner);
   free ends near the world edge are extended to it; short leftovers are
   dropped.

numpy and cv2 only. Pixel ↔ world as in ``extract.py``.

Invariants:
- Output segments are wall **centre lines**; the parsed world then carries
  the generator's ``wall_thickness`` (``extract`` sets it), never 0.
- A doorway is never closed: collinear pieces are only joined across a gap
  shorter than ``line_join`` or covered by obstacle / wall labels.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import _paths  # noqa: F401
from scene import boundary_walls
from targets import px_to_world


# ------------------------------------------------------------------ raster

def stroke_width_px(dt: np.ndarray, scale: float, border_px: int, fallback_px: float) -> float:
    """Typical wall stroke width (px): twice the median distance-transform ridge
    value over free-standing strokes. Ridges thicker than 2 units, or within
    2.5 units of such a block, belong to filled dead space and do not count; a
    scene with no free-standing stroke gets `fallback_px` (the world's wall
    thickness, exact for the generator's own style)."""
    core = (dt > 2.0 * scale).astype(np.uint8)
    r = int(math.ceil(2.5 * scale))
    near = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & ~near
    b = border_px
    ridge[:b], ridge[-b:], ridge[:, :b], ridge[:, -b:] = False, False, False, False
    if ridge.sum() < 3 * scale:                      # less than ~3 units of free-standing stroke
        return fallback_px
    return float(2.0 * np.median(dt[ridge]))


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang–Suen thinning (vectorised); uint8 0/1 in, 0/1 out."""
    img = (mask > 0).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(ring[:8])
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img


_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # 4-neighbours first


def branches(skel: np.ndarray, min_px: int) -> list:
    """Skeleton -> ordered pixel paths [(n, 2) (x, y)], split at junctions."""
    s = skel.astype(np.uint8)
    nb = cv2.filter2D(s, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - s
    junction = (s > 0) & (nb >= 3)
    junction = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    br = ((s > 0) & ~junction).astype(np.uint8)
    n, cc = cv2.connectedComponents(br, connectivity=8)
    out = []
    for k in range(1, n):
        ys, xs = np.nonzero(cc == k)
        if len(xs) < min_px:
            continue
        pts = set(zip(ys.tolist(), xs.tolist()))
        deg = {p: sum((p[0] + dy, p[1] + dx) in pts for dy, dx in _NB) for p in pts}
        while pts:
            ends = [p for p in pts if deg[p] <= 1]
            cur = min(ends) if ends else min(pts)
            path = [cur]
            pts.discard(cur)
            while True:
                nxt = next(((cur[0] + dy, cur[1] + dx) for dy, dx in _NB
                            if (cur[0] + dy, cur[1] + dx) in pts), None)
                if nxt is None:
                    break
                pts.discard(nxt)
                path.append(nxt)
                cur = nxt
            if len(path) >= min_px:
                out.append(np.array([(x, y) for y, x in path], np.float64))
    return out


def _rdp(pts: np.ndarray, eps: float) -> list:
    """Douglas–Peucker on an ordered path -> kept indices."""
    keep = {0, len(pts) - 1}
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        L = math.hypot(*d)
        seg = pts[i + 1:j] - a
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L if L > 1e-9 else np.hypot(*seg.T)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep.add(m)
            stack += [(i, m), (m, j)]
    return sorted(keep)


def _fit(pts: np.ndarray):
    """Least-squares line through pts; endpoints = projections of the first / last point."""
    mu = pts.mean(0)
    u = np.linalg.svd(pts - mu, full_matrices=False)[2][0] if len(pts) > 1 else np.array([1.0, 0.0])
    t0, t1 = (pts[0] - mu) @ u, (pts[-1] - mu) @ u
    return mu + t0 * u, mu + t1 * u


# --------------------------------------------------------------- segments

def _dir(s):
    d = s[1] - s[0]
    return d / max(np.linalg.norm(d), 1e-9)


def _length(s):
    return float(np.linalg.norm(s[1] - s[0]))


def _angle(a, b) -> float:
    """Unsigned angle between two segments' lines, degrees in [0, 90]."""
    c = abs(float(_dir(a) @ _dir(b)))
    return math.degrees(math.acos(min(1.0, c)))


def _axis_snap(s, max_deg):
    d = _dir(s)
    ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))       # 0 = horizontal
    s = s.copy()
    if ang <= max_deg:
        s[:, 1] = s[:, 1].mean()
    elif ang >= 90 - max_deg:
        s[:, 0] = s[:, 0].mean()
    return s


def _covered(labels_ok, a, b, max_gap_px) -> bool:
    """True when the straight path a -> b never leaves `labels_ok` for more than
    `max_gap_px` in a row. (A covered *fraction* is not enough: a doorway
    followed by a long wall would pass.)"""
    n = max(2, int(math.ceil(np.linalg.norm(b - a))) + 1)
    t = np.linspace(0, 1, n)
    p = np.round(a[None] + t[:, None] * (b - a)[None]).astype(int)
    S = labels_ok.shape[0]
    p = np.clip(p, 0, S - 1)
    ok = labels_ok[p[:, 1], p[:, 0]]
    run = best = 0
    for v in ok:
        run = 0 if v else run + 1
        best = max(best, run)
    return best <= max_gap_px


def _try_merge(a, b, perp_tol, gap_tol, bridge, short_px, bridge_gap=2.0) -> np.ndarray | None:
    """One segment for two collinear ones, or None.

    A piece shorter than `short_px` has an unreliable direction (an obstacle
    drawn over half a stroke bends its skeleton for a few pixels), so it only
    needs to lie within twice the tolerance of the longer one's line.
    """
    La, Lb = _length(a), _length(b)
    short = min(La, Lb) < short_px
    if _angle(a, b) > (30.0 if short else 12.0):
        return None
    if short:
        perp_tol = 2 * perp_tol
    base = a if La >= Lb else b
    u = _dir(base)
    n = np.array([-u[1], u[0]])
    o = base[0]
    if max(abs((p - o) @ n) for p in np.vstack([a, b])) > perp_tol:
        return None
    ta, tb = sorted((a - o) @ u), sorted((b - o) @ u)
    gap = max(ta[0], tb[0]) - min(ta[1], tb[1])
    if gap > gap_tol:
        ea = a[int(np.argmax((a - o) @ u))] if ta[1] < tb[0] else a[int(np.argmin((a - o) @ u))]
        eb = b[int(np.argmin((b - o) @ u))] if ta[1] < tb[0] else b[int(np.argmax((b - o) @ u))]
        if bridge is None or not _covered(bridge, ea, eb, bridge_gap):
            return None
    offset = (La * ((a[0] - o) @ n + (a[1] - o) @ n) + Lb * ((b[0] - o) @ n + (b[1] - o) @ n)) / (2 * (La + Lb))
    t = [(p - o) @ u for p in np.vstack([a, b])]
    return np.array([o + min(t) * u + offset * n, o + max(t) * u + offset * n])


def merge_collinear(segs, perp_tol, gap_tol, bridge=None, short_px=0.0, bridge_gap=2.0) -> list:
    segs = [s.copy() for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                m = _try_merge(segs[i], segs[j], perp_tol, gap_tol, bridge, short_px, bridge_gap)
                if m is not None:
                    segs[i] = m
                    del segs[j]
                    changed = True
                    break
            if changed:
                break
    return segs


def _intersect(a, b):
    d1, d2 = a[1] - a[0], b[1] - b[0]
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((b[0][0] - a[0][0]) * d2[1] - (b[0][1] - a[0][1]) * d2[0]) / den
    return a[0] + t * d1


def _param(s, p) -> float:
    """Position of p along s: 0 at s[0], 1 at s[1]."""
    d = s[1] - s[0]
    return float((p - s[0]) @ d / max(d @ d, 1e-9))


def snap_junctions(segs, join_px, overshoot_px) -> list:
    """Move every end that lies near another wall onto the lines' intersection:
    corners (both ends meet), T junctions (the stem meets the bar), and a
    corner drawn past its end (trimmed back by up to `overshoot_px`)."""
    segs = [s.copy() for s in segs]
    for i in range(len(segs)):
        for e in (0, 1):
            best, best_d = None, math.inf
            for j in range(len(segs)):
                if j == i or _angle(segs[i], segs[j]) < 30.0:
                    continue
                X = _intersect(segs[i], segs[j])
                if X is None:
                    continue
                d = float(np.linalg.norm(segs[i][e] - X))
                inside = 0.0 < _param(segs[i], X) < 1.0
                if d > join_px and not (inside and d <= overshoot_px):
                    continue                  # far ends are only trimmed back, never extended
                reach = join_px / max(_length(segs[j]), 1e-9)
                if not -reach <= _param(segs[j], X) <= 1 + reach:
                    continue                  # the other wall does not get there
                if d < best_d:
                    best, best_d = (j, X), d
            if best is not None:
                j, X = best
                segs[i][e] = X
                for f in (0, 1):              # the other wall's end at this corner meets it too
                    if np.linalg.norm(segs[j][f] - X) <= overshoot_px and \
                            (np.linalg.norm(segs[j][f] - X) <= join_px or 0.0 < _param(segs[j], X) < 1.0):
                        segs[j][f] = X
    return segs


def _free_end(segs, i, e, tol) -> bool:
    """End e of segment i touches no other segment (not a corner or junction)."""
    p = segs[i][e]
    for j, s in enumerate(segs):
        if j == i:
            continue
        d = s[1] - s[0]
        t = np.clip((p - s[0]) @ d / max(d @ d, 1e-9), 0, 1)
        if np.linalg.norm(p - (s[0] + t * d)) <= tol:
            return False
    return True


def extend_to_border(segs, S, reach_px, bridge=None, bridge_gap=2.0, tol=1.0) -> list:
    """Ends within reach of the world edge are extended onto it (generated walls
    that touch the boundary run exactly to 0 or to the world size); so are ends
    farther away when the whole way there is wall or obstacle label (a wall
    running under an obstacle to the boundary)."""
    out = []
    for i, s in enumerate(segs):
        s = s.copy()
        d = _dir(s)
        for e in (0, 1):
            if not _free_end(segs, i, e, tol):
                continue                      # a corner: it ends there on purpose
            p = s[e]
            other = s[1 - e]
            for axis, edge in ((0, 0.0), (0, float(S)), (1, 0.0), (1, float(S))):
                if abs(d[axis]) < 0.5 or abs(other[axis] - edge) < abs(p[axis] - edge):
                    continue                  # not heading there, or the other end is the one at this edge
                t = (edge - other[axis]) / (p[axis] - other[axis])
                q = other + t * (p - other)
                if abs(p[axis] - edge) <= reach_px or (bridge is not None and _covered(bridge, p, q, bridge_gap)):
                    s[e] = q
        out.append(s)
    return out


def _near_border(s, S, b) -> bool:
    """Both ends within b of the same image edge: a piece of the boundary band."""
    for axis in (0, 1):
        for edge in (0.0, float(S)):
            if abs(s[0][axis] - edge) <= b and abs(s[1][axis] - edge) <= b:
                return True
    return False


# ------------------------------------------------------------------- API

def wall_lines(wall_mask, obst_mask, world: dict, ecfg) -> list:
    """Straight wall centre lines [((x, y), (x, y))] in world units, boundary included."""
    S = wall_mask.shape[0]
    W = float(world["size"])
    scale = S / W
    border_px = max(2, int(math.ceil(ecfg.line_border * scale)))
    m = cv2.morphologyEx(wall_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    obst = obst_mask.astype(bool) if obst_mask is not None else np.zeros_like(m, bool)
    segs = []
    if m.any():
        # distance to *free* space: an obstacle drawn over a wall is neither, so the
        # wall/obstacle edge never becomes a wall line
        dt = cv2.distanceTransform(((m > 0) | obst).astype(np.uint8), cv2.DIST_L2, 5)
        w = stroke_width_px(np.where(m > 0, dt, 0).astype(np.float32), scale, border_px,
                            float(world["wall_thickness"]) * scale)
        rim = (m > 0) & (dt <= w)
        skel = thin(rim)
        b = border_px
        skel[:b], skel[-b:], skel[:, :b], skel[:, -b:] = 0, 0, 0, 0
        tol = max(1.0, ecfg.line_tol * scale)
        for path in branches(skel, max(3, int(ecfg.line_min_len * scale / 2))):
            idx = _rdp(path, tol)
            for i, j in zip(idx[:-1], idx[1:]):
                if j - i >= 2:
                    segs.append(np.array(_fit(path[i:j + 1])))

        # gaps may be bridged over pen strokes and obstacles, never through the
        # inside of a filled dead-space block (that would run walls to the border)
        # widened by the tolerance: a straightened line drifts off a wobbly stroke
        r = max(1, int(round(tol)))
        bridge = cv2.dilate((rim | obst).astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        join, over = ecfg.line_join * scale, ecfg.line_overshoot * scale
        segs = [_axis_snap(s, ecfg.line_axis_deg) for s in segs]
        short, bgap = 3 * join, max(2.0, 0.5 * scale)
        segs = merge_collinear(segs, tol, join, bridge, short, bgap)
        segs = [_axis_snap(s, ecfg.line_axis_deg) for s in segs]
        segs = merge_collinear(segs, tol, join, bridge, short, bgap)
        segs = snap_junctions(segs, join, over)
        segs = extend_to_border(segs, S, ecfg.line_border_reach * scale, bridge, bgap, tol)
        segs = [_axis_snap(s, ecfg.line_axis_deg) for s in segs]      # exact again after snapping
        segs = [s for s in segs if _length(s) >= ecfg.line_min_len * scale
                and not _near_border(s, S, border_px)]

    out = []
    for s in segs:
        a = px_to_world(W, S, float(s[0][0]), float(s[0][1]))
        c = px_to_world(W, S, float(s[1][0]), float(s[1][1]))
        out.append(((float(a[0]), float(a[1])), (float(c[0]), float(c[1]))))
    return boundary_walls(W) + out
