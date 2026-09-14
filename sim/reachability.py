"""Configuration-space reachability, clearance and bottleneck width.

The scene is rasterised once and a Euclidean distance transform gives
`clearance[i, j]`, the distance from cell (i, j) to the nearest solid cell.
A disk of radius R fits at (i, j) iff clearance[i, j] >= R, so every
reachability question is a threshold on one precomputed array.

Grid convention: grid[iy, ix], world x = (ix + 0.5) * cell, y = (iy + 0.5) * cell.
Row 0 is the bottom of the world; render.py flips for image space.
"""
from __future__ import annotations

import cv2
import numpy as np

from geometry import point_seg_dist

_GRID_CACHE = {}


def grid_shape(cfg):
    return int(round(cfg.world / cfg.cell))


def grid_coords(cfg):
    """Cached (px, py) world-coordinate arrays for every cell centre."""
    key = (cfg.world, cfg.cell)
    if key not in _GRID_CACHE:
        n = grid_shape(cfg)
        idx = (np.arange(n) + 0.5) * cfg.cell
        px, py = np.meshgrid(idx, idx)          # px[iy, ix], py[iy, ix]
        _GRID_CACHE[key] = (px, py)
    return _GRID_CACHE[key]


def circle_mask(cfg, c, r) -> np.ndarray:
    px, py = grid_coords(cfg)
    return (px - c[0]) ** 2 + (py - c[1]) ** 2 < r * r


def wall_mask(cfg, walls) -> np.ndarray:
    px, py = grid_coords(cfg)
    m = np.zeros(px.shape, bool)
    for seg in walls:
        m |= point_seg_dist(px, py, seg) < cfg.wall_half
    return m


def occupancy(cfg, scene) -> np.ndarray:
    """True where the world is solid (no inflation)."""
    m = wall_mask(cfg, scene.walls)
    for o in scene.obstacles:
        m |= circle_mask(cfg, o.c, o.r)
    return m


def clearance(cfg, occ: np.ndarray) -> np.ndarray:
    """Distance (world units) from each free cell to the nearest solid cell."""
    free = (~occ).astype(np.uint8)
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    return dt * cfg.cell


def free_mask(clear: np.ndarray, radius: float) -> np.ndarray:
    return clear >= radius


def _labels(mask: np.ndarray):
    """4-connected components (8-connectivity would pass diagonal pinholes)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    return lab


def cell_of(cfg, p):
    n = grid_shape(cfg)
    ix = int(p[0] / cfg.cell)
    iy = int(p[1] / cfg.cell)
    return (min(max(iy, 0), n - 1), min(max(ix, 0), n - 1))


def region_slice(cfg, region):
    n = grid_shape(cfg)
    x0 = min(max(int(region.x0 / cfg.cell), 0), n - 1)
    x1 = min(max(int(np.ceil(region.x1 / cfg.cell)), 1), n)
    y0 = min(max(int(region.y0 / cfg.cell), 0), n - 1)
    y1 = min(max(int(np.ceil(region.y1 / cfg.cell)), 1), n)
    return (slice(y0, y1), slice(x0, x1))


def _labels_in_region(lab, cfg, region):
    sub = lab[region_slice(cfg, region)]
    vals = np.unique(sub)
    return set(int(v) for v in vals if v > 0)


def reachable_labels(cfg, scene, radius, clear=None, target=None):
    """Component labels of the free space touching the target region."""
    if clear is None:
        clear = clearance(cfg, occupancy(cfg, scene))
    lab = _labels(free_mask(clear, radius))
    target = target or scene.goal_region
    return lab, _labels_in_region(lab, cfg, target)


def is_solvable(cfg, scene, radius=None, clear=None) -> bool:
    """Can every agent reach its own goal region as a disk of `radius`?"""
    radius = cfg.agent_radius if radius is None else radius
    if clear is None:
        clear = clearance(cfg, occupancy(cfg, scene))
    lab = _labels(free_mask(clear, radius))

    cache = {}
    for a in scene.agents:
        goal = scene.goal_for(a)
        if goal is None:
            return False
        key = tuple(goal.as_list())
        if key not in cache:
            cache[key] = _labels_in_region(lab, cfg, goal)
        iy, ix = cell_of(cfg, a.pos)
        if int(lab[iy, ix]) not in cache[key]:
            return False
    return True


def regions_connected(cfg, scene, radius, clear=None) -> bool:
    """Whether start_region and goal_region are linked for a disk of `radius`.

    Used for bottleneck search, where per-agent placement is irrelevant.
    """
    if scene.start_region is None or scene.goal_region is None:
        return is_solvable(cfg, scene, radius, clear)
    if clear is None:
        clear = clearance(cfg, occupancy(cfg, scene))
    lab = _labels(free_mask(clear, radius))
    start = _labels_in_region(lab, cfg, scene.start_region)
    goal = _labels_in_region(lab, cfg, scene.goal_region)
    return bool(start & goal)


def bottleneck_width(cfg, scene, clear=None, hi=15.0, iters=8) -> float:
    """Diameter of the largest disk that can travel start -> goal (binary search)."""
    if clear is None:
        clear = clearance(cfg, occupancy(cfg, scene))
    if not regions_connected(cfg, scene, 0.0, clear):
        return 0.0
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if regions_connected(cfg, scene, mid, clear):
            lo = mid
        else:
            hi = mid
    return 2.0 * lo


class OccupancyBuilder:
    """Incremental occupancy: the wall mask is rasterised once, obstacles are added on top."""

    def __init__(self, cfg, walls):
        self.cfg = cfg
        self.base = wall_mask(cfg, walls)
        self.occ = self.base.copy()

    def trial(self, c, r) -> np.ndarray:
        """Occupancy as it would be with one more obstacle -- nothing committed."""
        return self.occ | circle_mask(self.cfg, c, r)

    def commit(self, c, r):
        self.occ |= circle_mask(self.cfg, c, r)

    def clearance(self) -> np.ndarray:
        return clearance(self.cfg, self.occ)


def unreachable_free(cfg, scene, clear=None, anchor=None) -> np.ndarray:
    """Free cells not connected to the start/goal regions or any agent (dead space)."""
    if clear is None:
        clear = clearance(cfg, occupancy(cfg, scene))
    free = clear > 0.0
    lab = _labels(free)
    anchor = anchor or scene.start_region or scene.goal_region
    keep = set()
    for reg in filter(None, [anchor, scene.goal_region, scene.start_region]):
        keep |= _labels_in_region(lab, cfg, reg)
    for a in scene.agents:
        iy, ix = cell_of(cfg, a.pos)
        if lab[iy, ix] > 0:
            keep.add(int(lab[iy, ix]))
    if not keep:
        return np.zeros_like(free)
    return free & ~np.isin(lab, list(keep))
