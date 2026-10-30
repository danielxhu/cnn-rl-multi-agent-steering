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
from scene import members

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


def _labels_in_goal(lab, cfg, goal):
    """Labels touching a Region, or any member of a RegionSet (an exit set)."""
    out = set()
    for r in members(goal):
        out |= _labels_in_region(lab, cfg, r)
    return out


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
        key = tuple(tuple(r.as_list()) for r in members(goal))
        if key not in cache:
            cache[key] = _labels_in_goal(lab, cfg, goal)
        iy, ix = cell_of(cfg, a.pos)
        if int(lab[iy, ix]) not in cache[key]:
            return False
    return True


def regions_connected(cfg, scene, radius, clear=None) -> bool:
    """Whether start_region and goal_region are linked for a disk of `radius`.

    Used for bottleneck search, where per-agent placement is irrelevant. A scene
    with several boxes (`starts` / `exits`) is linked when every start box, and
    every agent outside the start boxes, reaches at least one exit.
    """
    if scene.starts or scene.exits:
        if clear is None:
            clear = clearance(cfg, occupancy(cfg, scene))
        lab = _labels(free_mask(clear, radius))
        exits = set()
        for g in scene.exit_boxes():
            exits |= _labels_in_region(lab, cfg, g)
        sources = [_labels_in_region(lab, cfg, s) for s in scene.start_boxes()]
        # agents drawn outside every start box (sketch v7) are sources of their own
        boxes = scene.start_boxes()
        sources += [{int(lab[cell_of(cfg, a.pos)])} for a in scene.agents
                    if not any(b.contains(a.pos) for b in boxes)]
        return bool(sources) and all(src & exits for src in sources)
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
    boxes = list(scene.starts or []) + list(scene.exits or [])     # several boxes: every one
    for reg in filter(None, [anchor, scene.goal_region, scene.start_region] + boxes):
        keep |= _labels_in_region(lab, cfg, reg)
    for a in scene.agents:
        iy, ix = cell_of(cfg, a.pos)
        if lab[iy, ix] > 0:
            keep.add(int(lab[iy, ix]))
    if not keep:
        return np.zeros_like(free)
    return free & ~np.isin(lab, list(keep))


# --------------------------------------------------------------- geodesic
#
# Path length from every coarse cell to a goal, for the "geodesic" progress
# reward (policy/DESIGN_E3.md §1.2). An 8-neighbour wavefront overestimates the
# straight-line distance by up to 8 %, so moves reach out to (3, 2): the 32
# directions are at most 18.4 deg apart, which bounds the error by 1.3 %. A long
# move is legal only when every cell its segment crosses is free, so it can
# never jump a wall corner.
#
# Relaxation runs along *lines*: for each of the 16 line directions the grid is
# partitioned into chains p, p + v, p + 2v, ... (index arrays, cached per grid
# size), and one segmented running minimum per orientation carries the front
# along every free stretch of every chain at once. A sweep over all directions
# therefore crosses any straight free stretch; sweeps repeat until nothing
# improves, typically a handful. The fixed point is that of plain 32-neighbour
# relaxation (checked in sim/tests).

def _geo_moves(reach=3):
    """[(dy, dx, length / cell, crossed cells)] for the primitive offsets up to ``reach``."""
    moves = []
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if (dy, dx) == (0, 0) or np.gcd(abs(dy), abs(dx)) != 1:
                continue
            ts = np.linspace(0.0, 1.0, 8 * reach + 1)[1:-1]
            # both roundings of the segment, so a move and its reverse cross the same cells
            crossed = {(int(np.floor(t * dy + 0.5)), int(np.floor(t * dx + 0.5))) for t in ts}
            crossed |= {(dy - int(np.floor((1 - t) * dy + 0.5)), dx - int(np.floor((1 - t) * dx + 0.5)))
                        for t in ts}
            crossed -= {(0, 0), (dy, dx)}
            moves.append((dy, dx, float(np.hypot(dy, dx)), sorted(crossed)))
    return moves


_GEO_MOVES = _geo_moves()
_GEO_PAD = 3
_GEO_BIG = 1e6                  # separates runs in the segmented running minimum
_CHAINS = {}


def _chains(m, dy, dx):
    """(flat index (S, L), valid (S, L)) of the chains of direction (dy, dx) on an m x m grid."""
    key = (m, dy, dx)
    if key not in _CHAINS:
        iy, ix = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        py, px = iy - dy, ix - dx                         # predecessor outside -> chain start
        start = (py < 0) | (py >= m) | (px < 0) | (px >= m)
        sy, sx = iy[start], ix[start]
        k = np.arange(m)
        cy = sy[:, None] + dy * k[None, :]
        cx = sx[:, None] + dx * k[None, :]
        valid = (cy >= 0) & (cy < m) & (cx >= 0) & (cx < m)
        L = int(valid.sum(1).max())
        cy, cx, valid = cy[:, :L], cx[:, :L], valid[:, :L]
        idx = np.where(valid, cy * m + cx, 0)
        _CHAINS[key] = (idx, valid)
    return _CHAINS[key]


def _coarse_free(cfg, scene, radius, cell, clear=None):
    """Configuration-space free mask at ``cell``: a coarse cell is free if any fine cell is."""
    if clear is None:
        clear = clearance(cfg, occupancy(cfg, scene))
    fine = free_mask(clear, radius)
    f = int(round(cell / cfg.cell))
    m = fine.shape[0] // f
    return fine[:m * f, :m * f].reshape(m, f, m, f).any(axis=(1, 3))


def _scan(V, ramp, big_seg):
    """Path lengths along chains: segmented running minimum of ``V - ramp``.

    ``big_seg`` = ``_GEO_BIG`` x run id (grows by one at every break), which
    keeps earlier runs from winning; where a run has no finite prefix the
    result is >= _GEO_BIG / 2 and the caller ignores it.
    """
    return np.minimum.accumulate(V - ramp - big_seg, axis=1) + big_seg + ramp


def geodesic_field(cfg, scene, goal, radius=None, cell=1.0, clear=None) -> np.ndarray:
    """(m, m) float32: path length (world units) from each cell centre to `goal`
    (a Region or RegionSet) for a disk of `radius` (default cfg.agent_radius);
    inf where unreachable. Grid convention as the module's (row 0 = bottom).

    ``clear``: a precomputed ``clearance`` (several goals of one scene share it).
    """
    radius = cfg.agent_radius if radius is None else radius
    free = _coarse_free(cfg, scene, radius, cell, clear)
    m = free.shape[0]
    idx1 = (np.arange(m) + 0.5) * cell
    px, py = np.meshgrid(idx1, idx1)

    # sources: free cells within one cell of the goal, at their exact distance
    # (0 inside); seeding only the cells inside would add up to a cell of error
    dist = np.full((m, m), np.inf)
    for r in members(goal):
        ddx = np.maximum(np.maximum(r.x0 - px, px - r.x1), 0.0)
        ddy = np.maximum(np.maximum(r.y0 - py, py - r.y1), 0.0)
        dist = np.minimum(dist, np.hypot(ddx, ddy))
    D = np.where(free & (dist <= cell), dist, np.inf).ravel()

    # legal moves: ok[(dy, dx)][p] = the move from p - (dy, dx) into p is allowed
    P = _GEO_PAD
    fp = np.zeros((m + 2 * P, m + 2 * P), bool)
    fp[P:P + m, P:P + m] = free
    ok = {}
    for dy, dx, c, crossed in _GEO_MOVES:
        o = free & fp[P - dy:P - dy + m, P - dx:P - dx + m]
        for cy, cx in crossed:
            o &= fp[P - cy:P - cy + m, P - cx:P - cx + m]
        ok[(dy, dx)] = o.ravel()

    # one entry per line direction (dy, dx) with its reverse (-dy, -dx); invalid
    # chain positions index an extra slot of D that is always inf
    n = m * m
    lines = []
    for dy, dx, c, _ in _GEO_MOVES:
        if (dy, dx) < (-dy, -dx):
            continue
        idx, valid = _chains(m, dy, dx)
        gidx = np.where(valid, idx, n)
        fwd = ~(ok[(dy, dx)][idx] & valid)                 # break before position k
        bwd = ~(ok[(-dy, -dx)][idx] & valid)[:, ::-1]      # reversed chain: move k + 1 -> k
        big_f = _GEO_BIG * np.cumsum(fwd, axis=1)
        big_b = _GEO_BIG * np.cumsum(bwd, axis=1)
        ramp = np.arange(idx.shape[1]) * (c * cell)
        lines.append((gidx, big_f, big_b, ramp))

    De = np.append(D, np.inf)
    cap = _GEO_BIG / 4
    while True:
        before = De.copy()
        for gidx, big_f, big_b, ramp in lines:
            V = De[gidx]
            Rf = _scan(V, ramp, big_f)
            Rb = _scan(V[:, ::-1], ramp, big_b)[:, ::-1]
            best = np.minimum(np.minimum(Rf, Rb), V)
            De[gidx] = np.where(best < cap, best, V)
            De[n] = np.inf
        if not (De < before - 1e-6).any():
            return De[:n].reshape(m, m).astype(np.float32)


def geodesic_distance(cfg, field, p, cell=1.0, goal=None) -> float:
    """Bilinear lookup of `field` at world point p.

    Only finite corners count (weights re-normalised). With none finite (the
    point is pressed against an obstacle, inside the pooled boundary): the
    smallest finite value of the 5 x 5 neighbourhood plus the straight-line
    distance to that cell's centre; failing that, the straight-line distance
    to `goal` (inf when no goal is given).
    """
    m = field.shape[0]
    u, v = p[0] / cell - 0.5, p[1] / cell - 0.5
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - x0, v - y0
    num = den = 0.0
    for iy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
        for ix, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
            if 0 <= iy < m and 0 <= ix < m:
                val = field[iy, ix]
                w = wy * wx
                if np.isfinite(val) and w > 0.0:
                    num += w * float(val)
                    den += w
    if den > 1e-9:
        return num / den
    cy, cx = int(round(v)), int(round(u))
    best = np.inf
    for iy in range(max(cy - 2, 0), min(cy + 3, m)):
        for ix in range(max(cx - 2, 0), min(cx + 3, m)):
            val = field[iy, ix]
            if np.isfinite(val):
                d = float(val) + float(np.hypot(p[0] - (ix + 0.5) * cell, p[1] - (iy + 0.5) * cell))
                best = min(best, d)
    if np.isfinite(best):
        return best
    return goal.distance_to(tuple(p)) if goal is not None else np.inf
