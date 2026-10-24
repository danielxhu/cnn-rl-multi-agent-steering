"""Layout families and the scene sampler.

    open          free space, scattered obstacles
    corridor      two parallel walls
    room          four walls around the playable area
    doorway       one blocking wall with a single gap
    two_doorway   two gaps of different width
    cul_de_sac    two gaps, one opening into a sealed pocket
    crossing      two perpendicular corridors, two groups of agents

Free-form families, not in LAYOUTS (so ``--layout mixed`` and every dataset made
with it are unchanged); ``MIXES["mixed2"]`` adds them for robustness to layouts
nobody planned. Start and goal boxes go anywhere, in any orientation:

    walls         0-6 random walls, mostly axis-parallel, some diagonal,
                  some attached to the boundary
    rooms         one or two walls with doors splitting the world into rooms

Multi-box free-form families (DESIGN_SKETCH §9.1), not in LAYOUTS or mixed2;
``MIXES["mixed3"]`` adds them. Each has k in {1, 2, 3} start boxes and m in
{1, 2, 3} exit boxes (k = m = 1 in a third of scenes); an agent succeeds by
reaching any exit (``Scene.exits``):

    boxrooms      1-4 small rooms of walls (sides 10-30, often against the
                  frame, one or two doors each); boxes inside some rooms and
                  outside others, and in more than half the scenes a room
                  that holds no box (a decoy that looks like a box)
    walls_multi   ``walls`` with several boxes
    rooms_multi   ``rooms`` with several boxes

A scene with one start and one exit comes back in the single-box form
(``groups``, no ``starts`` / ``exits``). ``close=True`` (sketch v5) packs the
agents of a start box at the hard spacing 2R + hard_gap.

Hard cases (sketch v6, DESIGN_SKETCH §10.2) are ``LayoutOptions``; the
defaults change nothing (no extra random draws), and ``MIX_OPTIONS`` turns them
on for ``--layout mixed4`` (the ``mixed3`` families):

    box_short_min       short side of free-form boxes U(box_short_min, 16)
    box_near_wall_prob  per box: moved to U(0.3, 2.0) from a wall's / the frame's surface
    corner_gap_prob     per room corner (``room``, ``boxrooms``): the walls stop
                        U(0.5, 3) short of each other, or one runs U(0.5, 4) past
    slit_prob           per scene: two parallel walls U(3, 8) apart, U(8, 25) long,
                        alone or alongside an existing wall

Sketch v7 (DESIGN_SKETCH §11.2) adds, again off by default, for ``--layout mixed5``:

    agent_mixed_prob,   per scene: some agents in start boxes and the others
    agent_free_prob     anywhere in free space ("mixed"), or no start box at all
                        ("free"); else every agent in a start box ("boxes")
    slant_prob          per free-form wall: any angle (default: 0.25, as before)
    room_tilt_prob      per room (``room``, ``boxrooms``, ``rooms*``): one or two
                        walls tilted by U(5°, 25°), the room still closed
    touch_prob          per scene: one obstacle touching a box from outside or a wall
    long_line_prob      per scene: a free-standing wall U(25, 70) long parallel
                        to a box's long side at U(4, 20)

Every scene is validated before it is returned: start must reach goal for a
disk of the agent's radius, and the tightest passage must exceed min_bottleneck.
With several boxes every agent must reach at least one exit, every exit must
be reachable from an agent, and the bottleneck is the widest disk with which
every start box reaches an exit.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, replace

import numpy as np

from config import Config
from reachability import (
    OccupancyBuilder,
    _labels,
    _labels_in_region,
    bottleneck_width,
    cell_of,
    clearance,
    free_mask,
    is_solvable,
    regions_connected,
)
from scene import Agent, Obstacle, Region, Scene, boundary_walls

LAYOUTS = ["open", "corridor", "room", "doorway", "two_doorway", "cul_de_sac", "crossing"]
FREE_LAYOUTS = ["walls", "rooms"]
MULTI_LAYOUTS = ["boxrooms", "walls_multi", "rooms_multi"]       # several start / exit boxes
# layout cycles generate.py samples from; "mixed2" is ~40 % free-form; "mixed3"
# (sketch v5) adds a third of rooms-next-to-boxes and multi-box scenes
MIXES = {"mixed": LAYOUTS,
         "mixed2": LAYOUTS + ["walls", "walls", "walls", "rooms", "rooms"]}
MIXES["mixed3"] = MIXES["mixed2"] + ["boxrooms", "boxrooms", "walls_multi", "walls_multi",
                                     "rooms_multi", "rooms_multi"]
# sketch v6: the mixed3 families with the hard cases of MIX_OPTIONS["mixed4"]
MIXES["mixed4"] = list(MIXES["mixed3"])
# sketch v7: the same families with MIX_OPTIONS["mixed5"], and its three probes
for _name in ("mixed5", "mixed5_free", "mixed5_tilt", "mixed5_lines"):
    MIXES[_name] = list(MIXES["mixed4"])


@dataclass(frozen=True)
class LayoutOptions:
    """Hard cases (DESIGN_SKETCH §10.2). The defaults reproduce every older layout
    exactly: no option draws a random number unless it is switched on."""

    box_short_min: float = 10.0       # short side of free-form boxes, U(this, 16)
    box_near_wall_prob: float = 0.0   # per box: a gap of U(0.3, 2.0) to a wall surface
    corner_gap_prob: float = 0.0      # per room corner: left open or crossed
    slit_prob: float = 0.0            # per scene: two parallel walls 3-8 apart
    # sketch v7 (DESIGN_SKETCH §11.2)
    agent_mixed_prob: float = 0.0     # per scene: some agents in boxes, the rest anywhere
    agent_free_prob: float = 0.0      # per scene: no start box, every agent anywhere
    slant_prob: float = 0.0           # per free-form wall: any angle (0: 0.25, the old rate)
    room_tilt_prob: float = 0.0       # per room: one or two walls tilted 5-25°
    touch_prob: float = 0.0           # per scene: an obstacle touching a box or a wall
    long_line_prob: float = 0.0       # per scene: a long wall parallel to a box's long side


DEFAULT_OPTIONS = LayoutOptions()
MIX_OPTIONS = {"mixed4": LayoutOptions(box_short_min=6.0, box_near_wall_prob=0.3,
                                       corner_gap_prob=0.5, slit_prob=0.3)}
MIX_OPTIONS["mixed5"] = replace(MIX_OPTIONS["mixed4"], agent_mixed_prob=0.2, agent_free_prob=0.2,
                                slant_prob=0.4, room_tilt_prob=0.3, touch_prob=0.3, long_line_prob=0.3)
MIX_OPTIONS["mixed5_free"] = replace(MIX_OPTIONS["mixed5"], agent_mixed_prob=0.5, agent_free_prob=0.5)
MIX_OPTIONS["mixed5_tilt"] = replace(MIX_OPTIONS["mixed5"], slant_prob=1.0, room_tilt_prob=1.0)   # with --style sketch7tilt
MIX_OPTIONS["mixed5_lines"] = replace(MIX_OPTIONS["mixed5"], long_line_prob=1.0)


@dataclass
class LayoutSpec:
    walls: list
    groups: list          # [(start_region, goal_region), ...]
    place_zone: Region    # where obstacle centres may be sampled
    name: str
    starts: list | None = None     # several boxes (MULTI_LAYOUTS); then groups is unused
    exits: list | None = None
    rooms: list | None = None      # boxrooms: each room's outline (for tests)
    hard: dict | None = None       # hard cases applied (counts; DESIGN_SKETCH §10.2)


def rect_outline(x0, y0, x1, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def _margin(cfg):
    """Smallest distance from a wall centre line at which an agent fits."""
    return cfg.wall_half + cfg.agent_radius + 1.0


# ------------------------------------------------------------------ families

def _open(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    return LayoutSpec(
        walls=boundary_walls(W),
        groups=[(Region(m, m, m + 14, W - m), Region(W - m - 14, m, W - m, W - m))],
        place_zone=Region(28, m, W - 28, W - m),
        name="open",
    )


def _corridor(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    h = rng.uniform(30.0, 46.0)
    y_lo = (W - h) / 2 + rng.uniform(-8.0, 8.0)
    y_hi = y_lo + h
    walls = boundary_walls(W) + [((0.0, y_lo), (W, y_lo)), ((0.0, y_hi), (W, y_hi))]
    inner = (y_lo + m, y_hi - m)
    return LayoutSpec(
        walls=walls,
        groups=[(Region(m, inner[0], m + 14, inner[1]),
                 Region(W - m - 14, inner[0], W - m, inner[1]))],
        place_zone=Region(28, inner[0], W - 28, inner[1]),
        name="corridor",
    )


def _room(rng, cfg, opts=DEFAULT_OPTIONS):
    W, m = cfg.world, _margin(cfg)
    x0, y0 = rng.uniform(6.0, 16.0), rng.uniform(6.0, 16.0)
    x1, y1 = W - rng.uniform(6.0, 16.0), W - rng.uniform(6.0, 16.0)
    inner = rect_outline(x0, y0, x1, y1)
    hard = {}
    ix0, iy0, ix1, iy1 = x0 + m, y0 + m, x1 - m, y1 - m
    groups = [(Region(ix0, iy0, ix0 + 13, iy1), Region(ix1 - 13, iy0, ix1, iy1))]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if opts.room_tilt_prob > 0:                        # v7: tilt walls, keeping off the boxes
        def ok(c):
            return all(cfg.wall_half <= v <= W - cfg.wall_half for q in c for v in q) and all(
                _seg_region_dist(sd, b) >= cfg.wall_half + 1.0 for sd in _quad_sides(c) for b in groups[0])
        corners = _tilt_corners(rng, corners, [False] * 4, opts, ok, hard, outward_only=False)
        inner = _quad_sides(corners)
    if opts.corner_gap_prob > 0:
        inner = _open_corners(rng, cfg, inner, corners, opts, hard)
    walls = boundary_walls(W) + inner
    return LayoutSpec(
        walls=walls,
        groups=groups,
        place_zone=Region(ix0 + 20, iy0, ix1 - 20, iy1),
        name="room", hard=hard or None,
    )


def _gap_wall(x, gaps, world):
    """Vertical wall at x, broken by `gaps` = [(centre, width), ...]."""
    cuts = sorted((c - w / 2, c + w / 2) for c, w in gaps)
    segs, y = [], 0.0
    for lo, hi in cuts:
        if lo > y:
            segs.append(((x, y), (x, lo)))
        y = max(y, hi)
    if y < world:
        segs.append(((x, y), (x, world)))
    return segs


def _doorway(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    xd = rng.uniform(42.0, 58.0)
    w = rng.uniform(10.0, 18.0)
    yd = rng.uniform(24.0, W - 24.0)
    walls = boundary_walls(W) + _gap_wall(xd, [(yd, w)], W)
    return LayoutSpec(
        walls=walls,
        groups=[(Region(m, m, m + 13, W - m), Region(W - m - 13, m, W - m, W - m))],
        place_zone=Region(20, m, W - 20, W - m),
        name="doorway",
    )


def _two_doorway(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    xd = rng.uniform(42.0, 58.0)
    y1 = rng.uniform(18.0, 40.0)
    y2 = rng.uniform(60.0, W - 18.0)
    w1 = rng.uniform(10.0, 13.0)
    w2 = rng.uniform(16.0, 22.0)          # deliberately unequal: is the wider one preferred?
    if rng.random() < 0.5:
        w1, w2 = w2, w1
    walls = boundary_walls(W) + _gap_wall(xd, [(y1, w1), (y2, w2)], W)
    return LayoutSpec(
        walls=walls,
        groups=[(Region(m, m, m + 13, W - m), Region(W - m - 13, m, W - m, W - m))],
        place_zone=Region(20, m, W - 20, W - m),
        name="two_doorway",
    )


def _cul_de_sac(rng, cfg):
    """Two gaps in one wall; behind one of them, a sealed pocket."""
    W, m = cfg.world, _margin(cfg)
    xd = rng.uniform(38.0, 52.0)
    y_real = rng.uniform(18.0, 40.0)
    y_fake = rng.uniform(60.0, W - 18.0)
    if rng.random() < 0.5:
        y_real, y_fake = y_fake, y_real
    w_real = rng.uniform(12.0, 18.0)
    w_fake = rng.uniform(12.0, 18.0)
    depth = rng.uniform(12.0, 22.0)

    walls = boundary_walls(W) + _gap_wall(xd, [(y_real, w_real), (y_fake, w_fake)], W)
    lo, hi = y_fake - w_fake / 2, y_fake + w_fake / 2
    walls += [                              # the pocket: floor, back, ceiling
        ((xd, lo), (xd + depth, lo)),
        ((xd + depth, lo), (xd + depth, hi)),
        ((xd + depth, hi), (xd, hi)),
    ]
    return LayoutSpec(
        walls=walls,
        groups=[(Region(m, m, m + 13, W - m), Region(W - m - 13, m, W - m, W - m))],
        place_zone=Region(18, m, xd - 8, W - m),
        name="cul_de_sac",
    )


def _crossing(rng, cfg):
    """A plus-shaped intersection with two groups of agents crossing."""
    W, m = cfg.world, _margin(cfg)
    wx = rng.uniform(24.0, 34.0)          # vertical corridor width
    wy = rng.uniform(24.0, 34.0)          # horizontal corridor height
    xc = W / 2 + rng.uniform(-6.0, 6.0)
    yc = W / 2 + rng.uniform(-6.0, 6.0)
    L, Rr = xc - wx / 2, xc + wx / 2
    B, T = yc - wy / 2, yc + wy / 2

    walls = boundary_walls(W) + [
        ((0.0, B), (L, B)), ((L, B), (L, 0.0)),          # bottom-left corner
        ((W, B), (Rr, B)), ((Rr, B), (Rr, 0.0)),         # bottom-right
        ((0.0, T), (L, T)), ((L, T), (L, W)),            # top-left
        ((W, T), (Rr, T)), ((Rr, T), (Rr, W)),           # top-right
    ]
    horiz = (B + m, T - m)
    vert = (L + m, Rr - m)
    return LayoutSpec(
        walls=walls,
        groups=[
            (Region(m, horiz[0], m + 12, horiz[1]),          # left  -> right
             Region(W - m - 12, horiz[0], W - m, horiz[1])),
            (Region(vert[0], m, vert[1], m + 12),            # bottom -> top
             Region(vert[0], W - m - 12, vert[1], W - m)),
        ],
        place_zone=Region(L + m, B + m, Rr - m, T - m),
        name="crossing",
    )


# ------------------------------------------------------- free-form families

def _box_dims(rng, short_min=10.0):
    """A start / goal box: one side short_min-16, the other 12-40, either way round."""
    a, b = rng.uniform(short_min, 16.0), rng.uniform(12.0, 40.0)
    return (a, b) if rng.random() < 0.5 else (b, a)


def _free_pair(rng, cfg, zones=None, opts=DEFAULT_OPTIONS):
    """(start, goal) anywhere in the world, apart; `zones` restricts each to a Region."""
    W, m = cfg.world, _margin(cfg)
    lo = opts.box_short_min
    for _ in range(200):
        regs = []
        for k in range(2):
            z = zones[k] if zones else Region(m, m, W - m, W - m)
            w, h = _box_dims(rng, lo)
            w, h = min(w, z.width), min(h, z.height)
            if w < lo or h < lo:
                break
            x0, y0 = rng.uniform(z.x0, z.x1 - w), rng.uniform(z.y0, z.y1 - h)
            regs.append(Region(x0, y0, x0 + w, y0 + h))
        if len(regs) < 2:
            continue
        s, g = regs
        if math.dist(s.center, g.center) < 0.35 * W:
            continue
        if not (s.x1 + 8 < g.x0 or g.x1 + 8 < s.x0 or s.y1 + 8 < g.y0 or g.y1 + 8 < s.y0):
            continue                                   # overlapping or touching
        return s, g
    return None


def _seg_region_dist(seg, reg, step=0.5) -> float:
    (ax, ay), (bx, by) = seg
    n = max(2, int(math.hypot(bx - ax, by - ay) / step))
    return min(reg.distance_to((ax + (bx - ax) * t, ay + (by - ay) * t)) for t in np.linspace(0, 1, n))


def _seg_seg_dist(a, b, step=0.5) -> float:
    (ax, ay), (bx, by) = a
    n = max(2, int(math.hypot(bx - ax, by - ay) / step))
    (cx, cy), (dx, dy) = b
    best = math.inf
    for t in np.linspace(0, 1, n):
        px, py = ax + (bx - ax) * t, ay + (by - ay) * t
        ex, ey = dx - cx, dy - cy
        u = max(0.0, min(1.0, ((px - cx) * ex + (py - cy) * ey) / max(ex * ex + ey * ey, 1e-12)))
        best = min(best, math.hypot(px - cx - u * ex, py - cy - u * ey))
    return best


def _walls(rng, cfg, opts=DEFAULT_OPTIONS):
    W = cfg.world
    pair = _free_pair(rng, cfg, opts=opts)
    if pair is None:
        return _open(rng, cfg)
    s, g = pair
    walls = _random_walls(rng, cfg, [s, g], opts.slant_prob or 0.25)
    m = _margin(cfg)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(s, g)],
                      place_zone=Region(m, m, W - m, W - m), name="walls")


def _random_walls(rng, cfg, boxes, slant_prob=0.25):
    """0-6 random walls kept off every box (the wall sampler of ``walls``); each at
    any angle with p = slant_prob, else axis-parallel."""
    W = cfg.world
    keep_off = cfg.wall_half + cfg.region_margin
    walls, want = [], int(rng.integers(0, 7))
    for _ in range(80):
        if len(walls) >= want:
            break
        L = rng.uniform(12.0, 55.0)
        diag = rng.random() < slant_prob
        if rng.random() < 0.35:                        # attached to the boundary, pointing inward
            edge = int(rng.integers(4))
            t = rng.uniform(10.0, W - 10.0)
            a = [(t, 0.0), (W, t), (t, W), (0.0, t)][edge]
            inward = [math.pi / 2, math.pi, -math.pi / 2, 0.0][edge]
            ang = inward + (rng.uniform(-0.6, 0.6) if diag else 0.0)
        else:
            a = (rng.uniform(8.0, W - 8.0), rng.uniform(8.0, W - 8.0))
            ang = rng.uniform(0, math.pi) if diag else rng.choice([0.0, math.pi / 2])
        b = (a[0] + L * math.cos(ang), a[1] + L * math.sin(ang))
        b = (min(max(b[0], 0.0), W), min(max(b[1], 0.0), W))
        seg = (a, b)
        if math.dist(a, b) < 8.0:
            continue
        if min(_seg_region_dist(seg, r) for r in boxes) < keep_off:
            continue
        # no near-miss slivers: another wall is either crossed or well apart
        if any(2.0 < _seg_seg_dist(seg, w) < 6.0 for w in walls):
            continue
        walls.append(seg)
    return walls


def _split_wall(axis_x, pos, lo, hi, n_doors, rng):
    """A wall at x = pos (axis_x) or y = pos from lo to hi, cut by n_doors doors."""
    span = hi - lo
    doors = []
    for _ in range(n_doors):
        w = rng.uniform(10.0, 16.0)
        c = rng.uniform(lo + w / 2 + 4, hi - w / 2 - 4) if span > w + 8 else (lo + hi) / 2
        doors.append((c, w))
    cuts = sorted((c - w / 2, c + w / 2) for c, w in doors)
    pieces, y = [], lo
    for a, b in cuts:
        if a > y + 1.0:
            pieces.append((y, a))
        y = max(y, b)
    if y < hi - 1.0:
        pieces.append((y, hi))
    if axis_x:
        return [((pos, a), (pos, b)) for a, b in pieces]
    return [((a, pos), (b, pos)) for a, b in pieces]


def _rooms(rng, cfg, opts=DEFAULT_OPTIONS):
    W, m = cfg.world, _margin(cfg)
    hard = {}
    walls, rooms = _room_walls(rng, cfg, opts, hard)
    i, j = rng.choice(len(rooms), 2, replace=False)
    pair = _free_pair(rng, cfg, zones=(rooms[i], rooms[j]), opts=opts)
    if pair is None:
        return _open(rng, cfg)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[pair],
                      place_zone=Region(m, m, W - m, W - m), name="rooms", hard=hard or None)


def _room_walls(rng, cfg, opts=DEFAULT_OPTIONS, hard=None):
    """(walls, rooms inset by the margin): the wall sampler of ``rooms``. With
    room_tilt_prob (v7) a single dividing wall is tilted by U(5°, 25°) about its
    middle (doors and all), its rooms shrunk by the shift so boxes keep off it."""
    W, m = cfg.world, _margin(cfg)
    mode = rng.choice(["v", "h", "both"])
    xs = rng.uniform(30.0, 70.0) if mode in ("v", "both") else None
    ys = rng.uniform(30.0, 70.0) if mode in ("h", "both") else None
    walls = []
    if xs is not None:
        if ys is None:
            walls += _split_wall(True, xs, 0.0, W, int(rng.integers(1, 3)), rng)
        else:                                          # two halves meeting the other wall
            walls += _split_wall(True, xs, 0.0, ys, int(rng.integers(0, 2)) + 1, rng)
            walls += _split_wall(True, xs, ys, W, int(rng.integers(0, 2)), rng)
    if ys is not None:
        if xs is None:
            walls += _split_wall(False, ys, 0.0, W, int(rng.integers(1, 3)), rng)
        else:
            walls += _split_wall(False, ys, 0.0, xs, int(rng.integers(0, 2)) + 1, rng)
            walls += _split_wall(False, ys, xs, W, int(rng.integers(0, 2)), rng)
    # rooms as Regions, inset so boxes keep off the walls
    gx = [0.0] + ([xs] if xs is not None else []) + [W]
    gy = [0.0] + ([ys] if ys is not None else []) + [W]
    rooms = [Region(gx[i] + m, gy[j] + m, gx[i + 1] - m, gy[j + 1] - m)
             for i in range(len(gx) - 1) for j in range(len(gy) - 1)]
    if opts.room_tilt_prob > 0 and mode != "both" and rng.random() < opts.room_tilt_prob:
        t = math.tan(math.radians(rng.uniform(5.0, 25.0))) * (1 if rng.random() < 0.5 else -1)
        d = abs(t) * W / 2                             # the largest shift of the divider
        line = xs if mode == "v" else ys
        bounds = []
        for r in rooms:                                # each room loses d on the divider's side
            lo, hi = (r.x0, r.x1) if mode == "v" else (r.y0, r.y1)
            lo, hi = (lo + d, hi) if lo > line else (lo, hi - d)
            bounds.append((r, lo, hi))
        if all(hi - lo >= 12.0 for _, lo, hi in bounds):   # else too steep for these rooms: no tilt
            if hard is not None:
                hard["rooms_tilted"] = hard.get("rooms_tilted", 0) + 1
            if mode == "v":
                walls = [tuple((x + (y - W / 2) * t, y) for x, y in w) for w in walls]
                rooms = [Region(lo, r.y0, hi, r.y1) for r, lo, hi in bounds]
            else:
                walls = [tuple((x, y + (x - W / 2) * t) for x, y in w) for w in walls]
                rooms = [Region(r.x0, lo, r.x1, hi) for r, lo, hi in bounds]
    return walls, rooms


def _quad_sides(c):
    """The four sides of a room given as corners [c0, c1, c2, c3] (c0 bottom-left,
    counter-clockwise), in the order and direction of ``_room_sides``."""
    return [(c[0], c[1]), (c[1], c[2]), (c[3], c[2]), (c[0], c[3])]


_SIDE_ENDS = {0: (0, 1), 1: (1, 2), 2: (3, 2), 3: (0, 3)}     # side -> its corners
_CORNER_SIDES = {0: (0, 3), 1: (0, 1), 2: (1, 2), 3: (2, 3)}  # corner -> the sides meeting there


def _tilt_corners(rng, corners, on_frame, opts, ok, hard, outward_only):
    """With p = room_tilt_prob: one or two of the room's inner sides tilted by
    U(5°, 25°). A side tilts when one of its ends slides along the other side
    meeting there, so that side stays straight and the room stays closed (a
    frame side keeps its ends on the frame). `ok(corners)` must accept the
    result; outward_only: only slides that enlarge the room."""
    if rng.random() >= opts.room_tilt_prob:
        return corners
    out = [tuple(c) for c in corners]
    inner = [i for i in range(4) if not on_frame[i]]
    want, done = 1 + int(rng.random() < 0.5), 0
    for _ in range(10):
        if done >= want or not inner:
            break
        i = inner[int(rng.integers(len(inner)))]
        c = _SIDE_ENDS[i][int(rng.integers(2))]
        j = [q for q in _CORNER_SIDES[c] if q != i][0]
        other = [q for q in _SIDE_ENDS[j] if q != c][0]
        far = [q for q in _SIDE_ENDS[i] if q != c][0]
        L = math.dist(out[c], out[far])
        Lj = math.dist(out[c], out[other])
        if L < 1e-6 or Lj < 1e-6:
            continue
        ux, uy = (out[c][0] - out[other][0]) / Lj, (out[c][1] - out[other][1]) / Lj
        t = L * math.tan(math.radians(rng.uniform(5.0, 25.0)))
        if not outward_only and rng.random() < 0.5:
            t = -min(t, 0.5 * Lj)
        new = list(out)
        new[c] = (out[c][0] + ux * t, out[c][1] + uy * t)
        if ok(new):
            out = new
            done += 1
    if done:
        hard["rooms_tilted"] = hard.get("rooms_tilted", 0) + 1
    return out


# ------------------------------------------- multi-box families (sketch v5)

def _box_counts(rng):
    """(k starts, m exits): k = m = 1 in a third of scenes, else any other combination."""
    if rng.random() < 1 / 3:
        return 1, 1
    combos = [(a, b) for a in (1, 2, 3) for b in (1, 2, 3) if (a, b) != (1, 1)]
    return combos[int(rng.integers(len(combos)))]


def _apart(a, b, gap) -> bool:
    return a.x1 + gap < b.x0 or b.x1 + gap < a.x0 or a.y1 + gap < b.y0 or b.y1 + gap < a.y0


def _free_boxes(rng, cfg, k, m, zone_for, avoid=None, tries=200, short_min=10.0):
    """(starts, exits) or None. `zone_for(rng, j)` gives the Region box j must lie in
    (None: skip this try); boxes are ≥ 6 apart, starts ≥ 0.3 world from exits,
    every box ≥ wall_half + region_margin from the regions `avoid(box, j)` (rooms)."""
    W = cfg.world
    n = k + m
    long_max = 40.0 if n <= 3 else 28.0                # several boxes: shorter ones
    keep = cfg.wall_half + cfg.region_margin
    starts, exits = [], []
    for j in range(n):
        is_exit = j >= k
        for _ in range(tries):
            z = zone_for(rng, j)
            if z is None:
                continue
            a, b = rng.uniform(short_min, 16.0), rng.uniform(12.0, long_max)
            w, h = (a, b) if rng.random() < 0.5 else (b, a)
            w, h = min(w, z.width), min(h, z.height)
            if w < short_min or h < short_min:
                continue
            x0, y0 = z.x0 + rng.uniform(0.0, z.width - w), z.y0 + rng.uniform(0.0, z.height - h)
            r = Region(x0, y0, x0 + w, y0 + h)
            if not all(_apart(r, q, 6.0) for q in starts + exits):
                continue
            others = starts if is_exit else exits
            if any(math.dist(r.center, q.center) < 0.3 * W for q in others):
                continue
            if avoid and any(not _apart(r, q, keep) for q in avoid(r, j)):
                continue
            (exits if is_exit else starts).append(r)
            break
        else:
            return None
    return starts, exits


def _walls_multi(rng, cfg, opts=DEFAULT_OPTIONS):
    W, m = cfg.world, _margin(cfg)
    k, n_exits = _box_counts(rng)
    whole = Region(m, m, W - m, W - m)
    boxes = _free_boxes(rng, cfg, k, n_exits, lambda r, j: whole, avoid=lambda b, j: [],
                        short_min=opts.box_short_min)
    if boxes is None:
        return _open(rng, cfg)
    starts, exits = boxes
    walls = _random_walls(rng, cfg, starts + exits, opts.slant_prob or 0.25)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(starts[0], exits[0])],
                      place_zone=whole, name="walls_multi", starts=starts, exits=exits)


def _rooms_multi(rng, cfg, opts=DEFAULT_OPTIONS):
    W, m = cfg.world, _margin(cfg)
    hard = {}
    walls, rooms = _room_walls(rng, cfg, opts, hard)
    k, n_exits = _box_counts(rng)
    boxes = _free_boxes(rng, cfg, k, n_exits, lambda r, j: rooms[int(r.integers(len(rooms)))],
                        avoid=lambda b, j: [], short_min=opts.box_short_min)
    if boxes is None:
        return _open(rng, cfg)
    starts, exits = boxes
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(starts[0], exits[0])],
                      place_zone=Region(m, m, W - m, W - m), name="rooms_multi",
                      starts=starts, exits=exits, hard=hard or None)


def _rect_gap(a, b) -> float:
    """Gap between two rectangles (0 when they touch or overlap)."""
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


def _small_room(rng, cfg, others, lo=10.0, hi=30.0):
    """One room, sides lo-hi (at least one side ≥ 14 for a door), as the Region of
    its wall centre lines: in a corner (35 %), against one frame edge (30 %) or
    free-standing; ≥ 10 units from every other room. None if none fits."""
    W = cfg.world
    for _ in range(60):
        w, h = rng.uniform(lo, hi, 2)
        if max(w, h) < 14.0:
            continue
        u = rng.random()
        if u < 0.35:                                   # a corner: two sides are the frame
            cx, cy = int(rng.integers(2)), int(rng.integers(2))
            x0, y0 = (0.0 if cx == 0 else W - w), (0.0 if cy == 0 else W - h)
        elif u < 0.65:                                 # one side is the frame
            edge = int(rng.integers(4))
            x0, y0 = rng.uniform(10.0, W - 10.0 - w), rng.uniform(10.0, W - 10.0 - h)
            if edge == 0:
                y0 = 0.0
            elif edge == 1:
                x0 = W - w
            elif edge == 2:
                y0 = W - h
            else:
                x0 = 0.0
        else:
            x0, y0 = rng.uniform(10.0, W - 10.0 - w), rng.uniform(10.0, W - 10.0 - h)
        room = Region(x0, y0, x0 + w, y0 + h)
        if all(_rect_gap(room, q) >= 10.0 for q in others):
            return room
    return None


def _room_sides(cfg, room, rng, opts=DEFAULT_OPTIONS, hard=None, corners=None):
    """The room's walls: every side not on the frame; one door (75 %) or two on
    different sides, 9.5-12 wide (the free gap beats min_bottleneck) with at
    least 2 units of wall on either side, so the room still reads as a rectangle.
    `corners` (v7): the room's tilted corners instead of the Region's."""
    W = cfg.world
    x0, y0, x1, y1 = room.x0, room.y0, room.x1, room.y1
    sides = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))]
    on_frame = [y0 <= 0.0, x1 >= W, y1 >= W, x0 <= 0.0]
    if corners is not None:
        sides = _quad_sides(corners)
    inner = [i for i in range(4) if not on_frame[i]]
    doorable = [i for i in inner if math.dist(*sides[i]) >= 14.0]
    n_doors = 1 + int(rng.random() < 0.25 and len(doorable) > 1)
    door_sides = [doorable[i] for i in rng.choice(len(doorable), n_doors, replace=False)]
    walls = []
    for i in inner:
        (ax, ay), (bx, by) = sides[i]
        if i not in door_sides:
            walls.append(sides[i])
            continue
        L = math.dist((ax, ay), (bx, by))
        ux, uy = (bx - ax) / L, (by - ay) / L
        dw = rng.uniform(9.5, min(12.0, L - 4.0))
        c = rng.uniform(dw / 2 + 2.0, L - dw / 2 - 2.0)
        lo, hi = c - dw / 2, c + dw / 2
        walls.append(((ax, ay), (ax + ux * lo, ay + uy * lo)))
        walls.append(((ax + ux * hi, ay + uy * hi), (bx, by)))
    if opts.corner_gap_prob > 0:
        walls = _open_corners(rng, cfg, walls, corners or [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], opts, hard)
    return walls


def _open_corners(rng, cfg, walls, corners, opts, hard=None):
    """Hand-drawn room corners (§10.2): with p = corner_gap_prob per corner, the
    walls meeting there stop U(0.5, 3) short of each other (one is pulled back),
    or one runs U(0.5, 4) past the other. A wall meeting the frame can only stop
    short of it. Centre-line gaps under 3 units leave the capsules touching, so
    no corner opens a passage for an agent."""
    W = cfg.world
    walls = [tuple(w) for w in walls]
    hard = hard if hard is not None else {}
    for q in corners:
        ends = [(i, e) for i, w in enumerate(walls) for e in (0, 1) if math.dist(w[e], q) < 1e-6]
        if not ends or rng.random() >= opts.corner_gap_prob:
            continue
        cross = len(ends) >= 2 and rng.random() < 0.5
        i, e = ends[int(rng.integers(len(ends)))]
        a, b = walls[i][e], walls[i][1 - e]                # a: the corner end
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        ux, uy = (a[0] - b[0]) / L, (a[1] - b[1]) / L      # pointing out through the corner
        if cross:
            t = rng.uniform(0.5, 4.0)
            hard["corners_crossed"] = hard.get("corners_crossed", 0) + 1
        else:
            t = -min(rng.uniform(0.5, 3.0), 0.5 * L)
            hard["corners_open"] = hard.get("corners_open", 0) + 1
        na = (min(max(a[0] + ux * t, 0.0), W), min(max(a[1] + uy * t, 0.0), W))
        walls[i] = (na, b) if e == 0 else (b, na)
    return walls


def _boxrooms(rng, cfg, tries=10, opts=DEFAULT_OPTIONS):
    """Small rooms next to (and around) the start / exit boxes. Some rooms hold a
    box (at least one box always stays outside), and in 60 % of scenes room 0
    holds none: a room alone must not read as a box."""
    W, m = cfg.world, _margin(cfg)
    inset = cfg.wall_half + 2.5                        # a box in a room keeps off its walls
    whole = Region(m, m, W - m, W - m)
    for _ in range(tries):
        k, n_exits = _box_counts(rng)
        n_rooms = int(rng.integers(1, 5))
        decoy = rng.random() < 0.6
        # which boxes go into a room: each non-decoy room takes one with p = 0.7
        order = [int(j) for j in rng.permutation(k + n_exits)]
        plan = [None] * (k + n_exits)
        rooms, rects, walls, holders, hard = [], [], [], 0, {}
        lo = opts.box_short_min
        for i in range(n_rooms):
            hold = (i > 0 or not decoy) and holders < k + n_exits - 1 and rng.random() < 0.7
            room = _small_room(rng, cfg, rooms, *((lo + 2 * inset, 30.0) if hold else (10.0, 30.0)))
            if room is None:
                break
            if hold:
                plan[order[holders]] = len(rooms)
                holders += 1
            corners = None
            if opts.room_tilt_prob > 0:                # v7: tilted outward, the room only grows
                on_frame = [room.y0 <= 0.0, room.x1 >= W, room.y1 >= W, room.x0 <= 0.0]

                def ok(c, others=list(rooms)):
                    bb = Region(min(q[0] for q in c), min(q[1] for q in c), max(q[0] for q in c), max(q[1] for q in c))
                    return (all(0.0 <= v <= W for q in c for v in q)
                            and all(_rect_gap(bb, o) >= 10.0 for o in others))
                corners = _tilt_corners(rng, [(room.x0, room.y0), (room.x1, room.y0), (room.x1, room.y1),
                                              (room.x0, room.y1)], on_frame, opts, ok, hard, outward_only=True)
            rects.append(room)
            if corners is not None:                    # spacing and "keep off" use the outline's box
                room = Region(min(q[0] for q in corners), min(q[1] for q in corners),
                              max(q[0] for q in corners), max(q[1] for q in corners))
            rooms.append(room)
            walls += _room_sides(cfg, rects[-1], rng, opts, hard, corners)
        if not rooms:
            continue
        inner = [Region(r.x0 + inset, r.y0 + inset, r.x1 - inset, r.y1 - inset) for r in rects]
        boxes = _free_boxes(rng, cfg, k, n_exits,
                            lambda r, j: whole if plan[j] is None else inner[plan[j]],
                            lambda box, j: rooms if plan[j] is None else [], short_min=lo)
        if boxes is None:
            continue
        starts, exits = boxes
        return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(starts[0], exits[0])],
                          place_zone=whole, name="boxrooms", starts=starts, exits=exits, rooms=rooms,
                          hard=hard or None)
    return _open(rng, cfg)


BUILDERS = {
    "open": _open,
    "corridor": _corridor,
    "room": _room,
    "doorway": _doorway,
    "two_doorway": _two_doorway,
    "cul_de_sac": _cul_de_sac,
    "crossing": _crossing,
    "walls": _walls,
    "rooms": _rooms,
    "boxrooms": _boxrooms,
    "walls_multi": _walls_multi,
    "rooms_multi": _rooms_multi,
}


# --------------------------------------------- hard cases (sketch v6, §10.2)

def _boxes_of(spec) -> list:
    if spec.starts is not None:
        return list(spec.starts) + list(spec.exits)
    return [r for pair in spec.groups for r in pair]


def _with_boxes(spec, old, new):
    """The spec with box `old` replaced by `new` everywhere it appears."""
    sub = lambda r: new if r == old else r  # noqa: E731
    return replace(spec, groups=[(sub(a), sub(b)) for a, b in spec.groups],
                   starts=[sub(r) for r in spec.starts] if spec.starts is not None else None,
                   exits=[sub(r) for r in spec.exits] if spec.exits is not None else None)


def _seg_box_dist(seg, box) -> float:
    return _seg_region_dist(seg, box, step=0.25)


def _near_walls(rng, cfg, spec, opts):
    """With p = box_near_wall_prob per box, slide the box (perpendicular to an
    axis-parallel wall it faces) until its side is U(0.3, 2.0) from that wall's
    surface (the frame counts). The swept area must cross no wall and every
    other wall stays ≥ that gap away; boxes stay apart. Only the box moves:
    agents are placed later with the simulator's wall clearance."""
    hw = cfg.wall_half
    n = 0
    for box in _boxes_of(spec):
        if rng.random() >= opts.box_near_wall_prob:
            continue
        g = rng.uniform(0.3, 2.0)
        others = [b for b in _boxes_of(spec) if b != box]
        cands = []
        for (ax, ay), (bx, by) in spec.walls:
            if abs(ax - bx) < 1e-6:                    # vertical wall at x = ax
                lo, hi = min(ay, by), max(ay, by)
                if min(hi, box.y1) - max(lo, box.y0) < 3.0:
                    continue
                for nx0 in (ax + hw + g, ax - hw - g - box.width):
                    cands.append(Region(nx0, box.y0, nx0 + box.width, box.y1))
            elif abs(ay - by) < 1e-6:                  # horizontal wall at y = ay
                lo, hi = min(ax, bx), max(ax, bx)
                if min(hi, box.x1) - max(lo, box.x0) < 3.0:
                    continue
                for ny0 in (ay + hw + g, ay - hw - g - box.height):
                    cands.append(Region(box.x0, ny0, box.x1, ny0 + box.height))
        cands.sort(key=lambda r: math.dist(r.center, box.center))
        for r in cands:
            if math.dist(r.center, box.center) > 25.0:
                break
            if r.x0 < 0 or r.y0 < 0 or r.x1 > cfg.world or r.y1 > cfg.world:
                continue
            swept = Region(min(r.x0, box.x0), min(r.y0, box.y0), max(r.x1, box.x1), max(r.y1, box.y1))
            if any(_seg_box_dist(w, swept) < hw + g - 0.05 for w in spec.walls):
                continue
            if not all(_apart(r, q, 6.0) for q in others):
                continue
            spec = _with_boxes(spec, box, r)
            n += 1
            break
    return spec, n


def _add_slit(rng, cfg, spec):
    """Two parallel walls U(3, 8) apart and U(8, 25) long: alone, or one of them an
    existing axis-parallel wall (a slit along a room wall). Kept off the boxes and
    ≥ 9 units from every other wall, so no passage gets narrower than before."""
    W = cfg.world
    keep = cfg.wall_half + cfg.region_margin
    boxes = _boxes_of(spec)
    inner = [w for w in spec.walls[4:] if abs(w[0][0] - w[1][0]) < 1e-6 or abs(w[0][1] - w[1][1]) < 1e-6]
    for _ in range(60):
        d, L = rng.uniform(3.0, 8.0), rng.uniform(8.0, 25.0)
        if inner and rng.random() < 0.5:               # alongside an existing wall
            (ax, ay), (bx, by) = inner[int(rng.integers(len(inner)))]
            vert = abs(ax - bx) < 1e-6
            lo, hi = (min(ay, by), max(ay, by)) if vert else (min(ax, bx), max(ax, bx))
            if hi - lo < 8.0:
                continue
            L = min(L, hi - lo)
            t0 = lo + rng.uniform(0.0, max(0.0, hi - lo - L))
            side = d if rng.random() < 0.5 else -d
            new = [((ax + side, t0), (ax + side, t0 + L))] if vert else [((t0, ay + side), (t0 + L, ay + side))]
            parent = [((ax, ay), (bx, by))]
        else:                                          # alone
            vert = rng.random() < 0.5
            c, t0 = rng.uniform(8.0, W - 8.0), rng.uniform(8.0, W - 8.0 - L)
            if vert:
                new = [((c, t0), (c, t0 + L)), ((c + d, t0), (c + d, t0 + L))]
            else:
                new = [((t0, c), (t0 + L, c)), ((t0, c + d), (t0 + L, c + d))]
            parent = []
        if any(not (0.0 <= v <= W) for seg in new for pt in seg for v in pt):
            continue
        if any(_seg_region_dist(seg, b) < keep for seg in new for b in boxes):
            continue
        rest = [w for w in spec.walls if w not in parent]
        if any(_seg_seg_dist(seg, w) < 9.0 for seg in new for w in rest):
            continue
        return replace(spec, walls=spec.walls + new), True
    return spec, False


def _add_long_line(rng, cfg, spec):
    """A free-standing wall U(25, 70) long parallel to a box's long side, U(4, 20)
    from it (sketch v7; the C12B drawing): locally just like the side of a
    narrow box. Kept off the other boxes; no near-miss slivers with other walls."""
    W = cfg.world
    boxes = _boxes_of(spec)
    keep = cfg.wall_half + cfg.region_margin
    for _ in range(40):
        b = boxes[int(rng.integers(len(boxes)))]
        L, d = rng.uniform(25.0, 70.0), rng.uniform(4.0, 20.0)
        horiz = b.width >= b.height                    # parallel to the long side
        side = 1 if rng.random() < 0.5 else -1
        if horiz:
            y = b.y1 + d if side > 0 else b.y0 - d
            cx = b.center[0] + rng.uniform(-0.25, 0.25) * L
            seg = ((max(3.0, cx - L / 2), y), (min(W - 3.0, cx + L / 2), y))
            off = not (3.0 <= y <= W - 3.0)
        else:
            x = b.x1 + d if side > 0 else b.x0 - d
            cy = b.center[1] + rng.uniform(-0.25, 0.25) * L
            seg = ((x, max(3.0, cy - L / 2)), (x, min(W - 3.0, cy + L / 2)))
            off = not (3.0 <= x <= W - 3.0)
        if off or math.dist(*seg) < 20.0:
            continue
        if _seg_region_dist(seg, b) < 4.0 - 1e-6:
            continue
        if any(_seg_region_dist(seg, q) < keep for q in boxes if q != b):
            continue
        if any(2.0 < _seg_seg_dist(seg, w) < 6.0 for w in spec.walls):
            continue
        return replace(spec, walls=spec.walls + [seg]), True
    return spec, False


def _build(layout, rng, cfg, opts):
    """BUILDERS[layout] with the options for the families that take them."""
    fn = BUILDERS[layout]
    if opts == DEFAULT_OPTIONS or "opts" not in inspect.signature(fn).parameters:
        return fn(rng, cfg)
    return fn(rng, cfg, opts=opts)


def _hard_cases(rng, cfg, spec, opts):
    """Slit, then boxes near walls, then (v7) a long line next to a box where
    the box ended up; counts in spec.hard."""
    hard = dict(spec.hard or {})
    if opts.slit_prob > 0 and rng.random() < opts.slit_prob:
        spec, ok = _add_slit(rng, cfg, spec)
        hard["slit"] = int(ok)
    if opts.box_near_wall_prob > 0:
        spec, n = _near_walls(rng, cfg, spec, opts)
        hard["near_wall"] = n
    if opts.long_line_prob > 0 and rng.random() < opts.long_line_prob:
        spec, ok = _add_long_line(rng, cfg, spec)
        hard["long_line"] = int(ok)
    return replace(spec, hard=hard)


# ------------------------------------------------------------------ sampling

def _links_ok(cfg, spec, clear) -> bool:
    """The spec's boxes still linked for an agent (every start to an exit, every pair)."""
    if spec.starts is not None:
        return _boxes_linked(cfg, spec.starts, spec.exits, cfg.agent_radius, clear)
    probe = Scene(walls=spec.walls)
    for start, goal in spec.groups:
        probe.start_region, probe.goal_region = start, goal
        if not regions_connected(cfg, probe, cfg.agent_radius, clear=clear):
            return False
    return True


def _touching_obstacle(rng, cfg, scene, spec, builder) -> bool:
    """Sketch v7: one obstacle touching a box from outside, or a wall (a gap of
    U(-0.3, 0.5) units between the drawn outlines: the box's rectangle, the wall's
    centre line), kept off the other boxes, the boxes still linked."""
    W = cfg.world
    boxes = _boxes_of(spec)
    walls = spec.walls[4:]
    for _ in range(60):
        r = rng.uniform(cfg.obstacle_r_min, cfg.obstacle_r_max)
        gap = rng.uniform(-0.3, 0.5)
        if walls and rng.random() < 0.5:               # a wall
            (ax, ay), (bx, by) = walls[int(rng.integers(len(walls)))]
            L = math.hypot(bx - ax, by - ay)
            if L < 6.0:
                continue
            t, sgn = rng.uniform(0.15, 0.85), (1 if rng.random() < 0.5 else -1)
            nx, ny = -(by - ay) / L * sgn, (bx - ax) / L * sgn
            c = (ax + (bx - ax) * t + nx * (r + gap), ay + (by - ay) * t + ny * (r + gap))
            target = None
        else:                                          # a box, from outside
            target = boxes[int(rng.integers(len(boxes)))]
            side = int(rng.integers(4))
            u = rng.uniform(0.15, 0.85)
            if side == 0:
                c = (target.x0 + u * target.width, target.y0 - r - gap)
            elif side == 1:
                c = (target.x1 + r + gap, target.y0 + u * target.height)
            elif side == 2:
                c = (target.x0 + u * target.width, target.y1 + r + gap)
            else:
                c = (target.x0 - r - gap, target.y0 + u * target.height)
        if not (r + cfg.wall_half <= c[0] <= W - r - cfg.wall_half and r + cfg.wall_half <= c[1] <= W - r - cfg.wall_half):
            continue
        if any(math.dist(c, o.c) < r + o.r + cfg.obstacle_gap for o in scene.obstacles):
            continue
        if any(b.distance_to(c) < r + (gap - 1e-6 if b is target else cfg.obstacle_clear_goal) for b in boxes):
            continue
        if not _links_ok(cfg, spec, clearance(cfg, builder.trial(c, r))):
            continue
        builder.commit(c, r)
        scene.obstacles.append(Obstacle(c, r))
        return True
    return False


def _place_obstacles(rng, cfg, scene, spec, n, builder, opts=DEFAULT_OPTIONS):
    """Add obstacles one at a time, rejecting any that seal off the goal."""
    zone = spec.place_zone
    if zone.width <= 0 or zone.height <= 0:
        return
    if opts.touch_prob > 0 and rng.random() < opts.touch_prob:   # v7: one touching, extra
        if _touching_obstacle(rng, cfg, scene, spec, builder):
            n += 1
            spec.hard = dict(spec.hard or {}, touch=1)
    tries = 0
    while len(scene.obstacles) < n and tries < cfg.max_place_tries:
        tries += 1
        r = rng.uniform(cfg.obstacle_r_min, min(cfg.obstacle_r_max, max(zone.width, zone.height) / 2))
        c = (rng.uniform(zone.x0, zone.x1), rng.uniform(zone.y0, zone.y1))

        if any(math.dist(c, o.c) < r + o.r + cfg.obstacle_gap for o in scene.obstacles):
            continue
        if spec.starts is not None:                    # several boxes: keep off every one
            if any(b.distance_to(c) < r + cfg.obstacle_clear_agents for b in spec.starts) or \
                    any(b.distance_to(c) < r + cfg.obstacle_clear_goal for b in spec.exits):
                continue
            trial = builder.trial(c, r)
            if not _boxes_linked(cfg, spec.starts, spec.exits, cfg.agent_radius,
                                 clearance(cfg, trial)):
                continue
            builder.commit(c, r)
            scene.obstacles.append(Obstacle(c, r))
            continue
        bad = False
        for start, goal in spec.groups:
            if start.distance_to(c) < r + cfg.obstacle_clear_agents:
                bad = True
            if goal.distance_to(c) < r + cfg.obstacle_clear_goal:
                bad = True
        if bad:
            continue

        trial = builder.trial(c, r)
        clear = clearance(cfg, trial)
        probe = Scene(walls=scene.walls, obstacles=scene.obstacles + [Obstacle(c, r)])
        ok = True
        for start, goal in spec.groups:
            probe.start_region, probe.goal_region = start, goal
            if not regions_connected(cfg, probe, cfg.agent_radius, clear=clear):
                ok = False
                break
        if not ok:
            continue

        builder.commit(c, r)
        scene.obstacles.append(Obstacle(c, r))


def _boxes_linked(cfg, starts, exits, radius, clear) -> bool:
    """Every start box reaches an exit and every exit is reached from a start box."""
    lab = _labels(free_mask(clear, radius))
    s_lab = [_labels_in_region(lab, cfg, b) for b in starts]
    e_lab = [_labels_in_region(lab, cfg, b) for b in exits]
    return all(any(a & b for b in e_lab) for a in s_lab) and all(any(a & b for a in s_lab) for b in e_lab)


def _close_spot(rng, cfg, start, mates, d):
    """A point in `start` at distance d from a random agent already in it (sketch v5)."""
    ax, ay = mates[int(rng.integers(len(mates)))].pos
    th = rng.uniform(0, 2 * math.pi)
    p = (ax + d * math.cos(th), ay + d * math.sin(th))
    return p if start.contains(p) else None


def _place_agents_multi(rng, cfg, scene, spec, n, clear, hard, close):
    """Several start boxes: every box gets an agent first (while there are agents),
    the rest go to random boxes, so one box may be crowded; each agent faces its
    nearest exit. Agents carry no goal of their own: any exit counts."""
    R2 = 2 * cfg.agent_radius
    gap = cfg.hard_gap if (hard or close) else cfg.agent_gap
    k = len(spec.starts)
    order = list(range(min(n, k))) + [int(rng.integers(k)) for _ in range(n - min(n, k))]
    placed, box_of = [], []
    for si in order:
        start = spec.starts[si]
        mates = [a for a, b in zip(placed, box_of) if b == si]
        for t in range(cfg.max_place_tries):
            if close and mates and t < cfg.max_place_tries // 2:
                p = _close_spot(rng, cfg, start, mates, R2 + gap + 0.01)
                if p is None:
                    continue
            elif hard and placed and rng.random() < 0.6:
                p = _close_spot(rng, cfg, start, placed, R2 + gap)
                if p is None:
                    continue
            else:
                p = (rng.uniform(start.x0, start.x1), rng.uniform(start.y0, start.y1))
            iy, ix = cell_of(cfg, p)
            if clear[iy, ix] < cfg.agent_radius:
                continue
            if any(math.dist(p, q.pos) < R2 + gap for q in placed):
                continue
            gx, gy = min(spec.exits, key=lambda g: g.distance_to(p)).center
            heading = math.atan2(gy - p[1], gx - p[0]) + rng.uniform(-0.45, 0.45)
            placed.append(Agent(pos=p, heading=heading))
            box_of.append(si)
            break
    scene.agents = placed
    return sorted(set(box_of))


def _place_free_agents(rng, cfg, placed, boxes, exits, n, clear, gap):
    """Sketch v7: n agents anywhere in free space that reaches an exit, with the
    simulator's spacing and wall clearance, off every box, any heading."""
    W, R = cfg.world, cfg.agent_radius
    lab = _labels(free_mask(clear, R))
    ok_labels = set()
    for g in exits:
        ok_labels |= _labels_in_region(lab, cfg, g)
    out = []
    for _ in range(n):
        for _t in range(cfg.max_place_tries):
            p = (rng.uniform(R, W - R), rng.uniform(R, W - R))
            iy, ix = cell_of(cfg, p)
            if clear[iy, ix] < R or int(lab[iy, ix]) not in ok_labels:
                continue
            if any(b.distance_to(p) < R for b in boxes):
                continue
            if any(math.dist(p, q.pos) < 2 * R + gap for q in placed + out):
                continue
            out.append(Agent(pos=p, heading=rng.uniform(-math.pi, math.pi)))
            break
    return out


def _place_agents(rng, cfg, scene, spec, n, clear, hard, close=False):
    """Spread agents over the groups' start regions, honouring physical size.

    ``close`` (sketch v5): an agent joining a start region that already holds
    one is placed at the hard spacing 2R + hard_gap from one of them."""
    R2 = 2 * cfg.agent_radius
    gap = cfg.hard_gap if (hard or close) else cfg.agent_gap
    placed = []
    order = [i % len(spec.groups) for i in range(n)]

    for gi in order:
        start, goal = spec.groups[gi]
        mates = [a for a in placed if start.contains(a.pos)] if close else []
        for t in range(cfg.max_place_tries):
            if close and mates and t < cfg.max_place_tries // 2:
                p = _close_spot(rng, cfg, start, mates, R2 + gap + 0.01)
                if p is None:
                    continue
            elif hard and placed and rng.random() < 0.6:
                # deliberately near-tangent to an existing agent
                ax, ay = placed[rng.integers(len(placed))].pos
                th = rng.uniform(0, 2 * math.pi)
                d = R2 + gap
                p = (ax + d * math.cos(th), ay + d * math.sin(th))
                if not start.contains(p):
                    continue
            else:
                p = (rng.uniform(start.x0, start.x1), rng.uniform(start.y0, start.y1))

            iy, ix = cell_of(cfg, p)
            if clear[iy, ix] < cfg.agent_radius:
                continue
            if any(math.dist(p, q.pos) < R2 + gap for q in placed):
                continue

            gx, gy = goal.center
            face = math.atan2(gy - p[1], gx - p[0])
            heading = face + rng.uniform(-0.45, 0.45)
            placed.append(Agent(pos=p, heading=heading,
                                goal_region=goal if gi > 0 else None))
            break
    scene.agents = placed


def random_scene(rng, cfg=None, layout="open", n_agents=1, n_obstacles=0,
                 hard=False, attempts=12, close=False, opts=None, info=None):
    """Sample one validated scene; None if every attempt fails validation.

    ``close``: pack the agents of each start box at the hard spacing (sketch v5).
    ``opts``: ``LayoutOptions`` (hard cases, sketch v6); None = the defaults.
    ``info``: a dict that receives the hard cases of the returned scene."""
    cfg = cfg or Config()
    opts = opts or DEFAULT_OPTIONS
    for _ in range(attempts):
        spec = _build(layout, rng, cfg, opts)
        if opts != DEFAULT_OPTIONS:
            spec = _hard_cases(rng, cfg, spec, opts)
        placement = "boxes"
        if opts.agent_mixed_prob + opts.agent_free_prob > 0:      # v7: agents outside boxes
            u = rng.random()
            placement = ("free" if u < opts.agent_free_prob else
                         "mixed" if u < opts.agent_free_prob + opts.agent_mixed_prob else "boxes")
        scene = Scene(walls=spec.walls, layout=layout, groups=list(spec.groups),
                      start_region=spec.groups[0][0], goal_region=spec.groups[0][1])
        builder = OccupancyBuilder(cfg, spec.walls)

        _place_obstacles(rng, cfg, scene, spec, n_obstacles, builder, opts)
        clear = builder.clearance()
        if placement != "boxes":
            if not _place_outside(rng, cfg, scene, spec, n_agents, clear, hard, close, placement):
                continue
        elif spec.starts is not None:
            used = _place_agents_multi(rng, cfg, scene, spec, n_agents, clear, hard, close)
            if not scene.agents:
                continue
            starts = [spec.starts[i] for i in used]          # a start box without agents is no start
            if not _boxes_linked(cfg, starts, spec.exits, cfg.agent_radius, clear):
                continue
            if len(starts) == 1 and len(spec.exits) == 1:   # one of each: the single-box form
                scene.groups = [(starts[0], spec.exits[0])]
            else:
                scene.groups, scene.starts, scene.exits = [], starts, list(spec.exits)
            scene.start_region, scene.goal_region = starts[0], spec.exits[0]
        else:
            _place_agents(rng, cfg, scene, spec, n_agents, clear, hard, close)
        if not scene.agents:
            continue

        if not is_solvable(cfg, scene, clear=clear):
            continue
        bw = bottleneck_width(cfg, scene, clear=clear)
        if cfg.check_bottleneck and bw < cfg.min_bottleneck:
            continue
        scene.bottleneck = bw
        if info is not None:
            info.clear()
            info.update(spec.hard or {})
            info["placement"] = placement
        return scene
    return None


def _unique(regs) -> list:
    out = []
    for r in regs:
        if r not in out:
            out.append(r)
    return out


def _place_outside(rng, cfg, scene, spec, n, clear, hard, close, placement) -> bool:
    """Sketch v7 placements: "mixed" puts 30-70 % of the agents in start boxes and
    the rest anywhere; "free" drops every start box and puts every agent
    anywhere. The scene keeps several-box form (``starts`` may be empty); every
    exit must be reachable from an agent. False: try another scene."""
    starts = list(spec.starts) if spec.starts is not None else _unique([s for s, _ in spec.groups])
    exits = list(spec.exits) if spec.exits is not None else _unique([g for _, g in spec.groups])
    n_box = 0 if placement == "free" else min(n, max(1, int(round(n * rng.uniform(0.3, 0.7)))))
    used = []
    if n_box:
        used = _place_agents_multi(rng, cfg, scene, replace(spec, starts=starts, exits=exits),
                                   n_box, clear, hard, close)
    else:
        scene.agents = []
    gap = cfg.hard_gap if (hard or close) else cfg.agent_gap
    kept = [starts[i] for i in used]
    scene.agents = scene.agents + _place_free_agents(rng, cfg, scene.agents, kept + exits, exits,
                                                     n - len(scene.agents), clear, gap)
    if not scene.agents:
        return False
    lab = _labels(free_mask(clear, cfg.agent_radius))
    reached = {int(lab[cell_of(cfg, a.pos)]) for a in scene.agents}
    if not all(_labels_in_region(lab, cfg, g) & reached for g in exits):
        return False                                   # an exit nobody can reach
    scene.groups, scene.starts, scene.exits = [], kept, exits
    scene.start_region, scene.goal_region = (kept[0] if kept else None), exits[0]
    return True
