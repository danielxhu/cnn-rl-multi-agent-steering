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

Every scene is validated before it is returned: start must reach goal for a
disk of the agent's radius, and the tightest passage must exceed min_bottleneck.
With several boxes every agent must reach at least one exit, every exit must
be reachable from an agent, and the bottleneck is the widest disk with which
every start box reaches an exit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass
class LayoutSpec:
    walls: list
    groups: list          # [(start_region, goal_region), ...]
    place_zone: Region    # where obstacle centres may be sampled
    name: str
    starts: list | None = None     # several boxes (MULTI_LAYOUTS); then groups is unused
    exits: list | None = None
    rooms: list | None = None      # boxrooms: each room's outline (for tests)


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


def _room(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    x0, y0 = rng.uniform(6.0, 16.0), rng.uniform(6.0, 16.0)
    x1, y1 = W - rng.uniform(6.0, 16.0), W - rng.uniform(6.0, 16.0)
    walls = boundary_walls(W) + rect_outline(x0, y0, x1, y1)
    ix0, iy0, ix1, iy1 = x0 + m, y0 + m, x1 - m, y1 - m
    return LayoutSpec(
        walls=walls,
        groups=[(Region(ix0, iy0, ix0 + 13, iy1), Region(ix1 - 13, iy0, ix1, iy1))],
        place_zone=Region(ix0 + 20, iy0, ix1 - 20, iy1),
        name="room",
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

def _box_dims(rng):
    """A start / goal box: one side 10-16, the other 12-40, either way round."""
    a, b = rng.uniform(10.0, 16.0), rng.uniform(12.0, 40.0)
    return (a, b) if rng.random() < 0.5 else (b, a)


def _free_pair(rng, cfg, zones=None):
    """(start, goal) anywhere in the world, apart; `zones` restricts each to a Region."""
    W, m = cfg.world, _margin(cfg)
    for _ in range(200):
        regs = []
        for k in range(2):
            z = zones[k] if zones else Region(m, m, W - m, W - m)
            w, h = _box_dims(rng)
            w, h = min(w, z.width), min(h, z.height)
            if w < 10 or h < 10:
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


def _walls(rng, cfg):
    W = cfg.world
    pair = _free_pair(rng, cfg)
    if pair is None:
        return _open(rng, cfg)
    s, g = pair
    walls = _random_walls(rng, cfg, [s, g])
    m = _margin(cfg)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(s, g)],
                      place_zone=Region(m, m, W - m, W - m), name="walls")


def _random_walls(rng, cfg, boxes):
    """0-6 random walls kept off every box (the wall sampler of ``walls``)."""
    W = cfg.world
    keep_off = cfg.wall_half + cfg.region_margin
    walls, want = [], int(rng.integers(0, 7))
    for _ in range(80):
        if len(walls) >= want:
            break
        L = rng.uniform(12.0, 55.0)
        diag = rng.random() < 0.25
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


def _rooms(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    walls, rooms = _room_walls(rng, cfg)
    i, j = rng.choice(len(rooms), 2, replace=False)
    pair = _free_pair(rng, cfg, zones=(rooms[i], rooms[j]))
    if pair is None:
        return _open(rng, cfg)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[pair],
                      place_zone=Region(m, m, W - m, W - m), name="rooms")


def _room_walls(rng, cfg):
    """(walls, rooms inset by the margin): the wall sampler of ``rooms``."""
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
    return walls, rooms


# ------------------------------------------- multi-box families (sketch v5)

def _box_counts(rng):
    """(k starts, m exits): k = m = 1 in a third of scenes, else any other combination."""
    if rng.random() < 1 / 3:
        return 1, 1
    combos = [(a, b) for a in (1, 2, 3) for b in (1, 2, 3) if (a, b) != (1, 1)]
    return combos[int(rng.integers(len(combos)))]


def _apart(a, b, gap) -> bool:
    return a.x1 + gap < b.x0 or b.x1 + gap < a.x0 or a.y1 + gap < b.y0 or b.y1 + gap < a.y0


def _free_boxes(rng, cfg, k, m, zone_for, avoid=None, tries=200):
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
            a, b = rng.uniform(10.0, 16.0), rng.uniform(12.0, long_max)
            w, h = (a, b) if rng.random() < 0.5 else (b, a)
            w, h = min(w, z.width), min(h, z.height)
            if w < 10 or h < 10:
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


def _walls_multi(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    k, n_exits = _box_counts(rng)
    whole = Region(m, m, W - m, W - m)
    boxes = _free_boxes(rng, cfg, k, n_exits, lambda r, j: whole, avoid=lambda b, j: [])
    if boxes is None:
        return _open(rng, cfg)
    starts, exits = boxes
    walls = _random_walls(rng, cfg, starts + exits)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(starts[0], exits[0])],
                      place_zone=whole, name="walls_multi", starts=starts, exits=exits)


def _rooms_multi(rng, cfg):
    W, m = cfg.world, _margin(cfg)
    walls, rooms = _room_walls(rng, cfg)
    k, n_exits = _box_counts(rng)
    boxes = _free_boxes(rng, cfg, k, n_exits, lambda r, j: rooms[int(r.integers(len(rooms)))],
                        avoid=lambda b, j: [])
    if boxes is None:
        return _open(rng, cfg)
    starts, exits = boxes
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(starts[0], exits[0])],
                      place_zone=Region(m, m, W - m, W - m), name="rooms_multi",
                      starts=starts, exits=exits)


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


def _room_sides(cfg, room, rng):
    """The room's walls: every side not on the frame; one door (75 %) or two on
    different sides, 9.5-12 wide (the free gap beats min_bottleneck) with at
    least 2 units of wall on either side, so the room still reads as a rectangle."""
    W = cfg.world
    x0, y0, x1, y1 = room.x0, room.y0, room.x1, room.y1
    sides = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))]
    on_frame = [y0 <= 0.0, x1 >= W, y1 >= W, x0 <= 0.0]
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
    return walls


def _boxrooms(rng, cfg, tries=10):
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
        rooms, walls, holders = [], [], 0
        for i in range(n_rooms):
            hold = (i > 0 or not decoy) and holders < k + n_exits - 1 and rng.random() < 0.7
            room = _small_room(rng, cfg, rooms, *((10.0 + 2 * inset, 30.0) if hold else (10.0, 30.0)))
            if room is None:
                break
            if hold:
                plan[order[holders]] = len(rooms)
                holders += 1
            rooms.append(room)
            walls += _room_sides(cfg, room, rng)
        if not rooms:
            continue
        inner = [Region(r.x0 + inset, r.y0 + inset, r.x1 - inset, r.y1 - inset) for r in rooms]
        boxes = _free_boxes(rng, cfg, k, n_exits,
                            lambda r, j: whole if plan[j] is None else inner[plan[j]],
                            lambda box, j: rooms if plan[j] is None else [])
        if boxes is None:
            continue
        starts, exits = boxes
        return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(starts[0], exits[0])],
                          place_zone=whole, name="boxrooms", starts=starts, exits=exits, rooms=rooms)
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


# ------------------------------------------------------------------ sampling

def _place_obstacles(rng, cfg, scene, spec, n, builder):
    """Add obstacles one at a time, rejecting any that seal off the goal."""
    zone = spec.place_zone
    if zone.width <= 0 or zone.height <= 0:
        return
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
                 hard=False, attempts=12, close=False):
    """Sample one validated scene; None if every attempt fails validation.

    ``close``: pack the agents of each start box at the hard spacing (sketch v5)."""
    cfg = cfg or Config()
    for _ in range(attempts):
        spec = BUILDERS[layout](rng, cfg)
        scene = Scene(walls=spec.walls, layout=layout, groups=list(spec.groups),
                      start_region=spec.groups[0][0], goal_region=spec.groups[0][1])
        builder = OccupancyBuilder(cfg, spec.walls)

        _place_obstacles(rng, cfg, scene, spec, n_obstacles, builder)
        clear = builder.clearance()
        if spec.starts is not None:
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
        return scene
    return None
