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

Every scene is validated before it is returned: start must reach goal for a
disk of the agent's radius, and the tightest passage must exceed min_bottleneck.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import Config
from reachability import (
    OccupancyBuilder,
    bottleneck_width,
    clearance,
    is_solvable,
    regions_connected,
)
from scene import Agent, Obstacle, Region, Scene, boundary_walls

LAYOUTS = ["open", "corridor", "room", "doorway", "two_doorway", "cul_de_sac", "crossing"]
FREE_LAYOUTS = ["walls", "rooms"]
# layout cycles generate.py samples from; "mixed2" is ~40 % free-form
MIXES = {"mixed": LAYOUTS,
         "mixed2": LAYOUTS + ["walls", "walls", "walls", "rooms", "rooms"]}


@dataclass
class LayoutSpec:
    walls: list
    groups: list          # [(start_region, goal_region), ...]
    place_zone: Region    # where obstacle centres may be sampled
    name: str


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
        if min(_seg_region_dist(seg, s), _seg_region_dist(seg, g)) < keep_off:
            continue
        # no near-miss slivers: another wall is either crossed or well apart
        if any(2.0 < _seg_seg_dist(seg, w) < 6.0 for w in walls):
            continue
        walls.append(seg)
    m = _margin(cfg)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[(s, g)],
                      place_zone=Region(m, m, W - m, W - m), name="walls")


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
    i, j = rng.choice(len(rooms), 2, replace=False)
    pair = _free_pair(rng, cfg, zones=(rooms[i], rooms[j]))
    if pair is None:
        return _open(rng, cfg)
    return LayoutSpec(walls=boundary_walls(W) + walls, groups=[pair],
                      place_zone=Region(m, m, W - m, W - m), name="rooms")


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


def _place_agents(rng, cfg, scene, spec, n, clear, hard):
    """Spread agents over the groups' start regions, honouring physical size."""
    R2 = 2 * cfg.agent_radius
    gap = cfg.hard_gap if hard else cfg.agent_gap
    placed = []
    order = [i % len(spec.groups) for i in range(n)]

    from reachability import cell_of

    for gi in order:
        start, goal = spec.groups[gi]
        for _ in range(cfg.max_place_tries):
            if hard and placed and rng.random() < 0.6:
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
                 hard=False, attempts=12):
    """Sample one validated scene; None if every attempt fails validation."""
    cfg = cfg or Config()
    for _ in range(attempts):
        spec = BUILDERS[layout](rng, cfg)
        scene = Scene(walls=spec.walls, layout=layout, groups=list(spec.groups),
                      start_region=spec.groups[0][0], goal_region=spec.groups[0][1])
        builder = OccupancyBuilder(cfg, spec.walls)

        _place_obstacles(rng, cfg, scene, spec, n_obstacles, builder)
        clear = builder.clearance()
        _place_agents(rng, cfg, scene, spec, n_agents, clear, hard)
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
