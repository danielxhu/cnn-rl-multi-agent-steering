"""ORCA baseline controller (DESIGN_ORCA.md): optimal reciprocal collision avoidance.

Implemented from van den Berg, Guy, Lin and Manocha, "Reciprocal n-body
collision avoidance" (ISRR 2011) and the publicly documented RVO2 algorithm;
no third-party code. Units are world units and steps (dt = 1).

Owns: ``linear_program`` (2-D incremental LP with a circular speed limit and
the 3-D fallback), ``orca_velocity`` (one agent's half-planes and LP),
``ORCAController`` (the evaluation controller: preferred velocity, neighbour
search, unicycle conversion).

Lines (half-planes) use the RVO2 convention: ``(point, direction)`` with a
unit direction; the permitted side is the *left* of the direction, i.e.
``{x : det(direction, point - x) <= 0}``. ``det(d, p - x)`` is how far x lies
outside the line.

Invariants:
- Every agent's velocity is computed from the same world state; the
  controller returns all actions at once and the simulator steps them together.
- Deterministic: constraints are processed in a fixed order (static first,
  then agents by distance); no randomness anywhere.
- Static half-planes are hard; when the program is infeasible only the agent
  half-planes are relaxed (the paper's 3-D program).
- ORCA reads exact positions and velocities from the simulator (more than the
  learned policy's 50 numbers); that is the point of the baseline (§0).
- ``map_scene`` (DESIGN_GAP.md §5) replaces only the walls and obstacles the
  controller plans around; agents, goals and physics stay the simulator's. None
  (the default) plans on the simulator's own scene.
"""
from __future__ import annotations

import _paths  # noqa: F401

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-9
N_PATH_DIRS = 32                       # candidate directions of orca_path (§1)
# Exact symmetry deadlocks ORCA: heading straight at a disk's centre, or into
# mirror-image partners (four agents swapping across a point), puts the LP's
# optimum on a vertex or line exactly on the axis, and the agents slow to a
# halt there. RVO2's examples break it with a random perturbation of the
# preferred velocity; here, deterministically, an agent that ORCA slows below
# half its preferred speed re-plans with the preferred velocity turned
# PASS_RIGHT_DEG to its right (a pass-on-the-right convention) and follows
# that answer. Keeping the better of the two by progress does not work: on
# the axis the straight answer always wins by a little, and everyone
# decelerates forever.
PASS_RIGHT_DEG = 30.0


@dataclass
class ORCAParams:
    tau: float = 5.0                   # agent-agent time horizon, steps
    tau_obs: float = 3.0               # static time horizon, steps
    margin: float = 0.3                # added to every combined radius
    neighbor_dist: float = 22.0        # = Config.fov_range
    max_neighbors: int = 10


# ----------------------------------------------------------------- LP (§2.3)

def _det(ax, ay, bx, by):
    return ax * by - ay * bx


def _lp1(lines, i, radius, opt, direction_opt):
    """Optimise on line i subject to lines[:i] and the speed disk; None if infeasible."""
    (px, py), (dx, dy) = lines[i]
    dot = px * dx + py * dy
    disc = dot * dot + radius * radius - (px * px + py * py)
    if disc < 0.0:
        return None                                    # the line misses the speed disk
    s = math.sqrt(disc)
    t_left, t_right = -dot - s, -dot + s
    for j in range(i):
        (qx, qy), (ex, ey) = lines[j]
        denom = _det(dx, dy, ex, ey)
        numer = _det(ex, ey, px - qx, py - qy)
        if abs(denom) <= EPS:                          # parallel lines
            if numer < 0.0:
                return None
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        t = t_right if opt[0] * dx + opt[1] * dy > 0.0 else t_left
    else:
        t = min(max(dx * (opt[0] - px) + dy * (opt[1] - py), t_left), t_right)
    return (px + t * dx, py + t * dy)


def _lp2(lines, radius, opt, direction_opt):
    """(index of the first line that could not be satisfied or len(lines), result)."""
    if direction_opt:                                  # opt is a unit direction
        res = (opt[0] * radius, opt[1] * radius)
    elif opt[0] * opt[0] + opt[1] * opt[1] > radius * radius:
        n = math.hypot(opt[0], opt[1])
        res = (opt[0] / n * radius, opt[1] / n * radius)
    else:
        res = (opt[0], opt[1])
    for i, ((px, py), (dx, dy)) in enumerate(lines):
        if _det(dx, dy, px - res[0], py - res[1]) > 0.0:   # res violates line i
            new = _lp1(lines, i, radius, opt, direction_opt)
            if new is None:
                return i, res
            res = new
    return len(lines), res


def _lp3(lines, n_hard, begin, radius, res):
    """Keep lines[:n_hard] hard, minimise the largest violation of the others."""
    distance = 0.0
    for i in range(begin, len(lines)):
        (px, py), (dx, dy) = lines[i]
        if _det(dx, dy, px - res[0], py - res[1]) <= distance:
            continue
        proj = list(lines[:n_hard])
        for j in range(n_hard, i):
            (qx, qy), (ex, ey) = lines[j]
            d = _det(dx, dy, ex, ey)
            if abs(d) <= EPS:
                if dx * ex + dy * ey > 0.0:
                    continue                           # same direction: j is implied
                point = (0.5 * (px + qx), 0.5 * (py + qy))
            else:
                t = _det(ex, ey, px - qx, py - qy) / d
                point = (px + t * dx, py + t * dy)
            ux, uy = ex - dx, ey - dy
            n = math.hypot(ux, uy)
            proj.append((point, (ux / n, uy / n)))
        fail, new = _lp2(proj, radius, (-dy, dx), True)
        if fail == len(proj):                          # else keep res (rounding only)
            res = new
        distance = _det(dx, dy, px - res[0], py - res[1])
    return res


def linear_program(lines, radius, v_pref, n_hard=0) -> np.ndarray:
    """The velocity closest to ``v_pref`` inside every line and ``|v| <= radius``.

    ``lines``: sequence of ``(point, direction)`` pairs (module docstring);
    the first ``n_hard`` are hard. Infeasible: the 3-D fallback.
    """
    lines = [((float(p[0]), float(p[1])), (float(d[0]), float(d[1]))) for p, d in lines]
    opt = (float(v_pref[0]), float(v_pref[1]))
    fail, res = _lp2(lines, radius, opt, False)
    if fail < len(lines):
        res = _lp3(lines, n_hard, fail, radius, res)
    return np.array(res)


def max_violation(lines, x) -> float:
    """Largest distance by which x lies outside a line (<= 0: x satisfies all)."""
    if not len(lines):
        return -np.inf
    return max(_det(d[0], d[1], p[0] - x[0], p[1] - x[1]) for p, d in lines)


# ----------------------------------------------------------- half-planes (§2.2)

def _orca_line(rel_pos, rel_vel, v_self, R, tau, share):
    """The ORCA line of one neighbour: ``share`` = 0.5 reciprocal, 1.0 static."""
    px, py = rel_pos
    vx, vy = rel_vel
    dist_sq = px * px + py * py
    r_sq = R * R
    if dist_sq > r_sq:
        wx, wy = vx - px / tau, vy - py / tau          # from the cutoff centre
        w_len_sq = wx * wx + wy * wy
        dot1 = wx * px + wy * py
        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # project on the cutoff circle
            w_len = math.sqrt(w_len_sq)
            ux_, uy_ = wx / w_len, wy / w_len
            direction = (uy_, -ux_)
            k = R / tau - w_len
            u = (k * ux_, k * uy_)
        else:
            # project on a leg of the cone
            leg = math.sqrt(dist_sq - r_sq)
            if _det(px, py, wx, wy) > 0.0:             # left leg
                direction = ((px * leg - py * R) / dist_sq, (px * R + py * leg) / dist_sq)
            else:                                      # right leg
                direction = (-(px * leg + py * R) / dist_sq, -(-px * R + py * leg) / dist_sq)
            dot2 = vx * direction[0] + vy * direction[1]
            u = (dot2 * direction[0] - vx, dot2 * direction[1] - vy)
    else:
        # already overlapping: the cone of a one-step horizon (the paper's rule)
        wx, wy = vx - px, vy - py
        w_len = math.hypot(wx, wy)
        if w_len < EPS:                                # exactly on top: push along -rel_pos
            n = max(math.hypot(px, py), EPS)
            ux_, uy_ = -px / n, -py / n
        else:
            ux_, uy_ = wx / w_len, wy / w_len
        direction = (uy_, -ux_)
        k = R - w_len
        u = (k * ux_, k * uy_)
    point = (v_self[0] + share * u[0], v_self[1] + share * u[1])
    return point, direction


def orca_velocity(p, v, v_pref, neighbors, statics, cfg, params) -> np.ndarray:
    """New velocity of one agent (pure function, §2.1 steps 5).

    ``neighbors``: [(position, velocity)] of other agents, nearest first.
    ``statics``: [(centre, radius)] of static disks (obstacles, wall disks),
    nearest first. Returns the chosen 2-D velocity.
    """
    lines = []
    R_agent = 2 * cfg.agent_radius + params.margin
    for c, r in statics:
        rel = (c[0] - p[0], c[1] - p[1])
        lines.append(_orca_line(rel, (v[0], v[1]), v, cfg.agent_radius + r + params.margin,
                                params.tau_obs, 1.0))
    n_hard = len(lines)
    for q, w in neighbors:
        rel = (q[0] - p[0], q[1] - p[1])
        lines.append(_orca_line(rel, (v[0] - w[0], v[1] - w[1]), v, R_agent, params.tau, 0.5))
    return linear_program(lines, cfg.max_speed, v_pref, n_hard)


# ------------------------------------------------------------- unicycle (§2.4)

def velocity_to_action(v_star, heading, v_pref, cfg) -> np.ndarray:
    """(turn, speed) in the simulator's normalised units for a free velocity.

    With no velocity to follow (|v*| ~ 0) the agent turns toward ``v_pref``
    at speed 0, so a blocked agent keeps facing its goal instead of freezing.
    """
    sp = math.hypot(v_star[0], v_star[1])
    if sp > 1e-6:
        theta = math.atan2(v_star[1], v_star[0])
    elif math.hypot(v_pref[0], v_pref[1]) > 1e-6:
        theta = math.atan2(v_pref[1], v_pref[0])
    else:
        return np.zeros(2)
    e = (theta - heading + math.pi) % (2 * math.pi) - math.pi
    turn = min(max(e / cfg.max_turn, -1.0), 1.0)
    speed = min(max(sp * max(0.0, math.cos(e)) / cfg.max_speed, 0.0), 1.0)
    return np.array([turn, speed])


# ------------------------------------------------------------ controller

def _seg_nearest(p, a, b):
    e = b - a
    ll = float(e @ e)
    t = 0.0 if ll < 1e-12 else min(max(float((p - a) @ e) / ll, 0.0), 1.0)
    return a + t * e


class ORCAController:
    """evaluate.py's controller protocol: ``reset(sim)``, ``act(sim, obs51)``."""

    def __init__(self, mode="straight", tau=5.0, tau_obs=3.0, margin=0.3, neighbor_dist=None,
                 max_neighbors=10, map_scene=None):
        if mode not in ("straight", "path"):
            raise ValueError(f"mode must be 'straight' or 'path', not {mode!r}")
        self.mode = mode
        self.name = f"orca_{mode}"
        self._kw = dict(tau=tau, tau_obs=tau_obs, margin=margin, max_neighbors=max_neighbors)
        self._neighbor_dist = neighbor_dist
        self.map_scene = map_scene

    def reset(self, sim):
        cfg = sim.cfg
        self.params = ORCAParams(neighbor_dist=self._neighbor_dist or cfg.fov_range, **self._kw)
        world = sim.scene if self.map_scene is None else self.map_scene
        self.obs_c = np.array([o.c for o in world.obstacles], float).reshape(-1, 2)
        self.obs_r = np.array([o.r for o in world.obstacles], float).reshape(-1)
        self.walls = [(np.asarray(a, float), np.asarray(b, float)) for a, b in world.walls]
        self.goals = [sim.scene.goal_for(a) for a in sim.scene.agents]
        self.fields = {}
        self.no_path_steps = 0                         # agent-steps with no finite path (read only)
        if self.mode == "path":
            from reachability import clearance, geodesic_field, occupancy
            clear = clearance(cfg, occupancy(cfg, world))
            for g in self.goals:
                if g not in self.fields:
                    self.fields[g] = geodesic_field(cfg, world, g, cell=cfg.geo_cell,
                                                    clear=clear)
        c, s_ = math.cos(math.radians(-PASS_RIGHT_DEG)), math.sin(math.radians(-PASS_RIGHT_DEG))
        self.rot = np.array([[c, -s_], [s_, c]])
        ang = np.arange(N_PATH_DIRS) * (2 * math.pi / N_PATH_DIRS)
        self.dirs = np.stack([np.cos(ang), np.sin(ang)], 1)

    # -- preferred velocity (§1) ----------------------------------------
    def _straight(self, p, goal, cfg):
        gx, gy = goal.nearest_point((float(p[0]), float(p[1])))
        dx, dy = gx - p[0], gy - p[1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return np.zeros(2)
        return np.array([dx, dy]) / d * cfg.max_speed

    def _path(self, p, goal, cfg):
        """Direction among 32 whose point at max_speed has the lowest path distance."""
        straight = self._straight(p, goal, cfg)
        field = self.fields[goal]
        cand = p[None, :] + cfg.max_speed * self.dirs
        vals = _bilinear_finite(field, cand, cfg.geo_cell)
        if not np.isfinite(vals).any():
            self.no_path_steps += 1
            return straight                            # no finite path around: fall back
        k = int(np.argmin(vals))
        if goal.contains((float(cand[k, 0]), float(cand[k, 1]))):
            return straight
        return self.dirs[k] * cfg.max_speed

    # -- neighbours (§2.1 steps 3-4) -------------------------------------
    def _statics(self, p, cfg):
        nd = self.params.neighbor_dist
        out = []
        for c, r in zip(self.obs_c, self.obs_r):
            d = math.hypot(c[0] - p[0], c[1] - p[1]) - r
            if d <= nd:
                out.append((d, (float(c[0]), float(c[1])), float(r)))
        for a, b in self.walls:                        # one disk per wall, at its nearest point
            q = _seg_nearest(p, a, b)
            d = math.hypot(q[0] - p[0], q[1] - p[1]) - cfg.wall_half
            if d <= nd:
                out.append((d, (float(q[0]), float(q[1])), cfg.wall_half))
        out.sort(key=lambda t: t[0])
        return [(c, r) for _, c, r in out]

    def act(self, sim, obs51):
        cfg = sim.cfg
        n = len(sim.pos)
        pos = sim.pos.copy()
        vel = sim.speed[:, None] * np.stack([np.cos(sim.heading), np.sin(sim.heading)], 1)
        active = sim.active.copy()
        actions = np.zeros((n, 2))
        idx = np.flatnonzero(active)
        for i in idx:
            p = pos[i]
            goal = self.goals[i]
            v_pref = self._path(p, goal, cfg) if self.mode == "path" else self._straight(p, goal, cfg)
            others = idx[idx != i]
            if len(others):
                d = np.hypot(*(pos[others] - p).T)
                keep = d <= self.params.neighbor_dist
                order = others[keep][np.argsort(d[keep], kind="stable")][:self.params.max_neighbors]
                neigh = [((float(pos[j, 0]), float(pos[j, 1])), (float(vel[j, 0]), float(vel[j, 1])))
                         for j in order]
            else:
                neigh = []
            pi_, vi_ = (float(p[0]), float(p[1])), (float(vel[i, 0]), float(vel[i, 1]))
            statics = self._statics(p, cfg)
            v_star = orca_velocity(pi_, vi_, v_pref, neigh, statics, cfg, self.params)
            pref = math.hypot(v_pref[0], v_pref[1])
            if pref > 1e-9 and math.hypot(*v_star) < 0.5 * pref:       # symmetry breaker
                v_star = orca_velocity(pi_, vi_, self.rot @ v_pref, neigh, statics, cfg,
                                       self.params)
            actions[i] = velocity_to_action(v_star, float(sim.heading[i]), v_pref, cfg)
        return actions


def _bilinear_finite(field, pts, cell):
    """Bilinear lookup at (k, 2) world points using finite corners only; inf if none.

    Unlike ``reachability.geodesic_distance`` there is no neighbourhood
    fallback: a candidate point inside a wall must not borrow a value from
    the wall's far side.
    """
    m = field.shape[0]
    u = pts[:, 0] / cell - 0.5
    v = pts[:, 1] / cell - 0.5
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    fx, fy = u - x0, v - y0
    num = np.zeros(len(pts))
    den = np.zeros(len(pts))
    for oy, wy in ((0, 1.0 - fy), (1, fy)):
        for ox, wx in ((0, 1.0 - fx), (1, fx)):
            iy, ix = y0 + oy, x0 + ox
            inside = (iy >= 0) & (iy < m) & (ix >= 0) & (ix < m)
            val = np.full(len(pts), np.inf)
            val[inside] = field[iy[inside], ix[inside]]
            ok = np.isfinite(val) & (wy * wx > 0)
            num += np.where(ok, wy * wx * np.where(ok, val, 0.0), 0.0)
            den += np.where(ok, wy * wx, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / den, np.inf)
