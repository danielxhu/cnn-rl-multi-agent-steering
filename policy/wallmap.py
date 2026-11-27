"""E7 wall memory: each agent's own map of the walls it has seen (DESIGN_E7.md §1).

Owns ``WallMemory``: per agent, a raster of the static hits its rays reported,
placed by dead reckoning; the height field (path distance around the
remembered walls to a point far along the goal bearing); the downhill
direction; and the lock-and-switch rule for multi-exit goals.

Invariants:
- Reads only what the agent senses: the ray readings and the bearing of the
  observation it is given, its own speed and turn (odometry), and the bearing
  of each exit of its goal (no distances). The exit ranking at a lock or a
  switch is the simulator's own rule for choosing which exit the bearing
  points at (straight-line distance, §1.4).
- With ``odo_noise = "none"`` the estimated pose *is* the simulator's pose.
  With noise, all randomness comes from the generator given at construction.
- ``observe`` is called exactly once per simulator step of a scene (the first
  call right after ``reset``), for every agent; agents no longer active are
  left untouched and get zero extra columns.
- The memory of one scene never leaks into the next: ``reset`` clears it.
- ``exit_rule = "orbit"`` (evaluation option, 20 Nov 2026; default
  ``"enclosed"`` is the trained rule): besides the enclosure test, an agent
  whose direction to its target exit (estimated heading + bearing) has turned
  through ``ORBIT_TURN`` (180°) in one sense has gone round the exit without
  getting in; the exit is marked blocked and the next one taken. Uses only the bearing and
  odometry, like the rest of the memory.
"""
from __future__ import annotations

import _paths  # noqa: F401

import math

import cv2
import numpy as np

from config import E_OBSTACLE, E_WALL
from reachability import geodesic_distance, geodesic_wavefront
from scene import RegionSet

EXTRA = 2                   # added observation columns: cos, sin of the downhill direction
RASTER = 0.5                # memory cell, world units
FIELD_CELL = 2.0            # height-field cell
BOX_MARGIN = 4.0            # virtual goal stays in [margin, world - margin]^2
SOURCE_R = 4.0              # field sources: free cells within this of the virtual goal
LEAK_GUARD = 0.6            # a field cell with any point this close to a seen surface is shut
DESCENT_STEPS = 3           # downhill: steepest-descent steps on the field (6 units)
_NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
FIELD_REACH = 2             # wavefront moves up to (2, 1): 16 directions (reachability.geodesic_wavefront)
ORBIT_TURN = math.pi           # exit_rule "orbit": switch after the target direction turns this far
ODO_SIGMA = {"none": (0.0, 0.0), "low": (0.02, 0.5), "high": (0.05, 2.0)}   # (speed frac, turn deg)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def members(goal) -> list:
    return list(goal) if isinstance(goal, RegionSet) else [goal]


def bearing_to(region, pos, heading) -> float:
    """Bearing / pi of ``region`` from (pos, heading): the simulator's formula."""
    gx, gy = region.nearest_point(tuple(pos))
    rel = math.atan2(gy - pos[1], gx - pos[0]) - heading
    return ((rel + math.pi) % (2 * math.pi) - math.pi) / math.pi


def _segment_points(p, q):
    """Points along the segment p-q, a quarter raster cell apart."""
    n = max(2, int(math.ceil(math.dist(p, q) / (RASTER / 4))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = p[None, :] * (1 - t) + q[None, :] * t
    return pts


class WallMemory:
    def __init__(self, sim_cfg, env_cfg, rng=None, exit_rule="enclosed"):
        if exit_rule not in ("enclosed", "orbit"):
            raise ValueError(f"exit_rule must be 'enclosed' or 'orbit', not {exit_rule!r}")
        self.exit_rule = exit_rule
        self.c = sim_cfg
        self.every = int(env_cfg.wallmap_every)
        self.visit = float(getattr(env_cfg, "wallmap_visit", 0.0))   # §14.3; 0 = off
        sv, st = ODO_SIGMA[env_cfg.odo_noise]
        self.sig_v, self.sig_t = sv, math.radians(st)
        if (sv or st) and rng is None:
            raise ValueError("odometry noise needs a generator")
        self.rng = rng
        self.m = int(round(sim_cfg.world / RASTER))
        self.mc = int(round(sim_cfg.world / FIELD_CELL))
        self.f = self.m // self.mc
        idx = (np.arange(self.mc) + 0.5) * FIELD_CELL
        self.px, self.py = np.meshgrid(idx, idx)              # px[iy, ix]

    # ---------------------------------------------------------------- state
    def reset(self, sim):
        n = len(sim.pos)
        S = self.c.n_sectors
        self.sector_off = np.asarray(sim._sector_off).reshape(S, -1).mean(1)
        self.n = n
        self.est_pos = sim.pos.astype(float).copy()
        self.est_head = sim.heading.astype(float).copy()
        self.prev_head = sim.heading.astype(float).copy()
        self.prev_active = sim.active.copy()
        self.occ = np.zeros((n, self.m, self.m), bool)
        self.goals = [members(sim.scene.goal_for(a)) for a in sim.scene.agents]
        self.target = [int(np.argmin([r.distance_to(tuple(sim.pos[i])) for r in g]))
                       for i, g in enumerate(self.goals)]
        self.blocked = [set() for _ in range(n)]
        self.switches = np.zeros(n, int)
        self.orbit = np.zeros(n)                  # turning of the direction to the target (rad)
        self.prev_dir = [None] * n
        self.field = [None] * n
        self.computed = np.zeros(n, bool)
        self.started = False
        self.visits = np.zeros((n, self.mc, self.mc), np.float32) if self.visit > 0 else None

    def _odometry(self, sim):
        moved = self.prev_active
        if self.sig_v == 0.0 and self.sig_t == 0.0:
            self.est_pos[moved] = sim.pos[moved]
            self.est_head[moved] = sim.heading[moved]
        else:
            for i in np.flatnonzero(moved):
                turn = wrap(sim.heading[i] - self.prev_head[i])
                self.est_head[i] = wrap(self.est_head[i] + turn + self.rng.normal(0.0, self.sig_t))
                v = sim.speed[i] * (1.0 + self.rng.normal(0.0, self.sig_v))
                self.est_pos[i] += v * np.array([math.cos(self.est_head[i]),
                                                 math.sin(self.est_head[i])])
        self.prev_head = sim.heading.astype(float).copy()
        self.prev_active = sim.active.copy()

    # --------------------------------------------------------------- memory
    def _mark(self, i, pts):
        ix = np.floor(pts[:, 0] / RASTER).astype(int)
        iy = np.floor(pts[:, 1] / RASTER).astype(int)
        ok = (ix >= 0) & (ix < self.m) & (iy >= 0) & (iy < self.m)
        self.occ[i, iy[ok], ix[ok]] = True

    def insert(self, i, o):
        """Remember the static hits of agent i's rays (``o``: its observation row)."""
        c, S = self.c, self.c.n_sectors
        ang = self.est_head[i] + self.sector_off
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        chunks = []
        for k in (E_OBSTACLE, E_WALL):
            r = np.asarray(o[k * S:(k + 1) * S], float)
            hit = r < 1.0 - 1e-6
            if not hit.any():
                continue
            pts = self.est_pos[i][None, :] + (r * c.fov_range)[:, None] * dirs
            chunks.append(pts[hit])
            for j in range(S - 1):            # join neighbours closer than any legal passage
                if hit[j] and hit[j + 1] and math.dist(pts[j], pts[j + 1]) < c.min_bottleneck:
                    chunks.append(_segment_points(pts[j], pts[j + 1]))
        if chunks:
            self._mark(i, np.concatenate(chunks))

    # ---------------------------------------------------------------- field
    def free_mask(self, i) -> np.ndarray:
        """(mc, mc) field cells an agent centre may use, from agent i's memory."""
        occ = self.occ[i]
        if not occ.any():
            return np.ones((self.mc, self.mc), bool)
        clear = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * RASTER
        mc, f = self.mc, self.f
        blocks = clear.reshape(mc, f, mc, f)
        return (blocks >= self.c.agent_radius).any(axis=(1, 3)) & \
               (blocks >= LEAK_GUARD).all(axis=(1, 3))

    def virtual_goal(self, i, bearing):
        """Method A: where the ray along ``bearing`` (/pi, relative) leaves the box."""
        lo, hi = BOX_MARGIN, self.c.world - BOX_MARGIN
        q = np.clip(self.est_pos[i], lo, hi)
        a = self.est_head[i] + bearing * math.pi
        u = np.array([math.cos(a), math.sin(a)])
        ts = []
        for k in range(2):
            if u[k] > 1e-12:
                ts.append((hi - q[k]) / u[k])
            elif u[k] < -1e-12:
                ts.append((lo - q[k]) / u[k])
        t = max(0.0, min(ts)) if ts else 0.0
        return q + t * u

    def field_for(self, free, goal_pt):
        dist = np.hypot(self.px - goal_pt[0], self.py - goal_pt[1])
        D = np.where(free & (dist <= SOURCE_R), dist, np.inf).ravel()
        if not np.isfinite(D).any():
            return None
        return geodesic_wavefront(free, D, FIELD_CELL, reach=FIELD_REACH)

    def _value(self, field, p) -> float:
        return geodesic_distance(self.c, field, p, FIELD_CELL)

    def update_field(self, i, sim):
        """New field for agent i; switch exit when the target is cut off (§1.4)."""
        free = self.free_mask(i)
        g = self.goals[i]
        for _ in range(len(g) + 1):
            b = bearing_to(g[self.target[i]], sim.pos[i], sim.heading[i])
            fld = self.field_for(free, self.virtual_goal(i, b))
            self.field[i] = fld
            if len(g) == 1 or fld is None or np.isfinite(self._value(fld, self.est_pos[i])):
                return
            if not self._switch(i, sim):
                return

    def _switch(self, i, sim) -> bool:
        """Mark the target blocked and take the nearest other exit; False if unchanged."""
        g = self.goals[i]
        self.blocked[i].add(self.target[i])
        if len(self.blocked[i]) >= len(g):
            self.blocked[i] = set()
        p = tuple(sim.pos[i])
        cand = [k for k in range(len(g)) if k not in self.blocked[i]]
        new = min(cand, key=lambda k: g[k].distance_to(p))
        self.orbit[i], self.prev_dir[i] = 0.0, None
        if new == self.target[i]:
            return False
        self.target[i] = new
        self.switches[i] += 1
        return True

    def _orbit_check(self, i, sim, bearing) -> bool:
        """Accumulate the turning of the direction to the target; switch at ``ORBIT_TURN``."""
        a = self.est_head[i] + bearing * math.pi
        if self.prev_dir[i] is not None:
            self.orbit[i] += wrap(a - self.prev_dir[i])
        self.prev_dir[i] = a
        if abs(self.orbit[i]) >= ORBIT_TURN and self._switch(i, sim):
            self.update_field(i, sim)
            return True
        return False

    def downhill(self, i, bearing) -> np.ndarray:
        """(cos, sin) of the downhill direction relative to the estimated heading.

        Steepest descent on the field cells for ``DESCENT_STEPS`` steps from
        the agent's cell (8 neighbours, no corner cutting past an
        unreachable cell); the direction points at the cell reached. One
        cell of the field is too coarse to compare neighbours directly. With
        no field, no finite cell around the agent, or no descent, the bearing
        is used.

        With ``visit`` > 0 (DESIGN_E7 §14.3, after LRTA*) the agent first adds
        ``visit`` to the field cell it is in and the descent runs on field +
        visits, so an area it keeps circling rises until the descent leads
        out; the field itself (and the exit switch's reachability) is unchanged."""
        fld = self.field[i]
        if fld is not None and self.visits is not None:
            m = self.mc
            p = self.est_pos[i]
            cx = min(max(int(p[0] // FIELD_CELL), 0), m - 1)
            cy = min(max(int(p[1] // FIELD_CELL), 0), m - 1)
            self.visits[i, cy, cx] += self.visit
            fld = fld + self.visits[i]
        ang = None
        if fld is not None:
            target = self._descend(fld, self.est_pos[i])
            if target is not None:
                v = target - self.est_pos[i]
                if math.hypot(v[0], v[1]) > 1e-6:
                    ang = math.atan2(v[1], v[0]) - self.est_head[i]
        if ang is None:
            ang = bearing * math.pi
        return np.array([math.cos(ang), math.sin(ang)], np.float32)

    def _descend(self, fld, p):
        m = fld.shape[0]
        cx = min(max(int(p[0] // FIELD_CELL), 0), m - 1)
        cy = min(max(int(p[1] // FIELD_CELL), 0), m - 1)
        own = (cy, cx)
        fin = np.isfinite(fld)
        if not fin[cy, cx]:                               # pressed against a remembered wall
            best = None
            for dy, dx in _NBRS:
                y, x = cy + dy, cx + dx
                if 0 <= y < m and 0 <= x < m and fin[y, x] and (best is None or fld[y, x] < fld[best]):
                    best = (y, x)
            if best is None:
                return None
            cy, cx = best
        start = (cy, cx)
        for _ in range(DESCENT_STEPS):
            best, bv = None, fld[cy, cx]
            for dy, dx in _NBRS:
                y, x = cy + dy, cx + dx
                if not (0 <= y < m and 0 <= x < m) or not fin[y, x]:
                    continue
                if dy and dx and not (fin[cy + dy, cx] and fin[cy, cx + dx]):
                    continue
                if fld[y, x] < bv:
                    best, bv = (y, x), fld[y, x]
            if best is None:
                break
            cy, cx = best
        if (cy, cx) == start == own:
            return None                                   # already at the bottom: use the bearing
        return np.array([(cx + 0.5) * FIELD_CELL, (cy + 0.5) * FIELD_CELL])

    # ------------------------------------------------------------------ API
    def observe(self, sim, obs) -> np.ndarray:
        """(n, d) policy observation of this step -> (n, d + 2); see the module invariants."""
        if self.started:
            self._odometry(sim)
        self.started = True
        obs = np.asarray(obs, np.float32)
        n, d = obs.shape
        bcol = d - 2                                      # [..., bearing, speed]
        out = np.zeros((n, d + EXTRA), np.float32)
        out[:, :d] = obs
        update = sim.steps % self.every == 0
        for i in np.flatnonzero(sim.active):
            self.insert(i, obs[i])
            if update or not self.computed[i]:
                self.update_field(i, sim)
                self.computed[i] = True
            if len(self.goals[i]) > 1:
                out[i, bcol] = bearing_to(self.goals[i][self.target[i]], sim.pos[i],
                                          sim.heading[i])
                if self.exit_rule == "orbit" and self._orbit_check(i, sim, float(out[i, bcol])):
                    out[i, bcol] = bearing_to(self.goals[i][self.target[i]], sim.pos[i],
                                              sim.heading[i])
            out[i, d:] = self.downhill(i, float(out[i, bcol]))
        return out
