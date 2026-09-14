"""Multi-agent steering simulation: sensing, dynamics, reward and statistics.

Observation per agent (n_sectors * 3 + 3 numbers, all normalised to [0, 1] or [-1, 1]):

    [ 0:16]  nearest obstacle distance per FOV sector
    [16:32]  same, for walls
    [32:48]  same, for other agents
    [48]     distance to goal region
    [49]     bearing to goal, relative to heading
    [50]     current speed

Agents step synchronously: all actions are computed from one world state,
then applied together.
"""
from __future__ import annotations

import math

import numpy as np

from config import Config, E_AGENT, E_OBSTACLE, E_WALL, N_ENTITY_CLASSES

INF = np.inf


def _ray_circles(p, dirs, centers, radii):
    """(S,) nearest hit distance for S rays against M circles."""
    if len(centers) == 0:
        return np.full(len(dirs), INF)
    f = p[None, :] - centers                       # (M,2)
    b = dirs @ f.T                                 # (S,M)
    c = (f * f).sum(1) - radii * radii             # (M,)
    disc = b * b - c[None, :]
    ok = disc >= 0.0
    s = np.sqrt(np.where(ok, disc, 0.0))
    t1, t2 = -b - s, -b + s
    t = np.where(t1 > 1e-9, t1, np.where(t2 > 1e-9, t2, INF))
    return np.where(ok, t, INF).min(1)


def _ray_segments(p, dirs, a, e):
    """(S,) nearest hit distance for S rays against K segments A + u*E."""
    if len(a) == 0:
        return np.full(len(dirs), INF)
    dx, dy = dirs[:, 0:1], dirs[:, 1:2]            # (S,1)
    ex, ey = e[None, :, 0], e[None, :, 1]          # (1,K)
    det = ex * dy - ey * dx                        # (S,K)
    w = a - p[None, :]                             # (K,2)
    wx, wy = w[None, :, 0], w[None, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ex * wy - ey * wx) / det
        u = (dx * wy - dy * wx) / det
    good = (np.abs(det) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    return np.where(good, t, INF).min(1)


class Simulator:
    """Steps every agent in a Scene.  Holds all mutable state; Scene stays pure."""

    def __init__(self, scene, cfg=None):
        self.scene = scene
        self.cfg = cfg or Config()
        c = self.cfg

        half = math.radians(c.fov_deg) / 2.0
        step = math.radians(c.fov_deg) / c.n_sectors
        self._sector_off = np.array(
            [-half + (k + 0.5) * step for k in range(c.n_sectors)])

        self._obs_c = np.array([o.c for o in scene.obstacles], float).reshape(-1, 2)
        self._obs_r = np.array([o.r for o in scene.obstacles], float).reshape(-1)

        # walls as capsules: two offset lines plus two round caps
        segs, caps_c = [], []
        for (ax, ay), (bx, by) in scene.walls:
            ex, ey = bx - ax, by - ay
            ll = math.hypot(ex, ey)
            if ll < 1e-12:
                caps_c.append((ax, ay))
                continue
            nx, ny = -ey / ll * c.wall_half, ex / ll * c.wall_half
            for s in (1.0, -1.0):
                segs.append(((ax + s * nx, ay + s * ny), (bx + s * nx, by + s * ny)))
            caps_c += [(ax, ay), (bx, by)]
        self._seg_a = np.array([s[0] for s in segs], float).reshape(-1, 2)
        self._seg_e = np.array([(s[1][0] - s[0][0], s[1][1] - s[0][1]) for s in segs],
                               float).reshape(-1, 2)
        self._cap_c = np.array(caps_c, float).reshape(-1, 2)
        self._cap_r = np.full(len(self._cap_c), c.wall_half)
        self._wall_lines = [(np.array(s[0]), np.array(s[1])) for s in scene.walls]

        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        n = len(self.scene.agents)
        self.pos = np.array([a.pos for a in self.scene.agents], float).reshape(n, 2)
        self.heading = np.array([a.heading for a in self.scene.agents], float)
        self.speed = np.zeros(n)
        self.active = np.ones(n, bool)
        self.outcome = [None] * n
        self.collisions = np.zeros(n, int)      # running violation count
        self.finish_step = np.full(n, -1, int)
        self.trajectories = [[tuple(p)] for p in self.pos]
        self.steps = 0
        return self.observe()

    @property
    def done(self) -> bool:
        return not self.active.any() or self.steps >= self.cfg.max_steps

    def _goal(self, i):
        return self.scene.goal_for(self.scene.agents[i])

    # -- sensing ----------------------------------------------------------
    def _sector_dirs(self, i):
        ang = self.heading[i] + self._sector_off
        return np.stack([np.cos(ang), np.sin(ang)], 1)

    def observe_one(self, i) -> np.ndarray:
        c = self.cfg
        p = self.pos[i]
        dirs = self._sector_dirs(i)
        out = np.empty(c.n_sectors * N_ENTITY_CLASSES + 3, np.float32)

        d_obs = _ray_circles(p, dirs, self._obs_c, self._obs_r)
        d_wall = np.minimum(_ray_segments(p, dirs, self._seg_a, self._seg_e),
                            _ray_circles(p, dirs, self._cap_c, self._cap_r))
        others = np.array([j for j in range(len(self.pos)) if j != i and self.active[j]],
                          int)
        if len(others):
            d_ag = _ray_circles(p, dirs, self.pos[others],
                                np.full(len(others), c.agent_radius))
        else:
            d_ag = np.full(c.n_sectors, INF)

        S = c.n_sectors
        for k, d in ((E_OBSTACLE, d_obs), (E_WALL, d_wall), (E_AGENT, d_ag)):
            out[k * S:(k + 1) * S] = np.minimum(d, c.fov_range) / c.fov_range

        gx, gy = self._goal(i).nearest_point(tuple(p))
        dist = math.hypot(gx - p[0], gy - p[1])
        rel = math.atan2(gy - p[1], gx - p[0]) - self.heading[i]
        rel = (rel + math.pi) % (2 * math.pi) - math.pi
        out[-3] = min(dist, c.world * 1.5) / (c.world * 1.5)
        out[-2] = rel / math.pi
        out[-1] = self.speed[i] / c.max_speed
        return out

    def observe(self) -> np.ndarray:
        return np.stack([self.observe_one(i) for i in range(len(self.pos))])

    # -- collision --------------------------------------------------------
    def _hits_static(self, i) -> bool:
        c = self.cfg
        p = self.pos[i]
        for j in range(len(self._obs_c)):
            if np.hypot(*(p - self._obs_c[j])) < self._obs_r[j] + c.agent_radius:
                return True
        for a, b in self._wall_lines:
            e = b - a
            ll = float(e @ e)
            t = 0.0 if ll < 1e-12 else float(np.clip(((p - a) @ e) / ll, 0.0, 1.0))
            if np.hypot(*(p - (a + t * e))) < c.agent_radius + c.wall_half:
                return True
        return False

    def _hits_agent(self, i) -> bool:
        r2 = 2 * self.cfg.agent_radius
        for j in range(len(self.pos)):
            if j != i and self.active[j] and np.hypot(*(self.pos[i] - self.pos[j])) < r2:
                return True
        return False

    # -- dynamics ---------------------------------------------------------
    def step(self, actions):
        """actions: (n, 2) array of (turn, speed), both normalised.

        turn in [-1, 1] scales max_turn_deg; speed in [0, 1] scales max_speed.
        Every agent is advanced from the same pre-step world state.
        """
        c = self.cfg
        actions = np.asarray(actions, float).reshape(-1, 2)
        prev_d = np.array([self._goal(i).distance_to(tuple(self.pos[i]))
                           for i in range(len(self.pos))])

        moving = self.active.copy()
        self.heading = np.where(
            moving, self.heading + np.clip(actions[:, 0], -1, 1) * c.max_turn, self.heading)
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        self.speed = np.where(moving, np.clip(actions[:, 1], 0, 1) * c.max_speed, 0.0)
        self.pos = self.pos + (moving * self.speed)[:, None] * np.stack(
            [np.cos(self.heading), np.sin(self.heading)], 1)
        self.steps += 1

        rewards = np.zeros(len(self.pos))
        for i in range(len(self.pos)):
            if not moving[i]:
                continue
            self.trajectories[i].append(tuple(self.pos[i]))
            d = self._goal(i).distance_to(tuple(self.pos[i]))
            rewards[i] = -c.step_cost + c.progress_weight * (prev_d[i] - d)

            static = self._hits_static(i)
            peer = self._hits_agent(i)
            if static or peer:
                self.collisions[i] += 1
                rewards[i] -= c.collision_penalty if static else c.agent_collision_penalty
                if c.terminate_on_collision:
                    self.active[i] = False
                    self.outcome[i] = "collision"
                    continue
            if self._goal(i).contains(tuple(self.pos[i])):
                rewards[i] += c.goal_bonus
                self.active[i] = False
                self.outcome[i] = "goal"
                self.finish_step[i] = self.steps

        if self.steps >= c.max_steps:
            for i in range(len(self.pos)):
                if self.active[i]:
                    self.active[i] = False
                    self.outcome[i] = "timeout"

        return self.observe(), rewards, self.done

    # -- reporting --------------------------------------------------------
    def stats(self) -> dict:
        """Both metric families: RL success rates and evacuation counters."""
        n = len(self.pos)
        reached = [i for i in range(n) if self.outcome[i] == "goal"]
        lengths = [sum(math.dist(a, b) for a, b in zip(t, t[1:]))
                   for t in self.trajectories]
        return {
            "n_agents": n,
            "success_rate": len(reached) / n if n else 0.0,
            "collision_rate": sum(o == "collision" for o in self.outcome) / n if n else 0.0,
            "timeout_rate": sum(o == "timeout" for o in self.outcome) / n if n else 0.0,
            "collision_violations": int(self.collisions.sum()),
            "finish_time": int(self.finish_step[reached].max()) if reached else -1,
            "remaining_agents": int(n - len(reached)),
            "mean_path_length": float(np.mean(lengths)) if lengths else 0.0,
            "steps": self.steps,
        }


def greedy_action(sim, i):
    """Turn toward the goal at full speed -- the null model, for calibration."""
    obs = sim.observe_one(i)
    return np.array([np.clip(obs[-2] * math.pi / sim.cfg.max_turn, -1, 1), 1.0])
