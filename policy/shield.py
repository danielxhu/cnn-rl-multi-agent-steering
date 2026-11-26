"""Safety filter after the policy network (fixed, not learned).

    python evaluate.py --ckpt models/e7_map_seed0/ckpt_best.pt --shield 0.3 --out ...

After the network picks (turn, speed) for an agent, the swept step is checked
against what that agent's rays sensed THIS step: obstacle and wall hit points,
points on the segments joining adjacent wall/obstacle hits closer than
``min_bottleneck`` (the wallmap rule: no legal passage between them), and agent
hit points. If the step comes within ``agent_radius + margin`` of any of them,
it is replaced by the safe step whose end point is nearest the wanted one:
heading offsets 0, ±5, ..., ±25° (within ``max_turn``) x speeds 1, 0.75, 0.5,
0.25 of the requested speed. The margin is kept only while some step keeps it: if none does, the
search is repeated with clearance ``agent_radius + SQUEEZE``, so an agent can
pass a gap narrower than 2 (R + margin) but wider than 2 R; an agent already
closer than R + SQUEEZE must only not get closer. Under the 21 Nov rule
("not closer" once inside R + margin, no second search) agents stood in front
of or inside such gaps between round obstacles until timeout. If no step is
safe even then:

- ``stuck="rotate"`` (default since 25 Nov 2026): turn in place by the full
  ``max_turn`` towards the side whose rays see more free space, and keep that
  side until a safe step exists again. Under ``"hold"`` the network kept
  asking to go straight into the wall and the agent stood there until timeout
  (34 of 64 e7_map seed-0 test4 timeouts, all blocked by static surfaces).
- ``stuck="hold"``: turn as the network asks, at speed 0 (the 21 Nov rule;
  reproduces DESIGN_E7 §13).

Invariants:
- Uses the agent's own observation and pose only (as the wall memory does);
  no map, no other agent's state.
- Off unless asked for (``evaluate.py --shield``, or a checkpoint trained
  with ``EnvConfig.shield >= 0``, which always runs with its filter): every
  earlier result is unchanged. Wrapping never changes the inner controller's state or actions
  that are already safe.
- Origin: fork session of 20–21 Nov 2026 (e7_map test4 0.925 -> 0.964 at
  margin 0.3 with agents included).
"""
from __future__ import annotations

import math

import numpy as np

from config import E_AGENT, E_OBSTACLE, E_WALL

TURNS = np.radians([0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25])   # offsets from the wanted heading
SPEEDS = (1.0, 0.75, 0.5, 0.25)                                       # fractions of the wanted speed
SQUEEZE = 0.05                                                        # clearance kept once inside R + margin
JOIN_POINTS = np.linspace(0, 1, 8)[1:-1, None]                        # interior points of a joining segment


class Shield:
    """Controller wrapper: ``Shield(inner, sim_cfg, margin)``; same reset/act interface.

    ``inner=None`` + ``begin``/``filter``: the training environment's use
    (``EnvConfig.shield``), where the actions come from the rollout."""

    def __init__(self, inner, sim_cfg, margin=0.3, agents=True, stuck="rotate"):
        if margin < 0:
            raise ValueError("margin must be >= 0")
        if stuck not in ("rotate", "hold"):
            raise ValueError(f"stuck must be 'rotate' or 'hold', not {stuck!r}")
        self.inner, self.c, self.margin, self.agents = inner, sim_cfg, float(margin), agents
        self.stuck = stuck
        self.name = f"{getattr(inner, 'name', 'policy')}+shield{margin:g}"
        self.n_mod = self.n_steps = 0

    def __getattr__(self, k):                 # exit_switches etc. of the inner controller
        if k == "inner":
            raise AttributeError(k)
        return getattr(self.inner, k)

    def reset(self, sim):
        self.inner.reset(sim)
        self.begin(sim)

    def begin(self, sim):
        """Per-scene state; ``reset`` without an inner controller (training env)."""
        self.off = np.asarray(sim._sector_off).reshape(self.c.n_sectors, -1).mean(1)
        self.escape = np.zeros(len(sim.pos))           # rotate: committed turn side, 0 = none

    def _free_side(self, o):
        """+1 (left) or -1 (right): the side whose wall/obstacle rays see further."""
        S = self.c.n_sectors
        r = np.minimum(np.asarray(o[E_OBSTACLE * S:(E_OBSTACLE + 1) * S], float),
                       np.asarray(o[E_WALL * S:(E_WALL + 1) * S], float))
        rel = (self.off + math.pi) % (2 * math.pi) - math.pi
        return 1.0 if r[rel > 0].sum() >= r[rel < 0].sum() else -1.0

    def sensed(self, p, h, o):
        """World-frame points the agent must keep clear of, from its observation row ``o``."""
        S, c = self.c.n_sectors, self.c
        ang = h + self.off
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        pts = []
        for k in (E_OBSTACLE, E_WALL) + ((E_AGENT,) if self.agents else ()):
            r = np.asarray(o[k * S:(k + 1) * S], float)
            hit = r < 1.0 - 1e-6
            q = p[None, :] + (r * c.fov_range)[:, None] * dirs
            pts.append(q[hit])
            if k == E_AGENT:
                continue
            for j in range(S - 1):
                if hit[j] and hit[j + 1] and math.dist(q[j], q[j + 1]) < c.min_bottleneck:
                    pts.append(q[j] + JOIN_POINTS * (q[j + 1] - q[j]))
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    @staticmethod
    def clear_of(p, q, pts, need):
        """The swept segment p->q stays at least ``need`` from every point."""
        if len(pts) == 0:
            return True
        e = q - p
        ll = float(e @ e)
        t = np.clip(((pts - p) @ e) / ll, 0, 1) if ll > 1e-12 else np.zeros(len(pts))
        return bool(np.hypot(*(pts - (p + t[:, None] * e)).T).min() >= need)

    def _search(self, p, h, want_h, sp, want, pts, nd):
        """The safe (turn, speed) nearest ``want`` at clearance ``nd``; None if none.
        The wanted step itself first (``"ok"``)."""
        c = self.c
        if self.clear_of(p, want, pts, nd):
            return "ok"
        best, bd = None, np.inf
        for f in SPEEDS:
            for dh in TURNS:
                hh = want_h + dh
                if abs((hh - h + math.pi) % (2 * math.pi) - math.pi) > c.max_turn + 1e-9:
                    continue
                q = p + f * sp * c.max_speed * np.array([math.cos(hh), math.sin(hh)])
                if self.clear_of(p, q, pts, nd):
                    d = math.dist(q, want)
                    if d < bd:
                        best, bd = ((hh - h) / c.max_turn, f * sp), d
        return best

    def act(self, sim, obs51):
        return self.filter(sim, obs51, self.inner.act(sim, obs51))[0]

    def filter(self, sim, obs51, actions):
        """Simulator actions (n, 2) -> (filtered copy, bool mask of edited agents)."""
        a = np.array(actions, float)
        edited = np.zeros(len(a), bool)
        c = self.c
        need = c.agent_radius + self.margin
        floor = c.agent_radius + SQUEEZE
        for i in np.flatnonzero(sim.active):
            self.n_steps += 1
            p, h = sim.pos[i].astype(float), float(sim.heading[i])
            pts = self.sensed(p, h, obs51[i])
            turn, sp = float(np.clip(a[i, 0], -1, 1)), float(np.clip(a[i, 1], 0, 1))
            want_h = h + turn * c.max_turn
            want = p + sp * c.max_speed * np.array([math.cos(want_h), math.sin(want_h)])
            cur = np.hypot(*(pts - p).T).min() if len(pts) else np.inf
            if self.stuck == "hold":                       # the 21 Nov rule
                levels = [min(need, cur - 1e-6)]
            else:                                          # margin first, then squeeze
                levels = [need, floor] if cur >= floor else [cur - 1e-6]
            res = None
            for nd in levels:
                res = self._search(p, h, want_h, sp, want, pts, nd)
                if res is not None:
                    break
            if res == "ok":
                self.escape[i] = 0
                continue
            self.n_mod += 1
            edited[i] = True
            if res is not None:
                self.escape[i] = 0
                a[i] = res
            elif self.stuck == "hold":
                a[i] = (turn, 0.0)
            else:
                if self.escape[i] == 0:
                    self.escape[i] = self._free_side(obs51[i])
                a[i] = (self.escape[i], 0.0)
        return a, edited
