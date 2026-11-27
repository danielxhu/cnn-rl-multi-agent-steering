"""Pure-JAX port of ``sim/physics.py`` on one packed level (``jaxsim.bank``).

    lv = {k: jnp.asarray(v) for k, v in bank.pack(scene, cfg).items()}
    st = reset(lv)
    obs = observe(P, lv, st)                       # (N, 51), as Simulator.observe
    st, obs, rew = step(P, lv, st, actions)        # actions (N, 2): turn, speed

``P = params(cfg)`` holds the Config numbers as Python constants (static).
Same order of operations as ``Simulator.step``: progress distances before
the move, synchronous move, swept agent-agent test, static test, goal test,
timeout. Padded agents are never active. ``outcome``: 0 running, 1 goal,
2 collision, 3 timeout. Batch with ``jax.vmap(step, (None, 0, 0, 0))``.

Precision: float32 by default; ``jax.config.update("jax_enable_x64", True)``
reproduces the numpy simulator to rounding (``port_check.py``).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

RUNNING, GOAL, COLLISION, TIMEOUT = 0, 1, 2, 3


class Params(NamedTuple):
    R: float
    wall_half: float
    fov_range: float
    n_sectors: int
    sector_off: tuple
    max_turn: float
    max_speed: float
    max_steps: int
    world: float
    goal_bonus: float
    collision_penalty: float
    agent_collision_penalty: float
    step_cost: float
    progress_weight: float
    timeout_penalty: float
    geodesic: bool
    geo_cell: float


def params(cfg) -> Params:
    if not cfg.terminate_on_collision:
        raise ValueError("jaxsim implements terminate_on_collision = True only")
    if cfg.progress_metric not in ("euclid", "geodesic"):
        raise ValueError(cfg.progress_metric)
    half = math.radians(cfg.fov_deg) / 2.0
    stp = math.radians(cfg.fov_deg) / cfg.n_sectors
    K = max(1, int(cfg.rays_per_sector))
    off = tuple(-half + (k + (j + 0.5) / K) * stp for k in range(cfg.n_sectors) for j in range(K))
    return Params(cfg.agent_radius, cfg.wall_half, cfg.fov_range, cfg.n_sectors, off,
                  cfg.max_turn, cfg.max_speed, cfg.max_steps, cfg.world, cfg.goal_bonus,
                  cfg.collision_penalty, cfg.agent_collision_penalty, cfg.step_cost,
                  cfg.progress_weight, cfg.timeout_penalty, cfg.progress_metric == "geodesic",
                  cfg.geo_cell)


class State(NamedTuple):
    pos: jax.Array        # (N, 2)
    heading: jax.Array    # (N,)
    speed: jax.Array      # (N,)
    active: jax.Array     # (N,) bool
    outcome: jax.Array    # (N,) int32
    finish: jax.Array     # (N,) int32, -1 until the goal
    steps: jax.Array      # () int32


def reset(lv) -> State:
    n = lv["valid"].shape[0]
    f = lv["pos0"].dtype
    return State(lv["pos0"], lv["head0"], jnp.zeros(n, f), lv["valid"],
                 jnp.zeros(n, jnp.int32), jnp.full(n, -1, jnp.int32), jnp.int32(0))


# ------------------------------------------------------------------ geometry
def _ray_circles(p, dirs, centers, radii):
    """(S,) nearest hit of S rays from p against M circles (inf if none)."""
    f = p[None, :] - centers                                   # (M, 2)
    b = dirs @ f.T                                             # (S, M)
    c = (f * f).sum(1) - radii * radii
    disc = b * b - c[None, :]
    ok = disc >= 0.0
    s = jnp.sqrt(jnp.where(ok, disc, 0.0))
    t1, t2 = -b - s, -b + s
    t = jnp.where(t1 > 1e-9, t1, jnp.where(t2 > 1e-9, t2, jnp.inf))
    return jnp.where(ok, t, jnp.inf).min(1)


def _ray_segments(p, dirs, a, e):
    dx, dy = dirs[:, 0:1], dirs[:, 1:2]
    ex, ey = e[None, :, 0], e[None, :, 1]
    det = ex * dy - ey * dx
    w = a - p[None, :]
    wx, wy = w[None, :, 0], w[None, :, 1]
    good_det = jnp.abs(det) > 1e-12
    sd = jnp.where(good_det, det, 1.0)
    t = (ex * wy - ey * wx) / sd
    u = (dx * wy - dy * wx) / sd
    good = good_det & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    return jnp.where(good, t, jnp.inf).min(1)


def _nearest_point(rects, p):
    """Nearest point of a RegionSet (G, 4) to p; the first nearest member wins."""
    q = jnp.stack([jnp.clip(p[0], rects[:, 0], rects[:, 2]),
                   jnp.clip(p[1], rects[:, 1], rects[:, 3])], 1)
    d = jnp.hypot(p[0] - q[:, 0], p[1] - q[:, 1])
    return q[jnp.argmin(d)], d.min()


def _contains(rects, p):
    return ((rects[:, 0] <= p[0]) & (p[0] <= rects[:, 2]) &
            (rects[:, 1] <= p[1]) & (p[1] <= rects[:, 3])).any()


def _geo_dist(P, look, m, p, euclid):
    """``reachability.geodesic_distance``: bilinear over finite corners, else the
    best of the 5 x 5 neighbourhood plus the straight line, else ``euclid``.
    ``look(iy, ix)``: the field value (indices already clipped to [0, m))."""
    cell = P.geo_cell
    u, v = p[0] / cell - 0.5, p[1] / cell - 0.5
    x0, y0 = jnp.floor(u), jnp.floor(v)
    fx, fy = u - x0, v - y0
    x0, y0 = x0.astype(jnp.int32), y0.astype(jnp.int32)
    num = den = 0.0
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            iy, ix = y0 + dy, x0 + dx
            inb = (iy >= 0) & (iy < m) & (ix >= 0) & (ix < m)
            val = look(jnp.clip(iy, 0, m - 1), jnp.clip(ix, 0, m - 1))
            w = wy * wx
            use = inb & jnp.isfinite(val) & (w > 0.0)
            num = num + jnp.where(use, w * jnp.where(use, val, 0.0), 0.0)
            den = den + jnp.where(use, w, 0.0)
    bil = num / jnp.where(den > 1e-9, den, 1.0)
    cy, cx = jnp.round(v).astype(jnp.int32), jnp.round(u).astype(jnp.int32)
    best = jnp.inf
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            iy, ix = cy + oy, cx + ox
            inb = (iy >= 0) & (iy < m) & (ix >= 0) & (ix < m)
            val = look(jnp.clip(iy, 0, m - 1), jnp.clip(ix, 0, m - 1))
            dd = val + jnp.hypot(p[0] - (ix + 0.5) * cell, p[1] - (iy + 0.5) * cell)
            best = jnp.minimum(best, jnp.where(inb & jnp.isfinite(val), dd, jnp.inf))
    return jnp.where(den > 1e-9, bil, jnp.where(jnp.isfinite(best), best, euclid))


def _progress_dist(P, lv, pos, geo_bank=None):
    """(N,) distance of the progress reward (straight line or geodesic).

    The fields are ``lv["geo"]`` (F, m, m), or with ``geo_bank`` (levels, F,
    m, m) row ``lv["level"]`` of it, read cell by cell (no per-env copy)."""
    rects = lv["rects"][lv["goal_id"]]                              # (N, G, 4)
    _, eu = jax.vmap(_nearest_point)(rects, pos)
    if not P.geodesic:
        return eu
    if geo_bank is None:
        m = lv["geo"].shape[-1]
        look = lambda g: (lambda iy, ix: lv["geo"][g, iy, ix])     # noqa: E731
    else:
        m = geo_bank.shape[-1]
        look = lambda g: (lambda iy, ix: geo_bank[lv["level"], g, iy, ix])  # noqa: E731
    return jax.vmap(lambda g, p, e: _geo_dist(P, look(g), m, p, e))(lv["goal_id"], pos, eu)


# ------------------------------------------------------------------ sensing
def observe(P, lv, st):
    """(N, 51) observations, as ``Simulator.observe`` (padded rows: garbage)."""
    S = P.n_sectors
    off = jnp.asarray(P.sector_off, st.pos.dtype)
    n = st.pos.shape[0]
    rad = jnp.full(n, P.R, st.pos.dtype)
    caps_r = jnp.full(lv["cap_c"].shape[0], P.wall_half, st.pos.dtype)

    def one(i):
        p = st.pos[i]
        ang = st.heading[i] + off
        dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], 1)
        d_obs = _ray_circles(p, dirs, lv["obs_c"], lv["obs_r"])
        d_wall = jnp.minimum(_ray_segments(p, dirs, lv["seg_a"], lv["seg_e"]),
                             _ray_circles(p, dirs, lv["cap_c"], caps_r))
        others = st.active & (jnp.arange(n) != i)
        d_ag = _ray_circles(p, dirs, jnp.where(others[:, None], st.pos, 1.0e4), rad)
        rows = [d_obs, d_wall, d_ag]
        if len(P.sector_off) != S:
            rows = [d.reshape(S, -1).min(1) for d in rows]
        rays = jnp.concatenate([jnp.minimum(d, P.fov_range) / P.fov_range for d in rows])
        q, dist = _nearest_point(lv["rects"][lv["goal_id"][i]], p)
        rel = jnp.arctan2(q[1] - p[1], q[0] - p[0]) - st.heading[i]
        rel = jnp.mod(rel + math.pi, 2 * math.pi) - math.pi
        tail = jnp.stack([jnp.minimum(dist, P.world * 1.5) / (P.world * 1.5), rel / math.pi,
                          st.speed[i] / P.max_speed])
        return jnp.concatenate([rays, tail])

    return jax.vmap(one)(jnp.arange(n))


# ---------------------------------------------------------------- collision
def _hits_static(P, lv, p):
    ob = (jnp.hypot(*(p[None, :] - lv["obs_c"]).T) < lv["obs_r"] + P.R).any()
    a, b = lv["wall_a"], lv["wall_b"]
    e = b - a
    ll = (e * e).sum(1)
    t = jnp.where(ll < 1e-12, 0.0,
                  jnp.clip(((p[None, :] - a) * e).sum(1) / jnp.where(ll < 1e-12, 1.0, ll), 0.0, 1.0))
    q = a + t[:, None] * e
    wl = (jnp.hypot(*(p[None, :] - q).T) < P.R + P.wall_half).any()
    return ob | wl


def _peer_hits(P, prev, pos, moving):
    d0 = prev[:, None, :] - prev[None, :, :]
    dv = (pos[:, None, :] - pos[None, :, :]) - d0
    vv = (dv * dv).sum(-1)
    t = jnp.where(vv > 1e-12, jnp.clip(-(d0 * dv).sum(-1) / jnp.maximum(vv, 1e-12), 0.0, 1.0), 0.0)
    dmin = jnp.linalg.norm(d0 + t[..., None] * dv, axis=-1)
    pair = moving[:, None] & moving[None, :] & ~jnp.eye(len(moving), dtype=bool)
    return moving & (pair & (dmin < 2 * P.R)).any(1)


# ----------------------------------------------------------------- dynamics
def step(P, lv, st, actions, geo_bank=None):
    """One synchronous step -> (state, obs (N, 51), rewards (N,))."""
    actions = actions.astype(st.pos.dtype)
    prev_d = _progress_dist(P, lv, st.pos, geo_bank)
    moving = st.active
    head = jnp.where(moving, st.heading + jnp.clip(actions[:, 0], -1, 1) * P.max_turn, st.heading)
    head = jnp.mod(head + math.pi, 2 * math.pi) - math.pi
    speed = jnp.where(moving, jnp.clip(actions[:, 1], 0, 1) * P.max_speed, 0.0)
    pos = st.pos + (moving * speed)[:, None] * jnp.stack([jnp.cos(head), jnp.sin(head)], 1)
    steps = st.steps + 1

    peer = _peer_hits(P, st.pos, pos, moving)
    d = _progress_dist(P, lv, pos, geo_bank)
    rew = -P.step_cost + P.progress_weight * (prev_d - d)
    static = jax.vmap(lambda p: _hits_static(P, lv, p))(pos)
    hit = moving & (static | peer)
    rew = rew - jnp.where(hit, jnp.where(static, P.collision_penalty, P.agent_collision_penalty), 0.0)
    rects = lv["rects"][lv["goal_id"]]
    inside = jax.vmap(_contains)(rects, pos)
    goal = moving & ~hit & inside
    rew = rew + jnp.where(goal, P.goal_bonus, 0.0)
    outcome = jnp.where(hit, COLLISION, jnp.where(goal, GOAL, st.outcome))
    finish = jnp.where(goal, steps, st.finish)
    active = moving & ~hit & ~goal
    tout = active & (steps >= P.max_steps)
    rew = rew - jnp.where(tout, P.timeout_penalty, 0.0)
    outcome = jnp.where(tout, TIMEOUT, outcome)
    active = active & ~tout
    rew = jnp.where(moving, rew, 0.0)
    st = State(pos, head, speed, active, outcome, finish, steps)
    return st, observe(P, lv, st), rew


def done(st):
    return ~st.active.any()
