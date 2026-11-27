"""Scenes -> fixed-size arrays for ``jaxsim.sim`` (numpy only, runs on CPU).

    python jaxsim/bank.py --n 50000 --out data/jaxbank/train.npz --workers 9

A level holds at most ``N`` agents, ``O`` obstacles, ``W`` walls, ``F``
distinct goals of ``G`` regions each (training scenes: 8 / 6 / 23 / 2 / 3).
Padding never interacts with anything:

- missing agents: ``valid`` False (never active, never seen);
- missing obstacles / caps / wall lines: far away (``FAR``), radius 0;
- missing ray segments: zero direction (determinant 0: never hit);
- missing regions of a goal: copies of its first region (min / any unchanged);
- missing goals: copies of goal 0 (no agent points at them).

``geo`` holds the progress reward's geodesic field per goal, exactly as
``physics.Simulator`` builds it (``progress_metric = "geodesic"``).
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

N, O, W, F, G = 8, 8, 32, 2, 4
FAR = 1.0e4


def pack(scene, cfg, geo=True):
    """One Scene -> dict of numpy arrays (see module doc)."""
    from scene import members
    ag = scene.agents
    if not 1 <= len(ag) <= N or len(scene.obstacles) > O or len(scene.walls) > W:
        raise ValueError(f"scene too large: {len(ag)} agents, {len(scene.obstacles)} "
                         f"obstacles, {len(scene.walls)} walls")
    d = {}
    d["valid"] = np.zeros(N, bool)
    d["valid"][:len(ag)] = True
    d["pos0"] = np.zeros((N, 2))
    d["pos0"][:len(ag)] = [a.pos for a in ag]
    d["pos0"][len(ag):] = FAR                       # never near anything
    d["head0"] = np.zeros(N)
    d["head0"][:len(ag)] = [a.heading for a in ag]

    d["obs_c"] = np.full((O, 2), FAR)
    d["obs_r"] = np.zeros(O)
    for k, o in enumerate(scene.obstacles):
        d["obs_c"][k], d["obs_r"][k] = o.c, o.r

    hw = cfg.wall_half
    d["seg_a"] = np.zeros((2 * W, 2))
    d["seg_e"] = np.zeros((2 * W, 2))
    d["cap_c"] = np.full((2 * W, 2), FAR)
    d["wall_a"] = np.full((W, 2), FAR)
    d["wall_b"] = np.full((W, 2), FAR)
    ns = nc = 0
    for k, ((ax, ay), (bx, by)) in enumerate(scene.walls):
        d["wall_a"][k], d["wall_b"][k] = (ax, ay), (bx, by)
        ex, ey = bx - ax, by - ay
        ll = math.hypot(ex, ey)
        if ll < 1e-12:
            d["cap_c"][nc] = (ax, ay)
            nc += 1
            continue
        nx, ny = -ey / ll * hw, ex / ll * hw
        for s in (1.0, -1.0):
            d["seg_a"][ns] = (ax + s * nx, ay + s * ny)
            d["seg_e"][ns] = (ex, ey)
            ns += 1
        d["cap_c"][nc], d["cap_c"][nc + 1] = (ax, ay), (bx, by)
        nc += 2

    goals = []
    for a in ag:
        g = scene.goal_for(a)
        if g not in goals:
            goals.append(g)
    if len(goals) > F:
        raise ValueError(f"{len(goals)} distinct goals > {F}")
    d["goal_id"] = np.zeros(N, np.int32)
    for i, a in enumerate(ag):
        d["goal_id"][i] = goals.index(scene.goal_for(a))
    d["rects"] = np.zeros((F, G, 4))
    for f in range(F):
        rs = members(goals[min(f, len(goals) - 1)])
        if len(rs) > G:
            raise ValueError(f"goal with {len(rs)} regions > {G}")
        for k in range(G):
            r = rs[min(k, len(rs) - 1)]
            d["rects"][f, k] = (r.x0, r.y0, r.x1, r.y1)
    if geo:
        from reachability import clearance, geodesic_field, occupancy
        clear = clearance(cfg, occupancy(cfg, scene))
        fl = [geodesic_field(cfg, scene, g, cell=cfg.geo_cell, clear=clear) for g in goals]
        fl += [fl[0]] * (F - len(fl))
        d["geo"] = np.stack(fl).astype(np.float32)
    return d


def stack(levels):
    return {k: np.stack([lv[k] for lv in levels]) for k in levels[0]}


def _build(args):
    seed, n = args
    from config import Config
    from env import sample_scene
    from policyconfig import EnvConfig
    rng = np.random.default_rng(seed)
    ec, cfg = EnvConfig(), Config()
    out, scenes = [], []
    while len(out) < n:
        sc = sample_scene(ec, cfg, rng)
        try:
            out.append(pack(sc, cfg))
        except ValueError:
            continue
        scenes.append(sc.layout)
    return out, scenes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=POLICY / "data" / "jaxbank" / "train.npz")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    import multiprocessing as mp
    k = max(1, a.workers) * 4
    jobs = [((a.seed, j), a.n // k + (j < a.n % k)) for j in range(k)]
    if a.workers:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            parts = pool.map(_build, jobs)
    else:
        parts = [_build(j) for j in jobs]
    levels = [lv for p, _ in parts for lv in p]
    layouts = np.array([s for _, p in parts for s in p])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, layout=layouts, **stack(levels))
    print(f"{len(levels)} levels -> {a.out}")


if __name__ == "__main__":
    main()
