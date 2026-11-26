"""Do agents loop where the goal bearing disagrees with the way to the goal?

    python scripts/loop_bearing.py --model models/e7_map_seed0/ckpt_best.pt --suite data/suites/test4 --out results/loop_bearing/e7_map_seed0.json

Per step of every agent: ``mismatch`` = angle between the observed goal
bearing (straight line to the goal's nearest point, obs[49]) and the geodesic
direction (steepest descent of the simulator's geodesic field over 16
directions at 2 cells, about 11 deg resolution). Per agent: the loop flag of
``evaluate.turning``, and for loop agents the loop window, the shortest stretch
whose cumulative heading change spans 360 deg (the first one).

Prints and writes: mismatch > 45 / 90 deg share of steps inside loop windows,
outside them on loop agents, and on agents that never loop; loop rate by the
largest mismatch an agent met.

Invariants: read-only on models and suites; the policy's own episodes (no
filter, deterministic), as ``evaluate.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

RING = 2


def descent_dir(field):
    """(m, m) steepest-descent angle (rad) of a geodesic field, NaN where none."""
    m = field.shape[0]
    F = np.where(np.isfinite(field), field, np.inf).astype(float)
    offs = sorted({(int(round(RING * math.sin(a))), int(round(RING * math.cos(a))))
                   for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)})
    best = np.full((m, m), np.inf)
    ang = np.full((m, m), np.nan)
    P = RING
    Fp = np.full((m + 2 * P, m + 2 * P), np.inf)
    Fp[P:P + m, P:P + m] = F
    for dy, dx in offs:
        nb = Fp[P + dy:P + dy + m, P + dx:P + dx + m]
        slope = (nb - F) / math.hypot(dy, dx)
        better = np.isfinite(slope) & (slope < best)
        best = np.where(better, slope, best)
        ang = np.where(better, math.atan2(dy, dx), ang)
    ang[~(best < 0)] = np.nan                        # at the goal or no finite neighbour
    return ang


def loop_window(traj, min_step=0.2):
    """(start, end) step indices of the first 360 deg stretch, or None."""
    p = np.asarray(traj, float)
    d = np.diff(p, axis=0)
    keep = np.flatnonzero(np.hypot(d[:, 0], d[:, 1]) > min_step)
    if len(keep) < 3:
        return None
    h = np.arctan2(d[keep, 1], d[keep, 0])
    cum = np.concatenate([[0.0], np.cumsum((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)])
    for j in range(len(cum)):
        lo = np.flatnonzero(np.abs(cum[:j] - cum[j]) >= 2 * math.pi)
        if len(lo):
            return int(keep[lo[-1]]), int(keep[j]) + 1
    return None


def agent_mismatch(traj, goal, ang, cell):
    m = ang.shape[0]
    out = []
    for p in np.asarray(traj, float)[:-1]:
        i = min(m - 1, max(0, int(p[1] / cell)))
        j = min(m - 1, max(0, int(p[0] / cell)))
        g = ang[i, j]
        if not np.isfinite(g):
            out.append(np.nan)
            continue
        gx, gy = goal.nearest_point((float(p[0]), float(p[1])))
        if math.hypot(gx - p[0], gy - p[1]) < 1e-6:
            out.append(np.nan)
            continue
        b = math.atan2(gy - p[1], gx - p[0])
        out.append(abs((b - g + math.pi) % (2 * math.pi) - math.pi))
    return np.degrees(np.array(out))


def _chunk(args):
    model, items = args
    from evaluate import LOOP_TURN, load_policy, run_episode, turning
    from policyconfig import default_reward
    from reachability import clearance, geodesic_field, occupancy
    from scene import Scene
    ctrl, _, tr, _ = load_policy(model)
    cfg = default_reward(tr)
    res = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        clear = clearance(cfg, occupancy(cfg, sc))
        angs = {}
        for i, (a, t, o) in enumerate(zip(sc.agents, r["trajectories"], r["outcomes"])):
            g = sc.goal_for(a)
            if g not in angs:
                angs[g] = descent_dir(geodesic_field(cfg, sc, g, cell=cfg.geo_cell, clear=clear))
            mm = agent_mismatch(t, g, angs[g], cfg.geo_cell)
            loop = bool(turning(t) >= LOOP_TURN)
            w = loop_window(t) if loop else None
            res.append(dict(scene=name, layout=sc.layout, agent=i, outcome=o, loop=loop,
                            window=w, mismatch=[None if not np.isfinite(x) else round(float(x), 1)
                                                for x in mm]))
    return res


def summary(rows):
    def share(vals, thr):
        v = np.array([x for x in vals if x is not None], float)
        return float((v > thr).mean()) if len(v) else float("nan"), len(v)
    inside, outside, never = [], [], []
    for r in rows:
        mm = r["mismatch"]
        if r["loop"] and r["window"]:
            s, e = r["window"]
            inside += mm[s:e]
            outside += mm[:s] + mm[e:]
        elif not r["loop"]:
            never += mm
    out = {}
    for name, v in (("loop window", inside), ("loop agents, outside window", outside),
                    ("agents that never loop", never)):
        out[name] = {f">{t}": share(v, t)[0] for t in (45, 90)} | {"steps": share(v, 0)[1]}
    bins = [(0, 45), (45, 90), (90, 135), (135, 181)]
    by = {}
    for lo, hi in bins:
        sel = [r for r in rows if lo <= max([x for x in r["mismatch"] if x is not None] or [0]) < hi]
        by[f"{lo}-{min(hi, 180)}"] = {"agents": len(sel),
                                      "loop_rate": float(np.mean([r["loop"] for r in sel])) if sel else None,
                                      "success": float(np.mean([r["outcome"] == "goal" for r in sel])) if sel else None}
    out["by max mismatch"] = by
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from suites import load_suite
    items = [(n, s.as_dict()) for n, s in load_suite(a.suite)][:a.limit]
    k = max(1, a.workers) * 4
    jobs = [(a.model, items[i::k]) for i in range(k) if items[i::k]]
    if a.workers > 0:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            parts = pool.map(_chunk, jobs)
    else:
        parts = [_chunk(j) for j in jobs]
    rows = [r for p in parts for r in p]
    s = summary(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": s, "agents": rows}))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
