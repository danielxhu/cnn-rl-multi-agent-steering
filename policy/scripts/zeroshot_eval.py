"""Zero-shot evaluation (DESIGN_ZEROSHOT.md): evaluate.py's per-scene rows plus
the four new metrics, for a checkpoint or an ORCA baseline.

    python scripts/zeroshot_eval.py --model models/e7_map_seed0/ckpt_best.pt --suites novel,crowd16 --data-root data/suites --out results/zeroshot/runs/e7_map_seed0.csv
    python scripts/zeroshot_eval.py --model orca_path --suites novel --data-root data/suites --out ...

Extra columns (per agent, then the scene mean), DESIGN_ZEROSHOT.md §3:

- ``stuck_rate``: share of agents that, while active, gain less than
  ``max_speed`` of geodesic progress over some ``STUCK_WINDOW`` consecutive steps;
- ``pause_steps``: mean number of active steps with speed < ``PAUSE_SPEED`` x max_speed;
- ``throughput``: arrivals per 100 steps, from step 0 to the last arrival (0 if none);
- ``path_ratio``: path length / geodesic start-to-goal distance, successful agents (NaN if none).

Geodesic distances use the simulator's field (agent-radius clearance, the
agent's own goal), so a detour round a wall counts as progress.

Invariants:
- Read-only on models and suites; episodes are those of ``evaluate.py``
  (same ``run_episode``, deterministic policy), so the shared columns agree.
- Paths through ``pathlib``; process pool via ``spawn`` (Windows-safe).
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

STUCK_WINDOW = 30
PAUSE_SPEED = 0.10
EXTRA = ["stuck_rate", "pause_steps", "throughput", "path_ratio"]


def agent_metrics(traj, goal_field, goal, cfg):
    """(stuck, pause_steps, path_length, geodesic start distance) of one trajectory."""
    from reachability import geodesic_distance
    t = np.asarray(traj, float)
    geo = np.array([geodesic_distance(cfg, goal_field, p, cfg.geo_cell, goal) for p in t])
    step = np.hypot(*np.diff(t, axis=0).T) if len(t) > 1 else np.zeros(0)
    stuck = False
    if len(geo) > STUCK_WINDOW:
        prog = geo[:-STUCK_WINDOW] - geo[STUCK_WINDOW:]
        stuck = bool((prog < cfg.max_speed).any())
    pause = int((step < PAUSE_SPEED * cfg.max_speed).sum())
    return stuck, pause, float(step.sum()), float(geo[0])


def scene_extras(scene, row, cfg):
    from reachability import clearance, geodesic_field, occupancy
    clear = clearance(cfg, occupancy(cfg, scene))
    fields = {}
    stuck, pause, ratios, arrivals = [], [], [], []
    for i, (a, traj, out) in enumerate(zip(scene.agents, row["trajectories"], row["outcomes"])):
        g = scene.goal_for(a)
        if g not in fields:
            fields[g] = geodesic_field(cfg, scene, g, cell=cfg.geo_cell, clear=clear)
        s, p, length, d0 = agent_metrics(traj, fields[g], g, cfg)
        stuck.append(s)
        pause.append(p)
        if out == "goal":
            arrivals.append(len(traj) - 1)
            if d0 > 1e-6 and math.isfinite(d0):
                ratios.append(length / d0)
    last = max(arrivals) if arrivals else 0
    return {"stuck_rate": float(np.mean(stuck)), "pause_steps": float(np.mean(pause)),
            "throughput": 100.0 * len(arrivals) / last if last > 0 else 0.0,
            "path_ratio": float(np.mean(ratios)) if ratios else float("nan")}


def make_controller(model, sim_cfg, exit_rule, shield):
    if model.startswith("orca_"):
        from orca import ORCAController
        return ORCAController(model[len("orca_"):])
    from evaluate import load_policy
    from evaluate import trained_shield
    ctrl, run_cfg, _, _ = load_policy(model)
    if exit_rule != "enclosed":
        ctrl.exit_rule = exit_rule
    if shield is None:
        shield = trained_shield(run_cfg)
    if shield is not None:
        from shield import Shield
        ctrl = Shield(ctrl, sim_cfg, shield)
    return ctrl


def _chunk(args):
    model, suite, items, sim_dict, exit_rule, shield = args
    from evaluate import run_episode, sim_config_from_dict
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl = make_controller(model, cfg, exit_rule, shield)
    rows = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        r.update(scene_extras(sc, r, cfg))
        for k in ("trajectories", "outcomes", "collision_pos"):
            r.pop(k)
        r.update(suite=suite, scene=name, layout=sc.layout)
        rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="checkpoint path, orca_straight or orca_path")
    ap.add_argument("--suites", required=True)
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--out", type=Path, required=True, help="per-scene CSV")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--exit-rule", choices=["enclosed", "orbit"], default="enclosed")
    ap.add_argument("--shield", type=float, default=None, metavar="MARGIN")
    ap.add_argument("--limit", type=int, default=None, help="first N scenes per suite (smoke runs)")
    a = ap.parse_args(argv)
    from dataclasses import asdict
    import json
    from evaluate import ROW_METRICS, E7_METRICS, default_reward, sim_config_from_dict
    from suites import load_suite
    rows = []
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers > 0 else None
    try:
        for s in a.suites.split(","):
            sdir = a.data_root / s
            sim_cfg = default_reward(sim_config_from_dict(
                json.loads((sdir / "suite.json").read_text())["sim"]))
            items = [(n, sc.as_dict()) for n, sc in load_suite(sdir)][:a.limit]
            k = max(1, a.workers) * 4
            jobs = [(a.model, s, items[i::k], asdict(sim_cfg), a.exit_rule, a.shield)
                    for i in range(k) if items[i::k]]
            parts = pool.map(_chunk, jobs) if pool else [_chunk(j) for j in jobs]
            order = {n: j for j, (n, _) in enumerate(items)}
            got = sorted((r for p in parts for r in p), key=lambda r: order[r["scene"]])
            rows += got
            print(f"{s}: {len(got)} scenes, success {np.mean([r['success_rate'] for r in got]):.3f}",
                  flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    cols = ["suite", "scene", "layout", "n_agents", "bottleneck"] + ROW_METRICS + E7_METRICS + \
        ["multi_exit"] + EXTRA
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
