"""Causal check of DESIGN_E7 §14: give the trained policy the true way to the goal.

    python scripts/oracle_dir.py --variant e7_map --oracle downhill --suites test4,novel --workers 9

At evaluation only, no retraining, the direction inputs are replaced by the
geodesic direction (steepest descent of the simulator's own geodesic field at
the agent's position; privileged, a ceiling, not a deployable policy):

- ``--oracle downhill``: the two wall-map columns (cos, sin) — e7 checkpoints;
- ``--oracle bearing``: the goal-bearing column;
- ``--oracle both``; ``--oracle none`` reproduces ``evaluate.py``;
- ``--oracle goalpoint``: not the direction but the wall map's target point:
  the goal's nearest point (needs the goal distance, which the policy does not
  get) instead of method A's far point on the world edge; the wall map's
  field and descent are otherwise unchanged — a deployable feature-layer change
  if the robot knows where its goal is.

If loops and timeouts mostly vanish, the direction signal causes them; if
not, the network's driving does. Writes ``<out>/<variant>_<oracle>_seed<k>.csv``
(per scene: success, collisions, timeout, loop rate) and prints the means.

Invariants: read-only on models and suites; the policy and its other inputs
are those of ``evaluate.py``; process pool via ``spawn``.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

N_DIRS = 32
PROBE = 2.0          # world units


def geodesic_dir(sim, i):
    """World angle of steepest geodesic descent at agent i, or None."""
    from reachability import geodesic_distance
    g = sim._goal(i)
    fld = sim._geo[g]
    p = sim.pos[i]
    here = geodesic_distance(sim.cfg, fld, p, sim.cfg.geo_cell, g)
    best, ang = here, None
    for a in np.linspace(-math.pi, math.pi, N_DIRS, endpoint=False):
        q = p + PROBE * np.array([math.cos(a), math.sin(a)])
        v = geodesic_distance(sim.cfg, fld, q, sim.cfg.geo_cell, g)
        if v < best - 1e-6:
            best, ang = v, a
    return ang


def oracle_act(self, sim, obs51):
    """PolicyController.act with the direction inputs replaced (see module doc)."""
    import torch
    from env import actions_to_sim, to_policy_obs
    obs = to_policy_obs(obs51, self.cfg.env)
    d0 = obs.shape[1]
    if self.wm is not None:
        self.wm._sim = sim
        obs = self.wm.observe(sim, obs)
    bcol = d0 - 2
    for i in np.flatnonzero(sim.active):
        if self.oracle == "goalpoint":
            continue
        a = geodesic_dir(sim, i)
        if a is None:
            continue
        rel = (a - sim.heading[i] + math.pi) % (2 * math.pi) - math.pi
        if self.oracle in ("bearing", "both"):
            obs[i, bcol] = rel / math.pi
        if self.oracle in ("downhill", "both") and self.wm is not None:
            obs[i, d0:d0 + 2] = (math.cos(rel), math.sin(rel))
    with torch.no_grad():
        dist, _, self.state = self.model.step(torch.from_numpy(obs), self.state, self.first)
        a = dist.mean
    self.first = torch.zeros_like(self.first)
    return actions_to_sim(a.numpy())


def _chunk(args):
    ckpt, oracle, exit_rule, items, sim_dict = args
    from evaluate import LOOP_TURN, load_policy, run_episode, sim_config_from_dict, turning
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl, _, _, _ = load_policy(ckpt)
    ctrl.exit_rule = exit_rule
    ctrl.oracle = oracle
    if oracle != "none":
        ctrl.act = types.MethodType(oracle_act, ctrl)
    if oracle == "goalpoint":
        orig_reset = ctrl.reset

        def reset(sim):
            orig_reset(sim)
            wm = ctrl.wm
            wm.virtual_goal = lambda i, bearing: np.array(
                wm.goals[i][wm.target[i]].nearest_point(tuple(wm._sim.pos[i])), float)
        ctrl.reset = reset
    rows = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        loops = [turning(t) >= LOOP_TURN for t in r["trajectories"]]
        rows.append(dict(scene=name, layout=sc.layout, success_rate=r["success_rate"],
                         static_collision_rate=r["static_collision_rate"],
                         agent_collision_rate=r["agent_collision_rate"],
                         timeout_rate=r["timeout_rate"], loop_rate=float(np.mean(loops))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="e7_map")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--oracle", choices=["none", "downhill", "bearing", "both", "goalpoint"], default="downhill")
    ap.add_argument("--exit-rule", choices=["enclosed", "orbit"], default="orbit")
    ap.add_argument("--suites", default="test4,novel")
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "oracle_dir")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from dataclasses import asdict, replace
    from evaluate import default_reward, sim_config_from_dict
    from suites import load_suite
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers > 0 else None
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        for k in a.seeds.split(","):
            ckpt = str(POLICY / "models" / f"{a.variant}_seed{k}" / "ckpt_best.pt")
            out = a.out / f"{a.variant}_{a.oracle}_seed{k}.csv"
            rows = []
            for s in a.suites.split(","):
                sdir = a.data_root / s
                sim_cfg = default_reward(sim_config_from_dict(
                    json.loads((sdir / "suite.json").read_text())["sim"]))
                # the field only feeds the oracle (and the reward, which no action sees)
                sim_cfg = replace(sim_cfg, progress_metric="geodesic")
                items = [(n, sc.as_dict()) for n, sc in load_suite(sdir)][:a.limit]
                kk = max(1, a.workers) * 4
                jobs = [(ckpt, a.oracle, a.exit_rule, items[j::kk], asdict(sim_cfg))
                        for j in range(kk) if items[j::kk]]
                parts = pool.map(_chunk, jobs) if pool else [_chunk(j) for j in jobs]
                got = [dict(r, suite=s) for p in parts for r in p]
                rows += got
                m = {c: 100 * np.mean([r[c] for r in got]) for c in
                     ("success_rate", "loop_rate", "timeout_rate", "static_collision_rate",
                      "agent_collision_rate")}
                print(f"{a.variant} {a.oracle} seed{k} {s}: " +
                      " ".join(f"{c.split('_')[0]} {v:.1f}" for c, v in m.items()), flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
