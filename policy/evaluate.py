"""Evaluate a controller on fixed suites (DESIGN.md §5.2–§5.3).

    python evaluate.py --ckpt RUN/ckpt_best.pt --suites test4,test_mixed --data-root D --out OUT
    python evaluate.py --greedy                --suites test4,test_mixed --data-root D --out OUT

Owns: the two controllers (``GreedyController`` = ``physics.greedy_action``,
``PolicyController`` = a trained network), ``run_episode`` (one scene, one
episode, the §5.3 row), the CLI that writes ``per_scene.csv``, ``metrics.json``
and ``trajectories/``, and ``Validator`` (the same code on a suite, for
``train.py``).

Invariants:
- One scene = one episode with the simulator exactly as trained (collision ends
  the agent). No noise, deterministic policy unless ``--stochastic`` (then a
  fixed-seed generator).
- ``run_episode`` steps the ``Simulator`` itself; the collision split asks
  ``Simulator._hits_static`` at the step the agent was removed (the simulator
  gives static hits precedence, and so does this).
- Episodes are independent, so a process pool changes nothing but speed.
"""
from __future__ import annotations

import _paths  # noqa: F401

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from env import (ObsStack, actions_to_sim, base_obs_dim, obs_dim, to_policy_obs)
from nets import ActorCritic
from physics import Simulator, greedy_action
from policyconfig import RunConfig, sim_config_from_dict
from scene import Scene
from suites import load_suite

BOTTLENECK_BINS = [(5.2, 7.0), (7.0, 10.0), (10.0, 15.0), (15.0, math.inf)]
ROW_METRICS = ["success_rate", "collision_rate", "static_collision_rate",
               "agent_collision_rate", "timeout_rate", "collision_violations",
               "remaining_agents", "finish_time", "mean_finish_step", "mean_path_length",
               "all_reached"]


# ------------------------------------------------------------ controllers

class GreedyController:
    name = "greedy"

    def reset(self, sim):
        pass

    def act(self, sim, obs51):
        return np.stack([greedy_action(sim, i) for i in range(len(sim.pos))])


class PolicyController:
    """A trained actor-critic acting on the 50-number (or stacked) observation."""

    def __init__(self, run_cfg: RunConfig, weights: dict, deterministic=True, seed=0):
        self.cfg, self.det = run_cfg, deterministic
        torch.set_num_threads(1)
        self.model = ActorCritic(obs_dim(run_cfg.env), run_cfg.net)
        self.model.set_weights(weights)
        self.model.eval()
        self.gen = torch.Generator().manual_seed(seed)
        self.name = run_cfg.net.memory

    def reset(self, sim):
        n = len(sim.pos)
        self.state = self.model.initial_state(n)
        self.first = torch.ones(n, dtype=torch.bool)
        ec = self.cfg.env
        self.stack = (ObsStack(n, ec.stack_k, base_obs_dim(ec)) if ec.memory == "stack" else None)

    def act(self, sim, obs51):
        obs = to_policy_obs(obs51, self.cfg.env)
        if self.stack is not None:
            if self.first.any():
                self.stack.clear(self.first.numpy())
            obs = self.stack.push(obs)
        with torch.no_grad():
            dist, _, self.state = self.model.step(torch.from_numpy(obs), self.state, self.first)
            a = dist.mean
            if not self.det:
                a = a + dist.stddev * torch.randn(a.shape, generator=self.gen)
        self.first = torch.zeros_like(self.first)
        return actions_to_sim(a.numpy())


def load_policy(ckpt_path, deterministic=True, seed=0):
    from train import load_ckpt
    ck = load_ckpt(Path(ckpt_path))
    cfg = RunConfig.from_dict(ck["config"])
    ctrl = PolicyController(cfg, {k: v.numpy() for k, v in ck["model"].items()},
                            deterministic, seed)
    return ctrl, cfg, sim_config_from_dict(ck["config"]["sim"]), ck


# ---------------------------------------------------------------- episode

def run_episode(scene, controller, sim_cfg, env_cfg=None, record=False) -> dict:
    """One scene, one episode; the §5.3 row (plus trajectories when ``record``)."""
    sim = Simulator(scene, sim_cfg)
    controller.reset(sim)
    obs = sim.observe()
    n = len(scene.agents)
    static_hit = np.zeros(n, bool)
    was_active = sim.active.copy()
    done = sim.done
    while not done:
        obs, _, done = sim.step(controller.act(sim, obs))
        for i in range(n):
            if was_active[i] and sim.outcome[i] == "collision":
                static_hit[i] = sim._hits_static(i)       # position is frozen at the hit
        was_active = sim.active.copy()
    st = sim.stats()
    coll = np.array([o == "collision" for o in sim.outcome])
    reached = [i for i in range(n) if sim.outcome[i] == "goal"]
    row = {"layout": scene.layout, "n_agents": n, "bottleneck": scene.bottleneck,
           "success_rate": st["success_rate"], "collision_rate": st["collision_rate"],
           "static_collision_rate": float((coll & static_hit).sum() / n),
           "agent_collision_rate": float((coll & ~static_hit).sum() / n),
           "timeout_rate": st["timeout_rate"], "collision_violations": st["collision_violations"],
           "remaining_agents": st["remaining_agents"], "finish_time": st["finish_time"],
           "mean_finish_step": (float(np.mean([sim.finish_step[i] for i in reached]))
                                if reached else float("nan")),
           "mean_path_length": st["mean_path_length"],
           "all_reached": int(len(reached) == n)}
    if record:
        row["trajectories"] = [[list(map(float, p)) for p in t] for t in sim.trajectories]
        row["outcomes"] = list(sim.outcome)
        row["collision_pos"] = [list(map(float, sim.pos[i])) if coll[i] else None
                                for i in range(n)]
    return row


# ------------------------------------------------------------ aggregation

def _mean(rows, key):
    v = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float)
                                                              and math.isnan(r[key]))]
    return float(np.mean(v)) if v else float("nan")


def summarise(rows: list) -> dict:
    return {"scenes": len(rows), **{k: _mean(rows, k) for k in ROW_METRICS}}


def metrics_for(rows: list) -> dict:
    """``overall``, ``per_layout`` and ``per_bottleneck`` of one suite's rows."""
    layouts = sorted({r["layout"] for r in rows})
    out = {"overall": summarise(rows),
           "per_layout": {l: summarise([r for r in rows if r["layout"] == l]) for l in layouts},
           "per_bottleneck": {}}
    for lo, hi in BOTTLENECK_BINS:
        sel = [r for r in rows if r["bottleneck"] is not None and lo <= r["bottleneck"] < hi]
        key = f"[{lo:g},{'inf' if hi == math.inf else format(hi, 'g')})"
        out["per_bottleneck"][key] = summarise(sel) if sel else {"scenes": 0}
    return out


# --------------------------------------------------------------- pool code

def _eval_chunk(args):
    """Worker: a list of scene dicts -> rows. Module-level so ``spawn`` can pickle it."""
    cfg_dict, weights, scene_dicts, sim_dict, deterministic, seed = args
    cfg = RunConfig.from_dict(cfg_dict)
    sim_cfg = sim_config_from_dict(sim_dict)
    ctrl = PolicyController(cfg, weights, deterministic, seed)
    return [run_episode(Scene.from_dict(d), ctrl, sim_cfg) for d in scene_dicts]


def _greedy_chunk(args):
    scene_dicts, sim_dict = args
    sim_cfg = sim_config_from_dict(sim_dict)
    return [run_episode(Scene.from_dict(d), GreedyController(), sim_cfg) for d in scene_dicts]


class Validator:
    """Deterministic policy on a suite, optionally in a persistent process pool."""

    def __init__(self, suite_dir, sim_cfg, workers=0):
        from dataclasses import asdict
        self.names, scenes = zip(*load_suite(suite_dir))
        self.scene_dicts = [s.as_dict() for s in scenes]
        self.sim_dict = asdict(sim_cfg)
        self.workers = workers
        self.pool = mp.get_context("spawn").Pool(workers) if workers > 0 else None

    def run(self, run_cfg, weights) -> dict:
        n = len(self.scene_dicts)
        k = max(1, self.workers) * 2
        chunks = [self.scene_dicts[i::k] for i in range(k)]
        jobs = [(run_cfg.to_dict(), weights, c, self.sim_dict, True, 0) for c in chunks if c]
        parts = self.pool.map(_eval_chunk, jobs) if self.pool else [_eval_chunk(j) for j in jobs]
        rows = [r for p in parts for r in p]
        assert len(rows) == n
        m = metrics_for(rows)
        out = dict(m["overall"])
        out.pop("scenes")
        for l, v in m["per_layout"].items():
            out[f"success_{l}"] = v["success_rate"]
        return out

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


# ------------------------------------------------------------------- CLI

def evaluate_suites(controller_factory, chunk_fn_args, suites: dict, workers: int, out: Path,
                    identity: dict, sim_cfg):
    """Evaluate ``suites`` ({name: dir}) and write the outputs of §5.3."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(exist_ok=True)
    pool = mp.get_context("spawn").Pool(workers) if workers > 0 else None
    all_rows, metrics = [], {"identity": identity, "suites": {}}
    try:
        for sname, sdir in suites.items():
            names, scenes = zip(*load_suite(sdir))
            dicts = [s.as_dict() for s in scenes]
            k = max(1, workers) * 4
            idx = [list(range(i, len(dicts), k)) for i in range(k)]
            jobs = [chunk_fn_args([dicts[j] for j in ix]) for ix in idx if ix]
            fn = jobs[0][0]
            parts = [pool.map(fn, [j[1] for j in jobs])] if pool else [[fn(j[1]) for j in jobs]]
            rows = [None] * len(dicts)
            for ix, part in zip([i for i in idx if i], parts[0]):
                for j, r in zip(ix, part):
                    rows[j] = r
            for name, sc, r in zip(names, scenes, rows):
                r.update(suite=sname, scene=name, layout=sc.layout)
            metrics["suites"][sname] = metrics_for(rows)
            all_rows += rows
            # trajectories of the first scene of each layout, for the path figures (§5.3)
            ctrl = controller_factory()
            seen = set()
            for name, sc in zip(names, scenes):
                if sc.layout in seen:
                    continue
                seen.add(sc.layout)
                rec = run_episode(sc, ctrl, sim_cfg, record=True)
                (out / "trajectories" / f"{sname}_{name}.json").write_text(json.dumps({
                    "suite": sname, "scene": name, "layout": sc.layout,
                    "scene_data": sc.as_dict(), "trajectories": rec["trajectories"], "outcomes": rec["outcomes"],
                    "collision_pos": rec["collision_pos"]}))
    finally:
        if pool:
            pool.close()
            pool.join()
    cols = ["suite", "scene", "layout", "n_agents", "bottleneck"] + ROW_METRICS
    with open(out / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path)
    g.add_argument("--greedy", action="store_true")
    ap.add_argument("--suites", default="test4,test_mixed")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed with --stochastic")
    a = ap.parse_args(argv)
    suites = {s: a.data_root / s for s in a.suites.split(",")}
    first = json.loads((next(iter(suites.values())) / "suite.json").read_text())
    from dataclasses import asdict
    if a.greedy:
        sim_cfg = sim_config_from_dict(first["sim"])
        identity = {"controller": "greedy", "seed": None, "ckpt": None}
        factory = GreedyController
        make_args = lambda dicts: (_greedy_chunk, (dicts, first["sim"]))
    else:
        _, cfg, sim_cfg, ck = load_policy(a.ckpt, not a.stochastic, a.seed)
        weights = {k: v.numpy() for k, v in ck["model"].items()}
        identity = {"controller": cfg.net.memory, "seed": cfg.seed, "ckpt": str(a.ckpt),
                    "iteration": ck.get("iteration"), "agent_steps": ck.get("agent_steps"),
                    "deterministic": not a.stochastic}
        factory = lambda: PolicyController(cfg, weights, not a.stochastic, a.seed)
        make_args = lambda dicts: (_eval_chunk, (cfg.to_dict(), weights, dicts, asdict(sim_cfg),
                                                 not a.stochastic, a.seed))
    m = evaluate_suites(factory, make_args, suites, a.workers, a.out, identity, sim_cfg)
    for sname, sm in m["suites"].items():
        print(f"\n{sname}  ({sm['overall']['scenes']} scenes)   success  collision  timeout")
        for l, v in {**sm["per_layout"], "OVERALL": sm["overall"]}.items():
            print(f"  {l:12s} {v['success_rate']:8.3f} {v['collision_rate']:9.3f} "
                  f"{v['timeout_rate']:8.3f}")


if __name__ == "__main__":
    main()
