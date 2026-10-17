"""Fixed evaluation scene sets (DESIGN.md §5.1).

Owns: ``make_suite`` (generate once, write ``scene_NNNNN.json`` + ``suite.json``)
and ``load_suite``.

Invariants:
- A suite is a pure function of its parameters: scene ``k`` of layout ``l`` uses
  ``default_rng(SeedSequence([seed, l, k]))``, so building it twice gives
  identical files (test 12).
- Headings are randomised at generation time with that same generator and
  stored, so evaluation itself needs no randomness.
- The seeds of the suites (7000/8000/9000) are roots of their own
  ``SeedSequence`` streams, disjoint from the training streams
  ``[run_seed, worker, env]`` by construction (different entropy length/values).
"""
from __future__ import annotations

import _paths  # noqa: F401

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from env import randomise_headings
from layouts import random_scene
from scene import Scene

SUITE_SPECS = {                      # name -> (per_layout, agents, seed)  (§5.1)
    "val": (20, (2, 8), 7000),
    "test4": (100, (4, 4), 8000),
    "test_mixed": (100, (2, 8), 9000),
}


def _scene_for(layout_index, layout, k, seed, agents, obstacles, sim_cfg):
    rng = np.random.default_rng(np.random.SeedSequence([seed, layout_index, k]))
    n = int(rng.integers(agents[0], agents[1] + 1))
    m = int(rng.integers(obstacles[0], obstacles[1] + 1))
    while True:
        scene = random_scene(rng, sim_cfg, layout=layout, n_agents=n, n_obstacles=m)
        if scene is not None and len(scene.agents) >= 1:
            return randomise_headings(scene, rng)


def make_suite(out_dir, layouts, per_layout, agents, obstacles, seed, sim_cfg, name=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    idx = 0
    for li, layout in enumerate(layouts):
        for k in range(per_layout):
            scene = _scene_for(li, layout, k, seed, agents, obstacles, sim_cfg)
            scene.seed = None
            fname = f"scene_{idx:05d}"
            (out / f"{fname}.json").write_text(
                json.dumps(scene.as_dict(sim_cfg, fname), indent=1))
            names.append(fname)
            idx += 1
    meta = {"name": name or out.name, "seed": seed, "layouts": list(layouts),
            "per_layout": per_layout, "agents": list(agents), "obstacles": list(obstacles),
            "heading": "random", "n_scenes": idx, "scenes": names, "sim": asdict(sim_cfg)}
    (out / "suite.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_suite(path) -> list:
    """``[(scene_name, Scene), ...]`` in generation order."""
    path = Path(path)
    meta = json.loads((path / "suite.json").read_text())
    return [(n, Scene.from_dict(json.loads((path / f"{n}.json").read_text())))
            for n in meta["scenes"]]
