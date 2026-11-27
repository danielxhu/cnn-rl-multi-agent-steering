"""JAX-trained weights -> a checkpoint ``evaluate.py`` loads (torch environment).

    python jaxsim/export_ckpt.py runs/frontier_seed0/params_best.npz models/frontier_seed0/ckpt_best.pt

The config is the e3_both observation (50 numbers, bearing only, no wall
map) with a GRU of the trained width; reward = the training reward
(geodesic progress, timeout 25); ``iteration`` / ``agent_steps`` from
``args.json`` / ``log.csv`` next to the weights when present.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))

import _paths  # noqa: F401,E402

import csv  # noqa: E402

import numpy as np  # noqa: E402


def export(src: Path, dst: Path):
    import torch
    from config import Config
    from nets import ActorCritic
    from policyconfig import EnvConfig, NetConfig, RunConfig
    z = np.load(src)
    w = {k: z[k] for k in z.files}
    H = int(w["actor_body.fc1.weight"].shape[0])
    cfg = RunConfig(env=EnvConfig(memory="gru", goal_obs="bearing", wallmap="off"),
                    net=NetConfig(memory="gru", hidden=H),
                    reward={"timeout_penalty": 25.0, "progress_metric": "geodesic"})
    model = ActorCritic(50, cfg.net)
    model.set_weights(w)                                # fails loudly on any shape mismatch
    it = steps = None
    log = src.parent / "log.csv"
    if log.exists():
        rows = list(csv.DictReader(open(log)))
        if rows:
            it, steps = int(rows[-1]["update"]), int(rows[-1]["agent_steps"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": cfg.to_dict(Config()), "iteration": it, "agent_steps": steps,
                "val": {}, "source": str(src)}, dst)
    print(f"{src} -> {dst} (hidden {H})")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
