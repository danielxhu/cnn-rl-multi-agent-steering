"""End-to-end pipeline: image -> U-Net parse -> simulator -> policy -> red path.

The policy runs closed-loop inside the PARSED scene (perception errors
propagate honestly); the resulting trajectory is then validated against the
GROUND-TRUTH scene: success = reaches the true goal with no true collision.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from perception import SIZE, UNet, extract_scene, predict
from sim import (AGENT_R, GOAL_REACH_DIST, STATEMENT_HALLWAY, STATEMENT_SINGLE,
                 WORLD, Scene, Simulator, random_scene, render, render_small)

LABEL_COLORS = np.array([
    [255, 255, 255],  # background
    [40, 40, 40],     # wall
    [200, 213, 163],  # fov
    [120, 120, 120],  # obstacle
    [20, 20, 230],    # goal
    [230, 120, 20],   # agent
], dtype=np.uint8)


def load_models(seed=0):
    net = UNet()
    net.load_state_dict(torch.load("results/unet.pt", weights_only=True))
    net.eval()
    model = PPO.load(f"results/ppo_seed{seed}")
    return net, model


def rollout(model, scene: Scene, goal_tol=None):
    sim = Simulator(scene) if goal_tol is None else Simulator(scene, goal_tol=goal_tol)
    obs = sim.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        turn = float(action[0])
        speed = (float(action[1]) + 1.0) / 2.0
        obs, _, done = sim.step(turn, speed)
    return sim


def validate_on_gt(traj, gt: Scene) -> str:
    """Replay a trajectory against the ground-truth scene."""
    if gt.hallway:
        y_lo, y_hi = gt.hallway
    else:
        y_lo, y_hi = 0.0, WORLD
    pts = []
    for a, b in zip(traj, traj[1:]):
        seg = max(2, int(math.dist(a, b) / 0.5) + 1)
        for t in np.linspace(0, 1, seg):
            pts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    for x, y in pts:
        if not (AGENT_R < x < WORLD - AGENT_R and y_lo + AGENT_R < y < y_hi - AGENT_R):
            return "collision"
        for cx, cy, r in gt.obstacles:
            if math.dist((x, y), (cx, cy)) < r + AGENT_R:
                return "collision"
    if math.dist(traj[-1], gt.goal) < GOAL_REACH_DIST:
        return "goal"
    return "timeout"


def pipeline(net, model, gt: Scene):
    """Full image->path pipeline on one scene. Returns (outcome, sim, parsed)."""
    img_small, _ = render_small(gt, SIZE)
    pred = predict(net, img_small[None])[0]
    parsed = extract_scene(pred)
    if parsed is None:
        return "parse_failure", None, None, pred
    # stop 2.0 from the parsed goal: leaves margin for perception error while
    # the true success criterion stays at GOAL_REACH_DIST around the TRUE goal
    sim = rollout(model, parsed, goal_tol=2.0)
    outcome = validate_on_gt(sim.trajectory, gt)
    return outcome, sim, parsed, pred


# ------------------------------------------------------------------ figures

def fig_showcase(net, model):
    rng = np.random.default_rng(7)
    scenes = [
        ("single obstacle (statement Fig. 1)", STATEMENT_SINGLE),
        ("multi-obstacle (random held-out)", random_scene(rng, 3)),
        ("hallway, zero-shot (statement Fig. 3)", STATEMENT_HALLWAY),
    ]
    fig, axes = plt.subplots(len(scenes), 2, figsize=(9, 13))
    outcomes = {}
    for row, (name, gt) in enumerate(scenes):
        outcome, sim, parsed, _ = pipeline(net, model, gt)
        outcomes[name] = outcome
        img_in, _ = render(gt)
        img_out, _ = render(gt, path=sim.trajectory if sim else None)
        axes[row, 0].imshow(img_in)
        axes[row, 0].set_title(f"input — {name}", fontsize=9)
        axes[row, 1].imshow(img_out)
        axes[row, 1].set_title(f"output — {outcome}", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/fig_end_to_end.png", dpi=150)
    plt.close(fig)
    return outcomes


def fig_perception(net):
    rng = np.random.default_rng(21)
    scenes = [random_scene(rng, 2), random_scene(rng, 3, hallway=True)]
    fig, axes = plt.subplots(len(scenes), 3, figsize=(10, 7))
    for row, sc in enumerate(scenes):
        img, lab = render_small(sc, SIZE)
        pred = predict(net, img[None])[0]
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("input (128x128)", fontsize=9)
        axes[row, 1].imshow(LABEL_COLORS[lab])
        axes[row, 1].set_title("ground-truth labels", fontsize=9)
        axes[row, 2].imshow(LABEL_COLORS[pred])
        axes[row, 2].set_title("U-Net prediction", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/fig_perception.png", dpi=150)
    plt.close(fig)


def fig_training_curve():
    import pandas as pd
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in [0, 1, 2]:
        series = []
        for i in range(8):
            path = f"results/monitor_seed{seed}_env{i}.monitor.csv"
            if os.path.exists(path):
                series.append(pd.read_csv(path, skiprows=1))
        if not series:
            continue
        df = pd.concat(series).sort_values("t").reset_index(drop=True)
        df["cum_steps"] = df["l"].cumsum()
        roll = df["r"].rolling(100).mean()
        ax.plot(df["cum_steps"], roll, label=f"seed {seed}", linewidth=1.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return (rolling mean, w=100)")
    ax.set_title("PPO training on randomized open scenes (1-3 obstacles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/fig_training_curve.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------- quantitative end-to-end

def quantitative(net, model, n=100):
    rng = np.random.default_rng(99)
    out = {}
    for family in ["single", "multi", "hallway"]:
        res = {"pipeline": [], "oracle": []}
        for ep in range(n):
            if family == "single":
                gt = random_scene(rng, 1)
            elif family == "multi":
                gt = random_scene(rng, int(rng.integers(2, 4)))
            else:
                gt = random_scene(rng, int(rng.integers(1, 4)), hallway=True)
            outcome, _, _, _ = pipeline(net, model, gt)
            res["pipeline"].append(outcome)
            oracle_sim = rollout(model, gt)
            res["oracle"].append(oracle_sim.outcome)
        out[family] = {
            mode: {
                "success_rate": v.count("goal") / n,
                "collision_rate": v.count("collision") / n,
                "timeout_rate": v.count("timeout") / n,
                "parse_failure_rate": v.count("parse_failure") / n,
            }
            for mode, v in res.items()
        }
        print("e2e", family, "done", flush=True)
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # pick the seed with best mean success across families
    with open("results/policy_metrics.json") as f:
        pm = json.load(f)
    best_seed = 0
    best = -1
    for s in [0, 1, 2]:
        m = np.mean([pm["ppo"][f]["per_seed"][s]["success_rate"]
                     for f in ["single", "multi", "hallway"]])
        if m > best:
            best, best_seed = m, s
    print("best seed:", best_seed, "mean success:", best)

    net, model = load_models(best_seed)
    showcase = fig_showcase(net, model)
    fig_perception(net)
    fig_training_curve()
    e2e = quantitative(net, model)
    result = {"best_seed": best_seed, "showcase_outcomes": showcase,
              "end_to_end": e2e}
    with open("results/end_to_end_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
