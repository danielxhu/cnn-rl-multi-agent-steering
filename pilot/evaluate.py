"""Evaluate trained policies vs. a go-straight baseline across layout families.

Layout families:
  single   - 1 obstacle, open scene (training family)
  multi    - 2-3 obstacles, open scene (training family)
  hallway  - 1-3 obstacles inside corridor walls (NEVER seen in training)
"""
import json
import math

import numpy as np
from stable_baselines3 import PPO

from rl_env import SteeringEnv
from sim import Simulator, random_scene

N_EPISODES = 100
SEEDS = [0, 1, 2]


def sampler_for(family):
    def f(rng):
        if family == "single":
            return random_scene(rng, 1, hallway=False)
        if family == "multi":
            return random_scene(rng, int(rng.integers(2, 4)), hallway=False)
        if family == "hallway":
            return random_scene(rng, int(rng.integers(1, 4)), hallway=True)
        raise ValueError(family)
    return f


def path_length(traj):
    return sum(math.dist(a, b) for a, b in zip(traj, traj[1:]))


def eval_policy(model, family, eval_seed=12345):
    env = SteeringEnv(seed=eval_seed, scene_sampler=sampler_for(family))
    outcomes, lengths = [], []
    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        outcomes.append(env.sim.outcome)
        if env.sim.outcome == "goal":
            lengths.append(path_length(env.sim.trajectory))
    return summarize(outcomes, lengths)


def eval_straight_baseline(family, eval_seed=12345):
    """Hand-crafted baseline: always turn toward the goal, full speed."""
    rng = np.random.default_rng(eval_seed)
    sampler = sampler_for(family)
    outcomes, lengths = [], []
    for ep in range(N_EPISODES):
        sim = Simulator(sampler(rng))
        obs = sim.reset()
        done = False
        while not done:
            rel_angle = obs[17] * math.pi  # de-normalize
            turn = np.clip(rel_angle / math.radians(25.0), -1, 1)
            obs, _, done = sim.step(turn, 1.0)
        outcomes.append(sim.outcome)
        if sim.outcome == "goal":
            lengths.append(path_length(sim.trajectory))
    return summarize(outcomes, lengths)


def summarize(outcomes, lengths):
    n = len(outcomes)
    return {
        "success_rate": outcomes.count("goal") / n,
        "collision_rate": outcomes.count("collision") / n,
        "timeout_rate": outcomes.count("timeout") / n,
        "avg_path_length": float(np.mean(lengths)) if lengths else float("nan"),
        "n_episodes": n,
    }


if __name__ == "__main__":
    results = {"baseline": {}, "ppo": {}}
    for family in ["single", "multi", "hallway"]:
        results["baseline"][family] = eval_straight_baseline(family)
        per_seed = []
        for s in SEEDS:
            model = PPO.load(f"results/ppo_seed{s}")
            per_seed.append(eval_policy(model, family))
        agg = {}
        for k in per_seed[0]:
            vals = [p[k] for p in per_seed if not (isinstance(p[k], float) and math.isnan(p[k]))]
            agg[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                      "max": float(np.max(vals))}
        results["ppo"][family] = {"per_seed": per_seed, "aggregate": agg}
        print(family, "done", flush=True)

    with open("results/policy_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
