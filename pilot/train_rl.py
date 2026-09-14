"""Train PPO policies on randomized open scenes (3 seeds)."""
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env import SteeringEnv

TIMESTEPS = 300_000


def train_seed(seed: int):
    os.makedirs("results", exist_ok=True)
    env = DummyVecEnv([
        (lambda i=i: Monitor(SteeringEnv(seed=seed * 100 + i),
                             filename=f"results/monitor_seed{seed}_env{i}"))
        for i in range(8)
    ])
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        n_steps=512, batch_size=512, learning_rate=3e-4,
        gamma=0.995, ent_coef=0.005,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    model.save(f"results/ppo_seed{seed}")
    print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    for s in seeds:
        train_seed(s)
