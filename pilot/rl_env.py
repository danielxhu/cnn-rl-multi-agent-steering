"""Gymnasium wrapper around the steering simulator.

Training scenes are randomized OPEN layouts (1-3 obstacles, no hallway);
hallway layouts are held out entirely for zero-shot generalization tests.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim import MAX_STEPS, N_SECTORS, Scene, Simulator, random_scene


class SteeringEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, scene_sampler=None, fixed_scene: Scene | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.fixed_scene = fixed_scene
        self.scene_sampler = scene_sampler or self._default_sampler
        self.observation_space = spaces.Box(-1.0, 1.5, shape=(N_SECTORS + 3,), dtype=np.float32)
        # action: [turn in [-1,1], speed in [-1,1] -> mapped to [0,1]]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.sim = None

    def _default_sampler(self, rng):
        return random_scene(rng, int(rng.integers(1, 4)), hallway=False)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scene = self.fixed_scene or self.scene_sampler(self.rng)
        self.sim = Simulator(scene)
        return self.sim.reset(), {}

    def step(self, action):
        turn = float(action[0])
        speed = (float(action[1]) + 1.0) / 2.0
        obs, reward, done = self.sim.step(turn, speed)
        terminated = done and self.sim.outcome in ("goal", "collision")
        truncated = done and self.sim.outcome == "timeout"
        info = {"outcome": self.sim.outcome} if done else {}
        return obs, reward, terminated, truncated, info
