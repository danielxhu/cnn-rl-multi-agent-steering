"""Multi-agent RL environment around ``sim/physics.Simulator`` (DESIGN.md §1).

Owns: scene sampling (layout, agent and obstacle counts, random headings), the
50-number policy observation, the optional frame stack and noise, action
mapping, slot bookkeeping and auto-reset.

Invariants:
- Arrays returned by ``MultiAgentEnv`` always have ``max_agents`` rows (slots).
  Slot ``i`` is ``scene.agents[i]`` for the whole scene; slots beyond
  ``n_agents`` are never alive.
- A transition belongs to a slot only on steps where ``alive`` was True
  *before* the step. ``terminal`` (goal, collision) and ``truncated``
  (timeout) are mutually exclusive and only set on such steps. With
  ``timeout_penalty > 0`` a timeout is part of the task and is ``terminal``
  too (no bootstrap; DESIGN_E3.md §1.1).
- On the step that ends a scene the env samples the next scene in place and
  returns *its* first observation, with ``first`` set on the filled slots;
  ``final_obs`` still holds the observation of the old scene, which truncated
  slots need for bootstrapping.
- Noise only touches the policy's copy of the observation, never the
  simulator's state.
- All randomness comes from the ``rng`` given at construction.
"""
from __future__ import annotations

import _paths  # noqa: F401

import math
from dataclasses import replace

import numpy as np

from layouts import random_scene
from physics import Simulator
from scene import Scene

SIM_OBS_DIM = 51
GOAL_DIST_COL = 48


def make_env_rng(seed: int, worker: int, env: int, epoch: int | None = None):
    """Generator of one env (§1.7); ``epoch`` re-seeds after a resume (§4)."""
    entropy = [seed, worker, env] + ([] if epoch is None else [epoch])
    return np.random.default_rng(np.random.SeedSequence(entropy))


def base_obs_dim(env_cfg) -> int:
    return SIM_OBS_DIM if env_cfg.goal_obs == "bearing_distance" else SIM_OBS_DIM - 1


def obs_dim(env_cfg) -> int:
    k = env_cfg.stack_k if env_cfg.memory == "stack" else 1
    return base_obs_dim(env_cfg) * k


def to_policy_obs(obs51: np.ndarray, env_cfg) -> np.ndarray:
    """§1.3 without stacking and without noise: drop the goal distance."""
    obs51 = np.asarray(obs51, np.float32)
    if env_cfg.goal_obs == "bearing_distance":
        return obs51.copy()
    return np.delete(obs51, GOAL_DIST_COL, axis=-1)


def add_obs_noise(obs: np.ndarray, env_cfg, rng) -> np.ndarray:
    """Noise on a (n, 50 or 51) base observation; returns a new array."""
    if env_cfg.obs_noise <= 0 and env_cfg.bearing_noise_deg <= 0:
        return obs
    obs = obs.copy()
    if env_cfg.obs_noise > 0:
        obs[:, :48] = np.clip(obs[:, :48] + rng.normal(0, env_cfg.obs_noise, obs[:, :48].shape),
                              0.0, 1.0)
    if env_cfg.bearing_noise_deg > 0:
        col = 49 if env_cfg.goal_obs == "bearing_distance" else 48
        b = obs[:, col] * math.pi + rng.normal(0, math.radians(env_cfg.bearing_noise_deg),
                                               len(obs))
        obs[:, col] = ((b + math.pi) % (2 * math.pi) - math.pi) / math.pi
    return obs


def actions_to_sim(a: np.ndarray, env_cfg=None, rng=None) -> np.ndarray:
    """§1.6: unbounded sample (n, 2) -> (turn in [-1, 1], speed in [0, 1])."""
    a = np.asarray(a, np.float64)
    if env_cfg is not None and env_cfg.action_noise > 0:
        a = a + rng.normal(0, env_cfg.action_noise, a.shape)
    return np.stack([np.clip(a[:, 0], -1, 1), np.clip((a[:, 1] + 1) / 2, 0, 1)], 1)


def randomise_headings(scene: Scene, rng) -> Scene:
    """§1.4: new scene with uniform headings; everything else untouched."""
    agents = [replace(a, heading=float(rng.uniform(-math.pi, math.pi))) for a in scene.agents]
    return replace(scene, agents=agents)


class ObsStack:
    """Last k observations per slot, oldest first, zero-padded (§1.3)."""

    def __init__(self, n_slots: int, k: int, d: int):
        self.k, self.d = k, d
        self.buf = np.zeros((n_slots, k, d), np.float32)

    def clear(self, mask):
        self.buf[np.asarray(mask, bool)] = 0.0

    def push(self, obs, mask=None) -> np.ndarray:
        """Shift in ``obs`` (n, d) for the slots in ``mask`` (all if None); return the stack."""
        if mask is None:
            mask = np.ones(len(self.buf), bool)
        self.buf[mask, :-1] = self.buf[mask, 1:]
        self.buf[mask, -1] = obs[mask]
        return self.view()

    def view(self) -> np.ndarray:
        return self.buf.reshape(len(self.buf), self.k * self.d).copy()

    def peek(self, obs) -> np.ndarray:
        """The stack as it would be after pushing ``obs`` for every slot (no state change)."""
        out = np.concatenate([self.buf[:, 1:], obs[:, None, :]], 1)
        return out.reshape(len(self.buf), self.k * self.d)


def sample_scene(env_cfg, sim_cfg, rng) -> Scene:
    """§1.2: one validated scene, headings per ``env_cfg.heading``."""
    while True:
        layout = str(rng.choice(env_cfg.layouts))
        n = int(rng.integers(env_cfg.agents[0], env_cfg.agents[1] + 1))
        m = int(rng.integers(env_cfg.obstacles[0], env_cfg.obstacles[1] + 1))
        scene = random_scene(rng, sim_cfg, layout=layout, n_agents=n, n_obstacles=m)
        if scene is None or not 1 <= len(scene.agents) <= env_cfg.max_agents:
            continue
        if env_cfg.heading == "random":
            scene = randomise_headings(scene, rng)
        return scene


class MultiAgentEnv:
    def __init__(self, env_cfg, sim_cfg, rng):
        self.cfg, self.sim_cfg, self.rng = env_cfg, sim_cfg, rng
        self.S = env_cfg.max_agents
        self.D = obs_dim(env_cfg)
        self._d = base_obs_dim(env_cfg)
        self.stack = (ObsStack(self.S, env_cfg.stack_k, self._d)
                      if env_cfg.memory == "stack" else None)
        self._records: list = []
        self.sim = None
        self.n = 0

    # -- helpers --------------------------------------------------------
    def _policy_obs(self, obs51_n: np.ndarray) -> np.ndarray:
        """(n, 51) simulator obs -> (S, base_dim) with noise, zero rows for empty slots."""
        base = add_obs_noise(to_policy_obs(obs51_n, self.cfg), self.cfg, self.rng)
        out = np.zeros((self.S, self._d), np.float32)
        out[:self.n] = base
        return out

    def _new_scene(self):
        scene = sample_scene(self.cfg, self.sim_cfg, self.rng)
        self.scene = scene
        self.sim = Simulator(scene, self.sim_cfg)
        self.n = len(scene.agents)
        self.ep_return = np.zeros(self.S)
        return self.sim.observe()

    def _present(self, base, alive, first):
        """Stacked (or plain) observation with dead and empty slots zeroed."""
        if self.stack is not None:
            self.stack.clear(first)
            obs = self.stack.push(base, alive | first)
        else:
            obs = base
        obs = obs.copy()
        obs[~alive] = 0.0
        return obs

    # -- API ------------------------------------------------------------
    def reset(self) -> dict:
        obs51 = self._new_scene()
        alive = np.zeros(self.S, bool)
        alive[:self.n] = True
        base = self._policy_obs(obs51)
        if self.stack is not None:
            self.stack.buf[:] = 0.0
        obs = self._present(base, alive, alive.copy())
        return {"obs": obs, "alive": alive, "first": alive.copy()}

    def step(self, actions: np.ndarray) -> dict:
        S, n = self.S, self.n
        sim = self.sim
        was_alive = np.zeros(S, bool)
        was_alive[:n] = sim.active
        sim_a = actions_to_sim(np.asarray(actions)[:n], self.cfg, self.rng)
        obs51, rew, done = sim.step(sim_a)

        reward = np.zeros(S, np.float32)
        reward[:n] = rew
        reward[~was_alive] = 0.0
        terminal = np.zeros(S, bool)
        truncated = np.zeros(S, bool)
        for i in range(n):
            if was_alive[i] and not sim.active[i]:
                if sim.outcome[i] == "timeout" and self.sim_cfg.timeout_penalty <= 0:
                    truncated[i] = True
                else:
                    terminal[i] = True
        self.ep_return += reward

        now_alive = np.zeros(S, bool)
        now_alive[:n] = sim.active
        base = self._policy_obs(obs51)
        # what the agents would see next, for bootstrapping at a timeout (§1.5)
        if self.stack is not None:
            final_obs = self.stack.peek(base)
        else:
            final_obs = base.copy()
        final_obs[~truncated] = 0.0

        first = np.zeros(S, bool)
        if done:
            self._records.append(self._record())
            obs51 = self._new_scene()
            now_alive = np.zeros(S, bool)
            now_alive[:self.n] = True
            first = now_alive.copy()
            base = self._policy_obs(obs51)
            if self.stack is not None:
                self.stack.buf[:] = 0.0
        obs = self._present(base, now_alive, first)
        return {"obs": obs, "alive": now_alive, "reward": reward, "terminal": terminal,
                "truncated": truncated, "first": first, "final_obs": final_obs,
                "was_alive": was_alive}

    def _record(self) -> dict:
        out = self.sim.outcome
        return {"layout": self.scene.layout, "n_agents": self.n,
                "goal": sum(o == "goal" for o in out),
                "collision": sum(o == "collision" for o in out),
                "timeout": sum(o == "timeout" for o in out),
                "return_sum": float(self.ep_return[:self.n].sum())}

    def episode_records(self) -> list:
        """Outcomes of scenes finished since the last call (then cleared)."""
        r, self._records = self._records, []
        return r
