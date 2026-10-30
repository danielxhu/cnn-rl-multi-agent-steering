"""Parameters of the policy stage: environment, network, PPO and the run.

Owns every tunable of E2 (DESIGN.md §1–§4, §8.1). Nothing here imports torch,
so workers and the CLIs can read a config cheaply.

Invariants:
- Every field is JSON-serialisable (tuples are stored as lists and restored on
  load), so a ``config.json`` fully describes a run.
- ``EnvConfig.memory`` and ``NetConfig.memory`` must agree; ``RunConfig``
  keeps them in sync (``RunConfig.make``).
- ``RunConfig.reward`` holds overrides of the simulator's reward fields
  (``REWARD_FIELDS``, DESIGN_E3.md §3.2); ``{}`` is E2's default reward. It is
  the only way a run changes the simulator, and it is saved with the run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

LAYOUTS_E2 = ("open", "corridor", "room", "doorway", "two_doorway", "cul_de_sac",
              "crossing", "boxrooms", "walls_multi", "rooms_multi")
MEMORIES = ("mlp", "stack", "gru")
# sim Config fields that shape the reward only (never the observation or the
# dynamics): a run may override them, evaluation always resets them
REWARD_FIELDS = ("goal_bonus", "collision_penalty", "agent_collision_penalty", "step_cost",
                 "progress_weight", "timeout_penalty", "progress_metric", "geo_cell")


@dataclass
class EnvConfig:
    layouts: tuple = LAYOUTS_E2
    agents: tuple = (2, 8)                  # n_agents uniform in [lo, hi] per scene
    obstacles: tuple = (0, 6)               # n_obstacles uniform in [lo, hi]
    max_agents: int = 8                     # slots per env
    heading: str = "random"                 # "random" | "generator"
    goal_obs: str = "bearing"               # "bearing" (50 numbers) | "bearing_distance" (51)
    memory: str = "gru"                     # only "stack" changes the observation
    stack_k: int = 4
    obs_noise: float = 0.0
    bearing_noise_deg: float = 0.0
    action_noise: float = 0.0


@dataclass
class NetConfig:
    memory: str = "gru"
    hidden: int = 128
    log_std_init: float = -0.5


@dataclass
class PPOConfig:
    lr: float = 3e-4
    epochs: int = 4
    n_minibatches: int = 4
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    T: int = 128                            # steps per iteration per env
    L: int = 32                             # sequence length of the update
    W: int = 10                             # rollout worker processes (0 = inline)
    E: int = 8                              # envs per worker
    total_steps: int = 50_000_000           # budget in alive agent transitions
    device: str = "auto"                    # learner device: auto | cpu | cuda


@dataclass
class RunConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 0
    val_every: int = 25
    ckpt_every: int = 25
    reward: dict = field(default_factory=dict)      # sim reward overrides (E3)

    @staticmethod
    def make(memory="gru", seed=0, reward=None, **ppo_overrides) -> "RunConfig":
        ppo = PPOConfig(**ppo_overrides)
        return RunConfig(env=EnvConfig(memory=memory), net=NetConfig(memory=memory),
                         ppo=ppo, seed=seed, reward=dict(reward or {}))

    def to_dict(self, sim_cfg=None) -> dict:
        d = asdict(self)
        if sim_cfg is not None:
            d["sim"] = asdict(sim_cfg)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        def build(cls, raw):
            names = {f.name for f in fields(cls)}
            kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()
                  if k in names}
            return cls(**kw)
        return RunConfig(env=build(EnvConfig, d["env"]), net=build(NetConfig, d["net"]),
                         ppo=build(PPOConfig, d["ppo"]), seed=d["seed"],
                         val_every=d.get("val_every", 25), ckpt_every=d.get("ckpt_every", 25),
                         reward=dict(d.get("reward") or {}))

    def save(self, path, sim_cfg=None):
        Path(path).write_text(json.dumps(self.to_dict(sim_cfg), indent=2))

    @staticmethod
    def load(path) -> "RunConfig":
        return RunConfig.from_dict(json.loads(Path(path).read_text()))


def sim_config_from_dict(d: dict):
    """Rebuild a sim ``Config`` from ``RunConfig.to_dict(sim_cfg)['sim']``."""
    from config import Config, StyleConfig
    d = dict(d)
    style = StyleConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                           for k, v in d.pop("style", {}).items()})
    return Config(style=style, **{k: (tuple(v) if isinstance(v, list) else v)
                                  for k, v in d.items()})


def sim_config(reward=None) -> "Config":
    """The simulator config of a run: defaults, collisions terminate the agent,
    plus the run's reward overrides (``RunConfig.reward``)."""
    from dataclasses import replace
    from config import Config
    cfg = Config()
    assert cfg.terminate_on_collision, "E2 needs terminate_on_collision=True"
    if reward:
        unknown = set(reward) - set(REWARD_FIELDS)
        if unknown:
            raise ValueError(f"not reward fields of sim Config: {sorted(unknown)}")
        cfg = replace(cfg, **reward)
    return cfg


def default_reward(sim_cfg) -> "Config":
    """``sim_cfg`` with every reward field back at its default: the physics alone.

    Evaluation metrics count outcomes, which the reward never changes, so every
    controller is evaluated on this (and no geodesic fields are built)."""
    from dataclasses import replace
    from config import Config
    base = Config()
    return replace(sim_cfg, **{k: getattr(base, k) for k in REWARD_FIELDS})


def physics_fields(sim_cfg) -> dict:
    """The fields of a sim Config that are not reward or rendering style."""
    from dataclasses import asdict
    d = asdict(sim_cfg) if not isinstance(sim_cfg, dict) else dict(sim_cfg)
    return {k: v for k, v in d.items() if k not in REWARD_FIELDS and k != "style"}
