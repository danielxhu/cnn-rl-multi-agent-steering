"""All tunable parameters for the scene generator.

Everything the generator can vary lives here. `generate.py` exposes the most
commonly tuned fields on the command line; anything else can be overridden with
a JSON file passed via ``--config``.  See README.md for a tuning guide.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace

# ------------------------------------------------------------------ classes

# Pixel classes in the label map (draw order; later classes overwrite earlier).
C_BG, C_WALL, C_OBST, C_START, C_GOAL, C_AGENT = range(6)
N_CLASSES = 6
CLASS_NAMES = ["background", "wall", "obstacle", "start_region", "goal_region", "agent"]

# Entity categories of the per-sector observation; order fixes the obs layout.
E_OBSTACLE, E_WALL, E_AGENT = range(3)
N_ENTITY_CLASSES = 3
ENTITY_NAMES = ["obstacle", "wall", "agent"]


@dataclass
class StyleConfig:
    """Rendering appearance.  Jitter fields drive domain randomisation."""

    background: tuple = (255, 255, 255)
    wall_color: tuple = (0, 0, 0)
    obstacle_color: tuple = (0, 0, 0)
    agent_color: tuple = (92, 45, 145)          # dark purple: stays visible on black
    start_color: tuple = (232, 151, 58)         # orange
    start_alpha: int = 64                       # ~25%
    goal_color: tuple = (43, 92, 230)           # blue
    goal_alpha: int = 89                        # ~35%

    agent_line_width: float = 4.0               # px at img_size=512
    heading_len_mult: float = 2.2               # heading line length, in agent radii
    region_border_width: float = 2.0
    region_dash: float = 10.0                   # dash length in px, start region

    # --- domain randomisation (all disabled by default) ---
    jitter_line_width: float = 0.0              # +/- px
    jitter_color: float = 0.0                   # +/- per RGB channel, 0-255
    jitter_background: float = 0.0              # +/- grey level on the paper
    noise_std: float = 0.0                      # gaussian pixel noise, 0-255
    blur_sigma: float = 0.0                     # gaussian blur radius in px


@dataclass
class Config:
    """Geometry, generation and evaluation parameters."""

    # --- world -----------------------------------------------------------
    world: float = 100.0                        # square world side, world units
    img_size: int = 512                         # rendered image side, px

    # --- agent -----------------------------------------------------------
    agent_radius: float = 2.0
    fov_deg: float = 220.0                      # matches the research statement
    fov_range: float = 22.0
    n_sectors: int = 16
    max_turn_deg: float = 25.0
    max_speed: float = 2.5
    max_steps: int = 300

    # --- walls -----------------------------------------------------------
    wall_thickness: float = 3.0                 # capsule thickness; shared by render and physics

    # --- obstacles -------------------------------------------------------
    obstacle_r_min: float = 4.0
    obstacle_r_max: float = 10.0
    obstacle_gap: float = 4.0                   # min free gap between obstacles
    obstacle_clear_agents: float = 8.0          # min gap to any agent at spawn
    obstacle_clear_goal: float = 6.0            # min gap to the goal region

    # --- reward / episode ------------------------------------------------
    goal_bonus: float = 100.0
    collision_penalty: float = 50.0
    agent_collision_penalty: float = 25.0
    step_cost: float = 0.01
    progress_weight: float = 0.1
    terminate_on_collision: bool = True         # False: count collisions, keep stepping

    # --- agent spawn spacing ---------------------------------------------
    agent_gap: float = 1.5                      # extra clearance beyond 2 * agent_radius

    # --- regions ---------------------------------------------------------
    region_margin: float = 4.0                  # keep regions off the walls

    # --- reachability ----------------------------------------------------
    cell: float = 0.25                          # occupancy grid cell, world units
    min_bottleneck: float = 5.2                 # 2.6 * agent_radius; reject below
    check_bottleneck: bool = True

    # --- generation ------------------------------------------------------
    hard_ratio: float = 0.0                     # fraction of near-tangent scenes
    hard_gap: float = 0.15                      # extra spacing in hard scenes
    max_place_tries: int = 300

    style: StyleConfig = field(default_factory=StyleConfig)

    # --- derived ---------------------------------------------------------
    @property
    def scale(self) -> float:
        """World units -> pixels."""
        return self.img_size / self.world

    @property
    def max_turn(self) -> float:
        return math.radians(self.max_turn_deg)

    @property
    def wall_half(self) -> float:
        return self.wall_thickness / 2.0

    @property
    def obs_dim(self) -> int:
        return self.n_sectors * N_ENTITY_CLASSES + 3

    # --- serialisation ---------------------------------------------------
    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            raw = json.load(fh)
        style = StyleConfig(**raw.pop("style", {}))
        return cls(style=style, **raw)

    def merged(self, **overrides):
        """Return a copy with the given non-None fields replaced."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        style_over = clean.pop("style", None)
        cfg = replace(self, **clean)
        if style_over:
            cfg = replace(cfg, style=replace(cfg.style, **style_over))
        return cfg


DEFAULT = Config()
