"""Scene data structures.

A Scene is pure geometry: walls, obstacles, regions and agents.  It carries no
simulation state -- `physics.Simulator` owns that -- so a Scene can be rendered,
serialised and re-loaded without side effects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

Point = tuple  # (x, y)
Segment = tuple  # ((x1, y1), (x2, y2)) -- wall centre line


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle, world coordinates, y increasing upward."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        object.__setattr__(self, "x0", min(self.x0, self.x1))
        object.__setattr__(self, "x1", max(self.x0, self.x1))
        object.__setattr__(self, "y0", min(self.y0, self.y1))
        object.__setattr__(self, "y1", max(self.y0, self.y1))

    @property
    def center(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    def contains(self, p) -> bool:
        return self.x0 <= p[0] <= self.x1 and self.y0 <= p[1] <= self.y1

    def nearest_point(self, p):
        """Closest point of the rectangle to p (p itself when inside)."""
        return (min(max(p[0], self.x0), self.x1), min(max(p[1], self.y0), self.y1))

    def distance_to(self, p) -> float:
        nx, ny = self.nearest_point(p)
        return math.hypot(p[0] - nx, p[1] - ny)

    def shrink(self, m: float) -> "Region":
        return Region(self.x0 + m, self.y0 + m, self.x1 - m, self.y1 - m)

    def as_list(self):
        return [self.x0, self.y0, self.x1, self.y1]

    @staticmethod
    def from_list(v):
        return Region(*v)


@dataclass
class Obstacle:
    c: Point
    r: float

    def as_dict(self):
        return {"c": [self.c[0], self.c[1]], "r": self.r}

    @staticmethod
    def from_dict(d):
        return Obstacle(c=tuple(d["c"]), r=d["r"])


@dataclass
class Agent:
    """One agent. `goal_region` overrides the scene goal; None means use the scene's."""

    pos: Point
    heading: float                      # radians, 0 = +x, CCW positive
    goal_region: Region | None = None

    def as_dict(self):
        d = {"pos": [self.pos[0], self.pos[1]], "heading": self.heading}
        if self.goal_region is not None:
            d["goal_region"] = self.goal_region.as_list()
        return d

    @staticmethod
    def from_dict(d):
        g = d.get("goal_region")
        return Agent(
            pos=tuple(d["pos"]),
            heading=d["heading"],
            goal_region=Region.from_list(g) if g else None,
        )


@dataclass
class Scene:
    walls: list = field(default_factory=list)        # list[Segment], centre lines
    obstacles: list = field(default_factory=list)    # list[Obstacle]
    agents: list = field(default_factory=list)       # list[Agent]
    start_region: Region | None = None
    goal_region: Region | None = None
    layout: str = "open"
    # every (start, goal) pair; `crossing` holds two
    groups: list = field(default_factory=list)
    seed: int | None = None
    bottleneck: float | None = None                  # widest passable diameter

    def goal_for(self, agent: Agent) -> Region:
        return agent.goal_region or self.goal_region

    def with_obstacle(self, obs: Obstacle) -> "Scene":
        return replace(self, obstacles=self.obstacles + [obs])

    # --- serialisation ---------------------------------------------------
    def as_dict(self, cfg=None, sid=None):
        d = {
            "id": sid,
            "seed": self.seed,
            "layout": self.layout,
            "bottleneck": self.bottleneck,
            "walls": [[list(a), list(b)] for a, b in self.walls],
            "obstacles": [o.as_dict() for o in self.obstacles],
            "start_region": self.start_region.as_list() if self.start_region else None,
            "goal_region": self.goal_region.as_list() if self.goal_region else None,
            "groups": [[a.as_list(), b.as_list()] for a, b in self.groups],
            "agents": [a.as_dict() for a in self.agents],
        }
        if cfg is not None:
            d["world"] = {
                "size": cfg.world,
                "img_size": cfg.img_size,
                "agent_radius": cfg.agent_radius,
                "wall_thickness": cfg.wall_thickness,
                "fov_deg": cfg.fov_deg,
                "fov_range": cfg.fov_range,
                "note": "world y increases upward; image row 0 is y = world",
            }
        return d

    @staticmethod
    def from_dict(d):
        return Scene(
            walls=[(tuple(a), tuple(b)) for a, b in d["walls"]],
            obstacles=[Obstacle.from_dict(o) for o in d["obstacles"]],
            agents=[Agent.from_dict(a) for a in d["agents"]],
            start_region=Region.from_list(d["start_region"]) if d.get("start_region") else None,
            goal_region=Region.from_list(d["goal_region"]) if d.get("goal_region") else None,
            groups=[(Region.from_list(a), Region.from_list(b))
                    for a, b in d.get("groups", [])],
            layout=d.get("layout", "open"),
            seed=d.get("seed"),
            bottleneck=d.get("bottleneck"),
        )


def boundary_walls(world: float) -> list:
    """The four world edges as wall centre lines (they are real walls)."""
    return [
        ((0.0, 0.0), (world, 0.0)),
        ((world, 0.0), (world, world)),
        ((world, world), (0.0, world)),
        ((0.0, world), (0.0, 0.0)),
    ]
