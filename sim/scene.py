"""Scene data structures.

A Scene is pure geometry: walls, obstacles, regions and agents.  It carries no
simulation state -- `physics.Simulator` owns that -- so a Scene can be rendered,
serialised and re-loaded without side effects.

Goals (DESIGN_SKETCH §9.4): an agent's goal is its own `goal_region` if set
(`crossing`), else the scene's exit set `RegionSet(exits)` if the scene
declares `exits` (several start / exit boxes; an agent succeeds by reaching any
exit), else the scene's `goal_region` (every scene made before exits existed).
`RegionSet` has `Region`'s goal interface, so the simulator treats both alike.
Optional fields (`Agent.radius`, `Scene.starts` / `exits`) are written to JSON
only when set, so older scenes serialise exactly as before.
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


class RegionSet:
    """Several regions acting as one goal: reaching any member counts.

    Same goal interface as `Region` (`contains`, `distance_to`, `nearest_point`),
    so `physics.Simulator` needs no change: the observation's goal distance and
    bearing, and the reward's progress, refer to the nearest member.
    """

    def __init__(self, regions):
        self.regions = tuple(regions)
        if not self.regions:
            raise ValueError("RegionSet needs at least one region")

    def contains(self, p) -> bool:
        return any(r.contains(p) for r in self.regions)

    def distance_to(self, p) -> float:
        return min(r.distance_to(p) for r in self.regions)

    def nearest_point(self, p):
        """The nearest point of the nearest member (p itself when inside one)."""
        return min((r.nearest_point(p) for r in self.regions),
                   key=lambda q: math.hypot(p[0] - q[0], p[1] - q[1]))

    def __iter__(self):
        return iter(self.regions)

    def __len__(self):
        return len(self.regions)

    def __eq__(self, other):
        return isinstance(other, RegionSet) and self.regions == other.regions

    def __hash__(self):
        return hash(self.regions)

    def __repr__(self):
        return "RegionSet(%r)" % (list(self.regions),)


def members(goal) -> list:
    """The regions of a goal: a Region's [itself], a RegionSet's members."""
    return list(goal.regions) if isinstance(goal, RegionSet) else [goal]


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
    """One agent. `goal_region` overrides the scene goal; None means use the scene's.

    `radius`: the radius it was *drawn* with (sketch v5, world units); None
    for every other scene, whose agents are drawn at the configured radius.
    """

    pos: Point
    heading: float                      # radians, 0 = +x, CCW positive
    goal_region: Region | None = None
    radius: float | None = None

    def as_dict(self):
        d = {"pos": [self.pos[0], self.pos[1]], "heading": self.heading}
        if self.goal_region is not None:
            d["goal_region"] = self.goal_region.as_list()
        if self.radius is not None:
            d["radius"] = self.radius
        return d

    @staticmethod
    def from_dict(d):
        g = d.get("goal_region")
        return Agent(
            pos=tuple(d["pos"]),
            heading=d["heading"],
            goal_region=Region.from_list(g) if g else None,
            radius=d.get("radius"),
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
    # several start / exit boxes (None: the single start_region / goal_region);
    # when set, start_region / goal_region are their first members
    starts: list | None = None
    exits: list | None = None
    # sketch v6: one {"kind": "S" | "G", "pos": [x, y], "height": h} per drawn box
    letters: list | None = None

    def goal_for(self, agent: Agent):
        """The agent's own goal, else any exit, else the scene goal (DESIGN_SKETCH §9.4)."""
        if agent.goal_region is not None:
            return agent.goal_region
        if self.exits:
            return RegionSet(self.exits)
        return self.goal_region

    def start_boxes(self) -> list:
        """Every start box: `starts`, else the groups' starts, else start_region."""
        if self.starts is not None:
            return list(self.starts)
        if self.groups:
            return [s for s, _ in self.groups if s is not None]
        return [self.start_region] if self.start_region is not None else []

    def exit_boxes(self) -> list:
        """Every goal box: `exits`, else the groups' goals, else goal_region."""
        if self.exits is not None:
            return list(self.exits)
        if self.groups:
            return [g for _, g in self.groups if g is not None]
        return [self.goal_region] if self.goal_region is not None else []

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
        if self.starts is not None:
            d["starts"] = [r.as_list() for r in self.starts]
        if self.exits is not None:
            d["exits"] = [r.as_list() for r in self.exits]
        if self.letters is not None:
            d["letters"] = [dict(v) for v in self.letters]
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
            starts=[Region.from_list(r) for r in d["starts"]] if d.get("starts") is not None else None,
            exits=[Region.from_list(r) for r in d["exits"]] if d.get("exits") is not None else None,
            letters=[dict(v) for v in d["letters"]] if d.get("letters") is not None else None,
        )


def boundary_walls(world: float) -> list:
    """The four world edges as wall centre lines (they are real walls)."""
    return [
        ((0.0, 0.0), (world, 0.0)),
        ((world, 0.0), (world, world)),
        ((world, world), (0.0, world)),
        ((0.0, world), (0.0, 0.0)),
    ]
