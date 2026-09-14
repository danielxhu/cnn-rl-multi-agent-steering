"""2D steering-behavior simulator, scene generation, and rendering.

World is a 100x100 continuous square. Scenes follow the research statement:
hollow-circle agent with a heading line and a green FOV arc, black circular
obstacles, a blue goal dot, and (optionally) black hallway walls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageDraw

WORLD = 100.0
IMG = 512
SCALE = IMG / WORLD

AGENT_R = 2.0
GOAL_R = 1.4
GOAL_REACH_DIST = 3.0
FOV_DEG = 220.0
FOV_RANGE = 22.0
N_SECTORS = 16
MAX_TURN = math.radians(25.0)
MAX_SPEED = 2.5
MAX_STEPS = 300

# pixel classes
C_BG, C_WALL, C_FOV, C_OBST, C_GOAL, C_AGENT = 0, 1, 2, 3, 4, 5
CLASS_NAMES = ["background", "wall", "fov", "obstacle", "goal", "agent"]


@dataclass
class Scene:
    agent: tuple  # (x, y)
    heading: float  # radians, 0 = +x, CCW positive
    obstacles: list  # [(x, y, r)]
    goal: tuple  # (x, y)
    hallway: tuple | None = None  # (y_lo, y_hi) corridor open interval


def random_scene(rng: np.random.Generator, n_obstacles: int, hallway: bool = False) -> Scene:
    if hallway:
        h = rng.uniform(30.0, 46.0)
        y_lo = (WORLD - h) / 2 + rng.uniform(-8, 8)
        y_hi = y_lo + h
        y_band = (y_lo + AGENT_R + 2, y_hi - AGENT_R - 2)
        r_max = min(9.0, h / 2 - AGENT_R - 2.5)
    else:
        y_lo = y_hi = None
        y_band = (12.0, WORLD - 12.0)
        r_max = 10.0

    agent = (rng.uniform(6, 16), rng.uniform(*y_band))
    goal = (rng.uniform(WORLD - 16, WORLD - 5), rng.uniform(*y_band))
    heading = rng.uniform(-math.pi / 5, math.pi / 5)

    obstacles = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < 300:
        tries += 1
        r = rng.uniform(4.0, r_max)
        x = rng.uniform(25, WORLD - 25)
        if hallway:
            y = rng.uniform(y_lo + r + 1, y_hi - r - 1)
        else:
            y = rng.uniform(r + 2, WORLD - r - 2)
        if math.dist((x, y), agent) < r + AGENT_R + 8:
            continue
        if math.dist((x, y), goal) < r + GOAL_REACH_DIST + 6:
            continue
        if any(math.dist((x, y), (ox, oy)) < r + orr + 4 for ox, oy, orr in obstacles):
            continue
        obstacles.append((x, y, r))
    hw = (y_lo, y_hi) if hallway else None
    return Scene(agent=agent, heading=heading, obstacles=obstacles, goal=goal, hallway=hw)


# ---------------------------------------------------------------- geometry

def _ray_circle(px, py, dx, dy, cx, cy, r):
    """Nearest positive t where ray (p + t*d) hits circle, or inf."""
    fx, fy = px - cx, py - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return math.inf
    s = math.sqrt(disc)
    t1, t2 = -b - s, -b + s
    if t1 > 1e-9:
        return t1
    if t2 > 1e-9:
        return t2
    return math.inf


def _ray_to_bounds(px, py, dx, dy, x_lo, x_hi, y_lo, y_hi):
    """Distance along ray until leaving the open box (interior assumed)."""
    ts = []
    if dx > 1e-12:
        ts.append((x_hi - px) / dx)
    elif dx < -1e-12:
        ts.append((x_lo - px) / dx)
    if dy > 1e-12:
        ts.append((y_hi - py) / dy)
    elif dy < -1e-12:
        ts.append((y_lo - py) / dy)
    return min(ts) if ts else math.inf


class Simulator:
    """Steps an agent through a Scene with (turn, speed) actions.

    goal_tol: stop radius around the goal. When the scene comes from
    perception, a smaller tolerance leaves margin for parse error.
    """

    def __init__(self, scene: Scene, goal_tol: float = GOAL_REACH_DIST):
        self.scene = scene
        self.goal_tol = goal_tol
        self.reset()

    def reset(self):
        self.x, self.y = self.scene.agent
        self.heading = self.scene.heading
        self.speed = 0.0
        self.steps = 0
        self.trajectory = [(self.x, self.y)]
        self.done = False
        self.outcome = None  # "goal" | "collision" | "timeout"
        return self.observe()

    # -- sensing ------------------------------------------------------
    def _corridor(self):
        if self.scene.hallway:
            y_lo, y_hi = self.scene.hallway
            return 0.0, WORLD, y_lo, y_hi
        return 0.0, WORLD, 0.0, WORLD

    def observe(self) -> np.ndarray:
        obs = np.zeros(N_SECTORS + 3, dtype=np.float32)
        x_lo, x_hi, y_lo, y_hi = self._corridor()
        half = math.radians(FOV_DEG) / 2
        for k in range(N_SECTORS):
            ang = self.heading - half + (k + 0.5) * math.radians(FOV_DEG) / N_SECTORS
            dx, dy = math.cos(ang), math.sin(ang)
            t = _ray_to_bounds(self.x, self.y, dx, dy, x_lo, x_hi, y_lo, y_hi)
            for cx, cy, r in self.scene.obstacles:
                t = min(t, _ray_circle(self.x, self.y, dx, dy, cx, cy, r))
            obs[k] = min(t, FOV_RANGE) / FOV_RANGE
        gx, gy = self.scene.goal
        d = math.dist((self.x, self.y), (gx, gy))
        rel = math.atan2(gy - self.y, gx - self.x) - self.heading
        rel = (rel + math.pi) % (2 * math.pi) - math.pi
        obs[N_SECTORS] = min(d, WORLD * 1.5) / (WORLD * 1.5)
        obs[N_SECTORS + 1] = rel / math.pi
        obs[N_SECTORS + 2] = self.speed / MAX_SPEED
        return obs

    # -- dynamics -----------------------------------------------------
    def _collided(self) -> bool:
        x_lo, x_hi, y_lo, y_hi = self._corridor()
        if not (x_lo + AGENT_R < self.x < x_hi - AGENT_R):
            return True
        if not (y_lo + AGENT_R < self.y < y_hi - AGENT_R):
            return True
        return any(
            math.dist((self.x, self.y), (cx, cy)) < r + AGENT_R
            for cx, cy, r in self.scene.obstacles
        )

    def step(self, turn: float, speed: float):
        """turn, speed in [-1, 1] / [0, 1] normalized units."""
        assert not self.done
        prev_goal_dist = math.dist((self.x, self.y), self.scene.goal)
        self.heading += float(np.clip(turn, -1, 1)) * MAX_TURN
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        self.speed = float(np.clip(speed, 0, 1)) * MAX_SPEED
        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)
        self.steps += 1
        self.trajectory.append((self.x, self.y))

        goal_dist = math.dist((self.x, self.y), self.scene.goal)
        reward = -0.01 + 0.1 * (prev_goal_dist - goal_dist)
        if self._collided():
            self.done, self.outcome, reward = True, "collision", reward - 50.0
        elif goal_dist < self.goal_tol:
            self.done, self.outcome, reward = True, "goal", reward + 100.0
        elif self.steps >= MAX_STEPS:
            self.done, self.outcome = True, "timeout"
        return self.observe(), reward, self.done


# ---------------------------------------------------------------- rendering

def _to_px(x, y):
    return x * SCALE, (WORLD - y) * SCALE


def render(scene: Scene, path=None):
    """Render scene to (RGB image, label map). path: optional [(x,y)] red polyline."""
    img = Image.new("RGB", (IMG, IMG), "white")
    lab = Image.new("P", (IMG, IMG), C_BG)
    d, dl = ImageDraw.Draw(img), ImageDraw.Draw(lab)

    if scene.hallway:
        y_lo, y_hi = scene.hallway
        for y0w, y1w in [(y_hi, WORLD), (0.0, y_lo)]:
            x0, y0 = _to_px(0, y1w)
            x1, y1 = _to_px(WORLD, y0w)
            d.rectangle([x0, y0, x1, y1], fill="black")
            dl.rectangle([x0, y0, x1, y1], fill=C_WALL)

    # FOV arc (pie slice). PIL angles are degrees, clockwise from +x (y down).
    ax, ay = _to_px(*scene.agent)
    h_deg = -math.degrees(scene.heading)
    box = [ax - FOV_RANGE * SCALE, ay - FOV_RANGE * SCALE,
           ax + FOV_RANGE * SCALE, ay + FOV_RANGE * SCALE]
    a0, a1 = h_deg - FOV_DEG / 2, h_deg + FOV_DEG / 2
    d.pieslice(box, a0, a1, fill=(200, 213, 163), outline=(90, 100, 60))
    dl.pieslice(box, a0, a1, fill=C_FOV)

    for cx, cy, r in scene.obstacles:
        px, py = _to_px(cx, cy)
        b = [px - r * SCALE, py - r * SCALE, px + r * SCALE, py + r * SCALE]
        d.ellipse(b, fill="black")
        dl.ellipse(b, fill=C_OBST)

    gx, gy = _to_px(*scene.goal)
    gb = [gx - GOAL_R * SCALE, gy - GOAL_R * SCALE, gx + GOAL_R * SCALE, gy + GOAL_R * SCALE]
    d.ellipse(gb, fill=(20, 20, 230))
    dl.ellipse(gb, fill=C_GOAL)

    # label mask for the agent is drawn thicker than the visual stroke so the
    # tiny agent survives NEAREST-downsampling of the label map
    ab = [ax - AGENT_R * SCALE, ay - AGENT_R * SCALE, ax + AGENT_R * SCALE, ay + AGENT_R * SCALE]
    d.ellipse(ab, outline="black", width=3)
    dl.ellipse(ab, outline=C_AGENT, width=7)
    hx = ax + 1.6 * AGENT_R * SCALE * math.cos(scene.heading)
    hy = ay - 1.6 * AGENT_R * SCALE * math.sin(scene.heading)
    d.line([ax, ay, hx, hy], fill="black", width=3)
    dl.line([ax, ay, hx, hy], fill=C_AGENT, width=6)

    d.rectangle([1, 1, IMG - 2, IMG - 2], outline=(60, 60, 60), width=2)

    if path is not None and len(path) > 1:
        pts = [_to_px(x, y) for x, y in path]
        d.line(pts, fill=(230, 30, 30), width=3)
    return img, lab


def render_small(scene: Scene, size=128):
    """(image, labels) downsampled for the CNN: uint8 HxWx3 and HxW."""
    img, lab = render(scene)
    img = img.resize((size, size), Image.BILINEAR)
    lab = lab.resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8), np.asarray(lab, dtype=np.uint8)


# Showcase scenes echoing the two figures in the research statement.
STATEMENT_SINGLE = Scene(agent=(14, 52), heading=-0.35, obstacles=[(52, 52, 10)], goal=(88, 53))
STATEMENT_HALLWAY = Scene(
    agent=(8, 50), heading=0.0,
    obstacles=[(28, 46, 5.0), (52, 52, 7.5), (74, 58, 6.0)],
    goal=(94, 49), hallway=(28.0, 72.0),
)
