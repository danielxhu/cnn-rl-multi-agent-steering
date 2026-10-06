"""Self-contained test suite (no pytest needed):  python tests/run_tests.py"""
from __future__ import annotations

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import C_AGENT, C_GOAL, Config
from geometry import ray_capsule, ray_circle, ray_segment
from layouts import LAYOUTS, random_scene
from physics import Simulator, greedy_action
from reachability import bottleneck_width, is_solvable
from render import render
from scene import Agent, Obstacle, Region, Scene, boundary_walls

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# --------------------------------------------------------------- geometry

@test
def test_ray_primitives():
    seg = ((5.0, -1.0), (5.0, 1.0))
    assert abs(ray_segment(0, 0, 1, 0, seg) - 5.0) < 1e-9
    assert ray_segment(0, 0, -1, 0, seg) == math.inf, "hits behind the ray must miss"
    assert ray_segment(0, 0, 0, 1, seg) == math.inf
    assert abs(ray_circle(0, 0, 1, 0, 5, 0, 1) - 4.0) < 1e-9
    # a capsule is hit at its surface, half a thickness before the centre line
    assert abs(ray_capsule(0, 0, 1, 0, seg, 1.5) - 3.5) < 1e-9


@test
def test_vectorised_rays_match_the_scalar_reference():
    """physics.py re-implements ray casting in batched form for the hot loop.

    Two implementations of the same maths drift apart silently, so the batched
    one is pinned against the readable scalar one in geometry.py.
    """
    from physics import _ray_circles, _ray_segments

    rng = np.random.default_rng(0)
    p = np.array([31.0, 47.0])
    ang = rng.uniform(-math.pi, math.pi, 16)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)

    centers = rng.uniform(10, 90, (5, 2))
    radii = rng.uniform(3, 9, 5)
    batched = _ray_circles(p, dirs, centers, radii)
    for k, (dx, dy) in enumerate(dirs):
        ref = min(ray_circle(p[0], p[1], dx, dy, c[0], c[1], r)
                  for c, r in zip(centers, radii))
        assert abs(batched[k] - ref) < 1e-6 or (batched[k] == ref == math.inf), \
            f"circle ray {k}: batched {batched[k]} vs scalar {ref}"

    a = rng.uniform(5, 95, (6, 2))
    b = rng.uniform(5, 95, (6, 2))
    batched = _ray_segments(p, dirs, a, b - a)
    for k, (dx, dy) in enumerate(dirs):
        ref = min(ray_segment(p[0], p[1], dx, dy, (tuple(u), tuple(v)))
                  for u, v in zip(a, b))
        assert abs(batched[k] - ref) < 1e-6 or (batched[k] == ref == math.inf), \
            f"segment ray {k}: batched {batched[k]} vs scalar {ref}"


# ----------------------------------------------------------- reachability

def _slit(cfg, gap):
    """Two huge circles leaving a horizontal slit of exactly `gap` at y=50."""
    rb = 60.0
    return Scene(
        walls=boundary_walls(cfg.world),
        obstacles=[Obstacle((50, 50 + gap / 2 + rb), rb),
                   Obstacle((50, 50 - gap / 2 - rb), rb)],
        agents=[Agent((10, 50), 0.0)],
        start_region=Region(4, 44, 16, 56),
        goal_region=Region(84, 44, 96, 56),
    )


@test
def test_narrow_slit_is_not_passable():
    """The case a naive centre-in-obstacle grid check gets wrong.

    A 1.5-radius gap is open on the grid but a disk of radius R cannot fit.
    """
    cfg = Config()
    gap = 1.5 * cfg.agent_radius            # 3.0 < agent diameter 4.0
    assert not is_solvable(cfg, _slit(cfg, gap)), "1.5R slit must be impassable"
    assert bottleneck_width(cfg, _slit(cfg, gap)) < 2 * cfg.agent_radius


@test
def test_wide_slit_is_passable():
    cfg = Config()
    s = _slit(cfg, 3.0 * cfg.agent_radius)
    assert is_solvable(cfg, s)
    assert abs(bottleneck_width(cfg, s) - 6.0) < 0.5


@test
def test_obstacle_sealing_a_doorway_is_detected():
    """A door wide enough on its own, plugged by one obstacle."""
    cfg = Config()
    walls = boundary_walls(cfg.world) + [((50, 0), (50, 38)), ((50, 62), (50, 100))]
    base = Scene(walls=walls, agents=[Agent((12, 50), 0.0)],
                 start_region=Region(4, 40, 18, 60), goal_region=Region(84, 40, 96, 60))
    assert is_solvable(cfg, base), "the empty doorway should be passable"
    plugged = base.with_obstacle(Obstacle((50, 50), 13.0))
    assert not is_solvable(cfg, plugged), "an obstacle filling the door must close it"


# ------------------------------------------------------------ coordinates

@test
def test_world_y_up_maps_to_image_y_down():
    """World y increases upward; image row 0 is the top.  Easy to invert."""
    cfg = Config()
    s = Scene(walls=boundary_walls(100) + [((0, 20), (100, 20)), ((0, 40), (100, 40))],
              agents=[Agent((10, 30), 0.0)],
              start_region=Region(5, 24, 18, 36), goal_region=Region(82, 24, 96, 36),
              layout="corridor")
    _, lab = render(s, cfg)
    n = cfg.img_size
    rows = np.flatnonzero(lab[:, n // 2] == 0)
    assert rows.mean() > n / 2, "a corridor low in the world must sit low in the image"
    assert abs(rows.mean() - (100 - 30) / 100 * n) < 20
    cols = np.flatnonzero((lab == C_GOAL).any(0))
    assert cols.min() > n / 2, "a goal on the world's right must be on the image's right"


@test
def test_scene_json_round_trip():
    cfg = Config()
    rng = np.random.default_rng(3)
    s = random_scene(rng, cfg, layout="crossing", n_agents=6, n_obstacles=2)
    back = Scene.from_dict(s.as_dict(cfg, "t"))
    assert len(back.agents) == len(s.agents) and len(back.groups) == len(s.groups)
    assert abs(back.agents[0].heading - s.agents[0].heading) < 1e-12
    assert back.agents[-1].goal_region is not None, "per-agent goals must survive"


# ---------------------------------------------------------------- scenes

@test
def test_every_layout_returns_solvable_scenes():
    cfg = Config()
    rng = np.random.default_rng(0)
    for name in LAYOUTS:
        for _ in range(6):
            s = random_scene(rng, cfg, layout=name, n_agents=4, n_obstacles=3)
            assert s is not None, f"{name}: sampler gave up"
            assert is_solvable(cfg, s), f"{name}: returned an unsolvable scene"
            assert s.bottleneck >= cfg.min_bottleneck, f"{name}: bottleneck too tight"
            assert len(s.agents) >= 1


@test
def test_agents_never_overlap_at_spawn():
    cfg = Config()
    rng = np.random.default_rng(5)
    for hard in (False, True):
        for name in LAYOUTS:
            s = random_scene(rng, cfg, layout=name, n_agents=6, n_obstacles=2, hard=hard)
            for i, a in enumerate(s.agents):
                for b in s.agents[i + 1:]:
                    assert math.dist(a.pos, b.pos) >= 2 * cfg.agent_radius, \
                        f"{name}: agents overlap at spawn (hard={hard})"


@test
def test_observation_shape_and_bounds():
    cfg = Config()
    rng = np.random.default_rng(1)
    s = random_scene(rng, cfg, layout="corridor", n_agents=4, n_obstacles=3)
    obs = Simulator(s, cfg).reset()
    assert obs.shape == (4, cfg.obs_dim) == (4, 51)
    assert obs[:, :48].min() >= 0.0 and obs[:, :48].max() <= 1.0 + 1e-6
    assert -1.0 <= obs[:, 49].min() <= obs[:, 49].max() <= 1.0


@test
def test_simulation_terminates_and_reports_both_metric_families():
    cfg = Config()
    rng = np.random.default_rng(2)
    s = random_scene(rng, cfg, layout="open", n_agents=4, n_obstacles=3)
    sim = Simulator(s, cfg)
    sim.reset()
    while not sim.done:
        sim.step(np.stack([greedy_action(sim, i) for i in range(len(sim.pos))]))
    st = sim.stats()
    assert sim.steps <= cfg.max_steps
    assert all(o is not None for o in sim.outcome)
    for key in ("success_rate", "collision_violations", "finish_time", "remaining_agents"):
        assert key in st


@test
def test_collisions_can_be_counted_instead_of_terminating():
    cfg = Config()
    cfg.terminate_on_collision = False
    rng = np.random.default_rng(2)
    s = random_scene(rng, cfg, layout="corridor", n_agents=5, n_obstacles=3)
    sim = Simulator(s, cfg)
    sim.reset()
    while not sim.done:
        sim.step(np.stack([greedy_action(sim, i) for i in range(len(sim.pos))]))
    assert "collision" not in sim.outcome, "agents should survive collisions in this mode"


@test
def test_label_map_contains_every_class():
    cfg = Config()
    rng = np.random.default_rng(7)
    s = random_scene(rng, cfg, layout="corridor", n_agents=4, n_obstacles=3)
    _, lab = render(s, cfg)
    for cls in range(6):
        assert (lab == cls).any(), f"class {cls} missing from the label map"
    assert (lab == C_AGENT).sum() > 0


# ---------------------------------------------------------------- sketch

@test
def test_sketch_is_deterministic():
    """Same scene and seed -> identical image, labels and drawn JSON."""
    from sketch import sketch
    cfg = Config()
    s = random_scene(np.random.default_rng(11), cfg, layout="crossing", n_agents=6, n_obstacles=3)
    s.seed = 1234
    img1, lab1, d1 = sketch(s, cfg)
    img2, lab2, d2 = sketch(s, cfg, rng=np.random.default_rng(1234))
    assert np.array_equal(np.asarray(img1), np.asarray(img2))
    assert np.array_equal(lab1, lab2)
    assert d1.as_dict(cfg, "t") == d2.as_dict(cfg, "t")
    s.seed = 1235
    img3, _, d3 = sketch(s, cfg)
    assert not np.array_equal(np.asarray(img1), np.asarray(img3)), "the seed must matter"
    assert d1.as_dict(cfg, "t") != d3.as_dict(cfg, "t")

    # the widened style (v2) is deterministic too, and its digital medium has no camera
    from dataclasses import replace
    from sketch import SKETCH_V2, SKETCH_V3, SKETCH_V4
    for st in (SKETCH_V2, SKETCH_V3, SKETCH_V4):
        a = sketch(s, cfg, style=st)
        b = sketch(s, cfg, style=st)
        assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and np.array_equal(a[1], b[1])
        assert a[2].as_dict(cfg, "t") == b[2].as_dict(cfg, "t")
    img, lab, d, parts = sketch(s, cfg, style=replace(SKETCH_V2, digital_prob=1.0), parts=True)
    assert np.allclose(parts["H"], np.eye(3)), "a tablet drawing is not photographed"
    assert np.median(np.asarray(img)) >= 250, "a tablet drawing is white"


@test
def test_sketch_labels_contain_every_class():
    import render as render_mod
    import sketch as sketch_mod
    cfg = Config()
    for x, y in ((0, 0), (12.5, 80.0), (100, 100)):
        assert sketch_mod._to_px(cfg, x, y) == render_mod._to_px(cfg, x, y), "one y flip"
    rng = np.random.default_rng(7)
    s = random_scene(rng, cfg, layout="corridor", n_agents=4, n_obstacles=3)
    s.seed = 7
    img, lab, _ = sketch_mod.sketch(s, cfg, style=sketch_mod.SketchStyle(hatch_prob=1.0))
    assert lab.shape == (cfg.img_size, cfg.img_size) and np.asarray(img).shape == lab.shape + (3,)
    for cls in range(6):
        assert (lab == cls).any(), f"class {cls} missing from the sketch label map"
    # corridor: the dead space above and below is hatched, so labelled wall
    assert (lab == 1).mean() > 0.2, "hatched dead space must be labelled wall"
    _, lab_blank, _ = sketch_mod.sketch(s, cfg, style=sketch_mod.SketchStyle(hatch_prob=0.0))
    assert (lab_blank == 1).mean() < 0.1, "blank dead space must stay background"


@test
def test_free_layouts_are_valid_and_varied():
    """walls / rooms (not in LAYOUTS): solvable, start and goal boxes off the
    walls and in varied places, so no 'start is on the left' shortcut."""
    from layouts import FREE_LAYOUTS, MIXES
    cfg = Config()
    rng = np.random.default_rng(5)
    assert MIXES["mixed"] == LAYOUTS, "the old mix must not change"
    for name in FREE_LAYOUTS:
        starts_x = []
        for _ in range(8):
            s = random_scene(rng, cfg, layout=name, n_agents=4, n_obstacles=2)
            assert s is not None, f"{name}: sampler gave up"
            assert is_solvable(cfg, s), name
            for reg in (s.start_region, s.goal_region):
                for a, b in s.walls[4:]:
                    for t in np.linspace(0, 1, 50):
                        q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                        assert reg.distance_to(q) > cfg.wall_half, f"{name}: a wall runs through a region"
            starts_x.append(s.start_region.center[0])
        assert np.ptp(starts_x) > 30, f"{name}: start boxes always in the same place"


def main():
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("  PASS  %s" % fn.__name__)
        except Exception:
            failed += 1
            print("  FAIL  %s" % fn.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
