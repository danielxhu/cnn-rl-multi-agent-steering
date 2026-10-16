"""Self-contained test suite (no pytest needed):  python tests/run_tests.py

    python tests/run_tests.py --write-hashes   # re-record tests/reference_hashes.json

``reference_hashes.json`` pins the scenes every existing style and layout
generates (DESIGN_SKETCH §9.7). It was recorded with the code *before* sketch
v5 (15 Oct 2026); re-record it only on purpose. Scene JSON hashes (floats
rounded to 1e-6) are compared everywhere; image and label hashes only where
numpy, OpenCV and Pillow have the recorded versions, since anti-aliasing and
JPEG may differ between library versions.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

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
    from sketch import SKETCH_V2, SKETCH_V3, SKETCH_V4, SKETCH_V5
    for st in (SKETCH_V2, SKETCH_V3, SKETCH_V4, SKETCH_V5):
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


# ------------------------------------------------- sketch v5 / several exits

REFERENCE = Path(__file__).with_name("reference_hashes.json")
_COMMON = ["--agents", "1-8", "--obstacles", "0-6"]
HASH_SETS = {
    # (DESIGN_SKETCH §9.7) 14 scenes per older style, each on its own dataset layout
    "sketch": ["--n", "14", "--style", "sketch", "--layout", "mixed", "--seed", "5"] + _COMMON,
    "sketch2": ["--n", "14", "--style", "sketch2", "--layout", "mixed", "--seed", "5"] + _COMMON,
    "sketch3": ["--n", "14", "--style", "sketch3", "--layout", "mixed", "--seed", "5"] + _COMMON,
    "sketch4": ["--n", "14", "--style", "sketch4", "--layout", "mixed2", "--seed", "5"] + _COMMON,
    # the default renderer: every family, mixed2, and hard / augmented scenes
    **{"render_" + name: ["--n", "4", "--layout", name, "--seed", "7"] + _COMMON
       for name in LAYOUTS + ["walls", "rooms"]},
    "render_mixed2": ["--n", "14", "--layout", "mixed2", "--seed", "340"] + _COMMON,
    "render_mixed_hard_aug3": ["--n", "14", "--layout", "mixed", "--agents", "4-8", "--obstacles", "0-6",
                               "--hard-ratio", "0.5", "--jitter-color", "45", "--jitter-line-width", "1.5",
                               "--jitter-background", "35", "--noise-std", "12", "--blur-sigma", "1.5",
                               "--seed", "301"],
}


def _versions() -> dict:
    import cv2
    import PIL
    return {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__}


def _rounded(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, list):
        return [_rounded(v) for v in o]
    if isinstance(o, dict):
        return {k: _rounded(v) for k, v in o.items()}
    return o


def scene_set_hashes(args) -> dict:
    """{"json", "pixels"}: sha256 over what `generate.py args` writes (in a temp dir)."""
    import generate
    hj, hp = hashlib.sha256(), hashlib.sha256()
    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            generate.main(list(args) + ["--out", d])
        for p in sorted(Path(d).glob("scene_*.json")):
            hj.update(json.dumps(_rounded(json.loads(p.read_text())), sort_keys=True).encode())
            hp.update(p.with_suffix(".png").read_bytes())
            hp.update(p.with_name(p.stem + "_labels.png").read_bytes())
    return {"json": hj.hexdigest(), "pixels": hp.hexdigest()}


def write_reference_hashes():
    rec = {"versions": _versions(), "sets": {k: scene_set_hashes(a) for k, a in HASH_SETS.items()}}
    REFERENCE.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote %s (%d sets)" % (REFERENCE, len(rec["sets"])))


def _check_hashes(names):
    ref = json.loads(REFERENCE.read_text())
    pixels = ref["versions"] == _versions()
    for name in names:
        h = scene_set_hashes(HASH_SETS[name])
        assert h["json"] == ref["sets"][name]["json"], f"{name}: scenes differ from the reference"
        if pixels:
            assert h["pixels"] == ref["sets"][name]["pixels"], f"{name}: images / labels differ"
    if not pixels:
        print("        (pixel hashes not compared: library versions %s differ from the reference's %s)"
              % (_versions(), ref["versions"]))


@test
def test_older_sketch_styles_unchanged():
    """v1-v4 draw exactly what they drew before sketch v5 (14 scenes each)."""
    _check_hashes(["sketch", "sketch2", "sketch3", "sketch4"])


@test
def test_existing_layouts_unchanged_by_exits():
    """Every LAYOUTS family, walls / rooms, mixed2 and hard augmented scenes hash
    as before exits existed: E5 and the RL stage stay comparable."""
    from layouts import MIXES
    assert MIXES["mixed2"] == LAYOUTS + ["walls", "walls", "walls", "rooms", "rooms"]
    assert MIXES["mixed3"][:len(MIXES["mixed2"])] == MIXES["mixed2"]
    _check_hashes([k for k in HASH_SETS if k.startswith("render_")])


@test
def test_region_set_goal():
    """Two exits: the observation points at the nearer one, yet an agent that
    walks into the farther one reaches its goal (any exit counts)."""
    from scene import RegionSet
    cfg = Config()
    near, far = Region(40, 44, 52, 56), Region(84, 44, 96, 56)
    s = Scene(walls=boundary_walls(cfg.world), agents=[Agent((30.0, 50.0), 0.0)],
              start_region=Region(24, 44, 36, 56), goal_region=near,
              starts=[Region(24, 44, 36, 56)], exits=[near, far])
    goal = s.goal_for(s.agents[0])
    assert isinstance(goal, RegionSet) and goal.distance_to((30.0, 50.0)) == 10.0
    assert goal.nearest_point((70.0, 50.0)) == (84, 50.0) and goal.contains((90.0, 50.0))
    assert is_solvable(cfg, s) and bottleneck_width(cfg, s) > cfg.min_bottleneck

    # heading north: the nearer exit is due west (bearing +90°), the farther one north-east
    s.agents[0] = Agent((46.0, 30.0), math.pi / 2)
    s.exits = [Region(20, 24, 32, 36), Region(70, 80, 82, 92)]
    sim = Simulator(s, cfg)
    obs = sim.reset()
    assert obs.shape == (1, cfg.obs_dim) == (1, 51)
    assert abs(obs[0, 48] - 14.0 / (cfg.world * 1.5)) < 1e-6, "distance to the nearer exit"
    assert abs(obs[0, 49] - 0.5) < 1e-6, "bearing to the nearer (west) exit"
    # walk north-east into the farther exit
    target = math.atan2(86 - 30, 76 - 46)
    sim.heading[0] = target
    while not sim.done:
        sim.step(np.array([[0.0, 1.0]]))
    assert sim.outcome[0] == "goal", sim.outcome
    assert s.exits[1].contains(tuple(sim.pos[0])) and sim.stats()["success_rate"] == 1.0
    # JSON round trip keeps the boxes; an older scene has neither key
    back = Scene.from_dict(json.loads(json.dumps(s.as_dict(cfg, "t"))))
    assert back.exits == s.exits and back.starts == s.starts
    assert "exits" not in random_scene(np.random.default_rng(0), cfg, "open", 2, 1).as_dict(cfg)


@test
def test_boxrooms_valid():
    """boxrooms: solvable for every agent (any exit), every start box holds an
    agent; rooms with no box (decoys) in at least half the scenes; boxes inside
    rooms and outside them; several starts and exits in some scenes."""
    import layouts
    from layouts import MULTI_LAYOUTS
    cfg = Config()
    rng = np.random.default_rng(12)
    decoys = inside = outside = 0
    n = 30
    for _ in range(n):
        spec = layouts._boxrooms(rng, cfg)
        assert spec.name == "boxrooms" and spec.rooms, "boxrooms fell back to open"
        boxes = spec.starts + spec.exits
        held = [any(r.contains(b.center) for b in boxes) for r in spec.rooms]
        decoys += not all(held)
        for b in boxes:
            if any(r.contains(b.center) for r in spec.rooms):
                inside += 1
            else:
                outside += 1
    assert decoys >= n / 2, f"decoy rooms in only {decoys} of {n} scenes"
    assert inside >= 5 and outside >= 5, (inside, outside)
    multi = 0
    for name in MULTI_LAYOUTS:
        for i in range(8):
            s = random_scene(rng, cfg, layout=name, n_agents=int(rng.integers(1, 9)),
                             n_obstacles=int(rng.integers(0, 5)), close=i % 2 == 0)
            assert s is not None, f"{name}: sampler gave up"
            assert is_solvable(cfg, s), f"{name}: an agent cannot reach any exit"
            assert s.bottleneck >= cfg.min_bottleneck
            for b in s.start_boxes():
                assert any(b.contains(a.pos) for a in s.agents), f"{name}: a start box without agents"
            for i_, a in enumerate(s.agents):
                for b_ in s.agents[i_ + 1:]:
                    assert math.dist(a.pos, b_.pos) >= 2 * cfg.agent_radius
            if s.exits:
                multi += 1
                assert s.groups == [] and s.goal_region == s.exits[0]
            else:
                assert len(s.groups) == 1 and s.starts is None
    assert multi >= 8, f"only {multi} of 24 scenes have several boxes"


@test
def test_close_agents_and_sketch5_json():
    """sketch v5 packs a start box at the hard spacing (world positions stay
    ≥ 2R apart), records drawn radii, and a several-box scene draws every box
    and writes them back as starts / exits."""
    from sketch import SKETCH_V5, SKETCH_V4, sketch
    cfg = Config()
    rng = np.random.default_rng(2)
    s = random_scene(rng, cfg, layout="open", n_agents=8, n_obstacles=0, close=True)
    d = sorted(min(math.dist(a.pos, b.pos) for b in s.agents if b is not a) for a in s.agents)
    hard = 2 * cfg.agent_radius + cfg.hard_gap
    assert d[0] >= 2 * cfg.agent_radius and sum(x < hard + 0.05 for x in d) >= 6, d
    s.seed = 5
    _, _, drawn = sketch(s, cfg, style=SKETCH_V5)
    assert all(a.radius is not None and 0.3 < a.radius < 5.0 for a in drawn.agents)
    assert all("radius" not in a for a in sketch(s, cfg, style=SKETCH_V4)[2].as_dict()["agents"])
    for _ in range(20):
        m = random_scene(rng, cfg, layout="walls_multi", n_agents=6, n_obstacles=2)
        if m.exits and len(m.exits) + len(m.starts) >= 4:
            break
    m.seed = 6
    _, lab, drawn = sketch(m, cfg, style=SKETCH_V5)
    assert len(drawn.starts) == len(m.starts) and len(drawn.exits) == len(m.exits) and drawn.groups == []
    for got, want in zip(drawn.starts + drawn.exits, m.starts + m.exits):
        assert abs(got.center[0] - want.center[0]) < 3 and abs(got.center[1] - want.center[1]) < 3
    from config import C_START
    import cv2
    n_lab, _ = cv2.connectedComponents((lab == C_START).astype(np.uint8))
    assert n_lab - 1 >= len(m.starts), "every start box is drawn"


def main():
    if "--write-hashes" in sys.argv:
        write_reference_hashes()
        return 0
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
