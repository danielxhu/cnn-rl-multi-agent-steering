"""Self-contained test suite (no pytest needed):  python tests/run_tests.py

CPU only.  Scenes come from ``sim.layouts.random_scene`` with fixed seeds and
labels from ``sim.render.render``, so ground-truth label maps stand in for a
perfect network and the extractor is tested in isolation from training.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _paths  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import C_AGENT, C_OBST, C_WALL, N_CLASSES, Config  # noqa: E402
from layouts import LAYOUTS, random_scene  # noqa: E402
from physics import Simulator, greedy_action  # noqa: E402
from render import _to_px, render  # noqa: E402
from scene import Agent, Obstacle, Region, Scene, boundary_walls  # noqa: E402

from extract import (detect_agents_ring, extract, pixel_geometry, sim_config,  # noqa: E402
                     parsed_to_json, parsed_from_json)
from metrics import (aggregate, heading_err_deg, match_points, region_iou,  # noqa: E402
                     scene_metrics, wall_surface_error)
from targets import instance_targets, world_to_px  # noqa: E402

EXAMPLES = _paths.SIM_DIR / "examples"
SCRATCH = _paths.PERCEPTION_DIR / "runs" / "_tests"
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def _world(cfg):
    return {"size": cfg.world, "img_size": cfg.img_size, "agent_radius": cfg.agent_radius,
            "wall_thickness": cfg.wall_thickness, "fov_deg": cfg.fov_deg, "fov_range": cfg.fov_range}


def _downsample(lab, size):
    return cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST)


_SCENES = {}


def _roundtrip_scenes(cfg, per_layout=3, seed=42):
    """3 scenes per layout, ≤ 8 agents, rendered once and cached across tests."""
    key = (per_layout, seed)
    if key not in _SCENES:
        rng = np.random.default_rng(seed)
        out = []
        for layout in LAYOUTS:
            for _ in range(per_layout):
                s = random_scene(rng, cfg, layout=layout, n_agents=int(rng.integers(1, 9)),
                                 n_obstacles=int(rng.integers(0, 5)))
                assert s is not None, f"{layout}: sampler gave up"
                _, lab = render(s, cfg)
                out.append((s, lab))
        _SCENES[key] = out
    return _SCENES[key]


def _match(gt_scene, parsed, tol=2.0):
    return match_points([a.pos for a in gt_scene.agents], [a.pos for a in parsed.scene.agents], tol)


# ---------------------------------------------------------------- targets

@test
def test_targets_peak_at_agent_centres():
    cfg = Config()
    rng = np.random.default_rng(1)
    s = random_scene(rng, cfg, layout="open", n_agents=6, n_obstacles=2)
    world = _world(cfg)
    for size in (128, 256, 512):
        heat, dirs, mask = instance_targets(s, world, size)
        assert heat.shape == (1, size, size) and dirs.shape == (2, size, size) and mask.shape == (1, size, size)
        h = heat[0]
        r_px = cfg.agent_radius * size / cfg.world
        centres = [world_to_px(cfg.world, size, *a.pos) for a in s.agents]
        # every local maximum of the heat map is within 1 px of an agent centre
        peaks = np.argwhere((h >= cv2.dilate(h, np.ones((3, 3), np.uint8))) & (h > 0.5))
        for iy, ix in peaks:
            assert min(math.hypot(ix - cx, iy - cy) for cx, cy in centres) <= 1.0 + 1e-6, \
                f"{size}: heat peak at ({ix},{iy}) is not at an agent centre"
        for a, (cx, cy) in zip(s.agents, centres):
            iy, ix = int(round(cy)), int(round(cx))
            assert h[iy, ix] == 1.0, f"{size}: no peak at agent centre"
            assert abs(dirs[0, iy, ix] - math.cos(a.heading)) < 1e-6
            assert abs(dirs[1, iy, ix] - math.sin(a.heading)) < 1e-6
            assert mask[0, iy, ix] == 1.0
        expected = len(s.agents) * math.pi * r_px ** 2
        assert abs(mask.sum() - expected) < 0.2 * expected + 4 * len(s.agents), \
            f"{size}: mask area {mask.sum()} vs {expected:.0f}"


# ------------------------------------------------------------ extraction

@test
def test_extract_roundtrip_512():
    cfg = Config()
    world = _world(cfg)
    for s, lab in _roundtrip_scenes(cfg):
        p = extract(lab, world)
        assert p is not None, f"{s.layout}: parse failed on ground truth"
        pairs = _match(s, p, tol=0.5)
        assert len(pairs) == len(s.agents) == len(p.scene.agents), \
            f"{s.layout}: {len(pairs)} of {len(s.agents)} agents matched, {len(p.scene.agents)} predicted"
        for gi, pi in pairs:
            err = heading_err_deg(p.scene.agents[pi].heading, s.agents[gi].heading)
            assert err < 5.0, f"{s.layout}: heading error {err:.1f} deg"
        assert len(p.scene.obstacles) == len(s.obstacles), \
            f"{s.layout}: {len(p.scene.obstacles)} obstacles vs {len(s.obstacles)}"
        for o in s.obstacles:
            q = min(p.scene.obstacles, key=lambda q: math.dist(q.c, o.c))
            assert math.dist(q.c, o.c) < 0.5 and abs(q.r - o.r) < 0.5, \
                f"{s.layout}: obstacle centre err {math.dist(q.c, o.c):.2f} radius err {abs(q.r - o.r):.2f}"
        for gs, gg in s.groups:
            assert max(region_iou(gs, ps) for ps, _ in p.scene.groups) > 0.95, f"{s.layout}: start region"
            assert max(region_iou(gg, pg) for _, pg in p.scene.groups) > 0.95, f"{s.layout}: goal region"
        werr = wall_surface_error(lab == C_WALL, p.scene.walls, cfg.world)
        assert werr < 0.5, f"{s.layout}: wall surface error {werr:.2f}"


@test
def test_extract_roundtrip_128():
    cfg = Config()
    world = _world(cfg)
    n_gt = n_matched = 0
    for s, lab512 in _roundtrip_scenes(cfg):
        p = extract(_downsample(lab512, 128), world)
        assert p is not None, f"{s.layout}: parse failed at 128"
        n_gt += len(s.agents)
        n_matched += len(_match(s, p))
        assert len(p.scene.obstacles) == len(s.obstacles), f"{s.layout}: obstacle count at 128"
    assert n_matched >= 0.9 * n_gt, f"agent recall at 128: {n_matched}/{n_gt}"


@test
def test_ring_detector_separates_touching_agents():
    cfg = Config()
    d = 2 * cfg.agent_radius + cfg.hard_gap                 # 4.15: `hard` spacing
    s = Scene(walls=boundary_walls(cfg.world),
              agents=[Agent((30.0, 50.0), 0.3), Agent((30.0 + d, 50.0), -0.2)],
              start_region=Region(20, 40, 45, 60), goal_region=Region(80, 40, 95, 60),
              groups=[(Region(20, 40, 45, 60), Region(80, 40, 95, 60))])
    _, lab = render(s, cfg, fill_unreachable=False)
    geo = pixel_geometry(_world(cfg), cfg.img_size)
    dets = detect_agents_ring(lab == C_AGENT, geo["r_px"], geo["lw_px"])
    assert len(dets) == 2, f"expected 2 detections, got {len(dets)}"
    p = extract(lab, _world(cfg))
    pairs = _match(s, p, tol=0.5)
    assert len(pairs) == 2 and len(p.scene.agents) == 2


@test
def test_crossing_groups_and_goal_assignment():
    cfg = Config()
    rng = np.random.default_rng(3)
    s = random_scene(rng, cfg, layout="crossing", n_agents=6, n_obstacles=2)
    _, lab = render(s, cfg)
    p = extract(lab, _world(cfg))
    assert len(p.scene.groups) == 2, f"{len(p.scene.groups)} groups recovered"
    pairs = _match(s, p, tol=0.5)
    assert len(pairs) == len(s.agents)
    for gi, pi in pairs:
        iou = region_iou(s.goal_for(s.agents[gi]), p.scene.goal_for(p.scene.agents[pi]))
        assert iou > 0.9, f"agent {gi}: parsed goal region IoU {iou:.2f}"
    # the JSON form round-trips
    back = parsed_from_json(parsed_to_json(p, "t"))
    assert len(back.scene.agents) == len(p.scene.agents) and back.world["wall_thickness"] == 0.0


@test
def test_obstacle_fit_ignores_interior_and_wall_contact():
    cfg = Config()
    world = _world(cfg)
    # (a) interior relabelled as wall
    rng = np.random.default_rng(5)
    s = random_scene(rng, cfg, layout="open", n_agents=2, n_obstacles=3)
    _, lab = render(s, cfg)
    lab = lab.copy()
    yy, xx = np.mgrid[0:cfg.img_size, 0:cfg.img_size]
    for o in s.obstacles:
        cx, cy = _to_px(cfg, *o.c)
        inner = np.hypot(xx - cx, yy - cy) < o.r * cfg.scale - 4
        lab[inner & (lab == C_OBST)] = C_WALL
    p = extract(lab, world)
    assert len(p.scene.obstacles) == len(s.obstacles)
    for o in s.obstacles:
        q = min(p.scene.obstacles, key=lambda q: math.dist(q.c, o.c))
        assert math.dist(q.c, o.c) < 0.5 and abs(q.r - o.r) < 0.5, "interior relabelling moved the fit"
    # (b) an obstacle overlapping a wall
    obs = Obstacle((50.0, 34.0), 6.0)
    s2 = Scene(walls=boundary_walls(cfg.world) + [((50.0, 0.0), (50.0, 36.0)), ((50.0, 64.0), (50.0, 100.0))],
               obstacles=[obs], agents=[Agent((10.0, 50.0), 0.0)],
               start_region=Region(4.5, 20, 18, 80), goal_region=Region(82, 20, 95.5, 80),
               groups=[(Region(4.5, 20, 18, 80), Region(82, 20, 95.5, 80))])
    _, lab2 = render(s2, cfg)
    p2 = extract(lab2, world)
    assert len(p2.scene.obstacles) == 1, f"{len(p2.scene.obstacles)} obstacles"
    q = p2.scene.obstacles[0]
    assert math.dist(q.c, obs.c) < 0.5 and abs(q.r - obs.r) < 0.5, \
        f"wall contact: centre err {math.dist(q.c, obs.c):.2f} radius err {abs(q.r - obs.r):.2f}"


@test
def test_wall_surface_matches_gt():
    cfg = Config()
    world = _world(cfg)
    rng = np.random.default_rng(11)
    for layout in ("corridor", "room"):
        s = random_scene(rng, cfg, layout=layout, n_agents=3, n_obstacles=1)
        _, lab = render(s, cfg)
        p = extract(lab, world)
        err = wall_surface_error(lab == C_WALL, p.scene.walls, cfg.world)
        assert err < 0.5, f"{layout}: symmetric wall surface error {err:.2f}"
        # no parsed edge lies away from the GT wall surface (dead space makes no interior walls)
        from metrics import rasterise_walls, surface_of_mask
        gt_surf = surface_of_mask(lab == C_WALL)
        dt = cv2.distanceTransform((gt_surf == 0).astype(np.uint8), cv2.DIST_L2, 5)
        pr = rasterise_walls(p.scene.walls, cfg.world, cfg.img_size)
        worst = float(dt[pr > 0].max()) / cfg.scale
        assert worst < 1.0, f"{layout}: a parsed wall edge is {worst:.2f} units from any GT surface"
        assert 4 <= len(p.scene.walls) <= 40, f"{layout}: {len(p.scene.walls)} segments"


@test
def test_parsed_scene_runs_in_simulator():
    cfg = Config()
    rng = np.random.default_rng(8)
    s = random_scene(rng, cfg, layout="doorway", n_agents=4, n_obstacles=2)
    _, lab = render(s, cfg)
    p = extract(lab, _world(cfg))
    sim = Simulator(p.scene, sim_config(cfg))
    assert sim.cfg.wall_thickness == 0.0
    obs = sim.reset()
    assert obs.shape == (len(p.scene.agents), cfg.obs_dim) and np.isfinite(obs).all()
    for i in range(len(p.scene.agents)):
        assert not sim._hits_static(i), f"agent {i} collides with parsed geometry at t=0"
    for _ in range(20):
        obs, rewards, done = sim.step(np.stack([greedy_action(sim, i) for i in range(len(sim.pos))]))
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        if done:
            break


# ------------------------------------------------------------------ data

@test
def test_dataset_shapes_and_downsampling():
    import torch
    from data import SceneDataset, class_frequencies
    present512 = {}
    for size in (512, 256, 128):
        ds = SceneDataset(EXAMPLES, size, instance_targets=(size == 128))
        assert len(ds) == 7 and len(ds.scenes) == 7 and len(ds.worlds) == 7 and len(set(ds.layouts)) == 7
        for i in range(len(ds)):
            b = ds[i]
            assert b["image"].shape == (3, size, size) and b["image"].dtype == torch.float32
            assert 0.0 <= float(b["image"].min()) and float(b["image"].max()) <= 1.0
            assert b["labels"].shape == (size, size) and b["labels"].dtype == torch.int64
            assert int(b["labels"].max()) < N_CLASSES
            classes = set(np.unique(b["labels"].numpy()).tolist())
            if size == 512:
                present512[i] = classes
            elif size == 128:
                assert present512[i] <= classes, f"scene {i}: classes lost at 128: {present512[i] - classes}"
                for k, shape in (("heat", (1, size, size)), ("dir", (2, size, size)), ("dir_mask", (1, size, size))):
                    assert b[k].shape == shape and b[k].dtype == torch.float32
    freq = class_frequencies(SceneDataset(EXAMPLES, 128))
    assert freq.shape == (N_CLASSES,) and abs(freq.sum() - 1.0) < 1e-6
    cache = EXAMPLES / "class_freq.json"
    if cache.exists():
        cache.unlink()                           # keep sim/examples pristine


# ----------------------------------------------------------------- model

@test
def test_model_shapes_and_param_count():
    import torch
    from model import build_model, count_params
    from runconfig import ModelConfig
    n = count_params(build_model(ModelConfig(arch="unet24x4")))
    assert 1_080_000 <= n <= 1_090_000, f"unet24x4 has {n} params"
    assert count_params(build_model(ModelConfig(arch="unet24x4", instance_head=True))) == n + 75
    for inst in (False, True):
        m = build_model(ModelConfig(arch="unet16x4", instance_head=inst)).eval()
        for size in (128, 256, 512):
            with torch.no_grad():
                out = m(torch.zeros(1, 3, size, size))
            assert out["seg"].shape == (1, 6, size, size)
            if inst:
                assert out["heat"].shape == (1, 1, size, size) and out["dir"].shape == (1, 2, size, size)
                assert float(out["heat"].min()) >= 0.0 and float(out["heat"].max()) <= 1.0
                norm = out["dir"].pow(2).sum(1).sqrt()
                assert float((norm - 1).abs().max()) < 1e-4, "dir must be unit length"
            else:
                assert "heat" not in out and "dir" not in out


# --------------------------------------------------------------- metrics

@test
def test_metrics_matching():
    gt = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)]
    pred = [(10.5, 10.0), (20.0, 11.0), (50.0, 50.0)]
    pairs = match_points(gt, pred, tol=2.0)
    assert sorted(pairs) == [(0, 0), (1, 1)], pairs
    assert match_points([], pred, 2.0) == [] and match_points(gt, [], 2.0) == []
    # nearest-first: the closest pair wins even when listed last
    assert match_points([(0.0, 0.0)], [(1.5, 0.0), (0.1, 0.0)], 2.0) == [(0, 1)]

    cfg = Config()
    rng = np.random.default_rng(21)
    s = random_scene(rng, cfg, layout="two_doorway", n_agents=5, n_obstacles=3)
    _, lab = render(s, cfg)
    p = extract(lab, _world(cfg))
    m, rows = scene_metrics(s, lab, p, lab, _world(cfg), 512, sid="t")
    assert m["agent_f1"] == 1.0 and m["agent_precision"] == 1.0 and m["agent_recall"] == 1.0
    assert m["agent_count_ok"] == 1.0 and m["agent_center_err"] < 0.5 and m["heading_err_deg"] < 5.0
    assert m["obst_count_ok"] == 1.0 and m["obst_center_err"] < 0.5 and m["obst_radius_err"] < 0.5
    assert m["pixacc"] == 1.0 and m["miou"] == 1.0 and m["wall_iou"] == 1.0
    assert m["region_iou"] > 0.95 and m["group_pairing_ok"] == 1.0 and m["scene_usable"] == 1.0
    assert m["wall_surface_err"] < 0.5 and m["parse_fail"] == 0.0
    assert len(rows) == 5 and all(r["matched"] == 1.0 for r in rows)
    # synthetic mismatch: drop one predicted agent and add a far one
    p.scene.agents[0].pos = (99.0, 1.0)
    m2, _ = scene_metrics(s, lab, p, lab, _world(cfg), 512)
    assert abs(m2["agent_precision"] - 4 / 5) < 1e-9 and abs(m2["agent_recall"] - 4 / 5) < 1e-9
    assert m2["agent_count_ok"] == 1.0 and m2["scene_usable"] == 0.0
    m3, rows3 = scene_metrics(s, lab, None, lab, _world(cfg), 512)
    assert m3["parse_fail"] == 1.0 and m3["agent_recall"] == 0.0 and all(r["matched"] == 0.0 for r in rows3)
    agg = aggregate([m, m2, m3], rows + rows3)
    assert "overall" in agg and "per_layout" in agg and len(agg["spacing"]) == 5
    assert abs(agg["overall"]["parse_fail"] - 1 / 3) < 1e-9


# ------------------------------------------------------------ checkpoint

@test
def test_checkpoint_roundtrip():
    import torch
    from data import SceneDataset
    from losses import total_loss
    from model import build_model
    from predict import Predictor
    from runconfig import RunConfig
    from train import save_checkpoint

    cfg = RunConfig()
    cfg.data.root, cfg.data.train, cfg.data.val, cfg.data.size = str(_paths.SIM_DIR), "examples", "examples", 128
    cfg.model.arch, cfg.model.instance_head = "unet16x4", True
    torch.manual_seed(0)
    ds = SceneDataset(EXAMPLES, 128, instance_targets=True, limit=4)
    model = build_model(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    w = torch.ones(N_CLASSES)
    for _ in range(2):
        batch = torch.utils.data.default_collate([ds[i] for i in range(4)])
        loss, _ = total_loss(model(batch["image"]), batch, w, cfg.train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "ckpt_test.pt"
    save_checkpoint(path, model, cfg, opt, epoch=1, world=ds.worlds[0])

    pr = Predictor(path, device="cpu")
    assert pr.size == 128 and pr.cfg.model.arch == "unet16x4"
    img = ds.image_uint8(0)
    out = pr(img)
    model.eval()
    with torch.no_grad():
        ref = model(ds[0]["image"][None])
    ref_probs = torch.softmax(ref["seg"], 1)[0].numpy()
    assert out["labels"].shape == (128, 128) and out["labels"].dtype == np.uint8
    assert np.array_equal(out["labels"], ref_probs.argmax(0).astype(np.uint8))
    assert np.allclose(out["probs"], ref_probs, atol=1e-6)
    assert np.allclose(out["heat"], ref["heat"][0, 0].numpy(), atol=1e-6)
    assert out["dir"].shape == (2, 128, 128)
    shutil.rmtree(SCRATCH, ignore_errors=True)


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
