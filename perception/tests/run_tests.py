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

from config import C_AGENT, C_GOAL, C_OBST, C_START, C_WALL, N_CLASSES, Config  # noqa: E402
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


# ------------------------------------------------------------ drawing sheet

@test
def test_rectify_recovers_sheet_and_id():
    """A tilted, unevenly lit, JPEG-compressed photo of a drawing sheet is
    rectified to the world square within 2 px of 512, and its id is read."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(_paths.PERCEPTION_DIR / "scripts"))
    import make_sheets
    import sheet
    from rectify import rectify, world_square_matrix

    cfg = Config()
    s = random_scene(np.random.default_rng(4), cfg, layout="doorway", n_agents=3, n_obstacles=2)
    fig, ax = make_sheets.new_page()
    make_sheets.draw_furniture(ax, 37, "t", cfg)
    make_sheets.draw_scene(ax, s, cfg)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, format="png")
    plt.close(fig)
    page = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[120, 90], [w + 60, 150], [w + 20, h + 170], [60, h + 110]])
    G = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(page, G, (w + 200, h + 260), borderValue=(60, 80, 70))
    yy, xx = np.mgrid[0:photo.shape[0], 0:photo.shape[1]] / max(photo.shape)
    photo = np.clip(photo * (0.6 + 0.4 * (1 - (xx - .2) ** 2 - (yy - .3) ** 2))[..., None], 0, 255).astype(np.uint8)
    photo = cv2.imdecode(cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], cv2.IMREAD_COLOR)

    img, info = rectify(photo)
    assert img.shape == (512, 512, 3)
    assert info["sheet_id"] == 37, f"id read as {info['sheet_id']}"
    s_ = 120 / 25.4
    raster_to_mm = np.array([[1 / s_, 0, 0], [0, -1 / s_, sheet.PAGE_H], [0, 0, 1]])
    M_true = world_square_matrix(512) @ raster_to_mm @ np.linalg.inv(G)
    q = np.float32([[x, y] for x in (0, 256, 512) for y in (0, 256, 512)])
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(q[None], np.linalg.inv(M_true)), info["homography"])[0]
    err = float(np.abs(back - q).max())
    assert err < 2.0, f"rectification off by {err:.2f} px"


@test
def test_multi_root_training_set():
    """--train accepts several directories: datasets concatenate, class
    frequencies combine by pixel counts, and configs whose data.train is a
    string (every run before this change) still load and train the same."""
    import json
    from data import SceneDataset, class_frequencies, concat_datasets, parts_of
    from runconfig import RunConfig, train_dirs

    old = RunConfig().to_dict()
    old["data"]["train"] = "train_aug3"
    cfg = RunConfig.from_dict(json.loads(json.dumps(old)))
    assert cfg.data.train == "train_aug3" and train_dirs(cfg.data) == ["train_aug3"]
    one = RunConfig.from_args(["--train", "train_aug3", "--out", "x"])
    assert one.data.train == "train_aug3", "one --train stays a string"
    two = RunConfig.from_args(["--train", "train_sketch", "train_aug3", "--out", "x"])
    assert two.data.train == ["train_sketch", "train_aug3"]
    assert RunConfig.from_dict(two.to_dict()).data.train == ["train_sketch", "train_aug3"]

    a = SceneDataset(EXAMPLES, 128)
    b = SceneDataset(EXAMPLES, 128, limit=3)
    ab = concat_datasets([a, b])
    assert len(ab) == len(a) + len(b) and parts_of(ab) == [a, b]
    assert concat_datasets([a]) is a
    assert ab[len(a)]["labels"].shape == (128, 128)
    counts = sum(np.bincount(d.labels_uint8(i).ravel(), minlength=N_CLASSES)[:N_CLASSES]
                 for d in (a, b) for i in range(len(d)))
    assert np.allclose(class_frequencies(ab), counts / counts.sum())


# -------------------------------------------------------- sketch (hand-drawn)

_SKETCHES = []
_SKETCHES_BY_STYLE = {}
SKETCH_STYLE_NAMES = ("sketch", "sketch2", "sketch3", "sketch4", "sketch5", "sketch6", "sketch7")


def _sketch_scenes(cfg, style="sketch"):
    """One sketch scene per layout (2–8 agents), with the per-agent ring and
    tail maps the renderer drew; rendered once per style and cached across tests.
    v4 adds the free-form layouts; v5 adds the multi-box ones too, and packs
    every second scene's start boxes (close agents); v6 draws the same families
    with the hard cases of mixed4."""
    cache = _SKETCHES if style == "sketch" else _SKETCHES_BY_STYLE.setdefault(style, [])
    if not cache:
        from layouts import FREE_LAYOUTS, MIX_OPTIONS, MULTI_LAYOUTS
        from sketch import STYLES, sketch
        rng = np.random.default_rng(77)
        extra = {"sketch4": FREE_LAYOUTS, "sketch5": FREE_LAYOUTS + MULTI_LAYOUTS * 2,
                 "sketch6": FREE_LAYOUTS + MULTI_LAYOUTS * 2,
                 "sketch7": FREE_LAYOUTS + MULTI_LAYOUTS * 2}.get(style, [])
        for k, layout in enumerate(LAYOUTS + extra):
            s = random_scene(rng, cfg, layout=layout, n_agents=int(rng.integers(2, 9)),
                             n_obstacles=int(rng.integers(1, 5)),
                             close=style in ("sketch5", "sketch6", "sketch7") and k % 2 == 0,
                             opts={"sketch6": MIX_OPTIONS["mixed4"], "sketch7": MIX_OPTIONS["mixed5"]}.get(style))
            assert s is not None, f"{layout}: sampler gave up"
            s.seed = 9000 + k
            img, lab, drawn, parts = sketch(s, cfg, parts=True, style=STYLES[style])
            cache.append((drawn, lab, parts))
    return cache


def _tail_off_centre(parts, k, cx, cy) -> bool:
    """v7: the tail starts at the circle's edge or beyond it (not at the centre).
    Such a tail is short or crosses the ring, so a line fit to its label pixels
    says little; the test fits the drawn stroke instead."""
    q = parts["tail_pts"][k]
    r = float(np.linalg.norm(parts["ring_pts"][k] - (cx, cy), axis=1).mean())
    return math.hypot(q[0][0] - cx, q[0][1] - cy) > 0.3 * r


def _buried_box(drawn, lab, world, frac=0.75) -> bool:
    """A start box whose area is ≥ `frac` agent ink (close agents drawn larger
    than their box allows): its visible pieces cannot be told from separate
    boxes, at any style (sketch6 loses 2 of 20 redraws of such a scene too)."""
    from extract import _box_slice
    return any(float((lab[_box_slice(b, world, lab.shape[0])] == C_AGENT).mean()) >= frac
               for b in drawn.start_boxes())


def _touches(drawn, k, cfg) -> bool:
    """Whether agent k's drawn ring may touch another's (v5 only, which records
    drawn radii: radii + 2 units apart or closer). Older styles never touch."""
    a = drawn.agents[k]
    if a.radius is None:
        return False
    return any(math.dist(a.pos, b.pos) < a.radius + b.radius + 2.0
               for j, b in enumerate(drawn.agents) if j != k)


def _circle_fit(xs, ys):
    """Algebraic (Kåsa) circle fit -> centre (x, y)."""
    A = np.c_[xs, ys, np.ones_like(xs)]
    sol = np.linalg.lstsq(A, xs * xs + ys * ys, rcond=None)[0]
    return sol[0] / 2, sol[1] / 2


@test
def test_sketch_json_matches_drawing():
    """The written JSON is the drawn geometry (DESIGN_SKETCH §3.1 step 4): every
    agent centre within ~1 px of the centre of its drawn label ring, every heading
    within 5° of its drawn label tail, measured on the label strokes themselves.

    The ring's centre is measured by a circle fit to its label pixels: the raw
    pixel centroid is pulled up to ~2 px off by the ±35 % pen-pressure width
    variation (one thick side), so it is only held to 2.5 px (DESIGN_SKETCH §8).
    Style v4 draws agents from 0.6x to 2x, each element with its own pen: the
    centre bound is 1.25 px (0.25 units; v4's per-element widths make the label
    ring's width vary more), scaling with the drawn ring (6 % / 25 % of its
    radius when that is more), and the heading bound scales with the drawn tail
    (1 px at 0.6 of its length when that is more than 5°).
    """
    cfg = Config()
    n_agents = 0
    for drawn, lab, parts in [x for st in SKETCH_STYLE_NAMES for x in _sketch_scenes(cfg, st)]:
        rings, tails = parts["rings"], parts["tails"]
        assert (lab[(rings > 0) | (tails > 0)] == C_AGENT).all(), "ring/tail maps must be agent label"
        for k, a in enumerate(drawn.agents):
            cx, cy = world_to_px(cfg.world, cfg.img_size, *a.pos)
            ys, xs = np.nonzero(rings == k + 1)
            fx, fy = _circle_fit(xs.astype(float), ys.astype(float))
            if _touches(drawn, k, cfg):                         # v5: the drawn curve itself
                q = parts["ring_pts"][k]
                fx, fy = _circle_fit(q[:, 0], q[:, 1])
            r_draw = float(np.hypot(xs - fx, ys - fy).mean())
            err = math.hypot(fx - cx, fy - cy)
            assert err <= max(1.25, 0.06 * r_draw), \
                f"{drawn.layout} agent {k}: JSON centre {err:.2f} px from the drawn ring (r {r_draw:.1f})"
            raw = math.hypot(xs.mean() - cx, ys.mean() - cy)
            # (not where rings touch, v5: the neighbour's ring is drawn over part of this one
            # in the ring map, which moves the raw centroid but not the circle fit)
            if not _touches(drawn, k, cfg):
                assert raw <= max(2.5, 0.25 * r_draw), \
                    f"{drawn.layout} agent {k}: ring pixel centroid {raw:.2f} px away"
            ty, tx = np.nonzero(tails == k + 1)
            if _touches(drawn, k, cfg) or _tail_off_centre(parts, k, cx, cy):   # the drawn curve itself
                tx, ty = parts["tail_pts"][k][:, 0], parts["tail_pts"][k][:, 1]
            tail_len = float(np.hypot(tx - cx, ty - cy).max())
            P = np.c_[tx, -ty].astype(float)                    # y up, like the heading
            mu = P.mean(0)
            u = np.linalg.svd(P - mu, full_matrices=False)[2][0]
            if np.dot(u, mu - np.array([cx, -cy])) < 0:
                u = -u
            herr = heading_err_deg(math.atan2(u[1], u[0]), a.heading)
            htol = max(5.0, math.degrees(math.atan2(1.0, 0.6 * tail_len)))
            assert herr <= htol, f"{drawn.layout} agent {k}: heading {herr:.1f}° off the drawn tail"
            n_agents += 1
        # obstacles and regions: the JSON circle / box covers the drawn label
        # (regions to the pixel grid: 1.5 px measured max excursion 1.12 px)
        for o in drawn.obstacles:
            cx, cy = world_to_px(cfg.world, cfg.img_size, *o.c)
            assert lab[int(round(cy)), int(round(cx))] == C_OBST, f"{drawn.layout}: obstacle centre off its label"
        for boxes, cls in ((drawn.start_boxes(), C_START), (drawn.exit_boxes(), C_GOAL)):
            ys, xs = np.nonzero(lab == cls)
            inside = np.zeros(len(xs), bool)
            for reg in boxes:
                x0, y0 = world_to_px(cfg.world, cfg.img_size, reg.x0, reg.y1)
                x1, y1 = world_to_px(cfg.world, cfg.img_size, reg.x1, reg.y0)
                assert x1 - x0 > 20 and y1 - y0 > 20, f"{drawn.layout}: degenerate region {reg}"
                inside |= (xs >= x0 - 1.5) & (xs <= x1 + 1.5) & (ys >= y0 - 1.5) & (ys <= y1 + 1.5)
            assert inside.all(), f"{drawn.layout}: region label outside its JSON box"
    assert n_agents >= 20


@test
def test_sketch_instance_targets_align():
    """Heat targets rendered from the sketch JSON peak on agent-labelled pixels
    (the tail starts at the drawn centre), at 512 and at 256: the sub-pixel peak
    is within 1 px of an agent pixel's centre (rounding it to the pixel grid can
    land on the neighbour of a 4 px tail). A v7 tail from the edge or through
    the circle leaves the centre blank: there the peak lies inside the drawn ring."""
    cfg = Config()
    world = _world(cfg)
    for drawn, lab, parts in [x for st in SKETCH_STYLE_NAMES for x in _sketch_scenes(cfg, st)]:
        for size in (512, 256):
            heat, dirs, mask = instance_targets(drawn, world, size)
            ay, ax = np.nonzero(_downsample(lab, size) == C_AGENT)
            for k, a in enumerate(drawn.agents):
                cx, cy = world_to_px(cfg.world, size, *a.pos)
                iy, ix = int(round(cy)), int(round(cx))
                assert heat[0, iy, ix] == 1.0
                n = lab.shape[0]
                if _tail_off_centre(parts, k, *world_to_px(cfg.world, n, *a.pos)):
                    # v7 edge / through glyph: no ink at the centre; the peak lies inside the drawn ring
                    ring = (parts["ring_pts"][k] * (size / n)).astype(np.float32).reshape(-1, 1, 2)
                    assert cv2.pointPolygonTest(ring, (cx, cy), False) >= 0, \
                        f"{drawn.layout} @{size}: heat peak of agent {k} outside its drawn ring"
                else:
                    d = float(np.hypot(ax - cx, ay - cy).min())
                    assert d <= 1.0, f"{drawn.layout} @{size}: heat peak of agent {k} {d:.2f} px from agent pixels"
                assert abs(math.atan2(dirs[1, iy, ix], dirs[0, iy, ix]) - a.heading) < 1e-5


@test
def test_extract_on_sketch_labels():
    """Extraction on sketch ground-truth labels with ground-truth heat and
    direction maps: every agent within 1.0 unit and 10°, obstacle count exact."""
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    from targets import letter_targets
    # v5: its own training targets and extractor (σ 0.5 x drawn radius, NMS 0.6, exits);
    # v6 adds the letter gate (fed the ground-truth letter heat) and min_box_side 3
    own = {"sketch5": ExtractConfig(heat_nms_factor=0.6, goal_mode="exits"),
           "sketch6": ExtractConfig(heat_nms_factor=0.6, goal_mode="exits", letter_gate=True,
                                    min_box_side=3.0),
           # v7 adds the wall-on-box / stub filter and slanted-line joining (§11.3)
           "sketch7": ExtractConfig(heat_nms_factor=0.6, goal_mode="exits", letter_gate=True,
                                    min_box_side=3.0, wall_mode="lines", wall_on_box_tol=1.5,
                                    stub_len=3.0, line_axis_dev=2.0)}
    for size in (512, 256):
        for st in SKETCH_STYLE_NAMES:
            for drawn, lab, _ in _sketch_scenes(cfg, st):
                heat, dirs, _ = instance_targets(drawn, world, size, 0.5 if st in own else 1.0)
                letter = letter_targets(drawn, world, size)[0] if st in ("sketch6", "sketch7") else None
                p = extract(_downsample(lab, size), world, heat=heat[0], dir=dirs,
                            ecfg=own.get(st, ExtractConfig()), letter=letter)
                tag = f"{st} {drawn.layout} @{size}"
                assert p is not None, f"{tag}: parse failed"
                pairs = _match(drawn, p, tol=1.0)
                assert len(pairs) == len(drawn.agents) == len(p.scene.agents), \
                    f"{tag}: {len(pairs)} of {len(drawn.agents)} agents matched"
                for gi, pi in pairs:
                    err = heading_err_deg(p.scene.agents[pi].heading, drawn.agents[gi].heading)
                    assert err < 10.0, f"{tag}: heading error {err:.1f}°"
                assert len(p.scene.obstacles) == len(drawn.obstacles), \
                    f"{tag}: {len(p.scene.obstacles)} obstacles vs {len(drawn.obstacles)}"
                if st in own and not _buried_box(drawn, lab, world):   # every start and exit, as a set
                    from metrics import region_set_ok
                    assert region_set_ok(drawn, p.scene), f"{tag}: start / exit boxes not recovered"


@test
def test_wall_lines_like_generated():
    """wall_mode="lines": walls come back as a few straight, axis-parallel centre
    lines with the generator's thickness, close to the scene's own, with every
    doorway still open; the parse runs in the simulator and re-renders like a
    generated scene. On default-renderer labels (21 scenes) and on hand-drawn
    sketch labels (7 scenes), at 512 and 256.

    The error is measured on interior walls away from obstacles: a wall drawn
    under an obstacle is not in the label map at all, and one half-covered by
    an obstacle shifts its visible half by up to ~0.6 units."""
    def visible_err(gt, pr, obstacles):
        def keep(pts):
            return np.array([all(math.dist(q, o.c) > o.r + 1.5 for o in obstacles) for q in pts], bool)
        a, b = _sample_segments(gt), _sample_segments(pr)
        a, b = a[keep(a)], b[keep(b)]
        return 0.5 * (_dist_to_segments(a, pr).mean() + _dist_to_segments(b, gt).mean())

    from metrics import _dist_to_segments, _sample_segments
    from reachability import is_solvable
    from runconfig import ExtractConfig
    from viz import render_clean
    cfg = Config()
    world = _world(cfg)
    ecfg = ExtractConfig(wall_mode="lines")
    boundary = set(boundary_walls(cfg.world))
    # (style, scenes, per-scene bound, median bound), units; measured medians 0.10 / 0.15
    # (render 512 / 256) and 0.27 / 0.28 (sketch), where the pen wobble itself is ~0.5
    sets = [("render", [(s, lab) for s, lab in _roundtrip_scenes(cfg)], 0.5, 0.2),
            ("sketch", [(d, lab) for d, lab, _ in _sketch_scenes(cfg)], 1.0, 0.4)]
    for style, items, tol, tol_median in sets:
        for size in (512, 256):
            errs = []
            for s, lab in items:
                heat, dirs, _ = instance_targets(s, world, size)
                p = extract(_downsample(lab, size), world, heat=heat[0], dir=dirs, ecfg=ecfg)
                tag = f"{style} {s.layout} @{size}"
                assert p.world["wall_thickness"] == cfg.wall_thickness, tag
                assert set(p.scene.walls[:4]) == boundary, f"{tag}: boundary walls not exact"
                inner = p.scene.walls[4:]
                assert all(a[0] == b[0] or a[1] == b[1] for a, b in inner), f"{tag}: wall off-axis"
                assert abs(len(p.scene.walls) - len(s.walls)) <= 1, \
                    f"{tag}: {len(p.scene.walls)} wall segments for {len(s.walls)}"
                if s.walls[4:]:
                    err = visible_err(s.walls[4:], inner, s.obstacles)
                    assert err < tol, f"{tag}: interior wall line error {err:.2f}"
                    errs.append(err)
                assert is_solvable(cfg, p.scene), f"{tag}: a doorway was closed"
            assert np.median(errs) < tol_median, f"{style} @{size}: median wall line error {np.median(errs):.2f}"
    sim = Simulator(p.scene, sim_config(cfg, p))
    assert sim.cfg.wall_thickness == cfg.wall_thickness
    sim.reset()
    for _ in range(5):
        sim.step(np.stack([greedy_action(sim, i) for i in range(len(sim.pos))]))
    img = render_clean(p)
    assert np.asarray(img).shape == (p.size, p.size, 3)


@test
def test_region_by_agents_fixes_a_misread_letter():
    """A start box labelled as goal (the S read as G) comes back as the start
    with region_by_agents, because it holds the agents; without it the scene
    has no start-goal pair."""
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    for layout in ("doorway", "crossing"):
        s = random_scene(np.random.default_rng(31), cfg, layout=layout, n_agents=6, n_obstacles=2)
        _, lab = render(s, cfg)
        bad = lab.copy()
        bad[bad == C_START] = C_GOAL                       # every S misread
        heat, dirs, _ = instance_targets(s, world, cfg.img_size)
        assert extract(bad, world, heat=heat[0], dir=dirs) is None or layout == "crossing"
        p = extract(bad, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(region_by_agents=True))
        assert p is not None, f"{layout}: no parse with region_by_agents"
        for gs, gg in s.groups:
            if not any(gs.contains(a.pos) for a in s.agents):
                continue                                   # an empty start keeps its (wrong) letter
            assert max(region_iou(gs, ps) for ps, _ in p.scene.groups) > 0.95, f"{layout}: start not recovered"
            assert max(region_iou(gg, pg) for _, pg in p.scene.groups) > 0.95, f"{layout}: goal lost"


# ----------------------------------------- sketch v5: close agents, exits

@test
def test_sketch5_close_agents_labels():
    """Touching drawn agents (v5 close scenes) still give one heat peak each, at
    the JSON centres, with v5's targets (σ = 0.5 x drawn radius) and NMS (0.6)."""
    from extract import detect_agents_heat
    from runconfig import ExtractConfig
    from sketch import SKETCH_V5, sketch
    cfg = Config()
    world = _world(cfg)
    rng = np.random.default_rng(50)
    ecfg = ExtractConfig(heat_nms_factor=0.6)
    touching = 0
    for k in range(6):
        s = random_scene(rng, cfg, layout="open" if k % 2 else "boxrooms", n_agents=8, n_obstacles=1,
                         close=True)
        s.seed = 600 + k
        _, lab, drawn, parts = sketch(s, cfg, style=SKETCH_V5, parts=True)
        rings = parts["rings"]
        grown = [cv2.dilate((rings == i + 1).astype(np.uint8), np.ones((3, 3), np.uint8))
                 for i in range(len(drawn.agents))]
        touching += sum(int((grown[i] & grown[j]).any()) for i in range(len(grown))
                        for j in range(i + 1, len(grown)))
        for size in (512, 256):
            heat, _, _ = instance_targets(drawn, world, size, heat_sigma=0.5)
            r_px = cfg.agent_radius * size / cfg.world
            dets = detect_agents_heat(heat[0], r_px, ecfg)
            centres = [world_to_px(cfg.world, size, *a.pos) for a in drawn.agents]
            assert len(dets) == len(centres), f"scene {k} @{size}: {len(dets)} peaks for {len(centres)} agents"
            pairs = match_points(centres, [(x, y) for x, y, _ in dets], 1.0)
            assert len(pairs) == len(centres), f"scene {k} @{size}: a peak is off its JSON centre"
    assert touching >= 6, f"only {touching} touching pairs of drawn rings: the test checks nothing"


@test
def test_exits_mode_drops_empty_start():
    """goal_mode="exits": a start box without agents is dropped (the fake start
    of a room), both exits are kept, the agents' goal is the exit set, and the
    parse is scored by region_set_ok. "pairs" mode is unchanged."""
    from metrics import region_set_ok
    from runconfig import ExtractConfig
    from scene import RegionSet
    cfg = Config()
    world = _world(cfg)
    start, empty = Region(8, 40, 22, 60), Region(8, 75, 22, 92)
    g1, g2 = Region(80, 10, 94, 25), Region(80, 70, 94, 88)
    s = Scene(walls=boundary_walls(cfg.world), agents=[Agent((15.0, 45.0), 0.0), Agent((15.0, 55.0), 0.3)],
              start_region=start, goal_region=g1, starts=[start, empty], exits=[g1, g2])
    _, lab = render(s, cfg, fill_unreachable=False)
    heat, dirs, _ = instance_targets(s, world, cfg.img_size)
    p = extract(lab, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(goal_mode="exits"))
    assert p is not None and p.diagnostics["dropped_empty_starts"] == 1
    assert len(p.scene.starts) == 1 and region_iou(p.scene.starts[0], start) > 0.9
    assert len(p.scene.exits) == 2 and p.scene.groups == []
    goal = p.scene.goal_for(p.scene.agents[0])
    assert isinstance(goal, RegionSet) and len(goal) == 2 and all(a.goal_region is None for a in p.scene.agents)
    assert region_set_ok(s, p.scene), "the empty start box is no start in the ground truth either"
    m, _ = scene_metrics(s, lab, p, lab, world, 512, goal_mode="exits")
    assert m["region_set_ok"] == 1.0 and m["scene_usable"] == 1.0 and math.isnan(m["group_pairing_ok"])
    back = parsed_from_json(parsed_to_json(p, "t"))
    assert len(back.scene.exits) == 2 and isinstance(back.scene.goal_for(back.scene.agents[0]), RegionSet)
    sim = Simulator(p.scene, sim_config(cfg, p))
    assert sim.reset().shape == (2, cfg.obs_dim)
    # no exit at all: a parse failure; pairs mode still pairs one-to-one
    lab_no_goal = lab.copy()
    lab_no_goal[lab_no_goal == C_GOAL] = 0
    assert extract(lab_no_goal, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(goal_mode="exits")) is None
    pp = extract(lab, world, heat=heat[0], dir=dirs)
    assert pp.scene.exits is None and len(pp.scene.groups) >= 1 and "dropped_empty_starts" not in pp.diagnostics
    # crossing ground truth (per-agent goals) scored as exit sets
    c = random_scene(np.random.default_rng(3), cfg, layout="crossing", n_agents=6, n_obstacles=2)
    _, lab_c = render(c, cfg)
    heat, dirs, _ = instance_targets(c, world, cfg.img_size)
    pc = extract(lab_c, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(goal_mode="exits"))
    mc, _ = scene_metrics(c, lab_c, pc, lab_c, world, 512, goal_mode="exits")
    assert len(pc.scene.exits) == 2 and mc["region_set_ok"] == 1.0 and mc["scene_usable"] == 1.0


@test
def test_heat_nms_separates_touching():
    """Two heat peaks 1.2 nominal radii apart: one detection with the default NMS
    (1.6 radii), both with heat_nms_factor 0.6; the default is unchanged."""
    from extract import detect_agents_heat
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    assert ExtractConfig().heat_nms_factor is None and ExtractConfig().goal_mode == "pairs"
    for size in (512, 256):
        r = cfg.agent_radius
        s = Scene(agents=[Agent((50.0, 50.0), 0.0, radius=0.6 * r),
                          Agent((50.0 + 1.2 * r, 50.0), 0.0, radius=0.6 * r)])
        heat, _, _ = instance_targets(s, world, size, heat_sigma=0.5)
        r_px = r * size / cfg.world
        assert len(detect_agents_heat(heat[0], r_px, ExtractConfig())) == 1
        dets = detect_agents_heat(heat[0], r_px, ExtractConfig(heat_nms_factor=0.6))
        assert len(dets) == 2, f"@{size}: {len(dets)} detections"
        want = [world_to_px(cfg.world, size, *a.pos) for a in s.agents]
        assert len(match_points(want, [(x, y) for x, y, _ in dets], 0.75)) == 2
    # targets: heat_sigma 1 and no drawn radius is the E5 target; a drawn radius sharpens it
    s = random_scene(np.random.default_rng(1), cfg, layout="open", n_agents=4, n_obstacles=1)
    h1 = instance_targets(s, world, 256)[0]
    assert np.array_equal(h1, instance_targets(s, world, 256, 1.0)[0])
    for a in s.agents:
        a.radius = cfg.agent_radius
    assert np.array_equal(h1, instance_targets(s, world, 256, 1.0)[0])
    assert instance_targets(s, world, 256, 0.5)[0].sum() < 0.3 * h1.sum()


# ------------------------------------------------ sketch v6: letter head and gate

def _two_box_scene(cfg):
    """A start box with two agents and a goal box, rendered (no letters drawn)."""
    s = Scene(walls=boundary_walls(cfg.world), agents=[Agent((15.0, 45.0), 0.0), Agent((15.0, 55.0), 0.3)],
              start_region=Region(8, 40, 22, 60), goal_region=Region(78, 40, 92, 60),
              groups=[(Region(8, 40, 22, 60), Region(78, 40, 92, 60))])
    _, lab = render(s, cfg, fill_unreachable=False)
    return s, lab


def _letter_map(cfg, size, letters):
    """(2, S, S) letter heat from [(kind, (x, y), height)] via the training targets."""
    from targets import letter_targets
    s = Scene(letters=[{"kind": k, "pos": list(p), "height": h} for k, p, h in letters])
    return letter_targets(s, _world(cfg), size)[0]


@test
def test_letter_targets():
    """Letter heat peaks of value 1 at the letter centres, in the letter's channel
    (0 = S, 1 = G); a scene without letters gives zeros and mask 0, and its
    letter loss is masked out (not trained towards "no letter")."""
    import torch
    from losses import masked_focal_heat
    from targets import letter_targets
    cfg = Config()
    world = _world(cfg)
    s = Scene(letters=[{"kind": "S", "pos": [20.0, 30.0], "height": 4.0},
                       {"kind": "G", "pos": [70.5, 80.25], "height": 6.0}])
    for size in (512, 256):
        heat, mask = letter_targets(s, world, size)
        assert heat.shape == (2, size, size) and mask == 1.0
        for ch, L in ((0, s.letters[0]), (1, s.letters[1])):
            cx, cy = world_to_px(cfg.world, size, *L["pos"])
            assert heat[ch, int(round(cy)), int(round(cx))] == 1.0
            assert heat[1 - ch, int(round(cy)), int(round(cx))] < 1e-3, "letter heat in the wrong channel"
            peaks = np.argwhere(heat[ch] == heat[ch].max())
            assert len(peaks) == 1
        assert int((heat == 1.0).sum()) == 2
    empty, m0 = letter_targets(Scene(), world, 256)
    assert m0 == 0.0 and not empty.any()
    pred = torch.full((2, 2, 16, 16), 0.3)
    gt = torch.zeros((2, 2, 16, 16))
    assert masked_focal_heat(pred, gt, torch.tensor([0.0, 0.0])) is None
    gt[0, 0, 5, 5] = 1.0
    a = masked_focal_heat(pred, gt, torch.tensor([1.0, 0.0]))
    gt2 = gt.clone()
    gt2[1, 1, 9, 9] = 1.0                                # a letter in the masked-out sample changes nothing
    assert torch.allclose(a, masked_focal_heat(pred, gt2, torch.tensor([1.0, 0.0])))


@test
def test_init_adds_letter_head():
    """--init from a checkpoint without a letter head: every tensor loads, the new
    head keeps its initialisation (bias -2.19); any other mismatch still raises.
    The letter head adds 50 parameters at unet24x4 and changes nothing else."""
    import torch
    from model import build_model, count_params, load_init_weights
    from runconfig import ModelConfig
    torch.manual_seed(0)
    v5 = build_model(ModelConfig(arch="unet16x4", instance_head=True))
    torch.manual_seed(0)
    v6 = build_model(ModelConfig(arch="unet16x4", instance_head=True, letter_head=True))
    for k, t in v5.state_dict().items():
        assert torch.equal(t, v6.state_dict()[k]), f"{k}: the letter head changed another layer's init"
    torch.manual_seed(1)
    v5b = build_model(ModelConfig(arch="unet16x4", instance_head=True))
    fresh = load_init_weights(v6, v5b.state_dict())
    assert fresh == ["letter.weight", "letter.bias"]
    assert all(torch.equal(t, v6.state_dict()[k]) for k, t in v5b.state_dict().items())
    assert torch.allclose(v6.letter.bias, torch.full((2,), -2.19))
    with torch.no_grad():
        out = v6.eval()(torch.zeros(1, 3, 64, 64))
    assert out["letter"].shape == (1, 2, 64, 64)
    for bad in (build_model(ModelConfig(arch="unet24x4", instance_head=True)).state_dict(),   # arch
                build_model(ModelConfig(arch="unet16x4")).state_dict()):                    # no inst head
        try:
            load_init_weights(build_model(ModelConfig(arch="unet16x4", instance_head=True, letter_head=True)), bad)
        except RuntimeError:
            continue
        raise AssertionError("a mismatch other than the letter head must raise")
    n24 = count_params(build_model(ModelConfig(arch="unet24x4", instance_head=True)))
    assert count_params(build_model(ModelConfig(arch="unet24x4", instance_head=True, letter_head=True))) == n24 + 50


@test
def test_letter_gate():
    """letter_gate: a box without letter heat is dropped; a box with G heat but the
    start class becomes an exit; with the gate off the result is v5's. Rendered
    sets switch the gate off in evaluate.py."""
    import json
    import tempfile
    from pathlib import Path
    from dataclasses import replace
    from evaluate import set_ecfg
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    s, lab = _two_box_scene(cfg)
    heat, dirs, _ = instance_targets(s, world, cfg.img_size)
    base = ExtractConfig(goal_mode="exits")
    gate = replace(base, letter_gate=True)
    off = extract(lab, world, heat=heat[0], dir=dirs, ecfg=base)
    # the start box carries a G, the goal box no letter at all
    letter = _letter_map(cfg, cfg.img_size, [("G", (15.0, 50.0), 4.0)])
    p = extract(lab, world, heat=heat[0], dir=dirs, ecfg=gate, letter=letter)
    assert p is not None and p.diagnostics["dropped_no_letter"] == 1
    assert len(p.scene.exits) == 1 and region_iou(p.scene.exits[0], s.start_region) > 0.9
    assert p.scene.starts == [] and len(p.scene.agents) == 2
    # with the right letters the gate changes nothing; off, the letter map is ignored
    good = _letter_map(cfg, cfg.img_size, [("S", (15.0, 50.0), 4.0), ("G", (85.0, 50.0), 4.0)])
    pg = extract(lab, world, heat=heat[0], dir=dirs, ecfg=gate, letter=good)
    assert pg.scene.starts == off.scene.starts and pg.scene.exits == off.scene.exits
    po = extract(lab, world, heat=heat[0], dir=dirs, ecfg=base, letter=letter)
    assert parsed_to_json(po) == parsed_to_json(off), "gate off: the letter map must not matter"
    try:
        extract(lab, world, heat=heat[0], dir=dirs, ecfg=gate)
        raise AssertionError("the gate without a letter map must raise")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "dataset.json").write_text(json.dumps({"args": {"style": "render"}}))
        assert set_ecfg(gate, Path(d), "t").letter_gate is False
        (Path(d) / "dataset.json").write_text(json.dumps({"args": {"style": "sketch6"}}))
        assert set_ecfg(gate, Path(d), "t").letter_gate is True


@test
def test_letter_gate_keeps_box_with_agents():
    """Gate step (iii): a box with no letter heat that holds a parsed agent (an S
    too small or faint to read) is kept as a start (kept_by_agents); a box with
    neither letter nor agent is still dropped; a lettered box keeps its letter's kind."""
    from dataclasses import replace
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    s, lab = _two_box_scene(cfg)
    heat, dirs, _ = instance_targets(s, world, cfg.img_size)
    gate = ExtractConfig(goal_mode="exits", letter_gate=True)
    from extract import extract_boxes
    none = np.zeros((2, cfg.img_size, cfg.img_size), np.float32)          # no letter read anywhere
    diag = {}
    starts, goals = extract_boxes(lab == C_START, lab == C_GOAL, cfg.scale, world, gate,
                                  agent_pos=[a.pos for a in s.agents], letter=none, diag=diag)
    assert diag["kept_by_agents"] == 1 and diag["dropped_no_letter"] == 1 and goals == []
    assert len(starts) == 1 and region_iou(starts[0], s.start_region) > 0.9
    # no exit read at all: the parse falls back to the segmentation's exits instead of failing
    f = extract(lab, world, heat=heat[0], dir=dirs, ecfg=gate, letter=none)
    assert f is not None and f.diagnostics["gate_fallback"] and len(f.scene.exits) == 1
    assert region_iou(f.scene.exits[0], s.goal_region) > 0.9 and len(f.scene.starts) == 1
    # with a G in the goal box: one start kept by its agents, one exit by its letter
    g_only = _letter_map(cfg, cfg.img_size, [("G", (85.0, 50.0), 4.0)])
    q = extract(lab, world, heat=heat[0], dir=dirs, ecfg=gate, letter=g_only)
    assert q.diagnostics["kept_by_agents"] == 1 and q.diagnostics["dropped_no_letter"] == 0
    assert len(q.scene.starts) == 1 and len(q.scene.exits) == 1
    assert region_iou(q.scene.exits[0], s.goal_region) > 0.9
    # a letter beats the agents: a G in the box holding the agents makes it an exit (step ii first)
    g_on_start = _letter_map(cfg, cfg.img_size, [("G", (15.0, 50.0), 4.0)])
    r = extract(lab, world, heat=heat[0], dir=dirs, ecfg=replace(gate, letter_suppress_agents=False),
                letter=g_on_start)
    assert r.diagnostics["kept_by_agents"] == 0 and len(r.scene.exits) == 1 and r.scene.starts == []


@test
def test_agent_on_letter_suppressed():
    """With the gate, an agent peak on a stronger letter peak is dropped (a letter
    read as an agent); an agent 3 radii from a letter is kept."""
    from extract import detect_agents_heat, suppress_agents_on_letters
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    size = 256
    r = cfg.agent_radius
    s = Scene(agents=[Agent((30.0, 50.0), 0.0), Agent((60.0, 50.0), 0.0)])
    heat = instance_targets(s, world, size)[0][0] * 0.6           # agent peaks scoring 0.6
    letter = _letter_map(cfg, size, [("S", (30.0, 50.0), 4.0), ("G", (60.0 + 3 * r, 50.0), 4.0)])
    ecfg = ExtractConfig(letter_gate=True)
    r_px = r * size / cfg.world
    dets = detect_agents_heat(heat, r_px, ecfg)
    assert len(dets) == 2
    diag = {}
    kept = suppress_agents_on_letters(dets, letter, r_px, ecfg, diag)
    assert diag["agents_on_letters"] == 1 and len(kept) == 1
    assert abs(kept[0][0] - world_to_px(cfg.world, size, 60.0, 50.0)[0]) < 1.0
    # a weaker letter does not remove a confident agent
    assert len(suppress_agents_on_letters(dets, letter * 0.5, r_px, replace_thresh(ecfg, 0.2), {})) == 2


def replace_thresh(ecfg, t):
    from dataclasses import replace
    return replace(ecfg, letter_thresh=t)


@test
def test_min_box_side():
    """A 2 x 10 box (a corner sliver) is dropped at min_box_side 3.0, kept at 0."""
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    sliver = Region(80, 10, 82, 20)
    s = Scene(walls=boundary_walls(cfg.world), agents=[Agent((15.0, 50.0), 0.0)],
              start_region=Region(8, 40, 22, 60), goal_region=Region(78, 40, 92, 60),
              starts=[Region(8, 40, 22, 60)], exits=[Region(78, 40, 92, 60), sliver])
    _, lab = render(s, cfg, fill_unreachable=False)
    heat, dirs, _ = instance_targets(s, world, cfg.img_size)
    keep = extract(lab, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(goal_mode="exits"))
    assert len(keep.scene.exits) == 2 and "dropped_thin_boxes" not in keep.diagnostics
    drop = extract(lab, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(goal_mode="exits", min_box_side=3.0))
    assert len(drop.scene.exits) == 1 and drop.diagnostics["dropped_thin_boxes"] == 1
    assert region_iou(drop.scene.exits[0], Region(78, 40, 92, 60)) > 0.9


# ------------------------------------- sketch v7: context, walls on boxes, no S

@test
def test_context_block_identity():
    """A v6-shaped checkpoint initialises a context model (--context dilated) with
    identical outputs: the block is the identity at initialisation. Only the
    context block may be missing; an arch mismatch still raises. ~ +1.0 M
    parameters at unet24x4."""
    import torch
    from model import build_model, count_params, load_init_weights
    from runconfig import ModelConfig
    torch.manual_seed(0)
    v6 = build_model(ModelConfig(arch="unet16x4", instance_head=True, letter_head=True)).eval()
    v7 = build_model(ModelConfig(arch="unet16x4", instance_head=True, letter_head=True, context="dilated"))
    fresh = load_init_weights(v7, v6.state_dict())
    assert fresh and all(k.startswith("context.") for k in fresh)
    v7.eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        a, b = v6(x), v7(x)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k}: the context block is not the identity at initialisation"
    # ... and it can learn: its zero weight gets a gradient (no ReLU after it)
    v7.train()
    out = v7(x)
    sum(v.float().pow(2).mean() for v in out.values()).backward()
    g = v7.context[-1].weight.grad
    assert g is not None and float(g.abs().sum()) > 0, "the context block gets no gradient"
    # a v4 / v5 checkpoint (no letter head) loads too; another arch does not
    v5 = build_model(ModelConfig(arch="unet16x4", instance_head=True))
    assert set(load_init_weights(v7, v5.state_dict())) >= {"letter.weight", "letter.bias"}
    try:
        load_init_weights(v7, build_model(ModelConfig(arch="unet24x4", instance_head=True)).state_dict())
        raise AssertionError("an arch mismatch must raise")
    except RuntimeError:
        pass
    n6 = count_params(build_model(ModelConfig(arch="unet24x4", instance_head=True, letter_head=True)))
    n7 = count_params(build_model(ModelConfig(arch="unet24x4", instance_head=True, letter_head=True,
                                              context="dilated")))
    assert 0.9e6 <= n7 - n6 <= 1.1e6, n7 - n6
    assert build_model(ModelConfig(arch="unet24x4")).context is None, "context defaults to none"


@test
def test_wall_on_box_dropped():
    """wall_on_box_tol / stub_len: a wall piece along a box side and a 2-unit stub
    on a box are dropped; a wall 4 units off the box and the boundary are kept;
    both off (0) keep everything. Also end to end on a rendered label map."""
    from extract import drop_walls_on_boxes
    from runconfig import ExtractConfig
    cfg = Config()
    box = Region(20, 30, 34, 60)
    along = ((20.5, 32.0), (20.8, 58.0))           # on the box's left side, slightly slanted
    stub = ((34.0, 45.0), (36.0, 45.0))            # 2 units sticking out of the right side
    off = ((38.0, 30.0), (38.0, 60.0))             # parallel, 4 units off
    walls = boundary_walls(cfg.world) + [along, stub, off]
    e = ExtractConfig(wall_on_box_tol=1.5, stub_len=3.0)
    diag = {}
    kept = drop_walls_on_boxes(walls, [box], e, diag, keep=set(boundary_walls(cfg.world)))
    assert kept == boundary_walls(cfg.world) + [off], kept
    assert diag["walls_on_boxes"] == 1 and diag["wall_stubs"] == 1
    assert drop_walls_on_boxes(walls, [box], ExtractConfig(), {}) == walls
    # end to end: a wall drawn over a box side comes back without it
    world = _world(cfg)
    s = Scene(walls=boundary_walls(cfg.world) + [((20.0, 30.0), (20.0, 60.0)), ((60.0, 20.0), (60.0, 80.0))],
              agents=[Agent((27.0, 45.0), 0.0)], start_region=box, goal_region=Region(80, 40, 94, 60),
              starts=[box], exits=[Region(80, 40, 94, 60)])
    _, lab = render(s, cfg, fill_unreachable=False)
    heat, dirs, _ = instance_targets(s, world, cfg.img_size)
    base = ExtractConfig(goal_mode="exits", wall_mode="lines")
    p0 = extract(lab, world, heat=heat[0], dir=dirs, ecfg=base)
    p1 = extract(lab, world, heat=heat[0], dir=dirs, ecfg=ExtractConfig(goal_mode="exits", wall_mode="lines",
                                                                       wall_on_box_tol=1.5, stub_len=3.0))
    inner0, inner1 = p0.scene.walls[4:], p1.scene.walls[4:]
    assert any(abs(a[0] - 20.0) < 1.0 and abs(b[0] - 20.0) < 1.0 for a, b in inner0), "the side wall is parsed"
    assert not any(abs(a[0] - 20.0) < 1.0 and abs(b[0] - 20.0) < 1.0 for a, b in inner1), "and then dropped"
    assert any(abs(a[0] - 60.0) < 1.0 for a, _ in inner1) and p1.diagnostics["walls_on_boxes"] >= 1


@test
def test_slanted_wall_one_segment():
    """A wall drawn at 15° in a generated (sketch v7) label map extracts as one
    segment at about 15° with line_axis_dev 2; a hand-drawn axis-parallel wall
    still comes out axis-parallel."""
    from runconfig import ExtractConfig
    from sketch import SKETCH_V7, sketch
    cfg = Config()
    world = _world(cfg)
    t = math.radians(15.0)
    slant = ((30.0, 15.0), (30.0 + 70.0 * math.sin(t), 15.0 + 70.0 * math.cos(t)))
    upright = ((75.0, 15.0), (75.0, 85.0))
    s = Scene(walls=boundary_walls(cfg.world) + [slant, upright], agents=[Agent((10.0, 50.0), 0.0)],
              start_region=Region(4.5, 40, 16, 60), goal_region=Region(84, 40, 95.5, 60),
              groups=[(Region(4.5, 40, 16, 60), Region(84, 40, 95.5, 60))], seed=21)
    e = ExtractConfig(wall_mode="lines", line_axis_dev=2.0)
    for seed in (21, 22, 23):
        s.seed = seed
        _, lab, drawn = sketch(s, cfg, style=SKETCH_V7)
        for size in (512, 256):
            heat, dirs, _ = instance_targets(drawn, world, size)
            p = extract(_downsample(lab, size), world, heat=heat[0], dir=dirs, ecfg=e)
            inner = [w for w in p.scene.walls[4:] if math.dist(*w) > 3.0]
            near = [w for w in inner if min(math.dist(w[0], (40.0, 50.0)), math.dist(w[1], (40.0, 50.0))) < 45
                    and abs(w[0][0] - 75.0) > 3.0]
            assert len(near) == 1, f"seed {seed} @{size}: the 15° wall came out as {near}"
            (ax, ay), (bx, by) = near[0]
            ang = math.degrees(math.atan2(abs(bx - ax), abs(by - ay)))
            assert 11.0 <= ang <= 19.0, f"seed {seed} @{size}: angle {ang:.1f}"
            ups = [w for w in inner if abs(w[0][0] - 75.0) < 3.0]
            assert len(ups) == 1 and abs(ups[0][0][0] - ups[0][1][0]) < 1e-9, "axis-parallel wall not snapped"


@test
def test_agents_without_start_box():
    """Agents drawn with no S box (sketch v7) parse in exits mode, with and
    without the letter gate; region_set_ok scores the scene right with no
    start parsed, and wrong when a start is invented."""
    from metrics import region_set_ok
    from runconfig import ExtractConfig
    cfg = Config()
    world = _world(cfg)
    g = Region(80, 40, 94, 60)
    s = Scene(walls=boundary_walls(cfg.world), agents=[Agent((30.0, 30.0), 0.0), Agent((40.0, 70.0), 1.0)],
              start_region=None, goal_region=g, starts=[], exits=[g])
    _, lab = render(s, cfg, fill_unreachable=False)
    heat, dirs, _ = instance_targets(s, world, cfg.img_size)
    letter = _letter_map(cfg, cfg.img_size, [("G", (87.0, 50.0), 4.0)])
    for e in (ExtractConfig(goal_mode="exits"),
              ExtractConfig(goal_mode="exits", letter_gate=True, region_by_agents=True), ):
        p = extract(lab, world, heat=heat[0], dir=dirs, ecfg=e, letter=letter if e.letter_gate else None)
        assert p is not None and len(p.scene.agents) == 2 and p.scene.starts == [] and len(p.scene.exits) == 1
        assert region_set_ok(s, p.scene)
        m, _ = scene_metrics(s, lab, p, lab, world, 512, goal_mode="exits")
        assert m["region_set_ok"] == 1.0 and m["scene_usable"] == 1.0
    p.scene.starts = [Region(25, 25, 35, 35)]      # an invented start box
    assert not region_set_ok(s, p.scene)


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
