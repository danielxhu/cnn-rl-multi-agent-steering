"""Dataset generator.

    python generate.py --n 3000 --layout mixed --agents 3-8 --obstacles 2-6 \
                       --hard-ratio 0.2 --size 512 --seed 0 --out data/train/

Each sample is written as three files that share a stem:

    scene_00001.png          RGB input
    scene_00001_labels.png   palette PNG, pixel value == class index
    scene_00001.json         structured ground truth

`--style sketch` / `sketch2` / ... / `sketch5` draws the scene hand-drawn
instead (sketch.py): the JSON then holds the drawn geometry, and the
domain-randomisation flags are ignored. `sketch5` also packs the agents of a
start box at the hard spacing in 40 % of scenes and writes each agent's drawn
radius. `--layout mixed3` adds rooms next to boxes and several start / exit
boxes (layouts.MULTI_LAYOUTS; DESIGN_SKETCH §9). `--style sketch6` draws every
line from one width law and writes the boxes' letters; `--layout mixed4` is
mixed3 with the hard cases of layouts.MIX_OPTIONS (narrow boxes, boxes near
walls, open / crossed room corners, slits; DESIGN_SKETCH §10). `--style sketch7`
adds tilted boxes, sides past the corners, three agent glyphs and big letters,
and `--layout mixed5` agents outside boxes, slanted and tilted walls, touching
obstacles and long lines next to boxes (DESIGN_SKETCH §11); `mixed5_free`,
`mixed5_tilt` (with `--style sketch7tilt`) and `mixed5_lines` are its probes.

plus one `dataset.json` recording the exact configuration, so a run is
reproducible from the output directory alone.  See README.md for the knobs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from config import Config, StyleConfig
from layouts import BUILDERS, LAYOUTS, MIX_OPTIONS, MIXES, random_scene
from render import PALETTE, render
from sketch import STYLES as SKETCH_STYLES, sketch


def parse_range(text, cast=int):
    """'3-8' -> (3, 8);  '4' -> (4, 4)."""
    if text is None:
        return None
    if "-" in str(text)[1:]:
        lo, hi = str(text).split("-", 1)
        return (cast(lo), cast(hi))
    v = cast(text)
    return (v, v)


def save_labels(labels: np.ndarray, path):
    """Palette PNG: viewable as colour, but the raw pixel value is the class."""
    img = Image.fromarray(labels.astype(np.uint8), "P")
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE)] = PALETTE
    img.putpalette(pal.flatten().tolist())
    img.save(path)


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate steering-behaviour scene images with pixel-perfect labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="number of scenes")
    p.add_argument("--out", default="data/train", help="output directory")
    p.add_argument("--layout", default="mixed",
                   help="one of %s, or 'mixed' for an even spread of those; 'mixed2' adds the "
                        "free-form families walls / rooms (~40 percent); 'mixed3' adds to mixed2 "
                        "boxrooms / walls_multi / rooms_multi (a third): small rooms next to boxes, "
                        "several start and exit boxes; 'mixed4' is mixed3 with hard cases (narrow boxes, "
                        "boxes near walls, open / crossed corners, slits); 'mixed5' adds agents outside boxes, "
                        "slanted / tilted walls, touching obstacles, long lines by boxes "
                        "(probes mixed5_free / mixed5_tilt / mixed5_lines)" % ", ".join(BUILDERS))
    p.add_argument("--agents", default="1-6", help="agents per scene, e.g. 3-8 or 4")
    p.add_argument("--obstacles", default="1-4", help="obstacles per scene, e.g. 2-6")
    p.add_argument("--hard-ratio", type=float, default=0.0,
                   help="fraction of scenes with near-tangent agent pairs")
    p.add_argument("--size", type=int, default=512, help="image side in pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None, help="JSON file overriding any Config field")
    p.add_argument("--style", default="render", choices=["render", "sketch", "sketch2", "sketch3", "sketch4", "sketch5",
                            "sketch6", "sketch6swap", "sketch7", "sketch7tilt"],
                   help="render: the default renderer; sketch: hand-drawn, one black pen, "
                        "photographed (sketch.py); sketch2: the same widened to tablet drawings, "
                        "thin pens and sparse fills (sketch.SKETCH_V2); sketch3: v2 plus S / G in "
                        "many hands, a harsher camera, pen dropouts and stray marks (SKETCH_V3); sketch4: v3 "
                        "plus agents 0.6-2x and a pen width per element (SKETCH_V4); sketch5: v4 plus smaller, "
                        "thinner letters, touching agents and drawn radii in the JSON (SKETCH_V5); sketch6: v5 "
                        "with one width law for every line and the letters in the JSON (SKETCH_V6); "
                        "sketch6swap: a probe with thin walls and thick box sides; sketch7: v6 with tilted boxes, "
                        "sides past corners, tails from the edge or through the circle, big letters "
                        "(SKETCH_V7); sketch7tilt: every box tilted. "
                        "The jitter flags are ignored for the sketch styles")

    g = p.add_argument_group("world")
    g.add_argument("--world", type=float, default=None, help="world side, world units")
    g.add_argument("--agent-radius", type=float, default=None)
    g.add_argument("--wall-thickness", type=float, default=None)
    g.add_argument("--obstacle-radius", default=None, help="min-max, e.g. 4-10")
    g.add_argument("--min-bottleneck", type=float, default=None,
                   help="reject scenes whose tightest passage is narrower than this")
    g.add_argument("--cell", type=float, default=None,
                   help="reachability grid cell; smaller is stricter and slower")

    g = p.add_argument_group("appearance / domain randomisation")
    g.add_argument("--jitter-color", type=float, default=None, help="+/- per RGB channel")
    g.add_argument("--jitter-line-width", type=float, default=None, help="+/- pixels")
    g.add_argument("--jitter-background", type=float, default=None, help="+/- grey level")
    g.add_argument("--noise-std", type=float, default=None, help="gaussian pixel noise")
    g.add_argument("--blur-sigma", type=float, default=None, help="gaussian blur radius")

    p.add_argument("--preview", type=int, default=0,
                   help="also write preview.png, a contact sheet of the first N scenes")
    p.add_argument("--no-fill-unreachable", action="store_true",
                   help="leave dead space white instead of painting it as wall")
    return p


def config_from_args(a) -> Config:
    cfg = Config.from_json(a.config) if a.config else Config()
    obs_r = parse_range(a.obstacle_radius, float)
    cfg = cfg.merged(
        img_size=a.size, world=a.world, agent_radius=a.agent_radius,
        wall_thickness=a.wall_thickness, min_bottleneck=a.min_bottleneck, cell=a.cell,
        hard_ratio=a.hard_ratio,
        obstacle_r_min=obs_r[0] if obs_r else None,
        obstacle_r_max=obs_r[1] if obs_r else None,
        style={k: v for k, v in dict(
            jitter_color=a.jitter_color, jitter_line_width=a.jitter_line_width,
            jitter_background=a.jitter_background, noise_std=a.noise_std,
            blur_sigma=a.blur_sigma).items() if v is not None},
    )
    return cfg


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    n_agents = parse_range(a.agents)
    n_obst = parse_range(a.obstacles)
    layouts = MIXES[a.layout] if a.layout in MIXES else [a.layout]
    for name in layouts:
        if name not in BUILDERS:
            raise SystemExit("unknown layout %r; choose from %s or %s"
                             % (name, ", ".join(BUILDERS), " / ".join(MIXES)))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    index, previews, skipped = [], [], 0
    t0 = time.time()
    close_prob = SKETCH_STYLES[a.style].close_prob if a.style in SKETCH_STYLES else 0.0
    opts = MIX_OPTIONS.get(a.layout)                  # hard cases (mixed4); None: none
    for i in range(a.n):
        layout = layouts[i % len(layouts)] if len(layouts) > 1 else layouts[0]
        # sketch v5 only: agents packed at the hard spacing (drawn only then, so
        # every other style keeps its random stream)
        close = bool(rng.random() < close_prob) if close_prob > 0 else False
        scene = random_scene(
            rng, cfg, layout=layout,
            n_agents=int(rng.integers(n_agents[0], n_agents[1] + 1)),
            n_obstacles=int(rng.integers(n_obst[0], n_obst[1] + 1)),
            hard=bool(rng.random() < cfg.hard_ratio), close=close, opts=opts,
        )
        if scene is None:
            skipped += 1
            continue
        scene.seed = int(rng.integers(2 ** 31))
        sid = "scene_%05d" % (len(index) + 1)
        if a.style in SKETCH_STYLES:
            # own rng from the scene seed: the drawing is reproducible from it
            img, lab, drawn = sketch(scene, cfg, rng=np.random.default_rng(scene.seed),
                                     fill_unreachable=not a.no_fill_unreachable,
                                     style=SKETCH_STYLES[a.style])
            rec = drawn.as_dict(cfg, sid)             # what was drawn, not what was sampled
            rec["style"] = a.style
        else:
            img, lab = render(scene, cfg, rng=rng,
                              fill_unreachable=not a.no_fill_unreachable)
            rec = scene.as_dict(cfg, sid)
        img.save(out / f"{sid}.png")
        save_labels(lab, out / f"{sid}_labels.png")
        with open(out / f"{sid}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        entry = {"id": sid, "layout": layout, "n_agents": len(scene.agents),
                 "n_obstacles": len(scene.obstacles),
                 "bottleneck": scene.bottleneck}
        if close_prob > 0:
            entry["close"] = close
        index.append(entry)
        if len(previews) < a.preview:
            previews.append((np.asarray(img), PALETTE[lab]))
        if (i + 1) % 200 == 0:
            print("  %d/%d  (%.1f scenes/s)" % (i + 1, a.n, (i + 1) / (time.time() - t0)),
                  file=sys.stderr)

    from dataclasses import asdict
    with open(out / "dataset.json", "w") as fh:
        json.dump({"config": asdict(cfg), "args": vars(a),
                   "n_written": len(index), "n_skipped": skipped,
                   "scenes": index}, fh, indent=1)

    if previews:
        h = 240
        rows = [np.concatenate(
            [np.asarray(Image.fromarray(x).resize((h, h))),
             np.full((h, 4, 3), 255, np.uint8),
             np.asarray(Image.fromarray(y).resize((h, h)))], 1) for x, y in previews]
        Image.fromarray(np.concatenate(rows, 0)).save(out / "preview.png")

    dt = time.time() - t0
    print("wrote %d scenes to %s in %.1fs (%.1f/s); %d rejected as unsolvable"
          % (len(index), out, dt, len(index) / max(dt, 1e-9), skipped))


if __name__ == "__main__":
    main()
