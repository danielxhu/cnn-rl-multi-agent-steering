"""Printable pages for the real hand-drawn test set.

    python scripts/make_sheets.py                    # 42 scenes, 6 per layout
    python scripts/make_sheets.py --n-per-layout 3   # a smaller first batch

Writes, under ``data/real_sketch/``:

    reference.pdf     one page per scene: the scene in the single-pen convention
    drawing.pdf       one page per scene: only the frame, corner markers and id boxes
    scenes/real_NNNN.json   the ground truth, in the generator's schema (+ "sheet_id")

How the pages are used: print both at 100 % ("actual size", no fit-to-page),
put a drawing sheet on top of its reference page, hold both against a window
or a light, and trace the scene with one black marker (~1 mm). The frame,
corner squares and id boxes are already printed on the drawing sheet and are
not traced. Photograph the drawing sheet flat with all four corner squares in
view. The id boxes let the photo be matched to its JSON automatically.

Page geometry lives in ``sheet.py`` so the rectifier reads the same numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: E402,F401
import sheet  # noqa: E402
from config import Config  # noqa: E402
from layouts import LAYOUTS, random_scene  # noqa: E402

MM_PT = 72.0 / 25.4          # points per millimetre
PEN_MM = 1.0                 # line width of traced strokes on the reference


# ------------------------------------------------------------------ drawing

def new_page():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(sheet.PAGE_W / 25.4, sheet.PAGE_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, sheet.PAGE_W)
    ax.set_ylim(0, sheet.PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_furniture(ax, sid: int, title: str, cfg: Config):
    """Frame (the boundary walls), corner markers, id boxes, header."""
    from matplotlib.patches import Rectangle
    band = sheet.FRAME_BAND                                     # boundary wall inside the square
    assert abs(band - cfg.wall_thickness / 2 * sheet.SQ / sheet.WORLD) < 1e-9, "sheet.FRAME_BAND out of sync"
    x0, y0, s = sheet.SQ_X0, sheet.SQ_Y0, sheet.SQ
    for rect in ((x0, y0, s, band), (x0, y0 + s - band, s, band),
                 (x0, y0, band, s), (x0 + s - band, y0, band, s)):
        ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], color="black", lw=0))
    for name, (cx, cy) in sheet.marker_centres().items():
        m = sheet.marker_size(name)
        ax.add_patch(Rectangle((cx - m / 2, cy - m / 2), m, m, color="black", lw=0))
    for (cx, cy), bit in zip(sheet.id_box_centres(), sheet.encode_id(sid)):
        b = sheet.ID_BOX
        ax.add_patch(Rectangle((cx - b / 2, cy - b / 2), b, b, facecolor="black" if bit else "white",
                               edgecolor="black", lw=0.6 * MM_PT * 0.35))
    ax.text(sheet.PAGE_W / 2, sheet.PAGE_H - 10, title, ha="center", va="center", fontsize=11)


def pick_letter_spot(reg, agents, h):
    """Letter centre inside the region, as far from every agent as possible (world units)."""
    long_x = reg.width >= reg.height
    best, best_d = reg.center, -1.0
    for f in np.linspace(0.18, 0.82, 9):
        c = ((reg.x0 + f * reg.width, reg.center[1]) if long_x else (reg.center[0], reg.y0 + f * reg.height))
        d = min([math.dist(c, a.pos) for a in agents] or [1e9])
        if d > best_d:
            best, best_d = c, d
    return best


def draw_scene(ax, scene, cfg: Config):
    """The scene in the single-pen convention: walls as lines, filled obstacles,
    hollow agents with a heading tail, region boxes with S / G."""
    from matplotlib.patches import Circle, Rectangle
    P = sheet.world_to_page
    k = sheet.SQ / sheet.WORLD
    lw = PEN_MM * MM_PT
    boundary = {((0.0, 0.0), (cfg.world, 0.0)), ((cfg.world, 0.0), (cfg.world, cfg.world)),
                ((cfg.world, cfg.world), (0.0, cfg.world)), ((0.0, cfg.world), (0.0, 0.0))}
    for a, b in scene.walls:
        if (tuple(a), tuple(b)) in boundary:
            continue                                  # printed as the frame already
        (ax_, ay_), (bx_, by_) = P(*a), P(*b)
        ax.plot([ax_, bx_], [ay_, by_], color="black", lw=lw, solid_capstyle="round")
    for o in scene.obstacles:
        cx, cy = P(*o.c)
        ax.add_patch(Circle((cx, cy), o.r * k, color="black", lw=0))
    for start, goal in (scene.groups or [(scene.start_region, scene.goal_region)]):
        for reg, ch in ((start, "S"), (goal, "G")):
            (x0, y0), (x1, y1) = P(reg.x0, reg.y0), P(reg.x1, reg.y1)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black", lw=lw * 0.8))
            h = min(reg.width, reg.height, 16.0) * 0.45            # letter height, world units
            lx, ly = P(*pick_letter_spot(reg, scene.agents, h))
            ax.text(lx, ly, ch, ha="center", va="center", fontsize=h * k * MM_PT / 0.72,
                    fontweight="bold", family="DejaVu Sans")
    r = cfg.agent_radius * k
    for a in scene.agents:
        cx, cy = P(*a.pos)
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="black", lw=lw * 0.8))
        L = cfg.style.heading_len_mult * r
        ax.plot([cx, cx + L * math.cos(a.heading)], [cy, cy + L * math.sin(a.heading)],
                color="black", lw=lw * 0.8, solid_capstyle="round")


INSTRUCTIONS = [
    "Put this sheet on top of its REFERENCE page (same scene id), hold both against a window, and trace.",
    "One black marker (about 1 mm). Do not trace the frame, the corner squares or the id boxes.",
    "Agents: small hollow circle + a line from its centre (heading).   Obstacles: outline, then fill in.",
    "Walls: one line.   Start / goal: a box with S or G inside.   Dead space: hatch it or leave it blank.",
    "Photograph the whole sheet flat, with all four corner squares visible.",
]


def draw_instructions(ax):
    for i, line in enumerate(INSTRUCTIONS):
        ax.text(15, 70 - i * 6, line, ha="left", va="center", fontsize=8.5)


# ------------------------------------------------------------------- scenes

def sample_scenes(n_per_layout: int, seed: int, cfg: Config) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for layout in LAYOUTS:
        got = 0
        while got < n_per_layout:
            s = random_scene(rng, cfg, layout=layout, n_agents=int(rng.integers(2, 7)),
                             n_obstacles=int(rng.integers(1, 5)))
            if s is None:
                continue
            s.seed = int(rng.integers(2 ** 31))
            out.append(s)
            got += 1
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Printable reference and drawing pages for hand-drawn scenes.")
    p.add_argument("--out", default=str(_paths.PERCEPTION_DIR / "data" / "real_sketch"))
    p.add_argument("--n-per-layout", type=int, default=6)
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--preview", action="store_true", help="also write PNG previews of the first scene")
    a = p.parse_args(argv)

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    cfg = Config()
    out = Path(a.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    scenes = sample_scenes(a.n_per_layout, a.seed, cfg)
    if len(scenes) >= 2 ** sheet.ID_BITS:
        raise SystemExit(f"at most {2 ** sheet.ID_BITS - 1} scenes fit in the id boxes")

    with PdfPages(out / "reference.pdf") as ref_pdf, PdfPages(out / "drawing.pdf") as draw_pdf:
        for k, scene in enumerate(scenes, 1):
            name = f"real_{k:04d}"
            fig, ax = new_page()
            draw_furniture(ax, k, f"REFERENCE   {name}   ({scene.layout})   - put this UNDER the drawing sheet", cfg)
            draw_scene(ax, scene, cfg)
            ref_pdf.savefig(fig)
            if a.preview and k == 1:
                fig.savefig(out / "preview_reference.png", dpi=120)
            plt.close(fig)

            fig, ax = new_page()
            draw_furniture(ax, k, f"DRAWING SHEET   {name}", cfg)
            draw_instructions(ax)
            draw_pdf.savefig(fig)
            if a.preview and k == 1:
                fig.savefig(out / "preview_drawing.png", dpi=120)
            plt.close(fig)

            rec = scene.as_dict(cfg, name)
            rec["sheet_id"] = k
            with open(out / "scenes" / f"{name}.json", "w") as fh:
                json.dump(rec, fh, indent=1)

    print(f"{len(scenes)} scenes -> {out}/reference.pdf, drawing.pdf, scenes/*.json")


if __name__ == "__main__":
    main()
