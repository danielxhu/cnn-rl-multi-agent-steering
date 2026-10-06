"""Hand-drawn ("sketch") rendering: one black marker on paper, photographed.

    img, lab, drawn = sketch(scene, cfg, rng)     # PIL RGB, uint8 labels, Scene

The scene is drawn in the single-pen convention of
``perception/DESIGN_SKETCH.md`` §1: walls are pen lines, dead space is hatched
or left blank, obstacles are filled or scribbled blobs, agents are small
hollow circles with a heading tail, start / goal regions are boxes with an
S / G inside. The world boundary is the printed band of the drawing sheet.

Three stages, in this order and with one rng:

1. ``perturb``  every primitive is perturbed exactly once (wobbly polylines,
   stroke width profiles, fill style, letter placement) into a ``Drawing``;
2. ``draw``     the same ``Drawing`` is drawn into the image (pen style) and
   into the label map (class indices), in the default draw order
   (dead space, regions, walls, obstacles, agents);
3. ``photo``    residual perspective, lighting, blur, noise, JPEG. The
   perspective is applied to the label map (nearest) and to every coordinate
   of the returned scene.

Invariants:
- Labels are exact by construction: image and label map are drawn from the
  same perturbed points and widths.
- The returned ``Scene`` describes what was *drawn*, not what was sampled
  (DESIGN_SKETCH §3.1 step 4): agent ``pos`` is the centroid of the drawn
  circle, ``heading`` the drawn tail's least-squares direction, obstacles the centroid and
  mean radius of the drawn curve, regions the axis-aligned box of the drawn
  quadrilateral, walls the sampled centre lines -- all through the same
  perspective as the image. "Centroid" is that of the area enclosed by one
  turn of the drawn curve (a gap is closed by its chord, an overshoot cut).
  The tail is drawn from that centroid, so the heat peak sits on ink. The instance head is trained from it.
- One y flip: ``_to_px`` mirrors ``render._to_px`` (pinned by
  ``tests/run_tests.py::test_sketch_labels_contain_every_class``).
- Pixel sizes in ``SketchStyle`` are at 512 px and scale with ``img_size``.
- Same scene and rng seed -> identical image, labels and scene. Re-rendering a
  written sketch JSON does *not* reproduce the image: that JSON is already
  the perturbed geometry.
- ``SketchStyle()`` is the first style (v1, ``generate.py --style sketch``),
  pixel-identical to the datasets made with it. ``SKETCH_V2``
  (``--style sketch2``) widens it after the first real drawing (DESIGN_SKETCH
  §8): a "digital" medium (tablet drawing: white, no photo effects), thinner
  pens, sparse-hatch and zigzag obstacle fills, elongated obstacles, region
  boxes drawn past their corners, smaller letters in the pen's own width. Its
  extra random draws only happen when those options are on, which is what
  keeps v1 unchanged. ``SKETCH_V3`` (``--style sketch3``) adds what the first
  v2 trial still missed: S / G in many hands (eight Hershey faces, italic,
  and procedural pen-stroke letters, all sheared, stretched and elastically
  warped), a harsher camera (noise, blur, JPEG, shadow, low resolution,
  gamma), pen dropouts and stray marks. The same rule holds: v1 and v2 stay
  pixel-identical.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PIL import Image

from config import C_AGENT, C_GOAL, C_OBST, C_START, C_WALL, Config
from reachability import unreachable_free
from scene import Agent, Obstacle, Region, Scene, boundary_walls

REF = 512.0                                     # SketchStyle pixel sizes are at this image size
SUBPIX = 4                                      # cv2 fixed-point bits for sub-pixel strokes


@dataclass
class SketchStyle:
    """Pen, paper and photo ranges (DESIGN_SKETCH §3.1, §3.2). Pixels at 512."""

    ink: tuple = (15.0, 55.0)                   # uniform per RGB channel: one near-black marker
    paper: tuple = (246, 244, 238)
    paper_jitter: tuple = (-12.0, 8.0)          # per channel
    pen_width: tuple = (3.0, 5.0)               # base stroke width
    wall_width_mult: tuple = (1.0, 2.2)         # single or gone over twice
    width_var: float = 0.35                     # smooth variation along a stroke
    printed_ink: tuple = (18, 18, 18)           # the printed frame band

    wall_wobble: tuple = (1.5, 3.0)             # perpendicular noise amplitude, px
    wall_overshoot: float = 0.05                # fraction of the segment length, per end
    obstacle_wobble: float = 0.06               # relative radius noise
    stretch: float = 0.12                       # independent x / y stretch, obstacles and agents
    agent_radius_mult: tuple = (0.8, 1.25)
    agent_wobble: float = 0.08
    agent_close: tuple = (-0.35, 0.5)           # rad: gap (<0) or overshoot (>0) of the circle
    tail_mult: tuple = (0.8, 1.2)
    heading_sigma: float = 0.06                 # rad
    region_corner_sigma: float = 3.0            # px

    solid_fill_prob: float = 0.45               # else scribbled
    scribble_spacing: tuple = (3.5, 6.0)
    hatch_prob: float = 0.6                     # dead space hatched (labelled wall), else blank
    hatch_spacing: tuple = (9.0, 15.0)
    letter_height_frac: float = 0.55            # of the region's shorter side
    letter_max_px: float = 40.0
    letter_rot_deg: float = 15.0

    persp_px: float = 8.0                       # residual perspective, per corner
    light_falloff: tuple = (0.05, 0.30)
    blur_sigma: tuple = (0.4, 1.0)
    noise_std: float = 4.0
    jpeg_quality: tuple = (50, 85)

    label_extra_agent: int = 3                  # label strokes wider than the ink, as in render.py
    label_extra_wall: int = 2

    # --- v2 options (defaults reproduce v1 exactly; see SKETCH_V2) ---
    digital_prob: float = 0.0                   # tablet drawing instead of a photographed page
    digital_pen_width: tuple = (1.2, 3.5)
    digital_ink: tuple = (0.0, 30.0)
    digital_paper_drop: tuple = (0.0, 4.0)      # white minus this, per channel
    digital_blur: tuple = (0.0, 0.5)
    sparse_fill_prob: float = 0.0               # of the non-solid obstacles: sparse hatch ...
    zigzag_fill_prob: float = 0.0               # ... or one back-and-forth scribble; else dense
    sparse_spacing: tuple = (7.0, 14.0)
    zigzag_spacing: tuple = (5.0, 10.0)
    fill_overshoot: tuple = (0.0, 0.0)          # px a non-solid fill may cross the outline
    obstacle_stretch: float | None = None       # None: `stretch`
    region_overshoot: float = 0.03              # region sides drawn past the corners
    letter_scale: tuple = (0.8, 1.0)            # of the largest letter that fits
    letter_in_pen: bool = False                 # letter stroke = the pen's width, not h / 12

    # --- v3 options (defaults off; see SKETCH_V3) ---
    letter_variety: bool = False                # many faces + procedural strokes, warped
    letter_proc_prob: float = 0.5               # of those: pen-stroke letters, else a font
    letter_shear: float = 0.35                  # tan of the slant, +/-
    letter_aspect: tuple = (0.7, 1.35)          # width / height stretch
    letter_rot_v3: float = 25.0                 # deg
    letter_warp: float = 0.06                   # elastic warp amplitude, fraction of h
    harsh_camera: bool = False
    noise_std_range: tuple = (2.0, 14.0)
    blur_range: tuple = (0.3, 2.0)
    jpeg_range: tuple = (30, 90)
    light_range: tuple = (0.05, 0.50)
    persp_px_v3: float = 12.0
    shadow_prob: float = 0.3
    shadow_strength: tuple = (0.55, 0.9)        # brightness kept in the shadow
    lowres_prob: float = 0.3
    lowres_scale: tuple = (0.35, 0.8)
    gamma_range: tuple = (0.7, 1.4)
    dropout_prob: float = 0.0                   # pen running dry: gaps along strokes
    dropout_frac: tuple = (0.05, 0.2)
    stray_prob: float = 0.0                     # short stray marks, labelled background
    stray_count: tuple = (1, 4)


SKETCH_V2 = SketchStyle(
    pen_width=(2.0, 5.0), digital_prob=0.35,
    sparse_fill_prob=0.30, zigzag_fill_prob=0.15, fill_overshoot=(0.0, 3.0),
    obstacle_stretch=0.22, agent_radius_mult=(0.7, 1.3), region_overshoot=0.10,
    letter_scale=(0.55, 1.0), letter_in_pen=True,
)
SKETCH_V3 = replace(
    SKETCH_V2, letter_variety=True, letter_scale=(0.4, 1.0), harsh_camera=True,
    dropout_prob=0.35, stray_prob=0.4,
)
STYLES = {"sketch": SketchStyle(), "sketch2": SKETCH_V2, "sketch3": SKETCH_V3}


# --------------------------------------------------------------- primitives

def _to_px(cfg, x, y):
    """World (y up) -> image px (y down); the same flip as render._to_px."""
    return (x * cfg.scale, (cfg.world - y) * cfg.scale)


def _to_world(cfg, px, py):
    return (px / cfg.scale, cfg.world - py / cfg.scale)


def _lowfreq(n, rng, knots, amp):
    """Smooth noise: piecewise-linear interpolation of knots + 2 normal samples."""
    ctrl = rng.normal(0.0, 1.0, knots + 2)
    return np.interp(np.linspace(0, knots + 1, n), np.arange(knots + 2), ctrl) * amp


def _wobbly_line(p0, p1, rng, amp, overshoot=0.0, step=3.0, anchored=False):
    """Hand-drawn straight line: points every ~step px, perpendicular smooth noise.
    ``anchored``: no offset at p0 (the pen is put down exactly there)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    n = max(8, int(L / step))
    t = np.linspace(-rng.uniform(0, overshoot), 1 + rng.uniform(0, overshoot), n)
    u = (p1 - p0) / (L + 1e-9)
    off = _lowfreq(n, rng, max(2, int(L / 60)), amp) + rng.normal(0, amp * 0.15, n)
    if anchored:
        off = off - off[0] * np.linspace(1, 0, n)
    return p0 + np.outer(t, p1 - p0) + np.outer(off, [-u[1], u[0]])


def _wobbly_closed(c, r, rng, amp_rel, stretch, close=0.0, drift=0.0):
    """Hand-drawn circle: radius noise, x/y stretch, a radius drift along the
    stroke and `close` extra radians (negative: a gap; positive: overshoot)."""
    n = max(24, int(r * 1.2))
    th = rng.uniform(0, 2 * math.pi) + np.linspace(0, 2 * math.pi + close, n)
    rr = r * (1 + _lowfreq(n, rng, 4, amp_rel)) * np.linspace(1, 1 + rng.uniform(-drift, drift), n)
    sx, sy = 1 + rng.uniform(-stretch, stretch), 1 + rng.uniform(-stretch, stretch)
    return np.stack([c[0] + rr * np.cos(th) * sx, c[1] + rr * np.sin(th) * sy], 1), th


def _area_centroid(poly):
    """Centroid of the area a closed polygon encloses (shoelace)."""
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return poly.mean(0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * a)


def _widths(n, rng, w, var):
    """Per-segment stroke width with smooth variation (the pen pressure)."""
    return np.clip(w * (1 + _lowfreq(max(2, n - 1), rng, 3, var)), 1.0, None)


@dataclass
class Stroke:
    pts: np.ndarray                              # (n, 2) px, pre-perspective
    widths: np.ndarray                           # (n - 1,) px, one per segment


def _stroke(pts, rng, w, var):
    return Stroke(pts, _widths(len(pts), rng, w, var))


@dataclass
class DrawnObstacle:
    poly: np.ndarray                             # closed curve, px
    outline: Stroke
    solid: bool
    scribble: list = field(default_factory=list)  # [Stroke], clipped to the poly when drawn
    overshoot: float = 0.0                       # px the scribble may cross the outline


@dataclass
class DrawnAgent:
    ring: Stroke
    ring_turn: int                               # points of `ring` within one full turn
    tail: Stroke
    centre: np.ndarray                           # where the tail starts, px
    tip: np.ndarray


@dataclass
class DrawnRegion:
    quad: np.ndarray                             # (4, 2) px
    sides: list                                  # 4 Strokes
    cls: int
    letter: str
    letter_centre: tuple = (0.0, 0.0)
    letter_h: float = 0.0
    letter_rot: float = 0.0
    group: int = 0
    letter_th: int = 0                           # stroke width; 0: h / 12
    letter_form: dict | None = None              # v3: face / strokes, shear, aspect, warp seed


@dataclass
class Drawing:
    """Every perturbed primitive of one scene; drawn once into image and labels."""

    n: int
    ink: tuple
    paper: np.ndarray
    band_px: float                               # printed frame band width
    hatch: list                                  # [Stroke]; empty when dead space is blank
    dead: np.ndarray | None                      # bool (n, n): dead space, grown to the wall centre
    regions: list
    walls: list                                  # [(Stroke, sampled segment)]
    obstacles: list
    agents: list
    printed_ink: tuple = (18, 18, 18)
    label_extra_wall: int = 2
    label_extra_agent: int = 3
    digital: bool = False                        # tablet drawing: no photo effects
    stray: list = field(default_factory=list)    # v3: [Stroke] marks that are nothing
    dropout_seed: int | None = None              # v3: gaps along the ink
    dropout_frac: float = 0.0


# ----------------------------------------------------------------- perturb

def _dead_mask(cfg, scene, n):
    mask = unreachable_free(cfg, scene)
    if not mask.any():
        return None
    big = np.flipud(cv2.resize(mask.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST))
    # grow up to the wall centre line, where the pen line is drawn (the drawn line is
    # much thinner than the 3-unit wall, so render.py's quarter-thickness seam fill
    # would leave a blank strip between the hatching and the line)
    grow = max(1, int(round(cfg.wall_half * cfg.scale)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(big, k).astype(bool)


def _hatch_strokes(dead, rng, st, k, mk):
    """Parallel wobbly lines over the bounding box of the dead space."""
    ys, xs = np.nonzero(dead)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    R = 0.5 * math.hypot(x1 - x0, y1 - y0) + 4
    ang = math.radians(rng.uniform(25, 65)) * rng.choice([-1, 1]) + (math.pi if rng.random() < 0.5 else 0)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    step = rng.uniform(*st.hatch_spacing) * k
    out = []
    for off in np.arange(-R, R, step):
        a, b = c + off * nrm - R * d, c + off * nrm + R * d
        out.append(_stroke(_wobbly_line(a, b, rng, 1.5 * k), rng, mk * 0.6, st.width_var))
    return out


def _scribble_strokes(c, R, rng, st, k, mk, spacing=None, width=0.8):
    """Parallel strokes at a random angle covering the whole disk of radius R
    around c (the fix for the prototype's white wedge: the strokes are laid out
    in a frame rotated with them, so every offset across the disk is covered)."""
    ang = rng.uniform(0, math.pi)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    step = rng.uniform(*(spacing or st.scribble_spacing)) * k
    out = []
    for off in np.arange(-R + rng.uniform(0, step), R, step):
        half = math.sqrt(max(R * R - off * off, 0.0)) + 2 * k
        a, b = c + off * nrm - half * d, c + off * nrm + half * d
        out.append(_stroke(_wobbly_line(a, b, rng, 1.2 * k, step=3.0 * k), rng, mk * width, st.width_var))
    return out


def _zigzag_stroke(c, R, rng, st, k, mk):
    """One continuous back-and-forth scribble over the disk of radius R."""
    ang = rng.uniform(0, math.pi)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    step = rng.uniform(*st.zigzag_spacing) * k
    pts, side = [], 1.0
    for off in np.arange(-R + rng.uniform(0, step), R, step):
        half = math.sqrt(max(R * R - off * off, 0.0)) * rng.uniform(0.75, 1.0)
        pts.append(c + off * nrm + side * half * d)
        side = -side
    if len(pts) < 2:
        return []
    pts = np.array(pts) + rng.normal(0, 1.0 * k, (len(pts), 2))
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / (3 * k))),
                                        endpoint=False) for a, b in zip(pts[:-1], pts[1:])] + [pts[-1:]])
    return [_stroke(dense, rng, mk * 0.7, st.width_var)]


HERSHEY_FACES = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                 cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                 cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

# pen-stroke skeletons in a unit box (x right, y down); each list is one stroke
LETTER_SKELETONS = {
    "S": [[[(0.85, 0.15), (0.5, 0.0), (0.15, 0.18), (0.25, 0.42), (0.75, 0.58), (0.85, 0.82),
            (0.5, 1.0), (0.12, 0.85)]],
          [[(0.8, 0.1), (0.35, 0.02), (0.2, 0.3), (0.8, 0.65), (0.6, 0.98), (0.15, 0.9)]]],
    "G": [[[(0.85, 0.18), (0.55, 0.0), (0.18, 0.15), (0.05, 0.5), (0.2, 0.86), (0.55, 1.0),
            (0.85, 0.85), (0.9, 0.58), (0.55, 0.58)]],
          [[(0.85, 0.2), (0.5, 0.0), (0.12, 0.25), (0.1, 0.75), (0.5, 1.0), (0.9, 0.8), (0.9, 0.55)],
           [(0.55, 0.55), (0.95, 0.55)]],
          [[(0.8, 0.15), (0.45, 0.0), (0.1, 0.35), (0.25, 0.9), (0.7, 0.95), (0.85, 0.6),
            (0.6, 0.6), (0.95, 0.6)]]],
}


def _catmull_rom(pts, n_per=8):
    """Smooth curve through the control points (end points repeated)."""
    P = np.vstack([pts[:1], pts, pts[-1:]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def _letter_form(ch, rng, st):
    """Random hand for one letter (v3): a face or pen strokes, plus the warp."""
    form = {"shear": rng.uniform(-st.letter_shear, st.letter_shear),
            "aspect": rng.uniform(*st.letter_aspect),
            "warp": rng.uniform(0, st.letter_warp), "seed": int(rng.integers(2 ** 31))}
    if rng.random() < st.letter_proc_prob:
        sk = LETTER_SKELETONS[ch][int(rng.integers(len(LETTER_SKELETONS[ch])))]
        form["strokes"] = [np.array(stroke) + rng.normal(0, 0.05, (len(stroke), 2)) for stroke in sk]
    else:
        form["face"] = HERSHEY_FACES[int(rng.integers(len(HERSHEY_FACES)))] | (
            cv2.FONT_ITALIC if rng.random() < 0.3 else 0)
    return form


def _letter_alpha_v3(n, ch, centre, h, rot, th, form):
    """Coverage (n, n) uint8 of one letter in the hand `form`, height ~h px."""
    th = max(1, min(int(th or round(h / 12)), int(h / 7)))   # never a blob: at most h / 7
    pad = int(h) + 4 * th + 8
    S = 3 * pad
    tile = np.zeros((S, S), np.uint8)
    if "strokes" in form:
        for stroke in form["strokes"]:
            curve = _catmull_rom(stroke) * h + (pad, pad)
            q = np.round(curve * (1 << SUBPIX)).astype(np.int32)
            cv2.polylines(tile, [q], False, 255, th, cv2.LINE_AA, SUBPIX)
    else:
        scale = h / 22.0
        (w, hh), _ = cv2.getTextSize(ch, form["face"], scale, th)
        if hh > 0:
            scale *= h / hh                           # faces differ in cap height
        cv2.putText(tile, ch, (pad, pad + int(h)), form["face"], scale, 255, th, cv2.LINE_AA)
    ys, xs = np.nonzero(tile)
    if len(xs) == 0:
        return np.zeros((n, n), np.uint8)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    # elastic warp: smooth random displacement, amplitude form["warp"] * h
    if form["warp"] > 0:
        r = np.random.default_rng(form["seed"])
        g = max(3, int(S / 8))
        dx = cv2.resize(r.normal(0, 1, (g, g)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(r.normal(0, 1, (g, g)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        a = np.float32(form["warp"] * h)
        tile = cv2.remap(tile, (xx + a * dx).astype(np.float32), (yy + a * dy).astype(np.float32),
                         cv2.INTER_LINEAR)
    # shear, aspect and rotation about the ink centre, then onto the canvas
    A = np.array([[form["aspect"], form["shear"], 0], [0, 1, 0]], np.float64)
    R = cv2.getRotationMatrix2D((0, 0), rot, 1.0)
    M = R[:, :2] @ A[:, :2]
    M = np.hstack([M, (np.array(centre) - M @ np.array([cx, cy]))[:, None]])
    return cv2.warpAffine(tile, M, (n, n))


def _stray_strokes(n, rng, st, k, mk):
    """A few short marks that are not part of the scene (labels untouched)."""
    out = []
    for _ in range(int(rng.integers(st.stray_count[0], st.stray_count[1] + 1))):
        a = rng.uniform(0.08 * n, 0.92 * n, 2)
        L = rng.uniform(4, 25) * k
        ang = rng.uniform(0, 2 * math.pi)
        b = a + L * np.array([math.cos(ang), math.sin(ang)])
        out.append(_stroke(_wobbly_line(a, b, rng, 0.8 * k, step=2.0 * k), rng, mk * rng.uniform(0.5, 1.0),
                           st.width_var))
    return out


def _seg_dist(p, a, b):
    """Distance from each point of p (m, 2) to the segment a-b."""
    ab = b - a
    t = np.clip((p - a) @ ab / max(float(ab @ ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _letter_spot(quad, h, agents, k):
    """Letter centre inside the drawn box, as far as possible from every drawn
    agent (its circle and its tail). The prototype put letters on agents."""
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    m = 0.6 * h + 2 * k
    mid = quad.mean(0)
    if x1 - x0 <= 2 * m or y1 - y0 <= 2 * m or not agents:
        return mid
    gx, gy = np.meshgrid(np.linspace(x0 + m, x1 - m, 7), np.linspace(y0 + m, y1 - m, 15))
    cand = np.stack([gx.ravel(), gy.ravel()], 1)
    d = np.full(len(cand), np.inf)
    for ag in agents:
        r = float(np.linalg.norm(ag.ring.pts - ag.centre, axis=1).max())
        d = np.minimum(d, np.linalg.norm(cand - ag.centre, axis=1) - r)
        d = np.minimum(d, _seg_dist(cand, ag.centre, ag.tip))
    # beyond ~2 letter heights an agent no longer matters: centre the letter
    d = np.minimum(d, 2 * h) - 0.02 * np.linalg.norm(cand - mid, axis=1)
    return cand[int(np.argmax(d))]


def perturb(scene, cfg, rng, st: SketchStyle, fill_unreachable=True) -> Drawing:
    """Step 1: every primitive perturbed once, in pixel coordinates."""
    n = cfg.img_size
    k = n / REF
    P = lambda x, y: np.array(_to_px(cfg, x, y))  # noqa: E731
    digital = st.digital_prob > 0 and rng.random() < st.digital_prob
    if digital:                                        # tablet: near-pure black on white
        ink = tuple(float(v) for v in rng.uniform(*st.digital_ink, 3))
        paper = 255.0 - rng.uniform(*st.digital_paper_drop, 3)
        mk = rng.uniform(*st.digital_pen_width) * k
    else:
        ink = tuple(float(v) for v in rng.uniform(*st.ink, 3))
        paper = np.clip(np.array(st.paper, float) + rng.uniform(*st.paper_jitter, 3), 0, 255)
        mk = rng.uniform(*st.pen_width) * k

    dead = _dead_mask(cfg, scene, n) if fill_unreachable else None
    hatch = []
    if dead is not None and rng.random() < st.hatch_prob:
        hatch = _hatch_strokes(dead, rng, st, k, mk)
    else:
        dead = None                                    # blank dead space is background

    regions = []
    pairs = scene.groups or [(scene.start_region, scene.goal_region)]
    for gi, (start, goal) in enumerate(pairs):
        for reg, ch, cls in ((start, "S", C_START), (goal, "G", C_GOAL)):
            if reg is None:
                continue
            a, b = P(reg.x0, reg.y1), P(reg.x1, reg.y0)
            quad = np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])
            quad = quad + rng.normal(0, st.region_corner_sigma * k, (4, 2))
            sides = [_stroke(_wobbly_line(quad[i], quad[(i + 1) % 4], rng, 1.3 * k, st.region_overshoot),
                             rng, mk * 0.6, st.width_var) for i in range(4)]
            regions.append(DrawnRegion(quad, sides, cls, ch, group=gi))

    boundary = {tuple(map(tuple, s)) for s in boundary_walls(cfg.world)}
    walls = []
    for a, b in scene.walls:
        if (tuple(a), tuple(b)) in boundary:
            continue                                   # printed, drawn crisp in `draw`
        pts = _wobbly_line(P(*a), P(*b), rng, rng.uniform(*st.wall_wobble) * k, st.wall_overshoot)
        w = mk * rng.uniform(*st.wall_width_mult)
        walls.append((_stroke(pts, rng, w, st.width_var), (a, b)))

    obstacles = []
    for o in scene.obstacles:
        c, r = P(*o.c), o.r * cfg.scale
        stretch = st.stretch if st.obstacle_stretch is None else st.obstacle_stretch
        poly, _ = _wobbly_closed(c, r, rng, st.obstacle_wobble, stretch)
        outline = _stroke(np.vstack([poly, poly[:2]]), rng, mk * 0.7, st.width_var)
        solid = rng.random() < st.solid_fill_prob
        scr, over = [], 0.0
        if not solid:
            R = float(np.linalg.norm(poly - c, axis=1).max())
            kind = rng.random() if (st.sparse_fill_prob or st.zigzag_fill_prob) else 1.0
            if kind < st.sparse_fill_prob:
                scr = _scribble_strokes(c, R, rng, st, k, mk, spacing=st.sparse_spacing, width=0.6)
            elif kind < st.sparse_fill_prob + st.zigzag_fill_prob:
                scr = _zigzag_stroke(c, R, rng, st, k, mk)
            else:
                scr = _scribble_strokes(c, R, rng, st, k, mk)
            if st.fill_overshoot[1] > 0:
                over = rng.uniform(*st.fill_overshoot) * k
        obstacles.append(DrawnObstacle(poly, outline, solid, scr, over))

    agents = []
    r0 = cfg.agent_radius * cfg.scale
    for ag in scene.agents:
        c = P(*ag.pos)
        ring_pts, th = _wobbly_closed(c, r0 * rng.uniform(*st.agent_radius_mult), rng,
                                      st.agent_wobble, st.stretch,
                                      close=rng.uniform(*st.agent_close), drift=0.1)
        turn = int(np.searchsorted(th - th[0], 2 * math.pi, side="right"))
        c = _area_centroid(ring_pts[: max(3, turn)])   # the tail starts at the drawn circle's centre
        L = cfg.style.heading_len_mult * r0 * rng.uniform(*st.tail_mult)
        h = ag.heading + rng.normal(0, st.heading_sigma)
        tip = c + L * np.array([math.cos(h), -math.sin(h)])
        agents.append(DrawnAgent(_stroke(ring_pts, rng, mk * 0.6, st.width_var), turn,
                                 _stroke(_wobbly_line(c, tip, rng, 0.8 * k, anchored=True), rng, mk * 0.6, st.width_var),
                                 c, tip))

    for reg in regions:                                # letters last: they avoid the drawn agents
        x0, y0 = reg.quad.min(0)
        x1, y1 = reg.quad.max(0)
        reg.letter_h = min(st.letter_height_frac * min(x1 - x0, y1 - y0), st.letter_max_px * k)
        reg.letter_h *= rng.uniform(*st.letter_scale)
        if st.letter_in_pen:
            reg.letter_th = max(1, int(round(mk * 0.7)))
        if st.letter_variety:
            reg.letter_form = _letter_form(reg.letter, rng, st)
        reg.letter_centre = tuple(_letter_spot(reg.quad, reg.letter_h, agents, k)
                                  + rng.normal(0, 1.5 * k, 2))
        reg.letter_rot = rng.uniform(-st.letter_rot_deg, st.letter_rot_deg)
        if st.letter_variety:
            reg.letter_rot = rng.uniform(-st.letter_rot_v3, st.letter_rot_v3)

    stray = _stray_strokes(n, rng, st, k, mk) if st.stray_prob > 0 and rng.random() < st.stray_prob else []
    drop_seed, drop_frac = None, 0.0
    if st.dropout_prob > 0 and rng.random() < st.dropout_prob:
        drop_seed, drop_frac = int(rng.integers(2 ** 31)), rng.uniform(*st.dropout_frac)

    return Drawing(n, ink, paper, cfg.wall_half * cfg.scale, hatch, dead, regions, walls,
                   obstacles, agents, st.printed_ink, st.label_extra_wall, st.label_extra_agent,
                   digital, stray, drop_seed, drop_frac)


# -------------------------------------------------------------------- draw

def _polyline(canvas, s: Stroke, value, extra=0, aa=True):
    q = np.round(s.pts * (1 << SUBPIX)).astype(np.int64)
    kind = cv2.LINE_AA if aa else cv2.LINE_8
    for i, w in enumerate(s.widths):
        cv2.line(canvas, (int(q[i, 0]), int(q[i, 1])), (int(q[i + 1, 0]), int(q[i + 1, 1])),
                 value, max(1, int(round(w + extra))), kind, SUBPIX)


def _poly_mask(n, poly):
    m = np.zeros((n, n), np.uint8)
    cv2.fillPoly(m, [np.round(poly * (1 << SUBPIX)).astype(np.int32)], 1, cv2.LINE_8, SUBPIX)
    return m.astype(bool)


def _letter_alpha(n, ch, centre, h, rot, th=0):
    """Coverage (n, n) uint8 of one Hershey-script letter of height ~h px,
    stroke `th` px (0: h / 12, at least 2)."""
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    scale = h / 22.0
    th = th or max(2, int(round(h / 12)))
    (w, hh), _ = cv2.getTextSize(ch, font, scale, th)
    tile = np.zeros((hh * 3, w * 3), np.uint8)
    cv2.putText(tile, ch, (w, 2 * hh), font, scale, 255, th, cv2.LINE_AA)
    ys, xs = np.nonzero(tile)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2   # ink centre, not the box
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    M[:, 2] += (centre[0] - cx, centre[1] - cy)
    return cv2.warpAffine(tile, M, (n, n))


def draw(d: Drawing, parts=False):
    """Step 2: (image float32 (n,n,3), labels uint8 (n,n)[, parts]).

    Labels are painted in the default draw order. The image needs no order:
    every mark is the same ink, so all strokes accumulate into one coverage
    mask that is laid over the paper and the printed band once.

    With ``parts=True`` also returns two uint8 maps, each agent's drawn *ring*
    and drawn *tail* (value i + 1 for agent i), drawn with the label strokes;
    tests use them to check the written poses against the drawing.
    """
    n = d.n
    lab = np.zeros((n, n), np.uint8)
    ink = np.zeros((n, n), np.uint8)                   # ink coverage, 0..255

    # dead space: hatched and labelled wall (blank dead space stays background)
    if d.dead is not None:
        lab[d.dead] = C_WALL
        h = np.zeros((n, n), np.uint8)
        for s in d.hatch:
            _polyline(h, s, 255)
        keep = d.dead.copy()
        for o in d.obstacles:                          # nobody hatches inside an obstacle
            keep &= ~_poly_mask(n, o.poly)
        h[~keep] = 0
        np.maximum(ink, h, out=ink)

    for reg in d.regions:
        lab[_poly_mask(n, reg.quad)] = reg.cls
        for s in reg.sides:
            _polyline(ink, s, 255)
        if reg.letter_form is not None:
            alpha = _letter_alpha_v3(n, reg.letter, reg.letter_centre, reg.letter_h, reg.letter_rot,
                                     reg.letter_th, reg.letter_form)
        else:
            alpha = _letter_alpha(n, reg.letter, reg.letter_centre, reg.letter_h, reg.letter_rot,
                                  reg.letter_th)
        np.maximum(ink, alpha, out=ink)

    # printed boundary band: crisp, in the image layer below the ink
    b = int(round(d.band_px))
    band = np.zeros((n, n), bool)
    band[:b], band[n - b:], band[:, :b], band[:, n - b:] = True, True, True, True
    lab[band] = C_WALL
    for s, _ in d.walls:
        _polyline(ink, s, 255)
        _polyline(lab, s, C_WALL, extra=d.label_extra_wall, aa=False)

    for o in d.obstacles:
        m = _poly_mask(n, o.poly)
        lab[m] = C_OBST
        a = np.zeros((n, n), np.uint8)
        if o.solid:
            cv2.fillPoly(a, [np.round(o.poly * (1 << SUBPIX)).astype(np.int32)], 255, cv2.LINE_AA, SUBPIX)
        else:
            for s in o.scribble:
                _polyline(a, s, 255)
            clip = m
            if o.overshoot >= 1:                       # a hand fill crosses its outline a little
                r = int(round(o.overshoot))
                clip = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
            a[~clip] = 0
        _polyline(a, o.outline, 255)
        np.maximum(ink, a, out=ink)

    rings = np.zeros((n, n), np.uint8) if parts else None
    tails = np.zeros((n, n), np.uint8) if parts else None
    extra = d.label_extra_agent
    for i, ag in enumerate(d.agents):
        for s in (ag.ring, ag.tail):
            _polyline(ink, s, 255)
            _polyline(lab, s, C_AGENT, extra=extra, aa=False)
        if parts:
            _polyline(rings, ag.ring, i + 1, extra=extra, aa=False)
            _polyline(tails, ag.tail, i + 1, extra=extra, aa=False)

    for s in d.stray:                                  # marks that are nothing: ink only
        _polyline(ink, s, 255)
    if d.dropout_seed is not None:                     # pen running dry: smooth gaps in the ink
        r = np.random.default_rng(d.dropout_seed)
        g = max(4, n // 24)
        field_ = cv2.resize(r.random((g, g)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        ink = np.where(field_ < np.quantile(field_, d.dropout_frac), ink * 0.15, ink).astype(np.uint8)

    img = np.empty((n, n, 3), np.float32)
    img[:] = d.paper
    img[band] = d.printed_ink
    a = ink.astype(np.float32)[..., None] / 255
    img = img * (1 - a) + np.asarray(d.ink, np.float32) * a
    return (img, lab, (rings, tails)) if parts else (img, lab)


# ------------------------------------------------------------------- photo

def photo(img, lab, rng, st: SketchStyle, extra_maps=(), digital=False):
    """Step 3: (uint8 image, labels, [extra maps], H). H maps drawn px -> photo px.

    A digital drawing (tablet) has no camera: H is the identity and only a
    light blur is applied (anti-aliasing of the export), no light, noise or JPEG.

    Where the perspective pulls the page edge inward, the uncovered border is
    filled with the printed band (ink, labelled wall), not replicated: replicating
    would smear any stroke that touches the edge along the whole border.
    """
    n = img.shape[0]
    k = n / REF
    if digital:
        sigma = rng.uniform(*st.digital_blur) * k
        if sigma > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        return (np.clip(img, 0, 255).astype(np.uint8), lab, list(extra_maps),
                np.eye(3, dtype=np.float64))
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    pp = st.persp_px_v3 if st.harsh_camera else st.persp_px
    dst = src + rng.uniform(-pp * k, pp * k, (4, 2)).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    def warp(m, interp, fill):
        return cv2.warpPerspective(m, H, (n, n), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=fill)

    img = warp(img, cv2.INTER_LINEAR, tuple(float(v) for v in st.printed_ink))
    lab = warp(lab, cv2.INTER_NEAREST, C_WALL)
    extra_maps = [warp(m, cv2.INTER_NEAREST, 0) for m in extra_maps]

    yy, xx = np.mgrid[0:n, 0:n] / n
    d2 = (xx - rng.uniform(0, 1)) ** 2 + (yy - rng.uniform(0, 1)) ** 2
    if not st.harsh_camera:
        img = img * (1 - rng.uniform(*st.light_falloff) * d2 / d2.max())[..., None]
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*st.blur_sigma) * k)
        img = np.clip(img + rng.normal(0, st.noise_std, img.shape), 0, 255).astype(np.uint8)
        q = int(rng.integers(st.jpeg_quality[0], st.jpeg_quality[1] + 1))
    else:
        img = _harsh_camera(img, rng, st, k, xx, yy, d2)
        q = int(rng.integers(st.jpeg_range[0], st.jpeg_range[1] + 1))
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"), np.uint8)
    return img, lab, extra_maps, H


def _harsh_camera(img, rng, st, k, xx, yy, d2):
    """v3 photo: stronger light fall-off, a soft shadow edge, gamma, low
    resolution, heavier blur and noise. Geometry is untouched (labels stay)."""
    n = img.shape[0]
    img = img * (1 - rng.uniform(*st.light_range) * d2 / d2.max())[..., None]
    if rng.random() < st.shadow_prob:                  # a hand or phone casting a shadow
        ang = rng.uniform(0, 2 * math.pi)
        t = (xx - 0.5) * math.cos(ang) + (yy - 0.5) * math.sin(ang) - rng.uniform(-0.3, 0.3)
        edge = 1 / (1 + np.exp(-t / rng.uniform(0.01, 0.08)))
        img = img * (1 - (1 - rng.uniform(*st.shadow_strength)) * edge)[..., None]
    g = rng.uniform(*st.gamma_range)
    img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** g
    if rng.random() < st.lowres_prob:                  # a far or cheap camera
        m = max(32, int(n * rng.uniform(*st.lowres_scale)))
        img = cv2.resize(cv2.resize(img, (m, m), interpolation=cv2.INTER_AREA), (n, n),
                         interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*st.blur_range) * k)
    img = img + rng.normal(0, rng.uniform(*st.noise_std_range), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------- drawn -> scene JSON

def _warp(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(1, -1, 2), H)[0]


def _line_angle(pts):
    """World heading of a drawn stroke: its least-squares direction, oriented
    first point -> last. The wobble moves the stroke's end points too, so the
    angle of (tip - root) is not what was drawn."""
    mu = pts.mean(0)
    u = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
    if np.dot(u, pts[-1] - pts[0]) < 0:
        u = -u
    return math.atan2(-u[1], u[0])                     # image y is down


def drawn_scene(d: Drawing, H, scene, cfg) -> Scene:
    """Step 4: the scene as it was drawn, through the photo's perspective."""
    W = lambda p: _to_world(cfg, float(p[0]), float(p[1]))  # noqa: E731
    walls = []
    for a, b in scene.walls:                           # centre lines as sampled, warped
        pa, pb = _warp(H, [_to_px(cfg, *a), _to_px(cfg, *b)])
        walls.append((W(pa), W(pb)))

    obstacles = []
    for o in d.obstacles:
        poly = _warp(H, o.poly)
        c = _area_centroid(poly)
        r = float(np.linalg.norm(poly - c, axis=1).mean()) / cfg.scale
        obstacles.append(Obstacle(c=W(c), r=r))

    regs = {}
    for reg in d.regions:
        q = _warp(H, reg.quad)
        (x0, y1), (x1, y0) = W(q.min(0)), W(q.max(0))  # px min is world top-left
        # sorted on purpose: Region.__post_init__ loses y1 when given y0 > y1
        regs[(reg.group, reg.cls)] = Region(x0, y0, x1, y1)
    pairs = scene.groups or [(scene.start_region, scene.goal_region)]
    groups = [(regs.get((gi, C_START)), regs.get((gi, C_GOAL))) for gi in range(len(pairs))]
    goal_map = {g: groups[gi][1] for gi, (_, g) in enumerate(pairs)}

    agents = []
    for ag, da in zip(scene.agents, d.agents):
        ring = _warp(H, da.ring.pts[: max(3, da.ring_turn)])
        c = _area_centroid(ring)
        tail = _warp(H, da.tail.pts)
        heading = _line_angle(tail)
        goal = goal_map.get(ag.goal_region) if ag.goal_region is not None else None
        agents.append(Agent(pos=W(c), heading=heading, goal_region=goal))

    return Scene(walls=walls, obstacles=obstacles, agents=agents,
                 start_region=groups[0][0], goal_region=groups[0][1], layout=scene.layout,
                 groups=groups if scene.groups else [], seed=scene.seed,
                 bottleneck=scene.bottleneck)


# -------------------------------------------------------------------- API

def sketch(scene, cfg=None, rng=None, fill_unreachable=True, style=None, parts=False):
    """(PIL RGB image, uint8 labels (n, n), drawn Scene[, parts dict]).

    `rng` defaults to ``default_rng(scene.seed)``. ``parts=True`` adds
    ``{"rings", "tails": uint8 (n, n) per-agent maps, "H": perspective}`` for tests.
    """
    cfg = cfg or Config()
    st = style or SketchStyle()
    if rng is None:
        rng = np.random.default_rng(scene.seed)
    d = perturb(scene, cfg, rng, st, fill_unreachable)
    out = draw(d, parts=parts)
    img, lab = out[0], out[1]
    img, lab, extra, H = photo(img, lab, rng, st, extra_maps=list(out[2]) if parts else [],
                               digital=d.digital)
    drawn = drawn_scene(d, H, scene, cfg)
    if parts:
        return Image.fromarray(img, "RGB"), lab, drawn, {"rings": extra[0], "tails": extra[1], "H": H}
    return Image.fromarray(img, "RGB"), lab, drawn
