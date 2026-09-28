"""PROTOTYPE of the single-black-pen sketch renderer (DESIGN_SKETCH.md §3).

Not project code: a starting point to port into sim/sketch.py, then delete.
Written 28 Sep 2026 to show the look; known problems the port must fix:

- letters S / G can land on top of agents (place them farthest from agents);
- the scribble fill of obstacles sometimes leaves a large white wedge;
- the JSON is not updated to the drawn geometry (DESIGN_SKETCH §3.1 step 4);
- it imports render._to_px, a private helper.

    python perception/prototypes/sketch_prototype.py     # writes sketch_prototype.png next to this file

Convention: wall = pen line (dead space hatched or blank); obstacle = filled or
scribbled blob; agent = small hollow circle + tail; start / goal = box with S / G.
"""
import math, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
from config import Config, C_BG, C_WALL, C_OBST, C_START, C_GOAL, C_AGENT
from layouts import random_scene
from render import _to_px, render, colorize
from reachability import unreachable_free

N = 512

def lowfreq(n, rng, knots, amp):
    ctrl = rng.normal(0, 1, knots + 2)
    return np.interp(np.linspace(0, knots + 1, n), np.arange(knots + 2), ctrl) * amp

def wobbly_line(p0, p1, rng, amp):
    L = math.dist(p0, p1); n = max(8, int(L / 3))
    t = np.linspace(-rng.uniform(0, .04), 1 + rng.uniform(0, .05), n)   # slight overshoot
    ux, uy = (p1[0]-p0[0]) / (L+1e-9), (p1[1]-p0[1]) / (L+1e-9)
    off = lowfreq(n, rng, max(2, int(L/60)), amp) + rng.normal(0, amp*.15, n)
    x = p0[0] + (p1[0]-p0[0])*t - uy*off; y = p0[1] + (p1[1]-p0[1])*t + ux*off
    return np.stack([x, y], 1)

def wobbly_circle(c, r, rng, amp_rel, gap=True):
    extra = rng.uniform(-0.35, 0.5) if gap else 0.15          # gap or overshoot, radians
    n = max(24, int(r * 1.2)); th0 = rng.uniform(0, 2*math.pi)
    th = th0 + np.linspace(0, 2*math.pi + extra, n)
    rr = r * (1 + lowfreq(n, rng, 4, amp_rel)) * np.linspace(1, 1 + rng.uniform(-.1, .1), n)
    sx, sy = 1 + rng.uniform(-.12, .12), 1 + rng.uniform(-.12, .12)   # not quite round
    return np.stack([c[0] + rr*np.cos(th)*sx, c[1] + rr*np.sin(th)*sy], 1)

def stroke(img, lab, pts, color, cls, w, rng, lab_extra=0):
    ws = np.clip(w * (1 + lowfreq(len(pts), rng, 3, .35)), 1, None)
    for (a, b), wi in zip(zip(pts[:-1], pts[1:]), ws):
        A, B = tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int))
        cv2.line(img, A, B, color, int(round(wi)), cv2.LINE_AA)
        if lab is not None:
            cv2.line(lab, A, B, cls, int(round(wi)) + lab_extra)


def letter(img, lab, ch, centre, h, pen, rng):
    """Hand-ish letter: Hershey script font, jittered size, slant and position."""
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    scale = h / 22 * rng.uniform(.8, 1.2); th = max(2, int(h / 12))
    (w, hh), _ = cv2.getTextSize(ch, font, scale, th)
    tile = np.zeros((hh * 3, w * 3), np.uint8)
    cv2.putText(tile, ch, (w, hh * 2), font, scale, 255, th, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((tile.shape[1] / 2, tile.shape[0] / 2), rng.uniform(-15, 15), 1)
    tile = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]))
    x0 = int(centre[0] - tile.shape[1] / 2 + rng.normal(0, 3)); y0 = int(centre[1] - tile.shape[0] / 2 + rng.normal(0, 3))
    ys, xs = np.nonzero(tile > 80)
    ys, xs = ys + y0, xs + x0
    ok = (ys >= 0) & (ys < N) & (xs >= 0) & (xs < N)
    img[ys[ok], xs[ok]] = pen

def sketch_mono(scene, cfg, rng):
    paper = np.array([246, 244, 238]) + rng.uniform(-12, 8, 3)
    img = np.empty((N, N, 3), np.uint8); img[:] = paper.astype(np.uint8)
    lab = np.zeros((N, N), np.uint8)
    P = lambda x, y: _to_px(cfg, x, y)
    pen = tuple(int(v) for v in rng.uniform(15, 55, 3))      # one black-ish marker
    mk = rng.uniform(3.0, 5.0)                                # marker width at 512 px

    dead = unreachable_free(cfg, scene)
    if dead.any() and rng.random() < 0.6:                     # hatched dead space
        big = np.flipud(cv2.resize(dead.astype(np.uint8), (N, N), interpolation=cv2.INTER_NEAREST)).astype(bool)
        lab[big] = C_WALL
        h = np.zeros((N, N), np.uint8); step = rng.integers(9, 15)
        for k in range(-N, 2 * N, step):
            stroke(h, None, wobbly_line((k, 0), (k + N, N), rng, 1.5), 255, 0, mk * .6, rng)
        img[(h > 0) & big] = pen

    for (start, goal) in (scene.groups or [(scene.start_region, scene.goal_region)]):
        for reg, ch, cls in ((start, "S", C_START), (goal, "G", C_GOAL)):
            x0, y0 = P(reg.x0, reg.y1); x1, y1 = P(reg.x1, reg.y0)
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) + rng.normal(0, 3, (4, 2))
            m = np.zeros((N, N), np.uint8); cv2.fillPoly(m, [quad.astype(np.int32)], 1)
            lab[m.astype(bool) & (lab == C_BG)] = cls
            for i in range(4):
                stroke(img, None, wobbly_line(quad[i], quad[(i + 1) % 4], rng, 1.3), pen, 0, mk * .6, rng)
            cx, cy = quad.mean(0); size = min(x1 - x0, y1 - y0)
            # put the letter away from agents: top or bottom third of the box
            cy = cy + rng.choice([-1, 1]) * (y1 - y0) * .33 if (y1 - y0) > 2 * size else cy
            letter(img, lab, ch, (cx, cy), min(size * .55, 40), pen, rng)

    for (a, b) in scene.walls:
        pts = wobbly_line(P(*a), P(*b), rng, 2.0)
        w = mk * rng.uniform(1.0, 2.2)                        # single or doubled stroke
        stroke(img, lab, pts, pen, C_WALL, w, rng, lab_extra=2)

    for o in scene.obstacles:
        c = P(*o.c); r = o.r * cfg.scale
        poly = wobbly_circle(c, r, rng, .06, gap=False)
        cv2.fillPoly(lab, [poly.astype(np.int32)], C_OBST)
        if rng.random() < .45:
            cv2.fillPoly(img, [poly.astype(np.int32)], pen, cv2.LINE_AA)
        else:
            m = np.zeros((N, N), np.uint8); cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            scr = np.zeros((N, N), np.uint8); ang = rng.uniform(-.6, .6)
            for k in np.arange(-r, r, rng.uniform(3.5, 6)):
                p0 = (c[0] - r * 1.1, c[1] + k); p1 = (c[0] + r * 1.1, c[1] + k + 2 * r * math.tan(ang))
                stroke(scr, None, wobbly_line(p0, p1, rng, 1.2), 255, 0, mk * .8, rng)
            img[(scr > 0) & (m > 0)] = pen
        stroke(img, None, poly, pen, 0, mk * .7, rng)

    for ag in scene.agents:
        c = np.array(P(*ag.pos)); r = cfg.agent_radius * cfg.scale * rng.uniform(.8, 1.25)
        circ = wobbly_circle(c, r, rng, .08)
        L = 2.2 * cfg.agent_radius * cfg.scale * rng.uniform(.8, 1.2)
        th = ag.heading + rng.normal(0, .06)
        tip = c + L * np.array([math.cos(th), -math.sin(th)])
        stroke(img, lab, circ, pen, C_AGENT, mk * .6, rng, lab_extra=3)
        stroke(img, lab, wobbly_line(tuple(c), tuple(tip), rng, .8), pen, C_AGENT, mk * .6, rng, lab_extra=3)

    # phone photo after rectification: residual tilt, uneven light, blur, noise, jpeg
    yy, xx = np.mgrid[0:N, 0:N] / N
    g = 1 - rng.uniform(.05, .3) * ((xx - rng.uniform(0, 1)) ** 2 + (yy - rng.uniform(0, 1)) ** 2)
    img = np.clip(img * g[..., None], 0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [N, 0], [N, N], [0, N]]); dst = src + rng.uniform(-8, 8, (4, 2)).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, H, (N, N), borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.warpPerspective(lab, H, (N, N), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(.4, 1.0))
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 85))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1], lab

if __name__ == "__main__":
    cfg = Config(); tiles = []
    for s, L in ((3, "doorway"), (11, "crossing"), (5, "cul_de_sac")):
        sc = random_scene(np.random.default_rng(s), cfg, layout=L, n_agents=5, n_obstacles=3)
        clean, _ = render(sc, cfg)
        img, lab = sketch_mono(sc, cfg, np.random.default_rng(s + 200))
        tiles.append([np.asarray(clean), img, np.asarray(colorize(lab))])
    S = 400; pad = 8
    rows = [np.concatenate(sum([[cv2.resize(t, (S, S), interpolation=cv2.INTER_AREA), np.full((S, pad, 3), 252, np.uint8)] for t in r], [])[:-1], 1) for r in tiles]
    grid = np.concatenate(sum([[r, np.full((pad, r.shape[1], 3), 252, np.uint8)] for r in rows], [])[:-1], 0)
    out = Path(__file__).with_name("sketch_prototype.png")
    cv2.imwrite(str(out), grid[..., ::-1]); print(out)
