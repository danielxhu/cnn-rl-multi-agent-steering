"""Perception module: small U-Net pixel classifier + geometry extraction.

Trains on simulator-rendered (image, label-map) pairs — no manual labels —
then converts a predicted label map into a structured scene description.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from sim import (AGENT_R, C_AGENT, C_FOV, C_GOAL, C_OBST, C_WALL, CLASS_NAMES,
                 WORLD, Scene, random_scene, render_small)

SIZE = 128
N_CLASSES = 6
DEVICE = "cpu"


class UNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )
        self.e1 = block(3, ch)
        self.e2 = block(ch, ch * 2)
        self.e3 = block(ch * 2, ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.d2 = block(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.d1 = block(ch * 2, ch)
        self.out = nn.Conv2d(ch, N_CLASSES, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def make_dataset(rng, n_scenes):
    """Random scenes spanning open (1-3 obstacles) and hallway layouts."""
    imgs, labs, scenes = [], [], []
    for i in range(n_scenes):
        hallway = rng.random() < 0.35
        n_obs = int(rng.integers(1, 4))
        sc = random_scene(rng, n_obs, hallway=hallway)
        img, lab = render_small(sc, SIZE)
        imgs.append(img)
        labs.append(lab)
        scenes.append(sc)
    return np.stack(imgs), np.stack(labs), scenes


def augment(img, rng):
    img = img.astype(np.float32)
    img *= rng.uniform(0.85, 1.15)                       # brightness
    img += rng.normal(0, 6.0, img.shape)                 # sensor noise
    return np.clip(img, 0, 255)


def train(seed=0, n_train=400, n_val=60, epochs=8, batch=8, lr=1e-3):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x_tr, y_tr, _ = make_dataset(rng, n_train)
    x_va, y_va, va_scenes = make_dataset(rng, n_val)

    net = UNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # class weights: rare classes (agent, goal) matter most for parsing
    w = torch.tensor([0.5, 1.0, 1.0, 2.0, 20.0, 40.0])
    loss_fn = nn.CrossEntropyLoss(weight=w)

    n = len(x_tr)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = np.stack([augment(x_tr[j], rng) for j in idx]) / 255.0
            xb = torch.from_numpy(xb.transpose(0, 3, 1, 2)).float()
            yb = torch.from_numpy(y_tr[idx]).long()
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep + 1}/{epochs}  loss {tot / n:.4f}", flush=True)

    # validation metrics
    net.eval()
    preds = predict(net, x_va)
    pix_acc = float((preds == y_va).mean())
    ious = {}
    for c in range(N_CLASSES):
        inter = int(((preds == c) & (y_va == c)).sum())
        union = int(((preds == c) | (y_va == c)).sum())
        ious[CLASS_NAMES[c]] = inter / union if union else float("nan")
    metrics = {"pixel_accuracy": pix_acc, "iou": ious}
    torch.save(net.state_dict(), "results/unet.pt")
    return net, metrics, (x_va, y_va, va_scenes, preds)


def predict(net, imgs):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            xb = torch.from_numpy(
                (imgs[i:i + 16] / 255.0).transpose(0, 3, 1, 2)).float()
            out.append(net(xb).argmax(dim=1).numpy())
    return np.concatenate(out).astype(np.uint8)


# -------------------------------------------------------- geometry extraction

def _px_to_world(px, py):
    s = WORLD / SIZE
    return px * s, WORLD - py * s


def extract_scene(pred: np.ndarray) -> Scene | None:
    """Predicted label map (SIZE x SIZE) -> structured Scene, or None."""
    s = WORLD / SIZE

    def centroid(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    # agent: outline circle + heading line share one class. Separate the line
    # tip (pixels far from the pixel centroid) from the circle outline, take
    # heading from the tip direction, then fit the center on outline pixels.
    agent_mask = (pred == C_AGENT).astype(np.uint8)
    if agent_mask.sum() < 4:
        return None
    ys, xs = np.nonzero(agent_mask)
    c0x, c0y = float(xs.mean()), float(ys.mean())
    dist = np.hypot(xs - c0x, ys - c0y)
    med = float(np.median(dist))
    tip = dist > 1.22 * med
    outline = ~tip

    pts = np.stack([xs[outline], ys[outline]], axis=1).astype(np.float32)
    (acx, acy), _ = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2))

    # heading: direction to the FOV-arc centroid (the arc is symmetric about
    # the heading). The heading line itself spans too few pixels at this
    # resolution to be a reliable estimator.
    fov_c = centroid(pred == C_FOV)
    if fov_c is None:
        return None
    heading = math.atan2(-(fov_c[1] - acy), fov_c[0] - acx)

    goal_c = centroid(pred == C_GOAL)
    if goal_c is None:
        return None

    obstacles = []
    n, cc = cv2.connectedComponents((pred == C_OBST).astype(np.uint8))
    for i in range(1, n):
        m = cc == i
        area = int(m.sum())
        if area < 12:  # speckle
            continue
        cx, cy = centroid(m)
        r = math.sqrt(area / math.pi) * s
        wx, wy = _px_to_world(cx, cy)
        obstacles.append((wx, wy, r))

    hallway = None
    wall = pred == C_WALL
    if wall.sum() > 0.02 * pred.size:
        rows = wall.mean(axis=1) > 0.5  # rows mostly wall
        open_rows = np.nonzero(~rows)[0]
        if len(open_rows) > 2:
            y_hi = WORLD - open_rows[0] * s
            y_lo = WORLD - open_rows[-1] * s
            hallway = (float(y_lo), float(y_hi))

    ax, ay = _px_to_world(acx, acy)
    gx, gy = _px_to_world(*goal_c)
    return Scene(agent=(ax, ay), heading=heading, obstacles=obstacles,
                 goal=(gx, gy), hallway=hallway)


def geometry_errors(gt: Scene, est: Scene) -> dict:
    d = {}
    d["agent_center_err"] = math.dist(gt.agent, est.agent)
    dh = (est.heading - gt.heading + math.pi) % (2 * math.pi) - math.pi
    d["heading_err_deg"] = abs(math.degrees(dh))
    d["goal_center_err"] = math.dist(gt.goal, est.goal)
    d["n_obstacles_gt"] = len(gt.obstacles)
    d["n_obstacles_est"] = len(est.obstacles)
    cerrs, rerrs = [], []
    used = set()
    for ox, oy, orr in gt.obstacles:
        best, bi = None, None
        for i, (ex, ey, er) in enumerate(est.obstacles):
            if i in used:
                continue
            dd = math.dist((ox, oy), (ex, ey))
            if best is None or dd < best:
                best, bi = dd, i
        if bi is not None:
            used.add(bi)
            cerrs.append(best)
            rerrs.append(abs(est.obstacles[bi][2] - orr))
    d["obstacle_center_err"] = float(np.mean(cerrs)) if cerrs else float("nan")
    d["obstacle_radius_err"] = float(np.mean(rerrs)) if rerrs else float("nan")
    return d


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    net, metrics, (x_va, y_va, va_scenes, preds) = train()

    errs = []
    parse_fail = 0
    for sc, pr in zip(va_scenes, preds):
        est = extract_scene(pr)
        if est is None:
            parse_fail += 1
            continue
        errs.append(geometry_errors(sc, est))
    if not errs:
        raise SystemExit(f"all {len(va_scenes)} val scenes failed to parse")
    agg = {}
    for k in errs[0]:
        vals = [e[k] for e in errs if not (isinstance(e[k], float) and math.isnan(e[k]))]
        agg[k] = float(np.mean(vals))
    count_ok = sum(1 for e in errs if e["n_obstacles_gt"] == e["n_obstacles_est"])
    metrics["geometry"] = agg
    metrics["obstacle_count_accuracy"] = count_ok / len(errs)
    metrics["parse_failures"] = parse_fail
    metrics["n_val"] = len(va_scenes)
    with open("results/perception_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
