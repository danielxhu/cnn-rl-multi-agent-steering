"""Per-scene metrics, greedy matching and aggregation (DESIGN.md §5).

numpy and cv2 only, no torch.  All geometric errors are in world units,
angles in degrees.  ``scene_metrics`` is a pure function of a ground-truth
``Scene`` (+ its 512 label map), a ``Parsed`` (or ``None``) and the predicted
label map at the working resolution.

Matching is greedy nearest-first with a fixed tolerance (``AGENT_TOL`` /
``OBST_TOL`` = 2.0 units, one agent radius).  Greedy is exact for agents
because true centres are ≥ 4 units apart.

``wall_surface_err`` is symmetric: parsed wall edges are rasterised on a 512
canvas and compared with the contour of the 512 ground-truth wall mask through
distance transforms, in both directions.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import _paths  # noqa: F401
from config import C_WALL, CLASS_NAMES, N_CLASSES
from targets import world_to_px

SPACING_BINS = [4.0, 5.0, 6.0, 8.0, 12.0, float("inf")]
AGENT_TOL = 2.0
OBST_TOL = 2.0
REGION_MATCH_IOU = 0.5

SCENE_KEYS = [
    "parse_fail", "pixacc", *[f"iou_{n}" for n in CLASS_NAMES], "miou",
    "agent_precision", "agent_recall", "agent_f1", "agent_count_ok",
    "agent_center_err", "heading_err_deg",
    "obst_precision", "obst_recall", "obst_count_ok", "obst_center_err", "obst_radius_err",
    "region_iou", "region_count_ok", "group_pairing_ok",
    "wall_iou", "wall_surface_err", "scene_usable",
    "n_agents", "n_pred_agents", "n_obstacles", "n_pred_obstacles",
]
AGENT_KEYS = ["scene", "layout", "agent", "d_nn", "matched", "center_err", "heading_err", "score"]

NAN = float("nan")


# ----------------------------------------------------------------- pixels

def pixel_metrics(pred, gt, n_classes=N_CLASSES) -> dict:
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()
    out = {"pixacc": float((pred == gt).mean()) if gt.size else NAN}
    ious = []
    for c in range(n_classes):
        p, g = pred == c, gt == c
        union = int((p | g).sum())
        iou = float((p & g).sum() / union) if union else NAN
        out[f"iou_{CLASS_NAMES[c]}"] = iou
        if union:
            ious.append(iou)
    out["miou"] = float(np.mean(ious)) if ious else NAN
    return out


# --------------------------------------------------------------- matching

def match_points(gt_pts, pred_pts, tol) -> list:
    """Greedy nearest-first one-to-one matching within `tol`: [(gi, pi)]."""
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return []
    G = np.asarray(gt_pts, float).reshape(-1, 2)
    P = np.asarray(pred_pts, float).reshape(-1, 2)
    d = np.sqrt(((G[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    pairs = []
    used_g, used_p = set(), set()
    for flat in np.argsort(d, axis=None, kind="stable"):
        gi, pi = divmod(int(flat), len(P))
        if d[gi, pi] > tol:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return pairs


def heading_err_deg(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def nearest_neighbour_distances(pts) -> np.ndarray:
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 2:
        return np.full(len(P), np.inf)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(1)


def region_iou(a, b) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _pr(n_matched, n_gt, n_pred):
    prec = n_matched / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    rec = n_matched / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _pairs_of(scene) -> list:
    if scene.groups:
        return list(scene.groups)
    if scene.start_region is not None and scene.goal_region is not None:
        return [(scene.start_region, scene.goal_region)]
    return []


# ------------------------------------------------------------------ walls

def rasterise_walls(walls, world_size: float, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    for (ax, ay), (bx, by) in walls:
        p0 = world_to_px(world_size, size, ax, ay)
        p1 = world_to_px(world_size, size, bx, by)
        cv2.line(m, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), 1, 1)
    return m


def surface_of_mask(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (outer and inner contours)."""
    out = np.zeros(mask.shape, np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if cs:
        cv2.drawContours(out, cs, -1, 1, 1)
    return out


def _mean_distance(from_mask: np.ndarray, to_mask: np.ndarray) -> float:
    if not from_mask.any() or not to_mask.any():
        return NAN
    dt = cv2.distanceTransform((to_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dt[from_mask > 0].mean())


def wall_surface_error(gt_wall_mask512: np.ndarray, parsed_walls, world_size: float) -> float:
    """Symmetric mean distance (world units) between parsed wall edges and the GT wall surface."""
    S = gt_wall_mask512.shape[0]
    gt_surf = surface_of_mask(gt_wall_mask512)
    pr_surf = rasterise_walls(parsed_walls, world_size, S)
    if not gt_surf.any() and not pr_surf.any():
        return 0.0
    fwd = _mean_distance(pr_surf, gt_surf)
    rev = _mean_distance(gt_surf, pr_surf)
    vals = [v for v in (fwd, rev) if not math.isnan(v)]
    if not vals:
        return NAN
    return float(np.mean(vals)) / (S / world_size)


# ------------------------------------------------------------------ scene

def scene_metrics(gt_scene, gt_labels512, parsed, pred_labels, world, size, sid=None):
    """(scene-level dict, per-GT-agent rows) for one scene."""
    W = float(world["size"])
    layout = getattr(gt_scene, "layout", "unknown")
    m = {k: NAN for k in SCENE_KEYS}
    m["parse_fail"] = float(parsed is None)
    m["n_agents"] = len(gt_scene.agents)
    m["n_obstacles"] = len(gt_scene.obstacles)

    # --- pixels --------------------------------------------------------------
    if pred_labels is not None and gt_labels512 is not None:
        gt_small = gt_labels512 if gt_labels512.shape[0] == size else cv2.resize(
            gt_labels512, (size, size), interpolation=cv2.INTER_NEAREST)
        m.update(pixel_metrics(pred_labels, gt_small))
        m["wall_iou"] = m[f"iou_{CLASS_NAMES[C_WALL]}"]

    # --- agents ------------------------------------------------------------
    gt_pos = [a.pos for a in gt_scene.agents]
    d_nn = nearest_neighbour_distances(gt_pos)
    rows = [{"scene": sid, "layout": layout, "agent": i, "d_nn": float(d_nn[i]), "matched": 0.0,
             "center_err": NAN, "heading_err": NAN, "score": NAN}
            for i in range(len(gt_pos))]

    if parsed is None:
        m["agent_precision"] = m["agent_recall"] = m["agent_f1"] = 0.0
        m["agent_count_ok"] = 0.0
        m["obst_precision"] = m["obst_recall"] = 0.0
        m["obst_count_ok"] = 0.0
        m["region_iou"] = 0.0
        m["region_count_ok"] = m["group_pairing_ok"] = 0.0
        m["scene_usable"] = 0.0
        m["n_pred_agents"] = m["n_pred_obstacles"] = 0
        return m, rows

    ps = parsed.scene
    scores = parsed.diagnostics.get("agent_scores", [])
    pred_pos = [a.pos for a in ps.agents]
    pairs = match_points(gt_pos, pred_pos, AGENT_TOL)
    cerrs, herrs = [], []
    for gi, pi in pairs:
        ce = math.dist(gt_pos[gi], pred_pos[pi])
        he = heading_err_deg(ps.agents[pi].heading, gt_scene.agents[gi].heading)
        cerrs.append(ce)
        herrs.append(he)
        rows[gi].update(matched=1.0, center_err=ce, heading_err=he,
                        score=float(scores[pi]) if pi < len(scores) else NAN)
    prec, rec, f1 = _pr(len(pairs), len(gt_pos), len(pred_pos))
    m.update(agent_precision=prec, agent_recall=rec, agent_f1=f1,
             agent_count_ok=float(len(pred_pos) == len(gt_pos)),
             agent_center_err=float(np.mean(cerrs)) if cerrs else NAN,
             heading_err_deg=float(np.mean(herrs)) if herrs else NAN,
             n_pred_agents=len(pred_pos))

    # --- obstacles -----------------------------------------------------------
    gt_o = [o.c for o in gt_scene.obstacles]
    pr_o = [o.c for o in ps.obstacles]
    opairs = match_points(gt_o, pr_o, OBST_TOL)
    oprec, orec, _ = _pr(len(opairs), len(gt_o), len(pr_o))
    m.update(obst_precision=oprec, obst_recall=orec,
             obst_count_ok=float(len(pr_o) == len(gt_o)),
             obst_center_err=float(np.mean([math.dist(gt_o[g], pr_o[p]) for g, p in opairs]))
             if opairs else NAN,
             obst_radius_err=float(np.mean([abs(gt_scene.obstacles[g].r - ps.obstacles[p].r)
                                            for g, p in opairs])) if opairs else NAN,
             n_pred_obstacles=len(pr_o))

    # --- regions and groups -------------------------------------------------
    gt_pairs, pr_pairs = _pairs_of(gt_scene), _pairs_of(ps)
    gt_starts, gt_goals = [p[0] for p in gt_pairs], [p[1] for p in gt_pairs]
    pr_starts, pr_goals = [p[0] for p in pr_pairs], [p[1] for p in pr_pairs]
    ious = []
    for gt_list, pr_list in ((gt_starts, pr_starts), (gt_goals, pr_goals)):
        for g in gt_list:
            ious.append(max([region_iou(g, p) for p in pr_list], default=0.0))
    m["region_iou"] = float(np.mean(ious)) if ious else NAN
    m["region_count_ok"] = float(len(pr_starts) == len(gt_starts) and len(pr_goals) == len(gt_goals))
    pairing_ok = len(pr_pairs) == len(gt_pairs) and all(
        any(region_iou(gs, s) > REGION_MATCH_IOU and region_iou(gg, g) > REGION_MATCH_IOU
            for s, g in pr_pairs)
        for gs, gg in gt_pairs)
    m["group_pairing_ok"] = float(pairing_ok)

    # --- walls -------------------------------------------------------------
    if gt_labels512 is not None:
        m["wall_surface_err"] = wall_surface_error(gt_labels512 == C_WALL, ps.walls, W)

    m["scene_usable"] = float(len(pairs) == len(gt_pos) == len(pred_pos)
                              and m["obst_count_ok"] == 1.0 and pairing_ok)
    return m, rows


# -------------------------------------------------------------- aggregate

def _nanmean(vals) -> float:
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else NAN


def _summarise(rows: list) -> dict:
    keys = [k for k in SCENE_KEYS if not k.startswith("n_")]
    out = {k: _nanmean([r.get(k, NAN) for r in rows]) for k in keys}
    out["n_scenes"] = len(rows)
    out["n_agents"] = int(sum(r.get("n_agents", 0) for r in rows))
    return out


def spacing_curve(agent_rows: list, bins=SPACING_BINS) -> list:
    """Recall and centre error per nearest-neighbour-distance bin."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in agent_rows if lo <= r["d_nn"] < hi]
        matched = [r for r in sel if r["matched"] > 0]
        out.append({
            "bin": f"[{lo:g}, {hi:g})", "lo": lo, "hi": hi, "n": len(sel),
            "agent_recall": (len(matched) / len(sel)) if sel else NAN,
            "agent_center_err": _nanmean([r["center_err"] for r in matched]),
            "heading_err_deg": _nanmean([r["heading_err"] for r in matched]),
        })
    return out


def aggregate(scene_rows: list, agent_rows: list) -> dict:
    """Overall means / rates, per-layout means, and the spacing curve."""
    layouts = sorted({r.get("layout", "unknown") for r in scene_rows})
    return {
        "overall": _summarise(scene_rows),
        "per_layout": {L: _summarise([r for r in scene_rows if r.get("layout") == L])
                       for L in layouts},
        "spacing": spacing_curve(agent_rows),
    }
