"""Evaluation CLI: checkpoint × test directories -> metrics.json, per_scene.csv, figures.

    python evaluate.py --ckpt runs/dev_s256/ckpt_best.pt \\
                       --data data/test data/test_aug3 data/test_spacing data/test_dense
    python evaluate.py --ckpt ... --data data/val --tune-extract      # grid over thresholds
    python evaluate.py --ckpt ... --data data/test_sketch5 --goal-mode exits --heat-thresh 0.3
                                                                       # override the run's extractor

A directory that holds no scenes itself but sub-directories that do (for
example ``data/test_spacing/gap*``) expands to one test set per sub-directory,
named ``test_spacing/gap1.0``. A set whose name is already taken is qualified
with its parent (``data/test`` and ``data/real_sketch/test`` -> ``test`` and
``real_sketch/test``).

Outputs, next to the checkpoint unless ``--out`` is given (DESIGN.md §3.4, §5):
    metrics.json     one block per test set: overall, per layout, spacing curve
    per_scene.csv    one row per (test set, scene) with every scene-level metric
    agent_rows.csv   one row per ground-truth agent (the raw material of the spacing curve)
    fig_spacing_<set>.png, fig_examples_<set>.png

Every set is evaluated at the checkpoint's own resolution, with the run's own
extractor settings unless ``--goal-mode`` / ``--heat-thresh`` /
``--heat-nms-factor`` / ``--region-by-agents`` / ``--letter-gate`` /
``--no-letter-gate`` / ``--letter-thresh`` / ``--min-box-side`` override them
(for instance a tuned threshold, or the v4 model scored as exit sets on sketch
v5 scenes). The letter gate (sketch v6) is switched off, with a printed note,
for any set whose ``dataset.json`` style is not a sketch style: rendered images
have no letters.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from config import CLASS_NAMES
from data import SceneDataset, scene_files
from extract import extract
from metrics import AGENT_KEYS, SCENE_KEYS, aggregate, scene_metrics
from predict import Predictor, predict_dataset, predict_iter
from runconfig import default_batch
from viz import grid, preview_row

DEFAULT_SETS = ["data/test", "data/test_aug3", "data/test_spacing", "data/test_dense"]


def expand_data_dirs(dirs) -> list:
    """[(set name, path)]: a directory of scenes, or one entry per scene sub-directory."""
    out = []
    for d in dirs:
        d = Path(d)
        if scene_files(d):
            out.append((d.name, d))
            continue
        subs = [s for s in sorted(d.iterdir()) if s.is_dir() and scene_files(s)] if d.is_dir() else []
        if not subs:
            raise FileNotFoundError(f"no scenes under {d}")
        out.extend((f"{d.name}/{s.name}", s) for s in subs)
    # two sets with one name (data/test and data/real_sketch/test) would overwrite each
    # other's block in metrics.json: qualify the later ones with their parent directory
    names, unique = set(), []
    for name, path in out:
        if name in names:
            name = f"{path.parent.name}/{name}"
        names.add(name)
        unique.append((name, path))
    return unique


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def example_indices(n: int, k: int) -> list:
    """`k` indices spread evenly over `n` scenes (the ones the example figure shows)."""
    return list(range(0, n, max(1, n // max(1, k))))[:k] if n and k else []


# metrics computed from the label map; NaN on sets whose labels are not traced from
# the image (ingested photos, DESIGN_SKETCH §5), so no table can average them in
LABEL_MAP_KEYS = ["pixacc", *[f"iou_{n}" for n in CLASS_NAMES], "miou", "wall_iou", "wall_surface_err"]


def dataset_style(path: Path):
    """The generator style recorded in <set>/dataset.json ("render", "sketch5", ...), or None."""
    p = Path(path) / "dataset.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("args", {}).get("style")
    except (OSError, ValueError):
        return None


def set_ecfg(ecfg, path: Path, name: str):
    """The extractor config for one set: no letter gate on rendered images (no letters drawn)."""
    style = dataset_style(path)
    if ecfg.letter_gate and style is not None and not str(style).startswith("sketch"):
        print(f"{name}: style {style!r} draws no letters -> letter gate off for this set")
        return replace(ecfg, letter_gate=False)
    return ecfg


def score_scene(ds: SceneDataset, i: int, pred: dict, ecfg, name: str):
    """Extract + score one prediction -> (scene row, agent rows)."""
    parsed = extract(pred["labels"], ds.worlds[i], pred.get("heat"), pred.get("dir"), ecfg,
                     letter=pred.get("letter"))
    m, ar = scene_metrics(ds.scenes[i], ds.labels512(i), parsed, pred["labels"], ds.worlds[i],
                          ds.size, sid=ds.ids[i], goal_mode=ecfg.goal_mode)
    if not ds.labels_exact:
        m.update({k: math.nan for k in LABEL_MAP_KEYS})
    for r in ar:
        r["set"] = name
    return {"set": name, "scene": ds.ids[i], "layout": ds.layouts[i], **m}, ar


def evaluate_set(pr: Predictor, name: str, path: Path, batch: int, limit=None, ecfg=None,
                 n_examples: int = 0):
    """Predict + extract + score one directory, **one scene at a time**.

    Returns (dataset, {index: prediction} for the example figure, scene rows,
    agent rows).  Each prediction is scored and dropped as it arrives: holding a
    whole 512 test set (1000 scenes) costs several GB, and the previous set is
    still referenced while the next one is built, which exhausted the 16 GB
    training machine (DESIGN.md §11.7).
    """
    ds = SceneDataset(path, pr.size, limit=limit)
    ecfg = set_ecfg(ecfg or pr.ecfg, path, name)
    keep = set(example_indices(len(ds), n_examples))
    kept, rows, arows = {}, [], []
    for i, p in predict_iter(pr.model, ds, range(len(ds)), pr.device, batch):
        m, ar = score_scene(ds, i, p, ecfg, name)
        rows.append(m)
        arows.extend(ar)
        if i in keep:
            kept[i] = p
    return ds, kept, rows, arows, ecfg


def score_predictions(ds: SceneDataset, preds: list, ecfg, name: str):
    """Score a list of predictions already in memory (threshold tuning)."""
    rows, arows = [], []
    for i, p in enumerate(preds):
        m, ar = score_scene(ds, i, p, ecfg, name)
        rows.append(m)
        arows.extend(ar)
    return rows, arows


# ---------------------------------------------------------------- figures

def spacing_figure(agg: dict, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = agg["spacing"]
    labels = [b["bin"] for b in bins]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(x, [b["agent_recall"] for b in bins], color="#4C72B0")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("agent recall")
    axes[1].bar(x, [b["agent_center_err"] for b in bins], color="#DD8452")
    axes[1].set_ylabel("centre error (world units)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("GT nearest-neighbour distance")
        for xi, b in zip(x, bins):
            ax.text(xi, ax.get_ylim()[1] * 0.02, f"n={b['n']}", ha="center", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def examples_figure(ds: SceneDataset, kept: dict, ecfg, path: Path):
    """input / GT labels / predicted labels / parsed overlay for the kept scenes."""
    rows = []
    for i in sorted(kept):
        p = kept[i]
        parsed = extract(p["labels"], ds.worlds[i], p.get("heat"), p.get("dir"), ecfg,
                         letter=p.get("letter"))
        rows.append(preview_row(ds.image_uint8(i), ds.labels_uint8(i), p["labels"], parsed, ds.scenes[i]))
    if rows:
        grid(rows).save(path)


# --------------------------------------------------------------- evaluate

def extract_overrides(ecfg, goal_mode=None, heat_thresh=None, heat_nms_factor=None,
                      region_by_agents=None, letter_gate=None, letter_thresh=None, min_box_side=None):
    """The run's ExtractConfig with the given (non-None) fields replaced."""
    over = {k: v for k, v in dict(goal_mode=goal_mode, heat_thresh=heat_thresh,
                                  heat_nms_factor=heat_nms_factor, region_by_agents=region_by_agents,
                                  letter_gate=letter_gate, letter_thresh=letter_thresh,
                                  min_box_side=min_box_side).items() if v is not None}
    return replace(ecfg, **over) if over else ecfg


def _check_gate(pr: Predictor):
    if pr.ecfg.letter_gate and not getattr(pr.cfg.model, "letter_head", False):
        raise SystemExit("--letter-gate needs a model with a letter head (sketch v6)")


def evaluate(ckpt, data_dirs, out_dir=None, figures=True, limit=None, device=None, batch=None,
             overrides=None) -> dict:
    pr = Predictor(ckpt, device)
    pr.ecfg = extract_overrides(pr.ecfg, **(overrides or {}))
    _check_gate(pr)
    batch = batch or default_batch(pr.size)
    out_dir = Path(out_dir or Path(ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"_meta": {"ckpt": str(ckpt), "size": pr.size, "arch": pr.cfg.model.arch,
                         "instance_head": pr.cfg.model.instance_head,
                         "epoch": pr.model._ckpt_meta.get("epoch"),
                         "extract": pr.ecfg.__dict__, "sets": []}}
    all_rows, all_arows = [], []
    for name, path in expand_data_dirs(data_dirs):
        t0 = time.time()
        ds, kept, rows, arows, set_cfg = evaluate_set(pr, name, path, batch, limit,
                                                      n_examples=4 if figures else 0)
        agg = aggregate(rows, arows)
        agg["letter_gate"] = set_cfg.letter_gate
        agg["seconds"] = time.time() - t0
        agg["path"] = str(path)
        if not ds.labels_exact:
            agg["note"] = ("labels rendered from the scene JSON, not traced from the photo: "
                           "pixel metrics and wall_surface_err are not meaningful (NaN)")
            print(f"{name}: {agg['note']}")
        metrics[name] = agg
        metrics["_meta"]["sets"].append(name)
        all_rows.extend(rows)
        all_arows.extend(arows)
        if figures:
            spacing_figure(agg, f"{name} @ {pr.size}", out_dir / f"fig_spacing_{_safe(name)}.png")
            examples_figure(ds, kept, set_cfg, out_dir / f"fig_examples_{_safe(name)}.png")
        o = agg["overall"]
        print(f"{name:<24s} n={o['n_scenes']:<5d} agent_f1 {o['agent_f1']:.3f}  "
              f"cerr {o['agent_center_err']:.3f}  head {o['heading_err_deg']:.1f}°  "
              f"obst_ok {o['obst_count_ok']:.3f}  usable {o['scene_usable']:.3f}  "
              f"fail {o['parse_fail']:.3f}  ({agg['seconds']:.0f}s)", flush=True)

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)
    with open(out_dir / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "scene", "layout", *SCENE_KEYS])
        w.writeheader()
        w.writerows(all_rows)
    with open(out_dir / "agent_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", *AGENT_KEYS])
        w.writeheader()
        w.writerows(all_arows)
    return metrics


HEAT_NMS_GRID = [0.5, 0.6, 0.8, 1.0, 1.2]


LETTER_GRID = [0.2, 0.3, 0.4, 0.5]


def tune_extract(ckpt, data_dir, out_dir=None, ring_grid=None, heat_grid=None, limit=None,
                 device=None, batch=None, nms_grid=None, overrides=None, letter_grid=None) -> dict:
    """Grid over ring_thresh / heat_thresh (/ heat_nms_factor) on one set; predictions
    are computed once.

    heat_nms_factor is gridded (``HEAT_NMS_GRID``) for runs that set it (sketch v5)
    or with ``nms_grid``; other runs keep their one value, so their grid is as
    before. Ring thresholds only matter for the heat fallback in such runs, so
    their ring grid is the run's own value unless ``ring_grid`` is given.

    A run with a letter head and the letter gate on (sketch v6) then grids
    letter_thresh (``LETTER_GRID``) at the best heat thresholds, choosing by
    scene_usable (then agent F1): a second, short stage instead of a 4x larger
    grid. The result is in ``best`` like the others."""
    pr = Predictor(ckpt, device)
    pr.ecfg = extract_overrides(pr.ecfg, **(overrides or {}))
    _check_gate(pr)
    batch = batch or default_batch(pr.size)
    out_dir = Path(out_dir or Path(ckpt).parent)
    name, path = expand_data_dirs([data_dir])[0]
    pr.ecfg = set_ecfg(pr.ecfg, path, name)
    ds = SceneDataset(path, pr.size, limit=limit)
    preds = predict_dataset(pr.model, ds, range(len(ds)), pr.device, batch)
    heat = "heat" in preds[0]
    sized = heat and (pr.ecfg.heat_nms_factor is not None or nms_grid is not None)
    ring_grid = ring_grid or ([pr.ecfg.ring_thresh] if sized
                             else [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])
    heat_grid = heat_grid or ([0.2, 0.3, 0.4, 0.5, 0.6] if heat else [pr.ecfg.heat_thresh])
    nms_grid = nms_grid or (HEAT_NMS_GRID if sized else [pr.ecfg.heat_nms_factor])
    results = []
    for rt in ring_grid:
        for ht in heat_grid:
            for nf in nms_grid:
                ecfg = replace(pr.ecfg, ring_thresh=rt, heat_thresh=ht, heat_nms_factor=nf)
                rows, arows = score_predictions(ds, preds, ecfg, name)
                o = aggregate(rows, arows)["overall"]
                r = {"ring_thresh": rt, "heat_thresh": ht, "agent_f1": o["agent_f1"],
                     "agent_precision": o["agent_precision"], "agent_recall": o["agent_recall"],
                     "agent_center_err": o["agent_center_err"], "scene_usable": o["scene_usable"]}
                nms_txt = ""
                if sized:
                    r["heat_nms_factor"] = nf
                    nms_txt = f" nms {nf:.2f}"
                results.append(r)
                print(f"ring {rt:.2f} heat {ht:.2f}{nms_txt}: f1 {o['agent_f1']:.4f} "
                      f"P {o['agent_precision']:.3f} R {o['agent_recall']:.3f} usable {o['scene_usable']:.3f}")
    best = max(results, key=lambda r: (r["agent_f1"] if not math.isnan(r["agent_f1"]) else -1))
    lettered = "letter" in preds[0] and pr.ecfg.letter_gate
    if lettered:                                       # stage 2: letter_thresh at the best heat
        stage = []
        for lt in letter_grid or LETTER_GRID:
            ecfg = replace(pr.ecfg, ring_thresh=best["ring_thresh"], heat_thresh=best["heat_thresh"],
                           heat_nms_factor=best.get("heat_nms_factor", pr.ecfg.heat_nms_factor),
                           letter_thresh=lt)
            rows, arows = score_predictions(ds, preds, ecfg, name)
            o = aggregate(rows, arows)["overall"]
            r = dict(best, letter_thresh=lt, agent_f1=o["agent_f1"], agent_precision=o["agent_precision"],
                     agent_recall=o["agent_recall"], agent_center_err=o["agent_center_err"],
                     scene_usable=o["scene_usable"], region_set_ok=o["region_set_ok"])
            stage.append(r)
            print(f"letter {lt:.2f}: usable {o['scene_usable']:.3f} region_set_ok {o['region_set_ok']:.3f} "
                  f"f1 {o['agent_f1']:.4f}")
        results += stage
        nan0 = lambda v: -1 if math.isnan(v) else v  # noqa: E731
        best = max(stage, key=lambda r: (nan0(r["scene_usable"]), nan0(r["agent_f1"])))
    rec = {"set": name, "n_scenes": len(ds), "goal_mode": pr.ecfg.goal_mode, "results": results,
           "best": best}
    with open(out_dir / "tune_extract.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"best: ring_thresh {best['ring_thresh']} heat_thresh {best['heat_thresh']}"
          + (f" heat_nms_factor {best['heat_nms_factor']}" if sized else "")
          + (f" letter_thresh {best['letter_thresh']}" if lettered else "")
          + f" (agent F1 {best['agent_f1']:.4f}) -> {out_dir / 'tune_extract.json'}")
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on test sets.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", default=DEFAULT_SETS, help="test directories")
    p.add_argument("--out", default=None, help="output directory (default: the checkpoint's)")
    p.add_argument("--limit", type=int, default=None, help="scenes per set (smoke runs)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--tune-extract", action="store_true",
                   help="grid over ring/heat thresholds on the first --data set instead")
    p.add_argument("--ring-grid", nargs="+", type=float, default=None)
    p.add_argument("--heat-grid", nargs="+", type=float, default=None)
    p.add_argument("--nms-grid", nargs="+", type=float, default=None,
                   help="heat_nms_factor values (default: %s for runs that set it, else the "
                        "run's own)" % HEAT_NMS_GRID)
    g = p.add_argument_group("extractor overrides (default: the run's own)")
    g.add_argument("--goal-mode", choices=["pairs", "exits"], default=None)
    g.add_argument("--heat-thresh", type=float, default=None)
    g.add_argument("--heat-nms-factor", type=float, default=None)
    g.add_argument("--region-by-agents", action="store_const", const=True, default=None)
    from runconfig import add_letter_flags
    add_letter_flags(g)
    p.add_argument("--letter-grid", nargs="+", type=float, default=None,
                   help="letter_thresh values for --tune-extract (default %s; letter head + gate)" % LETTER_GRID)
    a = p.parse_args(argv)
    over = dict(goal_mode=a.goal_mode, heat_thresh=a.heat_thresh, heat_nms_factor=a.heat_nms_factor,
                region_by_agents=a.region_by_agents, letter_gate=a.letter_gate,
                letter_thresh=a.letter_thresh, min_box_side=a.min_box_side)
    if a.tune_extract:
        tune_extract(a.ckpt, a.data[0], a.out, a.ring_grid, a.heat_grid, a.limit, a.device, a.batch,
                     nms_grid=a.nms_grid, overrides=over, letter_grid=a.letter_grid)
    else:
        evaluate(a.ckpt, a.data, a.out, figures=not a.no_figures, limit=a.limit,
                 device=a.device, batch=a.batch, overrides=over)


if __name__ == "__main__":
    main()
