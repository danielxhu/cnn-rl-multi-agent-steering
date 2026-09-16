"""Evaluation CLI: checkpoint × test directories -> metrics.json, per_scene.csv, figures.

    python evaluate.py --ckpt runs/dev_s256/ckpt_best.pt \\
                       --data data/test data/test_aug3 data/test_spacing data/test_dense
    python evaluate.py --ckpt ... --data data/val --tune-extract      # grid over thresholds

A directory that holds no scenes itself but sub-directories that do (for
example ``data/test_spacing/gap*``) expands to one test set per sub-directory,
named ``test_spacing/gap1.0``.

Outputs, next to the checkpoint unless ``--out`` is given (DESIGN.md §3.4, §5):
    metrics.json     one block per test set: overall, per layout, spacing curve
    per_scene.csv    one row per (test set, scene) with every scene-level metric
    agent_rows.csv   one row per ground-truth agent (the raw material of the spacing curve)
    fig_spacing_<set>.png, fig_examples_<set>.png

Every set is evaluated at the checkpoint's own resolution.
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
from data import SceneDataset, scene_files
from extract import extract
from metrics import AGENT_KEYS, SCENE_KEYS, aggregate, scene_metrics
from predict import Predictor, predict_dataset
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
    return out


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def evaluate_set(pr: Predictor, name: str, path: Path, batch: int, limit=None, ecfg=None):
    """Predict + extract + score one directory -> (dataset, predictions, scene rows, agent rows)."""
    ds = SceneDataset(path, pr.size, limit=limit)
    idx = list(range(len(ds)))
    preds = predict_dataset(pr.model, ds, idx, pr.device, batch)
    rows, arows = score_predictions(ds, preds, ecfg or pr.ecfg, name)
    return ds, preds, rows, arows


def score_predictions(ds: SceneDataset, preds: list, ecfg, name: str):
    rows, arows = [], []
    for i, p in enumerate(preds):
        parsed = extract(p["labels"], ds.worlds[i], p.get("heat"), p.get("dir"), ecfg)
        m, ar = scene_metrics(ds.scenes[i], ds.labels512(i), parsed, p["labels"], ds.worlds[i],
                              ds.size, sid=ds.ids[i])
        rows.append({"set": name, "scene": ds.ids[i], "layout": ds.layouts[i], **m})
        for r in ar:
            r["set"] = name
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


def examples_figure(ds: SceneDataset, preds: list, ecfg, path: Path, n=4):
    idx = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    rows = []
    for i in idx:
        p = preds[i]
        parsed = extract(p["labels"], ds.worlds[i], p.get("heat"), p.get("dir"), ecfg)
        rows.append(preview_row(ds.image_uint8(i), ds.labels_uint8(i), p["labels"], parsed, ds.scenes[i]))
    grid(rows).save(path)


# --------------------------------------------------------------- evaluate

def evaluate(ckpt, data_dirs, out_dir=None, figures=True, limit=None, device=None, batch=None) -> dict:
    pr = Predictor(ckpt, device)
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
        ds, preds, rows, arows = evaluate_set(pr, name, path, batch, limit)
        agg = aggregate(rows, arows)
        agg["seconds"] = time.time() - t0
        agg["path"] = str(path)
        metrics[name] = agg
        metrics["_meta"]["sets"].append(name)
        all_rows.extend(rows)
        all_arows.extend(arows)
        if figures:
            spacing_figure(agg, f"{name} @ {pr.size}", out_dir / f"fig_spacing_{_safe(name)}.png")
            examples_figure(ds, preds, pr.ecfg, out_dir / f"fig_examples_{_safe(name)}.png")
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


def tune_extract(ckpt, data_dir, out_dir=None, ring_grid=None, heat_grid=None, limit=None,
                 device=None, batch=None) -> dict:
    """Grid over ring_thresh / heat_thresh on one set; predictions are computed once."""
    pr = Predictor(ckpt, device)
    batch = batch or default_batch(pr.size)
    out_dir = Path(out_dir or Path(ckpt).parent)
    name, path = expand_data_dirs([data_dir])[0]
    ds = SceneDataset(path, pr.size, limit=limit)
    preds = predict_dataset(pr.model, ds, range(len(ds)), pr.device, batch)
    ring_grid = ring_grid or [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    heat_grid = heat_grid or ([0.2, 0.3, 0.4, 0.5, 0.6] if "heat" in preds[0] else [pr.ecfg.heat_thresh])
    results = []
    for rt in ring_grid:
        for ht in heat_grid:
            ecfg = replace(pr.ecfg, ring_thresh=rt, heat_thresh=ht)
            rows, arows = score_predictions(ds, preds, ecfg, name)
            o = aggregate(rows, arows)["overall"]
            results.append({"ring_thresh": rt, "heat_thresh": ht, "agent_f1": o["agent_f1"],
                            "agent_precision": o["agent_precision"], "agent_recall": o["agent_recall"],
                            "agent_center_err": o["agent_center_err"], "scene_usable": o["scene_usable"]})
            print(f"ring {rt:.2f} heat {ht:.2f}: f1 {o['agent_f1']:.4f} P {o['agent_precision']:.3f} "
                  f"R {o['agent_recall']:.3f} usable {o['scene_usable']:.3f}")
    best = max(results, key=lambda r: (r["agent_f1"] if not math.isnan(r["agent_f1"]) else -1))
    rec = {"set": name, "n_scenes": len(ds), "results": results, "best": best}
    with open(out_dir / "tune_extract.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"best: ring_thresh {best['ring_thresh']} heat_thresh {best['heat_thresh']} "
          f"(agent F1 {best['agent_f1']:.4f}) -> {out_dir / 'tune_extract.json'}")
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
    a = p.parse_args(argv)
    if a.tune_extract:
        tune_extract(a.ckpt, a.data[0], a.out, a.ring_grid, a.heat_grid, a.limit, a.device, a.batch)
    else:
        evaluate(a.ckpt, a.data, a.out, figures=not a.no_figures, limit=a.limit,
                 device=a.device, batch=a.batch)


if __name__ == "__main__":
    main()
