"""Aggregate E5 runs: runs/e5/*/metrics.json -> results/e5/ tables and figures.

    python aggregate.py --runs runs/e5 --out results/e5

Outputs (DESIGN.md §6.2):
    summary.csv              one row per run × test set with every aggregate metric
    table_res_aug.md         agent F1 / centre error / heading error / scene_usable,
                             rows = resolution, columns = training augmentation,
                             on `test` and `test_aug3`, mean ± std over seeds
    table_arch.md            seg-only vs instance head, per resolution
    fig_spacing_curve.png    agent recall and centre error vs d_nn bin, one line per
                             resolution (best augmentation), both architectures
    fig_res_aug.png          heat-map of agent F1 over resolution × augmentation
    fig_per_layout.png       scene_usable rate per layout at 256
    fig_training_curves.png  val agent F1 vs epoch, all runs, coloured by resolution

Run attributes come from each run's config.json (size, train set -> aug,
arch, instance head, seed), not from the directory name, so ad-hoc runs
aggregate too.  Off-axis cells of the star design are left empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

TABLE_METRICS = [("agent_f1", "F1", "{:.3f}"), ("agent_center_err", "cerr", "{:.2f}"),
                 ("heading_err_deg", "head°", "{:.1f}"), ("scene_usable", "usable", "{:.2f}")]
SIZE_COLORS = {128: "#4C72B0", 256: "#DD8452", 512: "#55A868"}


def _aug_of(train_name: str):
    m = re.search(r"aug(\d+)", train_name or "")
    return int(m.group(1)) if m else (train_name or "?")


def load_runs(runs_dir: Path) -> list:
    """One dict per run directory that has metrics.json (config.json and log.csv optional)."""
    runs = []
    for d in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        with open(mp) as fh:
            metrics = json.load(fh)
        meta = metrics.get("_meta", {})
        cfg = {}
        if (d / "config.json").exists():
            with open(d / "config.json") as fh:
                cfg = json.load(fh)
        log = []
        if (d / "log.csv").exists():
            with open(d / "log.csv") as fh:
                log = list(csv.DictReader(fh))
        runs.append({
            "name": d.name,
            "size": int(cfg.get("data", {}).get("size", meta.get("size", 0))),
            "aug": _aug_of(cfg.get("data", {}).get("train", "")),
            "arch": cfg.get("model", {}).get("arch", meta.get("arch", "?")),
            "inst": bool(cfg.get("model", {}).get("instance_head", meta.get("instance_head", False))),
            "seed": int(cfg.get("train", {}).get("seed", 0)),
            "sets": {k: v for k, v in metrics.items() if not k.startswith("_")},
            "log": log,
        })
    return runs


def primary_sets(runs) -> tuple:
    """(main test set, augmented test set): `test`/`test_aug3` when present, else what exists."""
    names = []
    for r in runs:
        names += [n for n in r["sets"] if n not in names]
    main = "test" if "test" in names else (names[0] if names else "test")
    aug = "test_aug3" if "test_aug3" in names else (names[1] if len(names) > 1 else None)
    return (main, aug) if aug else (main,)


def _aug_label(aug) -> str:
    return f"aug{aug}" if isinstance(aug, int) else str(aug)


def _arch_label(r) -> str:
    return f"{r['arch']}{'+inst' if r['inst'] else ''}"


def _mean_std(vals):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if v.size == 0:
        return math.nan, math.nan
    return float(v.mean()), float(v.std()) if v.size > 1 else 0.0


def _cell(runs, metric, fmt) -> str:
    m, s = _mean_std([r["sets"][r["_set"]]["overall"].get(metric) for r in runs])
    if math.isnan(m):
        return "–"
    return f"{fmt.format(m)} ± {fmt.format(s)}" if len(runs) > 1 else fmt.format(m)


def _with_set(runs, set_name):
    return [dict(r, _set=set_name) for r in runs if set_name in r["sets"]]


# ------------------------------------------------------------------ tables

def table_res_aug(runs, sets=None) -> str:
    sets = sets or primary_sets(runs)
    sizes = sorted({r["size"] for r in runs})
    augs = sorted({r["aug"] for r in runs}, key=lambda a: (isinstance(a, str), a))
    archs = sorted({_arch_label(r) for r in runs})
    lines = ["# Resolution × training augmentation", "",
             "Each cell: agent F1 · centre error (units) · heading error (deg) · scene_usable rate, "
             "mean ± std over seeds. Empty cells are off the star design.", ""]
    for set_name in sets:
        for arch in archs:
            sub = [r for r in _with_set(runs, set_name) if _arch_label(r) == arch]
            lines += [f"## `{set_name}` — {arch}", "",
                      "| resolution | " + " | ".join(_aug_label(a) for a in augs) + " |",
                      "|---|" + "---|" * len(augs)]
            for size in sizes:
                cells = []
                for aug in augs:
                    grp = [r for r in sub if r["size"] == size and r["aug"] == aug]
                    if not grp:
                        cells.append("")
                        continue
                    cells.append(" · ".join(_cell(grp, m, fmt) for m, _, fmt in TABLE_METRICS)
                                 + f" (n={len(grp)})")
                lines.append(f"| {size} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def table_arch(runs, set_name=None) -> str:
    set_name = set_name or primary_sets(runs)[0]
    sizes = sorted({r["size"] for r in runs})
    lines = ["# Architecture: seg-only vs instance head", "",
             f"On `{set_name}`, best augmentation per cell by agent F1; mean ± std over seeds.", "",
             "| resolution | variant | aug | " + " | ".join(l for _, l, _ in TABLE_METRICS) + " | n |",
             "|---|---|---|" + "---|" * (len(TABLE_METRICS) + 1)]
    for size in sizes:
        for arch in sorted({r["arch"] for r in runs}):
            for inst in (False, True):
                cand = [r for r in _with_set(runs, set_name)
                        if r["size"] == size and r["arch"] == arch and r["inst"] == inst]
                if not cand:
                    continue
                by_aug = {}
                for r in cand:
                    by_aug.setdefault(r["aug"], []).append(r)
                aug, grp = max(by_aug.items(), key=lambda kv: _mean_std(
                    [r["sets"][set_name]["overall"].get("agent_f1") for r in kv[1]])[0])
                lines.append(f"| {size} | {arch}{' + inst' if inst else ''} | {aug} | "
                             + " | ".join(_cell(grp, m, fmt) for m, _, fmt in TABLE_METRICS)
                             + f" | {len(grp)} |")
    return "\n".join(lines) + "\n"


def summary_rows(runs) -> list:
    rows = []
    for r in runs:
        for set_name, block in r["sets"].items():
            o = block.get("overall", {})
            rows.append({"run": r["name"], "size": r["size"], "aug": r["aug"], "arch": r["arch"],
                         "instance_head": int(r["inst"]), "seed": r["seed"], "set": set_name, **o})
    return rows


# ----------------------------------------------------------------- figures

def _best_aug(runs, size, inst, set_name=None):
    set_name = set_name or primary_sets(runs)[0]
    cand = [r for r in _with_set(runs, set_name) if r["size"] == size and r["inst"] == inst]
    if not cand:
        return None
    by_aug = {}
    for r in cand:
        by_aug.setdefault(r["aug"], []).append(r)
    return max(by_aug.items(), key=lambda kv: _mean_std(
        [r["sets"][set_name]["overall"].get("agent_f1") for r in kv[1]])[0])[0]


def fig_spacing_curve(runs, path: Path, set_prefix="test_spacing"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    drawn = False
    for size in sorted({r["size"] for r in runs}):
        for inst, ls in ((False, "-"), (True, "--")):
            aug = _best_aug(runs, size, inst)
            if aug is None:
                continue
            grp = [r for r in runs if r["size"] == size and r["inst"] == inst and r["aug"] == aug]
            # pool the spacing bins over every spacing set (or every set if none) and seeds
            bins = {}
            for r in grp:
                names = [s for s in r["sets"] if s.startswith(set_prefix)] or list(r["sets"])
                for s in names:
                    for b in r["sets"][s].get("spacing", []):
                        acc = bins.setdefault(b["bin"], {"n": 0, "hit": 0.0, "err": [], "lo": b["lo"]})
                        if b["n"]:
                            acc["n"] += b["n"]
                            acc["hit"] += b["agent_recall"] * b["n"]
                            if not math.isnan(b["agent_center_err"]):
                                acc["err"].append((b["agent_center_err"], b["n"]))
            if not bins:
                continue
            keys = sorted(bins, key=lambda k: bins[k]["lo"])
            x = np.arange(len(keys))
            rec = [bins[k]["hit"] / bins[k]["n"] if bins[k]["n"] else math.nan for k in keys]
            err = [sum(e * n for e, n in bins[k]["err"]) / max(1, sum(n for _, n in bins[k]["err"]))
                   if bins[k]["err"] else math.nan for k in keys]
            label = f"{size}px {_aug_label(aug)}{' +inst' if inst else ''}"
            axes[0].plot(x, rec, ls, marker="o", color=SIZE_COLORS.get(size), label=label)
            axes[1].plot(x, err, ls, marker="o", color=SIZE_COLORS.get(size), label=label)
            for ax in axes:
                ax.set_xticks(x)
                ax.set_xticklabels(keys, fontsize=8)
            drawn = True
    axes[0].set_ylabel("agent recall")
    axes[1].set_ylabel("centre error (units)")
    for ax in axes:
        ax.set_xlabel("GT nearest-neighbour distance (units)")
        ax.grid(alpha=0.3)
    if drawn:
        axes[0].legend(fontsize=8)
    fig.suptitle("Detection vs agent spacing (solid: seg-only, dashed: +instance head)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_res_aug(runs, path: Path, set_name=None):
    import matplotlib.pyplot as plt
    set_name = set_name or primary_sets(runs)[0]
    sizes = sorted({r["size"] for r in runs})
    augs = sorted({r["aug"] for r in runs}, key=lambda a: (isinstance(a, str), a))
    archs = sorted({_arch_label(r) for r in runs})
    fig, axes = plt.subplots(1, max(1, len(archs)), figsize=(4.2 * max(1, len(archs)), 3.6), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        M = np.full((len(sizes), len(augs)), np.nan)
        for i, size in enumerate(sizes):
            for j, aug in enumerate(augs):
                grp = [r for r in _with_set(runs, set_name)
                       if r["size"] == size and r["aug"] == aug and _arch_label(r) == arch]
                M[i, j] = _mean_std([r["sets"][set_name]["overall"].get("agent_f1") for r in grp])[0]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(sizes)):
            for j in range(len(augs)):
                if not math.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
        ax.set_xticks(range(len(augs)))
        ax.set_xticklabels([_aug_label(a) for a in augs])
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels([str(s) for s in sizes])
        ax.set_xlabel("training augmentation")
        ax.set_ylabel("resolution")
        ax.set_title(f"agent F1 on {set_name}: {arch}")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_per_layout(runs, path: Path, size=256, set_name=None):
    import matplotlib.pyplot as plt
    set_name = set_name or primary_sets(runs)[0]
    sub = [r for r in _with_set(runs, set_name) if r["size"] == size]
    if not sub:                                            # smoke runs: fall back to any size
        sub = _with_set(runs, set_name)
        size = sub[0]["size"] if sub else size
    layouts = sorted({L for r in sub for L in r["sets"][set_name].get("per_layout", {})})
    series = []
    for inst in (False, True):
        aug = _best_aug(sub, size, inst, set_name)
        grp = [r for r in sub if r["inst"] == inst and r["aug"] == aug]
        if grp:
            vals = [_mean_std([r["sets"][set_name]["per_layout"].get(L, {}).get("scene_usable") for r in grp])[0]
                    for L in layouts]
            series.append((f"{size}px {_aug_label(aug)}{' +inst' if inst else ''}", vals))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(layouts))
    w = 0.8 / max(1, len(series))
    for k, (label, vals) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("scene_usable rate")
    ax.set_title(f"Usable parses per layout ({set_name})")
    if series:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_training_curves(runs, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.8))
    seen = set()
    for r in runs:
        pts = [(int(row["epoch"]), float(row["val_agent_f1"])) for row in r["log"]
               if row.get("val_agent_f1") not in (None, "")]
        if not pts:
            continue
        label = f"{r['size']}px" if r["size"] not in seen else None
        seen.add(r["size"])
        ax.plot([e for e, _ in pts], [f for _, f in pts], marker=".", alpha=0.7,
                color=SIZE_COLORS.get(r["size"], "gray"), label=label,
                linestyle="--" if r["inst"] else "-")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val agent F1 (subset)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    if seen:
        ax.legend(fontsize=8)
    ax.set_title("Validation agent F1 (solid: seg-only, dashed: +instance head)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------- main

def aggregate_runs(runs_dir, out_dir) -> list:
    import matplotlib
    matplotlib.use("Agg")
    runs = load_runs(Path(runs_dir))
    if not runs:
        raise SystemExit(f"no run with metrics.json under {runs_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = summary_rows(runs)
    keys = ["run", "size", "aug", "arch", "instance_head", "seed", "set"]
    keys += sorted({k for r in rows for k in r if k not in keys})
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "table_res_aug.md").write_text(table_res_aug(runs), encoding="utf-8")
    (out / "table_arch.md").write_text(table_arch(runs), encoding="utf-8")
    fig_spacing_curve(runs, out / "fig_spacing_curve.png")
    fig_res_aug(runs, out / "fig_res_aug.png")
    fig_per_layout(runs, out / "fig_per_layout.png")
    fig_training_curves(runs, out / "fig_training_curves.png")
    print(f"aggregated {len(runs)} runs ({len(rows)} run × set rows) -> {out}")
    return runs


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate E5 runs into tables and figures.")
    p.add_argument("--runs", default="runs/e5")
    p.add_argument("--out", default="results/e5")
    a = p.parse_args(argv)
    aggregate_runs(a.runs, a.out)


if __name__ == "__main__":
    main()
