"""Collect the E4 folds into tables and figures (DESIGN_E4.md §6).

    python aggregate_e4.py --evals E:\\policy-eval --runs E:\\policy-runs --out results/e4

Owns ``table_e4.md``, ``table_e4_mixed.md``, ``summary_bottleneck.csv`` and
``fig_per_layout.png``, ``fig_bottleneck.png``; reuses ``aggregate`` for
``summary.csv``, ``summary_per_layout.csv``, ``fig_training_curves.png``.

Invariants:
- Folds are ``<evals>/e4/e4_holdout_<layout>_seed<k>``; the all-layouts baseline
  is ``<evals>/e3_both_seed<k>``. A fold whose ``config.json`` says it trained on
  its held-out layout stops the report.
- The bottleneck split comes from ``per_scene.csv``, filtered by layout:
  ``per_bottleneck`` in ``metrics.json`` pools all layouts of a suite, which
  would mix a fold's held-out scenes with the nine it trained on.
- The baseline is always restricted to the layouts that have a fold, so a
  partial set of folds compares like with like.
- Missing pieces (no baseline, no ``per_scene.csv``) print as "–" instead of
  failing, so a partial set of runs still aggregates.
"""
from __future__ import annotations

import _paths  # noqa: F401

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import aggregate
import plots
from evaluate import BOTTLENECK_BINS, ROW_METRICS, metrics_for
from policyconfig import LAYOUTS_E2

PREFIX = "e4_holdout_"
BASE = "e3_both"
HELD = "held-out"
METRICS = (("success_rate", "success"), ("collision_rate", "collision"), ("timeout_rate", "timeout"))
BINS = [f"[{lo:g},{'inf' if hi == math.inf else format(hi, 'g')})" for lo, hi in BOTTLENECK_BINS]
NUMERIC = ["n_agents", "bottleneck"] + ROW_METRICS
FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load_folds(evals: Path) -> dict:
    """{held-out layout: [evaluation of each seed]}, in the order of LAYOUTS_E2."""
    found = {}
    for e in aggregate.load_evals(evals / "e4", "variant", "^" + PREFIX):
        held = e["variant"][len(PREFIX):]
        if held not in LAYOUTS_E2:
            raise SystemExit(f"{e['dir']}: unknown held-out layout {held!r}")
        found.setdefault(held, []).append(e)
    return {l: found[l] for l in LAYOUTS_E2 if l in found}


def check_folds(folds: dict, runs_root):
    for held, evals in folds.items():
        for e in evals:
            d = aggregate.run_dir_of(e, runs_root)
            if d is None or not (d / "config.json").exists():
                continue
            trained = json.loads((d / "config.json").read_text(encoding="utf-8"))["env"]["layouts"]
            if held in trained:
                raise SystemExit(f"{d}: fold {held} trained on {held}: {trained}")


def values(evals, suite, layout, metric) -> list:
    return [e["suites"][suite]["per_layout"][layout][metric] for e in evals
            if suite in e["suites"] and layout in e["suites"][suite]["per_layout"]]


def gap(base, held) -> str:
    return f"{np.mean(base) - np.mean(held):+.3f}" if base and held else "–"


def mean_cell(vals) -> str:
    return f"{np.mean(vals):.3f}" if vals else "–"


def transfer_table(folds, base, suite) -> str:
    head = ["held-out layout", "seeds"]
    for _, name in METRICS:
        head += [f"{name}: all layouts", f"{name}: held out"]
        if name == "success":
            head.append("success gap")
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---|" * (len(head) - 1)]
    means = {m: ([], []) for m, _ in METRICS}
    for layout, evals in folds.items():
        row = [layout, str(len(values(evals, suite, layout, "success_rate")))]
        for m, name in METRICS:
            b, h = values(base, suite, layout, m), values(evals, suite, layout, m)
            row += [aggregate.fmt(b), aggregate.fmt(h)]
            if name == "success":
                row.append(gap(b, h))
            if b:
                means[m][0].append(float(np.mean(b)))
            if h:
                means[m][1].append(float(np.mean(h)))
        lines.append("| " + " | ".join(row) + " |")
    row = ["**mean over folds**", ""]
    for m, name in METRICS:
        b, h = means[m]
        row += [mean_cell(b), mean_cell(h)]
        if name == "success":
            row.append(gap(b, h))
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def trained_mean(e, suite, layouts) -> float | None:
    """Mean success of one evaluation over ``layouts``; None if it has not got them all."""
    m = e["suites"].get(suite)
    if m is None or not all(l in m["per_layout"] for l in layouts):
        return None
    return float(np.mean([m["per_layout"][l]["success_rate"] for l in layouts]))


def control_table(folds, base, suite) -> str:
    """Did leaving a layout out cost anything on the nine layouts the fold did train on?"""
    lines = ["| held-out layout | all layouts, on those 9 | fold, on its 9 trained layouts | gap |",
             "|---|---|---|---|"]
    for layout, evals in folds.items():
        trained = [l for l in LAYOUTS_E2 if l != layout]
        b = [v for v in (trained_mean(e, suite, trained) for e in base) if v is not None]
        h = [v for v in (trained_mean(e, suite, trained) for e in evals) if v is not None]
        lines.append(f"| {layout} | {aggregate.fmt(b)} | {aggregate.fmt(h)} | {gap(b, h)} |")
    return "\n".join(lines)


# ------------------------------------------------------------ bottleneck split

def _number(text):
    return float(text) if text not in ("", None) else None


def scene_rows(e, suite) -> list:
    """Rows of one suite from ``per_scene.csv``; [] when the file is missing."""
    p = e["dir"] / "per_scene.csv"
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return [{**r, **{k: _number(r[k]) for k in NUMERIC}} for r in csv.DictReader(fh)
                if r["suite"] == suite]


def per_bin(runs: list) -> dict:
    """``runs``: rows of each run. {bin: {"scenes": n, metric: [value per run]}}, empty bins absent."""
    out = {}
    for rows in runs:
        if not rows:
            continue
        for b, v in metrics_for(rows)["per_bottleneck"].items():
            if v["scenes"]:
                cell = out.setdefault(b, {"scenes": v["scenes"], **{m: [] for m, _ in METRICS}})
                for m, _ in METRICS:
                    cell[m].append(v[m])
    return out


def bottleneck_cells(folds, base, suite) -> dict:
    """{layout or "ALL": {"base": per_bin, "held": per_bin}}; ALL pools the held-out scenes per seed."""
    base_rows = [scene_rows(e, suite) for e in base]
    out, pooled = {}, {}
    for layout, evals in folds.items():
        held = []
        for e in evals:
            rows = [r for r in scene_rows(e, suite) if r["layout"] == layout]
            held.append(rows)
            pooled.setdefault(e["seed"], []).extend(rows)
        out[layout] = {"held": per_bin(held),
                       "base": per_bin([[r for r in rows if r["layout"] == layout]
                                        for rows in base_rows])}
    out["ALL"] = {"held": per_bin(list(pooled.values())),
                  "base": per_bin([[r for r in rows if r["layout"] in folds] for rows in base_rows])}
    return out


def bottleneck_table(cells, metric) -> str:
    lines = ["| held-out layout | " + " | ".join(BINS) + " |", "|---|" + "---|" * len(BINS)]
    for layout, c in cells.items():
        row = []
        for b in BINS:
            h, a = c["held"].get(b), c["base"].get(b)
            if h is None and a is None:
                row.append("–")
                continue
            left = f"{np.mean(a[metric]):.2f}" if a else "–"
            right = f"{np.mean(h[metric]):.2f}" if h else "–"
            row.append(f"{left} → {right} ({(h or a)['scenes']})")
        name = "**all folds**" if layout == "ALL" else layout
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_bottleneck_csv(all_cells: dict, path: Path):
    cols = ["suite", "layout", "bin", "series", "runs", "scenes", "success", "collision", "timeout"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for suite, cells in all_cells.items():
            for layout, c in cells.items():
                for series, key in ((BASE, "base"), (HELD, "held")):
                    for b, v in c[key].items():
                        w.writerow({"suite": suite, "layout": layout, "bin": b, "series": series,
                                    "runs": len(v["success_rate"]), "scenes": v["scenes"],
                                    **{name: float(np.mean(v[m])) for m, name in METRICS}})


# --------------------------------------------------------------------- report

def write_table(folds, base, suite, cells, path: Path):
    seeds = sorted({e["seed"] for evals in folds.values() for e in evals})
    parts = [f"# E4: {suite}", "",
             f"Leave-one-layout-out, {len(folds)} of {len(LAYOUTS_E2)} folds, fold seeds {seeds}; "
             f"\"all layouts\" is the E3 winner `{BASE}` ({len(base)} seeds). Cells are mean ± std "
             "over seeds; one value means one seed. A positive gap means holding the layout out "
             "cost success. Read a gap against the std of the all-layouts column: with one fold "
             "seed there is no spread of its own.", "",
             "## Held-out layout", "", transfer_table(folds, base, suite), "",
             "## Control: the layouts each fold did train on (success)", "",
             control_table(folds, base, suite), "",
             "## Held-out scenes by bottleneck width", "",
             "Each cell: all layouts → held out (scenes in the bin, per run)."]
    for m, name in METRICS:
        parts += ["", f"### {name.capitalize()}", "", bottleneck_table(cells, m)]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")     # "±": Windows defaults to GBK


def figures(folds, base, cells, out: Path):
    layouts = list(folds)
    v = {BASE: {l: values(base, "test4", l, "success_rate") for l in layouts},
         HELD: {l: values(e, "test4", l, "success_rate") for l, e in folds.items()}}
    v = {s: {l: x for l, x in d.items() if x} for s, d in v.items()}
    plots.per_layout_bars(v, layouts, out / "fig_per_layout.png",
                          title="test4: success on each layout, trained on vs held out",
                          order=[BASE, HELD])
    c = cells["ALL"]
    v = {BASE: {b: x["success_rate"] for b, x in c["base"].items()},
         HELD: {b: x["success_rate"] for b, x in c["held"].items()}}
    plots.bottleneck_curves(v, BINS, out / "fig_bottleneck.png",
                            title="test4: held-out scenes, success vs bottleneck width",
                            order=[BASE, HELD])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=None)
    a = ap.parse_args(argv)
    folds = load_folds(a.evals)
    if not folds:
        raise SystemExit(f"no E4 evaluations under {a.evals / 'e4'}")
    check_folds(folds, a.runs)
    base = aggregate.load_evals(a.evals, "variant", rf"^{BASE}_seed\d+$")
    if not base:
        print(f"note: no {BASE}_seed<k> under {a.evals}: the all-layouts columns stay empty")
    evals = [e for fold in folds.values() for e in fold]
    a.out.mkdir(parents=True, exist_ok=True)
    aggregate.write_summaries(evals, a.out, with_variant=True)
    all_cells = {}
    for suite, fname in (("test4", "table_e4.md"), ("test_mixed", "table_e4_mixed.md")):
        if any(suite in e["suites"] for e in evals):
            all_cells[suite] = bottleneck_cells(folds, base, suite)
            write_table(folds, base, suite, all_cells[suite], a.out / fname)
    write_bottleneck_csv(all_cells, a.out / "summary_bottleneck.csv")
    if "test4" in all_cells:
        figures(folds, base, all_cells["test4"], a.out)
    variants = [PREFIX + l for l in folds]
    plots.COLORS.update(zip(variants, FOLD_COLORS))        # ten folds: _SPARE would repeat colours
    aggregate.training_curves(evals, a.runs, a.out, variants)
    print(f"wrote {sorted(p.name for p in a.out.iterdir())} to {a.out}")


if __name__ == "__main__":
    main()
