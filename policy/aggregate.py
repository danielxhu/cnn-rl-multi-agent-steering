"""Collect evaluation directories into the E2 tables and figures (DESIGN.md §6).

    python aggregate.py --evals E:\\policy-eval --out results/e2 [--runs E:\\policy-runs]

Owns ``summary.csv``, ``summary_per_layout.csv``, ``table_e2.md``,
``table_e2_mixed.md`` and the four ``fig_*.png``.

Invariants:
- An evaluation directory is any sub-directory of ``--evals`` with a
  ``metrics.json``; its identity (``controller`` = greedy / mlp / gru, ``seed``)
  comes from that file, not from the directory name.
- Tables are mean ± std over seeds (population std); greedy has one run and is
  shown without a spread.
- Missing pieces (no gru runs yet, no trajectories) skip what needs them
  instead of failing, so a partial set of runs still aggregates.
"""
from __future__ import annotations

import _paths  # noqa: F401

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import plots
from evaluate import BOTTLENECK_BINS, ROW_METRICS
from policyconfig import LAYOUTS_E2
from scene import Scene

CTRLS = ["greedy", "mlp", "gru"]


def load_evals(root: Path) -> list:
    out = []
    for m in sorted(root.glob("*/metrics.json")):
        data = json.loads(m.read_text())
        ident = data["identity"]
        out.append({"dir": m.parent, "name": m.parent.name, "controller": ident["controller"],
                    "seed": ident.get("seed"), "ckpt": ident.get("ckpt"),
                    "suites": data["suites"]})
    return out


def write_summaries(evals: list, out: Path):
    cols = ["controller", "seed", "run", "suite", "scenes"] + ROW_METRICS
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in evals:
            for s, m in e["suites"].items():
                w.writerow({"controller": e["controller"], "seed": e["seed"], "run": e["name"],
                            "suite": s, **{k: m["overall"].get(k) for k in ["scenes"] + ROW_METRICS}})
    cols = ["controller", "seed", "run", "suite", "layout", "scenes"] + ROW_METRICS
    with open(out / "summary_per_layout.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in evals:
            for s, m in e["suites"].items():
                for l, v in m["per_layout"].items():
                    w.writerow({"controller": e["controller"], "seed": e["seed"], "run": e["name"],
                                "suite": s, "layout": l,
                                **{k: v.get(k) for k in ["scenes"] + ROW_METRICS}})


def collect(evals, suite, metric, getter):
    """{controller: {key: [value per run]}} for a per-layout / overall metric."""
    out = {}
    for e in evals:
        m = e["suites"].get(suite)
        if m is None:
            continue
        for key, v in getter(m):
            out.setdefault(e["controller"], {}).setdefault(key, []).append(v[metric])
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def table(evals, suite, metric, title) -> str:
    per_layout = lambda m: list(m["per_layout"].items()) + [("OVERALL", m["overall"])]
    vals = collect(evals, suite, metric, per_layout)
    ctrls = [c for c in CTRLS if c in vals]
    layouts = [l for l in LAYOUTS_E2 if any(l in vals[c] for c in ctrls)]
    layouts += sorted({l for c in ctrls for l in vals[c]} - set(layouts) - {"OVERALL"})
    n = {c: len(vals[c].get("OVERALL", [])) for c in ctrls}
    lines = [f"**{title}** (`{suite}`, mean ± std over seeds; runs: "
             + ", ".join(f"{c} {n[c]}" for c in ctrls) + ")", "",
             "| layout | " + " | ".join(ctrls) + " |", "|---|" + "---|" * len(ctrls)]
    for l in layouts + ["OVERALL"]:
        name = "**overall**" if l == "OVERALL" else l
        lines.append(f"| {name} | " + " | ".join(fmt(vals[c].get(l, [])) for c in ctrls) + " |")
    return "\n".join(lines)


def write_tables(evals, out: Path):
    for suite, fname in (("test4", "table_e2.md"), ("test_mixed", "table_e2_mixed.md")):
        if not any(suite in e["suites"] for e in evals):
            continue
        text = (f"# E2: {suite}\n\n" + table(evals, suite, "success_rate", "Success rate")
                + "\n\n" + table(evals, suite, "collision_rate", "Collision rate") + "\n")
        (out / fname).write_text(text)


def run_dir_of(e, runs_root):
    if e["ckpt"] is None:
        return None
    p = Path(e["ckpt"]).parent
    if runs_root is not None and (runs_root / p.name).exists():
        return runs_root / p.name
    return p if p.exists() else None


def training_curves(evals, runs_root, out: Path):
    curves = {}
    for e in evals:
        d = run_dir_of(e, runs_root)
        if e["controller"] == "greedy" or d is None or not (d / "val.csv").exists():
            continue
        with open(d / "val.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            curves.setdefault(e["controller"], []).append(
                (np.array([float(r["agent_steps"]) for r in rows]),
                 np.array([float(r["success_rate"]) for r in rows])))
    if curves:
        plots.training_curves(curves, out / "fig_training_curves.png")


def path_figure(evals, out: Path):
    from policyconfig import sim_config
    pick = {}
    for e in evals:
        if e["controller"] == "greedy" or e["seed"] == 0:
            pick[e["controller"]] = e
    panels = {}
    for c, e in pick.items():
        for f in sorted((e["dir"] / "trajectories").glob("test4_*.json")):
            rec = json.loads(f.read_text())
            item = panels.setdefault(rec["layout"], {"scene": Scene.from_dict(rec["scene_data"]),
                                                     "paths": {}})
            item["paths"][c] = rec
    if panels:
        layouts = [l for l in LAYOUTS_E2 if l in panels]
        plots.path_panels(panels, layouts, sim_config(), out / "fig_paths.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=None)
    a = ap.parse_args(argv)
    evals = load_evals(a.evals)
    if not evals:
        raise SystemExit(f"no evaluation directories (metrics.json) under {a.evals}")
    a.out.mkdir(parents=True, exist_ok=True)
    write_summaries(evals, a.out)
    write_tables(evals, a.out)
    training_curves(evals, a.runs, a.out)
    layouts = list(LAYOUTS_E2)
    if any("test4" in e["suites"] for e in evals):
        v = collect(evals, "test4", "success_rate", lambda m: m["per_layout"].items())
        plots.per_layout_bars(v, layouts, a.out / "fig_per_layout.png")
        bins = [f"[{lo:g},{'inf' if hi == math.inf else format(hi, 'g')})"
                for lo, hi in BOTTLENECK_BINS]
        v = collect(evals, "test4", "success_rate",
                    lambda m: [(b, x) for b, x in m["per_bottleneck"].items() if x["scenes"]])
        plots.bottleneck_curves(v, bins, a.out / "fig_bottleneck.png")
    path_figure(evals, a.out)
    print(f"wrote {sorted(p.name for p in a.out.iterdir())} to {a.out}")


if __name__ == "__main__":
    main()
