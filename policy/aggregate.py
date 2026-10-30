"""Collect evaluation directories into the E2 / E3 tables and figures (DESIGN.md §6, DESIGN_E3.md §6).

    python aggregate.py --evals E:\\policy-eval --out results/e2 [--runs E:\\policy-runs]
    python aggregate.py --evals E:\\policy-eval --out results/e3 --runs E:\\policy-runs \\
        --group variant --variants greedy e2_gru e3_timeout e3_geo e3_both --tag e3

Owns ``summary.csv``, ``summary_per_layout.csv``, ``table_<tag>.md``,
``table_<tag>_mixed.md``, the four ``fig_*.png`` and, with ``--group variant``,
``winner.json``.

Invariants:
- An evaluation directory is any sub-directory of ``--evals`` with a
  ``metrics.json``; its identity (``controller`` = greedy / mlp / gru, ``seed``,
  ``variant``) comes from that file, not from the directory name. Evaluations
  written before E3 have no ``variant``: it is then the directory name without
  ``_seed<k>``.
- Columns are groups: ``--group controller`` (default, E2) or ``variant``
  (E3); ``--variants`` / the default order fixes them, and evaluations of other
  groups are left out of tables and figures (not of the summaries).
- The E3 winner is chosen on *validation* (``val.csv`` of the runs, best row
  per run as ``ckpt_best.pt``), never on the test suites.
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
import re
from pathlib import Path

import numpy as np

import plots
from evaluate import BOTTLENECK_BINS, ROW_METRICS
from policyconfig import LAYOUTS_E2
from scene import Scene

CTRLS = ["greedy", "mlp", "gru"]


def load_evals(root: Path, group: str = "controller") -> list:
    out = []
    for m in sorted(root.glob("*/metrics.json")):
        data = json.loads(m.read_text(encoding="utf-8"))
        ident = data["identity"]
        variant = ident.get("variant") or ("greedy" if ident["controller"] == "greedy"
                                           else re.sub(r"_seed\d+$", "", m.parent.name))
        e = {"dir": m.parent, "name": m.parent.name, "controller": ident["controller"],
             "variant": variant, "seed": ident.get("seed"), "ckpt": ident.get("ckpt"),
             "suites": data["suites"]}
        e["group"] = e[group]
        out.append(e)
    return out


def write_summaries(evals: list, out: Path, with_variant=False):
    ident = ["controller", "variant"] if with_variant else ["controller"]
    cols = ident + ["seed", "run", "suite", "scenes"] + ROW_METRICS
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in evals:
            for s, m in e["suites"].items():
                w.writerow({**{k: e[k] for k in ident}, "seed": e["seed"], "run": e["name"],
                            "suite": s, **{k: m["overall"].get(k) for k in ["scenes"] + ROW_METRICS}})
    cols = ident + ["seed", "run", "suite", "layout", "scenes"] + ROW_METRICS
    with open(out / "summary_per_layout.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in evals:
            for s, m in e["suites"].items():
                for l, v in m["per_layout"].items():
                    w.writerow({**{k: e[k] for k in ident}, "seed": e["seed"], "run": e["name"],
                                "suite": s, "layout": l,
                                **{k: v.get(k) for k in ["scenes"] + ROW_METRICS}})


def collect(evals, suite, metric, getter):
    """{group: {key: [value per run]}} for a per-layout / overall metric."""
    out = {}
    for e in evals:
        m = e["suites"].get(suite)
        if m is None:
            continue
        for key, v in getter(m):
            out.setdefault(e["group"], {}).setdefault(key, []).append(v[metric])
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def table(evals, suite, metric, title, order=CTRLS) -> str:
    per_layout = lambda m: list(m["per_layout"].items()) + [("OVERALL", m["overall"])]
    vals = collect(evals, suite, metric, per_layout)
    ctrls = [c for c in order if c in vals]
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


def write_tables(evals, out: Path, order=CTRLS, tag="e2", timeouts=False):
    for suite, fname in (("test4", f"table_{tag}.md"), ("test_mixed", f"table_{tag}_mixed.md")):
        if not any(suite in e["suites"] for e in evals):
            continue
        parts = [table(evals, suite, "success_rate", "Success rate", order),
                 table(evals, suite, "collision_rate", "Collision rate", order)]
        if timeouts:                                       # the quantity E3 targets
            parts.append(table(evals, suite, "timeout_rate", "Timeout rate", order))
        text = f"# {tag.upper()}: {suite}\n\n" + "\n\n".join(parts) + "\n"
        (out / fname).write_text(text, encoding="utf-8")   # "±": Windows defaults to GBK


def run_dir_of(e, runs_root):
    if e["ckpt"] is None:
        return None
    p = Path(e["ckpt"]).parent
    if runs_root is not None and (runs_root / p.name).exists():
        return runs_root / p.name
    return p if p.exists() else None


def val_rows(e, runs_root) -> list:
    d = run_dir_of(e, runs_root)
    if e["controller"] == "greedy" or d is None or not (d / "val.csv").exists():
        return []
    with open(d / "val.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def training_curves(evals, runs_root, out: Path, order=None):
    curves = {}
    for e in evals:
        rows = val_rows(e, runs_root)
        if rows:
            curves.setdefault(e["group"], []).append(
                (np.array([float(r["agent_steps"]) for r in rows]),
                 np.array([float(r["success_rate"]) for r in rows])))
    if curves:
        series = [g for g in (order or ["mlp", "gru"]) if g != "greedy"]
        plots.training_curves(curves, out / "fig_training_curves.png", series)


def pick_winner(evals, runs_root, order) -> dict | None:
    """DESIGN_E3.md §2: highest mean validation success over seeds, ties -> lower collision.

    A run's validation score is its best ``val.csv`` row by the same rule, the
    one ``ckpt_best.pt`` was saved at."""
    per = {}
    for e in evals:
        if e["group"] == "greedy" or e["group"] not in order:
            continue
        rows = val_rows(e, runs_root)
        if not rows:
            continue
        best = max(rows, key=lambda r: (float(r["success_rate"]), -float(r["collision_rate"])))
        per.setdefault(e["group"], []).append(
            (e["seed"], float(best["success_rate"]), float(best["collision_rate"])))
    if not per:
        return None
    score = {g: (float(np.mean([s for _, s, _ in v])), float(np.mean([c for _, _, c in v])))
             for g, v in per.items()}
    g = max(score, key=lambda k: (score[k][0], -score[k][1]))
    return {"variant": g, "val_success_mean": score[g][0], "val_collision_mean": score[g][1],
            "seeds": sorted(s for s, _, _ in per[g]),
            "all": {k: {"val_success_mean": v[0], "val_collision_mean": v[1],
                        "seeds": sorted(s for s, _, _ in per[k])} for k, v in score.items()}}


def path_figure(evals, out: Path, groups=None):
    """One scene per layout; ``groups`` limits the series (default: every group)."""
    from policyconfig import sim_config
    pick = {}
    for e in evals:
        if groups is not None and e["group"] not in groups:
            continue
        if e["group"] == "greedy" or e["seed"] == 0:
            pick[e["group"]] = e
    panels = {}
    for c, e in pick.items():
        for f in sorted((e["dir"] / "trajectories").glob("test4_*.json")):
            rec = json.loads(f.read_text())
            item = panels.setdefault(rec["layout"], {"scene": Scene.from_dict(rec["scene_data"]),
                                                     "paths": {}})
            item["paths"][c] = rec
    if panels:
        layouts = [l for l in LAYOUTS_E2 if l in panels]
        plots.path_panels(panels, layouts, sim_config(), out / "fig_paths.png",
                          order=[g for g in (groups or plots.ORDER)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--group", choices=["controller", "variant"], default="controller")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="columns and their order (default: greedy mlp gru, or every variant)")
    ap.add_argument("--tag", default=None, help="table names: table_<tag>.md (default e2 / e3)")
    a = ap.parse_args(argv)
    evals = load_evals(a.evals, a.group)
    if not evals:
        raise SystemExit(f"no evaluation directories (metrics.json) under {a.evals}")
    by_variant = a.group == "variant"
    if a.variants:
        order = list(a.variants)
    elif by_variant:
        order = sorted({e["group"] for e in evals}, key=lambda g: (g != "greedy", g))
    else:
        order = list(CTRLS)
    tag = a.tag or ("e3" if by_variant else "e2")
    shown = [e for e in evals if e["group"] in order]
    a.out.mkdir(parents=True, exist_ok=True)
    write_summaries(evals, a.out, with_variant=by_variant)
    write_tables(shown, a.out, order, tag, timeouts=by_variant)
    training_curves(shown, a.runs, a.out, order if by_variant else None)
    layouts = list(LAYOUTS_E2)
    if any("test4" in e["suites"] for e in shown):
        v = collect(shown, "test4", "success_rate", lambda m: m["per_layout"].items())
        plots.per_layout_bars(v, layouts, a.out / "fig_per_layout.png", order=order)
        bins = [f"[{lo:g},{'inf' if hi == math.inf else format(hi, 'g')})"
                for lo, hi in BOTTLENECK_BINS]
        v = collect(shown, "test4", "success_rate",
                    lambda m: [(b, x) for b, x in m["per_bottleneck"].items() if x["scenes"]])
        plots.bottleneck_curves(v, bins, a.out / "fig_bottleneck.png", order=order)
    if by_variant:
        win = pick_winner(shown, a.runs, order)
        if win is not None:
            (a.out / "winner.json").write_text(json.dumps(win, indent=2), encoding="utf-8")
            print(f"winner (validation): {win['variant']}  success {win['val_success_mean']:.3f}  "
                  f"collision {win['val_collision_mean']:.3f}  seeds {win['seeds']}")
        # E3 path figure: the default reward against the winner (DESIGN_E3.md §6)
        pair = ["e2_gru"] + ([win["variant"]] if win and win["variant"] != "e2_gru" else ["greedy"])
        path_figure(shown, a.out, pair)
    else:
        path_figure(shown, a.out)
    print(f"wrote {sorted(p.name for p in a.out.iterdir())} to {a.out}")


if __name__ == "__main__":
    main()
