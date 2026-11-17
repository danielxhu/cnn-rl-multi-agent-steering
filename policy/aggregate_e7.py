"""Collect the E7 evaluations into tables, figures and the verdict (DESIGN_E7.md §5–§6).

    python aggregate_e7.py --evals E:\\policy-eval\\e7 --out results/e7 [--check-against E:\\policy-eval]

Owns ``table_e7.md`` / ``table_e7_mixed.md`` (per layout: success, static and
agent collision, timeout, loop rate; per bottleneck; the multi-exit subset),
``table_e7_longgru.md`` (seed 0 only), ``table_e7_odo.md`` (odometry noise),
``summary_e7.csv``, ``fig_per_layout.png``, ``fig_bottleneck.png``,
``fig_loop.png`` and ``verdict.json``.

Invariants:
- Groups are variants (``metrics.json`` identity), except that an evaluation
  with ``odo_noise`` is the group ``<variant>_odo_<level>``: the noise rows
  never mix with the clean ones.
- The verdict follows DESIGN_E7.md §6 exactly and uses ``test4`` only; it
  needs ``e3_both`` and ``e7_map`` (otherwise it is written as ``null``).
- ``--check-against`` compares the re-evaluated baselines (``e3_both_seed*``,
  ``orca_*``) with the evaluations of the same name there, on every metric
  that existed before E7; ``baseline_reproduced`` is False on any difference.
- Missing groups (a partial set of runs) skip what needs them.
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
from aggregate import collect, fmt, load_evals
from evaluate import BOTTLENECK_BINS, E7_METRICS, ROW_METRICS
from policyconfig import LAYOUTS_E2

ORDER = ["orca_straight", "orca_path", "e3_both", "e7_longgru", "e7_map"]
ODO = ["e7_map", "e7_map_odo_low", "e7_map_odo_high"]
LAYOUT_METRICS = [("success_rate", "Success rate"),
                  ("static_collision_rate", "Static (wall / obstacle) collision rate"),
                  ("agent_collision_rate", "Agent collision rate"),
                  ("timeout_rate", "Timeout rate"),
                  ("loop_rate", "Loop rate (cumulative heading change spans >= 360 deg)")]
BINS = [f"[{lo:g},{'inf' if hi == math.inf else format(hi, 'g')})" for lo, hi in BOTTLENECK_BINS]


def load(root: Path) -> list:
    evals = load_evals(root, "variant")
    for e in evals:
        ident = json.loads((e["dir"] / "metrics.json").read_text(encoding="utf-8"))["identity"]
        if ident.get("odo_noise") not in (None, "none"):
            e["group"] = f"{e['variant']}_odo_{ident['odo_noise']}"
    return evals


def _table(evals, suite, metric, title, order, getter, rows_name, rows) -> str:
    vals = collect(evals, suite, metric, getter)
    cols = [c for c in order if c in vals]
    if not cols:
        return ""
    n = {c: len(next(iter(vals[c].values()), [])) for c in cols}
    lines = [f"**{title}** (`{suite}`, mean ± std over seeds; runs: "
             + ", ".join(f"{c} {n[c]}" for c in cols) + ")", "",
             f"| {rows_name} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        if not any(r in vals[c] for c in cols):
            continue
        name = "**overall**" if r == "OVERALL" else r
        lines.append(f"| {name} | " + " | ".join(fmt(vals[c].get(r, [])) for c in cols) + " |")
    return "\n".join(lines)


def per_layout(m):
    return list(m["per_layout"].items()) + [("OVERALL", m["overall"])]


def per_bin(m):
    return [(b, x) for b, x in m["per_bottleneck"].items() if x.get("scenes")]


def multi(m):
    x = m.get("multi_exit")
    return [("multi-exit scenes", x)] if x and x.get("scenes") else []


def suite_tables(evals, suite, order) -> str:
    layouts = list(LAYOUTS_E2) + ["OVERALL"]
    parts = [_table(evals, suite, k, t, order, per_layout, "layout", layouts)
             for k, t in LAYOUT_METRICS]
    parts.append(_table(evals, suite, "success_rate", "Success rate per bottleneck width",
                        order, per_bin, "bottleneck", BINS))
    for k, t in (("success_rate", "Multi-exit subset: success"),
                 ("loop_rate", "Multi-exit subset: loop rate"),
                 ("exit_switches", "Multi-exit subset: exit switches per agent")):
        parts.append(_table(evals, suite, k, t, order, multi, "subset", ["multi-exit scenes"]))
    for k, t in (("loop_success_rate", "Success of looping agents"),
                 ("noloop_success_rate", "Success of the other agents")):
        parts.append(_table(evals, suite, k, t, order, lambda m: [("OVERALL", m["overall"])],
                            "", ["OVERALL"]))
    return f"# E7: {suite}\n\n" + "\n\n".join(p for p in parts if p) + "\n"


def write_summary(evals, out: Path):
    cols = ["group", "variant", "seed", "run", "suite", "scenes"] + ROW_METRICS + E7_METRICS
    with open(out / "summary_e7.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in evals:
            for s, m in e["suites"].items():
                w.writerow({"group": e["group"], "variant": e["variant"], "seed": e["seed"],
                            "run": e["name"], "suite": s,
                            **{k: m["overall"].get(k) for k in ["scenes"] + ROW_METRICS + E7_METRICS}})


def overall(evals, group, metric, suite="test4", seed=None) -> list:
    return [e["suites"][suite]["overall"][metric] for e in evals
            if e["group"] == group and suite in e["suites"]
            and (seed is None or e["seed"] == seed)
            and metric in e["suites"][suite]["overall"]]


def verdict(evals) -> dict | None:
    """DESIGN_E7.md §6 on test4."""
    s3, s7 = overall(evals, "e3_both", "success_rate"), overall(evals, "e7_map", "success_rate")
    l3, l7 = overall(evals, "e3_both", "loop_rate"), overall(evals, "e7_map", "loop_rate")
    if not (s3 and s7 and l3 and l7):
        return None
    margin = float(np.std(s3) + np.std(s7))
    d_succ = float(np.mean(s7) - np.mean(s3))
    loop_drop = float(1.0 - np.mean(l7) / np.mean(l3)) if np.mean(l3) > 0 else 0.0
    rule_a = d_succ > margin
    rule_b = loop_drop >= 1 / 3 and d_succ >= -margin
    out = {"e3_both_success": float(np.mean(s3)), "e7_map_success": float(np.mean(s7)),
           "success_gain": d_succ, "margin_sum_of_stds": margin,
           "e3_both_loop_rate": float(np.mean(l3)), "e7_map_loop_rate": float(np.mean(l7)),
           "loop_rate_drop": loop_drop, "rule_success": bool(rule_a), "rule_loops": bool(rule_b),
           "e7_map_helps": bool(rule_a or rule_b),
           "seeds": {"e3_both": len(s3), "e7_map": len(s7)}}
    for k in ("static_collision_rate", "agent_collision_rate"):
        a, b = overall(evals, "e3_both", k), overall(evals, "e7_map", k)
        out[f"e3_both_{k}"], out[f"e7_map_{k}"] = float(np.mean(a)), float(np.mean(b))
    high = overall(evals, "e7_map_odo_high", "success_rate")
    if high:
        loss = float(np.mean(s7) - np.mean(high))
        out["odo_high_success_loss"] = loss
        out["propose_training_with_drift"] = bool(d_succ > 0 and loss > d_succ / 2)
    return out


def longgru_table(evals, suite="test4") -> str:
    cols = ["e3_both", "e7_longgru", "e7_map"]
    lines = [f"**Seed 0 only** (`{suite}`; one run each, indicative, DESIGN_E7.md §6.4)", "",
             "| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in ("success_rate", "static_collision_rate", "agent_collision_rate", "timeout_rate",
              "loop_rate"):
        cells = []
        for c in cols:
            v = overall(evals, c, k, suite, seed=0)
            cells.append(f"{v[0]:.3f}" if v else "–")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "# E7: longer GRU memory vs wall memory\n\n" + "\n".join(lines) + "\n"


def odo_table(evals, suite="test4") -> str:
    lines = [f"**Odometry noise at evaluation** (`{suite}`, mean ± std over seeds)", "",
             "| noise | success | loop rate | static collision |", "|---|---|---|---|"]
    for g, name in zip(ODO, ("none", "low", "high")):
        if not overall(evals, g, "success_rate", suite):
            continue
        lines.append(f"| {name} | " + " | ".join(
            fmt(overall(evals, g, k, suite))
            for k in ("success_rate", "loop_rate", "static_collision_rate")) + " |")
    return "# E7: odometry noise\n\n" + "\n".join(lines) + "\n"


def loop_figure(evals, order, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals = collect(evals, "test4", "loop_rate", lambda m: m["per_layout"].items())
    cols = [c for c in order if c in vals]
    if not cols:
        return
    layouts = [l for l in LAYOUTS_E2 if any(l in vals[c] for c in cols)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.8 / len(cols)
    x = np.arange(len(layouts))
    for k, c in enumerate(cols):
        mu = [np.mean(vals[c].get(l, [np.nan])) for l in layouts]
        sd = [np.std(vals[c].get(l, [np.nan])) for l in layouts]
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, mu, w, yerr=sd, label=c,
               color=plots.color(c, k), capsize=2)
    ax.set_xticks(x, layouts, rotation=30, ha="right")
    ax.set_ylabel("loop rate")
    ax.set_title("test4: share of agents that loop (>= 360 deg)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def check_baselines(evals, ref_root: Path) -> dict:
    """Re-evaluated baselines against the evaluations of the same name in ``ref_root``."""
    out = {}
    for e in evals:
        if e["group"] not in ("e3_both", "orca_straight", "orca_path"):
            continue
        ref = ref_root / e["name"] / "metrics.json"
        if not ref.exists():
            out[e["name"]] = "no reference"
            continue
        r = json.loads(ref.read_text(encoding="utf-8"))["suites"]
        diffs = []
        for s, m in e["suites"].items():
            if s not in r:
                continue
            for k in ROW_METRICS:
                a, b = m["overall"].get(k), r[s]["overall"].get(k)
                if a is None or b is None:
                    continue
                if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                        or (math.isnan(a) and math.isnan(b))):
                    diffs.append(f"{s}.{k}: {b} -> {a}")
        out[e["name"]] = diffs or "identical"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-against", type=Path, default=None,
                    help="evaluation root holding the pre-E7 baseline evaluations")
    a = ap.parse_args(argv)
    evals = load(a.evals)
    if not evals:
        raise SystemExit(f"no evaluation directories (metrics.json) under {a.evals}")
    a.out.mkdir(parents=True, exist_ok=True)
    write_summary(evals, a.out)
    clean = [e for e in evals if e["group"] in ORDER]
    for suite, fname in (("test4", "table_e7.md"), ("test_mixed", "table_e7_mixed.md")):
        if any(suite in e["suites"] for e in clean):
            (a.out / fname).write_text(suite_tables(clean, suite, ORDER), encoding="utf-8")
    (a.out / "table_e7_longgru.md").write_text(longgru_table(evals), encoding="utf-8")
    (a.out / "table_e7_odo.md").write_text(odo_table(evals), encoding="utf-8")
    if any("test4" in e["suites"] for e in clean):
        v = collect(clean, "test4", "success_rate", lambda m: m["per_layout"].items())
        plots.per_layout_bars(v, list(LAYOUTS_E2), a.out / "fig_per_layout.png", order=ORDER)
        v = collect(clean, "test4", "success_rate", per_bin)
        plots.bottleneck_curves(v, BINS, a.out / "fig_bottleneck.png", order=ORDER)
        loop_figure(clean, ORDER, a.out / "fig_loop.png")
    ver = verdict(evals)
    if a.check_against is not None:
        chk = check_baselines(evals, a.check_against)
        ver = dict(ver or {}, baseline_check=chk,
                   baseline_reproduced=all(v in ("identical", "no reference") for v in chk.values()))
    (a.out / "verdict.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")
    if ver and "e7_map_helps" in ver:
        print(f"e7_map {ver['e7_map_success']:.3f} vs e3_both {ver['e3_both_success']:.3f} "
              f"(margin {ver['margin_sum_of_stds']:.3f}); loop rate {ver['e7_map_loop_rate']:.3f} "
              f"vs {ver['e3_both_loop_rate']:.3f}; helps: {ver['e7_map_helps']}")
    print(f"wrote {sorted(p.name for p in a.out.iterdir())} to {a.out}")


if __name__ == "__main__":
    main()
