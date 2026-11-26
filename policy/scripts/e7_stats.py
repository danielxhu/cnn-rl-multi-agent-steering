"""Robust statistics for the E7 comparison (Agarwal et al. 2021, "rliable"), numpy only.

    python scripts/e7_stats.py --runs results/e7_stats/per_scene --out results/e7_stats

``--runs`` holds one ``<variant>_seed<k>.csv`` per checkpoint: the ``per_scene.csv``
that ``evaluate.py`` writes (one row per scene, rates over that scene's agents).

Per suite and metric, every run is reduced to a [seed, layout] score matrix (the
mean over the layout's scenes). On that matrix:

- mean and IQM (interquartile mean: the middle 50 % of the seed x layout
  values, as rliable does over runs x tasks);
- 95 % percentile CIs from a two-level stratified bootstrap: resample seeds
  with replacement, then scenes with replacement *within each layout*. The
  scene draw is shared by the two variants of a comparison (paired: the
  suites are identical), the seed draws are independent;
- differences (B - A) with the same bootstrap, and for loop rate the relative
  reduction 1 - B/A;
- probability of improvement P(B > A): rliable's definition, per layout the
  Mann-Whitney statistic over seed pairs (ties 1/2), averaged over layouts;
  plus a scene-level version (pairs of runs, scene by scene), which uses the
  scenes instead of only the seeds;
- per layout: difference of means with its CI. A layout's gain is called
  credible when the CI excludes 0.

Invariants:
- Read-only on its inputs; deterministic for a given ``--seed``.
- A variant with one seed gets no seed resampling: its CI covers scene noise
  only, so it is too narrow. The tables flag this.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = {                       # column -> (label, higher is better)
    "success_rate": ("success", True),
    "static_collision_rate": ("wall crash", False),
    "agent_collision_rate": ("agent crash", False),
    "timeout_rate": ("timeout", False),
    "loop_rate": ("loop", False),
}
EXTRA_METRICS = {                 # zeroshot_eval.py columns, used when every run has them
    "stuck_rate": ("stuck", False),
    "pause_steps": ("pause steps", False),
    "throughput": ("throughput /100 steps", True),
    "path_ratio": ("path ratio", False),
}
COMPARE = [("e3_both", "e7_map"), ("e3_both", "e7_longgru")]


def load_runs(folder: Path):
    """{variant: {seed: {suite: {"scene": [...], "layout": [...], metric: array}}}}.

    Adds ``EXTRA_METRICS`` to ``METRICS`` when every file has those columns."""
    files = sorted(folder.glob("*_seed*.csv"))
    heads = [set(next(csv.reader(open(f, newline="")))) for f in files]
    for k, v in EXTRA_METRICS.items():
        if heads and all(k in h for h in heads):
            METRICS[k] = v
    runs = defaultdict(dict)
    for f in files:
        m = re.fullmatch(r"(.+)_seed(\d+)", f.stem)
        by_suite = defaultdict(lambda: defaultdict(list))
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                d = by_suite[r["suite"]]
                d["scene"].append(r["scene"])
                d["layout"].append(r["layout"])
                for k in METRICS:
                    d[k].append(float(r[k]))
        runs[m.group(1)][int(m.group(2))] = {
            s: {k: (np.asarray(v) if k in METRICS else v) for k, v in d.items()}
            for s, d in by_suite.items()}
    return runs


def iqm(x):
    x = np.sort(np.ravel(x))
    x = x[np.isfinite(x)]
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return x[lo:hi].mean()


def layout_matrix(scores, seeds, groups, picks):
    """scores [n_seed, n_scene] -> [len(seeds), n_layout] means over picked scenes."""
    s = scores[seeds]
    with np.errstate(all="ignore"):                       # path_ratio is NaN without arrivals
        return np.stack([np.nanmean(s[:, g[p]], 1) for g, p in zip(groups, picks)], 1)


def poi(a, b):
    """rliable probability of improvement of b over a; a, b: [seed, layout]."""
    gt = (b[:, None, :] > a[None, :, :]).mean((0, 1))
    eq = (b[:, None, :] == a[None, :, :]).mean((0, 1))
    return float((gt + 0.5 * eq).mean())


def poi_scene(a, b):
    """Scene-level: a, b [seed, scene]; P(b-run beats a-run on a scene), ties 1/2."""
    gt = (b[:, None, :] > a[None, :, :]).mean()
    eq = (b[:, None, :] == a[None, :, :]).mean()
    return float(gt + 0.5 * eq)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def analyse(runs, suite, B, rng, compare=COMPARE):
    variants = sorted(runs)
    ref = runs[variants[0]][min(runs[variants[0]])][suite]
    names, layouts = ref["scene"], ref["layout"]
    for v in variants:                                     # same scenes, same order
        for sd in runs[v].values():
            assert sd[suite]["scene"] == names, f"{v}: scene order differs on {suite}"
    L = sorted(set(layouts))
    lay = np.asarray(layouts)
    groups = [np.flatnonzero(lay == l) for l in L]
    seeds = {v: sorted(runs[v]) for v in variants}
    S = {(v, k): np.stack([runs[v][s][suite][k] for s in seeds[v]]) for v in variants for k in METRICS}

    full = [np.arange(len(g)) for g in groups]
    point = {}
    for v in variants:
        for k in METRICS:
            M = layout_matrix(S[(v, k)], np.arange(len(seeds[v])), groups, full)
            point[(v, k)] = dict(mean=np.nanmean(M), iqm=iqm(M), lay=np.nanmean(M, 0), M=M,
                                 seed_sd=float(np.nanmean(M, 1).std(ddof=1)) if len(M) > 1 else float("nan"))

    boot = defaultdict(list)
    for _ in range(B):
        picks = [rng.integers(0, len(g), len(g)) for g in groups]
        for v in variants:
            sd = rng.integers(0, len(seeds[v]), len(seeds[v]))
            for k in METRICS:
                M = layout_matrix(S[(v, k)], sd, groups, picks)
                boot[(v, k)].append((np.nanmean(M), iqm(M), np.nanmean(M, 0)))
    out = {"layouts": L, "seeds": seeds, "variants": {}, "compare": {}}
    for v in variants:
        out["variants"][v] = {}
        for k in METRICS:
            b = boot[(v, k)]
            out["variants"][v][k] = dict(
                mean=float(point[(v, k)]["mean"]), mean_ci=ci([x[0] for x in b]),
                iqm=float(point[(v, k)]["iqm"]), iqm_ci=ci([x[1] for x in b]),
                seed_sd=point[(v, k)]["seed_sd"],
                per_layout={l: float(x) for l, x in zip(L, point[(v, k)]["lay"])})
    for a, b_ in compare:
        if a not in runs or b_ not in runs:
            continue
        res = {}
        for k, (_, higher) in METRICS.items():
            A, Bv = boot[(a, k)], boot[(b_, k)]
            d_mean = [y[0] - x[0] for x, y in zip(A, Bv)]
            d_iqm = [y[1] - x[1] for x, y in zip(A, Bv)]
            d_lay = np.array([y[2] - x[2] for x, y in zip(A, Bv)])
            pa, pb = point[(a, k)], point[(b_, k)]
            sa, sb = S[(a, k)], S[(b_, k)]
            if not higher:                                  # P(improvement) = P(lower)
                pa_M, pb_M, sa_, sb_ = -pa["M"], -pb["M"], -sa, -sb
            else:
                pa_M, pb_M, sa_, sb_ = pa["M"], pb["M"], sa, sb
            r = dict(
                diff_mean=float(pb["mean"] - pa["mean"]), diff_mean_ci=ci(d_mean),
                diff_iqm=float(pb["iqm"] - pa["iqm"]), diff_iqm_ci=ci(d_iqm),
                poi=poi(pa_M, pb_M), poi_scene=poi_scene(sa_, sb_),
                per_layout={l: dict(diff=float(pb["lay"][j] - pa["lay"][j]),
                                    ci=ci(d_lay[:, j])) for j, l in enumerate(L)})
            if k == "loop_rate":
                rel = [1 - y[0] / x[0] for x, y in zip(A, Bv) if x[0] > 0]
                r["rel_reduction"] = float(1 - pb["mean"] / pa["mean"])
                r["rel_reduction_ci"] = ci(rel)
            res[k] = r
        out["compare"][f"{b_}_vs_{a}"] = res
    return out


def pct(x, k="success_rate"):
    """Rates in %, other metrics (steps, ratios, throughput) as they are."""
    return f"{100 * x:.1f}" if k.endswith("_rate") else f"{x:.2f}"


def cis(c, k="success_rate"):
    return f"[{pct(c[0], k)}, {pct(c[1], k)}]"


def markdown(stats, B):
    lines = ["# E7 robust statistics", "",
             f"Two-level stratified bootstrap (seeds, then scenes within layout), B = {B}; "
             "95 % percentile CIs; rates in %, other metrics as they are. Method: `scripts/e7_stats.py`.", ""]
    for suite, st in stats.items():
        lines += [f"## {suite}", "", "Seeds per variant: " +
                  ", ".join(f"{v} {len(s)}" for v, s in st["seeds"].items()) +
                  ". A variant with one seed has a scene-only CI (too narrow).", "",
                  "| variant | metric | mean [CI] | IQM [CI] |", "|---|---|---|---|"]
        for v, d in st["variants"].items():
            for k, (lab, _) in METRICS.items():
                x = d[k]
                lines.append(f"| {v} | {lab} | {pct(x['mean'], k)} {cis(x['mean_ci'], k)} | "
                             f"{pct(x['iqm'], k)} {cis(x['iqm_ci'], k)} |")
        for name, res in st["compare"].items():
            lines += ["", f"### {name}", "",
                      "| metric | diff of means [CI] | diff of IQM [CI] | P(improve) seeds | P(improve) scenes |",
                      "|---|---|---|---|---|"]
            for k, (lab, _) in METRICS.items():
                r = res[k]
                lines.append(f"| {lab} | {pct(r['diff_mean'], k)} {cis(r['diff_mean_ci'], k)} | "
                             f"{pct(r['diff_iqm'], k)} {cis(r['diff_iqm_ci'], k)} | {r['poi']:.2f} | "
                             f"{r['poi_scene']:.2f} |")
            if "rel_reduction" in res["loop_rate"]:
                r = res["loop_rate"]
                lines += ["", f"Loop rate relative reduction: {pct(r['rel_reduction'])} "
                              f"{cis(r['rel_reduction_ci'])}."]
            lines += ["", "Per layout, difference of means [CI]; * = CI excludes 0:", "",
                      "| layout | " + " | ".join(METRICS[k][0] for k in METRICS) + " |",
                      "|---" * (len(METRICS) + 1) + "|"]
            for l in st["layouts"]:
                cells = []
                for k in METRICS:
                    p = res[k]["per_layout"][l]
                    star = "*" if p["ci"][0] > 0 or p["ci"][1] < 0 else ""
                    cells.append(f"{pct(p['diff'], k)} {cis(p['ci'], k)}{star}")
                lines.append(f"| {l} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suites", default="test4,test_mixed")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", default=None,
                    help="comma list of A:B pairs (B vs A); default e7_map and e7_longgru vs e3_both")
    a = ap.parse_args(argv)
    runs = load_runs(a.runs)
    rng = np.random.default_rng(a.seed)
    compare = [tuple(p.split(":")) for p in a.compare.split(",")] if a.compare else COMPARE
    stats = {s: analyse(runs, s, a.boot, rng, compare) for s in a.suites.split(",")}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    (a.out / "table_stats.md").write_text(markdown(stats, a.boot))
    print(a.out / "table_stats.md")


if __name__ == "__main__":
    main()
