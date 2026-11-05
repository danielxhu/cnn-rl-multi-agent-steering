"""Behaviour gap: one policy on true scenes and on the scenes perception parsed (DESIGN_GAP.md).

    python gap.py --ckpt models/e3_both_seed0/ckpt_best.pt --data-root E:\\cnn-data \\
                  --parsed WORK/parsed --perception WORK/perception --sets test_sketch7 --out OUT

Expects perception's outputs (``scripts/run_gap.py`` makes them):
``<parsed>/<set>/<scene>.json`` and ``<perception>/<set>/per_scene.csv``, whose
rows decide which scenes count. Writes to ``--out``: ``per_scene.csv``,
``gap.md``, ``metrics.json``, ``fig_gap.png``, ``fig_gap_map.png``, ``identity.json``.

Runs per scene: the policy in the true and in the parsed scene; ``orca_path`` in
the true and in the parsed scene (arm 1); ``orca_path`` in the true scene planning
on the parsed walls and obstacles (arm 2, the "wrong map").

Invariants:
- ``per_scene.csv`` gets one complete row per scene as it finishes. A second run
  drops a cut-off last line, skips finished scenes and rebuilds the tables.
- ``identity.json`` refuses a folder holding another policy's or parse's rows.
- A failed parse has no parsed episode: empty ``parsed_*`` and ``gap_*`` cells,
  reported on their own, never averaged into a gap.
- Episodes are deterministic and independent, so a process pool changes only speed.
- An error in an ORCA run leaves that run's cells empty and fills ``orca_error``;
  it never changes whether a scene counts as parsed, nor the policy's numbers.
"""
from __future__ import annotations

import _paths  # noqa: F401

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from demo_e2e import start_check, to_scene  # noqa: E402

from evaluate import load_policy, run_episode  # noqa: E402
from orca import N_PATH_DIRS, ORCAController, _bilinear_finite  # noqa: E402
from policyconfig import default_reward  # noqa: E402

SIM_KEYS = ["world", "agent_radius", "wall_thickness", "max_speed", "max_steps", "max_turn_deg",
            "fov_deg", "fov_range", "n_sectors", "rays_per_sector"]
EPISODE = {"success": "success_rate", "collision": "collision_rate",
           "static_coll": "static_collision_rate", "agent_coll": "agent_collision_rate",
           "timeout": "timeout_rate", "finish_time": "finish_time",
           "path_length": "mean_path_length", "all_reached": "all_reached"}
GAP_METRICS = ["success", "collision", "timeout"]
PERCEPTION = ["wall_iou", "wall_line_err", "agent_f1", "agent_center_err", "heading_err_deg",
              "obst_count_ok", "region_set_ok", "scene_usable"]
COLUMNS = (["set", "scene", "layout", "per_agent_goals", "n_agents", "n_pred_agents",
            "agent_count_err", "parse_ok", "parse_reason"]
           + PERCEPTION
           + [f"true_{k}" for k in EPISODE] + ["true_start_touch"]
           + [f"parsed_{k}" for k in EPISODE] + ["parsed_start_touch"]
           + [f"gap_{k}" for k in GAP_METRICS]
           + [f"orca_true_{k}" for k in EPISODE]
           + [f"orca_parsed_{k}" for k in EPISODE]
           + [f"orca_gap_{k}" for k in GAP_METRICS]
           + [f"orca_wrongmap_{k}" for k in EPISODE]
           + [f"map_gap_{k}" for k in GAP_METRICS]
           + ["wrongmap_no_path_agents", "wrongmap_no_path_steps", "orca_error"]
           + ["parse_differs"])                 # last: a line that has it is complete
TEXT_COLUMNS = {"set", "scene", "layout", "parse_reason", "orca_error"}
NAN = float("nan")

_WORKER = {}


# --------------------------------------------------------------- episodes

def init_worker(ckpt):
    """Load the policy once per process (module level: ``spawn`` pickles it by name)."""
    ctrl, _, trained, _ = load_policy(ckpt, True, 0)
    _WORKER.update(ctrl=ctrl, sim=default_reward(trained))


def episode(scene, prefix, row, ctrl=None):
    """Run one deterministic episode and write its numbers to ``row``."""
    res = run_episode(scene, ctrl or _WORKER["ctrl"], _WORKER["sim"])
    for k, src in EPISODE.items():
        row[f"{prefix}_{k}"] = res[src]
    if ctrl is None:
        row[f"{prefix}_start_touch"] = len(start_check(scene, _WORKER["sim"]))


def no_path_at_start(ctrl, scene, sim_cfg) -> int:
    """Agents whose first step has no finite path on the controller's map (orca._path's test)."""
    count = 0
    step = 2 * math.pi / N_PATH_DIRS
    ang = np.arange(N_PATH_DIRS) * step
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1) * sim_cfg.max_speed
    for a in scene.agents:
        cand = np.asarray(a.pos, float)[None, :] + dirs
        vals = _bilinear_finite(ctrl.fields[scene.goal_for(a)], cand, sim_cfg.geo_cell)
        count += not np.isfinite(vals).any()
    return count


def orca_arms(true_scene, parsed_scene, row):
    """``orca_path`` runs of one scene (DESIGN_GAP.md §5); parsed_scene None: true run only."""
    try:
        episode(true_scene, "orca_true", row, ORCAController("path"))
        if parsed_scene is None:
            return
        episode(parsed_scene, "orca_parsed", row, ORCAController("path"))
        wrong = ORCAController("path", map_scene=parsed_scene)
        episode(true_scene, "orca_wrongmap", row, wrong)
        row["wrongmap_no_path_agents"] = no_path_at_start(wrong, true_scene, _WORKER["sim"])
        row["wrongmap_no_path_steps"] = wrong.no_path_steps
        for k in GAP_METRICS:
            row[f"orca_gap_{k}"] = row[f"orca_parsed_{k}"] - row[f"orca_true_{k}"]
            row[f"map_gap_{k}"] = row[f"orca_wrongmap_{k}"] - row[f"orca_true_{k}"]
    except Exception as e:      # reported in its own column, the policy's row stands
        row["orca_error"] = f"{type(e).__name__}: {e}"


def run_scene(job) -> dict:
    """(set, scene, true JSON, parsed JSON, perception row) -> one ``per_scene.csv`` row."""
    set_name, stem, true_path, parsed_path, prow = job
    d = json.loads(Path(true_path).read_text(encoding="utf-8"))
    true_scene, _ = to_scene(d, _WORKER["sim"])
    rec = json.loads(Path(parsed_path).read_text(encoding="utf-8"))
    n_pred = len(rec.get("agents", []))
    row = {"set": set_name, "scene": stem, "layout": true_scene.layout,
           "per_agent_goals": int(any(a.goal_region is not None for a in true_scene.agents)),
           "n_agents": len(true_scene.agents), "n_pred_agents": n_pred,
           "agent_count_err": n_pred - len(true_scene.agents), "parse_ok": 0, "parse_reason": ""}
    row.update({k: prow.get(k, "") for k in PERCEPTION})
    row["parse_differs"] = int((rec.get("parsed") is False) != (num(prow.get("parse_fail")) == 1.0)
                               or (rec.get("parsed") is not False
                                   and num(prow.get("n_pred_agents")) != n_pred))
    episode(true_scene, "true", row)
    try:
        parsed_scene, _ = to_scene(rec, _WORKER["sim"])
        episode(parsed_scene, "parsed", row)
    except Exception as e:      # one bad parse is a reported failure, not a stopped run
        row["parse_reason"] = f"{type(e).__name__}: {e}"
        orca_arms(true_scene, None, row)
        return row
    row["parse_ok"] = 1
    for k in GAP_METRICS:
        row[f"gap_{k}"] = row[f"parsed_{k}"] - row[f"true_{k}"]
    orca_arms(true_scene, parsed_scene, row)
    return row


# ------------------------------------------------------------- rows on disk

def num(v) -> float:
    return float(v) if v not in ("", None) else NAN


def cell(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else v


def read_rows(path: Path) -> list:
    """The complete rows of ``per_scene.csv`` (raw strings); [] if there is none."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path}: other columns than this gap.py writes; move it away")
        return [r for r in reader if r[COLUMNS[-1]] not in (None, "")]


def write_rows(path: Path, rows: list):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def numeric(rows: list) -> list:
    return [{k: (v if k in TEXT_COLUMNS else num(v)) for k, v in r.items()} for r in rows]


# -------------------------------------------------------------- aggregation

def _mean(values) -> float:
    v = [x for x in values if not math.isnan(x)]
    return float(np.mean(v)) if v else NAN


def arm_stats(rows: list, base: str, other: str) -> dict:
    """``other`` minus ``base``, paired over the rows where both ran."""
    out = {}
    for m in GAP_METRICS + ["static_coll"]:
        pairs = [(r[f"{base}_{m}"], r[f"{other}_{m}"]) for r in rows
                 if not math.isnan(r.get(f"{base}_{m}", NAN))
                 and not math.isnan(r.get(f"{other}_{m}", NAN))]
        d = np.array([b - a for a, b in pairs], float)
        out["scenes"] = len(pairs)
        out[f"base_{m}"] = float(np.mean([a for a, _ in pairs])) if pairs else NAN
        out[f"other_{m}"] = float(np.mean([b for _, b in pairs])) if pairs else NAN
        out[f"gap_{m}"] = float(d.mean()) if len(d) else NAN
        out[f"gap_{m}_ci95"] = float(1.96 * d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else NAN
        out[f"abs_gap_{m}"] = float(np.abs(d).mean()) if len(d) else NAN
    return out


def block(rows: list) -> dict:
    """Numeric rows -> the numbers of one table row (DESIGN_GAP.md §1-§2)."""
    ok = [r for r in rows if r["parse_ok"] == 1.0]
    failed = [r for r in rows if r["parse_ok"] != 1.0]
    out = {"scenes": len(rows), "parsed_scenes": len(ok), "failed_parses": len(failed)}
    for m in GAP_METRICS:
        d = np.array([r[f"gap_{m}"] for r in ok], float)
        out[f"true_{m}"] = _mean([r[f"true_{m}"] for r in ok])
        out[f"parsed_{m}"] = _mean([r[f"parsed_{m}"] for r in ok])
        out[f"gap_{m}"] = float(d.mean()) if len(d) else NAN
        out[f"gap_{m}_ci95"] = float(1.96 * d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else NAN
        out[f"abs_gap_{m}"] = float(np.abs(d).mean()) if len(d) else NAN
        out[f"true_all_{m}"] = _mean([r[f"true_{m}"] for r in rows])
    out["true_success_on_failed"] = _mean([r["true_success"] for r in failed])
    out["end_to_end_success"] = (float(sum(r["parsed_success"] for r in ok) / len(rows))
                                 if rows else NAN)
    out["orca"] = arm_stats(ok, "orca_true", "orca_parsed")
    out["map"] = arm_stats(ok, "orca_true", "orca_wrongmap")
    ran = [r for r in ok if not math.isnan(r.get("orca_wrongmap_success", NAN))]
    out["map"]["policy_true_success"] = _mean([r["true_success"] for r in ran])
    out["map"]["policy_true_collision"] = _mean([r["true_collision"] for r in ran])
    out["map"]["policy_true_timeout"] = _mean([r["true_timeout"] for r in ran])
    out["map"]["no_path_scenes"] = sum(r["wrongmap_no_path_agents"] > 0 for r in ran)
    out["map"]["no_path_step_scenes"] = sum(r["wrongmap_no_path_steps"] > 0 for r in ran)
    out["orca_errors"] = sum(bool(r.get("orca_error")) for r in rows)
    return out


def count_err_label(err) -> str:
    e = int(err)
    return "0" if e == 0 else ("<= -2" if e <= -2 else (">= +2" if e >= 2 else f"{e:+d}"))


COUNT_ERR_ORDER = ["<= -2", "-1", "0", "+1", ">= +2"]


def metrics_of(rows: list, sets: list) -> dict:
    ok = [r for r in rows if r["parse_ok"] == 1.0]
    by = lambda sel, key, keys: {k: block([r for r in sel if key(r) == k]) for k in keys}  # noqa: E731
    return {"overall": block(rows),
            "per_set": by(rows, lambda r: r["set"], [s for s in sets
                                                     if any(r["set"] == s for r in rows)]),
            "per_layout": by(rows, lambda r: r["layout"], sorted({r["layout"] for r in rows})),
            "scene_usable": by(ok, lambda r: "yes" if r["scene_usable"] == 1.0 else "no",
                               [k for k in ("yes", "no")
                                if any(("yes" if r["scene_usable"] == 1.0 else "no") == k
                                       for r in ok)]),
            "agent_count_err": by(ok, lambda r: count_err_label(r["agent_count_err"]),
                                  [k for k in COUNT_ERR_ORDER
                                   if any(count_err_label(r["agent_count_err"]) == k
                                          for r in ok)]),
            "no_path": by(ok, lambda r: "no path at start" if r.get("wrongmap_no_path_agents", 0) > 0
                          else "path at start", ["path at start", "no path at start"])
            if any(r.get("wrongmap_no_path_agents", 0) > 0 for r in ok) else {}}


# ------------------------------------------------------------------- tables

def f3(v) -> str:
    return "-" if v is None or math.isnan(v) else f"{v:.3f}"


def signed(v, ci=None) -> str:
    if v is None or math.isnan(v):
        return "-"
    return f"{v:+.3f}" + ("" if ci is None or math.isnan(ci) else f" ± {ci:.3f}")


def two_arm_table(blocks: dict, first: str) -> list:
    """Arm 1: policy and orca_path, each true vs parsed, row by row."""
    lines = [f"| {first} | controller | scenes | true S | parsed S | gap S (95% CI) | mean abs gap S | "
             "true C | parsed C | gap C | true T | parsed T | gap T |", "|" + "---|" * 13]
    for name, b in blocks.items():
        o = b["orca"]
        lines.append(
            f"| {name} | policy | {b['parsed_scenes']} | {f3(b['true_success'])} | {f3(b['parsed_success'])} | "
            f"{signed(b['gap_success'], b['gap_success_ci95'])} | {f3(b['abs_gap_success'])} | "
            f"{f3(b['true_collision'])} | {f3(b['parsed_collision'])} | {signed(b['gap_collision'])} | "
            f"{f3(b['true_timeout'])} | {f3(b['parsed_timeout'])} | {signed(b['gap_timeout'])} |")
        lines.append(
            f"| {name} | orca_path | {o['scenes']} | {f3(o['base_success'])} | {f3(o['other_success'])} | "
            f"{signed(o['gap_success'], o['gap_success_ci95'])} | {f3(o['abs_gap_success'])} | "
            f"{f3(o['base_collision'])} | {f3(o['other_collision'])} | {signed(o['gap_collision'])} | "
            f"{f3(o['base_timeout'])} | {f3(o['other_timeout'])} | {signed(o['gap_timeout'])} |")
    return lines


def wrong_map_table(blocks: dict, first: str) -> list:
    """Arm 2: everything happens in the true scene; only orca_path's map differs."""
    lines = [f"| {first} | scenes | policy S (no map) | ORCA right map S | ORCA wrong map S | "
             "map cost S (95% CI) | mean abs S | ORCA right C | wrong C | of which static | "
             "ORCA right T | wrong T | no path at start |", "|" + "---|" * 13]
    for name, b in blocks.items():
        o = b["map"]
        lines.append(
            f"| {name} | {o['scenes']} | {f3(o['policy_true_success'])} | {f3(o['base_success'])} | "
            f"{f3(o['other_success'])} | {signed(o['gap_success'], o['gap_success_ci95'])} | "
            f"{f3(o['abs_gap_success'])} | {f3(o['base_collision'])} | {f3(o['other_collision'])} | "
            f"{f3(o['other_static_coll'])} | {f3(o['base_timeout'])} | {f3(o['other_timeout'])} | "
            f"{o['no_path_scenes']} |")
    return lines


def write_table(path: Path, m: dict, rows: list, identity: dict, trained: list):
    sets_and_all = {**m["per_set"], "overall": m["overall"]}
    layouts = {(k if k in trained else f"{k} *"): v for k, v in m["per_layout"].items()}
    failed = [r for r in rows if r["parse_ok"] != 1.0]
    reasons = {}
    for r in failed:
        reasons[r["parse_reason"]] = reasons.get(r["parse_reason"], 0) + 1
    usable = {f"usable: {k}": v for k, v in m["scene_usable"].items()}
    counts = {f"agents {k}": v for k, v in m["agent_count_err"].items()}
    lines = [f"# Behaviour gap: {identity['policy']} on true vs parsed scenes", "",
             f"Perception `{identity['perception_ckpt']}`. One deterministic episode per scene. "
             "S = success, C = collision, T = timeout, as rates over a scene's agents, averaged "
             "over scenes. gap = parsed - true, on the scenes with a parse (DESIGN_GAP.md).", "",
             "## Arm 1: gap per set, each controller in the true and in the parsed world", ""]
    lines += two_arm_table(sets_and_all, "set")
    lines += ["", "`overall` pools the sets unweighted; `test_sketch7` is the headline, "
              "the others are probes of hard cases.", "",
              "## Arm 2: orca_path with the parsed map, in the true world", "",
              "Physics, agents and goals are the true scene's, also for the policy, which reads "
              "the true world through its rays and uses no map. Map cost = ORCA wrong map - ORCA "
              "right map. This tests map errors only, not goal or exit errors. `no path at start` "
              "= scenes where at least one agent had no finite path on the parsed map "
              "(it then follows the straight bearing).", ""]
    lines += wrong_map_table({**sets_and_all, **usable, **m["no_path"]}, "group")
    lines += ["", "## Failed parses", "",
              "End-to-end S counts a failed parse as 0 success.", "",
              "| set | scenes | failed | failed rate | true S, all scenes | true S, failed scenes | "
              "end-to-end S |", "|" + "---|" * 7]
    for name, b in sets_and_all.items():
        lines.append(f"| {name} | {b['scenes']} | {b['failed_parses']} | "
                     f"{f3(b['failed_parses'] / b['scenes'])} | {f3(b['true_all_success'])} | "
                     f"{f3(b['true_success_on_failed'])} | {f3(b['end_to_end_success'])} |")
    if reasons:
        lines += [""] + [f"- {n} x `{k}`" for k, n in sorted(reasons.items(), key=lambda x: -x[1])]
    lines += ["", "## Gap by perception error", "",
              "`usable` = perception's `scene_usable`; `agents` = parsed minus true agent count.", ""]
    lines += two_arm_table({**usable, **counts}, "group")
    lines += ["", "## Gap per layout", "",
              f"`*` = not among the policy's training layouts. `crossing`: true scenes give each "
              f"agent its own goal, parsed scenes any exit "
              f"({sum(int(r['per_agent_goals']) for r in rows)} scenes with per-agent goals).", ""]
    lines += two_arm_table(layouts, "layout")
    lines += ["", "### Arm 2 per layout", ""]
    lines += wrong_map_table(layouts, "layout")
    differs = sum(r["parse_differs"] == 1.0 for r in rows)
    lines += ["", f"Check: `predict.py` and `evaluate.py` disagree on {differs} parse(s) (expected 0).",
              f"Check: {m['overall']['orca_errors']} scene(s) with an error in an ORCA run (expected 0)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- figure

def quantile_bins(rows: list, key: str, k: int = 5) -> list:
    """Up to ``k`` near-equal-count bins of ``key``: [(label, median, rows)]."""
    sel = sorted((r for r in rows if not math.isnan(r[key])), key=lambda r: r[key])
    parts = [p for p in np.array_split(np.arange(len(sel)), min(k, len(sel))) if len(p)] if sel else []
    return [(f"{sel[p[0]][key]:.2f}-{sel[p[-1]][key]:.2f}",
             float(np.median([sel[i][key] for i in p])), [sel[i] for i in p]) for p in parts]


def write_figure(path: Path, rows: list, title: str, ylabel: str, base: str, other: str,
                 count_panel: bool):
    """Mean ``other - base`` success (with CI) and its mean absolute value, by perception error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["parse_ok"] == 1.0]
    stats = lambda sel: arm_stats(sel, base, other)  # noqa: E731
    usable = lambda r: "yes" if r["scene_usable"] == 1.0 else "no"  # noqa: E731
    count = lambda r: count_err_label(r["agent_count_err"])  # noqa: E731
    groups = lambda key, order: [(k, stats([r for r in ok if key(r) == k])) for k in order  # noqa: E731
                                 if any(key(r) == k for r in ok)]
    panels = [("wall IoU of the parse (5 equal-count bins)",
               [(lab, stats(sel)) for lab, _, sel in quantile_bins(ok, "wall_iou")])]
    if count_panel:
        panels.append(("parsed minus true agent count", groups(count, COUNT_ERR_ORDER)))
    panels.append(("scene_usable", groups(usable, ["yes", "no"])))
    ratios = [5, 5, 2] if count_panel else [5, 2]
    fig, axes = plt.subplots(1, len(panels), figsize=(12 if count_panel else 8, 3.8), sharey=True,
                             gridspec_kw={"width_ratios": ratios})
    for ax, (xlabel, data) in zip(axes, panels):
        x = np.arange(len(data))
        ax.axhline(0, color="#555555", lw=1)
        ax.errorbar(x, [b["gap_success"] for _, b in data],
                    yerr=[b["gap_success_ci95"] for _, b in data], fmt="o", color="#1f77b4", ms=6,
                    lw=1.5, capsize=3, label="mean gap (95% CI)")
        ax.plot(x, [b["abs_gap_success"] for _, b in data], "D", mfc="none", color="#ff7f0e",
                ms=6, label="mean |gap|")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\nn={b['scenes']}" for lab, b in data], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim(-0.6, max(len(data) - 0.4, 0.6))
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main

def perception_rows(root: Path, set_name: str, limit) -> list:
    p = root / set_name / "per_scene.csv"
    if not p.exists():
        raise SystemExit(f"{p} is missing: run scripts/run_gap.py (it makes perception's outputs)")
    with open(p, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["set"] == set_name]
    return rows[:limit] if limit else rows


def check_physics(data_root: Path, sets: list, sim_cfg):
    """Stop if a set was generated with other physics than the policy runs on."""
    for s in sets:
        p = data_root / s / "dataset.json"
        cfg = json.loads(p.read_text(encoding="utf-8")).get("config", {}) if p.exists() else {}
        diff = {k: (cfg[k], getattr(sim_cfg, k)) for k in SIM_KEYS
                if k in cfg and cfg[k] != getattr(sim_cfg, k)}
        if diff:
            raise SystemExit(f"{s}: physics differ from the checkpoint's (set, policy): {diff}")


def identity_of(ckpt: Path, perception: Path, sets: list) -> dict:
    meta = json.loads((perception / sets[0] / "metrics.json").read_text(encoding="utf-8"))["_meta"]
    return {"policy": ckpt.resolve().parent.name,
            "policy_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
            "deterministic": True,
            "arms": ["policy_true", "policy_parsed", "orca_path_true", "orca_path_parsed",
                     "orca_path_wrong_map"],
            "orca_sha256": hashlib.sha256((Path(__file__).resolve().parent / "orca.py").read_bytes()
                                          .replace(b"\r\n", b"\n")).hexdigest()[:16],
            "perception_ckpt": Path(meta["ckpt"].replace("\\", "/")).parent.name,
            "extract": meta["extract"]}


def check_identity(out: Path, identity: dict):
    p = out / "identity.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")) != identity:
        raise SystemExit(f"{out} holds a run of another policy or parse: use another --out "
                         f"or delete that folder")
    p.write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--parsed", type=Path, required=True)
    ap.add_argument("--perception", type=Path, required=True)
    ap.add_argument("--sets", default="test_sketch7,test_sketch7_free,test_sketch7_tilt,test_sketch7_lines")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N scenes per set (smoke runs)")
    a = ap.parse_args(argv)
    sets = a.sets.split(",")

    _, run_cfg, trained, _ = load_policy(a.ckpt, True, 0)
    check_physics(a.data_root, sets, default_reward(trained))
    identity = identity_of(a.ckpt, a.perception, sets)
    a.out.mkdir(parents=True, exist_ok=True)
    check_identity(a.out, identity)

    csv_path = a.out / "per_scene.csv"
    kept = read_rows(csv_path)
    write_rows(csv_path, kept)                 # drops a line cut off by an interruption
    done = {(r["set"], r["scene"]) for r in kept}
    wanted, jobs = [], []
    for s in sets:
        for prow in perception_rows(a.perception, s, a.limit):
            stem = prow["scene"]
            wanted.append((s, stem))
            if (s, stem) in done:
                continue
            parsed = a.parsed / s / f"{stem}.json"
            if not parsed.exists():
                raise SystemExit(f"{parsed} is missing: run scripts/run_gap.py (it parses the set)")
            jobs.append((s, stem, str(a.data_root / s / f"{stem}.json"), str(parsed), prow))
    print(f"{len(wanted)} scenes, {len(wanted) - len(jobs)} already done, {len(jobs)} to run", flush=True)

    if jobs:
        t0 = time.time()
        pool = (mp.get_context("spawn").Pool(a.workers, init_worker, (str(a.ckpt),))
                if a.workers > 0 else None)
        if pool is None:
            init_worker(a.ckpt)
        results = pool.imap_unordered(run_scene, jobs, chunksize=4) if pool else map(run_scene, jobs)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=COLUMNS)
                for i, row in enumerate(results, 1):
                    w.writerow({k: cell(row.get(k, "")) for k in COLUMNS})
                    fh.flush()
                    if i % 100 == 0 or i == len(jobs):
                        rate = (time.time() - t0) / i
                        print(f"  {i}/{len(jobs)}  {rate:.2f} s per scene, "
                              f"{rate * (len(jobs) - i) / 60:.0f} min left", flush=True)
        finally:
            if pool:
                pool.terminate()
                pool.join()

    order = {k: i for i, k in enumerate(wanted)}
    all_rows = read_rows(csv_path)
    all_rows.sort(key=lambda r: order.get((r["set"], r["scene"]), len(order)))
    write_rows(csv_path, all_rows)
    rows = numeric([r for r in all_rows if (r["set"], r["scene"]) in order])
    m = metrics_of(rows, sets)
    (a.out / "metrics.json").write_text(json.dumps({"identity": identity, **m}, indent=1) + "\n",
                                        encoding="utf-8")
    write_table(a.out / "gap.md", m, rows, identity, list(run_cfg.env.layouts))
    n_ok = m["overall"]["parsed_scenes"]
    write_figure(a.out / "fig_gap.png", rows,
                 f"{identity['policy']} on parsed vs true scenes ({n_ok} parsed)",
                 "success: parsed - true", "true", "parsed", True)
    write_figure(a.out / "fig_gap_map.png", rows,
                 f"orca_path with the parsed map, in the true world ({n_ok} parsed)",
                 "success: wrong map - right map", "orca_true", "orca_wrongmap", False)
    print(f"\n{'set':22s} {'scenes':>6s} {'failed':>6s} {'true S':>7s} {'parsed S':>8s} {'gap S':>7s} "
          f"{'ORCA gap S':>10s} {'map cost S':>10s}")
    for name, b in {**m["per_set"], "overall": m["overall"]}.items():
        print(f"{name:22s} {b['scenes']:6d} {b['failed_parses']:6d} {f3(b['true_success']):>7s} "
              f"{f3(b['parsed_success']):>8s} {signed(b['gap_success']):>7s} "
              f"{signed(b['orca']['gap_success']):>10s} {signed(b['map']['gap_success']):>10s}")
    print(f"-> {a.out / 'gap.md'}")


if __name__ == "__main__":
    main()
