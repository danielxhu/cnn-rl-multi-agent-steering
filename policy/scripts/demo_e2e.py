"""End-to-end demo: a hand-drawn scene -> perception parse -> the learned policy in the simulator.

    python scripts/demo_e2e.py photo.jpg --policy models/e3_both_seed0/ckpt_best.pt
    python scripts/demo_e2e.py ../perception/results/drawings_v7/*.json --policy CKPT --gif
    python scripts/demo_e2e.py ../perception/results/drawings_v7/*.json          # greedy controller

Each input is either a photo of a drawing (parsed by ``perception/scripts/demo.py``
with the newest model in ``perception/models``) or a scene JSON that perception
already wrote. The scene is loaded with ``Scene.from_dict``, one deterministic
episode is run with ``evaluate.run_episode`` and ``<out>/<name>_e2e.png`` is written:
photo | parsed scene | trajectories (plus ``<name>_e2e.gif`` with ``--gif``).
Several inputs also give ``<out>/summary.md`` and ``summary.csv``.

Invariants:
- Nothing in ``sim/``, ``perception/`` or the training code changes; any
  mismatch between perception's JSON and the simulator's scene is fixed here
  (``to_scene``), and every fix is reported.
- The simulator runs with the checkpoint's own physics and the default reward,
  exactly as ``evaluate.py`` does; without ``--policy`` the greedy controller
  runs on the default ``Config``.
- CPU only, no process pool, no shell: runs the same on Windows and macOS.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from config import Config  # noqa: E402
from evaluate import GreedyController, load_policy, run_episode  # noqa: E402
from physics import Simulator  # noqa: E402
from policyconfig import default_reward  # noqa: E402
from render import render  # noqa: E402
from scene import Scene  # noqa: E402

REPO = POLICY.parent
PERCEPTION = REPO / "perception"
# where perception keeps the drawings it was tested on (photo of a parsed JSON)
PHOTO_DIRS = [PERCEPTION / "results" / d / "input"
              for d in ("drawings_meeting", "drawings_v5", "old_drawings_v4")]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp"}
OUTCOME_COLOR = {"goal": "#2ca02c", "collision": "#d62728", "timeout": "#ff7f0e", None: "#ff7f0e"}
AGENT_COLORS = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
                "#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728"]


# ------------------------------------------------------------- inputs

def parse_photo(photo: Path, out: Path) -> Path:
    """Run perception's demo on a photo; return the scene JSON it wrote."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(PERCEPTION / "scripts" / "demo.py"), str(photo),
                    "--out", str(out), "--no-open"], cwd=PERCEPTION, check=True)
    js = out / f"{photo.stem}.json"
    if not js.exists():                      # demo.py converts HEIC to <stem>.jpg first
        raise SystemExit(f"{photo}: perception wrote no scene JSON")
    return js


def find_photo(js: Path) -> Path | None:
    """The photo a parsed JSON came from: same stem in perception's drawing folders."""
    for d in [js.parent] + PHOTO_DIRS:
        for q in sorted(d.glob(js.stem + ".*")) if d.exists() else []:
            if q.suffix.lower() in IMAGE_EXT and not q.stem.endswith(("_clean", "_overlay")):
                return q
    return None


def to_scene(rec: dict, sim_cfg) -> tuple[Scene, list]:
    """Perception's scene JSON -> a simulator ``Scene``; returns (scene, notes on every fix)."""
    notes = []
    d = dict(rec)
    if d.get("parsed") is False:
        raise ValueError("perception marked this drawing as not parsed")
    w = (d.get("world") or {}).get("size", sim_cfg.world)
    if abs(w - sim_cfg.world) > 1e-9:          # never happens with the v7 model; kept for safety
        k = sim_cfg.world / w
        sc = lambda p: [p[0] * k, p[1] * k]
        box = lambda b: [v * k for v in b]
        d["walls"] = [[sc(a), sc(b)] for a, b in d["walls"]]
        d["obstacles"] = [{**o, "c": sc(o["c"]), "r": o["r"] * k} for o in d["obstacles"]]
        d["agents"] = [{**a, "pos": sc(a["pos"])} for a in d["agents"]]
        for key in ("start_region", "goal_region"):
            if d.get(key):
                d[key] = box(d[key])
        for key in ("starts", "exits"):
            if d.get(key):
                d[key] = [box(b) for b in d[key]]
        notes.append(f"world {w:g} rescaled to {sim_cfg.world:g}")
    r = (d.get("world") or {}).get("agent_radius")
    if r is not None and abs(r - sim_cfg.agent_radius) > 1e-9:
        notes.append(f"drawn agent radius {r:g}, simulated at {sim_cfg.agent_radius:g}")
    if not d.get("layout"):
        d["layout"] = "drawing"                # perception leaves it None; metrics key on it
    if not d.get("exits") and not d.get("goal_region"):
        raise ValueError("no exit in the parsed scene")
    if not d.get("agents"):
        raise ValueError("no agent in the parsed scene")
    return Scene.from_dict(d), notes


def start_check(scene: Scene, sim_cfg) -> list:
    """Agents that start already touching a wall / obstacle or another agent."""
    sim = Simulator(scene, sim_cfg)
    return [f"agent {i} starts {'on a wall/obstacle' if sim._hits_static(i) else 'touching an agent'}"
            for i in range(len(scene.agents)) if sim._hits_static(i) or sim._hits_agent(i)]


# ------------------------------------------------------------- figures

def to_px(xy, sim_cfg):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return xy[:, 0] * sim_cfg.scale, (sim_cfg.world - xy[:, 1]) * sim_cfg.scale


def figure(photo, scene, rec, sim_cfg, title, path):
    """photo | parsed scene | trajectories, one row."""
    bg = np.asarray(render(scene, sim_cfg)[0])
    n = 3 if photo is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0))
    for ax in axes:
        ax.axis("off")
    k = 0
    if photo is not None:
        axes[0].imshow(Image.open(photo).convert("RGB"))
        axes[0].set_title("hand-drawn photo", fontsize=11)
        k = 1
    axes[k].imshow(bg)
    axes[k].set_title("parsed scene", fontsize=11)
    ax = axes[k + 1]
    ax.imshow(bg, alpha=0.55)
    for i, (tr, out) in enumerate(zip(rec["trajectories"], rec["outcomes"])):
        x, y = to_px(tr, sim_cfg)
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(x, y, color=c, lw=1.8, alpha=0.95)
        ax.plot(x[0], y[0], "o", color=c, ms=5)
        mk = {"goal": "*", "collision": "X"}.get(out, "s")
        ax.plot(x[-1], y[-1], mk, color=OUTCOME_COLOR[out], ms=11 if mk == "*" else 8,
                mec="black", mew=0.6)
    ax.set_title("policy rollout", fontsize=11)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", color="#555", label="start"),
               Line2D([], [], ls="", marker="*", color=OUTCOME_COLOR["goal"], mec="black", ms=11,
                      label="reached exit"),
               Line2D([], [], ls="", marker="X", color=OUTCOME_COLOR["collision"], mec="black",
                      ms=8, label="collision"),
               Line2D([], [], ls="", marker="s", color=OUTCOME_COLOR["timeout"], mec="black", ms=7,
                      label="timeout")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gif(scene, rec, sim_cfg, path, every=2, ms=60):
    """Agents moving over the parsed scene; an agent stays where it finished."""
    bg = render(scene, sim_cfg)[0].convert("RGB")
    trs = [np.asarray(t, float) for t in rec["trajectories"]]
    T = max(len(t) for t in trs)
    rad = sim_cfg.agent_radius * sim_cfg.scale
    frames = []
    for s in list(range(0, T, every)) + [T - 1]:
        im = bg.copy()
        d = ImageDraw.Draw(im)
        for i, t in enumerate(trs):
            x, y = to_px(t[:min(s, len(t) - 1) + 1], sim_cfg)
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            if len(x) > 1:
                d.line(list(zip(x, y)), fill=c, width=2)
            done = s >= len(t) - 1 and rec["outcomes"][i] is not None
            fill = OUTCOME_COLOR[rec["outcomes"][i]] if done else c
            d.ellipse([x[-1] - rad, y[-1] - rad, x[-1] + rad, y[-1] + rad], fill=fill,
                      outline="black")
        d.text((6, 6), f"step {s}", fill=(0, 0, 0))
        frames.append(im)
    frames += [frames[-1]] * 15                 # hold the last frame
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path, help="photos and/or perception scene JSONs")
    ap.add_argument("--policy", type=Path, default=None,
                    help="checkpoint (ckpt_best.pt); omitted: the greedy controller")
    ap.add_argument("--photo", type=Path, default=None,
                    help="the photo of a single JSON input (default: looked up by name)")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "demo_e2e")
    ap.add_argument("--gif", action="store_true", help="also write <name>_e2e.gif")
    ap.add_argument("--stochastic", action="store_true", help="sample actions (seed --seed)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.photo is not None and len(a.inputs) != 1:
        ap.error("--photo goes with a single input")
    a.out.mkdir(parents=True, exist_ok=True)

    if a.policy is not None:
        ctrl, run_cfg, trained, ck = load_policy(a.policy, not a.stochastic, a.seed)
        sim_cfg = default_reward(trained)
        max_agents = run_cfg.env.max_agents
        cname = f"{a.policy.parent.name} ({run_cfg.net.memory}, {ck.get('agent_steps', 0):,} steps)"
    else:
        ctrl, sim_cfg, max_agents, cname = GreedyController(), Config(), None, "greedy"
    print(f"controller: {cname}")

    rows = []
    for inp in a.inputs:
        inp = inp.resolve()
        if inp.suffix.lower() == ".json":
            js, photo = inp, (a.photo or find_photo(inp))
        else:
            js, photo = parse_photo(inp, a.out / "parsed"), inp
        name = js.stem
        row = {"drawing": name, "agents": 0, "starts": 0, "exits": 0, "inner_walls": 0,
               "obstacles": 0, "success": float("nan"), "collision": float("nan"),
               "static_coll": float("nan"), "agent_coll": float("nan"),
               "timeout": float("nan"), "steps": 0, "notes": ""}
        try:
            rec_in = json.loads(js.read_text())
            scene, notes = to_scene(rec_in, sim_cfg)
        except (ValueError, KeyError, TypeError) as e:
            row["notes"] = f"not loaded: {e}"
            print(f"{name}: {row['notes']}")
            rows.append(row)
            continue
        if max_agents is not None and len(scene.agents) > max_agents:
            notes.append(f"{len(scene.agents)} agents > {max_agents} trained")
        if len(scene.exit_boxes()) > 1 or len(scene.start_boxes()) > 1:
            notes.append(f"{len(scene.start_boxes())} starts / {len(scene.exit_boxes())} exits")
        notes += start_check(scene, sim_cfg)
        res = run_episode(scene, ctrl, sim_cfg, record=True)
        steps = max(len(t) for t in res["trajectories"]) - 1
        row.update(agents=len(scene.agents), starts=len(scene.start_boxes()),
                   exits=len(scene.exit_boxes()), inner_walls=len(scene.walls) - 4,
                   obstacles=len(scene.obstacles), success=res["success_rate"],
                   collision=res["collision_rate"], static_coll=res["static_collision_rate"],
                   agent_coll=res["agent_collision_rate"], timeout=res["timeout_rate"],
                   steps=steps, notes="; ".join(notes))
        n = len(scene.agents)
        title = (f"{name[:40]}   |   {cname}\n"
                 f"{round(res['success_rate'] * n)}/{n} reached an exit, "
                 f"{round(res['collision_rate'] * n)} collided, "
                 f"{round(res['timeout_rate'] * n)} timed out  ({steps} steps)")
        figure(photo, scene, res, sim_cfg, title, a.out / f"{name}_e2e.png")
        if a.gif:
            gif(scene, res, sim_cfg, a.out / f"{name}_e2e.gif")
        print(f"{name[:40]:40s} agents {n}  success {res['success_rate']:.2f}  "
              f"collision {res['collision_rate']:.2f}  timeout {res['timeout_rate']:.2f}"
              + (f"   [{row['notes']}]" if row["notes"] else ""))
        rows.append(row)

    if len(rows) > 1:
        write_summary(rows, cname, a.out)


def write_summary(rows, cname, out: Path):
    cols = list(rows[0])
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) and not math.isnan(v) else (
        "-" if isinstance(v, float) else str(v))
    ok = [r for r in rows if not math.isnan(r["success"])]
    lines = [f"# End-to-end demo: drawings -> {cname}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    if ok:
        tot = sum(r["agents"] for r in ok)
        lines.append("")
        lines += [f"- {k}, pooled over {tot} agents: {sum(r[k] * r['agents'] for r in ok) / tot:.3f}"
                  for k in ("success", "collision", "timeout")]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nsummary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
