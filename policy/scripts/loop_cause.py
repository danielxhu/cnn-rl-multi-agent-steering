"""What is around an agent while it loops? Classifies every loop window.

    python scripts/loop_cause.py --oracle downhill --suites test4,novel --workers 9

e7_map + orbit rule, no filter, 3 seeds; ``--oracle downhill`` gives the
network the true direction (``oracle_dir.py``), so the loops left are the
network's own (DESIGN_E7 §14.1); ``--oracle none`` is the plain policy.

For the first loop window of every looping agent (``loop_bearing.loop_window``,
360 deg of heading change):
- ``agent``: another active agent came within ``NEAR_AGENT`` during it;
- ``obstacle``: the window winds once or more round a round obstacle's
  centre (it circled an obstacle);
- ``wall``: else, the agent was within ``NEAR_WALL`` of a static surface
  for at least half of the window;
- ``open``: none of these.
(first match in this order). Control: for every agent that does not loop,
the same ``agent`` test on a window of ``CONTROL_LEN`` steps in the middle of
its trajectory (the base rate of meeting someone). Also per class: outcome of the agent and the
window's length in steps. Writes ``results/loop_cause/<oracle>.json``.
Invariants: read-only on models and suites; uses privileged state only to
classify, never to act (except the oracle direction).
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(POLICY / "scripts"))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

NEAR_AGENT = 7.0      # 2R + 3
NEAR_WALL = 4.0       # R + 2
CONTROL_LEN = 48      # about the median loop window
CLASSES = ("agent", "obstacle", "wall", "open")


def winding(path, c):
    d = path - np.asarray(c, float)
    a = np.arctan2(d[:, 1], d[:, 0])
    return abs(float(np.sum((np.diff(a) + math.pi) % (2 * math.pi) - math.pi))) / (2 * math.pi)


def _chunk(args):
    ckpt, oracle, items, sim_dict = args
    from evaluate import LOOP_TURN, load_policy, run_episode, sim_config_from_dict, turning
    from loop_bearing import loop_window
    from oracle_dir import oracle_act
    from reachability import clearance, occupancy
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl, _, _, _ = load_policy(ckpt)
    ctrl.exit_rule = "orbit"
    ctrl.oracle = oracle
    if oracle != "none":
        ctrl.act = types.MethodType(oracle_act, ctrl)
    out = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        T = [np.asarray(t, float) for t in r["trajectories"]]
        clear = None
        for i, t in enumerate(T):
            if turning(t.tolist()) < LOOP_TURN:
                if len(t) > CONTROL_LEN:
                    s = (len(t) - CONTROL_LEN) // 2
                    near = any(math.dist(t[k], T[j][k]) < NEAR_AGENT for j in range(len(T)) if j != i
                               for k in range(s, min(s + CONTROL_LEN + 1, len(T[j]))))
                    out.append(dict(scene=name, layout=sc.layout, agent=i, cls="control",
                                    near=bool(near), outcome=r["outcomes"][i], n_agents=len(T)))
                continue
            w = loop_window(t)
            if w is None:
                continue
            s, e = w
            seg = t[s:e + 1]
            others = [T[j] for j in range(len(T)) if j != i]
            dmin = min((math.dist(t[k], o[k]) for o in others for k in range(s, min(e + 1, len(o)))),
                       default=np.inf)
            if dmin < NEAR_AGENT:
                cls = "agent"
            elif any(winding(seg, o.c) >= 1.0 for o in sc.obstacles):
                cls = "obstacle"
            else:
                if clear is None:
                    clear = clearance(cfg, occupancy(cfg, sc))
                m = clear.shape[0]
                iy = np.clip((seg[:, 1] / cfg.cell).astype(int), 0, m - 1)
                ix = np.clip((seg[:, 0] / cfg.cell).astype(int), 0, m - 1)
                near = clear[iy, ix] < NEAR_WALL
                cls = "wall" if near.mean() >= 0.5 else "open"
            out.append(dict(scene=name, layout=sc.layout, agent=i, cls=cls,
                            outcome=r["outcomes"][i], length=int(e - s), start=int(s),
                            n_agents=len(T)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=["none", "downhill"], default="downhill")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--suites", default="test4,novel")
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "loop_cause")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from dataclasses import asdict, replace
    from evaluate import default_reward, sim_config_from_dict
    from suites import load_suite
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers else None
    rows = []
    try:
        for s in a.suites.split(","):
            sdir = a.data_root / s
            cfg = default_reward(sim_config_from_dict(json.loads((sdir / "suite.json").read_text())["sim"]))
            cfg = replace(cfg, progress_metric="geodesic")      # the oracle's field
            items = [(n, sc.as_dict()) for n, sc in load_suite(sdir)][:a.limit]
            kk = max(1, a.workers) * 4
            for sd in a.seeds.split(","):
                ck = str(POLICY / "models" / f"e7_map_seed{sd}" / "ckpt_best.pt")
                jobs = [(ck, a.oracle, items[j::kk], asdict(cfg)) for j in range(kk) if items[j::kk]]
                parts = pool.map(_chunk, jobs) if pool else [_chunk(j) for j in jobs]
                rows += [dict(r, suite=s, seed=sd) for p in parts for r in p]
    finally:
        if pool:
            pool.close()
            pool.join()
    res = {}
    for s in a.suites.split(","):
        groups = {s: [r for r in rows if r["suite"] == s]}
        if s == "novel":
            groups["novel ordinary"] = [r for r in groups[s]
                                        if r["layout"] not in ("u_trap", "blocked_exit")]
        for g, rr in groups.items():
            ctl = [r for r in rr if r["cls"] == "control"]
            rr = [r for r in rr if r["cls"] != "control"]
            n = len(rr)
            res[g] = {"loop windows": n,
                      "control: another agent within NEAR_AGENT %":
                          round(100 * float(np.mean([r["near"] for r in ctl])), 1) if ctl else None} | {
                c: {"share %": round(100 * sum(r["cls"] == c for r in rr) / max(1, n), 1),
                    "goal %": round(100 * float(np.mean([r["outcome"] == "goal" for r in rr if r["cls"] == c])), 1)
                    if any(r["cls"] == c for r in rr) else None,
                    "median length": float(np.median([r["length"] for r in rr if r["cls"] == c]))
                    if any(r["cls"] == c for r in rr) else None}
                for c in CLASSES}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.oracle}.json").write_text(json.dumps({"summary": res, "windows": rows}))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
