"""Inside loops: does the agent follow the wall map's downhill direction, and does that direction flip?

    python scripts/loop_downhill.py --model models/e7_map_seed0/ckpt_best.pt --suite data/suites/test4 --workers 9

Per active step of every agent: ``follow`` = |downhill direction relative to
the heading| (deg; the policy input), ``flip`` = change of the downhill world
direction since the previous step > 90 deg. Compared inside loop windows
(``loop_bearing.loop_window``), elsewhere on looping agents, and on agents
that never loop. Follows the downhill & no flips -> the map/field leads the
agent round; ignores it -> the network's driving.

Invariants: read-only; episodes as ``evaluate.py`` (``--exit-rule orbit``).
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

import numpy as np  # noqa: E402


def _chunk(args):
    ckpt, items = args
    from evaluate import LOOP_TURN, load_policy, run_episode, turning
    from loop_bearing import loop_window
    from policyconfig import default_reward
    from scene import Scene
    ctrl, _, tr, _ = load_policy(ckpt)
    ctrl.exit_rule = "orbit"
    cfg = default_reward(tr)
    orig_reset = ctrl.reset
    log = {}

    def reset(sim):
        orig_reset(sim)
        wm, obs_fn = ctrl.wm, ctrl.wm.observe
        log.clear()

        def observe(sim, obs):
            out = obs_fn(sim, obs)
            d = obs.shape[1]
            for i in np.flatnonzero(sim.active):
                rel = math.atan2(out[i, d + 1], out[i, d])
                log.setdefault(i, []).append((rel, sim.heading[i] + rel))
            return out
        wm.observe = observe
    ctrl.reset = reset
    res = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        for i, t in enumerate(r["trajectories"]):
            L = np.array(log.get(i, []))
            if len(L) < 2:
                continue
            follow = np.degrees(np.abs(L[:, 0]))
            dw = np.abs((np.diff(L[:, 1]) + math.pi) % (2 * math.pi) - math.pi)
            flip = np.concatenate([[False], dw > math.pi / 2])
            loop = turning(t) >= LOOP_TURN
            w = loop_window(t) if loop else None
            res.append(dict(loop=bool(loop), window=w, follow=follow.round(1).tolist(),
                            flip=flip.tolist()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from suites import load_suite
    items = [(n, s.as_dict()) for n, s in load_suite(a.suite)][:a.limit]
    k = max(1, a.workers) * 4
    jobs = [(a.model, items[i::k]) for i in range(k) if items[i::k]]
    if a.workers:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            parts = pool.map(_chunk, jobs)
    else:
        parts = [_chunk(j) for j in jobs]
    rows = [r for p in parts for r in p]
    groups = {"loop window": ([], []), "loopers, elsewhere": ([], []), "never loop": ([], [])}
    for r in rows:
        f, fl = r["follow"], r["flip"]
        if r["loop"] and r["window"]:
            s, e = r["window"]
            groups["loop window"][0].extend(f[s:e]); groups["loop window"][1].extend(fl[s:e])
            groups["loopers, elsewhere"][0].extend(f[:s] + f[e:])
            groups["loopers, elsewhere"][1].extend(fl[:s] + fl[e:])
        elif not r["loop"]:
            groups["never loop"][0].extend(f); groups["never loop"][1].extend(fl)
    out = {}
    for g, (f, fl) in groups.items():
        f = np.array(f)
        out[g] = {"steps": len(f), "median |downhill| deg": float(np.median(f)),
                  "share |downhill| < 30": float((f < 30).mean()),
                  "share > 90 (downhill behind)": float((f > 90).mean()),
                  "flips per 100 steps": 100 * float(np.mean(fl))}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
