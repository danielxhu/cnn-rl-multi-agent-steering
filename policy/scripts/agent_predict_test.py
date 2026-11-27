"""Would knowing other agents' motion stop agent-agent crashes? Two CPU tests, no training.

    python scripts/agent_predict_test.py filter --suites test4,crowd16 --workers 9
    python scripts/agent_predict_test.py foresee --suites test4,crowd16 --workers 9

``filter``: e7_map + orbit rule + safety filter (margin 0.3), three ways of
treating the other agents inside the filter:
  - ``rays``: the agent ray hit points (the filter as it is);
  - ``disk``: every agent within ``fov_range`` and the 220 deg FOV as a full
    disk at its current position (no occlusion; privileged geometry);
  - ``cv``: as ``disk`` plus disks along its constant-velocity next step
    (its own heading and speed now; privileged velocity);
  - ``est``: ``cv`` from the agent's own sensing only. Agent ray hits are
    grouped into blobs (adjacent sectors, hit points closer than 2R + 0.5);
    a blob's centre is the mean of its hit points pushed R further along
    their rays; the velocity is the difference to the nearest centre of the
    previous step (within ``GATE``), clamped to ``max_speed``; unmatched: 0.
    The ray hit points are kept as well;
  - ``est_ema``: as ``est`` with the velocity smoothed over steps
    (``v = EMA v_new + (1 - EMA) v_old``).
``est``/``est_ema`` also log how far the estimates are from the truth
(position and velocity error of each blob against the nearest real agent).
If ``cv`` removes most agent crashes and ``disk`` does not, motion is the
missing information; if neither does, the filter is not the place to fix it.

``foresee`` (no filter): for every agent-agent crash, was it predictable
``k`` steps before from both agents' current velocities (constant-velocity
closest approach < 2R within the next ``HORIZON`` steps)? And how often does
the same test fire with no crash following (false alarms per 100 agent-steps)?

Writes ``results/agent_predict/<test>.json``. Read-only on models and suites.
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

RING = np.stack([np.cos(np.linspace(0, 2 * math.pi, 12, endpoint=False)),
                 np.sin(np.linspace(0, 2 * math.pi, 12, endpoint=False))], 1)
HORIZON = 5
CV_SAMPLES = (0.0, 0.5, 1.0)
GATE = 7.0
EMA = 0.5
KINDS = ("rays", "disk", "cv")


def make_shield(kind, ctrl, cfg):
    from shield import Shield
    from config import E_AGENT

    class AgentShield(Shield):
        def begin(self, sim):
            super().begin(sim)
            self.prev = {}                              # agent -> [(centre, velocity)]

        def filter(self, sim, obs51, actions):
            self._sim = sim
            return super().filter(sim, obs51, actions)

        def blobs(self, p, h, o):
            """Estimated centres of the agents seen by the agent rays."""
            S, c = self.c.n_sectors, self.c
            R = c.agent_radius
            r = np.asarray(o[E_AGENT * S:(E_AGENT + 1) * S], float)
            ang = h + self.off
            dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
            q = p[None, :] + (r * c.fov_range)[:, None] * dirs
            groups = []
            for k in np.flatnonzero(r < 1.0 - 1e-6):
                if groups and groups[-1][-1] == k - 1 and math.dist(q[k], q[k - 1]) < 2 * R + 0.5:
                    groups[-1].append(k)
                else:
                    groups.append([k])
            return [np.mean(q[g] + R * dirs[g], 0) for g in groups]

        def sensed(self, p, h, o):
            if kind == "rays":
                return super().sensed(p, h, o)
            if kind in ("est", "est_ema"):
                sim = self._sim
                i = int(np.argmin(np.hypot(*(sim.pos - p).T)))
                prev = self.prev.get(i, [])
                now, pts = [], [super().sensed(p, h, o)]
                for cen in self.blobs(p, h, o):
                    v = np.zeros(2)
                    matched = False
                    if prev:
                        d = [math.dist(cen, pc) for pc, _ in prev]
                        k = int(np.argmin(d))
                        if d[k] < GATE:
                            matched = True
                            v = cen - prev[k][0]
                            if kind == "est_ema":
                                v = EMA * v + (1 - EMA) * prev[k][1]
                            sp = math.hypot(*v)
                            if sp > self.c.max_speed:
                                v *= self.c.max_speed / sp
                    now.append((cen, v))
                    for f in CV_SAMPLES:
                        pts.append(cen + f * v + self.c.agent_radius * RING)
                    act = [j for j in np.flatnonzero(sim.active) if j != i]
                    if act:                                      # error log, truth only here
                        dj = [math.dist(cen, sim.pos[j]) for j in act]
                        j = act[int(np.argmin(dj))]
                        tv = sim.speed[j] * np.array([math.cos(sim.heading[j]), math.sin(sim.heading[j])])
                        self.err.append((min(dj), math.dist(v, tv), math.hypot(*tv), matched))
                self.prev[i] = now
                return np.concatenate(pts)
            S = self.c.n_sectors
            o = np.array(o, float)
            o[E_AGENT * S:(E_AGENT + 1) * S] = 1.0          # drop the agent rays
            pts = [super().sensed(p, h, o)]
            sim = self._sim
            i = int(np.argmin(np.hypot(*(sim.pos - p).T)))
            half = math.radians(self.c.fov_deg) / 2
            for j in np.flatnonzero(sim.active):
                if j == i:
                    continue
                d = sim.pos[j] - p
                if math.hypot(*d) > self.c.fov_range + self.c.agent_radius:
                    continue
                if abs((math.atan2(d[1], d[0]) - h + math.pi) % (2 * math.pi) - math.pi) > half:
                    continue
                v = sim.speed[j] * np.array([math.cos(sim.heading[j]), math.sin(sim.heading[j])])
                for f in (CV_SAMPLES if kind == "cv" else (0.0,)):
                    pts.append(sim.pos[j] + f * v + self.c.agent_radius * RING)
            return np.concatenate(pts)
    sh = AgentShield(ctrl, cfg, 0.3)
    sh.err = []
    return sh


def _filter_chunk(args):
    kind, ckpt, items, sim_dict = args
    from evaluate import load_policy, run_episode, sim_config_from_dict
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl, _, _, _ = load_policy(ckpt)
    ctrl.exit_rule = "orbit"
    sh = make_shield(kind, ctrl, cfg)
    out = []
    for name, d in items:
        r = run_episode(Scene.from_dict(d), sh, cfg)
        out.append({k: r[k] for k in ("success_rate", "agent_collision_rate",
                                      "static_collision_rate", "timeout_rate")})
    if out:
        out[0]["_err"] = sh.err
    return out


def _foresee_chunk(args):
    _, ckpt, items, sim_dict = args
    from evaluate import load_policy, run_episode, sim_config_from_dict
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl, _, _, _ = load_policy(ckpt)
    ctrl.exit_rule = "orbit"
    R2 = 2 * cfg.agent_radius
    crashes = {k: [] for k in range(1, HORIZON + 1)}
    alarms = steps = 0
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        T = [np.asarray(t, float) for t in r["trajectories"]]
        n = len(T)

        def state(i, t):                       # position and velocity at step t, or None
            if t < 1 or t >= len(T[i]):
                return None
            return T[i][t], T[i][t] - T[i][t - 1]

        def predicts(i, j, t):
            a, b = state(i, t), state(j, t)
            if a is None or b is None:
                return False
            dp, dv = b[0] - a[0], b[1] - a[1]
            vv = float(dv @ dv)
            s = 0.0 if vv < 1e-12 else min(max(-float(dp @ dv) / vv, 0.0), HORIZON)
            return math.hypot(*(dp + s * dv)) < R2
        # agent-agent crashes: last position within 2R (+ a step) of another agent then
        for i in range(n):
            if r["outcomes"][i] != "collision":
                continue
            Ti = len(T[i]) - 1
            others = [j for j in range(n) if j != i and len(T[j]) > Ti
                      and math.dist(T[i][Ti], T[j][Ti]) < R2 + 0.5]
            if not others:
                continue                       # a static crash
            j = others[0]
            for k in crashes:
                crashes[k].append(predicts(i, j, Ti - k))
        # false alarms: pairs predicted to meet, at agent-steps not within HORIZON of a crash
        for i in range(n):
            end = len(T[i]) - 1 - (HORIZON if r["outcomes"][i] == "collision" else 0)
            for t in range(1, end):
                steps += 1
                alarms += any(predicts(i, j, t) for j in range(n)
                              if j != i and len(T[j]) - 1 - HORIZON > t)
    return {"crashes": crashes, "alarms": alarms, "steps": steps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("test", choices=["filter", "foresee"])
    ap.add_argument("--suites", default="test4,crowd16")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kinds", default=",".join(KINDS),
                    help="filter test: comma list of rays,disk,cv,est,est_ema")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "agent_predict")
    a = ap.parse_args(argv)
    from dataclasses import asdict
    from evaluate import default_reward, sim_config_from_dict
    from suites import load_suite
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers else None
    kinds = a.kinds.split(",") if a.test == "filter" else ["none"]
    fn = _filter_chunk if a.test == "filter" else _foresee_chunk
    res = {}
    try:
        for s in a.suites.split(","):
            sdir = a.data_root / s
            cfg = default_reward(sim_config_from_dict(json.loads((sdir / "suite.json").read_text())["sim"]))
            items = [(n, sc.as_dict()) for n, sc in load_suite(sdir)][:a.limit]
            kk = max(1, a.workers) * 4
            for kind in kinds:
                parts = []
                for sd in a.seeds.split(","):
                    ck = str(POLICY / "models" / f"e7_map_seed{sd}" / "ckpt_best.pt")
                    jobs = [(kind, ck, items[j::kk], asdict(cfg)) for j in range(kk) if items[j::kk]]
                    parts += pool.map(fn, jobs) if pool else [fn(j) for j in jobs]
                if a.test == "filter":
                    rows = [r for p in parts for r in p]
                    err = np.array([e for r in rows for e in r.pop("_err", [])], float)
                    m = {k: round(100 * float(np.mean([r[k] for r in rows])), 2) for k in rows[0]}
                    if len(err):
                        mt = err[err[:, 3] > 0]
                        m["blobs"] = len(err)
                        m["centre error median / p90"] = [round(float(x), 2) for x in
                                                          np.percentile(err[:, 0], [50, 90])]
                        m["velocity error median / p90 (matched)"] = [
                            round(float(x), 2) for x in np.percentile(mt[:, 1], [50, 90])]
                        m["true speed median (matched)"] = round(float(np.median(mt[:, 2])), 2)
                else:
                    m = {f"crash foreseen {k} steps before (%)":
                         round(100 * float(np.mean(sum((p["crashes"][k] for p in parts), []))), 1)
                         for k in range(1, HORIZON + 1)}
                    m["crashes"] = len(sum((p["crashes"][1] for p in parts), []))
                    m["false alarms per 100 agent-steps"] = round(
                        100 * sum(p["alarms"] for p in parts) / max(1, sum(p["steps"] for p in parts)), 2)
                res[f"{s} {kind}"] = m
                print(s, kind, m, flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.test != "filter" or tuple(kinds) == KINDS else "_" + "_".join(kinds)
    (a.out / f"{a.test}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
