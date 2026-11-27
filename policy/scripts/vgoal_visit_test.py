"""Loop fixes at evaluation only: a fixed virtual goal and a visit cost on the field.

    python scripts/vgoal_visit_test.py --fix-angle 45 --suites test4,novel --workers 9
    python scripts/vgoal_visit_test.py --visit 1.0 --suites test4,novel --workers 9

Two changes to ``wallmap.WallMemory`` (no retraining, e7_map checkpoints,
orbit exit rule, no safety filter, as ``route_memory_test.py``):

1. Fixed virtual goal (``--fix-angle DEG`` and/or ``--fix-every STEPS``).
   Method A recomputes the far point along the bearing at every field update,
   so the field's bottom moves with the agent. Here the far point is kept and
   recomputed only when the bearing's world direction has turned more than
   ``DEG`` away from the direction it was set with, when ``STEPS`` steps have
   passed since it was set, or when the exit target changed. Also a diagnosis:
   if loops drop, the moving goal causes them.
2. Visit cost (``--visit W``; after LRTA* / Moving Target Search). Each step
   the agent adds ``W`` (world units) to the field cell it is in (its own
   estimated position); the downhill direction is taken on field + W x visits.
   An area the agent keeps circling rises until the descent leads out of it.
   Reachability checks (exit switch) use the plain field.
3. Full map (``--full-map``, diagnostic, privileged): the memory starts with
   the scene's true occupancy (downsampled to its raster) instead of empty;
   method A's target is unchanged. The loop gap to the baseline is the part
   caused by the incomplete map.
4. Inertia (``--inertia N``): a new downhill direction more than 90 deg from
   the one followed is held off (the old direction is kept) until it has
   persisted for N steps or a wall/obstacle ray within ``BLOCK_D`` is seen
   along the old direction; smaller changes are followed at once.
5. Dead-end seal (``--seal C``, Daniel's DFS idea): the cells walked since
   the last turn-back form the trail; a field update that turns the descent
   by more than 90 deg (same exit target) is a turn-back: the trail is
   sealed, + C on those cells, once the agent is more than one cell away from
   it (so it is not pushed sideways while retracing), and a new trail starts.

Writes ``results/vgoal_visit/e7_map_<tag>_seed<s>.csv`` per scene and
prints means per suite and for the u_trap / blocked_exit layouts.
Invariants: read-only on models and suites; uses only what WallMemory uses.
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
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

BLOCK_D = 6.0        # inertia: a wall this close along the held direction ends the hold


def make_memory(o):
    """o: dict fix_angle, fix_every, visit, full_map, inertia, seal (see module doc)."""
    import wallmap
    from config import E_OBSTACLE, E_WALL
    from wallmap import FIELD_CELL, wrap
    # the unpatched class: a pool worker may already carry an earlier patch
    WallMemory = wallmap.__dict__.setdefault("_PLAIN_WALLMEMORY", wallmap.WallMemory)
    thr = math.radians(o["fix_angle"]) if o["fix_angle"] > 0 else None
    fix_every, visit, inertia, seal = o["fix_every"], o["visit"], o["inertia"], o["seal"]

    class Memory(WallMemory):
        def reset(self, sim):
            super().reset(sim)
            if o["full_map"]:
                from reachability import occupancy
                occ = occupancy(self.c, sim.scene)
                f = occ.shape[0] // self.m
                self.occ[:] = occ[:self.m * f, :self.m * f].reshape(self.m, f, self.m, f).any(axis=(1, 3))
            self.vg = [None] * self.n               # (point, world dir, step, target)
            self.visits = np.zeros((self.n, self.mc, self.mc), np.float32)
            self.sealed = np.zeros((self.n, self.mc, self.mc), np.float32)
            self.trail = [[] for _ in range(self.n)]       # cells since the last turn-back
            self.pending = [None] * self.n                 # trail to seal once left
            self.cur_dir = [None] * self.n                 # inertia: world direction followed
            self.wait = np.zeros(self.n, int)
            self.t = 0
            self.n_set = self.n_kept = self.n_turnback = self.n_held = 0
            self._obs = None

        # ------------------------------------------------------------ helpers
        def _cell(self, p):
            m = self.mc
            return (min(max(int(p[1] // FIELD_CELL), 0), m - 1),
                    min(max(int(p[0] // FIELD_CELL), 0), m - 1))

        def _extra(self, i):
            e = None
            if visit > 0:
                e = self.visits[i]
            if seal > 0:
                e = self.sealed[i] if e is None else e + self.sealed[i]
            return e

        def _dir(self, i, fld):
            """World angle of the descent on fld (+ extras), or None."""
            if fld is None:
                return None
            e = self._extra(i)
            q = self._descend(fld if e is None else fld + e, self.est_pos[i])
            if q is None:
                return None
            v = q - self.est_pos[i]
            return math.atan2(v[1], v[0]) if math.hypot(*v) > 1e-6 else None

        def _seen_blocked(self, i, world_dir):
            """A wall/obstacle ray within BLOCK_D near world_dir this step."""
            if self._obs is None:
                return False
            S, c = self.c.n_sectors, self.c
            rel = (world_dir - self.est_head[i] - self.sector_off + math.pi) % (2 * math.pi) - math.pi
            k = int(np.argmin(np.abs(rel)))
            if abs(rel[k]) > math.radians(20):
                return False                           # outside the field of view
            row = self._obs[i]
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < S:
                    r = min(float(row[E_OBSTACLE * S + kk]), float(row[E_WALL * S + kk]))
                    if r * c.fov_range < BLOCK_D:
                        return True
            return False

        # -------------------------------------------------------- overrides
        def observe(self, sim, obs):
            self._obs = np.asarray(obs, np.float32)
            return super().observe(sim, obs)

        def update_field(self, i, sim):
            self.t = int(sim.steps)
            before = self._dir(i, self.field[i]) if seal > 0 else None
            tgt = self.target[i]
            super().update_field(i, sim)
            if seal > 0 and before is not None and self.target[i] == tgt:
                after = self._dir(i, self.field[i])
                if after is not None and abs(wrap(after - before)) > math.pi / 2:
                    self.n_turnback += 1               # dead end: seal its trail once left
                    if self.trail[i]:
                        self.pending[i] = list(self.trail[i])
                    self.trail[i] = []

        def virtual_goal(self, i, bearing):
            if thr is None and fix_every <= 0:
                return super().virtual_goal(i, bearing)
            a = self.est_head[i] + bearing * math.pi
            v = self.vg[i]
            if v is not None and v[3] == self.target[i] \
                    and (thr is None or abs(wrap(a - v[1])) <= thr) \
                    and (fix_every <= 0 or self.t - v[2] < fix_every):
                self.n_kept += 1
                return v[0]
            q = super().virtual_goal(i, bearing)
            self.vg[i] = (q, a, self.t, self.target[i])
            self.n_set += 1
            return q

        def downhill(self, i, bearing):
            cy, cx = self._cell(self.est_pos[i])
            if visit > 0:
                self.visits[i, cy, cx] += visit
            if seal > 0:
                if not self.trail[i] or self.trail[i][-1] != (cy, cx):
                    self.trail[i].append((cy, cx))
                pd = self.pending[i]
                if pd is not None and min(max(abs(y - cy), abs(x - cx)) for y, x in pd) > 1:
                    for y, x in set(pd):
                        self.sealed[i, y, x] += seal
                    self.pending[i] = None
            e = self._extra(i)
            plain = self.field[i]
            if plain is not None and e is not None:
                self.field[i] = plain + e
            try:
                out = super().downhill(i, bearing)
            finally:
                self.field[i] = plain
            if inertia <= 0:
                return out
            new = self.est_head[i] + math.atan2(float(out[1]), float(out[0]))
            cur = self.cur_dir[i]
            if cur is None or abs(wrap(new - cur)) <= math.pi / 2 \
                    or self.wait[i] >= inertia or self._seen_blocked(i, cur):
                self.cur_dir[i], self.wait[i] = new, 0
                return out
            self.wait[i] += 1                          # a turn-back not yet confirmed: hold
            self.n_held += 1
            rel = cur - self.est_head[i]
            return np.array([math.cos(rel), math.sin(rel)], np.float32)

    return Memory


def _chunk(args):
    o, ckpt, items, sim_dict = args
    import wallmap
    plain = wallmap.__dict__.setdefault("_PLAIN_WALLMEMORY", wallmap.WallMemory)
    on = any(o[k] for k in ("fix_angle", "fix_every", "visit", "full_map", "inertia", "seal"))
    wallmap.WallMemory = make_memory(o) if on else plain
    from evaluate import LOOP_TURN, load_policy, run_episode, sim_config_from_dict, turning
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl, _, _, _ = load_policy(ckpt)
    ctrl.exit_rule = "orbit"
    rows = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, ctrl, cfg, record=True)
        loops = [turning(t) >= LOOP_TURN for t in r["trajectories"]]
        wm = ctrl.wm
        rows.append(dict(scene=name, layout=sc.layout, success_rate=r["success_rate"],
                         static_collision_rate=r["static_collision_rate"],
                         agent_collision_rate=r["agent_collision_rate"],
                         timeout_rate=r["timeout_rate"], loop_rate=float(np.mean(loops)),
                         vg_set=getattr(wm, "n_set", 0), vg_kept=getattr(wm, "n_kept", 0),
                         turnbacks=getattr(wm, "n_turnback", 0), held=getattr(wm, "n_held", 0)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fix-angle", type=float, default=0.0, help="deg; 0 = off")
    ap.add_argument("--fix-every", type=int, default=0, help="steps; 0 = off")
    ap.add_argument("--visit", type=float, default=0.0, help="cost per step in a cell; 0 = off")
    ap.add_argument("--full-map", action="store_true", help="diagnostic: start from the true map")
    ap.add_argument("--inertia", type=int, default=0, help="steps a turn-back is held; 0 = off")
    ap.add_argument("--seal", type=float, default=0.0, help="cost added to a dead-end trail; 0 = off")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--suites", default="test4,novel")
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "vgoal_visit")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from dataclasses import asdict
    from evaluate import default_reward, sim_config_from_dict
    from suites import load_suite
    tag = "_".join(t for t in (f"angle{a.fix_angle:g}" if a.fix_angle > 0 else "",
                               f"every{a.fix_every}" if a.fix_every > 0 else "",
                               f"visit{a.visit:g}" if a.visit > 0 else "",
                               "fullmap" if a.full_map else "",
                               f"inertia{a.inertia}" if a.inertia > 0 else "",
                               f"seal{a.seal:g}" if a.seal > 0 else "") if t) or "plain"
    opts = dict(fix_angle=a.fix_angle, fix_every=a.fix_every, visit=a.visit,
                full_map=a.full_map, inertia=a.inertia, seal=a.seal)
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers else None
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ("success_rate", "loop_rate", "timeout_rate", "static_collision_rate",
            "agent_collision_rate")
    try:
        for sd in a.seeds.split(","):
            ck = str(POLICY / "models" / f"e7_map_seed{sd}" / "ckpt_best.pt")
            rows = []
            for s in a.suites.split(","):
                sdir = a.data_root / s
                cfg = default_reward(sim_config_from_dict(
                    json.loads((sdir / "suite.json").read_text())["sim"]))
                items = [(n, sc.as_dict()) for n, sc in load_suite(sdir)][:a.limit]
                kk = max(1, a.workers) * 4
                jobs = [(opts, ck, items[j::kk], asdict(cfg))
                        for j in range(kk) if items[j::kk]]
                parts = pool.map(_chunk, jobs) if pool else [_chunk(j) for j in jobs]
                got = [dict(r, suite=s) for p in parts for r in p]
                rows += got
                groups = {s: got} | {lay: [r for r in got if r["layout"] == lay]
                                     for lay in ("u_trap", "blocked_exit")}
                for g, rr in groups.items():
                    if not rr:
                        continue
                    m = {c: 100 * np.mean([r[c] for r in rr]) for c in cols}
                    print(f"{tag} seed{sd} {g}: " + " ".join(
                        f"{c.split('_')[0]} {v:.1f}" for c, v in m.items()), flush=True)
            with open(a.out / f"e7_map_{tag}_seed{sd}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
