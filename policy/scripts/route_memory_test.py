"""Route memory test: keep the previous route unless it is blocked or clearly worse.

    python scripts/route_memory_test.py --keep 1.0,1.2,1.5 --suites test4,novel --workers 9

The wall memory (``wallmap.WallMemory``) recomputes its field every
``wallmap_every`` steps and hands the network the downhill direction of the
new field: the route is thrown away each time, and when a newly seen wall
makes the other side look shorter the direction flips (DESIGN_E7 §14.1).

``RouteMemory`` (evaluation only, no retraining) also remembers the route, a
list of field-cell centres from the agent to the target:

- every step the passed part is cut off; the downhill direction points at the
  route point ``AHEAD`` units on (the old 3-cell descent, 6 units);
- at every field update the old route's next ``LOOK`` units are checked on the
  new field: blocked (a cell the new field cannot use) -> replaced; otherwise
  its cost (that stretch + field value at its end) is compared with the new
  shortest; kept when cost_old <= ``keep`` x cost_new, the rest of the route
  re-planned from the end of the stretch;
- dropped when the agent has strayed more than ``STRAY`` from it or the exit
  target changed.

``keep = 1.0`` keeps the route only on ties (close to the current rule).
Writes ``results/route_memory/e7_map_keep<k>_seed<s>.csv`` per scene.
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

AHEAD = 6.0
LOOK = 20.0
STRAY = 4.0
MAX_CELLS = 400


def make_route_memory(keep):
    import wallmap
    from wallmap import FIELD_CELL, _NBRS
    # the unpatched class: a pool worker may already carry an earlier patch
    WallMemory = wallmap.__dict__.setdefault("_PLAIN_WALLMEMORY", wallmap.WallMemory)

    class RouteMemory(WallMemory):
        def reset(self, sim):
            super().reset(sim)
            self.route = [None] * self.n
            self.route_target = list(self.target)
            self.n_kept = self.n_replaced = 0

        def _path(self, fld, p):
            """Steepest-descent cell path from world point p to the field's bottom."""
            m = fld.shape[0]
            fin = np.isfinite(fld)
            cx = min(max(int(p[0] // FIELD_CELL), 0), m - 1)
            cy = min(max(int(p[1] // FIELD_CELL), 0), m - 1)
            if not fin[cy, cx]:
                c = [(cy + dy, cx + dx) for dy, dx in _NBRS
                     if 0 <= cy + dy < m and 0 <= cx + dx < m and fin[cy + dy, cx + dx]]
                if not c:
                    return None
                cy, cx = min(c, key=lambda t: fld[t])
            out = [(cy, cx)]
            for _ in range(MAX_CELLS):
                best, bv = None, fld[cy, cx]
                for dy, dx in _NBRS:
                    y, x = cy + dy, cx + dx
                    if not (0 <= y < m and 0 <= x < m) or not fin[y, x]:
                        continue
                    if dy and dx and not (fin[cy + dy, cx] and fin[cy, cx + dx]):
                        continue
                    if fld[y, x] < bv:
                        best, bv = (y, x), fld[y, x]
                if best is None:
                    break
                cy, cx = best
                out.append(best)
            return np.array([((x + 0.5) * FIELD_CELL, (y + 0.5) * FIELD_CELL) for y, x in out])

        def _trim(self, i):
            r = self.route[i]
            if r is None:
                return None
            d = np.hypot(*(r - self.est_pos[i]).T)
            k = int(np.argmin(d))
            if d[k] > STRAY:
                self.route[i] = None
                return None
            self.route[i] = r[k:]
            return self.route[i]

        def update_field(self, i, sim):
            super().update_field(i, sim)
            fld = self.field[i]
            if fld is None:
                self.route[i] = None
                return
            if self.target[i] != self.route_target[i]:
                self.route[i] = None
                self.route_target[i] = self.target[i]
            p = self.est_pos[i]
            new = self._path(fld, p)
            old = self._trim(i)
            if old is not None and len(old) > 1 and new is not None:
                seg = np.hypot(*np.diff(old, axis=0).T)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                k = max(1, int(np.searchsorted(cum, LOOK)))
                k = min(k, len(old) - 1)
                stretch = old[:k + 1]
                m = fld.shape[0]
                ix = np.clip((stretch[:, 0] // FIELD_CELL).astype(int), 0, m - 1)
                iy = np.clip((stretch[:, 1] // FIELD_CELL).astype(int), 0, m - 1)
                if np.isfinite(fld[iy, ix]).all():
                    c_old = math.dist(p, stretch[0]) + cum[k] + float(fld[iy[-1], ix[-1]])
                    c_new = self._value(fld, p)
                    if math.isfinite(c_new) and c_old <= keep * c_new + 1e-9:
                        rest = self._path(fld, stretch[-1])
                        self.route[i] = stretch if rest is None else np.concatenate([stretch, rest[1:]])
                        self.n_kept += 1
                        return
            self.route[i] = new
            self.n_replaced += 1

        def downhill(self, i, bearing):
            r = self._trim(i)
            if r is None or len(r) < 2:
                return super().downhill(i, bearing)
            p = self.est_pos[i]
            d = np.hypot(*(r - p).T)
            far = np.flatnonzero(d >= AHEAD)
            q = r[far[0]] if len(far) else r[-1]
            v = q - p
            if math.hypot(*v) < 1e-6:
                return super().downhill(i, bearing)
            ang = math.atan2(v[1], v[0]) - self.est_head[i]
            return np.array([math.cos(ang), math.sin(ang)], np.float32)

    return RouteMemory


def _chunk(args):
    keep, ckpt, shield, items, sim_dict = args
    import wallmap
    plain = wallmap.__dict__.setdefault("_PLAIN_WALLMEMORY", wallmap.WallMemory)
    wallmap.WallMemory = make_route_memory(keep) if keep > 0 else plain
    from evaluate import LOOP_TURN, load_policy, run_episode, sim_config_from_dict, turning
    from scene import Scene
    cfg = sim_config_from_dict(sim_dict)
    ctrl, _, _, _ = load_policy(ckpt)
    ctrl.exit_rule = "orbit"
    c = ctrl
    if shield is not None:
        from shield import Shield
        c = Shield(ctrl, cfg, shield)
    rows = []
    for name, d in items:
        sc = Scene.from_dict(d)
        r = run_episode(sc, c, cfg, record=True)
        loops = [turning(t) >= LOOP_TURN for t in r["trajectories"]]
        wm = ctrl.wm
        rows.append(dict(scene=name, layout=sc.layout, success_rate=r["success_rate"],
                         static_collision_rate=r["static_collision_rate"],
                         agent_collision_rate=r["agent_collision_rate"],
                         timeout_rate=r["timeout_rate"], loop_rate=float(np.mean(loops)),
                         kept=getattr(wm, "n_kept", 0), replaced=getattr(wm, "n_replaced", 0)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", default="1.0,1.2,1.5", help="comma list; 0 = plain WallMemory")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--suites", default="test4,novel")
    ap.add_argument("--shield", type=float, default=None)
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "route_memory")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from dataclasses import asdict
    from evaluate import default_reward, sim_config_from_dict
    from suites import load_suite
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers else None
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.shield is None else f"_shield{a.shield}"
    try:
        for keep in [float(x) for x in a.keep.split(",")]:
            for sd in a.seeds.split(","):
                ck = str(POLICY / "models" / f"e7_map_seed{sd}" / "ckpt_best.pt")
                rows = []
                for s in a.suites.split(","):
                    sdir = a.data_root / s
                    cfg = default_reward(sim_config_from_dict(
                        json.loads((sdir / "suite.json").read_text())["sim"]))
                    items = [(n, sc.as_dict()) for n, sc in load_suite(sdir)][:a.limit]
                    kk = max(1, a.workers) * 4
                    jobs = [(keep, ck, a.shield, items[j::kk], asdict(cfg))
                            for j in range(kk) if items[j::kk]]
                    parts = pool.map(_chunk, jobs) if pool else [_chunk(j) for j in jobs]
                    got = [dict(r, suite=s) for p in parts for r in p]
                    rows += got
                    m = {c: 100 * np.mean([r[c] for r in got]) for c in
                         ("success_rate", "loop_rate", "timeout_rate", "static_collision_rate",
                          "agent_collision_rate")}
                    kp = sum(r["kept"] for r in got) / max(1, sum(r["kept"] + r["replaced"] for r in got))
                    print(f"keep {keep} seed{sd} {s}: " + " ".join(
                        f"{c.split('_')[0]} {v:.1f}" for c, v in m.items()) + f" kept {100 * kp:.0f}%",
                        flush=True)
                with open(a.out / f"e7_map_keep{keep}{tag}_seed{sd}.csv", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
