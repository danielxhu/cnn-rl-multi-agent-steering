"""Quick look at E2 training runs: best validation per run and per layout.

    python scripts/peek_e2.py                       # runs under E:\\policy-runs
    python scripts/peek_e2.py --runs runs --only e2_mlp_seed1 e2_gru_seed1

Reads only ``val.csv`` / ``log.csv``; safe while training is running.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path(r"E:\policy-runs"))
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args(argv)
    dirs = sorted(d for d in a.runs.iterdir() if (d / "val.csv").exists())
    if a.only:
        dirs = [d for d in dirs if d.name in a.only]
    best = {}
    for d in dirs:
        rows = list(csv.DictReader(open(d / "val.csv")))
        if not rows:
            continue
        b = max(rows, key=lambda r: (float(r["success_rate"]), -float(r["collision_rate"])))
        log = list(csv.DictReader(open(d / "log.csv")))
        best[d.name] = (b, int(float(log[-1]["agent_steps"])) if log else 0)
    if not best:
        print(f"no val.csv under {a.runs}")
        return
    names = list(best)
    layouts = sorted(k[8:] for k in best[names[0]][0] if k.startswith("success_") and k != "success_rate")
    w = max(len(n) for n in names) + 2
    print("run".ljust(w) + "steps".rjust(12) + "  best@".rjust(12) + "success".rjust(9) + "collision".rjust(11))
    for n in names:
        b, steps = best[n]
        print(n.ljust(w) + f"{steps:>12,}" + f"{int(float(b['agent_steps'])):>12,}"
              + f"{float(b['success_rate']):>9.3f}" + f"{float(b['collision_rate']):>11.3f}")
    print("\nsuccess per layout at the best validation")
    print("layout".ljust(14) + "".join(n.replace("e2_", "")[:12].rjust(13) for n in names))
    for l in layouts:
        print(l.ljust(14) + "".join(f"{float(best[n][0]['success_' + l]):>13.3f}" for n in names))


if __name__ == "__main__":
    main()
