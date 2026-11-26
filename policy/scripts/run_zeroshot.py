"""Run every zero-shot evaluation of DESIGN_ZEROSHOT.md, skipping finished ones.

    python scripts/run_zeroshot.py --workers 9            # all runs
    python scripts/run_zeroshot.py --dry-run               # print the commands

Each run writes ``results/zeroshot/runs/<variant>_seed<k>.csv`` (one row per
scene of every suite in ``SUITES``) through ``zeroshot_eval.py``; a run whose
CSV exists is skipped, so an interrupted batch resumes. Then
``e7_stats.py --suites ...`` makes the tables (``--stats``).

Invariants:
- Evaluation only; models and suites are read, never written.
- Same command on Windows (pathlib, ``sys.executable``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
SUITES = "test4,novel,crowd4,crowd8,crowd12,crowd16"
OUT = POLICY / "results" / "zeroshot"


def runs():
    """{run name: zeroshot_eval.py arguments}."""
    r = {}
    for k in range(3):
        for v in ("e3_both", "e7_map"):
            r[f"{v}_seed{k}"] = ["--model", f"models/{v}_seed{k}/ckpt_best.pt"]
        r[f"e3_both_shield_seed{k}"] = r[f"e3_both_seed{k}"] + ["--shield", "0.3"]
        r[f"e7_map_orbit_seed{k}"] = r[f"e7_map_seed{k}"] + ["--exit-rule", "orbit"]
        r[f"e7_map_full_seed{k}"] = r[f"e7_map_orbit_seed{k}"] + ["--shield", "0.3"]
    for m in ("orca_straight", "orca_path"):
        r[f"{m}_seed0"] = ["--model", m]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--data-root", type=Path, default=POLICY / "data" / "suites")
    ap.add_argument("--only", default=None, help="comma list of run names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true", help="only make the tables")
    a = ap.parse_args(argv)
    if not a.stats:
        for name, args in runs().items():
            if a.only and name not in a.only.split(","):
                continue
            out = OUT / "runs" / f"{name}.csv"
            if out.exists():
                print(f"{name}: done, skipped", flush=True)
                continue
            cmd = [sys.executable, str(POLICY / "scripts" / "zeroshot_eval.py"), *args,
                   "--suites", SUITES, "--data-root", str(a.data_root),
                   "--out", str(out), "--workers", str(a.workers)]
            print(" ".join(cmd), flush=True)
            if not a.dry_run:
                subprocess.run(cmd, check=True, cwd=POLICY)
    if a.dry_run:
        return
    cmp = ["e3_both:e7_map", "e3_both:e3_both_shield", "e7_map:e7_map_orbit",
           "e7_map:e7_map_full", "e3_both:e7_map_full", "orca_path:e7_map_full"]
    subprocess.run([sys.executable, str(POLICY / "scripts" / "e7_stats.py"),
                    "--runs", str(OUT / "runs"), "--out", str(OUT), "--suites", SUITES,
                    "--compare", ",".join(cmp)], check=True, cwd=POLICY)


if __name__ == "__main__":
    main()
