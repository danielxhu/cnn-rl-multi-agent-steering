"""Generalisation suites (E8): layouts the policy never trained on, and larger crowds.

    python scripts/make_novel_suites.py --data-root E:\\policy-data

    novel      the five test-only families of ``sim.layouts.NOVEL_LAYOUTS``
               (l_turn, switchback, maze, u_trap, blocked_exit), 100 scenes
               each, 4 agents, 0-6 obstacles (as test4), seed 10000
    crowd4 / crowd8 / crowd12 / crowd16
               the ten training families (``LAYOUTS_E2``), 30 scenes each,
               exactly N agents asked for (small start boxes may hold fewer;
               the scene's own count is reported), seed 11000 + N

Skips a suite whose ``suite.json`` exists. Same generator contract as
``make_suites.py`` (``suites.make_suite``), so the suites are reproducible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _paths  # noqa: E402,F401
from layouts import NOVEL_LAYOUTS  # noqa: E402
from policyconfig import LAYOUTS_E2, sim_config  # noqa: E402
from suites import make_suite  # noqa: E402

NOVEL_SPECS = {"novel": (tuple(NOVEL_LAYOUTS), 100, (4, 4), 10000)}
for _n in (4, 8, 12, 16):
    NOVEL_SPECS[f"crowd{_n}"] = (LAYOUTS_E2, 30, (_n, _n), 11000 + _n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--only", default=None, help="comma list of suites")
    a = ap.parse_args(argv)
    for name, (layouts, per, agents, seed) in NOVEL_SPECS.items():
        if a.only and name not in a.only.split(","):
            continue
        out = a.data_root / name
        if (out / "suite.json").exists():
            print(f"{name}: exists, skipped", flush=True)
            continue
        meta = make_suite(out, layouts, per, agents, (0, 6), seed, sim_config(), name=name)
        print(f"{name}: {meta['n_scenes']} scenes -> {out}", flush=True)


if __name__ == "__main__":
    main()
