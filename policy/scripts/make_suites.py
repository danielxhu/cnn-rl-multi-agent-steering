"""Generate the three E2 evaluation suites (DESIGN.md §5.1) into a data root.

    python scripts/make_suites.py --data-root E:\\policy-data

Skips a suite whose ``suite.json`` exists. ``--only val`` builds just one;
``--per-layout N`` overrides the count (small suites for trials).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: E402,F401

from policyconfig import LAYOUTS_E2, sim_config  # noqa: E402
from suites import SUITE_SPECS, make_suite  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--only", default=None, help="comma list of suites")
    ap.add_argument("--per-layout", type=int, default=None)
    ap.add_argument("--layouts", default=None)
    a = ap.parse_args(argv)
    layouts = tuple(a.layouts.split(",")) if a.layouts else LAYOUTS_E2
    for name, (per, agents, seed) in SUITE_SPECS.items():
        if a.only and name not in a.only.split(","):
            continue
        out = a.data_root / name
        if (out / "suite.json").exists():
            print(f"{name}: exists, skipped", flush=True)
            continue
        meta = make_suite(out, layouts, a.per_layout or per, agents, (0, 6), seed,
                          sim_config(), name=name)
        print(f"{name}: {meta['n_scenes']} scenes -> {out}", flush=True)


if __name__ == "__main__":
    main()
