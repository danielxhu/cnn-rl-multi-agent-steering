"""Generate every dataset of DESIGN.md §2 with ``sim/generate.py``.

    python scripts/make_datasets.py --dry-run          # print the exact commands
    python scripts/make_datasets.py --jobs 4           # ≈ 6 min on 6 cores
    python scripts/make_datasets.py --n-scale 0.01     # 1 % of every count (smoke)

Python rather than a shell script so it runs unchanged on Windows.  Up to
``--jobs`` ``generate.py`` processes run in parallel, one per dataset.  The
table below is the specification: every set is ``--layout mixed`` and 512 px;
the 128 / 256 conditions come from downsampling at load time.  ``agent_gap``
is not a ``generate.py`` flag, so each spacing set gets a one-line JSON config.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: E402,F401

BASE = ["--layout", "mixed", "--agents", "1-8", "--obstacles", "0-6"]
AUG = {
    0: [],
    1: ["--jitter-color", "10", "--jitter-line-width", "0.5", "--jitter-background", "8",
        "--noise-std", "2", "--blur-sigma", "0.3"],
    2: ["--jitter-color", "25", "--jitter-line-width", "1.0", "--jitter-background", "20",
        "--noise-std", "6", "--blur-sigma", "0.8"],
    3: ["--jitter-color", "45", "--jitter-line-width", "1.5", "--jitter-background", "35",
        "--noise-std", "12", "--blur-sigma", "1.5"],
}
GAPS = [0.0, 0.5, 1.0, 2.0, 4.0]

# (directory, n scenes, generator arguments without --n/--out)
DATASETS = [
    ("train_aug0", 5000, BASE + AUG[0] + ["--seed", "100"]),
    ("train_aug1", 5000, BASE + AUG[1] + ["--seed", "101"]),
    ("train_aug2", 5000, BASE + AUG[2] + ["--seed", "102"]),
    ("train_aug3", 5000, BASE + AUG[3] + ["--seed", "103"]),
    ("val", 500, BASE + ["--seed", "200"]),
    ("test", 1000, BASE + ["--seed", "300"]),
    ("test_aug3", 1000, BASE + AUG[3] + ["--seed", "301"]),
] + [
    (f"test_spacing/gap{g:.1f}", 300,
     ["--layout", "mixed", "--agents", "4-8", "--obstacles", "0-6", "--hard-ratio", "0.5",
      "--seed", str(400 + k), "--config", f"test_spacing/gap{g:.1f}.json"])
    for k, g in enumerate(GAPS)
] + [
    ("test_dense", 300, ["--layout", "mixed", "--agents", "12-16", "--obstacles", "2-4", "--seed", "500"]),
]


def gap_configs(data_root: Path) -> list:
    """Write test_spacing/gap<g>.json = {"agent_gap": g}; return the paths."""
    out = []
    for g in GAPS:
        p = data_root / "test_spacing" / f"gap{g:.1f}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump({"agent_gap": g}, fh)
        out.append(p)
    return out


def commands(data_root: Path, sim_dir: Path, n_scale: float = 1.0, python=None) -> list:
    """[(name, argv)] for every dataset; paths are given relative to the cwd when possible."""
    python = python or sys.executable
    gen = sim_dir / "generate.py"
    out = []
    for name, n, args in DATASETS:
        n_eff = max(1, int(round(n * n_scale)))
        args = list(args)
        if "--config" in args:
            k = args.index("--config") + 1
            args[k] = (data_root / args[k]).as_posix()
        out.append((name, [python, gen.as_posix(), "--n", str(n_eff), "--out",
                           (data_root / name).as_posix()] + args))
    return out


def run_all(cmds: list, jobs: int, quiet: bool = True) -> int:
    """Run the commands with up to `jobs` concurrent processes; return the number of failures."""
    pending = list(cmds)
    running = []
    failed = 0
    t0 = time.time()
    while pending or running:
        while pending and len(running) < jobs:
            name, cmd = pending.pop(0)
            print(f"start {name}: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL if quiet else None,
                                    stderr=subprocess.STDOUT if quiet else None)
            running.append((name, proc, time.time()))
        time.sleep(0.5)
        still = []
        for name, proc, ts in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, ts))
                continue
            status = "done" if rc == 0 else f"FAILED (exit {rc})"
            failed += rc != 0
            print(f"{status} {name} in {time.time() - ts:.0f}s", flush=True)
        running = still
    print(f"all datasets finished in {time.time() - t0:.0f}s, {failed} failure(s)")
    return failed


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the perception datasets of DESIGN.md §2.")
    p.add_argument("--data-root", default=str(_paths.PERCEPTION_DIR / "data"))
    p.add_argument("--sim-dir", default=str(_paths.SIM_DIR))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--n-scale", type=float, default=1.0, help="multiply every scene count")
    p.add_argument("--only", default=None, help="glob on the dataset name, e.g. 'test_*'")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    p.add_argument("--verbose", action="store_true", help="show generate.py output")
    a = p.parse_args(argv)

    data_root, sim_dir = Path(a.data_root), Path(a.sim_dir)
    try:
        data_root = data_root.relative_to(Path.cwd())
        sim_dir = sim_dir.relative_to(Path.cwd())
    except ValueError:
        pass                                          # keep absolute paths
    cmds = commands(data_root, sim_dir, a.n_scale)
    if a.only:
        cmds = [(n, c) for n, c in cmds if fnmatch.fnmatch(n, a.only)]
    if a.dry_run:
        for name, cmd in cmds:
            print(" ".join(cmd))
        print(f"{len(cmds)} datasets, {sum(int(c[c.index('--n') + 1]) for _, c in cmds)} scenes")
        return 0
    gap_configs(data_root)
    return 1 if run_all(cmds, a.jobs, quiet=not a.verbose) else 0


if __name__ == "__main__":
    sys.exit(main())
