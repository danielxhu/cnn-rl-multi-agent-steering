"""E5 sweep launcher: the grid of DESIGN.md §6.2, run sequentially and resumably.

    python sweep.py --grid e5 --out runs/e5 --dry-run       # 24 commands + projected time
    python sweep.py --grid e5 --out runs/e5                 # run them (safe to Ctrl-C and relaunch)
    python sweep.py --grid e5 --out runs/e5 --only 's128_*'

Run directory name: ``runs/e5/s{size}_a{aug}_{arch}{_inst}_seed{k}``.  For each
run: skip if ``metrics.json`` exists; resume if ``ckpt_last.pt`` exists; else
train from scratch; then evaluate on ``test``, ``test_aug3``, ``test_spacing/*``
and ``test_dense``.  Every outcome is appended to ``runs/e5/sweep_status.csv``
(name, status, start, end, seconds, epoch_seconds).

Grids:
    e5       star design, 24 runs: resolution arm (128/256/512 at aug2) and
             augmentation arm (aug0/1/3 at 256), × {seg-only, +instance head} × seeds {0, 1}
    e5_full  full 3 × 4 × 2 × 2 factorial, 48 runs
512 trains for 30 epochs, everything else for 60 (§6.1).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import _paths  # noqa: F401

SIZES = [128, 256, 512]
# Every size trains for the same number of epochs, so the resolution arm is not
# confounded with training length (measured speed allows it; DESIGN.md §6.1).
EPOCHS = 60
AUGS = [0, 1, 2, 3]
ARCHS = ["unet24x4"]
SEEDS = [0, 1]
EVAL_SETS = ["test", "test_aug3", "test_spacing", "test_dense"]
STATUS_COLUMNS = ["name", "status", "start", "end", "seconds", "epoch_seconds"]
# Seconds per epoch, measured on the GTX 1070 over 5000 training scenes
# (2026-09-22, second epoch of a dev run at each size; the first epoch is
# slower: cuDNN algorithm selection and a cold page cache).
ESTIMATED_EPOCH_SECONDS = {128: 20.0, 256: 74.0, 512: 281.0}


@dataclass
class RunSpec:
    size: int
    aug: int
    arch: str = "unet24x4"
    instance_head: bool = False
    seed: int = 0
    extra: list = field(default_factory=list)      # extra train.py flags (e.g. --dice 0.5)
    tag: str = ""                                  # name suffix for extras

    @property
    def epochs(self) -> int:
        return EPOCHS

    @property
    def name(self) -> str:
        return (f"s{self.size}_a{self.aug}_{self.arch}{'_inst' if self.instance_head else ''}"
                f"{self.tag}_seed{self.seed}")


def e5_grid() -> list:
    runs = []
    for size in SIZES:                                     # resolution arm at aug2
        for arch in ARCHS:
            for inst in (False, True):
                for seed in SEEDS:
                    runs.append(RunSpec(size, 2, arch, inst, seed))
    for aug in (0, 1, 3):                                  # augmentation arm at 256
        for arch in ARCHS:
            for inst in (False, True):
                for seed in SEEDS:
                    runs.append(RunSpec(256, aug, arch, inst, seed))
    return sorted(runs, key=_order)


def e5_full_grid() -> list:
    runs = [RunSpec(size, aug, arch, inst, seed)
            for size in SIZES for aug in AUGS for arch in ARCHS
            for inst in (False, True) for seed in SEEDS]
    return sorted(runs, key=_order)


def e5_extras_grid() -> list:
    """Optional extras of §6.2, in priority order."""
    runs = [RunSpec(512, 2, "unet24x5", inst, 0) for inst in (False, True)]
    runs += [RunSpec(512, 3, "unet24x4", inst, 0) for inst in (False, True)]
    runs += [RunSpec(128, 2, "unet24x4", inst, 0, ["--dice", "0.5"], "_dice") for inst in (False, True)]
    return runs


GRIDS = {"e5": e5_grid, "e5_full": e5_full_grid, "e5_extras": e5_extras_grid}


def _order(r: RunSpec):
    return (r.size, r.aug, r.arch, r.instance_head, r.tag, r.seed)


# ---------------------------------------------------------------- commands

def train_command(spec: RunSpec, out_root: Path, data_root: Path, python: str, resume: bool) -> list:
    cmd = [python, "train.py", "--data-root", data_root.as_posix(),
           "--train", f"train_aug{spec.aug}", "--val", "val",
           "--size", str(spec.size), "--arch", spec.arch, "--seed", str(spec.seed),
           "--epochs", str(spec.epochs), "--out", (out_root / spec.name).as_posix()]
    if spec.instance_head:
        cmd.append("--instance-head")
    cmd += spec.extra
    if resume:
        cmd.append("--resume")
    return cmd


def eval_command(spec: RunSpec, out_root: Path, data_root: Path, python: str) -> list:
    return [python, "evaluate.py", "--ckpt", (out_root / spec.name / "ckpt_best.pt").as_posix(),
            "--data"] + [(data_root / s).as_posix() for s in EVAL_SETS]


def epoch_seconds(run_dir: Path):
    """Mean seconds per epoch from a run's log.csv, or None."""
    log = run_dir / "log.csv"
    if not log.exists():
        return None
    with open(log) as fh:
        vals = [float(r["seconds"]) for r in csv.DictReader(fh) if r.get("seconds")]
    return sum(vals) / len(vals) if vals else None


def measured_epoch_seconds(out_root: Path) -> dict:
    """size -> mean epoch seconds over every run of that size with a log."""
    acc = {}
    for d in sorted(out_root.glob("s*_a*_seed*")) if out_root.exists() else []:
        try:
            size = int(d.name[1:].split("_")[0])
        except ValueError:
            continue
        s = epoch_seconds(d)
        if s:
            acc.setdefault(size, []).append(s)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def append_status(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STATUS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in STATUS_COLUMNS})


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


# -------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description="Run the E5 perception sweep.")
    p.add_argument("--grid", choices=sorted(GRIDS), default="e5")
    p.add_argument("--out", default="runs/e5")
    p.add_argument("--data-root", default="data")
    p.add_argument("--only", default=None, help="glob on the run name, e.g. 's256_*'")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args(argv)

    out_root, data_root = Path(a.out), Path(a.data_root)
    runs = GRIDS[a.grid]()
    if a.only:
        runs = [r for r in runs if fnmatch.fnmatch(r.name, a.only)]
    status_path = out_root / "sweep_status.csv"

    if a.dry_run:
        measured = measured_epoch_seconds(out_root)
        total = 0.0
        for spec in runs:
            run_dir = out_root / spec.name
            done = (run_dir / "metrics.json").exists()
            resume = (run_dir / "ckpt_last.pt").exists() and not done
            print(" ".join(train_command(spec, out_root, data_root, a.python, resume)))
            print(" ".join(eval_command(spec, out_root, data_root, a.python)))
            if not done:
                total += spec.epochs * measured.get(spec.size, ESTIMATED_EPOCH_SECONDS[spec.size])
        src = ", ".join(f"{k}: {v:.0f}s/epoch (measured)" for k, v in sorted(measured.items())) or \
            "no measurements yet, FLOP estimates of §6.1"
        print(f"\n{len(runs)} runs in grid '{a.grid}'; projected training time for the runs not yet "
              f"done: {total / 3600:.1f} h  [{src}]")
        return 0

    cwd = _paths.PERCEPTION_DIR
    for k, spec in enumerate(runs, 1):
        run_dir = out_root / spec.name
        if (run_dir / "metrics.json").exists():
            print(f"[{k}/{len(runs)}] {spec.name}: metrics.json exists, skipping")
            continue
        resume = (run_dir / "ckpt_last.pt").exists()
        print(f"[{k}/{len(runs)}] {spec.name}: {'resuming' if resume else 'starting'}", flush=True)
        t0 = time.time()
        status = "ok"
        try:
            rc = subprocess.run(train_command(spec, out_root, data_root, a.python, resume), cwd=cwd).returncode
            if rc != 0:
                status = f"train_failed_{rc}"
            else:
                rc = subprocess.run(eval_command(spec, out_root, data_root, a.python), cwd=cwd).returncode
                if rc != 0:
                    status = f"eval_failed_{rc}"
        except KeyboardInterrupt:
            append_status(status_path, {"name": spec.name, "status": "interrupted",
                                        "start": _stamp(t0), "end": _stamp(time.time()),
                                        "seconds": f"{time.time() - t0:.0f}",
                                        "epoch_seconds": epoch_seconds(run_dir) or ""})
            print("interrupted; relaunch to resume")
            return 130
        t1 = time.time()
        append_status(status_path, {"name": spec.name, "status": status, "start": _stamp(t0),
                                    "end": _stamp(t1), "seconds": f"{t1 - t0:.0f}",
                                    "epoch_seconds": f"{epoch_seconds(run_dir):.1f}" if epoch_seconds(run_dir) else ""})
        print(f"[{k}/{len(runs)}] {spec.name}: {status} in {(t1 - t0) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
