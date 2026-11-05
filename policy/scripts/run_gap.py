"""Behaviour gap on the training machine: parse, score, roll out, push (DESIGN_GAP.md).

    python scripts/run_gap.py              # everything; safe to Ctrl-C and run again
    python scripts/run_gap.py --dry-run    # print the commands only
    python scripts/run_gap.py --limit 20   # smoke run: no commit, outputs under --work
    python scripts/run_gap.py --policy E:\\policy-runs\\e3_geo_seed1\\ckpt_best.pt

Steps, each skipped when its output already exists:

1. check the perception model, the policy and the test sets (stop if one is missing)
2. ``perception/predict.py`` on each set with the tuned thresholds
   -> ``<work>/<model>/parsed/<set>``
3. ``perception/evaluate.py`` on each set with the same thresholds
   -> ``<work>/<model>/perception/<set>/per_scene.csv``
4. ``gap.py``: the policy and ``orca_path`` in the true and parsed scene, and
   ``orca_path`` on the parsed map in the true scene, resuming scene by scene
   -> ``policy/results/gap/<policy>/``
5. ``git pull --rebase --autostash``, add, commit and push that folder
   (``--no-push`` skips the pull and the push; nothing to commit ends cleanly)

Defaults are the training machine's paths (E:\\cnn-data, E:\\policy-eval\\gap).
Steps 2-3 use the GPU if there is one; step 4 uses CPU workers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # policy/
REPO = HERE.parent
PERCEPTION = REPO / "perception"
PY = sys.executable
SETS = ["test_sketch7", "test_sketch7_free", "test_sketch7_tilt", "test_sketch7_lines"]
DRIVE = re.compile(r"^[A-Za-z]:\\")


def banner(text):
    print(f"\n=== {text}", flush=True)


def shown(c) -> str:
    """A Windows path built on another OS still prints with Windows separators."""
    s = str(c)
    return s.replace("/", "\\") if DRIVE.match(s) else s


def run(cmd, dry, cwd=HERE) -> int:
    print(f"  (in {cwd})\n  $ " + subprocess.list2cmdline([shown(c) for c in cmd]), flush=True)
    if dry:
        return 0
    rc = subprocess.run([str(c) for c in cmd], cwd=cwd).returncode
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {' '.join(str(c) for c in cmd)}")
    return rc


def absolute(p: Path) -> Path:
    """Steps run in other folders, so paths must be absolute; E:\\... stays as typed on any OS."""
    return p if DRIVE.match(str(p)) else p.absolute()


def tuned_flags(model: Path) -> list:
    """The model's tuned extractor thresholds as flags for predict.py and evaluate.py."""
    best = json.loads((model / "tune_extract.json").read_text(encoding="utf-8"))["best"]
    flags = []
    for key, flag in (("heat_thresh", "--heat-thresh"), ("heat_nms_factor", "--heat-nms-factor"),
                      ("letter_thresh", "--letter-thresh")):
        if best.get(key) is not None:
            flags += [flag, best[key]]
    return flags


def scene_count(set_dir: Path) -> int:
    return len(list(set_dir.glob("scene_*.json")))


def scored_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def parsed_count(parsed_dir: Path) -> int:
    """Scenes of a parse that ran to its end; a cut-off parse has no marker and counts 0."""
    marker = parsed_dir / "_complete.txt"
    return int(marker.read_text(encoding="utf-8")) if marker.exists() else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path(r"E:\cnn-data"))
    ap.add_argument("--work", type=Path, default=Path(r"E:\policy-eval\gap"),
                    help="parsed scenes and perception scores")
    ap.add_argument("--model", type=Path, default=PERCEPTION / "models" / "sketch7",
                    help="folder with ckpt_best.pt and tune_extract.json")
    ap.add_argument("--policy", type=Path, default=HERE / "models" / "e3_both_seed0" / "ckpt_best.pt")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--results", type=Path, default=None,
                    help="default: policy/results/gap/<policy folder>; with --limit, under --work")
    ap.add_argument("--workers", type=int, default=6, help="CPU workers for the episodes")
    ap.add_argument("--device", default=None, help="perception's device (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=None, help="first N scenes per set (smoke run, no commit)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dry = a.dry_run
    data, work = absolute(a.data_root), absolute(a.work)
    model, policy = absolute(a.model), absolute(a.policy)
    sets = a.sets.split(",")
    base = work / model.name
    name = policy.parent.name
    results = absolute(a.results) if a.results else (
        base / "smoke" / name if a.limit else HERE / "results" / "gap" / name)
    limit = ["--limit", a.limit] if a.limit else []
    device = ["--device", a.device] if a.device else []

    banner("1. inputs")
    needed = [model / "ckpt_best.pt", model / "tune_extract.json", policy]
    needed += [data / s / "dataset.json" for s in sets]
    missing = [p for p in needed if not p.exists()]
    for p in needed:
        print(f"  {'MISSING' if p in missing else 'ok     '} {shown(p)}")
    if missing and not dry:
        raise SystemExit("inputs are missing (the test sets come from perception/scripts/run_v7.py)")
    flags = tuned_flags(model)
    print("  tuned thresholds: " + " ".join(str(f) for f in flags))
    ckpt = model / "ckpt_best.pt"

    def need(s):
        n = scene_count(data / s)
        return min(a.limit, n) if a.limit else n

    banner("2. parse each set")
    for s in sets:
        out = base / "parsed" / s
        if not dry and parsed_count(out) >= need(s):
            print(f"  {s}: parsed, skipped")
            continue
        run([PY, "predict.py", "--ckpt", ckpt, "--dir", data / s, "--out", out, *flags, *limit, *device],
            dry, cwd=PERCEPTION)
        if not dry:
            (out / "_complete.txt").write_text(str(need(s)), encoding="utf-8")

    banner("3. score each parse")
    for s in sets:
        out = base / "perception" / s
        if not dry and scored_count(out / "per_scene.csv") >= need(s):
            print(f"  {s}: scored, skipped")
            continue
        run([PY, "evaluate.py", "--ckpt", ckpt, "--data", data / s, "--out", out, "--no-figures",
             *flags, *limit, *device], dry, cwd=PERCEPTION)

    banner("4. episodes: policy and orca_path, true and parsed, wrong map")
    run([PY, "gap.py", "--ckpt", policy, "--data-root", data, "--parsed", base / "parsed",
         "--perception", base / "perception", "--sets", ",".join(sets), "--out", results,
         "--workers", a.workers, *limit], dry)

    banner("5. commit and push")
    if a.limit:
        print(f"  --limit: a smoke run, nothing committed; results in {shown(results)}")
        return
    if a.no_push:
        print("  --no-push: no pull, commit only")
    else:
        run(["git", "pull", "--rebase", "--autostash"], dry, cwd=REPO)
    run(["git", "add", results], dry, cwd=REPO)
    if not dry and subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        print("  nothing new to commit")
        return
    run(["git", "commit", "-m", f"Behaviour gap: {name} on true vs parsed scenes"], dry, cwd=REPO)
    if not a.no_push:
        run(["git", "push"], dry, cwd=REPO)


if __name__ == "__main__":
    main()
