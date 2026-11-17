"""Experiment E7 (wall memory) end to end on the training machine (DESIGN_E7.md §8).

    python scripts/run_e7.py               # pilot, then stop and report the speed
    python scripts/run_e7.py --full        # pilot (skipped when done), full runs, eval, aggregate, push
    python scripts/run_e7.py --dry-run     # print the commands only (add --full for every step)
    python scripts/run_e7.py --train-only e7_map_seed0 e7_map_seed1   # this machine's share only

Steps, each skipped when its output already exists (so a second run resumes):

1. check the inputs: the three suites and ``<runs>/e3_both_seed{0,1,2}/ckpt_best.pt``
2. pilot: ``e7_map`` and ``e7_longgru``, seed 0, ``--pilot-steps`` (2 M) agent-steps
   each into ``<runs>/pilot_e7/``; print agent-steps/s, the projected hours of a
   50 M run, and the pilot ``e7_map``'s loop rate and exit switches on
   ``test4`` (``<eval>/e7_pilot``). Without
   ``--full`` the driver stops here.
3. full runs: ``e7_map_seed{0,1,2}``, ``e7_longgru_seed0`` (``--resume`` when
   ``ckpt_last.pt`` exists)
4. evaluation into ``<eval>/e7/``: the four runs; ``e3_both_seed{0,1,2}`` and
   both ORCA baselines re-evaluated (for the loop metric); ``e7_map`` with
   odometry noise ``low`` and ``high``
5. ``aggregate_e7.py`` -> ``policy/results/e7`` (``--check-against <eval>``
   compares the re-evaluated baselines with E3's evaluations)
6. ``git pull --rebase``, then add, commit and push ``policy/results/e7``
   (``--no-push`` skips pull and push)

Defaults are the training machine's paths (E:\\policy-data, E:\\policy-runs,
E:\\policy-eval). On the Mac, a drive path stays a Windows path, so
``--dry-run`` prints exactly what the training machine will run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

HERE = Path(__file__).resolve().parent.parent          # policy/
REPO = HERE.parent

REWARD_FLAGS = ["--timeout-penalty", "25", "--progress-metric", "geodesic"]   # e3_both
VARIANTS = {                                           # DESIGN_E7.md §2
    "e7_map": REWARD_FLAGS + ["--wallmap", "on"],
    "e7_longgru": REWARD_FLAGS + ["--L", "128"],
}
SEEDS = {"e7_map": (0, 1, 2), "e7_longgru": (0,)}
BASELINE = "e3_both"
BASE_SEEDS = (0, 1, 2)
SUITES = ("val", "test4", "test_mixed")
NOISE = ("low", "high")
FULL_STEPS = 50_000_000
SLOW = 1200                                            # agent-steps/s: below it, §4 says every = 8


def root_path(text):
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        return PureWindowsPath(text)
    return Path(text)


def exists(p) -> bool:
    return isinstance(p, Path) and p.exists()


def python(a) -> str:
    return "python" if os.name != "nt" and isinstance(a.runs, PureWindowsPath) else sys.executable


def banner(text):
    print(f"\n=== {text}", flush=True)


def run(cmd, dry, cwd=HERE) -> int:
    cmd = [str(c) for c in cmd]
    print("  $ " + subprocess.list2cmdline(cmd), flush=True)
    if dry:
        return 0
    child = subprocess.Popen(cmd, cwd=cwd)
    try:
        rc = child.wait()
    except KeyboardInterrupt:              # subprocess.run would kill train.py while it saves
        rc = child.wait()
    if rc == 130:
        raise SystemExit("interrupted: run the same command again to resume")
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {subprocess.list2cmdline(cmd)}")
    return rc


def finished(out, steps) -> bool:
    """A run is done when its last log row reached the budget."""
    log = out / "log.csv"
    if not exists(log):
        return False
    rows = list(csv.DictReader(open(log, newline="")))
    return bool(rows) and int(rows[-1]["agent_steps"]) >= steps


def train_cmd(a, variant, seed, out, steps=None):
    cmd = [python(a), "train.py", "--memory", "gru", "--seed", seed, "--out", out,
           "--val-suite", a.data_root / "val", "--val-workers", a.eval_workers,
           "--workers", a.workers] + VARIANTS[variant]
    if variant == "e7_map" and a.wallmap_every is not None:
        cmd += ["--wallmap-every", a.wallmap_every]
    if steps is not None:
        cmd += ["--total-steps", steps]
    if exists(out / "ckpt_last.pt"):
        cmd.append("--resume")
    return cmd


def eval_cmd(a, ckpt, out, extra=()):
    return [python(a), "evaluate.py", "--ckpt", ckpt, "--suites", "test4,test_mixed",
            "--data-root", a.data_root, "--out", out, "--workers", a.eval_workers] + list(extra)


def check_inputs(a, dry):
    need = [a.data_root / s / "suite.json" for s in SUITES]
    need += [a.runs / f"{BASELINE}_seed{s}" / "ckpt_best.pt" for s in BASE_SEEDS]
    missing = [p for p in need if not exists(p)]
    for p in need:
        print(f"  {'ok     ' if p not in missing else 'MISSING'} {p}")
    if missing and not dry:
        raise SystemExit(f"E7 needs E3's suites and e3_both checkpoints ({len(missing)} missing)")


def pilot_report(a):
    """Speed and exit switches of the pilot runs (DESIGN_E7.md §4)."""
    for v in VARIANTS:
        log = a.runs / "pilot_e7" / f"{v}_seed0" / "log.csv"
        if not exists(log):
            continue
        rows = list(csv.DictReader(open(log, newline="")))
        rate = [float(r["agent_steps_per_s"]) for r in rows[len(rows) // 4:]]   # skip warm-up
        if not rate:
            continue
        sps = sum(rate) / len(rate)
        hours = FULL_STEPS / sps / 3600
        print(f"  {v:11s} {sps:7.0f} agent-steps/s  ->  {hours:4.1f} h per 50 M-step run")
        if v == "e7_map" and sps < SLOW:
            print(f"  e7_map is below {SLOW} agent-steps/s: DESIGN_E7.md §4 says run the full "
                  "runs with --wallmap-every 8 (and note it in §10)")
    m = a.eval / "e7_pilot" / "e7_map_seed0" / "metrics.json"
    if exists(m):
        o = json.loads(m.read_text(encoding="utf-8"))["suites"]["test4"]["overall"]
        print(f"  pilot e7_map on test4: success {o['success_rate']:.3f}, loop rate "
              f"{o['loop_rate']:.3f}, exit switches per agent {o['exit_switches']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=root_path, default=root_path(r"E:\policy-data"))
    ap.add_argument("--runs", type=root_path, default=root_path(r"E:\policy-runs"))
    ap.add_argument("--eval", type=root_path, default=root_path(r"E:\policy-eval"))
    ap.add_argument("--full", action="store_true", help="go past the pilot")
    ap.add_argument("--pilot-steps", type=int, default=2_000_000)
    ap.add_argument("--total-steps", type=int, default=None, help="full runs (default 50 M)")
    ap.add_argument("--wallmap-every", type=int, default=None,
                    help="height-field update interval of e7_map (default 4, §1.3)")
    ap.add_argument("--workers", type=int, default=10, help="rollout processes per run")
    ap.add_argument("--eval-workers", type=int, default=6)
    ap.add_argument("--train-only", nargs="+", default=None, metavar="RUN",
                    help="train only these full runs (e.g. e7_map_seed0), then stop: "
                         "for splitting the runs between machines (AZURE_E7.md)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dry = a.dry_run
    ev = a.eval / "e7"
    results = Path("policy") / "results" / "e7"          # relative to the repository

    banner("1. inputs")
    check_inputs(a, dry)

    if a.train_only:
        names = {f"{v}_seed{s}" for v in VARIANTS for s in SEEDS[v]}
        bad = sorted(set(a.train_only) - names)
        if bad:
            raise SystemExit(f"--train-only: unknown runs {bad}; choose from {sorted(names)}")
        banner("3. full runs (--train-only)")
        for v in VARIANTS:
            for s in SEEDS[v]:
                if f"{v}_seed{s}" in a.train_only:
                    run(train_cmd(a, v, s, a.runs / f"{v}_seed{s}", a.total_steps), dry)
        print("\n  --train-only: done. When every run is in place, run: python scripts/run_e7.py --full")
        return

    banner("2. pilot")
    pilot = a.runs / "pilot_e7"
    for v in VARIANTS:
        out = pilot / f"{v}_seed0"
        if finished(out, a.pilot_steps):
            print(f"  {out.name}: pilot done, skipped")
            continue
        run(train_cmd(a, v, 0, out, a.pilot_steps), dry)
    pe = a.eval / "e7_pilot" / "e7_map_seed0"          # outside <eval>/e7: not aggregated
    if exists(pe / "metrics.json"):
        print("  pilot evaluation: exists, skipped")
    else:
        run([python(a), "evaluate.py", "--ckpt", pilot / "e7_map_seed0" / "ckpt_best.pt",
             "--suites", "test4", "--data-root", a.data_root, "--out", pe,
             "--workers", a.eval_workers], dry)
    if not dry:
        pilot_report(a)
    if not a.full:
        print("\n  pilot finished. Check the speed above, then run:\n"
              "    python scripts/run_e7.py --full")
        return

    runs = [(v, s) for v in VARIANTS for s in SEEDS[v]]
    banner("3. full runs")
    for v, s in runs:
        out = a.runs / f"{v}_seed{s}"
        run(train_cmd(a, v, s, out, a.total_steps), dry)

    banner("4. evaluation")
    jobs = [(a.runs / f"{v}_seed{s}" / "ckpt_best.pt", ev / f"{v}_seed{s}", ()) for v, s in runs]
    jobs += [(a.runs / f"{BASELINE}_seed{s}" / "ckpt_best.pt", ev / f"{BASELINE}_seed{s}", ())
             for s in BASE_SEEDS]
    jobs += [(a.runs / f"e7_map_seed{s}" / "ckpt_best.pt", ev / f"e7_map_seed{s}_odo_{n}",
              ("--odo-noise", n)) for s in SEEDS["e7_map"] for n in NOISE]
    for ckpt, out, extra in jobs:
        if exists(out / "metrics.json"):
            print(f"  {out.name}: exists, skipped")
            continue
        run(eval_cmd(a, ckpt, out, extra), dry)
    for mode in ("straight", "path"):
        out = ev / f"orca_{mode}"
        if exists(out / "metrics.json"):
            print(f"  {out.name}: exists, skipped")
            continue
        run([python(a), "evaluate.py", "--orca", mode, "--suites", "test4,test_mixed",
             "--data-root", a.data_root, "--out", out, "--workers", a.eval_workers], dry)

    banner("5. aggregate")
    run([python(a), "aggregate_e7.py", "--evals", ev, "--out", Path("results") / "e7",
         "--check-against", a.eval], dry)

    banner("6. commit and push")
    if a.no_push:
        print("  --no-push: no pull, commit only")
    else:
        run(["git", "pull", "--rebase", "--autostash"], dry, cwd=REPO)
    run(["git", "add", results], dry, cwd=REPO)
    if not dry and subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        print("  nothing new to commit")
        return
    run(["git", "commit", "-m", "E7: wall memory results"], dry, cwd=REPO)
    if not a.no_push:
        run(["git", "push"], dry, cwd=REPO)


if __name__ == "__main__":
    main()
