"""Experiment E4 (leave-one-layout-out) end to end on the training machine (DESIGN_E4.md §4).

    python scripts/run_e4.py                    # first pass: 10 folds, seed 0; safe to Ctrl-C and run again
    python scripts/run_e4.py --layouts doorway  # one fold (--layouts names the HELD-OUT layouts)
    python scripts/run_e4.py --seeds 1 2        # more seeds later; finished folds are skipped
    python scripts/run_e4.py --dry-run          # print the commands only
    python scripts/run_e4.py --dev              # tiny trial on small suites in a temp folder: no E: paths, no git

Steps, each skipped when its output already exists (so a second run resumes):

1. check what E4 needs: the E3 winner's reward (``results/e3/winner.json``), the
   suites, and the baseline evaluations ``<eval>/e3_both_seed<k>`` (stop if not)
2. per seed, per held-out layout: train ``e4_holdout_<layout>_seed<k>`` on the
   other nine into ``--runs`` (``--resume`` when ``ckpt_last.pt`` exists, nothing
   when it is finished), then evaluate ``ckpt_best.pt`` on ``test4`` and
   ``test_mixed`` -> ``<eval>/e4/<run>``, and print the held-out line
3. ``aggregate_e4.py`` over every fold evaluated so far -> ``policy/results/e4``
4. ``git pull --rebase --autostash``, then add, commit and push
   ``policy/results/e4`` (``--no-push`` skips the pull and the push; nothing to
   commit ends cleanly)

Fold evaluations live one level down, in ``<eval>/e4``: ``aggregate.py`` only
looks one level deep, so E2 and E3 tables never pick them up.

Defaults are the training machine's paths (E:\\policy-data, E:\\policy-runs,
E:\\policy-eval). Budget: about 3 h per fold on that machine.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath

HERE = Path(__file__).resolve().parent.parent          # policy/
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from policyconfig import LAYOUTS_E2  # noqa: E402

PREFIX = "e4_holdout"
WINNER = "e3_both"                                     # DESIGN_E4.md §2
REWARD = {"timeout_penalty": 25.0, "progress_metric": "geodesic"}
SUITES = ("val", "test4", "test_mixed")
DEV_STEPS = 20_000
DEV_LAYOUTS = ["open", "doorway"]


def root_path(text):
    """A drive path stays a Windows path on any OS, so --dry-run on the Mac prints
    exactly what the training machine will run."""
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


def train_layouts(held: str) -> list:
    return [l for l in LAYOUTS_E2 if l != held]


def run_name(held: str, seed: int) -> str:
    return f"{PREFIX}_{held}_seed{seed}"


def train_cmd(a, held, seed):
    out = a.runs / run_name(held, seed)
    cmd = [python(a), "train.py", "--memory", "gru", "--seed", seed, "--out", out,
           "--layouts", ",".join(train_layouts(held)),
           "--val-suite", a.data_root / "val", "--val-workers", a.eval_workers,
           "--workers", a.workers,
           "--timeout-penalty", f"{REWARD['timeout_penalty']:g}",
           "--progress-metric", REWARD["progress_metric"]]
    if a.total_steps is not None:
        cmd += ["--total-steps", a.total_steps]
    if exists(out / "ckpt_last.pt"):
        cmd.append("--resume")
    return cmd


def eval_cmd(a, held, seed):
    name = run_name(held, seed)
    return [python(a), "evaluate.py", "--ckpt", a.runs / name / "ckpt_best.pt",
            "--suites", "test4,test_mixed", "--data-root", a.data_root,
            "--out", a.eval / "e4" / name, "--workers", a.eval_workers]


def fold_state(a, held, seed) -> str:
    """``new``, ``partial`` or ``finished``, from ``ckpt_last.pt``; a checkpoint that
    is not this fold stops the driver instead of being resumed."""
    last = a.runs / run_name(held, seed) / "ckpt_last.pt"
    if not exists(last):
        return "new"
    import torch
    ck = torch.load(last, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    wrong = []
    if set(cfg["env"]["layouts"]) != set(train_layouts(held)):
        wrong.append(f"layouts {cfg['env']['layouts']}")
    if cfg.get("reward") != REWARD:
        wrong.append(f"reward {cfg.get('reward')}")
    if cfg["seed"] != seed:
        wrong.append(f"seed {cfg['seed']}")
    if wrong:
        raise SystemExit(f"{last} is not fold {held} seed {seed}: " + ", ".join(wrong))
    budget = a.total_steps or cfg["ppo"]["total_steps"]
    return "finished" if ck["agent_steps"] >= budget else "partial"


def eval_done(a, held, seed) -> bool:
    """A complete ``metrics.json`` at least as new as the checkpoint it evaluated."""
    name = run_name(held, seed)
    m = a.eval / "e4" / name / "metrics.json"
    if not exists(m):
        return False
    try:
        suites = json.loads(m.read_text(encoding="utf-8"))["suites"]
    except (ValueError, KeyError):
        return False
    if not {"test4", "test_mixed"} <= set(suites):
        return False
    best = a.runs / name / "ckpt_best.pt"
    return not exists(best) or best.stat().st_mtime <= m.stat().st_mtime


def report_fold(a, held, seed):
    if not eval_done(a, held, seed):
        return
    m = a.eval / "e4" / run_name(held, seed) / "metrics.json"
    suites = json.loads(m.read_text(encoding="utf-8"))["suites"]
    for s in ("test4", "test_mixed"):
        v = suites[s]["per_layout"].get(held)
        if v:
            print(f"  held-out {held} seed {seed} on {s}: success {v['success_rate']:.3f}  "
                  f"collision {v['collision_rate']:.3f}  timeout {v['timeout_rate']:.3f}", flush=True)


def run_fold(a, held, seed, dry):
    name = run_name(held, seed)
    banner(f"fold {name}")
    state = fold_state(a, held, seed)
    done = eval_done(a, held, seed)
    if state == "finished":
        print(f"  {name}: training finished, skipped")
    else:
        run(train_cmd(a, held, seed), dry)
    if done and state == "finished":
        print(f"  {name}: evaluation exists, skipped")
    else:
        run(eval_cmd(a, held, seed), dry)
    if not dry:
        report_fold(a, held, seed)


def winner_seeds() -> list:
    win = json.loads((HERE / "results" / "e3" / "winner.json").read_text(encoding="utf-8"))
    if win["variant"] != WINNER:
        raise SystemExit(f"results/e3/winner.json names {win['variant']}, E4 is written for "
                         f"{WINNER}: update WINNER and REWARD in run_e4.py and DESIGN_E4.md")
    return win["seeds"]


def check_inputs(a, seeds, dry):
    """The suites and the all-layouts baseline; stops when one is missing."""
    need = [a.data_root / s / "suite.json" for s in SUITES]
    for k in seeds:
        d = a.eval / f"{WINNER}_seed{k}"
        need += [d / "metrics.json", d / "per_scene.csv"]
    missing = [p for p in need if not exists(p)]
    for p in need:
        print(f"  {'ok     ' if p not in missing else 'MISSING'} {p}")
    if missing and not dry:
        raise SystemExit("E4 needs the suites and the E3 evaluations: run scripts/run_e2.py and "
                         f"scripts/run_e3.py first ({len(missing)} missing, listed above)")
    val = a.data_root / "val" / "suite.json"
    if exists(val):
        lacking = set(LAYOUTS_E2) - set(json.loads(val.read_text(encoding="utf-8"))["layouts"])
        if lacking:
            raise SystemExit(f"{val} has no scenes of {sorted(lacking)}")


def has_staged() -> bool:
    return subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode != 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=root_path, default=None, help=r"default E:\policy-data")
    ap.add_argument("--runs", type=root_path, default=None, help=r"default E:\policy-runs")
    ap.add_argument("--eval", type=root_path, default=None, help=r"default E:\policy-eval")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="default 0 (first pass); seeds outer, folds inner")
    ap.add_argument("--layouts", nargs="+", default=None, choices=list(LAYOUTS_E2),
                    help="the layouts to HOLD OUT, one fold each (default: all ten)")
    ap.add_argument("--total-steps", type=int, default=None, help="default: 50 M (config)")
    ap.add_argument("--workers", type=int, default=None, help="rollout processes per run (10)")
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="processes for validation / evaluation (6)")
    ap.add_argument("--dev", action="store_true",
                    help=f"tiny trial: {DEV_STEPS:,} steps, folds {DEV_LAYOUTS}, small suites, "
                         "temp folder, no git")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dev_root = Path(tempfile.gettempdir()) / "policy-e4-dev"
    for attr, machine, dev in (("data_root", r"E:\policy-data", "data"),
                               ("runs", r"E:\policy-runs", "runs"),
                               ("eval", r"E:\policy-eval", "eval")):
        if getattr(a, attr) is None:
            setattr(a, attr, dev_root / dev if a.dev else root_path(machine))
    a.layouts = a.layouts or (DEV_LAYOUTS if a.dev else list(LAYOUTS_E2))
    a.workers = a.workers if a.workers is not None else (2 if a.dev else 10)
    a.eval_workers = a.eval_workers if a.eval_workers is not None else (0 if a.dev else 6)
    if a.dev and a.total_steps is None:
        a.total_steps = DEV_STEPS
    return a


def main(argv=None):
    a = parse_args(argv)
    dry = a.dry_run
    folds = [(h, s) for s in a.seeds for h in a.layouts]              # seeds outer
    print(f"E4: {len(folds)} fold run(s); commands run from policy/, git from the repository root")

    banner("1. inputs")
    seeds = winner_seeds()
    print(f"  ok      results/e3/winner.json: {WINNER}, reward {REWARD}")
    if a.dev:
        run([python(a), "scripts/make_suites.py", "--data-root", a.data_root, "--per-layout", 3], dry)
    else:
        check_inputs(a, seeds, dry)

    banner(f"2. folds: held-out {a.layouts}, seeds {a.seeds}")
    for held, seed in folds:
        run_fold(a, held, seed, dry)

    banner("3. aggregate")
    results = a.eval.parent / "results" if a.dev else "results/e4"
    run([python(a), "aggregate_e4.py", "--evals", a.eval, "--runs", a.runs, "--out", results], dry)
    if a.dev:
        print(f"\ndev run finished: tables in {results}. No git step.")
        return

    banner("4. commit and push")
    if a.no_push:
        print("  --no-push: no pull, commit only")
    else:
        run(["git", "pull", "--rebase", "--autostash"], dry, cwd=REPO)
    run(["git", "add", "policy/results/e4"], dry, cwd=REPO)
    if not dry and not has_staged():
        print("  nothing new to commit")
        return
    run(["git", "commit", "-m", "E4: leave-one-layout-out results"], dry, cwd=REPO)
    if not a.no_push:
        run(["git", "push"], dry, cwd=REPO)


if __name__ == "__main__":
    main()
