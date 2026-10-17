"""Experiment E2 end to end on the training machine (DESIGN.md §7).

    python scripts/run_e2.py              # everything; safe to Ctrl-C and run again
    python scripts/run_e2.py --dev        # 2 M-step trial runs: throughput and validation, no push
    python scripts/run_e2.py --dry-run    # print the commands only

Steps, each skipped when its output already exists (so a second run resumes):

1. the suites (``make_suites.py``) into ``--data-root``
2. greedy on ``test4`` and ``test_mixed`` -> ``<eval>/greedy``
3. the training runs ``e2_<memory>_seed<k>`` into ``--runs``, one after another,
   each with ``--resume`` when ``ckpt_last.pt`` exists (a finished run exits at once)
4. ``ckpt_best.pt`` of each run on both suites -> ``<eval>/<run>``
5. ``aggregate.py`` -> ``policy/results/e2``
6. ``git add policy/results/e2``, commit, push (``--no-push`` skips the push)

Defaults are the training machine's paths (E:\\policy-data, E:\\policy-runs,
E:\\policy-eval).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # policy/
REPO = HERE.parent
PY = sys.executable


def banner(text):
    print(f"\n=== {text}", flush=True)


def run(cmd, dry, cwd=HERE) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    rc = subprocess.run([str(c) for c in cmd], cwd=cwd).returncode
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {' '.join(str(c) for c in cmd)}")
    return rc


def train_cmd(a, memory, seed, name, total, val_dir):
    out = a.runs / name
    cmd = [PY, "train.py", "--memory", memory, "--seed", seed, "--out", out,
           "--val-suite", val_dir, "--val-workers", a.eval_workers, "--workers", a.workers]
    if total is not None:
        cmd += ["--total-steps", total]
    if (out / "ckpt_last.pt").exists():
        cmd.append("--resume")
    return cmd


def eval_cmd(a, ckpt, out):
    return [PY, "evaluate.py", "--ckpt", ckpt, "--suites", "test4,test_mixed",
            "--data-root", a.data_root, "--out", out, "--workers", a.eval_workers]


def dev_report(a, names, dry):
    if dry:
        return
    for n in names:
        try:
            log = list(csv.DictReader(open(a.runs / n / "log.csv")))
            val = list(csv.DictReader(open(a.runs / n / "val.csv")))
        except OSError:
            continue
        rates = [float(r["agent_steps_per_s"]) for r in log[1:]] or [0.0]
        print(f"  {n}: {sum(rates) / len(rates):,.0f} agent-steps/s (mean of iterations 2..), "
              f"validation success {float(val[-1]['success_rate']):.3f} after "
              f"{int(float(val[-1]['agent_steps'])):,} steps" if val else f"  {n}: no validation", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path(r"E:\policy-data"))
    ap.add_argument("--runs", type=Path, default=Path(r"E:\policy-runs"))
    ap.add_argument("--eval", type=Path, default=Path(r"E:\policy-eval"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--memories", nargs="+", default=["mlp", "gru"])
    ap.add_argument("--total-steps", type=int, default=None, help="default: 50 M (config)")
    ap.add_argument("--workers", type=int, default=10, help="rollout processes per run")
    ap.add_argument("--eval-workers", type=int, default=6, help="processes for validation / evaluation")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dry = a.dry_run
    results = HERE / "results" / "e2"

    banner("1. suites")
    only = ["--only", "val"] if a.dev else []
    run([PY, "scripts/make_suites.py", "--data-root", a.data_root] + only, dry)

    if a.dev:
        banner("dev: 2 M agent-steps per memory variant")
        names = []
        for m in a.memories:
            name = f"e2_{m}_dev"
            names.append(name)
            run(train_cmd(a, m, 0, name, 2_000_000, a.data_root / "val"), dry)
        dev_report(a, names, dry)
        print("\nIf agent-steps/s makes 50 M take more than ~4 h per GRU run, "
              "rerun without --dev and lower --total-steps.")
        return

    banner("2. greedy baseline")
    if (a.eval / "greedy" / "metrics.json").exists():
        print("  exists, skipped")
    else:
        run([PY, "evaluate.py", "--greedy", "--suites", "test4,test_mixed", "--data-root",
             a.data_root, "--out", a.eval / "greedy", "--workers", a.eval_workers], dry)

    runs = [(m, s, f"e2_{m}_seed{s}") for s in a.seeds for m in a.memories]
    banner("3. training runs")
    for m, s, name in runs:
        run(train_cmd(a, m, s, name, a.total_steps, a.data_root / "val"), dry)

    banner("4. evaluation of each run")
    for m, s, name in runs:
        if (a.eval / name / "metrics.json").exists():
            print(f"  {name}: exists, skipped")
            continue
        run(eval_cmd(a, a.runs / name / "ckpt_best.pt", a.eval / name), dry)

    banner("5. aggregate")
    run([PY, "aggregate.py", "--evals", a.eval, "--runs", a.runs, "--out", results], dry)

    banner("6. commit and push")
    run(["git", "add", results], dry, cwd=REPO)
    run(["git", "commit", "-m", "E2: baseline policy results (mlp vs gru, greedy)"], dry, cwd=REPO)
    if a.no_push:
        print("  --no-push: not pushing")
    else:
        run(["git", "push"], dry, cwd=REPO)


if __name__ == "__main__":
    main()
