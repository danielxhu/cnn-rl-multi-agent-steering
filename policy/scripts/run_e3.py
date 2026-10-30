"""Experiment E3 (reward redesign) end to end on the training machine (DESIGN_E3.md §4).

    python scripts/run_e3.py              # everything; safe to Ctrl-C and run again
    python scripts/run_e3.py --dry-run    # print the commands only

Steps, each skipped when its output already exists (so a second run resumes):

1. check that E2 left what E3 compares against: the three suites,
   ``<eval>/greedy`` and the five ``<eval>/e2_gru_seed*`` (stop if not)
2. the nine runs ``<variant>_seed<k>`` into ``--runs``, seeds outer, variants
   inner, each with ``--resume`` when ``ckpt_last.pt`` exists (a finished run
   exits at once; the reward comes from the checkpoint then)
3. ``ckpt_best.pt`` of each run on ``test4`` and ``test_mixed`` -> ``<eval>/<run>``
4. ``aggregate.py --group variant`` over ``<eval>`` -> ``policy/results/e3``
   (tables, figures, ``winner.json`` chosen on validation)
5. ``git pull --rebase``, then add, commit and push ``policy/results/e3``
   (``--no-push`` skips pull and push; nothing to commit ends cleanly)

Defaults are the training machine's paths (E:\\policy-data, E:\\policy-runs,
E:\\policy-eval).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # policy/
REPO = HERE.parent
PY = sys.executable

# DESIGN_E3.md §2: every variant is gru; only the reward differs
VARIANTS = {
    "e3_timeout": ["--timeout-penalty", "25"],
    "e3_geo": ["--progress-metric", "geodesic"],
    "e3_both": ["--timeout-penalty", "25", "--progress-metric", "geodesic"],
}
BASELINE = "e2_gru"
E2_SEEDS = (0, 1, 2, 3, 4)


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


def train_cmd(a, variant, seed):
    out = a.runs / f"{variant}_seed{seed}"
    cmd = [PY, "train.py", "--memory", "gru", "--seed", seed, "--out", out,
           "--val-suite", a.data_root / "val", "--val-workers", a.eval_workers,
           "--workers", a.workers] + VARIANTS[variant]
    if a.total_steps is not None:
        cmd += ["--total-steps", a.total_steps]
    if (out / "ckpt_last.pt").exists():
        cmd.append("--resume")
    return cmd


def eval_cmd(a, run_name):
    return [PY, "evaluate.py", "--ckpt", a.runs / run_name / "ckpt_best.pt",
            "--suites", "test4,test_mixed", "--data-root", a.data_root,
            "--out", a.eval / run_name, "--workers", a.eval_workers]


def check_e2(a, dry):
    need = [a.data_root / s / "suite.json" for s in ("val", "test4", "test_mixed")]
    need += [a.eval / "greedy" / "metrics.json"]
    need += [a.eval / f"{BASELINE}_seed{s}" / "metrics.json" for s in E2_SEEDS]
    missing = [p for p in need if not p.exists()]
    for p in need:
        print(f"  {'ok     ' if p not in missing else 'MISSING'} {p}")
    if missing and not dry:
        raise SystemExit("E3 compares against E2: run scripts/run_e2.py first "
                         f"({len(missing)} missing, listed above)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path(r"E:\policy-data"))
    ap.add_argument("--runs", type=Path, default=Path(r"E:\policy-runs"))
    ap.add_argument("--eval", type=Path, default=Path(r"E:\policy-eval"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--total-steps", type=int, default=None, help="default: 50 M (config)")
    ap.add_argument("--workers", type=int, default=10, help="rollout processes per run")
    ap.add_argument("--eval-workers", type=int, default=6,
                    help="processes for validation / evaluation")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dry = a.dry_run
    results = HERE / "results" / "e3"

    banner("1. E2 outputs E3 compares against")
    check_e2(a, dry)

    runs = [(v, s) for s in a.seeds for v in a.variants]          # seeds outer
    banner("2. training runs")
    for v, s in runs:
        run(train_cmd(a, v, s), dry)

    banner("3. evaluation of each run")
    for v, s in runs:
        name = f"{v}_seed{s}"
        if (a.eval / name / "metrics.json").exists():
            print(f"  {name}: exists, skipped")
            continue
        run(eval_cmd(a, name), dry)

    banner("4. aggregate")
    run([PY, "aggregate.py", "--evals", a.eval, "--runs", a.runs, "--out", results,
         "--group", "variant", "--tag", "e3",
         "--variants", "greedy", BASELINE] + a.variants, dry)

    banner("5. commit and push")
    if a.no_push:
        print("  --no-push: no pull, commit only")
    else:
        run(["git", "pull", "--rebase", "--autostash"], dry, cwd=REPO)
    run(["git", "add", results], dry, cwd=REPO)
    if not dry and subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        print("  nothing new to commit")
        return
    run(["git", "commit", "-m", "E3: reward redesign results (timeout penalty, geodesic progress)"],
        dry, cwd=REPO)
    if not a.no_push:
        run(["git", "push"], dry, cwd=REPO)


if __name__ == "__main__":
    main()
