"""ORCA baseline on the training machine: evaluate, re-aggregate E2 / E3, push (DESIGN_ORCA.md §3).

    python scripts/run_orca.py              # everything; safe to Ctrl-C and run again
    python scripts/run_orca.py --dry-run    # print the commands only

Steps, each skipped when its output already exists:

1. check that the E2 suites ``test4`` and ``test_mixed`` exist (stop if not)
2. ``evaluate.py --orca straight`` and ``--orca path`` on both suites
   -> ``<eval>/orca_straight``, ``<eval>/orca_path``
3. re-aggregate E2 with the ORCA columns -> ``policy/results/e2``
   (``--variants greedy orca_straight orca_path mlp gru``; only the
   ``greedy``, ``orca_*`` and ``e2_*_seed<k>`` evaluations, so E3's gru runs
   stay out of E2's gru column)
4. if E3 evaluations exist, re-aggregate E3 with the ORCA columns
   -> ``policy/results/e3``
5. ``git pull --rebase --autostash``, add, commit and push ``policy/results``
   (``--no-push`` skips the pull and the push; nothing to commit ends cleanly)

Defaults are the training machine's paths (E:\\policy-data, E:\\policy-runs,
E:\\policy-eval). ORCA needs no training and no GPU.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # policy/
REPO = HERE.parent
PY = sys.executable
MODES = ("straight", "path")
E2_INCLUDE = r"^(greedy|orca_(straight|path)|e2_(mlp|gru)_seed\d+)$"
E3_VARIANTS = ["greedy", "orca_straight", "orca_path", "e2_gru", "e3_timeout", "e3_geo", "e3_both"]


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path(r"E:\policy-data"))
    ap.add_argument("--runs", type=Path, default=Path(r"E:\policy-runs"))
    ap.add_argument("--eval", type=Path, default=Path(r"E:\policy-eval"))
    ap.add_argument("--eval-workers", type=int, default=6)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dry = a.dry_run
    results = HERE / "results"

    banner("1. suites")
    missing = [s for s in ("test4", "test_mixed") if not (a.data_root / s / "suite.json").exists()]
    for s in ("test4", "test_mixed"):
        print(f"  {'MISSING' if s in missing else 'ok     '} {a.data_root / s / 'suite.json'}")
    if missing and not dry:
        raise SystemExit("the E2 suites are missing: run scripts/run_e2.py (or make_suites.py) first")

    banner("2. ORCA evaluations")
    for mode in MODES:
        out = a.eval / f"orca_{mode}"
        if (out / "metrics.json").exists():
            print(f"  orca_{mode}: exists, skipped")
            continue
        run([PY, "evaluate.py", "--orca", mode, "--suites", "test4,test_mixed",
             "--data-root", a.data_root, "--out", out, "--workers", a.eval_workers], dry)

    banner("3. re-aggregate E2 with the ORCA columns")
    run([PY, "aggregate.py", "--evals", a.eval, "--runs", a.runs, "--out", results / "e2",
         "--variants", "greedy", "orca_straight", "orca_path", "mlp", "gru",
         "--include", E2_INCLUDE], dry)

    banner("4. re-aggregate E3 with the ORCA columns")
    has_e3 = any(a.eval.glob("e3_*_seed*/metrics.json")) if a.eval.exists() else False
    if has_e3 or dry:
        if not has_e3:
            print("  (no E3 evaluations here; on the training machine this runs when they exist)")
        run([PY, "aggregate.py", "--evals", a.eval, "--runs", a.runs, "--out", results / "e3",
             "--group", "variant", "--tag", "e3", "--variants"] + E3_VARIANTS, dry)
    else:
        print("  no E3 evaluations yet, skipped")

    banner("5. commit and push")
    if a.no_push:
        print("  --no-push: no pull, commit only")
    else:
        run(["git", "pull", "--rebase", "--autostash"], dry, cwd=REPO)
    run(["git", "add", results], dry, cwd=REPO)
    if not dry and subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        print("  nothing new to commit")
        return
    run(["git", "commit", "-m", "ORCA baseline: orca_straight / orca_path columns in E2 (and E3)"],
        dry, cwd=REPO)
    if not a.no_push:
        run(["git", "push"], dry, cwd=REPO)


if __name__ == "__main__":
    main()
