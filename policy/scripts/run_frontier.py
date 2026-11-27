"""Frontier comparison end to end on AutoDL (DESIGN_FRONTIER.md §5).

    python scripts/run_frontier.py --step setup          # check jax / GPU / torch, build banks and val suite
    python scripts/run_frontier.py --step pilot          # 3 M steps of each new arm, print speed and cost
    python scripts/run_frontier.py --step frontier --seeds 0,1,2
    python scripts/run_frontier.py --step compromise --seeds 0,1,2 --workers 30
    python scripts/run_frontier.py --step eval           # export, zero-shot suites, statistics
    python scripts/run_frontier.py --step all --dry-run  # print every command

The compromise arm uses e3_both's rollout size (10 processes x 8 envs), so
one seed needs ~11 CPU cores; run the seeds side by side in separate
terminals (``--step compromise --seeds 1``) on an instance with >= 32 cores.
Every step skips finished outputs and resumes unfinished runs (``--resume``),
so the same command can be repeated after an interruption or a new instance.

Paths (under ``--root``, default ``/root/autodl-tmp/frontier``, AutoDL's data
disk): ``jaxbank/{train,val}.npz``, ``suites/val``, ``runs/<arm>_seed<k>/``,
``models/<arm>_seed<k>/ckpt_best.pt`` (frontier exports), ``zeroshot/``.
Results that go into git are copied to ``policy/results/frontier/``.

Arms (3 seeds each): ``current`` = the committed ``e7_map`` checkpoints with
the orbit rule; ``compromise`` = ``e3_both`` flags + ``--plr on`` (torch, 50 M);
``frontier`` = ``jaxsim/train_ippo.py`` (1 B, params kept at 50 M / 200 M /
500 M). All evaluated with the safety filter (margin 0.3) on
test4, novel and crowd4-16.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
REPO = POLICY.parent
SUITES = "test4,novel,crowd4,crowd8,crowd12,crowd16"
E3_FLAGS = ["--timeout-penalty", "25", "--progress-metric", "geodesic"]
COMPROMISE_STEPS = 50_000_000
FRONTIER_STEPS = 1_000_000_000
MILESTONES = (50_000_000, 200_000_000, 500_000_000)
PILOT_STEPS = 3_000_000


def sh(cmd, dry, cwd=POLICY):
    cmd = [str(c) for c in cmd]
    print("  $ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=cwd)


def last_steps(run: Path) -> int:
    log = run / "log.csv"
    if not log.exists():
        return 0
    rows = list(csv.DictReader(open(log)))
    return int(rows[-1]["agent_steps"]) if rows else 0


def frontier_cmd(a, seed, out, steps):
    cmd = [sys.executable, "jaxsim/train_ippo.py", "--bank", a.root / "jaxbank" / "train.npz",
           "--val-bank", a.root / "jaxbank" / "val.npz", "--out", out, "--seed", seed,
           "--total-steps", steps, "--envs", a.envs,
           "--save-at", ",".join(str(m) for m in MILESTONES)]
    if (out / "state.pkl").exists():
        cmd.append("--resume")
    return cmd


def compromise_cmd(a, seed, out, steps):
    cmd = [sys.executable, "train.py", "--memory", "gru", "--seed", seed, "--out", out,
           "--val-suite", a.root / "suites" / "val", "--val-workers", a.eval_workers,
           "--W", 10, "--E", 8, "--total-steps", steps, "--plr", "on"] + E3_FLAGS   # as e3_both
    if (out / "ckpt_last.pt").exists():
        cmd.append("--resume")
    return cmd


def step_setup(a):
    print("== setup")
    if not a.dry_run:
        import jax
        print("  jax", jax.__version__, "devices:", jax.devices())
        import torch
        print("  torch", torch.__version__)
    bank = a.root / "jaxbank"
    if not (bank / "train.npz").exists():
        sh([sys.executable, "jaxsim/bank.py", "--n", a.bank, "--out", bank / "train.npz",
            "--workers", a.workers], a.dry_run)
    if not (bank / "val.npz").exists():
        sh([sys.executable, "jaxsim/bank.py", "--n", 1024, "--bank-seed", 7,
            "--out", bank / "val.npz", "--workers", a.workers], a.dry_run)
    if not (a.root / "suites" / "val" / "suite.json").exists():
        sh([sys.executable, "scripts/make_suites.py", "--data-root", a.root / "suites",
            "--only", "val"], a.dry_run)


def step_pilot(a):
    print("== pilot")
    for arm, fn in (("frontier", frontier_cmd), ("compromise", compromise_cmd)):
        out = a.root / "runs" / f"pilot_{arm}"
        if last_steps(out) < PILOT_STEPS:
            sh(fn(a, 0, out, PILOT_STEPS), a.dry_run)
        if a.dry_run:
            continue
        rows = list(csv.DictReader(open(out / "log.csv")))
        sps = sum(float(r["agent_steps_per_s"]) for r in rows[-10:]) / min(10, len(rows))
        full = FRONTIER_STEPS if arm == "frontier" else COMPROMISE_STEPS
        print(f"  {arm}: {sps:,.0f} agent-steps/s -> one seed of {full:.0e} steps "
              f"~ {full / sps / 3600:.1f} h", flush=True)


def step_train(a, arm):
    print(f"== {arm}")
    fn, steps = ((frontier_cmd, FRONTIER_STEPS) if arm == "frontier"
                 else (compromise_cmd, COMPROMISE_STEPS))
    for s in a.seeds.split(","):
        out = a.root / "runs" / f"{arm}_seed{s}"
        if last_steps(out) >= steps:
            print(f"  {out.name}: done")
            continue
        sh(fn(a, s, out, steps), a.dry_run)


def step_eval(a):
    print("== eval")
    models, runs = a.root / "models", {}
    for s in a.seeds.split(","):
        runs[f"e3_both_seed{s}"] = ["--model", POLICY / "models" / f"e3_both_seed{s}" / "ckpt_best.pt"]
        runs[f"current_seed{s}"] = ["--model", POLICY / "models" / f"e7_map_seed{s}" / "ckpt_best.pt",
                                    "--exit-rule", "orbit"]
        comp = a.root / "runs" / f"compromise_seed{s}" / "ckpt_best.pt"
        runs[f"compromise_seed{s}"] = ["--model", comp]
        fr = a.root / "runs" / f"frontier_seed{s}"
        for tag, src in [("frontier", fr / "params_best.npz")] + \
                [(f"frontier{m // 1_000_000}M", fr / f"params_{m}.npz") for m in MILESTONES]:
            dst = models / f"{tag}_seed{s}" / "ckpt_best.pt"
            if (a.dry_run or src.exists()) and not dst.exists():
                sh([sys.executable, "jaxsim/export_ckpt.py", src, dst], a.dry_run)
            if a.dry_run or dst.exists():
                runs[f"{tag}_seed{s}"] = ["--model", dst]
    out = a.root / "zeroshot" / "runs"
    for name, args in runs.items():
        if (out / f"{name}.csv").exists():
            continue
        sh([sys.executable, "scripts/zeroshot_eval.py", *args, "--shield", "0.3", "--suites", SUITES,
            "--data-root", POLICY / "data" / "suites", "--out", out / f"{name}.csv",
            "--workers", a.eval_workers], a.dry_run)
    cmp = ["e3_both:compromise", "current:compromise", "current:frontier", "compromise:frontier",
           "current:frontier50M", "frontier50M:frontier"]
    sh([sys.executable, "scripts/e7_stats.py", "--runs", out, "--out", a.root / "zeroshot",
        "--suites", SUITES, "--compare", ",".join(cmp)], a.dry_run)
    if not a.dry_run:
        dst = POLICY / "results" / "frontier"
        shutil.copytree(a.root / "zeroshot", dst, dirs_exist_ok=True)
        for r in (a.root / "runs").glob("*_seed*"):
            for f in ("log.csv", "val.csv", "args.json", "config.json"):
                if (r / f).exists():
                    (dst / "train" / r.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy(r / f, dst / "train" / r.name / f)
        print(f"  results -> {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", required=True,
                    choices=["setup", "pilot", "frontier", "compromise", "eval", "all"])
    ap.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/frontier"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bank", type=int, default=50000)
    ap.add_argument("--envs", type=int, default=2048, help="frontier: parallel environments")
    ap.add_argument("--workers", type=int, default=16, help="CPU processes (banks, compromise rollouts)")
    ap.add_argument("--eval-workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    steps = ["setup", "pilot", "frontier", "compromise", "eval"] if a.step == "all" else [a.step]
    for s in steps:
        if s == "setup":
            step_setup(a)
        elif s == "pilot":
            step_pilot(a)
        elif s in ("frontier", "compromise"):
            step_train(a, s)
        else:
            step_eval(a)


if __name__ == "__main__":
    main()
