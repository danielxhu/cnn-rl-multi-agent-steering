"""New hand drawings through the newest model, then push, in one command.

    python scripts/test_drawings.py              # photos in results/drawings_v5/input
    python scripts/test_drawings.py --dry-run    # print the commands only

``--run`` defaults to the v6 run (``E:\\cnn-runs\\sketch6_ft_seed0``) once it has a
checkpoint, else the v5 run. With a v6 run (letter head) the parses go to
``results/drawings_v6``, ``compare_<name>.png`` is photo | v5 clean | v6
overlay | v6 clean, and the tuned letter threshold is used too
(DESIGN_SKETCH §10.6); with the v5 run everything is as before.

For every photo in ``--input`` not parsed yet: ``predict.py --photo`` (retried
without ``--photo`` when the corner markers are not found) with the thresholds
tuned in ``tune_extract.json``, then ``compare_<name>.png`` = photo | overlay |
clean re-rendering. ``summary.md`` lists what each drawing was read as, with
agents below ``--low`` confidence flagged. The folder is then committed and
pushed (photos included, so the parses can be checked against them).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v5 import HERE, IMAGE_EXT, PY, REPO, banner, run, side_by_side, tuned  # noqa: E402
import run_v6  # noqa: E402

V6_RUN = r"E:\cnn-runs\sketch6_ft_seed0"
V5_RUN = r"E:\cnn-runs\sketch5_ft_seed0"


def default_run() -> str:
    """The v6 run once it has a checkpoint, else the v5 run."""
    return V6_RUN if (Path(V6_RUN) / "ckpt_best.pt").exists() else V5_RUN


def has_letter_head(run_dir: Path) -> bool:
    p = Path(run_dir) / "config.json"
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("model", {}).get("letter_head"))
    except (OSError, ValueError):
        return False


def summary_row(name: str, rec: dict, low: float) -> str:
    if not rec.get("parsed", True):
        return f"| {name} | parse failed | | | | | | |"
    diag = rec.get("diagnostics", {})
    scores = diag.get("agent_scores", [])
    weak = ", ".join(f"{s:.2f}" for s in scores if s < low) or "-"
    return (f"| {name} | {len(rec.get('agents', []))} | {weak} | {len(rec.get('starts', []))} | "
            f"{len(rec.get('exits', []))} | {len(rec.get('walls', []))} | {len(rec.get('obstacles', []))} | "
            f"{diag.get('dropped_empty_starts', 0)} |")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default=str(HERE / "results" / "drawings_v5" / "input"))
    p.add_argument("--out", default=None,
                   help="default: results/drawings_v6 for a v6 run, results/drawings_v5 for v5")
    p.add_argument("--run", default=None, help="default: the v6 run if it exists, else the v5 run")
    p.add_argument("--low", type=float, default=0.6, help="flag agents below this confidence")
    p.add_argument("--redo", action="store_true", help="parse every photo again")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    dry = a.dry_run
    run_dir = Path(a.run or default_run())
    if has_letter_head(run_dir) or (dry and run_dir.name.startswith("sketch6")):
        return main_v6(a, run_dir)
    a.out = a.out or str(HERE / "results" / "drawings_v5")
    src, dst = Path(a.input).resolve(), Path(a.out).resolve()
    ckpt = run_dir / "ckpt_best.pt"

    src.mkdir(parents=True, exist_ok=True)
    photos = sorted(q for q in src.iterdir() if q.suffix in IMAGE_EXT)
    if not photos:
        print(f"no photos in {src}\nput the drawings there (png/jpg) and run this again")
        return 1
    ht, nf = tuned(run_dir, dry)
    banner(f"{len(photos)} drawing(s), heat_thresh {ht}, heat_nms_factor {nf}")

    for q in photos:
        if (dst / f"{q.stem}.json").exists() and not a.redo:
            print(f"  {q.name}: done")
            continue
        base = [PY, "predict.py", "--ckpt", ckpt, "--image", q, "--wall-mode", "lines",
                "--region-by-agents", "--goal-mode", "exits", "--heat-thresh", ht,
                "--heat-nms-factor", nf, "--out", dst]
        (dst / f"{q.stem}.json").unlink(missing_ok=True)
        run(base[:6] + ["--photo"] + base[6:], dry)
        if not dry and not (dst / f"{q.stem}.json").exists():   # predict prints REJECTED, exits 0
            print(f"  {q.name}: no corner markers found, retrying without --photo")
            run(base, dry)
        if not dry:
            side_by_side([("photo", q), ("overlay", dst / f"{q.stem}_overlay.png"),
                          ("clean", dst / f"{q.stem}_clean.png")], dst / f"compare_{q.stem}.png")

    if dry:
        return 0
    rows = ["| drawing | agents | weak agents (< %.2f) | starts | exits | walls | obstacles | "
            "empty starts dropped |" % a.low, "|---|---|---|---|---|---|---|---|"]
    for q in photos:
        f = dst / f"{q.stem}.json"
        rows.append(summary_row(q.name, json.loads(f.read_text()), a.low) if f.exists()
                    else f"| {q.name} | not parsed | | | | | | |")
    (dst / "summary.md").write_text("# v5 on new drawings\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print("\n" + "\n".join(rows))

    banner("push")
    if a.no_push:
        print("  skipped (--no-push)")
        return 0
    run(["git", "add", dst], dry, cwd=REPO)
    if run(["git", "commit", "-m", f"v5 on {len(photos)} new drawing(s)"], dry, check=False, cwd=REPO) == 0:
        run(["git", "push"], dry, cwd=REPO)
    else:
        print("  nothing new to commit")
    return 0


def main_v6(a, run_dir: Path) -> int:
    """The same with a v6 run: letter threshold too, the four-panel compare sheet."""
    dry = a.dry_run
    src = Path(a.input).resolve()
    dst = Path(a.out or HERE / "results" / "drawings_v6").resolve()
    ckpt = run_dir / "ckpt_best.pt"
    src.mkdir(parents=True, exist_ok=True)
    photos = sorted(q for q in src.iterdir() if q.suffix in IMAGE_EXT)
    if not photos:
        print(f"no photos in {src}\nput the drawings there (png/jpg) and run this again")
        return 1
    flags = run_v6.tuned6(run_dir, dry)
    banner(f"{len(photos)} drawing(s) through {run_dir.name}: "
           + ", ".join(f"{k[2:]} {v}" for k, v in flags.items()))
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)
    for q in photos:
        if (dst / f"{q.stem}.json").exists() and not a.redo:
            print(f"  {q.name}: done")
            continue
        run_v6.parse_drawing(q, dst, ckpt, flags, dry)
        if not dry:
            run_v6.compare_v6(q, dst)
    if dry:
        return 0
    print("\n" + "\n".join(run_v6.write_summary(photos, dst, a.low, f"{run_dir.name} on new drawings")))

    banner("push")
    if a.no_push:
        print("  skipped (--no-push)")
        return 0
    run(["git", "add", dst], dry, cwd=REPO)
    if run(["git", "commit", "-m", f"v6 on {len(photos)} new drawing(s)"], dry, check=False, cwd=REPO) == 0:
        run(["git", "push"], dry, cwd=REPO)
    else:
        print("  nothing new to commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
