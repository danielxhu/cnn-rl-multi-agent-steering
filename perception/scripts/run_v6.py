"""Sketch v6 end to end on the training machine: data -> train -> tune -> evaluate -> push.

    python scripts/run_v6.py              # everything; safe to Ctrl-C and run again
    python scripts/run_v6.py --dry-run    # print the commands only
    python scripts/run_v6.py --no-push    # everything but the git step

Steps, each skipped when its output already exists (so a second run resumes):

1. ``train_sketch6`` / ``val_sketch6`` / ``test_sketch6`` / ``test_sketch6swap``
   (``make_datasets.py``)
2. fine-tune the v5 model on sketch6 + aug3 with a new letter head
   (DESIGN_SKETCH §10.5); resumes from ``ckpt_last.pt`` when present
3. ``evaluate.py --tune-extract`` on ``val_sketch6`` -> ``tune_extract.json``
   (heat threshold, NMS factor, then the letter threshold)
4. evaluations (DESIGN_SKETCH §10.6): v6 on test, test_aug3, test_sketch4,
   test_sketch5, test_sketch6, test_sketch6swap; v6 without the letter gate on
   the three sketch-5/6 sets (``sketch6_nogate``); v5 on test_sketch6 and
   test_sketch6swap (``sketch5_on_sketch6``)
5. ``aggregate.py`` over every evaluation -> ``results/sketch6``
6. every drawing in ``results/old_drawings_v4/input`` and
   ``results/drawings_v5/input`` -> ``results/drawings_v6``, with
   ``compare_<name>.png`` = photo | v5 clean | v6 overlay | v6 clean and
   ``summary.md`` (a drawing without corner markers is retried without ``--photo``)
7. ``git add`` / ``commit`` / ``push`` of the two result folders

Defaults are the training machine's paths (E:\\cnn-data, E:\\cnn-runs, E:\\cnn-eval).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v5 import HERE, IMAGE_EXT, PY, REPO, banner, run, side_by_side  # noqa: E402

SETS6 = ["train_sketch6", "val_sketch6", "test_sketch6", "test_sketch6swap"]
DRAWING_INPUTS = [HERE / "results" / "old_drawings_v4" / "input", HERE / "results" / "drawings_v5" / "input"]
V5_PARSES = [HERE / "results" / "old_drawings_v5", HERE / "results" / "drawings_v5"]


def tuned6(run_dir: Path, dry: bool, letter: bool = True) -> dict:
    """The tuned extractor thresholds of a run (``tune_extract.json``'s ``best``)
    as predict / evaluate flags: heat threshold, NMS factor, letter threshold
    when the run tuned one. A v5 run gives the first two."""
    p = Path(run_dir) / "tune_extract.json"
    if dry and not p.exists():
        out = {"--heat-thresh": "<tuned>", "--heat-nms-factor": "<tuned>"}
        return {**out, "--letter-thresh": "<tuned>"} if letter else out
    best = json.loads(p.read_text(encoding="utf-8"))["best"]
    flags = {"--heat-thresh": best["heat_thresh"], "--heat-nms-factor": best.get("heat_nms_factor", 0.6)}
    if "letter_thresh" in best:
        flags["--letter-thresh"] = best["letter_thresh"]
    return flags


def flag_list(flags: dict) -> list:
    return [x for k, v in flags.items() for x in (k, v)]


def find_drawings(dirs=DRAWING_INPUTS) -> list:
    """Every photo in the input folders, by name (a name in two folders once)."""
    seen, out = set(), []
    for d in dirs:
        for q in sorted(d.glob("*")) if d.exists() else []:
            if q.suffix in IMAGE_EXT and q.stem not in seen:
                seen.add(q.stem)
                out.append(q)
    return out


def v5_clean(stem: str):
    """The v5 model's clean re-rendering of a drawing, wherever it was parsed."""
    for d in V5_PARSES:
        if (d / f"{stem}_clean.png").exists():
            return d / f"{stem}_clean.png"
    return None


def parse_drawing(q: Path, dst: Path, ckpt, flags: dict, dry: bool):
    """predict.py on one drawing (photo of a drawing sheet first, else the plain image)."""
    base = [PY, "predict.py", "--ckpt", ckpt, "--image", q, "--wall-mode", "lines",
            "--region-by-agents", "--goal-mode", "exits", *flag_list(flags), "--out", dst]
    if not dry:
        (dst / f"{q.stem}.json").unlink(missing_ok=True)
    run(base[:6] + ["--photo"] + base[6:], dry)
    if not dry and not (dst / f"{q.stem}.json").exists():   # predict prints REJECTED, exits 0
        print(f"  {q.name}: no corner markers found, retrying without --photo")
        run(base, dry)


def compare_v6(q: Path, dst: Path):
    """compare_<name>.png = photo | v5 clean | v6 overlay | v6 clean (DESIGN_SKETCH §10.6)."""
    side_by_side([("photo", q), ("v5 clean", v5_clean(q.stem)),
                  ("v6 overlay", dst / f"{q.stem}_overlay.png"), ("v6 clean", dst / f"{q.stem}_clean.png")],
                 dst / f"compare_{q.stem}.png")


SUMMARY_HEAD = ("| drawing | agents | weak agents (< {low:.2f}) | starts | exits | walls | obstacles | "
                "empty starts dropped | no-letter boxes dropped | thin boxes dropped | agents on letters |")


def summary_row(name: str, rec: dict, low: float) -> str:
    """One summary.md row: what a drawing was read as, with the v6 dropped-box counts."""
    if not rec.get("parsed", True):
        return f"| {name} | parse failed | | | | | | | | | |"
    diag = rec.get("diagnostics", {})
    weak = ", ".join(f"{s:.2f}" for s in diag.get("agent_scores", []) if s < low) or "-"
    return (f"| {name} | {len(rec.get('agents', []))} | {weak} | {len(rec.get('starts') or [])} | "
            f"{len(rec.get('exits') or [])} | {len(rec.get('walls', []))} | {len(rec.get('obstacles', []))} | "
            f"{diag.get('dropped_empty_starts', 0)} | {diag.get('dropped_no_letter', 0)} | "
            f"{diag.get('dropped_thin_boxes', 0)} | {diag.get('agents_on_letters', 0)} |")


def write_summary(photos, dst: Path, low: float, title: str) -> list:
    rows = [SUMMARY_HEAD.format(low=low), "|" + "---|" * 11]
    for q in photos:
        f = dst / f"{q.stem}.json"
        rows.append(summary_row(q.name, json.loads(f.read_text(encoding="utf-8")), low) if f.exists()
                    else f"| {q.name} | not parsed | | | | | | | | | |")
    (dst / "summary.md").write_text(f"# {title}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=r"E:\cnn-data")
    p.add_argument("--runs", default=r"E:\cnn-runs")
    p.add_argument("--eval", default=r"E:\cnn-eval")
    p.add_argument("--v5", default=r"E:\cnn-runs\sketch5_ft_seed0\ckpt_best.pt")
    p.add_argument("--name", default="sketch6_ft_seed0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="scenes per evaluated set (smoke runs)")
    p.add_argument("--n-scale", type=float, default=None, help="dataset size factor (smoke runs)")
    p.add_argument("--low", type=float, default=0.6, help="flag agents below this confidence")
    p.add_argument("--results", default=str(HERE / "results"),
                   help="where sketch6/ and drawings_v6/ go (default perception/results; smoke runs elsewhere)")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    dry = a.dry_run
    data, runs, evals = Path(a.data_root), Path(a.runs), Path(a.eval)
    run_dir = runs / a.name
    ckpt = run_dir / "ckpt_best.pt"
    v5_dir = Path(a.v5).parent
    limit = ["--limit", a.limit] if a.limit else []

    banner("1/7 datasets")
    if all((data / s / "dataset.json").exists() for s in SETS6):
        print("  already there")
    else:
        run([PY, "scripts/make_datasets.py", "--only", "*sketch6*", "--data-root", data, "--jobs", a.jobs]
            + (["--n-scale", a.n_scale] if a.n_scale else []), dry)

    banner("2/7 train (fine-tune the v5 model, new letter head)")
    cmd = [PY, "train.py", "--data-root", data, "--train", "train_sketch6", "train_aug3",
           "--val", "val_sketch6", "--size", "256", "--arch", "unet24x4", "--instance-head",
           "--letter-head", "--wall-mode", "lines", "--region-by-agents", "--goal-mode", "exits",
           "--heat-sigma", "0.5", "--heat-nms-factor", "0.6", "--min-box-side", "3", "--letter-gate",
           "--lambda-letter", "0.3",
           "--init", a.v5, "--epochs", a.epochs, "--lr", "5e-4", "--num-workers", "2", "--out", run_dir]
    log = run_dir / "log.csv"
    done = len(log.read_text().splitlines()) - 1 if log.exists() else 0
    if done >= a.epochs and ckpt.exists():
        print(f"  already trained ({done} epochs)")
    else:
        if (run_dir / "ckpt_last.pt").exists():
            cmd.append("--resume")
        run(cmd, dry)

    banner("3/7 tune the extractor thresholds on val_sketch6")
    if (run_dir / "tune_extract.json").exists():
        print("  already tuned")
    else:
        run([PY, "evaluate.py", "--ckpt", ckpt, "--data", data / "val_sketch6", "--tune-extract"] + limit, dry)
    flags = tuned6(run_dir, dry)
    print("  " + "  ".join(f"{k[2:]} {v}" for k, v in flags.items()))

    banner("4/7 evaluate")
    jobs = [
        (evals / a.name, ckpt, ["test", "test_aug3", "test_sketch4", "test_sketch5", "test_sketch6",
                                "test_sketch6swap"], flag_list(flags)),
        (evals / "sketch6_nogate", ckpt, ["test_sketch5", "test_sketch6", "test_sketch6swap"],
         flag_list(flags) + ["--no-letter-gate"]),
        (evals / "sketch5_on_sketch6", a.v5, ["test_sketch6", "test_sketch6swap"],
         flag_list(tuned6(v5_dir, dry, letter=False)) if (v5_dir / "tune_extract.json").exists() or dry else []),
    ]
    for out, which, sets, extra in jobs:
        if (out / "metrics.json").exists():
            print(f"  {out} already evaluated")
            continue
        run([PY, "evaluate.py", "--ckpt", which, "--data", *[data / s for s in sets], *extra, "--out", out]
            + limit, dry)

    banner("5/7 aggregate -> results/sketch6")
    results = Path(a.results)
    run([PY, "aggregate.py", "--runs", evals, "--out", results / "sketch6"], dry)

    banner("6/7 the drawings -> results/drawings_v6")
    dst = results / "drawings_v6"
    photos = find_drawings()
    if not photos:
        print("  no drawings in " + " or ".join(str(d) for d in DRAWING_INPUTS))
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)
    for q in photos:
        if (dst / f"{q.stem}.json").exists():
            print(f"  {q.name}: done")
            continue
        parse_drawing(q, dst, ckpt, flags, dry)
        if not dry:
            compare_v6(q, dst)
    if photos and not dry:
        print("\n".join(write_summary(photos, dst, a.low, "v6 on the drawings")))

    banner("7/7 push")
    if a.no_push:
        print("  skipped (--no-push)")
        return 0
    run(["git", "add", results / "sketch6", dst], dry, cwd=REPO)
    if run(["git", "commit", "-m", "Sketch v6: training, evaluation and the drawings"],
           dry, check=False, cwd=REPO) == 0:
        run(["git", "push"], dry, cwd=REPO)
    else:
        print("  nothing new to commit")
    print("\nall done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
