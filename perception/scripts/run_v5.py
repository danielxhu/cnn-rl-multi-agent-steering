"""Sketch v5 end to end on the training machine: data -> train -> tune -> evaluate -> push.

    python scripts/run_v5.py              # everything; safe to Ctrl-C and run again
    python scripts/run_v5.py --dry-run    # print the commands only

Steps, each skipped when its output already exists (so a second run resumes):

1. ``train_sketch5`` / ``val_sketch5`` / ``test_sketch5`` (``make_datasets.py``)
2. fine-tune the v4 model on sketch5 + aug3 (DESIGN_SKETCH §9.2); resumes from
   ``ckpt_last.pt`` when present
3. ``evaluate.py --tune-extract`` on ``val_sketch5`` -> ``tune_extract.json``
4. the v5 model on test, test_aug3, test_sketch4, test_sketch5 with the tuned
   ``heat_thresh`` / ``heat_nms_factor``; the v4 model on test_sketch5 as exit sets
5. ``aggregate.py`` over every evaluation -> ``results/sketch5``
6. the v5 model on every drawing in ``results/old_drawings_v4/input`` ->
   ``results/old_drawings_v5``, with ``compare_<name>.png`` = drawing | v4 | v5
   (a drawing without corner markers is retried without ``--photo``; put any
   other drawings into that ``input`` folder first)
7. ``git add`` / ``commit`` / ``push`` of the two result folders

Defaults are the training machine's paths (E:\\cnn-data, E:\\cnn-runs, E:\\cnn-eval).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # perception/
REPO = HERE.parent
PY = sys.executable
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def banner(text):
    print(f"\n=== {text}", flush=True)


def run(cmd, dry, check=True, cwd=HERE) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    rc = subprocess.run([str(c) for c in cmd], cwd=cwd).returncode
    if check and rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {' '.join(str(c) for c in cmd)}")
    return rc


def tuned(run_dir: Path, dry: bool):
    """(heat_thresh, heat_nms_factor) from tune_extract.json."""
    p = run_dir / "tune_extract.json"
    if dry and not p.exists():
        return "<tuned>", "<tuned>"
    best = json.loads(p.read_text())["best"]
    return best["heat_thresh"], best.get("heat_nms_factor", 0.6)


def side_by_side(panels, out: Path, h: int = 480):
    """[(label, image path)] -> one sheet, each panel scaled to height h; missing paths skipped."""
    from PIL import Image, ImageDraw
    ims = []
    for t, q in panels:
        if q is not None and Path(q).exists():
            im = Image.open(q).convert("RGB")
            ims.append((t, im.resize((max(1, int(im.width * h / im.height)), h))))
    sheet = Image.new("RGB", (sum(im.width for _, im in ims) + 12 * (len(ims) - 1), h + 26), "white")
    d = ImageDraw.Draw(sheet)
    x = 0
    for t, im in ims:
        sheet.paste(im, (x, 26))
        d.text((x + 6, 6), t, fill=(30, 30, 30))
        x += im.width + 12
    sheet.save(out)


def compare_image(drawing: Path, v4_clean: Path | None, v5_clean: Path, out: Path):
    side_by_side([("drawing", drawing), ("v4", v4_clean), ("v5", v5_clean)], out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=r"E:\cnn-data")
    p.add_argument("--runs", default=r"E:\cnn-runs")
    p.add_argument("--eval", default=r"E:\cnn-eval")
    p.add_argument("--v4", default=r"E:\cnn-runs\sketch4_ft_seed0\ckpt_best.pt")
    p.add_argument("--name", default="sketch5_ft_seed0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    dry = a.dry_run
    data, runs, evals = Path(a.data_root), Path(a.runs), Path(a.eval)
    run_dir = runs / a.name
    ckpt = run_dir / "ckpt_best.pt"

    banner("1/7 datasets")
    sets = ["train_sketch5", "val_sketch5", "test_sketch5"]
    if all((data / s / "dataset.json").exists() for s in sets):
        print("  already there")
    else:
        run([PY, "scripts/make_datasets.py", "--only", "*sketch5", "--data-root", data, "--jobs", a.jobs], dry)

    banner("2/7 train (fine-tune the v4 model)")
    cmd = [PY, "train.py", "--data-root", data, "--train", "train_sketch5", "train_aug3",
           "--val", "val_sketch5", "--size", "256", "--arch", "unet24x4", "--instance-head",
           "--wall-mode", "lines", "--region-by-agents", "--goal-mode", "exits",
           "--heat-sigma", "0.5", "--heat-nms-factor", "0.6", "--init", a.v4,
           "--epochs", a.epochs, "--lr", "5e-4", "--num-workers", "2", "--out", run_dir]
    log = run_dir / "log.csv"
    done = len(log.read_text().splitlines()) - 1 if log.exists() else 0
    if done >= a.epochs and (run_dir / "ckpt_best.pt").exists():
        print(f"  already trained ({done} epochs)")
    else:
        if (run_dir / "ckpt_last.pt").exists():
            cmd.append("--resume")
        run(cmd, dry)

    banner("3/7 tune the extractor thresholds on val_sketch5")
    if (run_dir / "tune_extract.json").exists():
        print("  already tuned")
    else:
        run([PY, "evaluate.py", "--ckpt", ckpt, "--data", data / "val_sketch5", "--tune-extract"], dry)
    ht, nf = tuned(run_dir, dry)
    print(f"  heat_thresh {ht}, heat_nms_factor {nf}")

    banner("4/7 evaluate")
    out5 = evals / a.name
    if (out5 / "metrics.json").exists():
        print(f"  {out5} already evaluated")
    else:
        run([PY, "evaluate.py", "--ckpt", ckpt, "--data", data / "test", data / "test_aug3",
             data / "test_sketch4", data / "test_sketch5", "--heat-thresh", ht,
             "--heat-nms-factor", nf, "--out", out5], dry)
    out4 = evals / "sketch4_on_sketch5"
    if (out4 / "metrics.json").exists():
        print(f"  {out4} already evaluated")
    else:
        run([PY, "evaluate.py", "--ckpt", a.v4, "--data", data / "test_sketch5",
             "--goal-mode", "exits", "--out", out4], dry)

    banner("5/7 aggregate -> results/sketch5")
    run([PY, "aggregate.py", "--runs", evals, "--out", HERE / "results" / "sketch5"], dry)

    banner("6/7 the earlier drawings -> results/old_drawings_v5")
    src = HERE / "results" / "old_drawings_v4" / "input"
    dst = HERE / "results" / "old_drawings_v5"
    drawings = sorted(q for q in src.glob("*") if q.suffix in IMAGE_EXT) if src.exists() else []
    if not drawings:
        print(f"  no drawings in {src}")
    for q in drawings:
        if (dst / f"{q.stem}.json").exists():
            print(f"  {q.name}: done")
            continue
        base = [PY, "predict.py", "--ckpt", ckpt, "--image", q, "--wall-mode", "lines",
                "--region-by-agents", "--goal-mode", "exits", "--heat-thresh", ht,
                "--heat-nms-factor", nf, "--out", dst]
        run(base[:6] + ["--photo"] + base[6:], dry)
        if not dry and not (dst / f"{q.stem}.json").exists():   # predict prints REJECTED, exits 0
            print(f"  {q.name}: no corner markers found, retrying without --photo")
            run(base, dry)
        v5 = dst / f"{q.stem}_clean.png"
        if not dry and v5.exists():
            compare_image(q, src.parent / f"{q.stem}_clean.png", v5, dst / f"compare_{q.stem}.png")

    banner("7/7 push")
    if a.no_push:
        print("  skipped (--no-push)")
        return 0
    run(["git", "add", "perception/results/sketch5", "perception/results/old_drawings_v5"], dry, cwd=REPO)
    if run(["git", "commit", "-m", "Sketch v5: training, evaluation and the earlier drawings"],
           dry, check=False, cwd=REPO) == 0:
        run(["git", "push"], dry, cwd=REPO)
    else:
        print("  nothing new to commit")
    print("\nall done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
