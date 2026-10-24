"""Sketch v7 end to end on the training machine: data -> train -> tune -> evaluate -> push.

    python scripts/run_v7.py              # everything; safe to Ctrl-C and run again
    python scripts/run_v7.py --dry-run    # print the commands only
    python scripts/run_v7.py --no-push    # everything but the git step

Steps, each skipped when its output already exists (so a second run resumes):

1. ``train_sketch7`` / ``val_sketch7`` / ``test_sketch7`` and the probes
   ``test_sketch7_free`` / ``_tilt`` / ``_lines`` (``make_datasets.py``)
2. fine-tune the v6 model on sketch7 + sketch6 + aug3 with a new context block
   (``--context dilated``, identity at the start; DESIGN_SKETCH §11.1, §11.4);
   resumes from ``ckpt_last.pt`` when present
3. ``evaluate.py --tune-extract`` on ``val_sketch7`` -> ``tune_extract.json``
4. evaluations (DESIGN_SKETCH §11.4): v7 on every earlier test set there is and
   the four v7 sets; v6 on the four v7 sets (``sketch6_on_sketch7``) and on the
   earlier sets its own run did not evaluate (``sketch6_on_earlier``)
5. ``aggregate.py`` over every evaluation -> ``results/sketch7``, plus
   ``regression.md``: v7 against v6 on each earlier set (must hold: no drop of
   more than 0.5 point of ``scene_usable``)
6. every drawing in ``old_drawings_v4/input``, ``drawings_v5/input`` and
   ``drawings_meeting/input`` -> ``results/drawings_v7``, with
   ``compare_<name>.png`` = photo | v6 clean | v7 overlay | v7 clean and
   ``summary.md``; a drawing v6 has no clean rendering of is parsed with v6
   into ``drawings_v7/v6`` first
7. the model for ``demo.py`` -> ``models/sketch7`` (best checkpoint, config, tuned thresholds)
8. ``git add`` / ``commit`` / ``push`` of the result folders and ``models/sketch7``

Defaults are the training machine's paths (E:\\cnn-data, E:\\cnn-runs, E:\\cnn-eval).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v5 import HERE, PY, REPO, banner, run, side_by_side  # noqa: E402
from run_v6 import DRAWING_INPUTS, find_drawings, flag_list, parse_drawing, tuned6  # noqa: E402

SETS7 = ["train_sketch7", "val_sketch7", "test_sketch7", "test_sketch7_free", "test_sketch7_tilt",
         "test_sketch7_lines"]
TESTS7 = SETS7[2:]
# every earlier test set (DESIGN_SKETCH §11.4); the v6 run evaluated the ones in V6_EVALUATED
EARLIER = ["test", "test_aug3", "test_sketch", "test_sketch2", "test_sketch3", "test_sketch4", "test_sketch5",
           "test_sketch6", "test_sketch6swap"]
V6_EVALUATED = ["test", "test_aug3", "test_sketch4", "test_sketch5", "test_sketch6", "test_sketch6swap"]
DRAWING_INPUTS7 = DRAWING_INPUTS + [HERE / "results" / "drawings_meeting" / "input"]
V6_PARSES = [HERE / "results" / "drawings_v6"]
# the extractor filters of §11.3: box sides / stubs are not walls; slanted walls stay one segment
V7_EXTRACT = ["--wall-on-box-tol", "1.5", "--stub-len", "3", "--line-axis-dev", "2"]
MAX_DROP = 0.005             # scene_usable may drop by at most 0.5 point on an earlier set


def v6_clean(stem: str, own: Path):
    """The v6 model's clean re-rendering of a drawing: its v6 run's, else the one made here."""
    for d in V6_PARSES + [own]:
        if (d / f"{stem}_clean.png").exists():
            return d / f"{stem}_clean.png"
    return None


def compare_v7(q: Path, dst: Path):
    """compare_<name>.png = photo | v6 clean | v7 overlay | v7 clean (DESIGN_SKETCH §11.4)."""
    side_by_side([("photo", q), ("v6 clean", v6_clean(q.stem, dst / "v6")),
                  ("v7 overlay", dst / f"{q.stem}_overlay.png"), ("v7 clean", dst / f"{q.stem}_clean.png")],
                 dst / f"compare_{q.stem}.png")


SUMMARY_HEAD = ("| drawing | agents | weak agents (< {low:.2f}) | starts | exits | walls | obstacles | "
                "empty starts dropped | no-letter boxes dropped | thin boxes dropped | agents on letters | "
                "walls on boxes dropped | stubs dropped |")


def summary_row(name: str, rec: dict, low: float) -> str:
    """One summary.md row: what a drawing was read as, with the v6 and v7 dropped counts."""
    if not rec.get("parsed", True):
        return f"| {name} | parse failed |" + " |" * 11
    diag = rec.get("diagnostics", {})
    weak = ", ".join(f"{s:.2f}" for s in diag.get("agent_scores", []) if s < low) or "-"
    return (f"| {name} | {len(rec.get('agents', []))} | {weak} | {len(rec.get('starts') or [])} | "
            f"{len(rec.get('exits') or [])} | {len(rec.get('walls', []))} | {len(rec.get('obstacles', []))} | "
            f"{diag.get('dropped_empty_starts', 0)} | {diag.get('dropped_no_letter', 0)} | "
            f"{diag.get('dropped_thin_boxes', 0)} | {diag.get('agents_on_letters', 0)} | "
            f"{diag.get('walls_on_boxes', 0)} | {diag.get('wall_stubs', 0)} |")


def write_summary(photos, dst: Path, low: float, title: str) -> list:
    rows = [SUMMARY_HEAD.format(low=low), "|" + "---|" * 13]
    for q in photos:
        f = dst / f"{q.stem}.json"
        rows.append(summary_row(q.name, json.loads(f.read_text(encoding="utf-8")), low) if f.exists()
                    else f"| {q.name} | not parsed |" + " |" * 11)
    (dst / "summary.md").write_text(f"# {title}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return rows


def usable(metrics_json: Path) -> dict:
    """{set: overall scene_usable} of one evaluation (empty when it is not there)."""
    if not metrics_json.exists():
        return {}
    m = json.loads(metrics_json.read_text(encoding="utf-8"))
    return {k: v["overall"]["scene_usable"] for k, v in m.items() if k != "_meta" and "overall" in v}


def write_regression(v7: dict, v6: dict, out: Path) -> list:
    """regression.md: v7 against v6 on every earlier set both were evaluated on."""
    rows = ["| set | v6 scene_usable | v7 scene_usable | change (points) | holds (≥ −0.5) |", "|---|---|---|---|---|"]
    for s in EARLIER:
        if s in v7 and s in v6:
            d = v7[s] - v6[s]
            rows.append(f"| {s} | {v6[s]:.3f} | {v7[s]:.3f} | {100 * d:+.1f} | {'yes' if d >= -MAX_DROP else 'NO'} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "regression.md").write_text("# v7 against v6 on the earlier test sets (DESIGN_SKETCH §11.4)\n\n"
                                       + "\n".join(rows) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=r"E:\cnn-data")
    p.add_argument("--runs", default=r"E:\cnn-runs")
    p.add_argument("--eval", default=r"E:\cnn-eval")
    p.add_argument("--v6", default=r"E:\cnn-runs\sketch6_ft_seed0\ckpt_best.pt")
    p.add_argument("--name", default="sketch7_ft_seed0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=None, help="training batch (default: train.py's, 8 at 256 px)")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="scenes per evaluated set (smoke runs)")
    p.add_argument("--n-scale", type=float, default=None, help="dataset size factor (smoke runs)")
    p.add_argument("--low", type=float, default=0.6, help="flag agents below this confidence")
    p.add_argument("--results", default=str(HERE / "results"),
                   help="where sketch7/ and drawings_v7/ go (default perception/results; smoke runs elsewhere)")
    p.add_argument("--models", default=str(HERE / "models"),
                   help="where the demo model folder sketch7/ goes (default perception/models)")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    dry = a.dry_run
    data, runs, evals = Path(a.data_root), Path(a.runs), Path(a.eval)
    run_dir = runs / a.name
    ckpt = run_dir / "ckpt_best.pt"
    v6_dir = Path(a.v6).parent
    limit = ["--limit", a.limit] if a.limit else []

    banner("1/8 datasets")
    if all((data / s / "dataset.json").exists() for s in SETS7):
        print("  already there")
    else:
        run([PY, "scripts/make_datasets.py", "--only", "*sketch7*", "--data-root", data, "--jobs", a.jobs]
            + (["--n-scale", a.n_scale] if a.n_scale else []), dry)

    banner("2/8 train (fine-tune the v6 model, new context block)")
    cmd = [PY, "train.py", "--data-root", data, "--train", "train_sketch7", "train_sketch6", "train_aug3",
           "--val", "val_sketch7", "--size", "256", "--arch", "unet24x4", "--instance-head",
           "--letter-head", "--context", "dilated", "--wall-mode", "lines", "--region-by-agents",
           "--goal-mode", "exits", "--heat-sigma", "0.5", "--heat-nms-factor", "0.6", "--min-box-side", "3",
           "--letter-gate", "--lambda-letter", "0.3", *V7_EXTRACT,
           "--init", a.v6, "--epochs", a.epochs, "--lr", "5e-4", "--num-workers", "2", "--out", run_dir]
    cmd += ["--batch", a.batch] if a.batch else []
    log = run_dir / "log.csv"
    done = len(log.read_text().splitlines()) - 1 if log.exists() else 0
    if done >= a.epochs and ckpt.exists():
        print(f"  already trained ({done} epochs)")
    else:
        if (run_dir / "ckpt_last.pt").exists():
            cmd.append("--resume")
        run(cmd, dry)

    banner("3/8 tune the extractor thresholds on val_sketch7")
    if (run_dir / "tune_extract.json").exists():
        print("  already tuned")
    else:
        run([PY, "evaluate.py", "--ckpt", ckpt, "--data", data / "val_sketch7", "--tune-extract"] + limit, dry)
    flags = tuned6(run_dir, dry)
    print("  " + "  ".join(f"{k[2:]} {v}" for k, v in flags.items()))
    v6_flags = flag_list(tuned6(v6_dir, dry)) if (v6_dir / "tune_extract.json").exists() or dry else []

    banner("4/8 evaluate")
    there = [s for s in EARLIER if dry or (data / s / "dataset.json").exists()]
    for s in sorted(set(EARLIER) - set(there)):
        print(f"  {s}: not in {data}, left out")
    jobs = [
        (evals / a.name, ckpt, there + TESTS7, flag_list(flags)),
        (evals / "sketch6_on_sketch7", a.v6, TESTS7, v6_flags),
        (evals / "sketch6_on_earlier", a.v6, [s for s in there if s not in V6_EVALUATED], v6_flags),
    ]
    for out, which, sets, extra in jobs:
        if not sets:
            continue
        if (out / "metrics.json").exists():
            print(f"  {out} already evaluated")
            continue
        run([PY, "evaluate.py", "--ckpt", which, "--data", *[data / s for s in sets], *extra, "--out", out]
            + limit, dry)

    banner("5/8 aggregate -> results/sketch7")
    results = Path(a.results)
    run([PY, "aggregate.py", "--runs", evals, "--out", results / "sketch7"], dry)
    if not dry:
        v6 = {**usable(evals / "sketch6_on_earlier" / "metrics.json"),
              **usable(evals / v6_dir.name / "metrics.json")}
        rows = write_regression(usable(evals / a.name / "metrics.json"), v6, results / "sketch7")
        print("\n".join(rows))

    banner("6/8 the drawings -> results/drawings_v7")
    dst = results / "drawings_v7"
    photos = find_drawings(DRAWING_INPUTS7)
    if not photos:
        print("  no drawings in " + " or ".join(str(d) for d in DRAWING_INPUTS7))
    if not dry:
        (dst / "v6").mkdir(parents=True, exist_ok=True)
    v6_tuned = tuned6(v6_dir, dry) if (v6_dir / "tune_extract.json").exists() or dry else {}
    for q in photos:
        if v6_clean(q.stem, dst / "v6") is None:          # e.g. the meeting drawings: v6 parsed them live
            parse_drawing(q, dst / "v6", a.v6, v6_tuned, dry)
        if (dst / f"{q.stem}.json").exists():
            print(f"  {q.name}: done")
            continue
        parse_drawing(q, dst, ckpt, flags, dry)
        if not dry:
            compare_v7(q, dst)
    if photos and not dry:
        print("\n".join(write_summary(photos, dst, a.low, "v7 on the drawings")))

    banner("7/8 the demo model -> models/sketch7")
    model = Path(a.models) / "sketch7"
    if (model / "ckpt_best.pt").exists():
        print("  already there")
    elif dry:
        print(f"  copy ckpt_best.pt, config.json, tune_extract.json from {run_dir} to {model}")
    else:
        model.mkdir(parents=True, exist_ok=True)
        for f in ("ckpt_best.pt", "config.json", "tune_extract.json"):
            shutil.copy2(run_dir / f, model / f)

    banner("8/8 push")
    if a.no_push:
        print("  skipped (--no-push)")
        return 0
    run(["git", "add", results / "sketch7", dst, model], dry, cwd=REPO)
    if run(["git", "commit", "-m", "Sketch v7: training, evaluation and the drawings"],
           dry, check=False, cwd=REPO) == 0:
        run(["git", "push"], dry, cwd=REPO)
    else:
        print("  nothing new to commit")
    print("\nall done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
