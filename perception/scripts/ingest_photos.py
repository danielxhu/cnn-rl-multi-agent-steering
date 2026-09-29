"""Photos of traced drawing sheets -> the real hand-drawn test set.

    python scripts/ingest_photos.py PHOTO_DIR
    python scripts/ingest_photos.py PHOTO_DIR --scenes data/real_sketch/scenes --out data/real_sketch/test

For every image in PHOTO_DIR (jpg / jpeg / png, any case): rectify it to the
world square (``rectify.py``), read the scene id from the id boxes, and write

    real_NNNN.png          the rectified photo, 512 x 512 (network input)
    real_NNNN.json         the scene JSON written by make_sheets.py (ground truth)
    real_NNNN_labels.png   labels rendered from that JSON by the default renderer
    dataset.json           config, ``"labels_exact": false``, accepted and rejected photos

The label PNG exists only so ``SceneDataset`` can load the set. It is not
traced from the photo, so pixel metrics and the wall surface error on this set
are not meaningful: ``dataset.json`` says so and ``evaluate.py`` reports them
as NaN. Agent, obstacle and region metrics compare against the JSON, which the
tracing workflow makes the ground truth (DESIGN_SKETCH §2).

Every rejected photo is printed with its reason (no verified markers, id
unreadable, no scene for the id, a second photo of the same sheet). Photos
are taken in sorted file-name order; the first photo of a sheet wins.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths  # noqa: E402,F401
from config import Config  # noqa: E402
from generate import save_labels  # noqa: E402
from rectify import RectifyError, load_photo, rectify  # noqa: E402
from render import render  # noqa: E402
from scene import Scene  # noqa: E402

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_scenes(scene_dir: Path) -> dict:
    """{sheet_id: (name, json dict)} from make_sheets.py's scenes/ directory."""
    out = {}
    for p in sorted(scene_dir.glob("real_*.json")):
        with open(p) as fh:
            d = json.load(fh)
        out[int(d["sheet_id"])] = (p.stem, d)
    return out


def config_for(world: dict) -> Config:
    return Config().merged(world=world["size"], img_size=world["img_size"],
                           agent_radius=world["agent_radius"],
                           wall_thickness=world["wall_thickness"])


def ingest(photo_dir, scene_dir, out_dir, size: int = 512) -> tuple:
    """(accepted [(photo, name, sheet_id)], rejected [(photo, reason)])."""
    photo_dir, scene_dir, out_dir = Path(photo_dir), Path(scene_dir), Path(out_dir)
    scenes = load_scenes(scene_dir)
    if not scenes:
        raise SystemExit(f"no real_*.json scenes in {scene_dir}; run scripts/make_sheets.py first")
    photos = sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not photos:
        raise SystemExit(f"no photos (.jpg / .jpeg / .png) in {photo_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted, rejected, seen = [], [], {}
    for path in photos:
        try:
            img, info = rectify(load_photo(path), out=size)
        except RectifyError as e:
            rejected.append((path.name, str(e)))
            continue
        sid = info["sheet_id"]
        if sid is None:
            rejected.append((path.name, "scene id unreadable (parity check failed); "
                                        "are the id boxes in focus and unobstructed?"))
            continue
        if sid not in scenes:
            rejected.append((path.name, f"scene id {sid} has no JSON in {scene_dir}"))
            continue
        if sid in seen:
            rejected.append((path.name, f"second photo of sheet {sid} (first: {seen[sid]})"))
            continue
        seen[sid] = path.name
        name, d = scenes[sid]
        rec = dict(d, id=name, photo=path.name)
        cfg = config_for(rec["world"])
        _, lab = render(Scene.from_dict(rec), cfg)
        Image.fromarray(img, "RGB").save(out_dir / f"{name}.png")
        save_labels(lab, out_dir / f"{name}_labels.png")
        with open(out_dir / f"{name}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        accepted.append((path.name, name, sid))

    cfg = config_for(next(iter(scenes.values()))[1]["world"])
    with open(out_dir / "dataset.json", "w") as fh:
        json.dump({"config": asdict(cfg), "labels_exact": False,
                   "note": "labels rendered from the scene JSON, not traced from the photos; "
                           "pixel metrics and wall surface error are not meaningful",
                   "n_written": len(accepted), "n_rejected": len(rejected),
                   "scenes": [{"id": n, "photo": ph, "sheet_id": s} for ph, n, s in accepted],
                   "rejected": [{"photo": ph, "reason": r} for ph, r in rejected]}, fh, indent=1)
    return accepted, rejected


def main(argv=None):
    real = _paths.PERCEPTION_DIR / "data" / "real_sketch"
    p = argparse.ArgumentParser(description="Rectify photographed drawing sheets into a test set.")
    p.add_argument("photos", help="directory of photos")
    p.add_argument("--scenes", default=str(real / "scenes"), help="make_sheets.py's scenes/ directory")
    p.add_argument("--out", default=str(real / "test"))
    p.add_argument("--clean", action="store_true", help="empty --out first")
    a = p.parse_args(argv)

    out = Path(a.out)
    if a.clean and out.exists():
        shutil.rmtree(out)
    accepted, rejected = ingest(a.photos, a.scenes, out)
    for ph, name, sid in accepted:
        print(f"ok        {ph} -> {name} (sheet {sid})")
    for ph, reason in rejected:
        print(f"REJECTED  {ph}: {reason}")
    n_scenes = len(load_scenes(Path(a.scenes)))
    missing = sorted(set(load_scenes(Path(a.scenes))) - {s for _, _, s in accepted})
    print(f"{len(accepted)} ingested, {len(rejected)} rejected -> {out}"
          + (f"; no photo yet for {len(missing)} of {n_scenes} sheets: {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
