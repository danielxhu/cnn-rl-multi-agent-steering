"""Inference: checkpoint -> label map (+ heat / dir) for any input image.

    python predict.py --ckpt runs/dev_s256/ckpt_best.pt --image data/test/scene_00007.png --out parsed/
    python predict.py --ckpt runs/dev_s256/ckpt_best.pt --dir data/test --out parsed/ --limit 20

Writes ``<stem>.json`` (``extract.parsed_to_json``) and ``<stem>_overlay.png``
per image.  The checkpoint carries its ``RunConfig``, so the resolution,
architecture and extractor thresholds are always the run's own; the world
block (size, agent radius, ...) is the training set's, stored at save time.

Invariants:
- ``Predictor`` resizes any input to the run's size with ``INTER_AREA`` and
  scales to [0, 1], exactly like ``data.load_image`` does for training.
- ``device`` is resolved once (CUDA if available, else CPU); nothing calls ``.cuda()``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from config import Config
from data import load_image, resize_image
from extract import extract, parsed_to_json
from model import build_model
from runconfig import RunConfig


def resolve_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_world(cfg: Config = None) -> dict:
    cfg = cfg or Config()
    return {"size": cfg.world, "img_size": cfg.img_size, "agent_radius": cfg.agent_radius,
            "wall_thickness": cfg.wall_thickness, "fov_deg": cfg.fov_deg,
            "fov_range": cfg.fov_range}


def load_checkpoint(path, device=None):
    """(model in eval mode on `device`, RunConfig).  The raw record is on model._ckpt_meta."""
    device = resolve_device(device)
    rec = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = RunConfig.from_dict(rec["config"])
    model = build_model(cfg.model)
    model.load_state_dict(rec["model"])
    model.to(device).eval()
    model._ckpt_meta = {k: rec.get(k) for k in ("epoch", "best_metric", "world")}
    return model, cfg


@torch.no_grad()
def predict_tensor(model, x: torch.Tensor, device) -> list:
    """(B,3,S,S) float in [0,1] -> list of dicts (labels u8, probs f32, [heat, dir])."""
    was_training = model.training
    model.eval()
    out = model(x.to(device, non_blocking=True))
    if was_training:
        model.train()
    probs = torch.softmax(out["seg"].float(), 1).cpu().numpy()
    labels = probs.argmax(1).astype(np.uint8)
    res = []
    for b in range(len(labels)):
        r = {"labels": labels[b], "probs": probs[b]}
        if "heat" in out:
            r["heat"] = out["heat"][b, 0].float().cpu().numpy()
            r["dir"] = out["dir"][b].float().cpu().numpy()
        res.append(r)
    return res


def predict_dataset(model, ds, indices, device, batch: int) -> list:
    """Predictions for `indices` of a SceneDataset, in order, batched (no DataLoader)."""
    out = []
    indices = list(indices)
    for k in range(0, len(indices), batch):
        chunk = indices[k:k + batch]
        x = torch.from_numpy(np.stack([ds.image_uint8(i).transpose(2, 0, 1) for i in chunk])
                             ).float() / 255.0
        out.extend(predict_tensor(model, x, device))
    return out


class Predictor:
    def __init__(self, ckpt_path, device=None):
        self.device = resolve_device(device)
        self.model, self.cfg = load_checkpoint(ckpt_path, self.device)
        self.size = int(self.cfg.data.size)
        self.world = self.model._ckpt_meta.get("world") or default_world()
        self.ecfg = self.cfg.extract

    def predict_tensor(self, x: torch.Tensor) -> list:
        return predict_tensor(self.model, x, self.device)

    def predict_batch(self, images_uint8: list) -> list:
        x = torch.from_numpy(np.stack([
            resize_image(np.asarray(im, np.uint8), self.size).transpose(2, 0, 1)
            for im in images_uint8])).float() / 255.0
        return self.predict_tensor(x)

    def __call__(self, image_uint8) -> dict:
        return self.predict_batch([image_uint8])[0]

    def parse(self, image_uint8, world=None):
        """(prediction dict, Parsed or None)."""
        pred = self(image_uint8)
        parsed = extract(pred["labels"], world or self.world, pred.get("heat"), pred.get("dir"),
                         self.ecfg)
        return pred, parsed


def main(argv=None):
    p = argparse.ArgumentParser(description="Parse scene images with a trained checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", default=None, help="one PNG")
    p.add_argument("--dir", default=None, help="a directory of scene_*.png")
    p.add_argument("--out", default="parsed", help="output directory for JSON + overlays")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    if not a.image and not a.dir:
        p.error("give --image or --dir")

    from viz import draw_parsed
    pr = Predictor(a.ckpt, a.device)
    paths = [Path(a.image)] if a.image else sorted(
        q for q in Path(a.dir).glob("scene_*.png") if not q.stem.endswith("_labels"))
    if a.limit:
        paths = paths[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for q in paths:
        img = load_image(q, pr.size)
        pred, parsed = pr.parse(img)
        if parsed is None:
            n_fail += 1
            rec = {"id": q.stem, "parsed": False, "parse_fail": True}
        else:
            rec = parsed_to_json(parsed, q.stem)
            draw_parsed(img, parsed).save(out / f"{q.stem}_overlay.png")
        with open(out / f"{q.stem}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
    print(f"parsed {len(paths)} image(s) -> {out}  ({n_fail} parse failures)")


if __name__ == "__main__":
    main()
