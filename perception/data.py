"""Dataset loading: generator output on disk -> tensors at the working resolution.

Everything on disk is 512×512 (``sim/generate.py``); the 128 / 256 conditions
are produced here by downsampling at load time (``INTER_AREA`` for the image,
``INTER_NEAREST`` for the label indices).  The training code does no
augmentation: augmentation lives in the renderer (DESIGN.md §2).

Invariants:
- Images are float32 in [0, 1], layout (3, S, S); no normalisation constants
  to keep in sync between training and inference.
- Labels are read as raw palette indices (mode "P", no conversion) and are
  int64 (S, S) with values < 6.
- The dataset is picklable (no lambdas, no open handles), so it survives
  ``spawn`` DataLoader workers on Windows.  With ``cache=True`` every image is
  decoded once at construction and ``num_workers`` must be 0.
- ``scenes[i]``, ``worlds[i]``, ``ids[i]`` and ``layouts[i]`` are the ground
  truth for sample ``i`` so evaluation can reach it by index.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import _paths  # noqa: F401
from config import N_CLASSES, Config
from scene import Scene
from targets import instance_targets

MIN_SCENES_TO_CACHE = 100                      # class_freq.json is only worth writing for real sets


# ----------------------------------------------------------------- readers

def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """uint8 (H, W, 3) -> (size, size, 3) with area averaging (no-op at size)."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_labels(lab: np.ndarray, size: int) -> np.ndarray:
    """uint8 (H, W) class indices -> (size, size), nearest (indices stay exact)."""
    if lab.shape[0] == size and lab.shape[1] == size:
        return lab
    return cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST)


def load_image(path, size: int) -> np.ndarray:
    """PNG -> uint8 (S, S, 3) RGB."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), np.uint8)
    return resize_image(arr, size)


def load_labels(path, size: int) -> np.ndarray:
    """Palette PNG -> uint8 (S, S) of raw class indices (no palette lookup)."""
    with Image.open(path) as im:
        if im.mode != "P" and im.mode != "L":
            raise ValueError(f"{path}: expected a palette label PNG, got mode {im.mode}")
        arr = np.asarray(im, np.uint8)
    return resize_labels(arr, size)


def world_from_manifest(root: Path) -> dict | None:
    """The `world` block a scene JSON would carry, from dataset.json's config."""
    p = root / "dataset.json"
    if not p.exists():
        return None
    with open(p) as fh:
        c = json.load(fh).get("config", {})
    if not c:
        return None
    return {"size": c["world"], "img_size": c["img_size"],
            "agent_radius": c["agent_radius"], "wall_thickness": c["wall_thickness"],
            "fov_deg": c["fov_deg"], "fov_range": c["fov_range"]}


def scene_files(root: Path) -> list:
    """Sorted scene stems (`scene_00001`) that have image, labels and JSON."""
    stems = []
    for p in sorted(root.glob("scene_*.json")):
        stem = p.stem
        if (root / f"{stem}.png").exists() and (root / f"{stem}_labels.png").exists():
            stems.append(stem)
    return stems


# ----------------------------------------------------------------- dataset

class SceneDataset(Dataset):
    def __init__(self, root, size, instance_targets=False, layouts=None, limit=None,
                 cache=False):
        self.root = Path(root)
        self.size = int(size)
        self.instance_targets = bool(instance_targets)
        self.cache = bool(cache)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {self.root}")

        default_world = world_from_manifest(self.root)
        self.ids, self.scenes, self.worlds, self.layouts = [], [], [], []
        for stem in scene_files(self.root):
            with open(self.root / f"{stem}.json") as fh:
                d = json.load(fh)
            layout = d.get("layout") or "unknown"
            if layouts is not None and layout not in layouts:
                continue
            world = d.get("world") or default_world
            if world is None:
                raise ValueError(f"{stem}: no `world` block and no dataset.json")
            self.ids.append(stem)
            self.scenes.append(Scene.from_dict(d))
            self.worlds.append(dict(world))
            self.layouts.append(layout)
            if limit is not None and len(self.ids) >= limit:
                break
        if not self.ids:
            raise ValueError(f"no scenes found in {self.root}")

        self._images = self._labels = None
        if self.cache:
            self._images = np.stack([load_image(self.image_path(i), self.size)
                                     for i in range(len(self))])
            self._labels = np.stack([load_labels(self.labels_path(i), self.size)
                                     for i in range(len(self))])

    # -- paths ------------------------------------------------------------
    def image_path(self, i) -> Path:
        return self.root / f"{self.ids[i]}.png"

    def labels_path(self, i) -> Path:
        return self.root / f"{self.ids[i]}_labels.png"

    def json_path(self, i) -> Path:
        return self.root / f"{self.ids[i]}.json"

    # -- raw access (numpy, no torch) ----------------------------------------
    def image_uint8(self, i) -> np.ndarray:
        if self._images is not None:
            return self._images[i]
        return load_image(self.image_path(i), self.size)

    def labels_uint8(self, i) -> np.ndarray:
        if self._labels is not None:
            return self._labels[i]
        return load_labels(self.labels_path(i), self.size)

    def labels512(self, i) -> np.ndarray:
        """Full-resolution ground-truth labels (for wall surface metrics)."""
        with Image.open(self.labels_path(i)) as im:
            return np.asarray(im, np.uint8)

    # -- torch ------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i) -> dict:
        img = self.image_uint8(i)
        lab = self.labels_uint8(i)
        out = {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0,
            "labels": torch.from_numpy(lab.astype(np.int64)),
            "index": i,
        }
        if self.instance_targets:
            heat, dirs, mask = instance_targets(self.scenes[i], self.worlds[i], self.size)
            out["heat"] = torch.from_numpy(heat)
            out["dir"] = torch.from_numpy(dirs)
            out["dir_mask"] = torch.from_numpy(mask)
        return out

    def sim_config(self) -> Config:
        """A sim Config matching the generator's world (for ground-truth simulation)."""
        p = self.root / "dataset.json"
        if p.exists():
            with open(p) as fh:
                raw = json.load(fh).get("config")
            if raw:
                from config import StyleConfig
                style = StyleConfig(**raw.pop("style", {}))
                return Config(style=style, **raw)
        w = self.worlds[0]
        return Config(world=w["size"], img_size=w["img_size"], agent_radius=w["agent_radius"],
                      wall_thickness=w["wall_thickness"], fov_deg=w["fov_deg"],
                      fov_range=w["fov_range"])


# ----------------------------------------------------------------- loaders

def make_loader(ds: SceneDataset, batch: int, shuffle: bool, num_workers: int,
                seed: int = 0, drop_last: bool = False) -> DataLoader:
    """Spawn-safe DataLoader; RAM-cached datasets never fork workers."""
    if ds.cache:
        num_workers = 0
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers,
        persistent_workers=num_workers > 0, drop_last=drop_last, generator=gen,
        pin_memory=torch.cuda.is_available(),
    )


def class_frequencies(ds: SceneDataset) -> np.ndarray:
    """Fraction of pixels per class over the whole set; cached in <root>/class_freq.json.

    Frequencies depend on the working resolution (thin classes shrink when
    downsampled), so the cache is keyed by size.
    """
    key = str(ds.size)
    cache_path = ds.root / "class_freq.json"
    cached = {}
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
        except (OSError, ValueError):
            cached = {}
        if key in cached and len(cached[key]) == N_CLASSES:
            return np.asarray(cached[key], np.float64)

    counts = np.zeros(N_CLASSES, np.int64)
    for i in range(len(ds)):
        counts += np.bincount(ds.labels_uint8(i).ravel(), minlength=N_CLASSES)[:N_CLASSES]
    freq = counts / max(1, counts.sum())
    cached[key] = freq.tolist()
    if len(ds) >= MIN_SCENES_TO_CACHE:         # tiny sets (sim/examples) recompute in no time
        try:
            with open(cache_path, "w") as fh:
                json.dump(cached, fh, indent=1)
        except OSError:
            pass                               # read-only dataset dir: recompute next time
    return freq
