"""Run configuration: one dataclass per concern, JSON round-trip, argparse merge.

``RunConfig`` is the single resolved description of a training run.  It is
written to ``runs/<name>/config.json`` together with the git hash and the
command line, stored inside every checkpoint, and read back by ``predict.py``
so inference always uses the run's own resolution, architecture and extractor
thresholds.

This module is called ``runconfig`` and not ``config`` because ``sim/config.py``
owns that name (DESIGN.md §11.3).

Invariants:
- Every field has a default except the three dataset paths, so a config can be
  built from a JSON file that omits fields added later.
- ``from_namespace`` only overrides a field when the flag was given (``None``
  means "keep the default"), so ``--config base.json`` plus flags composes.
- Batch and AMP defaults are resolved by ``default_batch`` / ``resolve_amp``
  at run time, not stored, so a config stays portable between machines.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import _paths  # noqa: F401

SIZES = (128, 256, 512)
DEFAULT_BATCH = {128: 16, 256: 8, 512: 4}


@dataclass
class DataConfig:
    root: str = "data"
    train: str = "train_aug2"
    val: str = "val"
    size: int = 256
    num_workers: int = 4
    cache: bool = False                    # --cache ram: decode once, forces num_workers=0
    layouts: list | None = None            # restrict to these layout names


@dataclass
class ModelConfig:
    arch: str = "unet24x4"
    instance_head: bool = False
    n_classes: int = 6


@dataclass
class TrainConfig:
    epochs: int = 60
    batch: int | None = None               # None: DEFAULT_BATCH[size]
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool | None = None                # None/False: fp32 (GTX 1070 has no tensor cores)
    seed: int = 0
    dice: float = 0.0
    lambda_heat: float = 1.0
    lambda_dir: float = 1.0
    extract_every: int = 5
    extract_subset: int = 100
    preview_every: int = 10
    deterministic: bool = False


@dataclass
class ExtractConfig:
    ring_thresh: float = 0.45
    nms_factor: float = 1.6
    heat_thresh: float = 0.4
    poly_eps_px: float = 1.5
    min_wall_area_px: int = 25
    min_region_area_px: int = 100


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    out_dir: str = "runs/run"

    # --- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(
            data=_from_dict(DataConfig, d.get("data", {})),
            model=_from_dict(ModelConfig, d.get("model", {})),
            train=_from_dict(TrainConfig, d.get("train", {})),
            extract=_from_dict(ExtractConfig, d.get("extract", {})),
            out_dir=d.get("out_dir", "runs/run"),
        )

    def save(self, path, argv=None):
        """Write config.json: the resolved config plus provenance (git hash, argv)."""
        rec = self.to_dict()
        rec["git_hash"] = git_hash()
        rec["argv"] = list(sys.argv if argv is None else argv)
        rec["python"] = sys.version.split()[0]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2)

    @classmethod
    def load(cls, path) -> "RunConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    # --- argparse ----------------------------------------------------------
    @classmethod
    def from_args(cls, argv=None) -> "RunConfig":
        return cls.from_namespace(build_parser().parse_args(argv))

    @classmethod
    def from_namespace(cls, ns) -> "RunConfig":
        cfg = cls.load(ns.config) if getattr(ns, "config", None) else cls()
        for section in ("data", "model", "train", "extract"):
            sub = getattr(cfg, section)
            for f in fields(sub):
                v = getattr(ns, f.name, None)
                if v is not None:
                    setattr(sub, f.name, v)
        if getattr(ns, "cache", None) is not None:
            cfg.data.cache = ns.cache == "ram"
        if getattr(ns, "out", None):
            cfg.out_dir = ns.out
        return cfg

    # --- derived ---------------------------------------------------------
    @property
    def batch(self) -> int:
        return default_batch(self.data.size, self.train.batch)


def _from_dict(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def default_batch(size: int, override: int | None = None) -> int:
    if override:
        return int(override)
    return DEFAULT_BATCH.get(int(size), max(1, 2048 // int(size)))


def resolve_amp(tcfg: TrainConfig, device) -> bool:
    """AMP is a memory fallback only, and only exists on CUDA."""
    return bool(tcfg.amp) and getattr(device, "type", str(device)) == "cuda"


def git_hash() -> str | None:
    """Current commit from .git/HEAD without invoking git (read-only)."""
    try:
        head = (_paths.REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = _paths.REPO / ".git" / head[4:].strip()
            if ref.exists():
                return ref.read_text().strip()
            packed = _paths.REPO / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == head[4:].strip():
                        return parts[0]
            return None
        return head
    except OSError:
        return None


def build_parser(description="Train one perception run.") -> argparse.ArgumentParser:
    """Flags for every RunConfig field. Defaults are None so unset flags keep the config."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="JSON RunConfig to start from")
    p.add_argument("--out", default=None, help="run directory (runs/<name>)")

    g = p.add_argument_group("data")
    g.add_argument("--data-root", dest="root", default=None)
    g.add_argument("--train", default=None, help="training set, relative to --data-root")
    g.add_argument("--val", default=None, help="validation set, relative to --data-root")
    g.add_argument("--size", type=int, default=None, choices=SIZES)
    g.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    g.add_argument("--cache", choices=["none", "ram"], default=None,
                   help="ram: decode the whole set once (forces --num-workers 0)")
    g.add_argument("--layouts", nargs="+", default=None)

    g = p.add_argument_group("model")
    g.add_argument("--arch", default=None)
    g.add_argument("--instance-head", dest="instance_head", action="store_const", const=True,
                   default=None, help="add the heat + heading head (variant B)")

    g = p.add_argument_group("train")
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--batch", type=int, default=None, help="default 16@128, 8@256, 4@512")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    g.add_argument("--amp", action="store_const", const=True, default=None,
                   help="fp16 autocast on CUDA; a memory fallback only")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--dice", type=float, default=None, help="soft-Dice weight, 0 = off")
    g.add_argument("--lambda-heat", dest="lambda_heat", type=float, default=None)
    g.add_argument("--lambda-dir", dest="lambda_dir", type=float, default=None)
    g.add_argument("--extract-every", dest="extract_every", type=int, default=None)
    g.add_argument("--extract-subset", dest="extract_subset", type=int, default=None)
    g.add_argument("--preview-every", dest="preview_every", type=int, default=None)
    g.add_argument("--deterministic", action="store_const", const=True, default=None)

    g = p.add_argument_group("extract")
    g.add_argument("--ring-thresh", dest="ring_thresh", type=float, default=None)
    g.add_argument("--nms-factor", dest="nms_factor", type=float, default=None)
    g.add_argument("--heat-thresh", dest="heat_thresh", type=float, default=None)
    g.add_argument("--poly-eps-px", dest="poly_eps_px", type=float, default=None)
    g.add_argument("--min-wall-area-px", dest="min_wall_area_px", type=int, default=None)
    g.add_argument("--min-region-area-px", dest="min_region_area_px", type=int, default=None)
    return p
