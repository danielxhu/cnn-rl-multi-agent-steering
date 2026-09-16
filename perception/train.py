"""Training CLI: one run -> runs/<name>/ (DESIGN.md §3.4).

    python train.py --data-root data --train train_aug2 --val val --size 256 \\
                    --arch unet24x4 --seed 0 --out runs/dev_s256
    python train.py ... --instance-head            # variant B
    python train.py ... --resume                   # continue from ckpt_last.pt

Run directory contents:
    config.json     resolved RunConfig + git hash + argv
    log.csv         one row per epoch (LOG_COLUMNS)
    ckpt_last.pt    every epoch: model, optimiser, scaler, scheduler, epoch, RNG
    ckpt_best.pt    on the selection metric (agent F1 on the val subset when the
                    extractor ran this epoch, else agent-class IoU)
    previews/       epoch_NN.png: input / GT labels / predicted labels / parsed overlay

Invariants:
- fp32 by default; ``--amp`` is a CUDA-only memory fallback (GTX 1070 has no
  tensor cores).  No bf16, no torch.compile.
- The schedule (1 epoch linear warm-up, cosine to 1e-5) is stepped per
  iteration and expressed in epochs, so it scales with ``--epochs``.
- ``device`` is resolved once and passed down; nothing calls ``.cuda()``.
- Resuming restores optimiser, scaler, scheduler and RNG state, so an
  interruption costs at most one epoch.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from config import C_AGENT, N_CLASSES
from data import SceneDataset, class_frequencies, make_loader
from extract import extract
from losses import class_weights_from_freq, total_loss
from metrics import aggregate, scene_metrics
from model import build_model, count_params
from predict import predict_dataset, resolve_device
from runconfig import RunConfig, build_parser, resolve_amp
from viz import grid, preview_row

LOG_COLUMNS = ["epoch", "lr", "train_loss", "loss_seg", "loss_heat", "loss_dir",
               "val_pixacc", "val_miou", "val_iou_agent", "val_agent_f1",
               "val_agent_center_err", "seconds"]
LR_MIN = 1e-5


# ------------------------------------------------------------- utilities

def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def make_scheduler(opt, total_iters: int, warm_iters: int, lr: float):
    floor = LR_MIN / lr

    def factor(it):
        if it < warm_iters:
            return (it + 1) / max(1, warm_iters)
        t = (it - warm_iters) / max(1, total_iters - warm_iters)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def save_checkpoint(path, model, cfg: RunConfig, optimizer=None, scheduler=None, scaler=None,
                    epoch=0, best_metric=None, world=None):
    rec = {"model": model.state_dict(), "config": cfg.to_dict(), "epoch": int(epoch),
           "best_metric": best_metric, "world": world, "rng": rng_state()}
    if optimizer is not None:
        rec["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        rec["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        rec["scaler"] = scaler.state_dict()
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(rec, tmp)
    tmp.replace(path)                       # never leave a half-written checkpoint


def make_grad_scaler(enabled: bool):
    """torch.amp.GradScaler (torch ≥ 2.3) with the older torch.cuda.amp fallback."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# ------------------------------------------------------------ validation

@torch.no_grad()
def validate_pixels(model, loader, device) -> dict:
    """Dataset-level pixel accuracy, per-class IoU and mIoU from one confusion matrix."""
    model.eval()
    conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int64, device=device)
    for batch in loader:
        b = to_device(batch, device)
        pred = model(b["image"])["seg"].argmax(1)
        idx = b["labels"].reshape(-1) * N_CLASSES + pred.reshape(-1)
        conf += torch.bincount(idx, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    conf = conf.cpu().numpy().astype(np.float64)
    tp = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return {"pixacc": float(tp.sum() / max(1.0, conf.sum())),
            "miou": float(np.nanmean(iou)), "iou_agent": float(iou[C_AGENT])}


def validate_extract(model, ds: SceneDataset, indices, device, cfg: RunConfig, batch: int):
    """Full extractor on a fixed val subset -> (agent_f1, agent_center_err, predictions)."""
    preds = predict_dataset(model, ds, indices, device, batch)
    rows, arows = [], []
    for i, pr in zip(indices, preds):
        parsed = extract(pr["labels"], ds.worlds[i], pr.get("heat"), pr.get("dir"), cfg.extract)
        m, ar = scene_metrics(ds.scenes[i], ds.labels512(i), parsed, pr["labels"], ds.worlds[i],
                              ds.size, sid=ds.ids[i])
        m["layout"] = ds.layouts[i]
        rows.append(m)
        arows.extend(ar)
    agg = aggregate(rows, arows)["overall"]
    return agg["agent_f1"], agg["agent_center_err"], preds


def write_preview(model, ds: SceneDataset, indices, path, device, cfg: RunConfig, batch: int):
    preds = predict_dataset(model, ds, indices, device, batch)
    rows = []
    for i, pr in zip(indices, preds):
        parsed = extract(pr["labels"], ds.worlds[i], pr.get("heat"), pr.get("dir"), cfg.extract)
        rows.append(preview_row(ds.image_uint8(i), ds.labels_uint8(i), pr["labels"], parsed,
                                ds.scenes[i]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid(rows).save(path)


def subset_indices(n: int, k: int) -> list:
    """`k` indices spread evenly over `n` scenes (all of them when n <= k)."""
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


# --------------------------------------------------------------- training

def train(cfg: RunConfig, resume: bool = False, device=None) -> Path:
    run_dir = Path(cfg.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    tcfg, dcfg = cfg.train, cfg.data
    seed_everything(tcfg.seed, tcfg.deterministic)
    amp = resolve_amp(tcfg, device)
    batch = cfg.batch

    root = Path(dcfg.root)
    train_ds = SceneDataset(root / dcfg.train, dcfg.size, instance_targets=cfg.model.instance_head,
                            layouts=dcfg.layouts, cache=dcfg.cache)
    val_ds = SceneDataset(root / dcfg.val, dcfg.size, instance_targets=False, layouts=dcfg.layouts)
    class_weights = class_weights_from_freq(class_frequencies(train_ds))
    workers = 0 if dcfg.cache else dcfg.num_workers
    train_loader = make_loader(train_ds, batch, True, workers, tcfg.seed, drop_last=len(train_ds) > batch)
    val_loader = make_loader(val_ds, batch, False, workers, tcfg.seed)
    subset = subset_indices(len(val_ds), tcfg.extract_subset)
    preview_idx = subset[:: max(1, len(subset) // 4)][:4]

    model = build_model(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    iters_per_epoch = max(1, len(train_loader))
    sched = make_scheduler(opt, tcfg.epochs * iters_per_epoch, iters_per_epoch, tcfg.lr)
    scaler = make_grad_scaler(amp)

    start_epoch, best = 0, -math.inf
    last = run_dir / "ckpt_last.pt"
    if resume and last.exists():
        rec = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(rec["model"])
        if "optimizer" in rec:
            opt.load_state_dict(rec["optimizer"])
        if "scheduler" in rec:
            sched.load_state_dict(rec["scheduler"])
        if "scaler" in rec and amp:
            scaler.load_state_dict(rec["scaler"])
        if rec.get("rng"):
            set_rng_state(rec["rng"])
        start_epoch = int(rec["epoch"])
        best = rec.get("best_metric") if rec.get("best_metric") is not None else -math.inf
        print(f"resumed from epoch {start_epoch}")
    cfg.save(run_dir / "config.json")
    world = train_ds.worlds[0]

    log_path = run_dir / "log.csv"
    new_log = not (resume and start_epoch > 0 and log_path.exists())
    log_fh = open(log_path, "w" if new_log else "a", newline="")
    log = csv.DictWriter(log_fh, fieldnames=LOG_COLUMNS)
    if new_log:
        log.writeheader()

    print(f"{cfg.model.arch}{' +inst' if cfg.model.instance_head else ''}: "
          f"{count_params(model) / 1e6:.3f} M params, size {dcfg.size}, batch {batch}, "
          f"{len(train_ds)} train / {len(val_ds)} val scenes, device {device}, amp {amp}")

    for epoch in range(start_epoch, tcfg.epochs):
        t0 = time.time()
        model.train()
        sums = {"loss": 0.0, "seg": 0.0, "heat": 0.0, "dir": 0.0, "lr": 0.0}
        n_batches = 0
        for batch_cpu in train_loader:
            b = to_device(batch_cpu, device)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(b["image"])
                loss, parts = total_loss(out, b, class_weights, tcfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums["lr"] += opt.param_groups[0]["lr"]              # mean lr over the epoch
            sums["loss"] += float(loss.detach())
            for k in ("seg", "heat", "dir"):
                sums[k] += parts[k]
            n_batches += 1
        n_batches = max(1, n_batches)

        pix = validate_pixels(model, val_loader, device)
        is_last = epoch + 1 == tcfg.epochs
        do_extract = tcfg.extract_every > 0 and ((epoch + 1) % tcfg.extract_every == 0 or is_last)
        f1 = cerr = None
        if do_extract:
            f1, cerr, _ = validate_extract(model, val_ds, subset, device, cfg, batch)
        metric = f1 if do_extract else (pix["iou_agent"] if tcfg.extract_every <= 0 else None)

        row = {"epoch": epoch + 1, "lr": sums["lr"] / n_batches,
               "train_loss": sums["loss"] / n_batches, "loss_seg": sums["seg"] / n_batches,
               "loss_heat": sums["heat"] / n_batches, "loss_dir": sums["dir"] / n_batches,
               "val_pixacc": pix["pixacc"], "val_miou": pix["miou"],
               "val_iou_agent": pix["iou_agent"],
               "val_agent_f1": "" if f1 is None else f1,
               "val_agent_center_err": "" if cerr is None else cerr,
               "seconds": time.time() - t0}
        log.writerow(row)
        log_fh.flush()

        if metric is not None and not math.isnan(metric) and metric > best:
            best = metric
            save_checkpoint(run_dir / "ckpt_best.pt", model, cfg, epoch=epoch + 1,
                            best_metric=best, world=world)
        save_checkpoint(last, model, cfg, opt, sched, scaler, epoch + 1,
                        best if best > -math.inf else None, world)
        if tcfg.preview_every > 0 and ((epoch + 1) % tcfg.preview_every == 0 or is_last):
            write_preview(model, val_ds, preview_idx, run_dir / "previews" / f"epoch_{epoch + 1:02d}.png",
                          device, cfg, batch)
        print(f"epoch {epoch + 1}/{tcfg.epochs}  loss {row['train_loss']:.4f}  "
              f"pixacc {pix['pixacc']:.4f}  miou {pix['miou']:.3f}  iou_agent {pix['iou_agent']:.3f}"
              + (f"  agent_f1 {f1:.3f}  cerr {cerr:.3f}" if f1 is not None else "")
              + f"  {row['seconds']:.0f}s", flush=True)

    log_fh.close()
    if not (run_dir / "ckpt_best.pt").exists():      # no selection epoch ran: last is best
        save_checkpoint(run_dir / "ckpt_best.pt", model, cfg, epoch=tcfg.epochs, world=world)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
        with open(run_dir / "memory.json", "w") as fh:
            json.dump({"max_memory_allocated_gb": peak}, fh)
        print(f"peak GPU memory {peak:.2f} GB")
    return run_dir


def main(argv=None):
    p = build_parser()
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt_last.pt")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ns = p.parse_args(argv)
    cfg = RunConfig.from_namespace(ns)
    if ns.out is None:
        p.error("--out is required")
    train(cfg, resume=ns.resume, device=ns.device)


if __name__ == "__main__":
    main()
