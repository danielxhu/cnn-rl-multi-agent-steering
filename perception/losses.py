"""Training losses (DESIGN.md §3.3).

    L = L_seg + λ_heat · L_heat + λ_dir · L_dir

- ``L_seg``: cross-entropy with per-class weights ``w_c ∝ 1/sqrt(freq_c)``,
  normalised to mean 1 and clipped to [0.5, 20]; optional soft-Dice term.
- ``L_heat``: CenterNet penalty-reduced focal loss (α = 2, β = 4), summed over
  pixels and divided by the number of positives (agent centres).
- ``L_dir``: L1 between the predicted unit vector and ``(cos θ, sin θ)``
  inside ``dir_mask``, averaged over masked pixels.

``total_loss`` reads ``heat``/``dir`` from the model output and the batch only
when both are present, so the same function serves both variants.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import _paths  # noqa: F401


def class_weights_from_freq(freq, lo=0.5, hi=20.0) -> torch.Tensor:
    f = np.clip(np.asarray(freq, np.float64), 1e-6, None)
    w = 1.0 / np.sqrt(f)
    w = w / w.mean()
    return torch.tensor(np.clip(w, lo, hi), dtype=torch.float32)


def soft_dice(logits, target, n_classes) -> torch.Tensor:
    """1 - mean over classes of the soft Dice coefficient (classes absent from both ignored)."""
    p = torch.softmax(logits, 1)
    t = F.one_hot(target, n_classes).permute(0, 3, 1, 2).to(p.dtype)
    inter = (p * t).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


def focal_heat(pred, gt, alpha=2.0, beta=4.0) -> torch.Tensor:
    """CenterNet focal loss on sigmoid heat maps; positives are pixels with gt == 1."""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = (gt >= 1.0).to(pred.dtype)
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - gt) ** beta * neg
    n_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / torch.clamp(n_pos, min=1.0)


def masked_dir_l1(pred, gt, mask) -> torch.Tensor:
    diff = (pred - gt).abs() * mask
    return diff.sum() / torch.clamp(mask.sum() * 2, min=1.0)


def total_loss(out: dict, batch: dict, class_weights, tcfg):
    """Return (loss, {"seg": float, "heat": float, "dir": float}); missing parts are 0."""
    logits, target = out["seg"], batch["labels"]
    w = class_weights.to(logits.device) if class_weights is not None else None
    seg = F.cross_entropy(logits, target, weight=w)
    if tcfg.dice > 0:
        seg = seg + tcfg.dice * soft_dice(logits, target, logits.shape[1])
    loss = seg
    parts = {"seg": float(seg.detach()), "heat": 0.0, "dir": 0.0}
    if "heat" in out and "heat" in batch:
        lh = focal_heat(out["heat"], batch["heat"])
        loss = loss + tcfg.lambda_heat * lh
        parts["heat"] = float(lh.detach())
    if "dir" in out and "dir" in batch:
        ld = masked_dir_l1(out["dir"], batch["dir"], batch["dir_mask"])
        loss = loss + tcfg.lambda_dir * ld
        parts["dir"] = float(ld.detach())
    return loss, parts
