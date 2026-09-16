"""U-Net pixel classifier with an optional instance head.

Backbone: ``levels`` encoder blocks of two ``3×3 conv → BN → ReLU``, widths
doubling per level (24 → 48 → 96 → 192 for ``unet24x4``), ``MaxPool2d(2)``
between them, ``ConvTranspose2d(2, stride 2)`` up, skip concatenation.
``levels=4`` therefore means three poolings: the input side must be divisible
by 2^(levels-1), which 128 / 256 / 512 all are, so there is no padding logic.

Heads on the final (B, base_ch, S, S) features:
    seg  : 1×1 conv → (B, 6, S, S) logits, always present
    inst : 1×1 conv → (B, 3, S, S) = [heat, cos, sin], optional;
           heat → sigmoid, (cos, sin) → L2-normalised

``forward`` returns a dict so the training loop and the extractor are
indifferent to the variant.  Parameter counts (DESIGN.md §3.1):
unet24x4 1.086 M, unet24x5 4.37 M, unet16x4 0.48 M; the instance head adds 75.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import _paths  # noqa: F401

ARCHS = {
    "unet24x4": dict(base_ch=24, levels=4),
    "unet24x5": dict(base_ch=24, levels=5),
    "unet16x4": dict(base_ch=16, levels=4),
}


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_ch=24, levels=4, n_classes=6, instance_head=False):
        super().__init__()
        if levels < 2:
            raise ValueError("a U-Net needs at least two levels")
        self.base_ch, self.levels = int(base_ch), int(levels)
        self.n_classes, self.instance_head = int(n_classes), bool(instance_head)
        widths = [base_ch * 2 ** k for k in range(levels)]

        self.enc = nn.ModuleList()
        cin = 3
        for w in widths:
            self.enc.append(_block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for k in range(levels - 1, 0, -1):          # deepest first
            self.up.append(nn.ConvTranspose2d(widths[k], widths[k - 1], 2, stride=2))
            self.dec.append(_block(2 * widths[k - 1], widths[k - 1]))
        self.seg = nn.Conv2d(widths[0], n_classes, 1)
        self.inst = nn.Conv2d(widths[0], 3, 1) if instance_head else None
        if self.inst is not None:
            # CenterNet prior: start the heat map near 0.1 so the focal loss is not
            # dominated by 99 % negative pixels at initialisation.
            nn.init.constant_(self.inst.bias[0:1], -2.19)

    @property
    def divisor(self) -> int:
        return 2 ** (self.levels - 1)

    def forward(self, x) -> dict:
        if x.shape[-1] % self.divisor or x.shape[-2] % self.divisor:
            raise ValueError(f"input side must be divisible by {self.divisor}, got {tuple(x.shape)}")
        skips = []
        for k, block in enumerate(self.enc):
            x = block(x if k == 0 else self.pool(x))
            skips.append(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        out = {"seg": self.seg(x)}
        if self.inst is not None:
            h = self.inst(x)
            out["heat"] = torch.sigmoid(h[:, 0:1])
            out["dir"] = F.normalize(h[:, 1:3], dim=1, eps=1e-6)
        return out


def build_model(mcfg) -> UNet:
    if mcfg.arch not in ARCHS:
        raise ValueError(f"unknown arch {mcfg.arch!r}; choose from {sorted(ARCHS)}")
    return UNet(n_classes=mcfg.n_classes, instance_head=mcfg.instance_head, **ARCHS[mcfg.arch])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
