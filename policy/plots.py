"""Figures of the E2 report (DESIGN.md §6). Pure drawing: aggregate.py hands in numbers.

Owns the four figures; each function takes plain Python / numpy data and a
target path and writes a PNG.

Invariants:
- Controller colours are fixed across figures: greedy grey, mlp blue, gru orange
  (colour-blind safe), so a reader never has to re-learn the legend.
- World -> pixel for paths follows ``render``: ``px = x * scale``,
  ``py = (world - y) * scale`` (image row 0 is the top of the world).
- No display is needed (Agg backend).
"""
from __future__ import annotations

import _paths  # noqa: F401

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"greedy": "#7f7f7f", "mlp": "#1f77b4", "gru": "#ff7f0e"}
ORDER = ["greedy", "mlp", "gru"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def training_curves(curves: dict, path):
    """curves: {memory: [(agent_steps array, success array), ...] one per seed}."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for mem in ("mlp", "gru"):
        runs = curves.get(mem) or []
        if not runs:
            continue
        top = min(r[0][-1] for r in runs)
        grid = np.linspace(0, top, 100)
        ys = np.stack([np.interp(grid, r[0], r[1]) for r in runs])
        mu, sd = ys.mean(0), ys.std(0)
        ax.plot(grid / 1e6, mu, color=COLORS[mem], label=f"{mem} (n={len(runs)})")
        ax.fill_between(grid / 1e6, mu - sd, mu + sd, color=COLORS[mem], alpha=0.2)
    ax.set_xlabel("agent-steps (millions)")
    ax.set_ylabel("validation success rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_layout_bars(values: dict, layouts: list, path, title="test4: success per layout"):
    """values: {controller: {layout: [per-seed success, ...]}}."""
    ctrls = [c for c in ORDER if c in values]
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(layouts)), 4))
    w = 0.8 / max(1, len(ctrls))
    for k, c in enumerate(ctrls):
        mu = [np.mean(values[c].get(l, [np.nan])) for l in layouts]
        sd = [np.std(values[c].get(l, [np.nan])) for l in layouts]
        ax.bar(np.arange(len(layouts)) + (k - (len(ctrls) - 1) / 2) * w, mu, w,
               yerr=sd, color=COLORS[c], label=c, capsize=2)
    ax.set_xticks(range(len(layouts)))
    ax.set_xticklabels(layouts, rotation=30, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bottleneck_curves(values: dict, bins: list, path, title="test4: success vs bottleneck width"):
    """values: {controller: {bin: [per-seed success]}}; bins in display order."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in [c for c in ORDER if c in values]:
        mu = np.array([np.mean(values[c].get(b, [np.nan])) for b in bins])
        sd = np.array([np.std(values[c].get(b, [np.nan])) for b in bins])
        ax.errorbar(range(len(bins)), mu, yerr=sd, color=COLORS[c], marker="o", capsize=3,
                    label=c)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_xlabel("bottleneck width (world units)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def path_panels(panels: list, layouts: list, sim_cfg, path):
    """panels: {layout: {"scene": Scene, "paths": {controller: record}}}.

    record = {"trajectories": [[(x, y), ...] per agent], "collision_pos": [...]}.
    """
    from render import render
    n = len(layouts)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.7 * rows), squeeze=False)
    sc = sim_cfg.scale
    for ax in axes.ravel():
        ax.axis("off")
    for ax, layout in zip(axes.ravel(), layouts):
        item = panels.get(layout)
        if item is None:
            continue
        img, _ = render(item["scene"], sim_cfg)
        ax.imshow(np.asarray(img))
        for c in ORDER:
            rec = item["paths"].get(c)
            if rec is None:
                continue
            for k, tr in enumerate(rec["trajectories"]):
                xy = np.array(tr)
                ax.plot(xy[:, 0] * sc, (sim_cfg.world - xy[:, 1]) * sc, color=COLORS[c],
                        lw=1.3, alpha=0.9, label=c if k == 0 else None)
            for cp in rec["collision_pos"]:
                if cp is not None:
                    ax.plot(cp[0] * sc, (sim_cfg.world - cp[1]) * sc, "x", color=COLORS[c],
                            ms=7, mew=2)
        ax.set_title(layout, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
