"""Side-by-side GIFs: a policy without wall memory | one with it, on the same scenes.

    python scripts/e7_gifs.py --suite E:\\policy-data\\test4 --rank cul_de_sac,doorway,corridor,crossing
    python scripts/e7_gifs.py --suite E:\\policy-data\\test4 --scenes scene_00131,scene_00505

``--rank`` runs both checkpoints on the first ``--per-layout`` scenes of each
layout and prints, per layout, the scenes where the second model gains or
loses most (success, then loop count). ``--scenes`` writes
``<out>/<layout>_<scene>.gif`` and a ``_final.png`` of the last frame.

Invariants:
- Episodes are deterministic (``load_policy`` default), so ``--scenes`` after
  ``--rank`` replays exactly the ranked episodes.
- Read-only on models and suites; CPU only, no process pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(POLICY / "scripts"))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from demo_e2e import AGENT_COLORS, OUTCOME_COLOR, to_px  # noqa: E402
from evaluate import LOOP_TURN, load_policy, run_episode, turning  # noqa: E402
from policyconfig import default_reward  # noqa: E402
from render import render  # noqa: E402
from suites import load_suite  # noqa: E402

LABEL = {"goal": "goal", "collision": "crash", "timeout": "timeout", None: "timeout"}
HEAD = 64


def font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def episodes(models, scenes):
    """{(model_dir_name, scene_name): run_episode row + per-agent ``loops``}."""
    res, cfg = {}, None
    for m in models:
        ctrl, _, trained, _ = load_policy(POLICY / "models" / m / "ckpt_best.pt")
        cfg = default_reward(trained)
        for n, s in scenes:
            r = run_episode(s, ctrl, cfg, record=True)
            r["loops"] = [turning(t) >= LOOP_TURN for t in r["trajectories"]]
            res[(m, n)] = r
    return res, cfg


def panel(bg, rec, s, title, cfg, F, Fs):
    im = bg.copy()
    dr = ImageDraw.Draw(im)
    rad = cfg.agent_radius * cfg.scale
    for i, t in enumerate(rec["trajectories"]):
        t = np.asarray(t, float)
        x, y = to_px(t[:min(s, len(t) - 1) + 1], cfg)
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        if len(x) > 1:
            dr.line(list(zip(x, y)), fill=c, width=3)
        done = s >= len(t) - 1
        fill = OUTCOME_COLOR[rec["outcomes"][i]] if done else c
        dr.ellipse([x[-1] - rad, y[-1] - rad, x[-1] + rad, y[-1] + rad], fill=fill,
                   outline="black", width=2)
        if done and rec["outcomes"][i] == "collision":
            k = rad + 3
            dr.line([x[-1] - k, y[-1] - k, x[-1] + k, y[-1] + k], fill="black", width=3)
            dr.line([x[-1] - k, y[-1] + k, x[-1] + k, y[-1] - k], fill="black", width=3)
    full = Image.new("RGB", (im.width, im.height + HEAD), "white")
    hd = ImageDraw.Draw(full)
    hd.text((8, 4), title, fill="black", font=F)
    xx = 8
    for i, o in enumerate(rec["outcomes"]):
        done = s >= len(rec["trajectories"][i]) - 1
        lab = f"A{i}: " + (LABEL[o] if done else "...") + (" (loop)" if rec["loops"][i] else "")
        hd.ellipse([xx, 30, xx + 10, 40], fill=AGENT_COLORS[i % len(AGENT_COLORS)])
        hd.text((xx + 14, 27), lab, fill=OUTCOME_COLOR[o] if done else "black", font=Fs)
        xx += 14 + int(hd.textlength(lab, font=Fs)) + 12
    hd.text((8, 46), f"step {s}", fill="gray", font=Fs)
    full.paste(im, (0, HEAD))
    return full


def gif(scene, name, a, b, titles, cfg, out, every=3, ms=50):
    F, Fs = font(15), font(12)
    bg = render(scene, cfg)[0].convert("RGB")
    T = max(len(t) for t in a["trajectories"] + b["trajectories"])
    frames = []
    for s in list(range(0, T, every)) + [T - 1]:
        p1 = panel(bg, a, s, f"{titles[0]}  {a['success_rate']:.0%}", cfg, F, Fs)
        p2 = panel(bg, b, s, f"{titles[1]}  {b['success_rate']:.0%}", cfg, F, Fs)
        fr = Image.new("RGB", (p1.width * 2 + 12, p1.height), "white")
        fr.paste(p1, (0, 0))
        fr.paste(p2, (p1.width + 12, 0))
        frames.append(fr.quantize(64))
    frames += [frames[-1]] * 25                       # hold the last frame
    path = out / f"{scene.layout}_{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0,
                   optimize=True)
    frames[-1].convert("RGB").save(out / f"{scene.layout}_{name}_final.png")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--models", default="e3_both_seed0,e7_map_seed0")
    ap.add_argument("--titles", default="e3_both (no wall memory),e7_map (wall memory)")
    ap.add_argument("--out", type=Path, default=POLICY / "results" / "e7_gifs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rank", help="comma list of layouts")
    g.add_argument("--scenes", help="comma list of scene names")
    ap.add_argument("--per-layout", type=int, default=40)
    a = ap.parse_args(argv)
    models, titles = a.models.split(","), a.titles.split(",")
    suite = load_suite(a.suite)

    if a.rank:
        layouts = a.rank.split(",")
        pick = []
        for L in layouts:
            pick += [(n, s) for n, s in suite if s.layout == L][:a.per_layout]
        res, _ = episodes(models, pick)
        diffs = []
        for n, s in pick:
            r0, r1 = res[(models[0], n)], res[(models[1], n)]
            diffs.append((r1["success_rate"] - r0["success_rate"],
                          sum(r0["loops"]) - sum(r1["loops"]), s.layout, n))
        for L in layouts:
            d = sorted([x for x in diffs if x[2] == L], key=lambda x: (-x[0], -x[1]))
            print(L)
            for x in d[:3] + d[-1:]:
                print(f"  {x[3]}  success {x[0]:+.2f}  loops {x[1]:+d}")
        print(f"{models[1]} better / tie / worse: {sum(x[0] > 0 for x in diffs)} / "
              f"{sum(x[0] == 0 for x in diffs)} / {sum(x[0] < 0 for x in diffs)}")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    want = a.scenes.split(",")
    pick = [(n, s) for n, s in suite if n in want]
    res, cfg = episodes(models, pick)
    for n, s in pick:
        print(gif(s, n, res[(models[0], n)], res[(models[1], n)], titles, cfg, a.out))


if __name__ == "__main__":
    main()
