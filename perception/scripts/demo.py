"""Live demo: one photo of a drawing -> the parsed scene, shown side by side.

    python scripts/demo.py photo.jpg                 # model in models/sketch6
    python scripts/demo.py photo.jpg --model E:\\cnn-runs\\sketch6_ft_seed0

Runs ``predict.py`` (with ``--photo``, retried without it when the corner
markers are not found) using the thresholds tuned for the model
(``tune_extract.json``), prints what was read, writes
``demo/<name>_demo.png`` = photo | overlay | clean re-rendering, and opens it.
CPU is enough (a few seconds per photo).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v5 import HERE, PY, side_by_side  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("photo")
    p.add_argument("--model", default=str(HERE / "models" / "sketch6"))
    p.add_argument("--out", default=str(HERE / "demo"))
    p.add_argument("--no-open", action="store_true")
    a = p.parse_args(argv)
    photo, model, out = Path(a.photo).resolve(), Path(a.model), Path(a.out).resolve()
    if photo.suffix.lower() in (".heic", ".heif"):           # iPhone photos: PIL cannot read them
        out.mkdir(parents=True, exist_ok=True)
        jpg = out / (photo.stem + ".jpg")
        if platform.system() != "Darwin":
            raise SystemExit("HEIC photo: convert it to JPG first")
        subprocess.run(["sips", "-s", "format", "jpeg", str(photo), "--out", str(jpg)],
                       check=True, capture_output=True)
        photo = jpg
    best = json.loads((model / "tune_extract.json").read_text())["best"]
    cmd = [PY, "predict.py", "--ckpt", str(model / "ckpt_best.pt"), "--image", str(photo),
           "--device", "cpu", "--out", str(out)]
    for k, flag in (("heat_thresh", "--heat-thresh"), ("heat_nms_factor", "--heat-nms-factor"),
                    ("letter_thresh", "--letter-thresh")):
        if best.get(k) is not None:
            cmd += [flag, str(best[k])]
    js = out / f"{photo.stem}.json"
    js.unlink(missing_ok=True)
    t0 = time.time()
    subprocess.run(cmd[:6] + ["--photo"] + cmd[6:], cwd=HERE)
    if not js.exists():
        print("corner markers not found: reading the image as it is")
        subprocess.run(cmd, cwd=HERE)
    if not js.exists():
        raise SystemExit("could not parse the photo")
    rec = json.loads(js.read_text())
    if not rec.get("parsed", True):
        raise SystemExit("parse failed: no agent or no exit found")
    d = rec.get("diagnostics", {})
    print(f"\n{photo.name}  ({time.time() - t0:.1f} s)")
    print(f"  agents    {len(rec['agents'])}   confidence "
          + " ".join(f"{s:.2f}" for s in d.get("agent_scores", [])))
    print(f"  starts    {len(rec.get('starts', []))}")
    print(f"  exits     {len(rec.get('exits', []))}")
    print(f"  walls     {len(rec.get('walls', []))}")
    print(f"  obstacles {len(rec.get('obstacles', []))}")
    sheet = out / f"{photo.stem}_demo.png"
    side_by_side([("photo", photo), ("what the model sees", out / f"{photo.stem}_overlay.png"),
                  ("parsed scene", out / f"{photo.stem}_clean.png")], sheet, h=600)
    print(f"  -> {sheet}")
    if not a.no_open:
        opener = {"Darwin": ["open"], "Windows": ["cmd", "/c", "start", ""]}.get(platform.system(), ["xdg-open"])
        subprocess.run(opener + [str(sheet)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
