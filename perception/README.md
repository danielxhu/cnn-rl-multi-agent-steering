# Perception: Image → Label Map → Scene

Turns a rendered scene image into a `Scene` the simulator can step: a U-Net
labels every pixel, then deterministic geometry extraction recovers agent
poses, obstacle circles, wall surfaces and start / goal regions from the label
map. Both stages train and test on `sim/` output alone; no hand annotation.
The design and its reasons are in [DESIGN.md](DESIGN.md); this file is the
how-to.

---

## Run

Python 3.10+ and the repository's `requirements.txt`. On the Windows training
machine install the CUDA build of PyTorch afterwards (the PyPI wheel is
CPU-only, and the GTX 1070 needs a cu126 / cu118 wheel):

```bash
pip install -r ../requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Everything runs from this directory.

```bash
python tests/run_tests.py                          # 12 tests, CPU, ~10 s
python scripts/make_datasets.py --jobs 4           # every dataset of DESIGN.md §2, ~6 min
python train.py --data-root data --train train_aug2 --val val --size 256 --out runs/dev_s256
python evaluate.py --ckpt runs/dev_s256/ckpt_best.pt
python sweep.py --grid e5 && python aggregate.py   # the E5 sweep, then tables and figures
```

No system Python with cv2? `uv run --python 3.12 --with numpy --with opencv-python-headless --with pillow --with torch --with matplotlib python <script>`

---

## Input and output

**Input:** a dataset directory written by `sim/generate.py` (`scene_*.png`,
`scene_*_labels.png`, `scene_*.json`, `dataset.json`), always 512 px; the
128 / 256 conditions are downsampled at load time.

**Output of a training run** (`runs/<name>/`):

| File | Contents |
|---|---|
| `config.json` | every parameter of the run, git hash, argv |
| `log.csv` | one row per epoch: losses, val pixel accuracy / mIoU / agent IoU, agent F1 |
| `ckpt_last.pt`, `ckpt_best.pt` | resumable state; best by agent F1 on a fixed val subset |
| `metrics.json`, `per_scene.csv`, `agent_rows.csv` | written by `evaluate.py`: per set, per scene, per agent |
| `fig_*.png`, `previews/` | spacing curve, example parses, predicted vs. true labels |

**Output of `predict.py`:** one JSON per image in the generator's schema, plus
an overlay PNG.

```json
{
  "id": "scene_00007", "parsed": true, "layout": null, "bottleneck": null,
  "world": {"size": 100.0, "img_size": 256, "agent_radius": 2.0,
            "wall_thickness": 0.0, "fov_deg": 220.0, "fov_range": 22.0},
  "walls": [[[x, y], [x, y]], "..."],
  "obstacles": [{"c": [x, y], "r": 6.1}],
  "start_region": [x0, y0, x1, y1], "goal_region": [x0, y0, x1, y1],
  "groups": [["start", "goal"], "..."],
  "agents": [{"pos": [x, y], "heading": 0.31}],
  "diagnostics": {"agent_scores": [0.83], "heading_uncertain": [false], "...": "..."}
}
```

- `walls` are **surface** edges of the predicted wall mask, thickness 0. Build
  the simulator with `extract.sim_config(cfg)`, never with the generator's
  config, and never re-render a parsed scene with `render.render`.
- `layout` and `bottleneck` are unknown; `fov_*` are copied from the
  generator config (the field of view is not drawn).

```python
import _paths
from predict import Predictor
from extract import sim_config
from physics import Simulator
from config import Config

pred, parsed = Predictor("runs/dev_s256/ckpt_best.pt").parse(image_uint8)
sim = Simulator(parsed.scene, sim_config(Config()))
```

`extract.py` and `metrics.py` need only numpy and cv2, so parsing runs without torch.

---

## Model

| Name | Base width | Levels | Params | Use |
|---|---|---|---|---|
| `unet24x4` | 24 | 4 | 1.09 M | default; the proposal's model |
| `unet24x5` | 24 | 5 | 4.37 M | twice the receptive field, for 512 |
| `unet16x4` | 16 | 4 | 0.48 M | smoke runs |

Every model outputs 6-class logits. `--instance-head` adds three channels
(agent-centre heatmap, cos θ, sin θ) on the same features (+75 parameters);
with it the extractor reads agent poses from the heatmap instead of the
label map. Losses: class-weighted cross-entropy (+ optional Dice), CenterNet
focal loss on the heatmap, L1 on the heading vector inside agent disks.
AdamW, cosine schedule, fp32 (the training GPU has no tensor cores).

---

## Extraction

Label map → `Scene`, all thresholds in `runconfig.ExtractConfig`:

| Class | Method |
|---|---|
| agent | fixed-radius ring vote (the Hough accumulator for a known radius) + non-maximum suppression, so touching outlines still give two peaks; circle fit refines the centre; heading from the pixels of the line sticking out of the ring |
| obstacle | connected components, least-squares circle on the **outer boundary** (interior pixels the net may call "wall" do not matter) |
| wall | contour polygons of the wall mask, each edge a zero-thickness segment (works for thin walls and filled dead space alike) |
| start / goal | bounding boxes of the region components; paired by axis overlap for `crossing`; agents join the start region that contains them |

Returns `None` (a parse failure) only when no start–goal pair or no agent is found.

Measured on ground-truth label maps (a perfect network): max centre error
0.30 / 0.41 / 0.65 units and max heading error 3.9° / 6.7° / — at 512 / 256 /
128; at 128 about 7 % of headings flip sign because the heading line is one
pixel wide.

---

## Architecture

```
       runconfig.py            targets.py
     (all parameters)     (heat / heading targets)
             |                     |
     +-------+-------+-------------+
     |               |             |
  data.py        model.py      extract.py ---- metrics.py
(datasets,      (U-Net,       (label map      (matching,
 downsampling)   heads)        -> Scene)       errors, bins)
     |               |             |               |
     +-------+-------+------+------+-------+-------+
             |              |              |
         train.py      predict.py     evaluate.py
      (one run ->     (image ->      (ckpt x sets ->
       runs/<name>)    JSON + PNG)    metrics, figs)
             |                             |
         sweep.py -------------------- aggregate.py
      (E5 grid, resumable)          (runs/ -> results/)

  _paths.py appends ../sim to sys.path; sim/ is never modified.
```

**Invariants.** One world ↔ pixel mapping (`targets.world_to_px` /
`px_to_world`, mirrors `render._to_px`). Extraction is a pure function of the
label map, so it is tested on rendered ground truth before any GPU time.
Metric names and run-directory files are the interface later stages read.

| To add… | Touch |
|---|---|
| an architecture | `model.ARCHS` |
| a metric | `metrics.scene_metrics`, then its column in `aggregate.py` |
| an extraction threshold | `runconfig.ExtractConfig` (it becomes a CLI flag) |
| a dataset | `scripts/make_datasets.DATASETS` |
| a sweep grid | `sweep.GRIDS` |

---

## Experiments (E5)

`sweep.py --grid e5` is a star design: the resolution axis at one
augmentation level, the augmentation axis at one resolution.

| Arm | Resolution | Training augmentation | Architectures | Seeds | Runs |
|---|---|---|---|---|---|
| resolution | 128, 256, 512 | aug2 | seg-only, +instance head | 0, 1 | 12 |
| augmentation | 256 | aug0, aug1, aug3 | seg-only, +instance head | 0, 1 | 12 |

512 trains 30 epochs, the rest 60. ≈ 34 h on a GTX 1070; `--dry-run` prints
the commands and a projection from measured epoch times. `e5_full` is the
48-run factorial, `e5_extras` the optional variants.

Every run is evaluated on `test`, `test_aug3`, `test_spacing/*` and
`test_dense`. `aggregate.py` writes `results/e5/`: `summary.csv`,
`table_res_aug.md`, `table_arch.md`, the error-vs-spacing curve, the
resolution × augmentation heat-map, per-layout usability and training curves.

---

## Tuning

| I want to… | Use |
|---|---|
| train at another resolution | `--size 128` (datasets stay at 512) |
| the learned instance head | `--instance-head` |
| fit 512 into memory | `--batch 2`, then `--amp` (memory only; no speed-up on Pascal) |
| faster epochs at 128 / 256 | `--cache ram` (decodes once; workers drop to 0) |
| train on some layouts | `--layouts corridor doorway` |
| continue an interrupted run | `train.py ... --resume` (same `--out`) |
| tune extraction thresholds | `evaluate.py --ckpt ... --data data/val --tune-extract` |
| a subset of the sweep | `sweep.py --only "s128_*"` |
| a smoke dataset | `scripts/make_datasets.py --n-scale 0.01` |
| run on CPU only | `--device cpu` |

---

## Debugging

**Extractor without a network.** Render a scene, extract from its own label
map; anything off here is an extraction bug, not a training problem.

```python
import _paths, json
from config import Config
from layouts import random_scene
from render import render
from extract import extract
import numpy as np

cfg = Config()
scene = random_scene(np.random.default_rng(0), cfg, layout="crossing", n_agents=6, n_obstacles=3)
_, lab = render(scene, cfg)
parsed = extract(lab, scene.as_dict(cfg)["world"])
print(len(parsed.scene.agents), parsed.diagnostics)
```

**One image, end to end.** `python predict.py --ckpt <ckpt> --image <png> --out parsed/`
writes the JSON and an overlay to look at.

**A run that stopped.** Re-run the same `sweep.py` command: runs with
`metrics.json` are skipped, runs with `ckpt_last.pt` are resumed.
`runs/e5/sweep_status.csv` says what happened to each.

**Low agent F1 but high pixel accuracy.** The ring vote threshold is off for
this model's mask width; `--tune-extract` grids `ring_thresh` (and
`heat_thresh`) on `val`.

**Two agents parsed as one.** Expected below ≈ 5 units of spacing at 128;
`agent_rows.csv` has `d_nn` per agent so the failure shows up as a curve,
not a mystery.

**Windows.** DataLoader workers use `spawn`; every entry point is guarded by
`if __name__ == "__main__":` and the dataset is picklable. Keep `data/` and
`runs/` on the SSD and off synced folders.

---

## Limitations

- Trained on the generator's visual style only; hand-drawn or photographed
  inputs need new training data, not a new extractor.
- Heading at 128 px is unreliable by construction (one-pixel line).
- Parsed walls are polygon surfaces, not centre lines: `Simulator` runs on
  them, `reachability` and `render` do not.
- Greedy agent matching in the metrics; exact only because true agents are
  ≥ 4 units apart.
