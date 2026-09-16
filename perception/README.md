# Perception: image → label map → `Scene`

U-Net pixel classifier, label-map → geometry extraction, training / evaluation
CLIs and the E5 robustness sweep. The specification is [DESIGN.md](DESIGN.md);
this file is the how-to. Everything runs from this directory and imports
`../sim` through `_paths.py`, so `sim/` is never modified.

## Install

Python 3.10+ with the repository's `requirements.txt` (numpy, opencv-python-headless,
pillow, torch, matplotlib). On the Windows training machine install the CUDA build
of PyTorch afterwards (see DESIGN.md §9, the GTX 1070 needs a cu126 / cu118 wheel):

```bash
cd perception
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate on macOS/Linux
pip install -r ../requirements.txt
```

No system Python with cv2? `uv run --python 3.12 --with numpy --with opencv-python-headless --with pillow --with torch --with matplotlib python <script>`.

## Tests (CPU, ≈ 1 min)

Extraction is tested on ground-truth label maps rendered from `sim`, so it is
checked before any GPU time is spent:

```bash
python tests/run_tests.py
```

## Make the datasets (≈ 6 min on 6 cores)

Every set of DESIGN.md §2 (four training augmentation levels, val, test sets,
the spacing series and the dense set) at 512 px; `--dry-run` prints the exact
`generate.py` commands, `--n-scale 0.01` makes a 1 % smoke copy:

```bash
python scripts/make_datasets.py --jobs 4
```

## Train one run

Writes `runs/dev_s256/{config.json, log.csv, ckpt_last.pt, ckpt_best.pt, previews/}`;
`--instance-head` adds the heat + heading head (variant B), `--resume` continues from `ckpt_last.pt`:

```bash
python train.py --data-root data --train train_aug2 --val val --size 256 --arch unet24x4 --seed 0 --out runs/dev_s256
```

CPU smoke run on the seven example scenes:

```bash
python train.py --data-root ../sim --train examples --val examples --size 128 --arch unet16x4 --epochs 2 --batch 2 --out runs/smoke
```

## Evaluate

Every set at the checkpoint's own resolution; writes `metrics.json`, `per_scene.csv`,
`agent_rows.csv` and figures next to the checkpoint (`--tune-extract` grids the
ring / heat thresholds on the first set instead):

```bash
python evaluate.py --ckpt runs/dev_s256/ckpt_best.pt --data data/test data/test_aug3 data/test_spacing data/test_dense
```

## Parse images

JSON in the generator's schema (`parsed: true`, surface walls with thickness 0) plus an overlay PNG per image:

```bash
python predict.py --ckpt runs/dev_s256/ckpt_best.pt --image data/test/scene_00007.png --out parsed/
```

## E5 sweep

The 24-run star design of DESIGN.md §6.2 (`--grid e5_full` is the 48-run factorial),
sequential and resumable; `--dry-run` lists the commands and the projected time,
`--only 's128_*'` filters:

```bash
python sweep.py --grid e5 --out runs/e5 --dry-run
```

```bash
python sweep.py --grid e5 --out runs/e5
```

## Aggregate

Tables (`table_res_aug.md`, `table_arch.md`), `summary.csv` and the four figures of §6.2 into `results/e5/`:

```bash
python aggregate.py --runs runs/e5 --out results/e5
```

## Using a parsed scene downstream

```python
import _paths
from predict import Predictor
from extract import sim_config
from physics import Simulator
from config import Config

pred, parsed = Predictor("runs/dev_s256/ckpt_best.pt").parse(image_uint8)
sim = Simulator(parsed.scene, sim_config(Config()))     # wall_thickness = 0: surface edges
```

`extract.py` and `metrics.py` import only numpy and cv2, so the parsing side
runs without torch.
