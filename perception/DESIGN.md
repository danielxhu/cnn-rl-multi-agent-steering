# Perception: model architecture and code structure

Design document for the perception stage (pipeline stages 2–3 in the README:
image → 6-class label map → structured `Scene`) and for the E5 perception
robustness sweeps (Detailed Proposal §5 E5, milestone weeks 2–3). Written
before the code; the code should follow it, and this file should be updated
when the code diverges.

Scope of this iteration:

| In | Out (later, needs the policy) |
|---|---|
| U-Net pixel classifier, optional instance head | pipeline-vs-oracle gap (E5 part 4) |
| Training / evaluation CLI, checkpoints, local logs | policy training, RL environment |
| Label map → `Scene` geometry extraction | path rendering over the input image |
| E5 sweep launcher (resolution × augmentation × architecture × seed) | E1–E4 |
| Result aggregation: tables, error-vs-spacing curve | |
| CPU unit tests for extraction and data loading | |

Decisions already taken (from discussion, 2026-09-14):

- Data is pre-generated to disk with `sim/generate.py`, always at 512×512;
  the 128 / 256 conditions are produced by downsampling at load time.
- Augmentation lives entirely in the renderer (`--jitter-*` flags): one
  training set per augmentation level. The training code does no augmentation.
- Instance separation: classical fixed-radius ring detection is the baseline
  (matches the proposal); a learned centre-heatmap + heading head is an
  optional architecture variant compared in E5.
- Parsed walls are surface polygons with thickness 0, not centre lines.
- Logging is plain local files (CSV / JSON / PNG). No tracking service.
- Code lives in `perception/`, importing `sim/` through a `sys.path` shim;
  `sim/` is not modified.
- Training machine (confirmed 2026-09-15): Windows 10 Pro, i7-8700K
  (6 cores / 12 threads), 16 GB RAM, **GeForce GTX 1070 8 GB** (Pascal,
  sm_61, no tensor cores), 238 GB SSD + 932 GB HDD. Consequences in §3.4,
  §6 and §9: fp32 training, star-shaped E5 grid instead of the full
  factorial, Windows-safe data loading, CUDA-wheel pinning.

---

## 1. Interfaces

### 1.1 Input

Files written by `sim/generate.py` (see `sim/README.md`):

| File | Used for |
|---|---|
| `scene_NNNNN.png` | network input, RGB 512×512 |
| `scene_NNNNN_labels.png` | segmentation target, palette PNG, pixel value = class 0–5 |
| `scene_NNNNN.json` | instance targets (agent poses), evaluation ground truth |
| `dataset.json` | `config` block: world size, agent radius, wall thickness, style |

Rendering facts the design depends on (`sim/config.py`, `sim/render.py`),
at 512 px, world = 100 units, scale = 5.12 px/unit:

| Quantity | World | 512 px | 256 px | 128 px |
|---|---|---|---|---|
| agent radius | 2.0 | 10.2 | 5.1 | 2.6 |
| agent ink line width | – | 4 | 2 | 1 |
| agent label line width (ink + 3) | – | 7 | 3.5 | 1.75 |
| heading line length (2.2 r) | 4.4 | 22.5 | 11.3 | 5.6 |
| wall thickness | 3.0 | 15.4 | 7.7 | 3.8 |
| obstacle diameter | 8–20 | 41–102 | 20–51 | 10–26 |
| min agent centre spacing, normal | 5.5 | 28 | 14 | 7 |
| min agent centre spacing, `hard` | 4.15 | 21 | 10.6 | 5.3 |

Two consequences drive the design:

1. **Outline merging.** Measured on rendered label maps, the gap between
   two agent outlines at the normal minimum spacing is 7 px at 512, 4 px at
   256 and 3 px at 128; at `hard` spacing it is 2 px at 512 and 1 px at 128.
   One mispredicted pixel merges two agents into a single component, so the
   extractor must not rely on connected components for agents. This is the
   curve E5 has to measure.
2. **Obstacle / wall ambiguity.** Both are solid black. A pixel deep inside a
   102 px obstacle sees no edge within a 4-level U-Net's receptive field
   (≈ 100 px), so the network can mislabel obstacle interiors as wall at 512.
   The extractor must fit obstacle circles from boundaries, not from filled
   area, and a deeper variant is kept as an architecture option.

### 1.2 Output

`extract.extract(...)` returns a `Parsed` object wrapping a `sim.scene.Scene`
that `sim.physics.Simulator` can step directly, plus a JSON form in the
generator's schema so parsed and ground-truth scenes are interchangeable in
later evaluation code:

```json
{
  "id": "scene_00042", "layout": null, "bottleneck": null, "parsed": true,
  "world": {"size": 100.0, "img_size": 256, "agent_radius": 2.0,
            "wall_thickness": 0.0, "fov_deg": 220.0, "fov_range": 22.0},
  "walls": [[[x, y], [x, y]], "..."],
  "obstacles": [{"c": [x, y], "r": r}],
  "start_region": [x0, y0, x1, y1], "goal_region": [x0, y0, x1, y1],
  "groups": [[start, goal], "..."],
  "agents": [{"pos": [x, y], "heading": theta, "goal_region": [...]?}],
  "diagnostics": {"agent_scores": [...], "n_dropped_components": 0, "..."}
}
```

Differences from a generated scene, all deliberate:

- `walls` are **surface** polygon edges of the predicted wall mask, and
  `world.wall_thickness` is `0.0`. Instantiate the simulator with
  `extract.sim_config(base_cfg)` (which sets `wall_thickness=0.0`), never
  with the generator's config. Do not pass a parsed scene to
  `render.render`; use `viz.draw_parsed` instead.
- `layout` and `bottleneck` are unknown and set to `null`.
- `agents[i].heading` is estimated; `fov_deg` / `fov_range` are copied from
  the generator config since they are not drawn.

---

## 2. Data plan

All sets are generated on the training machine (`sim/data/` is
git-ignored; 15 KB/scene, ≈ 20 scenes/s per core, ≈ 20 min for everything
below). `perception/scripts/make_datasets.py` holds the exact commands
(Python rather than a shell script so it runs unchanged on Windows; it
launches up to `--jobs 4` `generate.py` processes in parallel, one per
dataset, which brings the 6-core machine to ≈ 6 min total). Put `data/` and
`runs/` on the SSD, not the HDD. The table is the specification.

| Directory | Scenes | Generator arguments | Purpose |
|---|---|---|---|
| `data/train_aug0` | 5000 | `--layout mixed --agents 1-8 --obstacles 0-6 --seed 100` | training, no augmentation |
| `data/train_aug1` | 5000 | same + `--jitter-color 10 --jitter-line-width 0.5 --jitter-background 8 --noise-std 2 --blur-sigma 0.3`, seed 101 | low |
| `data/train_aug2` | 5000 | same + `--jitter-color 25 --jitter-line-width 1.0 --jitter-background 20 --noise-std 6 --blur-sigma 0.8`, seed 102 | medium (README's example) |
| `data/train_aug3` | 5000 | same + `--jitter-color 45 --jitter-line-width 1.5 --jitter-background 35 --noise-std 12 --blur-sigma 1.5`, seed 103 | high |
| `data/val` | 500 | clean, `--agents 1-8 --obstacles 0-6 --seed 200` | model selection |
| `data/test` | 1000 | clean, seed 300 | main test |
| `data/test_aug3` | 1000 | high augmentation, seed 301 | appearance robustness |
| `data/test_spacing/gap{0.0,0.5,1.0,2.0,4.0}` | 5 × 300 | `--hard-ratio 0.5 --agents 4-8`, `agent_gap` set through `--config`, seeds 400–404 | error-vs-nearest-neighbour curve |
| `data/test_dense` | 300 | `--agents 12-16 --obstacles 2-4`, seed 500 | E3-style density, perception side |

Notes:

- Every set is `--layout mixed`, so all seven families appear in equal
  proportion; metrics are reported per layout as well as overall.
- `test_spacing` exists so that nearest-neighbour distance covers the whole
  range 4.0–15 units with enough samples per bin. The curve is binned on the
  **measured** nearest-neighbour distance from the ground-truth JSON, so how
  the spacing was produced does not matter.
- All models, regardless of training augmentation, are evaluated on every
  test set, giving a (train augmentation × test perturbation) matrix.
- `agent_gap` is not a CLI flag of `generate.py`; the script writes a small
  JSON per gap value and passes it with `--config`.

### 2.1 Loading and resolution

`data.SceneDataset(root, size, instance_targets)`:

- Image: PNG → uint8 → resized to `size` with `cv2.INTER_AREA` → float32
  in [0, 1], layout `(3, S, S)`. No normalisation constants to keep in sync
  between train and inference.
- Labels: palette PNG read as raw indices (`Image.open(...)`, mode `P`, no
  conversion) → resized with `cv2.INTER_NEAREST` → int64 `(S, S)`.
- Instance targets (only when the instance head is on) are rendered from the
  JSON **at the working resolution**, not downsampled (`targets.py`):
  - `heat` `(1, S, S)`: max over agents of a Gaussian with σ = agent radius in
    px at that resolution, peak value 1 at the exact (sub-pixel) centre.
  - `dir` `(2, S, S)`: `(cos θ, sin θ)` constant over each agent's disk.
  - `dir_mask` `(1, S, S)`: union of agent disks; the heading loss is only
    computed there.
- The dataset keeps the parsed `Scene` objects and `world` dicts in memory
  (they are small) so evaluation can reach ground truth by index.
- Default loading is PNG decode in `num_workers=4` DataLoader workers with
  `persistent_workers=True`. On Windows workers are started with `spawn`,
  so the dataset object must be picklable (no lambdas, no open file
  handles) and every CLI entry point sits behind `if __name__ ==
  "__main__":`. Decoding a 512 PNG costs ≈ 3–5 ms; four workers deliver
  ≈ 1000 img/s, far above what the GPU consumes.
- `--cache ram` decodes the whole set once into uint8 arrays (5000 × 512² ×
  3 = 3.9 GB at 512, 1 GB at 256, 0.25 GB at 128). Because `spawn` copies
  the dataset into every worker, `--cache ram` forces `num_workers=0`; with
  16 GB RAM it is a sensible option at 128/256 and off by default.
- Class frequencies for loss weighting are computed once per training set
  and cached as `<root>/class_freq.json`.

---

## 3. Model

### 3.1 Backbone

Standard U-Net, encoder/decoder blocks of two `3×3 conv → BN → ReLU`,
`MaxPool2d(2)` down, `ConvTranspose2d(2, stride 2)` up, skip concatenation,
`1×1` output conv. Named variants:

| Name | base ch | levels | params | notes |
|---|---|---|---|---|
| `unet24x4` | 24 | 4 | 1.086 M | **the proposal's model** ("4-level, 1.09 M"); default |
| `unet24x5` | 24 | 5 | 4.37 M | roughly doubles the receptive field; candidate for 512 |
| `unet16x4` | 16 | 4 | 0.48 M | cheap variant for quick checks |

Widths double per level (24 → 48 → 96 → 192). Input `(B, 3, S, S)` for
S ∈ {128, 256, 512}; all three divide by 2⁴ (and 2⁵), so no padding logic.

### 3.2 Heads

```
                    ┌── seg  : 1×1 conv → (B, 6, S, S)   logits, always present
final features ─────┤
(B, 24, S, S)       └── inst : 1×1 conv → (B, 3, S, S)   [heat, cos, sin], optional
                                heat → sigmoid; (cos, sin) → L2-normalised
```

`model.UNet(base_ch, levels, n_classes=6, instance_head=False)` returns a
dict `{"seg": ..., "heat": ..., "dir": ...}` (`heat`/`dir` absent when the
head is off), so the training loop and extractor are indifferent to the
variant. Both heads share every backbone parameter; the instance head adds
75 parameters.

### 3.3 Losses

- Segmentation: cross-entropy with per-class weights
  `w_c ∝ 1 / sqrt(freq_c)`, normalised to mean 1 and clipped to [0.5, 20]
  (the pilot's hand-set weights were [0.5, 1, 1, 2, 20, 40]; agent pixels are
  ≈ 1 % of a 512 image with four agents, less after downsampling). Optional soft-Dice term
  with weight `--dice 0.0` (off by default; a knob for E5 if agent IoU
  stalls).
- Heat: CenterNet penalty-reduced focal loss (α = 2, β = 4), normalised by
  the number of agents.
- Direction: L1 between predicted unit vector and `(cos θ, sin θ)` inside
  `dir_mask`, averaged over masked pixels.
- Total: `L = L_seg + λ_heat L_heat + λ_dir L_dir`, λ = 1, 1.

### 3.4 Training loop

| Item | Default |
|---|---|
| optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| schedule | 1 epoch linear warm-up, cosine to 1e-5 |
| epochs | 60 (proposal) |
| batch | auto: 16 @128, 8 @256, 4 @512; `--batch` overrides |
| precision | **fp32 by default.** The GTX 1070 (Pascal) has no tensor cores and runs fp16 at 1/64 of its fp32 rate, so `torch.autocast` cannot make it faster; `--amp` stays available purely as a memory fallback if 512 does not fit. No bf16 (unsupported on sm_61), no `torch.compile` (Triton has no Windows build) |
| seed | `--seed`, seeds Python / NumPy / torch; `cudnn.benchmark=True` unless `--deterministic` |
| checkpoints | `ckpt_last.pt` every epoch (model, optimiser, scaler, scheduler, epoch, RNG), `ckpt_best.pt` on the selection metric; `--resume` continues from `ckpt_last.pt` |
| validation | every epoch: pixel accuracy, per-class IoU, mIoU on `val`; every `--extract-every 5` epochs additionally run the full extractor on a fixed 100-scene val subset for agent F1 / centre error |
| selection metric | agent F1 on the val subset when available, else agent-class IoU |
| previews | every 10 epochs, `previews/epoch_NN.png`: input / GT labels / predicted labels / parsed scene overlay for 4 fixed val scenes |

Memory estimate for `unet24x4` in fp32 (saved activations ≈ 0.6 GB per
512 image, roughly double that with backward temporaries): batch 4 @512
≈ 5 GB, batch 8 @256 ≈ 2.5 GB, batch 16 @128 ≈ 1.3 GB. The 8 GB card also
drives the Windows desktop (≈ 0.5–1 GB), so 512 stays at batch 4; if the
dev run still OOMs, `--batch 2` first, `--amp` second.

Run directory (`runs/<name>/`):

```
config.json        full resolved RunConfig (data, model, train, extract), git hash, argv
log.csv            epoch, lr, train_loss, loss_seg, loss_heat, loss_dir, val_pixacc,
                   val_miou, val_iou_agent, val_agent_f1, val_agent_center_err, seconds
ckpt_last.pt  ckpt_best.pt
metrics.json       written by evaluate.py: one block per test set
per_scene.csv      one row per (test set, scene) with every scene-level metric
previews/          epoch_NN.png
```

---

## 4. Geometry extraction (`extract.py`)

Input: predicted label map `(S, S)` uint8, the world config (size, agent
radius, wall thickness — copied from the dataset's `dataset.json`), and
optionally `heat` and `dir`. All thresholds live in `ExtractConfig` with the
defaults below; they are tuned once on `val` at 256 and then frozen.

Pixel ↔ world: `x = px / scale`, `y = world − py / scale`, `scale = S / world`
(the one y-flip, mirroring `render._to_px`).

### 4.1 Agents

Baseline (**A**, classical), used whenever `heat` is absent:

1. `agent_mask = labels == C_AGENT`.
2. Fixed-radius ring vote: cross-correlate the mask with a ring template of
   radius `r_px` (agent radius × scale) and width `max(1, lw_px)`
   (`cv2.filter2D`, normalised so a perfect isolated ring scores 1). This is
   the Hough circle accumulator for a known radius, computed exactly and
   deterministically instead of through `cv2.HoughCircles`' internal
   thresholds.
3. Non-maximum suppression: peaks above `ring_thresh = 0.45` with minimum
   separation `nms_factor × r_px`, `nms_factor = 1.6` (two agents are never
   closer than `2 r`). Sub-pixel refinement by a 3×3 quadratic fit around
   each peak.
4. Heading per detection: take agent pixels with distance in
   `(r_px + lw_px, 1.3 × heading_len_px)` from the centre that are closer to
   this centre than to any other detection (Voronoi restriction so a
   neighbour's line is not counted). θ = `atan2` of their mean offset vector,
   with the image-y flip. If fewer than 2 such pixels exist (possible at
   128), fall back to the principal axis of the whole blob and flag
   `heading_uncertain`.

Learned (**B**), used when `heat` is present:

1. Peaks of `heat` above `heat_thresh = 0.4` with the same NMS radius.
2. θ = `atan2(mean sin, mean cos)` over `dir` within `0.8 r_px` of the peak.
3. Falls back to A when the head is present but produces no peak and the
   agent mask is non-empty (logged in `diagnostics`).

Every agent carries a `score` (ring response or heat peak) in
`diagnostics["agent_scores"]` for later precision/recall-vs-threshold plots.

### 4.2 Obstacles

1. `obst_mask = labels == C_OBST`; morphological opening with a 3×3 kernel;
   fill holes (interior pixels the network may have called wall).
2. Connected components; drop components with area <
   `π (0.5 × obstacle_r_min × scale)²` (speckle).
3. Outer contour of each component; drop contour points 8-adjacent to a
   wall pixel (obstacles may overlap walls in the generator, and the
   obstacle is drawn on top, so the true boundary is where obstacle meets
   background). Algebraic least-squares circle fit (Kåsa) on the remaining
   points; `cv2.minEnclosingCircle` as fallback when fewer than 6 points
   survive.

Fitting from the boundary rather than `sqrt(area/π)` is what makes the
interior mislabelling in §1.1 harmless.

### 4.3 Walls

1. `wall_mask = labels == C_WALL`; closing with a 3×3 kernel to seal one-pixel
   cracks.
2. `cv2.findContours(RETR_LIST, CHAIN_APPROX_NONE)` — outer and inner
   contours, so a room's inside surface and a corridor's two walls are all
   represented; drop contours with area < 25 px².
3. `cv2.approxPolyDP` with `poly_eps_px = 1.5`; every polygon edge becomes one
   `((x, y), (x, y))` wall segment in world units.
4. `wall_thickness = 0.0` for the parsed scene. `Simulator` treats each edge
   as a zero-width capsule, so the collision and sensing surface is exactly
   the predicted black/white boundary. This handles thin walls and filled
   dead space the same way; a medial-axis approach would put the surface in
   the middle of dead-space blocks.

A corridor scene yields ≈ 8–20 segments; `Simulator` is vectorised over
segments, so cost is not a concern.

### 4.4 Regions and groups

1. Connected components of `labels == C_START` and `labels == C_GOAL`,
   minimum area 100 px² (regions are ≥ 12 units wide in every layout); each component's
   bounding box → `Region` (regions are axis-aligned by construction; agents
   and walls drawn over them punch holes but do not move the box).
2. Pairing: with one start and one goal, trivial. With more (crossing), pair
   a start with the goal whose interval overlaps it on one axis (left/right
   share a y-range, bottom/top an x-range); unpaired regions are dropped and
   counted in `diagnostics`.
3. `groups` = the pairs; `start_region` / `goal_region` = the first pair.
4. Agent → group: the start region containing the agent centre, else the
   nearest start region. Agents in group index > 0 get `goal_region` set,
   matching the generator's convention (`Scene.goal_for`).

### 4.5 Failure

`extract` returns `None` (a **parse failure**) if there is no start–goal
pair or no agent. Anything else is returned with diagnostics; downstream
metrics decide how wrong it is.

---

## 5. Metrics (`metrics.py`)

All geometric errors are in world units; angles in degrees. Per scene:

| Group | Metric | Definition |
|---|---|---|
| pixels | `pixacc`, `iou_<class>`, `miou` | on the label map at the working resolution |
| agents | `agent_precision`, `agent_recall`, `agent_f1` | greedy nearest matching, tolerance `tol = 2.0` (one radius); greedy is exact here because true centres are ≥ 4 units apart |
| | `agent_count_ok` | `n_pred == n_gt` |
| | `agent_center_err` | mean over matched |
| | `heading_err_deg` | circular absolute error over matched |
| | per-agent rows | `d_nn` (GT nearest-neighbour centre distance), `matched`, `center_err`, `heading_err`, `score` — the raw material for the spacing curve |
| obstacles | `obst_precision`, `obst_recall`, `obst_count_ok`, `obst_center_err`, `obst_radius_err` | matching tolerance 2.0 |
| regions | `region_iou` (mean over GT regions), `region_count_ok`, `group_pairing_ok` | |
| walls | `wall_iou` | raster IoU of wall class |
| | `wall_surface_err` | mean distance from sampled points on parsed wall edges to the GT wall surface (distance transform of the 512 GT wall mask), and the reverse direction; reported as the symmetric mean |
| scene | `parse_fail` | extractor returned `None` |
| | `scene_usable` | not failed ∧ all agents matched ∧ obstacle count ok ∧ pairing ok — "the policy could be run in this parse" |

Aggregates over a test set: means and rates overall and per layout; plus the
**spacing curve**: `agent_recall` and `agent_center_err` per `d_nn` bin,
bins `[4, 5), [5, 6), [6, 8), [8, 12), [12, ∞)`.

---

## 6. E5 sweep

### 6.1 Cost on the GTX 1070

FLOP count of `unet24x4` (forward, per image): 2.6 G @128, 10.4 G @256,
41.6 G @512; a training step is ≈ 3× forward. At 60 epochs × 5000 images
and an effective 1.5–2.5 TFLOPS (the card peaks at 6.5 fp32, but
24-channel convolutions are memory-bound):

| size | per 60-epoch run | note |
|---|---|---|
| 128 | ≈ 15–25 min | |
| 256 | ≈ 1–1.7 h | the proposal's 20–40 min assumed a T4/A10 with tensor cores |
| 512 | ≈ 4–7 h | 4× the pixels of 256 |

The full 3 × 4 × 2 × 2 factorial (48 runs) would be ≈ 5 days of continuous
GPU time, almost all of it at 512. Two changes bring it to about a day and
a half without losing either E5 question:

1. **512 trains for 30 epochs.** Every 512 image carries 4× the pixels, so
   30 epochs see more label pixels than 60 epochs at 256; the schedule
   (warm-up, cosine) is expressed in epochs and scales with it.
2. **Star design instead of the full factorial.** The augmentation axis is
   swept at 256 only (the resolution the pipeline will most likely use);
   the resolution axis is swept at aug2 only. Both axes still have 2
   architectures × 2 seeds.

### 6.2 Grid

`sweep.py --grid e5`, run directory name
`runs/e5/s{size}_a{aug}_{arch}{_inst}_seed{k}`:

| Arm | resolution | training augmentation | architecture | seeds | runs | ≈ time |
|---|---|---|---|---|---|---|
| resolution | 128, 256, 512 | aug2 | seg-only, +instance head | 0, 1 | 12 | 1.4 h + 5.5 h + 11 h |
| augmentation | 256 | aug0, aug1, aug3 (aug2 shared with the arm above) | seg-only, +instance head | 0, 1 | 12 | 16 h |
| | | | | | **24 runs** | **≈ 34 h** |

Optional extras, in priority order, each only if the arms above are done:
`unet24x5` at 512, aug2, both heads, seed 0 (2 runs, ≈ 12 h); 512 × aug3
(2 runs); `--dice 0.5` at 128 (2 runs, 40 min). The full factorial remains
available as `--grid e5_full` should a faster GPU appear.

The estimates are replaced by measurement: the first run of each size
writes its epoch time to `sweep_status.csv`, and `sweep.py --dry-run`
prints the projected total from whatever measurements exist.

The launcher runs jobs **sequentially** on the one GPU, skips any run whose
`metrics.json` exists, resumes any run that has `ckpt_last.pt` but no
`metrics.json`, and appends to `runs/e5/sweep_status.csv` (name, status,
start, end, seconds). `--dry-run` prints the commands; `--only 's256_*'`
filters by glob. It is safe to Ctrl-C and re-launch.

Each run is evaluated by `evaluate.py` on `test`, `test_aug3`,
`test_spacing/*` and `test_dense` at its own resolution.

Aggregation (`aggregate.py --runs runs/e5 --out results/e5`) writes:

| File | Content |
|---|---|
| `summary.csv` | one row per run × test set with every aggregate metric |
| `table_res_aug.md` | agent F1, centre error, heading error, `scene_usable` rate: rows = resolution, columns = training augmentation, on `test` and on `test_aug3`, mean ± std over seeds |
| `table_arch.md` | seg-only vs instance head, per resolution |
| `fig_spacing_curve.png` | agent recall and centre error vs `d_nn` bin, one line per resolution (best augmentation), both architectures |
| `fig_res_aug.png` | heat-map of agent F1 over resolution × augmentation |
| `fig_per_layout.png` | `scene_usable` rate per layout at 256 |
| `fig_training_curves.png` | val agent F1 vs epoch, all runs, coloured by resolution |

`results/e5/` is small (CSV/MD/PNG) and is committed; `runs/` and `data/`
are git-ignored. Checkpoints (≈ 4.4 MB each for `unet24x4`, ≈ 0.2 GB for
the whole sweep) stay on the training machine.

In `table_res_aug.md` the star design leaves the off-axis cells empty;
the table is still rendered as a full grid so it reads the same way if the
extras are run later.

---

## 7. Code structure

```
perception/
├── DESIGN.md              this file
├── README.md              how to run (written with the code)
├── .gitignore             data/  runs/
├── _paths.py              appends ../sim to sys.path; exposes REPO, SIM_DIR, PERCEPTION_DIR;
│                          imported first by every module (`import _paths  # noqa`)
├── runconfig.py           dataclasses DataConfig / ModelConfig / TrainConfig /
│                          ExtractConfig / RunConfig; JSON round-trip; argparse merge
│                          (not `config.py`: that name belongs to sim/config.py, see §11.3)
├── data.py                SceneDataset, loaders, class_frequencies, resize helpers
├── targets.py             heat / dir targets from a Scene; world↔pixel helpers
├── model.py               UNet, ARCHS registry, build_model, count_params
├── losses.py              weighted CE (+dice), focal heat loss, masked dir loss, total
├── train.py               CLI: one run → runs/<name>/ (log.csv, ckpts, previews)
├── predict.py             load_checkpoint, Predictor (any image size → labels/heat/dir),
│                          CLI for a directory or a single image (writes JSON + overlay)
├── extract.py             label map (+heat/dir) → Parsed; sim_config; parsed_to_json
├── metrics.py             per-scene metrics, matching, aggregation, spacing bins
├── evaluate.py            CLI: checkpoint × test dirs → metrics.json, per_scene.csv, figures
├── sweep.py               E5 grid, sequential resumable runner, sweep_status.csv
├── aggregate.py           runs/ → results/e5/ tables and figures
├── viz.py                 colorize (re-exported from sim), side-by-side panels,
│                          draw_parsed (parsed scene over the input image)
├── scripts/
│   └── make_datasets.py   every generate.py call in §2, plus the agent_gap configs;
│                          runs them as parallel subprocesses; Windows-safe
└── tests/
    └── run_tests.py       plain-assert tests in the style of sim/tests, CPU only
```

Dependency direction: `train/evaluate/sweep` → `data, model, losses,
metrics, extract, viz` → `targets, runconfig` → `sim/*`. `extract.py` and
`metrics.py` import torch nowhere, so the pipeline code that later runs the
policy can use them without a GPU stack.

### 7.1 Key signatures

```python
# runconfig.py
@dataclass
class DataConfig:  root: str; train: str; val: str; size: int = 256
                   num_workers: int = 4; cache: bool = False; layouts: list | None = None
@dataclass
class ModelConfig: arch: str = "unet24x4"; instance_head: bool = False; n_classes: int = 6
@dataclass
class TrainConfig: epochs: int = 60; batch: int | None = None; lr: float = 1e-3
                   weight_decay: float = 1e-4; amp: bool | None = None; seed: int = 0
                   dice: float = 0.0; lambda_heat: float = 1.0; lambda_dir: float = 1.0
                   extract_every: int = 5; extract_subset: int = 100
                   preview_every: int = 10; deterministic: bool = False
@dataclass
class ExtractConfig: ring_thresh: float = 0.45; nms_factor: float = 1.6
                     heat_thresh: float = 0.4; poly_eps_px: float = 1.5
                     min_wall_area_px: int = 25; min_region_area_px: int = 100
@dataclass
class RunConfig:   data: DataConfig; model: ModelConfig; train: TrainConfig
                   extract: ExtractConfig; out_dir: str
                   def save(path) / load(path) / from_args(argv)

# data.py
class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, root, size, instance_targets=False, layouts=None, limit=None, cache=False)
    def __getitem__(self, i) -> dict   # image (3,S,S) f32, labels (S,S) i64,
                                       # [heat (1,S,S), dir (2,S,S), dir_mask (1,S,S)], index
    scenes: list[Scene]; worlds: list[dict]; ids: list[str]; layouts: list[str]
def class_frequencies(ds) -> np.ndarray          # cached in <root>/class_freq.json
def load_image(path, size) -> np.ndarray         # uint8 (S,S,3)
def load_labels(path, size) -> np.ndarray        # uint8 (S,S)

# targets.py
def instance_targets(scene, world, size) -> (heat, dir, mask)   # numpy, at `size`
def world_to_px(world_size, size, x, y) / px_to_world(world_size, size, px, py)

# model.py
ARCHS = {"unet24x4": dict(base_ch=24, levels=4), "unet24x5": ..., "unet16x4": ...}
class UNet(nn.Module):
    def __init__(self, base_ch=24, levels=4, n_classes=6, instance_head=False)
    def forward(self, x) -> dict        # {"seg": (B,6,S,S), "heat": (B,1,S,S)?, "dir": (B,2,S,S)?}
def build_model(mcfg) -> UNet;  def count_params(m) -> int

# losses.py
def total_loss(out, batch, class_weights, tcfg) -> (loss, {"seg":…, "heat":…, "dir":…})

# predict.py
def load_checkpoint(path, device) -> (UNet, RunConfig)
class Predictor:
    def __init__(self, ckpt_path, device=None)
    def __call__(self, image_uint8) -> dict    # resizes to the run's size; labels (S,S) u8,
                                               # probs (6,S,S), [heat (S,S), dir (2,S,S)]

# extract.py
@dataclass
class Parsed: scene: Scene; size: int; world: dict; diagnostics: dict
def extract(labels, world, heat=None, dir=None, ecfg=ExtractConfig()) -> Parsed | None
def detect_agents_ring(agent_mask, r_px, lw_px, ecfg) -> list[(cx, cy, score)]
def detect_agents_heat(heat, r_px, ecfg) -> list[(cx, cy, score)]
def estimate_headings(agent_mask, centers, r_px, lw_px, hlen_px) -> (thetas, uncertain_flags)
def extract_obstacles(obst_mask, wall_mask, scale, world, ecfg) -> list[Obstacle]
def extract_walls(wall_mask, scale, world, ecfg) -> list[Segment]
def extract_regions(start_mask, goal_mask, scale, world, ecfg) -> list[(Region, Region)]
def sim_config(base_cfg) -> Config               # base_cfg.merged(wall_thickness=0.0)
def parsed_to_json(parsed, sid=None) -> dict;   def parsed_from_json(d) -> Parsed

# metrics.py
def pixel_metrics(pred, gt, n_classes=6) -> dict
def match_points(gt_pts, pred_pts, tol) -> list[(gi, pi)]
def scene_metrics(gt_scene, gt_labels512, parsed, pred_labels, world, size) -> (dict, list[agent_rows])
def aggregate(scene_rows, agent_rows) -> dict     # overall, per layout, spacing bins
SPACING_BINS = [4.0, 5.0, 6.0, 8.0, 12.0, float("inf")]

# evaluate.py
def evaluate(ckpt, data_dirs, out_dir, figures=True) -> dict      # also the CLI entry
# sweep.py
def e5_grid() -> list[RunSpec];  main(--grid e5 --out runs/e5 [--dry-run] [--only GLOB])
# aggregate.py
main(--runs runs/e5 --out results/e5)
```

### 7.2 CLI examples

```bash
cd perception
python scripts/make_datasets.py --jobs 4                       # ~6 min on 6 cores
python train.py --data-root data --train train_aug2 --val val --size 256 \
                --arch unet24x4 --seed 0 --out runs/dev_s256           # one run
python train.py ... --instance-head                                    # variant B
python evaluate.py --ckpt runs/dev_s256/ckpt_best.pt \
                   --data data/test data/test_aug3 data/test_spacing data/test_dense
python predict.py --ckpt runs/dev_s256/ckpt_best.pt --image data/test/scene_00007.png \
                  --out parsed/                                        # JSON + overlay PNG
python sweep.py --grid e5 --out runs/e5 --dry-run                      # list the 48 commands
python sweep.py --grid e5 --out runs/e5                                # run them, resumable
python aggregate.py --runs runs/e5 --out results/e5
python tests/run_tests.py                                              # CPU, ~1 min
```

---

## 8. Tests (`tests/run_tests.py`)

Same style as `sim/tests/run_tests.py`: plain functions, asserts, one
summary line. Scenes come from `sim.layouts.random_scene` with a fixed seed;
labels come from `sim.render.render`, so ground-truth label maps stand in
for a perfect network and the extractor is tested in isolation from
training.

| Test | Checks |
|---|---|
| `test_targets_peak_at_agent_centres` | `heat` argmax is within 1 px of every agent centre at 128/256/512; `dir` equals `(cos θ, sin θ)` inside disks; mask area ≈ n × π r² |
| `test_extract_roundtrip_512` | 3 scenes per layout, ≤ 8 agents: every agent matched within 0.5 units, heading within 5°, obstacle count exact, radius error < 0.5, region IoU > 0.95, wall surface error < 0.5 |
| `test_extract_roundtrip_128` | same scenes, labels downsampled with nearest: no exception, agent recall ≥ 0.9 at normal spacing, obstacle count exact |
| `test_ring_detector_separates_touching_agents` | two agents at centre distance 4.15 (hard spacing) at 512 → two detections |
| `test_crossing_groups_and_goal_assignment` | crossing scene: two pairs recovered, every agent's parsed goal region overlaps its true one |
| `test_obstacle_fit_ignores_interior_and_wall_contact` | obstacle whose interior is relabelled wall, and one overlapping a wall: fitted centre/radius within 0.5 |
| `test_wall_surface_matches_gt` | corridor and room: symmetric surface error < 0.5 units; dead-space blocks do not create interior walls |
| `test_parsed_scene_runs_in_simulator` | `Simulator(parsed.scene, sim_config(cfg))` steps 20 greedy actions, observations finite, no collision at t = 0 |
| `test_dataset_shapes_and_downsampling` | `sim/examples` at 128/256/512: tensor shapes, dtypes, label values < 6, every class present at 512 survives at 128 |
| `test_model_shapes_and_param_count` | forward at each size for both heads, output shapes; `unet24x4` params in [1.08 M, 1.09 M] |
| `test_metrics_matching` | synthetic GT/pred sets: precision/recall/count/centre error computed by hand |
| `test_checkpoint_roundtrip` | tiny `unet16x4` trained for 2 steps on 4 example scenes on CPU, saved, reloaded through `Predictor`, identical outputs |

---

## 9. Workflow on the training machine

1. `git clone`, `python -m venv .venv`, activate, `pip install -r
   requirements.txt`. **Then install the CUDA build of PyTorch
   explicitly**: on Windows the PyPI `torch` wheel is CPU-only, and the
   newest CUDA 12.8+ wheels no longer include Pascal (sm_61). Use the
   PyTorch index for a CUDA 12.6 or 11.8 build, for example
   `pip install torch --index-url https://download.pytorch.org/whl/cu126`
   (pin the newest version that still ships cu126 wheels; the NVIDIA driver
   must be recent enough for that CUDA runtime — update it from GeForce
   Experience / nvidia.com if `torch.cuda.is_available()` is `False`).
   Verify with
   `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`
   → `GeForce GTX 1070 (6, 1)`, and run one convolution on the GPU.
2. `cd perception && python scripts/make_datasets.py --jobs 4` (into
   `data/` on the SSD).
3. `python tests/run_tests.py` — confirms the sim shim and extractor work
   there before any GPU time is spent.
4. One short dev run at each size (`--epochs 2`) to measure seconds per
   epoch and peak memory (`torch.cuda.max_memory_allocated`); the numbers
   go into `sweep_status.csv` and, if a batch default has to change, into
   `runconfig.py`; commit.
5. Windows housekeeping for overnight runs: set the power plan so the
   machine never sleeps, disable automatic restart for updates during
   active hours, and keep `runs/` off OneDrive-synced folders. The sweep is
   resumable, so an interruption costs at most one epoch.
6. Tune `ExtractConfig` thresholds once on `val` predictions of a full
   256 / aug2 run (`evaluate.py --tune-extract`, grid over `ring_thresh`
   / `heat_thresh`); commit the values.
7. `python sweep.py --grid e5 --out runs/e5` (≈ 1.5 days, resumable;
   `--only 's128_*'` first to see complete results early).
8. `python aggregate.py`, commit `results/e5/`, copy the two or three
   `ckpt_best.pt` files that later stages will use.

---

## 10. Open items and assumptions

- **Python, NVIDIA driver and CUDA version on the training machine** are
  still unknown; they only affect the install line in §9 step 1.
- The per-run times in §6 are FLOP-based estimates for the GTX 1070 and
  must be replaced by the dev-run measurement; if 256 comes in well under
  an hour, the full factorial (`--grid e5_full`) is back on the table.
- 30 epochs at 512 (vs. 60 elsewhere) is a budget decision; the 512 arm
  is the one to extend if its validation curve has not flattened.
- Agent matching tolerance 2.0 units and the spacing bins are my choice;
  both are single constants in `metrics.py`.
- `test_dense` (12–16 agents) is included because E3 will need perception at
  that density; it costs 300 scenes and one extra evaluation pass.
- `extract.py` is written for the generator's visual style only. Hand-drawn
  or photographed inputs (proposal §10 item 4) would change the network's
  training data, not the extractor, as long as the label semantics hold.

---

## 11. Instructions for the implementing agent

This section is the brief for whoever writes the code. Sections 1–10 are
the specification; this section says how to turn them into code without
re-deriving the decisions.

### 11.1 Read first, in this order

1. This file, all of it. It is the contract: interfaces (§1), datasets
   (§2), model and training (§3), extraction (§4), metrics (§5), sweep
   (§6), file layout and signatures (§7), tests (§8), machine (§9).
2. `sim/README.md`, then `sim/config.py`, `sim/scene.py`, `sim/render.py`,
   `sim/physics.py` (the `Simulator` constructor and `observe_one`) and
   `sim/generate.py`. Everything the extractor produces must be consumable
   by `Simulator`; everything the loader reads is written by `generate.py`.
3. `sim/tests/run_tests.py` — the test harness style to copy (`@test`
   decorator, plain asserts, `main()` prints PASS/FAIL and a count).
4. `pilot/perception.py` — prior art for the U-Net and for OpenCV-based
   extraction. Read it for the idioms; do not port it. It is single-agent,
   has an FOV class that no longer exists, and estimates heading from the
   FOV arc.
5. The top-level `README.md` and `Proposal_DanielHu_Detailed.md` §5 E5 and
   §6.1, only for context.

### 11.2 Ground rules

- **Do not modify anything under `sim/`.** If the design needs something
  from `sim/` that is not there, implement it on the perception side and
  note it in §11.7. `pilot/` is frozen too.
- **No new dependencies.** `requirements.txt` already has numpy, opencv,
  pillow, torch, matplotlib. No scipy (matching is greedy by design),
  no tqdm, no Lightning, no hydra, no wandb.
- **Plain PyTorch, plain Python.** Match the style of `sim/`: a module
  docstring that states what the module owns and its invariants,
  dataclasses for parameters, small functions, numpy for geometry, type
  hints where they help, comments that explain why rather than what. No
  class hierarchies for their own sake; no global mutable state.
- **The spec wins over convenience.** Signatures in §7.1, file names in
  §7, run-directory contents in §3.4, metric names in §5 and the grid in
  §6.2 are fixed so that later stages (policy training, pipeline
  evaluation) can be written against them. If something in the spec is
  wrong or impossible, change the code *and* this document together and
  list the change in §11.7. Do not silently diverge.
- **Cross-platform from the first line.** The code is written on macOS
  and run on Windows: `pathlib` everywhere, no shell scripts, no `/tmp`,
  no `os.fork`, every CLI behind `if __name__ == "__main__":`, DataLoader
  datasets picklable (module-level functions, no lambdas, no open handles
  stored on `self`), `persistent_workers=True` when `num_workers > 0`.
- **CPU-first.** Every test and every smoke run in §11.4 must pass on a
  laptop CPU. `device` is resolved once (`cuda` if available, else `cpu`)
  and passed down; nothing calls `.cuda()` directly.
- **Determinism where cheap.** Seed Python, NumPy and torch from
  `--seed`; the extractor and metrics are pure functions of their inputs.
- **No git operations.** The author commits and pushes himself. Do not
  create branches, stage, commit or push; leave the working tree with the
  finished files in place and list them in the final report.

### 11.3 The `sim/` import shim and the `config` name clash

`sim/` is not a package: its modules import each other by bare name
(`from config import Config`). The only way to use them unchanged is to
put `<repo>/sim` on `sys.path`. Two rules follow:

1. `perception/_paths.py` does `sys.path.append(str(SIM_DIR))` (append,
   not insert, so `perception/` — the script directory — stays first) and
   every module in `perception/` starts with `import _paths  # noqa: F401`
   before any `from config import ...`. `tests/run_tests.py` first inserts
   `perception/` itself into `sys.path`, as `sim/tests/run_tests.py` does
   for `sim/`.
2. **No module in `perception/` may share a name with a module in
   `sim/`** (`config`, `scene`, `render`, `physics`, `geometry`,
   `layouts`, `reachability`, `generate`). That is why the run
   configuration lives in `runconfig.py`. A `perception/config.py` would
   shadow `sim/config.py` for `sim`'s own modules and break them in a
   confusing way.

### 11.4 Implementation order and the check at each step

Build bottom-up; run the check before moving on.

| Step | Files | Check |
|---|---|---|
| 1 | `_paths.py`, `runconfig.py`, `.gitignore` | `python -c "import _paths; from config import Config; import runconfig"` from `perception/` prints nothing |
| 2 | `targets.py`, `data.py` | `SceneDataset("../sim/examples", size)` for 128/256/512 returns the shapes and dtypes of §2.1; `test_dataset_shapes_and_downsampling`, `test_targets_peak_at_agent_centres` |
| 3 | `model.py`, `losses.py` | `test_model_shapes_and_param_count` (1.08–1.09 M for `unet24x4`); one forward/backward on CPU at 128 with both heads |
| 4 | `extract.py` | all extraction tests of §8 on **ground-truth** label maps rendered from `random_scene` — this is the step where most bugs live; do not proceed until the 512 round-trip tolerances hold |
| 5 | `metrics.py` | `test_metrics_matching`; `scene_metrics` on a GT-vs-GT parse gives F1 = 1, errors ≈ 0 |
| 6 | `tests/run_tests.py` complete | every test in §8 present and passing; `python tests/run_tests.py` exits 0 |
| 7 | `predict.py`, `viz.py`, `train.py` | 2-epoch CPU smoke run: `python train.py --data-root ../sim --train examples --val examples --size 128 --arch unet16x4 --epochs 2 --batch 2 --out runs/smoke`; run dir contains every file of §3.4; `test_checkpoint_roundtrip` |
| 8 | `evaluate.py` | on `runs/smoke` against `../sim/examples`: `metrics.json` has every metric name of §5, `per_scene.csv` has one row per scene, figures render |
| 9 | `scripts/make_datasets.py` | `--dry-run` prints the exact `generate.py` commands of §2; a real run with `--n-scale 0.01` (1 % of every count) completes in under a minute |
| 10 | `sweep.py`, `aggregate.py` | `sweep.py --grid e5 --dry-run` lists 24 runs with the names of §6.2 and `--grid e5_full` lists 48; `aggregate.py` on two smoke runs produces every file of §6 |
| 11 | `README.md` (perception), §11.7 of this file | README: install, make datasets, train one run, evaluate, sweep, aggregate, tests — the commands of §7.2 with one sentence each |

Steps 2–6 need no torch beyond the model test; keep the extractor and
metrics importable without a GPU stack (`extract.py` and `metrics.py`
import numpy and cv2 only).

### 11.5 Local development environment

The development machine is a Mac without a GPU and without `cv2` in the
system Python. Use either a venv (`python3 -m venv .venv && .venv/bin/pip
install -r requirements.txt`) or `uv`, which is installed:

```bash
cd perception
uv run --python 3.12 --with numpy --with opencv-python-headless --with pillow --with torch --with matplotlib python tests/run_tests.py
```

`sim/examples/` holds seven scenes (one per layout) with images, labels
and JSON; it is enough for every test and smoke run. For anything larger
generate into `perception/data/dev` with `sim/generate.py` (git-ignored).
Never commit data, checkpoints or run directories.

### 11.6 Definition of done

- Every file in the §7 tree exists and does what its line says.
- `python tests/run_tests.py` passes on CPU; the §11.4 smoke runs for
  steps 7, 8 and 10 succeed and leave the specified files behind.
- `python sweep.py --grid e5 --dry-run` prints 24 commands that would run
  unchanged on the Windows machine (forward slashes via `pathlib`, no
  shell features).
- `perception/README.md` exists; §11.7 below lists every deviation from
  §1–§10, or says there are none.
- Nothing has been committed; the final report lists every file created
  or changed so the author can review and commit.
- The final report to the user states, in this order: what was built, what
  the tests and smoke runs printed (paste the summary lines), what was
  not done and why, and the divergence list.

### 11.7 Divergence log

Filled in by the implementing agent. One line per deviation from §1–§10:
what changed, why, and which section was updated.

- (none yet)
