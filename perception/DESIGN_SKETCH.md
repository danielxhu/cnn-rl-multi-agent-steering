# Hand-drawn input: design

Extends the perception stage (`DESIGN.md`) from rendered images to scenes
drawn by hand with **one black marker** and photographed. Written on
28 September 2026, after E5 and before implementation. Parts already built
are marked **(built)**; everything else is specification.

Decisions taken (28 Sep):

- Input: a hand-drawn scene on paper, photographed with a phone. One black
  marker (~1 mm felt tip, not a ballpoint: a 0.5 mm line is under one pixel
  at 256). Classes are told apart **by shape only**.
- Training data is **generated**, not drawn: a procedural "sketch" renderer
  perturbs every primitive once and draws the same perturbed geometry into
  the image and the label map, so labels stay exact and unannotated.
- A small **real test set** is drawn by tracing printed scenes on a light
  source; the scene JSON is the ground truth, so no annotation there either.
- This work comes before the RL environment, which moves back by 1–2 weeks.
- Configuration carried over from E5: 256 px, instance head, heavy
  photometric augmentation (aug3) in the mix.

---

## 1. Drawing convention

| Class | Drawn as | Distinguished by |
|---|---|---|
| wall | a pen line, single or gone over twice | long line not attached to a small circle |
| dead space | hatched, **or** left blank | both allowed; labelled wall when hatched, background when blank |
| obstacle | a closed outline, then filled in or scribbled in | solid, large |
| agent | a small **hollow** circle and a line from its centre (heading) | hollow, small, has a tail |
| start / goal | a rectangle with the letter **S** / **G** inside | the letter |
| world boundary | printed on the sheet (not drawn) | — |

Why dead space may stay blank: agents cannot reach it, so for simulation the
drawn wall line is the only surface that matters (walls are extracted as
surfaces with thickness 0, `DESIGN.md` §4.3). Requiring people to hatch it
would be extra work for no simulation benefit.

---

## 2. The drawing sheet and rectification **(built)**

**Page** (`sheet.py`): A4 portrait. The world square is 160 × 160 mm
(1.6 mm per world unit; an agent is a 6.4 mm circle), bottom-left corner at
(25, 100) mm, with the boundary wall printed as a 2.4 mm band inside it (half
of the simulator's 3.0-unit wall thickness, so the printed frame is exactly
what the generator draws). Four filled squares sit diagonally 12 mm outside
the square's corners; the top-left one is 12 mm, the others 8 mm, which fixes
orientation. Eight 5 mm boxes below the square encode the scene id (7 bits,
MSB first, plus even parity). All page geometry lives in `sheet.py`; the
generator and the rectifier read the same numbers.

**Pages** (`scripts/make_sheets.py`): for each scene a *reference* page (the
scene drawn in the convention above, plus frame, markers and id) and a
*drawing sheet* (frame, markers, id boxes and instructions only), as two PDFs,
plus the scene JSON with a `sheet_id`. Default: 42 scenes, 6 per layout,
2–6 agents, 1–4 obstacles, normal spawn spacing, seed 700. Print at 100 %.

**Rectification** (`rectify.py`): photo → 512 × 512 world square + scene id.

1. Grey, flatten illumination (divide by a dilated, blurred copy).
2. Find the paper: largest region brighter than 0.75 × Otsu, convex hull,
   slightly eroded. Everything outside is set to paper before binarising.
   Without this, a dark desk next to the page edge binarises as a thick band
   of "ink" that swallows the 7 mm-from-edge markers. The 0.75 factor keeps
   paper in a shadow inside the mask.
3. Otsu binarisation; candidate markers are solid four-cornered blobs
   (`RETR_LIST` so nothing nested is missed; the contour's area close to its
   four-corner approximation's, which rejects filled circles at 2/π; fill
   ≥ 0.85, which rejects hollow boxes and the frame).
4. Every set of four candidates whose areas match the printed pattern (one
   ≈ 2.25× the others) and that forms a large convex quadrilateral is
   **verified**: its homography must put ink on ≥ 70 % of 60 points along
   the printed frame. The best-verified set wins; otherwise `RectifyError`.
5. Warp the world square out; read the id boxes through the same homography.

Measured on 100 simulated phone photos (rotation ±20°, corner jitter
±110 px, scale 0.8–1.05, desk colours 20–140, lighting gradient, hard shadow,
blur σ 0.6–1.6, noise σ 7, JPEG 65): **93 accepted, 0 wrong**; all 93 ids
correct; rectification error over a 9 × 9 grid median 0.9 px, worst 3.0 px
of 512 (0.6 world units). The 7 rejections all had the top-left marker cut
off by the photo edge — the error message says so, and the fix is to retake
the photo. Pinned by `test_rectify_recovers_sheet_and_id`.

**Tracing workflow.** Put a drawing sheet on its reference page, align the
frames against a window or a light, trace everything except the printed
frame, markers and id boxes, photograph the drawing sheet flat with all four
corner squares in view. Tracing keeps the drawn positions within ~1 mm
(≈ 0.6 units) of the JSON, so the JSON is the ground truth; tolerance 2.0
units in the metrics is unchanged.

---

## 3. Synthetic sketch renderer (`sim/sketch.py`)

`generate.py --style sketch` writes the same three files per scene as the
default renderer. This is the first change to `sim/`: one new module and one
CLI flag; `render.py` is untouched and the 13 `sim` tests must still pass.

### 3.1 Algorithm

A working prototype of steps 1–3 is `perception/prototypes/sketch_prototype.py`
(its docstring lists what it gets wrong). The ranges below are the prototype's.


For each scene sampled by `layouts.random_scene` (unchanged):

1. **Perturb every primitive once**, with the scene's seed:
   - wall segment → polyline: points every ~3 px, offset perpendicular by
     smooth low-frequency noise (amplitude 1.5–3 px at 512) plus small jitter;
     ends overshoot by up to 5 %;
   - obstacle → closed curve: radius modulated by smooth noise (± 6 %),
     independent x/y stretch (± 12 %);
   - agent → open curve: radius × U(0.8, 1.25), ± 8 % wobble, stretch ± 12 %,
     ending with a gap or an overshoot of −0.35…+0.5 rad; tail length ×
     U(0.8, 1.2), heading + N(0, 0.06 rad);
   - region → quadrilateral: corners jittered by N(0, 3 px); letter S / G
     placed where it is **farthest from every agent** in the region (the
     prototype placed letters over agents);
   - boundary walls → the printed band, drawn crisp (it is printed, not drawn).
2. **Draw the perturbed geometry twice** — into the image with the pen style
   below, and into the label map with class indices, in the default draw
   order (dead space, regions, walls, obstacles, agents). Agent label strokes
   are 3 px wider than the ink, as in the default renderer.
3. **Photo effects**, applied after drawing: residual perspective (corners
   ± 8 px), lighting gradient (5–30 % fall-off), blur σ U(0.4, 1.0), noise σ 4,
   JPEG quality 50–85. The perspective is applied to the label map (nearest)
   and to every JSON coordinate.
4. **Write the drawn geometry to the JSON**, not the pre-perturbation one:
   agent `pos` = the drawn circle's centre, `heading` = the drawn tail's
   angle, obstacle = centre and mean radius of the drawn curve, region = the
   axis-aligned box of the drawn quadrilateral; all after the perspective of
   step 3. Wall centre lines stay as sampled (the ~0.5-unit wobble is below
   the metric tolerances). The instance head's targets are rendered from this
   JSON, so a mismatch here would teach it offset centres.

### 3.2 Pen style (randomised per image)

| Property | Range |
|---|---|
| ink colour | uniform in RGB [15, 55]³ (one near-black marker) |
| paper colour | (246, 244, 238) ± U(−12, 8) per channel |
| base stroke width | 3–5 px at 512; walls × U(1.0, 2.2) (single or doubled) |
| stroke width along a stroke | ± 35 % smooth variation |
| obstacle fill | solid (45 %) or scribbled parallel strokes 3.5–6 px apart at a random angle |
| dead space | hatched (60 %) with lines 9–15 px apart, or blank |
| letters | Hershey script font, height ≤ 0.55 × box width, rotation ± 15° |

### 3.3 Label semantics (differences from the default renderer)

- Region label = the interior of the drawn quadrilateral (letters included).
- Hatched dead space = wall; blank dead space = background.
- Agent label = drawn circle and tail strokes, +3 px.
- Obstacle label = the filled drawn curve, regardless of scribble gaps.

### 3.4 Cost

Prototype: 143 ms per render on one Mac core (≈ 3 × the default renderer,
because strokes are drawn as many short segments). 5,000 scenes on four
processes of the training machine: ≈ 30 min.

---

## 4. Datasets

| Directory | Scenes | Style | Seed | Purpose |
|---|---|---|---|---|
| `train_sketch` | 5,000 | sketch | 110 | training, mixed with `train_aug3` |
| `val_sketch` | 500 | sketch | 210 | model selection |
| `test_sketch` | 1,000 | sketch | 310 | synthetic hand-drawn test |
| `real_sketch/test` | 42 (fewer if photos are rejected) | photographed | 700 | the real test |

Scene parameters as in `DESIGN.md` §2 (mixed layouts, 1–8 agents, 0–6
obstacles). Added to `scripts/make_datasets.py`.

---

## 5. Training and evaluation

- `--train` accepts several directories (`--train train_sketch train_aug3`);
  `SceneDataset`s are concatenated, class frequencies computed over the union
  (cache per root and size, combined by pixel counts). `DataConfig.train`
  becomes `str | list`; old configs with a string still load.
- Model: `unet24x4` + instance head, 256 px, 60 epochs, seeds 0 and 1.
  10,000 training images ≈ 150 s/epoch on the GTX 1070 (2 × 75 s measured
  at 5,000) → ≈ 2.5 h per run.
- Extraction: unchanged. With the instance head, agent centres come from the
  heat map, which does not assume a fixed radius; the ring vote (fallback
  only) will be weaker on hand-drawn agents and that is acceptable.
- `scripts/ingest_photos.py`: a folder of photos → `rectify` → decode id →
  `real_sketch/test/real_NNNN.png` (the rectified image), the scene JSON, and
  a label PNG rendered from the JSON by the default renderer (only so the
  loader works — **pixel metrics and wall surface error on the real set are
  not meaningful** and are reported as such). Prints every rejected photo
  with its reason.

**Comparison** (one table): rows = the E5 model (256 + inst, aug3, never saw
a sketch) and the new model; columns = `test`, `test_aug3`, `test_sketch`,
`real_sketch/test`; metrics = agent F1, centre error, heading error, obstacle
count, region IoU, `scene_usable`. The first row shows how far the rendered
style transfers; the second what the sketch data buys; the `test` columns
check that nothing was lost on rendered images.

---

## 6. Tests to add

| Test | Checks |
|---|---|
| `sim`: `test_sketch_is_deterministic` | same seed → identical image, labels and JSON |
| `sim`: `test_sketch_labels_contain_every_class` | as for the default renderer (with hatched dead space forced on) |
| `perception`: `test_sketch_json_matches_drawing` | on 7 sketch scenes, every JSON agent centre is within 1 px of the centroid of that agent's label ring, every JSON heading within 5° of the drawn tail direction |
| `perception`: `test_sketch_instance_targets_align` | heat-map peaks fall on agent-labelled pixels |
| `perception`: `test_extract_on_sketch_labels` | extraction on sketch ground-truth labels with ground-truth heat and direction maps recovers every agent within 1.0 unit and 10°, obstacle count exact |
| `perception`: `test_rectify_recovers_sheet_and_id` | **(built)** |

---

## 7. Implementation order

| Step | Work | Check |
|---|---|---|
| 1 | `sim/sketch.py`, ported from `perception/prototypes/sketch_prototype.py` (fix the problems listed in its docstring; delete the prototype afterwards), `generate.py --style sketch` | `sim` tests 13/13; the two new `sim` tests; a 7-scene contact sheet looks right |
| 2 | JSON = drawn geometry (§3.1 step 4) | `test_sketch_json_matches_drawing`, `test_sketch_instance_targets_align` |
| 3 | multi-root `--train`; datasets in `make_datasets.py` | 1 % smoke generation; a 2-epoch smoke run on `train_sketch` + `train_aug3` |
| 4 | `ingest_photos.py` | on simulated photos of the 42 reference pages: every one ingested with the right id |
| 5 | `test_extract_on_sketch_labels` | passes |
| 6 | README section; divergence log below | — |

Ground rules as in `DESIGN.md` §11.2 (CPU-first, Windows-safe, no new
dependencies, no git operations).

## 8. Divergence log

- (none yet)
