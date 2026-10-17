# Hand-drawn input: design

Extends the perception stage (`DESIGN.md`) from rendered images to scenes
drawn by hand with **one black marker** and photographed. Written on
28 September 2026, after E5 and before implementation. Parts already built
are marked **(built)**; §3–§6 were implemented on 28 September 2026 and are
marked **(built)** too, with every deviation in §8.

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

## 3. Synthetic sketch renderer (`sim/sketch.py`) **(built)**

`generate.py --style sketch` writes the same three files per scene as the
default renderer. This is the first change to `sim/`: one new module and one
CLI flag; `render.py` is untouched and the 13 `sim` tests must still pass.

### 3.1 Algorithm

Ported from the prototype `perception/prototypes/sketch_prototype.py` (now
deleted) with its four listed problems fixed (letters on agents, white wedges
in scribbled obstacles, JSON not updated, the private `render._to_px`). The
ranges below are the prototype's and live in `sketch.SketchStyle`. How the
code refines steps 1–4 is in §8.


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
because strokes are drawn as many short segments). Built version: ≈ 100 ms
per render; `generate.py --style sketch` writes 5.5 scenes/s against 12/s for
the default style (one core of the Linux development container), so
`train_sketch` takes ≈ 15 min in its own process and `make_datasets.py`
starts it first.

---

## 4. Datasets **(built)**

| Directory | Scenes | Style | Seed | Purpose |
|---|---|---|---|---|
| `train_sketch` | 5,000 | sketch | 110 | training, mixed with `train_aug3` |
| `val_sketch` | 500 | sketch | 210 | model selection |
| `test_sketch` | 1,000 | sketch | 310 | synthetic hand-drawn test |
| `real_sketch/test` | 42 (fewer if photos are rejected) | photographed | 700 | the real test |

Scene parameters as in `DESIGN.md` §2 (mixed layouts, 1–8 agents, 0–6
obstacles). Added to `scripts/make_datasets.py`.

---

## 5. Training and evaluation (code **built**; runs pending on the training machine)

- `--train` accepts several directories (`--train train_sketch train_aug3`);
  `SceneDataset`s are concatenated, class frequencies computed over the union
  (cache per root and size, combined by pixel counts). `DataConfig.train`
  becomes `str | list`; old configs with a string still load.
- Model: `unet24x4` + instance head, 256 px, 60 epochs, seeds 0 and 1.
  10,000 training images ≈ 150 s/epoch on the GTX 1070 (2 × 75 s measured
  at 5,000) → ≈ 2.5 h per run.
- Extraction: agents, obstacles and regions unchanged — they already come out
  ideal (poses; Kåsa circles; axis-aligned boxes). With the instance head,
  agent centres come from the heat map, which does not assume a fixed radius;
  the ring vote (fallback only) will be weaker on hand-drawn agents and that
  is acceptable. **Walls (added 29 Sep):** `--wall-mode lines`
  (`walllines.py`) turns the wall mask into straight, axis-parallel centre
  lines with the generator's 3.0 thickness, so a wobbly drawing parses into the
  same form as a generated scene (surface walls gave 13–113 jagged segments per
  sketch against 4–12 generated). Sketch runs use it: pass `--wall-mode lines`
  to `train.py` (it is stored in the run and used by `evaluate.py`), or to
  `predict.py`, which then also writes `<name>_clean.png`, the parse re-drawn by
  the generator's renderer. `predict.py --photo` rectifies phone photos first.
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

## 6. Tests **(built)**

| Test | Checks |
|---|---|
| `sim`: `test_sketch_is_deterministic` | same seed → identical image, labels and JSON |
| `sim`: `test_sketch_labels_contain_every_class` | as for the default renderer (with hatched dead space forced on) |
| `perception`: `test_sketch_json_matches_drawing` | on 7 sketch scenes, every JSON agent centre is within 1 px of the centre of that agent's label ring (circle fit; raw pixel centroid within 2.5 px, §8), every JSON heading within 5° of the drawn tail direction (line fit to the tail's label pixels); obstacle centres on obstacle label; every region label pixel inside its JSON box ± 1.5 px |
| `perception`: `test_sketch_instance_targets_align` | heat-map peaks fall on agent-labelled pixels (sub-pixel peak within 1 px of one), at 512 and 256 |
| `perception`: `test_multi_root_training_set` | added: string and list `data.train` both load; concatenation; class frequencies combined by pixel counts |
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

Implemented 28 Sep 2026. One entry per deviation from §1–§7 or addition to
them; sections that state a changed number were updated too.

**Renderer (§3.1, §3.2)**

- *Centre of a drawn circle* (step 4) = centroid of the area enclosed by one
  turn of the drawn curve (a gap closed by its chord, an overshoot cut at 2π),
  after the perspective. Obstacles: the same centroid; radius = mean distance
  of the warped curve from it.
- *Tail* starts at that centre, not at the sampled position, and its wobble is
  anchored there (the pen is put down at the centre). Before this, the wobble
  moved the root up to ~2 px off and heat peaks missed the ink.
- *Heading* (step 4) = least-squares direction of the drawn tail polyline,
  oriented root → tip, after the perspective. "Angle of tip − root" was up to
  8° off the ink, because the wobble moves the end points too.
- *Walls* (step 4): centre lines stay as sampled but go through the perspective
  like every other coordinate (step 3 says "every JSON coordinate"; left
  unwarped they would be off by up to 8 px = 1.6 units). Only the wobble is not
  modelled. Boundary walls are warped too, so they are no longer exactly at
  0 / 100.
- *Perspective border* (step 3): the strip the warp uncovers is filled with the
  printed band (ink colour; label wall), not replicated. Replication smeared any
  stroke touching the edge along the whole border (found as a 1.1 px centre
  error on an agent near the edge).
- *Dead space*: its mask is grown to the wall centre line (1.5 units), not by
  `render.py`'s quarter thickness. The drawn line is much thinner than the
  3-unit wall, so the smaller growth left a blank strip between hatching and
  line. Hatching is clipped out of obstacles; its angle is random, 25–65° on
  either diagonal (the prototype had a fixed 45°).
- *Scribble fill*: parallel strokes laid out in a frame rotated with them over
  the obstacle's whole disk (the prototype's offsets stopped covering the disk
  once tilted: the white wedge). Obstacle label = the filled curve, as
  specified; the ink outline therefore sticks out ≈ 1.5 px beyond it.
- *Regions*: labels are painted in draw order (the prototype only painted
  background pixels). Letters go to the grid point in the drawn box farthest
  from every drawn circle and tail, with distances capped at two letter
  heights so a box with no agents nearby centres its letter; letter height
  × U(0.8, 1.0) under the 0.55 × box limit.
- *Image composition*: every mark is the same ink, so all strokes accumulate
  into one coverage mask laid over paper and band once (205 → 100 ms per
  render). Label maps are still painted in the default draw order.
- *Randomness*: the sketch uses `default_rng(scene.seed)`, not the generator's
  stream, so a scene's drawing depends only on its seed. The JSON carries
  `"style": "sketch"`. Re-rendering a sketch JSON does not reproduce the
  image: that JSON already holds the perturbed geometry.
- *Test hook*: `sketch(..., parts=True)` also returns per-agent ring and tail
  maps, drawn with the label strokes and warped like the labels.
- *`sim.scene.Region` bug, worked around*: `Region(x0, y0, x1, y1)` with
  y0 > y1 sets both y's to the smaller value (`__post_init__` overwrites y0
  before computing y1). `sketch.py` always passes sorted bounds. Not fixed
  here, because `sim/` is frozen apart from `sketch.py` and `--style`.

**Tests (§6)**

- `test_sketch_json_matches_drawing` measures the ring's centre with an
  algebraic circle fit to the ring's label pixels, not their centroid. The
  ±35 % pen-pressure width makes one side of a ring thicker, which pulls the
  raw pixel centroid up to 2.3 px off a correctly written centre; it is still
  checked, at 2.5 px. Heading is measured as the line fit of the tail's label
  pixels. Over 1,047 agents in 210 scenes (beyond the test's 7): circle-fit
  centre error median 0.28 px, p99 0.82, max 1.11 px; heading error median
  0.7°, p99 3.3°, max 6.3° (a short, thick tail whose round caps dominate the
  pixel fit); heat peak to nearest agent pixel max 0.68 px. The 1 px / 5°
  bounds therefore hold on the fixed 7-scene test, and outliers beyond them
  are rare and come from the pixel measurement itself.
- The two new `sim` tests are in `sim/tests/run_tests.py` (15 tests now).
  `test_sketch_labels_contain_every_class` also pins `sketch._to_px` to
  `render._to_px` (one y flip) and checks that blank dead space stays
  background.

**Data, training, evaluation (§4, §5)**

- `make_datasets.py` starts the longest datasets first (a sketch scene counts
  as two), so `train_sketch` is not left to run alone at the end.
- `SceneDataset` also loads `real_*` stems. A `dataset.json` with
  `"labels_exact": false` (written by `ingest_photos.py`) makes `evaluate.py`
  set pixacc, the IoUs, mIoU, wall_iou and wall_surface_err to NaN and add a
  `note` to that set's block: this is how "reported as not meaningful" is done.
- `evaluate.py` qualifies a set name that is already taken with its parent
  directory: `data/test` and `data/real_sketch/test` would both be `test` and
  overwrite each other in `metrics.json`; the second is now `real_sketch/test`.
- `aggregate.py` labels a mixed training set by its names (`sketch+aug3`)
  instead of taking `aug3` from it, so the new model does not merge into the
  E5 aug3 cells.
- `ingest_photos.py` applies the EXIF orientation tag, keeps the first photo
  of a sheet and rejects later ones as duplicates, and copies the photo name
  into the JSON. Checked on 42 simulated phone photos of the reference pages
  (rotation ±15°, corner jitter ±50 px, desk colours 20–140, lighting
  gradient, a hard shadow in half of them, blur σ 0.6–1.4, noise σ 5, JPEG
  65–90, file names shuffled): 42 / 42 ingested with the right id. A cropped
  photo and a duplicate were rejected with their reasons.
- `sim/README.md` does not mention `--style` (the hard constraint allowed only
  `sketch.py` and the flag); the flag's help, `generate.py`'s docstring and
  the perception README document it.

**Straight-line output (29 Sep 2026)**

- Request: the parse of a hand-drawn scene should have the generator's form —
  straight walls, round obstacles — whatever the pen did. Obstacles (Kåsa
  circles), regions (boxes) and agents (poses) already were; walls were not
  (surface polygons follow every wobble). Added `walllines.py` and
  `ExtractConfig.wall_mode` (`"surface"` default, E5 unchanged; `"lines"`).
  The network and its label semantics are unchanged: labels stay exact to the
  drawing, the regularisation is a pure function in extraction, testable on
  ground-truth labels. (Training the network on idealised labels instead was
  rejected: it would have to invent the 3-unit thickness and blank dead space,
  and labels would stop matching the image.)
- Algorithm: rim of the wall mask within one stroke width of free space
  (obstacle pixels are neither) → Zhang–Suen skeleton → branches split at
  junctions → Douglas–Peucker (1.5 units) → least-squares pieces → axis snap
  (≤ 10°; every layout wall is axis-parallel) → collinear merge (gaps ≤ 4 units,
  or covered by stroke / obstacle labels with no hole over 0.5 units) → corner
  and T snapping, trimming a hand-drawn overshoot up to 6 units → free ends
  within 4.5 units of the edge extended onto it → exact boundary walls. Stroke
  width is measured on free-standing strokes; with none (dead space filled
  everywhere) it is the world's wall thickness.
- Measured on ground-truth labels, 70 scenes per style (10 per layout):
  default-renderer labels, mean centre-line error 0.034 units at 512, 0.049 at
  256; sketch labels 0.57 / 0.57, of which most is the pen wobble itself (the
  drawn stroke wanders around the sampled line; the parse is the best straight
  line through the drawing) and the photo perspective on the boundary.
  Segment counts equal the generated ones in 69 / 70 scenes at 512.
- Limits, by construction: a wall drawn *under* an obstacle is not in the label
  map, so a wall piece hidden entirely by one is lost, and a doorway an
  obstacle fills is bridged (the simulation is the same: nothing passes
  there). An obstacle drawn over half a wall's width shifts that stretch by up
  to ~0.6 units. In a hatched sketch with no free-standing stroke the stroke
  width falls back to the 3-unit wall, which may place those walls up to ~1 unit
  toward the dead space.
- Metrics: `wall_line_err` (centre lines vs the JSON's, NaN for surface
  parses); `wall_surface_err` uses the capsule outline for centre-line parses.
  `sim_config(cfg, parsed)` takes the parse's own wall thickness (0 for
  surface walls, as before when called without a parse).
- Test `test_wall_lines_like_generated`: 21 default-render and 7 sketch scenes
  at 512 and 256 — boundary exact, every wall axis-parallel, segment count
  within one of the generated, interior centre-line error away from obstacles
  per scene < 0.5 / 1.0 units and median < 0.2 / 0.4 (measured medians 0.10 /
  0.15 and 0.27 / 0.28), every parse still solvable (no doorway closed), the
  simulator runs with thickness 3, and `render_clean` draws it.
- Also: `predict.py --photo` (rectify first), `--wall-mode`, reads `real_*`;
  `rectify.load_photo` (EXIF-upright) moved there from `ingest_photos.py`.

**Rectification fallback (5 Oct 2026)**

- §2 step 2 (page mask) failed on an image cropped close to the sheet (a
  tablet drawing on the drawing-sheet PDF, 950 × 920 px): the largest bright
  region was the inside of the frame, not the paper around it, so the corner
  markers were masked away (`0 candidates`). `rectify` now retries marker
  finding on the unmasked image when the masked search fails; the frame check
  of step 4 still guards against a wrong set. Phone photos of the whole page
  take the first path as before (`test_rectify_recovers_sheet_and_id`
  unchanged and passing).

**Sketch style v2 (6 Oct 2026)**

- The first real input, a tablet drawing on the drawing-sheet PDF, parsed
  badly with the v1-trained model (obstacles broken into wall pieces, the S box
  missed): 1–2 px strokes at 512 against v1's 3–5, sparse thin hatching inside
  obstacles, a pure white page with no photo effects, elongated obstacles,
  small letters. Thickening its strokes by hand already turned two of three
  obstacles back into circles, so the gap is the training style, not the
  method.
- `sketch.SKETCH_V2` (`generate.py --style sketch2`) widens §3.2: a *digital*
  medium in 35 % of images (white page minus 0–4, ink 0–30, pen 1.2–3.5 px, no
  perspective / light / noise / JPEG, blur 0–0.5); photographed pens 2–5 px;
  non-solid obstacles split into dense (as v1), sparse (spacing 7–14 px, 0.6 ×
  pen) and one zigzag scribble (spacing 5–10 px), with fills crossing the
  outline by up to 3 px; obstacle stretch ± 22 %; agent radius × 0.7–1.3;
  region sides drawn up to 10 % past the corners; letters 0.55–1.0 of the
  largest that fits, in the pen's width. Labels keep §3.3's semantics.
- `SketchStyle()` is unchanged, and v2's extra random draws happen only when
  its options are on, so `--style sketch` is pixel-identical to before (checked
  by hashing 14 generated scenes before and after the change).
- Checked on 140 v2 scenes: JSON ↔ drawing, heat-target and extraction checks
  all pass; `perception` tests now run those three on v1 and v2 scenes, `sim`
  tests check v2 determinism and that the digital medium has no camera. Wall
  lines: in ~4 % of v2 (and v1) scenes a short, thick, bent pocket wall forks
  its skeleton into an extra off-axis piece; left as a known limit.
- Datasets `train_sketch2` / `val_sketch2` / `test_sketch2` (5000 / 500 / 1000,
  seeds 120 / 220 / 320). `train.py --init <ckpt>` starts from a trained
  model's weights with a fresh optimiser and schedule (fine-tuning).

**Sketch style v3 and region_by_agents (6 Oct 2026)**

- After v2 the real drawings were still not parsed reliably, and the S / G
  boxes never came out right. Cause: every letter in v1 / v2 is one Hershey
  script glyph, only scaled and rotated, while the region class rests entirely
  on that letter.
- `sketch.SKETCH_V3` (`--style sketch3`) = v2 plus: letters in many hands —
  half from the eight Hershey faces (30 % italic), half procedural pen strokes
  (2 S and 3 G skeletons, control points jittered, Catmull–Rom curves) — all
  sheared (± 0.35), stretched (0.7–1.35), rotated (± 25°), elastically warped
  (≤ 6 % of the height), 0.4–1.0 of the largest size, stroke ≤ h / 7; a harsher
  camera for the photographed medium (noise σ 2–14, blur 0.3–2.0, JPEG 30–90,
  light fall-off ≤ 50 %, a soft shadow edge in 30 %, low resolution 0.35–0.8 in
  30 %, gamma 0.7–1.4, residual perspective ± 12 px); pen dropouts in 35 %
  (5–20 % of the ink faded along smooth gaps; labels unchanged); 1–4 stray
  marks in 40 % (ink only, labelled background). v1 and v2 stay
  pixel-identical (hashes of 14 scenes each, before and after).
- `ExtractConfig.region_by_agents` (`--region-by-agents` in train / predict):
  region boxes are found on the union of both classes; a box holding a parsed
  agent is a start, others take their majority class. Agents are always drawn
  in their start box, so a misread letter no longer loses the pair. Default
  off (E5 unchanged); test `test_region_by_agents_fixes_a_misread_letter`.
- Checks on 140 v3 scenes: heat-target and extraction checks pass; JSON ↔
  drawing passes the 1 px circle-fit bound everywhere, and one agent of ~650
  exceeds the loose raw-centroid bound (2.71 > 2.5 px). `perception` tests run
  the sketch checks on v1, v2 and v3.
- Datasets `train_sketch3` / `val_sketch3` / `test_sketch3` (seeds 130 / 230 /
  330); 4.9 scenes/s on one core.

**Sketch style v4 and free-form layouts (6 Oct 2026)**

- Concern raised: agents drawn at any size, pens of inconsistent width, and
  layouts nobody planned. The seven families all keep start left / goal right
  (except crossing) and axis-parallel walls, so a model can learn "the left
  box is the start" and nothing about a diagonal wall.
- `sim/layouts.py` adds two free-form families, **not** in `LAYOUTS` (so
  `--layout mixed` and every dataset made with it are unchanged): `walls` (0–6
  random walls, 75 % axis-parallel and 25 % diagonal, 35 % attached to the
  boundary, no near-parallel slivers) and `rooms` (one or two walls with 1–2
  doors, 2–4 rooms). Both put the start and goal boxes anywhere, either way
  round (one side 10–16, the other 12–40), ≥ 35 units apart and off the walls.
  `MIXES["mixed2"]` (`generate.py --layout mixed2`) cycles the seven families
  plus 3 × walls and 2 × rooms (~40 % free-form). 30 / 30 scenes valid per
  family in a check; ~0.05–0.08 s per scene.
- `sketch.SKETCH_V4` (`--style sketch4`) = v3 plus one drawn agent size per
  drawing, 0.6–2 × the agent radius, each agent ± 15 % around it and capped at
  0.42 × its nearest neighbour's distance (no overlapping circles), tails 1.6–
  2.8 drawn radii; and every element (each wall, region box, obstacle, agent,
  letter, the hatching) with its own pen width, 0.55–1.7 × the drawing's pen.
  v1, v2 and v3 stay pixel-identical (hashes before / after).
- Checks on 140 v4 scenes over `mixed2`: heat-target and extraction checks
  pass; JSON ↔ drawing passes the circle-fit bound for every agent; 2 of ~800
  agents exceed a secondary pixel measure (raw centroid 4.9 px on a large ring;
  a 5.4° heading on a short tail). The test's centre bound is now 1.25 px
  (0.25 units; per-element widths make the label ring vary more), scaling with
  the drawn ring, and its heading bound scales with the drawn tail.
- Tests: `test_free_layouts_are_valid_and_varied` (sim: solvable, boxes off the
  walls, start boxes spread ≥ 30 units in x, `mixed` unchanged); v4
  determinism; the perception sketch checks run on v1–v4, v4 including the
  free-form families.
- Datasets `train_sketch4` / `val_sketch4` / `test_sketch4` (seeds 140 / 240 /
  340, `--layout mixed2`); metrics per layout include `walls` and `rooms`.

**Sketch v5: rooms, close agents, multiple exits (16 Oct 2026, §9)**

*Kept as specified.* `RegionSet`, `Scene.starts` / `exits` / `goal_for`, the
51-number observation, `--style sketch5`, `--layout mixed3`, JSON `"radius"` /
`"starts"` / `"exits"`, `TrainConfig.heat_sigma` (1.0), `ExtractConfig.heat_nms_factor`
(default = `nms_factor`) and `goal_mode` (`"pairs"`), `region_set_ok`, the
datasets of §9.5. v1–v4 and every older layout are unchanged: 15 sets
(14 scenes per sketch style on its dataset layout, 4 per layout family, 14 of
`mixed2`, 14 hard + aug3 scenes) hash identically before and after
(image, labels, JSON); the hashes, recorded with the pre-v5 code, are now
`sim/tests/reference_hashes.json`.

*Additions and choices the spec left open.*

- Layout names: `boxrooms`, `walls_multi`, `rooms_multi` (`layouts.MULTI_LAYOUTS`,
  not in `LAYOUTS` or `mixed2`). `mixed3` = `mixed2` + 2 × each (18 entries, a
  third of scenes new).
- `boxrooms`: a room is in a corner (35 %), against one frame edge (30 %) or
  free (35 %), ≥ 10 units from other rooms, sides 10–30 with one side ≥ 14
  (rooms meant to hold a box: 18–30). Sides on the frame are not drawn. One
  door (75 %) or two on different sides, 9.5–12 wide (free gap ≥ 6.5 >
  `min_bottleneck`), with ≥ 2 units of wall either side so the room still
  reads as a rectangle. Room 0 holds no box in 60 % of scenes; every other
  room takes a box with p = 0.7, but at least one box always stays outside the
  rooms. A box in a room keeps `wall_half + 2.5` off its walls; a box outside
  keeps `wall_half + region_margin` off every room. 10 retries before falling
  back to `open`. Measured on 300 layouts: a room with no box in 87 % of
  scenes; 31 % of boxes inside rooms.
- Several boxes: boxes ≥ 6 apart, start–exit centres ≥ 0.3 world (one pair:
  0.35); the long side 12–28 when there are more than three boxes. Agents go
  one per start box first, the rest to random start boxes (so one box may be
  crowded), each facing its nearest exit, with no goal of its own. A start box
  that gets no agent (fewer agents than boxes) is removed, and a scene left
  with one start and one exit is written in the single-box form (`groups`, no
  `starts` / `exits`), so k = m = 1 in 37–39 % of scenes, not a third. Every
  exit must also be reachable from a start box (otherwise it would sit in
  hatched dead space). The bottleneck is the widest disk with which every
  start box reaches some exit. 300 / 300 valid per family; 0.06–0.11 s per
  scene.
- JSON of a several-box scene: `groups` is `[]` and `start_region` /
  `goal_region` are the first start and exit, so older readers still work.
  `render.py` and `sketch.py` draw every start and exit; `reachability`
  checks each agent against its own goal set and keeps every box out of the
  dead space.
- Close agents: drawn per scene in `generate.py` from `SketchStyle.close_prob`
  (0.4 in v5, 0 otherwise, so other styles keep their random stream) and passed
  as `random_scene(close=True)`; this applies to every layout of `mixed3`, the
  seven fixed families included. An agent joining a start box that already holds
  one is placed 2R + `hard_gap` (+ 0.01) from a random one of them (first half
  of the tries; then anywhere at that spacing). `dataset.json` records `close`
  per scene for v5.
- Letters: stroke `U(0.5, 0.7)` × the element's pen (v4: 0.7).
- Drawn radius in the JSON = mean distance of one turn of the warped ring from
  its area centroid (the JSON centre).
- Heat targets: the Gaussian window is never smaller than the direction disk
  (nominal radius), which matters only for σ < r / 4.
- Extractor, exits mode: a box lying ≥ half inside a larger box is dropped
  first (`diagnostics["dropped_nested_boxes"]`). Found on generated v5 labels:
  the hole of a ring walled in by touching neighbours is cut off from its box
  (the seal closing bridges one ring, not two) and came back as a tiny start box
  holding that agent. Pairs mode is unchanged.
- "A start box holding an agent" (exits mode, extractor and metric) means an
  agent centre within one agent radius of the box: agents are placed anywhere
  in their box, edges included, and a drawn centre on the edge fell a pixel
  outside the parsed box (1 of 50 smoke scenes, found by an oracle run).
- Metrics: `region_set_ok` counts as true starts only the start boxes holding a
  true agent (in exits mode an empty start box is no start, on either side);
  it is reported in both modes. In exits mode `region_iou` and
  `region_count_ok` are taken over the box sets and `group_pairing_ok` is NaN.
  `scene_metrics(..., goal_mode=...)`; `train.py` and `evaluate.py` pass the
  run's.
- `--tune-extract` grids `heat_nms_factor` over 0.5, 0.6, 0.8, 1.0, 1.2 for runs
  that set it (or with `--nms-grid`); for those runs the ring threshold (heat
  fallback only) stays the run's unless `--ring-grid` is given. Other runs
  grid exactly as before. `tune_extract.json` records the goal mode.
- `evaluate.py` takes `--goal-mode`, `--heat-thresh`, `--heat-nms-factor`,
  `--region-by-agents`, and `predict.py` the first three, overriding the run's
  extractor: to apply tuned thresholds, and to score the v4 model as exit sets
  on `test_sketch5`.
- Tests (§9.7): `test_older_sketch_styles_unchanged` is the "v1–v4 hashes"
  test. Scene JSON hashes (floats rounded to 1e-6) are compared on any machine;
  image / label hashes only with the recorded numpy / OpenCV / Pillow versions
  (2.5.4 / 5.0.0 / 12.3.0 in the development container), because
  anti-aliasing and JPEG can differ between versions. Also added
  `test_close_agents_and_sketch5_json` (sim). The perception sketch checks
  now run on v5 too (fixed + free-form layouts + 2 × each multi-box family,
  every second scene packed), v5 extraction with its own σ 0.5, NMS 0.6 and
  exits mode, boxes checked with `region_set_ok`.
- Test hook: `sketch(..., parts=True)` also returns `ring_pts` / `tail_pts`,
  each agent's whole drawn ring and tail. Where v5 rings touch or tails cross,
  a later agent overwrites part of an earlier one in the `rings` / `tails`
  maps, which pulled the pixel circle fit up to 1.8 px off a correct centre and
  a tail fit 8° off; for those agents (drawn radii + 2 units apart or closer)
  the test fits the drawn curves and skips the raw-centroid bound. Agents of
  v1–v4 are measured as before.
- Checks on 140 v5 scenes drawn like `train_sketch5` (`mixed3`, 40 % packed,
  627 agents): JSON ↔ drawing passes in every scene; heat targets
  (σ 0.5 × drawn radius) peak on agent pixels everywhere; extraction on
  ground-truth labels with those targets, NMS 0.6 and exits mode matches all
  627 agents with no extra detection at 512 and at 256, obstacle counts exact,
  and `region_set_ok` holds in 140 / 140 scenes at 256 and 139 / 140 at 512.
  Known limit, the one failure: a packed row of agents drawn right across a
  narrow start box splits its label in two (the seal closing bridges one ring,
  not a row); pairs mode splits it the same way.

---

## 9. Sketch v5: rooms, close agents, multiple exits (spec, 15 Oct 2026; **built** 16 Oct, deviations in §8)

**Why.** Four real drawings parsed with the v4 model (`results/old_drawings_v4/`,
16 of 17 agents, 12 of 12 obstacles, every real start/goal right) showed the
remaining failures, all in the shapes v4 never generates:

| Observed on real drawings | Cause | Fix in v5 |
|---|---|---|
| A small room formed by walls and the frame was parsed as an S or G box, and the walls around it were lost (drawing 4) | regions are recognised by "rectangle + letter"; v4 has no rooms next to boxes and its letters are large, so "a rectangle" alone is enough | §9.1 rooms and decoy rooms; smaller, thinner letters |
| Two agents drawn touching: one detected (drawing 4) | v4 caps every drawn agent at 0.42 × the nearest neighbour's distance, so drawn agents never touch; heat peaks have σ = the *nominal* radius and blur together for small, close agents; heat NMS keeps peaks ≥ 1.6 nominal radii apart | §9.1 close agents; §9.2 sharper, size-aware heat targets and size-aware NMS |
| Agents crowded in a narrow box score 0.53–0.54 (threshold 0.4) | same as above | same; re-tune thresholds |
| One S with two G: the second G was dropped (drawing 2) | the extractor pairs boxes one-to-one | §9.3 multiple exits; §9.4 the simulator accepts any exit |
| Thin, light wall lines partly missed (drawings 2–4) | lines under one pixel at 256 | **out of scope**: the author draws with a ~1 mm marker |

Decisions (author, 15 Oct): one S may serve several G and several S may
share one G; **an agent succeeds by reaching any G** (multi-exit evacuation).
The author's own free drawings are **not** labelled and do not enter
training; they stay a diagnostic set, like `results/old_drawings_v4/`.

### 9.1 Generator: `sim/sketch.py` v5 and new layouts (`--style sketch5`, `--layout mixed3`)

v1–v4 must stay pixel-identical (hash 14 scenes per style before and after,
as for v2–v4).

- **Rooms next to boxes.** A new free-form family (`boxrooms`, not in
  `LAYOUTS`): 1–4 small rectangular rooms (sides 10–30 units) made of walls,
  often sharing walls with the frame, each with one or two doors; S and G
  boxes are placed **inside some rooms and outside others**, and at least
  one room in two scenes holds **no box at all** (a decoy). Rooms must not
  make the scene unsolvable.
- **Letters**: size 0.25–1.0 of the largest that fits (v4: 0.4–1.0), stroke
  as thin as 0.5 × the pen.
- **Close agents**: in 40 % of scenes, agents in a start box are placed at the
  generator's hard spacing (`2R + hard_gap`), and the drawn-size cap rises
  from 0.42 to **0.6 × the nearest neighbour's distance**, so drawn circles
  may touch or overlap slightly. Up to 8 agents per box. Agents' *world*
  positions still satisfy the simulator's ≥ 2R spacing; only the drawing
  touches.
- **Multiple starts and exits**: free-form scenes get k ∈ {1, 2, 3} start
  boxes and m ∈ {1, 2, 3} exit boxes in any combination (k = m = 1 in a
  third of scenes). Agents are only ever in start boxes. Every agent must be
  able to reach at least one exit (solvability per agent, any exit).
- **JSON**: every agent records its drawn radius (`"radius"`, world units);
  scenes with several boxes record `"starts"` and `"exits"` (lists of boxes).
  Single-box scenes keep `start_region` / `goal_region` as now.
- `MIXES["mixed3"]` = `mixed2` + 2 × `boxrooms` + multi-box variants of
  `walls` / `rooms`.

### 9.2 Perception model

- **Heat targets** (`targets.py`): σ = `heat_sigma` × the agent's **drawn**
  radius when the JSON has `"radius"`, else the nominal radius as now.
  `TrainConfig.heat_sigma` default 1.0 (E5 and v4 runs unchanged); v5
  trains with **0.5**.
- **Heat NMS** (`extract.py`): minimum peak distance in heat mode =
  `ExtractConfig.heat_nms_factor` × nominal radius, default = `nms_factor`
  (unchanged behaviour); v5 uses **0.6** and `--tune-extract` grids it
  together with `heat_thresh`.
- Train: `--init` the v4 model (`sketch4_ft_seed0/ckpt_best.pt`), train on
  `train_sketch5 train_aug3`, val `val_sketch5`, 256 px, instance head,
  `--wall-mode lines`, 30 epochs, lr 5e-4, `--heat-sigma 0.5`,
  `--goal-mode exits` (§9.3). ≈ 1.5 h on the GTX 1070.

### 9.3 Extractor: exits instead of pairs

`ExtractConfig.goal_mode`: `"pairs"` (default, current behaviour, E5
unchanged) or `"exits"`:

- every start box that **contains at least one parsed agent** is a start;
  start boxes with no agent are dropped and counted in
  `diagnostics["dropped_empty_starts"]` (this removes the fake start of
  drawing 4);
- every goal box is an exit; no pairing; unpaired boxes are not an error;
- the parsed `Scene` carries `starts` and `exits`; agents carry no
  `goal_region`, so `goal_for` returns the exit set (§9.4);
- a parse fails only when there is no agent or no exit.

`metrics.py`: a `region_set_ok` metric (every true start box and every true
exit matched one-to-one by a parsed box of the same kind, IoU > 0.5, no extra
boxes). In `goal_mode="exits"`, `scene_usable` uses it instead of
`group_pairing_ok`. Ground-truth scenes with explicit per-agent goals
(`crossing`) are scored as exit sets in this mode — a hand-drawn scene cannot
express pairing, so parsed scenes always mean "any exit".

### 9.4 Simulator: any exit counts (`sim/scene.py`, `sim/physics.py`)

- `scene.RegionSet(regions)`: `contains` = any, `distance_to` = min,
  `nearest_point` = the nearest point of the nearest member — the same
  interface as `Region`, so `physics.py` needs no change in its goal logic.
- `Scene.starts` / `Scene.exits` (optional lists). `Scene.goal_for(agent)`:
  the agent's own `goal_region` if set (crossing keeps its two groups), else
  `RegionSet(exits)` if `exits` is set, else `goal_region` (every existing
  scene).
- Observation (51 numbers) unchanged in size and layout: goal distance and
  bearing now refer to the nearest exit. Reward progress uses the same
  distance. `stats()` unchanged (an agent's outcome is `goal` on entering any
  exit).
- Rendering (`render.py`, `sketch.py`) draws every start and exit;
  `reachability` checks each agent against its own goal set.
- Every existing layout produces exactly the scenes it did before (hashes),
  and `sim` tests stay green; this is what keeps E5 and later RL comparable.

### 9.5 Datasets

`train_sketch5` / `val_sketch5` / `test_sketch5`: 5000 / 500 / 1000, seeds
150 / 250 / 350, `--style sketch5 --layout mixed3`.

### 9.6 Evaluation

One table, v4 model vs v5 model, on `test`, `test_aug3`, `test_sketch4`,
`test_sketch5` (rendered images must not regress; v4-style drawings must not
regress), plus per-layout rows for the new families. Then both models on the
four drawings in `results/old_drawings_v4/input` (and any the author adds),
with overlays and clean re-renders side by side for inspection.

Expected: fake boxes from rooms mostly gone as long as letters are legible;
touching agents mostly separated, heavy overlap (one circle over another)
still ambiguous; low-confidence agents rarer; one-S-many-G and many-S-one-G
parsed and simulated as exits. Thin lines are not addressed.

### 9.7 Tests to add

| Test | Checks |
|---|---|
| `sim`: v5 determinism; v1–v4 hashes unchanged | as for v2–v4 |
| `sim`: `test_region_set_goal` | an agent reaches the farther of two exits → outcome `goal`; observation points at the nearer exit |
| `sim`: `test_existing_layouts_unchanged_by_exits` | every `LAYOUTS` family and `mixed2` scene hashes as before |
| `sim`: `test_boxrooms_valid` | solvable for every agent; decoy rooms present; boxes inside and outside rooms |
| `perception`: `test_sketch5_close_agents_labels` | touching drawn agents still give one heat peak each at the JSON centres |
| `perception`: `test_exits_mode_drops_empty_start` | a start box without agents is dropped; two exits both kept; agent goal is the exit set |
| `perception`: `test_heat_nms_separates_touching` | two heat peaks 1.2 nominal radii apart are both detected with `heat_nms_factor` 0.6 |

---

## 10. Sketch v6: recognise by pattern, not by line width (spec, 16 Oct 2026)

**Why.** The v5 model on seven real drawings (`results/old_drawings_v5/`,
`results/drawings_v5/`): every agent count and every real start right, but
four failures, all from drawings that only the *shape* of the strokes can
explain:

| Observed (v5) | Drawing |
|---|---|
| a 5.5 × 6 G box read as a start box, its letter G read as an agent | `drawings_v5/30f1…` |
| the two sides of a 6-unit-wide G box read as two walls; the exit kept only a 4 × 9 corner | `drawings_v5/f2e8…` |
| a 2 × 10 sliver at a room corner read as an exit | `drawings_v5/823f…` |
| the 7 × 11 gap between an S box and a room wall read as an exit | `old_drawings_v5/a7e1…` |

Cause, from the generator: **line width tells walls from box sides.** A wall
is drawn at pen × U(1.0, 2.2) (`wall_width_mult`), a box side at pen × 0.6, so
walls are on average ≈ 2.7 × wider than box sides in every sketch set, and
the network uses the shortcut. It also never sees a box with a side under 10
units (16 mm on paper), a box closer than `wall_half + region_margin` = 5.5
units to a wall, or a corner sliver. The §9 advice "draw walls with a thick
marker" served the shortcut and is withdrawn: after v6 the author draws
walls and boxes with the same pen, thin or thick.

Decision (author, 16 Oct): recognise walls, boxes, agents and obstacles by
their **pattern** — a box is a closed rectangle with a letter in it, a wall
an open line, an agent a small circle with a tail, an obstacle a filled blob
— and make line width carry no information. Three parts: equal widths (§10.1),
hard cases only pattern separates (§10.2), and an explicit letter head whose
detections decide which closed rectangles are boxes and of which kind
(§10.3–10.4).

### 10.1 Generator: one width law for every line (`--style sketch6`)

`SKETCH_V6 = replace(SKETCH_V5, …)`; v1–v5 stay pixel-identical (hashes).

- Walls, box sides, agent circles and tails, obstacle outlines: each element's
  stroke width = pen × `ew()` × U(**0.45, 1.6**), drawn **independently for
  every element from the same law** (no `wall_width_mult`, no fixed 0.6 for
  box sides). Any element, box sides included, is "gone over twice" (width ×
  U(1.6, 2.0)) with probability 0.2.
- `element_width` widens to (0.5, 2.0), so one drawing holds thin walls next
  to thick boxes and the reverse.
- Probe style `sketch6swap` (test only): walls always at the thin end
  (U(0.45, 0.7)), box sides always at the thick end (U(1.3, 1.6)); the reverse
  of what v1–v5 taught. A model that has learned the pattern scores the same
  on it as on `test_sketch6`.

### 10.2 Generator: hard cases (`--layout mixed4`)

New layout options with defaults that keep every existing layout and mix
byte-identical (`sim` hashes): `box_short_min` (10), `box_near_wall_prob`
(0), `corner_gap_prob` (0), `slit_prob` (0). `MIXES["mixed4"]` = `mixed3`
families with:

- **small and narrow boxes**: short side U(`box_short_min` = **6**, 16),
  long side as now; a start box still holds every agent it is given at the
  simulator's spacing (fewer agents rather than a box they do not fit);
- **boxes near walls**: with probability **0.3** a box is placed with a gap
  of U(0.3, 2.0) units between its side and a wall's or the frame's surface,
  inside or outside a room (the drawn strokes touch or nearly); agents keep
  the simulator's wall clearance, only the box comes closer;
- **room corners left open or crossed**: with probability **0.5** per room
  corner, the two walls stop U(0.5, 3) short of each other or one runs U(0.5,
  4) past the other (the slivers and gaps of real drawings);
- **slits**: with probability **0.3** per scene, two parallel walls U(3, 8)
  units apart and U(8, 25) long, alone or as part of a room — the shape of a
  narrow box without a letter;
- decoy rooms as in v5. Every scene stays solvable for every agent.

### 10.3 Letter head (model, targets, loss)

- **JSON**: sketch6 scenes record `"letters"`: one `{"kind": "S"|"G",
  "pos": [x, y], "height": h}` per drawn box, world units, after the photo
  perspective, like every other coordinate. Earlier styles do not write it
  (JSON hashes unchanged).
- **Model**: `ModelConfig.letter_head` (default False): a 1 × 1 conv → 2
  channels, sigmoid = letter heat for S and for G, on the same features as
  the other heads (+ 50 parameters). Old checkpoints load unchanged.
- **Targets** (`targets.py`): per letter a Gaussian at `pos` with σ =
  0.3 × `height` (px), peak pixel = 1, in its kind's channel — the agent heat
  convention.
- **Loss**: the agent heat focal loss on each channel, weight
  `lambda_letter` (1.0), normalised by the number of letters; scenes without
  `"letters"` (rendered sets, earlier sketch sets) contribute **no** letter
  loss (a per-sample mask, not a zero target).
- **Fine-tuning**: `--init` a checkpoint without a letter head loads every
  matching tensor and leaves the new head at its initialisation (bias −2.19
  as for agent heat); any *other* mismatch still raises.

### 10.4 Extractor: boxes need a letter (`ExtractConfig`)

New fields, defaults = v5 behaviour:

- `letter_gate` (False; requires a letter head): every box the segmentation
  yields (start or goal) is kept only if the max letter heat inside it,
  over both channels, is ≥ `letter_thresh` (0.3); its **kind is the
  stronger channel there**, not the segmentation's class and not
  `region_by_agents`. Dropped boxes are counted in
  `diagnostics["dropped_no_letter"]`. Pixels of a dropped box become
  background for wall extraction (its sides are not turned into walls by
  this step; walls come from the wall mask as now).
- `letter_suppress_agents` (True, only with `letter_gate`): an agent peak
  within 1.5 nominal radii of a letter peak ≥ `letter_thresh` whose score is
  higher than the agent's is dropped (`diagnostics["agents_on_letters"]`).
- `min_box_side` (0 = off; v6 runs use **3.0** units): boxes whose shorter
  side is below it are dropped (`diagnostics["dropped_thin_boxes"]`).
- `evaluate.py --tune-extract` additionally grids `letter_thresh` ∈ {0.2,
  0.3, 0.4, 0.5} when the run has a letter head; the result is written to
  `best` like the other thresholds. `evaluate.py` / `predict.py` accept
  `--letter-gate`, `--no-letter-gate`, `--letter-thresh`, `--min-box-side`.
- The run's own `ExtractConfig` (saved with the checkpoint) sets
  `letter_gate` on for v6, so a plain `predict.py` uses it. On rendered
  images (no letters drawn) the gate must be off: `evaluate.py` turns it off
  for any set whose `dataset.json` style is not a sketch style, and says so.

### 10.5 Datasets and training

- `train_sketch6` / `val_sketch6` / `test_sketch6`: 5000 / 500 / 1000, seeds
  160 / 260 / 360, `--style sketch6 --layout mixed4`, agents 1–8, obstacles
  0–6; `test_sketch6swap`: 500, seed 361, `--style sketch6swap --layout
  mixed4`. All in `make_datasets.py`.
- Train: `--init` the v5 model (`sketch5_ft_seed0/ckpt_best.pt`), train
  `train_sketch6 train_aug3`, val `val_sketch6`, 256 px, `unet24x4`, instance
  head, **letter head**, `--wall-mode lines --region-by-agents --goal-mode
  exits --heat-sigma 0.5 --heat-nms-factor 0.6 --min-box-side 3
  --letter-gate`, 30 epochs, lr 5e-4. ≈ 1.5–2 h on the GTX 1070.

### 10.6 One command (`scripts/run_v6.py`)

Like `run_v5.py`, resumable, every step skipped when its output exists:
datasets → train → `--tune-extract` on `val_sketch6` → evaluations → aggregate
→ real drawings → git add / commit / push.

- Evaluations (`E:\cnn-eval`): v6 on `test`, `test_aug3`, `test_sketch4`,
  `test_sketch5`, `test_sketch6`, `test_sketch6swap` (tuned thresholds); v6
  with `--no-letter-gate` on the three sketch-5/6 sets (`sketch6_nogate`);
  v5 on `test_sketch6` and `test_sketch6swap` (`sketch5_on_sketch6`).
  Aggregate → `results/sketch6`.
- Real drawings: every photo in `results/old_drawings_v4/input` and
  `results/drawings_v5/input` → `results/drawings_v6/`, with
  `compare_<name>.png` = photo | v5 clean | v6 overlay | v6 clean, and
  `summary.md` as in `test_drawings.py` plus the dropped-box columns.
- `test_drawings.py` gains `--run` default = the v6 run once it exists
  (falls back to v5), so new drawings go through the newest model.

### 10.7 Evaluation and what to expect

One table: v5 vs v6 (gate on / off) on every set above, per-layout rows for
`mixed4`. Must hold: rendered sets and `test_sketch4/5` do not regress by
more than 0.5 point of `scene_usable`. Expected: v5 drops sharply on
`test_sketch6swap` (it reads by width), v6 scores there within ~1 point of
`test_sketch6`; on the seven real drawings the four failures in the table
above are gone or reduced to "box dropped" (never "box invented").

### 10.8 Tests to add

| Test | Checks |
|---|---|
| `sim`: v1–v5 and every layout / mix unchanged; v6 determinism | hashes, as for v5 |
| `sim`: `test_sketch6_width_law` | over 50 scenes, mean drawn wall width ≈ mean box side width (ratio 0.8–1.25); `sketch6swap` ratio < 0.6 |
| `sim`: `test_mixed4_hard_cases` | boxes with short side < 10 present; boxes within 2 units of a wall present; every agent fits its box and can reach an exit |
| `sim`: `test_sketch6_letters_json` | one letter per box, inside it, kind matches the box |
| `perception`: `test_letter_targets` | peaks of value 1 at letter centres, right channel; no letters → mask 0 |
| `perception`: `test_init_adds_letter_head` | a v5-shaped checkpoint initialises a letter-head model; an arch mismatch still raises |
| `perception`: `test_letter_gate` | box without letter heat dropped; box with G heat but start class becomes an exit; gate off = v5 result |
| `perception`: `test_agent_on_letter_suppressed` | agent peak on a stronger letter peak dropped, one 3 radii away kept |
| `perception`: `test_min_box_side` | a 2 × 10 box dropped at 3.0, kept at 0 |
