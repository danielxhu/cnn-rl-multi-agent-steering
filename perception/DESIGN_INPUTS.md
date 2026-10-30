# Multiple input types → one scene format (main line B, step 1)

Status: draft, 2026-10-28. Decisions were taken with Daniel (§1, §6); what is still open is in §7.

Main line B: (1) hand drawings, floor plans and map / satellite images all parse into **one scene format**; (2) the same policy runs in parsed and ground-truth scenes and the behaviour gap measures the cost of perception error; (3) a policy is trained to tolerate that error. This document covers step 1 only.

## 1. Decisions

| Question | Decision |
|---|---|
| What an agent is | **Scale is a parameter.** The scene stores metres per world unit; people, ground robots and drones are all expressible. |
| Polygonal obstacles (buildings, furniture) | **New obstacle type `polygon`.** Needs a simulator change (§3) and affects the policy line. |
| Training data for floor plans and maps | **Train on real datasets**, after converting their annotations to our label classes (§4). |
| Agents and start / goal on plans and maps | **Both:** use marks drawn on the image when present, otherwise place them automatically (§2.4). |

## 2. Scene format v2

Extends the current `scene.json` (`sim/scene.py`). Every v1 file stays valid: new fields are optional and default to v1 behaviour.

### 2.1 New top-level fields

```json
{
  "format": 2,
  "input_type": "sketch | floorplan | map | satellite | synthetic",
  "source": {"dataset": "CubiCasa5K", "item": "high_quality/1234", "license": "..."},
  "units": {"metres_per_unit": 0.125},
  "extent": [0.0, 0.0, 100.0, 72.0]
}
```

- `units.metres_per_unit`: physical scale. v1 default: unset (unitless). Agent radius, speed and `fov_range` stay in world units, so the policy is unchanged; the scale only says what the world means.
- `extent`: world rectangle `[x0, y0, x1, y1]`; v1 default `[0, 0, 100, 100]`. Allows non-square plans and map tiles.

### 2.2 Polygon obstacles

```json
"obstacles": [
  {"c": [58.7, 13.1], "r": 9.15},
  {"poly": [[10, 10], [24, 10], [24, 18], [10, 18]]}
]
```

Circle entries keep the v1 shape; `poly` entries are simple polygons, counter-clockwise, world units. Rendered and labelled as class `obstacle`.

### 2.3 Provenance and confidence (for step 2)

Every wall, obstacle, region and agent may carry:

```json
{"conf": 0.83, "from": "parsed"}
```

`from` ∈ `gt | parsed | placed` (placed = generated automatically, §2.4). `conf` is the parser's confidence in [0, 1]; ground truth has no `conf`. Step 2 uses these to split the behaviour gap by element type and confidence.

### 2.4 Agents, start and goal

1. If the image contains start / goal marks and agent circles (the hand-drawing vocabulary of DESIGN_SKETCH), parse them as today.
2. Otherwise `placement` describes how the simulator fills them in, and every placed item gets `"from": "placed"`:

```json
"placement": {"n_agents": 6, "start": "auto", "goal": "auto", "seed": 17}
```

`auto` start / goal: two free-space regions chosen by the reachability check in `sim/reachability.py`, as far apart as the layout allows.

## 3. Simulator change (affects the policy line)

- `physics.Simulator`: ray casting and collision for polygon obstacles (ray vs polygon edges; point-in-polygon or edge distance for collision). Polygons go in the **obstacle** channel of the observation.
- `reachability.py` and `render.py`: rasterise polygons.
- Policies trained only on circular obstacles may need retraining or fine-tuning on polygon scenes. **Tell the policy session before this lands.**

## 4. Data sources

| Input | Dataset | What it gives | Conversion to our classes | To check |
|---|---|---|---|---|
| Floor plan | CubiCasa5K (arXiv:1904.01920) — training | 5,000 plan images, polygon annotations, 80+ categories | wall → `wall`; furniture, fixtures → `obstacle`; door → `door`; everything else → `background` | CC BY-NC 4.0 |
| Floor plan | Modified Swiss Dwellings, MSD (arXiv:2407.10121, ECCV 2024) — test | 5.3K floor plans of building complexes, vector structure | walls and structural elements → `wall`; doors → `door` | licence; whether raster images come with it |
| Floor plan | RPLAN | 80K plans | — | access needs a request |
| Satellite | AI2 S2-NAIP (Hugging Face `allenai/s2-naip`) | 512×512 NAIP tiles, 1.25 m/px, with OSM buildings, roads and 30 other categories as GeoJSON in pixel coordinates | building → `obstacle` (polygon); everything else → `background` | licence of the bundle; OSM is ODbL |
| Satellite | SpaceNet (arXiv:1807.01232) | building footprints | building → `obstacle` | CC BY-SA 4.0: derived labels must keep it |
| Map | OpenStreetMap rendered by us | vector → rendered map style, pixel-exact labels | buildings → `obstacle`, barriers → `wall` | ODbL attribution |

### 4.1 The missing classes in real data

Real plans and maps contain no agents, start or goal marks, so `agent`, `start_region` and `goal_region` never appear in their labels. Since §2.4 lets users draw marks on these images, training composites marks onto the real images: paste rendered hand-drawing marks (agent circles with heading, S / G boxes) at random free positions and add them to the label map. Without this the U-Net would never see a mark on a plan or map background.

### 4.2 Scale

At 1.25 m/px a person is below one pixel, so satellite inputs supply walls and obstacles only; agents there are either placed (§2.4) or drawn as marks at drone / vehicle scale. Each dataset records its `metres_per_unit` from its known pixel size (plans: from the plan scale or door width; NAIP: 1.25 m/px).

## 5. What changes in perception

- Labels: the six classes of `sim/config.py` plus `door` (index 6, §6 item 4); polygons are rasterised into `obstacle`.
- Extraction (`extract.py`) must output polygons for obstacles that are not circles: fit a circle, keep it if the fit residual is small, otherwise simplify the contour to a polygon.
- One model per input type first, a unified model later (§6 item 1).

## 6. Decisions on the former open items (2026-10-28)

1. **Models: one per input type first, then merge.** Each input type gets its own U-Net (the sketch model stays as is); at the end one unified model is trained and compared with the per-type models. The comparison is itself an experiment. Until the merge, the caller states the input type.
2. **Floor-plan data: both datasets.** CubiCasa5K for training, MSD as a test set from a different source (does a parser trained on one dataset hold on another?).
3. **Satellite agents are drones**, drawable on the image (radius ≈ 2.5 m ≈ 4 px at 1.25 m/px), using the hand-drawing mark vocabulary. Assumption: low-altitude flight, so buildings are obstacles.
4. **Doors are a separate class.** Labels go from 6 to 7 classes (`door` appended as index 6 so existing class indices stay valid). In the scene format doors are segments in a new optional `doors` list; the simulator treats them as open (passable) for now, so physics and the policy are unchanged. Extraction must also leave a gap in the wall where a door is.

### 6.1 Licences (checked 2026-10-28)

| Dataset | Licence | Consequence |
|---|---|---|
| CubiCasa5K | **CC BY-NC 4.0** (LICENSE file in the GitHub repo) | Research use is fine; no commercial use; credit the authors. |
| S2-NAIP | Apache-2.0 on the Hugging Face card; its sources keep their own terms (OSM: ODbL, attribution required) | Fine for research; cite OSM. |
| MSD | **Not confirmed.** Paper says code and data are open; the 4TU record (doi:10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9) did not show a licence in a script-readable form | Read the licence on the 4TU page before downloading. |
| SpaceNet | CC BY-SA 4.0 | Labels derived from it must keep CC BY-SA. |

## 7. Still open

- Merge criterion: what counts as "the unified model is as good" (e.g. behaviour gap within a fixed margin of the per-type models).
- Floor-plan furniture: all furniture → `obstacle`, or only furniture above a height / size threshold.
