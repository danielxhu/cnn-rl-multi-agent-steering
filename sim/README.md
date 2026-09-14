# Steering Scene Simulator

Generates 2D steering scenes as an RGB image, a pixel-exact label map and a
JSON description of the geometry, so a perception model can be trained without
hand annotation. The same scene can be stepped by a small multi-agent
simulator, which is where the steering policy trains.

---

## Run

Needs Python 3.10+ with `numpy`, `opencv-python-headless` and `pillow`.

```bash
python generate.py --n 3000 --layout mixed --agents 3-8 --obstacles 2-6 --seed 0 --out data/train
```

```bash
python generate.py --n 7 --layout mixed --preview 7 --out data/preview   # one scene per layout
python tests/run_tests.py                                                # 13 tests, ~20 s
```

No system Python? `uv run --python 3.12 --with numpy --with opencv-python-headless --with pillow python generate.py ...`

---

## Input and output

**Input:** a layout name and a seed. Everything else comes from `Config`
(see [Tuning](#tuning)). Same seed and arguments → same dataset.

**Output:** three files per scene and one `dataset.json` manifest per run.

| File | Contents |
|---|---|
| `scene_00001.png` | RGB image, 512 × 512 by default |
| `scene_00001_labels.png` | palette PNG; the raw pixel value is the class index |
| `scene_00001.json` | exact geometry, see below |
| `dataset.json` | full config, CLI arguments, one summary row per scene |

Label classes, in draw order (later overwrites earlier):

| 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| background | wall | obstacle | start region | goal region | agent |

Dead space no agent can reach is painted as wall. The field of view is not
drawn; it follows from the agent pose and the constants in `dataset.json`.

```json
{
  "id": "scene_00001", "seed": 1738271, "layout": "doorway", "bottleneck": 9.62,
  "world": {"size": 100.0, "img_size": 512, "agent_radius": 2.0,
            "wall_thickness": 3.0, "fov_deg": 220.0, "fov_range": 22.0},
  "walls": [[[0.0, 0.0], [100.0, 0.0]], "..."],
  "obstacles": [{"c": [52.1, 48.3], "r": 6.2}],
  "start_region": [4, 30, 18, 70],
  "goal_region": [86, 30, 98, 70],
  "groups": [[[4, 30, 18, 70], [86, 30, 98, 70]]],
  "agents": [{"pos": [8.4, 52.1], "heading": 0.37}]
}
```

- `agents`: exact poses; heading in radians, 0 = +x, counter-clockwise.
  Agents in a second group carry their own `goal_region`.
- `groups`: every (start, goal) pair; only `crossing` has two.
- `bottleneck`: diameter of the widest disk that can travel start → goal.
  A free difficulty label.
- World y increases **upward**; image row 0 is the top.

---

## Layouts

| Layout | Geometry | Tests |
|---|---|---|
| `open` | scattered obstacles | baseline |
| `corridor` | two parallel walls | lateral constraint |
| `room` | four walls | boundaries on every side |
| `doorway` | one wall, one gap | committing to a passage |
| `two_doorway` | one wall, two gaps of unequal width | choosing a passage |
| `cul_de_sac` | two gaps, one into a sealed pocket | trap and exit look identical from outside |
| `crossing` | two perpendicular corridors, two agent groups | agent-to-agent avoidance |

Every scene is validated before it is written: start must reach goal for a
disk of the agent's radius, and the narrowest passage must exceed
`min_bottleneck`. Failures are resampled, never relaxed.

Greedy "turn toward the goal at full speed" success rate, 20 scenes per layout:

| open | room | two_doorway | doorway | corridor | cul_de_sac | crossing |
|---|---|---|---|---|---|---|
| 56% | 34% | 12% | 6% | 5% | 5% | 4% |

---

## Architecture

```
                          config.py          scene.py
                     (all parameters)    (Scene / Agent /
                             |            Obstacle / Region)
             +---------------+---------------+
             |               |               |
        geometry.py     reachability.py      |
      (rays, distances)  (C-space, DT,       |
             |            connectivity)      |
             |               |               |
    +--------+-------+-------+-------+-------+
    |                |               |
physics.py       layouts.py      render.py
(sensing,      (7 families,    (RGB + labels,
 dynamics,      validated       dead-space fill)
 metrics)       sampling)           |
    |                |               |
    |                +-------+-------+
    |                        |
    |                  generate.py
    |                (CLI, dataset I/O)
    |
 (RL loop, evaluation)
```

Two pipelines share one data type. **Generation:** `generate.py` →
`layouts.random_scene` → `Scene` → `render.render` → files. **Simulation:**
`Scene` → `physics.Simulator` → observations, rewards, stats. `Scene` holds
geometry only; all mutable state lives in `Simulator`.

**Sampling.** A layout builder returns walls, (start, goal) pairs and an
obstacle zone. Obstacles are placed one at a time with a connectivity check
after each, so a bad obstacle costs one sample rather than the whole scene.
Agents are placed at least `2 * agent_radius + agent_gap` apart, facing their
goal. The finished scene is checked for solvability and bottleneck width.

**Reachability.** The scene is rasterised and a distance transform gives each
free cell's clearance. A disk of radius R fits where clearance ≥ R, so every
"can it get through?" question is a threshold plus a 4-connected labelling;
`bottleneck_width` binary-searches the threshold. Raw cell checks would pass a
1.5-radius gap that no agent can pass.

**Rendering.** Image and label map are drawn from the same primitives in the
same order, so labels are exact by construction.

**Invariants.** One y flip (`render._to_px`). One wall thickness driving
drawing, ray stops and collision. One draw order. Synchronous stepping: all
actions computed from one world state, then applied together.

| To add… | Touch |
|---|---|
| a layout | a builder in `layouts.py`, registered in `BUILDERS` and `LAYOUTS` |
| a pixel class | enum in `config.py`, draw step in `render.py`, `PALETTE` |
| a sensed entity category | enum in `config.py`, one block in `Simulator.observe_one` |
| a non-circular obstacle | `geometry.py`, `reachability.occupancy`, `render.py`, `physics._hits_static` |
| a metric | `Simulator.stats()` |

---

## Simulator

```python
import json
from scene import Scene
from physics import Simulator, greedy_action

sim = Simulator(Scene.from_dict(json.load(open("data/train/scene_00001.json"))))
obs = sim.reset()                                   # (n_agents, 51)
while not sim.done:
    obs, rewards, done = sim.step([greedy_action(sim, i) for i in range(len(sim.pos))])
print(sim.stats())
```

**Observation** (51 per agent, normalised, no absolute position): 16 FOV
sectors × nearest {obstacle, wall, agent} distance, then goal distance, goal
bearing, own speed.

**Action:** `(turn, speed)`, `turn ∈ [-1, 1]` × `max_turn_deg`, `speed ∈ [0, 1]` × `max_speed`.

**Reward:** `+goal_bonus` on arrival, `-collision_penalty` (wall/obstacle) or
`-agent_collision_penalty`, `-step_cost` per step, `+progress_weight ×`
goal-distance decrease.

**End:** arrival, collision (if `terminate_on_collision`), or `max_steps`.
`stats()` gives success / collision / timeout rates and collision violations,
finish time, remaining agents.

---

## Tuning

Common knobs are flags; anything in `config.py` can be overridden with
`--config my.json`.

| I want to… | Use |
|---|---|
| change image size | `--size 256` |
| change counts | `--agents 3-8`, `--obstacles 2-6`, `--obstacle-radius 4-10` |
| make scenes harder | `--hard-ratio 0.3` (near-tangent agent pairs) |
| tighter passages | `--min-bottleneck 4.5` (must exceed `2 * agent_radius`) |
| domain randomisation | `--jitter-color 25 --jitter-line-width 1.5 --jitter-background 20 --noise-std 6 --blur-sigma 0.8` |
| generate faster | `--cell 0.5` |
| bigger world | `--world 200`, and scale `agent_radius`, `obstacle_radius`, `fov_range`, `wall_thickness` |
| leave dead space white | `--no-fill-unreachable` |
| fewer merged agent outlines | `agent_gap` in `config.py`: 1.5 → 18% of scenes merge, 3.0 → 0% |

Reward and episode parameters live in `config.py`.

---

## Debugging

**Re-render a scene.** The seed is in the JSON, so output is deterministic.

```python
from render import render, colorize
img, lab = render(Scene.from_dict(json.load(open("data/train/scene_00042.json"))))
img.save("debug.png"); colorize(lab).save("debug_labels.png")
```

**Check reachability directly.**

```python
from reachability import is_solvable, bottleneck_width, unreachable_free
```

**Fewer obstacles than asked.** Placement gives up after `max_place_tries`;
tight layouts and large `obstacle_r_max` reduce the count. See `dataset.json`.

**Many scenes rejected.** `min_bottleneck` too large for the obstacle sizes,
or the radius range too wide for the layout.

**Upside down.** Only `render._to_px` flips y; grid masks go through
`np.flipud`. `test_world_y_up_maps_to_image_y_down` catches a missing flip.

**Sensing and collision disagree.** Both derive from `wall_thickness`; never
hard-code a thickness. `test_vectorised_rays_match_the_scalar_reference`
pins the batched casts to `geometry.py`.

**Speed.** ~50 ms per scene end-to-end; 0.07 ms per agent-step; ~15 KB per
scene on disk. When re-rendering a fixed scene in a loop, pass the reachability
mask as `clear` instead of recomputing it.

---

## Limitations

- Static circular obstacles only; no moving obstacles, no leader.
- Straight walls, axis-aligned regions, one visual style.
- Multi-agent congestion is not validated; `bottleneck` and agent count are in
  the JSON so evaluation can account for it.
- `Simulator` favours clarity over speed; vectorise if it bottlenecks RL.
