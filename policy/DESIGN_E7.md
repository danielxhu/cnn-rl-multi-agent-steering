# Policy: E7, a memory of seen walls

Design document for experiment **E7**: does a memory of the walls an agent has
seen reduce looping and wall collisions? It builds on `policy/DESIGN.md` (E2)
and `DESIGN_E3.md`: everything not stated here is as there. Written before the
code; update §10 when the code diverges.

## 0. Why: what the failure analysis showed

`e3_both_seed0` on `test4`, the seven complex layouts, 2,799 agents
(16 Nov 2026; `RESEARCH_LOG.md` Part III §3, `policy/results/behave_complex/`):

- 13.2 % of agents **loop**: their cumulative signed heading change spans at
  least 360°. Loopers succeed 60 % of the time, other agents 92 %. Looping is
  worst in `cul_de_sac` (31 %), `crossing` (15 %), `doorway` (13 %) and
  `boxrooms` (12 %). In `corridor` only 26 % of loopers succeed.
- In the GIFs, agents get trapped in a start room whose door faces away from
  the goal, walk into the dead end of a cul-de-sac, dither at gaps 5.3–6.7
  wide, and jam at junctions.
- Most collisions are static: 5.0 % of agents hit a wall or obstacle, 1.2 %
  hit another agent.

The first two failures are failures of memory. The field of view is 220° and
22 deep in a 100-unit world, so a wall the agent has turned away from is gone
from the observation. The GRU could remember it, but PPO backpropagates
through only L = 32 steps (about 80 units at full speed) of a 300-step
episode. E3 showed that reward changes move success by less than seed noise.

**Not the cause: the goal bearing switching between exits.** For a
multi-exit goal the bearing points at the exit nearest in straight-line
distance, so it can switch while the agent walks. Only `boxrooms` and
`rooms_multi` have multi-exit goals. 19 % of their agents saw a switch, and
those agents succeed more often (96 % vs 91 %). Only 10 of the 369 loopers
ever saw one (Part III §4). E7 nevertheless adds a switching rule (§1.4),
because hand-drawn scenes have several exits more often than the suites do.

## 1. The change: a wall memory and a height field

One new option, `EnvConfig.wallmap` (default `"off"`, every existing variant).
With `"on"`, each agent keeps its own memory, and the policy observation grows
from 50 to 52 numbers. The simulator, the reward and the network code are
unchanged; only the input width of `fc1` follows `obs_dim`.

### 1.1 What an agent remembers

- **Only static geometry it has sensed itself.** At every step, each ray
  whose obstacle reading `[0:16]` or wall reading `[16:32]` is below 1 (a hit
  within `fov_range`) gives a hit point. The point is the agent's estimated
  position plus `d` along the ray direction (estimated heading + the sector's
  offset; with `rays_per_sector > 1` the sector centre is used). The memory
  reads the same observation the policy gets, noise included.
- **Joining.** The 16 rays are 13.75° apart, so the hit points of one wall
  are up to 5.3 apart at range 22, and a dotted wall would look passable. Hit
  points of two adjacent rays are joined by a segment when both rays hit a
  static entity and the points are less than `min_bottleneck` (5.2) apart.
  A real passage is at least 5.2 wide between surfaces, so a join can never
  close one. A gap narrower than 5.2 that the generator did not reject may be
  closed, which errs on the safe side.
- **Not remembered:** other agents (they move), the agent's own trail, and
  free space. Cells it has not seen count as free.
- **Lifetime:** one episode. The memory is cleared when the agent's scene
  starts, and ignored after it stops.
- **Raster.** A boolean grid, 0.5 units per cell (200 × 200 over the 100-unit
  world), with each hit point and joining segment drawn into it.

### 1.2 Position by dead reckoning

The agent places what it sees using its own estimate of where it is,
integrated from its own motion. The estimate starts at the true start pose;
the frame of the grid is the world's.

**Why this gives the agent no extra information.** With perfect odometry the
estimate equals the true pose. Choosing the world's axes and origin for the
frame is then only a choice of coordinates: the policy sees the downhill
direction (§1.3) relative to its own heading, which is the same in any frame
up to grid discretisation. No absolute position enters the observation.

- **Training:** perfect odometry (`odo_noise = "none"`).
- **Evaluation only** (§5.3): noisy odometry. Each step, the estimated
  heading changes by the true turn plus N(0, σ_θ), and the estimated position
  moves along the estimated heading by the true speed times (1 + N(0, σ_v)).
  Levels: `low` (σ_v = 0.02, σ_θ = 0.5°) and `high` (σ_v = 0.05, σ_θ = 2°).
  Hit points that land outside the grid are dropped. Training with drift is a
  later step, and only if evaluation shows a large loss.

### 1.3 The height field and the downhill direction

From the memory, a path-distance field over the agent's own map:

1. **Inflate by the agent radius.** Clearance of the memory raster (OpenCV
   distance transform). A cell is free when its clearance is at least
   `agent_radius` (2), so the agent can be treated as a point. A gap narrower
   than the agent's diameter is closed in the field. The gaps the generator
   allows (≥ 5.2) keep a free band of at least 1.2. This is the same
   configuration-space construction as E3's geodesic progress (`DESIGN_E3.md`
   §1.2).
2. **Coarsen to 2 units per cell** (50 × 50). A coarse cell is free if any of
   its fine cells has clearance ≥ 2 (as in E3, this keeps narrow passages
   open) **and** none of them is within 0.6 of a remembered surface. The
   second condition stops a thin remembered wall from leaking between two
   diagonal cells (§10.1). Gaps ≥ 5.2 stay open, gaps < 4 are shut, gaps of
   4–5.2 depend on how they sit on the grid.
3. **Goal: method A.** The observation has no goal distance, so the field's
   goal is a point far along the bearing. The point lies on the ray from the
   estimated position in the bearing's direction, where that ray leaves the
   box `[4, 96]²` (the world minus its boundary wall and an agent radius).
   This uses the world size, a constant of the environment. The source is
   every free coarse cell within 4 of that point, at its straight-line
   distance. With no such cell, the field is invalid (step 5).
4. **Propagate** with the E3 wavefront, a move is legal only through free
   cells, cost = path length, but with 16 directions instead of 32
   (`reach = 2`, half the cost, path lengths up to 2.7 % long). The wavefront
   code is shared with `sim/reachability.py`, refactored so that it takes a
   free mask and sources (§10).
5. **Downhill direction.** From the field cell of the agent's estimated
   position, take 3 steps of steepest descent over the 8 neighbouring cells,
   never cutting a corner past an unreachable cell. The direction points from
   the agent to the cell reached, about 6 units ahead. The policy receives its
   angle relative to the estimated heading as `(cos, sin)`. When the field is
   invalid, no cell around the agent is reachable, or there is no descent,
   the bearing itself is used. In the field, downhill is the path around the
   remembered walls; where nothing has been seen, downhill is the bearing.
6. **Update rate.** The field costs about 7 ms per computation (Mac, 50 × 50
   wavefront), so it is recomputed every `wallmap_every = 4` steps of a scene
   (steps 0, 4, 8, …) and whenever the target exit changes (§1.4). The
   downhill direction is looked up at every step on the newest field. The
   memory itself is updated every step.

### 1.4 Several exits: lock, and switch only when blocked

When the agent's goal is a `RegionSet`, the agent is told the bearing of
**every** exit, not only the nearest. It still gets no distance. A fixed rule,
not the network, decides which exit it heads for:

- **Lock.** At the scene's first step, the target is the exit nearest in
  straight-line distance (the one the simulator's bearing points at then).
- **Switch.** At a field update, if the agent's own cell is unreachable in
  the target's field (`inf`: the remembered walls cut it off), the target
  becomes the nearest exit not yet marked blocked. The old target is marked
  blocked, and the field is recomputed at once. When every exit is blocked,
  the marks are cleared and the nearest exit is used again.
- **What the policy sees.** The bearing column `[48]` of the 52 numbers is
  the bearing of the target exit, and the field is built for that exit.
  Single-goal scenes are unaffected (one exit, never switched).
- **Success is unchanged:** reaching any exit counts, as before. So does the
  reward (progress to the nearest exit).

With unseen cells counted as free, a switch needs the remembered walls to
enclose the agent or the target completely, so switches should be rare. Their
count is reported (§5).

### 1.5 The observation

| columns | content |
|---|---|
| `[0:48]` | rays, as E2 |
| `[48]` | bearing to the target exit / π (`wallmap = "on"`), else as E2 |
| `[49]` | own speed, as E2 |
| `[50:52]` | `cos`, `sin` of the downhill direction relative to the estimated heading |

The memory's state lives outside the network, in the env (training) and in
`PolicyController` (evaluation, validation, `gap.py`, `demo_e2e.py`). The
GRU's hidden state is unchanged in kind.

## 2. Variants and runs

| Variant | Change from `e3_both` | Runs |
|---|---|---|
| `e3_both` (reference, exists) | none | `e3_both_seed{0,1,2}` |
| `e7_map` (main) | `wallmap = "on"` | `e7_map_seed{0,1,2}` |
| `e7_longgru` (control) | `L = 128` (PPO sequence length; no memory map) | `e7_longgru_seed0` |

`e7_longgru` asks whether letting the GRU backpropagate through 128 steps
instead of 32 is enough on its own. With T = 128 every rollout row is one
sequence. One seed only, by the author's decision: it is compared with
`e7_map_seed0` and with `e3_both_seed0`, and the result is reported as
indicative, not as a finding.

## 3. What stays fixed

Everything of `e3_both`: reward (`timeout_penalty = 25`,
`progress_metric = "geodesic"`), ten layouts, 2–8 agents, random headings,
`gru` with hidden 128, 50 M agent-steps, PPO hyper-parameters (`L` only for
`e7_longgru`), validation suite and best-checkpoint rule, the `val`, `test4`
and `test_mixed` suites, and the simulator's physics.

**Old variants stay exact.** With `wallmap = "off"` (the default, and what
every existing `config.json` loads as), the env, the rollout, the controller
and the simulator produce the same numbers as before E7. Tests pin this
(§7). New `config.json` files gain the new `EnvConfig` fields, and the
evaluation outputs gain new columns (§5); no existing number changes.

## 4. Budget and runtime (GTX 1070 box, 10 rollout workers)

| Item | Estimate |
|---|---|
| `e3_both`-like GRU run | 4.5 h (about 3,000 agent-steps/s, measured in E3) |
| `e7_map` run | **7–8 h** (*estimate*): about 32 alive agents per worker step, a field every 4 steps at about 7 ms, so about +60 ms on about 107 ms per worker step, about 1,900 agent-steps/s |
| `e7_longgru` run | about 4.5–5 h (the same number of GRU steps per update) |
| Pilot (§8): 2 M steps of each new variant | about 30 min in total |
| Evaluations (2 suites × 4 runs, 3 `e3_both` re-evaluations, 6 noise evaluations) | about 2 h |
| **Total** | **about 30 h**: 3 × `e7_map` (≈ 23 h) + `e7_longgru` (≈ 5 h) + evaluation |

The pilot replaces the estimate with a measurement before the full runs
start. If `e7_map` comes in below 1,200 agent-steps/s (more than 12 h per
run), raise `wallmap_every` to 8 and note it in §10.

## 5. Metrics

Every metric is computed on `test4` and `test_mixed`. Learned rows are mean ±
std over seeds; `e7_longgru` is one run.

### 5.1 Added to `evaluate.py` (for every controller)

- **`loop_rate`**: the share of agents whose path loops. Steps shorter than
  0.2 are dropped. The heading of each remaining step is taken, and its signed
  change accumulated. An agent loops when the max − min of that sum is at
  least 360°. This is `turning()` of the 16 Nov analysis, moved into
  `evaluate.py`.
- **`loop_success_rate`** and **`noloop_success_rate`**: success among
  loopers and among the other agents.
- **`exit_switches`**: mean switches per agent of the §1.4 rule (0 for every
  other controller).
- Per scene, so they aggregate per layout and per bottleneck bin like the
  rest. They do not go into `val.csv` (validation columns stay as before).

### 5.2 Reported (existing)

Success, static and agent collision rates, timeout, per layout and per
bottleneck bin (`[5.2,7)`, `[7,10)`, `[10,15)`, `[15,inf)`). Additionally,
the **multi-exit subset**: scenes whose goals are `RegionSet`s with more than
one member.

### 5.3 Odometry noise (evaluation only)

`evaluate.py --odo-noise low|high` on the three `e7_map` checkpoints writes
`<eval>/e7_map_seed<k>_odo_<level>`. The table reports success and loop rate
at `none`, `low` and `high`.

### 5.4 Baseline re-evaluation

`e3_both_seed{0,1,2}` and the ORCA baselines are re-evaluated with the new
`evaluate.py` so that `loop_rate` exists for them. Their other numbers must
equal those already in `results/e3`; the driver checks this.

## 6. How the result is judged

Judged on `test4`, decided before the runs:

1. **`e7_map` helps** if its overall success exceeds `e3_both` by more than
   the sum of the two seed stds (E3's spread was ±0.013), **or** its loop rate
   falls by at least a third while success does not drop by more than that
   sum.
2. **Mechanism.** The gain should show where memory matters: `cul_de_sac`,
   `boxrooms`, `doorway`, and loopers. A gain only in `open` or `walls_multi`
   would point to something else (e.g. the extra input), and would be
   reported as such.
3. **Static collisions** are reported separately. A memory of walls should
   lower them; if it does not, the dithering at narrow gaps is a control
   problem, not a memory one.
4. **`e7_longgru`** (one seed) is set beside `e7_map_seed0` and
   `e3_both_seed0` and is described, not tested.
5. **Odometry noise:** the loss in success from `none` to `high` is reported.
   If it is more than half the gain of (1), training with drift is proposed as
   the next step.
6. **Reference for E4–E6.** If (1) holds, `e7_map` replaces `e3_both` as the
   reference policy. If not, `e3_both` stays and E7 is reported as a negative
   result.

## 7. Tests (CPU, `policy/tests/run_tests.py`)

1. `e7_1_off_is_unchanged`: with `wallmap = "off"`, a 400-step env stream
   (E2 and E3 reward: observation and reward totals, terminal / first /
   truncated counts) and a `PolicyController` rollout equal numbers recorded
   before E7 was written (main at `01ba8ff`).
2. `e7_2_geodesic_refactor_identical`: `geodesic_field` after the refactor
   equals fields recorded before it (E3 progress unchanged).
3. `e7_3_obs_dim_and_old_configs`: 52 with `wallmap = "on"`, 50 otherwise. A
   `config.json` without the new fields loads as `"off"`. Wall memory with
   the frame stack is refused.
4. `e7_4_hits_land_on_walls_and_a_5_2_gap_stays_open`: no remembered cell
   lies in free space more than one raster cell from a surface; a 5.2 gap
   stays open, and the downhill direction points through it.
5. `e7_5_downhill_goes_around_a_seen_wall`: with the goal behind a seen wall,
   the downhill direction points round it; with an empty memory it follows
   the bearing.
6. `e7_6_exit_lock_and_switch`: the target exit switches once when memory
   seals it; it never switches with nothing sealed or with one exit.
7. `e7_7_perfect_odometry_is_exact_and_noise_drifts`: `none` equals the
   simulator's pose exactly; `high` drifts and is reproducible from the seed.
8. `e7_8_env_and_controller_see_the_same_observation`: the env and a
   stand-alone `WallMemory` give the same 52 numbers on the same actions;
   `PolicyController` keeps its memory in step with the simulator.
9. `e7_9_loop_metric`: a circle loops, a line and an S-bend do not; equal to
   the 16 Nov reference implementation on random paths.
10. `e7_10_smoke_train_eval_aggregate`: two 2-iteration runs (`e3_both`-like
    and `e7_map`), their evaluation (and `--odo-noise high`; refused without
    wall memory), `val.csv` without E7 columns, and `aggregate_e7.py` with the
    baseline check.
11. `e7_11_dry_run_prints_the_windows_commands`: `scripts/run_e7.py
    --dry-run` with and without `--full`, `--no-push` and `--wallmap-every`.

## 8. Driver (`scripts/run_e7.py`)

Resumable, like `run_e3.py`. Every step is skipped when its output exists.
Defaults are the training machine's paths (`E:\policy-data`,
`E:\policy-runs`, `E:\policy-eval`).

1. Check that the inputs exist: suites, `e3_both_seed{0,1,2}/ckpt_best.pt`.
2. **Pilot**: `e7_map` and `e7_longgru` for 2 M steps each into
   `<runs>/pilot_e7/…`. Print agent-steps/s, the projected hours per full run,
   and the mean exit switches. Then stop and ask the author to start the full
   run with `--full`, unless `--full` was given.
3. Full runs: `e7_map_seed{0,1,2}`, `e7_longgru_seed0`, each with `--resume`
   when `ckpt_last.pt` exists.
4. Evaluation: the four new runs; `e3_both` and ORCA re-evaluated into
   `<eval>/e7/`; noise evaluations of `e7_map`.
5. Aggregation into `policy/results/e7/`: tables, figures (per layout, per
   bottleneck, loop rate, noise), `verdict.json` following §6.
6. Commit and push `policy/results/e7` (skipped with `--no-push`), as in E3.

`--dry-run` prints every command and does nothing else.

## 9. Out of scope

Learning the exit choice; mapping other agents; training with odometry
noise; a longer memory window for the GRU together with the map;
communication between agents; changing the reward.

## 10. Implementation notes and divergences

Implemented 16 Nov 2026. All 58 policy tests and 35 sim tests pass on the Mac.

1. **Leak guard in the coarse free mask (§1.3 step 2).** With the agent
   radius alone, a remembered wall is a blocked band 4 wide around a line of
   zero thickness. Any-pooling to 2-unit cells then leaves diagonal holes
   through a slanted wall. A coarse cell is now also shut when any of its
   fine cells is within `LEAK_GUARD = 0.6` of a remembered surface. That makes
   every remembered wall a 4-connected chain of shut cells, and keeps 5.2 gaps
   open. Explained to the author when implementation began.
2. **16-direction wavefront** (`reach = 2`) instead of E3's 32: about half the
   time per field. `geodesic_wavefront(free, D, cell, reach=3)` is the E3 code
   moved out of `geodesic_field`; with the default reach it is the same
   computation (test e7_2).
3. **Downhill by 3-step descent (§1.3 step 5)** instead of comparing the field
   at 16 points one cell away. Neighbouring cells differ by less than the
   field's own discretisation error, so the probe chose east for a wall whose
   way round is at 42°. The descent gives 45° there and −8° with an empty
   memory (bearing 0°): the policy still has the exact bearing in `[48]`.
4. **Field schedule.** A field is computed at steps 0, 4, 8, … of a scene, and
   on the first step an agent is seen; an invalid field (no source cell) is
   not retried before the next scheduled step.
5. **Exit ranking at a switch** uses straight-line distance, the simulator's
   own rule for choosing which exit the bearing points at (§1.4). The policy
   never sees a distance.
6. **Hits inside walls.** Joining two hits on a convex wall end (the capsule
   cap) draws a chord inside the wall. The test therefore checks that no
   remembered cell lies in free space; cells inside the solid are harmless.
7. **Speed on the Mac** (no network): the env alone runs at 437 agent-steps/s
   with the wall memory against 756 without (1.7× slower). Short training
   runs with 6 workers were too noisy to give a ratio (e3-like 1,040–2,224,
   `e7_map` 960–1,388 agent-steps/s in the first 5 iterations). The Windows
   pilot (§8) gives the real number.
8. **`fig_loop.png`** is drawn in `aggregate_e7.py`, not in `plots.py`
   (another branch edits `plots.py`).
9. **New output columns.** `per_scene.csv` and `metrics.json` gain
   `loop_rate`, `loop_success_rate`, `noloop_success_rate`, `exit_switches`
   and `multi_exit`, plus a `multi_exit` block per suite. `val.csv` and the
   `val` dict in `ckpt_best.pt` are unchanged. `config.json` gains
   `wallmap`, `wallmap_every` and `odo_noise`.

## 11. Exit rule "orbit" (20 Nov 2026, evaluation option)

**Why.** On the E8 `blocked_exit` scenes (the nearer exit sealed behind a slit)
`e7_map` scored 0.008: the lock (§1.4) never released. The field's target is
method A's far point at the world's edge, beyond the exit, so a sealed exit
never makes the agent's cell unreachable; the agent circles the sealed exit
until timeout. Without any lock (nearest exit at every field update) the same
checkpoint scores 0.862, so the lock itself was the failure.

**Rule (chosen by the author).** `exit_rule = "orbit"` keeps the lock and the
enclosure test and adds: per agent, accumulate the signed turning of the
direction to the target exit (estimated heading + bearing); when it reaches
`ORBIT_TURN` in one sense, the agent has gone round the exit without getting
in, so the exit is marked blocked and the next one taken (same ranking as
§1.4). Only the bearing and odometry are used. `evaluate.py --exit-rule
orbit`; the default `"enclosed"` is the trained rule, so every earlier result
is unchanged. No retraining: the rule is fixed, not learned.

**First numbers** (`e7_map_seed0`, first 40 `blocked_exit` scenes):

| rule | success | agent collisions | timeout |
|---|---|---|---|
| enclosed (trained) | 0.013 | – | 0.525 |
| orbit, 360° | 0.537 | 0.275 | 0.087 |
| orbit, 270° | 0.750 | 0.200 | 0.006 |
| orbit, 180° | 0.881 | 0.075 | 0.006 |
| no lock (reference) | 0.862 | – | 0.065 |

`ORBIT_TURN` is 180° (author's choice, 21 Nov 2026). A full turn lets four
agents circle together for long enough to collide. Per scene, 180° beats 360°
on 26 of the 40, ties on 12 and loses on 2: there agents that switched early
turn back and meet the ones still going round (`scene_00410`). Side-by-side
GIFs: `results/e7_gifs/orbit/`.

## 12. Robust statistics (21 Nov 2026)

Re-evaluated all seven checkpoints on CPU (deterministic; per-scene rows in
`results/e7_stats/per_scene/`; means agree with `results/e7/summary_e7.csv`
to within 0.5 points, float differences between machines) and analysed them
with `scripts/e7_stats.py`: mean and IQM over the seed x layout matrix,
95 % CIs from a two-level stratified bootstrap (seeds, then scenes within
layout, scene draw shared by the paired variants), probability of
improvement after Agarwal et al. 2021. Full tables: `results/e7_stats/table_stats.md`.

| e7_map vs e3_both (3 vs 3 seeds) | test4 | test_mixed |
|---|---|---|
| success, diff of means | +3.2 [+1.4, +5.1] | +3.0 [+1.0, +5.1] |
| success, diff of IQM | +3.7 [+1.6, +5.7] | +3.5 [+1.4, +5.7] |
| P(improvement), success | 0.80 | 0.77 |
| loop rate, relative reduction | 33 % [14, 48] | 35 % [16, 52] |
| wall crash, diff | -1.3 [-2.4, -0.1] | -1.3 [-2.2, -0.4] |

The success gain's CI excludes 0 on both suites. The loop reduction is
clearly above 0 but its CI straddles the 1/3 bar of §6, so that criterion is
neither met nor failed. Per layout (test4), the success gain is credible on
corridor, doorway and boxrooms; the loop reduction on crossing, cul_de_sac
and boxrooms; open, room and walls_multi show nothing (ceiling).
e7_longgru vs e3_both: success +0.7 [-1.0, +2.4], loop -5 % [-22, +9]: no
effect, so the gain comes from the map, not from a longer memory. With one
longgru seed its CI reflects scene noise only.

## 13. Safety filter after the network (21 Nov 2026)

`shield.py`, `evaluate.py --shield MARGIN` (off by default): a fixed filter
that replaces a step coming within `agent_radius + margin` of a surface
sensed this step (wall/obstacle hits, segments joining hits closer than
`min_bottleneck`, agent hits) by the nearest safe (turn, speed). Observation
and own pose only, no map, no retraining. Designed and diagnosed in a fork
session (static crashes came from wall-hugging: the progress field uses
radius R, so the shortest path touches walls). Margin 0.3, agents included.
Statistics as §12 (`results/e7_stats/shield/table_stats.md`):

| test4 (3 seeds each) | success | wall crash | timeout | loop |
|---|---|---|---|---|
| e3_both | 89.3 | 5.5 | 2.9 | 11.1 |
| e3_both + filter | 94.5 | 0.0 | 4.7 | 12.8 |
| e7_map | 92.5 | 4.2 | 1.8 | 7.4 |
| e7_map + filter | **96.4** [95.1, 97.6] | 0.0 | 3.0 | 8.4 |

| difference [95 % CI] | test4 | test_mixed |
|---|---|---|
| filter on e7_map | +3.9 [2.5, 5.3] | +4.0 [2.1, 5.8] |
| filter on e3_both | +5.3 [3.4, 6.9] | +5.5 [3.4, 7.8] |
| e7_map vs e3_both, both filtered | +1.9 [0.4, 3.4] | +1.5 [-0.6, 3.5] |
| loop, e7_map vs e3_both, both filtered | -4.4 [-6.9, -1.7] | -4.7 [-7.5, -1.8] |

The filter removes wall crashes entirely; part of the gain returns as
timeouts. With the filter, wall memory's success gain shrinks (on test_mixed
its CI includes 0) but its loop reduction stays clear: the filter fixes
crashes, the map fixes looping and dead ends.

The filter was fixed on 25 Nov 2026 (agents no longer stand still in gaps
narrower than 2 (R + margin)); e7_map + filter is now 97.2 [95.7, 98.4] on
test4. Current numbers and the training option: DESIGN_SHIELD.md.

## 14. Where agents loop (25 Nov 2026)

`scripts/loop_bearing.py`, results `results/loop_bearing/` (seed 0, no
filter). Mismatch = angle between the observed goal bearing (straight line)
and the geodesic direction to the goal.

| e7_map seed 0, test4 | steps with mismatch > 45° |
|---|---|
| inside the loop window (first 360° stretch) | 59 % |
| same looping agents, outside it | 28 % |
| agents that never loop | 14 % |

| largest mismatch an agent meets | 0–45° | 45–90° | 90–135° | 135–180° |
|---|---|---|---|---|
| e3_both loop rate, test4 | 0.9 % | 4.9 % | 20 % | 60 % |
| e7_map loop rate, test4 | 0.4 % | 3.3 % | 21 % | 41 % |
| e7_map loop rate, novel | 0 % | 5.5 % | 30 % | 63 % |

Loops happen where the straight-line bearing disagrees with the way round
the walls; the wall map lowers the worst bin (60 → 41 %) but does not remove
it. Correlation only: such places also need a detour. A causal check would
replace the bearing by the geodesic direction at evaluation.
