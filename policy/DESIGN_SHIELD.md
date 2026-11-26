# Safety filter (shield) — design

Status: evaluation filter in use since 21 Nov 2026; fixed 25 Nov 2026;
training with the filter implemented 25 Nov 2026, **runs pending compute**.

Architecture (Daniel, 21 Nov): sensing → designed features (wall memory,
height-field direction) → learned network → **fixed safety filter**.

## 1. The filter (`shield.py`)

After the network picks (turn, speed), the swept step is checked against what
the agent's rays sensed this step: wall and obstacle hit points, points on the
segments joining adjacent hits closer than `min_bottleneck`, and agent hit
points. Observation and own pose only.

| case | action |
|---|---|
| the wanted step keeps `R + margin` (margin 0.3) | unchanged |
| some candidate keeps `R + margin` | the candidate whose end point is nearest the wanted one: heading offsets 0, ±5 … ±25° (within `max_turn` = 25°) x speeds 1, 0.75, 0.5, 0.25 of the wanted speed |
| none does, some keeps `R + 0.05` (`SQUEEZE`) | that candidate (lets agents through gaps between 2R and 2(R + margin)) |
| agent already closer than `R + 0.05` | only steps that do not get closer |
| no candidate at all | turn in place by `max_turn` towards the side whose rays see more free space; keep that side until a safe step exists |

`stuck="hold"` (`evaluate.py --shield-stuck hold`) is the 21 Nov rule: no
second search, "not closer" once inside `R + margin`, and with no candidate
the network's turn at speed 0. It reproduces DESIGN_E7 §13.

### 1.1 Why the 25 Nov fix

e7_map seed 0, test4, 4000 agents, 64 timeouts under the 21 Nov rule. 34 of
them stood still for >= 20 steps because the filter found no safe step, every
time blocked by static surfaces: an agent in or in front of a gap between two
round obstacles narrower than 2(R + 0.3) but wider than 2R, where every
direction moved it closer to a ray point on one arc. The other 30 the filter
hardly touched (mostly the network's own looping).

| e7_map seed 0, test4 | success | timeouts | held by filter | collisions |
|---|---|---|---|---|
| no filter | 92.35 | | | |
| 21 Nov filter | 97.55 | 64 | 34 | 32 |
| + turn in place | 97.67 | 61 | 31 | |
| + squeeze search (25 Nov) | **98.40** | 34 | 7 | 30 |

Scene `two_doorway/scene_00460` (GIF `results/e7_gifs/shield/`), where the
filtered run loops and times out: the filter edited one step of the episode;
the agents loop without the filter as well (1139° and 1631°) and happened to
get out.

## 2. Training with the filter

`train.py --shield MARGIN [--shield-penalty P]` → `EnvConfig.shield`,
`EnvConfig.shield_penalty` (defaults -1 = off, 0).

| item | choice |
|---|---|
| where | `MultiAgentEnv.step`: network sample → `actions_to_sim` → filter → simulator; the same filter as in evaluation |
| what PPO learns from | the sampled action (its log-probability); the filter is part of the environment (shielded RL, Alshiekh et al. 2018) |
| penalty | `shield_penalty` subtracted per step the filter edits, so the network can learn not to need it |
| evaluation | a checkpoint trained with the filter is always evaluated (and validated during training) with it: `evaluate.trained_shield` |
| old runs | `shield < 0` in every older `config.json`; rollouts unchanged (`test_e7_1_off_is_unchanged`) |

Planned runs (after Thanksgiving, when compute is available): e7_map
configuration (`run_e7.py` flags, `--wallmap on`) plus `--shield 0.3`,
A: `--shield-penalty 0`, B: `--shield-penalty 0.05`, 3 seeds each, 50 M steps.
Compare against e7_map + evaluation-only filter (same checkpoints as §1)
with `scripts/e7_stats.py`.

Expected: wall crashes no longer end training episodes, so the network sees
more of the later part of an episode (loops, dead ends). Risk: without a
penalty it may rely on the filter and hug walls more.
Smoke run (CPU, 2 iterations): config, filtered rollout, validation and
evaluation with the trained filter all work.

## 3. Results of the fixed filter (3 seeds, 25 Nov 2026)

`results/e7_stats/shield_v2/table_stats.md` (method of DESIGN_E7 §12; the
21 Nov numbers stay in `results/e7_stats/shield/`).

| test4 | success | agent crash | timeout | loop |
|---|---|---|---|---|
| e3_both | 89.3 | 2.4 | 2.9 | 11.1 |
| e3_both + filter | 95.4 [94.1, 96.6] | 1.0 | 3.6 | 12.4 |
| e7_map | 92.5 | 1.4 | 1.8 | 7.4 |
| e7_map + filter | **97.2** [95.7, 98.4] | 0.6 | 2.2 | 8.3 |

Wall crashes are 0 with the filter. 21 Nov filter -> fixed filter: e7_map
96.4 -> 97.2, e3_both 94.5 -> 95.4, timeouts 3.0 -> 2.2 and 4.7 -> 3.6.

| difference [95 % CI] | test4 | test_mixed |
|---|---|---|
| filter on e7_map | +4.6 [3.0, 6.2] | +4.7 [2.7, 6.6] |
| filter on e3_both | +6.1 [4.2, 8.0] | +6.1 [4.0, 8.4] |
| e7_map vs e3_both, both filtered: success | +1.7 [-0.1, 3.6] | +1.6 [-0.4, 3.6] |
| e7_map vs e3_both, both filtered: loop | -4.1 [-6.7, -1.4] | -4.4 [-7.2, -1.6] |

With both filtered, the wall map's success gain is no longer clear (CI just
includes 0; IQM +2.1 [0.5, 3.7] on test4), its loop reduction is.
