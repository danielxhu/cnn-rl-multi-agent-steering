# Frontier comparison — design (draft, 26 Nov 2026)

Status: design approved by Daniel 26 Nov 2026 (JAX allowed, everything on AutoDL,
`policy/jaxsim/`). Done: simulator port + port check (§2.1). Next: level bank,
numpy controller for evaluation, IPPO + SFL trainer, PLR for the compromise arm.

Question: does the hand-designed feature layer (E7 wall memory, height-field
direction, orbit exit rule) still pay off against (a) the same learner with a
curriculum and (b) the current research recipe — GPU simulation, raw sensing,
a recurrent policy, an automatic curriculum and ~20x the training steps?

## 1. Arms

| arm | observation | curriculum | steps | code |
|---|---|---|---|---|
| **current** | 50 numbers + wall-map downhill (52) | uniform over the 10 layouts | 50 M | existing `e7_map` checkpoints, orbit rule |
| **compromise** | 50 numbers (`e3_both`, no wall map) | Robust PLR over a level bank | 50 M | existing torch pipeline + sampler (`--plr`, off by default) |
| **frontier** | 50 numbers, nothing designed | SFL (learnability sampling, Rutherford et al. 2024, "No Regrets") over a level bank | 1 B (checkpoints at 50 M, 200 M, 500 M) | new JAX simulator + IPPO |

All arms: train on the original 10 layouts only, 3 seeds, the same safety
filter at evaluation (`shield.py`, margin 0.3, plus the `est_ema` agent
velocity of DESIGN_SHIELD §4 if adopted by then). The frontier arm's 50 M
checkpoint separates "better recipe" from "more training".

## 2. Frontier arm

Recipe after JaxNav / SFL (Rutherford et al. 2024), Gigaflow (Cusumano-Towner
et al. 2025: scale + self-play, plain policy) and Wijmans et al. 2023 (memory
agents learn maps and shortcuts at scale).

| part | choice | why |
|---|---|---|
| simulator | `policy/jaxsim/`: a port of `sim/` (rays 16 sectors x 3 classes, 220 deg, range 22; physics; per-agent termination; E3 reward) as pure JAX, `vmap` over thousands of envs | the CPU sim gives ~1.4 k agent-steps/s (e7_map: 50 M in 13 h); 1 B steps needs >= 100 x |
| levels | a bank of ~100 k scenes from the existing generator (CPU, once), padded to fixed arrays (walls, obstacles, 8 agents) | generator and reachability checks stay in numpy; no level code in JAX |
| policy | shared actor-critic, MLP 256 -> GRU 256 -> heads; same action space | recurrent memory is the standard; Memory Gym / Ni et al. 2022 favour GRU |
| algorithm | IPPO (shared parameters, per-agent advantages); MAPPO critic as a later ablation | Yu et al. 2022: IPPO/MAPPO are strong |
| curriculum | every N updates score bank levels by learnability p(1 - p) (p = success over a few rollouts), train on the top fraction mixed with uniform | SFL beat PLR/ACCEL regret proxies on JaxNav |
| evaluation | the trained weights run in the **existing** `evaluate.py` on the CPU sim via a numpy GRU controller | same simulator and suites for every arm; also checks the port |

**Port check (CPU, before any GPU time):** 500 test4 scenes, fixed random
action sequences; JAX vs `sim/` per step: positions and ray observations
within 1e-4, identical outcomes and rewards within 1e-4. And the trained
policy's success on JAX sim vs CPU sim within 1 point.

### 2.1 Port check results (26 Nov 2026, Mac CPU)

`jaxsim/port_check.py`: 500 training-distribution scenes (2-8 agents, all 10
layouts, multi-exit), e3/e7 training reward, random goal-biased actions,
15.8 k steps.

| precision | worst obs / reward / position diff | obs entries off > 1e-3 | active or outcome mismatches |
|---|---|---|---|
| float64 | 3e-8 (numpy obs is float32) / 1e-14 / 9e-14 | 0 | 0 |
| float32 | 0.90 / 1e-5 / 1e-4 | 2 of 4.2 M (tangent rays) | 0 |

Speed of the simulator alone (no network), float32, Mac CPU: 35 k
env-steps/s at 2048 envs, about 175 k agent-steps/s, ~130 x the current
training loop. GPU numbers come from the AutoDL pilot.

Not yet covered: long episodes that time out (random actions end in ~32
steps); to be checked with a trained policy once the numpy controller exists.

## 3. Compromise arm

`train.py --plr` (default off; `--resume` works): Robust PLR (Jiang et al.
2021) over the same level bank; score = positive value loss; replay
probability 0.5, staleness 0.3, rank temperature 0.3. `e3_both` flags
otherwise unchanged. CPU test: the sampler's distribution and that `--plr`
off reproduces the old rollout stream.

## 4. Comparison

Suites: test4, test_mixed, novel (incl. u_trap, blocked_exit), crowd4–16.
Metrics: success, wall / agent crash, timeout, loop rate, path-length ratio.
Statistics: `scripts/e7_stats.py` (two-level stratified bootstrap, IQM,
probability of improvement). Also wall-clock and GPU-hours per arm.

Readings:
- frontier >= current at 50 M: the feature layer is not needed;
- frontier catches up only at 500 M–1 B: features substitute for scale (a
  result in itself for compute-limited robots);
- compromise ~ current: the curriculum, not the features, does the work.

## 5. Compute and where

Everything runs on AutoDL (Daniel, 26 Nov): one Linux GPU instance image with
torch and CUDA JAX, the repository cloned from git.

| work | where |
|---|---|
| JAX port, port check, numpy controller, PLR code, CPU tests | Mac (JAX on CPU), then re-run on AutoDL |
| compromise arm (torch; the CPU simulator is the bottleneck, so pick an instance with many CPU cores) | AutoDL, 3 seeds in parallel, ~13 h each |
| frontier pilot and runs | AutoDL GPU; pilot first to measure steps/s, then size the runs |
| evaluation of all three arms (incl. the existing e7_map checkpoints) | AutoDL, same instance type, so wall-clock numbers compare |

The GTX 1070 box is not needed for this comparison.

## 6. Dependencies

`jax` (and `optax`) only for `policy/jaxsim/`, in a separate
`requirements-jax.txt`; the rest of the repository keeps its dependencies.

## 7. Risks

- The port drifts from `sim/` (geometry edge cases): the port check gates
  everything.
- 1 B steps may plateau below the current arm on novel layouts; that is a
  valid result, report it.
- 3 seeds give wide CIs (as before).
