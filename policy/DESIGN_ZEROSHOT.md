# Zero-shot generalisation (no retraining) — design

Status: **draft, awaiting approval** (21 Nov 2026). Evaluation only, CPU only.

Question: how far do the trained policies carry beyond their training
distribution — new layout families, larger crowds, an exit that is walled
off — and does wall memory (E7) still help there?

## 1. What already exists (E8, 18–20 Nov)

- `sim/layouts.py`: five test-only families in `NOVEL_LAYOUTS` (`l_turn`,
  `switchback`, `maze`, `u_trap`, `blocked_exit`), outside `LAYOUTS` and
  `LAYOUTS_E2`; reachability and bottleneck checks as for every family;
  `LayoutSpec.dead_exits` lets `blocked_exit` keep one exit unreachable.
  CPU test `test_novel_layouts_test_only`; the reference hashes of the old
  suites are unchanged (36/36).
- `scripts/make_novel_suites.py`: `novel` (5 x 100 scenes, 4 agents) and
  `crowd4/8/12/16` (10 training families x 30 scenes).
- First numbers, GIFs (`results/e8_gifs/`), the `blocked_exit` diagnosis and
  the `orbit` exit rule (DESIGN_E7 §11, 180°).
- `scripts/e7_stats.py`: IQM / two-level bootstrap / probability of improvement.

## 2. Design

| item | choice |
|---|---|
| layout families | the five above; T-junction and office are not added (T is close to `crossing`, office to `rooms_multi`) |
| crowd sizes | 4 / 8 / 12 / 16 agents on the 10 training families; the scene's actual count is recorded (`n_agents`) and reported per size |
| blocked exit | `blocked_exit`: the exit nearest (straight line) to the start box is walled in; at least one exit is reachable |
| scenes | 100 per layout per suite, fixed seeds: `novel` seed 10000 (unchanged), `crowd{N}` seed 11000 + N, now 100 instead of 30 per layout |
| baseline in-distribution | `test4` (existing, 1000 scenes) |
| models | `e3_both_seed{0,1,2}`, `e7_map_seed{0,1,2}` (trained exit rule), `e7_map` + `--exit-rule orbit`, `orca_straight`, `orca_path` (chain walls) |
| no agent cap | `PolicyController` and `run_episode` loop over `n`; checked by a CPU test with 16 agents |
| new metrics | below, §3 |
| statistics | `e7_stats.py` method; strata = layouts (novel) or families (crowd); trained-vs-novel drop uses independent scene draws (different scenes) |
| GIFs | `scripts/e7_gifs.py --rank` on the novel families and crowd16, side by side e3_both / e7_map |
| output | `results/zeroshot/` (per-scene CSVs, `stats.json`, `table_zeroshot.md`, GIFs) |
| code | new `scripts/zeroshot_eval.py`; `evaluate.py`, `sim/` and the old suites unchanged |

## 3. New metrics (per agent, then mean per scene)

After SteerBench (Singh et al. 2009) and the deadlock measures of Chandra et
al. 2023. Progress is measured along the geodesic distance to the goal
(`sim.reachability.geodesic_field`, agent-radius clearance), so a detour
round a wall is not mistaken for no progress.

| metric | definition |
|---|---|
| stuck rate | share of agents that at some point gain less than `max_speed` (2.5) of geodesic progress over 30 consecutive steps while active |
| pause time | mean number of active steps with speed < 10 % of `max_speed` |
| throughput | agents reaching the goal per 100 steps, from step 0 to the last arrival (scenes with no arrival: 0) |
| path ratio | path length / geodesic distance start→goal, successful agents only |

Throughput is measured at the goal, not at the narrowest gap: the gap has
no stored location, and every agent that arrives has crossed it.

## 4. Questions the report answers

1. Drop from `test4` to `novel`, per model and per family (with CIs).
2. Does `e7_map` still beat `e3_both` on `novel`? (difference, CI, P(improvement))
3. From which crowd size does success fall clearly (CI of the change from 4 agents excludes 0)?
4. Does the exit switch fire on `blocked_exit` (`exit_switches`), and does
   the `orbit` rule help there?

## 5. Cost

Evaluation only. Estimated 1–3 h on the Mac (10 cores) for all models and
suites, crowd16 being the slowest.
