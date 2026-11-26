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

## 6. Results (25 Nov 2026)

All runs: `results/zeroshot/runs/` (3 seeds per checkpoint variant, ORCA
1 run); tables with CIs and P(improvement): `results/zeroshot/table_stats.md`
(`scripts/run_zeroshot.py --stats`). GIFs of typical failures:
`results/zeroshot/gifs/`. `full` = e7_map + orbit exit rule + filter 0.3.
`orca_path` is given the map (privileged); `orca_straight` is not.

Success, mean of seeds (%):

| model | test4 | novel | crowd4 | crowd8 | crowd12 | crowd16 |
|---|---|---|---|---|---|---|
| e3_both | 89.3 | 72.4 | 90.9 | 88.1 | 85.3 | 83.6 |
| e7_map | 92.5 | 64.9 | 92.8 | 91.1 | 89.4 | 88.4 |
| e7_map + orbit | 92.6 | 82.6 | 92.9 | 91.1 | 89.5 | 88.4 |
| e3_both + filter | 95.4 | 86.1 | 95.8 | 95.2 | 94.2 | 93.0 |
| **full** | **97.2** | **91.6** [88.4, 94.5] | **97.3** | **96.5** | **95.7** | **94.7** |
| orca_path | 92.7 | 81.4 | 92.8 | 86.4 | 81.4 | 77.3 |
| orca_straight | 70.5 | 11.2 | 69.4 | 67.3 | 65.7 | 64.7 |

Novel, per family (success %):

| model | blocked_exit | l_turn | maze | switchback | u_trap |
|---|---|---|---|---|---|
| e3_both | 82.0 | 72.2 | 50.5 | 62.4 | 94.7 |
| e7_map | 0.8 | 84.7 | 67.7 | 87.0 | 84.5 |
| e7_map + orbit | 88.9 | 84.7 | 67.7 | 87.0 | 84.5 |
| full | 94.8 | 93.1 | 85.8 | 95.3 | 88.9 |
| orca_path | 92.5 | 79.2 | 86.8 | 64.8 | 83.8 |

### Answers

1. **Drop test4 → novel.** full −5.6 points (97.2 → 91.6); e3_both −16.9;
   e7_map −27.6, all of it from blocked_exit (without it, e7_map's four other
   families average 81.0). Hardest family: maze (full 85.8).
2. **e7_map vs e3_both on novel.** Without the exit rule e7_map is worse
   overall (−7.4 [−12.0, −2.7]) because of blocked_exit (−81). Per family it
   is better on l_turn (+12.5), maze (+17.2), switchback (+24.6), all CIs
   exclude 0, and worse on u_trap (−10.2 [−21.2, −1.7]: more loops and
   timeouts, GIF `u_trap/`). full vs e3_both: +19.3 [14.0, 24.5].
   On crowd suites e7_map > e3_both by +2.0 to +4.8.
3. **Crowd size.** No cliff up to 16 agents. Change from crowd4 [95 % CI]:

   | model | 8 | 12 | 16 |
   |---|---|---|---|
   | e3_both | −2.8 [−4.4, −1.2] | −5.6 [−7.6, −3.8] | −7.3 [−9.6, −5.0] |
   | e7_map | −1.8 [−3.2, −0.3] | −3.5 [−4.9, −2.1] | −4.5 [−5.9, −2.9] |
   | full | −0.8 [−1.7, −0.0] | −1.5 [−2.7, −0.5] | −2.6 [−3.6, −1.6] |
   | orca_path | −6.4 [−8.0, −4.8] | −11.4 [−13.0, −9.8] | −15.5 [−17.1, −13.9] |

   The fall is already significant at 8 agents for every model but small for
   full (< 1 point); it grows about linearly. Agent–agent crashes for full:
   1.0 % (4) → 2.2 % (16), mostly crossing (GIF `crowd16/`).
4. **Exit switch.** The `enclosed` rule never fires on blocked_exit
   (0 switches; e7_map loops at the blocked exit, 85 % loop rate). The
   `orbit` rule fires there (0.91 switches per agent) and nowhere else in
   novel, and lifts blocked_exit 0.8 → 88.9 % (full 94.8). On test4 and the
   crowd suites it changes nothing (+0.0 to +0.1).

Other metrics (test4, e7_map vs e3_both): stuck −5.4 [−7.1, −3.7], path
ratio 1.30 vs 1.43, throughput +0.84 per 100 steps. orca_path (with the map)
has path ratio 1.04 and the highest throughput but the most agent crashes.

Remaining failures (GIFs `maze/`): agents loop at the start where the
bearing points through a wall (see DESIGN_E7 §14), and under the filter the
crashes left are agent–agent.
