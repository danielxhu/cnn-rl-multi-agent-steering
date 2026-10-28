# Policy: E3, reward redesign

Design document for experiment **E3** (Detailed Proposal §5: "reward
sensitivity … settles the reward before the policy is studied further").
It builds on `policy/DESIGN.md` (E2): everything not stated here is as there.
Written before the code; update §9 when the code diverges.

## 0. Why: what E2 showed

E2 (results in `policy/results/e2/`) trained one shared policy with the
default reward of `sim/config.py`:

```
r = −0.01 per step + 0.1·(d_prev − d)  (d = straight-line distance to the goal region)
    + 100 on reaching a goal,  −50 wall/obstacle hit,  −25 agent hit (both agents),  0 at timeout
```

The learned policies beat greedy everywhere (test4 success: greedy 0.24, mlp
0.82, gru 0.88), but the path figures and a traced crossing scene show two
failures that come from the reward, not from the network:

1. **Parking.** Standing still costs 0.01 per step and a timeout costs
   nothing, so an agent that cannot find the way stops: 260 steps of waiting
   cost 2.6, one risky attempt that ends in a wall costs 50. In a crossing
   scene whose centre is nearly blocked (bottleneck 6.2), both groups slid
   into the other group's arm and stood there until step 300. `mlp` times
   out on 8–12 % of agents.
2. **Detours are punished.** Progress is measured in straight-line
   distance, so backing out of a dead end, or walking away from the goal to
   go round a wall, is penalised at every step. The correct behaviour in
   `cul_de_sac`, `walls_multi` and a blocked `crossing` earns negative
   shaping until the agent is round the corner.

E3 changes the reward to remove each cause, one at a time and together, and
keeps the best version for E4–E6. E2 stays as the default-reward baseline.

## 1. The two changes

### 1.1 Timeout penalty

`Config.timeout_penalty` (default `0.0`, E2's behaviour). When an agent is
still active at `max_steps`, it receives `−timeout_penalty` on that step, in
addition to the step's ordinary reward, and its outcome is `"timeout"` as
before. E3 uses **25**: a timeout must cost more than waiting does today, but
less than a collision (50), or crashing early would be as good as searching.

A penalised timeout is part of the task, so in the policy environment it is
a **terminal** transition (no bootstrap from `V(final_obs)`), not a truncation:
`MultiAgentEnv` sets `terminal` instead of `truncated` for timeouts whenever
`timeout_penalty > 0`. Caveat: the observation carries no clock, so the value
of a state late in the episode is not Markov in the observation; the `gru`
policy can count steps, `mlp` cannot. E3 trains `gru` only.

### 1.2 Path-distance (geodesic) progress

`Config.progress_metric` = `"euclid"` (default, E2's behaviour) or
`"geodesic"`. With `"geodesic"`, `d` in the progress term is the length of the
shortest path an agent disk can walk from its position to its goal, around
walls and obstacles, instead of the straight-line distance. Everything else
in the reward is unchanged, and so is the observation: the policy still sees
the straight-line goal bearing only. The path distance is privileged
information used to score training transitions; nothing at test time or on a
robot needs a map.

**The field.** New in `sim/reachability.py`:

```python
def geodesic_field(cfg, scene, goal, radius=None, cell=1.0) -> np.ndarray:
    """(m, m) float32: path length (world units) from each cell centre to `goal`
    (a Region or RegionSet) for a disk of `radius` (default cfg.agent_radius);
    inf where unreachable. Grid convention as the module's (row 0 = bottom)."""
def geodesic_distance(cfg, field, p, cell=1.0) -> float:
    """Bilinear lookup of `field` at world point p; see the fallbacks below."""
```

- Free space is the configuration-space free mask of the module
  (`clearance(cfg, occupancy(cfg, scene)) >= radius`) at `cfg.cell` (0.25),
  max-pooled to `cell = 1.0` (a coarse cell is free if any of its 4 × 4 fine
  cells is). Max-pooling keeps narrow passages open (the narrowest legal
  passage, 5.2, leaves a free band of 1.2); it cannot leak through a wall
  or an obstacle, whose blocked bands are ≥ 7 wide.
- Sources: coarse free cells whose centre lies in the goal region, or in any
  member of a `RegionSet`, at distance 0.
- Propagation: 8-connected, step costs `cell` and `√2·cell`, iterated as a
  numpy wavefront (repeated `minimum` over the 8 shifted copies) until no cell
  changes. No scipy, no Python loop over cells. Target: under 30 ms per field
  on the Mac.
- Lookup: bilinear interpolation of the 4 surrounding cell centres, using only
  finite neighbours (re-normalised weights). If all 4 are infinite (the agent
  is pressed against an obstacle, inside the pooled boundary), use the
  smallest finite value in the 5 × 5 neighbourhood plus the straight-line
  distance to that cell's centre; if that is infinite too, fall back to the
  straight-line distance to the goal.

**In the simulator.** `Simulator.__init__` builds one field per distinct goal
of the scene (one for single-goal and exit-set scenes, two for `crossing`),
only when `progress_metric == "geodesic"`, keyed by the goal object.
`step()` computes `prev_d` and `d` with `geodesic_distance` instead of
`goal.distance_to`. Arrival is still `goal.contains(p)`; nothing else in
`physics.py` changes.

## 2. Variants and runs

| Variant | `timeout_penalty` | `progress_metric` | Runs |
|---|---|---|---|
| `e2_gru` (baseline, exists) | 0 | euclid | `e2_gru_seed{0..4}` |
| `e3_timeout` | 25 | euclid | `e3_timeout_seed{0,1,2}` |
| `e3_geo` | 0 | geodesic | `e3_geo_seed{0,1,2}` |
| `e3_both` | 25 | geodesic | `e3_both_seed{0,1,2}` |

All `gru`, every other setting exactly as E2 (layouts, agents 2–8, random
headings, 50 M agent-steps, PPO hyper-parameters, validation suite). Nine new
runs, about 3 h each on the training machine.

**Winner.** The variant with the highest mean **validation** success over its
seeds (ties: lower collision rate) becomes the reward for E4–E6. The test
suites only report; they never choose.

## 3. Code changes

### 3.1 `sim/` (allowed by this spec, nothing else in `sim/`)

| File | Change |
|---|---|
| `config.py` | `timeout_penalty: float = 0.0`, `progress_metric: str = "euclid"`, `geo_cell: float = 1.0` |
| `reachability.py` | `geodesic_field`, `geodesic_distance` (§1.2) |
| `physics.py` | fields at construction when geodesic; geodesic `prev_d` / `d`; timeout penalty on the timeout step |
| `README.md` | the reward section names both new fields |
| `tests/run_tests.py` | tests 1–5 of §5 |

With the defaults the simulator must behave exactly as before (§5 test 1).

### 3.2 `policy/`

| File | Change |
|---|---|
| `policyconfig.py` | `RunConfig.reward: dict` (overrides of sim `Config` reward fields, default `{}`), saved in `config.json` and the checkpoint; `sim_config(reward=None)` applies them |
| `env.py` | timeouts are `terminal` when `timeout_penalty > 0` (§1.1) |
| `train.py` | flags `--timeout-penalty`, `--progress-metric`; `--resume` keeps the checkpoint's reward |
| `evaluate.py` | evaluation always runs the default-reward physics (metrics do not depend on the reward; asserts the physics fields match the checkpoint's); writes `variant` = run name without `_seed<k>` into the identity of `metrics.json` |
| `aggregate.py` | `--group variant` groups by that identity instead of `controller`; `--variants` fixes column order; E2 behaviour unchanged by default |
| `scripts/run_e3.py` | the driver, §4 |

## 4. Driver (`scripts/run_e3.py`, resumable, same style as `run_e2.py`)

```
cd policy && python scripts/run_e3.py
```

1. check the suites and `E:\policy-eval\greedy` and the five `e2_gru_seed*`
   evaluations exist (they come from E2; stop with a clear message if not);
2. the nine runs into `E:\policy-runs`, seeds outer, variants inner, each with
   `--resume` when `ckpt_last.pt` exists;
3. evaluation of each on `test4` and `test_mixed` → `E:\policy-eval\<run>`;
4. `aggregate.py --group variant --variants greedy e2_gru e3_timeout e3_geo e3_both`
   over `E:\policy-eval` → `policy/results/e3`;
5. `git pull --rebase`, then add, commit and push `policy/results/e3`
   (`--no-push` skips; nothing to commit ends cleanly).

Flags as `run_e2.py`: `--data-root`, `--runs`, `--eval`, `--seeds`,
`--variants`, `--total-steps`, `--workers`, `--eval-workers`, `--no-push`,
`--dry-run`.

## 5. Tests

Simulator (`sim/tests/run_tests.py`):

1. `test_default_reward_unchanged` — on three fixed scenes, the summed greedy
   rewards per agent and the outcomes with the default `Config` equal values
   recorded **before** the change (record them first, write them into the test).
2. `test_geodesic_open_equals_euclid` — `open` scene, no obstacles: the field
   is within 1.5 units of the straight-line distance to the goal region
   everywhere it is finite.
3. `test_geodesic_goes_round_a_wall` — a hand-built doorway: from a point
   behind the wall, the field is within 2 units of the hand-computed path
   through the gap, and well above the straight-line distance.
4. `test_geodesic_progress_rewards_backing_out` — `cul_de_sac`-like hand
   scene: an agent inside the pocket that steps out (straight-line distance
   grows) gets positive progress with `"geodesic"`, negative with
   `"euclid"`.
5. `test_timeout_penalty_once` — `max_steps` tiny: every agent active at the
   end receives exactly `−timeout_penalty` once on top of its step reward;
   agents finished earlier receive nothing.

Policy (`policy/tests/run_tests.py`):

6. `test_timeout_terminal_with_penalty` — with the penalty the env reports
   timeouts as `terminal`, without it as `truncated` (E2 unchanged).
7. `test_reward_config_roundtrip` — `RunConfig.reward` survives
   `config.json` and the checkpoint; `--resume` restores it.
8. `test_variant_grouping` — `aggregate.py --group variant` on two fake
   evaluations of one variant and two seeds produces one column.

All existing tests keep passing.

## 6. Reporting

`policy/results/e3/` gets the E2 file set, with columns = variants:
`table_e3.md` (success and collision on `test4`), `table_e3_mixed.md`, plus a
**timeout table** (the quantity E3 targets), `fig_training_curves.png`,
`fig_per_layout.png`, `fig_bottleneck.png`, `fig_paths.png` (one scene per
layout: `e2_gru` vs the winner, seed 0), and `winner.json`
(`{"variant", "val_success_mean", "val_collision_mean", "seeds"}`).

## 7. Expected outcome (hypotheses, not acceptance criteria)

- `e3_timeout` lowers the timeout rate but raises collisions a little.
- `e3_geo` helps most on `cul_de_sac`, `walls_multi` and `crossing`.
- `e3_both` is best overall.

## 8. Instructions for the implementing agent

Read `policy/DESIGN.md` §11 first: its ground rules all apply (no git
operations, no new dependencies, Windows-safe, CPU-first, spec wins over
convenience, log every deviation). Differences for this task:

- `sim/` may be changed **only** as listed in §3.1. `perception/` and `pilot/`
  stay untouched.
- Order: record the reward values for §5 test 1 → `geodesic_field` with
  tests 2–3 → `physics.py` with tests 4–5 → `policy/` changes with tests
  6–8 → `run_e3.py --dry-run` → a 2 M-step `e3_both` CPU run on the Mac
  (4 workers) that prints training success and timeout rate next to the
  same budget with the default reward.
- Report: what was built; test output (sim and policy, all tests); the field
  timing (ms per field, Mac); the 2 M-step comparison; the divergence log;
  every file created or changed.

## 9. Divergence log

Filled in by the implementing agent: one line per deviation — what, why,
which section was updated.
