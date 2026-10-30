# Policy: ORCA baseline

Design document for a classical multi-agent collision-avoidance baseline,
ORCA (Optimal Reciprocal Collision Avoidance; van den Berg, Guy, Lin and
Manocha, "Reciprocal n-body collision avoidance", ISRR 2011), evaluated
next to greedy and the learned policies on the E2 suites. It builds on
`policy/DESIGN.md` (E2) and `policy/DESIGN_E3.md` (E3); everything not stated
here is as there. Update §8 when the code diverges.

## 0. Why

Greedy shows the learned policies do *something*; it does not show they are
competitive. ORCA is the standard hand-designed method for many agents
sharing space, and the usual baseline in learned multi-robot navigation
papers. It needs no training, so it costs minutes of evaluation.

What the comparison can and cannot say:

| | learned policy (`mlp`, `gru`) | ORCA |
|---|---|---|
| origin | learned from reward | geometric rule, no training |
| information | 50 numbers: 48 ray distances, goal bearing, own speed | **exact** positions and velocities of every agent within 22, all directions; exact obstacle and wall geometry nearby |
| global route | none (bearing only) | `orca_straight`: none (bearing only); `orca_path`: the geodesic field of E3 (a map) |
| kinematics | unicycle, turn ≤ 25° per step | plans a free 2-D velocity, then converted to (turn, speed) (§2.4) |

ORCA therefore has *more* information than the learned policy. A learned
policy that matches it does so with less; one that loses to it on agent–agent
avoidance shows what the ray observation costs.

## 1. Two variants

| Variant | Preferred velocity |
|---|---|
| `orca_straight` | full speed toward the nearest point of the goal (the straight-line bearing the policy sees) |
| `orca_path` | full speed along the shortest path: the direction, among 32 evenly spaced ones, whose point at distance `max_speed` has the lowest `geodesic_distance` (`sim/reachability.py`, E3); the straight bearing when that point is inside the goal |

`orca_straight` is the fair comparison (same goal information). `orca_path`
is how ORCA is used in practice, with a global planner on a map: the
reference for "what a map buys".

## 2. The controller (`policy/orca.py`, numpy only, no new dependency)

Implemented from the paper (and the publicly documented RVO2 algorithm); no
third-party code copied. Units: world units and steps (`dt = 1`).

### 2.1 Per step, for every active agent i

1. **Velocity now**: `v_i = speed_i · (cos h_i, sin h_i)` from `sim.speed`,
   `sim.heading`.
2. **Preferred velocity** `v_pref` (§1).
3. **Neighbours**: other *active* agents with centre within `neighbor_dist =
   fov_range` (22), any direction, nearest `max_neighbors = 10` first.
4. **Static neighbours**: obstacles (circles) and wall capsules whose surface
   is within `neighbor_dist`.
5. **Half-planes** (§2.2), then the **linear program** (§2.3): the velocity
   closest to `v_pref` inside every half-plane and within `|v| ≤ max_speed`.
6. **Action** from the chosen velocity (§2.4).

All agents are computed from the same state, then stepped together, as the
simulator requires.

### 2.2 Half-planes

Agent–agent (reciprocal, time horizon `tau = 5` steps): combined radius
`R = 2·agent_radius + margin` (`margin = 0.3`); relative position
`p = p_j − p_i`, relative velocity `v = v_i − v_j`; the velocity obstacle is
the truncated cone of `p`, `R`, `tau`; `u` is the smallest change of `v` that
leaves it, `n` its outward normal; the half-plane is
`{x : (x − (v_i + u/2)) · n ≥ 0}`. Already overlapping (`|p| < R`): use the
cone of horizon 1 step, as in the paper.

Static circles (obstacles; wall capsules approximated by a circle of radius
`wall_half` at the nearest point of the wall segment; horizon `tau_obs = 3`):
the same construction with `v_j = 0` and full responsibility:
`{x : (x − (v_i + u)) · n ≥ 0}`, radius `agent_radius + r + margin`. The
capsule approximation is the documented simplification of this baseline:
one disk per wall, re-taken every step, at the nearest point.

### 2.3 Linear program

The 2-D incremental randomized LP with a circular speed limit (the paper's
§5, as in RVO2's `linearProgram1/2`): constraints are added one at a time; when
the current optimum violates the next one, re-optimise on that constraint's
line. If the program is infeasible (dense crowds), the 3-D fallback
(`linearProgram3`): keep the static-obstacle half-planes hard, and minimise
the largest violation of the agent half-planes. Process constraints in a
fixed order for determinism (static first, then agents by distance).

### 2.4 From a free velocity to (turn, speed)

The simulator's agents turn at most `max_turn` (25°) per step and cannot move
sideways. With `θ` the direction of the chosen velocity `v*` and `e =
wrap(θ − h_i)`:

```
turn  = clip(e / max_turn, −1, 1)
speed = clip(|v*| · max(0, cos e) / max_speed, 0, 1)     # slow down while turning; stop when facing away
```

Actions are passed in the policy's action units through the same mapping the
evaluation uses for greedy (`(turn, speed)` straight to `Simulator.step`).

### 2.5 Interface

```python
class ORCAController:                           # evaluate.py's controller protocol
    name = "orca_straight" | "orca_path"
    def __init__(self, mode="straight", tau=5.0, tau_obs=3.0, margin=0.3, neighbor_dist=None, max_neighbors=10)
    def reset(self, sim)                        # caches geometry; for "path" one geodesic field per goal
    def act(self, sim, obs51) -> np.ndarray     # (n, 2) actions for all agents (inactive rows ignored)
def orca_velocity(p, v, v_pref, neighbors, statics, cfg, params) -> np.ndarray   # one agent, pure function
def linear_program(lines, radius, v_pref) -> np.ndarray                          # §2.3, pure function
```

## 3. Evaluation and reporting

- `evaluate.py --orca straight|path`, like `--greedy`: identity
  `{"controller": "orca", "variant": "orca_straight" | "orca_path", "seed":
  None}`, deterministic, same suites, same metrics, default-reward physics.
- `aggregate.py`: the special treatment of `greedy` (one run, no training
  curve, no seed std) applies to every **non-learned** variant:
  `NON_LEARNED = {"greedy", "orca_straight", "orca_path"}`. Default E2 column
  order becomes `greedy orca_straight orca_path mlp gru` when ORCA evaluations
  exist; `--variants` still fixes the order explicitly.
- `fig_paths.png` keeps its three controllers by default; `--path-variants`
  chooses them (e.g. `orca_straight mlp gru`).
- `scripts/run_orca.py` (resumable): evaluate both variants on `test4` and
  `test_mixed` into `E:\policy-eval\orca_straight` / `orca_path` (skip if
  present); re-aggregate E2 into `policy/results/e2` with the ORCA columns
  (`--variants greedy orca_straight orca_path mlp gru --group controller`
  semantics of E2 unchanged otherwise); if E3 evaluations exist, re-aggregate
  E3 with `--variants greedy orca_straight orca_path e2_gru e3_timeout e3_geo
  e3_both`; `git pull --rebase`, commit, push `policy/results`. Flags as the
  other drivers (`--data-root`, `--eval`, `--runs`, `--eval-workers`,
  `--no-push`, `--dry-run`).

## 4. Expected outcome (hypotheses)

- `orca_straight`: near-perfect on `open`; few agent–agent collisions; stuck
  (timeouts, not collisions) in `cul_de_sac`, `doorway`, `two_doorway`,
  `walls_multi`: it has no route, like greedy, but it does not crash.
- `orca_path`: high success on the static layouts; some wall collisions
  where the unicycle cannot follow ORCA's sideways velocity in a narrow gap.
- The learned `gru` beats `orca_straight` overall (it can find routes) and
  is beaten by `orca_path` on the static layouts (a map beats no map); the
  interesting comparison is `crossing` and `test_mixed` with 8 agents.

## 5. Tests (`policy/tests/run_tests.py`, CPU)

1. `test_orca_lp_matches_grid_search` — random feasible sets of 1–8
   half-planes and a speed disk: the LP's answer satisfies every constraint
   (1e-6) and is within 0.02 of the nearest feasible point found by a dense
   grid search; for infeasible sets the 3-D fallback's largest violation is
   within 0.02 of the grid search's best.
2. `test_orca_head_on_swap` — two agents facing each other 20 apart in an
   empty world, goals behind each other: both reach their goals, no
   collision, using `ORCAController("straight")` in the real simulator.
3. `test_orca_four_way_swap` — four agents at the compass points swapping:
   all reach their goals, no collision.
4. `test_orca_avoids_static_circle` — one agent, a single obstacle between it
   and the goal: reaches the goal, no collision.
5. `test_orca_path_heads_for_the_gap` — hand doorway scene, agent behind the
   wall off the gap's axis: `orca_path`'s preferred velocity points to the
   gap side (straight bearing would point at the wall).
6. `test_orca_action_conversion` — heading error 0 → full speed, no turn;
   error 120° → speed 0, full turn toward the velocity.
7. `test_orca_deterministic` — two runs of one scene give identical
   trajectories.
8. `test_orca_eval_and_aggregate` — `evaluate.py --orca straight` on a
   10-scene suite writes the §5.3 files of DESIGN.md; `aggregate.py` with it
   and greedy gives an `orca_straight` column without a std.

## 6. Instructions for the implementing agent

Read `policy/DESIGN.md` §11; its ground rules apply (no git operations, no new
dependencies, Windows-safe, CPU-first, spec wins, log every deviation). Do
not modify `sim/`, `perception/` or `pilot/`. In `policy/`, change only
`orca.py` (new), `evaluate.py`, `aggregate.py`, `plots.py`,
`scripts/run_orca.py` (new), `tests/run_tests.py`, `README.md`, and this file.

Order: `linear_program` with test 1 → `orca_velocity` and the controller with
tests 2–4, 6–7 → path mode with test 5 → evaluate / aggregate with test 8 →
`run_orca.py --dry-run` → a local run on the 200-scene `val` suite (build it
with `scripts/make_suites.py --only val`) for greedy, `orca_straight` and
`orca_path`, printing the per-layout table.

Report: what was built; all test output (policy and sim suites); the
per-layout `val` table of the three controllers; ORCA's speed (agent-steps
per second, one process); the divergence log; every file created or changed.

## 7. Open items

- ORCA sees all agents within 22 in every direction; the learned policy sees
  a 220° field of view through 16 rays. A 220° ORCA variant is possible later
  (`fov_only=True`) if a reviewer asks; not part of this iteration.
- Collided agents vanish (simulator semantics), as for every controller.

## 8. Divergence log

Filled in by the implementing agent: one line per deviation — what, why,
which section was updated.
