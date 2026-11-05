# Behaviour gap: the same policy on true and on parsed scenes

Main line B, step 2 (`perception/DESIGN_INPUTS.md`) and the second half of E1
(`docs/PROPOSAL_REVISIONS.md`). Code: `gap.py`, `scripts/run_gap.py`.

## 1. Measurement

For every scene of `test_sketch7`, `_free`, `_tilt`, `_lines`:

1. perception parses the scene's image (`perception/predict.py`, the model's
   tuned thresholds from `tune_extract.json`);
2. perception scores that parse (`perception/evaluate.py`, same thresholds):
   `wall_iou`, agent errors, `scene_usable`;
3. one deterministic episode in the true scene and one in the parsed scene, same
   policy, same physics (`evaluate.run_episode`, `default_reward` of the checkpoint).

`gap = parsed - true`, per scene, for success, collision and timeout (rates over
that scene's agents). Set values are means over scenes, as in `evaluate.py`.

## 2. Failed parses

A parse fails when perception returns no scene, or the scene has no agent or no
exit. It has no parsed episode, so it is never counted as a success or as a gap.

- The gap table uses only scenes with a parse, with the true column on **those
  same scenes**, so the difference is paired.
- The failed-parse table gives the count, the true success on those scenes, and
  `end-to-end success`: parsed success with a failed parse counted as 0.

## 3. Reading the numbers

- **The parsed run happens in the parsed world.** Observations, collisions and
  exits all come from the parsed scene. A missed wall makes the run easier, so the
  gap can be positive, and it measures how much behaviour changes, not how a
  parsed plan would do in the real room.
- **Errors cancel in the signed mean.** The tables also give the mean of `|gap|`.
- **Some of the gap is not perception's fault.** Multi-agent rollouts are
  sensitive to small changes, so `|gap|` stays above 0 even for good parses. The
  `scene_usable = yes` row is the practical floor.
- **True success here is not the 89-90 % of the E2 / E3 suites.** The sketch sets
  hold layouts the policy never trained on (`rooms`, `walls`, marked `*` in the
  layout table), and 1-agent scenes (training uses 2-8).
- **`crossing` is not like for like.** True scenes give each agent its own goal; a
  drawing cannot express that, so in the parse any exit counts. Compare the per-layout row.
- **Scenes per set are not weighted.** `overall` pools all four sets, and the
  three probes over-sample hard cases. `test_sketch7` is the headline row.

## 4. Invariants

- No change to `sim/`, `perception/` or training. `perception/` and `policy/` have
  modules with the same name (`evaluate`), so perception runs only as a subprocess.
- Perception JSON becomes a `Scene` through `demo_e2e.to_scene` for both worlds.
- Physics of the set's `dataset.json` must match the checkpoint, or `gap.py` stops.
- Resume: one complete CSV row per scene, appended as it finishes. `identity.json`
  (policy hash, perception checkpoint, thresholds) refuses mixing runs in one folder.
- `parse_differs` flags scenes where `predict.py` and `evaluate.py` disagree on
  the parse. They read the same model and thresholds, so this should stay 0.
- ORCA's cells are empty when its run raises, with the message in `orca_error`.
  That never changes which scenes count as parsed or the policy's numbers.
- `identity.json` also holds the list of arms and a hash of `orca.py`, so rows from
  before these arms, or from another ORCA, cannot be mixed in.

## 5. Why not just use `orca_path`?

A reviewer can say: the pipeline already parses a full map, so run a map-based
planner on it. Two arms test that. The hoped-for answer, that the policy holds up
better when the parse has errors, is **untested**; report it whichever way it falls.

**Arm 1, parsed world.** `orca_path` runs once in the true and once in the parsed
scene, on the same scenes as the policy, with the same failed-parse handling. It is
like for like with the policy's gap, and shares the limit of §3: the whole run
happens in the parsed world.

**Arm 2, wrong map in the true world.** This is the arm that answers the question.

| | comes from |
|---|---|
| physics, agents, goals, collisions, outcomes | true scene |
| other agents as ORCA sees them | true simulator, exactly |
| ORCA's geodesic field and static half-planes | **parsed** walls and obstacles |

Implemented by one parameter, `ORCAController(..., map_scene=parsed)`; the default
`None` is the previous behaviour.

- **Comparison.** The policy's true-scene run: its rays read the real world and it
  uses no map, so a wrong map cannot affect it. `map cost` = ORCA wrong map - ORCA
  right map.
- **The policy's goal is taken from the true scene here, and so is ORCA's.** This
  arm tests map errors only, not goal or exit errors. A parsed exit in the wrong
  place would hurt a real deployment of either controller, and arm 2 does not show that.
- **Parsed agent errors are out too:** the agents are the true ones. Arm 2 therefore
  flatters neither side on agents, and says nothing about missed or invented agents.
- **Map and true goal can disagree.** The goal may lie behind a parsed wall, or the
  agent's start inside one. The field then has no finite value near the agent and
  `orca_path` follows the straight bearing, as it always does in that case. Such
  scenes stay in the table: `wrongmap_no_path_agents` counts agents with no path at
  the start and `wrongmap_no_path_steps` counts agent-steps on the fallback, and
  `gap.md` splits arm 2 by `no path at start`.
- **The policy's own gap in this arm is 0 by construction, and that is a limit.** It
  does not use the map; its cost of a bad parse in deployment is the goal error this
  arm excludes. The fair question is whether `orca_path` falls below the policy
  once its map is wrong, not whether the policy's number moves. The policy can
  still lose: `orca_path` with a wrong map may stay above it.
