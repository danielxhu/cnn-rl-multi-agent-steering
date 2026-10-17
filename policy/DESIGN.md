# Policy: E2 baseline steering policy

Design document for the policy stage (pipeline stages 5–7 in the README:
ego-centric observation → shared policy → synchronous simulation) and for
experiment **E2, baseline policy** (Detailed Proposal §5). Written before the
code; the code should follow it, and this file should be updated when the
code diverges (§11.7).

Scope of this iteration:

| In | Out (later stages) |
|---|---|
| Multi-agent RL environment around `sim/physics.Simulator` | reward sweeps (E3), layout hold-out (E4), crowd-size sweep (E5) |
| One shared policy, two memory variants (none, GRU), own PPO | comparison with discovered models and distillation (E6) |
| Fixed evaluation suites, greedy baseline, per-layout metrics | image → parse → policy end to end, pipeline-vs-oracle gap |
| One resumable driver for the training machine | global planner / waypoints, neighbour communication |
| CPU unit tests | |

Decisions already taken (discussion with the author, 2026-10-16/17):

- **Training layouts: all ten families** — the seven of the proposal
  (`open, corridor, room, doorway, two_doorway, cul_de_sac, crossing`) plus
  the three multi-exit families of sketch v5 (`boxrooms, walls_multi,
  rooms_multi`, `layouts.MULTI_LAYOUTS`), so that scenes parsed from the
  author's drawings are inside the training distribution.
- **2–8 agents per scene**, drawn per scene. The proposal's 4-agent
  condition is reported as its own evaluation suite.
- **Initial headings fully random** (uniform on [−π, π)) in training and in
  every evaluation suite. The generator spawns agents facing their goal
  ±26° (`layouts._place_agents`), and in the seven original families the goal
  lies within ±45° of +x for 100 % of agents (crossing: 50 %), so a policy
  could learn "keep going straight" instead of navigating. Hand-drawn agents
  point anywhere.
- **Goal signal: bearing only, no distance**, matching Le & Hu (2022) §3.2,
  where agents get the angle towards a pre-set area each step but no
  distance. The observation is therefore **50 numbers**, not the 51 of
  `Simulator.observe` (§1.3). Goal distance stays available as a switch.
- **Continuous actions** (turn, speed), Gaussian policy.
- **Own PPO in plain PyTorch**, not Stable-Baselines3: SB3 is single-agent;
  agents that finish early would have to be padded with fake transitions.
- **Two memory variants**, five seeds each: `mlp` (no memory; the same
  information as the discovered symbolic models, which keep no history —
  the fair E6 counterpart) and `gru` (recurrent; the only variant that can in
  principle remember having been inside a dead end). A frame-stack mode
  (`stack`, last k observations) is implemented but not part of E2.
- **Observation and action noise**: implemented as switches, **off** by
  default and off in E2.
- **Collision ends the agent's episode** (`Config.terminate_on_collision =
  True`). The count-only mode of the simulator lets agents pass through
  walls, so it must not be used for training.
- **Reward unchanged** from `sim/config.py` (E3 varies it later).
- **Training machine**: the Windows box (i7-8700K, 6 cores / 12 threads,
  16 GB, GTX 1070). Simulation is the bottleneck, so rollouts run in CPU
  worker processes; the learner may use the GPU.
- Code lives in `policy/`, imports `sim/` through a `sys.path` shim; `sim/`,
  `perception/` and `pilot/` are not modified.

---

## 1. Environment (`env.py`)

### 1.1 What the simulator already provides

`sim/physics.Simulator(scene, cfg)` steps every agent of one `Scene`
synchronously. Facts the design depends on (read `physics.py`):

| Fact | Where |
|---|---|
| `reset()` / `step(actions)` return `obs` of shape `(n, 51)` for **all** `n` agents, inactive ones included | `observe()` |
| `actions` is `(n, 2)`: turn ∈ [−1, 1] × `max_turn` (25°), speed ∈ [0, 1] × `max_speed` (2.5) | `step()` |
| reward `−0.01 + 0.1·Δd`, `+100` on entering the goal, `−50` static hit, `−25` agent hit; 0 for inactive agents | `step()` |
| `active[i]` turns False on goal, collision or timeout (`max_steps = 300`); `outcome[i]` ∈ {goal, collision, timeout} | `step()` |
| inactive agents vanish: they are not sensed and cannot be hit | `observe_one`, `_hits_agent` |
| the goal of an agent is `scene.goal_for(agent)`: its group's region, or a `RegionSet` of all exits (reach any) | `scene.py` |
| `stats()` gives success / collision / timeout rates, `collision_violations`, `finish_time`, `remaining_agents`, `mean_path_length` | `stats()` |
| `greedy_action(sim, i)` is the null model | bottom of file |

Observation layout (51): `[0:16]` obstacle, `[16:32]` wall, `[32:48]` agent
sector distances (normalised by `fov_range`), `[48]` goal distance,
`[49]` goal bearing / π, `[50]` speed / `max_speed`.

### 1.2 Scene sampling

`random_scene(rng, cfg, layout, n_agents, n_obstacles)` from `sim/layouts.py`,
then the heading override of §1.4. Per scene: layout uniform over the ten
families (`EnvConfig.layouts`), `n_agents` uniform in `agents = (2, 8)`,
`n_obstacles` uniform in `obstacles = (0, 6)`, `hard=False`, `close=False`.
`random_scene` returns `None` when validation fails: draw again. Sampling
costs ≈ 17 ms per scene on one core (measured), an episode is up to 300
steps, so sampling on the fly inside the workers is cheap; no pre-generated
training set.

### 1.3 Observation (`obs_dim = 50`)

`obs50 = np.delete(obs51, 48, axis=-1)`: 48 sector distances, goal bearing,
own speed. `EnvConfig.goal_obs = "bearing"` (default) or
`"bearing_distance"` (the full 51). With `memory = "stack"` the env
concatenates the last `stack_k` observations of each agent, oldest first,
zero-padded at the start of an episode (`obs_dim = 50·k`).

Optional noise, applied to the policy's copy only (never to the simulator's
state), all default 0:

| Field | Effect |
|---|---|
| `obs_noise` | Gaussian, std in normalised units, added to the 48 sector values, clipped to [0, 1] |
| `bearing_noise_deg` | Gaussian on the bearing, in degrees, wrapped |
| `action_noise` | Gaussian added to (turn, speed) after the policy, before clipping |

### 1.4 Initial headings

`EnvConfig.heading = "random"` (default): after sampling, replace every
agent's heading by `rng.uniform(−π, π)` (`dataclasses.replace` on the
`Agent`s, a new `Scene`; positions and everything else untouched).
`"generator"` keeps the generator's goal-facing headings.

### 1.5 Slots, masks and resets

One env holds one scene with up to `max_agents = 8` **slots**. Arrays
returned by `MultiAgentEnv.step` are always shaped `(max_agents, …)`:

| Array | Meaning |
|---|---|
| `obs` `(8, obs_dim)` float32 | observation of every slot (zeros for empty slots) |
| `alive` `(8,)` bool | the slot holds an agent that acts **this** step |
| `reward` `(8,)` float32 | reward of the transition just taken |
| `terminal` `(8,)` bool | the agent's episode ended by goal or collision on this transition |
| `truncated` `(8,)` bool | it ended by timeout on this transition |
| `first` `(8,)` bool | the slot starts a new episode at this observation (recurrent state must be zeroed) |

An agent contributes a transition only on steps where it was alive before
the step. When every agent is inactive (`sim.done`), the env samples a new
scene in place, sets `first` for every filled slot and returns the new
observation in the same call (auto-reset). Slots beyond `n_agents` stay
empty (`alive = False`) for the whole scene.

Truncation: when an agent times out, the env also returns
`final_obs` `(8, obs_dim)`, the observation it would have seen next, so the
learner can bootstrap from `V(final_obs)` (§3.3). Goal and collision are
true terminals.

### 1.6 Actions

The policy outputs an unbounded Gaussian sample `a ∈ ℝ²`. The env maps
`turn = clip(a₀, −1, 1)`, `speed = clip((a₁ + 1) / 2, 0, 1)` (as in the
pilot). Log-probabilities are of the unclipped sample.

### 1.7 Seeding

Each env owns `np.random.default_rng(SeedSequence([run_seed, worker, env]))`.
Two runs with the same seed, worker count and env count produce the same
scenes and, on CPU, the same trajectories.

## 2. Networks (`nets.py`)

Separate actor and critic, same body shape, no shared weights (robust
default for PPO with continuous actions):

| `memory` | body (actor and critic each) | input |
|---|---|---|
| `mlp` | Linear(obs_dim, 128) – tanh – Linear(128, 128) – tanh | 50 |
| `stack` | same as `mlp` | 50·k |
| `gru` | Linear(obs_dim, 128) – tanh – GRU(128, 128) | 50 |

Heads: actor `Linear(128, 2)` → mean; a state-independent `log_std`
parameter, initialised to −0.5. Critic `Linear(128, 1)`. Orthogonal
initialisation (gain √2 for hidden layers, 0.01 actor head, 1 critic head).
The GRU version keeps one hidden state for the actor and one for the
critic, each `(1, n, 128)`, zeroed where `first` is set.

One interface for all three modes, so the update code has a single path:

```python
class ActorCritic(nn.Module):
    def initial_state(self, n: int) -> tuple[Tensor, Tensor]   # zeros; shape (1, n, 128) or (1, n, 0) for mlp/stack
    def step(self, obs, state, first) -> tuple[dist, value, state]          # one time step, obs (n, D)
    def sequence(self, obs, state0, first) -> tuple[dist, value]            # obs (L, n, D), first (L, n): resets inside
```

`sequence` must give the same outputs as calling `step` L times with the
same resets (§8, test 6). About 25 k parameters for `mlp`, 0.2 M for `gru`;
the size is not a variable of E2.

## 3. PPO (`rollout.py`, `ppo.py`)

### 3.1 Rollout collection

`W` worker processes (default 10, `multiprocessing` with the `spawn`
context), each stepping `E` envs (default 8) with its own CPU copy of the
actor and critic (`torch.set_num_threads(1)`). Per iteration:

1. the learner sends the current weights (state-dict as numpy arrays);
   rewards come back raw and are scaled in the learner (§3.2);
2. each worker runs `T = 128` steps on its `E` envs, carrying env state and
   recurrent state across iterations, and returns, per (env, slot, step):
   `obs, action, logp, value, reward, alive, terminal, truncated, first,
   bootstrap_value` plus the recurrent state at the start of every
   sequence chunk (§3.4) and per-episode outcome records for logging;
3. `W = 0` runs the same code inline in the learner process (tests, Mac,
   debugging), as `inline_workers` (default 1) `RolloutWorker`s one after
   another; for the same seeds it produces the same batch as a pool of that
   many workers.

One iteration therefore holds `W·E·T·8` slot-steps, of which the alive ones
(≈ 40 k with the defaults) are agent transitions. **Budget and progress are
counted in alive agent transitions** ("agent-steps").

### 3.2 Reward scaling

Rewards span −50 … +100. The learner divides rewards by the running standard
deviation of the per-slot discounted return (`ret ← γ·ret·(1 − done) + r`,
as in CleanRL's `NormalizeReward`), updated once per iteration from that
iteration's rewards in time order. The statistics and the per-slot `ret`
carry-over are part of the checkpoint. Reported returns and all metrics use
raw rewards.

### 3.3 Advantages

GAE per slot over the time axis, γ = 0.99, λ = 0.95, on scaled rewards:

```
δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) + γ·B_t·truncated_t − V(s_t)
done_t = terminal_t or truncated_t;   B_t = V(final_obs_t)  (0 unless truncated)
A_t   = δ_t + γ·λ·(1 − done_t)·A_{t+1}
```

Steps where the slot is not alive carry no transition: they are skipped in
the recursion (the next alive step of the same slot is "t+1" only within the
same episode; a `first` step starts a new recursion). `V(s_T)` after the last
step of the iteration comes from one extra critic call in the worker.
Returns = A + V. Advantages are normalised per minibatch over alive
entries.

### 3.4 Update

Data is cut into **sequences of length L = 32** along time for every
(worker, env, slot), each with its stored initial recurrent state. All three
memory modes use the same sequence batches (the `mlp` net simply ignores
the state). Each epoch shuffles sequences and splits them into
`n_minibatches = 4`. Loss, masked to alive entries:

```
L = −min(ρA, clip(ρ, 1 ± 0.2)·A) + 0.5·(V − R)² − c_ent·H
```

| Hyperparameter | Value |
|---|---|
| learning rate | 3e-4, linear decay to 0 over the budget |
| optimiser | Adam, eps 1e-5 |
| epochs per iteration | 4 |
| minibatches | 4 |
| clip ε | 0.2 (value loss not clipped) |
| entropy coefficient | 0.0 |
| value coefficient | 0.5 |
| max grad norm | 0.5 |
| γ / λ | 0.99 / 0.95 |
| T / L | 128 / 32 |
| W / E | 10 / 8 |
| budget | 50 M agent-steps (§7 adjusts after the dev run) |

Every value is a `PPOConfig` field and a CLI flag. Before the first update of
an iteration, re-evaluating the batch with `sequence` must reproduce the
stored `logp` and `value` (§8, test 7): this is the check that catches almost
every recurrent-PPO bug.

### 3.5 Logging

`log.csv`, one row per iteration: `iteration, agent_steps, seconds,
agent_steps_per_s, episodes, success_rate, collision_rate, timeout_rate,
mean_return, policy_loss, value_loss, entropy, approx_kl, clip_frac,
explained_variance, lr, log_std_turn, log_std_speed` (training-episode
rates over episodes that ended during the iteration).

## 4. Training run (`train.py`)

```
python train.py --memory gru --seed 0 --out E:\policy-runs\e2_gru_seed0 --val-suite E:\policy-data\val
```

Run directory:

| File | Contents |
|---|---|
| `config.json` | `RunConfig` = `EnvConfig` + `NetConfig` + `PPOConfig` + seed, plus `sim` `Config` as a dict |
| `log.csv` | §3.5 |
| `val.csv` | one row per validation: `iteration, agent_steps` + the overall metrics of §5.3 + `success_<layout>` for each layout |
| `ckpt_last.pt` | weights, optimiser, reward normaliser, per-slot return carry, iteration, agent-steps, RNG states, `config` |
| `ckpt_best.pt` | (only when `--val-suite` is given) weights + config at the best validation (highest `success_rate`; ties → lower `collision_rate`) |

Validation every `val_every = 25` iterations and at the end: deterministic
policy on the `val` suite (§5.1), in a process pool. `--resume` continues
from `ckpt_last.pt` (the run's config is read from the checkpoint; only
`--total-steps`, `--workers` and `--device` may be overridden) with every piece
of state restored (env state is not
restored: the workers start fresh scenes from re-seeded generators derived
from `(seed, iteration)`); a finished run exits at once. Checkpoint every
`ckpt_every = 25` iterations and on Ctrl-C.

## 5. Evaluation (`suites.py`, `evaluate.py`)

### 5.1 Suites

Fixed scene sets, generated once by `scripts/make_suites.py` into the data
root as `scene_NNNNN.json` (`Scene.as_dict`) plus `suite.json` (generator
parameters, `Config`). Headings are randomised at generation time with the
suite's own RNG and stored, so evaluation is deterministic.

| Suite | Scenes | Agents | Seed | Use |
|---|---|---|---|---|
| `val` | 20 per layout × 10 | 2–8 | 7000 | checkpoint selection |
| `test4` | 100 per layout × 10 | 4 | 8000 | **main E2 table** (proposal's condition) |
| `test_mixed` | 100 per layout × 10 | 2–8 | 9000 | the training distribution |

Obstacles 0–6 in every suite. Scene seeds are disjoint from the training
streams by construction (different `SeedSequence` roots).

### 5.2 Controllers

`evaluate.py` runs one controller on one or more suites:

- `--ckpt <run>/ckpt_best.pt`: the policy, deterministic (Gaussian mean) by
  default, `--stochastic` samples with a fixed seed. Noise switches off.
- `--greedy`: `physics.greedy_action`, the null model (it reads the
  simulator's own 51-number observation; that is intended).

Each scene is one episode. Episodes run in a process pool (`--workers`).

### 5.3 Metrics

Per scene (`per_scene.csv`, one row per scene): `suite, scene, layout,
n_agents, bottleneck` and

| Metric | Definition |
|---|---|
| `success_rate` | agents that reached a goal / n |
| `collision_rate` | agents removed by a collision / n |
| `static_collision_rate`, `agent_collision_rate` | split of the above by what was hit (checked with `Simulator._hits_static` at the collision step) |
| `timeout_rate` | agents still active at step 300 / n |
| `collision_violations`, `remaining_agents`, `finish_time` | from `Simulator.stats()` (framework counters) |
| `mean_finish_step` | mean step at which successful agents arrived |
| `mean_path_length` | from `stats()` |
| `all_reached` | 1 if every agent reached a goal |

`metrics.json`: per suite, `overall` (means over scenes) and `per_layout`;
`per_bottleneck` with bins `[5.2, 7), [7, 10), [10, 15), [15, ∞)` (E4 needs
the split). `trajectories/<suite>_<scene>.json` for the scenes chosen by
§6 (first scene of each layout), for the path figures; each file also embeds the
scene (`scene_data`) so the figure needs no data root.

## 6. Aggregation and figures (`aggregate.py`, `plots.py`)

`aggregate.py --evals E:\policy-eval --out results/e2 [--runs E:\policy-runs]` (`--runs` locates
`val.csv` for the training curves; default: the folder of each evaluated
checkpoint) reads every
evaluation directory and writes:

| Output | Contents |
|---|---|
| `summary.csv` | one row per (controller run, suite): identity columns (`controller` = greedy / mlp / gru, `seed`, `suite`) + overall metrics |
| `summary_per_layout.csv` | same, one row per layout |
| `table_e2.md` | `test4`: rows = layouts + overall, columns = greedy, mlp, gru as mean ± std over the five seeds, for success; a second table for collision rate |
| `table_e2_mixed.md` | the same on `test_mixed` |
| `fig_training_curves.png` | validation success vs agent-steps, mean ± std band over seeds, mlp vs gru |
| `fig_per_layout.png` | grouped bars, success per layout on `test4`, greedy / mlp / gru with seed std |
| `fig_bottleneck.png` | success vs bottleneck bin, three controllers |
| `fig_paths.png` | one scene per layout (10 panels): the scene rendered with `sim/render.render`, trajectories of greedy, mlp seed 0, gru seed 0 drawn over it in three colours, crosses where agents collided |

World → pixel for the path figure: `px = x·scale`, `py = (world − y)·scale`
(`Config.scale`; image row 0 is the top of the world).

## 7. The E2 experiment

Runs: `e2_mlp_seed{0..4}`, `e2_gru_seed{0..4}`, identical except `memory`.
Then each `ckpt_best.pt` on `test4` and `test_mixed`, deterministic; greedy
on both suites once.

Driver, one command on the training machine, resumable (re-run after any
interruption; every step is skipped when its output exists):

```
cd policy && python scripts/run_e2.py
```

1. suites (`make_suites.py`) into `E:\policy-data`;
2. greedy on `test4` and `test_mixed` → `E:\policy-eval\greedy`;
3. the ten training runs into `E:\policy-runs`, one after another, each
   with `--resume` when `ckpt_last.pt` exists;
4. evaluation of each run → `E:\policy-eval\<run>`;
5. `aggregate.py` → `policy/results/e2`;
6. `git add policy/results/e2`, commit, push (`--no-push` skips).

`--dev` instead trains `e2_mlp_dev` and `e2_gru_dev` for 2 M agent-steps
each, prints agent-steps/s and the validation success, and pushes nothing.
**Run `--dev` first**: if agent-steps/s on the Windows box makes 50 M
longer than ~4 h per GRU run, lower `--total-steps` (curves decide whether
the budget suffices; a plateau in `val.csv` well before the end means it
does). Flags: `--data-root`, `--runs`, `--eval`, `--seeds`, `--memories`,
`--total-steps`, `--workers`, `--no-push`, `--dry-run`.

Expected outcome (hypotheses the results will confirm or refute, not
acceptance criteria): both learned policies beat greedy on every layout;
the gap between `gru` and `mlp` is largest on `cul_de_sac` and smallest on
`open`.

## 8. Code structure

### 8.1 Key signatures

```python
# policyconfig.py
@dataclass class EnvConfig:   layouts, agents=(2, 8), obstacles=(0, 6), max_agents=8,
                              heading="random", goal_obs="bearing", memory="gru", stack_k=4,
                              obs_noise=0.0, bearing_noise_deg=0.0, action_noise=0.0
@dataclass class NetConfig:   memory="gru", hidden=128, log_std_init=-0.5
@dataclass class PPOConfig:   §3.4 table
@dataclass class RunConfig:   env, net, ppo, seed, val_every=25, ckpt_every=25;  save(path) / load(path)

# env.py
def obs_dim(env_cfg) -> int
def to_policy_obs(obs51: np.ndarray, env_cfg) -> np.ndarray          # §1.3 without stacking
class MultiAgentEnv:
    def __init__(self, env_cfg, sim_cfg, rng)
    def reset(self) -> dict                                            # obs, alive, first
    def step(self, actions: np.ndarray) -> dict                        # §1.5 arrays + final_obs
    def episode_records(self) -> list[dict]                            # outcomes of finished scenes since last call

# nets.py — §2
# rollout.py
def collect(model, envs, state, T) -> Batch                            # inline (W = 0)
class WorkerPool:  start(), collect(weights) -> Batch, close()
# ppo.py
def gae(batch, gamma, lam, reward=None) -> tuple[np.ndarray, np.ndarray]           # advantages, returns
class RewardScaler: update(rewards, dones, alive) / scale(rewards) / state_dict()
def replay(model, batch, L) -> (logp, value, entropy)                  # re-evaluate a batch
def update(model, optimiser, batch, adv, ret, cfg, device, rng) -> dict   # loss statistics
# suites.py
def make_suite(out_dir, layouts, per_layout, agents, obstacles, seed, sim_cfg)
def load_suite(path) -> list[tuple[str, Scene]]
# evaluate.py
def run_episode(scene, controller, sim_cfg, env_cfg=None, record=False) -> dict   # §5.3 row (+ trajectories)
```

Module names must not collide with `sim/` modules (`config, scene, render,
physics, geometry, layouts, reachability, generate, sketch`) — hence
`policyconfig.py`, `nets.py`.

### 8.2 Files

```
policy/
  DESIGN.md            this file
  README.md            how to run (§11.4 step 10)
  .gitignore           data/ runs/ eval/ __pycache__/
  _paths.py            sys.path shim to sim/
  policyconfig.py      EnvConfig, NetConfig, PPOConfig, RunConfig
  env.py               MultiAgentEnv (§1)
  nets.py              ActorCritic (§2)
  rollout.py           inline collection and WorkerPool (§3.1)
  ppo.py               RewardScaler, gae, update (§3.2–3.4)
  train.py             CLI, checkpoints, resume, validation (§4)
  suites.py            make_suite / load_suite (§5.1)
  evaluate.py          CLI, run_episode, metrics (§5.2–5.3)
  aggregate.py         CLI, tables (§6)
  plots.py             figures (§6)
  scripts/make_suites.py
  scripts/run_e2.py    the driver (§7)
  tests/run_tests.py
  results/e2/          written by the driver on the training machine
```

## 9. Tests (`tests/run_tests.py`, CPU, < 60 s total)

Same harness style as `sim/tests/run_tests.py`.

1. `test_obs_is_51_minus_distance` — `to_policy_obs` equals the simulator's
   observation with column 48 removed; `bearing_distance` gives all 51.
2. `test_heading_randomised_only` — headings change and are spread over
   [−π, π); positions, walls, obstacles, regions identical; same RNG seed →
   same headings.
3. `test_slots_and_auto_reset` — empty slots never alive; an agent that
   reached its goal is not alive afterwards; when the scene ends, the next
   observation comes from a new scene with `first` set on its filled slots.
4. `test_truncation_vs_termination` — with `max_steps` tiny, timeouts set
   `truncated` and a `final_obs`; goal and collision set `terminal`.
5. `test_gae_reference` — `gae` matches a hand-written loop on a small batch
   with a truncation, a terminal, a gap where the slot is empty and a new
   episode starting mid-batch.
6. `test_sequence_equals_steps` — for every memory mode, `sequence` over
   L steps with resets inside equals L calls of `step`.
7. `test_replay_matches_rollout` — for `mlp` and `gru`, re-evaluating a
   collected batch with stored initial states reproduces `logp` and `value`
   to 1e-5 before any update.
8. `test_stack_padding` — `stack` mode zero-pads the first k−1 frames and
   clears on `first`.
9. `test_workers_match_inline` — `WorkerPool` with 2 workers gives the same
   batch as inline collection for the same seeds (CPU).
10. `test_reward_scaler_reference` — matches a numpy reference.
11. `test_checkpoint_resume` — two iterations, interrupt, `--resume`, run to
    three: the restored iteration counter, agent-steps, weights, optimiser
    and scaler state equal what was saved, `log.csv` has exactly three rows,
    and a finished run resumed again exits without training.
12. `test_suite_deterministic` — building a suite twice gives identical
    JSON; headings are not the generator's.
13. `test_greedy_episode_matches_simulator` — `run_episode` with the greedy
    controller reproduces `Simulator.stats()` of a hand-stepped episode.
15. `test_update_is_finite_with_tiny_minibatches` — regression: a minibatch with a
    single alive entry once made the advantage std NaN.
14. `test_smoke_train_eval_aggregate` — a tiny run (W = 0, E = 2, T = 16,
    3 iterations) writes every §4 file; `evaluate.py` on a 10-scene suite and
    `aggregate.py` on that write every §5–§6 file.

## 10. Open items and assumptions

- The README and proposal describe a 51-number observation; E2 uses 50
  (bearing only). Update the README's pipeline table when E2 results are
  written up.
- Agents removed by a collision vanish from the scene (simulator semantics),
  so they do not block others afterwards. Accepted for E2; noted for E5,
  where crowds are denser.
- Agents in a multi-exit scene head for the nearest exit in straight-line
  distance (`RegionSet.nearest_point`); exit choice is not learned
  separately.
- The 50 M budget is a guess until the `--dev` run measures throughput.
- `sim/` greedy numbers in the README use generator headings and 4 agents;
  the E2 table recomputes greedy on the E2 suites (random headings).

## 11. Instructions for the implementing agent

### 11.1 Read first, in this order

1. This file, all of it.
2. `sim/README.md`, then `sim/config.py`, `sim/scene.py`, `sim/physics.py`
   (all of it), `sim/layouts.py` (`random_scene`, `LAYOUTS`,
   `MULTI_LAYOUTS`, `_place_agents*`), `sim/render.py` (`render`).
3. `sim/tests/run_tests.py` — the harness style to copy.
4. `perception/_paths.py`, `perception/train.py` (checkpoint and resume
   idioms), `perception/scripts/run_v5.py` (the driver style §7 asks for).
5. `pilot/rl_env.py`, `pilot/train_rl.py` — prior art only; do not port.

### 11.2 Ground rules

- **Do not modify anything under `sim/`, `perception/` or `pilot/`.** If
  something is missing there, implement it in `policy/` and log it in §11.7.
- **No new dependencies**: numpy, opencv, pillow, torch, matplotlib only. No
  SB3, gymnasium, scipy, tqdm, wandb.
- **Plain PyTorch, plain Python**, in the style of `sim/` and
  `perception/`: a module docstring stating what the module owns and its
  invariants, dataclasses for parameters, small functions, comments that say
  why.
- **The spec wins over convenience.** File names, run-directory contents,
  metric names and suite names are fixed so that E3–E6 can be written
  against them. If something here is wrong, change code and document
  together and log it in §11.7.
- **Windows-safe**: `pathlib`, no shell scripts, `spawn` multiprocessing
  with module-level worker functions, every CLI behind
  `if __name__ == "__main__":`, workers shut down cleanly on Ctrl-C.
- **CPU-first**: every test and smoke run passes on a laptop CPU; `device`
  resolved once and passed down.
- **No git operations.** The author commits and pushes. Leave the files in
  place and list them in the final report.

### 11.3 The `sim/` import shim

As in `perception/`: `policy/_paths.py` appends `<repo>/sim` to `sys.path`;
every module starts with `import _paths  # noqa: F401`; tests insert
`policy/` first. No module in `policy/` may share a name with a module in
`sim/` (§8.1).

### 11.4 Implementation order and the check at each step

| Step | Files | Check |
|---|---|---|
| 1 | `_paths.py`, `policyconfig.py`, `.gitignore` | `python -c "import _paths; from config import Config; import policyconfig"` prints nothing |
| 2 | `env.py` | tests 1–4, 8 |
| 3 | `nets.py` | test 6 |
| 4 | `ppo.py`, `rollout.py` (inline) | tests 5, 7, 10 |
| 5 | `rollout.py` (`WorkerPool`) | test 9 |
| 6 | `train.py` | test 11; a 20-iteration CPU run on `open` only with `--memory mlp` shows training success rising above the greedy rate — report the numbers |
| 7 | `suites.py`, `scripts/make_suites.py`, `evaluate.py` | tests 12, 13; greedy on a small suite prints per-layout rates |
| 8 | `aggregate.py`, `plots.py` | test 14; figures render |
| 9 | `scripts/run_e2.py` | `--dry-run` prints every command of §7 with Windows default paths |
| 10 | `README.md`, §11.7 | README: tests, dev run, full driver, one-run train/evaluate commands |

### 11.5 Local development environment

The Mac has no `cv2` in the system Python. Use `uv` (installed):

```bash
cd policy
uv run --python 3.12 --with numpy --with opencv-python-headless --with pillow --with torch --with matplotlib python tests/run_tests.py
```

Generate any scratch data into `policy/data/` (git-ignored).

### 11.6 Definition of done

- Every file of §8.2 exists and does what its line says.
- `python tests/run_tests.py` passes on CPU; the §11.4 checks were run and
  their output is in the final report.
- `python scripts/run_e2.py --dry-run` prints commands that run unchanged on
  Windows.
- `policy/README.md` exists; §11.7 lists every deviation from §1–§10, or
  says there are none.
- Nothing committed. The final report lists, in order: what was built, the
  test and smoke-run output, the measured CPU agent-steps/s (W = 0 and
  W = 4 on the Mac), what was not done and why, the divergence list, and
  every file created.

### 11.7 Divergence log

Filled in by the implementing agent: one line per deviation from §1–§10 —
what changed, why, which section was updated.

One line per deviation (section changed in brackets):

- `WorkerPool(W=0)` runs `inline_workers` (default 1) workers rather than "the pool's W", so the equality of §3.1 / test 9 is stated against a pool of the same size. [§3.1]
- `RewardScaler.update` takes `alive` (statistics only over acting agents); `gae` takes an optional scaled `reward`; `update` takes `adv, ret, device, rng`; added `replay`. [§8.1]
- Minibatches are `np.array_split` of the shuffled sequences (no small tail batch) and the advantage std is skipped for a single alive entry: the first draft produced NaN weights at iteration 15 of the dev run. Test 15 pins it. [§3.4, §9]
- The replay check (§3.4) runs at iteration 1 and every 25th (`--check-replay-every`), printing a warning above 1e-4 (1e-2 on GPU), not before every update, to save a forward pass. [§3.4]
- `ckpt_best.pt` exists only when a validation suite is given. [§4]
- On `--resume` the config comes from the checkpoint; only budget, workers and device can change. [§4]
- Trajectory files embed the scene; `aggregate.py` gets an optional `--runs`. [§5.3, §6]
- `run_episode`'s `env_cfg` argument is optional and unused (the controller carries it). [§8.1]
- `run_e2.py` has `--eval-workers` (validation / evaluation processes, default 6) and passes `--val-workers` to training. [§7]
- Added test 15. [§9]
