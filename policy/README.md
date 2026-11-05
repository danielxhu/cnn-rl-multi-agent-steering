# policy/ — E2 baseline steering policy

Multi-agent PPO (own implementation, plain PyTorch) for the shared steering
policy, the E2 experiment (`mlp` vs `gru`, five seeds, ten layouts) and its
evaluation. The specification is [DESIGN.md](DESIGN.md); code and spec are kept
in step (divergence log in §11.7).

Dependencies: numpy, opencv-python-headless, pillow, torch, matplotlib. All
commands run from `policy/`. On the Mac use `uv` (no `cv2` in the system Python):

```bash
uv run --python 3.12 --with numpy --with opencv-python-headless --with pillow --with torch --with matplotlib python tests/run_tests.py
```

On the Windows box replace that prefix with `python`.

## Tests (CPU, about 75 s)

```bash
python tests/run_tests.py            # all
python tests/run_tests.py test_7     # by name fragment
```

Test 7 (replay of a batch reproduces the rollout's logp and value, `mlp` and
`gru`) is the one that catches recurrent-PPO bugs; `train.py` repeats it every
25 iterations and warns if it fails.

## One run: train, evaluate

```bash
python scripts/make_suites.py --data-root data              # val, test4, test_mixed
python train.py --memory gru --seed 0 --out runs/e2_gru_seed0 --val-suite data/val --val-workers 4 --workers 4
python train.py --memory gru --seed 0 --out runs/e2_gru_seed0 --val-suite data/val --resume      # after an interruption
python evaluate.py --ckpt runs/e2_gru_seed0/ckpt_best.pt --suites test4,test_mixed --data-root data --out eval/e2_gru_seed0 --workers 4
python evaluate.py --greedy --suites test4,test_mixed --data-root data --out eval/greedy --workers 4
python aggregate.py --evals eval --runs runs --out results/e2
```

Useful flags of `train.py`: `--layouts open` (restrict layouts; validation follows), `--agents LO HI`,
`--total-steps`, `--workers W` (0 = inline), `--envs E`, `--device cpu|cuda`,
`--stop-after N` (iterations, smoke runs); every `PPOConfig` field is a flag.

Run directory: `config.json`, `log.csv`, `val.csv`, `ckpt_last.pt`,
`ckpt_best.pt` (DESIGN §4). Evaluation directory: `per_scene.csv`,
`metrics.json`, `trajectories/`.

## E4: leave-one-layout-out (training machine, resumable; `DESIGN_E4.md`)

```bash
python scripts/run_e4.py --dry-run            # print every command with the Windows default paths
python scripts/run_e4.py --dev                # tiny trial in a temp folder, no git
python scripts/run_e4.py --layouts doorway    # one fold; --layouts names the layouts to hold out
python scripts/run_e4.py                      # ten folds, seed 0, about 30 h
python scripts/run_e4.py --seeds 1 2          # more seeds; finished folds are skipped
```

It needs the E2 suites and the `e3_both_seed*` evaluations. Results go to `results/e4`.

## The whole of E2 (training machine, resumable)

```bash
python scripts/run_e2.py --dev        # first: 2 M steps per memory variant; prints agent-steps/s and validation
python scripts/run_e2.py --dry-run    # print every command with the Windows default paths
python scripts/run_e2.py              # suites, greedy, 10 runs, evaluations, aggregate, commit, push
python scripts/run_e2.py --no-push
```

Paths default to `E:\policy-data`, `E:\policy-runs`, `E:\policy-eval`
(`--data-root`, `--runs`, `--eval`). Other flags: `--seeds`, `--memories`,
`--total-steps`, `--workers`, `--eval-workers`. Re-run after any interruption:
finished steps are skipped and unfinished runs resume.

## ORCA baseline (no training)

Classical reciprocal collision avoidance (DESIGN_ORCA.md), plain numpy, in
`orca.py`. `orca_straight` heads for the goal's straight-line bearing (the
policy's information); `orca_path` follows the E3 geodesic field (a map).

```bash
python evaluate.py --orca straight --suites test4,test_mixed --data-root data --out eval/orca_straight --workers 4
python evaluate.py --orca path     --suites test4,test_mixed --data-root data --out eval/orca_path --workers 4
python scripts/run_orca.py --dry-run  # training machine: both variants, E2 / E3 tables with ORCA columns, push
```

`aggregate.py` treats greedy and both ORCA variants as non-learned (one run,
no spread, no training curve); `--path-variants` picks the series of
`fig_paths.png`, `--include REGEX` limits which evaluation directories count.

## Behaviour gap: true vs parsed scenes

The same policy on the true scenes of the sketch7 test sets and on the scenes
perception parsed from their images (DESIGN_GAP.md). `orca_path` runs next to it
twice: true vs parsed world, and in the true world with the parsed scene as its
map (`ORCAController(map_scene=...)`). No training; perception uses the GPU if
there is one, episodes use CPU workers.

```bash
python scripts/run_gap.py --dry-run    # print every command
python scripts/run_gap.py --limit 20   # smoke run: 20 scenes per set, nothing committed
python scripts/run_gap.py              # parse, score, episodes, tables, commit, push
python scripts/run_gap.py --policy E:\policy-runs\e3_geo_seed1\ckpt_best.pt --no-push
```

Paths default to `E:\cnn-data` (`--data-root`) and `E:\policy-eval\gap`
(`--work`: parsed scenes, overlays, perception scores, about 85 KB per scene).
Other flags: `--model`, `--sets`, `--workers`, `--device`, `--results`. Re-run after
any interruption: finished sets and scenes are skipped.

Results in `results/gap/<policy>/`: `gap.md` (both ORCA arms next to the policy,
failed parses, gap by perception error, per layout), `per_scene.csv`, `metrics.json`,
`fig_gap.png`, `fig_gap_map.png`, `identity.json`. Read DESIGN_GAP.md §3 and §5
before quoting the numbers.

## Files

`_paths.py` (import shim to `sim/`), `policyconfig.py`, `env.py`, `nets.py`,
`rollout.py`, `ppo.py`, `train.py`, `suites.py`, `evaluate.py`, `aggregate.py`,
`plots.py`, `orca.py`, `gap.py`, `scripts/make_suites.py`, `scripts/run_e2.py`, `scripts/run_e3.py`,
`scripts/run_orca.py`, `scripts/run_gap.py`, `tests/run_tests.py`.
`data/`, `runs/`, `eval/` are git-ignored scratch; `results/e2/` and `results/gap/` are written by the drivers.
