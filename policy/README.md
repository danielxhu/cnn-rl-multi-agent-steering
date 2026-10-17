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

## Tests (CPU, about 20 s)

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

Useful flags of `train.py`: `--layouts open` (restrict layouts), `--agents LO HI`,
`--total-steps`, `--workers W` (0 = inline), `--envs E`, `--device cpu|cuda`,
`--stop-after N` (iterations, smoke runs); every `PPOConfig` field is a flag.

Run directory: `config.json`, `log.csv`, `val.csv`, `ckpt_last.pt`,
`ckpt_best.pt` (DESIGN §4). Evaluation directory: `per_scene.csv`,
`metrics.json`, `trajectories/`.

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

## Files

`_paths.py` (import shim to `sim/`), `policyconfig.py`, `env.py`, `nets.py`,
`rollout.py`, `ppo.py`, `train.py`, `suites.py`, `evaluate.py`, `aggregate.py`,
`plots.py`, `scripts/make_suites.py`, `scripts/run_e2.py`, `tests/run_tests.py`.
`data/`, `runs/`, `eval/` are git-ignored scratch; `results/e2/` is written by the driver.
