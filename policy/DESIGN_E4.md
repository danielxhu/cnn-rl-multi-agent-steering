# Policy: E4, layout transfer

Design document for experiment **E4**: does a policy work on a layout family it
never trained on? `docs/PROPOSAL_REVISIONS.md` fixes the method as
leave-one-layout-out. It builds on `policy/DESIGN.md` and `policy/DESIGN_E3.md`:
everything not stated here is as there.

## 1. Folds

One fold per layout family, ten in all: `open`, `corridor`, `room`, `doorway`,
`two_doorway`, `cul_de_sac`, `crossing`, `boxrooms`, `walls_multi`, `rooms_multi`.
Fold `<layout>` trains on the other nine and is tested on `<layout>`.

| | |
|---|---|
| Run name | `e4_holdout_<layout>_seed<k>` |
| Network | `gru` only |
| Reward | E3 winner `e3_both`: `timeout_penalty` 25, `progress_metric` geodesic |
| Budget | 50 M agent-steps, same as E2 and E3 |
| Everything else | as E3 (agents 2–8, random headings, PPO settings) |
| Selection | `ckpt_best.pt`, chosen on validation scenes of the nine training layouts |
| Test | `test4` and `test_mixed`, the same fixed scenes as E2 and E3 |
| Baseline | `e3_both_seed{0,1,2}`: the same GRU and reward, trained on all ten |

The driver stops if `results/e3/winner.json` no longer names `e3_both`.

## 2. Seeds and budget

About 3 h per GRU run on the training machine, so ten folds at one seed take
about 30 h, plus evaluation.

| Pass | Where | Seeds | Runs | Time |
|---|---|---|---|---|
| First | Windows GTX 1070 | 0 | 10 | ~30 h |
| Later | Emory cluster | 3 or more (`--seeds 1 2 …`) | 10 per seed | |

`docs/COMPUTE_REQUEST.md` budgets 2 variants × 5 seeds on the cluster. This
driver covers `gru` only; adding `mlp` would need a variant loop in the driver.

Suggested start: `--layouts doorway` as a one-fold pilot, about 3 h, then the rest.
Finished folds are skipped, so the pilot is not wasted.

## 3. Keeping the held-out layout out of training

`train.py --layouts` already restricted the sampled label. That alone leaves
two leaks, both closed by one rule, `env.in_training_layouts`:

| Route | Problem | Fix |
|---|---|---|
| Scene sampling | Covered by `--layouts` | `sample_scene` keeps only `EnvConfig.layouts` |
| Validation | `train.py` validated on the whole suite, so the held-out layout's 20 scenes chose `ckpt_best.pt` | `Validator(layouts=cfg.env.layouts)` |
| `open` under another label | `walls_multi` and `rooms_multi` call the `open` generator when box placement fails and keep their own label; `walls_multi` can also draw no walls | When `open` is held out, a scene with only the frame walls is rejected, in sampling and in validation |

Measured over 400 draws per family: `walls_multi` called the `open` generator
about 0.75 % of the time and had frame-only scenes 13 % of the time; `rooms_multi`
3 % for both. `boxrooms` has the same fallback, which never fired in those draws;
the rule covers it too. Every such fallback goes to `open`, so no other fold is
affected.

**E2 and E3 are unchanged.** With all ten layouts every scene passes the rule,
which uses no randomness, so the sampled stream and the validation set are
identical (test 3). The one behaviour change is for runs with a layout subset:
`--layouts X --val-suite S` now validates on the scenes of `X` only, and refuses a
suite that has none.

`--resume` takes the layouts from the checkpoint. The driver also stops if
`ckpt_last.pt` does not match the fold's layouts, reward and seed.

### Code changes

| File | Change |
|---|---|
| `env.py` | `in_training_layouts`; `sample_scene` uses it |
| `evaluate.py` | `Validator(…, layouts=None)` |
| `train.py` | passes `cfg.env.layouts` to the validator |
| `aggregate_e4.py` | new: tables and figures, §6 |
| `scripts/run_e4.py` | new: the driver, §4 |
| `tests/run_tests.py` | §7 |

`sim/`, `aggregate.py` and `plots.py` are not changed.

## 4. Driver

```
cd policy
python scripts/run_e4.py --dry-run                 # print the commands
python scripts/run_e4.py --dev                     # tiny trial in a temp folder, no git
python scripts/run_e4.py --layouts doorway         # one fold
python scripts/run_e4.py                           # ten folds, seed 0
python scripts/run_e4.py --seeds 1 2               # more seeds
```

| Flag | Meaning | Default |
|---|---|---|
| `--seeds` | seeds, outer loop | `0` |
| `--layouts` | layouts to **hold out**, one fold each | all ten |
| `--data-root`, `--runs`, `--eval` | roots | `E:\policy-data`, `E:\policy-runs`, `E:\policy-eval` |
| `--workers`, `--eval-workers` | rollout, and validation or evaluation, processes | 10, 6 |
| `--total-steps` | budget | 50 M |
| `--dry-run` | print only; Windows paths stay Windows-style on the Mac | |
| `--no-push` | no pull, no push; commits locally | |
| `--dev` | 20,000 steps, folds `open` and `doorway`, 3 scenes per layout, temp folder, no git | |

Steps:

1. Check the winner, the suites and the baseline evaluations; stop if missing.
2. Per seed, per fold: train, evaluate, print the held-out line. Each fold is
   evaluated right after training, so a wrong result shows after 3 h, not 30 h.
3. `aggregate_e4.py` over every fold evaluated so far.
4. Pull, add, commit and push `policy/results/e4`.

**Resume.** `ckpt_last.pt` missing: train from scratch. Below budget: `--resume`.
At budget: skip training. Evaluation is skipped when `metrics.json` is complete and
at least as new as `ckpt_best.pt`. Ctrl-C waits for `train.py` to save.

**Where evaluations go.** `<eval>/e4/<run>`, one level below E2 and E3:
`aggregate.py` looks one level deep, so re-running the E2 or E3 report cannot put
fold runs into its `gru` column or summaries.

## 5. How to read the comparison

| Limit | Effect |
|---|---|
| One seed in the first pass | A fold has no spread of its own. The yardstick is the baseline's seed range per layout: `test4` mean 0.042, largest 0.075 (`crossing`); `test_mixed` mean 0.036, largest 0.137 (`crossing`). Do not read a gap under about 0.08 (`test4`) as an effect |
| Related families | `doorway` / `two_doorway`, `room` / `rooms_multi` / `boxrooms` and `corridor` / `crossing` share structure. A held-out family is new as a family, not as geometry, so small gaps there do not prove general transfer |
| Control table | Each fold is also compared with the baseline on its nine trained layouts. A gap counts as transfer only if that control stays near zero |
| Baseline selection | The baseline's `ckpt_best.pt` saw 20 validation scenes of the layout; a fold's did not. Slightly favours the baseline |
| Reward and PPO settings | Chosen in E2 and E3 with all ten layouts, and not re-chosen per fold |
| Exposure | Same 50 M budget over nine layouts: each gets about 11 % more than in the baseline |
| Ceiling | `open` (0.994), `walls_multi` (0.988) and `room` (0.979) have little room to show a gap on `test4` |
| `open` fold, `test` suites | Frame-only `walls_multi` and `rooms_multi` test scenes stay in those rows, so this fold's control includes a few effectively unseen scenes. The held-out row is unaffected |

## 6. Reporting

`aggregate_e4.py` writes to `policy/results/e4/`:

| File | Content |
|---|---|
| `table_e4.md`, `table_e4_mixed.md` | Per held-out layout: success, collision and timeout, all layouts vs held out, and the success gap. Then the control table, then the bottleneck split |
| `summary_bottleneck.csv` | The bottleneck split as data |
| `summary.csv`, `summary_per_layout.csv` | Every fold on every layout, from `aggregate.write_summaries` |
| `fig_per_layout.png` | Success per layout, all layouts vs held out |
| `fig_bottleneck.png` | Held-out scenes pooled, success vs bottleneck width |
| `fig_training_curves.png` | Validation success per fold, on its nine layouts |

**Bottleneck split.** `per_bottleneck` in `metrics.json` pools all layouts of a
suite, so in a fold it would mix held-out scenes with trained ones. The split is
computed from `per_scene.csv`, filtered to the held-out layout, with the same bins
(`evaluate.metrics_for`). Bins can be small: each cell shows its scene count.

Partial sets are fine: the baseline is restricted to layouts that have a fold, so
"mean over folds" compares the same layouts on both sides. No test suite chooses
anything; E4 has no winner.

## 7. Tests

`python tests/run_tests.py e4` (CPU, about 40 s), all in `tests/run_tests.py`:

1. `sampling_never_draws_the_held_out_layout`: each of the ten folds.
2. `open_fold_rejects_open_scenes_under_other_labels`: sampling and validation.
3. `all_ten_layouts_unchanged`: scene stream equals the pre-E4 code; validation set equals the whole suite.
4. `held_out_layout_absent_from_a_real_fold`: a real training run with resume; spies on every sampled and validated scene; `val.csv`, `config.json` and both checkpoints.
5. `worker_processes_keep_to_the_fold`: real worker processes.
6. `dry_run_prints_the_windows_commands`: exact commands; the reward flags equal `run_e3.VARIANTS[winner]`.
7. `driver_resumes_and_skips`: new, partial, finished, cut-off `metrics.json`, extended run, wrong checkpoint.
8. `aggregate_reports_held_out_scenes_only`: gap values, the filtered bottleneck split, baseline restricted to folds, mislabelled fold refused, folds invisible to `aggregate.py`.

Tests 1, 2, 4, 6 and 8 were each checked to fail when their guard is removed.

## 8. Expected outcome (hypotheses, not acceptance criteria)

- Largest gaps on families with structure no other family teaches: `cul_de_sac`
  (backing out of a sealed pocket) and `crossing` (two groups meeting).
- Small gaps on `doorway` and `two_doorway`, which teach each other, and on
  `rooms_multi`.
- `open` near the ceiling even when held out.

## 9. Divergence log

- Added `aggregate_e4.py` rather than extending `aggregate.py`, which keeps the E2 and E3 report untouched. [§6]
- Path figures are left out; `evaluate.py` still saves trajectories, so they can be added later from `<eval>/e4/*/trajectories`. [§6]
- `run_e4.py` evaluates each fold right after training (`run_e3.py` trains all first), and waits for `train.py` on Ctrl-C (`subprocess.run` would kill it about 0.25 s later, while it saves). [§4]
