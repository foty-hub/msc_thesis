# Tuning sparse-grid train-time robustness

This document describes a practical tuning protocol for the maintained
`traintime_robustness` experiment. It is based on the LunarLander tuning run
performed on 2026-07-29, but is intended to be reusable for Acrobot,
MountainCar, CartPole, and future discrete-action environments.

The main principle is to separate three questions:

1. **Can the calibration change the policy?** Check grid occupancy and the
   number of statistically usable cells.
2. **Does it appear helpful on development seeds?** Use cheap paired
   evaluations to reject bad configurations.
3. **Does the frozen configuration generalise?** Evaluate untouched training
   seeds over the full shift range.

Do not launch a large Bayesian search before answering the first question.
Calibration and occupancy diagnostics are cheap; shifted policy evaluation is
the expensive and high-variance part.

## Maintained experiment path

The relevant implementation is:

- [`src/crl/discretise/grid.py`](src/crl/discretise/grid.py): quantile-fitted Cartesian grid and sparse
  mixed-radix cell IDs.
- [`src/crl/calib.py`](src/crl/calib.py): rollout collection, TD/Monte Carlo scores, calibration
  sets, conformal quantiles, and fallback corrections.
- [`src/crl/experiment.py`](src/crl/experiment.py): grid calibration, corrected action selection, and
  paired shift evaluation.
- [`scripts/traintime_robustness.py`](scripts/traintime_robustness.py): multi-seed orchestration, result
  persistence, plots, timing, and coverage diagnostics.
- [`scripts/cli.py`](scripts/cli.py): command-line interface.
- `src/crl/experiment.py::SHIFT_SPECS`: environment shift ranges and default
  grid resolutions.

The corrected action is

```text
argmax_a [Q(s, a) - correction(s, a)].
```

The current global defaults are 10,000 calibration transitions,
`min_calib=100`, alpha 0.25, `obs_quantile=0.1`, TD scoring, and at most four
workers. The grid resolution comes from `SHIFT_SPECS` unless `--grid-bins` is
passed. These defaults were selected using LunarLander; their transfer to other
environments must be tested rather than assumed. The subsequent 25-seed
LunarLander evaluation found no reliable aggregate improvement, so these
defaults should be treated as an evaluated candidate rather than a settled
paper configuration.

Only visited state-action cells and their bounded score deques are allocated.
The number of possible IDs can therefore be large without causing a dense-grid
allocation. Statistical sparsity is still important even when memory sparsity
is solved.

## Important implementation details

### Unseen-cell fallback

`compute_corrections` calculates a quantile for every cell with at least
`min_calib` scores. The fallback for an unseen or under-populated cell is the
**maximum** correction among all qualified cells.

This has two consequences:

- If every candidate action receives the fallback, subtracting it changes
  nothing and the calibrated policy is identical to the baseline.
- As more transitions are collected, a rare high-error cell can cross
  `min_calib` and abruptly increase the global fallback. More calibration data
  is therefore not guaranteed to improve performance.

Always record:

- possible state-action cells;
- visited cells;
- cells meeting `min_calib`;
- fallback correction; and
- the fraction of evaluated shifts or actions for which the policy changes.

The first four are now included in or derivable from experiment results.

### Grid size and occupancy

For a uniform grid,

```text
possible state-action cells = bins ** observation_dimensions * actions.
```

This value describes the ID space, not allocated memory. A useful grid is not
one that covers a large fraction of the entire Cartesian product; it is one
that creates enough high-support cells to estimate local corrections while
retaining enough resolution to distinguish meaningfully different states.

Inspect occupancy before evaluation. For each candidate grid, calculate the
number of visited cells whose sample counts exceed plausible `min_calib`
values. A configuration that qualifies zero or one cell is unlikely to be a
useful local calibration, even though it may still induce behaviour through
the fallback.

### Cached models

The current model cache key contains environment, algorithm, and seed, but not
the requested training timesteps or the DQN YAML contents. Changing
`n_train_steps` without `--retrain` can silently load an older cached model.

Before comparing agent-training settings:

- inspect which model files already exist under `models/<env>/<algorithm>/`;
- use `--retrain` intentionally when training settings change; and
- avoid interpreting a recorded `n_train_steps` as proof that a cached model
  was trained for that budget.

Do not remove or overwrite caches casually: they are expensive experimental
artifacts and may be needed to reproduce earlier results.

### Monte Carlo buffer boundary

The current transition collector requests an exact number of transitions, so
the buffer can end partway through an episode. `compute_mc_returns` starts its
backward accumulator at zero; consequently, the final incomplete episode has a
truncated and biased return target.

TD scoring does not have this issue because terminal masks are applied
transition by transition. Before treating Monte Carlo scoring as a serious
candidate, change collection or preprocessing to retain only complete episodes
(or explicitly bootstrap the incomplete suffix) and add a regression test for
that boundary.

## Recommended tuning protocol

### 1. Define the estimand and success criteria

The primary tuning metric should be the paired difference

```text
mean calibrated return - mean baseline return
```

for the same trained policy, shift value, initial-state seed, and episode
budget.

Do not optimize calibrated return alone. Training-seed quality varies much more
than many calibration effects, so an objective based on raw calibrated return
mostly selects configurations evaluated on lucky policies.

Useful secondary criteria are:

- median paired delta across training seeds;
- number of training seeds with positive aggregate delta;
- number of seed-by-shift wins, losses, and ties;
- nominal-environment delta;
- worst seed delta;
- worst mean shift delta; and
- fallback size and number of calibrated cells.

Treat the **training seed** as the main independent unit. Shift values and
episodes from one trained policy are correlated and should not be presented as
independent replications.

### 2. Reserve development and validation seeds

Split model seeds before tuning. A reasonable small-compute split is:

- 5 development seeds for screening; and
- at least 10 untouched seeds for final validation.

For a paper run, increase the held-out set toward the repository default of 25
seeds if compute permits. Never allow Optuna, manual selection, or fallback
rule changes to observe validation-seed results before the configuration is
frozen.

Before collecting calibration transitions, qualify every original DQN with one
reproducible nominal evaluation batch. The Optuna tuner defaults to 25 episodes
per DQN and retains a seed when its mean return is greater than or equal to the
environment threshold:

```text
CartPole-v1      475
Acrobot-v1      -100
MountainCar-v0  -110
LunarLander-v3   200
```

These values are explicit in `src/crl/env.py` and match the Gymnasium 1.1.1
registry locked by this project. The old CartPole comment used 490, which was
not the registered threshold. The unsupported Pendulum comment was removed
because Gymnasium does not register a reward threshold for Pendulum-v1.

The nominal evaluation is a seed-qualification check, not an Optuna trial. Its
raw returns are cached against the model fingerprint. Failed seeds are excluded
before transition collection, baseline evaluation, pruning, and the objective.
Development and validation eligibility are recorded separately in
`development_eligibility.json` and `validation_eligibility.json`.

This filtering changes the estimand: the study measures robustness conditional
on the original DQN solving the nominal task. Always report both the requested
seed count and the eligible/excluded counts. With high-variance policies,
increase `--eligibility-eval-episodes` instead of repeatedly rerunning the
check until a borderline seed passes. Use `--nominal-reward-threshold` only
when the experiment intentionally needs a different criterion.

### 3. Establish a baseline and profile one process

Start with one cached model, few evaluation episodes, and all configured
shifts:

```bash
uv run python scripts/cli.py \
  --env-name LunarLander-v3 \
  --debug-seed 0 \
  --num-eval-episodes 5 \
  --results-out lunarlander_baseline
```

Record wall time and peak RSS. Check:

- the model is loaded rather than unexpectedly retrained;
- all returns are finite;
- calibration creates at least some qualified cells;
- calibrated and baseline returns are not accidentally identical everywhere;
  and
- the result plot and pickle are written.

For memory profiling on macOS, `/usr/bin/time -l` reports maximum resident set
size. On Linux, `/usr/bin/time -v` is the more common equivalent. Account for
one model and Python runtime per process when selecting `--max-workers`.

### 4. Run an occupancy-only sweep

Collect one large deterministic nominal rollout per development seed. Reuse
prefixes of that rollout for different `n_calib_steps` values so that a 5k
candidate is a true prefix of the corresponding 10k candidate.

For each combination of:

- calibration steps;
- grid bins;
- observation quantile; and
- scoring method,

fit the grid, fill calibration sets, and report the cell-count distribution.
This usually takes seconds and avoids spending evaluation time on settings
that cannot affect action selection sensibly.

A useful first table reports counts greater than or equal to:

```text
5, 10, 20, 40, 80, 100, 200
```

Choose `min_calib` candidates from observed occupancy rather than using the
same arbitrary threshold for every environment.

### 5. Screen failure modes cheaply

Select representative shifts containing:

- the hardest low endpoint;
- one or two values between the endpoint and nominal;
- the nominal value;
- one or two values beyond nominal; and
- the opposite endpoint.

Use only 2 episodes per shift for the first pass. This pass may reject:

- configurations that never change the policy;
- fallback explosions;
- catastrophic return collapses;
- grids with effectively no local corrections; and
- configurations that are obviously too slow.

Do **not** select a winner from two-episode results. Their only purpose is to
remove failure modes.

### 6. Compare a small candidate set on development seeds

Increase to roughly 10 episodes per representative shift and evaluate about
3-5 candidates across all development seeds.

Keep candidates deliberately diverse. For example:

- intended default;
- stricter `min_calib`;
- looser `min_calib`;
- smaller calibration budget; and
- one gentler or stronger alpha.

Prefer configurations that:

- improve most seeds rather than relying on one large gain;
- have a positive median seed delta;
- do not substantially damage nominal return;
- avoid extreme or unstable fallback values; and
- qualify a plausible number of cells across different policies.

Stop tuning when one family is consistently preferable. Searching more
variants after observing noisy development results increases selection bias.

### 7. Freeze and validate

Write down the complete configuration before evaluating validation seeds:

- model type and training budget;
- calibration steps;
- bins;
- observation quantile;
- `min_calib`;
- maximum samples per cell;
- alpha;
- score function and TD/Monte Carlo method;
- shift values;
- episodes per shift; and
- evaluation seed convention.

Then run the full shift range with at least 25 episodes per shift. Preserve raw
episode returns.

Report seed-level baseline, calibrated return, paired delta, calibrated-cell
count, fallback, and shift win/loss/tie counts. Aggregate with both mean and
median. A mean confidence interval and exact sign test can be useful, but keep
the exploratory nature of hyperparameter selection explicit.

### 8. Run tests and a post-change smoke experiment

After changing defaults or result schemas:

```bash
MPLCONFIGDIR=/tmp/mplconfig WANDB_MODE=offline uv run pytest -q
```

Run a one-episode-per-shift CLI smoke test using only defaults. Inspect the
saved configuration, timing fields, visited-cell count, and calibrated-cell
count. This catches differences between a manually constructed tuning script
and the maintained CLI path.

## Hyperparameter interpretation

| Parameter | Main effect | Common failure mode |
|---|---|---|
| `n_calib_steps` | More scores and potentially more qualified cells | A rare cell qualifies and inflates the maximum fallback |
| `grid_bins` | Higher spatial resolution | Counts fragment across an exponential ID space |
| `min_calib` | Controls statistical support per correction | Too high gives no policy change; too low admits unstable outlier cells |
| `alpha` | Controls correction quantile | Small alpha can make corrections and fallback very large |
| `obs_quantile` | Sets grid range from nominal observations | Extreme clipping or overly narrow ranges collapse shifted states into boundary cells |
| `max_calib_per_cell` | Bounds memory and limits dominance of common cells | A small cap discards useful history; a large cap matters little when few cells are dense |
| TD vs Monte Carlo | Changes the conformity target | MC scores have higher variance and depend on complete episode handling |

Tune coverage parameters (`n_calib_steps`, `grid_bins`, `min_calib`,
`obs_quantile`) before fine-tuning alpha. Alpha cannot rescue a calibration in
which every action receives the same fallback.

## LunarLander case study

### Initial failure

The original maintained defaults used:

```text
2,500 transitions, 4 bins, min_calib=80, alpha=0.25.
```

For seed 0 this produced:

- 262,144 possible state-action IDs;
- only 1 qualified calibrated cell; and
- identical baseline and calibrated policies at all 16 gravity values.

The sparse implementation did not have a grid-memory problem; it had a
statistical-coverage problem.

### Occupancy findings

With seed 0 and 10,000 transitions, a 4-bin grid had:

- 1,084 visited cells;
- 194 cells with at least 10 samples;
- 75 with at least 20;
- 27 with at least 40;
- 10 with at least 80; and
- 10 with at least 100 in the final rerun.

Looser thresholds did not necessarily perform better. They allowed rare
high-error cells to determine the global fallback.

In one deliberately aggressive configuration using 3 bins, `min_calib=20`,
and alpha 0.1, the fallback changed as follows:

| Calibration steps | Fallback |
|---:|---:|
| 5,000 | 1.95 |
| 10,000 | 24.34 |
| 20,000 | 29.50 |

Performance deteriorated at the larger budgets. This reproduced the prior
observation that too many calibration transitions can hurt.

### Development and frozen configuration

Seeds 0-4 were used for development. Cheap screens compared calibration
budgets, 2-4 bins, `min_calib` values from 20 to 120, alpha values from 0.05 to
0.4, and observation quantiles of 0.05, 0.1, and 0.2.

The frozen configuration was:

```yaml
n_calib_steps: 10000
grid_bins: 4
obs_quantile: 0.1
min_calib: 100
max_calib_per_cell: 500
alpha: 0.25
scoring_method: td
score_fn: signed_score
```

On the development seeds it produced seed deltas of approximately:

```text
+0.8, 0.0, +10.4, +21.9, +21.2
```

### Held-out validation

An initial validation on seeds 5-14 used 25 episodes at every integer gravity
from -16 through -1. It looked encouraging: 9/10 seeds improved and the mean
seed delta was +10.93. That result was an interim subset, not an adequate final
conclusion.

The same frozen configuration was then run on all 25 available seeds. Seeds
0-4 were the development seeds and seeds 5-24 formed the 20-seed held-out set.

Full results:

- all-seed baseline return: 98.61;
- all-seed calibrated return: 99.13;
- all-seed mean delta: +0.52;
- all-seed median delta: +1.53;
- all-seed wins/losses: 15/10;
- approximate 95% t interval: [-8.61, 9.65]; and
- two-sided exact sign-test p-value: 0.424.

On the 20 held-out seeds alone, mean delta was +0.53, median delta was +1.87,
and 12/20 seeds improved. Seed 24 was a valid severe failure with a delta of
-85.06. Excluding it raises the all-seed point estimate to +4.09 but still
leaves the approximate interval crossing zero; it must remain in the primary
analysis.

The complete evaluation therefore does not show a reliable aggregate
improvement. The contrast between the optimistic first 10 validation seeds and
the full result is an important stopping-rule lesson: do not report a favorable
prefix of a planned seed set.

The final table, CSVs, figure, and runtime notes are in the
[25-seed LunarLander report](results/lunarlander_sparse_25seed/summary.md). The
earlier
[10-seed validation summary](results/lunarlander_sparse_validation_summary.md)
is retained but explicitly marked as superseded.

### Runtime and memory

On the machine used for the validation:

- the four-worker 25-seed run completed in 5 minutes 33 seconds;
- mean per-seed worker time was 49.3 seconds, with a 24.8-77.5 second range;
- calibration averaged 0.92 seconds per seed;
- evaluation averaged 48.2 seconds per seed;
- peak single-process RSS was approximately 394-398 MiB; and
- no dense grid-sized allocation was observed.

The model and PyTorch runtime dominate RSS. Four workers are therefore a safer
default than eight on a laptop, at an expected aggregate of roughly 1.6 GiB
before operating-system sharing and overhead.

## Applying the method to Acrobot

The current shift changes `LINK_LENGTH_1` from 0.5 to 2.0, with nominal value
1.0. Acrobot has six observation dimensions and three actions. The current
6-bin specification has:

```text
6 ** 6 * 3 = 139,968 possible state-action IDs.
```

This ID space is smaller than LunarLander's 4-bin grid, but nominal trajectories
may occupy it differently. Do not copy LunarLander's `min_calib=100` without an
occupancy sweep.

Suggested initial coverage sweep:

```text
calibration steps: 2,500, 5,000, 10,000, 20,000
grid bins:         3, 4, 6, 8
min_calib probes:  20, 40, 80, 100, 200
obs_quantile:      0.05, 0.10, 0.20
```

Start with TD scores and signed error. Add Monte Carlo only after the TD
pipeline is behaving sensibly.

Acrobot returns are negative episode lengths. Higher (less negative) is better:
`-80` is better than `-150`. Use paired return differences directly. Do not use
a raw calibrated/reference return ratio, because division reverses or obscures
the meaning of negative rewards.

Useful Acrobot checks:

- confirm cached policies actually solve or nearly solve the nominal task;
- examine whether calibration helps by shortening episodes or merely changes
  truncation frequency;
- include both shorter and longer link shifts in the representative subset;
- preserve episode-level returns, since many runs may pile up at the time
  limit; and
- ensure shift attributes are applied to `env.unwrapped` as the current
  environment adapter expects.

Because Acrobot episodes can be long, evaluation rather than calibration is
likely to dominate runtime. A representative-shift screen saves substantial
compute.

## Applying the method to MountainCar

MountainCar has two observation dimensions and three actions. Its current
10-bin grid has only:

```text
10 ** 2 * 3 = 300 possible state-action IDs.
```

The main concern is not Cartesian explosion. It is whether a deterministic
policy repeatedly visits a narrow path and creates highly imbalanced cell
counts.

Suggested initial coverage sweep:

```text
calibration steps: 2,500, 5,000, 10,000, 20,000
grid bins:         6, 8, 10, 12, 20
min_calib probes:  20, 50, 100, 200, 500
obs_quantile:      0.00, 0.05, 0.10
```

The observation space is naturally bounded, so `obs_quantile=0.0` is a
reasonable candidate. Compare it with quantile-fitted bounds rather than
assuming nominal visitation should define the whole grid.

MountainCar returns are also negative episode lengths, commonly with a hard
floor at the time limit. As with Acrobot:

- higher return is better;
- use paired differences, not ratios;
- report the fraction of episodes reaching the goal; and
- distinguish a meaningful improvement from moving a few returns off the
  truncation boundary.

The gravity shift spans 0.001 to 0.005 with nominal 0.0025. A representative
subset should include both endpoints, nominal, and values on each side of
nominal. Check float matching carefully when joining results to reference
policies.

If the nominal DQN never solves MountainCar for some seeds, calibration results
for those seeds answer a different question: whether correction can rescue a
poor policy. Report baseline quality explicitly rather than pooling solved and
unsolved policies without comment.

## Optuna Bayesian optimization

The maintained tuner is [`scripts/optuna_tuner.py`](scripts/optuna_tuner.py).
It uses Optuna's multivariate TPE sampler and a median pruner. The historical
tuner remains useful for provenance:

```bash
git show 90219da:notebooks/experiments/optuna_tuner.py
```

The current script:

1. evaluates each original DQN once on the nominal environment and excludes
   seeds whose mean return is below the environment threshold;
2. caches one maximum-length chronological buffer per eligible development
   seed;
3. fingerprints model weights so stale eligibility checks, buffers, or resumed
   studies are rejected;
4. uses prefixes for trial-specific calibration budgets;
5. precomputes and caches paired baseline returns;
6. samples grid bins, calibration budget, `min_calib`, alpha, and
   `obs_quantile`;
7. evaluates eligible development seeds in one fixed shuffled order;
8. prunes only after the configured minimum number of eligible seeds;
9. records seed/shift deltas, calibrated and visited cells, and fallbacks as
   trial attributes;
10. stores the resumable study in SQLite; and
11. qualifies and evaluates optional validation seeds only after optimization
    is complete.

The default scalar objective is:

```text
0.5 * mean(seed_delta)
+ 0.5 * median(seed_delta)
- 0.25 * excess_nominal_loss
- 0.10 * excess_worst_seed_loss_beyond_10
```

All terms and tolerances are exposed on the CLI. Use `--objective-mode mean`
with zero penalty weights to recover a plain mean-delta objective.

### Recommended LunarLander study

All seeds 0-24 have now been observed, so a new paper-quality study needs new
validation policies. Seeds 0-9 can serve as development data, with newly
trained seeds 25-44 reserved for validation.

Run the study first without validation seeds:

```bash
uv run python scripts/optuna_tuner.py \
  --env-name LunarLander-v3 \
  --development-seeds 0 1 2 3 4 5 6 7 8 9 \
  --num-trials 80 \
  --num-tuning-shifts 8 \
  --tuning-eval-episodes 10 \
  --startup-trials 16 \
  --min-seeds-before-prune 4 \
  --n-jobs 2 \
  --results-out optuna/LunarLander-v3/sparse_grid_v1
```

Inspect `trials.json` and `best_config.yaml` before opening any validation
results. To validate the already-selected best trial, repeat the same command
with the target trial count unchanged and add:

```bash
--validation-seeds 25 26 27 28 29 30 31 32 33 34 \
  35 36 37 38 39 40 41 42 43 44
```

The resumed study will add no trials when it already contains the target count.
It will evaluate the best candidate over the full environment shift range,
using `validation_eval_episodes`, and save `best_validation.pkl` plus
`best_validation_summary.yaml`.

### Resume and parallelism

`--num-trials` is the target total number of stored trials, not the number to
append. The study records a search signature and model fingerprints. Changing
development seeds, the eligibility threshold or budget, shift selection,
objective, search bounds, evaluation budget, or model weights requires a new
study name or results directory.

`--n-jobs` uses Optuna threads within one process. The models and cached
buffers are shared read-only, which is much less memory-intensive than loading
one model per trial process. Start with one or two jobs for reproducibility and
raise this only after measuring CPU saturation. The default
`--torch-threads 1` avoids nested CPU oversubscription.

Bayesian optimization does not remove the need for a held-out design. Never
add validation seeds to `development_seeds`, select a different trial after
reading validation output, or silently resume a study against changed model
weights.

## Reporting checklist

Before describing an environment as successfully tuned, record:

- [ ] exact model cache/training provenance;
- [ ] requested, eligible, and excluded development and held-out seeds;
- [ ] nominal qualification threshold, episode count, and raw returns;
- [ ] shift values and nominal value;
- [ ] evaluation seed convention;
- [ ] raw episode returns;
- [ ] all calibration hyperparameters;
- [ ] possible, visited, and calibrated cell counts;
- [ ] fallback correction per seed;
- [ ] baseline and calibrated return per seed and shift;
- [ ] mean and median paired seed delta;
- [ ] seed and seed-by-shift win/loss/tie counts;
- [ ] nominal and worst-shift effects;
- [ ] wall time and peak RSS per worker;
- [ ] unsuccessful configurations and failure modes;
- [ ] full test-suite result; and
- [ ] a post-change default CLI smoke run.

Generated models and results belong under `models/` and `results/`. Stable
methodology and reusable analysis logic should remain in tracked Markdown,
scripts, or `src/crl/` modules rather than only in notebooks.
