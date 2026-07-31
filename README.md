# Conformal calibration for reinforcement learning

Research code for sparse-grid conformal calibration of value-based reinforcement
learning policies. The current paper-facing implementation uses a nominal-policy
rollout to fit a state-action grid and estimate conformal corrections, then
evaluates the corrected greedy policy under controlled dynamics shifts.

The maintained experiment covers Gymnasium classic-control environments and
MinAtar Breakout with DQN, Double DQN, and CQL-DQN. Actor–critic support and
reporting relative to per-shift reference policies remain planned extensions.

## Setup

The project requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Trained policies are cached under `models/` by default. Set `MODELS_DIR` in a
local `.env` file to use another location.


## Run Experiments

Run one seeded CartPole robustness experiment:

```bash
uv run python scripts/cli.py \
  --env-name CartPole-v1 \
  --debug-seed 0 \
  --results-out CartPole-v1/smoke
```

Omit `--debug-seed` for the configured multi-seed experiment. Add `--retrain`
to ignore a cached policy. Results and plots are written beneath `results/`,
which is intentionally gitignored.

Compare 500k- and 5M-step MinAtar policies over three seeds:

```bash
uv run python scripts/minatar_training_benchmark.py
```

Run a single MinAtar robustness seed with a four-dimensional, four-bin latent
grid and 50k calibration transitions:

```bash
uv run python scripts/cli.py \
  --env-name MinAtar/Breakout-v1 \
  --n-train-steps 500000 \
  --debug-seed 0
```

The tuned MinAtar configuration uses the policy's three action values as an
action-relevant representation instead of selecting a PCA variance cutoff:

```bash
uv run python scripts/cli.py \
  --env-name MinAtar/Breakout-v1 \
  --n-train-steps 500000 \
  --num-experiments 10 \
  --num-eval-episodes 25 \
  --representation-method q_values \
  --grid-bins 3 \
  --alpha 0.50 \
  --min-calib 100 \
  --n-representation-steps 10000 \
  --n-calib-steps 50000 \
  --obs-quantile 0.1
```

## Per-shift reference policies

The reference-policy runner trains fresh policies directly in every shifted
environment. It saves raw episode returns and an across-training-seed mean for
each shift:

```bash
uv run python scripts/optimal_policies.py \
  CartPole-v1 \
  --seeds 0 1 2 3 4
```

These are empirical reference returns, not guarantees of global optimality.
They are intended to support a normalized-regret robustness metric.

## Repository layout

- `src/crl/calib.py`: transition collection, batched conformity scores, and
  sparse conformal corrections.
- `src/crl/discretise/grid.py`: sparse mixed-radix state-action grid.
- `src/crl/experiment.py`: calibration and paired shift evaluation pipeline.
- `src/crl/agents/`: DQN, Double DQN, and CQL-DQN training/loading.
- `src/crl/configs/`: environment-specific DQN hyperparameters.
- `scripts/`: runnable experiment entry points.
- `tests/crl/`: deterministic unit and end-to-end tests.

Generated models, experiment results, profiles, and local environment files are
excluded from version control.


## Tests

```bash
uv run pytest -q
```
