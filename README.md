# Conformal calibration for reinforcement learning

Research code for sparse-grid conformal calibration of value-based reinforcement
learning policies. The current paper-facing implementation fits a state-action
grid from nominal-policy rollouts, estimates conformal corrections from a
disjoint calibration rollout, and evaluates the corrected greedy policy under
controlled dynamics shifts.

The maintained experiment currently covers Gymnasium classic-control
environments with DQN, Double DQN, and CQL-DQN. MinAtar, actor–critic support,
and reporting relative to per-shift reference policies are planned extensions,
not implemented paper results.

## Setup

The project requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional analysis and notebook dependencies can be installed with:

```bash
uv sync --group analysis --group notebooks
```

Trained policies are cached under `models/` by default. Set `MODELS_DIR` in a
local `.env` file to use another location.

## Validate the core method

Run the deterministic unit and end-to-end smoke tests:

```bash
uv run pytest -q
```

Run one seeded CartPole robustness experiment:

```bash
uv run python notebooks/experiments/cli.py \
  --env-name CartPole-v1 \
  --debug-seed 0 \
  --results-out CartPole-v1/smoke
```

Omit `--debug-seed` for the configured multi-seed experiment. Add `--retrain`
to ignore a cached policy. Results and plots are written beneath `results/`,
which is intentionally gitignored.

## Per-shift reference policies

The reference-policy runner trains fresh policies directly in every shifted
environment. It saves raw episode returns and an across-training-seed mean for
each shift:

```bash
uv run python notebooks/experiments/optimal_policies.py \
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
- `notebooks/experiments/`: runnable experiment entry points.
- `tests/crl/`: deterministic unit and end-to-end tests.

Generated models, experiment results, profiles, and local environment files are
excluded from version control.
