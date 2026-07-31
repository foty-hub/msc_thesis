# Conformal Calibration

Repo for my MSc Thesis devising and explaining '[Conformal Calibration](https://www.alexinch.com/assets/pdfs/ucl_msc_thesis.pdf)' - a method for calibrating the Q-function of a DQN-based agent, improving its robustness to domain shift on some tasks.

## Setup

The project requires Python 3.12 or newer and uses [`uv`](https://docs.astral.sh/uv/) for dependency management. To setup the virtual env, run

```bash
uv sync
```

Trained policies are cached under `models/` by default. Set `MODELS_DIR` in a local `.env` file to use another location.


## Run Experiments

### Basic Evaluation

To run a single seeded CartPole robustness experiment:

```bash
uv run scripts/cli.py \
  --env-name CartPole-v1 \
  --results-out CartPole-v1/RUN_NAME
```

If there are no cached models saved, this will train a new DQN for each of 25 seeds and then evaluate it. If there are saved models, then the script will load them first. To see a list of args to the CLI script, run


```bash
uv run scripts/cli.py --help
```


### MinAtar
To compare 500k- and 5M-step MinAtar policies over three seeds:

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


### Per-shift reference policies

The reference-policy runner trains fresh policies directly for each parameter shifts. It runs a few seeds and saves raw episode returns, so you can compute the calibrated returns as a ratio to an agent trained directly in that environment.

```bash
uv run python scripts/optimal_policies.py \
  CartPole-v1 \
  --seeds 0 1 2 3 4
```


## Repository layout

- `src/crl/calib.py`: transition collection, batched conformity scores, and
  sparse conformal corrections.
- `src/crl/discretise/grid.py`: sparse mixed-radix state-action grid.
- `src/crl/experiment.py`: calibration and paired shift evaluation pipeline.
- `src/crl/agents/`: DQN, DDQN, and CQL-DQN training/loading.
- `src/crl/configs/`: environment-specific DQN hyperparameters from SB Zoo.
- `scripts/`: runnable experiment entry points.
- `tests/crl/`: tests.

Generated models, experiment results, profiles, and local environment files are
excluded from version control.

## Tests

```bash
uv run pytest -q
```
