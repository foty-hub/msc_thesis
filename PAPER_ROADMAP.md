# Repository overview and paper roadmap

## What this repository currently does

The maintained experiment applies conformal corrections to the action values of
a trained DQN. It:

1. collects nominal-policy states and fits a fixed state-action grid;
2. collects a disjoint calibration rollout;
3. computes TD or Monte Carlo conformity scores;
4. stores conformal corrections only for visited grid cells; and
5. compares the original and corrected greedy policies under dynamics shifts.

The current implementation covers Gymnasium classic-control environments and
DQN, Double DQN, and CQL-DQN. CC-NN, FAISS, tile coding, and tree
discretisation have been removed.

The core path is:

- `src/crl/discretise/grid.py`: grid fitting and state-action cell IDs.
- `src/crl/calib.py`: rollouts, scores, calibration sets, and quantiles.
- `src/crl/experiment.py`: calibration, action selection, and shift evaluation.
- `notebooks/experiments/traintime_robustness.py`: multi-seed experiment.
- `notebooks/experiments/cli.py`: command-line entry point.
- `tests/crl/`: unit and end-to-end regression tests.

The cleanup checkpoint passes the test suite and a deterministic CartPole run.
For one trained seed, using 25 episodes at each of 15 pole-length values,
calibration improved 14 shifts and tied one. This is a functional regression
check, not sufficient evidence for the paper.

## Next paper steps

### 1. Finalise the grid-only experiment

Keep a single terminology throughout the code and paper, preferably
`grid calibration`. Before large runs, settle the grid resolution,
`obs_quantile`, calibration size, `min_calib`, alpha, scoring method, and
fallback rule. These choices currently live in:

- `GridCalibrationConfig` in `src/crl/experiment.py`;
- `RobustnessConfig` in
  `notebooks/experiments/traintime_robustness.py`; and
- environment grids and shift ranges in `SHIFT_SPECS`, in
  `src/crl/experiment.py`.

Add tests before changing the cell-ID scheme or conformal quantile calculation;
those are the easiest places to silently change the method.

### 2. Measure performance relative to a policy trained at each shift

`notebooks/experiments/optimal_policies.py` already trains fresh reference
policies in every shifted environment and stores raw returns. It reports the
mean across training seeds rather than selecting the luckiest seed.

The remaining work is to join those reference returns to the robustness results
by environment, shift parameter, and shift value. Add the metric and aggregation
in a new library module (for example `src/crl/metrics.py`) rather than embedding
it in a plotting notebook.

Avoid a raw ratio `return / reference_return`: Acrobot and MountainCar have
negative rewards, so that ratio is misleading. Reasonable candidates are:

- regret: `reference_return - evaluated_return`; or
- normalized return using an explicit lower anchor, such as a random-policy
  return.

If using fraction of the baseline-to-reference gap closed, handle shifts where
the baseline already matches or exceeds the empirical reference. Preserve raw
episode returns and compute uncertainty across both training and evaluation
seeds.

### 3. Add a larger state-space task such as MinAtar

Do not apply the current full Cartesian grid directly to MinAtar pixels. The
number of cells grows exponentially, and the current mixed-radix `int64` cell ID
can overflow in high dimensions even though storage is sparse.

First decide what representation will be discretised, for example:

- a small fixed latent representation from the policy network;
- a deliberately selected subset of state features; or
- a compact, reproducible projection fitted only on nominal grid data.

This will mainly touch:

- `src/crl/discretise/grid.py` for high-dimensional/hashable cell keys;
- `src/crl/calib.py` for model-independent feature extraction;
- `src/crl/experiment.py` for environment and shift adapters;
- `src/crl/types.py` and `src/crl/configs/` for the new environment; and
- `pyproject.toml` for MinAtar dependencies.

Keep representation fitting, conformal calibration, and evaluation data
separate to avoid leakage.

### 4. Support an actor–critic policy

The present method assumes a DQN-like model with `q_net`, a discount factor, and
one score per discrete action. A conventional PPO/A2C critic estimates
`V(s)`, not `Q(s, a)`, so simply subtracting corrections from policy logits
would change the meaning of the method.

Choose the statistical object first: action values, advantages, or policy
logits. Then introduce a small model adapter exposing operations such as:

- batched action scores;
- selected actions;
- discount/bootstrapping information; and
- the representation used by the grid.

Replace direct `model.q_net` access in `src/crl/calib.py` and
`src/crl/experiment.py` with that adapter. Add the actor–critic implementation
under `src/crl/agents/` and create adapter-level tests before running a large
experiment. A discrete-action actor–critic is the smallest first step.

## Compute priorities

Several inexpensive improvements are already present: sparse visited-cell
storage, batched score inference, `torch.inference_mode()`, headless training,
model caching for the value agents, bounded worker counts, and deterministic
calibration seeds.

The next useful optimisations are:

1. cache and resume the per-shift models in `optimal_policies.py`;
2. batch action selection across evaluation environments instead of doing one
   network forward pass per environment step;
3. set PyTorch threads per worker to avoid CPU oversubscription in multi-seed
   runs;
4. use small smoke budgets before launching the paper configuration; and
5. persist per-seed results progressively so an interrupted sweep can resume.

Generated models and results belong under `models/` and `results/`; both are
gitignored. Editable schematic notebooks and Draw.io sources are retained.
