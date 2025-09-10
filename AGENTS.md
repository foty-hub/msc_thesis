# Repository Guidelines

## Project Structure & Module Organization
- `src/crl/`: core library code (agents, discretise, utils, configs, env, buffer, types, calib).
- `src/crl/configs/`: YAML configs named by Gymnasium env IDs (e.g., `CartPole-v1.yml`).
- `notebooks/`: experiments and analyses; move stable logic into `src/crl/`.
- `tests/`: `pytest` suites grouped by area (`agents/`, `envs/`, `predictors/`).
- `results/`, `models/`, `wandb/`: generated artifacts; gitignored by default.

## Build, Test, and Development Commands
- Environment (Python ≥ 3.12) via `uv`:
  - `uv sync` — create env and install dependencies.
  - `uv pip install -e .` — editable install to expose `crl` package.
- Run tests:
  - `uv run pytest -q` — full suite.
  - `uv run pytest tests/crl/envs/test_frozen_lake.py::test_no_slip_with_zero_prob` — single test.
- Notebooks:
  - `uv run jupyter lab` — open notebooks with project kernel.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4‑space indentation and type hints.
- Names: modules/files `snake_case`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Keep config files small, declarative, and environment‑scoped (e.g., `src/crl/configs/LunarLander-v3.yml`).
- Prefer pure functions and small modules under `src/crl/`; move notebook‑only code into library modules when reused.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/<area>/test_*.py`; test functions `test_*`.
- Determinism: set RNG seeds (e.g., `np.random.seed(42)`); avoid network calls. If logging, set `WANDB_MODE=offline`.
- Scope: include unit tests for new modules and update affected tests when changing behaviors.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars), optional body. Group related changes. Example: `fix: stabilize slip wrapper`.
- Link issues/PRs using `#<id>` when relevant.
- PRs: clear description, motivation, reproduction steps (configs/commands), and test coverage. Include plots/screenshots for results changes and note any updated notebooks.

## Security & Configuration Tips
- Keep secrets in `.env` (e.g., W&B API key); never commit them.
- Large artifacts in `results/` and `models/` are ignored—do not override ignore rules.
