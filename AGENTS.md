# Repository Guidelines

## Project Structure & Module Organization
- `src/crl/`: library code
  - `agents/`, `predictors/`, `envs/`, `cons/` (configs, discretisers, RL algos), `utils/`.
- `tests/`: pytest suite mirroring `crl` (e.g., `tests/crl/envs/...`).
- `notebooks/`: exploratory experiments (being migrated into `src/`).
- `models/`, `results/`, `images/`, `data/`: saved artifacts, run outputs, figures, datasets.
- Top-level scripts (e.g., `2021_05_07_dqn_lunarlander.py`) are runnable experiment entry points.

## Build, Test, and Development Commands
- Install (uv): `uv sync && uv pip install -e .` — resolves deps (Python ≥3.12) and installs package in editable mode.
- Run tests: `uv run pytest -q` — executes the full test suite.
- Focused tests: `uv run pytest tests/crl/envs/test_frozen_lake.py -k slip` — run a subset quickly.

## Coding Style & Naming Conventions
- Python style: PEP 8, 4‑space indentation, 88–100 col soft limit.
- Naming: modules/files/functions `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE`.
- Imports: stdlib → third‑party → local; prefer absolute imports within `crl`.
- Typing: add type hints to new/modified public APIs; include short docstrings (what/why over how).
- Formatting: no strict tool pinned; keep consistent. If you use formatters, align with PEP 8 and sorted imports.

## Testing Guidelines
- Framework: `pytest` with simple, deterministic tests; seed randomness where applicable.
- Layout: mirror package paths under `tests/`; name files `test_*.py`, functions `test_*`.
- Coverage: not enforced; prioritize core logic (e.g., predictors, env wrappers). Add regression tests with minimal fixtures.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood; include scope when helpful (e.g., `agents: add ddqn implementation`, `envs: tweak cartpole lr`). Group related changes; avoid mixing refactors with features.
- PRs: clear description of problem and approach; link issues; include run commands and configs (e.g., YAMLs in `src/crl/cons/configs/`), and attach key results/plots under `results/` or `images/`.
- Checks: ensure `uv run pytest` passes; note breaking changes and migration steps.

## Configuration & Repro Tips
- Configs: prefer YAMLs in `src/crl/cons/configs/` for algorithm/env params; commit small configs.
- Repro: record seeds, package versions, and commands; avoid committing large artifacts; keep credentials out of git.

