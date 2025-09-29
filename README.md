# Conformal Prediction for RL - MSc Thesis

## Installation
I strongly recommend the use of `uv` to manage dependencies - download [here](https://docs.astral.sh/uv/):
```bash
>>> uv sync
>>> uv pip install -e .
```

Otherwise, the following default pip install should work (not yet tested):
```bash
>>> python -m venv .venv
>>> source .venv/bin/activate
>>> pip install .
``` 

## Repo Structure

Currently, all the experiments live in notebooks in the `notebooks/experiments` dir. These are being moved into proper `.py` files as the structure begins to coalesce. The primary notebook is `traintime_robustness.py`, which implements and tests conformal calibration. A single-file reference implementation is on the roadmap.


## Tests
```bash
>>> uv run pytest
```

## Models Directory
- Configure a single models cache root via `.env` with `MODELS_DIR`.
- `MODELS_DIR` may be relative; it is resolved against the project root (the folder containing `pyproject.toml` or `.git`).
- Defaults to `models/` at the project root if unset.

Example `.env`:
```
MODELS_DIR=models
```
This resolves to `<repo>/models`, regardless of where you run scripts or notebooks.
