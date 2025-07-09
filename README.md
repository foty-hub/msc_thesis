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

Currently, all the experiments live in notebooks in the `notebooks/` dir. These are being moved into proper `.py` files as the structure begins to coalesce.


## Tests
```bash
>>> uv run pytest
```