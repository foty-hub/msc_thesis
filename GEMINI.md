# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the project structure, dependencies, and key components of this MSc thesis project on Conformal Prediction for Reinforcement Learning.

## Project Overview

This project is a research codebase for an MSc thesis focused on applying Conformal Prediction to Reinforcement Learning (RL) to improve agent robustness under distributional shift. The core idea is to use conformal prediction to calibrate the value estimates of a Deep Q-Network (DQN), providing statistically rigorous lower bounds on the Q-values. This allows the agent to act more conservatively and safely when it encounters novel situations. The project also explores Conservative Q-Learning (CQL) as a complementary technique for learning robust policies.

The project is primarily written in Python and uses the following key technologies:

*   **Core ML/RL:** PyTorch, JAX, Flax, Optax, Gymnasium, Stable Baselines3
*   **Data Science & Plotting:** NumPy, Pandas, Matplotlib, Seaborn
*   **Experiment Tracking:** Weights & Biases (wandb)

The codebase is structured into several key directories:

*   `notebooks/`: Contains Jupyter notebooks for experiments and analysis.
*   `src/crl/`: Contains the core source code for the Conformal RL (CRL) library.
*   `tests/`: Contains unit tests for the `crl` library.
*   `results/`: Contains results from experiments, including figures and model checkpoints.

## Building and Running

### Installation

The project uses `uv` for dependency management. To install the required packages, run:

```bash
uv sync
uv pip install -e .
```

### Running Experiments

Experiments are primarily run through Python scripts and Jupyter notebooks. The main entry points for running experiments are in the `distribution_shift.py` file and the notebooks in the `notebooks/` directory.

For example, to run a training experiment with the DQN agent under a distributional shift, you can use the following command:

```bash
python distribution_shift.py train --env-name LunarLander-v3 --shift noise
```

### Running Tests

The project uses `pytest` for testing. To run the test suite, use the following command:

```bash
uv run pytest
```

## Development Conventions

*   **Code Style:** The code generally follows the PEP 8 style guide.
*   **Testing:** Unit tests are located in the `tests/` directory and are written using `pytest`.
*   **Dependencies:** Project dependencies are managed in the `pyproject.toml` file.
*   **Notebooks:** Experimental work and analysis are often done in Jupyter notebooks in the `notebooks/` directory before being moved to Python scripts.
*   **Modularity:** The core logic is encapsulated in the `src/crl/` directory, with clear separation of concerns for agents, environments, and conformal prediction logic.
