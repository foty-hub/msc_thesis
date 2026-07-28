from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import gymnasium as gym
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.types import ClassicControl
from crl.utils.paths import get_models_dir

Agent = TypeVar("Agent", bound=DQN)


def make_training_env(env_name: ClassicControl) -> gym.Env:
    """Create a headless environment; rendering slows training substantially."""
    return gym.make(env_name)


def load_dqn_args(env_name: ClassicControl) -> dict:
    config_path = (
        Path(__file__).resolve().parent / ".." / "configs" / f"{env_name}.yml"
    )
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def cached_model_path(
    env_name: ClassicControl,
    algorithm: str,
    basename: str,
    model_dir: str | Path | None,
) -> Path:
    base_dir = Path(model_dir) if model_dir is not None else get_models_dir()
    algorithm_dir = base_dir / env_name / algorithm
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    return algorithm_dir / basename


def load_cached_agent(
    agent_class: type[Agent],
    model_path: Path,
    env_name: ClassicControl,
) -> Agent:
    return agent_class.load(
        str(model_path),
        env=make_training_env(env_name),
    )


def require_vec_env(model: DQN) -> VecEnv:
    env = model.get_env()
    if env is None:
        raise RuntimeError("The trained model has no attached environment.")
    return env


def seeded_vec_env(model: DQN, seed: int) -> VecEnv:
    """Seed the next reset, including after loading a cached policy."""
    env = require_vec_env(model)
    env.seed(seed)
    return env
