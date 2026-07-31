from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

import gymnasium as gym
import torch
from torch import nn
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.env import MINATAR_BREAKOUT, register_minatar
from crl.types import ClassicControl
from crl.utils.paths import get_models_dir

Agent = TypeVar("Agent", bound=DQN)


class MinAtarCNN(BaseFeaturesExtractor):
    """The small convolutional network used for 10x10 MinAtar observations."""

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[-1]
        self.network = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.permute(0, 3, 1, 2))


def make_training_env(env_name: ClassicControl) -> gym.Env:
    """Create a headless environment; rendering slows training substantially."""
    if env_name == MINATAR_BREAKOUT:
        register_minatar()
    return gym.make(env_name)


def load_dqn_args(
    env_name: ClassicControl,
    total_timesteps: int | None = None,
) -> dict:
    config_path = (
        Path(__file__).resolve().parent / ".." / "configs" / f"{env_name}.yml"
    )
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    if env_name == MINATAR_BREAKOUT:
        config["exploration_fraction"] = min(
            1.0,
            100_000 / float(total_timesteps or 500_000),
        )
        config["policy_kwargs"]["features_extractor_class"] = MinAtarCNN
    return config


def model_basename(
    env_name: ClassicControl,
    seed: int,
    total_timesteps: int,
) -> str:
    if env_name != MINATAR_BREAKOUT and total_timesteps == 50_000:
        return f"model_{seed}"
    return f"model_{seed}_steps_{total_timesteps}"


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


def seeded_vec_env(model: DQN, seed: int) -> VecEnv:
    """Seed the next reset, including after loading a cached policy."""
    env = cast(VecEnv, model.get_env())
    env.seed(seed)
    return env
