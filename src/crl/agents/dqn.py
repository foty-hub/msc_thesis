from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.agents._common import (
    cached_model_path,
    load_cached_agent,
    load_dqn_args,
    make_training_env,
    model_basename,
    seeded_vec_env,
)
from crl.env import MINATAR_BREAKOUT
from crl.types import ClassicControl


def instantiate_vanilla_dqn(
    env_name: ClassicControl,
    seed: int = 0,
    total_timesteps: int | None = None,
) -> DQN:
    return DQN(
        env=make_training_env(env_name),
        seed=seed,
        **load_dqn_args(env_name, total_timesteps),
    )


def learn_dqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
    total_timesteps: int = 50_000,
    model_dir: str | Path | None = None,
    train_from_scratch: bool = False,
) -> tuple[DQN, VecEnv]:
    """Load a cached DQN or train and cache one without rendering overhead."""
    model_path = cached_model_path(
        env_name,
        "dqn",
        model_basename(env_name, seed, total_timesteps),
        model_dir,
    )
    if not train_from_scratch and model_path.with_suffix(".zip").exists():
        print(f"Loading DQN model: {seed}")
        model = load_cached_agent(DQN, model_path, env_name)
    else:
        print(f"Learning DQN from scratch: {seed}")
        model = instantiate_vanilla_dqn(env_name, seed, total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=env_name == MINATAR_BREAKOUT,
        )
        model.save(str(model_path))
    return model, seeded_vec_env(model, seed)
