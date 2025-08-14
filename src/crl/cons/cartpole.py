# %%
import gymnasium as gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import os
from typing import Literal
import yaml
from pathlib import Path

ClassicControl = Literal[
    "CartPole-v1", "Acrobot-v1", "LunarLander-v3", "MountainCar-v0"
]


def instantiate_vanilla_dqn(
    env_name: ClassicControl, seed: int = 0, discount: float = 0.99
) -> DQN:
    env = gym.make(env_name, render_mode="rgb_array")

    # using SB3 zoo suggested hyperparameters (path relative to this file)
    config_path = Path(__file__).resolve().parent / "configs" / f"{env_name}.yml"
    with open(config_path, "r") as f:
        dqn_args = yaml.safe_load(f)

    dqn_args["gamma"] = discount

    model = DQN(env=env, seed=seed, **dqn_args)
    return model


def instantiate_eval_env(
    env_name: ClassicControl, seed: int | None = None, **kwargs
) -> VecEnv:
    """
    Instantiate a classic control evaluation environment with custom parameters.

    Parameters
    ----------
    env_name : ClassicControl
        The name of the classic control environment to instantiate.
    seed : int | None
        Random seed for reproducibility. If ``None`` the environment is left unseeded.
    **kwargs
        Custom parameters for the environment.

        CartPole-v1:
            - gravity: float (default: 9.8)
            - masscart: float (default: 1.0)
            - masspole: float (default: 0.1)
            - length: float (default: 0.5)
            - force_mag: float (default: 10.0)

        Acrobot-v1:
            - link_length_1: float (default: 1.0)
            - link_length_2: float (default: 1.0)
            - link_mass_1: float (default: 1.0)
            - link_mass_2: float (default: 1.0)
            - link_com_pos_1: float (default: 0.5)
            - link_com_pos_2: float (default: 0.5)
            - link_moi: float (default: 1.0)
            - max_vel_1: float (default: 4 * pi)
            - max_vel_2: float (default: 9 * pi)

        LunarLander-v3:
            - gravity: float (default: -10, [-12, 0])
            - wind_power: float (default: 15.0, [0, 20])
            - turbulence_power: (default: 1.5, [0, 2])

        MountainCar-v0:
            - min_position: float (default: -1.2)
            - max_position: float (default: 0.6)
            - max_speed: float (default: 0.07)
            - goal_position: float (default: 0.5)
            - goal_velocity: float (default: 0)
            - force: float (default: 0.001)
            - gravity: float (default: 0.0025)
    """
    eval_env = gym.make(env_name)

    # Validate and set custom parameters
    for key, value in kwargs.items():
        if hasattr(eval_env.unwrapped, key):
            setattr(eval_env.unwrapped, key, value)
        else:
            raise ValueError(f"Invalid parameter '{key}' for environment '{env_name}'")

    if seed is not None:
        # Seed the environment and its spaces for deterministic roll‑outs
        eval_env.reset(seed=seed)
        eval_env.action_space.seed(seed)
        eval_env.observation_space.seed(seed)

    eval_vec_env = DQN("MlpPolicy", env=eval_env).get_env()
    return eval_vec_env


def learn_dqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
    discount: float = 0.99,
    total_timesteps: int = 50_000,
    model_dir: str = "models",
    train_from_scratch: bool = False,
) -> tuple[DQN, VecEnv]:
    # Path for caching the trained model
    model_dir = os.path.join(model_dir, env_name, "dqn")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{seed}")

    # Load cached model if available and not training from scratch
    if not train_from_scratch and os.path.exists(model_path + ".zip"):
        print(f"Loading model: {seed}")
        model = DQN.load(model_path)
        # Attach a fresh environment so the model is usable immediately
        env = gym.make(env_name, render_mode="rgb_array")
        model.set_env(env)
    else:
        # Train a new model from scratch
        print(f"Learning from scratch: {seed}")
        model = instantiate_vanilla_dqn(env_name, seed, discount)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_path)

    # Retrieve the vectorised environment
    vec_env = model.get_env() if model.get_env() is not None else model.env
    return model, vec_env


# "solve" thresholds:
# CartPole    >  490
# Acrobot     > -100
# MountainCar > -110
# Pendulum    > -200
# %%
def main():
    from stable_baselines3.common.evaluation import evaluate_policy

    model, vec_env = learn_dqn_policy(env_name="Acrobot-v1", seed=1, discount=0.99)

    print(evaluate_policy(model, vec_env))


if __name__ == "__main__":
    main()
# %%
