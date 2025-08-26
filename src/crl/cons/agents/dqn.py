# %%
import os
from pathlib import Path

import gymnasium as gym
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.cons._type import ClassicControl


def instantiate_vanilla_dqn(env_name: ClassicControl, seed: int = 0) -> DQN:
    env = gym.make(env_name, render_mode="rgb_array")

    # using SB3 zoo suggested hyperparameters (path relative to this file)
    config_path = Path(__file__).resolve().parent / ".." / "configs" / f"{env_name}.yml"
    with open(config_path, "r") as f:
        dqn_args = yaml.safe_load(f)

    model = DQN(env=env, seed=seed, **dqn_args)
    return model


def learn_dqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
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
        model = instantiate_vanilla_dqn(env_name, seed)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_path)

    # Retrieve the vectorised environment
    vec_env = model.get_env() if model.get_env() is not None else model.env
    return model, vec_env


# %%
def main():
    from stable_baselines3.common.evaluation import evaluate_policy

    model, vec_env = learn_dqn_policy(env_name="Acrobot-v1", seed=1)
    print(evaluate_policy(model, vec_env))


if __name__ == "__main__":
    main()
