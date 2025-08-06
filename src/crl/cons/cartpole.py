import gymnasium as gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import os


def instantiate_vanilla_dqn(seed: int = 0, discount: float = 0.99) -> DQN:
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # using SB3 zoo suggested hyperparameters
    dqn_args = {
        "policy": "MlpPolicy",
        "learning_rate": 2.3e-3,
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "gamma": discount,
        "target_update_interval": 10,
        "train_freq": 256,
        "gradient_steps": 128,
        "exploration_fraction": 0.16,
        "exploration_final_eps": 0.04,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }

    model = DQN(env=env, seed=seed, **dqn_args)
    return model


def instantiate_eval_env(length, masscart):
    # attrs = ['gravity', 'force_mag', 'masscart', 'masspole', 'length', 'polemass_length']
    eval_env = gym.make("CartPole-v1")
    eval_env.unwrapped.masscart = masscart  # default 1.0
    eval_env.unwrapped.length = length  # default 0.5
    eval_vec_env = DQN("MlpPolicy", env=eval_env).get_env()
    return eval_vec_env


def learn_dqn_policy(
    seed: int = 0,
    discount: float = 0.99,
    total_timesteps: int = 50_000,
    model_dir: str = "models/cartpole/dqn",
    train_from_scratch: bool = False,
) -> tuple[DQN, VecEnv]:
    # Path for caching the trained model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{seed}")

    # Load cached model if available and not training from scratch
    if not train_from_scratch and os.path.exists(model_path + ".zip"):
        print(f"Loading model: {seed}")
        model = DQN.load(model_path)
        # Attach a fresh environment so the model is usable immediately
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        model.set_env(env)
    else:
        # Train a new model from scratch
        print(f"Learning from scratch: {seed}")
        model = instantiate_vanilla_dqn(seed, discount)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_path)

    # Retrieve the vectorised environment
    vec_env = model.get_env() if model.get_env() is not None else model.env
    return model, vec_env
