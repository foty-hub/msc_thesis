import gymnasium as gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


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


def learn_policy(
    model: DQN,
    total_timesteps: int = 50_000,
) -> tuple[DQN, VecEnv]:
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    vec_env = model.get_env()
    return model, vec_env
