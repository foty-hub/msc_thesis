# %%
import os
import pickle
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from crl.cons.agents.cql import CQLDQN

# Cartpole args
# dqn_args = {
#     "policy": "MlpPolicy",
#     "learning_rate": 2.3e-3,
#     "batch_size": 64,
#     "learning_starts": 1_000,
#     "buffer_size": 100000,
#     "train_freq": 256,
#     "gamma": 0.99,
#     "target_update_interval": 10,  # gradient‑step interval
#     "gradient_steps": 128,
#     "exploration_fraction": 0.16,
#     "exploration_final_eps": 0.04,
#     "policy_kwargs": dict(net_arch=[256, 256]),
# }

# # MountainCar args
# dqn_args = {
#     "policy": "MlpPolicy",
#     "learning_rate": 4.0e-3,
#     "batch_size": 128,
#     "buffer_size": 10000,
#     "learning_starts": 1000,
#     "gamma": 0.98,
#     "target_update_interval": 600,
#     "train_freq": 16,
#     "gradient_steps": 8,
#     "exploration_fraction": 0.2,
#     "exploration_final_eps": 0.07,
#     "policy_kwargs": dict(net_arch=[256, 256]),
# }


# Lunar Lander args
dqn_args = {
    "policy": "MlpPolicy",
    "learning_rate": 6.3e-4,
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_starts": 0,
    "gamma": 0.99,
    "target_update_interval": 250,
    "train_freq": 4,
    "gradient_steps": -1,
    "exploration_fraction": 0.12,
    "exploration_final_eps": 0.1,
    "policy_kwargs": dict(net_arch=[256, 256]),
}


def single_steps_experiment(train_steps: int, log_dir: str) -> list[dict[str, Any]]:
    exp_results = []
    # for alpha in tqdm(np.arange(0.0, 2.1, 0.1), leave=False, desc=f"{train_steps:,}"):
    for alpha in tqdm([0.0, 0.1, 0.5, 1.0, 2.0], leave=False):
        env = gym.make("LunarLander-v3")
        env = Monitor(env, log_dir)
        agent = CQLDQN(
            env=env,
            seed=1,
            cql_alpha=alpha,
            **dqn_args,
        )

        agent.learn(total_timesteps=train_steps, progress_bar=True)

        returns, ep_lengths = evaluate_policy(
            agent,
            agent.get_env(),
            n_eval_episodes=100,
            return_episode_rewards=True,
        )
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        print(f"cql ({alpha}): {mean_r:.1f} ± {std_r:.1f}")

        # Helper from the library
        results_plotter.plot_results(
            [log_dir], train_steps, results_plotter.X_TIMESTEPS, task_name=alpha
        )

        exp_results.append(
            {
                "train_steps": train_steps,
                "alpha": alpha,
                "returns": returns,
            }
        )
    return exp_results


# %%

all_results = []
for train_steps in [100_000]:
    temp_dir = os.environ["TMPDIR"]  # macos temporary logging dir
    log_dir = f"{temp_dir}/gym/"
    os.makedirs(log_dir, exist_ok=True)
    exp_result = single_steps_experiment(train_steps, log_dir)
    all_results.extend(exp_result)

    # # Save results to pickle file
    # with open("cql_results_withreturns.pkl", "wb") as f:
    #     pickle.dump(all_results, f)

    # for seed in range(25):
    #     env = gym.make("CartPole-v1")  # No rendering during training
    #     env = Monitor(env, log_dir)
    #     agent = CQLDQN(
    #         env=env,
    #         seed=seed,
    #         **dqn_args,
    #     )

    #     agent.learn(total_timesteps=timesteps, progress_bar=True)

# %%
