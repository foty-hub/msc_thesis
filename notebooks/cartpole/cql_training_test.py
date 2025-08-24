# %%
import pickle
from typing import Any

import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from crl.cons.agents.cql import CQLDQN

dqn_args = {
    "policy": "MlpPolicy",
    "learning_rate": 2.3e-3,
    "batch_size": 64,
    "learning_starts": 1_000,
    "buffer_size": 100000,
    "train_freq": 256,
    "gamma": 0.99,
    "target_update_interval": 10,  # gradient‑step interval
    "gradient_steps": 128,
    "exploration_fraction": 0.16,
    "exploration_final_eps": 0.04,
    "policy_kwargs": dict(net_arch=[256, 256]),
}


def single_steps_experiment(train_steps: int) -> list[dict[str, Any]]:
    exp_results = []
    # for alpha in tqdm(np.arange(0.0, 2.1, 0.1), leave=False, desc=f"{train_steps:,}"):
    for alpha in tqdm([0.0, 1.0, 2.0], leave=False):
        env = gym.make("CartPole-v1")
        agent = CQLDQN(
            env=env,
            seed=1,  # known good seed
            cql_alpha=alpha,
            **dqn_args,
        )

        agent.learn(total_timesteps=train_steps, progress_bar=False)

        returns, ep_lengths = evaluate_policy(
            agent,
            agent.get_env(),
            n_eval_episodes=100,
            return_episode_rewards=True,
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
for train_steps in [50_000, 75_000, 80_000, 100_000]:
    exp_result = single_steps_experiment(train_steps)
    all_results.extend(exp_result)

# Save results to pickle file
with open("cql_results_withreturns.pkl", "wb") as f:
    pickle.dump(all_results, f)

# %%
