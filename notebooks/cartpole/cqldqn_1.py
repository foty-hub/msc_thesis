# %%
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer

# Import polyak_update for the soft target network update
from stable_baselines3.common.utils import polyak_update
from typing import Optional, List
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy


class CQLDQN(DQN):
    """
    Final Corrected Conservative Q-Learning (CQL) implementation.
    """

    def __init__(self, cql_alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = cql_alpha
        # SB3 uses tau for Polyak updates. Ensure it's set.
        if "tau" not in kwargs:
            self.tau = 1.0  # Default for DQN if not provided

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Main gradient step loop
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with torch.no_grad():
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values_target.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            q_values_in_distribution = torch.gather(
                current_q_values, dim=1, index=replay_data.actions
            )
            td_loss = F.smooth_l1_loss(q_values_in_distribution, target_q_values)

            # --- CQL regularizer ---
            log_sum_exp_q_values = torch.logsumexp(
                current_q_values, dim=1, keepdim=True
            )
            cql_regularizer = (log_sum_exp_q_values - q_values_in_distribution).mean()

            # --- Combined loss ---
            loss = td_loss + (self.cql_alpha * cql_regularizer)

            # --- Optimization step ---
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Increment the number of updates
            self._n_updates += 1

            # --- CORRECTED: Perform Polyak update of the target network ---
            # This is done inside the loop, after each gradient step.
            # The 'target_update_interval' in DQN is based on gradient steps, not env steps.
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    self.q_net.parameters(), self.q_net_target.parameters(), self.tau
                )


# %%
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    dqn_args = {
        "policy": "MlpPolicy",
        "learning_rate": 2.3e-3,
        "batch_size": 64,
        "learning_starts": 1000,
        "gamma": 0.99,
        "target_update_interval": 10,
        "train_freq": 256,
        "gradient_steps": 128,
        "exploration_fraction": 0.16,
        "exploration_final_eps": 0.04,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }

    # model = DQN(env=env, seed=seed, **dqn_args)

    # model.learn(total_timesteps=50_000, progress_bar=True)
    # vec_env = model.get_env()
    cql_agent = CQLDQN(
        env=env,
        cql_alpha=2.0,  # Manually-tuned alpha
        verbose=0,
        seed=3,
        **dqn_args,
    )

    # Train the agent. Note: `learn` will not step the environment because
    # `train_freq` is met at every call, and the buffer is already full.
    cql_agent.learn(total_timesteps=50_000, progress_bar=True)

# %%
env.close()
print("a")
# %%
from stable_baselines3.common.evaluation import evaluate_policy
from traintime_robustness import instantiate_eval_env

for length in np.linspace(0.1, 2.0, 20):
    eval_env = instantiate_eval_env(length, 0.5)
    mean_r, std_r = evaluate_policy(cql_agent, eval_env, n_eval_episodes=10)
    print(f"{length:.1f}: {mean_r:.1f} ± {std_r:.1f}")
# vec_env = cql_agent.get_env()
# evaluate_policy(cql_agent, vec_env)
# %%
