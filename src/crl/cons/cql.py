# %%
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN


class CQLDQN(DQN):
    """
    Corrected Conservative Q-Learning (CQL) implementation.
    """

    def __init__(self, cql_alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = cql_alpha
        # SB3 uses tau for Polyak updates. Ensure it's set.
        if "tau" not in kwargs:
            self.tau = 1.0  # Default for DQN if not provided

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # bookkeeping
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Online network Q(s, ·)
            q_values_all = self.q_net(replay_data.observations)  # [B, A]
            # Q(s, a_logged) for TD target comparison
            current_q_values = torch.gather(
                q_values_all, dim=1, index=replay_data.actions.long()
            )  # [B, 1]

            # Standard DQN TD loss
            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # --- CQL(min-Q) regularizer (discrete) ---
            # E_{(s,a)~D}[Q(s,a)]
            dataset_expec = current_q_values.mean()
            # E_s[log Σ_a exp(Q(s,a))]
            negative_sampling = torch.logsumexp(q_values_all, dim=1).mean()
            cql_loss = self.cql_alpha * (negative_sampling - dataset_expec)

            # Total loss
            loss = td_loss + cql_loss

            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps


# %%
def main():
    import numpy as np
    import gymnasium as gym
    from stable_baselines3.common.evaluation import evaluate_policy

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

    for alpha in np.arange(0.0, 1.6, 0.1):
        env = gym.make("CartPole-v1")  # No rendering during training
        agent = CQLDQN(
            env=env,
            seed=1,
            cql_alpha=alpha,
            **dqn_args,
        )

        agent.learn(total_timesteps=70_000, progress_bar=False)

        mean_r, std_r = evaluate_policy(agent, agent.get_env(), n_eval_episodes=50)
        print(f"alpha={alpha:.1f}: {mean_r:.1f} ± {std_r:.1f}")


if __name__ == "__main__":
    main()

# %%
