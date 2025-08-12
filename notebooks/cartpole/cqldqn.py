import torch
import torch.nn.functional as F
from stable_baselines3 import DQN


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

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # TODO: INJECT CQL REGULARISATION INTO THE LOSS CALCULATION
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps


# vanilla DQN train method
# def train(self, gradient_steps: int, batch_size: int = 100) -> None:
#     # Switch to train mode (this affects batch norm / dropout)
#     self.policy.set_training_mode(True)
#     # Update learning rate according to schedule
#     self._update_learning_rate(self.policy.optimizer)

#     losses = []
#     for _ in range(gradient_steps):
#         # Sample replay buffer
#         replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
#         # For n-step replay, discount factor is gamma**n_steps (when no early termination)
#         discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

#         with th.no_grad():
#             # Compute the next Q-values using the target network
#             next_q_values = self.q_net_target(replay_data.next_observations)
#             # Follow greedy policy: use the one with the highest value
#             next_q_values, _ = next_q_values.max(dim=1)
#             # Avoid potential broadcast issue
#             next_q_values = next_q_values.reshape(-1, 1)
#             # 1-step TD target
#             target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

#         # Get current Q-values estimates
#         current_q_values = self.q_net(replay_data.observations)

#         # Retrieve the q-values for the actions from the replay buffer
#         current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

#         # Compute Huber loss (less sensitive to outliers)
#         loss = F.smooth_l1_loss(current_q_values, target_q_values)
#         losses.append(loss.item())

#         # Optimize the policy
#         self.policy.optimizer.zero_grad()
#         loss.backward()
#         # Clip gradient norm
#         th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
#         self.policy.optimizer.step()

#     # Increase update counter
#     self._n_updates += gradient_steps

#     self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
#     self.logger.record("train/loss", np.mean(losses))
