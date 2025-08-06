# %%
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from traintime_robustness import instantiate_eval_env


class CQLDQN(DQN):
    """Conservative Q-Learning (CQL) agent built on SB3 DQN.

    Adds a CQL regulariser with adaptive α and optional random-action
    augmentation.  Suitable for both online and offline discrete-action
    tasks.
    """

    def __init__(
        self,
        *args,
        cql_alpha: float = 1.0,
        cql_target_gap: float = 0.0,
        cql_alpha_lr: float = 1e-4,
        num_random_actions: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # --- CQL hyper-parameters ---
        self.num_random_actions = num_random_actions
        self.cql_target_gap = cql_target_gap

        # α is trainable (dual gradient ascent) and kept non‑negative.
        self.log_cql_alpha = torch.tensor(
            np.log(cql_alpha),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.cql_alpha_optim = torch.optim.Adam([self.log_cql_alpha], lr=cql_alpha_lr)

        # Default τ (Polyak factor) – if the parent ctor did not set it.
        if not hasattr(self, "tau"):
            self.tau = 1.0

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # noqa: D401
        """Perform one optimisation phase consisting of *gradient_steps* mini‑batches."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # -----------------------------------------------------------
            # 1.  TD target
            # -----------------------------------------------------------
            with torch.no_grad():
                next_q = self.q_net_target(replay_data.next_observations)
                next_q_max = next_q.max(dim=1, keepdim=True)[0]
                target_q = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_max
                )

            # -----------------------------------------------------------
            # 2.  Current Q estimates
            # -----------------------------------------------------------
            current_q_all = self.q_net(replay_data.observations)
            batch_q = torch.gather(current_q_all, 1, replay_data.actions)

            td_loss = F.smooth_l1_loss(batch_q, target_q)

            # -----------------------------------------------------------
            # 3.  CQL regulariser
            # -----------------------------------------------------------
            logsumexp_q = torch.logsumexp(current_q_all, dim=1, keepdim=True)

            # Optional random‑action penalty (improves offline stability)
            if self.num_random_actions > 0:
                rand_actions = torch.randint(
                    0,
                    self.action_space.n,
                    (batch_size, self.num_random_actions, 1),
                    device=self.device,
                )
                # Broadcast observations to (B * k, obs_dim)
                obs_expanded = (
                    replay_data.observations.unsqueeze(1)
                    .expand(-1, self.num_random_actions, -1)
                    .reshape(-1, *replay_data.observations.shape[1:])
                )
                q_rand_all = self.q_net(obs_expanded)
                q_rand = torch.gather(q_rand_all, 1, rand_actions.reshape(-1, 1)).view(
                    batch_size, self.num_random_actions
                )
                rand_penalty = q_rand.mean(dim=1, keepdim=True)
            else:
                rand_penalty = 0.0

            cql_gap = logsumexp_q - batch_q - rand_penalty  # shape (B,1)
            cql_regulariser = cql_gap.mean()

            # -----------------------------------------------------------
            # 4.  Q‑network update (α treated as constant here)
            # -----------------------------------------------------------
            alpha = self.log_cql_alpha.exp().detach()
            loss_q = td_loss + alpha * cql_regulariser

            self.policy.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # -----------------------------------------------------------
            # 5.  α update (dual objective)
            # -----------------------------------------------------------
            alpha_loss = -(
                self.log_cql_alpha * (cql_regulariser.detach() - self.cql_target_gap)
            )
            self.cql_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.cql_alpha_optim.step()
            # Keep α ≥ 0
            with torch.no_grad():
                self.log_cql_alpha.data.clamp_(min=-20.0, max=20.0)

            # Book‑keeping
            self._n_updates += 1

            # -----------------------------------------------------------
            # 6.  Target‑network Polyak update
            # -----------------------------------------------------------
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    self.q_net.parameters(), self.q_net_target.parameters(), self.tau
                )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def cql_alpha(self) -> float:  # noqa: D401
        """Current value of α (after the log‑param transform)."""
        return float(self.log_cql_alpha.exp().item())


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------


def main() -> None:  # noqa: D401
    env = gym.make("CartPole-v1")  # No rendering during training

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

    agent = CQLDQN(
        env=env,
        verbose=0,
        cql_alpha=1.0,
        cql_target_gap=0.0,
        num_random_actions=10,
        **dqn_args,
    )

    try:
        agent.learn(total_timesteps=50_000, progress_bar=True)
    except Exception:
        agent.get_env().close()
        raise

    mean_r, std_r = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)
    print(f"Eval reward over 10 episodes: {mean_r:.1f} ± {std_r:.1f}")

    return agent


# %%
if __name__ == "__main__":
    agent = main()

    # %%
    for length in np.linspace(0.1, 2.0, 20):
        eval_env = instantiate_eval_env(length, 0.5)
        mean_r, std_r = evaluate_policy(agent, eval_env, n_eval_episodes=10)
        print(f"{length:.1f}: {mean_r:.1f} ± {std_r:.1f}")

# %%
