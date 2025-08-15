# %%
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
import os
from typing import Literal
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import yaml
from pathlib import Path


class CQLDQN(DQN):
    """
    Corrected Conservative Q-Learning (CQL) implementation.
    """

    def __init__(self, cql_alpha: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = cql_alpha

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


ClassicControl = Literal[
    "CartPole-v1", "Acrobot-v1", "LunarLander-v3", "MountainCar-v0"
]


def _alpha_to_fname(alpha: float) -> str:
    """
    Format alpha for filenames: 1.0 -> '1-0', 0.05 -> '0-05'.
    Avoids dots in filenames that can confuse some tooling.
    """
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "-")


def instantiate_cql_dqn(
    env_name: ClassicControl,
    seed: int = 0,
    cql_alpha: float = 0.0,
) -> CQLDQN:
    env = gym.make(env_name, render_mode="rgb_array")

    # using SB3 zoo suggested hyperparameters (path relative to this file)
    config_path = Path(__file__).resolve().parent / "configs" / f"{env_name}.yml"
    with open(config_path, "r") as f:
        dqn_args = yaml.safe_load(f)

    model = CQLDQN(env=env, seed=seed, cql_alpha=cql_alpha, **dqn_args)
    return model


def learn_cqldqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
    cql_alpha: float = 0.0,
    total_timesteps: int = 50_000,
    model_dir: str = "models",
    train_from_scratch: bool = False,
) -> tuple[CQLDQN, VecEnv]:
    # Path for caching the trained model
    model_dir = os.path.join(model_dir, env_name, "cqldqn")
    os.makedirs(model_dir, exist_ok=True)
    model_basename = f"model_{seed}_alpha_{_alpha_to_fname(cql_alpha)}"
    model_path = os.path.join(model_dir, model_basename)

    # Load cached model if available and not training from scratch
    if not train_from_scratch and os.path.exists(model_path + ".zip"):
        print(f"Loading CQLDQN model: {seed}, alpha: {cql_alpha}")
        model = CQLDQN.load(model_path)
        # Attach a fresh environment so the model is usable immediately
        env = gym.make(env_name, render_mode="rgb_array")
        model.set_env(env)
    else:
        # Train a new model from scratch
        print(f"Learning CQLDQN from scratch: {seed}, alpha: {cql_alpha}")
        model = instantiate_cql_dqn(env_name, seed, cql_alpha)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_path)

    # Retrieve the vectorised environment
    vec_env = model.get_env() if model.get_env() is not None else model.env
    return model, vec_env


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
