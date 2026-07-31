from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.agents._common import (
    cached_model_path,
    load_cached_agent,
    load_dqn_args,
    make_training_env,
    model_basename,
    seeded_vec_env,
)
from crl.env import MINATAR_BREAKOUT
from crl.types import ClassicControl


class CQLDQN(DQN):
    """Discrete DQN with a conservative Q-learning regulariser."""

    def __init__(self, *args, cql_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = cql_alpha

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        td_losses: list[float] = []
        cql_losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                next_q_values = self.q_net_target(
                    replay_data.next_observations
                ).max(dim=1, keepdim=True).values
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            q_values = self.q_net(replay_data.observations)
            selected_q_values = torch.gather(
                q_values,
                dim=1,
                index=replay_data.actions.long(),
            )
            td_loss = F.smooth_l1_loss(selected_q_values, target_q_values)
            cql_loss = self.cql_alpha * (
                torch.logsumexp(q_values, dim=1).mean()
                - selected_q_values.mean()
            )
            loss = td_loss + cql_loss

            losses.append(float(loss.item()))
            td_losses.append(float(td_loss.item()))
            cql_losses.append(float(cql_loss.item()))
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", sum(losses) / len(losses))
        self.logger.record("train/td_loss", sum(td_losses) / len(td_losses))
        self.logger.record("train/cql_loss", sum(cql_losses) / len(cql_losses))


def _alpha_to_filename(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "-")


def instantiate_cql_dqn(
    env_name: ClassicControl,
    seed: int = 0,
    cql_alpha: float = 0.0,
    total_timesteps: int | None = None,
) -> CQLDQN:
    return CQLDQN(
        env=make_training_env(env_name),
        seed=seed,
        cql_alpha=cql_alpha,
        **load_dqn_args(env_name, total_timesteps),
    )


def learn_cqldqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
    cql_alpha: float = 0.0,
    total_timesteps: int = 50_000,
    model_dir: str | Path | None = None,
    train_from_scratch: bool = False,
) -> tuple[CQLDQN, VecEnv]:
    basename = model_basename(env_name, seed, total_timesteps)
    model_path = cached_model_path(
        env_name,
        "cqldqn",
        f"{basename}_alpha_{_alpha_to_filename(cql_alpha)}",
        model_dir,
    )
    if not train_from_scratch and model_path.with_suffix(".zip").exists():
        print(f"Loading CQLDQN model: {seed}, alpha: {cql_alpha}")
        model = load_cached_agent(CQLDQN, model_path, env_name)
    else:
        print(f"Learning CQLDQN from scratch: {seed}, alpha: {cql_alpha}")
        model = instantiate_cql_dqn(
            env_name,
            seed,
            cql_alpha,
            total_timesteps,
        )
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=env_name == MINATAR_BREAKOUT,
        )
        model.save(str(model_path))
    return model, seeded_vec_env(model, seed)
