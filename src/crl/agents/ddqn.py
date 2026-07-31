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


class DDQN(DQN):
    """Double DQN with online selection and target-network evaluation."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
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
                next_actions = self.q_net(
                    replay_data.next_observations
                ).argmax(dim=1, keepdim=True)
                next_q_values = torch.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_actions,
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = torch.gather(
                self.q_net(replay_data.observations),
                dim=1,
                index=replay_data.actions.long(),
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))

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


def instantiate_ddqn(
    env_name: ClassicControl,
    seed: int = 0,
    total_timesteps: int | None = None,
) -> DDQN:
    return DDQN(
        env=make_training_env(env_name),
        seed=seed,
        **load_dqn_args(env_name, total_timesteps),
    )


def learn_ddqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
    total_timesteps: int = 50_000,
    model_dir: str | Path | None = None,
    train_from_scratch: bool = False,
) -> tuple[DDQN, VecEnv]:
    model_path = cached_model_path(
        env_name,
        "ddqn",
        model_basename(env_name, seed, total_timesteps),
        model_dir,
    )
    if not train_from_scratch and model_path.with_suffix(".zip").exists():
        print(f"Loading DDQN model: {seed}")
        model = load_cached_agent(DDQN, model_path, env_name)
    else:
        print(f"Learning DDQN from scratch: {seed}")
        model = instantiate_ddqn(env_name, seed, total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=env_name == MINATAR_BREAKOUT,
        )
        model.save(str(model_path))
    return model, seeded_vec_env(model, seed)
