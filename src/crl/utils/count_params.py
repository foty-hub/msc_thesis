from __future__ import annotations

import torch
from stable_baselines3 import DQN


def count_params(module: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def count_dqn_inference_params(model: DQN) -> tuple[int, dict[str, int]]:
    """Count parameters used by a DQN Q-network during action selection."""
    parts = {"q_net": count_params(model.policy.q_net)}
    return sum(parts.values()), parts


def count_dqn_training_step_params(model: DQN) -> tuple[int, dict[str, int]]:
    """Count online and target Q-network parameters used during training."""
    parts = {
        "q_net": count_params(model.policy.q_net),
        "q_net_target": count_params(model.policy.q_net_target),
    }
    return sum(parts.values()), parts
