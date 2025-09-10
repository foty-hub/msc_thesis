# %%
from typing import Dict, Tuple

from crl.cons.agents import learn_dqn_policy


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_dqn_inference_params(model) -> Tuple[int, Dict[str, int]]:
    """
    Parameters used for selecting an action with a DQN policy:
    features_extractor + q_net.
    """
    pi = model.policy
    if not hasattr(pi, "q_net"):
        raise TypeError("This helper expects an SB3 DQN model (policy with `q_net`).")
    parts: Dict[str, int] = {}
    # if hasattr(pi, "features_extractor"):
    #     parts["features_extractor"] = count_params(pi.features_extractor)
    parts["q_net"] = count_params(pi.q_net)
    return sum(parts.values()), parts


def count_dqn_training_step_params(model) -> Tuple[int, Dict[str, int]]:
    """
    Parameters touched in a typical DQN training step:
    features_extractor + q_net + q_net_target.
    """
    pi = model.policy
    if not hasattr(pi, "q_net"):
        raise TypeError("This helper expects an SB3 DQN model (policy with `q_net`).")
    parts: Dict[str, int] = {}
    # if hasattr(pi, "features_extractor"):
    #     parts["features_extractor"] = count_params(pi.features_extractor)
    parts["q_net"] = count_params(pi.q_net)
    if hasattr(pi, "q_net_target"):
        parts["q_net_target"] = count_params(pi.q_net_target)
    return sum(parts.values()), parts


# Usage:

env = "LunarLander-v3"
model, vec_env = learn_dqn_policy(env)
n_infer, breakdown_infer = count_dqn_inference_params(model)
n_train, breakdown_train = count_dqn_training_step_params(model)
n_policy_total = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

print("DQN inference params:", n_infer, breakdown_infer)
print("DQN training-step params:", n_train, breakdown_train)
print("Total trainable params in policy object:", n_policy_total)

# %%
