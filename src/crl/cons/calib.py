# %%
import torch
import numpy as np

from collections import deque
from typing import Callable
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.cons.buffer import ReplayBuffer


def collect_transitions(
    model: DQN,
    env: VecEnv,
    n_transitions: int = 100_000,
):
    "run episodes and record SARSA transitions to a replay buffer"
    # TODO: this could stop recording transitions more intelligently if it just updated calib_sets as it went.
    buffer = ReplayBuffer(capacity=n_transitions)
    obs = env.reset()
    for _ in range(n_transitions):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        # Next action required for SARSA
        next_action, _ = model.predict(next_obs, deterministic=True)

        # Store transition (s, a, r, s', a', done)
        buffer.push(obs, action, reward, next_obs, next_action, done)

        obs = next_obs
        if done:
            obs = env.reset()  # start new episode

    return buffer


# 1. build up a conformal set and conformalise calib_sets for each state-action pair (essentially measuring Bellman errors)
def score(y_pred, y_true):
    "Note that this is the signed error, *not* the absolute error as is typical. This only penalises overestimation"
    return np.asarray(y_pred) - np.asarray(y_true)


def fill_calib_sets(
    model: DQN,
    buffer: ReplayBuffer,
    discretise: Callable,
    n_discrete_states: int,
    discount: float,
    maxlen: int = 500,
):
    calib_sets = {}
    for (sa,) in np.ndindex((n_discrete_states)):
        calib_sets[sa] = dict(
            y_preds=deque(maxlen=maxlen),
            y_trues=deque(maxlen=maxlen),
            scores=deque(maxlen=maxlen),
        )

    # now construct the calibration sets
    for transition in buffer[:-1]:
        # extract a transition and add it to the calibration set
        with torch.no_grad():
            q_pred = model.q_net(model.policy.obs_to_tensor(transition.state)[0])
            y_pred = q_pred[0, transition.action]
            # Value of the terminal state is 0 by definition.
            if transition.done:
                y_true = transition.reward
            else:
                q_true = model.q_net(
                    model.policy.obs_to_tensor(transition.next_state)[0]
                )
                y_true = (
                    transition.reward[0] + discount * q_true[0, transition.next_action]
                )

            obs_disc = discretise(transition.state, transition.action)

            calib_sets[obs_disc]["y_preds"].append(y_pred)
            calib_sets[obs_disc]["y_trues"].append(y_true)
            calib_sets[obs_disc]["scores"].append(score(y_pred, y_true))

    return calib_sets


def compute_lower_bounds(calib_sets: dict, alpha: float, min_calib: int):
    n_calibs = []
    qhat_global = 0

    for sa, reg in calib_sets.items():
        n_calib = len(reg["y_preds"])
        n_calibs.append(n_calib)
        if n_calib < min_calib:
            continue

        # conformalise
        q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
        qhat = np.quantile(reg["scores"], q_level, method="higher")

        reg["qhat"] = qhat
        # Set a global, pessimistic correction for un-visited state action pairs.
        if qhat > qhat_global:
            qhat_global = qhat

    calib_sets["fallback"] = qhat_global
    return calib_sets, n_calibs
