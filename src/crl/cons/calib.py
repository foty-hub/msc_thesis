# %%
import torch
import numpy as np

from collections import deque
from typing import Callable, Literal
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.cons.buffer import ReplayBuffer

AggregationStrategy = Literal["max", "mean", "median"]


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


def signed_score(y_pred, y_true) -> np.ndarray:
    """Signed error e=y_pred - y_true.
    Note that this is the signed error, *not* the absolute error as is typical - to penalise overestimation"""
    return np.asarray(y_pred) - np.asarray(y_true)


def unsigned_score(y_pred, y_true) -> np.ndarray:
    "Unsigned variant of the score e=|y_pred - y_true|"
    return np.abs(signed_score(y_pred, y_true))


def fill_calib_sets(
    model: DQN,
    buffer: ReplayBuffer,
    discretise: Callable,
    n_discrete_states: int,
    maxlen: int = 500,
    score: Callable = signed_score,
):
    calib_sets = {}
    for (sa,) in np.ndindex((n_discrete_states)):
        calib_sets[sa] = dict(
            y_preds=deque(maxlen=maxlen),
            y_trues=deque(maxlen=maxlen),
            scores=deque(maxlen=maxlen),
        )

    # now construct the calibration sets
    discount = model.gamma
    for trans in buffer[:-1]:
        # extract a transition and add it to the calibration set
        with torch.no_grad():
            q_pred = model.q_net(model.policy.obs_to_tensor(trans.state)[0])
            y_pred = q_pred[0, trans.action]
            # Value of the terminal state is 0 by definition.
            if trans.done:
                y_true = trans.reward
            else:
                q_true = model.q_net(model.policy.obs_to_tensor(trans.next_state)[0])
                y_true = trans.reward[0] + discount * q_true[0, trans.next_action]

            obs_disc = discretise(trans.state, trans.action)

            # iterate over the discrete
            for idx in obs_disc:
                idx = int(idx)
                calib_sets[idx]["y_preds"].append(y_pred)
                calib_sets[idx]["y_trues"].append(y_true)
                calib_sets[idx]["scores"].append(score(y_pred, y_true))

    return calib_sets


def compute_corrections(calib_sets: dict, alpha: float, min_calib: int):
    qhats = np.full(len(calib_sets), fill_value=np.nan)
    visits = np.zeros_like(qhats)
    qhat_global = 0

    for sa, regressor in calib_sets.items():
        n_calib = len(regressor["y_preds"])
        visits[sa] = n_calib
        if n_calib < min_calib:
            continue

        # conformalise
        q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
        qhat = np.quantile(regressor["scores"], q_level, method="higher")

        qhats[sa] = qhat
        # Set a global, pessimistic correction for un-visited state action pairs.
        if qhat > qhat_global:
            qhat_global = qhat

    # Set all the biggest offsets in place
    np.nan_to_num(qhats, nan=qhat_global, copy=False)
    return qhats, visits


def correction_for(
    state: np.ndarray,
    action: np.ndarray,
    qhats: np.ndarray,
    discretise: Callable,
    agg: AggregationStrategy,
    clip_correction: bool = False,
) -> float:
    """Return a single correction for a (state, action) by aggregating the
    per-tile corrections corresponding to the active tile-coding indices.

    Parameters
    ----------
    state, action:
        Inputs to `discretise`.
    qhats: np.ndarray
        Vector of per-index corrections of length n_discrete_states.
    discretise: Callable
        Function mapping (state, action) -> int or sequence[int] of active indices.
    agg: {"max", "mean", "median"}
        Aggregation used to combine per-tile corrections.
    clip_correction:
        Optionally limit the minimum correction. If clip_correction=True,
        then the correction will not *increase* the Q value, even if the
        conformal predictor finds that the model is underestimating Q.

    Returns
    -------
    float
        The aggregated correction.
    """
    idxs = discretise(state, action)
    vals = qhats[idxs]

    if agg == "max":
        correction = np.max(vals)
    if agg == "mean":
        correction = np.mean(vals)
    if agg == "median":
        correction = np.median(vals)
    else:
        ValueError(f"Unknown agg='{agg}'. Use 'max', 'mean', or 'median'.")

    if clip_correction:
        correction = np.clip(correction, a_min=0, a_max=None)

    return correction
