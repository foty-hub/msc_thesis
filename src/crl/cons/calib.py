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
    n_transitions: int,
) -> ReplayBuffer:
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


# TODO: this doesn't work :~<
def collect_training_transitions(
    model: DQN,
    env: VecEnv,
    n_transitions: int,
) -> ReplayBuffer:
    """Fills a replay buffer by accessing the replay buffer from a trained
    stablebaslines3 DQN"""
    # Pull transitions directly from the SB3 replay buffer, rather than
    # interacting with the environment. We mirror the shapes used by
    # `collect_transitions`: state/next_state include a leading env
    # dimension (of size 1), and action/reward/done are arrays with shape
    # (1,).
    sb3_rb = getattr(model, "replay_buffer", None)
    if sb3_rb is None:
        raise ValueError(
            "DQN model has no replay_buffer. Ensure the model was trained and has a populated buffer."
        )

    # Determine how many transitions are available in the SB3 buffer
    if getattr(sb3_rb, "full", False):
        available = sb3_rb.buffer_size
    else:
        available = sb3_rb.pos

    k = min(n_transitions, available)
    if available < n_transitions:
        print(
            f"[collect_training_transitions] Warning: requested {n_transitions} transitions, "
            f"but only {available} available in the SB3 replay buffer; using all available."
        )

    # Build indices of the most recent `k` transitions in chronological order
    if getattr(sb3_rb, "full", False):
        buf_size = sb3_rb.buffer_size
        start = (sb3_rb.pos - k) % buf_size
        indices = [(start + i) % buf_size for i in range(k)]
    else:
        start = sb3_rb.pos - k
        indices = list(range(start, start + k))

    # Create our local replay buffer to hold the copied transitions
    buffer = ReplayBuffer(capacity=k)

    # Helper to coerce scalar/array to shape (1,)
    def _as_row_array(x, dtype=None):
        # x: B, N_env, D
        arr = np.asarray(x)
        if arr.shape == ():
            arr = np.array([arr], dtype=dtype)
        elif dtype is not None and arr.dtype != dtype:
            arr = arr[2].astype(dtype)
        return arr

    # Iterate through selected indices and push into our buffer
    for idx in indices:
        obs = sb3_rb.observations[idx]
        next_obs = sb3_rb.next_observations[idx]
        action = sb3_rb.actions[idx]
        reward = sb3_rb.rewards[idx]
        done_raw = sb3_rb.dones[idx]
        timeout = None
        if hasattr(sb3_rb, "timeouts") and sb3_rb.timeouts is not None:
            timeout = sb3_rb.timeouts[idx]

        # Only terminal if done and not timeout (time-limit truncation)
        done_bool = np.asarray(done_raw).astype(bool)
        if timeout is not None:
            done_bool = np.logical_and(
                done_bool, np.logical_not(np.asarray(timeout).astype(bool))
            )
        done_arr = _as_row_array(done_bool, dtype=bool)
        # Ensure shapes to match collect_transitions
        action_arr = _as_row_array(action)
        reward_arr = _as_row_array(reward)

        # Add leading env dimension to observations (size 1)
        state = np.expand_dims(np.asarray(obs), axis=0)
        next_state = np.expand_dims(np.asarray(next_obs), axis=0)

        # Next action required for SARSA: greedy with the current policy
        next_action, _ = model.predict(next_state, deterministic=True)

        buffer.push(state, action_arr, reward_arr, next_state, next_action, done_arr)
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


def fill_calib_sets_mc(
    model: DQN,
    buffer: ReplayBuffer,
    discretise: Callable,
    n_discrete_states: int,
    maxlen: int = 500,
    score: Callable = signed_score,
):
    """
    Variant of `fill_calib_sets` that uses Monte Carlo returns as the target
    rather than a 1-step Bellman bootstrap. For each transition t, the target is
    G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ... until the end of the episode.

    Parameters mirror `fill_calib_sets`.
    """
    # Initialise per-(state, action) deques
    calib_sets = {}
    for (sa,) in np.ndindex((n_discrete_states)):
        calib_sets[sa] = dict(
            y_preds=deque(maxlen=maxlen),
            y_trues=deque(maxlen=maxlen),
            scores=deque(maxlen=maxlen),
        )

    discount = model.gamma

    # Mirror the original behaviour of ignoring the final buffer slot
    n = len(buffer) - 1
    if n <= 0:
        return calib_sets

    # Precompute Monte Carlo returns for each transition in this buffer
    rewards = np.empty(n, dtype=float)
    dones = np.empty(n, dtype=bool)
    for i in range(n):
        trans = buffer[i]
        r = np.asarray(trans.reward).reshape(-1)[0]
        d = bool(np.asarray(trans.done).reshape(-1)[0])
        rewards[i] = float(r)
        dones[i] = d

    mc_returns = np.empty(n, dtype=float)
    G = 0.0
    for i in range(n - 1, -1, -1):
        # resets at episode boundary (done = True)
        if dones[i]:
            G = rewards[i]
        else:
            G = rewards[i] + discount * G
        mc_returns[i] = G

    # Populate calibration sets using MC targets
    for i in range(n):
        trans = buffer[i]
        with torch.no_grad():
            # Predicted Q(s,a)
            obs_tensor = model.policy.obs_to_tensor(trans.state)[0]
            q_pred = model.q_net(obs_tensor)

            # Ensure a scalar action index for tensor indexing
            act_scalar = int(np.asarray(trans.action).reshape(-1)[0])
            y_pred = q_pred[0, act_scalar]

            # MC target for this step
            y_true = mc_returns[i]

            # Convert to plain floats for scoring/deques
            y_pred_val = (
                float(y_pred.detach().cpu().numpy())
                if hasattr(y_pred, "detach")
                else float(y_pred)
            )
            y_true_val = float(y_true)

            obs_disc = discretise(trans.state, trans.action)
            for idx in obs_disc:
                idx = int(idx)
                calib_sets[idx]["y_preds"].append(y_pred_val)
                calib_sets[idx]["y_trues"].append(y_true_val)
                calib_sets[idx]["scores"].append(score(y_pred_val, y_true_val))

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
