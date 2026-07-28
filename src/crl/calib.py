from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.buffer import ReplayBuffer, Transition

AggregationStrategy = Literal["max", "mean", "median"]
CalibrationSets = dict[int, dict[str, deque[np.float32]]]
Corrections = dict[int | str, float]


def collect_transitions(
    model: DQN,
    env: VecEnv,
    n_transitions: int,
) -> ReplayBuffer:
    """Collect chronological SARSA transitions from one vectorised environment."""
    if n_transitions < 1:
        raise ValueError("n_transitions must be positive.")
    if getattr(env, "num_envs", 1) != 1:
        raise ValueError("Transition collection currently requires one environment.")

    buffer = ReplayBuffer(capacity=n_transitions)
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    for _ in range(n_transitions):
        next_obs, reward, done, _info = env.step(action)
        next_action, _ = model.predict(next_obs, deterministic=True)

        buffer.push(
            np.array(obs, copy=True),
            np.array(action, copy=True),
            np.array(reward, copy=True),
            np.array(next_obs, copy=True),
            np.array(next_action, copy=True),
            np.array(done, copy=True),
        )

        # SB3 VecEnv implementations automatically reset completed environments,
        # so next_obs is already the next episode's initial observation when done.
        obs = next_obs
        action = next_action

    return buffer


def signed_score(y_pred, y_true) -> np.ndarray:
    """One-sided error ``prediction - target`` used to penalise overestimation."""
    return np.asarray(y_pred) - np.asarray(y_true)


def unsigned_score(y_pred, y_true) -> np.ndarray:
    """Absolute prediction error."""
    return np.abs(signed_score(y_pred, y_true))


def _as_transitions(
    buffer: ReplayBuffer | Sequence[Transition],
) -> list[Transition]:
    transitions = list(buffer)
    if not transitions:
        raise ValueError("At least one transition is required for calibration.")
    return transitions


def _transition_arrays(
    transitions: Sequence[Transition],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.concatenate(
        [np.asarray(transition.state) for transition in transitions],
        axis=0,
    )
    next_states = np.concatenate(
        [np.asarray(transition.next_state) for transition in transitions],
        axis=0,
    )
    actions = np.asarray(
        [np.asarray(transition.action).reshape(-1)[0] for transition in transitions],
        dtype=np.int64,
    )
    next_actions = np.asarray(
        [
            np.asarray(transition.next_action).reshape(-1)[0]
            for transition in transitions
        ],
        dtype=np.int64,
    )
    rewards = np.asarray(
        [np.asarray(transition.reward).reshape(-1)[0] for transition in transitions],
        dtype=np.float32,
    )
    dones = np.asarray(
        [np.asarray(transition.done).reshape(-1)[0] for transition in transitions],
        dtype=bool,
    )
    return states, actions, rewards, next_states, next_actions, dones


def compute_td_scores(
    model: DQN,
    buffer: ReplayBuffer | Sequence[Transition],
    score: Callable = signed_score,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute one-step SARSA scores using batched network inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    transitions = _as_transitions(buffer)
    states, actions, rewards, next_states, next_actions, dones = _transition_arrays(
        transitions
    )
    scores = np.empty(len(transitions), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(transitions), batch_size):
            stop = min(start + batch_size, len(transitions))
            state_tensor = model.policy.obs_to_tensor(states[start:stop])[0]
            next_state_tensor = model.policy.obs_to_tensor(next_states[start:stop])[0]

            q_values = model.q_net(state_tensor)
            next_q_values = model.q_net(next_state_tensor)
            batch_indices = torch.arange(q_values.shape[0], device=q_values.device)

            predicted = q_values[
                batch_indices,
                torch.as_tensor(actions[start:stop], device=q_values.device),
            ]
            next_predicted = next_q_values[
                batch_indices,
                torch.as_tensor(next_actions[start:stop], device=q_values.device),
            ]

            target = torch.as_tensor(
                rewards[start:stop],
                dtype=predicted.dtype,
                device=predicted.device,
            )
            nonterminal = torch.as_tensor(
                ~dones[start:stop],
                dtype=predicted.dtype,
                device=predicted.device,
            )
            target = target + float(model.gamma) * nonterminal * next_predicted

            batch_scores = score(
                predicted.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
            )
            scores[start:stop] = np.asarray(batch_scores, dtype=np.float32)

    return scores


def compute_mc_returns(
    buffer: ReplayBuffer | Sequence[Transition],
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns, resetting the accumulator at episode ends."""
    transitions = _as_transitions(buffer)
    rewards = np.asarray(
        [np.asarray(transition.reward).reshape(-1)[0] for transition in transitions],
        dtype=np.float64,
    )
    dones = np.asarray(
        [np.asarray(transition.done).reshape(-1)[0] for transition in transitions],
        dtype=bool,
    )

    returns = np.empty(len(transitions), dtype=np.float64)
    running_return = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if dones[index]:
            running_return = rewards[index]
        else:
            running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def compute_mc_scores(
    model: DQN,
    buffer: ReplayBuffer | Sequence[Transition],
    score: Callable = signed_score,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute Monte Carlo return scores using batched Q-value inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    transitions = _as_transitions(buffer)
    states, actions, _rewards, _next_states, _next_actions, _dones = _transition_arrays(
        transitions
    )
    returns = compute_mc_returns(transitions, gamma=float(model.gamma))
    scores = np.empty(len(transitions), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(transitions), batch_size):
            stop = min(start + batch_size, len(transitions))
            state_tensor = model.policy.obs_to_tensor(states[start:stop])[0]
            q_values = model.q_net(state_tensor)
            batch_indices = torch.arange(q_values.shape[0], device=q_values.device)
            predicted = q_values[
                batch_indices,
                torch.as_tensor(actions[start:stop], device=q_values.device),
            ]
            batch_scores = score(
                predicted.detach().cpu().numpy(),
                returns[start:stop],
            )
            scores[start:stop] = np.asarray(batch_scores, dtype=np.float32)

    return scores


def _fill_calibration_sets(
    transitions: Sequence[Transition],
    scores: np.ndarray,
    discretise: Callable,
    maxlen: int,
) -> CalibrationSets:
    if maxlen < 1:
        raise ValueError("maxlen must be positive.")

    states, actions, _rewards, _next_states, _next_actions, _dones = _transition_arrays(
        transitions
    )
    cell_ids = np.asarray(discretise(states, actions), dtype=np.int64)
    if cell_ids.ndim == 1:
        cell_ids = cell_ids[:, None]
    if cell_ids.ndim != 2 or cell_ids.shape[0] != len(transitions):
        raise ValueError("discretise must return one row of cell IDs per transition.")

    calibration_sets: CalibrationSets = {}
    for row, value in zip(cell_ids, scores, strict=True):
        for cell_id in row:
            cell = int(cell_id)
            if cell not in calibration_sets:
                calibration_sets[cell] = {"scores": deque(maxlen=maxlen)}
            calibration_sets[cell]["scores"].append(np.float32(value))
    return calibration_sets


def fill_calib_sets_td(
    model: DQN,
    buffer: ReplayBuffer | Sequence[Transition],
    discretise: Callable,
    maxlen: int = 500,
    score: Callable = signed_score,
    *,
    batch_size: int = 4096,
) -> CalibrationSets:
    """Group batched one-step TD scores by visited state-action grid cell."""
    transitions = _as_transitions(buffer)
    scores = compute_td_scores(model, transitions, score, batch_size=batch_size)
    return _fill_calibration_sets(transitions, scores, discretise, maxlen)


def fill_calib_sets_mc(
    model: DQN,
    buffer: ReplayBuffer | Sequence[Transition],
    discretise: Callable,
    maxlen: int = 500,
    score: Callable = signed_score,
    *,
    batch_size: int = 4096,
) -> CalibrationSets:
    """Group batched Monte Carlo scores by visited state-action grid cell."""
    transitions = _as_transitions(buffer)
    scores = compute_mc_scores(model, transitions, score, batch_size=batch_size)
    return _fill_calibration_sets(transitions, scores, discretise, maxlen)


def compute_corrections(
    calib_sets: CalibrationSets,
    alpha: float,
    min_calib: int,
) -> Corrections:
    """Conformalise every sufficiently populated cell and add a fallback."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    if min_calib < 1:
        raise ValueError("min_calib must be positive.")

    corrections: Corrections = {}
    fallback = 0.0
    for state_action, calibration_set in calib_sets.items():
        scores = calibration_set["scores"]
        n_calib = len(scores)
        if n_calib < min_calib:
            continue

        q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
        qhat = float(np.quantile(scores, q_level, method="higher"))
        corrections[state_action] = qhat
        fallback = max(fallback, qhat)

    corrections["fallback"] = fallback
    return corrections


def correction_for(
    state: np.ndarray,
    action: np.ndarray | int,
    qhats: Mapping[int | str, float],
    discretise: Callable,
    agg: AggregationStrategy = "max",
    clip_correction: bool = False,
) -> float:
    """Look up and aggregate sparse corrections for one state-action pair."""
    fallback = float(qhats.get("fallback", 0.0))
    cell_ids = np.asarray(discretise(state, action), dtype=np.int64).reshape(-1)
    values = np.asarray(
        [qhats.get(int(cell_id), fallback) for cell_id in cell_ids],
        dtype=float,
    )

    if agg == "max":
        correction = float(np.max(values))
    elif agg == "mean":
        correction = float(np.mean(values))
    elif agg == "median":
        correction = float(np.median(values))
    else:
        raise ValueError("Unknown aggregation; use 'max', 'mean', or 'median'.")

    if clip_correction:
        correction = max(0.0, correction)
    return correction


def corrections_for_actions(
    state: np.ndarray,
    actions: np.ndarray,
    qhats: Mapping[int | str, float],
    discretise: Callable,
    *,
    agg: AggregationStrategy = "max",
    clip_correction: bool = False,
) -> np.ndarray:
    """Return sparse conformal corrections for every candidate action."""
    action_ids = np.asarray(actions, dtype=np.int64).reshape(-1)
    return np.asarray(
        [
            correction_for(
                state,
                int(action),
                qhats,
                discretise,
                agg=agg,
                clip_correction=clip_correction,
            )
            for action in action_ids
        ],
        dtype=np.float32,
    )
