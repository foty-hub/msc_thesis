from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.buffer import ReplayBuffer
from crl.calib import (
    Corrections,
    collect_transitions,
    compute_corrections,
    corrections_for_actions,
    fill_calib_sets_mc,
    fill_calib_sets_td,
    signed_score,
)
from crl.discretise import GridDiscretiser
from crl.env import instantiate_eval_env
from crl.types import ClassicControl, ScoringMethod


@dataclass(frozen=True)
class ShiftSpec:
    parameter: str
    values: tuple[float, ...]
    nominal_value: float
    grid_bins: int


SHIFT_SPECS: dict[ClassicControl, ShiftSpec] = {
    "CartPole-v1": ShiftSpec(
        parameter="length",
        values=tuple(float(value) for value in np.arange(0.1, 3.1, 0.2)),
        nominal_value=0.5,
        grid_bins=4,
    ),
    "Acrobot-v1": ShiftSpec(
        parameter="LINK_LENGTH_1",
        values=tuple(float(value) for value in np.linspace(0.5, 2.0, 16)),
        nominal_value=1.0,
        grid_bins=6,
    ),
    "MountainCar-v0": ShiftSpec(
        parameter="gravity",
        values=tuple(
            float(value)
            for value in np.arange(0.001, 0.005 + 0.00025, 0.00025)
        ),
        nominal_value=0.0025,
        grid_bins=10,
    ),
    "LunarLander-v3": ShiftSpec(
        parameter="gravity",
        values=tuple(float(value) for value in np.arange(-16.0, 0.0, 1.0)),
        nominal_value=-10.0,
        grid_bins=4,
    ),
}


@dataclass(frozen=True)
class GridCalibrationConfig:
    n_grid_steps: int = 2_500
    n_calib_steps: int = 2_500
    alpha: float = 0.25
    min_calib: int = 80
    max_calib_per_cell: int = 500
    obs_quantile: float = 0.1
    scoring_method: ScoringMethod = "td"
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = signed_score
    inference_batch_size: int = 4096


@dataclass(frozen=True)
class GridCalibration:
    discretiser: GridDiscretiser
    corrections: Corrections
    n_calibrated_cells: int
    fallback: float


def fit_grid_from_buffer(
    buffer: ReplayBuffer,
    *,
    n_bins: int,
    n_actions: int,
    obs_quantile: float,
) -> GridDiscretiser:
    observations = np.concatenate(
        [np.asarray(transition.state) for transition in buffer],
        axis=0,
    )
    return GridDiscretiser.fit(
        observations,
        n_bins=n_bins,
        n_actions=n_actions,
        obs_quantile=obs_quantile,
    )


def calibrate_grid_policy(
    model: DQN,
    env: VecEnv,
    *,
    n_bins: int,
    config: GridCalibrationConfig,
) -> GridCalibration:
    """Fit a nominal grid and conformal corrections on disjoint rollouts."""
    grid_buffer = collect_transitions(model, env, config.n_grid_steps)
    n_actions = int(getattr(env.action_space, "n"))
    discretiser = fit_grid_from_buffer(
        grid_buffer,
        n_bins=n_bins,
        n_actions=n_actions,
        obs_quantile=config.obs_quantile,
    )

    calibration_buffer = collect_transitions(model, env, config.n_calib_steps)
    if config.scoring_method == "td":
        calibration_sets = fill_calib_sets_td(
            model,
            calibration_buffer,
            discretiser,
            maxlen=config.max_calib_per_cell,
            score=config.score_fn,
            batch_size=config.inference_batch_size,
        )
    elif config.scoring_method == "monte_carlo":
        calibration_sets = fill_calib_sets_mc(
            model,
            calibration_buffer,
            discretiser,
            maxlen=config.max_calib_per_cell,
            score=config.score_fn,
            batch_size=config.inference_batch_size,
        )
    else:
        raise ValueError(f"Unknown scoring method: {config.scoring_method}")

    corrections = compute_corrections(
        calibration_sets,
        alpha=config.alpha,
        min_calib=config.min_calib,
    )
    return GridCalibration(
        discretiser=discretiser,
        corrections=corrections,
        n_calibrated_cells=len(corrections) - 1,
        fallback=float(corrections["fallback"]),
    )


def select_action(
    model: DQN,
    observation: np.ndarray,
    *,
    calibration: GridCalibration | None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Select the greedy raw or conformal-corrected action."""
    with torch.inference_mode():
        observation_tensor = model.policy.obs_to_tensor(observation)[0]
        raw_q_values = (
            model.q_net(observation_tensor)[0].detach().cpu().numpy().astype(float)
        )

    if calibration is None:
        corrections = np.zeros_like(raw_q_values)
    else:
        actions = np.arange(raw_q_values.size, dtype=np.int64)
        corrections = corrections_for_actions(
            observation,
            actions,
            calibration.corrections,
            calibration.discretiser,
        ).astype(float)

    action = int(np.argmax(raw_q_values - corrections))
    return action, raw_q_values, corrections


def evaluate_policy(
    model: DQN,
    env: VecEnv,
    *,
    n_episodes: int,
    calibration: GridCalibration | None,
    max_steps_per_episode: int = 10_000,
) -> list[float]:
    """Evaluate one policy without constructing autograd graphs."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive.")

    returns: list[float] = []
    observation = env.reset()
    episode_return = 0.0
    episode_steps = 0

    while len(returns) < n_episodes:
        action, _q_values, _corrections = select_action(
            model,
            observation,
            calibration=calibration,
        )
        observation, reward, done, _info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        episode_steps += 1

        if bool(np.asarray(done).reshape(-1)[0]):
            returns.append(episode_return)
            episode_return = 0.0
            episode_steps = 0
        elif episode_steps >= max_steps_per_episode:
            raise RuntimeError(
                "Evaluation episode exceeded max_steps_per_episode without ending."
            )

    return returns


def evaluate_shift(
    model: DQN,
    *,
    env_name: ClassicControl,
    parameter: str,
    value: float,
    n_episodes: int,
    eval_seed: int,
    calibration: GridCalibration,
) -> dict[str, float | list[float]]:
    """Evaluate paired baseline and calibrated policies at one shift value."""
    shift = {parameter: value}
    baseline_env = instantiate_eval_env(env_name, seed=eval_seed, **shift)
    calibrated_env = instantiate_eval_env(env_name, seed=eval_seed, **shift)
    try:
        baseline_returns = evaluate_policy(
            model,
            baseline_env,
            n_episodes=n_episodes,
            calibration=None,
        )
        calibrated_returns = evaluate_policy(
            model,
            calibrated_env,
            n_episodes=n_episodes,
            calibration=calibration,
        )
    finally:
        baseline_env.close()
        calibrated_env.close()

    return {
        parameter: float(value),
        "returns_noconf": baseline_returns,
        "returns_conf": calibrated_returns,
    }
