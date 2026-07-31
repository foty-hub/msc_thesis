from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
from crl.env import MINATAR_BREAKOUT, instantiate_eval_env
from crl.types import ClassicControl, RepresentationMethod, ScoringMethod


@dataclass(frozen=True)
class ShiftSpec:
    parameter: str
    values: tuple[float, ...]
    nominal_value: float
    grid_bins: int


# Defines the distribution shifts evaluated for each environment. For example, the Cartpole
# spec means we train the model on the nominal value (length = 0.5), then evaluate it on
# the range of lengths [0.1, 0.3, 0.5, ..., 2.7, 2.9]
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
            float(value) for value in np.arange(0.001, 0.005 + 0.00025, 0.00025)
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
    MINATAR_BREAKOUT: ShiftSpec(
        parameter="sticky_action_prob",
        values=tuple(float(value) for value in np.arange(0.0, 0.51, 0.05)),
        nominal_value=0.1,
        grid_bins=4,
    ),
}


@dataclass(frozen=True)
class GridCalibrationConfig:
    n_calib_steps: int = 10_000
    alpha: float = 0.25
    min_calib: int = 100
    max_calib_per_cell: int = 500
    obs_quantile: float = 0.1
    scoring_method: ScoringMethod = "td"
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = signed_score
    inference_batch_size: int = 4096
    representation_dim: int | None = None
    n_representation_steps: int = 10_000
    representation_method: RepresentationMethod = "pca"


@dataclass(frozen=True)
class PCARepresentation:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features) - self.mean) @ self.components.T

    @property
    def n_dimensions(self) -> int:
        return int(self.components.shape[0])


@dataclass(frozen=True)
class InputPCARepresentation(PCARepresentation):
    """PCA fitted to flattened observations before policy feature extraction."""


@dataclass(frozen=True)
class QValueRepresentation:
    """Use the policy's action values as an action-relevant state summary."""

    model: DQN

    def transform(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, device=self.model.device)
        with torch.inference_mode():
            q_values = self.model.q_net.q_net(feature_tensor)
        return q_values.detach().cpu().numpy()

    @property
    def n_dimensions(self) -> int:
        return int(self.model.action_space.n)


@dataclass(frozen=True)
class GridCalibration:
    discretiser: GridDiscretiser
    corrections: Corrections
    n_visited_cells: int
    n_calibrated_cells: int
    fallback: float
    representation: (
        PCARepresentation | InputPCARepresentation | QValueRepresentation | None
    ) = None


def extract_policy_features(
    model: DQN,
    observations: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    batches = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            observation_tensor = model.policy.obs_to_tensor(
                observations[start : start + batch_size]
            )[0]
            features = model.q_net.extract_features(
                observation_tensor,
                model.q_net.features_extractor,
            )
            batches.append(features.detach().cpu().numpy())
    return np.concatenate(batches)


def fit_pca_representation(
    features: np.ndarray,
    n_components: int,
) -> PCARepresentation:
    mean = features.mean(axis=0)
    _, _, components = np.linalg.svd(features - mean, full_matrices=False)
    components = components[:n_components]
    largest = np.abs(components).argmax(axis=1)
    signs = np.sign(components[np.arange(n_components), largest])
    components *= signs[:, None]
    return PCARepresentation(mean=mean, components=components)


def fit_input_pca_representation(
    features: np.ndarray,
    n_components: int,
) -> InputPCARepresentation:
    """Fit exact PCA without materialising the large left-singular matrix."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = centered.T @ centered
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, -n_components:][:, ::-1].T.copy()
    largest = np.abs(components).argmax(axis=1)
    signs = np.sign(components[np.arange(n_components), largest])
    components *= signs[:, None]
    return InputPCARepresentation(mean=mean, components=components)


def buffer_observations(buffer: ReplayBuffer) -> np.ndarray:
    return np.concatenate(
        [np.asarray(transition.state) for transition in buffer],
        axis=0,
    )


def flatten_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    return observations.reshape(len(observations), -1)


def fit_grid_from_buffer(
    buffer: ReplayBuffer,
    *,
    n_bins: int,
    n_actions: int,
    obs_quantile: float,
    observations: np.ndarray | None = None,
) -> GridDiscretiser:
    if observations is None:
        observations = buffer_observations(buffer)
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
    """Fit a nominal grid and conformal corrections."""
    representation = None
    grid_buffer = None
    if (
        config.representation_dim is not None
        or config.representation_method == "q_values"
    ):
        grid_buffer = collect_transitions(
            model,
            env,
            config.n_representation_steps,
        )
        representation_observations = buffer_observations(grid_buffer)
        if config.representation_method == "input_pca":
            input_features = flatten_observations(
                representation_observations
            )
            representation = fit_input_pca_representation(
                input_features,
                config.representation_dim,
            )
            grid_observations = representation.transform(input_features)
        else:
            latent_features = extract_policy_features(
                model,
                representation_observations,
                batch_size=config.inference_batch_size,
            )
        if config.representation_method == "pca":
            representation = fit_pca_representation(
                latent_features,
                config.representation_dim,
            )
            grid_observations = representation.transform(latent_features)
        elif config.representation_method == "q_values":
            representation = QValueRepresentation(model)
            grid_observations = representation.transform(latent_features)

    calibration_buffer = collect_transitions(model, env, config.n_calib_steps)
    if grid_buffer is None:
        grid_buffer = calibration_buffer
        grid_observations = buffer_observations(grid_buffer)

    # Defines the grid cell boundaries using a sparse radix encoding (so we don't have
    # to materialise the whole grid - useful for higher dimensional state spaces).
    discretiser = fit_grid_from_buffer(
        grid_buffer,
        n_bins=n_bins,
        n_actions=int(env.action_space.n),
        obs_quantile=config.obs_quantile,
        observations=grid_observations,
    )

    if representation is None:
        calibration_discretiser = discretiser
    elif isinstance(representation, InputPCARepresentation):

        def calibration_discretiser(observations, actions):
            input_features = flatten_observations(observations)
            return discretiser(
                representation.transform(input_features),
                actions,
            )

    else:

        def calibration_discretiser(observations, actions):
            features = extract_policy_features(
                model,
                observations,
                batch_size=config.inference_batch_size,
            )
            return discretiser(representation.transform(features), actions)

    # Given the grid, compute the calibration scores
    fill_calibration_sets = {
        "td": fill_calib_sets_td,
        "monte_carlo": fill_calib_sets_mc,
    }[config.scoring_method]
    calibration_sets = fill_calibration_sets(
        model,
        calibration_buffer,
        calibration_discretiser,
        maxlen=config.max_calib_per_cell,
        score=config.score_fn,
        batch_size=config.inference_batch_size,
    )

    # Compute corrections using the scores
    corrections = compute_corrections(
        calibration_sets,
        alpha=config.alpha,
        min_calib=config.min_calib,
    )
    return GridCalibration(
        discretiser=discretiser,
        corrections=corrections,
        n_visited_cells=len(calibration_sets),
        n_calibrated_cells=len(corrections) - 1,
        fallback=float(corrections["fallback"]),
        representation=representation,
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
        representation = (
            None if calibration is None else calibration.representation
        )
        if isinstance(representation, InputPCARepresentation):
            raw_q_tensor = model.q_net(observation_tensor)
            grid_observation = representation.transform(
                flatten_observations(observation)
            )
        elif representation is not None:
            features = model.q_net.extract_features(
                observation_tensor,
                model.q_net.features_extractor,
            )
            raw_q_tensor = model.q_net.q_net(features)
            grid_observation = representation.transform(
                features.detach().cpu().numpy()
            )
        else:
            raw_q_tensor = model.q_net(observation_tensor)
            grid_observation = observation
        raw_q_values = raw_q_tensor[0].detach().cpu().numpy().astype(float)

    if calibration is None:
        corrections = np.zeros_like(raw_q_values)
    else:
        actions = np.arange(raw_q_values.size, dtype=np.int64)
        corrections = corrections_for_actions(
            grid_observation,
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
    """Evaluate a policy without constructing autograd graphs."""
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
            returns.append(episode_return)
            observation = env.reset()
            episode_return = 0.0
            episode_steps = 0

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
