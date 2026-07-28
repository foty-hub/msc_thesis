from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _normalise_bins(n_bins: int | Sequence[int], n_dims: int) -> np.ndarray:
    if isinstance(n_bins, int):
        bins = np.full(n_dims, n_bins, dtype=np.int64)
    else:
        bins = np.asarray(n_bins, dtype=np.int64)
    if bins.shape != (n_dims,):
        raise ValueError("n_bins must be an int or contain one entry per dimension.")
    if np.any(bins < 1):
        raise ValueError("Every grid dimension must contain at least one bin.")
    return bins


def _get_unique_ids(
    binned_data: np.ndarray,
    num_bins: np.ndarray,
) -> np.ndarray:
    """Encode rows of mixed-radix bin coordinates as stable integer IDs."""
    binned = np.asarray(binned_data, dtype=np.int64)
    bins = np.asarray(num_bins, dtype=np.int64)
    if binned.ndim != 2 or binned.shape[1] != bins.size:
        raise ValueError("binned_data must have shape (n_samples, n_dimensions).")

    multipliers = np.cumprod(bins[::-1], dtype=np.int64)[:-1][::-1]
    multipliers = np.append(multipliers, np.int64(1))
    return binned @ multipliers


def discretise_observation_grid(
    obs: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    num_bins: np.ndarray,
) -> np.ndarray:
    """Map a batch of continuous observations to mixed-radix state IDs."""
    observations = np.asarray(obs, dtype=float)
    if observations.ndim == 1:
        observations = observations[None, :]

    mins_arr = np.asarray(mins, dtype=float)
    maxs_arr = np.asarray(maxs, dtype=float)
    bins = np.asarray(num_bins, dtype=np.int64)
    expected_shape = (bins.size,)
    if observations.ndim != 2 or observations.shape[1] != bins.size:
        raise ValueError("obs must have shape (n_samples, n_dimensions).")
    if mins_arr.shape != expected_shape or maxs_arr.shape != expected_shape:
        raise ValueError("mins, maxs, and num_bins must have matching shapes.")

    widths = (maxs_arr - mins_arr) / bins
    if np.any(widths <= 0) or not np.all(np.isfinite(widths)):
        raise ValueError("Every grid dimension must have a finite positive width.")

    coordinates = np.floor((observations - mins_arr) / widths)
    coordinates = np.clip(coordinates, 0, bins - 1).astype(np.int64)
    return _get_unique_ids(coordinates, bins)


@dataclass(frozen=True)
class GridDiscretiser:
    """A fitted state-action grid that only materialises visited cell IDs."""

    mins: np.ndarray
    maxs: np.ndarray
    num_bins: np.ndarray
    n_actions: int

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        *,
        n_bins: int | Sequence[int],
        n_actions: int,
        obs_quantile: float = 0.1,
    ) -> GridDiscretiser:
        obs = np.asarray(observations, dtype=float)
        if obs.ndim != 2 or obs.shape[0] == 0:
            raise ValueError("observations must have shape (n_samples, n_dimensions).")
        if not np.all(np.isfinite(obs)):
            raise ValueError("Grid fitting observations must all be finite.")
        if not 0.0 <= obs_quantile < 0.5:
            raise ValueError("obs_quantile must lie in [0, 0.5).")
        if n_actions < 1:
            raise ValueError("n_actions must be positive.")

        bins = _normalise_bins(n_bins, obs.shape[1])
        mins = np.quantile(obs, obs_quantile, axis=0)
        maxs = np.quantile(obs, 1.0 - obs_quantile, axis=0)

        # Constant dimensions carry no partitioning information but still need
        # finite widths for a well-defined mapping.
        scale = np.maximum(np.maximum(np.abs(mins), np.abs(maxs)), 1.0)
        maxs = np.maximum(maxs, mins + 1e-6 * scale)
        return cls(mins=mins, maxs=maxs, num_bins=bins, n_actions=n_actions)

    @property
    def n_state_cells(self) -> int:
        return int(np.prod(self.num_bins, dtype=np.int64))

    @property
    def n_state_action_cells(self) -> int:
        return self.n_state_cells * self.n_actions

    def state_ids(self, observations: np.ndarray) -> np.ndarray:
        return discretise_observation_grid(
            observations,
            self.mins,
            self.maxs,
            self.num_bins,
        )

    def __call__(
        self,
        observations: np.ndarray,
        actions: np.ndarray | Sequence[int] | int,
    ) -> np.ndarray:
        state_ids = self.state_ids(observations)
        action_ids = np.asarray(actions, dtype=np.int64).reshape(-1)

        if state_ids.size == 1 and action_ids.size > 1:
            state_ids = np.repeat(state_ids, action_ids.size)
        elif action_ids.size == 1 and state_ids.size > 1:
            action_ids = np.repeat(action_ids, state_ids.size)
        elif state_ids.size != action_ids.size:
            raise ValueError("Observation and action batch sizes are incompatible.")

        if np.any(action_ids < 0) or np.any(action_ids >= self.n_actions):
            raise ValueError("Action is outside the configured discrete action space.")
        return state_ids * self.n_actions + action_ids
