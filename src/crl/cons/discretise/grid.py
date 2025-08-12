import numpy as np
from typing import Callable
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# Classic Control obs spaces are Boxes in Gymnasium/Gym
from gymnasium.spaces import Box  # type: ignore


def run_test_episodes(model: DQN, vec_env: VecEnv, n_episodes: int = 50):
    """
    Run `n_episodes` using a vectorized env (n_envs == 1) and collect per-dimension
    observation traces. Works for generic Classic Control tasks that expose a
    1-D Box observation space.
    """
    obs_space = vec_env.observation_space
    n_dims = int(obs_space.shape[0])
    # Sanity checks
    assert getattr(vec_env, "num_envs", 1) == 1
    assert isinstance(obs_space, Box)

    # Generic, index-based labels
    stats = [{"label": f"dim_{i}", "vals": []} for i in range(n_dims)]

    for _ in range(n_episodes):
        obs = vec_env.reset()  # shape (1, n_dims)

        # Record initial observation at reset
        for i in range(n_dims):
            stats[i]["vals"].append(float(obs[0, i]))

        # Step until episode ends (terminated or truncated)
        dones = np.array([False], dtype=bool)
        while not bool(dones[0]):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            for i in range(n_dims):
                stats[i]["vals"].append(float(obs[0, i]))

    return stats


def build_tiling(model: DQN, vec_env: VecEnv, state_bins: list[int]):
    stats = run_test_episodes(model, vec_env)

    # compute mins and maxes of the tiling according to the quantiles of the distribution
    maxs, mins = compute_bin_ranges(stats, obs_quantile=0.1, state_bins=state_bins)
    n_actions = vec_env.action_space.n
    discretise, n_discrete_states = build_grid_tiling(
        mins,
        maxs,
        state_bins=state_bins,
        n_actions=n_actions,
    )
    return discretise, n_discrete_states


def compute_bin_ranges(
    stats: list[dict[str, str | list]],
    state_bins: list[int],
    obs_quantile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    D = len(state_bins)  # dimensionality of the state space
    maxs = np.zeros(D)
    mins = np.zeros(D)
    for dim in range(D):
        vals = stats[dim]["vals"]
        mins[dim], maxs[dim] = (
            np.quantile(vals, obs_quantile),
            np.quantile(vals, 1 - obs_quantile),
        )

    return maxs, mins


# Grid Tiling
def discretise_observation_grid(
    obs: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    num_bins: np.ndarray,
) -> np.ndarray:
    """
    Vectorized function to discretize N-D obs, and return it as an index for a specific tile
    in the state-space

    Args:
        obs (np.ndarray): Observation array of shape ((n_samples), n_dims).
        mins (np.ndarray): Array of minimums for each dimension, shape (n_dims,).
        maxs (np.ndarray): Array of maximums for each dimension, shape (n_dims,).
        num_bins (np.ndarray): Array of bin counts for each dimension, shape (n_dims,).

    Returns:
        np.ndarray: Array of tile indices, shape (n_samples,), dtype int.
    """

    bin_widths = (maxs - mins) / num_bins
    scaled_data = (obs - mins) / bin_widths
    discretized = np.floor(scaled_data)
    discretised_state = np.clip(discretized, 0, num_bins - 1).astype(int)
    flattened = get_unique_ids(discretised_state, num_bins)
    return flattened


def get_unique_ids(binned_data, num_bins):
    """
    Calculates a unique integer ID for each row of discretized data.

    Args:
        binned_data (np.ndarray): Array of bin indices, shape (n_samples, n_dims).
        num_bins (np.ndarray): Array of bin counts for each dimension, shape (n_dims,).

    Returns:
        np.ndarray: A 1D array of unique IDs, shape (n_samples,).
    """
    num_bins = np.asarray(num_bins)

    # Calculate multipliers (strides) for each dimension.
    # This is equivalent to converting from a mixed-radix number to base 10.
    # The multipliers are the cumulative product of bin counts from right to left.
    multipliers = np.cumprod(num_bins[::-1])[:-1][::-1]
    multipliers = np.append(multipliers, 1)

    # Dot product of each row with multipliers gives the unique ID
    ids = np.dot(binned_data, multipliers)
    return ids


def build_grid_tiling(
    mins: np.ndarray,
    maxs: np.ndarray,
    state_bins: list[int],
    n_actions: int,
) -> tuple[Callable[[np.ndarray, np.ndarray], int | np.ndarray], int]:
    """
    Fixed-width grid discretisation over (state, action).

    Parameters
    ----------
    buffer      : replay buffer with .state and .action
    mins, maxs  : lower / upper bounds for the four state variables
    num_bins    : number of bins per state dimension

    Returns
    -------
    ids         : discrete ID for every transition in *buffer*
    discretise  : callable mapping (obs, act) → ID (scalar or array)
    n_states    : total number of discrete (state, action) cells
    """
    # ---------- build mapping for the existing buffer
    # states = np.stack([tr.state[0] for tr in buffer])  # (N, 4)
    # actions = np.asarray([tr.action for tr in buffer], int)  # (N,)

    # state_ids = discretise_observation_grid(states, mins, maxs, num_bins)
    # ids = (state_ids * n_actions + actions).tolist()
    num_bins = np.array(state_bins)
    n_states = int(np.prod(num_bins)) * n_actions

    # ---------------- lookup for arbitrary (obs, act)
    def discretise(obs: np.ndarray, act: np.ndarray | int) -> int | np.ndarray:
        obs_arr = np.asarray(obs, float).reshape(
            -1, len(state_bins)
        )  # ensure (batch,4)
        act_arr = np.asarray(act, int).reshape(-1)  # ensure (batch,)

        state_batch = discretise_observation_grid(obs_arr, mins, maxs, num_bins)
        out = state_batch * n_actions + act_arr
        return out.item() if out.size == 1 else out

    return discretise, n_states
