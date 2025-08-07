import numpy as np
from typing import Sequence, Callable
from crl.cons.buffer import Transition
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def run_test_episodes(model: DQN, vec_env: VecEnv):
    stats = [
        {"label": "position", "vals": []},
        {"label": "velocity", "vals": []},
        {"label": "angle", "vals": []},
        {"label": "angular_velocity", "vals": []},
    ]
    for episode in range(50):
        obs = vec_env.reset()
        for _id in range(4):
            stats[_id]["vals"].append(obs[0, _id])

        for t in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            for _id in range(4):
                stats[_id]["vals"].append(obs[0, _id])
            if done:
                break

    return stats


def build_tiling(model: DQN, vec_env: VecEnv, state_bins: list[int]):
    stats = run_test_episodes(model, vec_env)

    # compute mins and maxes of the tiling according to the quantiles of the distribution
    maxs, mins = compute_bin_ranges(stats, obs_quantile=0.1, state_bins=state_bins)
    discretise, n_discrete_states = build_grid_tiling(mins, maxs, state_bins=state_bins)
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
    n_actions = 2

    n_states = int(np.prod(num_bins)) * n_actions

    # ---------------- lookup for arbitrary (obs, act)
    def discretise(obs: np.ndarray, act: np.ndarray | int) -> int | np.ndarray:
        obs_arr = np.asarray(obs, float).reshape(-1, 4)  # ensure (batch,4)
        act_arr = np.asarray(act, int).reshape(-1)  # ensure (batch,)

        state_batch = discretise_observation_grid(obs_arr, mins, maxs, num_bins)
        out = state_batch * n_actions + act_arr
        return out.item() if out.size == 1 else out

    # print(
    #     f"Grid tiling: {n_states:,} states ({np.prod(num_bins)} states x {n_actions} actions)"
    # )
    return discretise, n_states


# Adaptive Tree Tiling
class _KDNode:
    __slots__ = ("axis", "thresh", "left", "right", "leaf_id", "is_leaf")

    def __init__(
        self,
        *,
        axis: int | None = None,
        thresh: float | None = None,
        left: "_KDNode | None" = None,
        right: "_KDNode | None" = None,
        leaf_id: int | None = None,
        is_leaf: bool = False,
    ):
        self.axis = axis
        self.thresh = thresh
        self.left = left
        self.right = right
        self.leaf_id = leaf_id
        self.is_leaf = is_leaf


def build_kdtree_tiling(
    buffer: Sequence[Transition],
    min_leaf: int = 32,
    include_actions: bool = False,
) -> tuple[list[int], Callable[[np.ndarray, np.ndarray | float | None], int], int]:
    """
    Build a variance-split kd-tree over 4-D states (or 5-D state+action).

    Returns
    -------
    ids                 : leaf id for each transition in `buffer`
    discretise(obs, act): maps observation (+action) to a leaf id
    n_discrete_states   : total number of leaves
    """
    # ------------------------------------------------------------------ data
    if include_actions:
        states = np.stack(
            [np.concatenate([tr.state[0], tr.action]) for tr in buffer],
            axis=0,
        )
        expected_dim = 5
    else:
        states = np.stack([tr.state[0] for tr in buffer], axis=0)
        expected_dim = 4

    N, D = states.shape
    if D != expected_dim:
        raise ValueError(f"Expected {expected_dim}-D inputs but got {D}-D.")

    # ------------------------------------------------------------ tree build
    next_leaf_id = [0]

    def _build(idx: np.ndarray) -> _KDNode:
        if idx.size <= min_leaf:
            leaf = _KDNode(leaf_id=next_leaf_id[0], is_leaf=True)
            next_leaf_id[0] += 1
            return leaf

        subset = states[idx]
        for axis in subset.var(axis=0).argsort()[::-1]:  # try axes by variance
            thresh = float(np.median(subset[:, axis]))
            left_mask = subset[:, axis] <= thresh
            right_mask = ~left_mask
            if left_mask.any() and right_mask.any():  # valid split
                node = _KDNode(axis=int(axis), thresh=thresh)
                node.left = _build(idx[left_mask])
                node.right = _build(idx[right_mask])
                return node

        # no axis gives a valid split → make leaf
        leaf = _KDNode(leaf_id=next_leaf_id[0], is_leaf=True)
        next_leaf_id[0] += 1
        return leaf

    root = _build(np.arange(N))

    # -------------------------------------------------------------- lookup
    def _lookup(node: _KDNode, x: np.ndarray) -> int:
        while not node.is_leaf:
            node = node.left if x[node.axis] <= node.thresh else node.right
        return node.leaf_id

    # ----------------------------------------------------------- interface
    def discretise(
        obs: np.ndarray,
        action: np.ndarray | float | None = None,
    ) -> int | np.ndarray:
        """
        * include_actions=False: discretise(obs) where obs shape (4,) or (batch,4)
        * include_actions=True : discretise(obs, action)
              - obs shape (4,) & action scalar → int
              - obs shape (batch,4) & action scalar or (batch,) array → array
        """
        arr = np.asarray(obs, dtype=float)
        if action:
            action = action[0]

        if include_actions:
            if action is None:
                raise ValueError("Action must be provided when include_actions=True.")

            act_arr = np.asarray(action, dtype=float)
            if arr.ndim == 1:
                if arr.shape[0] != 4 or act_arr.ndim > 0:
                    raise ValueError("Expect obs (4,) and action scalar.")
                x = np.concatenate([arr, [act_arr.item()]])
                return _lookup(root, x)

            elif arr.ndim == 2 and arr.shape[1] == 4:
                if act_arr.ndim == 0:  # scalar → broadcast
                    act_arr = np.full(arr.shape[0], act_arr.item())
                if act_arr.shape != (arr.shape[0],):
                    raise ValueError("Action array must match batch length.")
                batch = np.hstack([arr, act_arr[:, None]])
                return np.fromiter(
                    (_lookup(root, row) for row in batch), int, count=batch.shape[0]
                )

            else:
                raise ValueError(
                    "Obs must be (4,) or (batch,4) when include_actions=True."
                )

        else:  # no actions
            if arr.ndim == 1:
                if arr.shape[0] != 4:
                    raise ValueError("Expected obs shape (4,).")
                return _lookup(root, arr)
            elif arr.ndim == 2 and arr.shape[1] == 4:
                return np.fromiter(
                    (_lookup(root, row) for row in arr), int, count=arr.shape[0]
                )
            else:
                raise ValueError("Obs must be (4,) or (batch,4).")

    # -------------------------------------------------------- existing ids
    ids = [_lookup(root, s) for s in states]
    n_discrete_states = max(ids) + 1
    print(f"Built tree with {n_discrete_states:,} nodes")
    return ids, discretise, n_discrete_states
