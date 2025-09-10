# %%
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.dqn import DQN

from crl.discretise.grid import run_test_episodes


class Node:
    def __init__(
        self, dimension=None, threshold=None, left=None, right=None, leaf_id=None
    ):
        self.dimension = dimension
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_id = leaf_id

    @property
    def is_leaf(self):
        return self.leaf_id is not None


def _best_split_1d(col: np.ndarray, min_leaf: int):
    """Return (threshold, gain) for the best 1D split by SSE reduction.
    If no valid split exists, return None.
    """
    v = np.asarray(col, dtype=float)
    n = v.size
    if n < 2 * min_leaf:
        return None

    order = np.argsort(v)
    v = v[order]

    # If constant, cannot split
    if v[0] == v[-1]:
        return None

    # Prefix sums for O(1) SSE queries
    s1 = np.cumsum(v, dtype=float)
    s2 = np.cumsum(v * v, dtype=float)

    def sse_prefix(m: int) -> float:
        # SSE of the first m elements (1-based m)
        sum1 = s1[m - 1]
        sum2 = s2[m - 1]
        return sum2 - (sum1 * sum1) / m

    n_float = float(n)
    total_sse = s2[-1] - (s1[-1] * s1[-1]) / n_float

    best_gain = 0.0
    best_k = None

    # Consider splits between distinct adjacent values, enforcing min_leaf
    for k in range(min_leaf, n - min_leaf + 1):
        if k < n and v[k - 1] == v[k]:
            continue
        left_sse = sse_prefix(k)
        right_n = n - k
        right_sum = s1[-1] - s1[k - 1]
        right_sum2 = s2[-1] - s2[k - 1]
        right_sse = right_sum2 - (right_sum * right_sum) / right_n
        gain = total_sse - (left_sse + right_sse)
        if gain > best_gain:
            best_gain = gain
            best_k = k

    if best_k is None or best_gain <= 0.0:
        return None

    # Threshold midway between the two neighboring values
    if best_k < n:
        t = 0.5 * (v[best_k - 1] + v[best_k])
    else:
        t = v[best_k - 1]
    return t, float(best_gain)


def _build_tree(
    data,
    dims,
    max_depth,
    min_samples_leaf,
    current_depth,
    leaf_id_counter,
    use_impurity_split: bool = False,
    min_impurity_decrease: float = 0.0,
):
    """Build a binary partitioning tree.

    Parameters
    ----------
    use_impurity_split : bool
        If False (default), use the existing KD-tree style axis selection
        (largest variance) with a median threshold. If True, select the
        (dimension, threshold) pair that maximizes within-node SSE reduction.
    min_impurity_decrease : float
        Minimum required SSE reduction (only used when use_impurity_split=True).
    """
    if current_depth >= max_depth or len(data) <= min_samples_leaf:
        leaf_id = leaf_id_counter[0]
        leaf_id_counter[0] += 1
        return Node(leaf_id=leaf_id)

    if use_impurity_split:
        # Search best (dim, threshold) by SSE reduction
        best = None  # (gain, dim, thr)
        for d in range(dims):
            res = _best_split_1d(data[:, d], min_samples_leaf)
            if res is None:
                continue
            thr, gain = res
            if best is None or gain > best[0]:
                best = (gain, d, thr)

        if best is None or best[0] < min_impurity_decrease:
            leaf_id = leaf_id_counter[0]
            leaf_id_counter[0] += 1
            return Node(leaf_id=leaf_id)

        _, dim_to_split, split_val = best
    else:
        # Existing behavior: choose axis by largest variance and split at median
        # dim_to_split = current_depth % dims
        dim_to_split = int(np.argmax(np.var(data, axis=0)))  # KD TREE axis choice
        split_val = float(np.median(data[:, dim_to_split]))

    left_mask = data[:, dim_to_split] < split_val
    right_mask = ~left_mask
    left_data = data[left_mask]
    right_data = data[right_mask]

    # If all data points fall on one side, make a leaf
    if len(left_data) == 0 or len(right_data) == 0:
        leaf_id = leaf_id_counter[0]
        leaf_id_counter[0] += 1
        return Node(leaf_id=leaf_id)

    left_child = _build_tree(
        left_data,
        dims,
        max_depth,
        min_samples_leaf,
        current_depth + 1,
        leaf_id_counter,
        use_impurity_split,
        min_impurity_decrease,
    )
    right_child = _build_tree(
        right_data,
        dims,
        max_depth,
        min_samples_leaf,
        current_depth + 1,
        leaf_id_counter,
        use_impurity_split,
        min_impurity_decrease,
    )

    return Node(
        dimension=dim_to_split, threshold=split_val, left=left_child, right=right_child
    )


def _get_leaf_id(obs, node: Node):
    while not node.is_leaf:
        if obs[node.dimension] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.leaf_id


def build_binary_partition(
    model: DQN,
    vec_env: VecEnv,
    *,
    max_depth: int,
    min_samples_leaf: int,
    use_impurity_split: bool = False,
    min_impurity_decrease: float = 0.0,
):
    stats = run_test_episodes(model, vec_env)
    dims = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n

    # stats is a list of dicts, each with a 'vals' key. Build an (N, D) matrix.
    data = np.column_stack([np.asarray(s["vals"]) for s in stats])

    # Build a single tree on the full data.
    leaf_id_counter = [0]
    tree = _build_tree(
        data,
        dims,
        max_depth,
        min_samples_leaf,
        0,
        leaf_id_counter,
        use_impurity_split,
        min_impurity_decrease,
    )

    n_leaves = leaf_id_counter[0]

    def discretise(obs: np.ndarray, action: np.ndarray):
        # Remove spurious batch dimension
        obs_vec = obs[0]
        a = int(action[0])
        leaf_id = _get_leaf_id(obs_vec, tree)
        # Return length-1 array; offset by action to make (state, action) unique
        return np.array([a * n_leaves + leaf_id])

    discretise.tree = tree
    discretise.n_leaves = n_leaves

    return discretise, n_leaves * n_actions


# %%
if __name__ == "__main__":
    from crl.cons.env import learn_dqn_policy

    env_name = "CartPole-v1"
    model, vec_env = learn_dqn_policy(
        env_name=env_name,
        seed=5,
        model_dir="models",
        total_timesteps=120_000,
    )

    discretise_tree, n_features_tree = build_binary_partition(
        model,
        vec_env,
        max_depth=6,
        min_samples_leaf=1,
        use_impurity_split=True,
        min_impurity_decrease=3e-1,
    )

    # %%
    stats = run_test_episodes(model, vec_env)
    data = np.column_stack([np.asarray(s["vals"]) for s in stats])

    # %%
    data.shape
    # %%
    from sklearn.neighbors import BallTree

    # %%
    tree = BallTree(data)
    dist, ind = tree.query(data[:3, :], k=5)
    # %%
    ind

    # %%
    tree.get_tree_stats()
    # %%
