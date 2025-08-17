# %%
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.dqn import DQN
from PyFixedReps import TileCoder, TileCoderConfig
from crl.cons.discretise.grid import run_test_episodes, compute_bin_ranges


# %%
def build_tile_coding(model: DQN, vec_env: VecEnv, tiles: int, tilings: int):
    stats = run_test_episodes(model, vec_env)
    dims = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n

    state_bins = [tiles] * dims

    # compute mins and maxes of the tiling according to the quantiles of the distribution
    maxs, mins = compute_bin_ranges(stats, obs_quantile=0.1, state_bins=state_bins)
    # rearrange into per-dimension (min, max) tuples for TileCoderConfig
    input_ranges = list(zip(mins, maxs))
    cfg = TileCoderConfig(
        tiles=tiles,
        tilings=tilings,
        dims=dims,
        offset="cascade",
        scale_output=False,
        input_ranges=input_ranges,
    )
    tc = TileCoder(config=cfg)

    n_state_features = tc.features()

    def discretise(obs: np.array, action: np.array):
        # the observation and array come with a spurious batch dimension
        state_vals = tc.get_indices(obs[0])
        # offset so there's a unique id for each action and state pair.
        state_vals = action * n_state_features + state_vals
        return state_vals

    return discretise, n_state_features * n_actions


# %%


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


def _collect_leaf_rectangles_2d(
    node: Node, i: int, j: int, x_bounds, y_bounds, out_rects
):
    """Accumulate rectangles (x_interval, y_interval, leaf_id) for a 2D projection."""
    if node.is_leaf:
        out_rects.append(
            ((x_bounds[0], x_bounds[1]), (y_bounds[0], y_bounds[1]), node.leaf_id)
        )
        return
    t = node.threshold
    if node.dimension == i:
        left_x = (x_bounds[0], min(x_bounds[1], t))
        right_x = (max(x_bounds[0], t), x_bounds[1])
        _collect_leaf_rectangles_2d(node.left, i, j, left_x, y_bounds, out_rects)
        _collect_leaf_rectangles_2d(node.right, i, j, right_x, y_bounds, out_rects)
    elif node.dimension == j:
        lower_y = (y_bounds[0], min(y_bounds[1], t))
        upper_y = (max(y_bounds[0], t), y_bounds[1])
        _collect_leaf_rectangles_2d(node.left, i, j, x_bounds, lower_y, out_rects)
        _collect_leaf_rectangles_2d(node.right, i, j, x_bounds, upper_y, out_rects)
    else:
        _collect_leaf_rectangles_2d(node.left, i, j, x_bounds, y_bounds, out_rects)
        _collect_leaf_rectangles_2d(node.right, i, j, x_bounds, y_bounds, out_rects)


def plot_tree_grid_and_leaves_2d(
    stats, tree: Node, i: int, j: int, tiles: int = 6, obs_quantile: float = 0.1
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib as mpl

    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    x, y = X[:, i], X[:, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Scatter data on both axes
    ax1.scatter(x, y, s=4, alpha=0.15)
    ax2.scatter(x, y, s=4, alpha=0.15)

    # --- Left subplot: uniform grid ---
    state_bins = [tiles] * X.shape[1]
    maxs, mins = compute_bin_ranges(
        stats, obs_quantile=obs_quantile, state_bins=state_bins
    )
    xi_edges = np.linspace(mins[i], maxs[i], tiles + 1)
    yj_edges = np.linspace(mins[j], maxs[j], tiles + 1)
    for xe in xi_edges:
        ax1.axvline(x=xe, linewidth=1, alpha=0.8, c="k")
    for ye in yj_edges:
        ax1.axhline(y=ye, linewidth=1, alpha=0.8, c="k")

    # --- Right subplot: tree leaves as filled rectangles ---
    x_bounds = (np.min(x), np.max(x))
    y_bounds = (np.min(y), np.max(y))
    rects = []
    _collect_leaf_rectangles_2d(tree, i, j, x_bounds, y_bounds, rects)

    # Occupancy-based coloring
    counts = {}
    for row in X:
        lid = _get_leaf_id(row, tree)
        counts[lid] = counts.get(lid, 0) + 1
    occ_vals = np.array(list(counts.values())) if counts else np.array([1.0])
    norm = mcolors.Normalize(vmin=occ_vals.min(), vmax=occ_vals.max())
    # cmap = cm.get_cmap("viridis")
    cmap = mpl.colormaps["viridis"]

    for xb, yb, lid in rects:
        w = xb[1] - xb[0]
        h = yb[1] - yb[0]
        color = cmap(norm(counts.get(lid, 0)))
        face_rgba = (color[0], color[1], color[2], 0.2)
        ax2.add_patch(
            Rectangle(
                (xb[0], yb[0]),
                w,
                h,
                fill=False,
                facecolor=face_rgba,
                edgecolor="k",
                linewidth=0.8,
            )
        )

    # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    # cbar = fig.colorbar(mappable, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    # cbar.set_label("Leaf occupancy")

    # Labels and titles
    ax1.set_title("Uniform grid (lines) with data")
    ax2.set_title("Tree leaves (filled) with data")
    for ax in (ax1, ax2):
        ax.set_xlabel(f"obs[{i}]")
        ax.set_ylabel(f"obs[{j}]")

    plt.tight_layout()
    plt.show()


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


# def main():
if __name__ == "__main__":
    # main()
    from crl.cons.cartpole import learn_dqn_policy

    model, vec_env = learn_dqn_policy(
        "CartPole-v1", seed=5, model_dir="../../../../models"
    )

    discretise_tile, n_features_tile = build_tile_coding(model, vec_env, 6, 1)

    discretise_tree, n_features_tree = build_binary_partition(
        model,
        vec_env,
        max_depth=6,
        min_samples_leaf=1,
        use_impurity_split=True,
        min_impurity_decrease=3e-1,
    )

    # Quick API check: both discretisers should accept (obs: (1, D)), (action: (1,))
    obs = vec_env.reset()

    # Action from the model and a random action
    act_model, _ = model.predict(obs, deterministic=True)
    act_rand = np.array([vec_env.action_space.sample()], dtype=np.int64)

    for label, a in [("model", act_model), ("random", act_rand)]:
        out_tile = discretise_tile(obs, a)
        out_tree = discretise_tree(obs, a)
        print(
            f"[API check:{label}] obs.shape={obs.shape}, action.shape={a.shape} -> "
            f"tile_out.shape={np.shape(out_tile)}, tree_out.shape={np.shape(out_tree)}"
        )

    print(f"{n_features_tile=}, {n_features_tree=}")
    # Visualise in 2D: compare uniform grid vs. learned tree leaves
    stats = run_test_episodes(model, vec_env)
    # %%
    plot_tree_grid_and_leaves_2d(
        stats, discretise_tree.tree, i=2, j=3, tiles=6, obs_quantile=0.1
    )

# %%
