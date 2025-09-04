# %%
import numpy as np

from crl.cons.agents import learn_dqn_policy
from crl.cons.discretise import (
    Node,
    build_binary_partition,
    build_tile_coding,
    compute_bin_ranges,
    run_test_episodes,
)

# %%
FONTSIZE = 16
PARAM_NAMES = {
    "Acrobot-v1": [
        r"$\cos(\theta_1)$",
        r"$\sin(\theta_1)$",
        r"$\cos(\theta_2)$",
        r"$\sin(\theta_2)$",
        r"$\omega_1$",
        r"$\omega_2$",
    ],
    "CartPole-v1": ["cartpos", "cartvel", r"$\theta$", r"$\omega$"],
    "MountainCar-v0": ["position", "velocity"],
    "LunarLander-v3": [
        r"$x$",
        r"$y$",
        r"$v_x$",
        r"$v_y$",
        r"$\theta$",
        r"$\omega$",
        "Left contact",
        "Right contact",
    ],
}


def plot_tree_grid_and_leaves_2d(
    stats,
    tree: Node,
    i: int,
    j: int,
    tiles: int = 6,
    obs_quantile: float = 0.1,
    param_names: list[str] | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    x, y = X[:, i], X[:, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Scatter data on both axes
    ax1.scatter(x, y, s=4, alpha=0.15)
    ax2.scatter(x, y, s=4, alpha=0.15)

    # --- Left subplot: uniform grid (quantile-bounded) ---
    _plot_grid(stats, i, j, tiles, obs_quantile, np, X, ax1)

    # --- Right subplot: tree partitions as lines (no fills) ---
    x_bounds = (np.min(x), np.max(x))
    y_bounds = (np.min(y), np.max(y))

    # Collect vertical and horizontal split lines for the (i, j) projection
    def _collect_split_lines_2d(node: Node, xb, yb, vlines, hlines):
        if node.is_leaf:
            return
        t = node.threshold
        d = node.dimension
        if d == i:
            # vertical line at x=t across current y-bounds (if inside x-bounds)
            if xb[0] < t < xb[1]:
                vlines.append((t, yb[0], yb[1]))
            left_x = (xb[0], min(xb[1], t))
            right_x = (max(xb[0], t), xb[1])
            _collect_split_lines_2d(node.left, left_x, yb, vlines, hlines)
            _collect_split_lines_2d(node.right, right_x, yb, vlines, hlines)
        elif d == j:
            # horizontal line at y=t across current x-bounds (if inside y-bounds)
            if yb[0] < t < yb[1]:
                hlines.append((t, xb[0], xb[1]))
            lower_y = (yb[0], min(yb[1], t))
            upper_y = (max(yb[0], t), yb[1])
            _collect_split_lines_2d(node.left, xb, lower_y, vlines, hlines)
            _collect_split_lines_2d(node.right, xb, upper_y, vlines, hlines)
        else:
            # split is on another dimension; propagate bounds unchanged
            _collect_split_lines_2d(node.left, xb, yb, vlines, hlines)
            _collect_split_lines_2d(node.right, xb, yb, vlines, hlines)

    vlines, hlines = [], []
    _collect_split_lines_2d(tree, x_bounds, y_bounds, vlines, hlines)

    # (Optional) de-duplicate exact duplicates
    vset = {(float(x0), float(y0), float(y1)) for (x0, y0, y1) in vlines}
    hset = {(float(y0), float(x0), float(x1)) for (y0, x0, x1) in hlines}

    for x0, y0, y1 in vset:
        ax2.vlines(x0, y0, y1, linewidth=1, colors="k")
    for y0, x0, x1 in hset:
        ax2.hlines(y0, x0, x1, linewidth=1, colors="k")

    # Labels and titles
    ax1.set_title("Uniform grid")
    ax2.set_title("Tree partitions")
    for ax in (ax1, ax2):
        if param_names:
            ax.set_xlabel(f"{param_names[i]}", fontsize=FONTSIZE)
            ax.set_ylabel(f"{param_names[j]}", fontsize=FONTSIZE)
        else:
            ax.set_xlabel(f"obs[{i}]", fontsize=FONTSIZE)
            ax.set_ylabel(f"obs[{j}]", fontsize=FONTSIZE)

    plt.tight_layout()
    plt.show()


def _plot_grid(stats, i, j, tiles, obs_quantile, np, X, ax1):
    state_bins = [tiles] * X.shape[1]
    maxs, mins = compute_bin_ranges(
        stats, obs_quantile=obs_quantile, state_bins=state_bins
    )
    xi_edges = np.linspace(mins[i], maxs[i], tiles + 1)
    yj_edges = np.linspace(mins[j], maxs[j], tiles + 1)
    for xe in xi_edges:
        ax1.axvline(x=xe, linewidth=1, alpha=0.8, color="k")
    for ye in yj_edges:
        ax1.axhline(y=ye, linewidth=1, alpha=0.8, color="k")


def _plot_tile_coding(stats, i, j, tiles: int, tilings: int, obs_quantile: float, ax):
    import numpy as np

    # Compute quantile-bounded ranges similar to _plot_grid
    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    state_bins = [tiles] * X.shape[1]
    maxs, mins = compute_bin_ranges(
        stats, obs_quantile=obs_quantile, state_bins=state_bins
    )

    # Step sizes per dimension
    dx = (maxs[i] - mins[i]) / tiles
    dy = (maxs[j] - mins[j]) / tiles

    # Draw each tiling with evenly spaced offsets within one tile width
    # First tiling is unshifted; subsequent are offset by t/tilings of a tile
    colors = ["k", "tab:orange", "tab:green", "tab:purple", "tab:red", "tab:blue"]
    for t in range(tilings):
        offx = (t / tilings) * dx
        offy = (t / tilings) * dy

        xi_edges = np.linspace(mins[i] + offx, maxs[i] + offx, tiles + 1)
        yj_edges = np.linspace(mins[j] + offy, maxs[j] + offy, tiles + 1)

        # Only draw lines that fall within the original bounds
        cx = colors[t % len(colors)]
        for xe in xi_edges:
            if mins[i] <= xe <= maxs[i]:
                ax.axvline(x=xe, linewidth=1, alpha=0.8, color=cx)
        for ye in yj_edges:
            if mins[j] <= ye <= maxs[j]:
                ax.axhline(y=ye, linewidth=1, alpha=0.8, color=cx)


def plot_discretisations_2x2(
    stats,
    tree: Node,
    i: int,
    j: int,
    grid_tiles: int = 6,
    tile_tiles: int = 4,
    tile_tilings: int = 2,
    obs_quantile: float = 0.1,
    param_names: list[str] | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    x, y = X[:, i], X[:, j]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    (ax_tl, ax_tr), (ax_bl, ax_br) = axs

    # Scatter on used axes (leave bottom-right empty for now)
    for ax in (ax_tl, ax_tr, ax_bl):
        ax.scatter(x, y, s=4, alpha=0.15)

    # Top-left: Uniform grid
    _plot_grid(stats, i, j, grid_tiles, obs_quantile, np, X, ax_tl)
    ax_tl.set_title("Uniform grid")

    # Top-right: Tile coding (tiles, tilings)
    _plot_tile_coding(stats, i, j, tile_tiles, tile_tilings, obs_quantile, ax_tr)
    ax_tr.set_title(f"Tile coding (tiles={tile_tiles}, tilings={tile_tilings})")

    # Bottom-left: Binary partition (tree splits)
    x_bounds = (np.min(x), np.max(x))
    y_bounds = (np.min(y), np.max(y))

    def _collect_split_lines_2d(node: Node, xb, yb, vlines, hlines):
        if node.is_leaf:
            return
        t = node.threshold
        d = node.dimension
        if d == i:
            if xb[0] < t < xb[1]:
                vlines.append((t, yb[0], yb[1]))
            left_x = (xb[0], min(xb[1], t))
            right_x = (max(xb[0], t), xb[1])
            _collect_split_lines_2d(node.left, left_x, yb, vlines, hlines)
            _collect_split_lines_2d(node.right, right_x, yb, vlines, hlines)
        elif d == j:
            if yb[0] < t < yb[1]:
                hlines.append((t, xb[0], xb[1]))
            lower_y = (yb[0], min(yb[1], t))
            upper_y = (max(yb[0], t), yb[1])
            _collect_split_lines_2d(node.left, xb, lower_y, vlines, hlines)
            _collect_split_lines_2d(node.right, xb, upper_y, vlines, hlines)
        else:
            _collect_split_lines_2d(node.left, xb, yb, vlines, hlines)
            _collect_split_lines_2d(node.right, xb, yb, vlines, hlines)

    vlines, hlines = [], []
    _collect_split_lines_2d(tree, x_bounds, y_bounds, vlines, hlines)

    vset = {(float(x0), float(y0), float(y1)) for (x0, y0, y1) in vlines}
    hset = {(float(y0), float(x0), float(x1)) for (y0, x0, x1) in hlines}

    for x0, y0, y1 in vset:
        ax_bl.vlines(x0, y0, y1, linewidth=1, colors="k")
    for y0, x0, x1 in hset:
        ax_bl.hlines(y0, x0, x1, linewidth=1, colors="k")
    ax_bl.set_title("Binary partition")

    # Bottom-right: left empty for now (scatter already drawn)
    ax_br.set_title("")

    # Labels (leave bottom-right without labels)
    for ax in (ax_tl, ax_tr, ax_bl):
        if param_names:
            ax.set_xlabel(f"{param_names[i]}", fontsize=FONTSIZE)
            ax.set_ylabel(f"{param_names[j]}", fontsize=FONTSIZE)
        else:
            ax.set_xlabel(f"obs[{i}]", fontsize=FONTSIZE)
            ax.set_ylabel(f"obs[{j}]", fontsize=FONTSIZE)

    plt.tight_layout()
    plt.show()


# (GMM clustering visualisation removed)


# %%

env_name = "CartPole-v1"
model, vec_env = learn_dqn_policy(
    env_name=env_name,
    seed=5,
    model_dir="models",
    total_timesteps=120_000,
)

# %%
discretise_tile_4x2, n_features_tile_4x2 = build_tile_coding(
    model, vec_env, 4, 2, obs_quantile=0.0005
)

discretise_tree, n_features_tree = build_binary_partition(
    model,
    vec_env,
    max_depth=5,
    min_samples_leaf=1,
    use_impurity_split=True,
    min_impurity_decrease=1e-4,
)
# %%
xaxis = 0
yaxis = 1

stats = run_test_episodes(model, vec_env)
plot_discretisations_2x2(
    stats,
    discretise_tree.tree,
    i=xaxis,
    j=yaxis,
    grid_tiles=6,
    tile_tiles=4,
    tile_tilings=2,
    obs_quantile=0.1,
    param_names=PARAM_NAMES[env_name],
)

# %%
