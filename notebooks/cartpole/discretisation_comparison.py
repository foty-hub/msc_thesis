# %%
import numpy as np
from ccnn import calibrate_ccnn
from sklearn.neighbors import KDTree

from crl.cons.agents import learn_dqn_policy
from crl.cons.calib import collect_transitions, signed_score
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


def plot_discretisations_3x1(
    stats,
    tree: Node,
    i: int,
    j: int,
    grid_tiles: int = 6,
    tile_tiles: int = 4,
    tile_tilings: int = 2,
    obs_quantile: float = 0.1,
    param_names: list[str] | None = None,
    # CCNN heatmap inputs (optional)
    ccnn_scores: np.ndarray | None = None,
    ccnn_state_tree: KDTree | None = None,
    ccnn_state_means: np.ndarray | None = None,
    ccnn_state_scales: np.ndarray | None = None,
    ccnn_max_dist: float | None = None,
    knn_k: int = 50,
    heat_res: int = 120,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    x, y = X[:, i], X[:, j]
    # Figure with 3 columns: left scatter, middle uniform grid (filled), right KNN heatmap
    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(16, 5), sharex=True, sharey=True
    )

    # Prepare consistent color mapping based on CCNN scores
    if ccnn_scores is None:
        raise ValueError("ccnn_scores required for colouring.")
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(np.min(ccnn_scores)), vmax=float(np.max(ccnn_scores)))

    # Left: scatter of stats, coloured by kNN mean residual at each point
    if ccnn_state_tree is None or ccnn_state_means is None or ccnn_state_scales is None:
        raise ValueError("CCNN state KDTree and scaler stats are required.")
    X_scaled = (X - ccnn_state_means) / ccnn_state_scales
    dists_pts, ids_pts = ccnn_state_tree.query(X_scaled, k=knn_k)
    vals_pts = np.mean(ccnn_scores[ids_pts], axis=1)
    if ccnn_max_dist is not None:
        far_mask_pts = dists_pts.max(axis=1) > float(ccnn_max_dist)
        fallback_value = float(np.min(ccnn_scores))
        vals_pts = np.where(far_mask_pts, fallback_value, vals_pts)
    ax_l.scatter(x, y, c=vals_pts, s=6, alpha=0.25, cmap=cmap, norm=norm)
    ax_l.set_title("Scatter coloured by residual score (kNN mean)")

    # Middle: uniform grid coloured blocks using same mapping
    state_bins = [grid_tiles] * X.shape[1]
    maxs, mins = compute_bin_ranges(
        stats, obs_quantile=obs_quantile, state_bins=state_bins
    )
    xi_edges = np.linspace(mins[i], maxs[i], grid_tiles + 1)
    yj_edges = np.linspace(mins[j], maxs[j], grid_tiles + 1)

    # Evaluate at cell centres
    centers = []
    rects = []
    for yi in range(grid_tiles):
        y0, y1 = yj_edges[yi], yj_edges[yi + 1]
        yc = 0.5 * (y0 + y1)
        for xi in range(grid_tiles):
            x0, x1 = xi_edges[xi], xi_edges[xi + 1]
            xc = 0.5 * (x0 + x1)
            centers.append((xc, yc))
            rects.append((x0, y0, x1 - x0, y1 - y0))

    # Build a batch of query states for centres, keeping other dims at empirical mean
    full_means = np.mean(X, axis=0)
    Q = np.tile(full_means, (len(centers), 1))
    Q[:, i] = [c[0] for c in centers]
    Q[:, j] = [c[1] for c in centers]
    Q_scaled = (Q - ccnn_state_means) / ccnn_state_scales
    dists_c, ids_c = ccnn_state_tree.query(Q_scaled, k=knn_k)
    vals_c = np.mean(ccnn_scores[ids_c], axis=1)
    if ccnn_max_dist is not None:
        far_mask_c = dists_c.max(axis=1) > float(ccnn_max_dist)
        fallback_value = float(np.min(ccnn_scores))
        vals_c = np.where(far_mask_c, fallback_value, vals_c)

    # Outside-grid falloff: set background to lowest grid-cell value
    fallback_grid_value = float(np.min(vals_c))
    ax_m.set_facecolor(cmap(norm(fallback_grid_value)))

    # Draw coloured rectangles
    for (x0, y0, w, h), v in zip(rects, vals_c):
        ax_m.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                facecolor=cmap(norm(float(v))),
                edgecolor="none",
                alpha=0.85,
            )
        )
    # Overlay black grid lines spanning full extent
    for xe in xi_edges:
        ax_m.axvline(x=xe, color="k", linewidth=1.0, alpha=0.9)
    for ye in yj_edges:
        ax_m.axhline(y=ye, color="k", linewidth=1.0, alpha=0.9)

    ax_m.set_title("Uniform grid (cells coloured by kNN mean)")

    # Right: CCNN kNN mean score heatmap
    xlo, xhi = float(np.min(x)), float(np.max(x))
    ylo, yhi = float(np.min(y)), float(np.max(y))

    full_means = np.mean(X, axis=0)
    base = np.tile(full_means, (heat_res * heat_res, 1))
    gx = np.linspace(xlo, xhi, heat_res)
    gy = np.linspace(ylo, yhi, heat_res)
    GX, GY = np.meshgrid(gx, gy)
    base[:, i] = GX.ravel()
    base[:, j] = GY.ravel()
    Q_scaled = (base - ccnn_state_means) / ccnn_state_scales
    dists, ids = ccnn_state_tree.query(Q_scaled, k=knn_k)
    knn_means = np.mean(ccnn_scores[ids], axis=1)
    if ccnn_max_dist is not None:
        far_mask = dists.max(axis=1) > float(ccnn_max_dist)
        fallback_value = float(np.min(ccnn_scores))
        knn_means = np.where(far_mask, fallback_value, knn_means)
    heat = knn_means.reshape(heat_res, heat_res)

    im = ax_r.imshow(
        heat,
        origin="lower",
        extent=(xlo, xhi, ylo, yhi),
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax_r.set_title(f"CCNN kNN mean score (k={knn_k})")

    # Labels
    for ax in (ax_l, ax_m, ax_r):
        if param_names:
            ax.set_xlabel(f"{param_names[i]}", fontsize=FONTSIZE)
            ax.set_ylabel(f"{param_names[j]}", fontsize=FONTSIZE)
        else:
            ax.set_xlabel(f"obs[{i}]", fontsize=FONTSIZE)
            ax.set_ylabel(f"obs[{j}]", fontsize=FONTSIZE)

    # Match axis limits to full scatter extents
    xlim = (float(np.min(x)), float(np.max(x)))
    ylim = (float(np.min(y)), float(np.max(y)))
    for ax in (ax_l, ax_m, ax_r):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Single shared colorbar below the plots (horizontal)
    # Add extra bottom margin so the colorbar does not clash
    fig.subplots_adjust(bottom=0.22, wspace=0.12)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        mappable,
        ax=(ax_l, ax_m, ax_r),
        orientation="horizontal",
        fraction=0.06,
        pad=0.1,
    )
    cbar.set_label("score", rotation=0)

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
    max_depth=6,
    min_samples_leaf=1,
    use_impurity_split=True,
    min_impurity_decrease=3e-1,
)
# %%
# Calibrate CCNN and prepare state-space KDTree for kNN heatmap
buffer = collect_transitions(model, vec_env, 5_000)
ccnn_scores, ccnn_scaler, ccnn_full_tree, _ = calibrate_ccnn(
    model, buffer, k=50, score_fn=signed_score, scoring_method="td"
)
# Build KDTree over state-only scaled features
ccnn_state_features_scaled = np.asarray(ccnn_full_tree.data)[:, :-1]
ccnn_state_tree = KDTree(ccnn_state_features_scaled)
ccnn_state_means = ccnn_scaler.mean_[:-1]
ccnn_state_scales = ccnn_scaler.scale_[:-1]
"""
Estimate a state-only max-distance threshold consistent with CCNN's falloff.
We mimic calibrate_ccnn's logic: for each point, take the max distance to its
K neighbours (excluding itself), then use a high quantile (0.99) as cutoff.
"""
_k = 15
_dists_train, _ = ccnn_state_tree.query(ccnn_state_features_scaled, k=_k + 1)
_per_point_max = _dists_train.max(axis=1)
ccnn_state_max_dist = float(np.quantile(_per_point_max, 0.99))

xaxis = 0
yaxis = 1

stats = run_test_episodes(model, vec_env)
plot_discretisations_3x1(
    stats,
    discretise_tree.tree,
    i=xaxis,
    j=yaxis,
    grid_tiles=6,
    tile_tiles=4,
    tile_tilings=2,
    obs_quantile=0.1,
    param_names=PARAM_NAMES[env_name],
    ccnn_scores=ccnn_scores,
    ccnn_state_tree=ccnn_state_tree,
    ccnn_state_means=ccnn_state_means,
    ccnn_state_scales=ccnn_state_scales,
    ccnn_max_dist=ccnn_state_max_dist,
    knn_k=_k,
    heat_res=120,
)

# %%
