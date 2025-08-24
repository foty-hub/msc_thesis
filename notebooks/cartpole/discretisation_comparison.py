# %%
import numpy as np

from crl.cons.agents import learn_dqn_policy
from crl.cons.discretise import (
    Node,
    build_binary_partition,
    build_cluster_partition,
    build_tile_coding,
    compute_bin_ranges,
    run_test_episodes,
)
from crl.cons.discretise.gmm import _scale_with_ranges

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


def plot_gmm_grid_and_clusters_2d(
    stats,
    discretise_fn,
    i: int,
    j: int,
    tiles: int = 6,
    obs_quantile: float = 0.1,
    param_names: list[str] | None = None,
):
    """Visualise grid vs. GMM clusters and flagged anomalies in 2D.

    Parameters
    ----------
    stats : list of dicts with key 'vals'
    discretise_fn : function returned by build_cluster_partition (must have .gmm, .mins, .maxs, .threshold)
    i, j : dimensions to plot
    tiles : grid lines for left subplot
    obs_quantile : passed to grid overlay for consistency
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    x, y = X[:, i], X[:, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Scatter raw data on both subplots
    ax1.scatter(x, y, s=4, alpha=0.15)
    ax2.scatter(x, y, s=4, alpha=0.15)

    # Left: uniform grid
    _plot_grid(stats, i, j, tiles, obs_quantile, np, X, ax1)
    ax1.set_title("Uniform grid")

    # Right: GMM clusters + anomalies
    gmm = discretise_fn.gmm
    mins = discretise_fn.mins
    maxs = discretise_fn.maxs
    tau = discretise_fn.threshold

    Xs = _scale_with_ranges(X, mins, maxs)
    ll = gmm.score_samples(Xs)
    labels = gmm.predict(Xs)
    is_anom = ll < tau

    # Plot inliers by cluster
    for k in range(gmm.n_components):
        mask = (~is_anom) & (labels == k)
        ax2.scatter(X[mask, i], X[mask, j], s=6, alpha=0.6, label=f"C{k}")

    # Plot anomalies
    if np.any(is_anom):
        ax2.scatter(
            X[is_anom, i],
            X[is_anom, j],
            s=20,
            marker="x",
            linewidths=1.0,
            label="anomaly",
        )

    # # Draw covariance ellipses for components (projected to (i,j)) in ORIGINAL units
    # means = gmm.means_
    # covs = gmm.covariances_

    # # scaling transform: x_orig = mins + denom * x_scaled
    # denom_all = np.where(maxs > mins, maxs - mins, 1.0)
    # denom_2d = np.diag(denom_all[[i, j]])  # 2x2 diagonal
    # mins_2d = mins[[i, j]]

    # # 95% equiprobability contour radius for 2D Gaussian
    # c95 = 2.447746830680816  # sqrt(chi2.ppf(0.95, df=2)) without SciPy dep

    # for k in range(gmm.n_components):
    #     # mean in original units
    #     mu_scaled_2d = means[k, [i, j]]
    #     mu_orig_2d = mins_2d + denom_all[[i, j]] * mu_scaled_2d

    #     # covariance in scaled space → original units via S * Σ * S
    #     if gmm.covariance_type == "full":
    #         cov2_scaled = covs[k][np.ix_([i, j], [i, j])]
    #     elif gmm.covariance_type == "diag":
    #         cov2_scaled = np.diag(covs[k][[i, j]])
    #     elif gmm.covariance_type == "spherical":
    #         cov2_scaled = np.eye(2) * covs[k]
    #     else:  # tied
    #         cov2_scaled = covs[np.ix_([i, j], [i, j])]

    #     cov2_orig = denom_2d @ cov2_scaled @ denom_2d

    #     vals, vecs = np.linalg.eigh(cov2_orig)
    #     order = vals.argsort()[::-1]
    #     vals, vecs = vals[order], vecs[:, order]
    #     angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    #     # width/height of 95% ellipse (factor 2*c95 because Ellipse expects full width/height)
    #     width, height = 2.0 * c95 * np.sqrt(vals)
    #     ell = Ellipse(
    #         xy=mu_orig_2d, width=width, height=height, angle=angle, fill=False
    #     )
    #     ax2.add_patch(ell)

    ax2.set_title("GMM clusters (ellipses) + anomalies")
    for ax in (ax1, ax2):
        if param_names:
            ax.set_xlabel(f"{param_names[i]}", fontsize=FONTSIZE)
            ax.set_ylabel(f"{param_names[j]}", fontsize=FONTSIZE)
        else:
            ax.set_xlabel(f"obs[{i}]", fontsize=FONTSIZE)
            ax.set_ylabel(f"obs[{j}]", fontsize=FONTSIZE)
    ax2.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


# %%

env_name = "Acrobot-v1"
model, vec_env = learn_dqn_policy(
    env_name=env_name,
    seed=5,
    model_dir="models",
    total_timesteps=120_000,
)

discretise_tile, n_features_tile = build_tile_coding(
    model, vec_env, 6, 1, obs_quantile=0.0005
)

discretise_tree, n_features_tree = build_binary_partition(
    model,
    vec_env,
    max_depth=6,
    min_samples_leaf=1,
    use_impurity_split=True,
    min_impurity_decrease=3e-1,
)

discretise_cluster, n_features_cluster = build_cluster_partition(
    model,
    vec_env,
    k_min=1,
    k_max=12,
    obs_quantile=0.1,
    scale_bins=6,
    covariance_type="full",
    reg_covar=1e-6,
    max_iter=500,
    n_init=1,
    anomaly_frac=0.1,
    random_state=5,
    use_bic=True,
)

# Quick API check: both discretisers should accept (obs: (1, D)), (action: (1,))
obs = vec_env.reset()

# Action from the model and a random action
act_model, _ = model.predict(obs, deterministic=True)
act_rand = np.array([vec_env.action_space.sample()], dtype=np.int64)
# %%
for label, a in [("model", act_model), ("random", act_rand)]:
    out_tile = discretise_tile(obs, a)
    out_tree = discretise_tree(obs, a)
    out_clust = discretise_cluster(obs, a)
    print(
        f"[API check:{label}] obs.shape={obs.shape}\n action.shape={a.shape} "
        f"\n tile_out.shape={np.shape(out_tile)}\n tree_out.shape={np.shape(out_tree)}, "
        f"\n cluster_out.shape={np.shape(out_clust)}"
    )

print("Feature Counts")
print(f"  Tile/Grid: {int(n_features_tile):,}")
print(f"  Tree     : {int(n_features_tree):,}")
print(f"  Cluster  : {int(n_features_cluster):,}")
# Visualise in 2D: compare uniform grid vs. learned tree leaves

# %%
xaxis = 0
yaxis = 1

stats = run_test_episodes(model, vec_env)
plot_tree_grid_and_leaves_2d(
    stats,
    discretise_tree.tree,
    i=xaxis,
    j=yaxis,
    tiles=6,
    obs_quantile=0.1,
    param_names=PARAM_NAMES[env_name],
)
plot_gmm_grid_and_clusters_2d(
    stats,
    discretise_cluster,
    i=xaxis,
    j=yaxis,
    tiles=6,
    obs_quantile=0.1,
    param_names=PARAM_NAMES[env_name],
)

# %%
