# %%
import matplotlib.pyplot as plt
import numpy as np
from ccnn import calibrate_ccnn
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from sklearn.neighbors import KDTree

from crl.cons.agents import learn_dqn_policy
from crl.cons.calib import collect_transitions, signed_score
from crl.cons.discretise import (
    compute_bin_ranges,
    run_test_episodes,
)

# %%

FONTSIZE = 12
PARAM_NAMES = {
    "Acrobot-v1": [
        r"$\cos(\theta_1)$",
        r"$\sin(\theta_1)$",
        r"$\cos(\theta_2)$",
        r"$\sin(\theta_2)$",
        r"$\omega_1$",
        r"$\omega_2$",
    ],
    "CartPole-v1": ["Cart Position", "Cart Velocity", r"$\theta$", r"$\omega$"],
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


def plot_discretisations_3x1(
    stats,
    i: int,
    j: int,
    grid_tiles: int = 6,
    obs_quantile: float = 0.1,
    param_names: list[str] | None = None,
    # CCNN heatmap inputs (optional)
    ccnn_scores: np.ndarray | None = None,
    ccnn_state_tree: KDTree | None = None,
    ccnn_state_means: np.ndarray | None = None,
    ccnn_state_scales: np.ndarray | None = None,
    ccnn_max_dist: float | None = None,
    use_knn_fallback: bool = True,
    knn_k: int = 50,
    knn_quantile: float = 0.9,
    heat_res: int = 120,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    scatter_alpha: float = 0.25,
):
    X = np.column_stack([np.asarray(s["vals"]) for s in stats])
    x, y = X[:, i], X[:, j]
    # Figure with 3 columns: left scatter, middle uniform grid (filled), right KNN heatmap
    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(8, 4), sharex=True, sharey=True
    )

    # Prepare consistent color mapping based on CCNN scores
    if ccnn_scores is None:
        raise ValueError("ccnn_scores required for colouring.")
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(np.min(ccnn_scores)), vmax=float(np.max(ccnn_scores)))

    # Left: scatter of stats, coloured by kNN quantile residual at each point (controlled by knn_quantile)
    if ccnn_state_tree is None or ccnn_state_means is None or ccnn_state_scales is None:
        raise ValueError(r"CCNN state KDTree and scaler stats are required.")
    X_scaled = (X - ccnn_state_means) / ccnn_state_scales
    dists_pts, ids_pts = ccnn_state_tree.query(X_scaled, k=knn_k)
    vals_pts = np.quantile(ccnn_scores[ids_pts], knn_quantile, axis=1)
    if use_knn_fallback and ccnn_max_dist is not None:
        far_mask_pts = dists_pts.max(axis=1) > float(ccnn_max_dist)
        fallback_value = float(np.max(ccnn_scores))
        vals_pts = np.where(far_mask_pts, fallback_value, vals_pts)
    ax_l.scatter(x, y, c=vals_pts, s=6, alpha=scatter_alpha, cmap=cmap, norm=norm)
    ax_l.set_title("")

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
    vals_c = np.quantile(ccnn_scores[ids_c], knn_quantile, axis=1)
    if use_knn_fallback and ccnn_max_dist is not None:
        far_mask_c = dists_c.max(axis=1) > float(ccnn_max_dist)
        fallback_value = float(np.max(ccnn_scores))
        vals_c = np.where(far_mask_c, fallback_value, vals_c)

    # Outside-grid falloff: set background to lowest grid-cell value
    fallback_grid_value = float(np.max(vals_c))
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

    ax_m.set_title("CC-Disc (Uniform)")

    # Right: CCNN kNN quantile score heatmap (controlled by knn_quantile)
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
    knn_vals = np.quantile(ccnn_scores[ids], knn_quantile, axis=1)
    if use_knn_fallback and ccnn_max_dist is not None:
        far_mask = dists.max(axis=1) > float(ccnn_max_dist)
        fallback_value = float(np.max(ccnn_scores))
        knn_vals = np.where(far_mask, fallback_value, knn_vals)
    heat = knn_vals.reshape(heat_res, heat_res)

    im = ax_r.imshow(
        heat,
        origin="lower",
        extent=(xlo, xhi, ylo, yhi),
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax_r.set_title(f"CC-NN (k={knn_k})")

    # Labels and ticks formatting
    if param_names:
        xlab = f"{param_names[i]}"
        ylab = f"{param_names[j]}"
    else:
        xlab = f"obs[{i}]"
        ylab = f"obs[{j}]"

    # Set labels: keep xlabels on left and right; remove for middle
    ax_l.set_xlabel(xlab, fontsize=FONTSIZE)
    ax_l.set_ylabel(ylab, fontsize=FONTSIZE)
    ax_m.set_xlabel("")
    ax_m.set_ylabel("")  # remove ylabels on middle
    ax_r.set_xlabel(xlab, fontsize=FONTSIZE)
    ax_r.set_ylabel("")  # remove ylabels on right

    # Remove x ticks on all three, and remove all y ticks
    for ax in (ax_l, ax_m, ax_r):
        ax.set_xticks([])
        ax.set_yticks([])

    # Match axis limits to full scatter extents, overridden by optional kwargs
    default_xlim = (float(np.min(x)), float(np.max(x)))
    default_ylim = (float(np.min(y)), float(np.max(y)))
    final_xlim = xlim if xlim is not None else default_xlim
    final_ylim = ylim if ylim is not None else default_ylim
    for ax in (ax_l, ax_m, ax_r):
        ax.set_xlim(final_xlim)
        ax.set_ylim(final_ylim)

    # Single shared colorbar below the plots (horizontal)
    # Add extra bottom margin so the colorbar does not clash
    fig.subplots_adjust(bottom=0.28, wspace=0.12)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        mappable,
        ax=(ax_l, ax_m, ax_r),
        orientation="horizontal",
        fraction=0.06,
        pad=0.1,
    )
    cbar.set_label("score", rotation=0)
    # Only show end ticks on the colorbar
    cbar.set_ticks([round(norm.vmin, 2), round(norm.vmax, 2)])


# %%

env_name = "CartPole-v1"
model, vec_env = learn_dqn_policy(
    env_name=env_name,
    seed=5,
    model_dir="models",
    total_timesteps=120_000,
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

# Estimate a state-only max-distance threshold consistent with CCNN's falloff.
# We mimic calibrate_ccnn's logic: for each point, take the max distance to its
# K neighbours (excluding itself), then use a high quantile (0.99) as cutoff.
stats = run_test_episodes(model, vec_env)
# %%
_k = 50
obs_quantile = 0.95
_dists_train, _ = ccnn_state_tree.query(ccnn_state_features_scaled, k=_k + 1)
_per_point_max = _dists_train.max(axis=1)
ccnn_state_max_dist = float(np.quantile(_per_point_max, obs_quantile))

xaxis = 0
yaxis = 1

plot_discretisations_3x1(
    stats,
    i=xaxis,
    j=yaxis,
    grid_tiles=6,
    obs_quantile=1 - obs_quantile,
    param_names=PARAM_NAMES[env_name],
    ccnn_scores=ccnn_scores,
    ccnn_state_tree=ccnn_state_tree,
    ccnn_state_means=ccnn_state_means,
    ccnn_state_scales=ccnn_state_scales,
    ccnn_max_dist=ccnn_state_max_dist,
    knn_k=_k,
    knn_quantile=0.8,
    heat_res=120,
    xlim=None,
    ylim=None,
    use_knn_fallback=True,
    scatter_alpha=0.1,
)

plt.savefig(
    f"../../results/figures/state_occupancy/{env_name}_{xaxis}_{yaxis}.pdf",
    bbox_inches="tight",
)
plt.show()

# %%
