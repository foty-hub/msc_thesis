# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple
from matplotlib.axes import Axes
from scipy.stats import trim_mean
from typing import Tuple, Dict, Any
from tqdm import trange


# --- Helper Functions ---
def despine(ax: Axes) -> None:
    """Removes the top and right spines from a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _iqm(data: np.ndarray) -> np.ndarray:
    """
    Calculates the Interquartile Mean (IQM) along the first axis.
    As recommended by Agarwal et al., IQM is the mean of the central 50% of the data.
    """
    return trim_mean(data, proportiontocut=0.25, axis=0)  # type: ignore


# --- Core Reusable Functions ---


def get_robust_perf_stats(
    data: np.ndarray,
    n_bootstrap_samples: int = 2000,
    ci_level: float = 0.95,
    pbar: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Computes robust performance statistics (IQM and Bootstrap CIs) for a set of runs.

    Args:
        data: A 2D numpy array of shape (n_runs, n_episodes).
        n_bootstrap_samples: The number of bootstrap samples to draw.
        ci_level: The confidence level for the interval (e.g., 0.95 for 95% CI).
        pbar: Whether or not to show a progress bar for the bootstrapping process.

    Returns:
        A dictionary containing the IQM curve, and the lower and upper CI bounds.
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (n_runs, n_episodes).")

    n_runs = data.shape[0]
    bootstrap_iqms = []

    if pbar:
        bar = trange(n_bootstrap_samples)
    else:
        bar = range(n_bootstrap_samples)
    for _ in range(n_bootstrap_samples):
        bootstrap_indices = np.random.choice(n_runs, size=n_runs, replace=True)
        bootstrap_sample = data[bootstrap_indices, :]
        bootstrap_iqms.append(_iqm(bootstrap_sample))

    bootstrap_iqms = np.asarray(bootstrap_iqms)

    # Calculate CIs from the bootstrap distribution
    lower_percentile = (1.0 - ci_level) / 2.0 * 100
    upper_percentile = (1.0 + ci_level) / 2.0 * 100
    ci_lower = np.percentile(bootstrap_iqms, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_iqms, upper_percentile, axis=0)

    return {"iqm": _iqm(data), "ci_lower": ci_lower, "ci_upper": ci_upper}


def plot_robust_perf_curve(
    ax: Axes,
    stats: Dict[str, np.ndarray],
    label: str,
    color: str = "C0",
    smooth: int = 1,
) -> Tuple[Any, str]:
    """
    Plots a performance curve with IQM and bootstrap CIs on a given matplotlib axis.

    Args:
        ax: The matplotlib axis object to plot on.
        stats: The dictionary of stats from get_robust_perf_stats.
        label: The label for the agent/run.
        color: The color for the plot.
        smooth: The window size for smoothing the curve for visualization.

    Returns:
        A tuple containing the handle for the legend and the formatted label string.
    """

    def _smooth_func(arr, w):
        if w > 1 and len(arr) >= w:
            return arr.reshape(-1, w).mean(1)
        return arr

    iqm_curve = stats["iqm"]
    num_points = len(iqm_curve)
    x = np.linspace(0, num_points, num_points // smooth if smooth > 1 else num_points)

    # Plot the shaded confidence interval
    ax.fill_between(
        x,
        _smooth_func(stats["ci_lower"], smooth),
        _smooth_func(stats["ci_upper"], smooth),
        alpha=0.2,
        color=color,
    )
    # Plot the main IQM line
    ax.plot(x, _smooth_func(iqm_curve, smooth), lw=2, color=color)

    # Create proxy artists for a clean legend
    proxy_line = Line2D([0], [0], lw=2, color=color)
    proxy_patch = Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2, edgecolor="none")
    legend_handle = (proxy_line, proxy_patch)
    legend_label = f"{label}"

    return legend_handle, legend_label


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # 1. Generate some sample data for two different agents
    np.random.seed(42)
    # Agent A: Good performance
    agent_a_returns = np.random.normal(
        loc=np.linspace(0, 5, 1000), scale=1.5, size=(50, 1000)
    )
    # Agent B: Slower learner with some outlier good runs
    agent_b_base = np.random.normal(
        loc=np.linspace(0, 3.5, 1000), scale=2, size=(50, 1000)
    )
    agent_b_returns = (
        agent_b_base + (np.random.random((50, 1)) > 0.9) * 10
    )  # Add some outliers

    # 2. Compute stats for both agents
    stats_a = get_robust_perf_stats(agent_a_returns)
    stats_b = get_robust_perf_stats(agent_b_returns)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    legend_handles = []
    legend_labels = []

    # Plot Agent A
    handle, label = plot_robust_perf_curve(
        ax, stats_a, label="Agent A", color="C0", smooth=10
    )
    legend_handles.append(handle)
    legend_labels.append(label)

    # Plot Agent B
    handle, label = plot_robust_perf_curve(
        ax, stats_b, label="Agent B", color="C1", smooth=10
    )
    legend_handles.append(handle)
    legend_labels.append(label)

    # 4. Finalize plot
    despine(ax)
    ax.set_title("Agent Performance Comparison using Robust Statistics")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Performance (IQM of Returns)")
    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
        frameon=False,
        loc="best",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# %%
