# %% [markdown]
# # Acrobot Test
# The aim of this nb is to investigate *why* conformal calibration fails on the Acrobot environment. Two current hypotheses are:
#
# 1. Acrobot is a particularly chaotic environment - small changes in the dynamics lead to drastically different vistation patterns, so the calibration is all in the constant fallback region and does nothing
# 2. The calibration actually *is* working, but the optimal reward on Acrobot varies massively
#
# To test these hypotheses, I want to do the following:
# 1. Look at the state-space visitation rate of a typical Acrobot policy, and compute the distance between the visitation on Acrobot (maybe a TV distance), compared to Cartpole. Sidenote, how do I guarantee that the distances are comparable? Standardise per-dimension?
# 2. Train a new agent from scratch for each shifted environment, and see how the reward it achieves varies compared to the nominally-trained policy.

# %%
# step 1
#   train DQN on the nominal env
#   track state-space occupancy
#

# %%
# from crl.agents import learn_dqn_policy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm

from crl.agents.dqn import instantiate_vanilla_dqn
from crl.env import instantiate_eval_env

# %%
env_name = "CartPole-v1"
seed = 6
trainsteps = 50_000
param_vals_fine = np.arange(0.1, 4.0, 0.025)
param_vals_coarse = np.arange(0.1, 4.0, 0.1)

param_name = "length"
nominal_val = 0.5
# param_name = "LINK_LENGTH_1"
# nominal_val = 1.0
# Plot theme: "light" or "dark"
theme = "dark"

# %%
model = instantiate_vanilla_dqn(env_name, seed)
model.learn(total_timesteps=trainsteps, progress_bar=True)
# eval_env = instantiate_eval_env('Acrobot-v1', seed=5)


# %%
def observe_occupancy(model: DQN, vec_env: VecEnv, obs_steps: int = 10_000):
    state_dim = vec_env.observation_space.shape[0]
    obs = vec_env.reset()
    states = np.zeros((obs_steps, state_dim))

    for t in range(obs_steps):
        action = model.policy.predict(obs)
        obs, reward, done, info = vec_env.step(action[0])

        states[t] = obs.flatten()

        if done:  # reset and just keep running episodes
            obs = vec_env.reset()

    return states


# %%

vec_env = model.get_env()
states = observe_occupancy(model, vec_env)

# %%


def default_state_labels(env_name: str, n_dims: int) -> list[str]:
    """Return sensible per-dimension labels (Matplotlib mathtext) for common envs."""
    if env_name.startswith("Acrobot"):
        labels = [
            r"$\cos(\theta_1)$",
            r"$\sin(\theta_1)$",
            r"$\cos(\theta_2)$",
            r"$\sin(\theta_2)$",
            r"$\omega_1$",
            r"$\omega_2$",
        ]
    elif env_name.startswith("CartPole"):
        labels = [
            r"$x$",
            r"$\dot{x}$",
            r"$\theta$",
            r"$\dot{\theta}$",
        ]
    else:
        labels = [f"$s_{i}$" for i in range(n_dims)]

    # Ensure the list length matches n_dims
    if len(labels) < n_dims:
        labels = labels + [f"$s_{i}$" for i in range(len(labels), n_dims)]
    else:
        labels = labels[:n_dims]
    return labels


def plot_states(
    states,
    xlims: list[tuple[float, float]] | None = None,
    ylims: list[tuple[float, float]] | None = None,
    labels: list[str] | None = None,
    bins_per_dim: int = 50,
    theme: str = "light",
) -> tuple[Figure, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Plot pairwise scatter (off-diagonal) and per-dimension histograms (diagonal)
    for `states`.

    If `xlims`/`ylims` are provided, they must be lists of (min, max) tuples of
    length equal to the state dimension. These limits are applied consistently
    across the grid so that each row/column shares the same ranges.

    If not provided, limits are inferred from the plotted nominal figure and
    returned so they can be reused for subsequent plots.

    `labels` optionally names each state dimension; when provided, column titles
    (top row) and y-labels (left column) are drawn accordingly using Matplotlib
    mathtext (LaTeX-like) syntax.

    When `xlims` is provided, fixed, equal-width bin edges are constructed using
    `bins_per_dim` so that histograms use consistent bins across frames.
    """
    n_dims = states.shape[1]
    if labels is None:
        labels = [f"$s_{i}$" for i in range(n_dims)]
    else:
        # Guard against length mismatches
        if len(labels) != n_dims:
            labels = (labels + [f"$s_{i}$" for i in range(len(labels), n_dims)])[
                :n_dims
            ]

    # Theme handling
    theme = theme.lower()
    if theme not in {"light", "dark"}:
        raise ValueError("theme must be 'light' or 'dark'")
    fg = "w" if theme == "dark" else "k"
    bg = "k" if theme == "dark" else "w"

    # If xlims are provided, precompute fixed, equal-width bin edges per dimension
    bin_edges = None
    if xlims is not None:
        bin_edges = [
            np.linspace(xmin, xmax, bins_per_dim + 1) for (xmin, xmax) in xlims
        ]

    fig, axes = plt.subplots(n_dims, n_dims, figsize=(12, 12))
    fig.patch.set_facecolor(bg)

    for x, y in np.ndindex((n_dims, n_dims)):
        ax = axes[x, y]
        ax.set_facecolor(bg)
        if x == y:
            bins_arg = bin_edges[x] if bin_edges is not None else bins_per_dim
            ax.hist(states[:, x], bins=bins_arg, color=fg, alpha=0.8)
            if xlims is not None:
                ax.set_xlim(*xlims[x])
        else:
            ax.scatter(states[:, x], states[:, y], alpha=0.01, s=3, c=fg)
            if xlims is not None:
                ax.set_xlim(*xlims[x])
            if ylims is not None:
                ax.set_ylim(*ylims[y])

        # Styling
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Labels along top row and left column
        if x == 0:
            axes[0, y].set_title(labels[y], color=fg)
        if y == 0:
            axes[x, 0].set_ylabel(
                labels[x], rotation=0, ha="right", va="center", labelpad=20, color=fg
            )

    if xlims is None:
        xlims = [axes[d, d].get_xlim() for d in range(n_dims)]
    if ylims is None:
        ylims = [axes[d, d].get_xlim() for d in range(n_dims)]

    return fig, xlims, ylims


# %%
# nominal states
labels = default_state_labels(env_name, states.shape[1])
fig, xlims, ylims = plot_states(states, labels=labels, theme=theme)
plt.show()


# %%
eval_env = instantiate_eval_env(env_name, seed, **{param_name: 1.5})
eval_states = observe_occupancy(model, eval_env)

# %%

scaler = StandardScaler()
scaler.fit(states)

states_sc = scaler.transform(states)
eval_states_sc = scaler.transform(eval_states)
plt.hist(np.linalg.norm(states_sc - eval_states_sc, axis=1, ord=1), bins=100)


plt.title("Standardised distance - between nominal and eval states")

# %%
plt.hist(states_sc.mean(axis=1), label="nominal", alpha=0.8, bins=50)
plt.hist(eval_states_sc.mean(axis=1), label="test", alpha=0.8, bins=50)
plt.legend(frameon=False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)


# %% Energy distance
def _mean_pairwise_dist(X, Y):
    # mean of all pairwise Euclidean distances between rows of X and Y (ordered pairs)
    X2 = np.sum(X * X, axis=1)[:, None]  # (M,1)
    Y2 = np.sum(Y * Y, axis=1)[None, :]  # (1,N)
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    np.maximum(D2, 0.0, out=D2)  # numerical guard
    return np.sqrt(D2).mean()


def energy_distance(p, q, *, return_squared=False, unbiased=False, normalize=False):
    """
    Energy distance between samples p (M,D) and q (N,D).

    If normalize=True, return the DISCO-style normalized energy distance in [0,1]:
        NED = B/T with B = (MN/(M+N)^2) * (2*mean d(p,q) - mean d(p,p) - mean d(q,q)),
        T = mean d(z,z) over the pooled sample z = [p; q].
    Notes:
      - Normalization follows the biased (V-statistic) definitions to guarantee [0,1].
        The 'unbiased' flag affects the unnormalized return only.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1]:
        raise ValueError("p and q must be 2D with the same number of columns.")

    M, N = len(p), len(q)
    a = _mean_pairwise_dist(p, q)  # cross mean
    b = _mean_pairwise_dist(p, p)  # within-P mean
    c = _mean_pairwise_dist(q, q)  # within-Q mean

    # 'Squared' energy distance in the literature:
    E2 = 2.0 * a - b - c

    if normalize:
        # DISCO normalization (always uses biased means to ensure [0,1])
        K = M + N
        z = np.vstack([p, q])
        t = _mean_pairwise_dist(z, z)  # pooled total dispersion
        if t <= 0:
            return 0.0  # degenerate case: all points identical
        ned = (M * N) / (K * K) * (E2 / t)
        return float(np.clip(ned, 0.0, 1.0))

    # Optional U-statistic correction for the unnormalized estimate
    if unbiased:
        if M > 1:
            b *= M / (M - 1)
        if N > 1:
            c *= N / (N - 1)
        E2 = 2.0 * a - b - c

    if return_squared:
        return E2
    return np.sqrt(max(E2, 0.0))


energy_distance(states, eval_states, normalize=True)
# %% GENERATE DISTANCE PLOT
distances = []
for ix, param_val in enumerate(tqdm(param_vals_coarse)):
    eval_env = instantiate_eval_env(env_name, seed, **{param_name: param_val})
    eval_states = observe_occupancy(model, eval_env)
    distances.append(energy_distance(states, eval_states, normalize=True))

# %%
plt.plot(param_vals_coarse, distances, c="k")
plt.axvline(nominal_val, linestyle="--", alpha=0.5, c="k")

ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylim(0, None)
ax.set_xlim(0, None)
ax.grid(visible=True, which="major", axis="y", alpha=0.3)
ax.set_yticks(ticks=[0.0, 0.1, 0.2, 0.3])
ax.set_xticks(ticks=[0.0, 2.0, 4.0])
ax.set_xlabel(param_name.title())
ax.set_ylabel("Normalised energy distance")
ax.text(nominal_val + 0.1, y=0.3 - 0.02, s="Nominal value")
ax.set_title("CartPole")
plt.show()

# %%


# %% GIF GENERATION
# generate gif over changing parameter values
for ix, param_val in enumerate(tqdm(param_vals_fine)):
    eval_env = instantiate_eval_env(env_name, seed, length=param_val)
    eval_states = observe_occupancy(model, eval_env)
    fig, _, _ = plot_states(
        eval_states, xlims, ylims, labels=labels, bins_per_dim=50, theme=theme
    )
    fig.suptitle(
        f"CartPole - pole length: {param_val:.2f}",
        color=("w" if theme == "dark" else "k"),
    )
    plt.savefig(
        f"results/gif/img_{ix + 1}.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()

# %%
# save a gif

# List your PNG files in the desired order
png_files = [f"results/gif/img_{i + 1}.png" for i, _ in enumerate(param_vals_fine)]

# Open the first image
frames = [Image.open(p) for p in png_files]

# Save as GIF
frames[0].save(
    "results/gif/output.gif",
    save_all=True,  # Save as an animation
    append_images=frames[1:],  # The rest of the frames
    duration=50,  # Frame duration in ms
    loop=0,  # 0 = infinite loop
)
