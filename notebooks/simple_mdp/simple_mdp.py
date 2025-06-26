# %%
import numpy as np
import matplotlib.pyplot as plt
from crl.graph_utils import despine
from itertools import cycle
from agent import SimpleAgent, ConformalAgent
from env import MDPEnv


# --- Experiment Runner ---


def run_experiment(agent, env, num_episodes: int) -> np.ndarray:
    """
    Runs a number of episodes and collects the returns.

    Args:
        agent: The agent to test.
        env: The environment to run in.
        num_episodes: The number of episodes to run.

    Returns:
        A list containing the total return for each episode.
    """
    episodic_returns = []
    for episode_idx in range(num_episodes):
        agent.reset()
        state = env.reset()

        done = False
        episode_return = 0.0
        trajectory = []

        # Max steps to prevent infinite loops in case of a bad policy
        for _ in range(100):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            episode_return += reward
            trajectory.append((state, action, reward))
            state = next_state

            if done:
                # need to add the terminal state S_T, but
                # a_T-1 and R_T are already accounted for in the previous
                # entry
                trajectory.append((next_state, None, None))
                break

        agent.update_at_episode_end(trajectory)
        episodic_returns.append(episode_return)

    return np.array(episodic_returns)


# %%

NUM_EPISODES = 2000
risky_reward = -1
env_delta = 0.9
model_delta = 0.9

# compute results for the simple agent
dumb_agent = SimpleAgent()  # always picks the safe path
env = MDPEnv(delta=env_delta, a_short_return_reward=risky_reward)
dumb_returns = run_experiment(dumb_agent, env, NUM_EPISODES)

# now simulate a few different environments for the smart agent
smart_agent = ConformalAgent(delta_belief=model_delta, use_conformal_prediction=False)


# reverse so colours are in descending order in the legend
def run_basic_experiment(env_delta: float, window: int = 100) -> np.ndarray:
    env = MDPEnv(delta=env_delta, a_short_return_reward=risky_reward)
    # --- Rolling average smoothing ---
    returns = run_experiment(smart_agent, env, NUM_EPISODES)
    window = 100  # size of the moving window
    kernel = np.ones(window) / window
    returns_smoothed = np.convolve(returns, kernel, mode="valid")
    return returns_smoothed


# EXPERIMENT - Comparing loss in reward due to distributional shift
window = 100
n_runs = 50
for env_delta in reversed([0.1, 0.3, 0.5, 0.7, 0.9]):
    results = np.zeros((NUM_EPISODES - window + 1, n_runs))
    for k in range(n_runs):
        results[:, k] = run_basic_experiment(env_delta, window)

    # average over the runs
    mean = results.mean(1)
    std = results.std(1)
    plt.plot(
        range(window - 1, NUM_EPISODES),
        mean,
        label=env_delta,
    )
    # add shading for standard deviation
    x = range(window - 1, NUM_EPISODES)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2,
        color=plt.gca().lines[-1].get_color(),
        linewidth=0,
    )

plt.plot(dumb_returns, label="Safe path", color="k", linestyle="--")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title=r"True $\delta$")
plt.tight_layout()
plt.title(rf"Average Reward of agent with belief that $\delta={model_delta}$")
plt.ylabel(r"Reward")
plt.xlabel(r"Episode #")
despine(plt.gca())
plt.grid(visible=True, which="major", axis="y", alpha=0.4)
CHARTS_DIR = "../../results/figures/basic_mdp"
plt.savefig(
    f"{CHARTS_DIR}/agent_belief_nruns_{n_runs}_smoothing_{window}_risk_reward_{int(np.abs(risky_reward))}.pdf",
    bbox_inches="tight",
)
plt.show()
print(f"note smoothing window of {window}")

# %%
# Analytic plot of value of the starting state
n_steps = 200
x_vals = np.linspace(0.01, 1, n_steps)
y_risky_vals = -1 / x_vals
y_safe_vals = np.zeros(n_steps) - 2  # 2 everywhere

plt.plot(x_vals, y_safe_vals, label=r"$\pi_{safe}$")
plt.plot(x_vals, y_risky_vals, label=r"$\pi_{risky}$")
plt.ylim(-5, 0)
ax = plt.gca()

despine(ax)

# Axis labels, title, and legend
ax.set_xlabel(r"Success probability $\delta$")
ax.set_ylabel("Value of start state")
ax.set_title(
    r"""An agent which mistakenly believes $\delta=0.9$
    can suffer arbitrarily low reward"""
)
ax.legend()
plt.savefig(f"{CHARTS_DIR}/risky_performance.pdf")
plt.show()

# %%
# DISTRIBUTION SHIFT EXPERIMENTS
# Each inner list is a *group* of experiments that will be overlaid on a single
# 2×2 figure.
# TODO: these dicts are fine for this adhoc MDP, but should write a proper experiment runner with
# defined dataclasses that saves configs.
SHIFT_EXPERIMENTS: list[list[dict]] = [
    [  # Comparing different levels of conformal prediction
        dict(
            use_conformal_prediction=False,
            cp_valid_actions=[0, 1],
            plot_title="No-CP",
        ),
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[0, 1],
            plot_title="CP-Uncond",
        ),
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            plot_title="CP-Cond",
        ),
    ],
    [  # Comparing different calibration set sizes
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            calibration_set_size=800,
            plot_title="CP-Cal-800",
        ),
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            calibration_set_size=100,
            plot_title="CP-Cal-100",
        ),
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            calibration_set_size=50,
            plot_title="CP-Cal-50",
        ),
    ],
    [  # Comparing different values of delta
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            calibration_set_size=100,
            delta_max=0.9,
            delta_min=0.5,
            plot_title="0.9 -> 0.5",
        ),
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            calibration_set_size=100,
            delta_max=0.9,
            delta_min=0.3,
            plot_title="0.9 -> 0.3",
        ),
        dict(
            use_conformal_prediction=True,
            cp_valid_actions=[1],
            calibration_set_size=100,
            delta_max=0.9,
            delta_min=0.1,
            plot_title="0.9 -> 0.1",
        ),
    ],
]


# ------------------------------------------------------------------------------ #
# Helper utilities for grouped / over-laid shift experiments
def _run_single_shift_experiment(
    schedule: list[float],
    *,
    num_episodes: int,
    n_runs: int,
    agent_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Run n_runs episodes of a given δ-schedule and return mean & std returns."""
    temp = np.zeros((num_episodes, n_runs))
    for k in range(n_runs):
        env = MDPEnv(delta=schedule)
        agent = ConformalAgent(
            delta_belief=0.9,
            use_conformal_prediction=agent_kwargs.get(
                "use_conformal_prediction", False
            ),
            cp_valid_actions=agent_kwargs.get("cp_valid_actions", [0, 1]),
            calibration_set_size=agent_kwargs.get("calibration_set_size", 400),
        )
        temp[:, k] = run_experiment(agent, env, num_episodes)
    return temp.mean(axis=1), temp.std(axis=1)


def run_shift_experiment_group(
    experiments: list[dict],
    *,
    num_episodes: int = 2000,
    n_runs: int = 50,
    file_suffix: str = "group",
) -> None:
    """Overlay a *group* of experiment configs on a single 2x2 figure."""
    # Figure & axes
    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(12, 6))
    colour_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    handles: list = []
    labels: list[str] = []

    for exp_cfg in experiments:
        colour = next(colour_cycle)
        label = exp_cfg.get("plot_title", "exp")
        labels.append(label)

        # Schedules for the two columns (match original logic)
        delta_max = exp_cfg.get("delta_max", 0.9)
        delta_min = exp_cfg.get("delta_min", 0.1)
        sched1 = [delta_max] * (num_episodes // 2) + [delta_min] * (num_episodes // 2)
        # sched2 = np.linspace(delta_max, 0.1, num_episodes).tolist()
        sched2 = (
            [delta_max] * (num_episodes // 4)
            + np.linspace(delta_max, delta_min, num_episodes // 2).tolist()
            + [delta_min] * (num_episodes // 4)
        )

        # --- Left column (schedule-1) ---
        mean1, std1 = _run_single_shift_experiment(
            sched1, num_episodes=num_episodes, n_runs=n_runs, agent_kwargs=exp_cfg
        )
        (h,) = axes[0, 0].plot(mean1, label=label, color=colour)
        axes[0, 0].fill_between(
            range(num_episodes),
            mean1 - std1,
            mean1 + std1,
            color=colour,
            alpha=0.2,
            linewidth=0,
        )
        axes[1, 0].plot(sched1, color=colour, linewidth=1)

        # --- Right column (schedule-2) ---
        mean2, std2 = _run_single_shift_experiment(
            sched2, num_episodes=num_episodes, n_runs=n_runs, agent_kwargs=exp_cfg
        )
        axes[0, 1].plot(mean2, label=label, color=colour)
        axes[0, 1].fill_between(
            range(num_episodes),
            mean2 - std2,
            mean2 + std2,
            color=colour,
            alpha=0.2,
            linewidth=0,
        )
        axes[1, 1].plot(sched2, color=colour, linewidth=1)

        handles.append(h)

    # Cosmetic tweaks (shared across experiments)
    for col in range(2):
        despine(axes[0, col])
        despine(axes[1, col])
        axes[0, col].set_ylim(-20, 0)
        axes[1, col].set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])

    axes[0, 0].set_ylabel("Return")
    axes[1, 0].set_ylabel(r"$\delta$", rotation=0)
    axes[1, 0].set_xlabel("Episode #")
    axes[1, 1].set_xlabel("Episode #")

    # Place the legend centred *below* the two columns
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(labels),
        frameon=False,
    )

    fig.subplots_adjust(wspace=0.25, bottom=0.25)

    # Title & save
    fig.savefig(
        f"{CHARTS_DIR}/distribution_shift_overlay_{file_suffix}.pdf",
        bbox_inches="tight",
    )
    plt.show()


for g_idx, group in enumerate(SHIFT_EXPERIMENTS, start=1):
    run_shift_experiment_group(group, file_suffix=f"group{g_idx}")

# %%
# %%
