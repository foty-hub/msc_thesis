# %%
import numpy as np
import matplotlib.pyplot as plt
from crl.envs.frozen_lake import make_env
from crl.utils.graphing import get_robust_perf_stats, plot_robust_perf_curve, despine
from crl.predictors.tabular import PredictorGlobal, PredictorSAConditioned, NoPredictor
from crl.agents.tabular.dyna import DynaVAgent, DynaAgentParams
from crl.agents.tabular.types import Reward, Agent
from crl.agents.tabular.adaptive import AdaptiveLearningAgent, AdaptiveAgentParams

from joblib import Parallel, delayed
from tqdm import tqdm


# %%
def train_agent(
    predictor_class,
    slip_prob_initial: float = 0.05,
    slip_prob_final: float = 0.6,
    n_episodes_initial: int = 2_000,
    n_episodes_shifted: int = 2_000,
    epsilon_init: float = 1.0,
    rng: int = 42,
    lr: float = 0.05,
    alpha: float = 0.05,
    start_cp: int = 1_500,
) -> tuple[Agent, list[Reward], list[float]]:
    # setup env
    env = make_env(seed=rng, slip_prob=slip_prob_initial)()

    # setup agent
    if predictor_class == PredictorSAConditioned:
        predictor = predictor_class(
            n_states=16,
            n_actions=4,
            alpha=alpha,
            n_calib=200,
        )
        adaptation_scale = 10
    elif predictor_class == PredictorGlobal:
        predictor = predictor_class(
            alpha=alpha,
            n_calib=200,
        )
        adaptation_scale = 10
    else:  # NoPredictor
        predictor = predictor_class()
        adaptation_scale = 1

    # params = DynaAgentParams(
    #     learning_rate=lr, epsilon=epsilon_init, discount=0.95, rng=rng
    # )
    # agent = DynaVAgent(env, params, predictor)
    params = AdaptiveAgentParams(
        learning_rate=lr,
        epsilon=epsilon_init,
        discount=0.95,
        rng=rng,
        adaptation_scale=adaptation_scale,
        use_predictor=True,
    )
    agent = AdaptiveLearningAgent(env, params, predictor)

    # train the agent
    returns = []
    lrs = []
    for episode in range(n_episodes_initial + n_episodes_shifted):
        if episode == n_episodes_initial:
            env.set_slip_prob(slip_prob_final)

        # simple epsilon schedule encourages heavy exploration early - linearly decays to 0 by the halfway mark
        if episode > 0:
            agent.epsilon = max(
                0, epsilon_init * (1 - (episode - 0) / (n_episodes_initial // 2))
            )

        # option to ignore conformal prediction during early learning
        if episode > start_cp:
            agent.use_predictor = True

        done = False
        state, info = env.reset()
        while not done:
            # execute an action to interact with the MDP
            action = agent.select_action(state)
            state_next, reward, terminated, truncated, info = env.step(action)
            obs = (state, action, reward, state_next)
            agent.observe(obs)  # type: ignore
            state = state_next

            # By default there's no planning happen - it complicates things
            agent.plan()

            done = terminated or truncated
            if done:
                returns.append(reward)
                lrs.append(agent.learning_rate)

    return agent, returns, lrs


# %%


# Define the function to be parallelized
def run_single_experiment(
    predictor_class,
    seed: int,
    alpha: float = 0.2,
    start_cp: int = 1_500,
    dist_shift_episode: int = 2_000,
) -> tuple[Agent, list[float], list[float]]:
    agent, exp_returns, lrs = train_agent(
        predictor_class=predictor_class,
        rng=seed,
        alpha=alpha,
        start_cp=start_cp,
        n_episodes_shifted=dist_shift_episode,
    )
    return agent, exp_returns, lrs


# Main loop
n_runs = 50
PREDICTORS = [NoPredictor, PredictorGlobal, PredictorSAConditioned]
all_returns = {p.__name__: [] for p in PREDICTORS}
all_lrs = {p.__name__: [] for p in PREDICTORS}
agents = {}
cp_start_episode = 1_500
dist_shift_episode = 2_000
alpha = 0.03

for predictor_class in tqdm(PREDICTORS):
    # print(f"Running experiments for predictor={predictor_class.__name__}...")
    tasks = (
        delayed(run_single_experiment)(
            predictor_class,
            run,
            alpha=alpha,
            start_cp=cp_start_episode,
            dist_shift_episode=dist_shift_episode,
        )
        for run in range(n_runs)
    )

    results = Parallel(n_jobs=-1)(tasks)
    all_returns[predictor_class.__name__] = [res[1] for res in results]  # type: ignore
    all_lrs[predictor_class.__name__] = [res[2] for res in results]  # type: ignore

    # save an agent from each run to play around with
    agents[predictor_class.__name__] = results[0][0]  # type: ignore

# %%
from matplotlib.legend_handler import HandlerTuple

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
legend_handles = []
legend_labels = []
metrics = {}
returns_hist = {}
learning_rate_metrics = {}

# --- top subplot: returns ---
for ix, predictor_name in enumerate(all_returns):
    exp_returns = np.array(all_returns[predictor_name])
    stats = get_robust_perf_stats(exp_returns, n_bootstrap_samples=2000, pbar=True)
    handle, label = plot_robust_perf_curve(
        ax1, stats, label=f"{predictor_name}", color=f"C{ix}", smooth=10
    )
    legend_handles.append(handle)
    legend_labels.append(label)

    metrics[predictor_name] = stats["iqm"][-100:].mean()
    returns_hist[predictor_name] = exp_returns[:, -100:].mean(1)

# --- bottom subplot: learning rate ---
for ix, predictor_name in enumerate(all_lrs):
    exp_lrs = np.array(all_lrs[predictor_name])
    stats_lr = get_robust_perf_stats(exp_lrs, n_bootstrap_samples=2000, pbar=True)
    plot_robust_perf_curve(
        ax2, stats_lr, label=f"{predictor_name}", color=f"C{ix}", smooth=10
    )
    learning_rate_metrics[predictor_name] = stats_lr["iqm"][-100:].mean()

# Optional summary prints
for predictor_name, asym_return in metrics.items():
    print(f"Predictor: {predictor_name}, Last 100 eps IQM: {asym_return:0.3f}")
for predictor_name, asym_lr in learning_rate_metrics.items():
    print(f"Predictor: {predictor_name}, Last 100 eps LR IQM: {asym_lr:0.6f}")

# Formatting
despine(ax1)
despine(ax2)
ax1.set_title("Performance of agents under distribution shift")
ax1.set_ylabel("Performance (IQM of Returns)")
ax2.set_title("Learning Rate over Training")
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Learning Rate (IQM)")
ax1.legend(
    legend_handles,
    legend_labels,
    handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
    frameon=False,
    loc="best",
)

# vertical reference lines
for ax in (ax1, ax2):
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axvline(x=cp_start_episode, linestyle="--", color="grey")
    ax.axvline(x=dist_shift_episode, linestyle="--", color="red")

plt.tight_layout()
plt.show()


# %%
# histogram of the 100 final returns
bins = np.linspace(0, 1, 20)

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(6, 9))

axes[0].hist(returns_hist["PredictorSAConditioned"], bins=bins, alpha=0.8)
axes[0].set_title("PredictorSAConditioned")

axes[1].hist(returns_hist["PredictorGlobal"], bins=bins, alpha=0.8)
axes[1].set_title("PredictorGlobal")

axes[2].hist(returns_hist["NoPredictor"], bins=bins, alpha=0.8)
axes[2].set_title("NoPredictor")

# Optional: enforce identical limits explicitly (sharex/sharey already sync them)
axes[0].set_xlim(bins[0], bins[-1])
axes[0].set_ylim(bottom=0)

plt.tight_layout()
plt.show()

# %%
