# %%
import numpy as np
import matplotlib.pyplot as plt
from crl.envs.frozen_lake import make_env
from crl.utils.graphing import get_robust_perf_stats, plot_robust_perf_curve, despine
from crl.predictors.tabular import PredictorGlobal, PredictorSAConditioned, NoPredictor
from crl.agents.tabular import DynaVAgent, AgentParams, Reward
from matplotlib.legend_handler import HandlerTuple

from joblib import Parallel, delayed
from tqdm import tqdm


# %%
def train_agent(
    slip_prob: float = 0.15,
    n_episodes: int = 2_000,
    epsilon_init: float = 1.0,
    rng: int = 42,
    lr: float = 0.1,
    alpha: float = 0.2,
    start_cp: int = 1_500,
) -> tuple[DynaVAgent, list["Reward"]]:
    # setup env
    env = make_env(seed=rng, slip_prob=slip_prob)()

    # setup agent
    params = AgentParams(learning_rate=lr, epsilon=epsilon_init, discount=0.99, rng=rng)
    predictor = PredictorSAConditioned(
        n_states=16, n_actions=4, alpha=alpha, n_calib=100
    )
    # predictor = PredictorGlobal(alpha=alpha, n_calib=1600)
    agent = DynaVAgent(env, params, predictor)

    # train the agent
    returns = []
    for episode in range(n_episodes):
        # simple epsilon schedule encourages heavy exploration early - linearly decays to 0 by the halfway mark
        if episode > 0:
            agent.epsilon = max(
                0, epsilon_init * (1 - (episode - 0) / (n_episodes // 2))
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

    return agent, returns


# %%


# Define the function to be parallelized
def run_single_experiment(
    slip_prob: float, seed: int, alpha: float = 0.2, start_cp: int = 1_500
) -> tuple[DynaVAgent, list[float]]:
    # This function should contain the logic for one run
    # Note: It's often better to instantiate the agent and env inside
    # the worker process to avoid pickling issues.
    agent, exp_returns = train_agent(
        slip_prob=slip_prob, rng=seed, alpha=alpha, start_cp=start_cp
    )
    return agent, exp_returns


# Main loop
n_runs = 50
SLIP_PROBS = [0.0, 0.15, 0.3]
all_returns = {sp: [] for sp in SLIP_PROBS}
agents = {}
cp_start_episode = 1_500
for slip_prob in tqdm(all_returns):
    # print(f"Running experiments for slip_prob={slip_prob}...")
    tasks = (
        delayed(run_single_experiment)(
            slip_prob, run, alpha=0.2, start_cp=cp_start_episode
        )
        for run in range(n_runs)
    )

    # Wrap the Parallel object, which is a generator of results
    results = Parallel(n_jobs=-1)(tasks)
    all_returns[slip_prob] = [res[1] for res in results]  # type: ignore

    # save an agent from each run to play around with
    agents[slip_prob] = results[0][0]  # type: ignore

# %%

fig, ax = plt.subplots(figsize=(10, 6))
legend_handles = []
legend_labels = []
metrics = {}

for ix, slip_prob in enumerate(SLIP_PROBS):
    exp_returns = np.array(all_returns[slip_prob])
    stats = get_robust_perf_stats(exp_returns, n_bootstrap_samples=2000, pbar=True)
    handle, label = plot_robust_perf_curve(
        ax, stats, label=f"p={slip_prob}", color=f"C{ix}", smooth=10
    )
    legend_handles.append(handle)
    legend_labels.append(label)

    metrics[slip_prob] = stats["iqm"][-100:].mean()

for slip_prob, asym_return in metrics.items():
    print(f"Slip prob: {slip_prob:.2f}, Last 100 eps IQM: {asym_return:0.3f}")


despine(ax)
ax.set_title("Performance of agents on different slip probability Frozen Lakes")
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
plt.axvline(x=cp_start_episode, linestyle="--")
plt.tight_layout()
plt.show()
