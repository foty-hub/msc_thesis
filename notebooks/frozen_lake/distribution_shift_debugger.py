# %%
from crl.envs.frozen_lake import make_env
from crl.predictors.tabular import PredictorGlobal, PredictorSAConditioned, NoPredictor
from crl.agents.tabular.dyna import DynaVAgent, DynaAgentParams
from crl.agents.tabular.adaptive import AdaptiveLearningAgent, AdaptiveAgentParams
from crl.agents.tabular.types import Reward, Agent


# %%
def train_agent(
    predictor_class,
    slip_prob_initial: float = 0.05,
    slip_prob_final: float = 0.3,
    n_episodes_initial: int = 2_000,
    n_episodes_shifted: int = 2_000,
    epsilon_init: float = 1.0,
    rng: int = 42,
    lr: float = 0.1,
    alpha: float = 0.1,
    start_cp: int = 1_500,
) -> tuple[Agent, list[Reward]]:
    # setup env
    env = make_env(seed=rng, slip_prob=slip_prob_initial)()

    # setup agent
    if predictor_class == PredictorSAConditioned:
        predictor = predictor_class(n_states=16, n_actions=4, alpha=alpha, n_calib=100)
    elif predictor_class == PredictorGlobal:
        predictor = predictor_class(alpha=alpha, n_calib=100)
    else:  # NoPredictor
        predictor = predictor_class()
    # params = DynaAgentParams(
    #     learning_rate=lr, epsilon=epsilon_init, discount=0.95, rng=rng
    # )
    # agent = DynaVAgent(env, params, predictor)

    params = AdaptiveAgentParams(
        learning_rate=lr,
        epsilon=epsilon_init,
        discount=0.95,
        rng=rng,
        adaptation_scale=10,
        use_predictor=True,
    )
    agent = AdaptiveLearningAgent(env, params, predictor)

    # train the agent
    returns = []
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

    return agent, returns


# %%


# Define the function to be parallelized
def run_single_experiment(
    predictor_class,
    seed: int,
    alpha: float = 0.2,
    start_cp: int = 1_500,
    dist_shift_episode: int = 2_000,
) -> tuple[Agent, list[float]]:
    agent, exp_returns = train_agent(
        predictor_class=predictor_class,
        rng=seed,
        alpha=alpha,
        start_cp=start_cp,
        n_episodes_shifted=dist_shift_episode,
    )
    return agent, exp_returns


# Main loop
n_runs = 50
PREDICTORS = [NoPredictor, PredictorGlobal, PredictorSAConditioned]
all_returns = {p.__name__: [] for p in PREDICTORS}
agents = {}
cp_start_episode = 1_500
dist_shift_episode = 2_000
alpha = 0.1

run_single_experiment(
    PredictorSAConditioned,
    seed=42,
    alpha=0.1,
    start_cp=cp_start_episode,
    dist_shift_episode=dist_shift_episode,
)
# %%
# from matplotlib.legend_handler import HandlerTuple

# fig, ax = plt.subplots(figsize=(10, 6))
# legend_handles = []
# legend_labels = []
# metrics = {}

# for ix, predictor_name in enumerate(all_returns):
#     exp_returns = np.array(all_returns[predictor_name])
#     stats = get_robust_perf_stats(exp_returns, n_bootstrap_samples=2000, pbar=True)
#     handle, label = plot_robust_perf_curve(
#         ax, stats, label=f"{predictor_name}", color=f"C{ix}", smooth=10
#     )
#     legend_handles.append(handle)
#     legend_labels.append(label)

#     metrics[predictor_name] = stats["iqm"][-100:].mean()

# for predictor_name, asym_return in metrics.items():
#     print(f"Predictor: {predictor_name}, Last 100 eps IQM: {asym_return:0.3f}")


# despine(ax)
# ax.set_title("Performance of agents under distribution shift")
# ax.set_xlabel("Training Steps")
# ax.set_ylabel("Performance (IQM of Returns)")
# ax.legend(
#     legend_handles,
#     legend_labels,
#     handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
#     frameon=False,
#     loc="best",
# )
# ax.grid(axis="y", linestyle="--", alpha=0.7)
# plt.axvline(x=cp_start_episode, linestyle="--", color="grey", label="CP Starts")
# plt.axvline(x=dist_shift_episode, linestyle="--", color="red", label="Dist. Shift")
# plt.tight_layout()
# plt.show()
