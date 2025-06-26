# %%
import numpy as np
import matplotlib.pyplot as plt
from crl.graph_utils import despine
from collections import deque

# Typing to make function signatures more legible
type State = int
type Action = int
type Reward = float
type Done = bool
type Observation = tuple[State, Reward, Action]

# --- Constants for Readability ---
STATE_MAP: dict[str, State] = {"start": 0, "middle": 1, "goal": 2}
ACTION_MAP: dict[str, Action] = {"a_long": 0, "a_short": 1}
# Reverse mapping for inspection
IDX_TO_STATE: dict[State, str] = {v: k for k, v in STATE_MAP.items()}


class MDPEnv:
    """
    The environment's transition probability `delta` can be a fixed scalar or a
    schedule of values that change on each new episode.
    """

    def __init__(
        self,
        delta: float | list[float],
        a_short_return_reward: Reward = -1.0,
    ):
        """
        Initializes the environment.

        Args:
            delta: The probability of reaching the goal with 'a_short'.
                   Can be a float or a list representing a schedule.
            a_short_return_reward: The reward for taking 'a_short' and returning
                                   to the start state.
        """
        if isinstance(delta, list):
            self.delta_schedule = delta
        else:
            self.delta_schedule = [delta]

        self.a_short_return_reward = a_short_return_reward
        self.current_delta = self.delta_schedule[0]
        self.episode_count = 0
        self.state = STATE_MAP["start"]

    def reset(self) -> State:
        """
        Resets the environment to the start state and updates delta if on a schedule.

        Returns:
            The initial state index.
        """
        self.state = STATE_MAP["start"]
        # Update delta based on the schedule for the new episode
        self.current_delta = self.delta_schedule[
            self.episode_count % len(self.delta_schedule)
        ]
        self.episode_count += 1
        return self.state

    def step(self, action: Action) -> tuple[State, Reward, Done]:
        """
        Executes one time step in the environment.

        Args:
            action: The action to take (0 for 'a_long', 1 for 'a_short').

        Returns:
            A tuple of (next_state, reward, done).
        """
        if self.state == STATE_MAP["start"]:
            if action == ACTION_MAP["a_long"]:
                self.state = STATE_MAP["middle"]
                return self.state, -1.0, False
            elif action == ACTION_MAP["a_short"]:
                if np.random.rand() < self.current_delta:
                    self.state = STATE_MAP["goal"]
                    return self.state, -1.0, True  # Reached goal
                else:
                    self.state = STATE_MAP["start"]
                    return (
                        self.state,
                        self.a_short_return_reward,
                        False,
                    )  # Returned to start

        elif self.state == STATE_MAP["middle"]:
            if action == ACTION_MAP["a_long"]:
                self.state = STATE_MAP["goal"]
                return self.state, -1.0, True  # Reached goal

        raise ValueError(f"Step called from a terminal state or with invalid action.")


class SimpleAgent:
    """An agent that always selects the 'a_long' action."""

    def __init__(self):
        self.action = ACTION_MAP["a_long"]

    def select_action(self, state: int) -> int:
        """Selects 'a_long' regardless of the state."""
        return self.action

    def update_at_episode_end(self, trajectory: list[tuple]):
        """This agent does not learn."""
        pass

    def reset(self):
        """No internal state to reset."""
        pass

    def __repr__(self) -> str:
        """Return a concise representation for interactive inspection."""
        return f"{self.__class__.__name__}(action={self.action})"


class ConformalAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 1.0,
        use_conformal_prediction: bool = False,
        calibration_set_size: int = 400,
        alpha: float = 0.9,
        delta_belief: float = 0.9,
        cp_valid_actions: None | list[Action] = None,
    ) -> None:
        # state value function: S->V
        self.V = np.zeros(len(STATE_MAP))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode_trajectory = []
        self.calibration_scores = deque(maxlen=calibration_set_size)
        self.do_conformal_prediction = use_conformal_prediction
        self.alpha = alpha
        self._init_world_model(delta_belief)
        self.calibration_set_is_full = False
        self.cp_valid_actions = cp_valid_actions or [0, 1]

    def select_action(self, state: State) -> Action:
        """Given the state, returns the next action using conformal prediction"""
        # first, predict the next state for each valid action
        next_action = -1
        next_value = -np.inf

        if state == STATE_MAP["middle"]:
            return ACTION_MAP["a_long"]  # only valid action

        for action in ACTION_MAP.values():
            preds = self.world_model[state, action]
            # perform conformal prediction to get an array of possible next states
            if self.do_conformal_prediction and self.calibration_set_is_full:
                s_pred = self._conformalise(preds)
            else:
                s_pred = [preds.argmax()]

            # for each possible next state, get the value of it
            worst_case_value = np.min([self.V[s_next] for s_next in s_pred])
            if worst_case_value > next_value:
                next_value = worst_case_value
                next_action = action

        return next_action

    def _conformalise(self, scores: np.ndarray) -> list[State]:
        """Output a conformal prediction set, given the world model's scores"""
        # takes world model predictions and returns the conformal prediction set
        prediction_sets = scores >= (1 - self.qhat)
        prediction_sets = np.flatnonzero(prediction_sets).tolist()
        if not prediction_sets:
            return [scores.argmax()]  # type: ignore

        return prediction_sets

    def update_at_episode_end(
        self, trajectory: list[tuple[State, Action, Reward]]
    ) -> None:
        """Run at the end of an episode to update the value function using the
        Monte Carlo return"""
        self._update_value_function(trajectory)

        # update parameters for the conformal predictor
        if self.do_conformal_prediction and self.calibration_set_is_full:
            self._update_conformal_predictor()

        # reset the episode trajectory
        self.episode_trajectory = []

    def _update_value_function(self, trajectory):
        G = 0  # Cumulative discounted return

        # Iterate backwards through the episode to update the value function and fill
        # in the calibration set
        next_state = None
        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            self.V[state] += self.learning_rate * (G - self.V[state])
            # p sure this violates exchangeability... hmmm.....
            if (next_state is not None) and (action in self.cp_valid_actions):
                # note 1-prob here - a low probability means high nonconformity score
                score = 1 - self.world_model[state, action, next_state]
                self.calibration_scores.append(score)

            next_state = state

        # check if the calibration set is full - to start conformalising
        self.calibration_set_is_full = (
            len(self.calibration_scores) == self.calibration_scores.maxlen
        )

    def _update_conformal_predictor(self):
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(self.calibration_scores, q_level, method="higher")

    def _init_world_model(self, delta: float) -> None:
        # world model: SxA->S'
        self.world_model = np.zeros((len(STATE_MAP), len(ACTION_MAP), len(STATE_MAP)))
        # fmt: off
        self.world_model[STATE_MAP['start'], ACTION_MAP["a_short"]] = 1-delta, 0, delta
        self.world_model[STATE_MAP['start'], ACTION_MAP["a_long"]]  =       0, 1, 0
        self.world_model[STATE_MAP['middle'], ACTION_MAP["a_long"]] =       0, 0, 1
        # fmt: on

    def reset(self, reset_conformal_predictor: bool = False) -> None:
        self.trajectory = []

        if reset_conformal_predictor:
            self.calibration_scores.clear()
            self.calibration_set_is_full = False

    def __repr__(self) -> str:
        """Return a concise representation for interactive inspection."""
        return (
            f"{self.__class__.__name__}("
            f"learning_rate={self.learning_rate}, "
            f"gamma={self.gamma}, "
            f"do_conformal_prediction={self.do_conformal_prediction}, "
            f"calibration_set_size={self.calibration_scores.maxlen}, "
            f"alpha={self.alpha}, "
            f"calibration_set_is_full={self.calibration_set_is_full}"
            ")"
        )


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
    for _ in range(num_episodes):
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
CHARTS_DIR = "../results/figures/basic_mdp"
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
# EXPERIMENT - observing agent reward under distribution shift
NUM_EPISODES = 2000
USE_CONFORMAL_PREDICTION = True
CP_VALID_ACTIONS = [1]  # which actions to allow into the conformal prediction set
plot_title = "Reward trajectories without conformal adaptation"


def run_shift_experiment(
    use_conformal_prediction: bool,
    cp_valid_actions: list[int],
    plot_title: str,
    calibration_set_size: int = 400,
    num_episodes: int = 2000,
    n_runs: int = 50,
    delta_max: float = 0.9,
    delta_min: float = 0.1,
):
    schedule1 = [delta_max] * 1000 + [delta_min] * 1000
    schedule2 = np.linspace(delta_max, 0.1, num_episodes).tolist()
    # schedule3 = (0.4 * np.cos(np.linspace(0, np.pi, num_episodes)) + 0.5).tolist()

    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(12, 6))
    for ix, schedule in enumerate([schedule1, schedule2]):
        temp = np.zeros((num_episodes, n_runs))
        for k in range(n_runs):
            env = MDPEnv(delta=schedule)
            agent = ConformalAgent(
                delta_belief=0.9,
                use_conformal_prediction=use_conformal_prediction,
                cp_valid_actions=cp_valid_actions,
                calibration_set_size=calibration_set_size,
            )
            temp[:, k] = run_experiment(agent, env, num_episodes)
        mean_results = temp.mean(axis=1)
        std_results = temp.std(axis=1)
        axes[0, ix].plot(mean_results, label=f"δ={ix}")
        axes[0, ix].fill_between(
            range(num_episodes),
            mean_results - std_results,
            mean_results + std_results,
            alpha=0.2,
            linewidth=0,
        )
        axes[1, ix].plot(schedule)
        despine(axes[0, ix])
        despine(axes[1, ix])
        axes[0, ix].set_ylim(-20, 0)
    axes[0, 0].set_ylabel("Return")
    axes[1, 0].set_ylabel(r"$\delta$", rotation=0)
    axes[1, 0].set_xlabel("Episode #")
    # Set specific y‑axis ticks for the δ plots
    for ax in axes[1]:
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    fig.suptitle(plot_title)
    fig.savefig(
        f"{CHARTS_DIR}/distribution_shift_usecp_{str(use_conformal_prediction).lower()}_validactions_{len(cp_valid_actions)}_calibsize_{calibration_set_size}.pdf"
    )


SHIFT_EXPERIMENTS = [
    dict(
        use_conformal_prediction=False,
        cp_valid_actions=[0, 1],
        plot_title="Agent performance under distributional shift without test-time adaptation",
    ),
    dict(
        use_conformal_prediction=True,
        cp_valid_actions=[0, 1],
        plot_title="ConformalAgent without state-action conditiong",
    ),
    dict(
        use_conformal_prediction=True,
        cp_valid_actions=[1],
        plot_title=r"ConformalAgent with state-action conditiong",
    ),
    dict(
        use_conformal_prediction=True,
        cp_valid_actions=[0, 1],
        calibration_set_size=50,
        plot_title="ConformalAgent with calibration set size 50",
    ),
    dict(
        use_conformal_prediction=True,
        cp_valid_actions=[0, 1],
        calibration_set_size=800,
        plot_title="ConformalAgent with calibration set size 800",
    ),
]

for exp in SHIFT_EXPERIMENTS:
    run_shift_experiment(**exp)  # type: ignore

# %%
