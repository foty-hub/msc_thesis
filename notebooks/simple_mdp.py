# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from crl.graph_utils import despine

# --- Constants for Readability ---
STATE_MAP: dict[str, int] = {"start": 0, "middle": 1, "goal": 2}
ACTION_MAP: dict[str, int] = {"a_long": 0, "a_short": 1}
# Reverse mapping for inspection
IDX_TO_STATE: dict[int, str] = {v: k for k, v in STATE_MAP.items()}


class MDPEnv:
    """
    Implements the MDP described in the image.

    The environment's transition probability `delta` can be a fixed scalar or a
    schedule of values that change on each new episode.
    """

    def __init__(self, delta: float | list[float], a_short_return_reward: float = -1.0):
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

    def reset(self) -> int:
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

    def step(self, action: int) -> tuple[int, float, bool]:
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

        # If in 'goal' state, episode is already done.
        # This part should ideally not be reached if used correctly.
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


class SmartAgent:
    """
    Learns a state-value function V(s) using first-visit Monte Carlo and
    selects actions by querying an internal world model.
    """

    def __init__(
        self,
        model_delta: float,
        learning_rate: float = 0.1,
        gamma: float = 1.0,
        a_short_return_reward: float = -1.0,
    ):
        """
        Args:
            model_delta: The agent's internal, fixed belief about the 'a_short'
                         success probability.
            learning_rate: The alpha parameter for the V-function update.
            gamma: The discount factor.
        """
        self.model_delta = model_delta
        self.alpha = learning_rate
        self.gamma = gamma
        self.a_short_return_reward = a_short_return_reward
        self.V = np.zeros(len(STATE_MAP))  # V[start], V[middle], V[goal]
        self.trajectory = []

    def reset(self):
        """Clears the trajectory for the new episode."""
        self.trajectory = []

    def select_action(self, state: int) -> int:
        """
        Selects an action based on a 1-step lookahead using the internal world model.
        """
        if state == STATE_MAP["middle"]:
            return ACTION_MAP["a_long"]

        if state == STATE_MAP["start"]:
            # Value of taking 'a_long'
            # Expected reward is -1, next state is 'middle'
            q_long = -1.0 + self.gamma * self.V[STATE_MAP["middle"]]

            # Value of taking 'a_short' according to the agent's model
            val_if_success = -1.0 + self.gamma * self.V[STATE_MAP["goal"]]
            val_if_fail = (
                self.a_short_return_reward + self.gamma * self.V[STATE_MAP["start"]]
            )
            q_short = (
                self.model_delta * val_if_success + (1 - self.model_delta) * val_if_fail
            )

            return ACTION_MAP["a_long"] if q_long >= q_short else ACTION_MAP["a_short"]

        raise Exception("No action selected")

    def update_at_episode_end(self, trajectory: list[tuple]):
        """
        Updates the value function using first-visit Monte Carlo.
        """
        G = 0  # Cumulative discounted return
        visited_states = set()

        # Iterate backwards through the episode
        for state, _, reward in reversed(trajectory):
            G = reward + self.gamma * G
            if state not in visited_states:
                # First-visit MC update
                self.V[state] += self.alpha * (G - self.V[state])
                visited_states.add(state)


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
smart_agent = SmartAgent(
    model_delta=model_delta,
    a_short_return_reward=risky_reward,
)


# reverse so colours are in descending order in the legend
def run_smart_experiment(env_delta: float, window: int = 100) -> np.ndarray:
    env = MDPEnv(delta=env_delta, a_short_return_reward=risky_reward)
    # --- Rolling average smoothing ---
    returns = run_experiment(smart_agent, env, NUM_EPISODES)
    window = 100  # size of the moving window
    kernel = np.ones(window) / window
    returns_smoothed = np.convolve(returns, kernel, mode="valid")
    return returns_smoothed


# %%
# EXPERIMENT - Comparing loss in reward due to distributional shift
window = 100
n_runs = 50
for env_delta in reversed([0.1, 0.3, 0.5, 0.7, 0.9]):
    results = np.zeros((NUM_EPISODES - window + 1, n_runs))
    for k in range(n_runs):
        results[:, k] = run_smart_experiment(env_delta, window)

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
# EXPERIMENT - observing agent reward under active distribution shift
NUM_EPISODES = 2000
schedule1 = [0.9] * 1000 + [0.1] * 1000
schedule2 = np.linspace(0.9, 0.1, NUM_EPISODES).tolist()
schedule3 = (0.4 * np.cos(np.linspace(0, np.pi, NUM_EPISODES)) + 0.5).tolist()

n_runs = 50  # number of repetitions for averaging

fig, axes = plt.subplots(2, 3, sharex="col", sharey="row", figsize=(12, 6))
results_matrix = np.zeros((NUM_EPISODES, n_runs, 3))
for ix, schedule in enumerate([schedule1, schedule2, schedule3]):
    temp = np.zeros((NUM_EPISODES, n_runs))
    for k in range(n_runs):
        env = MDPEnv(delta=schedule)
        agent = SmartAgent(model_delta=0.9)
        temp[:, k] = run_experiment(agent, env, NUM_EPISODES)
    mean_results = temp.mean(axis=1)
    std_results = temp.std(axis=1)
    axes[0, ix].plot(mean_results, label=f"δ={ix}")
    axes[0, ix].fill_between(
        range(NUM_EPISODES),
        mean_results - std_results,
        mean_results + std_results,
        alpha=0.2,
        linewidth=0,
    )
    axes[1, ix].plot(schedule)
    despine(axes[0, ix])
    despine(axes[1, ix])
axes[0, 0].set_ylabel("Reward")
axes[1, 0].set_ylabel(r"$\delta$", rotation=0)
fig.suptitle("Reward trajectories without conformal adaptation")
fig.savefig(f"{CHARTS_DIR}/distribution_shift_no_adaptation.pdf")
# %%
