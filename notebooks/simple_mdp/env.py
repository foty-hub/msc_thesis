import numpy as np

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

        raise ValueError("Step called from a terminal state or with invalid action.")
