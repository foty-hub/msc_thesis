import numpy as np
from collections import deque
from env import STATE_MAP, ACTION_MAP, Action, State, Reward


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
            if action is None and reward is None:
                # this is the terminal state, so just set it to the next state and keep going
                next_state = state
                continue
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
