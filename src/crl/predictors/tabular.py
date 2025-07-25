import numpy as np
from typing import Protocol, runtime_checkable
from collections import deque

type State = int
type Action = int
type Reward = float
type Observation = tuple[State, Action, Reward, State]


@runtime_checkable
class Predictor(Protocol):
    def conformalise(
        self, preds: np.ndarray, state: "State", action: "Action"
    ) -> list["State"]: ...

    def observe(self, obs: "Observation", wm_prob: float) -> None: ...

    def mean_score(self, state: State, action: Action) -> float: ...


class NoPredictor:
    "A no-op predictor - doesn't do any conformal prediction"

    def __init__(self) -> None:
        pass

    def conformalise(
        self, preds: np.ndarray, state: "State", action: "Action"
    ) -> list["State"]:
        return [int(preds.argmax())]

    def observe(self, obs: "Observation", wm_prob: float) -> None:
        pass

    def mean_score(self, state: State, action: Action) -> float:
        return 0.0


class PredictorSAConditioned:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        n_calib: int = 100,
        min_count: int | None = None,
    ):
        self.alpha = alpha
        self.q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)

        # if min_count is not specified, require a calibration set to be full to use it
        self.min_count = min_count or n_calib

        # sets up the calibration sets with a deque per state-action pair. Note this is very gnarly memory-wise for non trivial MDPs
        self.calib_set = np.empty((n_states, n_actions), dtype=object)
        for i, j in np.ndindex(n_states, n_actions):
            self.calib_set[i, j] = deque(maxlen=n_calib)
        self.qhat = np.zeros((n_states, n_actions)) + 1.0

    def conformalise(
        self, preds: np.ndarray, state: "State", action: "Action"
    ) -> list["State"]:
        calib_set = self.calib_set[state, action]

        # if the calibration set isn't full, just return the argmax of the scores
        if len(calib_set) < self.min_count:
            return [int(preds.argmax())]

        prediction_sets = preds >= (1 - self.qhat[state, action])
        prediction_sets = np.flatnonzero(prediction_sets).tolist()
        if not prediction_sets:
            return [int(preds.argmax())]

        return prediction_sets

    def observe(self, obs: "Observation", wm_prob: float) -> None:
        s, a, r, s_prime = obs
        # add a tiny offset to mitigate fp error overestimating 1 - qhat
        self.calib_set[s, a].append(1 - wm_prob + 1e-8)
        self.qhat[s, a] = np.quantile(
            self.calib_set[s, a], self.q_level, method="higher"
        )

    def mean_score(self, state: State, action: Action) -> float:
        return np.mean(self.calib_set[state, action])


class PredictorGlobal:
    def __init__(
        self,
        alpha: float,
        n_calib: int = 100,
        min_count: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)

        # if min_count is not specified, require a calibration set to be full to use it
        self.min_count = min_count or n_calib

        self.calib_set = deque(maxlen=n_calib)
        self.qhat = 1.0

    def conformalise(
        self, preds: np.ndarray, state: "State", action: "Action"
    ) -> list["State"]:
        if len(self.calib_set) < self.min_count:
            return [int(preds.argmax())]

        prediction_sets = preds >= (1 - self.qhat)
        prediction_sets = np.flatnonzero(prediction_sets).tolist()
        if not prediction_sets:
            return [int(preds.argmax())]

        return prediction_sets

    def observe(self, obs: "Observation", wm_prob: float) -> None:
        # add a tiny offset to mitigate fp error overestimating 1 - qhat
        self.calib_set.append(1 - wm_prob + 1e-8)
        self.qhat = np.quantile(self.calib_set, self.q_level, method="higher")

    def mean_score(self, state: State, action: Action) -> float:
        return np.mean(self.calib_set)  # type: ignore


assert issubclass(NoPredictor, Predictor)
assert issubclass(PredictorGlobal, Predictor)
assert issubclass(PredictorSAConditioned, Predictor)
