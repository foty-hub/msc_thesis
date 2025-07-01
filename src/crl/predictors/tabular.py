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


class PredictorSAConditioned:
    def __init__(
        self,
        S: int,
        A: int,
        alpha: float,
        n_calib: int = 100,
        min_count: int | None = None,
    ):
        self.alpha = alpha
        self.q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib

        # if min_count is not specified, require a calibration set to be full to use it
        self.min_count = min_count or n_calib

        # sets up the calibration sets with a deque per state-action pair. Note this is very gnarly memory-wise for non trivial MDPs
        self.calib_set = np.empty((S, A), dtype=object)
        for i, j in np.ndindex(S, A):
            self.calib_set[i, j] = deque(maxlen=n_calib)
        self.qhat = np.zeros((S, A)) + 1.0

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
            return [preds.argmax()]  # type: ignore

        return prediction_sets

    def observe(self, obs: "Observation", wm_prob: float) -> None:
        s, a, r, s_prime = obs
        self.calib_set[s, a].append(1 - wm_prob)
        self.qhat[s, a] = np.quantile(
            self.calib_set[s, a], self.q_level, method="higher"
        )


class PredictorGlobal:
    def __init__(
        self, alpha: float, n_calib: int = 100, min_count: int | None = None
    ) -> None:
        self.alpha = alpha
        self.q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib

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
            return [preds.argmax()]  # type: ignore

        return prediction_sets

    def observe(self, obs: "Observation", wm_prob: float) -> None:
        self.calib_set.append(1 - wm_prob)
        self.qhat = np.quantile(self.calib_set, self.q_level, method="higher")


assert issubclass(NoPredictor, Predictor)
assert issubclass(PredictorGlobal, Predictor)
assert issubclass(PredictorSAConditioned, Predictor)
