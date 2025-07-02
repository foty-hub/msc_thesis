from typing import Protocol
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from crl.predictors.tabular import Predictor, NoPredictor

type State = int
type Action = int
type Reward = float
type Observation = tuple[State, Action, Reward, State]


class Agent(Protocol):
    def select_action(self, state: State) -> Action: ...

    def observe(self, obs: Observation) -> None: ...

    def reset(self) -> None: ...


@dataclass
class AgentParams:
    # Agent Learning Params
    epsilon: float
    learning_rate: float
    # Conformal Prediction
    use_predictor: bool = False
    # Extra utility
    n_plan_steps: int = 0
    discount: float = 1.0
    rng: int | None = None


_nopred = NoPredictor()


class DynaVAgent:
    def __init__(
        self,
        env: gym.Env,
        params: AgentParams,
        predictor: Predictor,
    ) -> None:
        S: int = env.observation_space.n  # type: ignore
        A: int = env.action_space.n  # type: ignore

        # setup world model, reward model and value function
        self._setup_models(S, A)

        # rng
        if params.rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(params.rng)

        # learning parameters
        self.epsilon = params.epsilon
        self.discount = params.discount
        self.learning_rate = params.learning_rate
        self.n_plan_steps = params.n_plan_steps

        # conformal parameters
        self.predictor = predictor
        self.use_predictor = params.use_predictor

    def _setup_models(self, S: int, A: int) -> None:
        # define arrays for a simple counting world model
        # N: |S|x|A|x|S|
        # M: |S|x|A|x|S|
        # R: |S|x|A|
        self.actions = range(A)
        self.states = range(S)
        self.N = np.zeros((S, A, S))
        self.world_model = np.zeros((S, A, S)) + (1 / S)  # init to equal probs
        self.reward_model = np.zeros((S, A))

        # define the value function
        # V: |S|
        self.V = np.zeros(S)

    def select_action(self, s: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self._random_action()
        return self._greedy_action(s)

    def _random_action(self) -> Action:
        return self.rng.choice(self.actions)

    def _greedy_action(self, s: State) -> Action:
        a_next = -1
        v_next = -np.inf

        for a in self.actions:
            # s_pred = self.world_model[s, a].argmax()
            wm_predictions = self.world_model[s, a]
            if self.use_predictor:
                next_states = self.predictor.conformalise(wm_predictions, s, a)
            else:
                next_states = _nopred.conformalise(wm_predictions, s, a)
            G_worstcase = self.reward_model[s, a] + min(
                self.V[s_pred] for s_pred in next_states
            )
            if G_worstcase > v_next:
                v_next = G_worstcase
                a_next = a

        return a_next

    def observe(self, obs: Observation) -> None:
        """Takes in an observation (S, A, R, S') and updates the value function,
        world model and conformal predictor"""
        s, a, r, s_prime = obs
        # value function - TD(0) update
        self.V[s] += self.learning_rate * (
            r + self.discount * self.V[s_prime] - self.V[s]
        )

        # model learning - counting since tabular
        self.N[s, a, s_prime] += 1
        count_sa = self.N[s, a, :].sum()  # total (s, a) visits
        self.world_model[s, a, :] = self.N[s, a, :] / count_sa

        # Online update - expected reward
        self.reward_model[s, a] += (r - self.reward_model[s, a]) / count_sa

        self.predictor.observe(obs, self.world_model[s, a, s_prime])

    def plan(self) -> None:
        "Conduct imagined updates using the world model to refine the value function"
        # No planning until the model has correctly completed an episode
        if self.reward_model.max() < 1:
            return
        for _ in range(self.n_plan_steps):
            # Sample a previously observed transition
            coords = np.argwhere(self.N)
            s, a, s_prime = coords[self.rng.choice(len(coords))]

            # value function update - TD(0)
            model_r = self.reward_model[s, a]
            self.V[s] += self.learning_rate * (
                model_r + self.discount * self.V[s_prime] - self.V[s]
            )
