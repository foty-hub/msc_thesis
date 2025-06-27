from typing import Protocol

type Action = int
type State = tuple[int]  # ?
type Reward = float
type Observation = tuple[State, Action, Reward]
type Done = bool


class Agent(Protocol):
    def select_action(self, state: State) -> Action: ...

    def observe(self, obs: Observation) -> None: ...

    def observe_last(self, obs: Observation | State) -> None: ...

    def reset(self) -> None: ...


class ConformalAgent: ...
