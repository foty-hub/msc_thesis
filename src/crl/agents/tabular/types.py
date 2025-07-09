from typing import Protocol

type State = int
type Action = int
type Reward = float
type Observation = tuple[State, Action, Reward, State]


class Agent(Protocol):
    def select_action(self, state: State) -> Action: ...

    def observe(self, obs: Observation) -> None: ...

    def plan(self) -> None: ...
