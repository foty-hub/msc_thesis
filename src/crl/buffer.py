from collections import namedtuple
from collections.abc import Iterator

import numpy as np

# -----------------------------------------------------------------------------
# Replay buffer: simple ring buffer (fixed capacity, FIFO)
# -----------------------------------------------------------------------------
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "next_action", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.pos = 0
        self.full = False

    def push(self, *transition_fields):
        self.buffer[self.pos] = Transition(*transition_fields)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or (self.pos == 0)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def __iter__(self) -> Iterator[Transition]:
        if self.full:
            ordered = self.buffer[self.pos :] + self.buffer[: self.pos]
        else:
            ordered = self.buffer[: self.pos]
        return iter(ordered)

    def __getitem__(self, key: int | slice) -> Transition | list[Transition]:
        ordered = list(self)
        return ordered[key]

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self), batch_size, replace=False)
        transitions = list(self)
        return [transitions[i] for i in idx]
