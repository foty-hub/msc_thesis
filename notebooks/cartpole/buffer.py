import numpy as np
from collections import namedtuple

# -----------------------------------------------------------------------------
# 1. Replay buffer: simple ring buffer (fixed capacity, FIFO eviction)
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

    def __getitem__(self, key):
        return self.buffer[key]

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[i] for i in idx]
