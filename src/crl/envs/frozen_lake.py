import gymnasium as gym
import numpy as np
from typing import Sequence, Callable

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
_PERP = {LEFT: (UP, DOWN), RIGHT: (UP, DOWN), UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT)}

type Schedule = float | Sequence[float] | np.ndarray


class ContinuousSlipWrapper(gym.Wrapper):
    """
    Wrapper for a *FrozenLake*-style environment whose slip probability can vary
    continuously between 0 and 1 and—optionally—follow a user-supplied schedule.

    At every reset the wrapper:

    * Bumps an internal *episode* counter.
    * Sets the current *slip_prob* from the schedule (or leaves it unchanged if
      a scalar was supplied).
    * Injects diagnostic keys into the ``info`` dict returned by
      :py:meth:`reset`.

    During :py:meth:`step` it may replace the agent's chosen action by a
    perpendicular one with probability ``slip_prob`` and records whether a slip
    occurred along with the effective action.

    Attributes
    ----------
    episode : int
        Zero-based episode counter (-1 before the first reset).
    slip_prob : float
        Current probability that the intended action is replaced by a
        perpendicular one.
    schedule : Sequence[float] | numpy.ndarray | None
        Episode-wise schedule of slip probabilities or *None* when a scalar slip_prob was
        given.
    rng : numpy.random.Generator
        Optional random-number generator used for all stochastic decisions.

    Notes
    -----
    The wrapper assumes the underlying environment uses the canonical four
    actions: ``0:LEFT, 1:DOWN, 2:RIGHT, 3:UP`` (as in *FrozenLake*). Slips are
    sampled independently at every step.

    Examples
    --------
    >>> base_env = gym.make("FrozenLake-v1", is_slippery=False)
    >>> env = ContinuousSlipWrapper(base_env, slip_prob=[1.0, 0.5, 0.0])
    >>> obs, info = env.reset()
    >>> info["slip_prob"]
    1.0
    """

    def __init__(
        self,
        env: gym.Env,
        slip_prob: Schedule = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        env : gym.Env
            Environment to wrap. Must implement the four-action *FrozenLake* API.
        slip_prob : float | Sequence[float] | numpy.ndarray, default 0.2
            *Scalar* Fixed slip probability used for all episodes.

            *Sequence / ndarray* Schedule of probabilities applied episode by
            episode.  If the agent outlives the schedule, the final value is
            held constant thereafter.
        rng : numpy.random.Generator, optional
            Source of randomness.  When *None* (default) this falls back to
            :pyfunc:`numpy.random.default_rng`.

        Notes
        -----
        Only light bookkeeping happens here; the probability is applied in
        :py:meth:`step` via :py:meth:`_maybe_slip`.
        """
        super().__init__(env)
        self.episode = -1
        if isinstance(slip_prob, float):
            self.slip_prob = slip_prob  # current value (mutable)
            self.schedule = None
        elif isinstance(slip_prob, (Sequence, np.ndarray)):
            self.schedule = slip_prob
            self.slip_prob = slip_prob[0]

        self.rng = rng or np.random.default_rng()

    # ----- public helpers --------------------------------------------------
    def set_slip_prob(self, p: float) -> None:
        """Override slip probability"""
        self.slip_prob = p
        self.schedule = None

    # ----- gym API ----------------------------------------------------------
    def reset(self, **kwargs):
        self.episode += 1
        self.slip_prob = self._scheduled_prob()
        obs, info = self.env.reset(**kwargs)
        info.update(
            prob=1.0,
            slipped=False,
            effective_action=None,
            slip_prob=self.slip_prob,
            episode=self.episode,
        )
        return obs, info

    def step(self, action):
        eff_action, slipped, p = self._maybe_slip(action)
        obs, rew, term, trunc, info = self.env.step(eff_action)
        info.update(
            prob=p,
            slipped=slipped,
            effective_action=eff_action,
            slip_prob=self.slip_prob,
            episode=self.episode,
        )
        return obs, rew, term, trunc, info

    # ----- internals --------------------------------------------------------
    def _maybe_slip(self, action: int):
        if self.rng.random() < self.slip_prob:
            eff_action = self.rng.choice(_PERP[action])
            return eff_action, True, self.slip_prob / 2.0
        return action, False, 1.0 - self.slip_prob

    def _scheduled_prob(self) -> float:
        if self.schedule is None:
            return self.slip_prob

        if self.episode < len(self.schedule):
            return self.schedule[self.episode]
        # if the episode is beyond the end of the schedule, just take
        # the final value
        return self.schedule[-1]


def make_env(seed: int) -> Callable[[], ContinuousSlipWrapper]:
    "Returns a thunk which generates an environment"

    def _thunk():
        base = gym.make("FrozenLake-v1", is_slippery=False)
        return ContinuousSlipWrapper(
            base, slip_prob=0.15, rng=np.random.default_rng(seed)
        )

    return _thunk
