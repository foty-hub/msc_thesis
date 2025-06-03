"""Environment wrappers introducing distributional shift for LunarLander."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Adds gaussian noise to observations after a given episode."""

    def __init__(self, env: gym.Env, std: float = 0.0, activate_episode: int = 0):
        super().__init__(env)
        self.std = float(std)
        self.activate_episode = int(activate_episode)
        self._episode = 0

    def reset(self, **kwargs):
        self._episode += 1
        return super().reset(**kwargs)

    def observation(self, observation):
        if self._episode < self.activate_episode:
            return observation
        noise = np.random.normal(0.0, self.std, size=np.array(observation).shape)
        return observation + noise


class FlatlineWrapper(gym.ObservationWrapper):
    """Replaces one element of the observation vector with a constant value."""

    def __init__(self, env: gym.Env, index: int, value: float = 0.0, activate_episode: int = 0):
        super().__init__(env)
        self.index = index
        self.value = value
        self.activate_episode = int(activate_episode)
        self._episode = 0

    def reset(self, **kwargs):
        self._episode += 1
        return super().reset(**kwargs)

    def observation(self, observation):
        if self._episode < self.activate_episode:
            return observation
        obs = np.array(observation, copy=True)
        obs[self.index] = self.value
        return obs


class GradualForceShiftWrapper(gym.Wrapper):
    """Gradually modifies gravity and wind parameters of the environment."""

    def __init__(
        self,
        env: gym.Env,
        gravity_delta: float = 0.0,
        wind_delta: float = 0.0,
        steps: int = 1000,
        activate_episode: int = 0,
    ):
        super().__init__(env)
        self.gravity_delta = gravity_delta
        self.wind_delta = wind_delta
        self.steps = max(1, steps)
        self.base_gravity = float(env.world.gravity[1])
        self.base_wind = float(getattr(env, "wind_power", 0.0))
        self._step_count = 0
        self.activate_episode = int(activate_episode)
        self._episode = 0

    def reset(self, **kwargs):
        self._step_count = 0
        self._episode += 1
        self.env.world.gravity = (0.0, self.base_gravity)
        setattr(self.env, "wind_power", self.base_wind)
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._episode >= self.activate_episode:
            frac = min(1.0, self._step_count / self.steps)
            g = self.base_gravity + frac * self.gravity_delta
            w = self.base_wind + frac * self.wind_delta
            self.env.world.gravity = (0.0, g)
            setattr(self.env, "wind_power", w)
            self._step_count += 1
        return self.env.step(action)


__all__ = [
    "ObservationNoiseWrapper",
    "FlatlineWrapper",
    "GradualForceShiftWrapper",
]
