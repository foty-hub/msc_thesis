"""Utilities to run DQN experiments under different distributional shifts."""

import argparse
from typing import Optional

import torch
import gymnasium as gym
import numpy as np

from dqn_agent import DQNAgent, train_dqn, evaluate
from shift_wrappers import (
    ObservationNoiseWrapper,
    FlatlineWrapper,
    GradualForceShiftWrapper,
)


def make_env(name: str, shift: Optional[str] = None, activate_episode: int = 0) -> gym.Env:
    env = gym.make(name)
    if shift == "noise":
        env = ObservationNoiseWrapper(env, std=0.1, activate_episode=activate_episode)
    elif shift == "sensor_failure":
        env = FlatlineWrapper(env, index=0, value=0.0, activate_episode=activate_episode)
    elif shift == "force_change":
        env = GradualForceShiftWrapper(
            env, gravity_delta=-2.0, wind_delta=5.0, activate_episode=activate_episode
        )
    return env


def run(args: argparse.Namespace) -> None:
    env = make_env(args.env_name, args.shift, args.activate_episode)
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.mode == "train":
        scores = train_dqn(env, agent, n_episodes=args.episodes)
        np.save("scores.npy", np.array(scores))
        torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
    else:
        rewards = evaluate(agent, env, episodes=args.episodes)
        print("Evaluation rewards:", rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"], help="Run training or evaluation")
    parser.add_argument("--env-name", default="LunarLander-v3")
    parser.add_argument("--shift", choices=["noise", "sensor_failure", "force_change", None], default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--activate-episode", type=int, default=0, help="Episode on which to activate the shift")
    args = parser.parse_args()
    run(args)
