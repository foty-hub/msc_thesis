# %%
from typing import Callable, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from tqdm import tqdm

from crl.agents import learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    correction_for,
    fill_calib_sets,
)
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env

# %%

AggregationStrategy = Literal["max", "mean", "median"]
N_CALIB_TRANSITIONS = 50_000

model, vec_env = learn_dqn_policy("CartPole-v1", seed=5)
# %%


def run_eval(
    model: DQN,
    discretise: Callable,
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    qhats: np.ndarray,
    agg: str = "max",
    clip_correction: bool = False,
) -> list[float]:
    episodic_returns = []

    # Determine number of discrete actions
    num_actions = getattr(ep_env.action_space, "n", None)
    if num_actions is None:
        raise ValueError(
            f"run_eval expects a discrete action space; got {type(ep_env.action_space)}"
        )

    for ep in range(num_eps):
        obs = ep_env.reset()
        for t in range(1000):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

            if conformalise:
                # Adjust the qvalues of each action using
                # the correction from CP
                for a in range(num_actions):
                    correction = correction_for(
                        obs,
                        a,
                        qhats,
                        discretise,
                        agg=agg,
                        clip_correction=clip_correction,
                    )
                    q_vals[a] -= correction

            action = q_vals.argmax().numpy().reshape(1)

            obs, reward, done, info = ep_env.step(action)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break
    return episodic_returns


buffer = collect_transitions(model, vec_env, n_transitions=N_CALIB_TRANSITIONS)
# %%
NUM_EVAL_EPS = 100

tiling_settings = [
    dict(tiles=6, tilings=1),  # current champ
    dict(tiles=6, tilings=2),
    dict(tiles=6, tilings=4),
    dict(tiles=4, tilings=1),
    dict(tiles=4, tilings=2),
    dict(tiles=4, tilings=4),
    dict(tiles=4, tilings=6),
    dict(tiles=4, tilings=8),
]

all_returns = []
for tiling in tqdm(tiling_settings):
    discretise, n_discrete_states = build_tile_coding(model, vec_env, **tiling)
    calib_sets = fill_calib_sets(model, buffer, discretise, n_discrete_states)
    qhats, visits = compute_corrections(calib_sets, alpha=0.1, min_calib=50)

    lengths = np.linspace(0.1, 2.0, 21)

    eval_returns = []
    for length in lengths:
        eval_vec_env = instantiate_eval_env(env_name="CartPole-v1", length=length)
        returns = run_eval(
            model,
            discretise,
            num_eps=NUM_EVAL_EPS,
            conformalise=True,
            ep_env=eval_vec_env,
            qhats=qhats,
            agg="mean",
        )
        eval_returns.append(returns)

    # plt.hist(returns, bins=50, range=(0, 500))
    # plt.xlim(0, 500)
    # plt.ylim(0, NUM_EVAL_EPS)
    # plt.title(f"Tiles = {tiling['tiles']}, Tilings = {tiling['tilings']}")
    # plt.show()

    all_returns.append(eval_returns)
# %%
all_returns = np.array(all_returns)
means = np.mean(all_returns, axis=2)
for ix, tiling in enumerate(tiling_settings):
    print(
        f"Tiles = {tiling['tiles']}, Tilings = {tiling['tilings']}: {means[ix].mean():.1f}"
    )
# %%
# agg = max
# Tiles = 6, Tilings = 1: 147.9
# Tiles = 6, Tilings = 2: 126.6
# Tiles = 6, Tilings = 3: 110.6
# Tiles = 6, Tilings = 4: 103.4
# Tiles = 6, Tilings = 6: 103.5
# Tiles = 4, Tilings = 1: 158.9
# Tiles = 4, Tilings = 2: 132.0
# Tiles = 4, Tilings = 4: 163.3
# Tiles = 4, Tilings = 6: 141.8
# Tiles = 2, Tilings = 1: 52.1
# Tiles = 2, Tilings = 2: 140.5
# Tiles = 2, Tilings = 4: 143.5
# Tiles = 2, Tilings = 8: 158.1
# Tiles = 2, Tilings = 16: 154.3

# agg = mean
# Tiles = 6, Tilings = 1: 158.4
# Tiles = 6, Tilings = 2: 144.8
# Tiles = 4, Tilings = 1: 151.4
# Tiles = 4, Tilings = 4: 180.0
# Tiles = 4, Tilings = 6: 183.9
# Tiles = 2, Tilings = 2: 60.9
# Tiles = 2, Tilings = 4: 110.0
# Tiles = 2, Tilings = 8: 108.7
# Tiles = 2, Tilings = 16: 103.2
idxes = range(8)
# idxes = [0, 1, 2]
# idxes = [3, 4, 5, 6, 7]
idxes = [0, 5]
for idx in idxes:
    label = (tiling_settings[idx]["tiles"], tiling_settings[idx]["tilings"])
    plt.plot(lengths, means[idx], marker="o", label=f"{label}")

plt.legend()
# %%
means.shape
# %%
