# %%
import os
import pickle
import pprint
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN
from tqdm import tqdm, trange

from crl.agents import learn_cqldqn_policy, learn_ddqn_policy, learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    correction_for,
    fill_calib_sets_mc,
    fill_calib_sets_td,
    signed_score,
)
from crl.ccnn import run_ccnn_experiment
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env
from crl.types import AgentTypes, CalibMethods, ClassicControl, ScoringMethod
from crl.utils.graphing import despine

# %%
ALPHA_DISC = 0.01  # Conformal prediction miscoverage level (discrete algo)
ALPHA_NN = 0.1  # Conformal prediction miscoverage level (nearest neighbour algo)
CQL_ALPHA = 0.05
MIN_CALIB = 50  # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES = 250
N_CALIB_STEPS = 10_000
N_TRAIN_STEPS = 50_000
K = 50
SCORING_METHOD: ScoringMethod = "td"
AGENT_TYPE: AgentTypes = "vanilla"
SCORE_FN = signed_score
RETRAIN = False
CALIB_METHODS: list[CalibMethods] = ["nocalib", "ccdisc", "ccnn"]
CCNN_MAX_DISTANCE_QUANTILE = 0.9

# %%
import os

import ale_py
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack

ENV_ID = "BreakoutNoFrameskip-v4"  # classic ALE id suitable for DQN/Atari wrappers
SEED = 0

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
# 1) Make and wrap the env (DeepMind-style preprocessing + VecFrameStack(4))
train_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED, monitor_dir=log_dir)
train_env = VecFrameStack(train_env, n_stack=4)

# %%
model = DQN.load(
    "../../models/BreakoutNoFrameskip-v4/BreakoutNoFrameskip-v4_2/best_model.zip"
)

# %%
eval_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED + 1)
eval_env = VecFrameStack(eval_env, n_stack=4)

# %%
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RobustnessConfig:
    """Configuration for robustness experiments."""

    alpha_disc: float = ALPHA_DISC
    alpha_nn: float = ALPHA_NN
    cql_alpha: float = CQL_ALPHA
    min_calib: int = MIN_CALIB
    num_experiments: int = NUM_EXPERIMENTS
    num_eval_episodes: int = NUM_EVAL_EPISODES
    n_calib_steps: int = N_CALIB_STEPS
    n_train_steps: int = N_TRAIN_STEPS
    k: int = K
    scoring_method: ScoringMethod = SCORING_METHOD
    agent_type: AgentTypes = AGENT_TYPE
    score_fn: Callable = SCORE_FN
    retrain: bool = RETRAIN
    calib_methods: list[CalibMethods] = field(
        default_factory=lambda: CALIB_METHODS.copy()
    )
    ccnn_max_distance_quantile: float = CCNN_MAX_DISTANCE_QUANTILE
    debug_serial: bool = False
    debug_seed: int | None = 0


cfg = RobustnessConfig()

# %%
vec_env = make_atari_env(
    ENV_ID, n_envs=1, seed=SEED + 1, env_kwargs={"difficulty": 0, "mode": 0}
)
vec_env = VecFrameStack(vec_env, n_stack=4)

obs_space = vec_env.observation_space
n_dims = int(obs_space.shape[0])
# Sanity checks
assert getattr(vec_env, "num_envs", 1) == 1

# %%
n_dims

# %% [markdown]
# # trying cc-nn

# %%
buffer = collect_transitions(model, vec_env, n_transitions=cfg.n_calib_steps)

# %%
model.env = vec_env

# %%
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from crl.calib import signed_score
from crl.ccnn import _clip_scores, _compute_correction, _compute_max_dist


def _latest_frame(obs: np.ndarray) -> np.ndarray:
    obs_arr = np.asarray(obs)
    return obs_arr[..., -1]


def _sa_feature(obs: np.ndarray, action: np.ndarray | int) -> np.ndarray:
    frame = _latest_frame(obs)
    flat = frame.reshape(frame.shape[0], -1).astype(np.float32)
    action_arr = np.asarray(action).reshape(-1, 1).astype(np.float32)
    return np.concatenate([flat, action_arr], axis=1)


def compute_nn_scores_td_last_frame(
    model: DQN,
    buffer,
    score_fn,
):
    n_samples = len(buffer)
    first = buffer[0]
    n_features = _sa_feature(first.state, first.action).shape[1]
    scores = np.zeros(n_samples)
    features = np.zeros((n_samples, n_features), dtype=np.float32)

    for ix, transition in enumerate(buffer):
        with torch.no_grad():
            q_pred = model.q_net(model.policy.obs_to_tensor(transition.state)[0])
            act_scalar = int(np.asarray(transition.action).reshape(-1)[0])
            y_pred = q_pred[0, act_scalar]

            done_bool = bool(np.asarray(transition.done).reshape(-1)[0])
            reward_val = float(np.asarray(transition.reward).reshape(-1)[0])
            if done_bool:
                y_true = reward_val
            else:
                q_true = model.q_net(
                    model.policy.obs_to_tensor(transition.next_state)[0]
                )
                next_act = int(np.asarray(transition.next_action).reshape(-1)[0])
                y_true = reward_val + float(model.gamma) * q_true[0, next_act]

        score = score_fn(y_pred, y_true)
        scores[ix] = np.asarray(score).squeeze()
        features[ix] = _sa_feature(transition.state, transition.action).squeeze()

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return scores, features, scaler


score_fn = signed_score
score_clip_level = 0.01

scores, features, scaler = compute_nn_scores_td_last_frame(
    model,
    buffer,
    score_fn=score_fn,
)

# clip the scores to remove extreme outliers
scores = _clip_scores(score_clip_level, scores)

tree = KDTree(features)
max_dist = _compute_max_dist(cfg.k, cfg.ccnn_max_distance_quantile, features, tree)

# %%
num_eps = 10
ep_env = vec_env
alpha = cfg.alpha_nn


# %%
from collections import defaultdict

episodic_returns = defaultdict(list)
normal_episodic_returns = defaultdict(list)

# Determine number of discrete actions
num_actions = getattr(ep_env.action_space, "n")

fallback_value = scores.max()
for mode in range(0, 45, 4):
    ep_env = make_atari_env(
        ENV_ID, n_envs=1, seed=SEED + 1, env_kwargs={"difficulty": 0, "mode": mode}
    )
    ep_env = VecFrameStack(ep_env, n_stack=4)
    for ep in range(num_eps):
        obs = ep_env.reset()
        ep_return = 0.0
        print(f"Episode No. {ep + 1} (Calibrated) [{mode}]")
        with tqdm(leave=False) as pbar:
            while True:
                q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

                # Adjust the qvalues of each action using
                # the correction from CP
                for a in range(num_actions):
                    sa_feature = _sa_feature(obs, np.array([a]))
                    # Scale the feature before NN search
                    sa_feature = scaler.transform(sa_feature)
                    dists, ids = tree.query(sa_feature, k=cfg.k)
                    if np.max(dists) > max_dist:  # too far, anomalous
                        correction = fallback_value  # * np.maximum(1.0, sa_maxdist)
                    else:
                        neighbour_scores = scores[ids]
                        correction = _compute_correction(neighbour_scores, alpha=alpha)
                    q_vals[a] -= correction

                # take the action
                action = q_vals.argmax().numpy().reshape(1)
                obs, reward, done, info = ep_env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                pbar.update(1)
                if done and (info[0].get("lives", 0) == 0):
                    episodic_returns[mode].append(ep_return)
                    break

for mode in range(0, 45, 4):
    ep_env = make_atari_env(
        ENV_ID, n_envs=1, seed=SEED + 1, env_kwargs={"difficulty": 0, "mode": mode}
    )
    ep_env = VecFrameStack(ep_env, n_stack=4)
    # normal run
    for ep in range(num_eps):
        obs = ep_env.reset()
        ep_return = 0.0
        print(f"Episode No. {ep + 1} (Normal) [{mode}]")
        with tqdm(leave=False) as pbar:
            while True:
                q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()
                # take the action
                action = q_vals.argmax().numpy().reshape(1)
                obs, reward, done, info = ep_env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])

                pbar.update(1)

                if done and (info[0].get("lives", 0) == 0):
                    normal_episodic_returns[mode].append(ep_return)
                    break

# %%
# import os

# os.environ["RUST_BACKTRACE"] = "full"

# # %%
# discretise, n_features = build_tile_coding(model, vec_env, tiles=4, tilings=1)
# buffer = collect_transitions(model, vec_env, n_transitions=cfg.n_calib_steps)
# calib_sets = fill_calib_sets_td(
#     model,
#     buffer,
#     discretise,
# )
# qhats = compute_corrections(
#     calib_sets,
#     alpha=cfg.alpha_disc,
#     min_calib=cfg.min_calib,
# )

# %%
import pickle

results = {"baseline": normal_episodic_returns, "ccnn": episodic_returns}
with open("ccnn_no_tuning_comparison.pkl", "wb") as f:
    pickle.dump(results, f)
# %%
