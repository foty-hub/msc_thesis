# %%
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import ale_py
import gymnasium as gym
import line_profiler
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from tqdm import tqdm

from crl.calib import (
    collect_transitions,
    signed_score,
)
from crl.types import AgentTypes, CalibMethods, ScoringMethod

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
# STATE_REPR = "image"  # "image" uses last frame; "activation" uses intermediate model features
STATE_REPR = "activation"
# Optional module path for activation extraction; when None, uses features_extractor directly.
# Examples: None, "q_net.features_extractor", "q_net.q_net"
ACTIVATION_MODULE: str | None = None

# %%


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

# %% [markdown]
# # trying cc-nn

# %%
buffer = collect_transitions(model, vec_env, n_transitions=cfg.n_calib_steps)

# %%
model.env = vec_env

# %%
import faiss
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from crl.approx import FaissKDTree
from crl.calib import signed_score
from crl.ccnn import _clip_scores, _compute_correction, _compute_max_dist

faiss.omp_set_num_threads(1)


def _latest_frame(obs: np.ndarray) -> np.ndarray:
    obs_arr = np.asarray(obs)
    return obs_arr[..., -1]


def _resolve_module_path(root: object, module_path: str) -> torch.nn.Module:
    obj = root
    for part in module_path.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"Module path '{module_path}' not found at '{part}'")
        obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"Resolved object at '{module_path}' is not a torch module")
    return obj


def _activation_from_obs(
    model: DQN, obs: np.ndarray, module_path: str | None = None
) -> np.ndarray:
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    if module_path is None:
        with torch.no_grad():
            if hasattr(model.q_net, "extract_features") and hasattr(
                model.q_net, "features_extractor"
            ):
                feats = model.q_net.extract_features(
                    obs_tensor, model.q_net.features_extractor
                )
            elif hasattr(model.policy, "extract_features") and hasattr(
                model.policy, "features_extractor"
            ):
                feats = model.policy.extract_features(
                    obs_tensor, model.policy.features_extractor
                )
            elif hasattr(model.q_net, "features_extractor"):
                feats = model.q_net.features_extractor(obs_tensor)
            elif hasattr(model.policy, "features_extractor"):
                feats = model.policy.features_extractor(obs_tensor)
            else:
                raise ValueError("No features extractor found on the DQN model")
        return feats.detach().cpu().numpy()

    module = _resolve_module_path(model, module_path)
    activations: list[torch.Tensor] = []

    def _capture(_module, _inputs, outputs):
        activations.append(outputs)

    handle = module.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            _ = model.q_net(obs_tensor)
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError(f"No activations captured for module '{module_path}'")
    act = activations[-1]
    if isinstance(act, (tuple, list)):
        act = act[0]
    return act.detach().cpu().numpy()


def _state_feature(obs: np.ndarray, model: DQN) -> np.ndarray:
    if STATE_REPR == "image":
        frame = _latest_frame(obs)
        flat = frame.reshape(frame.shape[0], -1).astype(np.float32)
        return flat
    if STATE_REPR == "activation":
        feats = _activation_from_obs(model, obs, module_path=ACTIVATION_MODULE)
        feats = np.asarray(feats)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1)
        return feats.astype(np.float32)
    raise ValueError(f"Unknown STATE_REPR: {STATE_REPR}")


def _sa_feature(obs: np.ndarray, action: np.ndarray | int, model: DQN) -> np.ndarray:
    flat = _state_feature(obs, model=model)
    action_arr = np.asarray(action).reshape(-1, 1).astype(np.float32)
    if flat.shape[0] != action_arr.shape[0]:
        if flat.shape[0] == 1:
            flat = np.repeat(flat, action_arr.shape[0], axis=0)
        elif action_arr.shape[0] == 1:
            action_arr = np.repeat(action_arr, flat.shape[0], axis=0)
        else:
            raise ValueError(
                f"Batch mismatch: obs {flat.shape[0]} vs action {action_arr.shape[0]}"
            )
    return np.concatenate([flat, action_arr], axis=1)


def compute_nn_scores_td(
    model: DQN,
    buffer,
    score_fn,
):
    n_samples = len(buffer)
    first = buffer[0]
    n_features = _sa_feature(first.state, first.action, model=model).shape[1]
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
        features[ix] = _sa_feature(
            transition.state, transition.action, model=model
        ).squeeze()

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return scores, features, scaler


score_fn = signed_score
score_clip_level = 0.01

scores, features, scaler = compute_nn_scores_td(
    model,
    buffer,
    score_fn=score_fn,
)

# clip the scores to remove extreme outliers
scores = _clip_scores(score_clip_level, scores)
tree = FaissKDTree(features)
# %%
# tree = KDTree(features)
max_dist = _compute_max_dist(cfg.k, cfg.ccnn_max_distance_quantile, features, tree)

# %%
num_eps = 25
ep_env = vec_env
alpha = cfg.alpha_nn


# %%
# Determine number of discrete actions
num_actions = getattr(ep_env.action_space, "n")

fallback_value = scores.max()


@line_profiler.profile
def run_eval_loop(num_eps: int = 5, modes: Iterable[int] = range(0, 45, 4)):
    episodic_returns = defaultdict(list)

    for mode in modes:
        ep_env = make_atari_env(
            ENV_ID, n_envs=1, seed=SEED + 1, env_kwargs={"difficulty": 0, "mode": mode}
        )
        ep_env = VecFrameStack(ep_env, n_stack=4)
        for ep in range(num_eps):
            obs = ep_env.reset()
            ep_return = 0.0
            print(f"Episode No. {ep + 1} (Calibrated) [{mode}]")
            while True:
                with torch.inference_mode():
                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    q_vals = model.q_net(obs_tensor).flatten()

                # Adjust the qvalues of each action using a single NN lookup.
                actions = np.arange(num_actions, dtype=np.int64)
                sa_features = _sa_feature(obs, actions, model=model)
                sa_features = scaler.transform(sa_features)
                dists, ids = tree.query(sa_features, k=cfg.k)

                too_far = np.max(dists, axis=1) > max_dist
                corrections = np.empty(num_actions, dtype=np.float32)
                corrections[too_far] = fallback_value
                for idx in np.flatnonzero(~too_far):
                    neighbour_scores = scores[ids[idx]]
                    corrections[idx] = _compute_correction(
                        neighbour_scores, alpha=alpha
                    )
                q_vals = q_vals - torch.as_tensor(corrections, device=q_vals.device)

                # take the action
                action = np.array([int(q_vals.argmax().item())])
                obs, reward, done, info = ep_env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                if done and (info[0].get("lives", 0) == 0):
                    episodic_returns[mode].append(ep_return)
                    break
    return episodic_returns


if __name__ == "__main__":
    run_eval_loop(num_eps=25, modes=[0])

# for mode in range(0, 45, 4):
#     ep_env = make_atari_env(
#         ENV_ID, n_envs=1, seed=SEED + 1, env_kwargs={"difficulty": 0, "mode": mode}
#     )
#     ep_env = VecFrameStack(ep_env, n_stack=4)
#     # normal run
#     for ep in range(num_eps):
#         obs = ep_env.reset()
#         ep_return = 0.0
#         print(f"Episode No. {ep + 1} (Normal) [{mode}]")
#         with tqdm(leave=False) as pbar:
#             while True:
#                 with torch.inference_mode():
#                     obs_tensor = model.policy.obs_to_tensor(obs)[0]
#                     q_vals = model.q_net(obs_tensor).flatten()
#                 # take the action
#                 action = np.array([int(q_vals.argmax().item())])
#                 obs, reward, done, info = ep_env.step(action)
#                 ep_return += float(np.asarray(reward).reshape(-1)[0])

#                 pbar.update(1)

#                 if done and (info[0].get("lives", 0) == 0):
#                     normal_episodic_returns[mode].append(ep_return)
#                     break

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

# # %%
# import pickle

# results = {"baseline": normal_episodic_returns, "ccnn": episodic_returns}
# with open("ccnn_no_tuning_comparison_vectorised.pkl", "wb") as f:
#     pickle.dump(results, f)
# %%
