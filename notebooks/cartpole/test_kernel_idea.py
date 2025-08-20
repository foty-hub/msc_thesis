# %%
import os
import pickle
import numpy as np
import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
import torch

from crl.utils.graphing import despine
from tqdm import tqdm
from typing import Callable
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from crl.cons.calib import (
    collect_transitions,
    signed_score,
)
from crl.cons.cartpole import instantiate_eval_env, learn_dqn_policy
from crl.cons.cql import learn_cqldqn_policy
from crl.cons.buffer import ReplayBuffer, Transition

# fmt: off
ALPHA = 0.1                 # Conformal prediction miscoverage level
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES=250
N_CALIB_TRANSITIONS=50_000
N_TRAIN_EPISODES = 50_000

# data = np.column_stack([s['vals'] for s in stats])
# %%
# compute score for each point
# Output an array: scores
# at inference time - distance to k-nearest scores (with some radius limit)
# do quantile calc on that dataset

# # %%
# go through the buffer - collect the scores, and also 
# construct the features which will be used for the KDTree
def _compute_transition_score(model: DQN, transition: Transition, score: Callable = signed_score) -> float:
    with torch.no_grad():
        q_pred = model.q_net(model.policy.obs_to_tensor(transition.state)[0])
        y_pred = q_pred[0, transition.action]
        # Value of the terminal state is 0 by definition.
        if transition.done:
            y_true = transition.reward
        else:
            q_true = model.q_net(model.policy.obs_to_tensor(transition.next_state)[0])
            y_true = transition.reward[0] + model.gamma * q_true[0, transition.next_action]
    
    return score(y_pred, y_true)

def _compute_correction(scores: np.ndarray, alpha: float):
    # perform conformalisation
    # TODO: potentially add a distance weighting
    n_calib = scores.shape[0]
    q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
    qhat = np.quantile(scores, q_level, method="higher")
    return qhat

def compute_scores(model: DQN, vec_env: VecEnv, buffer: ReplayBuffer):
    n_samples = buffer.capacity
    scores = np.zeros(n_samples)
    n_features = vec_env.observation_space.shape[0] + 1
    features = np.zeros((n_samples, n_features))

    for ix, transition in enumerate(buffer):
        # compute score
        score = _compute_transition_score(model, transition)
        state, action = transition.state, transition.action
        sa_feature = np.concatenate([state.flatten(), action.flatten()])

        scores[ix] = np.asarray(score).squeeze()
        features[ix] = sa_feature

    # Fit scaler on features and transform them
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return scores, features, scaler

def plot_obs_scatter(
    scores: np.ndarray,
    features: np.ndarray,
    x: int = 0,
    y: int = 1,
    alpha: float = 0.25,
    per_point_max_dists: np.ndarray | None = None,
    max_dist: float | None = None,
) -> None:
    cmap = 'viridis'
    norm = mpl.colors.Normalize(vmin=float(np.min(scores)), vmax=float(np.max(scores)))

    xs = features[:, x]
    ys = features[:, y]

    if per_point_max_dists is not None and max_dist is not None:
        mask = np.asarray(per_point_max_dists) > max_dist

        if np.any(~mask):
            plt.scatter(
                xs[~mask], ys[~mask],
                c=scores[~mask], cmap=cmap, norm=norm,
                alpha=alpha, s=2, marker='o'
            )
        if np.any(mask):
            plt.scatter(
                xs[mask], ys[mask],
                c='red', cmap=cmap, norm=norm,
                alpha=1.0, s=2, marker='o'
            )
    else:
        plt.scatter(xs, ys, c=scores, cmap=cmap, norm=norm, alpha=alpha, s=2)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable, ax=plt.gca(), label='score')
    plt.xlabel(f'dim: {x}')
    plt.ylabel(f'dim: {y}')
    despine(plt.gca())
    plt.show()


# %%
# plot_obs_scatter(scores, features, 0, 2)
# plot_obs_scatter(scores, features, 0, 2, per_point_max_dists=per_point_max_dists, max_dist=max_dist)

# %%
# test on evals now
def run_eval(
    model: DQN,
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    scores: np.ndarray,
    tree: KDTree | BallTree,
    scaler: StandardScaler,
    k: int,
    max_dist: float,
    alpha: float
) -> list[float]:
    episodic_returns = []

    # Determine number of discrete actions
    num_actions = getattr(ep_env.action_space, "n")

    fallback_value = scores.max()

    for ep in range(num_eps):
        obs = ep_env.reset()
        for t in range(1000):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

            if conformalise:
                # Adjust the qvalues of each action using
                # the correction from CP
                for a in range(num_actions):
                    sa_feature = np.concatenate([obs.flatten(), np.array([a])]).reshape(1, -1)
                    # Scale the feature before NN search
                    sa_feature = scaler.transform(sa_feature)
                    dists, ids = tree.query(sa_feature, k=k)
                    if np.max(dists) > max_dist:  # too far, anomalous
                        correction = fallback_value
                    else:
                        neighbour_scores = scores[ids]
                        correction = _compute_correction(neighbour_scores, alpha=alpha)
                    q_vals[a] -= correction

            action = q_vals.argmax().numpy().reshape(1)

            # The state-visitation count should be for the state *before* we step
            # the environment.
            obs, reward, done, info = ep_env.step(action)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break

    return episodic_returns

# %%
# for length in np.arange(0.1, 3.1, 0.2):
    # "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), 6, 1),
def run_single_seed_experiment(alpha, env_name, model, k, scores, tree, scaler, max_dist, param_name, param_vals):
    allconf, allnoconf = [], []
    args = {
        'model': model,
        'num_eps': 100,
        'scores': scores,
        'tree': tree,
        'scaler': scaler,
        'k': k,
        'max_dist': max_dist,
        'alpha': alpha
    }
    for param_val in (pbar := tqdm(param_vals, leave=False)):
        pbar.set_description(f'{param_name}: {param_val:.2f}')
        eval_vec_env = instantiate_eval_env(env_name=env_name, **{param_name: param_val})

        noconf_returns = run_eval(
            conformalise=False,
            ep_env=eval_vec_env,
            **args,
        )
        conf_returns = run_eval(
            conformalise=True,
            ep_env=eval_vec_env,
            **args,
        )
        allconf.append(conf_returns)
        allnoconf.append(noconf_returns)
    return allconf, allnoconf

def _compute_max_dist(k, max_quantile, features, tree):
    dists, _ = tree.query(features, k=k + 1)
    per_point_max_dists = dists.max(axis=1)
    max_dist = np.quantile(per_point_max_dists, max_quantile)
    return max_dist

def _clip_scores(score_clip_level, scores):
    lo_quant = np.quantile(scores, q= score_clip_level)
    hi_quant = np.quantile(scores, q= 1 - score_clip_level)
    scores = scores.clip(lo_quant, hi_quant)
    return scores

# %%
# param_name, param_vals = 'LINK_LENGTH_1', np.linspace(0.5, 2.0, 16), -500
k = 100
score_clip_level = 0.01
MAX_DIST_QUANTILE = 0.99
N_CALIB_TRANSITIONS = 50_000
alpha=0.1
env_name = 'LunarLander-v3'
param_name, param_vals, min_return = "gravity", np.arange(-12, -0, 0.5), 0
# param_name, param_vals, min_return = 'length', np.arange(0.1, 3.1, 0.2), 0
# good_seeds = [0, 1, 2, 3, 4, 7, 8, 9] # acrobot
# good_seeds = [1, 3, 5, 6, 7, 9] # cartpole
# good_seeds = [0, 6, 7] # LunarLander
good_seeds = [7]

# good_seeds = list(range(10))

seeds = []
all_seed_results = []
for seed in good_seeds:
    model, vec_env = learn_dqn_policy(
        env_name=env_name,
        seed=seed,
        total_timesteps=N_TRAIN_EPISODES,
    )

    print(' Constructing tree')
    # Collect transitions and construct tree for nearest neighbours queries
    buffer = collect_transitions(model, vec_env, n_transitions=N_CALIB_TRANSITIONS)
    scores, features, scaler = compute_scores(model, vec_env, buffer)
    tree = KDTree(features)

    # clip the scores to remove extreme outliers
    scores = _clip_scores(score_clip_level, scores)

    # For each point, get its k nearest neighbours (excluding itself) and take the max distance.
    # Then set the threshold as the MAX_DIST_QUANTILE quantile of these max distances.
    max_dist = _compute_max_dist(k, MAX_DIST_QUANTILE, features, tree)

    allconf, allnoconf = run_single_seed_experiment(alpha=alpha, env_name=env_name, model=model, k=k, scores=scores, tree=tree, scaler=scaler, max_dist=max_dist, param_name=param_name, param_vals=param_vals)
    noconf_returns = np.array(allnoconf) - min_return
    conf_returns = np.array(allconf) - min_return
    all_seed_results.append({'seed': seed,
                        'noconf_returns': noconf_returns,
                        'conf_returns': conf_returns})

    print(f' NOCONF: {noconf_returns.mean():.1f}, CONF: {conf_returns.mean():.1f}')
# # %%
# plt.plot(param_vals, conf_returns.mean(axis=1) / noconf_returns.mean(axis=1), marker='o')
# plt.grid(linestyle='--', alpha=0.3)
# %%
# TODO: standard scale features

conf = np.array([r['conf_returns']for r in all_seed_results])
noconf = np.array([r['noconf_returns']for r in all_seed_results])
# %%
conf_plots = plt.plot(param_vals, conf.mean(axis=2).T, marker='o')
plt.legend()
plt.show()
# %%
print('seed, no conf, w/ conf')
for i, seed in enumerate(good_seeds):
    print(f'   {seed}:   {noconf[i].mean():.1f}    {conf[i].mean():.1f}')
good_seeds = (noconf[:, 3, :].mean(axis=1) > 200).nonzero()[0]
good_seeds = [int(s) for s in good_seeds]
avg_perf = (conf[good_seeds].mean(2) / noconf[good_seeds].mean(2)).mean(1).mean()
print(f'{avg_perf:.3f}x improvement')
# %%
import pickle
with open(f'kernel_results_{env_name}.pkl', 'wb') as f:
    pickle.dump(all_seed_results, f)

# %%
for i, seed in enumerate(good_seeds):
    conf_seed = conf[i].mean(1)
    noconf_seed = noconf[i].mean(1)
    plt.plot(param_vals, conf_seed, label='w/ Conf', marker='o')
    plt.plot(param_vals, noconf_seed, label='No Conf', marker='o')
    plt.legend()
    plt.grid()
    plt.title(f'Seed: {seed}')
    plt.show()


# %%
conf.shape
# %%
