from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import torch
from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN

from crl.cons._type import ScoringMethod
from crl.cons.buffer import ReplayBuffer, Transition
from crl.cons.env import instantiate_eval_env


def _compute_transition_score(
    model: DQN, transition: Transition, score_fn: Callable
) -> float:
    with torch.no_grad():
        q_pred = model.q_net(model.policy.obs_to_tensor(transition.state)[0])
        y_pred = q_pred[0, transition.action]
        # Value of the terminal state is 0 by definition.
        if transition.done:
            y_true = transition.reward
        else:
            q_true = model.q_net(model.policy.obs_to_tensor(transition.next_state)[0])
            y_true = (
                transition.reward[0] + model.gamma * q_true[0, transition.next_action]
            )

    return score_fn(y_pred, y_true)


def _compute_correction(scores: np.ndarray, alpha: float):
    # perform conformalisation
    # TODO: potentially add a distance weighting - spatial conformal prediction
    n_calib = scores.shape[0]
    q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
    qhat = np.quantile(scores, q_level, method="higher")
    return qhat


def compute_nn_scores(
    model: DQN,
    buffer: ReplayBuffer,
    score_fn: Callable,
    scoring_method: str,
):
    vec_env = model.get_env()
    n_samples = buffer.capacity
    scores = np.zeros(n_samples)
    n_features = vec_env.observation_space.shape[0] + 1
    features = np.zeros((n_samples, n_features))

    if scoring_method != "td":
        raise NotImplementedError(
            "Not yet implemented Monte Carlo return scoring for nearest neighbour"
        )
    for ix, transition in enumerate(buffer):
        # compute score
        score = _compute_transition_score(model, transition, score_fn)
        state, action = transition.state, transition.action
        sa_feature = np.concatenate([state.flatten(), action.flatten()])

        scores[ix] = np.asarray(score).squeeze()
        features[ix] = sa_feature

    # Fit scaler on features and transform them
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return scores, features, scaler


def _compute_mc_returns(buffer: ReplayBuffer, gamma: float) -> np.ndarray:
    """Compute discounted Monte Carlo returns G_t for every transition in a
    chronological buffer. Episodes are delimited by `transition.done` and are
    treated as terminal (no bootstrap beyond terminal).

    Assumes the buffer is full and ordered chronologically.
    """
    transitions = list(buffer)
    n = len(transitions)
    returns = np.zeros(n, dtype=float)

    G = 0.0
    # Backward pass to accumulate discounted returns, resetting at terminals
    for i in range(n - 1, -1, -1):
        tr = transitions[i]
        r = float(np.asarray(tr.reward).squeeze())
        if tr.done:
            G = r
        else:
            G = r + gamma * G
        returns[i] = G

    return returns


def compute_nn_scores_mc(
    model: DQN,
    buffer: ReplayBuffer,
    score_fn: Callable,
    scoring_method: str,
):
    """Compute nearest-neighbour features and scores using Monte Carlo returns.

    The score is defined via the provided `score_fn` as
        score_fn(Q_theta(s, a), G_t),
    where G_t is the discounted Monte Carlo return from this time step until
    episode termination, with discount factor taken from `model.gamma`.

    Returns
    -------
    scores : np.ndarray of shape (n_samples,)
        Nonconformity scores per (s, a) time step.
    features : np.ndarray of shape (n_samples, obs_dim + 1)
        Concatenated [state || action] features.
    scaler : StandardScaler
        Fitted scaler for the features.
    """
    if scoring_method != "mc":
        raise NotImplementedError("compute_nn_scores_mc expects scoring_method='mc'")

    vec_env = model.get_env()
    n_samples = buffer.capacity
    scores = np.zeros(n_samples)
    n_features = vec_env.observation_space.shape[0] + 1
    features = np.zeros((n_samples, n_features))

    # Precompute discounted MC returns aligned with buffer indices
    returns = _compute_mc_returns(buffer, gamma=float(model.gamma))

    # We need random access to returns by index; align enumeration with buffer order
    for ix, transition in enumerate(buffer):
        with torch.no_grad():
            q_pred = model.q_net(model.policy.obs_to_tensor(transition.state)[0])
            y_pred = q_pred[0, transition.action]

        y_true = returns[ix]
        score = score_fn(y_pred, y_true)

        state, action = transition.state, transition.action
        sa_feature = np.concatenate(
            [np.asarray(state).flatten(), np.asarray(action).flatten()]
        )

        scores[ix] = np.asarray(score).squeeze()
        features[ix] = sa_feature

    # Fit scaler on features and transform them
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return scores, features, scaler


def run_ccnn_eval(
    model: DQN,
    num_eps: int,
    ep_env: gym.Env,
    scores: np.ndarray,
    tree: KDTree | BallTree,
    scaler: StandardScaler,
    k: int,
    max_dist: float,
    alpha: float,
) -> list[float]:
    episodic_returns = []

    # Determine number of discrete actions
    num_actions = getattr(ep_env.action_space, "n")

    fallback_value = scores.max()

    for ep in range(num_eps):
        obs = ep_env.reset()
        for t in range(1000):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

            # Adjust the qvalues of each action using
            # the correction from CP
            for a in range(num_actions):
                sa_feature = np.concatenate([obs.flatten(), np.array([a])]).reshape(
                    1, -1
                )
                # Scale the feature before NN search
                sa_feature = scaler.transform(sa_feature)
                dists, ids = tree.query(sa_feature, k=k)
                if (sa_maxdist := np.max(dists)) > max_dist:  # too far, anomalous
                    correction = fallback_value  # * np.maximum(1.0, sa_maxdist)
                else:
                    neighbour_scores = scores[ids]
                    correction = _compute_correction(neighbour_scores, alpha=alpha)
                q_vals[a] -= correction

            # take the action
            action = q_vals.argmax().numpy().reshape(1)
            obs, reward, done, info = ep_env.step(action)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break

    return episodic_returns


def run_ccnn_experiment(
    model: DQN,
    env_name: str,
    alpha: float,
    k: int,
    ccnn_scores,
    tree,
    scaler,
    max_dist,
    param: str,
    param_val: float,
    num_eps: int,
) -> dict[str, Any]:
    ccnn_results = run_single_seed_ccnn_experiment(
        model=model,
        env_name=env_name,
        k=k,
        alpha=alpha,
        scores=ccnn_scores,
        tree=tree,
        scaler=scaler,
        max_dist=max_dist,
        param_name=param,
        param_val=param_val,
        num_eps=num_eps,
    )
    return {"returns_ccnn": ccnn_results}


def calibrate_ccnn(
    model: DQN,
    buffer: ReplayBuffer,
    k: int,
    score_fn: Callable,
    scoring_method: ScoringMethod,
    max_distance_quantile: float = 0.99,
    score_clip_level: float = 0.01,
):
    scores, features, scaler = compute_nn_scores(
        model,
        buffer,
        score_fn=score_fn,
        scoring_method=scoring_method,
    )

    # clip the scores to remove extreme outliers
    scores = _clip_scores(score_clip_level, scores)

    # For each point, get its k nearest neighbours (excluding itself) and take the max distance.
    # Then set the threshold as the MAX_DIST_QUANTILE quantile of these max distances.
    tree = KDTree(features)
    max_dist = _compute_max_dist(k, max_distance_quantile, features, tree)
    return scores, scaler, tree, max_dist


def run_single_seed_ccnn_experiment(
    model: DQN,
    env_name: str,
    alpha: float,
    k: int,
    scores: np.ndarray,
    tree: KDTree,
    scaler: StandardScaler,
    max_dist: float,
    param_name: str,
    param_val: float,
    num_eps: int,
) -> list[float]:
    eval_vec_env = instantiate_eval_env(env_name=env_name, **{param_name: param_val})

    ccnn_returns = run_ccnn_eval(
        model=model,
        num_eps=num_eps,
        ep_env=eval_vec_env,
        scores=scores,
        tree=tree,
        scaler=scaler,
        k=k,
        max_dist=max_dist,
        alpha=alpha,
    )
    return ccnn_returns


def _compute_max_dist(k, max_quantile, features, tree):
    dists, _ = tree.query(features, k=k + 1)
    per_point_max_dists = dists.max(axis=1)
    max_dist = np.quantile(per_point_max_dists, max_quantile)
    return max_dist


def _clip_scores(score_clip_level, scores):
    lo_quant = np.quantile(scores, q=score_clip_level)
    hi_quant = np.quantile(scores, q=1 - score_clip_level)
    scores = scores.clip(lo_quant, hi_quant)
    return scores
