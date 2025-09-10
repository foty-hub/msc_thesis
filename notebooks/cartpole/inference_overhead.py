"""
Experiment: inference-time overhead of action selection

Compares per-step compute time for three variants using a learned DQN policy:
  1) Baseline (unmodified DQN)
  2) Conformal calibration via discretisation + lookup (tile coding)
  3) Conformal calibration via nearest neighbours (CCNN)

Notes
- Uses an already trained policy (see models/<env>/dqn/model_<seed>.zip).
- No CLI: tweak top-level constants below.
- Isolates the action-selection compute path; environment stepping time is not
  included in the timing window.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN

from crl.agents import learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    fill_calib_sets,
    fill_calib_sets_mc,
    signed_score,
)

# Relative import of CCNN utilities from this folder
from crl.ccnn import calibrate_ccnn
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env
from crl.types import ScoringMethod

# =========================
# Top-level configuration
# =========================
ENV_NAME = "CartPole-v1"  # default single env
RUN_ALL_ENVS = True  # if True, runs for all 4 envs below
SEED = 0

# Calibration settings
N_CALIB_STEPS = 10_000
ALPHA_DISC = 0.01
ALPHA_NN = 0.10
K = 50
SCORING_METHOD: ScoringMethod = "td"  # {"monte_carlo", "td"}
SCORE_FN: Callable = signed_score
CCNN_MAX_DISTANCE_QUANTILE = 0.90

# Discretisation (tile coding) settings per env: (tiles, tilings)
DISC_SETTINGS = {
    "CartPole-v1": (6, 1),
    "Acrobot-v1": (6, 1),
    "MountainCar-v0": (10, 1),
    "LunarLander-v3": (4, 1),
}

# Timing/eval settings
NUM_EVAL_EPISODES = 50
MAX_STEPS_PER_EP = 1000
WARMUP_STEPS_PER_METHOD = 50  # ignore first N steps per method when computing stats


# =========================
# Helpers
# =========================
@dataclass
class TimingResult:
    label: str
    times_ns: np.ndarray

    def summary(self) -> dict[str, float]:
        arr = self.times_ns.astype(np.float64)
        return {
            "count": int(arr.size),
            "mean_us": float(arr.mean() / 1e3),
            "std_us": float(arr.std() / 1e3),
            "median_us": float(np.median(arr) / 1e3),
            "p95_us": float(np.quantile(arr, 0.95) / 1e3),
            "p99_us": float(np.quantile(arr, 0.99) / 1e3),
            "min_us": float(arr.min() / 1e3),
            "max_us": float(arr.max() / 1e3),
        }


def _select_action_baseline(model: DQN, obs) -> np.ndarray:
    q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()
    action = q_vals.argmax().numpy().reshape(1)
    return action


def _select_action_disc(
    model: DQN,
    obs,
    qhats: np.ndarray,
    discretise: Callable,
    n_state_features: int | None = None,
) -> np.ndarray:
    # Inference without autograd bookkeeping
    with torch.no_grad():
        q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

    # Derive #actions directly from network output (cheaper than env lookup)
    num_actions = q_vals.numel()

    # Reuse a cached arange for actions to avoid per-step allocation
    try:
        actions = _select_action_disc._A_CACHE[num_actions]  # type: ignore[attr-defined]
    except AttributeError:
        _select_action_disc._A_CACHE = {}  # type: ignore[attr-defined]
        actions = None
    except KeyError:
        actions = None
    if actions is None:
        actions = np.arange(num_actions, dtype=np.int64)
        _select_action_disc._A_CACHE[num_actions] = actions  # type: ignore[attr-defined]

    # Compute state block size once if not provided
    if n_state_features is None:
        n_state_features = int(qhats.size // num_actions)

    # Vectorised correction over all actions
    corrections = correction_for(
        obs,
        actions,
        qhats,
        discretise,
        agg="max",
        n_state_features=n_state_features,
    )

    # Subtract corrections from Q-values and pick greedy action
    q_vals -= torch.as_tensor(corrections, dtype=q_vals.dtype, device=q_vals.device)
    action_idx = int(torch.argmax(q_vals))
    return np.array([action_idx], dtype=np.int64)


def _select_action_ccnn(
    model: DQN,
    obs,
    scores: np.ndarray,
    tree,
    scaler: StandardScaler,
    k: int,
    j_idx: int,
    max_dist: float,
    fallback_value: float,
) -> np.ndarray:
    # Number of discrete actions
    num_actions = getattr(model.get_env().action_space, "n")

    # Forward pass without autograd bookkeeping
    with torch.no_grad():
        q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

    # Build scaled (s, a) features for all actions with minimal allocations
    actions = np.arange(num_actions)
    flat_obs = np.asarray(obs, dtype=np.float64).ravel()

    means = scaler.mean_
    scales = scaler.scale_
    state_means, action_mean = means[:-1], means[-1]
    state_scales, action_scale = scales[:-1], scales[-1]

    # Scale state once
    state_scaled = (flat_obs - state_means) / state_scales

    # Preallocate feature matrix: shape (A, state_dim + 1)
    sa_features = np.empty(
        (num_actions, state_scaled.size + 1), dtype=state_scaled.dtype
    )
    sa_features[:, :-1] = state_scaled  # row-broadcast
    sa_features[:, -1] = (actions - action_mean) / action_scale

    # Batched KDTree query; distances are sorted ascending, so last col is the max
    dists, ids = tree.query(sa_features, k=k)
    far_mask = dists[:, -1] > float(max_dist)

    # Gather neighbour scores (A, k)
    neighbour_scores = np.take(scores, ids)

    # j_idx is precomputed order statistic index for conformal cutoff
    partitioned = np.partition(neighbour_scores, j_idx, axis=1)
    corrs = partitioned[:, j_idx]

    # Apply global fallback when neighbourhood is too far
    corrs = np.where(far_mask, fallback_value, corrs)

    # Subtract corrections from Q-values and pick greedy action
    q_vals -= torch.as_tensor(corrs, dtype=q_vals.dtype, device=q_vals.device)
    action_idx = int(torch.argmax(q_vals))
    return np.array([action_idx], dtype=np.int64)


def _time_action_selection(
    model: DQN,
    eval_env,
    selector: Callable[[DQN, np.ndarray], np.ndarray],
    *,
    warmup_steps: int = 0,
    num_eps: int = 1,
    max_steps: int = 1000,
) -> np.ndarray:
    times: list[int] = []
    n_steps_seen = 0
    for _ in range(num_eps):
        obs = eval_env.reset()
        for _ in range(max_steps):
            t0 = time.perf_counter_ns()
            action = selector(model, obs)
            t1 = time.perf_counter_ns()
            if n_steps_seen >= warmup_steps:
                times.append(t1 - t0)
            obs, reward, done, info = eval_env.step(action)
            n_steps_seen += 1
            if done:
                break
    return np.asarray(times, dtype=np.int64)


def _calibrate_disc(
    model: DQN,
    vec_env,
    *,
    tiles: int,
    tilings: int,
    scoring_method: ScoringMethod,
    score_fn: Callable,
    n_calib_steps: int,
) -> tuple[np.ndarray, Callable]:
    discretise, n_features = build_tile_coding(model, vec_env, tiles, tilings)
    buffer = collect_transitions(model, vec_env, n_transitions=n_calib_steps)
    if scoring_method == "td":
        calib_sets = fill_calib_sets(
            model, buffer, discretise, n_features, score=score_fn
        )
    elif scoring_method == "monte_carlo":
        calib_sets = fill_calib_sets_mc(
            model, buffer, discretise, n_features, score=score_fn
        )
    else:
        raise NotImplementedError(f"Unknown scoring_method: {scoring_method}")
    qhats, _visits = compute_corrections(calib_sets, alpha=ALPHA_DISC, min_calib=50)
    return qhats, discretise


# =========================
# Batched discretisation + correction
# =========================
def discretise_batched(
    obs: np.ndarray,
    actions: np.ndarray,
    base_discretise: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_state_features: int,
) -> np.ndarray:
    """Return active tile indices for all actions in a single array.

    Parameters
    - obs: observation array as passed to the base discretiser (possibly with env dim).
    - actions: array of action indices shape (A,).
    - base_discretise: function mapping (obs, action[shape (1,)]) -> indices for that action.
    - n_state_features: number of features for a single action block.

    Returns
    - idxs: shape (A, T) where T is number of active tiles per state.
    """
    # Get base indices once for action 0; action only contributes an offset
    base_idxs = base_discretise(obs, np.array([0]))
    base_idxs = np.asarray(base_idxs).reshape(1, -1)
    actions = np.asarray(actions, dtype=int).reshape(-1, 1)
    # Broadcast offsets per action block
    return base_idxs + actions * int(n_state_features)


def correction_for(
    state: np.ndarray,
    action: np.ndarray,
    qhats: np.ndarray,
    discretise: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    agg: str,
    clip_correction: bool = False,
    n_state_features: int | None = None,
) -> np.ndarray:
    """Vectorised correction aggregator for all provided actions.

    For tile coding, the active feature indices for different actions are just
    offsets of the same base indices; we exploit this to compute all actions at
    once without a Python loop.
    """
    actions = np.asarray(action, dtype=int).reshape(-1)
    if n_state_features is None:
        # Infer using size of qhats and the maximum action index + 1
        # This assumes zero-based contiguous action indices.
        num_actions = int(actions.max()) + 1
        n_state_features = int(qhats.size // max(1, num_actions))

    idxs = discretise_batched(
        state, actions, discretise, n_state_features=int(n_state_features)
    )
    # Gather per-tile corrections for each action: shape (A, T)
    vals = qhats[idxs]

    if agg == "max":
        corrs = np.max(vals, axis=1)
    elif agg == "mean":
        corrs = np.mean(vals, axis=1)
    elif agg == "median":
        corrs = np.median(vals, axis=1)
    else:
        raise ValueError("Unknown agg. Use 'max', 'mean', or 'median'.")

    if clip_correction:
        corrs = np.clip(corrs, a_min=0, a_max=None)

    return corrs


def _calibrate_ccnn(
    model: DQN,
    vec_env,
    *,
    score_fn: Callable,
    scoring_method: ScoringMethod,
    n_calib_steps: int,
    k: int,
    max_distance_quantile: float,
):
    buffer = collect_transitions(model, vec_env, n_transitions=n_calib_steps)
    scores, scaler, tree, max_dist = calibrate_ccnn(
        model,
        buffer,
        k=k,
        score_fn=score_fn,
        scoring_method=scoring_method,
        max_distance_quantile=max_distance_quantile,
    )
    # Precompute a global fallback correction using the same conformal index,
    # but across the full calibration score distribution.
    n = scores.shape[0]
    q_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA_NN)) / n)
    fallback = float(np.quantile(scores, q_level, method="higher"))
    return scores, scaler, tree, max_dist, fallback


def main():
    def _env_list() -> Iterable[str]:
        if RUN_ALL_ENVS:
            return [
                "CartPole-v1",
                "MountainCar-v0",
                "Acrobot-v1",
                "LunarLander-v3",
            ]
        return [ENV_NAME]

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_summaries: dict[str, dict] = {}

    for env in _env_list():
        assert env in DISC_SETTINGS, f"No discretisation config for {env}"
        tiles, tilings = DISC_SETTINGS[env]

        # Load a learned policy (no retraining). Attaches a fresh env internally.
        model, vec_env = learn_dqn_policy(
            env_name=env,
            seed=SEED,
            train_from_scratch=False,
        )

        # Calibrate both variants once using the nominal env
        qhats, discretise = _calibrate_disc(
            model,
            vec_env,
            tiles=tiles,
            tilings=tilings,
            scoring_method=SCORING_METHOD,
            score_fn=SCORE_FN,
            n_calib_steps=N_CALIB_STEPS,
        )

        ccnn_scores, ccnn_scaler, ccnn_tree, ccnn_max_dist, ccnn_fallback = (
            _calibrate_ccnn(
                model,
                vec_env,
                score_fn=SCORE_FN,
                scoring_method=SCORING_METHOD,
                n_calib_steps=N_CALIB_STEPS,
                k=K,
                max_distance_quantile=CCNN_MAX_DISTANCE_QUANTILE,
            )
        )

        # Precompute j-th order statistic index used in conformal correction
        CCNN_J_IDX = int(np.ceil((K + 1) * (1 - float(ALPHA_NN))) - 1)
        CCNN_J_IDX = max(0, min(K - 1, CCNN_J_IDX))

        # Fresh eval env for timing loops
        eval_env = instantiate_eval_env(env_name=env)

        # Sanity: ensure policy in eval mode for consistent timing
        model.policy.eval()

        # Baseline timing
        baseline_times = _time_action_selection(
            model,
            eval_env,
            selector=_select_action_baseline,
            warmup_steps=WARMUP_STEPS_PER_METHOD,
            num_eps=NUM_EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EP,
        )

        # Discretised conformal timing
        disc_times = _time_action_selection(
            model,
            eval_env,
            selector=lambda m, o: _select_action_disc(
                m, o, qhats=qhats, discretise=discretise
            ),
            warmup_steps=WARMUP_STEPS_PER_METHOD,
            num_eps=NUM_EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EP,
        )

        # CCNN timing
        ccnn_times = _time_action_selection(
            model,
            eval_env,
            selector=lambda m, o: _select_action_ccnn(
                m,
                o,
                scores=ccnn_scores,
                tree=ccnn_tree,
                scaler=ccnn_scaler,
                k=K,
                j_idx=CCNN_J_IDX,
                max_dist=ccnn_max_dist,
                fallback_value=ccnn_fallback,
            ),
            warmup_steps=WARMUP_STEPS_PER_METHOD,
            num_eps=NUM_EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EP,
        )

        # Summaries (per env)
        results = [
            TimingResult("baseline_dqn", baseline_times),
            TimingResult("disc_cp", disc_times),
            TimingResult("ccnn_cp", ccnn_times),
        ]

        print(f"\nInference-time overhead results ({env}, seed={SEED})")
        print("-" * 72)
        basemed = np.median(baseline_times)
        basemean = baseline_times.mean()
        env_summary: dict[str, dict] = {}
        for res in results:
            stats = res.summary()
            overhead_median = stats["median_us"] / (basemed / 1e3)
            overhead_mean = stats["mean_us"] / (basemean / 1e3)
            print(
                f"{res.label:>12}: count={stats['count']:>5d}  "
                f"median={stats['median_us']:.2f}us  mean={stats['mean_us']:.2f}us  "
                f"std={stats['std_us']:.2f}us  "
                f"p95={stats['p95_us']:.2f}us  p99={stats['p99_us']:.2f}us  "
                f"overhead(x): median={overhead_median:.2f} mean={overhead_mean:.2f}"
            )
            env_summary[res.label] = {
                **stats,
                "overhead_median_x": float(overhead_median),
                "overhead_mean_x": float(overhead_mean),
            }

        # Accumulate per-env summary for YAML dump
        run_summaries[env] = env_summary

        # Persist YAML per env with timestamped filename
        out_dir = os.path.join("results", env)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"inference_overhead_{run_ts}.yaml")

        config_info = {
            "env": env,
            "seed": SEED,
            "n_calib_steps": N_CALIB_STEPS,
            "n_eval_episodes": NUM_EVAL_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EP,
            "warmup_steps_per_method": WARMUP_STEPS_PER_METHOD,
            "score_fn": getattr(SCORE_FN, "__name__", str(SCORE_FN)),
            "scoring_method": SCORING_METHOD,
            "disc": {
                "alpha": ALPHA_DISC,
                "tiles": tiles,
                "tilings": tilings,
                "aggregation": "max",
                "min_calib": 50,
            },
            "ccnn": {
                "alpha": ALPHA_NN,
                "k": K,
                "max_distance_quantile": CCNN_MAX_DISTANCE_QUANTILE,
            },
            "timestamp": run_ts,
        }
        with open(out_path, "w") as f:
            yaml.safe_dump(
                {
                    "config": config_info,
                    "summary": env_summary,
                },
                f,
                sort_keys=False,
            )

    # If multiple envs, also emit a combined summary file under results/
    if RUN_ALL_ENVS:
        combined_dir = os.path.join("results", "inference_overhead")
        os.makedirs(combined_dir, exist_ok=True)
        combined_path = os.path.join(
            combined_dir, f"inference_overhead_all_envs_{run_ts}.yaml"
        )
        global_cfg = {
            "seed": SEED,
            "n_calib_steps": N_CALIB_STEPS,
            "n_eval_episodes": NUM_EVAL_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EP,
            "warmup_steps_per_method": WARMUP_STEPS_PER_METHOD,
            "score_fn": getattr(SCORE_FN, "__name__", str(SCORE_FN)),
            "scoring_method": SCORING_METHOD,
            "disc_settings": DISC_SETTINGS,
            "disc_alpha": ALPHA_DISC,
            "ccnn_alpha": ALPHA_NN,
            "k": K,
            "ccnn_max_distance_quantile": CCNN_MAX_DISTANCE_QUANTILE,
            "timestamp": run_ts,
            "envs": list(run_summaries.keys()),
        }
        with open(combined_path, "w") as f:
            yaml.safe_dump(
                {
                    "config": global_cfg,
                    "summary": run_summaries,
                },
                f,
                sort_keys=False,
            )


if __name__ == "__main__":
    main()
