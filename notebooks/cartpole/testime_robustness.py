"""Test-time robustness experiments using current CRL APIs.

This script mirrors the configuration and multiprocessing scaffolding of
`traintime_robustness.py`, but implements test-time adaptation of
conformal corrections (discrete tile-coding) during evaluation.
"""

# %%
import os
import pickle
import pprint
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from tqdm import tqdm

from crl.agents import learn_cqldqn_policy, learn_ddqn_policy, learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    correction_for,
    fill_calib_sets,
    fill_calib_sets_mc,
    signed_score,
)
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env
from crl.types import AgentTypes, ClassicControl, ScoringMethod
from crl.utils.graphing import despine

# prevent deprecation spam when using Lunar Lander
warnings.filterwarnings("ignore", category=UserWarning)

# fmt: off
ALPHA_DISC = 0.01            # Conformal prediction miscoverage level (discrete algo)
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES = 250
N_CALIB_STEPS = 10_000
N_TRAIN_STEPS = 50_000
SCORING_METHOD: ScoringMethod = 'td'
AGENT_TYPE: AgentTypes = 'vanilla'
SCORE_FN = signed_score
RETRAIN = False
AGG = 'max'
ETA = 1.0                  # Base step size for test-time adaptation
# fmt: on


@dataclass
class RobustnessConfig:
    """Configuration for test-time robustness experiments."""

    alpha_disc: float = ALPHA_DISC
    min_calib: int = MIN_CALIB
    num_experiments: int = NUM_EXPERIMENTS
    num_eval_episodes: int = NUM_EVAL_EPISODES
    n_calib_steps: int = N_CALIB_STEPS
    n_train_steps: int = N_TRAIN_STEPS
    scoring_method: ScoringMethod = SCORING_METHOD
    agent_type: AgentTypes = AGENT_TYPE
    score_fn: Callable = SCORE_FN
    retrain: bool = RETRAIN
    agg: str = AGG
    eta: float = ETA


@dataclass
class ExperimentParams:
    env_name: str
    param: str
    param_vals: Sequence
    state_bins: list[int]
    nominal_value: float
    success_threshold: int = 0
    good_seeds: list[int] | None = None


EVAL_PARAMETERS = {
    # CartPole: vary pole length around nominal 0.5 value
    "CartPole-v1": ("length", np.arange(0.1, 3.1, 0.2), 6, 1),
    # Acrobot: vary link 1 length (0.5x–2.0x of default 1.0)
    "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), 6, 1),
    # MountainCar: vary gravity around default 0.0025
    "MountainCar-v0": ("gravity", np.arange(0.001, 0.005 + 0.00025, 0.00025), 10, 1),
    # LunarLander v3: vary gravity
    "LunarLander-v3": ("gravity", np.arange(-12, -0, 0.5), 4, 4),
}


def run_tta_eval(
    model: DQN,
    discretise: Callable,
    ep_env: gym.Env,
    qhats_offline: np.ndarray,
    num_eps: int,
    alpha: float,
    agg: str,
    eta: float,
) -> list[float]:
    """Run test-time adaptation using online updates of per‑tile corrections.

    The online correction vector is initialised from ``qhats_offline`` and
    updated after each transition using the Angelopoulos‑style rule:
        qhat_{t+1} = qhat_t + eta_t * (I[target < q_pred] - alpha)

    where eta_t = eta / sqrt(n_i + 1), with n_i a per‑tile counter.
    """
    episodic_returns = []

    # Determine number of discrete actions
    num_actions = getattr(ep_env.action_space, "n")

    # Online state: copy offline qhats and maintain per‑index counters
    online_qhats = np.array(qhats_offline, copy=True)
    update_counts = np.zeros_like(online_qhats, dtype=np.int64)

    for _ in range(num_eps):
        obs = ep_env.reset()

        prev_obs = None
        prev_action = None
        prev_reward = None
        prev_done = None
        prev_raw_q_vals = None

        for t in range(1000):
            raw_q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

            # Apply current (online) corrections for each action
            q_vals = raw_q_vals.clone()
            for a in range(num_actions):
                corr = correction_for(
                    obs,
                    np.asarray([a]),
                    online_qhats,
                    discretise,
                    agg=agg,
                    clip_correction=False,
                )
                q_vals[a] -= corr

            # Greedy action under corrected Qs
            action = q_vals.argmax().numpy().reshape(1)
            action_idx = int(action.item())

            # Step env
            next_obs, reward, done, info = ep_env.step(action)

            # Update online corrections for PREVIOUS transition
            if prev_obs is not None:
                max_q_next = 0.0 if prev_done else raw_q_vals.max().item()
                target = float(prev_reward) + float(model.gamma) * float(max_q_next)
                q_prev_sa = float(prev_raw_q_vals[prev_action].item())
                err = 1.0 if target < q_prev_sa else 0.0

                idxs = discretise(prev_obs, np.asarray([prev_action]))
                for idx in np.asarray(idxs).astype(int).ravel():
                    update_counts[idx] += 1
                    eta_t = eta / np.sqrt(update_counts[idx] + 1)
                    online_qhats[idx] += eta_t * (err - alpha)

            # Shift prev_* buffers
            prev_obs = obs
            prev_action = action_idx
            prev_reward = float(reward)
            prev_done = bool(done)
            prev_raw_q_vals = raw_q_vals

            # Advance state
            obs = next_obs

            # Terminal handling: final update without bootstrap
            if done:
                idxs = discretise(prev_obs, np.asarray([prev_action]))
                q_prev_sa = float(prev_raw_q_vals[prev_action].item())
                target = float(prev_reward)
                err = 1.0 if target < q_prev_sa else 0.0
                for idx in np.asarray(idxs).astype(int).ravel():
                    update_counts[idx] += 1
                    eta_t = eta / np.sqrt(update_counts[idx] + 1)
                    online_qhats[idx] += eta_t * (err - alpha)

                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break

    return episodic_returns


def run_shift_experiment_tta(
    model: DQN,
    qhats: np.ndarray,
    discretise: Callable,
    env_name: str,
    shift_params: dict,
    cfg: RobustnessConfig,
    num_eps: int,
):
    # instantiate the shifted env
    eval_vec_env = instantiate_eval_env(env_name=env_name, **shift_params)
    exp_result = {"num_episodes": num_eps}

    returns_tta = run_tta_eval(
        model,
        discretise,
        ep_env=eval_vec_env,
        qhats_offline=qhats,
        num_eps=num_eps,
        alpha=cfg.alpha_disc,
        agg=cfg.agg,
        eta=cfg.eta,
    )
    exp_result["returns_tta"] = returns_tta
    exp_result.update(shift_params)
    return exp_result


def run_single_seed_experiment(env_name: str, seed: int, cfg: RobustnessConfig):
    """Run one seed's test‑time adaptation experiment under the provided configuration."""
    # Train the nominal policy
    model, vec_env = train_agent(env_name, seed, cfg)

    # Discretiser and offline calibration
    param, param_values, tiles, tilings = EVAL_PARAMETERS[env_name]
    discretise, n_features = build_tile_coding(model, vec_env, tiles, tilings)
    # Ensure an int for downstream allocation loops
    n_features = int(n_features)
    buffer = collect_transitions(model, vec_env, n_transitions=cfg.n_calib_steps)
    if cfg.scoring_method == "td":
        calib_sets = fill_calib_sets(
            model,
            buffer,
            discretise,
            n_features,
            score=cfg.score_fn,
        )
    elif cfg.scoring_method == "monte_carlo":
        calib_sets = fill_calib_sets_mc(
            model,
            buffer,
            discretise,
            n_features,
            score=cfg.score_fn,
        )
    else:
        raise NotImplementedError(f"Unknown scoring_method: {cfg.scoring_method}")

    qhats, visits = compute_corrections(
        calib_sets,
        alpha=cfg.alpha_disc,
        min_calib=cfg.min_calib,
    )

    # Sweep shifted environments
    results = []
    for param_val in (pbar := tqdm(param_values)):
        pbar.set_description(f"{param}={round(param_val, 3)}")
        eval_parameters = {param: float(param_val)}
        shift_result = run_shift_experiment_tta(
            model,
            qhats,
            discretise,
            shift_params=eval_parameters,
            env_name=env_name,
            cfg=cfg,
            num_eps=cfg.num_eval_episodes,
        )
        shift_result[param] = float(param_val)
        results.append(shift_result)

    return results


def train_agent(env_name: ClassicControl, seed: int, cfg: RobustnessConfig):
    agent_type = cfg.agent_type
    if agent_type == "cql":
        model, vec_env = learn_cqldqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
        )
    elif agent_type == "ddqn":
        model, vec_env = learn_ddqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
        )
    elif agent_type == "vanilla":
        model, vec_env = learn_dqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
            train_from_scratch=cfg.retrain,
        )
    else:
        raise ValueError(f"{agent_type=} not recognised. Should be one of {AgentTypes}")
    return model, vec_env


def plot_robustness(
    seed: int, results: list[dict], env_name: str, cfg: RobustnessConfig, out_dir: str
):
    x_key = EVAL_PARAMETERS[env_name][0]
    xvals = np.array([res[x_key] for res in results])

    returns = np.array([res["returns_tta"] for res in results])
    mean = returns.mean(axis=1)
    se = returns.std(axis=1) / np.sqrt(cfg.num_eval_episodes)
    plt.errorbar(
        xvals,
        mean,
        yerr=se,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Calibrated (Test-time)",
        c="tab:blue",
    )

    plt.ylabel("Episodic Return")
    plt.xlabel(x_key)
    if x_key == "length":
        plt.axvline(0.5, linestyle="--", c="k", alpha=0.5)
        plt.xlim(0, max(EVAL_PARAMETERS[env_name][1]))
        plt.ylim(0, None)
    despine(plt.gca())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.savefig(os.path.join(out_dir, f"testtime_robustness_{seed}.png"))
    plt.close()


# %%
def main(
    env_name: str,
    config: RobustnessConfig | dict | None = None,
    results_out: str | None = None,
):
    """Run test‑time robustness experiments for a given environment.

    Parameters
    - env_name: Gymnasium environment id to evaluate.
    - config: Either a RobustnessConfig, a plain dict with matching keys, or
      None to use default values.
    - results_out: Optional override for results subdirectory under 'results/'.
      Defaults to '{env_name}_tt'.
    """
    # Normalise config
    if config is None:
        cfg = RobustnessConfig(
            alpha_disc=ALPHA_DISC,
            min_calib=MIN_CALIB,
            num_experiments=NUM_EXPERIMENTS,
            num_eval_episodes=NUM_EVAL_EPISODES,
            n_calib_steps=N_CALIB_STEPS,
            n_train_steps=N_TRAIN_STEPS,
            scoring_method=SCORING_METHOD,
            agent_type=AGENT_TYPE,
            score_fn=SCORE_FN,
            retrain=RETRAIN,
            agg=AGG,
            eta=ETA,
        )
    elif isinstance(config, dict):
        cfg = RobustnessConfig(**config)
    else:
        cfg = config

    all_results = []
    experiment_info = {
        "env": env_name,
        "agent_type": cfg.agent_type,
        "alpha_disc": cfg.alpha_disc,
        "disc_mincalib": cfg.min_calib,
        "n_train_steps": cfg.n_train_steps,
        "n_calib_steps": cfg.n_calib_steps,
        "n_eval_episodes": cfg.num_eval_episodes,
        "score_fn": getattr(cfg.score_fn, "__name__", str(cfg.score_fn)),
        "scoring_method": cfg.scoring_method,
        "disc_tiles": EVAL_PARAMETERS[env_name][2],
        "disc_tilings": EVAL_PARAMETERS[env_name][3],
        "tta_eta": cfg.eta,
        "tta_agg": cfg.agg,
    }

    # Determine results directory (defaults to env_name + "_tt")
    results_subdir = results_out or f"{env_name}_tt"
    exp_dir = os.path.join("results", results_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    # Pretty print config at start
    print("Test-time Experiment Configuration:")
    pprint.pprint(experiment_info, sort_dicts=False)

    # Save config to YAML
    yaml_path = os.path.join(exp_dir, "experiment_config.yaml")
    with open(yaml_path, "w") as f:
        import yaml

        yaml.safe_dump(experiment_info, f, sort_keys=False)

    seeds = list(range(cfg.num_experiments))
    max_workers = min(cfg.num_experiments, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_seed = {
            ex.submit(
                run_single_seed_experiment,
                env_name,
                seed,
                cfg,
            ): seed
            for seed in seeds
        }

        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            single_exp_result = future.result()
            # Plot per-seed robustness in the main process to avoid conflicts
            plot_robustness(seed, single_exp_result, env_name, cfg, exp_dir)
            all_results.append({"seed": seed, "results": single_exp_result})
            # Persist incrementally as results arrive
            with open(os.path.join(exp_dir, "testtime_robustness.pkl"), "wb") as f:
                pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    # env = "CartPole-v1"
    # env = "Acrobot-v1"
    # env = "MountainCar-v0"
    # env = "LunarLander-v3"
    env = "Acrobot-v1"
    cfg = RobustnessConfig()
    # Sanity: validate training steps similar to traintime script, if desired
    if env == "MountainCar-v0":
        assert cfg.n_train_steps in [50_000, 120_000]
    elif env == "Acrobot-v1":
        assert cfg.n_train_steps in [50_000, 100_000]
    elif env == "CartPole-v1":
        assert cfg.n_train_steps in [50_000, 100_000]
    main(env_name=env, config=cfg)

# %%
