# %%
import os
import pickle
import pprint
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN

from crl.agents import learn_cqldqn_policy, learn_ddqn_policy, learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    correction_for,
    fill_calib_sets,
    fill_calib_sets_mc,
    signed_score,
)
from crl.discretise import (
    build_binary_partition,
    build_tile_coding,
)
from crl.env import instantiate_eval_env
from crl.types import AgentTypes, ClassicControl, ScoringMethod
from crl.utils.graphing import despine

# ---------------- Defaults mirrored from traintime_robustness -----------------
ALPHA_DISC = 0.01
MIN_CALIB = 50
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES = 250
N_CALIB_STEPS = 10_000
N_TRAIN_STEPS = 50_000
SCORING_METHOD: ScoringMethod = "td"
AGENT_TYPE: AgentTypes = "vanilla"
SCORE_FN = signed_score
RETRAIN = False


@dataclass
class DiscExpConfig:
    """Configuration for discretisation experiments (discrete CP only).

    This mirrors training/eval defaults from traintime_robustness and adds
    grids of discretiser hyperparameters to sweep per environment.
    """

    # Conformal / eval config
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
    agg: str = "max"
    clip_correction: bool = False

    # Discretiser search spaces
    # Grid: numbers of bins per-dimension (we expand to [n]*D for each env)
    grid_bins: list[int] = field(default_factory=lambda: [4, 6])
    # Tile coding: (tiles, tilings)
    tile_options: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (4, 2),
            (4, 4),
            (6, 2),
            (6, 4),
            (8, 2),
        ]
    )
    tile_obs_quantile: float = 0.1
    # Tree partitioning: a small grid of depth/leaf/criterion
    tree_options: list[dict] = field(
        default_factory=lambda: [
            dict(max_depth=4, min_samples_leaf=1, use_impurity_split=False),
            dict(max_depth=6, min_samples_leaf=1, use_impurity_split=False),
            dict(
                max_depth=6,
                min_samples_leaf=5,
                use_impurity_split=True,
                min_impurity_decrease=0.05,
            ),
            dict(max_depth=8, min_samples_leaf=1, use_impurity_split=False),
            dict(
                max_depth=8,
                min_samples_leaf=10,
                use_impurity_split=True,
                min_impurity_decrease=0.10,
            ),
        ]
    )


@dataclass
class ExperimentParams:
    env_name: str
    param: str
    param_vals: Sequence
    state_bins: list[int]
    nominal_value: float
    success_threshold: int = 0
    good_seeds: list[int] | None = None


# Environment shift parameters (mirrors traintime_robustness)
EVAL_PARAMETERS = {
    "CartPole-v1": ("length", np.arange(0.1, 3.1, 0.2), 6, 1),
    "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), 6, 1),
    "MountainCar-v0": ("gravity", np.arange(0.001, 0.005 + 0.00025, 0.00025), 10, 1),
    "LunarLander-v3": ("gravity", np.arange(-12, -0, 0.5), 4, 4),
}


# ------------------------- Discretiser builders -------------------------------
def build_discretiser_variants(
    model: DQN, vec_env, env_name: ClassicControl, cfg: DiscExpConfig
):
    """Construct a list of discretiser variants to evaluate.

    Returns a list of dicts with keys: scheme, label, params, discretise, n_states.
    """
    variants = []

    def _ensure_index_array(
        disc_fn: Callable[[np.ndarray, np.ndarray], int | np.ndarray],
    ):
        def _wrapped(obs: np.ndarray, action: np.ndarray):
            # Ensure obs has leading batch dim
            obs_arr = np.asarray(obs)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr[None, :]

            # Ensure action is array with shape (1,)
            if isinstance(action, (int, np.integer)):
                act_arr = np.array([int(action)], dtype=int)
            else:
                act_arr = np.asarray(action)
                if act_arr.ndim == 0:
                    act_arr = np.array([int(act_arr)], dtype=int)

            out = disc_fn(obs_arr, act_arr)
            if isinstance(out, (int, np.integer)):
                return np.array([int(out)], dtype=int)
            arr = np.asarray(out).reshape(-1)
            return arr.astype(int, copy=False)

        return _wrapped

    # Uniform grid variants (implemented via tile coding with tilings=1)
    for n in cfg.grid_bins:
        discretise, n_states = build_tile_coding(
            model,
            vec_env,
            tiles=int(n),
            tilings=1,
            obs_quantile=cfg.tile_obs_quantile,
        )
        discretise = _ensure_index_array(discretise)
        variants.append(
            dict(
                scheme="grid",
                label=f"Grid[{n}]",
                params=dict(tiles=int(n), tilings=1),
                discretise=discretise,
                n_states=n_states,
            )
        )

    # Tile-coding variants
    for tiles, tilings in cfg.tile_options:
        discretise, n_states = build_tile_coding(
            model,
            vec_env,
            tiles=tiles,
            tilings=tilings,
            obs_quantile=cfg.tile_obs_quantile,
        )
        discretise = _ensure_index_array(discretise)
        variants.append(
            dict(
                scheme="tile",
                label=f"Tile[{tiles}x{tilings}]",
                params=dict(tiles=tiles, tilings=tilings),
                discretise=discretise,
                n_states=n_states,
            )
        )

    # Binary partition (tree) variants
    for opt in cfg.tree_options:
        opt = {**dict(min_impurity_decrease=0.0), **opt}
        discretise, n_states = build_binary_partition(
            model,
            vec_env,
            max_depth=int(opt.get("max_depth", 6)),
            min_samples_leaf=int(opt.get("min_samples_leaf", 1)),
            use_impurity_split=bool(opt.get("use_impurity_split", False)),
            min_impurity_decrease=float(opt.get("min_impurity_decrease", 0.0)),
        )
        discretise = _ensure_index_array(discretise)
        imp = "imp" if opt.get("use_impurity_split", False) else "med"
        label = f"Tree[d={opt['max_depth']},l={opt['min_samples_leaf']},{imp}]"
        variants.append(
            dict(
                scheme="tree",
                label=label,
                params=opt,
                discretise=discretise,
                n_states=n_states,
            )
        )

    return variants


# -------------------------- Training helper ----------------------------------
def train_agent(env_name: ClassicControl, seed: int, cfg: DiscExpConfig):
    agent_type = cfg.agent_type
    if agent_type == "cql":
        model, vec_env = learn_cqldqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
            cql_alpha=0.05,
        )
    elif agent_type == "ddqn":
        model, vec_env = learn_ddqn_policy(
            env_name=env_name, seed=seed, total_timesteps=cfg.n_train_steps
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


# -------------------------- Eval helper --------------------------------------
def run_eval(
    model: DQN,
    discretise: Callable,
    num_eps: int,
    ep_env: gym.Env,
    qhats: np.ndarray,
    agg: str = "max",
    clip_correction: bool = False,
) -> list[float]:
    """Evaluate the model with conformal correction applied (discrete only)."""
    episodic_returns = []
    num_actions = getattr(ep_env.action_space, "n")
    for _ in range(num_eps):
        obs = ep_env.reset()
        for _t in range(1000):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()
            # Apply discrete CP correction to all actions
            for a in range(num_actions):
                corr = correction_for(
                    obs, a, qhats, discretise, agg=agg, clip_correction=clip_correction
                )
                q_vals[a] -= corr
            action = q_vals.argmax().numpy().reshape(1)
            obs, reward, done, info = ep_env.step(action)
            if done:
                episodic_returns.append(info[0]["episode"]["r"])
                break
    return episodic_returns


# ---------------------- Single-seed experiment --------------------------------
def run_single_seed_experiment(env_name: str, seed: int, cfg: DiscExpConfig):
    # Train or load cached policy
    model, vec_env = train_agent(env_name, seed, cfg)

    # Collect one buffer and reuse across discretisers
    buffer = collect_transitions(model, vec_env, n_transitions=cfg.n_calib_steps)

    # Build discretiser variants
    variants = build_discretiser_variants(model, vec_env, env_name, cfg)

    # Calibrate corrections for each variant
    for v in variants:
        if cfg.scoring_method == "td":
            calib_sets = fill_calib_sets(
                model, buffer, v["discretise"], v["n_states"], score=cfg.score_fn
            )
        elif cfg.scoring_method == "monte_carlo":
            calib_sets = fill_calib_sets_mc(
                model, buffer, v["discretise"], v["n_states"], score=cfg.score_fn
            )
        else:
            raise ValueError(f"Unknown scoring_method={cfg.scoring_method}")
        qhats, visits = compute_corrections(
            calib_sets, alpha=cfg.alpha_disc, min_calib=cfg.min_calib
        )
        v["qhats"] = qhats
        v["visits"] = visits  # not used in eval but kept for completeness

    # Evaluate across shifts
    param, param_values, _tiles, _tilings = EVAL_PARAMETERS[env_name]
    results = []
    for param_val in param_values:
        eval_vec_env = instantiate_eval_env(
            env_name=env_name, **{param: float(param_val)}
        )
        entry = {param: float(param_val), "num_episodes": cfg.num_eval_episodes}
        for v in variants:
            key = v["label"]
            returns_conf = run_eval(
                model,
                discretise=v["discretise"],
                num_eps=cfg.num_eval_episodes,
                ep_env=eval_vec_env,
                qhats=v["qhats"],
                agg=cfg.agg,
                clip_correction=cfg.clip_correction,
            )
            entry[key] = returns_conf
        results.append(entry)

    metadata = [
        dict(scheme=v["scheme"], label=v["label"], params=v["params"]) for v in variants
    ]
    return results, metadata


# ------------------------------ Plotting --------------------------------------
SCHEME_COLOURS = {"grid": "tab:blue", "tile": "tab:orange", "tree": "tab:green"}


def plot_per_seed(
    results: list[dict],
    metadata: list[dict],
    env_name: str,
    cfg: DiscExpConfig,
    out_dir: str,
):
    x_key = EVAL_PARAMETERS[env_name][0]
    xvals = np.array([res[x_key] for res in results])

    # For consistent legend ordering
    for meta in metadata:
        label = meta["label"]
        scheme = meta["scheme"]
        colour = SCHEME_COLOURS.get(scheme, None)
        series = np.array([res[label] for res in results])  # (X, episodes)
        mean = series.mean(axis=1)
        se = series.std(axis=1) / np.sqrt(cfg.num_eval_episodes)
        plt.errorbar(
            xvals,
            mean,
            yerr=se,
            marker="o",
            linestyle="-",
            capsize=3,
            label=label,
            c=colour,
        )

    plt.ylabel("Episodic Return")
    plt.xlabel(x_key)
    if x_key == "length":
        plt.axvline(0.5, linestyle="--", c="k", alpha=0.5)
        plt.xlim(0, max(EVAL_PARAMETERS[env_name][1]))
        plt.ylim(0, None)
    despine(plt.gca())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "disc_robustness.png"))
    plt.close()


# ------------------------------- Main -----------------------------------------
def main(
    env_name: ClassicControl,
    config: DiscExpConfig | dict | None = None,
    results_out: str | None = None,
):
    # Config normalisation
    cfg = (
        DiscExpConfig(**config)
        if isinstance(config, dict)
        else (config or DiscExpConfig())
    )

    # Determine results dir
    exp_dir = os.path.join("results", results_out or env_name, "discretisers")
    os.makedirs(exp_dir, exist_ok=True)

    # Log and save config
    experiment_info = {
        "env": env_name,
        "alpha_disc": cfg.alpha_disc,
        "disc_mincalib": cfg.min_calib,
        "n_train_steps": cfg.n_train_steps,
        "n_calib_steps": cfg.n_calib_steps,
        "n_eval_episodes": cfg.num_eval_episodes,
        "score_fn": getattr(cfg.score_fn, "__name__", str(cfg.score_fn)),
        "scoring_method": cfg.scoring_method,
        "agent_type": cfg.agent_type,
        "agg": cfg.agg,
        "clip_correction": cfg.clip_correction,
        "grid_bins": cfg.grid_bins,
        "tile_options": cfg.tile_options,
        "tile_obs_quantile": cfg.tile_obs_quantile,
        "tree_options": cfg.tree_options,
    }
    print("Experiment Configuration:")
    pprint.pprint(experiment_info, sort_dicts=False)
    with open(os.path.join(exp_dir, "experiment_config.yaml"), "w") as f:
        yaml.safe_dump(experiment_info, f, sort_keys=False)

    seeds = list(range(cfg.num_experiments))
    max_workers = min(cfg.num_experiments, os.cpu_count() or 1)

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_seed = {
            ex.submit(run_single_seed_experiment, env_name, seed, cfg): seed
            for seed in seeds
        }

        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            single_results, metadata = future.result()
            plot_dir = os.path.join(exp_dir, f"seed_{seed}")
            plot_per_seed(single_results, metadata, env_name, cfg, plot_dir)
            all_results.append(
                {"seed": seed, "results": single_results, "metadata": metadata}
            )

            # Save incrementally as results arrive
            with open(
                os.path.join(exp_dir, "discretisation_experiment.pkl"), "wb"
            ) as f:
                pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    # Run all classic-control envs with defaults
    envs = list(EVAL_PARAMETERS.keys())
    for env in envs:
        cfg = DiscExpConfig()
        if env == "MountainCar-v0":
            assert cfg.n_train_steps == 120_000 or cfg.n_train_steps == N_TRAIN_STEPS
        elif env == "Acrobot-v1":
            assert cfg.n_train_steps == 100_000 or cfg.n_train_steps == N_TRAIN_STEPS
        elif env == "CartPole-v1":
            assert cfg.n_train_steps in [50_000, 100_000]
        main(env_name=env, config=cfg)

# %%
