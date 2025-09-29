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
import yaml
from stable_baselines3 import DQN
from tqdm import tqdm

from crl.agents import learn_cqldqn_policy, learn_ddqn_policy, learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    correction_for,
    fill_calib_sets_mc,
    fill_calib_sets_td,
    signed_score,
)
from crl.ccnn import calibrate_ccnn, run_ccnn_experiment
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env
from crl.types import AgentTypes, CalibMethods, ClassicControl, ScoringMethod
from crl.utils.graphing import despine

# prevent deprecation spam when using Lunar Lander
warnings.filterwarnings("ignore", category=UserWarning)

# fmt: off
ALPHA_DISC = 0.25           # Conformal prediction miscoverage level (discrete algo)
ALPHA_NN = 0.1              # Conformal prediction miscoverage level (nearest neighbour algo)
CQL_ALPHA = 0.05
MIN_CALIB = 80              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES = 250
N_CALIB_STEPS = 2_500
N_TRAIN_STEPS = 50_000
K = 50
SCORING_METHOD: ScoringMethod = 'td'
AGENT_TYPE: AgentTypes = 'vanilla'
SCORE_FN = signed_score
RETRAIN = False
CALIB_METHODS: list[CalibMethods] = ['nocalib', 'ccdisc', 'ccnn']
# CALIB_METHODS: list[CalibMethods] = ['ccdisc']
CCNN_MAX_DISTANCE_QUANTILE = 0.9


@dataclass
class RobustnessConfig:
    """Configuration for robustness experiments.
    """

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
    calib_methods: list[CalibMethods] = field(default_factory=lambda: CALIB_METHODS.copy())
    ccnn_max_distance_quantile: float = CCNN_MAX_DISTANCE_QUANTILE
    debug_serial: bool = False
    debug_seed: int | None = 0


# fmt: on
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
    "CartPole-v1": ("length", np.arange(0.1, 3.1, 0.2), 4, 1),
    # Acrobot: vary link 1 length (0.5x–2.0x of default 1.0)
    "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), 6, 1),
    # MountainCar: vary gravity around default 0.0025
    "MountainCar-v0": ("gravity", np.arange(0.001, 0.005 + 0.00025, 0.00025), 10, 1),
    "LunarLander-v3": ("gravity", np.arange(-16.0, 0.0, 1.0), 4, 1),
}


def run_eval(
    model: DQN,
    discretise: Callable,
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    qhats: np.ndarray,
) -> list[float]:
    episodic_returns = []
    fallback = qhats["fallback"]

    # Determine number of discrete actions
    num_actions = getattr(ep_env.action_space, "n")

    for ep in range(num_eps):
        obs = ep_env.reset()
        for t in range(1000):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

            if conformalise:
                # Adjust the qvalues of each action using
                # the correction from CP
                for a in range(num_actions):
                    sa_feature = discretise(obs, a)  # np scalar
                    correction = qhats.get(sa_feature[0], fallback)
                    q_vals[a] -= correction

            action = q_vals.argmax().numpy().reshape(1)

            obs, reward, done, info = ep_env.step(action)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break

    return episodic_returns


def run_shift_experiment(
    model: DQN,
    qhats: dict[int, float],
    discretise: Callable,
    env_name: str,
    shift_params: dict,
    cfg: RobustnessConfig,
    num_eps: int = NUM_EVAL_EPISODES,
):
    # instantiate the shifted env
    eval_vec_env = instantiate_eval_env(env_name=env_name, **shift_params)
    exp_result = {"num_episodes": num_eps}

    # run an experiment with and without the CP lower-bound correction
    if "ccdisc" in cfg.calib_methods:
        returns_conf = run_eval(
            model,
            discretise,
            num_eps=num_eps,
            conformalise=True,
            ep_env=eval_vec_env,
            qhats=qhats,
        )
        exp_result["returns_conf"] = returns_conf

    if "nocalib" in cfg.calib_methods:
        returns_noconf = run_eval(
            model,
            discretise,
            num_eps=num_eps,
            conformalise=False,
            ep_env=eval_vec_env,
            qhats=qhats,
        )
        exp_result["returns_noconf"] = returns_noconf

    exp_result.update(shift_params)
    return exp_result


def run_single_seed_experiment(
    env_name: str,
    seed: int,
    cfg: RobustnessConfig,
):
    """Run one seed's experiment under the provided configuration."""
    # train the nominal policy
    model, vec_env = train_agent(env_name, seed, cfg)
    # discretise the space and collect observations for the calibration sets
    param, param_values, tiles, tilings = EVAL_PARAMETERS[env_name]
    discretise, n_features = build_tile_coding(model, vec_env, tiles, tilings)
    buffer = collect_transitions(model, vec_env, n_transitions=cfg.n_calib_steps)
    if cfg.scoring_method == "td":
        calib_sets = fill_calib_sets_td(
            model,
            buffer,
            discretise,
            score=cfg.score_fn,
        )
    elif cfg.scoring_method == "monte_carlo":
        calib_sets = fill_calib_sets_mc(
            model,
            buffer,
            discretise,
            score=cfg.score_fn,
        )
    qhats = compute_corrections(
        calib_sets,
        alpha=cfg.alpha_disc,
        min_calib=cfg.min_calib,
    )
    # print(f"{len(qhats)} total calibrated features")

    ccnn_scores, scaler, tree, max_dist = calibrate_ccnn(
        model,
        buffer,
        k=cfg.k,
        score_fn=cfg.score_fn,
        scoring_method=cfg.scoring_method,
        max_distance_quantile=cfg.ccnn_max_distance_quantile,
    )

    # Test agent on shifted environments
    results = []

    for param_val in (pbar := tqdm(param_values)):
        pbar.set_description(f"{param}={param_val:.1f}")
        eval_parameters = {param: param_val}
        # Always record the evaluation parameter so plotting doesn't KeyError
        shift_result = {param: float(param_val)}
        if "nocalib" in cfg.calib_methods or "ccdisc" in cfg.calib_methods:
            disc_results = run_shift_experiment(
                model,
                qhats,
                discretise,
                shift_params=eval_parameters,
                env_name=env_name,
                cfg=cfg,
                num_eps=cfg.num_eval_episodes,
            )
            shift_result.update(disc_results)

        if "ccnn" in cfg.calib_methods:
            ccnn_results = run_ccnn_experiment(
                model,
                env_name=env_name,
                alpha=cfg.alpha_nn,
                k=cfg.k,
                ccnn_scores=ccnn_scores,
                tree=tree,
                scaler=scaler,
                max_dist=max_dist,
                param=param,
                param_val=param_val,
                num_eps=cfg.num_eval_episodes,
            )
            shift_result.update(ccnn_results)
        results.append(shift_result)

    return results


def train_agent(env_name: ClassicControl, seed: int, cfg: RobustnessConfig):
    agent_type = cfg.agent_type
    if agent_type == "cql":
        model, vec_env = learn_cqldqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
            cql_alpha=cfg.cql_alpha,
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


plot_params = {
    "nocalib": {"key": "returns_noconf", "label": "Uncalibrated", "c": "tab:orange"},
    "ccdisc": {"key": "returns_conf", "label": "Calibrated (Disc)", "c": "tab:blue"},
    "ccnn": {"key": "returns_ccnn", "label": "Calibrated (NN)", "c": "tab:green"},
}


def plot_robustness(
    seed: int,
    results: list[dict],
    env_name: str,
    cfg: RobustnessConfig,
    out_dir: str,
):
    x_key = EVAL_PARAMETERS[env_name][0]
    xvals = np.array([res[x_key] for res in results])

    for calib_method in cfg.calib_methods:
        key = plot_params[calib_method]["key"]
        label = plot_params[calib_method]["label"]
        colour = plot_params[calib_method]["c"]
        returns = np.array([res[key] for res in results])
        mean = returns.mean(axis=1)
        se = returns.std(axis=1) / np.sqrt(cfg.num_eval_episodes)
        plt.errorbar(
            xvals,
            mean,
            yerr=se,
            marker="o",
            linestyle="-",
            capsize=4,
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
    plt.legend(frameon=False)
    plt.savefig(os.path.join(out_dir, f"robustness_experiment_{seed}.png"))
    plt.close()


def plot_occupancy_histograms(
    single_exp_result, env_name, seed, out_dir: str | None = None
):
    x_key: str = EVAL_PARAMETERS[env_name][0]
    if out_dir is None:
        out_dir = os.path.join("results", env_name)
    exp_dir = os.path.join(out_dir, f"hists_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    for ix in range(len(single_exp_result)):
        for conf in ["visits_conf", "visits_noconf"]:
            visits = []
            nominal_visits = single_exp_result[ix][conf]
            for ep in range(len(nominal_visits)):
                visits.extend(nominal_visits[ep])
            mean_occupancy = np.mean(visits)
            # print(f"{conf}: {len(visits):,} total visits")
            plt.hist(visits, bins=50, label=conf, alpha=0.7, density=True)
            plt.axvline(mean_occupancy, linestyle="--", alpha=0.5, c="k")

        param_val = f"{single_exp_result[ix][x_key]:.1f}"
        param_name = x_key.replace("_", " ").title()
        plt.title(f"{param_name}={param_val}")
        plt.xlabel(param_name)
        plt.ylabel("Occupancy rate")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.3)
        despine(plt.gca())
        plt.savefig(f"{exp_dir}/hist_{param_val.replace('.', '-')}.png")
        plt.close()


# %%
def main(
    env_name: str,
    config: RobustnessConfig | dict | None = None,
    results_out: str | None = None,
):
    """Run robustness experiments for a given environment.

    Parameters
    - env_name: Gymnasium environment id to evaluate.
    - config: Either a RobustnessConfig, a plain dict with matching keys, or
      None to use default values. The configuration captures all parameters that
      were previously taken from module-level constants.
    - results_out: Optional name for the results subdirectory under 'results/'.
      If provided, outputs are written to 'results/{results_out}' instead of
      'results/{env_name}'.
    """
    # Normalise config
    if config is None:
        # Build a config from the current module-level globals for backward
        # compatibility with any code that still mutates these variables.
        cfg = RobustnessConfig(
            alpha_disc=ALPHA_DISC,
            alpha_nn=ALPHA_NN,
            cql_alpha=CQL_ALPHA,
            min_calib=MIN_CALIB,
            num_experiments=NUM_EXPERIMENTS,
            num_eval_episodes=NUM_EVAL_EPISODES,
            n_calib_steps=N_CALIB_STEPS,
            n_train_steps=N_TRAIN_STEPS,
            k=K,
            scoring_method=SCORING_METHOD,
            agent_type=AGENT_TYPE,
            score_fn=SCORE_FN,
            retrain=RETRAIN,
            calib_methods=CALIB_METHODS.copy(),
            ccnn_max_distance_quantile=CCNN_MAX_DISTANCE_QUANTILE,
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
        "alpha_nn": cfg.alpha_nn,
        "disc_mincalib": cfg.min_calib,
        "n_train_steps": cfg.n_train_steps,
        "n_calib_steps": cfg.n_calib_steps,
        "n_eval_episodes": cfg.num_eval_episodes,
        "score_fn": getattr(cfg.score_fn, "__name__", str(cfg.score_fn)),
        "scoring_method": cfg.scoring_method,
        "ccnn_k": cfg.k,
        "ccnn_max_distance_quantile": cfg.ccnn_max_distance_quantile,
        "disc_tiles": EVAL_PARAMETERS[env_name][2],
        "disc_tilings": EVAL_PARAMETERS[env_name][3],
        "cql_alpha": cfg.cql_alpha if cfg.agent_type == "cql" else None,
        "calib_methods": cfg.calib_methods,
        "debug_serial": cfg.debug_serial,
        "debug_seed": cfg.debug_seed,
    }

    # Determine results directory (override env_name if results_out provided)
    exp_dir = os.path.join("results", results_out or env_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Pretty print config at start
    print("Experiment Configuration:")
    pprint.pprint(experiment_info, sort_dicts=False)

    # Save config to YAML
    yaml_path = os.path.join(exp_dir, "experiment_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(experiment_info, f, sort_keys=False)

    seeds = list(range(cfg.num_experiments))
    max_workers = min(cfg.num_experiments, os.cpu_count() or 1)

    if cfg.debug_serial:
        # Run a single seed synchronously for debugging (easy to attach a debugger)
        debug_seed = cfg.debug_seed if cfg.debug_seed is not None else seeds[0]
        single_exp_result = run_single_seed_experiment(env_name, debug_seed, cfg)
        # Plot per-seed robustness in the main process to avoid conflicts
        plot_robustness(debug_seed, single_exp_result, env_name, cfg, exp_dir)
        all_results.append({"seed": debug_seed, "results": single_exp_result})
        # Persist incrementally as results arrive
        with open(os.path.join(exp_dir, "robustness_experiment.pkl"), "wb") as f:
            pickle.dump(all_results, f)
    else:
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
                with open(
                    os.path.join(exp_dir, "robustness_experiment.pkl"), "wb"
                ) as f:
                    pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    # envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    # env = "CartPole-v1"
    # env = "LunarLander-v3"
    cfg = RobustnessConfig()
    env = "LunarLander-v3"
    # cfg = RobustnessConfig(debug_serial=True, debug_seed=0)
    main(env_name="LunarLander-v3", config=cfg)
    # if env == "MountainCar-v0":
    #     assert cfg.n_train_steps == 120_000
    # elif env == "Acrobot-v1":
    #     assert cfg.n_train_steps == 100_000
    # elif env == "CartPole-v1":
    #     assert cfg.n_train_steps in [50_000, 100_000]
    # main(env_name=env, config=cfg)

# %%
