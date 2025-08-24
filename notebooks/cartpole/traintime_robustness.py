# %%
import os
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from crl.utils.graphing import despine
from tqdm import tqdm
from typing import Any, Callable
from stable_baselines3 import DQN
from dataclasses import dataclass
from typing import Sequence

from crl.cons.calib import (
    compute_corrections,
    collect_transitions,
    # collect_training_transitions,
    fill_calib_sets,
    fill_calib_sets_mc,
    signed_score,
    correction_for,
)
from crl.cons.cartpole import instantiate_eval_env, learn_dqn_policy
from crl.cons.cql import learn_cqldqn_policy
from crl.cons.discretise import build_tile_coding

# fmt: off
ALPHA = 0.1                 # Conformal prediction miscoverage level
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES=250
N_CALIB_TRANSITIONS=1_000
N_TRAIN_EPISODES = 50_000
USE_MC_RETURN_SCORE = True

@dataclass
class ExperimentParams:
    env_name: str
    param: str
    param_vals: Sequence
    state_bins: list[int]
    nominal_value: float
    success_threshold: int = 0
    good_seeds: list[int] | None = None


# EXPERIMENTS = [
#     ExperimentParams("CartPole-v1", "length", np.linspace(0.1, 2.0, 20), [6] * 4, 0.5),
#     ExperimentParams("Acrobot-v1", "LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), [6] * 6, 1.0),
#     # ExperimentParams("Acrobot-v1", "LINK_MASS_2", np.linspace(0.5, 2.0, 16), [4] * 6, 1.0),
#     ExperimentParams("MountainCar-v0", "gravity", np.linspace(0.0015, 0.0040, 21), [6] * 2, 9.8)
# ]
# fmt: on

EVAL_PARAMETERS = {
    # CartPole: vary pole length around nominal 0.5 value
    # "CartPole-v1": ("length", np.linspace(0.1, 2.0, 20), [6] * 4),
    "CartPole-v1": ("length", np.arange(0.1, 3.1, 0.2), 6, 1),
    # Acrobot: vary link 1 length (0.5x–2.0x of default 1.0)
    "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), 6, 1),
    # "Acrobot-v1": ("LINK_MASS_1", np.linspace(0.5, 2.0, 16), [4] * 6),
    # "Acrobot-v1": ("LINK_MOI", np.arange(0.5, 2.1, 0.1), 8, 1),
    # MountainCar: vary gravity around default 0.0025
    "MountainCar-v0": ("gravity", np.arange(0.001, 0.005 + 0.00025, 0.00025), 10, 1),
    "LunarLander-v3": ("gravity", np.arange(-12, -0, 0.5), 4, 4),
}


def run_eval(
    model: DQN,
    discretise: Callable,
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    qhats: np.ndarray,
    visits: np.ndarray,
    agg: str = "max",
    clip_correction: bool = False,
) -> list[float]:
    episodic_returns = []
    all_occupancies = []

    # Determine number of discrete actions
    num_actions = getattr(ep_env.action_space, "n")
    discount = model.gamma

    for ep in range(num_eps):
        obs = ep_env.reset()
        ep_occupancy = []
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

            # The state-visitation count should be for the state *before* we step
            # the environment.
            occupancy = correction_for(
                obs,
                action,
                visits,
                discretise,
                agg=agg,
                clip_correction=clip_correction,
            )
            obs, reward, done, info = ep_env.step(action)

            ep_occupancy.append(occupancy)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break

        all_occupancies.append(ep_occupancy)
    return episodic_returns, all_occupancies


def run_shift_experiment(
    model: DQN,
    qhats: np.ndarray,
    visits: np.ndarray,
    discretise: Callable,
    env_name: str,
    shift_params: dict,
    num_eps: int = NUM_EVAL_EPISODES,
):
    # instantiate the shifted env
    eval_vec_env = instantiate_eval_env(env_name=env_name, **shift_params)

    # run an experiment with and without the CP lower-bound correction
    returns_conf, visits_conf = run_eval(
        model,
        discretise,
        num_eps=num_eps,
        conformalise=True,
        ep_env=eval_vec_env,
        qhats=qhats,
        visits=visits,
    )
    returns_noconf, visits_noconf = run_eval(
        model,
        discretise,
        num_eps=num_eps,
        conformalise=False,
        ep_env=eval_vec_env,
        qhats=qhats,
        visits=visits,
    )

    exp_result = {
        "returns_conf": returns_conf,
        "returns_noconf": returns_noconf,
        "visits_conf": visits_conf,
        "visits_noconf": visits_noconf,
        "num_episodes": num_eps,
    }
    exp_result.update(shift_params)
    return exp_result


def run_single_seed_experiment(
    env_name: str, seed: int, cql_loss_weight: float | None = None
):
    # train the nominal policy
    if cql_loss_weight is not None:
        model, vec_env = learn_cqldqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=N_TRAIN_EPISODES,
            cql_alpha=cql_loss_weight,
        )
    else:
        model, vec_env = learn_dqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=N_TRAIN_EPISODES,
        )
    # discretise the space and collect observations for the calibration sets
    param, param_values, tiles, tilings = EVAL_PARAMETERS[env_name]
    # discretise, n_discrete_states = build_tiling(model, vec_env, state_bins=state_bins)
    discretise, n_discrete_states = build_tile_coding(
        model, vec_env, tiles=tiles, tilings=tilings
    )
    buffer = collect_transitions(model, vec_env, n_transitions=N_CALIB_TRANSITIONS)
    if not USE_MC_RETURN_SCORE:
        calib_sets = fill_calib_sets(
            model,
            buffer,
            discretise,
            n_discrete_states,
            score=signed_score,
        )
    else:
        calib_sets = fill_calib_sets_mc(
            model,
            buffer,
            discretise,
            n_discrete_states,
            score=signed_score,
        )
    qhats, visits = compute_corrections(
        calib_sets,
        alpha=ALPHA,
        min_calib=MIN_CALIB,
    )

    # Test agent on shifted environments
    results = []

    for param_val in (pbar := tqdm(param_values)):
        pbar.set_description(f"{param}={param_val:.1f}")
        eval_parameters = {param: param_val}
        results.append(
            run_shift_experiment(
                model,
                qhats,
                visits,
                discretise,
                shift_params=eval_parameters,
                env_name=env_name,
                num_eps=NUM_EVAL_EPISODES,
            )
        )

    return results


def plot_robustness(seed: int, results: list[dict], env_name: str):
    conf_returns = np.array([res["returns_conf"] for res in results])
    noconf_returns = np.array([res["returns_noconf"] for res in results])
    x_key = EVAL_PARAMETERS[env_name][0]
    xvals = np.array([res[x_key] for res in results])

    # Conformalised returns
    mean_conf = conf_returns.mean(axis=1)
    se_conf = conf_returns.std(axis=1) / np.sqrt(NUM_EVAL_EPISODES)
    plt.errorbar(
        xvals,
        mean_conf,
        yerr=se_conf,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Conformalised",
    )

    # Non-conformalised returns
    mean_no = noconf_returns.mean(axis=1)
    se_no = noconf_returns.std(axis=1) / np.sqrt(NUM_EVAL_EPISODES)
    plt.errorbar(
        xvals,
        mean_no,
        yerr=se_no,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Non-conformalised",
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
    exp_dir = f"results/{env_name}"
    os.makedirs(exp_dir, exist_ok=True)
    plt.savefig(f"{exp_dir}/robustness_experiment_{seed}.png")
    os.makedirs(exp_dir, exist_ok=True)
    plt.close()


def plot_occupancy_histograms(single_exp_result, env_name, seed):
    x_key: str = EVAL_PARAMETERS[env_name][0]
    exp_dir = f"results/{env_name}/hists_{seed}"
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
        # plt.show()


# %%
def main(env_name: str):
    all_results = []
    # cql_loss_weight = 1.0
    cql_loss_weight = None

    # for seed in [5]:
    for seed in range(NUM_EXPERIMENTS):
        single_exp_result = run_single_seed_experiment(
            env_name=env_name, seed=seed, cql_loss_weight=cql_loss_weight
        )
        plot_robustness(seed, single_exp_result, env_name)
        # plot_occupancy_histograms(single_exp_result, env_name, seed)
        all_results.append({"seed": seed, "results": single_exp_result})
        with open(f"results/{env_name}/robustness_experiment.pkl", "wb") as f:
            pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    # envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    env = "CartPole-v1"
    if env == "MountainCar-v0":
        assert N_TRAIN_EPISODES == 120_000
    elif env == "Acrobot-v1":
        assert N_TRAIN_EPISODES == 100_000
    results = main(env)
# %%

# # plot mean occupancy line chart
# exp_results = results[0]["results"]
# for conf in ["visits_conf", "visits_noconf"]:
#     mean_occupancies = []
#     for ix in range(len(exp_results)):
#         nominal_visits = exp_results[ix][conf]
#         visits = []
#         for ep in range(len(nominal_visits)):
#             visits.extend(nominal_visits[ep])
#         mean_occupancy = np.mean(visits)
#         mean_occupancies.append(mean_occupancy)

#     plt.plot(EVAL_PARAMETERS[env][1], mean_occupancies, marker="o", label=conf)
# plt.legend()
# plt.title("Occupancy rates")
# plt.grid(linestyle="--", alpha=0.3)
# despine(plt.gca())
# plt.show()

# %%
