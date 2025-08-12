# %%
import os
import pickle
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from typing import Any, Callable
from stable_baselines3 import DQN
from dataclasses import dataclass
from typing import Sequence

from crl.cons.calib import (
    compute_lower_bounds,
    collect_transitions,
    fill_calib_sets,
    signed_score,
)
from crl.cons.cartpole import instantiate_eval_env, learn_dqn_policy
from crl.cons.discretise import build_tiling, build_tile_coding

# fmt: off
_DISCOUNT = 0.99            # Gamma/discount factor for the DQN
ALPHA = 0.1                 # Conformal prediction miscoverage level
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES=250
N_CALIB_TRANSITIONS=100_000

@dataclass
class ExperimentParams:
    env_name: str
    param: str
    param_vals: Sequence
    state_bins: list[int]
    nominal_value: float
    success_threshold: int = 0
    good_seeds: list[int] | None = None


EXPERIMENTS = [
    ExperimentParams("CartPole-v1", "length", np.linspace(0.1, 2.0, 20), [6] * 4, 0.5),
    ExperimentParams("Acrobot-v1", "LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), [6] * 6, 1.0),
    # ExperimentParams("Acrobot-v1", "LINK_MASS_2", np.linspace(0.5, 2.0, 16), [4] * 6, 1.0),
    ExperimentParams("MountainCar-v0", "gravity", np.linspace(0.0015, 0.0040, 21), [6] * 2, 9.8)
]
# fmt: on

EVAL_PARAMETERS = {
    # CartPole: vary pole length as before
    "CartPole-v1": ("length", np.linspace(0.1, 2.0, 20), [6] * 4),
    # Acrobot: vary link 1 length (0.5x–2.0x of default 1.0)
    "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), [6] * 6),
    # "Acrobot-v1": ("LINK_MASS_1", np.linspace(0.5, 2.0, 16), [4] * 6),
    # MountainCar: vary gravity around default 0.0025
    "MountainCar-v0": ("gravity", np.linspace(0.0015, 0.0040, 21), [6] * 2),
}


def run_eval(
    model: DQN,
    discretise: Callable,
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    calib_sets: dict[dict[str, Any]],
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
        for t in range(500):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()
            qhat_global = calib_sets["fallback"]

            if conformalise:
                # Adjust each action using the conformal prediction lower bound
                for a in range(num_actions):
                    obs_disc = discretise(obs, np.array([a]))
                    qhat = calib_sets[obs_disc].get("qhat", qhat_global)
                    q_vals[a] = q_vals[a] - qhat

            action = q_vals.argmax().numpy().reshape(1)

            obs, reward, done, info = ep_env.step(action)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break
    return episodic_returns


def run_shift_experiment(
    model: DQN,
    calib_sets: dict,
    discretise: Callable,
    env_name: str,
    shift_params: dict,
    num_eps: int = NUM_EVAL_EPISODES,
):
    # instantiate the shifted env
    eval_vec_env = instantiate_eval_env(env_name=env_name, **shift_params)

    # run an experiment with and without the CP lower-bound correction
    returns_conf = run_eval(
        model,
        discretise,
        num_eps=num_eps,
        conformalise=True,
        ep_env=eval_vec_env,
        calib_sets=calib_sets,
    )
    returns_noconf = run_eval(
        model,
        discretise,
        num_eps=num_eps,
        conformalise=False,
        ep_env=eval_vec_env,
        calib_sets=calib_sets,
    )

    exp_result = {
        "returns_conf": returns_conf,
        "returns_noconf": returns_noconf,
        "num_episodes": num_eps,
    }
    exp_result.update(shift_params)
    return exp_result


def run_single_seed_experiment(env_name: str, seed: int):
    # train the nominal policy
    model, vec_env = learn_dqn_policy(
        env_name=env_name,
        seed=seed,
        discount=_DISCOUNT,
        total_timesteps=50_000,
    )
    # discretise the space and collect observations for the calibration sets
    param, param_values, state_bins = EVAL_PARAMETERS[env_name]
    # discretise, n_discrete_states = build_tiling(model, vec_env, state_bins=state_bins)
    discretise, n_discrete_states = build_tile_coding(
        model, vec_env, tiles=state_bins[0], tilings=1
    )
    buffer = collect_transitions(model, vec_env, n_transitions=N_CALIB_TRANSITIONS)
    calib_sets = fill_calib_sets(
        model,
        buffer,
        discretise,
        n_discrete_states,
        discount=_DISCOUNT,
        score=signed_score,
    )
    calib_sets, n_calibs = compute_lower_bounds(
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
                calib_sets,
                discretise,
                shift_params=eval_parameters,
                env_name=env_name,
                num_eps=NUM_EVAL_EPISODES,
            )
        )

    return results


def plot_single_experiment(seed: int, results: list[dict], env_name: str):
    conf_returns = np.array([res["returns_conf"] for res in results])
    noconf_returns = np.array([res["returns_noconf"] for res in results])
    # Determine which shift parameter was swept (the key that's not a return/stat field)
    _ignore_keys = {"returns_conf", "returns_noconf", "num_episodes"}
    shift_keys = [k for k in results[0].keys() if k not in _ignore_keys]
    assert len(shift_keys) == 1, (
        f"Expected a single shift parameter, found: {shift_keys}"
    )
    x_key = shift_keys[0]
    xvals = np.array([res[x_key] for res in results])
    import matplotlib.pyplot as plt
    from crl.utils.graphing import despine

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
        plt.xlim(0, 2.0)
        plt.ylim(0, None)
    despine(plt.gca())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    exp_dir = f"results/{env_name}"
    os.makedirs(exp_dir, exist_ok=True)
    plt.savefig(f"{exp_dir}/robustness_experiment_{seed}.png")
    plt.close()


# %%
def main(env_name: str):
    all_results = []

    # for seed in range(NUM_EXPERIMENTS):
    for seed in range(NUM_EXPERIMENTS):
        single_exp_result = run_single_seed_experiment(env_name=env_name, seed=seed)
        plot_single_experiment(seed, single_exp_result, env_name)
        all_results.append({"seed": seed, "results": single_exp_result})
        with open(f"results/{env_name}/robustness_experiment.pkl", "wb") as f:
            pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    # envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    # for env in envs:
    #     all_results = main(env)
    # main("CartPole-v1")
    results = main("Acrobot-v1")
# %%
