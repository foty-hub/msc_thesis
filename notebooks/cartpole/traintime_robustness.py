# %%
import pickle
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from typing import Any
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from crl.cons.calib import compute_lower_bounds, collect_transitions, fill_calib_sets
from crl.cons.cartpole import (
    instantiate_vanilla_dqn,
    instantiate_eval_env,
    learn_policy,
)
from crl.cons.discretise import build_tiling

# fmt: off
_DISCOUNT = 0.99            # Gamma/discount factor for the DQN
STATE_BINS = [6, 6, 6, 6]   # num. bins per dimension for coarse grid tiling
ALPHA = 0.1                 # Conformal prediction miscoverage level
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES=250
# fmt: on


def run_eval(
    model,
    discretise,
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    calib_sets: dict[dict[str, Any]],
) -> list[float]:
    episodic_returns = []

    for ep in range(num_eps):
        obs = ep_env.reset()
        for t in range(500):
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()
            # adjust the action using the conformal prediction lower bound
            obs_disc_0 = discretise(obs, np.array([0]))  # [0]
            obs_disc_1 = discretise(obs, np.array([1]))  # [0]
            qhat_global = calib_sets["fallback"]
            if conformalise:
                q_vals[0] = q_vals[0] - calib_sets[obs_disc_0].get("qhat", qhat_global)
                q_vals[1] = q_vals[1] - calib_sets[obs_disc_1].get("qhat", qhat_global)

            action = q_vals.argmax().numpy().reshape(1)

            obs, reward, done, info = ep_env.step(action)

            if done:
                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break
    return episodic_returns


def run_shift_experiment(
    model,
    calib_sets,
    discretise,
    length: float = 0.5,
    masscart: float = 1.0,
    num_eps: int = NUM_EVAL_EPISODES,
):
    # instantiate the shifted env
    eval_vec_env = instantiate_eval_env(length, masscart)

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
        "length": length,
        "masscart": masscart,
        "returns_conf": returns_conf,
        "returns_noconf": returns_noconf,
        "num_episodes": num_eps,
    }
    return exp_result


def run_single_seed_experiment(seed: int):
    # train the nominal policy
    model = instantiate_vanilla_dqn(seed=seed, discount=_DISCOUNT)
    model, vec_env = learn_policy(model=model, total_timesteps=50_000)
    # discretise the space and collect observations for the calibration sets
    discretise, n_discrete_states = build_tiling(model, vec_env, state_bins=STATE_BINS)
    buffer = collect_transitions(model, vec_env, n_transitions=500_000)
    calib_sets = fill_calib_sets(
        model,
        buffer,
        discretise,
        n_discrete_states,
        discount=_DISCOUNT,
    )
    calib_sets, n_calibs = compute_lower_bounds(
        calib_sets,
        alpha=ALPHA,
        min_calib=MIN_CALIB,
    )

    # Test agent on shifted environments
    results = []
    for length in (pbar := tqdm(np.linspace(0.1, 2.0, 20))):
        pbar.set_description(f"l={length:.1f}")
        results.append(
            run_shift_experiment(
                model,
                calib_sets,
                discretise,
                length=length,
                num_eps=NUM_EVAL_EPISODES,
            )
        )

    return results


def plot_single_experiment(seed: int, results: list[dict]):
    conf_returns = np.array([res["returns_conf"] for res in results])
    noconf_returns = np.array([res["returns_noconf"] for res in results])
    lengths = np.array([res["length"] for res in results])
    import matplotlib.pyplot as plt
    from crl.utils.graphing import despine

    # Conformalised returns
    mean_conf = conf_returns.mean(axis=1)
    se_conf = conf_returns.std(axis=1) / np.sqrt(250)
    plt.errorbar(
        lengths,
        mean_conf,
        yerr=se_conf,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Conformalised",
    )

    # Non-conformalised returns
    mean_no = noconf_returns.mean(axis=1)
    se_no = noconf_returns.std(axis=1) / np.sqrt(250)
    plt.errorbar(
        lengths,
        mean_no,
        yerr=se_no,
        marker="o",
        linestyle="-",
        capsize=4,
        label="Non-conformalised",
    )

    plt.ylabel("Episodic Return")
    plt.xlabel("Pole Length")
    plt.axvline(0.5, linestyle="--", c="k", alpha=0.5)
    plt.xlim(0, 2.0)
    plt.ylim(0, None)
    despine(plt.gca())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.savefig(f"results/robustness_experiment_{seed}.png")
    plt.close()


# %%
def main():
    all_results = []

    for seed in range(NUM_EXPERIMENTS):
        single_exp_result = run_single_seed_experiment(seed=seed)
        plot_single_experiment(seed, single_exp_result)
        all_results.append({"seed": seed, "results": single_exp_result})
        with open("robustness_experiment.pkl", "wb") as f:
            pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    all_results = main()
    # print(f"On average, CP is {np.mean(mean_conf / mean_no):.2f}x better")
# %%
