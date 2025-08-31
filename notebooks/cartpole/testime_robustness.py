# %%
import pickle
from copy import deepcopy
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from tqdm import tqdm

from crl.cons.calib import (
    collect_transitions,
    compute_lower_bounds,
    fill_calib_sets,
    unsigned_score,
)
from crl.cons.discretise import build_tiling
from crl.cons.env import instantiate_eval_env, learn_dqn_policy

# fmt: off
_DISCOUNT = 0.99            # Gamma/discount factor for the DQN
STATE_BINS = [6, 6, 6, 6]   # num. bins per dimension for coarse grid tiling
ALPHA = 0.1                 # Conformal prediction miscoverage level
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES=250
ETA = 0.01
# fmt: on


def run_eval(
    model: DQN,
    discretise: Callable[[np.ndarray, np.ndarray], int],
    num_eps: int,
    conformalise: bool,
    ep_env: gym.Env,
    calib_sets: dict[dict[str, Any]],
) -> list[float]:
    # copy calib_sets while leaving the original alone for further runs.
    calib_sets = deepcopy(calib_sets)
    episodic_returns = []

    # angelopoulos update: q_t1 = q_t + eta_t (I[y_true <= y_pred] - alpha)
    for ep in range(num_eps):
        obs = ep_env.reset()
        # initialise per-episode bookkeeping for online q̂ updates
        prev_obs = None
        prev_action = None
        prev_reward = None
        prev_done = None
        prev_raw_q_vals = None
        for t in range(500):
            # --- compute raw Q(s, ·) ------------------------------------------
            raw_q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

            # working copy for action selection
            q_vals = raw_q_vals.clone()

            # discretised keys for conformal adjustment
            obs_disc_0 = discretise(obs, np.array([0]))
            obs_disc_1 = discretise(obs, np.array([1]))

            if conformalise:
                q_vals[0] = q_vals[0] - calib_sets[obs_disc_0]["qhat"]
                q_vals[1] = q_vals[1] - calib_sets[obs_disc_1]["qhat"]

            # pick action
            action_arr = q_vals.argmax().numpy().reshape(1)
            action_idx = int(action_arr.item())

            # step environment
            next_obs, reward, done, info = ep_env.step(action_arr)

            # ------------------- update q̂ for PREVIOUS transition --------------
            if prev_obs is not None:
                # next-state value is 0 for the terminal state
                max_q_next = 0.0 if prev_done else raw_q_vals.max().item()
                target = prev_reward + _DISCOUNT * max_q_next
                q_prev_sa = prev_raw_q_vals[prev_action].item()
                err = 1.0 if target < q_prev_sa else 0.0
                sa_key = discretise(prev_obs, np.array([prev_action]))
                # increment counter for this region
                calib_sets[sa_key]["t"] = calib_sets[sa_key].get("t", 0) + 1
                eta_t = ETA / np.sqrt(calib_sets[sa_key]["t"] + 1)  # update
                calib_sets[sa_key]["qhat"] += eta_t * (err - ALPHA)

            # shift current transition into prev_* buffers
            prev_obs = obs
            prev_action = action_idx
            prev_reward = reward
            prev_done = done
            prev_raw_q_vals = raw_q_vals

            # advance state
            obs = next_obs

            # ----------------------- terminal‑state handling --------------------
            if done:
                sa_key = discretise(prev_obs, np.array([prev_action]))
                q_prev_sa = prev_raw_q_vals[prev_action].item()
                target = prev_reward  # no bootstrap from terminal state
                err = 1.0 if target < q_prev_sa else 0.0
                eta_t = ETA / np.sqrt(t + 1)
                calib_sets[sa_key]["qhat"] += eta_t * (err - ALPHA)

                ep_return = info[0]["episode"]["r"]
                episodic_returns.append(ep_return)
                break
    return episodic_returns


def run_test_adaptation_experiment(
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

    exp_result = {
        "length": length,
        "masscart": masscart,
        "returns_conf": returns_conf,
        "num_episodes": num_eps,
    }
    return exp_result


def run_single_seed_experiment(seed: int):
    # train the nominal policy
    model, vec_env = learn_dqn_policy(
        seed=seed,
        discount=_DISCOUNT,
        total_timesteps=50_000,
    )
    # discretise the space and collect observations for the calibration sets
    discretise, n_discrete_states = build_tiling(model, vec_env, state_bins=STATE_BINS)
    buffer = collect_transitions(model, vec_env, n_transitions=100_000)
    calib_sets = fill_calib_sets(
        model,
        buffer,
        discretise,
        n_discrete_states,
        discount=_DISCOUNT,
        score=unsigned_score,
    )
    calib_sets, n_calibs = compute_lower_bounds(
        calib_sets,
        alpha=ALPHA,
        min_calib=MIN_CALIB,
    )

    # assign the fallback to each state-action pair so they can be individually
    # tuned at test-time.
    for sa, regressor in calib_sets.items():
        if sa == "fallback":
            continue
        if regressor.get("qhat") is None:
            regressor["qhat"] = calib_sets["fallback"]

    # Test agent on shifted environments
    results = []
    for length in (pbar := tqdm(np.linspace(0.1, 2.0, 20))):
        pbar.set_description(f"l={length:.1f}")
        results.append(
            run_test_adaptation_experiment(
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

    plt.ylabel("Episodic Return")
    plt.xlabel("Pole Length")
    plt.axvline(0.5, linestyle="--", c="k", alpha=0.5)
    plt.xlim(0, 2.0)
    plt.ylim(0, None)
    despine(plt.gca())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.savefig(f"results/testtime_adaptation_{seed}.png")
    plt.close()


# %%
def main():
    all_results = []

    for seed in range(NUM_EXPERIMENTS):
        single_exp_result = run_single_seed_experiment(seed=seed)
        plot_single_experiment(seed, single_exp_result)
        all_results.append({"seed": seed, "results": single_exp_result})
        with open("testtime_adaptation.pkl", "wb") as f:
            pickle.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    all_results = main()

# %%
