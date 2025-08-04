# %%
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import torch
import numpy as np
import gymnasium as gym
import pickle

from typing import Callable, Any
from tqdm import tqdm
from stable_baselines3 import A2C, DQN

from buffer import ReplayBuffer
from collections import deque
from discretise import build_grid_tiling

# fmt: off
DISCOUNT = 0.99             # Gamma/discount factor for the DQN
STATE_BINS = [6, 6, 6, 6]   # num. bins per dimension for coarse grid tiling
ALPHA = 0.1                 # Conformal prediction miscoverage level
MIN_CALIB = 50              # Minimum threshold for a calibration set to be leveraged
NUM_EXPERIMENTS = 25
NUM_EVAL_EPISODES=250
# fmt: on


def learn_policy(seed: int = 0, discount=0.99) -> tuple[DQN, VecEnv]:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    dqn_args = {
        "policy": "MlpPolicy",
        "learning_rate": 2.3e-3,
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "gamma": discount,
        "target_update_interval": 10,
        "train_freq": 256,
        "gradient_steps": 128,
        "exploration_fraction": 0.16,
        "exploration_final_eps": 0.04,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }

    model = DQN(env=env, seed=seed, **dqn_args)

    model.learn(total_timesteps=50_000, progress_bar=True)
    vec_env = model.get_env()
    return model, vec_env


def build_tiling(model: DQN, vec_env: VecEnv):
    stats = run_test_episodes(model, vec_env)

    # compute mins and maxes of the tiling according to the quantiles of the distribution
    obs_quantile = 0.1  # 95% coverage
    maxs, mins = compute_bin_ranges(stats, obs_quantile)

    num_bins = np.array(STATE_BINS)  # Different bin count per dimension
    discretise, n_discrete_states = build_grid_tiling(mins, maxs, num_bins)
    return discretise, n_discrete_states


def compute_bin_ranges(
    stats: list[dict[str, str | list]],
    obs_quantile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    D = len(STATE_BINS)  # dimensionality of the state space
    maxs = np.zeros(D)
    mins = np.zeros(D)
    for dim in range(D):
        vals = stats[dim]["vals"]
        mins[dim], maxs[dim] = (
            np.quantile(vals, obs_quantile),
            np.quantile(vals, 1 - obs_quantile),
        )

    return maxs, mins


def run_test_episodes(model: DQN, vec_env: VecEnv):
    stats = [
        {"label": "position", "vals": []},
        {"label": "velocity", "vals": []},
        {"label": "angle", "vals": []},
        {"label": "angular_velocity", "vals": []},
    ]
    for episode in range(50):
        obs = vec_env.reset()
        for _id in range(4):
            stats[_id]["vals"].append(obs[0, _id])

        for t in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            for _id in range(4):
                stats[_id]["vals"].append(obs[0, _id])
            if done:
                break

    return stats


def collect_transitions(
    model: DQN,
    env: VecEnv,
    n_transitions: int = 100_000,
):
    "run episodes and record SARSA transitions to a replay buffer"
    # TODO: this could stop recording transitions more intelligently if it just updated calib_sets as it went.
    buffer = ReplayBuffer(capacity=n_transitions)
    obs = env.reset()
    for _ in range(n_transitions):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        # Next action required for SARSA
        next_action, _ = model.predict(next_obs, deterministic=True)

        # Store transition (s, a, r, s', a', done)
        buffer.push(obs, action, reward, next_obs, next_action, done)

        obs = next_obs
        if done:
            obs = env.reset()  # start new episode

    return buffer


# 1. build up a conformal set and conformalise calib_sets for each state-action pair (essentially measuring Bellman errors)
def score(y_pred, y_true):
    "Note that this is the signed error, *not* the absolute error as is typical. This only penalises overestimation"
    return np.asarray(y_pred) - np.asarray(y_true)


def fill_calib_sets(
    model: DQN,
    buffer: ReplayBuffer,
    discretise: Callable,
    n_discrete_states: int,
    maxlen: int = 500,
):
    calib_sets = {}
    for (sa,) in np.ndindex((n_discrete_states)):
        calib_sets[sa] = dict(
            y_preds=deque(maxlen=maxlen),
            y_trues=deque(maxlen=maxlen),
            scores=deque(maxlen=maxlen),
        )

    # now construct the calibration sets
    for transition in buffer[:-1]:
        # extract a transition and add it to the calibration set
        with torch.no_grad():
            q_pred = model.q_net(model.policy.obs_to_tensor(transition.state)[0])
            y_pred = q_pred[0, transition.action]
            # Value of the terminal state is 0 by definition.
            if transition.done:
                y_true = transition.reward
            else:
                q_true = model.q_net(
                    model.policy.obs_to_tensor(transition.next_state)[0]
                )
                y_true = (
                    transition.reward[0] + DISCOUNT * q_true[0, transition.next_action]
                )

            obs_disc = discretise(transition.state, transition.action)

            calib_sets[obs_disc]["y_preds"].append(y_pred)
            calib_sets[obs_disc]["y_trues"].append(y_true)
            calib_sets[obs_disc]["scores"].append(score(y_pred, y_true))

    return calib_sets


def compute_lower_bounds(calib_sets):
    n_calibs = []
    qhat_global = 0

    for sa, reg in calib_sets.items():
        n_calib = len(reg["y_preds"])
        n_calibs.append(n_calib)
        if n_calib < MIN_CALIB:
            continue

        # conformalise
        q_level = min(1.0, np.ceil((n_calib + 1) * (1 - ALPHA)) / n_calib)
        qhat = np.quantile(reg["scores"], q_level, method="higher")

        reg["qhat"] = qhat
        # Set a global, pessimistic correction for un-visited state action pairs.
        if qhat > qhat_global:
            qhat_global = qhat

    calib_sets["fallback"] = qhat_global
    return calib_sets, n_calibs


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


def instantiate_eval_env(length, masscart):
    # attrs = ['gravity', 'force_mag', 'masscart', 'masspole', 'length', 'polemass_length']
    eval_env = gym.make("CartPole-v1")
    eval_env.unwrapped.masscart = masscart  # default 1.0
    eval_env.unwrapped.length = length  # default 0.5
    eval_vec_env = DQN("MlpPolicy", env=eval_env).get_env()
    return eval_vec_env


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
    model, vec_env = learn_policy(seed=seed, discount=DISCOUNT)
    discretise, n_discrete_states = build_tiling(model, vec_env)
    buffer = collect_transitions(model, vec_env, n_transitions=500_000)
    calib_sets = fill_calib_sets(model, buffer, discretise, n_discrete_states)
    calib_sets, n_calibs = compute_lower_bounds(calib_sets)

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
