# %%
import pickle
from copy import deepcopy
from typing import Any, Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN
from tqdm import tqdm

from crl.cons.calib import collect_transitions, compute_lower_bounds, fill_calib_sets
from crl.cons.discretise import build_tiling
from crl.cons.env import instantiate_eval_env, learn_dqn_policy

DISCOUNT = 0.99
ALPHA = 0.1
NUM_EXPERIMENTS = 25
STATE_BINS = [6, 6, 6, 6]
MIN_CALIB = 50

# from plot.ipynb - filters out seeds where the
# agent gets less than 490 reward on the nominal env.
GOOD_SEEDS = [1, 3, 5, 6, 7, 9, 12, 14, 15, 17, 21, 22]


def measure_coverage(
    vec_env: gym.Env,
    model: DQN,
    discretise: Callable,
    calib_sets: dict[dict[str, Any]],
    n_transitions: int = 100_000,
) -> tuple[float, dict[dict[str, Any], Any]]:
    """
    Measures the coverage of the conformal prediction intervals.
    """
    calib_sets = deepcopy(calib_sets)
    obs = vec_env.reset()
    covered = 0

    for _ in tqdm(range(n_transitions), desc="Measuring coverage", leave=False):
        # Get predicted Q-values and action
        with torch.no_grad():
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()
        action, _ = model.predict(
            obs, deterministic=True
        )  # action is a np.ndarray of shape (1,)

        # Discretise state-action pair to get the calibration set
        obs_disc = discretise(obs, action)
        qhat = calib_sets[obs_disc].get("qhat", calib_sets["fallback"])
        calib_sets[obs_disc]["visits"] = calib_sets[obs_disc].get("visits", 0) + 1

        # Calculate the lower bound of the prediction interval
        q_pred = q_vals[action[0]]
        q_lower = q_pred - qhat

        # Take a step in the environment
        next_obs, reward, done, _ = vec_env.step(action)

        # Calculate the observed one-step Q-value (y_true)
        if done:
            y_true = reward
        else:
            with torch.no_grad():
                next_q_vals = model.q_net(
                    model.policy.obs_to_tensor(next_obs)[0]
                ).flatten()
            next_action, _ = model.predict(next_obs, deterministic=True)
            y_true = reward[0] + DISCOUNT * next_q_vals[next_action[0]].numpy()

        # Check if the true value is within the prediction interval
        if y_true >= np.asarray(q_lower):
            calib_sets[obs_disc]["covered"] = calib_sets[obs_disc].get("covered", 0) + 1
            covered += 1

        # Update state or reset environment
        obs = next_obs
        if done:
            obs = vec_env.reset()

    coverage_rate = covered / n_transitions

    return coverage_rate, calib_sets


def run_single_seed_experiment(seed: int) -> dict[str, Any]:
    """
    Runs a single experiment for a given seed.
    """
    model, vec_env = learn_dqn_policy(
        seed=seed,
        discount=DISCOUNT,
        total_timesteps=50_000,
    )
    discretise, n_discrete_states = build_tiling(model, vec_env, state_bins=STATE_BINS)
    buffer = collect_transitions(model, vec_env, n_transitions=100_000)
    calib_sets = fill_calib_sets(
        model, buffer, discretise, n_discrete_states, discount=DISCOUNT
    )
    calib_sets, _ = compute_lower_bounds(calib_sets, alpha=0.1, min_calib=MIN_CALIB)

    coverage, calib_sets = measure_coverage(vec_env, model, discretise, calib_sets)
    print(f"Seed {seed}: Coverage = {coverage:.4f} (target: {1 - ALPHA})")

    shifted_coverages = []
    for length in np.linspace(0.1, 2.0, 20):
        eval_env = instantiate_eval_env(length=length, masscart=1.0, seed=seed)
        shifted_coverage, _ = measure_coverage(
            eval_env, model, discretise, calib_sets, n_transitions=50_000
        )
        shifted_coverages.append(shifted_coverage)
        print(f" Length {length:.1f}: Coverage = {shifted_coverage:.4f}")
    return {
        "seed": seed,
        "coverage": coverage,
        "calib_sets": calib_sets,
        "shifted_coverage": shifted_coverages,
    }


# %%
def main():
    """
    Main function to run experiments for multiple seeds.
    """
    all_results = []
    for seed in GOOD_SEEDS:
        result = run_single_seed_experiment(seed=seed)
        all_results.append(result)
        with open("coverage_experiment.pkl", "wb") as f:
            pickle.dump(all_results, f)

    coverages = [r["coverage"] for r in all_results]
    print(f"\nAverage coverage over {NUM_EXPERIMENTS} seeds: {np.mean(coverages):.4f}")
    return all_results


if __name__ == "__main__":
    results = main()

# %%
coverages = []
visits = []
for ix, calib_set in results[1]["calib_sets"].items():
    try:
        visited = calib_set.get("covered", False)
    except AttributeError:
        # blows up on the fallback value - just ignore it
        continue

    if visited:
        covered = calib_set["covered"]
        visited = calib_set["visits"]
        coverage = covered / visited
        print(f"({ix}): {coverage:.1%} ({visited} visits)")
        coverages.append(coverage)
        visits.append(visited)

coverages = np.array(coverages)
visits = np.array(visits)

plt.hist(coverages, bins=100)
plt.axvline(1 - ALPHA, linestyle="--", c="k")
plt.text(1 - ALPHA + 0.005, y=140, s=r"$1-\alpha$")
plt.xlim(0.5, 1.05)
plt.title("Histogram of coverage rates, by state-action bin")
# %%
# visit weighted coverage
print((coverages * visits).sum() / visits.sum())
# %%
