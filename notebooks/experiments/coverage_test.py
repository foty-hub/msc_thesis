# %%
import pickle
from typing import Any, Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import beta
from stable_baselines3 import DQN
from tqdm import tqdm

from crl.agents import learn_dqn_policy
from crl.calib import (
    collect_transitions,
    compute_corrections,
    correction_for,
    fill_calib_sets,
)
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env
from crl.utils.graphing import despine

# %%
DISCOUNT = 0.99
ALPHA = 0.1
NUM_EXPERIMENTS = 25
TILES = 6
TILINGS = 1
MIN_CALIB = 250
MAX_CALIB = 250
EPS_JITTER = 1e-8  # tiny jitter to break potential ties in calibration scores
ENV_NAME = "CartPole-v1"
N_TRANSITIONS = 10_000_000
N_CALIB_EVALS = 100_000

# Environment-specific configuration to avoid CartPole-only assumptions.
# Each entry: (eval_param_name, eval_param_values, tiles, tilings)
# Values lists default to a single nominal value for quick checks.
EVAL_PARAMETERS: dict[str, tuple[str, list[float], int, int]] = {
    # CartPole: vary pole length (nominal 0.5)
    "CartPole-v1": ("length", [0.5], 6, 1),
    # Acrobot: vary link 1 length (nominal 1.0)
    "Acrobot-v1": ("LINK_LENGTH_1", [1.0], 6, 1),
    # MountainCar: vary gravity (nominal 0.0025)
    "MountainCar-v0": ("gravity", [0.0025], 20, 1),
}

# Optional: curated seeds per environment. If not provided, fall back to range(NUM_EXPERIMENTS).
GOOD_SEEDS_BY_ENV: dict[str, list[int]] = {
    # From prior CartPole experiments (threshold ~490 on nominal env)
    "CartPole-v1": [1, 3, 5, 6, 7, 9, 12, 14, 15, 17, 21, 22],
    # Example seeds for Acrobot (can be adjusted as needed)
    "Acrobot-v1": [0, 1, 2, 3, 4, 7, 8, 9],
    # Example seeds for MountainCar (no curation by default)
    "MountainCar-v0": [0, 1, 2, 3, 4, 5, 6, 7],
}


# %%
def measure_coverage(
    vec_env: gym.Env,
    model: DQN,
    discretise: Callable,
    qhats: np.ndarray,
    n_transitions: int = 100_000,
    agg: str = "max",
) -> tuple[float, dict[tuple[int, ...], dict[str, int]]]:
    """
    Measure coverage using the updated conformal correction API.

    Returns overall coverage rate and a per-bin stats dict keyed by the
    discretised feature tuple plus action, containing 'visits' and 'covered'.
    """
    stats: dict[tuple[int, ...], dict[str, int]] = {}
    obs = vec_env.reset()
    covered = 0

    for _ in tqdm(range(n_transitions), desc="Measuring coverage", leave=False):
        with torch.no_grad():
            q_vals = model.q_net(model.policy.obs_to_tensor(obs)[0]).flatten()

        # Greedy action under current Q-values
        action = q_vals.argmax().numpy().reshape(1)

        # Compute conformal correction for this state-action
        correction = correction_for(
            obs,
            int(action[0]),
            qhats,
            discretise,
            agg=agg,
            clip_correction=False,
        )

        # Discretised bin key (feature indices + action)
        feats = discretise(obs, action)
        # Ensure we can use as dict key
        if isinstance(feats, (list, np.ndarray)):
            feats_key = tuple(int(f) for f in feats)
        else:
            feats_key = (int(feats),)
        bin_key = feats_key + (int(action[0]),)
        stats.setdefault(bin_key, {"visits": 0, "covered": 0})
        stats[bin_key]["visits"] += 1

        # Lower bound and observed one-step target
        q_pred = q_vals[action[0]]
        q_lower = q_pred - correction

        next_obs, reward, done, _ = vec_env.step(action)

        if done:
            # Use scalar reward for the single env (VecEnv returns arrays)
            y_true = reward[0]
        else:
            with torch.no_grad():
                next_q_vals = model.q_net(
                    model.policy.obs_to_tensor(next_obs)[0]
                ).flatten()
            next_action = next_q_vals.argmax().numpy().reshape(1)
            y_true = reward[0] + DISCOUNT * next_q_vals[next_action[0]].numpy()

        if y_true >= np.asarray(q_lower):
            stats[bin_key]["covered"] += 1
            covered += 1

        obs = next_obs
        if done:  # if finished, start a new episode
            obs = vec_env.reset()

    coverage_rate = covered / n_transitions
    return coverage_rate, stats


def run_single_seed_experiment(seed: int) -> dict[str, Any]:
    """
    Runs a single experiment for a given seed.
    """
    model, vec_env = learn_dqn_policy(
        env_name=ENV_NAME,
        seed=seed,
        total_timesteps=50_000,
    )

    # Use per-environment discretisation if available
    _, _, tiles_cfg, tilings_cfg = EVAL_PARAMETERS.get(
        ENV_NAME, ("", [], TILES, TILINGS)
    )
    discretise, n_features = build_tile_coding(
        model, vec_env, tiles=tiles_cfg, tilings=tilings_cfg
    )
    buffer = collect_transitions(model, vec_env, n_transitions=N_TRANSITIONS)
    calib_sets = fill_calib_sets(
        model, buffer, discretise, n_features, maxlen=MAX_CALIB
    )
    # Add a tiny symmetric jitter to scores before quantile computation to soften ties
    if EPS_JITTER is not None and EPS_JITTER > 0:
        for reg in calib_sets.values():
            if len(reg["scores"]) > 0:
                arr = np.asarray(reg["scores"], dtype=float)
                arr = arr + np.random.uniform(-EPS_JITTER, EPS_JITTER, size=arr.shape)
                reg["scores"] = arr.tolist()

    qhats, visits = compute_corrections(calib_sets, alpha=ALPHA, min_calib=MIN_CALIB)

    coverage, calib_stats = measure_coverage(
        vec_env, model, discretise, qhats, n_transitions=N_CALIB_EVALS
    )
    print(f"Seed {seed}: Coverage = {coverage:.4f} (target: {1 - ALPHA})")

    shifted_coverages = []
    # Evaluate on nominal parameter setting(s) for the selected environment
    param_name, param_values, _, _ = EVAL_PARAMETERS.get(
        ENV_NAME, ("", [], TILES, TILINGS)
    )
    if param_name and param_values:
        for param_val in param_values:
            eval_env = instantiate_eval_env(
                env_name=ENV_NAME, **{param_name: param_val}
            )
            shifted_coverage, _ = measure_coverage(
                eval_env, model, discretise, qhats, n_transitions=50_000
            )
            shifted_coverages.append(shifted_coverage)
            print(f" {param_name}={param_val}: Coverage = {shifted_coverage:.4f}")
    else:
        # If we don't have a known eval parameter, just reuse the training env
        shifted_coverage, _ = measure_coverage(
            vec_env, model, discretise, qhats, n_transitions=50_000
        )
        shifted_coverages.append(shifted_coverage)
        print(f" Nominal eval: Coverage = {shifted_coverage:.4f}")
    return {
        "seed": seed,
        "coverage": coverage,
        "calib_sets": calib_stats,
        "calib_visits": visits,
        "shifted_coverage": shifted_coverages,
    }


# %%
def main():
    """
    Main function to run experiments for multiple seeds.
    """
    # Choose seeds based on environment curation if available
    seeds = GOOD_SEEDS_BY_ENV.get(ENV_NAME, list(range(NUM_EXPERIMENTS)))
    all_results = []
    for seed in seeds:
        result = run_single_seed_experiment(seed=seed)
        all_results.append(result)
        with open("coverage_experiment.pkl", "wb") as f:
            pickle.dump(all_results, f)

    coverages = [r["coverage"] for r in all_results]
    print(f"\nAverage coverage over {len(seeds)} seeds: {np.mean(coverages):.4f}")
    return all_results


if __name__ == "__main__":
    results = main()

# %%
coverages_fallback = []
coverages_normal = []
visits_fallback = []
visits_normal = []

for res in results:
    calib_visits = res.get("calib_visits")
    for ix, calib_set in res["calib_sets"].items():
        # Test-time visit/coverage for this (s,a) bin
        eval_visited = calib_set.get("visits", 0)
        if not eval_visited:
            continue

        covered = calib_set["covered"]
        coverage = covered / eval_visited
        print(f"({ix}): {coverage:.1%} ({eval_visited} visits)")

        # Map bin key to discrete (s,a) index (first element of feats tuple)
        if isinstance(ix, tuple):
            idx = int(ix[0])
        else:
            idx = int(ix)

        # Classify fallback using CALIBRATION visit counts (from compute_corrections)
        is_fallback = False
        if calib_visits is not None and 0 <= idx < len(calib_visits):
            is_fallback = calib_visits[idx] < MIN_CALIB

        if is_fallback:
            coverages_fallback.append(coverage)
            visits_fallback.append(eval_visited)
        else:
            coverages_normal.append(coverage)
            visits_normal.append(eval_visited)


# %%

coverages_normal = np.array(coverages_normal)
coverages_fallback = np.array(coverages_fallback)
visits_normal = np.array(visits_normal)
visits_fallback = np.array(visits_fallback)

# Use the per-environment discretiser config in output filenames/metadata
_, _, tiles_used, tilings_used = EVAL_PARAMETERS.get(ENV_NAME, ("", [], TILES, TILINGS))
filename = (
    f"results/coverage/coverage_{ENV_NAME}_{tiles_used}_{MIN_CALIB}-{MAX_CALIB}_v2"
)

bins = np.linspace(0.0, 1.0, 200)
fig, ax = plt.subplots(1, 1)
# plot the histograms
vals, bins, _ = ax.hist(
    coverages_normal,
    bins=bins,
    weights=visits_normal,
    label="Calibrated",
    density=False,
    alpha=0.9,
)
vals_fall, _, _ = ax.hist(
    coverages_fallback,
    bins=bins,
    weights=visits_fallback,
    label="Fallback Region",
    density=False,
    alpha=0.9,
)

# plot the theoretical curve
t = np.ceil((1 - ALPHA) * (MAX_CALIB + 1))

dist = beta(t, MAX_CALIB + 1 - t)
xs = np.linspace(0.0, 1, 200)
ys = dist.pdf(xs)
# scale theoretical curve to match
ys = ys / ys.max()
ys = ys * vals.max() * 1.23

ax.plot(xs, ys, label="Theoretical coverage", linestyle="--", c="tab:red")

# plot styling
ax.axvline(1 - ALPHA, linestyle="--", c="k", alpha=0.5)
txt_pos = np.maximum(ys.max(), vals_fall.max()) * 0.95
# ax.text(1 - ALPHA - 0.08, y=txt_pos, s=r"$1-\alpha$")
ax.set_xlim(0.5, 1.05)
# ax.set_title("Histogram of coverage rates, by state-action bin")
despine(ax)

plt.xlabel("Empirical coverage")
plt.ylabel("Count")
plt.gcf().set_size_inches(8, 4)
plt.legend(frameon=False)
plt.savefig(f"{filename}.pdf")

plt.show()
# %%
# visit weighted coverage
print("Occupancy-weighted coverage")
total_visits = visits_fallback.sum() + visits_normal.sum()
weighted_coverage = (coverages_fallback * visits_fallback).sum() + (
    coverages_normal * visits_normal
).sum()
weighted_coverage /= total_visits
fallback_weighted_cov = (
    coverages_fallback * visits_fallback
).sum() / visits_fallback.sum()
normal_weighted_cov = (coverages_normal * visits_normal).sum() / visits_normal.sum()
print(f"Overall weighted coverage: {weighted_coverage:.1%}")
print(f"Fallback weighted coverage: {fallback_weighted_cov}")
print(f"Calibrated weighted coverage: {normal_weighted_cov}")
# %%
exp_results = {
    "visits_normal": visits_normal,
    "visits_fallback": visits_fallback,
    "coverages_normal": coverages_normal,
    "coverages_fallback": coverages_fallback,
    "tiles": tiles_used,
    "tilings": tilings_used,
    "min_calib": MIN_CALIB,
    "env": ENV_NAME,
    "fallback_weighted_cov": fallback_weighted_cov,
    "overall_weighted cov": weighted_coverage,
    "calibrated_weighted cov": normal_weighted_cov,
}

with open(f"{filename}.pkl", "wb") as f:
    pickle.dump(exp_results, f)
# %%
