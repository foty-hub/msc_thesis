import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.dqn import DQN
from PyFixedReps import TileCoder, TileCoderConfig
from crl.cons.discretise.grid import run_test_episodes, compute_bin_ranges


# %%
def build_tile_coding(model: DQN, vec_env: VecEnv, tiles: int, tilings: int):
    stats = run_test_episodes(model, vec_env)
    dims = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n

    state_bins = [tiles] * dims

    # compute mins and maxes of the tiling according to the quantiles of the distribution
    maxs, mins = compute_bin_ranges(stats, obs_quantile=0.1, state_bins=state_bins)
    # rearrange into per-dimension (min, max) tuples for TileCoderConfig
    input_ranges = list(zip(mins, maxs))
    cfg = TileCoderConfig(
        tiles=tiles,
        tilings=tilings,
        dims=dims,
        offset="cascade",
        scale_output=False,
        input_ranges=input_ranges,
    )
    tc = TileCoder(config=cfg)

    n_state_features = tc.features()

    def discretise(obs: np.array, action: np.array):
        # the observation and array come with a spurious batch dimension
        state_vals = tc.get_indices(obs[0])
        # offset so there's a unique id for each action and state pair.
        state_vals = action * n_state_features + state_vals
        return state_vals

    return discretise, n_state_features * n_actions


# %%
