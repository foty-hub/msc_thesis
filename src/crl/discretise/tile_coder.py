import numpy as np
from PyFixedReps import TileCoder, TileCoderConfig
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.dqn import DQN
from tqdm import trange

LR = 1e-2
N_ESTIMATION_EPISODES = 1000


def get_obs_processor(vec_env: VecEnv):
    obs_shape = vec_env.observation_space.shape
    if len(obs_shape) == 3:
        # (N, C, H, W) -> Mean C -> (N, H*W)
        return lambda x: np.mean(x, axis=1).reshape(x.shape[0], -1), np.prod(
            obs_shape[1:]
        )
    return lambda x: x, obs_shape[0]


def build_tile_coding(
    model: DQN, vec_env: VecEnv, tiles: int, tilings: int, obs_quantile: float = 0.1
):
    process_obs, n_dims = get_obs_processor(vec_env)
    quantiles = [obs_quantile, 1.0 - obs_quantile]

    init_obs = vec_env.reset()
    flat_init = process_obs(init_obs).flatten()

    q_estimates = {q: np.copy(flat_init) for q in quantiles}

    for _ in trange(N_ESTIMATION_EPISODES, desc="Test episodes"):
        obs = vec_env.reset()
        # Initial step current_obs
        current_obs = process_obs(obs).flatten()

        dones = np.array([False])
        while not dones[0]:
            for q, est in q_estimates.items():
                # Robbins Monro quantile estimation
                update = LR * (q - (current_obs < est).astype(float))
                q_estimates[q] += update

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            current_obs = process_obs(obs).flatten()

    mins = q_estimates[quantiles[0]]
    maxs = q_estimates[quantiles[1]]

    # FIX: Ensure strict inequality max > min to avoid ZeroDiv
    # If a pixel is static (min==max), we artificially create a tiny range.
    epsilon = 0.1
    maxs = np.maximum(maxs, mins + epsilon)

    input_ranges = list(zip(mins, maxs))

    cfg = TileCoderConfig(
        tiles=tiles,
        tilings=tilings,
        dims=n_dims,
        offset="cascade",
        scale_output=False,
        input_ranges=input_ranges,
    )
    tc = TileCoder(config=cfg)

    n_actions = vec_env.action_space.n
    n_state_features = tc.features()

    def discretise(obs: np.array, action: np.array):
        processed = process_obs(obs)
        state_vals = tc.get_indices(processed[0])
        state_vals = action * n_state_features + state_vals
        return state_vals

    return discretise, n_state_features * n_actions
