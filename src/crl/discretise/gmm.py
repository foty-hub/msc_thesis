import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.dqn import DQN
from crl.cons.discretise.grid import run_test_episodes, compute_bin_ranges
from sklearn.mixture import GaussianMixture


def _scale_with_ranges(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Scale features to [0, 1] using provided per-dimension mins/maxs.
    Values are clipped to [0, 1]. Handles zero-range dimensions safely.
    Parameters
    ----------
    X : (N, D) or (D,) array
    mins, maxs : (D,) arrays
    """
    X = np.asarray(X, dtype=float)
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    Z = (X - mins) / denom
    return np.clip(Z, 0.0, 1.0)


def _select_gmm_by_ic(
    Xs: np.ndarray,
    k_min: int,
    k_max: int,
    *,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 500,
    n_init: int = 1,
    random_state: int | None = None,
    use_bic: bool = True,
) -> GaussianMixture:
    """Fit GMMs for K in [k_min, k_max] and select the one with lowest BIC/AIC."""
    best_model = None
    best_score = np.inf
    for k in range(max(1, k_min), max(1, k_max) + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        gm.fit(Xs)
        score = gm.bic(Xs) if use_bic else gm.aic(Xs)
        if score < best_score:
            best_score = score
            best_model = gm
    return best_model


def build_cluster_partition(
    model: DQN,
    vec_env: VecEnv,
    *,
    k_min: int = 1,
    k_max: int = 12,
    obs_quantile: float = 0.1,
    scale_bins: int = 6,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 500,
    n_init: int = 1,
    anomaly_frac: float = 0.01,
    random_state: int | None = None,
    use_bic: bool = True,
):
    """
    Cluster-based discretisation with an explicit anomaly bucket.

    Training:
      - Build X from `stats` (rollout with the provided model/env).
      - Scale features to [0,1] using quantile-based mins/maxs from `compute_bin_ranges`.
      - Fit a GMM; pick K by BIC (default) or AIC over K in [k_min, k_max].
      - Set anomaly threshold as the `anomaly_frac` quantile of in-sample log-likelihoods.

    Runtime mapping:
      - Scale obs with stored mins/maxs and clip to [0,1].
      - If log-likelihood < threshold ⇒ anomaly index (shared).
      - Else assign to argmax component.

    Returns
    -------
    discretise : callable
        Maps (obs:(1,D), action:(1,)) → np.array([feature_id]) where
        feature_id = action * (K+1) + cluster_idx (with cluster_idx == K for anomalies).
    n_features : int
        (K+1) * n_actions
    """
    stats = run_test_episodes(model, vec_env)
    dims = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n

    # Build data matrix
    X = np.column_stack([np.asarray(s["vals"]) for s in stats])

    # Quantile-based scaling to [0,1]
    state_bins = [scale_bins] * dims
    maxs, mins = compute_bin_ranges(
        stats, obs_quantile=obs_quantile, state_bins=state_bins
    )
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    Xs = _scale_with_ranges(X, mins, maxs)

    # Select and fit GMM by information criterion
    gmm = _select_gmm_by_ic(
        Xs,
        k_min,
        k_max,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        use_bic=use_bic,
    )

    # Global anomaly threshold to meet the requested false positive rate
    train_ll = gmm.score_samples(Xs)
    tau = np.quantile(train_ll, anomaly_frac)

    n_clusters = int(gmm.n_components)
    n_state_features = n_clusters + 1  # extra slot for anomaly

    def discretise(obs: np.ndarray, action: np.ndarray):
        # Remove spurious batch dim, scale, and gate by likelihood
        x = np.asarray(obs[0], dtype=float)
        a = int(action[0])
        xs = _scale_with_ranges(x, mins, maxs)
        ll = float(gmm.score_samples(xs.reshape(1, -1))[0])
        if ll < tau:
            idx = n_clusters  # anomaly bucket
        else:
            idx = int(gmm.predict(xs.reshape(1, -1))[0])
        return np.array([a * n_state_features + idx], dtype=np.int64)

    # Attach artefacts for inspection/visualisation
    discretise.gmm = gmm
    discretise.threshold = float(tau)
    discretise.mins = mins
    discretise.maxs = maxs
    discretise.n_clusters = n_clusters
    discretise.obs_quantile = obs_quantile
    discretise.scale_bins = scale_bins
    discretise.anomaly_frac = anomaly_frac
    discretise.random_state = random_state

    return discretise, n_state_features * n_actions
