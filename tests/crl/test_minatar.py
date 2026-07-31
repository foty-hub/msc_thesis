import crl.experiment as experiment_module
from crl.agents._common import load_dqn_args, model_basename
from crl.agents.cql import instantiate_cql_dqn
from crl.agents.ddqn import instantiate_ddqn
from crl.agents.dqn import instantiate_vanilla_dqn
from crl.env import MINATAR_BREAKOUT, instantiate_eval_env
from crl.experiment import (
    GridCalibrationConfig,
    calibrate_grid_policy,
    evaluate_shift,
)


def test_minatar_policy_and_sticky_action_shift():
    model = instantiate_vanilla_dqn(
        MINATAR_BREAKOUT,
        seed=7,
        total_timesteps=500_000,
    )
    observation = model.get_env().reset()
    observation_tensor = model.policy.obs_to_tensor(observation)[0]

    assert model.q_net(observation_tensor).shape == (1, 3)
    features = model.q_net.extract_features(
        observation_tensor,
        model.q_net.features_extractor,
    )
    assert features.shape == (1, 128)
    assert load_dqn_args(MINATAR_BREAKOUT, 500_000)["exploration_fraction"] == 0.2
    assert load_dqn_args(MINATAR_BREAKOUT, 5_000_000)["exploration_fraction"] == 0.02
    model.get_env().close()

    shifted_env = instantiate_eval_env(
        MINATAR_BREAKOUT,
        seed=11,
        sticky_action_prob=0.35,
    )
    assert shifted_env.envs[0].unwrapped.game.sticky_action_prob == 0.35
    shifted_env.close()


def test_minatar_uses_separate_representation_and_calibration_rollouts(monkeypatch):
    model = instantiate_vanilla_dqn(
        MINATAR_BREAKOUT,
        seed=3,
        total_timesteps=500_000,
    )
    env = model.get_env()
    collection_sizes = []
    collect_transitions = experiment_module.collect_transitions

    def record_collection_size(model, env, n_transitions):
        collection_sizes.append(n_transitions)
        return collect_transitions(model, env, n_transitions)

    monkeypatch.setattr(
        experiment_module,
        "collect_transitions",
        record_collection_size,
    )
    calibration = calibrate_grid_policy(
        model,
        env,
        n_bins=2,
        config=GridCalibrationConfig(
            n_calib_steps=64,
            n_representation_steps=32,
            representation_dim=2,
            min_calib=1,
            max_calib_per_cell=64,
        ),
    )
    env.close()

    assert collection_sizes == [32, 64]
    assert calibration.representation.components.shape == (2, 128)
    assert calibration.discretiser.n_state_action_cells == 12

    result = evaluate_shift(
        model,
        env_name=MINATAR_BREAKOUT,
        parameter="sticky_action_prob",
        value=0.2,
        n_episodes=1,
        eval_seed=101,
        calibration=calibration,
    )
    assert len(result["returns_noconf"]) == 1
    assert len(result["returns_conf"]) == 1


def test_minatar_can_discretise_policy_q_values():
    model = instantiate_vanilla_dqn(
        MINATAR_BREAKOUT,
        seed=3,
        total_timesteps=500_000,
    )
    env = model.get_env()
    calibration = calibrate_grid_policy(
        model,
        env,
        n_bins=3,
        config=GridCalibrationConfig(
            n_calib_steps=64,
            n_representation_steps=32,
            representation_method="q_values",
            min_calib=1,
            max_calib_per_cell=64,
        ),
    )
    env.close()

    assert calibration.representation.n_dimensions == 3
    assert calibration.discretiser.n_state_action_cells == 81


def test_minatar_can_apply_pca_to_flattened_input_observations():
    model = instantiate_vanilla_dqn(
        MINATAR_BREAKOUT,
        seed=3,
        total_timesteps=500_000,
    )
    env = model.get_env()
    calibration = calibrate_grid_policy(
        model,
        env,
        n_bins=3,
        config=GridCalibrationConfig(
            n_calib_steps=64,
            n_representation_steps=32,
            representation_dim=3,
            representation_method="input_pca",
            min_calib=1,
            max_calib_per_cell=64,
        ),
    )
    env.close()

    assert calibration.representation.components.shape == (3, 400)
    assert calibration.discretiser.n_state_action_cells == 81


def test_minatar_model_cache_names_include_training_budget():
    short = model_basename(MINATAR_BREAKOUT, seed=2, total_timesteps=500_000)
    long = model_basename(MINATAR_BREAKOUT, seed=2, total_timesteps=5_000_000)

    assert short != long
    assert "500000" in short
    assert "5000000" in long


def test_all_value_agents_accept_the_minatar_policy_config():
    models = [
        instantiate_ddqn(
            MINATAR_BREAKOUT,
            seed=1,
            total_timesteps=500_000,
        ),
        instantiate_cql_dqn(
            MINATAR_BREAKOUT,
            seed=1,
            cql_alpha=0.05,
            total_timesteps=500_000,
        ),
    ]

    for model in models:
        observation = model.get_env().reset()
        observation_tensor = model.policy.obs_to_tensor(observation)[0]
        assert model.q_net(observation_tensor).shape == (1, 3)
        model.get_env().close()
