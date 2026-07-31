import numpy as np
import torch
from stable_baselines3 import DQN

import crl.experiment as experiment_module
from crl.agents._common import seeded_vec_env
from crl.discretise import GridDiscretiser
from crl.experiment import (
    GridCalibration,
    GridCalibrationConfig,
    calibrate_grid_policy,
    evaluate_policy,
    evaluate_shift,
    fit_input_pca_representation,
    select_action,
)


class TensorPolicy:
    @staticmethod
    def obs_to_tensor(observation):
        return torch.as_tensor(observation, dtype=torch.float32), None


class ConstantQ(torch.nn.Module):
    def forward(self, observations):
        return torch.tensor(
            [[2.0, 1.0]],
            dtype=torch.float32,
            device=observations.device,
        ).repeat(observations.shape[0], 1)


def test_sparse_correction_can_change_the_greedy_action():
    model = type(
        "FakeDQN",
        (),
        {"policy": TensorPolicy(), "q_net": ConstantQ()},
    )()
    grid = GridDiscretiser(
        mins=np.asarray([0.0]),
        maxs=np.asarray([1.0]),
        num_bins=np.asarray([1]),
        n_actions=2,
    )
    calibration = GridCalibration(
        discretiser=grid,
        corrections={0: 1.5, 1: 0.0, "fallback": 1.5},
        n_visited_cells=2,
        n_calibrated_cells=2,
        fallback=1.5,
    )

    raw_action, _raw_q, _raw_corrections = select_action(
        model,
        np.asarray([[0.5]]),
        calibration=None,
    )
    corrected_action, _q, corrections = select_action(
        model,
        np.asarray([[0.5]]),
        calibration=calibration,
    )
    assert raw_action == 0
    assert corrected_action == 1
    np.testing.assert_allclose(corrections, np.asarray([1.5, 0.0]))


def test_cartpole_grid_calibration_pipeline_smoke(monkeypatch):
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        seed=7,
        learning_starts=10,
        buffer_size=100,
        policy_kwargs={"net_arch": [16]},
    )
    nominal_env = model.get_env()
    assert nominal_env is not None

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
        nominal_env,
        n_bins=2,
        config=GridCalibrationConfig(
            n_calib_steps=128,
            alpha=0.2,
            min_calib=1,
            max_calib_per_cell=50,
        ),
    )
    result = evaluate_shift(
        model,
        env_name="CartPole-v1",
        parameter="length",
        value=0.7,
        n_episodes=2,
        eval_seed=123,
        calibration=calibration,
    )
    nominal_env.close()

    assert collection_sizes == [128]
    assert calibration.n_visited_cells >= calibration.n_calibrated_cells
    assert calibration.n_calibrated_cells > 0
    assert len(result["returns_noconf"]) == 2
    assert len(result["returns_conf"]) == 2
    assert np.all(np.isfinite(result["returns_noconf"]))
    assert np.all(np.isfinite(result["returns_conf"]))


def test_seeded_vec_env_reproduces_the_next_reset():
    model = DQN("MlpPolicy", "CartPole-v1", seed=7)

    first = seeded_vec_env(model, 123).reset()
    second = seeded_vec_env(model, 123).reset()
    model.get_env().close()

    np.testing.assert_array_equal(first, second)


def test_evaluation_records_return_at_the_episode_step_limit():
    model = DQN("MlpPolicy", "CartPole-v1", seed=7)
    env = model.get_env()

    returns = evaluate_policy(
        model,
        env,
        n_episodes=2,
        calibration=None,
        max_steps_per_episode=1,
    )
    env.close()

    assert returns == [1.0, 1.0]


def test_input_pca_fit_returns_orthonormal_components():
    observations = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    representation = fit_input_pca_representation(
        observations,
        n_components=2,
    )

    assert representation.components.shape == (2, 3)
    np.testing.assert_allclose(
        representation.components @ representation.components.T,
        np.eye(2),
        atol=1e-12,
    )
