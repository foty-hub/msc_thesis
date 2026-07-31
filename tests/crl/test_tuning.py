import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import gymnasium as gym
from stable_baselines3 import DQN

from crl.env import NOMINAL_REWARD_THRESHOLDS
from crl.calib import collect_transitions
from crl.tuning import (
    GridTuningCandidate,
    calibrate_candidate,
    choose_shift_indices,
    objective_score,
    partition_seed_scores,
)
from scripts import optuna_tuner


def test_nominal_reward_thresholds_match_project_gymnasium_registry():
    assert NOMINAL_REWARD_THRESHOLDS == {
        "CartPole-v1": 475.0,
        "Acrobot-v1": -100.0,
        "MountainCar-v0": -110.0,
        "LunarLander-v3": 200.0,
    }
    for env_name, threshold in NOMINAL_REWARD_THRESHOLDS.items():
        assert gym.spec(env_name).reward_threshold == threshold


def test_partition_seed_scores_includes_threshold_boundary():
    eligible, excluded = partition_seed_scores(
        {3: 201.0, 5: 200.0, 8: 199.9},
        threshold=200.0,
    )

    assert eligible == (3, 5)
    assert excluded == (8,)


def test_nominal_eligibility_is_cached_by_model_fingerprint(
    tmp_path,
    monkeypatch,
):
    class FakeEnv:
        def close(self):
            pass

    models = {
        0: optuna_tuner.SeedModel(0, "passing", "fingerprint-0", 4),
        1: optuna_tuner.SeedModel(1, "failed", "fingerprint-1", 4),
    }
    evaluation_calls: list[str] = []

    monkeypatch.setattr(
        optuna_tuner,
        "instantiate_eval_env",
        lambda *_args, **_kwargs: FakeEnv(),
    )

    def fake_evaluate(model, *_args, **_kwargs):
        evaluation_calls.append(model)
        return [205.0, 201.0] if model == "passing" else [198.0, 196.0]

    monkeypatch.setattr(optuna_tuner, "evaluate_policy", fake_evaluate)
    config = optuna_tuner.OptunaTuningConfig(
        development_seeds=(0, 1),
        eligibility_eval_episodes=2,
        min_seeds_before_prune=1,
    )
    cache_path = tmp_path / "nominal_returns.pkl"
    report_path = tmp_path / "eligibility.json"

    first = optuna_tuner._prepare_eligibility(
        config,
        models,
        cache_path=cache_path,
        report_path=report_path,
    )
    second = optuna_tuner._prepare_eligibility(
        config,
        models,
        cache_path=cache_path,
        report_path=report_path,
    )

    assert evaluation_calls == ["passing", "failed"]
    assert first["eligible_seeds"] == [0]
    assert first["excluded_seeds"] == [1]
    assert second == first


def test_choose_shift_indices_retains_endpoints_and_nominal():
    values = tuple(float(value) for value in range(-16, 0))

    indices = choose_shift_indices(values, nominal_value=-10.0, count=6)

    assert len(indices) == 6
    assert indices == tuple(sorted(set(indices)))
    assert indices[0] == 0
    assert indices[-1] == len(values) - 1
    assert values.index(-10.0) in indices


def test_objective_score_applies_nominal_and_worst_seed_penalties():
    score = objective_score(
        [10.0, -2.0, 4.0],
        mode="mean_median",
        nominal_deltas=[-5.0, -1.0, -3.0],
        nominal_loss_penalty=0.25,
        worst_seed_loss_penalty=0.10,
        worst_seed_loss_tolerance=1.0,
    )

    assert score == pytest.approx(3.15)


def test_calibrate_candidate_uses_requested_transition_prefix():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        seed=7,
        policy_kwargs={"net_arch": [16]},
    )
    env = model.get_env()
    assert env is not None
    buffer = collect_transitions(model, env, 64)
    candidate = GridTuningCandidate(
        alpha=0.25,
        grid_bins=2,
        min_calib=1,
        n_calib_steps=32,
        obs_quantile=0.1,
    )

    calibration = calibrate_candidate(
        model,
        tuple(buffer),
        n_actions=2,
        candidate=candidate,
        max_calib_per_cell=50,
        scoring_method="td",
        inference_batch_size=64,
    )
    env.close()

    assert calibration.discretiser.n_state_action_cells == 2**4 * 2
    assert calibration.n_visited_cells >= calibration.n_calibrated_cells
    assert calibration.n_calibrated_cells > 0
    assert np.isfinite(calibration.fallback)


def test_optuna_cli_keeps_development_and_validation_seeds_explicit():
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "optuna_tuner.py"),
            "--env-name",
            "LunarLander-v3",
            "--development-seeds",
            "0",
            "1",
            "2",
            "--validation-seeds",
            "25",
            "26",
            "--grid-bins",
            "2",
            "3",
            "4",
            "--num-trials",
            "12",
            "--eligibility-eval-episodes",
            "30",
            "--nominal-reward-threshold",
            "210",
            "--print-config-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(completed.stdout)

    assert config["development_seeds"] == [0, 1, 2]
    assert config["validation_seeds"] == [25, 26]
    assert config["grid_bin_choices"] == [2, 3, 4]
    assert config["num_trials"] == 12
    assert config["objective_mode"] == "mean_median"
    assert config["eligibility_eval_episodes"] == 30
    assert config["nominal_reward_threshold"] == 210.0
