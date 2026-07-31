from types import SimpleNamespace

import numpy as np
import torch

from crl.buffer import ReplayBuffer
from crl.calib import (
    compute_corrections,
    compute_mc_returns,
    compute_td_scores,
    correction_for,
    fill_calib_sets_td,
)


class TensorPolicy:
    @staticmethod
    def obs_to_tensor(observation):
        return torch.as_tensor(observation, dtype=torch.float32), None


class IdentityQ(torch.nn.Module):
    def forward(self, observations):
        return observations


def make_fake_model(gamma: float = 0.5):
    return SimpleNamespace(
        gamma=gamma,
        policy=TensorPolicy(),
        q_net=IdentityQ(),
    )


def make_score_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=2)
    buffer.push(
        np.asarray([[4.0, 1.0]]),
        np.asarray([0]),
        np.asarray([1.0]),
        np.asarray([[2.0, 3.0]]),
        np.asarray([1]),
        np.asarray([False]),
    )
    buffer.push(
        np.asarray([[0.0, 5.0]]),
        np.asarray([1]),
        np.asarray([2.0]),
        np.asarray([[9.0, 9.0]]),
        np.asarray([0]),
        np.asarray([True]),
    )
    return buffer


def test_td_scores_match_one_step_sarsa_targets():
    scores = compute_td_scores(make_fake_model(), make_score_buffer())
    np.testing.assert_allclose(scores, np.asarray([1.5, 3.0]))


def test_mc_returns_reset_at_episode_boundaries():
    buffer = ReplayBuffer(capacity=3)
    for reward, done in [(1.0, False), (2.0, True), (3.0, False)]:
        buffer.push(
            np.asarray([[0.0, 0.0]]),
            np.asarray([0]),
            np.asarray([reward]),
            np.asarray([[0.0, 0.0]]),
            np.asarray([0]),
            np.asarray([done]),
        )
    np.testing.assert_allclose(
        compute_mc_returns(buffer, gamma=0.5),
        np.asarray([2.0, 2.0, 3.0]),
    )


def test_calibration_groups_scores_by_sparse_cell():
    calibration_sets = fill_calib_sets_td(
        make_fake_model(),
        make_score_buffer(),
        lambda _states, actions: np.asarray(actions),
        maxlen=10,
    )
    assert list(calibration_sets[0]["scores"]) == [np.float32(1.5)]
    assert list(calibration_sets[1]["scores"]) == [np.float32(3.0)]


def test_corrections_use_finite_sample_quantile_and_sparse_fallback():
    calibration_sets = {
        2: {"scores": [np.float32(value) for value in [1.0, 2.0, 3.0]]},
        5: {"scores": [np.float32(-2.0), np.float32(-1.0)]},
    }
    corrections = compute_corrections(
        calibration_sets,
        alpha=0.25,
        min_calib=3,
    )
    assert corrections == {2: 3.0, "fallback": 3.0}
    assert (
        correction_for(
            np.asarray([0.0]),
            0,
            corrections,
            lambda _state, _action: np.asarray([999]),
        )
        == 3.0
    )
