import numpy as np
import pytest

from crl.discretise.grid import GridDiscretiser, discretise_observation_grid


def test_grid_discretiser_maps_states_and_actions_to_stable_ids():
    observations = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.25],
            [0.75, 0.75],
            [1.0, 1.0],
        ]
    )
    grid = GridDiscretiser.fit(
        observations,
        n_bins=2,
        n_actions=2,
        obs_quantile=0.0,
    )

    np.testing.assert_array_equal(
        grid.state_ids(np.asarray([[0.0, 0.0], [0.75, 0.75]])),
        np.asarray([0, 3]),
    )
    np.testing.assert_array_equal(
        grid(np.asarray([0.75, 0.75]), np.asarray([0, 1])),
        np.asarray([6, 7]),
    )
    assert grid.n_state_action_cells == 8


def test_grid_clips_out_of_range_observations_to_edge_cells():
    ids = discretise_observation_grid(
        np.asarray([[-10.0], [0.49], [100.0]]),
        mins=np.asarray([0.0]),
        maxs=np.asarray([1.0]),
        num_bins=np.asarray([2]),
    )
    np.testing.assert_array_equal(ids, np.asarray([0, 0, 1]))


def test_grid_fit_handles_constant_dimensions():
    grid = GridDiscretiser.fit(
        np.ones((10, 2)),
        n_bins=4,
        n_actions=2,
    )
    assert np.all(grid.maxs > grid.mins)
    assert grid(np.ones(2), 0).shape == (1,)


def test_grid_rejects_invalid_action():
    grid = GridDiscretiser.fit(
        np.asarray([[0.0], [1.0]]),
        n_bins=2,
        n_actions=2,
        obs_quantile=0.0,
    )
    with pytest.raises(ValueError, match="outside"):
        grid(np.asarray([0.5]), 2)
