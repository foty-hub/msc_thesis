import pytest
import numpy as np
import gymnasium as gym

from crl.envs.frozen_lake import ContinuousSlipWrapper, make_env, _PERP


@pytest.fixture
def base_env():
    """Pytest fixture for a base, non-slippery FrozenLake environment."""
    return gym.make("FrozenLake-v1", is_slippery=False)


def test_wrapper_init(base_env):
    """Tests that the wrapper can be initialized with different slip_prob types."""
    # Test with a float
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=0.5)
    assert wrapper.slip_prob == 0.5
    assert wrapper.schedule is None

    # Test with a list
    schedule = [0.1, 0.2, 0.3]
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=schedule)
    assert wrapper.slip_prob == schedule[0]
    assert wrapper.schedule == schedule

    # Test with a numpy array
    schedule_np = np.array([0.4, 0.5, 0.6])
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=schedule_np)
    assert wrapper.slip_prob == schedule_np[0]
    assert np.array_equal(wrapper.schedule, schedule_np)  # type: ignore


def test_slip_prob_scheduling(base_env):
    """Tests that the slip_prob scheduling is working as advertised."""
    schedule = [0.1, 0.5, 0.9]
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=schedule)

    # Episode 0
    _, info = wrapper.reset()
    assert info["slip_prob"] == schedule[0]
    assert wrapper.slip_prob == schedule[0]

    # Episode 1
    _, info = wrapper.reset()
    assert info["slip_prob"] == schedule[1]
    assert wrapper.slip_prob == schedule[1]

    # Episode 2
    _, info = wrapper.reset()
    assert info["slip_prob"] == schedule[2]
    assert wrapper.slip_prob == schedule[2]

    # Episode 3 (should hold the last value)
    _, info = wrapper.reset()
    assert info["slip_prob"] == schedule[2]
    assert wrapper.slip_prob == schedule[2]


def test_no_slip_with_zero_prob(base_env):
    """Tests that no slip occurs when slip_prob is 0."""
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=0.0)
    wrapper.reset()
    for action in range(4):
        eff_action, slipped, _ = wrapper._maybe_slip(action)
        assert not slipped
        assert eff_action == action


def test_guaranteed_slip_with_one_prob(base_env):
    """Tests that a slip always occurs when slip_prob is 1."""
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=1.0)
    wrapper.reset()
    for action in range(4):
        eff_action, slipped, _ = wrapper._maybe_slip(action)
        assert slipped
        assert eff_action in _PERP[action]


def test_stochasticity_with_seed(base_env):
    """Tests that the wrapper is not stochastic given a random seed."""
    seed = 42

    # First run
    wrapper1 = ContinuousSlipWrapper(
        base_env, slip_prob=0.5, rng=np.random.default_rng(seed)
    )
    wrapper1.reset()
    actions1 = [wrapper1._maybe_slip(0)[0] for _ in range(10)]

    # Second run
    wrapper2 = ContinuousSlipWrapper(
        base_env, slip_prob=0.5, rng=np.random.default_rng(seed)
    )
    wrapper2.reset()
    actions2 = [wrapper2._maybe_slip(0)[0] for _ in range(10)]

    assert actions1 == actions2


def test_info_dict_updates(base_env):
    """Tests that the info dictionary is correctly updated."""
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=0.5)
    obs, info = wrapper.reset()

    assert "prob" in info
    assert "slipped" in info
    assert "effective_action" in info
    assert "slip_prob" in info
    assert "episode" in info
    assert info["episode"] == 0

    action = 0
    _, _, _, _, info = wrapper.step(action)

    assert "prob" in info
    assert "slipped" in info
    assert "effective_action" in info
    assert "slip_prob" in info
    assert "episode" in info


def test_set_slip_prob(base_env):
    """Tests that the slip probability can be manually set."""
    wrapper = ContinuousSlipWrapper(base_env, slip_prob=0.1)
    wrapper.reset()

    new_prob = 0.8
    wrapper.set_slip_prob(new_prob)

    assert wrapper.slip_prob == new_prob
    assert wrapper.schedule is None

    # Check that the new probability is used
    _, info = wrapper.reset()
    assert info["slip_prob"] == new_prob


def test_make_env():
    """Tests the make_env function."""
    seed = 123
    env_thunk = make_env(seed)
    env = env_thunk()

    assert isinstance(env, ContinuousSlipWrapper)
    assert env.slip_prob == 0.15

    # Test that the seed is used
    env1 = make_env(seed)()
    env1.reset()
    actions1 = [env1._maybe_slip(0)[0] for _ in range(10)]

    env2 = make_env(seed)()
    env2.reset()
    actions2 = [env2._maybe_slip(0)[0] for _ in range(10)]

    assert actions1 == actions2
