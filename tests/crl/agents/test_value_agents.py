import pytest

from crl.agents import CQLDQN, DDQN


@pytest.mark.parametrize(
    ("agent_class", "extra_kwargs"),
    [
        (DDQN, {}),
        (CQLDQN, {"cql_alpha": 0.05}),
    ],
)
def test_custom_value_agent_can_complete_training_update(
    agent_class,
    extra_kwargs,
):
    model = agent_class(
        "MlpPolicy",
        "CartPole-v1",
        seed=5,
        learning_starts=1,
        buffer_size=16,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [8]},
        **extra_kwargs,
    )

    model.learn(total_timesteps=4)
    model.get_env().close()

    assert model._n_updates > 0
