# This is just a quick script that calls a single shift experiment to trigger the debugger
from simple_mdp import _run_single_shift_experiment
import numpy as np

NUM_EPISODES = 2_000
# delta_schedule = [0.9] * 1000 + [0.1] * 1000  # same schedule as before
delta_schedule = [0.9] * 500 + np.linspace(0.9, 0.1, 1000).tolist() + [0.1] * 500

# Single run so we can step through it
mean, std = _run_single_shift_experiment(
    delta_schedule,
    num_episodes=NUM_EPISODES,
    n_runs=1,
    agent_kwargs=dict(
        use_conformal_prediction=True,
        calibration_set_size=100,
        cp_valid_actions=[0, 1],
    ),
)

print("Finished run")
