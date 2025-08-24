# %%
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
import os
from typing import Literal
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import yaml
from pathlib import Path


class DDQN(DQN):
    """
    Double DQN (DDQN) implementation: decouple action selection (online net)
    from action evaluation (target net) to reduce overestimation bias.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # bookkeeping
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                # Double DQN target: select action with online network, evaluate with target network
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_target, dim=1, index=next_actions)

                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps


ClassicControl = Literal[
    "CartPole-v1", "Acrobot-v1", "LunarLander-v3", "MountainCar-v0"
]


def instantiate_ddqn(
    env_name: ClassicControl,
    seed: int = 0,
) -> DDQN:
    env = gym.make(env_name, render_mode="rgb_array")

    # using SB3 zoo suggested hyperparameters (path relative to this file)
    config_path = Path(__file__).resolve().parent / "configs" / f"{env_name}.yml"
    with open(config_path, "r") as f:
        dqn_args = yaml.safe_load(f)

    model = DDQN(env=env, seed=seed, **dqn_args)
    return model


def learn_ddqn_policy(
    env_name: ClassicControl,
    seed: int = 0,
    total_timesteps: int = 50_000,
    model_dir: str = "models",
    train_from_scratch: bool = False,
) -> tuple[DDQN, VecEnv]:
    # Path for caching the trained model
    model_dir = os.path.join(model_dir, env_name, "ddqn")
    os.makedirs(model_dir, exist_ok=True)
    model_basename = f"model_{seed}"
    model_path = os.path.join(model_dir, model_basename)

    # Load cached model if available and not training from scratch
    if not train_from_scratch and os.path.exists(model_path + ".zip"):
        print(f"Loading DDQN model: {seed}")
        model = DDQN.load(model_path)
        # Attach a fresh environment so the model is usable immediately
        env = gym.make(env_name, render_mode="rgb_array")
        model.set_env(env)
    else:
        # Train a new model from scratch
        print(f"Learning DDQN from scratch: {seed}")
        model = instantiate_ddqn(env_name, seed)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_path)

    # Retrieve the vectorised environment
    vec_env = model.get_env() if model.get_env() is not None else model.env
    return model, vec_env


# %%
def main():
    import os
    import gymnasium as gym
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor

    # Create log dir
    temp_dir = os.environ["TMPDIR"]  # macos temporary logging dir
    log_dir = f"{temp_dir}/gym/"
    os.makedirs(log_dir, exist_ok=True)

    dqn_args = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "batch_size": 64,
        "learning_starts": 1_000,
        "buffer_size": 100000,
        "train_freq": 256,
        "gamma": 0.99,
        "target_update_interval": 30,  # gradient‑step interval
        "gradient_steps": 128,
        "exploration_fraction": 0.16,
        "exploration_final_eps": 0.04,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }

    for seed in range(25):
        env = gym.make("CartPole-v1")  # No rendering during training
        env = Monitor(env, log_dir)
        agent = DDQN(
            env=env,
            seed=seed,
            **dqn_args,
        )

        timesteps = 50_000
        agent.learn(total_timesteps=timesteps, progress_bar=True)

        mean_r, std_r = evaluate_policy(agent, agent.get_env(), n_eval_episodes=50)
        print(f"DDQN: {mean_r:.1f} ± {std_r:.1f}")

        from stable_baselines3.common import results_plotter

        # Helper from the library
        results_plotter.plot_results(
            [log_dir], timesteps, results_plotter.X_TIMESTEPS, "DDQN"
        )


#     def moving_average(values, window):
#     """
#     Smooth values by doing a moving average
#     :param values: (numpy array)
#     :param window: (int)
#     :return: (numpy array)
#     """
#     weights = np.repeat(1.0, window) / window
#     return np.convolve(values, weights, "valid")


# def plot_results(log_folder, title="Learning Curve"):
#     """
#     plot the results

#     :param log_folder: (str) the save location of the results to plot
#     :param title: (str) the title of the task to plot
#     """
#     x, y = ts2xy(load_results(log_folder), "timesteps")
#     y = moving_average(y, window=50)
#     # Truncate x
#     x = x[len(x) - len(y) :]

#     fig = plt.figure(title)
#     plt.plot(x, y)
#     plt.xlabel("Number of Timesteps")
#     plt.ylabel("Rewards")
#     plt.title(title + " Smoothed")
#     plt.show()


if __name__ == "__main__":
    main()

# %%
