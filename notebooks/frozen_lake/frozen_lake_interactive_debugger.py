import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from crl.envs.frozen_lake import make_env
from crl.agents.tabular import DynaVAgent, AgentParams
from crl.predictors.tabular import PredictorSAConditioned

ACTION_MAP = {"left": 0, "down": 1, "right": 2, "up": 3}
ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def train_agent_and_capture_trajectories(
    slip_prob: float = 0.15,
    n_episodes: int = 2000,
    start_capture_episode: int = 500,
    epsilon_init: float = 1.0,
    rng: int = 42,
    lr: float = 0.1,
    alpha: float = 0.2,
    start_cp: int = 1500,
):
    """Trains a Dyna-V agent and captures trajectories for episodes after a certain point."""
    env = make_env(seed=rng, slip_prob=slip_prob)()
    params = AgentParams(learning_rate=lr, epsilon=epsilon_init, discount=0.99, rng=rng)
    predictor = PredictorSAConditioned(
        n_states=16, n_actions=4, alpha=alpha, n_calib=100
    )
    agent = DynaVAgent(env, params, predictor)

    trajectories = {}
    value_functions = {}

    for episode in range(n_episodes):
        if episode > 0:
            agent.epsilon = max(
                0, epsilon_init * (1 - (episode - 0) / (n_episodes // 2))
            )
        if episode > start_cp:
            agent.use_predictor = True

        done = False
        state, info = env.reset()

        if episode >= start_capture_episode:
            is_capturing_episode = True
            episode_trajectory = []
        else:
            is_capturing_episode = False

        while not done:
            action = agent.select_action(state)

            if is_capturing_episode:
                wm_predictions = agent.world_model[state, action]
                if agent.use_predictor:
                    conformal_set = agent.predictor.conformalise(
                        wm_predictions, state, action
                    )
                else:
                    conformal_set = []
                episode_trajectory.append(
                    {
                        "state": state,
                        "action": action,
                        "world_model_predictions": wm_predictions.copy(),
                        "conformal_set": conformal_set,
                    }
                )

            state_next, reward, terminated, truncated, info = env.step(action)
            obs = (state, action, reward, state_next)
            agent.observe(obs)  # type: ignore
            state = state_next
            agent.plan()
            done = terminated or truncated

        if is_capturing_episode:
            # capture terminal state
            episode_trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "world_model_predictions": wm_predictions.copy(),
                    "conformal_set": conformal_set,
                }
            )
            trajectories[episode] = episode_trajectory
            value_functions[episode] = agent.V.copy()

    return agent, trajectories, value_functions


def visualize_trajectories(trajectories, value_functions):
    """Creates an interactive plot to view the agent's trajectories."""
    fig, (ax_grid, ax_preds) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.35)

    ax_episode = plt.axes([0.25, 0.15, 0.65, 0.03])  # type: ignore
    ax_step = plt.axes([0.25, 0.05, 0.65, 0.03])  # type: ignore

    episodes = sorted(trajectories.keys())

    episode_slider = Slider(
        ax=ax_episode,
        label="Episode",
        valmin=min(episodes),
        valmax=max(episodes),
        valinit=min(episodes),
        valstep=1,
    )

    step_slider = Slider(
        ax=ax_step,
        label="Step",
        valmin=0,
        valmax=len(trajectories[min(episodes)]) - 1,
        valinit=0,
        valstep=1,
    )

    cbar = None

    def update_step(val):
        nonlocal cbar
        episode = int(episode_slider.val)
        step_data = trajectories[episode][int(step_slider.val)]
        state = step_data["state"]
        action = step_data["action"]
        wm_predictions = step_data["world_model_predictions"]
        conformal_set = step_data["conformal_set"]
        V = value_functions[episode]

        if cbar:
            cbar.remove()

        ax_grid.clear()
        ax_preds.clear()

        ax_grid.set_xticks(np.arange(4))
        ax_grid.set_yticks(np.arange(4))
        ax_grid.set_xticks(np.arange(5) - 0.5, minor=True)
        ax_grid.set_yticks(np.arange(5) - 0.5, minor=True)
        ax_grid.grid(which="minor", color="k", linestyle="-", linewidth=2)
        ax_grid.set_title(f"Episode {episode}, Step {int(step_slider.val)}")

        # # Draw value function heatmap
        im = ax_grid.imshow(
            np.flipud(V.reshape(4, 4)),
            cmap="Greens",
            interpolation="nearest",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        # ax_grid.invert_yaxis()
        cbar = fig.colorbar(im, ax=ax_grid)

        for s in range(16):
            r, c = divmod(s, 4)
            if s in [5, 7, 11, 12]:
                ax_grid.add_patch(
                    plt.Rectangle((c - 0.5, 2.5 - r), 1, 1, color="black")  # type: ignore
                )
            if s == 15:
                ax_grid.add_patch(plt.Rectangle((c - 0.5, 2.5 - r), 1, 1, color="gold"))  # type: ignore

        r, c = divmod(state, 4)
        ax_grid.text(c, 3 - r, "A", ha="center", va="center", color="red", fontsize=20)
        ax_grid.set_title(
            f"Episode {episode}, Step {int(step_slider.val)} - Action: {ACTION_ARROWS[action]}"
        )

        ax_preds.bar(range(16), wm_predictions, color="skyblue")
        ax_preds.set_title("World Model Predictions")
        ax_preds.set_xlabel("Next State")
        ax_preds.set_ylabel("Probability")
        ax_preds.set_xticks(range(16))
        ax_preds.set_xticklabels(range(16), rotation=90)

        for s_next in conformal_set:
            ax_preds.get_children()[s_next].set_color("salmon")

        legend_patches = [
            mpatches.Patch(color="skyblue", label="Prediction"),
            mpatches.Patch(color="salmon", label="Conformal Set"),
        ]
        ax_preds.legend(handles=legend_patches)

        fig.canvas.draw_idle()

    def update_episode(val):
        episode = int(episode_slider.val)
        max_steps = len(trajectories[episode]) - 1
        step_slider.valmax = max_steps
        step_slider.ax.set_xlim(0, max_steps)
        step_slider.set_val(0)
        update_step(0)

    episode_slider.on_changed(update_episode)
    step_slider.on_changed(update_step)

    fig.episode_slider = episode_slider  # type: ignore
    fig.step_slider = step_slider  # type: ignore

    update_episode(min(episodes))
    plt.show()


if __name__ == "__main__":
    agent, trajectories, value_functions = train_agent_and_capture_trajectories()
    visualize_trajectories(trajectories, value_functions)
