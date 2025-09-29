# Optimal policies per-shift trainer
#
# Example runs:
# - uv run optimal_policies.py CartPole-v1
# - uv run optimal_policies.py LunarLander-v3 --agent ddqn --seeds 0 1 2 3 --max-workers 2
# - uv run optimal_policies.py MountainCar-v0 --agent cql --cql-alpha 0.05 --train-steps 100000 --eval-episodes 250
# - uv run optimal_policies.py Acrobot-v1 --out results/Acrobot-v1/optimal_policies
#
# Outputs are written to results/{env}/optimal_policies by default.

import os
import pickle
import pprint
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from crl.agents.cql import CQLDQN

# Agents from our library
from crl.agents.ddqn import DDQN
from crl.types import AgentTypes, ClassicControl
from crl.utils.paths import project_root

# Shift grids aligned with traintime_robustness.py
EVAL_PARAMETERS: dict[str, tuple[str, Iterable[float], int, int]] = {
    # (param_name, values, tiles, tilings) — tiles/tilings unused here but kept for parity
    "CartPole-v1": ("length", np.arange(0.1, 3.1, 0.2), 4, 1),
    "Acrobot-v1": ("LINK_LENGTH_1", np.linspace(0.5, 2.0, 16), 6, 1),
    "MountainCar-v0": (
        "gravity",
        np.arange(0.001, 0.005 + 0.00025, 0.00025),
        10,
        1,
    ),
    "LunarLander-v3": ("gravity", np.arange(-16.0, 0.0, 1.0), 4, 1),
}


@dataclass
class OptimalTrainConfig:
    env_name: ClassicControl
    agent_type: AgentTypes = "vanilla"
    cql_alpha: float = 0.05
    n_train_steps: int = 50_000
    n_eval_episodes: int = 200
    seeds: list[int] = field(default_factory=lambda: list(range(5)))
    # results directory; if None, defaults to results/{env_name}/optimal_policies
    results_out: str | None = None
    # execution options
    debug_serial: bool = False
    max_workers: int | None = None


def _make_env_with_shift(
    env_name: ClassicControl, seed: int | None, **kwargs
) -> gym.Env:
    """Create a Gymnasium env and set shifted parameters on unwrapped env."""
    env = gym.make(env_name, render_mode="rgb_array")
    for key, value in kwargs.items():
        if hasattr(env.unwrapped, key):
            setattr(env.unwrapped, key, value)
        else:
            raise ValueError(f"Invalid parameter '{key}' for environment '{env_name}'")
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def _load_dqn_args(env_name: ClassicControl) -> dict:
    """Load SB3 DQN kwargs from our YAML config for the given env."""
    # Resolve from project root: src/crl/configs/{env}.yml
    config_path = project_root() / "src" / "crl" / "configs" / f"{env_name}.yml"
    with open(config_path, "r") as f:
        dqn_args = yaml.safe_load(f)
    return dqn_args


def _instantiate_agent(
    agent_type: AgentTypes,
    env: gym.Env,
    seed: int,
    env_name: ClassicControl,
    cql_alpha: float,
):
    dqn_args = _load_dqn_args(env_name)
    if agent_type == "vanilla":
        model = DQN(env=env, seed=seed, **dqn_args)
    elif agent_type == "ddqn":
        model = DDQN(env=env, seed=seed, **dqn_args)
    elif agent_type == "cql":
        model = CQLDQN(env=env, seed=seed, cql_alpha=cql_alpha, **dqn_args)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    return model


def _fmt_val_for_path(x: float) -> str:
    # avoid dots in filenames: 0.05 -> 0-05, -12.0 -> -12-0
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "-")


def train_and_evaluate_one(
    env_name: ClassicControl,
    param: str,
    param_val: float,
    seed: int,
    cfg: OptimalTrainConfig,
) -> dict:
    """Train an agent on a shifted env and evaluate mean episodic return."""
    # Training env (shifted)
    train_env = _make_env_with_shift(env_name, seed=seed, **{param: param_val})
    model = _instantiate_agent(
        cfg.agent_type, train_env, seed=seed, env_name=env_name, cql_alpha=cfg.cql_alpha
    )
    model.learn(total_timesteps=cfg.n_train_steps, progress_bar=True)

    # Fresh evaluation env to avoid training-time state
    eval_env = _make_env_with_shift(env_name, seed=seed + 1, **{param: param_val})
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg.n_eval_episodes
    )

    # Cleanup
    try:
        train_env.close()
    except Exception:
        pass
    try:
        eval_env.close()
    except Exception:
        pass

    result = {
        "seed": seed,
        "param": param,
        "param_val": float(param_val),
        "mean_return": float(mean_r),
        "std_return": float(std_r),
    }
    return result


def run_optimal_training(cfg: OptimalTrainConfig):
    # Resolve results dir
    out_dir = cfg.results_out or os.path.join(
        "results", cfg.env_name, "optimal_policies"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Param sweep values
    if cfg.env_name not in EVAL_PARAMETERS:
        raise ValueError(f"No EVAL_PARAMETERS configured for env '{cfg.env_name}'.")
    param_name, values, _, _ = EVAL_PARAMETERS[cfg.env_name]
    values = list(values)

    # Save a copy of the config and sweep
    config_dump = {
        "env": cfg.env_name,
        "agent_type": cfg.agent_type,
        "cql_alpha": cfg.cql_alpha if cfg.agent_type == "cql" else None,
        "n_train_steps": cfg.n_train_steps,
        "n_eval_episodes": cfg.n_eval_episodes,
        "seeds": cfg.seeds,
        "param": param_name,
        "values": [float(v) for v in values],
        "debug_serial": cfg.debug_serial,
    }
    print("Optimal policy training configuration:")
    pprint.pprint(config_dump, sort_dicts=False)
    with open(os.path.join(out_dir, "optimal_config.yaml"), "w") as f:
        yaml.safe_dump(config_dump, f, sort_keys=False)

    all_results: list[dict] = []
    best_by_val: list[dict] = []

    # Determine level of parallelism (parallelize across seeds for each shift)
    workers = (
        1
        if cfg.debug_serial
        else min(
            (cfg.max_workers if cfg.max_workers is not None else len(cfg.seeds)),
            os.cpu_count() or 1,
        )
    )

    for val in values:
        seed_results: list[dict] = []
        print(f"Training at shift {param_name}={val:.4f}")
        if workers == 1:
            for seed in cfg.seeds:
                res = train_and_evaluate_one(
                    env_name=cfg.env_name,
                    param=param_name,
                    param_val=float(val),
                    seed=seed,
                    cfg=cfg,
                )
                seed_results.append(res)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        train_and_evaluate_one,
                        cfg.env_name,
                        param_name,
                        float(val),
                        seed,
                        cfg,
                    )
                    for seed in cfg.seeds
                ]
                for fut in as_completed(futures):
                    seed_results.append(fut.result())

        # pick the best mean return across seeds
        best_idx = int(np.argmax([r["mean_return"] for r in seed_results]))
        best = seed_results[best_idx]
        best_by_val.append(
            {
                "param": param_name,
                "param_val": float(val),
                "best_seed": int(best["seed"]),
                "best_mean_return": float(best["mean_return"]),
                "best_std_return": float(best["std_return"]),
                "all_seed_results": seed_results,
            }
        )
        # Persist progressively to avoid loss on long runs
        with open(os.path.join(out_dir, "optimal_per_shift.pkl"), "wb") as f:
            pickle.dump(best_by_val, f)

        all_results.extend(seed_results)

    # Save all raw results and the per-shift best summary
    with open(os.path.join(out_dir, "optimal_all_runs.pkl"), "wb") as f:
        pickle.dump(all_results, f)
    with open(os.path.join(out_dir, "optimal_per_shift.pkl"), "wb") as f:
        pickle.dump(best_by_val, f)

    # Also write a simple TSV for quick inspection
    tsv_path = os.path.join(out_dir, "optimal_summary.tsv")
    with open(tsv_path, "w") as f:
        f.write("param\tvalue\tbest_seed\tbest_mean_return\tbest_std_return\n")
        for row in best_by_val:
            f.write(
                f"{row['param']}\t{row['param_val']}\t{row['best_seed']}\t{row['best_mean_return']:.3f}\t{row['best_std_return']:.3f}\n"
            )

    return best_by_val


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train a policy for each shifted environment value "
            "across multiple seeds and report the best return."
        )
    )
    parser.add_argument("env", type=str, help="Gymnasium env id, e.g., CartPole-v1")
    parser.add_argument(
        "--agent",
        type=str,
        default="vanilla",
        choices=["vanilla", "ddqn", "cql"],
        help="Agent type",
    )
    parser.add_argument(
        "--train-steps", type=int, default=50_000, help="Training timesteps per run"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=200,
        help="Evaluation episodes per trained model",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(5)),
        help="Seed list (space-separated)",
    )
    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=0.05,
        help="CQL alpha (only for agent=cql)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Results directory (defaults to results/{env}/optimal_policies)",
    )
    parser.add_argument(
        "--debug-serial",
        action="store_true",
        help="Reserved for future parallelization toggles; no-op for now.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers (defaults to min(len(seeds), CPU count))",
    )

    args = parser.parse_args()

    cfg = OptimalTrainConfig(
        env_name=args.env,
        agent_type=args.agent,
        cql_alpha=args.cql_alpha,
        n_train_steps=args.train_steps,
        n_eval_episodes=args.eval_episodes,
        seeds=list(args.seeds),
        results_out=args.out,
        debug_serial=bool(args.debug_serial),
        max_workers=args.max_workers,
    )

    run_optimal_training(cfg)


if __name__ == "__main__":
    main()
