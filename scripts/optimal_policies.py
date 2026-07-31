"""Train reference policies directly in each shifted environment.

These returns provide the environment-specific denominator for a future
normalised-regret robustness metric. They are reference returns rather than a
claim of global optimality: each value is averaged over independently trained
policies and evaluation episodes.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import pprint
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from crl.agents.cql import CQLDQN
from crl.agents.ddqn import DDQN
from crl.experiment import SHIFT_SPECS
from crl.types import AgentTypes, ClassicControl
from crl.utils.paths import project_root


@dataclass(frozen=True)
class ReferencePolicyConfig:
    env_name: ClassicControl
    agent_type: AgentTypes = "vanilla"
    cql_alpha: float = 0.05
    n_train_steps: int = 50_000
    n_eval_episodes: int = 100
    seeds: tuple[int, ...] = tuple(range(5))
    results_out: str | None = None
    max_workers: int = 4


def make_shifted_env(
    env_name: ClassicControl,
    *,
    seed: int,
    parameter: str,
    value: float,
) -> gym.Env:
    env = gym.make(env_name, render_mode="rgb_array")
    setattr(env.unwrapped, parameter, value)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def load_dqn_args(env_name: ClassicControl) -> dict:
    path = project_root() / "src" / "crl" / "configs" / f"{env_name}.yml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def instantiate_agent(
    agent_type: AgentTypes,
    env: gym.Env,
    *,
    env_name: ClassicControl,
    seed: int,
    cql_alpha: float,
) -> DQN:
    kwargs = load_dqn_args(env_name)
    if agent_type == "ddqn":
        return DDQN(env=env, seed=seed, **kwargs)
    if agent_type == "cql":
        return CQLDQN(env=env, seed=seed, cql_alpha=cql_alpha, **kwargs)
    return DQN(env=env, seed=seed, **kwargs)


def train_and_evaluate_one(
    config: ReferencePolicyConfig,
    parameter: str,
    value: float,
    seed: int,
) -> dict:
    train_env = make_shifted_env(
        config.env_name,
        seed=seed,
        parameter=parameter,
        value=value,
    )
    eval_env = make_shifted_env(
        config.env_name,
        seed=100_000 + seed,
        parameter=parameter,
        value=value,
    )
    try:
        model = instantiate_agent(
            config.agent_type,
            train_env,
            env_name=config.env_name,
            seed=seed,
            cql_alpha=config.cql_alpha,
        )
        model.learn(total_timesteps=config.n_train_steps, progress_bar=False)
        returns, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
    finally:
        train_env.close()
        eval_env.close()

    return {
        "seed": seed,
        "returns": [float(value) for value in returns],
        "episode_lengths": [int(value) for value in episode_lengths],
        "mean_return": float(np.mean(returns)),
        "max_return": float(np.max(returns)),
    }


def summarise_shift(
    parameter: str,
    value: float,
    seed_results: list[dict],
) -> dict:
    ordered = sorted(seed_results, key=lambda result: result["seed"])
    seed_means = np.asarray(
        [result["mean_return"] for result in ordered],
        dtype=float,
    )
    return {
        "parameter": parameter,
        "value": float(value),
        "reference_return": float(seed_means.mean()),
        "between_seed_std": float(seed_means.std(ddof=1))
        if seed_means.size > 1
        else 0.0,
        "seed_results": ordered,
    }


def save_results(
    output_dir: Path,
    config: ReferencePolicyConfig,
    summaries: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reference_policies.pkl").open("wb") as handle:
        pickle.dump(summaries, handle)

    with (output_dir / "reference_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "value",
                "reference_return",
                "between_seed_std",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in writer.fieldnames})

    metadata = {
        **asdict(config),
        "shift": asdict(SHIFT_SPECS[config.env_name]),
        "interpretation": "mean return across independently trained policies",
    }
    with (output_dir / "reference_config.yaml").open("w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def run_reference_training(config: ReferencePolicyConfig) -> list[dict]:
    shift = SHIFT_SPECS[config.env_name]
    output_dir = (
        project_root()
        / "results"
        / (config.results_out or f"{config.env_name}/reference_policies")
    )
    pprint.pprint(asdict(config), sort_dicts=False)

    summaries: list[dict] = []
    workers = min(config.max_workers, len(config.seeds))
    for value in shift.values:
        if workers == 1:
            seed_results = [
                train_and_evaluate_one(config, shift.parameter, value, seed)
                for seed in config.seeds
            ]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        train_and_evaluate_one,
                        config,
                        shift.parameter,
                        value,
                        seed,
                    )
                    for seed in config.seeds
                ]
                seed_results = [future.result() for future in as_completed(futures)]

        summaries.append(summarise_shift(shift.parameter, value, seed_results))
        save_results(output_dir, config, summaries)
    return summaries


def parse_args() -> ReferencePolicyConfig:
    parser = argparse.ArgumentParser(
        description="Train reference policies in every shifted environment."
    )
    parser.add_argument(
        "env",
        choices=[
            "Acrobot-v1",
            "CartPole-v1",
            "LunarLander-v3",
            "MountainCar-v0",
        ],
    )
    parser.add_argument(
        "--agent",
        choices=["vanilla", "ddqn", "cql"],
        default="vanilla",
    )
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--cql-alpha", type=float, default=0.05)
    parser.add_argument("--out")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    return ReferencePolicyConfig(
        env_name=args.env,
        agent_type=args.agent,
        cql_alpha=args.cql_alpha,
        n_train_steps=args.train_steps,
        n_eval_episodes=args.eval_episodes,
        seeds=tuple(args.seeds),
        results_out=args.out,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    run_reference_training(parse_args())
