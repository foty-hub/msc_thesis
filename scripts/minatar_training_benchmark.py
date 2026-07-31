"""Compare nominal MinAtar policy performance at two training budgets."""

from __future__ import annotations

import argparse
import csv
import pickle
import time

import numpy as np
import yaml

if __package__:
    from .traintime_robustness import RobustnessConfig, train_agent
else:
    from traintime_robustness import RobustnessConfig, train_agent

from crl.env import MINATAR_BREAKOUT, instantiate_eval_env
from crl.experiment import evaluate_policy
from crl.utils.paths import project_root


def save_readable_results(output_dir, runs: list[dict]) -> None:
    fieldnames = [
        "budget",
        "seed",
        "episodes",
        "mean_return",
        "median_return",
        "min_return",
        "max_return",
        "nonzero_episodes",
        "seconds",
    ]
    with (output_dir / "benchmark.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for run in runs:
            returns = run["returns"]
            writer.writerow(
                {
                    "budget": run["budget"],
                    "seed": run["seed"],
                    "episodes": len(returns),
                    "mean_return": run["mean_return"],
                    "median_return": float(np.median(returns)),
                    "min_return": float(np.min(returns)),
                    "max_return": float(np.max(returns)),
                    "nonzero_episodes": sum(value > 0 for value in returns),
                    "seconds": run["seconds"],
                }
            )


def run_benchmark(
    *,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    n_eval_episodes: int,
    agent_type: str,
    cql_alpha: float,
    retrain: bool,
    results_out: str | None,
) -> dict:
    output_dir = (
        project_root()
        / "results"
        / (results_out or f"{MINATAR_BREAKOUT}/training_benchmark")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / "benchmark.pkl"
    if pickle_path.exists():
        with pickle_path.open("rb") as handle:
            runs = pickle.load(handle)
    else:
        runs = []

    for budget in budgets:
        for seed in seeds:
            existing = [
                run
                for run in runs
                if run["budget"] == budget and run["seed"] == seed
            ]
            if existing and not retrain:
                print(f"{budget} steps, seed {seed}: already complete, skipping")
                continue
            if existing:
                runs = [
                    run
                    for run in runs
                    if run["budget"] != budget or run["seed"] != seed
                ]

            config = RobustnessConfig(
                n_train_steps=budget,
                agent_type=agent_type,
                cql_alpha=cql_alpha,
                retrain=retrain,
            )
            start = time.perf_counter()
            model, training_env = train_agent(MINATAR_BREAKOUT, seed, config)
            training_env.close()

            eval_env = instantiate_eval_env(
                MINATAR_BREAKOUT,
                seed=100_000 + seed,
            )
            returns = evaluate_policy(
                model,
                eval_env,
                n_episodes=n_eval_episodes,
                calibration=None,
            )
            eval_env.close()
            runs.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "returns": returns,
                    "mean_return": float(np.mean(returns)),
                    "seconds": time.perf_counter() - start,
                }
            )
            runs.sort(key=lambda run: (run["budget"], run["seed"]))
            with pickle_path.open("wb") as handle:
                pickle.dump(runs, handle)
            save_readable_results(output_dir, runs)
            print(
                f"{budget} steps, seed {seed}: "
                f"mean={np.mean(returns):.2f}, "
                f"median={np.median(returns):.2f}, "
                f"range={np.min(returns):.0f}-{np.max(returns):.0f}, "
                f"nonzero={sum(value > 0 for value in returns)}/{len(returns)}"
            )

    completed_budgets = sorted({run["budget"] for run in runs})
    completed_seeds = sorted({run["seed"] for run in runs})
    summary = {}
    for budget in completed_budgets:
        seed_means = [
            run["mean_return"] for run in runs if run["budget"] == budget
        ]
        summary[budget] = {
            "seed_means": seed_means,
            "mean_return": float(np.mean(seed_means)),
            "between_seed_std": (
                float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            ),
        }
    metadata = {
        "env_name": MINATAR_BREAKOUT,
        "budgets": completed_budgets,
        "seeds": completed_seeds,
        "n_eval_episodes": n_eval_episodes,
        "agent_type": agent_type,
        "cql_alpha": cql_alpha,
        "retrain": retrain,
        "summary": summary,
    }
    with (output_dir / "benchmark.yaml").open("w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return {"runs": runs, **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MinAtar Breakout policy training budgets."
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[500_000, 5_000_000],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--agent-type",
        choices=["vanilla", "ddqn", "cql"],
        default="vanilla",
    )
    parser.add_argument("--cql-alpha", type=float, default=0.05)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--results-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        budgets=tuple(args.budgets),
        seeds=tuple(args.seeds),
        n_eval_episodes=args.eval_episodes,
        agent_type=args.agent_type,
        cql_alpha=args.cql_alpha,
        retrain=args.retrain,
        results_out=args.results_out,
    )
    print(yaml.safe_dump(result["summary"], sort_keys=False))


if __name__ == "__main__":
    main()
