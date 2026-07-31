from __future__ import annotations

import argparse
import json
from dataclasses import asdict

if __package__:
    from .traintime_robustness import RobustnessConfig
    from .traintime_robustness import main as run_main
else:
    from traintime_robustness import RobustnessConfig
    from traintime_robustness import main as run_main

from crl.experiment import SHIFT_SPECS


def parse_args() -> argparse.Namespace:
    defaults = RobustnessConfig()
    parser = argparse.ArgumentParser(
        description="Run sparse-grid conformal robustness experiments."
    )
    parser.add_argument(
        "--env-name",
        default="CartPole-v1",
        choices=sorted(SHIFT_SPECS),
    )
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--min-calib", type=int, default=defaults.min_calib)
    parser.add_argument(
        "--max-calib-per-cell",
        type=int,
        default=defaults.max_calib_per_cell,
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=defaults.num_experiments,
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=defaults.num_eval_episodes,
    )
    parser.add_argument("--n-calib-steps", type=int, default=defaults.n_calib_steps)
    parser.add_argument(
        "--n-representation-steps",
        type=int,
        default=defaults.n_representation_steps,
    )
    parser.add_argument(
        "--representation-dims",
        type=int,
        default=defaults.representation_dims,
    )
    parser.add_argument(
        "--representation-method",
        choices=["pca", "input_pca", "q_values"],
        default=defaults.representation_method,
    )
    parser.add_argument("--n-train-steps", type=int)
    parser.add_argument(
        "--obs-quantile",
        type=float,
        default=defaults.obs_quantile,
    )
    parser.add_argument(
        "--grid-bins",
        type=int,
        default=defaults.grid_bins,
        help="Override the environment's default bins per observation dimension.",
    )
    parser.add_argument(
        "--scoring-method",
        choices=["td", "monte_carlo"],
        default=defaults.scoring_method,
    )
    parser.add_argument(
        "--agent-type",
        choices=["vanilla", "ddqn", "cql"],
        default=defaults.agent_type,
    )
    parser.add_argument("--cql-alpha", type=float, default=defaults.cql_alpha)
    parser.add_argument("--max-workers", type=int, default=defaults.max_workers)
    parser.add_argument("--debug-seed", type=int, default=None)
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=defaults.eval_seed_offset,
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--results-out", default=None)
    parser.add_argument("--print-config-only", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> RobustnessConfig:
    return RobustnessConfig(
        alpha=args.alpha,
        min_calib=args.min_calib,
        max_calib_per_cell=args.max_calib_per_cell,
        num_experiments=args.num_experiments,
        num_eval_episodes=args.num_eval_episodes,
        n_calib_steps=(
            args.n_calib_steps
            if args.n_calib_steps is not None
            else 50_000
            if args.env_name == "MinAtar/Breakout-v1"
            else 10_000
        ),
        n_representation_steps=args.n_representation_steps,
        representation_dims=(
            None
            if args.representation_method == "q_values"
            else args.representation_dims
            if args.representation_dims is not None
            else 4
            if args.env_name == "MinAtar/Breakout-v1"
            else None
        ),
        representation_method=args.representation_method,
        n_train_steps=(
            args.n_train_steps
            if args.n_train_steps is not None
            else 500_000
            if args.env_name == "MinAtar/Breakout-v1"
            else 50_000
        ),
        obs_quantile=args.obs_quantile,
        grid_bins=(
            args.grid_bins
            if args.grid_bins is not None
            else 4
            if args.env_name == "MinAtar/Breakout-v1"
            else None
        ),
        scoring_method=args.scoring_method,
        agent_type=args.agent_type,
        cql_alpha=args.cql_alpha,
        retrain=args.retrain,
        max_workers=args.max_workers,
        debug_seed=args.debug_seed,
        eval_seed_offset=args.eval_seed_offset,
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    if args.print_config_only:
        print(
            json.dumps(
                {
                    "env_name": args.env_name,
                    "results_out": args.results_out,
                    **asdict(config),
                },
                indent=2,
            )
        )
        return
    run_main(args.env_name, config=config, results_out=args.results_out)


if __name__ == "__main__":
    main()
