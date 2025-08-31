"""
Command-line entry point for traintime_robustness.

Examples:
  # Vanilla DQN on CartPole with defaults
  python cli.py

  # CQL on Acrobot with custom steps and fewer seeds
  python cli.py --env-name Acrobot-v1 --agent-type cql --cql-alpha 0.5 \
               --n-train-steps 50000 --num-experiments 5

  # Run twice without overwriting by changing results directory name
  python cli.py --env-name CartPole-v1 --results-out cartpole_run1
  python cli.py --env-name CartPole-v1 --results-out cartpole_run2
"""

from __future__ import annotations

import argparse
from importlib import import_module

from traintime_robustness import (
    RobustnessConfig,
    EVAL_PARAMETERS,
    main as run_main,
)


def parse_args() -> argparse.Namespace:
    defaults = RobustnessConfig()
    parser = argparse.ArgumentParser(description="Run robustness experiments.")

    # Environment selection
    parser.add_argument(
        "--env-name",
        default="CartPole-v1",
        choices=sorted(EVAL_PARAMETERS.keys()),
        help="Environment to evaluate.",
    )

    # Core config mirrors RobustnessConfig
    parser.add_argument("--alpha-disc", "--alpha", type=float, default=defaults.alpha_disc)
    parser.add_argument("--alpha-nn", type=float, default=defaults.alpha_nn)
    parser.add_argument("--cql-alpha", type=float, default=defaults.cql_alpha)
    parser.add_argument("--min-calib", type=int, default=defaults.min_calib)
    parser.add_argument("--num-experiments", type=int, default=defaults.num_experiments)
    parser.add_argument(
        "--num-eval-episodes", type=int, default=defaults.num_eval_episodes
    )
    parser.add_argument("--n-calib-steps", type=int, default=defaults.n_calib_steps)
    parser.add_argument("--n-train-steps", type=int, default=defaults.n_train_steps)
    parser.add_argument("--k", type=int, default=defaults.k)
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
    parser.add_argument(
        "--score-fn",
        default=getattr(defaults.score_fn, "__name__", "signed_score"),
        help="Function name from crl.cons.calib (e.g., signed_score).",
    )

    retrain_group = parser.add_mutually_exclusive_group()
    retrain_group.add_argument("--retrain", dest="retrain", action="store_true")
    retrain_group.add_argument("--no-retrain", dest="retrain", action="store_false")
    parser.set_defaults(retrain=defaults.retrain)

    parser.add_argument(
        "--calib-methods",
        nargs="+",
        choices=["nocalib", "ccdisc", "ccnn"],
        default=defaults.calib_methods,
    )
    parser.add_argument(
        "--ccnn-max-distance-quantile",
        type=float,
        default=defaults.ccnn_max_distance_quantile,
    )

    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print the resolved config then exit.",
    )

    parser.add_argument(
        "--results-out",
        default=None,
        help="Optional subdirectory name under 'results/' to write outputs. If omitted, uses env name.",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> RobustnessConfig:
    # Resolve score function by name from crl.cons.calib
    calib_mod = import_module("crl.cons.calib")
    score_fn = getattr(calib_mod, args.score_fn)

    return RobustnessConfig(
        alpha_disc=args.alpha_disc,
        alpha_nn=args.alpha_nn,
        cql_alpha=args.cql_alpha,
        min_calib=args.min_calib,
        num_experiments=args.num_experiments,
        num_eval_episodes=args.num_eval_episodes,
        n_calib_steps=args.n_calib_steps,
        n_train_steps=args.n_train_steps,
        k=args.k,
        scoring_method=args.scoring_method,
        agent_type=args.agent_type,
        score_fn=score_fn,
        retrain=args.retrain,
        calib_methods=list(args.calib_methods),
        ccnn_max_distance_quantile=args.ccnn_max_distance_quantile,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)

    if args.print_config_only:
        # Lightweight JSON print without extra deps
        import json

        out = {
            "env_name": args.env_name,
            "results_out": args.results_out,
            **{
                k: getattr(cfg, k)
                for k in [
                    "alpha_disc",
                    "alpha_nn",
                    "cql_alpha",
                    "min_calib",
                    "num_experiments",
                    "num_eval_episodes",
                    "n_calib_steps",
                    "n_train_steps",
                    "k",
                    "scoring_method",
                    "agent_type",
                    "retrain",
                    "calib_methods",
                    "ccnn_max_distance_quantile",
                ]
            },
            "score_fn": getattr(cfg.score_fn, "__name__", str(cfg.score_fn)),
        }
        print(json.dumps(out, indent=2, sort_keys=False))
        return

    run_main(env_name=args.env_name, config=cfg, results_out=args.results_out)


if __name__ == "__main__":
    main()
