# cli.py
"""
Command-line front end for traintime_robustness.py

Usage examples:
  # vanilla DQN, default params, CartPole
  python cli.py

  # run CQL on Acrobot with fewer experiments and shorter training
  python cli.py --env-name Acrobot-v1 --agent-type cql --cql-alpha 0.5 \
                --num-experiments 5 --n-train-steps 50000

  # switch scoring & calibration methods
  python cli.py --scoring-method monte_carlo --calib-methods nocalib ccdisc
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module


def parse_args() -> argparse.Namespace:
    # Import here so defaults reflect whatever is in the current file
    tr = import_module("traintime_robustness")

    parser = argparse.ArgumentParser(
        description="Configure and run robustness experiments."
    )

    # Environment (kept separate because main() takes it as an argument)
    parser.add_argument(
        "--env-name",
        default="CartPole-v1",
        choices=["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3"],
        help="Environment to evaluate.",
    )

    # Top-level constants that can be overridden
    parser.add_argument(
        "--alpha",
        type=float,
        default=tr.ALPHA,
        help="Conformal miscoverage level (0<alpha<1).",
    )
    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=tr.CQL_ALPHA,
        help="CQL regularization strength (only used for agent-type=cql).",
    )
    parser.add_argument(
        "--min-calib",
        type=int,
        default=tr.MIN_CALIB,
        help="Minimum per-cell calibration count.",
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=tr.NUM_EXPERIMENTS,
        help="Number of random seeds to run.",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=tr.NUM_EVAL_EPISODES,
        help="Episodes per shifted setting.",
    )
    parser.add_argument(
        "--n-calib-steps",
        type=int,
        default=tr.N_CALIB_STEPS,
        help="Transitions for calibration buffer collection.",
    )
    parser.add_argument(
        "--n-train-steps",
        type=int,
        default=tr.N_TRAIN_STEPS,
        help="Training steps for the agent.",
    )
    parser.add_argument(
        "--k", type=int, default=tr.K, help="k for CCNN / nearest-neighbor calibration."
    )
    parser.add_argument(
        "--scoring-method",
        choices=["td", "monte_carlo"],
        default=tr.SCORING_METHOD,
        help="How to compute nonconformity scores.",
    )
    parser.add_argument(
        "--agent-type",
        choices=["vanilla", "ddqn", "cql"],
        default=tr.AGENT_TYPE,
        help="Which DQN variant to train.",
    )

    # Score function by name (defaults to signed_score). You can pass any symbol
    # exported by crl.cons.calib (e.g., 'signed_score' if that’s the only one available).
    parser.add_argument(
        "--score-fn",
        default=getattr(tr, "SCORE_FN").__name__
        if hasattr(tr, "SCORE_FN")
        else "signed_score",
        help="Score function symbol from crl.cons.calib (default: signed_score).",
    )

    # Boolean toggle for retraining from scratch
    retrain_group = parser.add_mutually_exclusive_group()
    retrain_group.add_argument(
        "--retrain",
        dest="retrain",
        action="store_true",
        help="Force training from scratch.",
    )
    retrain_group.add_argument(
        "--no-retrain",
        dest="retrain",
        action="store_false",
        help="Reuse any cached checkpoints if supported by code.",
    )
    parser.set_defaults(retrain=tr.RETRAIN)

    # Which calibration curves to compute/plot
    parser.add_argument(
        "--calib-methods",
        nargs="+",
        choices=["nocalib", "ccdisc", "ccnn"],
        default=tr.CALIB_METHODS,
        help="Subset of calibration methods to include.",
    )

    parser.add_argument(
        "--ccnn-max-distance-quantile",
        type=float,
        default=tr.CCNN_MAX_DISTANCE_QUANTILE,
        help="Max distance quantile for CCNN out-of-support gating (0<q≤1).",
    )

    # Convenience: print the resolved configuration and exit
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print the final config (after overrides) and exit.",
    )

    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    tr = import_module("traintime_robustness")

    # Basic sanity checks
    if not (0.0 < args.alpha < 1.0):
        sys.exit("--alpha must be in (0,1).")
    if not (0.0 < args.ccnn_max_distance_quantile <= 1.0):
        sys.exit("--ccnn-max-distance-quantile must be in (0,1].")
    for name in [
        "num_experiments",
        "num_eval_episodes",
        "n_calib_steps",
        "n_train_steps",
        "min_calib",
        "k",
    ]:
        if getattr(args, name) <= 0:
            sys.exit(f"--{name.replace('_', '-')} must be positive.")

    # Set module-level constants before calling main()
    tr.ALPHA = args.alpha
    tr.CQL_ALPHA = args.cql_alpha
    tr.MIN_CALIB = args.min_calib
    tr.NUM_EXPERIMENTS = args.num_experiments
    tr.NUM_EVAL_EPISODES = args.num_eval_episodes
    tr.N_CALIB_STEPS = args.n_calib_steps
    tr.N_TRAIN_STEPS = args.n_train_steps
    tr.K = args.k
    tr.SCORING_METHOD = args.scoring_method
    tr.AGENT_TYPE = args.agent_type
    tr.RETRAIN = args.retrain
    tr.CALIB_METHODS = args.calib_methods
    tr.CCNN_MAX_DISTANCE_QUANTILE = args.ccnn_max_distance_quantile

    # Resolve score function by name from crl.cons.calib
    try:
        calib_mod = import_module("crl.cons.calib")
        tr.SCORE_FN = getattr(calib_mod, args.score_fn)
    except Exception as e:
        sys.exit(
            f"Could not resolve score function '{args.score_fn}' from crl.cons.calib: {e}"
        )


def dump_config(args: argparse.Namespace) -> None:
    tr = import_module("traintime_robustness")
    cfg = {
        "env_name": args.env_name,
        "ALPHA": tr.ALPHA,
        "CQL_ALPHA": tr.CQL_ALPHA,
        "MIN_CALIB": tr.MIN_CALIB,
        "NUM_EXPERIMENTS": tr.NUM_EXPERIMENTS,
        "NUM_EVAL_EPISODES": tr.NUM_EVAL_EPISODES,
        "N_CALIB_STEPS": tr.N_CALIB_STEPS,
        "N_TRAIN_STEPS": tr.N_TRAIN_STEPS,
        "K": tr.K,
        "SCORING_METHOD": tr.SCORING_METHOD,
        "AGENT_TYPE": tr.AGENT_TYPE,
        "SCORE_FN": getattr(tr.SCORE_FN, "__name__", str(tr.SCORE_FN)),
        "RETRAIN": tr.RETRAIN,
        "CALIB_METHODS": tr.CALIB_METHODS,
        "CCNN_MAX_DISTANCE_QUANTILE": tr.CCNN_MAX_DISTANCE_QUANTILE,
    }
    # Pretty, but without adding a YAML dep here
    import json

    print(json.dumps(cfg, indent=2, sort_keys=False))


def main() -> None:
    args = parse_args()
    apply_overrides(args)

    if args.print_config_only:
        dump_config(args)
        return

    tr = import_module("traintime_robustness")
    # Run with the chosen environment
    tr.main(env_name=args.env_name)


if __name__ == "__main__":
    main()
