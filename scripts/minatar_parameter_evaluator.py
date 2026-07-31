"""Evaluate one MinAtar calibration candidate with reusable baselines.

This runner separates uncalibrated and calibrated evaluation so a high-budget
baseline can be computed once and reused across a bounded parameter search.
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml

if __package__:
    from .traintime_robustness import RobustnessConfig, train_agent
else:
    from traintime_robustness import RobustnessConfig, train_agent

from crl.env import MINATAR_BREAKOUT, instantiate_eval_env
from crl.experiment import (
    SHIFT_SPECS,
    GridCalibrationConfig,
    calibrate_grid_policy,
    evaluate_policy,
)
from crl.utils.paths import project_root


def _evaluate_seed(
    seed: int,
    *,
    mode: str,
    config: dict,
    shift_values: tuple[float, ...],
    n_eval_episodes: int,
    eval_seed_offset: int,
) -> dict:
    torch.set_num_threads(1)
    robustness = RobustnessConfig(
        n_train_steps=config["n_train_steps"],
        agent_type=config["agent_type"],
    )
    model, nominal_env = train_agent(MINATAR_BREAKOUT, seed, robustness)
    calibration = None
    try:
        if mode == "calibrated":
            calibration = calibrate_grid_policy(
                model,
                nominal_env,
                n_bins=config["grid_bins"],
                config=GridCalibrationConfig(
                    n_calib_steps=config["n_calib_steps"],
                    alpha=config["alpha"],
                    min_calib=config["min_calib"],
                    max_calib_per_cell=config["max_calib_per_cell"],
                    obs_quantile=config["obs_quantile"],
                    scoring_method=config["scoring_method"],
                    representation_dim=config["representation_dims"],
                    n_representation_steps=config["n_representation_steps"],
                    representation_method=config["representation_method"],
                ),
            )
    finally:
        nominal_env.close()

    returns = []
    for value in shift_values:
        env = instantiate_eval_env(
            MINATAR_BREAKOUT,
            seed=eval_seed_offset + seed,
            sticky_action_prob=value,
        )
        try:
            returns.append(
                evaluate_policy(
                    model,
                    env,
                    n_episodes=n_eval_episodes,
                    calibration=calibration,
                )
            )
        finally:
            env.close()

    calibration_summary = None
    if calibration is not None:
        calibration_summary = {
            "n_visited_cells": calibration.n_visited_cells,
            "n_calibrated_cells": calibration.n_calibrated_cells,
            "fallback": calibration.fallback,
            "n_state_action_cells": (
                calibration.discretiser.n_state_action_cells
            ),
            "representation_dims": calibration.representation.n_dimensions,
        }
    return {
        "seed": seed,
        "returns": returns,
        "calibration": calibration_summary,
    }


def _summary(
    payload: dict,
    *,
    baseline_payload: dict | None,
) -> dict:
    shift_values = np.asarray(payload["shift_values"], dtype=float)
    nominal = SHIFT_SPECS[MINATAR_BREAKOUT].nominal_value
    nominal_index = int(np.argmin(np.abs(shift_values - nominal)))
    adverse = shift_values >= nominal
    strict_adverse = shift_values > nominal
    severe = shift_values >= 0.30
    candidate_by_seed = {
        row["seed"]: np.asarray(row["returns"], dtype=float).mean(axis=1)
        for row in payload["results"]
    }
    summary = {
        "mode": payload["mode"],
        "seeds": sorted(candidate_by_seed),
        "shift_values": payload["shift_values"],
        "n_eval_episodes": payload["n_eval_episodes"],
        "eval_seed_offset": payload["eval_seed_offset"],
        "mean_returns_by_seed": {
            seed: values.tolist() for seed, values in candidate_by_seed.items()
        },
    }
    if baseline_payload is None:
        summary["nominal_mean_returns"] = {
            seed: float(values[nominal_index])
            for seed, values in candidate_by_seed.items()
        }
        return summary

    baseline_by_seed = {
        row["seed"]: np.asarray(row["returns"], dtype=float).mean(axis=1)
        for row in baseline_payload["results"]
    }
    missing_baselines = set(candidate_by_seed) - set(baseline_by_seed)
    if missing_baselines:
        raise ValueError(
            "Candidate seeds are missing from the baseline results: "
            f"{sorted(missing_baselines)}"
        )
    if baseline_payload["shift_values"] != payload["shift_values"]:
        raise ValueError("Candidate and baseline shift values do not match.")
    if baseline_payload["n_eval_episodes"] != payload["n_eval_episodes"]:
        raise ValueError("Candidate and baseline episode counts do not match.")
    if baseline_payload["eval_seed_offset"] != payload["eval_seed_offset"]:
        raise ValueError("Candidate and baseline evaluation offsets do not match.")

    seed_metrics = []
    for seed in sorted(candidate_by_seed):
        baseline = baseline_by_seed[seed]
        candidate = candidate_by_seed[seed]
        delta = candidate - baseline
        seed_metrics.append(
            {
                "seed": seed,
                "baseline_nominal_return": float(baseline[nominal_index]),
                "candidate_nominal_return": float(candidate[nominal_index]),
                "nominal_delta": float(delta[nominal_index]),
                "full_range_delta": float(delta.mean()),
                "adverse_delta": float(delta[adverse].mean()),
                "strict_adverse_delta": float(
                    delta[strict_adverse].mean()
                ),
                "severe_delta": float(delta[severe].mean()),
            }
        )
    adverse_deltas = np.asarray(
        [row["adverse_delta"] for row in seed_metrics],
        dtype=float,
    )
    nominal_deltas = np.asarray(
        [row["nominal_delta"] for row in seed_metrics],
        dtype=float,
    )
    mean_median = float(
        0.5 * adverse_deltas.mean() + 0.5 * np.median(adverse_deltas)
    )
    nominal_penalty = 0.5 * max(0.0, -float(nominal_deltas.mean()))
    worst_seed_penalty = 0.25 * max(
        0.0,
        -float(adverse_deltas.min()) - 1.0,
    )
    summary.update(
        {
            "seed_metrics": seed_metrics,
            "adverse_mean_delta": float(adverse_deltas.mean()),
            "adverse_median_seed_delta": float(np.median(adverse_deltas)),
            "adverse_seed_wins": int(np.sum(adverse_deltas > 0.0)),
            "nominal_mean_delta": float(nominal_deltas.mean()),
            "worst_seed_adverse_delta": float(adverse_deltas.min()),
            "objective": (
                mean_median - nominal_penalty - worst_seed_penalty
            ),
            "objective_components": {
                "mean_median_adverse": mean_median,
                "nominal_loss_penalty": nominal_penalty,
                "worst_seed_loss_penalty": worst_seed_penalty,
            },
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one cached-baseline MinAtar tuning candidate."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "calibrated"],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-eval-episodes", type=int, default=250)
    parser.add_argument("--eval-seed-offset", type=int, default=100_000)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--results-out", required=True)
    parser.add_argument("--baseline-results")
    parser.add_argument("--n-train-steps", type=int, default=500_000)
    parser.add_argument(
        "--representation-method",
        choices=["pca", "input_pca", "q_values"],
        default="q_values",
    )
    parser.add_argument("--representation-dims", type=int)
    parser.add_argument("--grid-bins", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--min-calib", type=int, default=100)
    parser.add_argument("--max-calib-per-cell", type=int, default=500)
    parser.add_argument("--n-representation-steps", type=int, default=10_000)
    parser.add_argument("--n-calib-steps", type=int, default=50_000)
    parser.add_argument("--obs-quantile", type=float, default=0.1)
    parser.add_argument(
        "--scoring-method",
        choices=["td", "monte_carlo"],
        default="td",
    )
    parser.add_argument(
        "--agent-type",
        choices=["vanilla", "ddqn", "cql"],
        default="vanilla",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.representation_method == "q_values":
        args.representation_dims = None

    shift_values = SHIFT_SPECS[MINATAR_BREAKOUT].values
    config = {
        "n_train_steps": args.n_train_steps,
        "representation_method": args.representation_method,
        "representation_dims": args.representation_dims,
        "grid_bins": args.grid_bins,
        "alpha": args.alpha,
        "min_calib": args.min_calib,
        "max_calib_per_cell": args.max_calib_per_cell,
        "n_representation_steps": args.n_representation_steps,
        "n_calib_steps": args.n_calib_steps,
        "obs_quantile": args.obs_quantile,
        "scoring_method": args.scoring_method,
        "agent_type": args.agent_type,
    }
    output_dir = project_root() / "results" / args.results_out
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_config.yaml").open("w") as handle:
        yaml.safe_dump(
            {
                "environment": MINATAR_BREAKOUT,
                "mode": args.mode,
                "seeds": args.seeds,
                "n_eval_episodes": args.n_eval_episodes,
                "eval_seed_offset": args.eval_seed_offset,
                **config,
            },
            handle,
            sort_keys=False,
        )

    results = []
    with ProcessPoolExecutor(
        max_workers=min(args.max_workers, len(args.seeds))
    ) as executor:
        futures = {
            executor.submit(
                _evaluate_seed,
                seed,
                mode=args.mode,
                config=config,
                shift_values=shift_values,
                n_eval_episodes=args.n_eval_episodes,
                eval_seed_offset=args.eval_seed_offset,
            ): seed
            for seed in args.seeds
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"Completed seed {row['seed']}", flush=True)
    results.sort(key=lambda row: row["seed"])
    payload = {
        "mode": args.mode,
        "config": config,
        "shift_values": list(shift_values),
        "n_eval_episodes": args.n_eval_episodes,
        "eval_seed_offset": args.eval_seed_offset,
        "results": results,
    }
    with (output_dir / "returns.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    baseline_payload = None
    if args.baseline_results is not None:
        baseline_path = Path(args.baseline_results)
        if not baseline_path.is_absolute():
            baseline_path = project_root() / "results" / baseline_path
        with (baseline_path / "returns.pkl").open("rb") as handle:
            baseline_payload = pickle.load(handle)
    summary = _summary(payload, baseline_payload=baseline_payload)
    with (output_dir / "summary.yaml").open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
