from __future__ import annotations

import os
import pickle
import pprint
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from crl.agents import learn_cqldqn_policy, learn_ddqn_policy, learn_dqn_policy
from crl.calib import signed_score
from crl.env import MINATAR_BREAKOUT
from crl.experiment import (
    SHIFT_SPECS,
    GridCalibrationConfig,
    calibrate_grid_policy,
    evaluate_shift,
)
from crl.types import (
    AgentTypes,
    ClassicControl,
    RepresentationMethod,
    ScoringMethod,
)
from crl.utils.graphing import despine
from crl.utils.paths import project_root


@dataclass(frozen=True)
class RobustnessConfig:
    alpha: float = 0.25
    min_calib: int = 100
    max_calib_per_cell: int = 500
    num_experiments: int = 25
    num_eval_episodes: int = 25
    n_calib_steps: int | None = None
    n_representation_steps: int = 10_000
    representation_dims: int | None = None
    representation_method: RepresentationMethod = "pca"
    n_train_steps: int = 50_000
    obs_quantile: float = 0.1
    grid_bins: int | None = None
    scoring_method: ScoringMethod = "td"
    agent_type: AgentTypes = "vanilla"
    cql_alpha: float = 0.05
    retrain: bool = False
    max_workers: int = 4
    debug_seed: int | None = None
    eval_seed_offset: int = 100_000


def effective_calibration_steps(
    env_name: ClassicControl,
    cfg: RobustnessConfig,
) -> int:
    if cfg.n_calib_steps is not None:
        return cfg.n_calib_steps
    return 50_000 if env_name == MINATAR_BREAKOUT else 10_000


def train_agent(env_name: ClassicControl, seed: int, cfg: RobustnessConfig):
    if cfg.agent_type == "cql":
        return learn_cqldqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
            cql_alpha=cfg.cql_alpha,
            train_from_scratch=cfg.retrain,
        )
    if cfg.agent_type == "ddqn":
        return learn_ddqn_policy(
            env_name=env_name,
            seed=seed,
            total_timesteps=cfg.n_train_steps,
            train_from_scratch=cfg.retrain,
        )
    return learn_dqn_policy(
        env_name=env_name,
        seed=seed,
        total_timesteps=cfg.n_train_steps,
        train_from_scratch=cfg.retrain,
    )


def run_single_seed_experiment(
    env_name: ClassicControl,
    seed: int,
    cfg: RobustnessConfig,
) -> dict:
    experiment_start = time.perf_counter()
    train_start = time.perf_counter()
    model, nominal_env = train_agent(env_name, seed, cfg)
    train_seconds = time.perf_counter() - train_start
    shift_spec = SHIFT_SPECS[env_name]
    grid_bins = shift_spec.grid_bins if cfg.grid_bins is None else cfg.grid_bins
    representation_dims = cfg.representation_dims
    if (
        env_name == MINATAR_BREAKOUT
        and representation_dims is None
        and cfg.representation_method in ("pca", "input_pca")
    ):
        representation_dims = 4

    # Observe the agent and compute all the calibrations
    calibration_start = time.perf_counter()
    calibration = calibrate_grid_policy(
        model,
        nominal_env,
        n_bins=grid_bins,
        config=GridCalibrationConfig(
            n_calib_steps=effective_calibration_steps(env_name, cfg),
            alpha=cfg.alpha,
            min_calib=cfg.min_calib,
            max_calib_per_cell=cfg.max_calib_per_cell,
            obs_quantile=cfg.obs_quantile,
            scoring_method=cfg.scoring_method,
            score_fn=signed_score,
            representation_dim=representation_dims,
            n_representation_steps=cfg.n_representation_steps,
            representation_method=cfg.representation_method,
        ),
    )
    calibration_seconds = time.perf_counter() - calibration_start

    # Now use the calibration to evaluate the agent on unseen distribution shifts
    evaluation_start = time.perf_counter()
    results = [
        evaluate_shift(
            model,
            env_name=env_name,
            parameter=shift_spec.parameter,
            value=value,
            n_episodes=cfg.num_eval_episodes,
            eval_seed=cfg.eval_seed_offset + seed,
            calibration=calibration,
        )
        for value in shift_spec.values
    ]
    evaluation_seconds = time.perf_counter() - evaluation_start
    nominal_env.close()
    return {
        "seed": seed,
        "calibration": {
            "n_visited_cells": calibration.n_visited_cells,
            "n_calibrated_cells": calibration.n_calibrated_cells,
            "fallback": calibration.fallback,
            "n_state_action_cells": calibration.discretiser.n_state_action_cells,
            "representation_dims": (
                None
                if calibration.representation is None
                else calibration.representation.n_dimensions
            ),
            "representation_method": cfg.representation_method,
        },
        "timing_seconds": {
            "train_or_load": train_seconds,
            "calibration": calibration_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - experiment_start,
        },
        "results": results,
    }


def plot_robustness(
    seed_result: dict,
    env_name: ClassicControl,
    out_dir: Path,
) -> None:
    shift_spec = SHIFT_SPECS[env_name]
    results = seed_result["results"]
    x_values = np.asarray([row[shift_spec.parameter] for row in results])

    for key, label, colour in [
        ("returns_noconf", "Uncalibrated", "tab:orange"),
        ("returns_conf", "Grid calibrated", "tab:blue"),
    ]:
        returns = np.asarray([row[key] for row in results], dtype=float)
        plt.plot(x_values, returns.mean(axis=1), marker="o", label=label, color=colour)

    plt.axvline(
        shift_spec.nominal_value,
        linestyle="--",
        color="black",
        alpha=0.5,
        label="Nominal",
    )
    plt.xlabel(shift_spec.parameter)
    plt.ylabel("Episodic return")
    plt.grid(linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    despine(plt.gca())
    plt.tight_layout()
    plt.savefig(out_dir / f"robustness_seed_{seed_result['seed']}.png")
    plt.close()


def main(
    env_name: ClassicControl,
    config: RobustnessConfig | None = None,
    results_out: str | None = None,
) -> list[dict]:
    cfg = config or RobustnessConfig()

    out_dir = project_root() / "results" / (results_out or env_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_info = {
        "env": env_name,
        **asdict(cfg),
        "effective_grid_bins": (
            SHIFT_SPECS[env_name].grid_bins if cfg.grid_bins is None else cfg.grid_bins
        ),
        "effective_n_calib_steps": effective_calibration_steps(env_name, cfg),
        "effective_representation_dims": (
            3
            if (
                env_name == MINATAR_BREAKOUT and cfg.representation_method == "q_values"
            )
            else cfg.representation_dims
            if (
                cfg.representation_dims is not None
                and cfg.representation_method in ("pca", "input_pca")
            )
            else 4
            if (
                env_name == MINATAR_BREAKOUT
                and cfg.representation_method in ("pca", "input_pca")
            )
            else None
        ),
        "effective_representation_method": cfg.representation_method,
        "shift": asdict(SHIFT_SPECS[env_name]),
        "score_fn": "signed_score",
        "calibration_method": "sparse_grid",
    }
    pprint.pprint(experiment_info, sort_dicts=False)
    with (out_dir / "experiment_config.yaml").open("w") as handle:
        yaml.safe_dump(experiment_info, handle, sort_keys=False)

    if cfg.debug_seed is not None:
        seeds = [cfg.debug_seed]
        max_workers = 1
    else:
        seeds = list(range(cfg.num_experiments))
        max_workers = min(cfg.max_workers, len(seeds), os.cpu_count() or 1)

    all_results: list[dict] = []
    if max_workers == 1:
        for seed in seeds:
            seed_result = run_single_seed_experiment(env_name, seed, cfg)
            all_results.append(seed_result)
            plot_robustness(seed_result, env_name, out_dir)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_seed_experiment, env_name, seed, cfg): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed_result = future.result()
                all_results.append(seed_result)
                plot_robustness(seed_result, env_name, out_dir)

    all_results.sort(key=lambda result: result["seed"])
    with (out_dir / "robustness_experiment.pkl").open("wb") as handle:
        pickle.dump(all_results, handle)
    return all_results


if __name__ == "__main__":
    main("CartPole-v1", RobustnessConfig(debug_seed=0))
