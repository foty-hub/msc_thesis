"""Bayesian optimization for the maintained sparse-grid robustness method.

The study sees development seeds only. Optional validation seeds are evaluated
once, after optimization, using the selected trial and the full shift range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import pprint
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml
from stable_baselines3 import DQN

if __package__:
    from .traintime_robustness import RobustnessConfig, train_agent
else:
    from traintime_robustness import RobustnessConfig, train_agent

from crl.calib import collect_transitions
from crl.env import instantiate_eval_env, nominal_reward_threshold
from crl.experiment import SHIFT_SPECS, evaluate_policy, evaluate_shift
from crl.tuning import (
    GridTuningCandidate,
    ObjectiveMode,
    calibrate_candidate,
    choose_shift_indices,
    objective_score,
    partition_seed_scores,
)
from crl.types import AgentTypes, ClassicControl, ScoringMethod
from crl.utils.paths import project_root

CACHE_VERSION = 1

GRID_CHOICES: dict[ClassicControl, tuple[int, ...]] = {
    "CartPole-v1": (2, 3, 4, 6, 8),
    "Acrobot-v1": (3, 4, 5, 6, 8),
    "MountainCar-v0": (6, 8, 10, 12, 16, 20),
    "LunarLander-v3": (2, 3, 4, 5, 6, 8, 10),
}


@dataclass(frozen=True)
class OptunaTuningConfig:
    env_name: ClassicControl = "LunarLander-v3"
    agent_type: AgentTypes = "vanilla"
    cql_alpha: float = 0.05
    n_train_steps: int = 50_000
    retrain: bool = False
    development_seeds: tuple[int, ...] = tuple(range(10))
    validation_seeds: tuple[int, ...] = ()
    eligibility_eval_episodes: int = 25
    nominal_reward_threshold: float | None = None
    tuning_eval_episodes: int = 10
    validation_eval_episodes: int = 25
    num_tuning_shifts: int = 6
    num_trials: int = 50
    n_jobs: int = 1
    torch_threads: int = 1
    sampler_seed: int = 42
    startup_trials: int = 10
    min_seeds_before_prune: int = 3
    study_name: str | None = None
    results_out: str | None = None
    grid_bin_choices: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10)
    calibration_step_choices: tuple[int, ...] = (2_500, 5_000, 10_000, 20_000)
    min_calib_choices: tuple[int, ...] = (10, 20, 40, 60, 80, 100, 120, 160, 200)
    alpha_min: float = 0.01
    alpha_max: float = 0.45
    obs_quantile_min: float = 0.0
    obs_quantile_max: float = 0.2
    max_calib_per_cell: int = 500
    inference_batch_size: int = 4096
    scoring_method: ScoringMethod = "td"
    objective_mode: ObjectiveMode = "mean_median"
    nominal_loss_penalty: float = 0.25
    nominal_loss_tolerance: float = 0.0
    worst_seed_loss_penalty: float = 0.10
    worst_seed_loss_tolerance: float = 10.0
    eligibility_seed_offset: int = 200_000
    eval_seed_offset: int = 100_000
    refresh_cache: bool = False


@dataclass(frozen=True)
class SeedModel:
    seed: int
    model: DQN
    model_fingerprint: str
    n_actions: int


@dataclass(frozen=True)
class SeedContext(SeedModel):
    transitions: tuple


def _validate_config(config: OptunaTuningConfig) -> None:
    if not config.development_seeds:
        raise ValueError("At least one development seed is required.")
    if len(set(config.development_seeds)) != len(config.development_seeds):
        raise ValueError("Development seeds must be unique.")
    if len(set(config.validation_seeds)) != len(config.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    overlap = set(config.development_seeds) & set(config.validation_seeds)
    if overlap:
        raise ValueError(f"Development and validation seeds overlap: {sorted(overlap)}")
    if (
        config.eligibility_eval_episodes < 1
        or config.tuning_eval_episodes < 1
        or config.validation_eval_episodes < 1
    ):
        raise ValueError("Evaluation episode counts must be positive.")
    if (
        config.nominal_reward_threshold is not None
        and not np.isfinite(config.nominal_reward_threshold)
    ):
        raise ValueError("nominal_reward_threshold must be finite.")
    if config.n_train_steps < 1:
        raise ValueError("n_train_steps must be positive.")
    if config.num_trials < 1 or config.n_jobs < 1 or config.torch_threads < 1:
        raise ValueError("Trial, job, and thread counts must be positive.")
    if not 1 <= config.min_seeds_before_prune <= len(config.development_seeds):
        raise ValueError("min_seeds_before_prune is outside the seed count.")
    if config.startup_trials < 0:
        raise ValueError("startup_trials cannot be negative.")
    if (
        not config.grid_bin_choices
        or min(config.grid_bin_choices) < 1
        or len(set(config.grid_bin_choices)) != len(config.grid_bin_choices)
    ):
        raise ValueError("Grid-bin choices must be positive.")
    if (
        not config.calibration_step_choices
        or min(config.calibration_step_choices) < 1
        or len(set(config.calibration_step_choices))
        != len(config.calibration_step_choices)
    ):
        raise ValueError("Calibration-step choices must be positive.")
    if (
        not config.min_calib_choices
        or min(config.min_calib_choices) < 1
        or len(set(config.min_calib_choices)) != len(config.min_calib_choices)
    ):
        raise ValueError("min_calib choices must be positive.")
    if config.max_calib_per_cell < 1 or config.inference_batch_size < 1:
        raise ValueError("Calibration capacity and batch size must be positive.")
    if max(config.min_calib_choices) > config.max_calib_per_cell:
        raise ValueError("min_calib cannot exceed max_calib_per_cell.")
    if max(config.min_calib_choices) > min(config.calibration_step_choices):
        raise ValueError("A min_calib choice exceeds a calibration-step choice.")
    if not 0.0 < config.alpha_min <= config.alpha_max < 1.0:
        raise ValueError("Alpha bounds must lie strictly between zero and one.")
    if not 0.0 <= config.obs_quantile_min <= config.obs_quantile_max < 0.5:
        raise ValueError("Observation-quantile bounds must lie in [0, 0.5).")
    if (
        config.nominal_loss_penalty < 0.0
        or config.worst_seed_loss_penalty < 0.0
        or config.nominal_loss_tolerance < 0.0
        or config.worst_seed_loss_tolerance < 0.0
    ):
        raise ValueError("Objective penalties and tolerances cannot be negative.")
    shift_count = len(SHIFT_SPECS[config.env_name].values)
    if not 1 <= config.num_tuning_shifts <= shift_count:
        raise ValueError("num_tuning_shifts is outside the configured shift range.")


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def _model_fingerprint(model: DQN) -> str:
    digest = hashlib.sha256()
    digest.update(type(model).__qualname__.encode())
    digest.update(str(float(model.gamma)).encode())
    for name, tensor in sorted(model.q_net.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _training_config(config: OptunaTuningConfig) -> RobustnessConfig:
    return RobustnessConfig(
        n_train_steps=config.n_train_steps,
        agent_type=config.agent_type,
        cql_alpha=config.cql_alpha,
        retrain=config.retrain,
    )


def _effective_nominal_threshold(config: OptunaTuningConfig) -> float:
    if config.nominal_reward_threshold is not None:
        return config.nominal_reward_threshold
    return nominal_reward_threshold(config.env_name)


def _prepare_models(
    config: OptunaTuningConfig,
    *,
    seeds: tuple[int, ...],
) -> dict[int, SeedModel]:
    models: dict[int, SeedModel] = {}
    for seed in seeds:
        model, nominal_env = train_agent(
            config.env_name,
            seed,
            _training_config(config),
        )
        try:
            models[seed] = SeedModel(
                seed=seed,
                model=model,
                model_fingerprint=_model_fingerprint(model),
                n_actions=int(model.action_space.n),
            )
        finally:
            nominal_env.close()
    return models


def _eligibility_metadata(
    config: OptunaTuningConfig,
    models: dict[int, SeedModel],
) -> dict:
    return {
        "version": CACHE_VERSION,
        "env_name": config.env_name,
        "agent_type": config.agent_type,
        "seeds": list(models),
        "model_fingerprints": {
            str(seed): seed_model.model_fingerprint
            for seed, seed_model in models.items()
        },
        "n_episodes": config.eligibility_eval_episodes,
        "eval_seed_offset": config.eligibility_seed_offset,
    }


def _prepare_eligibility(
    config: OptunaTuningConfig,
    models: dict[int, SeedModel],
    *,
    cache_path: Path,
    report_path: Path,
) -> dict:
    """Evaluate each original policy once and partition seeds for the run."""
    metadata = _eligibility_metadata(config, models)
    returns_by_seed: dict[int, list[float]] = {}
    if cache_path.exists() and not config.refresh_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata") == metadata:
            returns_by_seed = payload.get("returns", {})

    for seed, seed_model in models.items():
        if seed in returns_by_seed:
            continue
        env = instantiate_eval_env(
            config.env_name,
            seed=config.eligibility_seed_offset + seed,
        )
        try:
            returns_by_seed[seed] = evaluate_policy(
                seed_model.model,
                env,
                n_episodes=config.eligibility_eval_episodes,
                calibration=None,
            )
        finally:
            env.close()
        _atomic_pickle(
            cache_path,
            {"metadata": metadata, "returns": returns_by_seed},
        )

    threshold = _effective_nominal_threshold(config)
    mean_scores = {
        seed: float(np.mean(returns_by_seed[seed])) for seed in models
    }
    eligible, excluded = partition_seed_scores(
        mean_scores,
        threshold=threshold,
    )
    report = {
        **metadata,
        "threshold": threshold,
        "threshold_source": (
            "cli_override"
            if config.nominal_reward_threshold is not None
            else "src/crl/env.py"
        ),
        "comparison": "mean_nominal_return >= threshold",
        "eligible_seeds": list(eligible),
        "excluded_seeds": list(excluded),
        "seed_results": [
            {
                "seed": seed,
                "model_fingerprint": models[seed].model_fingerprint,
                "returns": [float(value) for value in returns_by_seed[seed]],
                "mean_nominal_return": mean_scores[seed],
                "eligible": seed in eligible,
            }
            for seed in models
        ],
    }
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return report


def _prepare_context(
    config: OptunaTuningConfig,
    *,
    seed_model: SeedModel,
    max_transitions: int,
    cache_dir: Path,
) -> SeedContext:
    seed = seed_model.seed
    cache_path = cache_dir / f"buffer_{config.agent_type}_seed_{seed}.pkl"
    transitions: tuple | None = None

    if cache_path.exists() and not config.refresh_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            payload.get("version") == CACHE_VERSION
            and payload.get("env_name") == config.env_name
            and payload.get("model_fingerprint")
            == seed_model.model_fingerprint
            and len(payload.get("transitions", ())) >= max_transitions
        ):
            transitions = tuple(payload["transitions"][:max_transitions])

    if transitions is None:
        nominal_env = instantiate_eval_env(config.env_name, seed=seed)
        try:
            buffer = collect_transitions(
                seed_model.model,
                nominal_env,
                max_transitions,
            )
            transitions = tuple(buffer)
            _atomic_pickle(
                cache_path,
                {
                    "version": CACHE_VERSION,
                    "env_name": config.env_name,
                    "agent_type": config.agent_type,
                    "seed": seed,
                    "model_fingerprint": seed_model.model_fingerprint,
                    "transitions": transitions,
                },
            )
        finally:
            nominal_env.close()
    return SeedContext(
        seed=seed,
        model=seed_model.model,
        transitions=transitions,
        model_fingerprint=seed_model.model_fingerprint,
        n_actions=seed_model.n_actions,
    )


def _prepare_contexts(
    config: OptunaTuningConfig,
    *,
    models: dict[int, SeedModel],
    seeds: tuple[int, ...],
    max_transitions: int,
    cache_dir: Path,
) -> dict[int, SeedContext]:
    contexts: dict[int, SeedContext] = {}
    for seed in seeds:
        contexts[seed] = _prepare_context(
            config,
            seed_model=models[seed],
            max_transitions=max_transitions,
            cache_dir=cache_dir,
        )
    return contexts


def _evaluate_returns(
    model: DQN,
    *,
    env_name: ClassicControl,
    parameter: str,
    values: tuple[float, ...],
    seed: int,
    eval_seed_offset: int,
    n_episodes: int,
    calibration,
) -> list[list[float]]:
    returns: list[list[float]] = []
    for value in values:
        env = instantiate_eval_env(
            env_name,
            seed=eval_seed_offset + seed,
            **{parameter: value},
        )
        try:
            returns.append(
                evaluate_policy(
                    model,
                    env,
                    n_episodes=n_episodes,
                    calibration=calibration,
                )
            )
        finally:
            env.close()
    return returns


def _baseline_metadata(
    config: OptunaTuningConfig,
    contexts: dict[int, SeedContext],
    shift_values: tuple[float, ...],
) -> dict:
    return {
        "version": CACHE_VERSION,
        "env_name": config.env_name,
        "agent_type": config.agent_type,
        "seeds": list(contexts),
        "model_fingerprints": {
            str(seed): context.model_fingerprint for seed, context in contexts.items()
        },
        "shift_values": list(shift_values),
        "n_episodes": config.tuning_eval_episodes,
        "eval_seed_offset": config.eval_seed_offset,
    }


def _prepare_baselines(
    config: OptunaTuningConfig,
    contexts: dict[int, SeedContext],
    *,
    shift_values: tuple[float, ...],
    cache_path: Path,
) -> dict[int, list[list[float]]]:
    metadata = _baseline_metadata(config, contexts, shift_values)
    baselines: dict[int, list[list[float]]] = {}
    if cache_path.exists() and not config.refresh_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata") == metadata:
            baselines = payload.get("returns", {})

    shift_spec = SHIFT_SPECS[config.env_name]
    for seed, context in contexts.items():
        if seed in baselines:
            continue
        baselines[seed] = _evaluate_returns(
            context.model,
            env_name=config.env_name,
            parameter=shift_spec.parameter,
            values=shift_values,
            seed=seed,
            eval_seed_offset=config.eval_seed_offset,
            n_episodes=config.tuning_eval_episodes,
            calibration=None,
        )
        _atomic_pickle(
            cache_path,
            {"metadata": metadata, "returns": baselines},
        )
    return baselines


def _trial_candidate(
    trial: optuna.Trial,
    config: OptunaTuningConfig,
) -> GridTuningCandidate:
    return GridTuningCandidate(
        alpha=trial.suggest_float("alpha", config.alpha_min, config.alpha_max),
        grid_bins=int(
            trial.suggest_categorical("grid_bins", list(config.grid_bin_choices))
        ),
        min_calib=int(
            trial.suggest_categorical("min_calib", list(config.min_calib_choices))
        ),
        n_calib_steps=int(
            trial.suggest_categorical(
                "n_calib_steps",
                list(config.calibration_step_choices),
            )
        ),
        obs_quantile=trial.suggest_float(
            "obs_quantile",
            config.obs_quantile_min,
            config.obs_quantile_max,
        ),
    )


def _set_trial_diagnostics(
    trial: optuna.Trial,
    *,
    seed_deltas: list[float],
    nominal_deltas: list[float],
    calibrated_cells: list[int],
    visited_cells: list[int],
    fallbacks: list[float],
    shift_deltas: dict[str, list[float]],
    score: float,
) -> None:
    trial.set_user_attr("seed_deltas", seed_deltas)
    trial.set_user_attr("mean_delta", float(np.mean(seed_deltas)))
    trial.set_user_attr("median_delta", float(np.median(seed_deltas)))
    trial.set_user_attr("worst_seed_delta", float(np.min(seed_deltas)))
    trial.set_user_attr("nominal_mean_delta", float(np.mean(nominal_deltas)))
    trial.set_user_attr("calibrated_cells", calibrated_cells)
    trial.set_user_attr("visited_cells", visited_cells)
    trial.set_user_attr("fallbacks", fallbacks)
    trial.set_user_attr("max_fallback", float(np.max(fallbacks)))
    trial.set_user_attr("shift_deltas", shift_deltas)
    trial.set_user_attr("objective_score", score)


def _build_objective(
    config: OptunaTuningConfig,
    *,
    contexts: dict[int, SeedContext],
    baselines: dict[int, list[list[float]]],
    shift_values: tuple[float, ...],
    seed_order: tuple[int, ...],
):
    nominal_position = int(
        np.argmin(
            np.abs(
                np.asarray(shift_values)
                - SHIFT_SPECS[config.env_name].nominal_value
            )
        )
    )

    def objective(trial: optuna.Trial) -> float:
        candidate = _trial_candidate(trial, config)
        seed_deltas: list[float] = []
        nominal_deltas: list[float] = []
        calibrated_cells: list[int] = []
        visited_cells: list[int] = []
        fallbacks: list[float] = []
        shift_deltas: dict[str, list[float]] = {}
        started = time.perf_counter()

        for completed, seed in enumerate(seed_order, start=1):
            context = contexts[seed]
            calibration = calibrate_candidate(
                context.model,
                context.transitions,
                n_actions=context.n_actions,
                candidate=candidate,
                max_calib_per_cell=config.max_calib_per_cell,
                scoring_method=config.scoring_method,
                inference_batch_size=config.inference_batch_size,
            )
            calibrated = _evaluate_returns(
                context.model,
                env_name=config.env_name,
                parameter=SHIFT_SPECS[config.env_name].parameter,
                values=shift_values,
                seed=seed,
                eval_seed_offset=config.eval_seed_offset,
                n_episodes=config.tuning_eval_episodes,
                calibration=calibration,
            )
            baseline_means = np.asarray(
                [np.mean(values) for values in baselines[seed]],
                dtype=float,
            )
            calibrated_means = np.asarray(
                [np.mean(values) for values in calibrated],
                dtype=float,
            )
            deltas = calibrated_means - baseline_means
            seed_deltas.append(float(np.mean(deltas)))
            nominal_deltas.append(float(deltas[nominal_position]))
            calibrated_cells.append(calibration.n_calibrated_cells)
            visited_cells.append(calibration.n_visited_cells)
            fallbacks.append(calibration.fallback)
            shift_deltas[str(seed)] = [float(value) for value in deltas]

            score = objective_score(
                seed_deltas,
                mode=config.objective_mode,
                nominal_deltas=nominal_deltas,
                nominal_loss_penalty=config.nominal_loss_penalty,
                nominal_loss_tolerance=config.nominal_loss_tolerance,
                worst_seed_loss_penalty=config.worst_seed_loss_penalty,
                worst_seed_loss_tolerance=config.worst_seed_loss_tolerance,
            )
            _set_trial_diagnostics(
                trial,
                seed_deltas=seed_deltas,
                nominal_deltas=nominal_deltas,
                calibrated_cells=calibrated_cells,
                visited_cells=visited_cells,
                fallbacks=fallbacks,
                shift_deltas=shift_deltas,
                score=score,
            )
            trial.set_user_attr("duration_seconds", time.perf_counter() - started)
            trial.report(score, step=completed - 1)
            if (
                completed >= config.min_seeds_before_prune
                and trial.should_prune()
            ):
                raise optuna.TrialPruned()
        return score

    return objective


def _study_signature(
    config: OptunaTuningConfig,
    *,
    shift_indices: tuple[int, ...],
    eligible_development_seeds: tuple[int, ...],
    seed_order: tuple[int, ...],
) -> dict:
    return {
        "env_name": config.env_name,
        "agent_type": config.agent_type,
        "requested_development_seeds": list(config.development_seeds),
        "eligible_development_seeds": list(eligible_development_seeds),
        "nominal_reward_threshold": _effective_nominal_threshold(config),
        "eligibility_eval_episodes": config.eligibility_eval_episodes,
        "eligibility_seed_offset": config.eligibility_seed_offset,
        "seed_order": list(seed_order),
        "shift_indices": list(shift_indices),
        "tuning_eval_episodes": config.tuning_eval_episodes,
        "grid_bin_choices": list(config.grid_bin_choices),
        "calibration_step_choices": list(config.calibration_step_choices),
        "min_calib_choices": list(config.min_calib_choices),
        "alpha": [config.alpha_min, config.alpha_max],
        "obs_quantile": [
            config.obs_quantile_min,
            config.obs_quantile_max,
        ],
        "max_calib_per_cell": config.max_calib_per_cell,
        "scoring_method": config.scoring_method,
        "objective_mode": config.objective_mode,
        "nominal_loss_penalty": config.nominal_loss_penalty,
        "nominal_loss_tolerance": config.nominal_loss_tolerance,
        "worst_seed_loss_penalty": config.worst_seed_loss_penalty,
        "worst_seed_loss_tolerance": config.worst_seed_loss_tolerance,
        "eval_seed_offset": config.eval_seed_offset,
    }


def _lock_study_metadata(
    study: optuna.Study,
    *,
    signature: dict,
    models: dict[int, SeedModel],
) -> None:
    fingerprints = {
        str(seed): seed_model.model_fingerprint
        for seed, seed_model in models.items()
    }
    existing_signature = study.user_attrs.get("search_signature")
    existing_fingerprints = study.user_attrs.get("model_fingerprints")
    if existing_signature is not None and existing_signature != signature:
        raise ValueError(
            "The resumed study uses a different search configuration. "
            "Choose a new --study-name or --results-out."
        )
    if existing_fingerprints is not None and existing_fingerprints != fingerprints:
        raise ValueError(
            "The resumed study uses different model weights. "
            "Choose a new --study-name or --results-out."
        )
    study.set_user_attr("search_signature", signature)
    study.set_user_attr("model_fingerprints", fingerprints)


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "intermediate_values": {
            str(step): value for step, value in trial.intermediate_values.items()
        },
        "datetime_start": (
            trial.datetime_start.isoformat() if trial.datetime_start else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ),
    }


def _save_study_outputs(
    output_dir: Path,
    config: OptunaTuningConfig,
    study: optuna.Study,
    *,
    shift_indices: tuple[int, ...],
    shift_values: tuple[float, ...],
) -> None:
    trials = [_trial_record(trial) for trial in study.get_trials(deepcopy=False)]
    with (output_dir / "trials.json").open("w") as handle:
        json.dump(trials, handle, indent=2)

    best = study.best_trial
    payload = {
        "config": asdict(config),
        "tuning_shift_indices": list(shift_indices),
        "tuning_shift_values": list(shift_values),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
    }
    with (output_dir / "best_config.yaml").open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _candidate_from_params(params: dict) -> GridTuningCandidate:
    return GridTuningCandidate(
        alpha=float(params["alpha"]),
        grid_bins=int(params["grid_bins"]),
        min_calib=int(params["min_calib"]),
        n_calib_steps=int(params["n_calib_steps"]),
        obs_quantile=float(params["obs_quantile"]),
    )


def _run_validation(
    config: OptunaTuningConfig,
    *,
    candidate: GridTuningCandidate,
    best_trial: optuna.trial.FrozenTrial,
    cache_dir: Path,
    output_dir: Path,
) -> list[dict]:
    if not config.validation_seeds:
        return []
    models = _prepare_models(config, seeds=config.validation_seeds)
    eligibility = _prepare_eligibility(
        config,
        models,
        cache_path=cache_dir / "validation_nominal_returns.pkl",
        report_path=output_dir / "validation_eligibility.json",
    )
    eligible_seeds = tuple(eligibility["eligible_seeds"])
    excluded_seeds = tuple(eligibility["excluded_seeds"])
    if excluded_seeds:
        print(
            "Excluded validation seeds below the nominal threshold: "
            f"{list(excluded_seeds)}"
        )
    if not eligible_seeds:
        summary = {
            "best_trial_number": best_trial.number,
            "candidate": asdict(candidate),
            "requested_validation_seeds": list(config.validation_seeds),
            "validation_seeds": [],
            "excluded_validation_seeds": list(excluded_seeds),
            "nominal_reward_threshold": eligibility["threshold"],
            "baseline_seed_means": [],
            "calibrated_seed_means": [],
            "seed_deltas": [],
            "baseline_mean": None,
            "calibrated_mean": None,
            "mean_delta": None,
            "median_delta": None,
            "seed_wins": 0,
            "seed_losses": 0,
        }
        with (output_dir / "best_validation_summary.yaml").open("w") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)
        print("No validation seeds reached the nominal reward threshold.")
        return []
    contexts = _prepare_contexts(
        config,
        models=models,
        seeds=eligible_seeds,
        max_transitions=candidate.n_calib_steps,
        cache_dir=cache_dir,
    )
    shift_spec = SHIFT_SPECS[config.env_name]
    results: list[dict] = []
    validation_path = output_dir / "best_validation.pkl"

    for seed in eligible_seeds:
        started = time.perf_counter()
        context = contexts[seed]
        calibration = calibrate_candidate(
            context.model,
            context.transitions,
            n_actions=context.n_actions,
            candidate=candidate,
            max_calib_per_cell=config.max_calib_per_cell,
            scoring_method=config.scoring_method,
            inference_batch_size=config.inference_batch_size,
        )
        shift_results = [
            evaluate_shift(
                context.model,
                env_name=config.env_name,
                parameter=shift_spec.parameter,
                value=value,
                n_episodes=config.validation_eval_episodes,
                eval_seed=config.eval_seed_offset + seed,
                calibration=calibration,
            )
            for value in shift_spec.values
        ]
        results.append(
            {
                "seed": seed,
                "model_fingerprint": context.model_fingerprint,
                "calibration": {
                    "n_visited_cells": calibration.n_visited_cells,
                    "n_calibrated_cells": calibration.n_calibrated_cells,
                    "fallback": calibration.fallback,
                    "n_state_action_cells": (
                        calibration.discretiser.n_state_action_cells
                    ),
                },
                "timing_seconds": time.perf_counter() - started,
                "results": shift_results,
            }
        )
        _atomic_pickle(
            validation_path,
            {
                "best_trial_number": best_trial.number,
                "best_trial_value": best_trial.value,
                "candidate": asdict(candidate),
                "requested_validation_seeds": config.validation_seeds,
                "validation_seeds": eligible_seeds,
                "excluded_validation_seeds": excluded_seeds,
                "nominal_reward_threshold": eligibility["threshold"],
                "results": results,
            },
        )

    baseline_seed_means = []
    calibrated_seed_means = []
    seed_deltas = []
    for seed_result in results:
        baseline = np.asarray(
            [
                np.mean(row["returns_noconf"])
                for row in seed_result["results"]
            ]
        )
        calibrated = np.asarray(
            [
                np.mean(row["returns_conf"])
                for row in seed_result["results"]
            ]
        )
        baseline_seed_means.append(float(np.mean(baseline)))
        calibrated_seed_means.append(float(np.mean(calibrated)))
        seed_deltas.append(float(np.mean(calibrated - baseline)))
    summary = {
        "best_trial_number": best_trial.number,
        "candidate": asdict(candidate),
        "requested_validation_seeds": list(config.validation_seeds),
        "validation_seeds": list(eligible_seeds),
        "excluded_validation_seeds": list(excluded_seeds),
        "nominal_reward_threshold": eligibility["threshold"],
        "baseline_seed_means": baseline_seed_means,
        "calibrated_seed_means": calibrated_seed_means,
        "seed_deltas": seed_deltas,
        "baseline_mean": float(np.mean(baseline_seed_means)),
        "calibrated_mean": float(np.mean(calibrated_seed_means)),
        "mean_delta": float(np.mean(seed_deltas)),
        "median_delta": float(np.median(seed_deltas)),
        "seed_wins": int(np.sum(np.asarray(seed_deltas) > 0)),
        "seed_losses": int(np.sum(np.asarray(seed_deltas) < 0)),
    }
    with (output_dir / "best_validation_summary.yaml").open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    return results


def run_tuning(config: OptunaTuningConfig) -> optuna.Study:
    _validate_config(config)
    torch.set_num_threads(config.torch_threads)
    if config.scoring_method == "monte_carlo":
        print(
            "WARNING: Monte Carlo scoring includes a biased incomplete final "
            "episode unless the buffer-boundary issue in tuning.md is fixed."
        )

    shift_spec = SHIFT_SPECS[config.env_name]
    shift_indices = choose_shift_indices(
        shift_spec.values,
        nominal_value=shift_spec.nominal_value,
        count=config.num_tuning_shifts,
    )
    shift_values = tuple(shift_spec.values[index] for index in shift_indices)
    output_dir = (
        project_root()
        / "results"
        / (
            config.results_out
            or f"optuna/{config.env_name}/sparse_grid"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    models = _prepare_models(config, seeds=config.development_seeds)
    eligibility = _prepare_eligibility(
        config,
        models,
        cache_path=cache_dir / "development_nominal_returns.pkl",
        report_path=output_dir / "development_eligibility.json",
    )
    eligible_seeds = tuple(eligibility["eligible_seeds"])
    excluded_seeds = tuple(eligibility["excluded_seeds"])
    if excluded_seeds:
        print(
            "Excluded development seeds below the nominal threshold: "
            f"{list(excluded_seeds)}"
        )
    if not eligible_seeds:
        raise ValueError(
            "No development seeds reached the nominal reward threshold of "
            f"{eligibility['threshold']}."
        )
    if len(eligible_seeds) < config.min_seeds_before_prune:
        print(
            f"WARNING: only {len(eligible_seeds)} development seeds are "
            "eligible, so trials cannot reach min_seeds_before_prune="
            f"{config.min_seeds_before_prune} and seed-based pruning is "
            "effectively disabled."
        )
    run_metadata = {
        **asdict(config),
        "effective_nominal_reward_threshold": eligibility["threshold"],
        "eligible_development_seeds": list(eligible_seeds),
        "excluded_development_seeds": list(excluded_seeds),
        "tuning_shift_indices": list(shift_indices),
        "tuning_shift_values": list(shift_values),
    }
    with (output_dir / "tuning_config.yaml").open("w") as handle:
        yaml.safe_dump(run_metadata, handle, sort_keys=False)
    pprint.pprint(
        {
            **run_metadata,
            "output_dir": str(output_dir),
        },
        sort_dicts=False,
    )
    contexts = _prepare_contexts(
        config,
        models=models,
        seeds=eligible_seeds,
        max_transitions=max(config.calibration_step_choices),
        cache_dir=cache_dir,
    )
    baselines = _prepare_baselines(
        config,
        contexts,
        shift_values=shift_values,
        cache_path=cache_dir / "tuning_baselines.pkl",
    )
    generator = np.random.default_rng(config.sampler_seed)
    seed_order = tuple(
        int(seed) for seed in generator.permutation(eligible_seeds)
    )
    study_name = config.study_name or f"{config.env_name}_sparse_grid"
    storage_url = f"sqlite:///{output_dir / 'study.sqlite3'}"
    sampler = optuna.samplers.TPESampler(
        seed=config.sampler_seed,
        multivariate=True,
        n_startup_trials=config.startup_trials,
        constant_liar=config.n_jobs > 1,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.startup_trials,
        n_warmup_steps=config.min_seeds_before_prune - 1,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    _lock_study_metadata(
        study,
        signature=_study_signature(
            config,
            shift_indices=shift_indices,
            eligible_development_seeds=eligible_seeds,
            seed_order=seed_order,
        ),
        models=models,
    )
    remaining_trials = max(0, config.num_trials - len(study.trials))
    if remaining_trials:
        study.optimize(
            _build_objective(
                config,
                contexts=contexts,
                baselines=baselines,
                shift_values=shift_values,
                seed_order=seed_order,
            ),
            n_trials=remaining_trials,
            n_jobs=config.n_jobs,
            gc_after_trial=True,
        )
    else:
        print(
            f"Study already contains {len(study.trials)} trials; "
            f"target is {config.num_trials}."
        )
    _save_study_outputs(
        output_dir,
        config,
        study,
        shift_indices=shift_indices,
        shift_values=shift_values,
    )
    _run_validation(
        config,
        candidate=_candidate_from_params(study.best_trial.params),
        best_trial=study.best_trial,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    print(
        f"Best trial {study.best_trial.number}: "
        f"value={study.best_value:.3f}, params={study.best_params}"
    )
    return study


def parse_args() -> OptunaTuningConfig:
    parser = argparse.ArgumentParser(
        description="Tune sparse-grid train-time robustness with Optuna TPE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-name",
        choices=sorted(SHIFT_SPECS),
        default="LunarLander-v3",
    )
    parser.add_argument(
        "--agent-type",
        choices=["vanilla", "ddqn", "cql"],
        default="vanilla",
    )
    parser.add_argument("--cql-alpha", type=float, default=0.05)
    parser.add_argument("--n-train-steps", type=int, default=50_000)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--development-seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seeds visible to Optuna.",
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Disjoint seeds evaluated once after the best trial is selected.",
    )
    parser.add_argument(
        "--eligibility-eval-episodes",
        type=int,
        default=25,
        help="Nominal episodes in the one-time original-policy seed check.",
    )
    parser.add_argument(
        "--nominal-reward-threshold",
        type=float,
        help="Override the environment threshold defined in src/crl/env.py.",
    )
    parser.add_argument("--tuning-eval-episodes", type=int, default=10)
    parser.add_argument("--validation-eval-episodes", type=int, default=25)
    parser.add_argument("--num-tuning-shifts", type=int, default=6)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Target total number of trials, including resumed trials.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Concurrent Optuna threads sharing read-only models and buffers.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--min-seeds-before-prune", type=int, default=3)
    parser.add_argument("--study-name")
    parser.add_argument(
        "--results-out",
        help="Directory relative to results/ for study storage and caches.",
    )
    parser.add_argument("--grid-bins", type=int, nargs="+")
    parser.add_argument(
        "--calibration-steps",
        type=int,
        nargs="+",
        default=[2_500, 5_000, 10_000, 20_000],
    )
    parser.add_argument(
        "--min-calib",
        type=int,
        nargs="+",
        default=[10, 20, 40, 60, 80, 100, 120, 160, 200],
    )
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=0.45)
    parser.add_argument("--obs-quantile-min", type=float, default=0.0)
    parser.add_argument("--obs-quantile-max", type=float, default=0.2)
    parser.add_argument("--max-calib-per-cell", type=int, default=500)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument(
        "--scoring-method",
        choices=["td", "monte_carlo"],
        default="td",
    )
    parser.add_argument(
        "--objective-mode",
        choices=["mean", "median", "mean_median"],
        default="mean_median",
    )
    parser.add_argument("--nominal-loss-penalty", type=float, default=0.25)
    parser.add_argument("--nominal-loss-tolerance", type=float, default=0.0)
    parser.add_argument("--worst-seed-loss-penalty", type=float, default=0.10)
    parser.add_argument("--worst-seed-loss-tolerance", type=float, default=10.0)
    parser.add_argument("--eligibility-seed-offset", type=int, default=200_000)
    parser.add_argument("--eval-seed-offset", type=int, default=100_000)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--print-config-only", action="store_true")
    args = parser.parse_args()

    grid_choices = (
        tuple(args.grid_bins)
        if args.grid_bins is not None
        else GRID_CHOICES[args.env_name]
    )
    config = OptunaTuningConfig(
        env_name=args.env_name,
        agent_type=args.agent_type,
        cql_alpha=args.cql_alpha,
        n_train_steps=args.n_train_steps,
        retrain=args.retrain,
        development_seeds=tuple(args.development_seeds),
        validation_seeds=tuple(args.validation_seeds),
        eligibility_eval_episodes=args.eligibility_eval_episodes,
        nominal_reward_threshold=args.nominal_reward_threshold,
        tuning_eval_episodes=args.tuning_eval_episodes,
        validation_eval_episodes=args.validation_eval_episodes,
        num_tuning_shifts=args.num_tuning_shifts,
        num_trials=args.num_trials,
        n_jobs=args.n_jobs,
        torch_threads=args.torch_threads,
        sampler_seed=args.sampler_seed,
        startup_trials=args.startup_trials,
        min_seeds_before_prune=args.min_seeds_before_prune,
        study_name=args.study_name,
        results_out=args.results_out,
        grid_bin_choices=grid_choices,
        calibration_step_choices=tuple(args.calibration_steps),
        min_calib_choices=tuple(args.min_calib),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        obs_quantile_min=args.obs_quantile_min,
        obs_quantile_max=args.obs_quantile_max,
        max_calib_per_cell=args.max_calib_per_cell,
        inference_batch_size=args.inference_batch_size,
        scoring_method=args.scoring_method,
        objective_mode=args.objective_mode,
        nominal_loss_penalty=args.nominal_loss_penalty,
        nominal_loss_tolerance=args.nominal_loss_tolerance,
        worst_seed_loss_penalty=args.worst_seed_loss_penalty,
        worst_seed_loss_tolerance=args.worst_seed_loss_tolerance,
        eligibility_seed_offset=args.eligibility_seed_offset,
        eval_seed_offset=args.eval_seed_offset,
        refresh_cache=args.refresh_cache,
    )
    _validate_config(config)
    if args.print_config_only:
        print(json.dumps(asdict(config), indent=2))
        raise SystemExit(0)
    return config


if __name__ == "__main__":
    run_tuning(parse_args())
