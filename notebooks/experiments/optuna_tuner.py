"""
Optuna-based hyperparameter tuner for traintime robustness experiments.

Key features
- Supports both CC-Disc (alpha, tiles, n_calib_steps, min_calib) and CCNN (k, max_dist_q, n_calib_steps)
- Optimizes mean episodic return aggregated across shift range (or nominal-only)
- Parallelizes across TRIALS (via Optuna n_jobs); evaluates SEEDS sequentially per trial
- Uses MedianPruner with careful warm-up to mitigate seed variance
- Reuses per-seed cached trained models and replay buffers (same directory layout)
- Writes a single `optuna_trials.json` and a `best_config.yaml` under results/{env}/tuning_{method}/

Usage examples
- CC-Disc on LunarLander nominal-only, 20 trials, 5 seeds, threading:
  uv run notebooks/experiments/optuna_tuner.py --env-name LunarLander-v3 --method ccdisc \
    --nominal-only --num-trials 20 --num-seeds 5 --num-eval-episodes 100 --n-jobs 4

- CCNN on CartPole with multi-process JournalStorage (recommended for RL):
  uv run notebooks/experiments/optuna_tuner.py --env-name CartPole-v1 --method ccnn \
    --num-trials 20 --num-seeds 5 --mp-workers 4
"""

import argparse
import json
import os
import pickle
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from traintime_robustness import (
    EVAL_PARAMETERS,
    RobustnessConfig,
    run_eval,
    train_agent,
)

from crl.buffer import ReplayBuffer
from crl.calib import (
    collect_transitions,
    compute_corrections,
    fill_calib_sets_mc,
    fill_calib_sets_td,
)
from crl.ccnn import calibrate_ccnn, run_ccnn_experiment
from crl.discretise import build_tile_coding
from crl.env import instantiate_eval_env

# -------------------------
# Helpers: metrics and utils (mirrors hparam_tuner)
# -------------------------

# Search space constants (tweak here)
# CC-Disc
CCDISC_ALPHA_MIN, CCDISC_ALPHA_MAX = 0.01, 0.25
CCDISC_TILES_CHOICES = [4, 6, 8, 10, 12]
CCDISC_TILINGS_FIXED = 1
# Shared across methods
N_CALIB_STEPS_MIN, N_CALIB_STEPS_MAX = 2_000, 20_000
MIN_CALIB_MIN, MIN_CALIB_MAX, MIN_CALIB_STEP = 25, 100, 5

# CCNN
CCNN_K_MIN, CCNN_K_MAX = 25, 100
CCNN_MAX_DIST_Q_MIN, CCNN_MAX_DIST_Q_MAX = 0.85, 0.99

# Buffer precollection budget factor (x upper bound of n_calib_steps)
PRECACHE_FACTOR = 2


def _mean_return_from_results(
    results: list[dict], key: str, nominal_only: bool, env_name: str
) -> float:
    if nominal_only:
        per_shift_means = [np.mean(results[0][key])]
    else:
        per_shift_means = [np.mean(r[key]) for r in results]
    return float(np.mean(per_shift_means))


def _eval_ccdisc_single_seed(
    seed: int,
    env_name: str,
    num_eval_episodes: int,
    tiles: int,
    tilings: int,
    n_calib_steps: int,
    alpha: float,
    min_calib: int,
    base_cfg: RobustnessConfig,
    nominal_only: bool,
    buffer_dir: str | None,
) -> dict[str, Any]:
    cfg = RobustnessConfig(**{**asdict(base_cfg)})
    model, vec_env = train_agent(env_name, seed, cfg)

    discretise, n_features = build_tile_coding(model, vec_env, tiles, tilings)
    buffer_path = (
        os.path.join(buffer_dir, f"buffer_seed_{seed}.pkl") if buffer_dir else None
    )
    if buffer_path is not None and os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            big_buffer: ReplayBuffer = pickle.load(f)
        use_n = min(n_calib_steps, len(big_buffer))
        buffer = ReplayBuffer(capacity=use_n)
        for i in range(use_n):
            tr = big_buffer[i]
            buffer.push(
                tr.state, tr.action, tr.reward, tr.next_state, tr.next_action, tr.done
            )
    else:
        buffer = collect_transitions(model, vec_env, n_transitions=n_calib_steps)

    if cfg.scoring_method == "td":
        calib_sets = fill_calib_sets_td(
            model,
            buffer,
            discretise,
            score=cfg.score_fn,
        )
    else:
        calib_sets = fill_calib_sets_mc(
            model,
            buffer,
            discretise,
            score=cfg.score_fn,
        )

    qhats = compute_corrections(calib_sets, alpha=alpha, min_calib=min_calib)

    results: list[dict] = []
    param, param_values, _, _ = EVAL_PARAMETERS[env_name]
    if nominal_only:
        ep_env = instantiate_eval_env(env_name=env_name)
        returns_conf = run_eval(
            model,
            discretise,
            num_eps=num_eval_episodes,
            conformalise=True,
            ep_env=ep_env,
            qhats=qhats,
        )
        results.append({param: None, "returns_conf": returns_conf})
    else:
        for param_val in param_values:
            ep_env = instantiate_eval_env(
                env_name=env_name, **{param: float(param_val)}
            )
            returns_conf = run_eval(
                model,
                discretise,
                num_eps=num_eval_episodes,
                conformalise=True,
                ep_env=ep_env,
                qhats=qhats,
            )
            results.append({param: float(param_val), "returns_conf": returns_conf})

    mean_ret = _mean_return_from_results(
        results, key="returns_conf", nominal_only=nominal_only, env_name=env_name
    )

    calibrated_count = max(0, len(qhats) - 1)  # subtract one for fallback
    fraction = float(calibrated_count) / float(n_features) if n_features > 0 else 0.0

    return dict(
        mean_return=float(mean_ret),
        calib_fraction=float(fraction),
        calib_count=int(calibrated_count),
        total_bins=int(n_features),
    )


def _eval_ccnn_single_seed(
    seed: int,
    env_name: str,
    num_eval_episodes: int,
    n_calib_steps: int,
    k: int,
    max_dist_q: float,
    base_cfg: RobustnessConfig,
    nominal_only: bool,
    buffer_dir: str | None,
) -> float:
    cfg = RobustnessConfig(**{**asdict(base_cfg)})
    model, vec_env = train_agent(env_name, seed, cfg)

    buffer_path = (
        os.path.join(buffer_dir, f"buffer_seed_{seed}.pkl") if buffer_dir else None
    )
    if buffer_path is not None and os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            big_buffer: ReplayBuffer = pickle.load(f)
        use_n = min(n_calib_steps, len(big_buffer))
        buffer = ReplayBuffer(capacity=use_n)
        for i in range(use_n):
            tr = big_buffer[i]
            buffer.push(
                tr.state, tr.action, tr.reward, tr.next_state, tr.next_action, tr.done
            )
    else:
        buffer = collect_transitions(model, vec_env, n_transitions=n_calib_steps)

    scores, scaler, tree, max_dist = calibrate_ccnn(
        model,
        buffer,
        k=k,
        score_fn=cfg.score_fn,
        scoring_method=cfg.scoring_method,
        max_distance_quantile=max_dist_q,
    )

    results: list[dict] = []
    param, param_values, _, _ = EVAL_PARAMETERS[env_name]
    if nominal_only:
        returns = run_ccnn_experiment(
            model,
            env_name=env_name,
            alpha=cfg.alpha_nn,
            k=k,
            ccnn_scores=scores,
            tree=tree,
            scaler=scaler,
            max_dist=max_dist,
            param=param,
            param_val=param_values[0],
            num_eps=num_eval_episodes,
        )["returns_ccnn"]
        results.append({param: None, "returns_ccnn": returns})
    else:
        for param_val in param_values:
            ccnn_results = run_ccnn_experiment(
                model,
                env_name=env_name,
                alpha=cfg.alpha_nn,
                k=k,
                ccnn_scores=scores,
                tree=tree,
                scaler=scaler,
                max_dist=max_dist,
                param=param,
                param_val=param_val,
                num_eps=num_eval_episodes,
            )
            results.append({param: float(param_val), **ccnn_results})

    return _mean_return_from_results(
        results, key="returns_ccnn", nominal_only=nominal_only, env_name=env_name
    )


# -------------------------
# Buffer preparation (reuse layout)
# -------------------------


def _prepare_buffers(
    env_name: str,
    seeds: list[int],
    base_cfg: RobustnessConfig,
    out_dir: str,
    precollect_n: int,
) -> str:
    buf_dir = os.path.join(out_dir, "buffers")
    os.makedirs(buf_dir, exist_ok=True)
    print("Ensuring replay buffers exist")
    for seed in seeds:
        buf_path = os.path.join(buf_dir, f"buffer_seed_{seed}.pkl")
        if os.path.exists(buf_path):
            continue
        model, vec_env = train_agent(env_name, seed, base_cfg)
        buffer = collect_transitions(model, vec_env, n_transitions=precollect_n)
        with open(buf_path, "wb") as f:
            pickle.dump(buffer, f)
    return buf_dir


# -------------------------
# Optuna objective
# -------------------------


def build_objective(
    *,
    method: str,
    env_name: str,
    nominal_only: bool,
    num_eval_episodes: int,
    seeds: list[int],
    base_cfg: RobustnessConfig,
    buffer_dir: str,
) -> Callable[[optuna.trial.Trial], float]:
    """Builds an Optuna objective that evaluates seeds sequentially and reports
    intermediate means at step=s (seeds completed)."""

    assert method in {"ccdisc", "ccnn"}

    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest hyperparameters (continuous/int as appropriate)
        if method == "ccdisc":
            alpha = trial.suggest_float("alpha", CCDISC_ALPHA_MIN, CCDISC_ALPHA_MAX)
            tiles = trial.suggest_categorical("tiles", CCDISC_TILES_CHOICES)
            tilings = CCDISC_TILINGS_FIXED  # fixed as requested
            n_calib_steps = trial.suggest_int(
                "n_calib_steps", N_CALIB_STEPS_MIN, N_CALIB_STEPS_MAX, log=True
            )
            min_calib = trial.suggest_int(
                "min_calib", MIN_CALIB_MIN, MIN_CALIB_MAX, step=MIN_CALIB_STEP
            )
            # Per-seed evaluation (sequential to enable pruning)
            seed_means: list[float] = []
            for idx, seed in enumerate(seeds):
                res = _eval_ccdisc_single_seed(
                    seed,
                    env_name,
                    num_eval_episodes,
                    tiles,
                    tilings,
                    n_calib_steps,
                    alpha,
                    min_calib,
                    base_cfg,
                    nominal_only,
                    buffer_dir,
                )
                seed_means.append(float(res["mean_return"]))
                running_mean = float(np.mean(seed_means))
                step = idx + 1
                trial.report(running_mean, step=step)
                # Prune check (MedianPruner compares at same step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(seed_means))

        else:  # method == 'ccnn'
            k = trial.suggest_int("k", CCNN_K_MIN, CCNN_K_MAX, step=5)
            max_dist_q = trial.suggest_float(
                "max_dist_q", CCNN_MAX_DIST_Q_MIN, CCNN_MAX_DIST_Q_MAX
            )
            n_calib_steps = trial.suggest_int(
                "n_calib_steps", N_CALIB_STEPS_MIN, N_CALIB_STEPS_MAX, log=True
            )
            seed_means: list[float] = []
            for idx, seed in enumerate(seeds):
                mean_ret = _eval_ccnn_single_seed(
                    seed,
                    env_name,
                    num_eval_episodes,
                    n_calib_steps,
                    k,
                    max_dist_q,
                    base_cfg,
                    nominal_only,
                    buffer_dir,
                )
                seed_means.append(float(mean_ret))
                running_mean = float(np.mean(seed_means))
                step = idx + 1
                trial.report(running_mean, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(seed_means))

    return objective


# -------------------------
# CLI and orchestration
# -------------------------


def parse_args() -> argparse.Namespace:
    defaults = RobustnessConfig()
    p = argparse.ArgumentParser(description="Optuna tuner for traintime robustness.")
    p.add_argument(
        "--env-name", default="CartPole-v1", choices=sorted(EVAL_PARAMETERS.keys())
    )
    p.add_argument("--method", choices=["ccdisc", "ccnn"], default="ccdisc")
    p.add_argument("--nominal-only", action="store_true")
    p.add_argument("--num-eval-episodes", type=int, default=100)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)

    # Base config (forwarded to RobustnessConfig/train_agent)
    p.add_argument(
        "--agent-type", choices=["vanilla", "ddqn", "cql"], default=defaults.agent_type
    )
    p.add_argument("--n-train-steps", type=int, default=defaults.n_train_steps)
    p.add_argument(
        "--scoring-method",
        choices=["td", "monte_carlo"],
        default=defaults.scoring_method,
    )
    p.add_argument("--alpha-nn", type=float, default=defaults.alpha_nn)  # for CCNN eval
    p.add_argument("--cql-alpha", type=float, default=defaults.cql_alpha)
    p.add_argument("--retrain", action="store_true")

    # Optuna/Study parameters
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel trials (defaults to CPU count)",
    )
    p.add_argument("--study-name", default=None, help="Optional study name (in-memory)")
    # Multiprocessing with JournalStorage
    p.add_argument(
        "--mp-workers",
        type=int,
        default=0,
        help="If >0, run multi-process optimization with JournalStorage using this many processes.",
    )
    p.add_argument(
        "--journal-path",
        default=None,
        help="Path to journal log file (default: results/{env}/tuning_{method}/optuna_journal.log)",
    )

    return p.parse_args()


def build_base_config_from_args(args: argparse.Namespace) -> RobustnessConfig:
    return RobustnessConfig(
        alpha_disc=0.1,  # not used directly for CC-Disc metric computation here
        alpha_nn=args.alpha_nn,
        cql_alpha=args.cql_alpha,
        min_calib=50,  # overridden per-eval for CC-Disc
        num_experiments=args.num_seeds,  # we manage seeds ourselves
        num_eval_episodes=args.num_eval_episodes,
        n_calib_steps=10_000,  # overridden per-trial suggestion
        n_train_steps=args.n_train_steps,
        k=50,  # overridden for CCNN
        scoring_method=args.scoring_method,
        agent_type=args.agent_type,
        retrain=args.retrain,
        calib_methods=["ccdisc"] if args.method == "ccdisc" else ["ccnn"],
    )


def _trial_to_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value,
        "params": trial.params,
        "intermediate_values": trial.intermediate_values,  # step -> value
        "user_attrs": trial.user_attrs,
    }
    return rec


def _mp_worker_entry(payload: dict[str, Any]) -> None:
    """Multi-process worker entrypoint. Rebuilds config and runs a slice of trials.

    Payload keys expected:
    - n_trials, method, env_name, nominal_only, num_eval_episodes, seeds, buffer_dir,
      agent_type, n_train_steps, scoring_method, alpha_nn, cql_alpha, retrain,
      study_name, journal_path, sampler_seed
    """
    n_trials: int = int(payload["n_trials"])
    method: str = payload["method"]
    env_name: str = payload["env_name"]
    nominal_only: bool = bool(payload["nominal_only"])
    num_eval_episodes: int = int(payload["num_eval_episodes"])
    seeds: list[int] = list(payload["seeds"])  # ensure plain list
    buffer_dir: str = payload["buffer_dir"]
    study_name: str = payload["study_name"]
    journal_path: str = payload["journal_path"]
    sampler_seed: int = int(payload.get("sampler_seed", 42))

    # Rebuild base config inside the worker to avoid pickling function objects
    worker_base_cfg = RobustnessConfig(
        alpha_disc=0.1,
        alpha_nn=float(payload["alpha_nn"]),
        cql_alpha=float(payload["cql_alpha"]),
        min_calib=50,
        num_experiments=len(seeds),
        num_eval_episodes=num_eval_episodes,
        n_calib_steps=10_000,
        n_train_steps=int(payload["n_train_steps"]),
        k=50,
        scoring_method=payload["scoring_method"],
        agent_type=payload["agent_type"],
        retrain=bool(payload["retrain"]),
        calib_methods=["ccdisc"] if method == "ccdisc" else ["ccnn"],
    )

    objective = build_objective(
        method=method,
        env_name=env_name,
        nominal_only=nominal_only,
        num_eval_episodes=num_eval_episodes,
        seeds=seeds,
        base_cfg=worker_base_cfg,
        buffer_dir=buffer_dir,
    )
    storage = JournalStorage(JournalFileBackend(file_path=journal_path))
    sampler = TPESampler(multivariate=True, n_startup_trials=5, seed=sampler_seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    # Use single-thread in each process
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)


def main() -> None:
    args = parse_args()
    base_cfg = build_base_config_from_args(args)

    seeds = list(range(args.seed_offset, args.seed_offset + args.num_seeds))
    out_dir = os.path.join("results", "optuna", args.env_name, f"tuning_{args.method}")
    os.makedirs(out_dir, exist_ok=True)

    # Ensure buffers exist up-front to avoid race conditions across trials
    # Precollect budget: PRECACHE_FACTOR x the upper bound of n_calib_steps
    buffer_dir = _prepare_buffers(
        env_name=args.env_name,
        seeds=seeds,
        base_cfg=base_cfg,
        out_dir=out_dir,
        precollect_n=int(PRECACHE_FACTOR * N_CALIB_STEPS_MAX),
    )

    # If mp-workers > 0, run with JournalStorage + multiprocessing; else use in-memory + threads
    study_name = args.study_name or f"optuna_{args.env_name}_{args.method}"
    journal_path = args.journal_path or os.path.join(out_dir, "optuna_journal.log")
    sampler_seed = 42
    if args.mp_workers and args.mp_workers > 0:
        from multiprocessing import Pool

        # Distribute trials across workers
        mp = min(args.mp_workers, max(1, args.num_trials))
        base = args.num_trials // mp
        rem = args.num_trials % mp
        per_worker = [base + (1 if i < rem else 0) for i in range(mp)]
        per_worker = [n for n in per_worker if n > 0]

        payload_common = dict(
            method=args.method,
            env_name=args.env_name,
            nominal_only=args.nominal_only,
            num_eval_episodes=args.num_eval_episodes,
            seeds=seeds,
            buffer_dir=buffer_dir,
            study_name=study_name,
            journal_path=journal_path,
            sampler_seed=sampler_seed,
            # base config fields
            alpha_nn=args.alpha_nn,
            cql_alpha=args.cql_alpha,
            n_train_steps=args.n_train_steps,
            scoring_method=args.scoring_method,
            agent_type=args.agent_type,
            retrain=args.retrain,
        )

        payloads = [dict(payload_common, n_trials=n) for n in per_worker]

        with Pool(processes=len(per_worker)) as pool:
            pool.map(_mp_worker_entry, payloads)
    else:
        # Threaded in-memory fallback
        objective = build_objective(
            method=args.method,
            env_name=args.env_name,
            nominal_only=args.nominal_only,
            num_eval_episodes=args.num_eval_episodes,
            seeds=seeds,
            base_cfg=base_cfg,
            buffer_dir=buffer_dir,
        )
        sampler = TPESampler(multivariate=True, n_startup_trials=5, seed=sampler_seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name
        )
        n_jobs = args.n_jobs or (os.cpu_count() or 1)
        study.optimize(
            objective, n_trials=args.num_trials, n_jobs=n_jobs, gc_after_trial=True
        )

    # Resolve study handle for reading results
    if args.mp_workers and args.mp_workers > 0:
        storage = JournalStorage(JournalFileBackend(file_path=journal_path))
        study = optuna.load_study(study_name=study_name, storage=storage)
    # else: study is already defined above

    # Persist results
    trials_path = os.path.join(out_dir, "optuna_trials.json")
    trials_payload = [_trial_to_record(t) for t in study.get_trials(deepcopy=False)]
    with open(trials_path, "w") as f:
        json.dump(trials_payload, f, indent=2)

    # Derive best config summary
    best = study.best_trial if study.best_trial is not None else None
    best_entry = None
    if best is not None:
        # Recover per-seed means from intermediate values if available
        # We stored running means at steps 1..num_seeds
        seed_means = []
        if best.intermediate_values:
            # Reconstruct per-step running means and approximate per-seed means by differences
            # Here we simply keep the running means; downstream use cares about aggregate value
            seed_means = [
                best.intermediate_values.get(i + 1)
                for i in range(args.num_seeds)
                if (i + 1) in best.intermediate_values
            ]
        best_entry = dict(
            params=best.params, metric=best.value, running_means=seed_means
        )

    summary = dict(
        env_name=args.env_name,
        method=args.method,
        nominal_only=args.nominal_only,
        num_eval_episodes=args.num_eval_episodes,
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        best=best_entry,
    )

    import yaml  # local import to avoid hard dependency at import time

    with open(os.path.join(out_dir, "best_config.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    # Also print a short summary to stdout
    if best is not None:
        print(f"Best value={best.value:.3f} params={best.params}")
    else:
        print("No completed trials.")


if __name__ == "__main__":
    main()
