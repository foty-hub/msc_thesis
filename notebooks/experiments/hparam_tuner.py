"""
Hyperparameter tuner for traintime robustness experiments.

Features
- Supports CC-Disc (alpha, tiles, tilings, n_calib_steps, min_calib)
- Supports CCNN (k, ccnn_max_distance_quantile, n_calib_steps)
- Mean-return objective across the shift range, or nominal-only with a flag
- Coarse + refine (top-k) search strategy
- Reuses cached trained models and calibration buffers per seed
- Parallelises across seeds per config

Usage examples
- CC-Disc coarse + refine on LunarLander nominal-only, 5 seeds, 100 eval eps:
  python hparam_tuner.py --env-name LunarLander-v3 --method ccdisc \
    --nominal-only --num-seeds 5 --num-eval-episodes 100

- CCNN on CartPole across shift range, 8 seeds, custom coarse space:
  python hparam_tuner.py --env-name CartPole-v1 --method ccnn \
    --num-seeds 8 --coarse-k 25 50 100 --coarse-n-calib-steps 2000 5000 10000

Notes
- Results are written to results/{env}/tuning_{method}/ with CSV and YAML.
"""

import argparse
import itertools
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
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
# Search space definitions
# -------------------------

# Defaults for CC-Disc (coarse stage)
COARSE_ALPHA = [0.01, 0.05, 0.1, 0.15, 0.25]
COARSE_TILES = [4, 6, 8, 10]
COARSE_TILINGS = [1]
COARSE_N_CALIB_STEPS = [2_000, 5_000, 10_000, 20_000]
COARSE_MIN_CALIB = [25, 50, 100]

# Defaults for CCNN (coarse stage)
COARSE_K = [25, 50, 100]
COARSE_MAX_DIST_Q = [0.85, 0.9, 0.95, 0.99]


@dataclass
class TuningConfig:
    env_name: str
    method: str  # "ccdisc" or "ccnn"
    # Evaluation
    nominal_only: bool = False
    num_eval_episodes: int = 100
    # Search strategy
    refine_top_k: int = 3
    num_seeds: int = 5
    seed_offset: int = 0
    # Core RobustnessConfig that we will copy and adjust per-eval
    base_cfg: RobustnessConfig = field(default_factory=RobustnessConfig)
    # Search spaces (editable)
    coarse_alpha: list[float] = None
    coarse_tiles: list[int] = None
    coarse_tilings: list[int] = None
    coarse_n_calib_steps: list[int] = None
    coarse_min_calib: list[int] = None
    coarse_k: list[int] = None
    coarse_max_dist_q: list[float] = None

    def __post_init__(self):
        # Fill defaults only if not passed
        if self.coarse_alpha is None:
            self.coarse_alpha = list(COARSE_ALPHA)
        if self.coarse_tiles is None:
            self.coarse_tiles = list(COARSE_TILES)
        if self.coarse_tilings is None:
            self.coarse_tilings = list(COARSE_TILINGS)
        if self.coarse_n_calib_steps is None:
            self.coarse_n_calib_steps = list(COARSE_N_CALIB_STEPS)
        if self.coarse_min_calib is None:
            self.coarse_min_calib = list(COARSE_MIN_CALIB)
        if self.coarse_k is None:
            self.coarse_k = list(COARSE_K)
        if self.coarse_max_dist_q is None:
            self.coarse_max_dist_q = list(COARSE_MAX_DIST_Q)


# -------------------------
# Helpers: metrics and utils
# -------------------------


def _mean_return_from_results(
    results: list[dict], key: str, nominal_only: bool, env_name: str
) -> float:
    """Compute mean return across shift range (or nominal-only) from results list.

    results: list of per-shift dicts from run_eval-like outputs.
    key: 'returns_conf' (CC-Disc) or 'returns_ccnn' (CCNN).
    """
    if nominal_only:
        # Evaluate on the nominal env by creating a single eval with no shift
        # If nominal_only was used during evaluation, results length is 1.
        per_shift_means = [np.mean(results[0][key])]
    else:
        per_shift_means = [np.mean(r[key]) for r in results]
    return float(np.mean(per_shift_means))


def _neighbour_int_values(v: int, candidates: Iterable[int]) -> list[int]:
    cand = sorted(set(int(x) for x in candidates))
    if v in cand:
        i = cand.index(v)
        return list(
            sorted(set([cand[max(0, i - 1)], cand[i], cand[min(len(cand) - 1, i + 1)]]))
        )
    # if v outside, pick 2 nearest
    cand_arr = np.array(cand)
    idx = int(np.argmin(np.abs(cand_arr - v)))
    j = int(np.clip(idx + 1, 0, len(cand_arr) - 1))
    return sorted(set([int(cand_arr[idx]), int(cand_arr[j])]))


def _neighbour_float_values(
    v: float, lo: float, hi: float, steps: int = 5
) -> list[float]:
    span = max(1e-6, 0.2 * (hi - lo))
    lo2, hi2 = max(lo, v - span), min(hi, v + span)
    return list(np.linspace(lo2, hi2, num=steps))


# -------------------------
# Evaluation per config
# -------------------------


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
    # Train/load model
    cfg = RobustnessConfig(**{**asdict(base_cfg)})
    model, vec_env = train_agent(env_name, seed, cfg)

    # Build discretiser and load a large precomputed buffer, then slice
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

    # Fill calibration sets (reuse score settings from cfg)
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

    # Evaluate
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

    # Coverage stats
    calibrated_count = max(0, len(qhats) - 1)  # subtract one for the fallback
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
    # Train/load model
    cfg = RobustnessConfig(**{**asdict(base_cfg)})
    model, vec_env = train_agent(env_name, seed, cfg)

    # Load precomputed buffer and slice
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

    # Evaluate
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
            param_val=param_values[0],  # ignored by nominal-only aggregation
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


def _parallel_seed_eval(
    eval_fn: Callable[..., Any],
    seeds: Sequence[int],
    args: tuple[Any, ...],
    max_workers: int | None = None,
) -> list[Any]:
    if max_workers is None:
        max_workers = min(len(seeds), os.cpu_count() or 1)
    out: list[float] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2seed = {ex.submit(eval_fn, s, *args): s for s in seeds}
        for fut in as_completed(fut2seed):
            out.append(fut.result())
    return out


# -------------------------
# Search strategy
# -------------------------


def coarse_configs_ccdisc(cfg: TuningConfig) -> list[dict[str, Any]]:
    combos = itertools.product(
        cfg.coarse_alpha,
        cfg.coarse_tiles,
        cfg.coarse_tilings,
        cfg.coarse_n_calib_steps,
        cfg.coarse_min_calib,
    )
    return [
        dict(alpha=a, tiles=t, tilings=ti, n_calib_steps=n, min_calib=m)
        for (a, t, ti, n, m) in combos
    ]


def refine_configs_ccdisc(
    cfg: TuningConfig, top_cfgs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # Generate small neighbourhoods around top configs
    refined: list[dict[str, Any]] = []
    a_lo, a_hi = float(min(cfg.coarse_alpha)), float(max(cfg.coarse_alpha))
    for c in top_cfgs:
        a_vals = _neighbour_float_values(c["alpha"], lo=a_lo, hi=a_hi, steps=5)
        t_vals = _neighbour_int_values(c["tiles"], cfg.coarse_tiles)
        ti_vals = _neighbour_int_values(c["tilings"], cfg.coarse_tilings)
        # For steps/min_calib explore multiplicative neighbours
        n_cands = sorted(
            set(
                cfg.coarse_n_calib_steps
                + [
                    int(max(1000, c["n_calib_steps"] // 2)),
                    c["n_calib_steps"],
                    int(c["n_calib_steps"] * 2),
                ]
            )
        )
        m_cands = sorted(
            set(
                cfg.coarse_min_calib
                + [max(5, c["min_calib"] // 2), c["min_calib"], c["min_calib"] * 2]
            )
        )
        for a, t, ti, n, m in itertools.product(
            a_vals, t_vals, ti_vals, n_cands, m_cands
        ):
            refined.append(
                dict(
                    alpha=float(a),
                    tiles=int(t),
                    tilings=int(ti),
                    n_calib_steps=int(n),
                    min_calib=int(m),
                )
            )
    # Deduplicate
    seen = set()
    uniq = []
    for c in refined:
        key = (
            round(c["alpha"], 6),
            c["tiles"],
            c["tilings"],
            c["n_calib_steps"],
            c["min_calib"],
        )
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def coarse_configs_ccnn(cfg: TuningConfig) -> list[dict[str, Any]]:
    combos = itertools.product(
        cfg.coarse_k,
        cfg.coarse_max_dist_q,
        cfg.coarse_n_calib_steps,
    )
    return [dict(k=k, max_dist_q=q, n_calib_steps=n) for (k, q, n) in combos]


def refine_configs_ccnn(
    cfg: TuningConfig, top_cfgs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    refined: list[dict[str, Any]] = []
    q_lo, q_hi = float(min(cfg.coarse_max_dist_q)), float(max(cfg.coarse_max_dist_q))
    for c in top_cfgs:
        k_vals = _neighbour_int_values(c["k"], cfg.coarse_k)
        q_vals = _neighbour_float_values(c["max_dist_q"], lo=q_lo, hi=q_hi, steps=5)
        n_cands = sorted(
            set(
                cfg.coarse_n_calib_steps
                + [
                    int(max(1000, c["n_calib_steps"] // 2)),
                    c["n_calib_steps"],
                    int(c["n_calib_steps"] * 2),
                ]
            )
        )
        for k, q, n in itertools.product(k_vals, q_vals, n_cands):
            refined.append(dict(k=int(k), max_dist_q=float(q), n_calib_steps=int(n)))
    # Deduplicate
    seen = set()
    uniq = []
    for c in refined:
        key = (c["k"], round(c["max_dist_q"], 6), c["n_calib_steps"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


# -------------------------
# Orchestration
# -------------------------


def run_search(cfg: TuningConfig) -> dict[str, Any]:
    seeds = list(range(cfg.seed_offset, cfg.seed_offset + cfg.num_seeds))
    out_dir = os.path.join("results", cfg.env_name, f"tuning_{cfg.method}")
    os.makedirs(out_dir, exist_ok=True)
    buf_dir = os.path.join(out_dir, "buffers")
    os.makedirs(buf_dir, exist_ok=True)

    # Stage 1: coarse
    if cfg.method == "ccdisc":
        coarse = coarse_configs_ccdisc(cfg)
    elif cfg.method == "ccnn":
        coarse = coarse_configs_ccnn(cfg)
    else:
        raise ValueError("method must be one of: ccdisc, ccnn")

    # Precompute per-seed large buffers once to avoid repeated env interaction
    # Use a budget that should cover refine stage as well: 2x max coarse steps
    coarse_max_n = max(cfg.coarse_n_calib_steps) if cfg.coarse_n_calib_steps else 10000
    precollect_n = max(1000, int(2 * coarse_max_n))

    # Build buffers serially to keep memory use predictable
    print("Saving replay buffers")
    for seed in seeds:
        buf_path = os.path.join(buf_dir, f"buffer_seed_{seed}.pkl")
        if os.path.exists(buf_path):
            continue
        # Train/load model and collect transitions
        model, vec_env = train_agent(cfg.env_name, seed, cfg.base_cfg)
        buffer = collect_transitions(model, vec_env, n_transitions=precollect_n)
        with open(buf_path, "wb") as f:
            pickle.dump(buffer, f)

    coarse_results = []
    print("Beginning search")
    coarse_total = len(coarse)
    for ix, params in enumerate(coarse, start=1):
        if cfg.method == "ccdisc":
            alpha, tiles, tilings, n_steps, mcal = (
                params["alpha"],
                params["tiles"],
                params["tilings"],
                params["n_calib_steps"],
                params["min_calib"],
            )
            args = (
                cfg.env_name,
                cfg.num_eval_episodes,
                tiles,
                tilings,
                n_steps,
                alpha,
                mcal,
                cfg.base_cfg,
                cfg.nominal_only,
                buf_dir,
            )
            seed_stats = _parallel_seed_eval(_eval_ccdisc_single_seed, seeds, args)
            seed_means = [float(s["mean_return"]) for s in seed_stats]
            mean_across_seeds = float(np.mean(seed_means))
            frac_list = [float(s["calib_fraction"]) for s in seed_stats]
            cnt_list = [int(s["calib_count"]) for s in seed_stats]
            total_bins = int(seed_stats[0]["total_bins"]) if seed_stats else None
            entry = dict(
                params=params,
                metric=mean_across_seeds,
                seed_means=seed_means,
                calib_fraction_mean=float(np.mean(frac_list)),
                calib_fraction_seeds=frac_list,
                calib_count_mean=float(np.mean(cnt_list)),
                calib_count_seeds=cnt_list,
                total_bins=total_bins,
            )
            coarse_results.append(entry)
            print(
                f"[coarse {ix}/{coarse_total}] mean={mean_across_seeds:.2f} "
                f"calib={entry['calib_fraction_mean'] * 100:.1f}% params={params}",
                flush=True,
            )
        else:  # ccnn
            k, q, n_steps = params["k"], params["max_dist_q"], params["n_calib_steps"]
            args = (
                cfg.env_name,
                cfg.num_eval_episodes,
                n_steps,
                k,
                q,
                cfg.base_cfg,
                cfg.nominal_only,
                buf_dir,
            )
            seed_means = _parallel_seed_eval(_eval_ccnn_single_seed, seeds, args)
            mean_across_seeds = float(np.mean(seed_means))
            entry = dict(params=params, metric=mean_across_seeds, seed_means=seed_means)
            coarse_results.append(entry)
            print(
                f"[coarse {ix}/{coarse_total}] mean={mean_across_seeds:.2f} params={params}",
                flush=True,
            )

        # Persist incremental progress
        with open(os.path.join(out_dir, "coarse_results.json"), "w") as f:
            json.dump(coarse_results, f, indent=2)

    # Pick top-k
    top_k = sorted(coarse_results, key=lambda d: d["metric"], reverse=True)[
        : cfg.refine_top_k
    ]
    top_cfgs = [r["params"] for r in top_k]

    # Stage 2: refine
    if cfg.method == "ccdisc":
        refine = refine_configs_ccdisc(cfg, top_cfgs)
    else:
        refine = refine_configs_ccnn(cfg, top_cfgs)

    refine_results = []
    refine_total = len(refine)
    for ix, params in enumerate(refine, start=1):
        if cfg.method == "ccdisc":
            alpha, tiles, tilings, n_steps, mcal = (
                params["alpha"],
                params["tiles"],
                params["tilings"],
                params["n_calib_steps"],
                params["min_calib"],
            )
            args = (
                cfg.env_name,
                cfg.num_eval_episodes,
                tiles,
                tilings,
                n_steps,
                alpha,
                mcal,
                cfg.base_cfg,
                cfg.nominal_only,
                buf_dir,
            )
            seed_stats = _parallel_seed_eval(_eval_ccdisc_single_seed, seeds, args)
            seed_means = [float(s["mean_return"]) for s in seed_stats]
            mean_across_seeds = float(np.mean(seed_means))
            frac_list = [float(s["calib_fraction"]) for s in seed_stats]
            cnt_list = [int(s["calib_count"]) for s in seed_stats]
            total_bins = int(seed_stats[0]["total_bins"]) if seed_stats else None
            entry = dict(
                params=params,
                metric=mean_across_seeds,
                seed_means=seed_means,
                calib_fraction_mean=float(np.mean(frac_list)),
                calib_fraction_seeds=frac_list,
                calib_count_mean=float(np.mean(cnt_list)),
                calib_count_seeds=cnt_list,
                total_bins=total_bins,
            )
            refine_results.append(entry)
            print(
                f"[refine {ix}/{refine_total}] mean={mean_across_seeds:.2f} "
                f"calib={entry['calib_fraction_mean'] * 100:.1f}% params={params}",
                flush=True,
            )
        else:
            k, q, n_steps = params["k"], params["max_dist_q"], params["n_calib_steps"]
            args = (
                cfg.env_name,
                cfg.num_eval_episodes,
                n_steps,
                k,
                q,
                cfg.base_cfg,
                cfg.nominal_only,
                buf_dir,
            )
            seed_means = _parallel_seed_eval(_eval_ccnn_single_seed, seeds, args)
            mean_across_seeds = float(np.mean(seed_means))
            entry = dict(params=params, metric=mean_across_seeds, seed_means=seed_means)
            refine_results.append(entry)
            print(
                f"[refine {ix}/{refine_total}] mean={mean_across_seeds:.2f} params={params}",
                flush=True,
            )

        with open(os.path.join(out_dir, "refine_results.json"), "w") as f:
            json.dump(refine_results, f, indent=2)

    # Select best overall
    all_results = coarse_results + refine_results
    best = max(all_results, key=lambda d: d["metric"]) if all_results else None
    summary = dict(
        env_name=cfg.env_name,
        method=cfg.method,
        nominal_only=cfg.nominal_only,
        num_eval_episodes=cfg.num_eval_episodes,
        num_seeds=cfg.num_seeds,
        seed_offset=cfg.seed_offset,
        best=best,
    )

    with open(os.path.join(out_dir, "best_config.yaml"), "w") as f:
        import yaml

        yaml.safe_dump(summary, f, sort_keys=False)

    return summary


def parse_args() -> argparse.Namespace:
    defaults = RobustnessConfig()
    p = argparse.ArgumentParser(
        description="Hyperparameter tuner for traintime robustness."
    )
    p.add_argument(
        "--env-name", default="CartPole-v1", choices=sorted(EVAL_PARAMETERS.keys())
    )
    p.add_argument("--method", choices=["ccdisc", "ccnn"], default="ccdisc")
    p.add_argument("--nominal-only", action="store_true")
    p.add_argument("--num-eval-episodes", type=int, default=100)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--refine-top-k", type=int, default=3)

    # Base config knobs (forwarded to RobustnessConfig)
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

    # Coarse spaces (override defaults)
    p.add_argument("--coarse-alpha", type=float, nargs="+")
    p.add_argument("--coarse-tiles", type=int, nargs="+")
    p.add_argument("--coarse-tilings", type=int, nargs="+")
    p.add_argument("--coarse-n-calib-steps", type=int, nargs="+")
    p.add_argument("--coarse-min-calib", type=int, nargs="+")
    p.add_argument("--coarse-k", type=int, nargs="+")
    p.add_argument("--coarse-max-dist-q", type=float, nargs="+")

    return p.parse_args()


def build_tuning_config(args: argparse.Namespace) -> TuningConfig:
    base = RobustnessConfig(
        alpha_disc=0.1,  # not used directly for CC-Disc metric computation
        alpha_nn=args.alpha_nn,
        cql_alpha=args.cql_alpha,
        min_calib=50,  # overridden per-eval
        num_experiments=args.num_seeds,  # not used here; we manage seeds
        num_eval_episodes=args.num_eval_episodes,
        n_calib_steps=10_000,  # overridden per-eval
        n_train_steps=args.n_train_steps,
        k=50,  # for CCNN calibration; overridden per-eval
        scoring_method=args.scoring_method,
        agent_type=args.agent_type,
        retrain=args.retrain,
        calib_methods=["ccdisc"] if args.method == "ccdisc" else ["ccnn"],
    )

    tcfg = TuningConfig(
        env_name=args.env_name,
        method=args.method,
        nominal_only=args.nominal_only,
        num_eval_episodes=args.num_eval_episodes,
        refine_top_k=args.refine_top_k,
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        base_cfg=base,
        coarse_alpha=args.coarse_alpha,
        coarse_tiles=args.coarse_tiles,
        coarse_tilings=args.coarse_tilings,
        coarse_n_calib_steps=args.coarse_n_calib_steps,
        coarse_min_calib=args.coarse_min_calib,
        coarse_k=args.coarse_k,
        coarse_max_dist_q=args.coarse_max_dist_q,
    )
    return tcfg


def main() -> None:
    args = parse_args()
    tcfg = build_tuning_config(args)
    summary = run_search(tcfg)
    # Pretty print to stdout
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
