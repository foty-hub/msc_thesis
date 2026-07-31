from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from stable_baselines3 import DQN

from crl.calib import (
    compute_corrections,
    fill_calib_sets_mc,
    fill_calib_sets_td,
    signed_score,
)
from crl.experiment import GridCalibration, fit_grid_from_buffer
from crl.types import ScoringMethod

ObjectiveMode = Literal["mean", "median", "mean_median"]


@dataclass(frozen=True)
class GridTuningCandidate:
    alpha: float
    grid_bins: int
    min_calib: int
    n_calib_steps: int
    obs_quantile: float


def partition_seed_scores(
    seed_scores: Mapping[int, float],
    *,
    threshold: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split seeds by whether their mean nominal return reaches a threshold."""
    eligible = tuple(
        seed for seed, score in seed_scores.items() if score >= threshold
    )
    excluded = tuple(
        seed for seed, score in seed_scores.items() if score < threshold
    )
    return eligible, excluded


def choose_shift_indices(
    values: Sequence[float],
    *,
    nominal_value: float,
    count: int,
) -> tuple[int, ...]:
    """Choose spread-out shift indices while retaining endpoints and nominal."""
    shift_values = np.asarray(values, dtype=float)
    if count >= shift_values.size:
        return tuple(range(shift_values.size))

    nominal_index = int(np.argmin(np.abs(shift_values - nominal_value)))
    selected = {0, nominal_index, shift_values.size - 1}
    while len(selected) < count:
        candidates = [
            index
            for index in range(shift_values.size)
            if index not in selected
        ]
        next_index = max(
            candidates,
            key=lambda index: (
                min(abs(index - chosen) for chosen in selected),
                -index,
            ),
        )
        selected.add(next_index)
    return tuple(sorted(selected))


def objective_score(
    seed_deltas: Sequence[float],
    *,
    mode: ObjectiveMode,
    nominal_deltas: Sequence[float] = (),
    nominal_loss_penalty: float = 0.0,
    nominal_loss_tolerance: float = 0.0,
    worst_seed_loss_penalty: float = 0.0,
    worst_seed_loss_tolerance: float = 0.0,
) -> float:
    """Aggregate paired seed deltas with optional safety penalties."""
    deltas = np.asarray(seed_deltas, dtype=float)
    if mode == "mean":
        score = float(np.mean(deltas))
    elif mode == "median":
        score = float(np.median(deltas))
    else:
        score = float(0.5 * np.mean(deltas) + 0.5 * np.median(deltas))

    if nominal_loss_penalty:
        nominal = np.asarray(nominal_deltas, dtype=float)
        excess_nominal_loss = max(
            0.0,
            -float(np.mean(nominal)) - nominal_loss_tolerance,
        )
        score -= nominal_loss_penalty * excess_nominal_loss

    if worst_seed_loss_penalty:
        excess_worst_loss = max(
            0.0,
            -float(np.min(deltas)) - worst_seed_loss_tolerance,
        )
        score -= worst_seed_loss_penalty * excess_worst_loss
    return score


def calibrate_candidate(
    model: DQN,
    transitions: Sequence,
    *,
    n_actions: int,
    candidate: GridTuningCandidate,
    max_calib_per_cell: int,
    scoring_method: ScoringMethod,
    inference_batch_size: int,
) -> GridCalibration:
    """Fit one sparse-grid candidate from a chronological transition prefix."""
    prefix = transitions[: candidate.n_calib_steps]
    discretiser = fit_grid_from_buffer(
        prefix,
        n_bins=candidate.grid_bins,
        n_actions=n_actions,
        obs_quantile=candidate.obs_quantile,
    )
    fill_calibration_sets = {
        "td": fill_calib_sets_td,
        "monte_carlo": fill_calib_sets_mc,
    }[scoring_method]
    calibration_sets = fill_calibration_sets(
        model,
        prefix,
        discretiser,
        maxlen=max_calib_per_cell,
        score=signed_score,
        batch_size=inference_batch_size,
    )

    corrections = compute_corrections(
        calibration_sets,
        alpha=candidate.alpha,
        min_calib=candidate.min_calib,
    )
    return GridCalibration(
        discretiser=discretiser,
        corrections=corrections,
        n_visited_cells=len(calibration_sets),
        n_calibrated_cells=len(corrections) - 1,
        fallback=float(corrections["fallback"]),
    )
