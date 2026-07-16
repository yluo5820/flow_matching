"""Representation-matched local functional audit for Observation 0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from fm_lab.diagnostics.long_tail_geometry.functional_audit_preregistration import (
    FunctionalGeometryAuditPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.functional_calibration import (
    projected_descent_direction,
    top_centered_covariance_direction,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import GradientRows


@dataclass(frozen=True)
class PairedAuditDirection:
    """One fitted basis and its orientation under the shared raw scale gradient."""

    basis_kind: str
    basis_vector: torch.Tensor
    vector: torch.Tensor
    eigenvalue: float
    explained_fraction: float
    projection_fraction: float
    fit_row_norm_cv: float
    basis_vector_sha256: str
    vector_sha256: str
    orientation_gradient_sha256: str


@dataclass(frozen=True)
class FunctionalGeometryAuditDecision:
    """Non-unlocking interpretation of the representation-matched audit."""

    stage1_unlocked: bool
    probe_b_opened: bool
    status: str
    layer_summaries: dict[str, dict[str, float | int | bool | None]]
    next_action: str


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().contiguous().cpu().numpy()
    hasher = hashlib.sha256()
    hasher.update(values.dtype.str.encode("ascii"))
    hasher.update(json.dumps(list(values.shape)).encode("ascii"))
    hasher.update(values.tobytes())
    return hasher.hexdigest()


def paired_projected_directions(
    rows: GradientRows,
    *,
    fit_positions: tuple[int, ...],
    scale_positions: tuple[int, ...],
    minimum_projection_fraction: float,
) -> dict[str, PairedAuditDirection]:
    """Fit raw and normalized bases, oriented by one raw scale mean."""

    if rows.raw.ndim != 2 or rows.normalized.shape != rows.raw.shape:
        raise ValueError("Paired audit gradient rows must be aligned matrices.")
    if rows.norms.shape != (rows.raw.shape[0],):
        raise ValueError("Paired audit row norms must align with gradient rows.")
    fit = tuple(int(value) for value in fit_positions)
    scale = tuple(int(value) for value in scale_positions)
    if len(fit) < 2 or not scale:
        raise ValueError("Paired audit requires fit and scale positions.")
    if len(set(fit)) != len(fit) or len(set(scale)) != len(scale):
        raise ValueError("Paired audit positions must be unique.")
    if set(fit) & set(scale):
        raise ValueError("Paired audit fit and scale positions must be disjoint.")
    if any(value < 0 or value >= rows.raw.shape[0] for value in fit + scale):
        raise ValueError("Paired audit positions are out of range.")
    if (
        not torch.isfinite(rows.raw).all()
        or not torch.isfinite(rows.normalized).all()
        or not torch.isfinite(rows.norms).all()
    ):
        raise ValueError("Paired audit gradient rows must be finite.")

    orientation = rows.raw[list(scale)].mean(dim=0)
    orientation_digest = _tensor_sha256(orientation)
    fit_norms = rows.norms[list(fit)].float()
    fit_norm_cv = float(
        fit_norms.std(unbiased=False) / fit_norms.mean()
    )
    results: dict[str, PairedAuditDirection] = {}
    for basis_kind, matrix in (
        ("raw", rows.raw),
        ("row_normalized", rows.normalized),
    ):
        rank1 = top_centered_covariance_direction(matrix[list(fit)])
        projected = projected_descent_direction(
            rank1.vector,
            orientation,
            minimum_projection_fraction=minimum_projection_fraction,
        )
        results[basis_kind] = PairedAuditDirection(
            basis_kind=basis_kind,
            basis_vector=rank1.vector,
            vector=projected.vector,
            eigenvalue=rank1.eigenvalue,
            explained_fraction=rank1.explained_fraction,
            projection_fraction=projected.projection_fraction,
            fit_row_norm_cv=fit_norm_cv,
            basis_vector_sha256=_tensor_sha256(rank1.vector),
            vector_sha256=_tensor_sha256(projected.vector),
            orientation_gradient_sha256=orientation_digest,
        )
    return results


def relative_benefit_slope(
    *,
    direction: torch.Tensor,
    evaluation_mean_gradient: torch.Tensor,
    parameter_norm: float,
    base_loss: float,
) -> float:
    """Derivative of relative benefit with respect to relative layer step."""

    if direction.ndim != 1 or evaluation_mean_gradient.ndim != 1:
        raise ValueError("Direction and evaluation mean gradient must be vectors.")
    if direction.shape != evaluation_mean_gradient.shape:
        raise ValueError("Direction and evaluation mean gradient must have the same shape.")
    direction = direction.detach().float().cpu()
    gradient = evaluation_mean_gradient.detach().float().cpu()
    if not torch.isfinite(direction).all() or not torch.isfinite(gradient).all():
        raise ValueError("Direction and evaluation mean gradient must be finite.")
    norm = torch.linalg.vector_norm(direction)
    if not torch.isclose(
        norm,
        torch.ones_like(norm),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Relative-benefit direction must have unit norm.")
    if not np.isfinite(parameter_norm) or float(parameter_norm) <= 0:
        raise ValueError("Relative-benefit parameter norm must be positive and finite.")
    if not np.isfinite(base_loss) or float(base_loss) <= 0:
        raise ValueError("Relative-benefit base loss must be positive and finite.")
    return -float(parameter_norm) * float(torch.dot(gradient, direction)) / float(
        base_loss
    )


_SLOPE_KEYS = (
    "checkpoint_step",
    "layer",
    "seed",
    "fold",
    "basis_kind",
    "direction_class",
    "evaluation_class",
)
_FINITE_KEYS = (
    "checkpoint_step",
    "layer",
    "seed",
    "fold",
    "basis_kind",
    "direction_class",
    "partition",
    "relative_step",
)


def analyze_functional_geometry_audit(
    slopes: pd.DataFrame,
    finite_steps: pd.DataFrame,
    preregistration: FunctionalGeometryAuditPreregistration,
) -> FunctionalGeometryAuditDecision:
    """Analyze complete cross-fitted tables without creating an unlock path."""

    _validate_slope_table(slopes, preregistration)
    _validate_finite_table(finite_steps, preregistration)
    fold_blocks = _fold_slope_blocks(slopes)
    blocks = (
        fold_blocks.groupby(
            [
                "checkpoint_step",
                "layer",
                "basis_kind",
                "seed",
                "direction_class",
            ],
            as_index=False,
        )[["target_slope", "selectivity_slope"]]
        .median()
    )
    finite_errors = _finite_evaluation_errors(slopes, finite_steps)
    primary = preregistration.primary_checkpoint_step
    layer_summaries: dict[str, dict[str, float | int | bool | None]] = {}
    for layer_index, layer in enumerate(preregistration.layers):
        summary: dict[str, float | int | bool | None] = {}
        for basis_index, basis_kind in enumerate(preregistration.basis_kinds):
            selected = blocks[
                (blocks["checkpoint_step"] == primary)
                & (blocks["layer"] == layer)
                & (blocks["basis_kind"] == basis_kind)
            ]
            values = selected["target_slope"].to_numpy(dtype=np.float64)
            lower, upper = _bootstrap_median_interval(
                values,
                resamples=preregistration.bootstrap_resamples,
                seed=(
                    preregistration.bootstrap_seed
                    + 100 * layer_index
                    + basis_index
                ),
                confidence_level=preregistration.confidence_level,
            )
            seed_repeats = int(
                (
                    selected.groupby("seed")["target_slope"].median()
                    > 0
                ).sum()
            )
            selectivity_median = float(selected["selectivity_slope"].median())
            prefix = "normalized" if basis_kind == "row_normalized" else "raw"
            summary[f"{prefix}_target_slope_median"] = float(np.median(values))
            summary[f"{prefix}_target_slope_ci_lower"] = lower
            summary[f"{prefix}_target_slope_ci_upper"] = upper
            summary[f"{prefix}_selectivity_slope_median"] = selectivity_median
            summary[f"{prefix}_positive_seed_repeats"] = seed_repeats
            summary[f"{prefix}_positive_local_transport"] = bool(
                lower > 0
                and selectivity_median > 0
                and seed_repeats >= preregistration.required_seed_repeats
            )
            summary[f"{prefix}_largest_concordant_evaluation_step"] = (
                _largest_concordant_step(
                    finite_errors,
                    checkpoint_step=primary,
                    layer=layer,
                    basis_kind=basis_kind,
                    tolerance=preregistration.local_linearity_relative_error_max,
                )
            )

        paired = _paired_block_differences(blocks, primary, layer)
        paired_lower, paired_upper = _bootstrap_median_interval(
            paired,
            resamples=preregistration.bootstrap_resamples,
            seed=preregistration.bootstrap_seed + 100 * layer_index + 9,
            confidence_level=preregistration.confidence_level,
        )
        summary["normalized_minus_raw_target_slope_median"] = float(
            np.median(paired)
        )
        summary["normalized_minus_raw_target_slope_ci_lower"] = paired_lower
        summary["normalized_minus_raw_target_slope_ci_upper"] = paired_upper
        summary["normalized_paired_advantage"] = bool(paired_lower > 0)
        layer_summaries[layer] = summary

    normalized_positive = all(
        bool(summary["normalized_positive_local_transport"])
        for summary in layer_summaries.values()
    )
    raw_positive = all(
        bool(summary["raw_positive_local_transport"])
        for summary in layer_summaries.values()
    )
    paired_advantage = all(
        bool(summary["normalized_paired_advantage"])
        for summary in layer_summaries.values()
    )
    normalized_nonpositive = all(
        float(summary["normalized_target_slope_ci_upper"]) <= 0
        for summary in layer_summaries.values()
    )
    if normalized_positive and paired_advantage:
        status = "normalized_representation_rescue"
        next_action = "review_separate_small_local_step_preregistration"
    elif normalized_positive and raw_positive:
        status = "representation_independent_local_transport"
        next_action = "study_partition_and_finite_step_curvature"
    elif normalized_nonpositive:
        status = "no_transferable_local_descent"
        next_action = "pivot_to_gradient_sign_transport_failure"
    else:
        status = "mixed_or_class_heterogeneous_transport"
        next_action = "analyze_class_and_seed_transport_heterogeneity"
    return FunctionalGeometryAuditDecision(
        stage1_unlocked=False,
        probe_b_opened=False,
        status=status,
        layer_summaries=layer_summaries,
        next_action=next_action,
    )


def _validate_slope_table(
    frame: pd.DataFrame,
    preregistration: FunctionalGeometryAuditPreregistration,
) -> None:
    required = set(_SLOPE_KEYS) | {"base_loss", "parameter_norm", "benefit_slope"}
    if not required.issubset(frame.columns):
        raise ValueError("Functional geometry audit has an incomplete slope table.")
    if frame.duplicated(list(_SLOPE_KEYS)).any():
        raise ValueError("Functional geometry audit has duplicate slope cells.")
    numeric = frame[[
        "checkpoint_step",
        "seed",
        "fold",
        "direction_class",
        "evaluation_class",
        "base_loss",
        "parameter_norm",
        "benefit_slope",
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Functional geometry audit slope values must be finite.")
    expected = pd.MultiIndex.from_product(
        [
            preregistration.checkpoint_steps,
            preregistration.layers,
            (0, 1, 2),
            range(len(preregistration.fold_offsets)),
            preregistration.basis_kinds,
            preregistration.classes,
            preregistration.classes,
        ],
        names=_SLOPE_KEYS,
    )
    observed = pd.MultiIndex.from_frame(frame[list(_SLOPE_KEYS)])
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("Functional geometry audit requires a complete slope table.")


def _validate_finite_table(
    frame: pd.DataFrame,
    preregistration: FunctionalGeometryAuditPreregistration,
) -> None:
    required = set(_FINITE_KEYS) | {"base_loss", "perturbed_loss", "benefit"}
    if not required.issubset(frame.columns):
        raise ValueError("Functional geometry audit has an incomplete finite-step table.")
    if frame.duplicated(list(_FINITE_KEYS)).any():
        raise ValueError("Functional geometry audit has duplicate finite-step cells.")
    numeric = frame[[
        "checkpoint_step",
        "seed",
        "fold",
        "direction_class",
        "relative_step",
        "base_loss",
        "perturbed_loss",
        "benefit",
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Functional geometry audit finite-step values must be finite.")
    expected = pd.MultiIndex.from_product(
        [
            preregistration.checkpoint_steps,
            preregistration.layers,
            (0, 1, 2),
            range(len(preregistration.fold_offsets)),
            preregistration.basis_kinds,
            preregistration.classes,
            ("scale", "evaluation"),
            preregistration.relative_step_grid,
        ],
        names=_FINITE_KEYS,
    )
    observed = pd.MultiIndex.from_frame(frame[list(_FINITE_KEYS)])
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError(
            "Functional geometry audit requires a complete finite-step table."
        )


def _fold_slope_blocks(slopes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "checkpoint_step",
        "layer",
        "seed",
        "fold",
        "basis_kind",
        "direction_class",
    ]
    for key, block in slopes.groupby(keys, sort=False):
        direction_class = int(key[-1])
        target = block[block["evaluation_class"] == direction_class]
        offclass = block[block["evaluation_class"] != direction_class]
        if len(target) != 1 or offclass.empty:
            raise ValueError("Functional geometry audit slope response block is incomplete.")
        target_slope = float(target.iloc[0]["benefit_slope"])
        harm = max(0.0, -float(offclass["benefit_slope"].min()))
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "target_slope": target_slope,
                "selectivity_slope": target_slope - harm,
            }
        )
    return pd.DataFrame(rows)


def _finite_evaluation_errors(
    slopes: pd.DataFrame,
    finite_steps: pd.DataFrame,
) -> pd.DataFrame:
    target = slopes[
        slopes["direction_class"] == slopes["evaluation_class"]
    ][
        [
            "checkpoint_step",
            "layer",
            "seed",
            "fold",
            "basis_kind",
            "direction_class",
            "benefit_slope",
        ]
    ]
    evaluation = finite_steps[finite_steps["partition"] == "evaluation"].merge(
        target,
        on=[
            "checkpoint_step",
            "layer",
            "seed",
            "fold",
            "basis_kind",
            "direction_class",
        ],
        validate="many_to_one",
    )
    prediction = evaluation["relative_step"] * evaluation["benefit_slope"]
    evaluation = evaluation.copy()
    evaluation["linearity_relative_error"] = (
        (evaluation["benefit"] - prediction).abs()
        / np.maximum(prediction.abs(), 1e-12)
    )
    return evaluation


def _largest_concordant_step(
    errors: pd.DataFrame,
    *,
    checkpoint_step: int,
    layer: str,
    basis_kind: str,
    tolerance: float,
) -> float | None:
    selected = errors[
        (errors["checkpoint_step"] == checkpoint_step)
        & (errors["layer"] == layer)
        & (errors["basis_kind"] == basis_kind)
    ]
    medians = selected.groupby("relative_step")[
        "linearity_relative_error"
    ].median()
    valid = medians[medians <= tolerance]
    return None if valid.empty else float(valid.index.max())


def _paired_block_differences(
    blocks: pd.DataFrame,
    checkpoint_step: int,
    layer: str,
) -> np.ndarray:
    selected = blocks[
        (blocks["checkpoint_step"] == checkpoint_step)
        & (blocks["layer"] == layer)
    ]
    index = ["seed", "direction_class"]
    raw = selected[selected["basis_kind"] == "raw"][index + ["target_slope"]]
    normalized = selected[selected["basis_kind"] == "row_normalized"][
        index + ["target_slope"]
    ]
    paired = normalized.merge(
        raw,
        on=index,
        suffixes=("_normalized", "_raw"),
        validate="one_to_one",
    )
    return (
        paired["target_slope_normalized"] - paired["target_slope_raw"]
    ).to_numpy(dtype=np.float64)


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Functional geometry audit bootstrap values are invalid.")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    medians = np.median(values[indices], axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    return float(np.quantile(medians, alpha)), float(
        np.quantile(medians, 1.0 - alpha)
    )
