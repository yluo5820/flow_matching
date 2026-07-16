"""Representation-matched local functional audit for Observation 0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.functional_audit_preregistration import (
    FunctionalGeometryAuditPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.functional_calibration import (
    CalibrationContext,
    _file_sha256,
    _load_completed_decision,
    _load_probe_components,
    _write_dataframe_atomic,
    _write_json_atomic,
    cell_microbatch_rows,
    prepare_calibration_context,
    projected_descent_direction,
    top_centered_covariance_direction,
    virtual_layer_update,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import (
    GradientRows,
    collect_gradient_rows,
    resolve_probe_layers,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeBatch,
    materialize_probe_batch,
)


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


@dataclass(frozen=True)
class FunctionalGeometryAuditChunk:
    """Complete metrics for one seed and checkpoint."""

    slopes: pd.DataFrame
    finite_steps: pd.DataFrame
    basis_comparison: pd.DataFrame


@dataclass(frozen=True)
class FunctionalGeometryAuditContext:
    """Validated audit inputs and its isolated artifact directory."""

    study_dir: Path
    artifact_dir: Path
    calibration: CalibrationContext | Any
    preregistration: FunctionalGeometryAuditPreregistration
    input_digests: dict[str, str]
    functional_lock_sha256: str


@dataclass(frozen=True)
class FunctionalGeometryAuditResult:
    """Completed non-unlocking audit and its artifact directory."""

    artifact_dir: Path
    decision: FunctionalGeometryAuditDecision


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


def validate_blocked_functional_lock(
    lock: dict[str, Any],
    preregistration: FunctionalGeometryAuditPreregistration,
) -> None:
    """Require the exact failed calibration state that motivated this audit."""

    valid = bool(
        lock.get("stage1_unlocked") is preregistration.required_stage1_unlocked
        and lock.get("probe_view") == "a"
        and lock.get("probe_b_opened") is False
        and lock.get("next_action")
        == preregistration.required_functional_next_action
        and lock.get("functional_preregistration_sha256")
        == preregistration.functional_preregistration_sha256
    )
    if not valid:
        raise ValueError(
            "Functional geometry audit requires the exact blocked functional calibration."
        )


def collect_audit_metrics(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    batches_by_class: dict[int, tuple[ProbeBatch, ...]],
    preregistration: FunctionalGeometryAuditPreregistration,
    seed: int,
    checkpoint_step: int,
) -> FunctionalGeometryAuditChunk:
    """Collect paired slopes and finite responses from one loaded checkpoint."""

    if set(batches_by_class) != set(preregistration.classes):
        raise ValueError("Functional geometry audit requires every locked class.")
    if any(
        len(batches) != preregistration.microbatches_per_cell
        for batches in batches_by_class.values()
    ):
        raise ValueError("Functional geometry audit requires all cell microbatches.")
    if int(checkpoint_step) not in preregistration.checkpoint_steps:
        raise ValueError("Functional geometry audit checkpoint is outside the contract.")

    gradients = {
        class_id: collect_gradient_rows(
            model=model,
            objective=objective,
            path=path,
            batches=batches,
            layer_names=preregistration.layers,
        )
        for class_id, batches in batches_by_class.items()
    }
    layers = resolve_probe_layers(model, preregistration.layers)
    parameter_norms = {
        layer.name: float(torch.linalg.vector_norm(layer.parameter.detach()))
        for layer in layers
    }
    slope_rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        for fold, positions in enumerate(preregistration.fold_positions):
            scale_positions = positions["scale"]
            evaluation_positions = positions["evaluation"]
            partition_batches = {
                class_id: {
                    "scale": tuple(batches[index] for index in scale_positions),
                    "evaluation": tuple(
                        batches[index] for index in evaluation_positions
                    ),
                }
                for class_id, batches in batches_by_class.items()
            }
            base_losses = {
                (class_id, partition): _mean_objective_loss(
                    model=model,
                    objective=objective,
                    path=path,
                    batches=batches,
                )
                for class_id, values in partition_batches.items()
                for partition, batches in values.items()
            }
            for direction_class in preregistration.classes:
                for layer in preregistration.layers:
                    paired = paired_projected_directions(
                        gradients[direction_class][layer],
                        fit_positions=positions["fit"],
                        scale_positions=scale_positions,
                        minimum_projection_fraction=(
                            preregistration.minimum_projection_fraction
                        ),
                    )
                    basis_cosine = abs(
                        float(
                            torch.dot(
                                paired["raw"].basis_vector,
                                paired["row_normalized"].basis_vector,
                            )
                        )
                    )
                    for basis_kind, direction in paired.items():
                        basis_rows.append(
                            {
                                "checkpoint_step": int(checkpoint_step),
                                "layer": layer,
                                "seed": int(seed),
                                "fold": int(fold),
                                "basis_kind": basis_kind,
                                "direction_class": int(direction_class),
                                "fit_explained_fraction": (
                                    direction.explained_fraction
                                ),
                                "projection_fraction": direction.projection_fraction,
                                "fit_row_norm_cv": direction.fit_row_norm_cv,
                                "basis_vector_sha256": (
                                    direction.basis_vector_sha256
                                ),
                                "direction_vector_sha256": direction.vector_sha256,
                                "orientation_gradient_sha256": (
                                    direction.orientation_gradient_sha256
                                ),
                                "raw_normalized_basis_abs_cosine": basis_cosine,
                            }
                        )
                        target_evaluation_slope: float | None = None
                        for evaluation_class in preregistration.classes:
                            evaluation_gradient = gradients[evaluation_class][
                                layer
                            ].raw[list(evaluation_positions)].mean(dim=0)
                            base_loss = base_losses[
                                (evaluation_class, "evaluation")
                            ]
                            slope = relative_benefit_slope(
                                direction=direction.vector,
                                evaluation_mean_gradient=evaluation_gradient,
                                parameter_norm=parameter_norms[layer],
                                base_loss=base_loss,
                            )
                            if evaluation_class == direction_class:
                                target_evaluation_slope = slope
                            slope_rows.append(
                                {
                                    "checkpoint_step": int(checkpoint_step),
                                    "layer": layer,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "basis_kind": basis_kind,
                                    "direction_class": int(direction_class),
                                    "evaluation_class": int(evaluation_class),
                                    "base_loss": base_loss,
                                    "parameter_norm": parameter_norms[layer],
                                    "benefit_slope": slope,
                                }
                            )
                        if target_evaluation_slope is None:
                            raise RuntimeError("Audit target slope was not collected.")
                        scale_gradient = gradients[direction_class][layer].raw[
                            list(scale_positions)
                        ].mean(dim=0)
                        scale_slope = relative_benefit_slope(
                            direction=direction.vector,
                            evaluation_mean_gradient=scale_gradient,
                            parameter_norm=parameter_norms[layer],
                            base_loss=base_losses[(direction_class, "scale")],
                        )
                        for partition, predicted_slope in (
                            ("scale", scale_slope),
                            ("evaluation", target_evaluation_slope),
                        ):
                            base_loss = base_losses[(direction_class, partition)]
                            batches = partition_batches[direction_class][partition]
                            for relative_step in preregistration.relative_step_grid:
                                with virtual_layer_update(
                                    model,
                                    layer_name=layer,
                                    direction=direction.vector,
                                    relative_step=relative_step,
                                ):
                                    perturbed_loss = _mean_objective_loss(
                                        model=model,
                                        objective=objective,
                                        path=path,
                                        batches=batches,
                                    )
                                finite_rows.append(
                                    {
                                        "checkpoint_step": int(checkpoint_step),
                                        "layer": layer,
                                        "seed": int(seed),
                                        "fold": int(fold),
                                        "basis_kind": basis_kind,
                                        "direction_class": int(direction_class),
                                        "partition": partition,
                                        "relative_step": float(relative_step),
                                        "base_loss": base_loss,
                                        "perturbed_loss": perturbed_loss,
                                        "benefit": -(
                                            perturbed_loss - base_loss
                                        )
                                        / base_loss,
                                        "predicted_slope": predicted_slope,
                                    }
                                )
    finally:
        model.train(was_training)
    return FunctionalGeometryAuditChunk(
        slopes=pd.DataFrame(slope_rows).sort_values(list(_SLOPE_KEYS)).reset_index(
            drop=True
        ),
        finite_steps=pd.DataFrame(finite_rows)
        .sort_values(list(_FINITE_KEYS))
        .reset_index(drop=True),
        basis_comparison=pd.DataFrame(basis_rows)
        .sort_values(
            [
                "checkpoint_step",
                "layer",
                "seed",
                "fold",
                "basis_kind",
                "direction_class",
            ]
        )
        .reset_index(drop=True),
    )


def _mean_objective_loss(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    batches: tuple[ProbeBatch, ...],
) -> float:
    if not batches:
        raise ValueError("Functional geometry audit loss requires probe batches.")
    weighted_loss = 0.0
    row_count = 0
    with torch.no_grad():
        for batch in batches:
            loss, _ = objective(
                model=model,
                path=path,
                x0=batch.x0,
                x1=batch.x1,
                t=batch.t,
                compute_diagnostics=False,
                class_labels=batch.labels,
                original_class_labels=batch.labels,
            )
            count = int(len(batch.labels))
            weighted_loss += float(loss.detach().cpu()) * count
            row_count += count
    result = weighted_loss / row_count
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Functional geometry audit requires a positive finite loss.")
    return result


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


def file_sha256(path: str | Path) -> str:
    """Return the content digest used by the public audit artifact contract."""

    return _file_sha256(path)


def prepare_functional_geometry_audit_context(
    *,
    study_dir: str | Path,
    audit_preregistration_path: str | Path,
) -> FunctionalGeometryAuditContext:
    """Validate the blocked functional calibration without opening Probe B."""

    root = Path(study_dir)
    preregistration = FunctionalGeometryAuditPreregistration.load(
        audit_preregistration_path
    )
    functional_dir = root / "aggregate" / "functional_calibration"
    functional_preregistration_path = functional_dir / "preregistration.yaml"
    calibration = prepare_calibration_context(
        study_dir=root,
        calibration_preregistration_path=functional_preregistration_path,
    )
    if calibration.observation0.digest != (
        preregistration.observation0_preregistration_sha256
    ):
        raise ValueError("Functional geometry audit Observation-0 identity changed.")
    if calibration.preregistration.digest != (
        preregistration.functional_preregistration_sha256
    ):
        raise ValueError("Functional geometry audit calibration identity changed.")
    scope_pairs = (
        (calibration.preregistration.checkpoint_steps, preregistration.checkpoint_steps),
        (calibration.preregistration.layers, preregistration.layers),
        (calibration.preregistration.classes, preregistration.classes),
    )
    if any(tuple(actual) != tuple(expected) for actual, expected in scope_pairs):
        raise ValueError("Functional geometry audit scope differs from calibration.")
    if (
        calibration.preregistration.probe_view != preregistration.probe_view
        or calibration.preregistration.stratum_id != preregistration.stratum_id
        or calibration.preregistration.rank != preregistration.rank
        or calibration.preregistration.microbatches_per_cell
        != preregistration.microbatches_per_cell
    ):
        raise ValueError("Functional geometry audit probe scope differs from calibration.")

    _load_completed_decision(calibration)
    lock_path = functional_dir / "functional_lock.json"
    lock = json.loads(lock_path.read_text())
    validate_blocked_functional_lock(lock, preregistration)
    names = (
        "preregistration.yaml",
        "scale_grid.csv",
        "responses.csv",
        "functional_lock.json",
        "complete.json",
    )
    input_digests = dict(calibration.input_digests)
    input_digests.update(
        {
            f"functional_calibration/{name}": _file_sha256(functional_dir / name)
            for name in names
        }
    )
    return FunctionalGeometryAuditContext(
        study_dir=root,
        artifact_dir=root / "aggregate" / "functional_geometry_audit",
        calibration=calibration,
        preregistration=preregistration,
        input_digests=input_digests,
        functional_lock_sha256=_file_sha256(lock_path),
    )


def collect_audit_chunk(
    *,
    context: FunctionalGeometryAuditContext,
    seed: int,
    checkpoint_step: int,
    device: torch.device,
) -> FunctionalGeometryAuditChunk:
    """Load one checkpoint and collect its complete paired audit metrics."""

    model, target, source, path, objective = _load_probe_components(
        context.calibration,
        seed=seed,
        checkpoint_step=checkpoint_step,
        device=device,
    )
    manifest = context.calibration.probe_a_manifest
    if manifest is None or manifest.split != "a":
        raise ValueError("Functional geometry audit context lacks Probe-A.")
    batches_by_class: dict[int, tuple[ProbeBatch, ...]] = {}
    for class_id in context.preregistration.classes:
        rows = cell_microbatch_rows(
            manifest,
            class_id=class_id,
            stratum_id=context.preregistration.stratum_id,
        )
        batches_by_class[class_id] = tuple(
            materialize_probe_batch(
                target,
                source,
                manifest,
                row_indices,
                device=device,
            )
            for row_indices in rows
        )
    return collect_audit_metrics(
        model=model,
        objective=objective,
        path=path,
        batches_by_class=batches_by_class,
        preregistration=context.preregistration,
        seed=seed,
        checkpoint_step=checkpoint_step,
    )


def run_functional_geometry_audit(
    *,
    study_dir: str | Path,
    audit_preregistration_path: str | Path,
    device: torch.device,
) -> FunctionalGeometryAuditResult:
    """Run or validate the separate non-unlocking functional geometry audit."""

    context = prepare_functional_geometry_audit_context(
        study_dir=study_dir,
        audit_preregistration_path=audit_preregistration_path,
    )
    artifact_dir = context.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context.preregistration.lock(artifact_dir / "preregistration.yaml")
    complete_path = artifact_dir / "complete.json"
    if complete_path.is_file():
        return FunctionalGeometryAuditResult(
            artifact_dir=artifact_dir,
            decision=_load_completed_audit(context),
        )

    chunks = []
    for seed in context.calibration.observation0.training_seeds:
        for checkpoint_step in context.preregistration.checkpoint_steps:
            chunk_dir = (
                artifact_dir
                / "chunks"
                / f"seed_{seed}_checkpoint_{checkpoint_step:06d}"
            )
            identity = {
                "schema_version": 1,
                "audit_preregistration_sha256": context.preregistration.digest,
                "input_digests": context.input_digests,
                "seed": int(seed),
                "checkpoint_step": int(checkpoint_step),
            }
            chunks.append(
                _load_or_collect_audit_chunk(
                    chunk_dir,
                    identity=identity,
                    collector=lambda seed=seed, checkpoint_step=checkpoint_step: (
                        collect_audit_chunk(
                            context=context,
                            seed=seed,
                            checkpoint_step=checkpoint_step,
                            device=device,
                        )
                    ),
                )
            )
    slopes = pd.concat([chunk.slopes for chunk in chunks], ignore_index=True)
    slopes = slopes.sort_values(list(_SLOPE_KEYS)).reset_index(drop=True)
    finite_steps = pd.concat(
        [chunk.finite_steps for chunk in chunks],
        ignore_index=True,
    ).sort_values(list(_FINITE_KEYS)).reset_index(drop=True)
    basis_comparison = pd.concat(
        [chunk.basis_comparison for chunk in chunks],
        ignore_index=True,
    ).sort_values(
        [
            "checkpoint_step",
            "layer",
            "seed",
            "fold",
            "basis_kind",
            "direction_class",
        ]
    ).reset_index(drop=True)
    _write_dataframe_atomic(slopes, artifact_dir / "slopes.csv")
    _write_dataframe_atomic(finite_steps, artifact_dir / "finite_steps.csv")
    _write_dataframe_atomic(
        basis_comparison,
        artifact_dir / "basis_comparison.csv",
    )
    decision = analyze_functional_geometry_audit(
        slopes,
        finite_steps,
        context.preregistration,
    )
    summary = _decision_to_dict(decision)
    summary.update(
        {
            "schema_version": 1,
            "audit_preregistration_sha256": context.preregistration.digest,
            "observation0_preregistration_sha256": (
                context.preregistration.observation0_preregistration_sha256
            ),
            "functional_preregistration_sha256": (
                context.preregistration.functional_preregistration_sha256
            ),
            "functional_lock_sha256": context.functional_lock_sha256,
            "input_digests": context.input_digests,
            "original_functional_lock_remains_blocked": True,
        }
    )
    _write_json_atomic(summary, artifact_dir / "audit_summary.json")
    files = {
        name: _file_sha256(artifact_dir / name)
        for name in (
            "preregistration.yaml",
            "slopes.csv",
            "finite_steps.csv",
            "basis_comparison.csv",
            "audit_summary.json",
        )
    }
    _write_json_atomic(
        {
            "schema_version": 1,
            "audit_preregistration_sha256": context.preregistration.digest,
            "input_digests": context.input_digests,
            "files": files,
        },
        complete_path,
    )
    return FunctionalGeometryAuditResult(
        artifact_dir=artifact_dir,
        decision=decision,
    )


def _load_or_collect_audit_chunk(
    directory: Path,
    *,
    identity: dict[str, Any],
    collector: Any,
) -> FunctionalGeometryAuditChunk:
    complete_path = directory / "complete.json"
    names = ("slopes.csv", "finite_steps.csv", "basis_comparison.csv")
    if directory.exists():
        if not complete_path.is_file() or any(
            not (directory / name).is_file() for name in names
        ):
            raise ValueError(f"Functional geometry audit has a partial chunk: {directory}")
        complete = json.loads(complete_path.read_text())
        if complete.get("identity") != identity:
            raise ValueError(f"Functional geometry audit chunk identity changed: {directory}")
        for name in names:
            if (complete.get("files") or {}).get(name) != _file_sha256(
                directory / name
            ):
                raise ValueError(
                    f"Functional geometry audit chunk contents changed: {directory}"
                )
        return FunctionalGeometryAuditChunk(
            slopes=pd.read_csv(directory / "slopes.csv", keep_default_na=False),
            finite_steps=pd.read_csv(
                directory / "finite_steps.csv",
                keep_default_na=False,
            ),
            basis_comparison=pd.read_csv(
                directory / "basis_comparison.csv",
                keep_default_na=False,
            ),
        )
    chunk = collector()
    if not isinstance(chunk, FunctionalGeometryAuditChunk) or any(
        frame.empty
        for frame in (chunk.slopes, chunk.finite_steps, chunk.basis_comparison)
    ):
        raise ValueError("Functional geometry audit collector returned no rows.")
    directory.mkdir(parents=True, exist_ok=False)
    _write_dataframe_atomic(chunk.slopes, directory / "slopes.csv")
    _write_dataframe_atomic(chunk.finite_steps, directory / "finite_steps.csv")
    _write_dataframe_atomic(
        chunk.basis_comparison,
        directory / "basis_comparison.csv",
    )
    _write_json_atomic(
        {
            "identity": identity,
            "files": {
                name: _file_sha256(directory / name)
                for name in names
            },
        },
        complete_path,
    )
    return chunk


def _load_completed_audit(
    context: FunctionalGeometryAuditContext,
) -> FunctionalGeometryAuditDecision:
    artifact_dir = context.artifact_dir
    complete = json.loads((artifact_dir / "complete.json").read_text())
    if (
        complete.get("schema_version") != 1
        or complete.get("audit_preregistration_sha256")
        != context.preregistration.digest
        or complete.get("input_digests") != context.input_digests
    ):
        raise ValueError("Completed functional geometry audit input identity changed.")
    expected_names = {
        "preregistration.yaml",
        "slopes.csv",
        "finite_steps.csv",
        "basis_comparison.csv",
        "audit_summary.json",
    }
    files = complete.get("files") or {}
    if set(files) != expected_names:
        raise ValueError("Completed functional geometry audit file index changed.")
    for name in expected_names:
        path = artifact_dir / name
        if not path.is_file() or files[name] != _file_sha256(path):
            raise ValueError(
                f"Functional geometry audit aggregate changed after completion: {name}"
            )
    summary = json.loads((artifact_dir / "audit_summary.json").read_text())
    if (
        summary.get("audit_preregistration_sha256")
        != context.preregistration.digest
        or summary.get("functional_lock_sha256")
        != context.functional_lock_sha256
        or summary.get("input_digests") != context.input_digests
        or summary.get("original_functional_lock_remains_blocked") is not True
        or summary.get("stage1_unlocked") is not False
        or summary.get("probe_b_opened") is not False
    ):
        raise ValueError("Completed functional geometry audit summary changed.")
    return _decision_from_dict(summary)


def _decision_to_dict(
    decision: FunctionalGeometryAuditDecision,
) -> dict[str, Any]:
    return {
        "stage1_unlocked": decision.stage1_unlocked,
        "probe_b_opened": decision.probe_b_opened,
        "status": decision.status,
        "layer_summaries": decision.layer_summaries,
        "next_action": decision.next_action,
    }


def _decision_from_dict(values: dict[str, Any]) -> FunctionalGeometryAuditDecision:
    return FunctionalGeometryAuditDecision(
        stage1_unlocked=bool(values["stage1_unlocked"]),
        probe_b_opened=bool(values["probe_b_opened"]),
        status=str(values["status"]),
        layer_summaries={
            str(layer): dict(summary)
            for layer, summary in dict(values["layer_summaries"]).items()
        },
        next_action=str(values["next_action"]),
    )
