"""Exact directions and local functional tests for Observation 0."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.gradients import resolve_probe_layers
from fm_lab.diagnostics.long_tail_geometry.functional_preregistration import (
    FunctionalCalibrationPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import ProbeManifest


@dataclass(frozen=True)
class Rank1Direction:
    """Top direction of an exact centered microbatch-gradient covariance."""

    vector: torch.Tensor
    eigenvalue: float
    explained_fraction: float


@dataclass(frozen=True)
class ProjectedDescentDirection:
    """Signed unit descent direction inside a rank-1 subspace."""

    vector: torch.Tensor
    projection_fraction: float


@dataclass(frozen=True)
class ScaleSelection:
    """One layer's shared primary-checkpoint perturbation scale."""

    layer: str
    relative_step: float
    median_benefit: float
    doubled_median_benefit: float
    local_linearity_relative_error: float
    valid: bool


@dataclass(frozen=True)
class FunctionalCalibrationDecision:
    """Fail-closed Stage-1 lock derived from held-out Probe-A responses."""

    stage1_unlocked: bool
    positive_control_pass: bool
    selected_relative_steps: dict[str, float]
    layer_summaries: dict[str, dict[str, float | int | bool]]
    next_action: str


def cell_microbatch_rows(
    manifest: ProbeManifest,
    *,
    class_id: int,
    stratum_id: int,
) -> tuple[np.ndarray, ...]:
    """Return one cell's microbatch row arrays in stable manifest order."""

    selected: list[np.ndarray] = []
    for rows in manifest.microbatch_row_indices():
        labels = np.unique(manifest.labels[rows])
        strata = np.unique(manifest.stratum_ids[rows])
        if len(labels) != 1 or len(strata) != 1:
            raise ValueError("Probe manifest contains a mixed class/stratum microbatch.")
        if int(labels[0]) == int(class_id) and int(strata[0]) == int(stratum_id):
            selected.append(np.asarray(rows, dtype=np.int64))
    if not selected:
        raise ValueError(
            f"Probe manifest has no microbatches for class {class_id}, "
            f"stratum {stratum_id}."
        )
    return tuple(selected)


def top_centered_covariance_direction(rows: torch.Tensor) -> Rank1Direction:
    """Compute the exact top right singular direction through a sample Gram matrix."""

    if rows.ndim != 2:
        raise ValueError("Exact gradient rows must form a matrix.")
    if rows.shape[0] < 2:
        raise ValueError("Centered covariance requires at least two rows.")
    if rows.shape[1] < 1:
        raise ValueError("Exact gradient rows must have at least one parameter.")
    values = rows.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("Exact gradient rows must be finite.")
    centered = values - values.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalue = eigenvalues[-1]
    scale = torch.linalg.matrix_norm(gram)
    tolerance = torch.finfo(gram.dtype).eps * max(1, int(rows.shape[0])) * scale
    if not torch.isfinite(eigenvalue) or float(eigenvalue) <= float(tolerance):
        raise ValueError("Exact gradient rows have zero centered rank.")
    vector = centered.T @ eigenvectors[:, -1]
    vector /= torch.linalg.vector_norm(vector)
    vector = vector.contiguous()
    total = torch.trace(gram)
    return Rank1Direction(
        vector=vector,
        eigenvalue=float(eigenvalue),
        explained_fraction=float(eigenvalue / total),
    )


def projected_descent_direction(
    subspace_direction: torch.Tensor,
    mean_gradient: torch.Tensor,
    *,
    minimum_projection_fraction: float = 1e-8,
) -> ProjectedDescentDirection:
    """Orient a normalized rank-1 projection against a disjoint mean gradient."""

    if subspace_direction.ndim != 1 or mean_gradient.ndim != 1:
        raise ValueError("Subspace direction and mean gradient must be vectors.")
    if subspace_direction.shape != mean_gradient.shape:
        raise ValueError("Subspace direction and mean gradient must have the same shape.")
    if not 0 < minimum_projection_fraction < 1:
        raise ValueError("minimum_projection_fraction must lie in (0, 1).")
    direction = subspace_direction.detach().float().cpu()
    gradient = mean_gradient.detach().float().cpu()
    if not torch.isfinite(direction).all() or not torch.isfinite(gradient).all():
        raise ValueError("Subspace direction and mean gradient must be finite.")
    direction_norm = torch.linalg.vector_norm(direction)
    gradient_norm = torch.linalg.vector_norm(gradient)
    if float(direction_norm) == 0.0 or float(gradient_norm) == 0.0:
        raise ValueError("Subspace direction and mean gradient must be nonzero.")
    unit = direction / direction_norm
    projection = unit * torch.dot(unit, gradient)
    projection_norm = torch.linalg.vector_norm(projection)
    fraction = float(projection_norm / gradient_norm)
    if not np.isfinite(fraction) or fraction < minimum_projection_fraction:
        raise ValueError("Projected mean gradient is numerically negligible.")
    return ProjectedDescentDirection(
        vector=(-projection / projection_norm).contiguous(),
        projection_fraction=fraction,
    )


def deterministic_random_unit_direction(
    dimension: int,
    *,
    base_seed: int,
    key: tuple[Any, ...],
) -> torch.Tensor:
    """Generate a platform-stable keyed Gaussian unit direction on CPU."""

    if int(dimension) < 1:
        raise ValueError("Random direction dimension must be positive.")
    payload = json.dumps(
        {"base_seed": int(base_seed), "key": list(key)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    seed %= 2**63 - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(int(dimension), generator=generator, dtype=torch.float32)
    return (vector / torch.linalg.vector_norm(vector)).contiguous()


@contextmanager
def virtual_layer_update(
    model: nn.Module,
    *,
    layer_name: str,
    direction: torch.Tensor,
    relative_step: float,
) -> Iterator[float]:
    """Apply one relative layerwise update and restore the parameter bit-exactly."""

    if not np.isfinite(relative_step) or float(relative_step) <= 0:
        raise ValueError("Virtual-update relative_step must be positive and finite.")
    layer = resolve_probe_layers(model, (layer_name,))[0]
    flat_direction = direction.detach().reshape(-1).float().cpu()
    if flat_direction.numel() != layer.parameter.numel():
        raise ValueError("Virtual-update direction has the wrong shape.")
    if not torch.isfinite(flat_direction).all():
        raise ValueError("Virtual-update direction must be finite.")
    direction_norm = torch.linalg.vector_norm(flat_direction)
    if not torch.isclose(
        direction_norm,
        torch.ones_like(direction_norm),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Virtual-update direction must be unit norm.")
    original = layer.parameter.detach().clone()
    parameter_norm = torch.linalg.vector_norm(original)
    if not torch.isfinite(parameter_norm) or float(parameter_norm) == 0.0:
        raise ValueError("Virtual-update layer must have a finite nonzero norm.")
    applied_norm = float(relative_step) * float(parameter_norm)
    update = flat_direction.to(
        device=layer.parameter.device,
        dtype=layer.parameter.dtype,
    ).reshape(layer.shape)
    with torch.no_grad():
        layer.parameter.add_(update, alpha=applied_norm)
    try:
        yield applied_norm
    finally:
        with torch.no_grad():
            layer.parameter.copy_(original)
        if not torch.equal(layer.parameter.detach(), original):
            raise RuntimeError("Virtual update failed to restore the base parameter.")


_SCALE_COLUMNS = {
    "checkpoint_step",
    "layer",
    "seed",
    "class_id",
    "relative_step",
    "benefit",
    "doubled_benefit",
}


def select_layer_scales(
    scale_table: pd.DataFrame,
    preregistration: FunctionalCalibrationPreregistration,
) -> dict[str, ScaleSelection]:
    """Choose one shared epsilon per layer from complete finite scale grids."""

    missing = _SCALE_COLUMNS - set(scale_table)
    if missing:
        raise ValueError(f"Scale table is missing columns: {sorted(missing)}")
    table = scale_table.loc[:, sorted(_SCALE_COLUMNS)].copy()
    numeric = table[
        [
            "checkpoint_step",
            "seed",
            "class_id",
            "relative_step",
            "benefit",
            "doubled_benefit",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Scale table values must be finite.")
    table = table[
        table["checkpoint_step"] == preregistration.primary_checkpoint_step
    ]
    if set(table["layer"]) != set(preregistration.layers):
        raise ValueError("Scale table does not contain the locked layers.")
    if set(int(value) for value in table["class_id"]) != set(
        preregistration.classes
    ):
        raise ValueError("Scale table does not contain the locked classes.")
    if table.duplicated(
        ["layer", "seed", "class_id", "relative_step"]
    ).any():
        raise ValueError("Scale table contains duplicate calibration cells.")

    selections: dict[str, ScaleSelection] = {}
    locked_steps = set(preregistration.relative_step_grid)
    for layer in preregistration.layers:
        layer_table = table[table["layer"] == layer]
        blocks = layer_table[["seed", "class_id"]].drop_duplicates()
        if len(set(int(value) for value in blocks["seed"])) < (
            preregistration.required_seed_repeats
        ):
            raise ValueError("Scale table has too few training seeds.")
        counts = layer_table.groupby(["seed", "class_id"]).size()
        step_sets = layer_table.groupby(["seed", "class_id"])[
            "relative_step"
        ].apply(lambda values: set(float(value) for value in values))
        if (
            counts.empty
            or set(int(value) for value in counts)
            != {len(preregistration.relative_step_grid)}
            or any(values != locked_steps for values in step_sets)
        ):
            raise ValueError("Every seed/class requires the complete scale grid.")
        aggregate = (
            layer_table.groupby("relative_step", as_index=False)[
                ["benefit", "doubled_benefit"]
            ]
            .median()
            .sort_values("relative_step")
        )
        aggregate["target_distance"] = (
            aggregate["benefit"]
            - preregistration.target_loss_change_fraction
        ).abs()
        minimum_distance = float(aggregate["target_distance"].min())
        tied = aggregate[
            np.isclose(
                aggregate["target_distance"].to_numpy(),
                minimum_distance,
                rtol=1e-9,
                atol=1e-12,
            )
        ]
        chosen = tied.sort_values("relative_step", kind="stable").iloc[0]
        benefit = float(chosen["benefit"])
        doubled = float(chosen["doubled_benefit"])
        denominator = max(abs(2.0 * benefit), np.finfo(np.float64).eps)
        linearity_error = abs(doubled - 2.0 * benefit) / denominator
        target_low, target_high = preregistration.target_benefit_interval
        relative_step = float(chosen["relative_step"])
        valid = bool(
            target_low <= benefit <= target_high
            and relative_step <= preregistration.max_relative_layer_step
            and linearity_error
            <= preregistration.local_linearity_relative_error_max
        )
        selections[layer] = ScaleSelection(
            layer=layer,
            relative_step=relative_step,
            median_benefit=benefit,
            doubled_median_benefit=doubled,
            local_linearity_relative_error=float(linearity_error),
            valid=valid,
        )
    return selections


_RESPONSE_COLUMNS = {
    "checkpoint_step",
    "layer",
    "seed",
    "direction_class",
    "evaluation_class",
    "direction_kind",
    "control_id",
    "benefit",
}


def response_block_metrics(
    responses: pd.DataFrame,
    preregistration: FunctionalCalibrationPreregistration,
) -> pd.DataFrame:
    """Reduce complete cross-class response matrices to seed/class blocks."""

    missing = _RESPONSE_COLUMNS - set(responses)
    if missing:
        raise ValueError(f"Response table is missing columns: {sorted(missing)}")
    table = responses.loc[:, sorted(_RESPONSE_COLUMNS)].copy()
    numeric = table[
        [
            "checkpoint_step",
            "seed",
            "direction_class",
            "evaluation_class",
            "control_id",
            "benefit",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Response table values must be finite.")
    if set(int(value) for value in table["checkpoint_step"]) != set(
        preregistration.checkpoint_steps
    ):
        raise ValueError("Response table does not contain both locked checkpoints.")
    if set(table["layer"]) != set(preregistration.layers):
        raise ValueError("Response table does not contain the locked layers.")
    if set(int(value) for value in table["direction_class"]) != set(
        preregistration.classes
    ) or set(int(value) for value in table["evaluation_class"]) != set(
        preregistration.classes
    ):
        raise ValueError("Response table does not contain the locked class matrix.")
    if set(table["direction_kind"]) != {"primary", "random"}:
        raise ValueError("Response table requires primary and random directions.")
    keys = [
        "checkpoint_step",
        "layer",
        "seed",
        "direction_class",
        "direction_kind",
        "control_id",
    ]
    if table.duplicated(keys + ["evaluation_class"]).any():
        raise ValueError("Response table contains duplicate response cells.")
    expected_classes = set(preregistration.classes)
    rows: list[dict[str, Any]] = []
    for key, group in table.groupby(keys, sort=True):
        if set(int(value) for value in group["evaluation_class"]) != expected_classes:
            raise ValueError("Every direction requires a complete class response matrix.")
        (
            checkpoint_step,
            layer,
            seed,
            direction_class,
            direction_kind,
            control_id,
        ) = key
        target = group[group["evaluation_class"] == direction_class]
        if len(target) != 1:
            raise ValueError("Every response block requires exactly one target class.")
        target_benefit = float(target.iloc[0]["benefit"])
        off_class = group[group["evaluation_class"] != direction_class]["benefit"]
        harm = max(0.0, -float(off_class.min()))
        rows.append(
            {
                "checkpoint_step": int(checkpoint_step),
                "layer": str(layer),
                "seed": int(seed),
                "direction_class": int(direction_class),
                "direction_kind": str(direction_kind),
                "control_id": int(control_id),
                "target_benefit": target_benefit,
                "non_target_harm": harm,
                "selectivity_margin": target_benefit - harm,
            }
        )
    metrics = pd.DataFrame(rows)
    primary = metrics[metrics["direction_kind"] == "primary"]
    if set(int(value) for value in primary["control_id"]) != {-1}:
        raise ValueError("Primary directions must use control_id -1.")
    random = metrics[metrics["direction_kind"] == "random"]
    expected_controls = set(range(preregistration.random_controls))
    control_sets = random.groupby(
        ["checkpoint_step", "layer", "seed", "direction_class"]
    )["control_id"].apply(lambda values: set(int(value) for value in values))
    if control_sets.empty or any(values != expected_controls for values in control_sets):
        raise ValueError("Every response block requires all matched random controls.")
    primary_counts = primary.groupby(["checkpoint_step", "layer"]).size()
    random_counts = random.groupby(["checkpoint_step", "layer"]).size()
    if primary_counts.empty or any(
        int(random_counts.loc[index])
        != int(count) * preregistration.random_controls
        for index, count in primary_counts.items()
    ):
        raise ValueError("Response table has incomplete primary or random blocks.")
    return metrics


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector.")
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, len(values), size=(int(resamples), len(values)))
    medians = np.median(values[indices], axis=1)
    alpha = 1.0 - float(confidence_level)
    return (
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def analyze_functional_calibration(
    *,
    scale_table: pd.DataFrame,
    responses: pd.DataFrame,
    preregistration: FunctionalCalibrationPreregistration,
) -> FunctionalCalibrationDecision:
    """Apply the locked two-layer functional selectivity decision."""

    selections = select_layer_scales(scale_table, preregistration)
    metrics = response_block_metrics(responses, preregistration)
    layer_summaries: dict[str, dict[str, float | int | bool]] = {}
    for layer_index, layer in enumerate(preregistration.layers):
        primary = metrics[
            (metrics["checkpoint_step"] == preregistration.primary_checkpoint_step)
            & (metrics["layer"] == layer)
            & (metrics["direction_kind"] == "primary")
        ]
        random = metrics[
            (metrics["checkpoint_step"] == preregistration.primary_checkpoint_step)
            & (metrics["layer"] == layer)
            & (metrics["direction_kind"] == "random")
        ]
        target_median = float(primary["target_benefit"].median())
        harm_median = float(primary["non_target_harm"].median())
        margin_median = float(primary["selectivity_margin"].median())
        target_lower, target_upper = _bootstrap_median_interval(
            primary["target_benefit"].to_numpy(),
            resamples=preregistration.bootstrap_resamples,
            seed=preregistration.bootstrap_seed + 2 * layer_index,
            confidence_level=preregistration.confidence_level,
        )
        harm_lower, harm_upper = _bootstrap_median_interval(
            primary["non_target_harm"].to_numpy(),
            resamples=preregistration.bootstrap_resamples,
            seed=preregistration.bootstrap_seed + 2 * layer_index + 1,
            confidence_level=preregistration.confidence_level,
        )
        random_margins = random.groupby("control_id")[
            "selectivity_margin"
        ].median()
        random_threshold = float(
            np.quantile(
                random_margins.to_numpy(),
                preregistration.random_control_quantile,
                method="higher",
            )
        )
        seed_repeats = int(
            (
                primary.groupby("seed")["target_benefit"].median()
                > 0
            ).sum()
        )
        selection = selections[layer]
        passed = bool(
            selection.valid
            and target_lower > 0
            and harm_upper
            < preregistration.maximum_harm_to_benefit_ratio * target_median
            and margin_median > random_threshold
            and seed_repeats >= preregistration.required_seed_repeats
        )
        layer_summaries[layer] = {
            "passed": passed,
            "scale_valid": selection.valid,
            "relative_step": selection.relative_step,
            "scale_median_benefit": selection.median_benefit,
            "local_linearity_relative_error": (
                selection.local_linearity_relative_error
            ),
            "target_benefit_median": target_median,
            "target_benefit_ci_lower": target_lower,
            "target_benefit_ci_upper": target_upper,
            "non_target_harm_median": harm_median,
            "non_target_harm_ci_lower": harm_lower,
            "non_target_harm_ci_upper": harm_upper,
            "selectivity_margin_median": margin_median,
            "random_margin_quantile": random_threshold,
            "seed_repeats": seed_repeats,
        }

    control_metrics = metrics[
        (
            metrics["checkpoint_step"]
            == preregistration.positive_control_checkpoint_step
        )
        & (metrics["direction_kind"] == "primary")
    ]
    control_medians = control_metrics.groupby("layer")["target_benefit"].median()
    positive_control_pass = bool(
        set(control_medians.index) == set(preregistration.layers)
        and (control_medians > 0).all()
    )
    primary_pass = all(
        bool(summary["passed"]) for summary in layer_summaries.values()
    )
    unlocked = bool(primary_pass and positive_control_pass)
    if unlocked:
        next_action = "stage1_unlocked_for_separate_preregistration"
    elif positive_control_pass and any(
        float(summary["target_benefit_ci_lower"]) <= 0
        for summary in layer_summaries.values()
    ):
        next_action = "stop_stage1_and_study_training_dynamics"
    else:
        next_action = "stop_stage1_and_revise_functional_geometry"
    return FunctionalCalibrationDecision(
        stage1_unlocked=unlocked,
        positive_control_pass=positive_control_pass,
        selected_relative_steps={
            layer: selection.relative_step for layer, selection in selections.items()
        },
        layer_summaries=layer_summaries,
        next_action=next_action,
    )
