"""Natural-image analysis and service for CIFAR-10-LT transport falsification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.diagnostics.long_tail_geometry.functional_audit import (
    _bootstrap_median_interval,
    _finite_evaluation_errors,
    _fold_slope_blocks,
    _largest_concordant_step,
    _paired_block_differences,
    _validate_finite_table,
    _validate_slope_table,
)
from fm_lab.diagnostics.long_tail_geometry.natural_image_preregistration import (
    NaturalImageTransportPreregistration,
)


@dataclass(frozen=True)
class NaturalImageTransportDecision:
    """Terminal scientific interpretation of the CIFAR transport study."""

    stage1_unlocked: bool
    method_opened: bool
    status: str
    baseline_learned: bool
    baseline_loss_ratio: float
    reliable_common_classes: tuple[int, ...]
    layer_summaries: dict[str, dict[str, float | int | bool | None]]
    next_action: str


@dataclass(frozen=True)
class NaturalImageTransportAnalysis:
    """Decision plus complete class, frequency, and interference outputs."""

    decision: NaturalImageTransportDecision
    class_transport: pd.DataFrame
    frequency_associations: pd.DataFrame
    interference_matrices: dict[str, np.ndarray]


def analyze_natural_image_transport(
    slopes: pd.DataFrame,
    finite_steps: pd.DataFrame,
    basis_comparison: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    class_counts: tuple[int, ...],
    class_ranks: tuple[int, ...],
    preregistration: NaturalImageTransportPreregistration,
) -> NaturalImageTransportAnalysis:
    """Apply the locked natural-image prerequisites and transport decision."""

    _validate_slope_table(slopes, preregistration)
    _validate_finite_table(finite_steps, preregistration)
    _validate_basis_table(basis_comparison, preregistration)
    reliable_common_classes = _reliable_common_classes(
        reliability,
        preregistration,
    )
    counts, ranks = _validate_class_frequency_inputs(
        class_counts,
        class_ranks,
        preregistration,
    )

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
        .sort_values(
            [
                "checkpoint_step",
                "layer",
                "basis_kind",
                "seed",
                "direction_class",
            ]
        )
        .reset_index(drop=True)
    )
    finite_errors = _finite_evaluation_errors(slopes, finite_steps)
    loss_ratios = _baseline_loss_ratios(slopes, preregistration)
    baseline_loss_ratio = float(max(loss_ratios.values()))
    baseline_learned = all(
        value <= preregistration.maximum_final_to_baseline_loss_ratio
        for value in loss_ratios.values()
    )
    layer_summaries = _layer_summaries(
        blocks,
        finite_errors,
        basis_comparison,
        loss_ratios,
        preregistration,
    )
    normalized_positive = all(
        bool(summary["normalized_positive_local_transport"])
        for summary in layer_summaries.values()
    )
    normalized_nonpositive = all(
        float(summary["normalized_target_slope_ci_upper"]) <= 0
        for summary in layer_summaries.values()
    )
    geometry_passed = (
        len(reliable_common_classes)
        >= preregistration.minimum_reliable_common_classes
    )
    if not baseline_learned:
        status = "baseline_not_learned"
        next_action = "repair_ordinary_cifar_baseline"
    elif not geometry_passed:
        status = "no_reliable_cifar_geometry"
        next_action = "pivot_from_spectral_gradient_geometry"
    elif normalized_positive:
        status = "natural_image_transport_confirmed"
        next_action = "develop_sign_transport_theory"
    elif normalized_nonpositive:
        status = "geometry_without_transport"
        next_action = "pivot_from_spectral_gradient_geometry"
    else:
        status = "heterogeneous_natural_image_transport"
        next_action = "explain_locked_transport_heterogeneity"

    class_transport = _class_transport_table(
        slopes,
        blocks,
        counts=counts,
        ranks=ranks,
        preregistration=preregistration,
    )
    return NaturalImageTransportAnalysis(
        decision=NaturalImageTransportDecision(
            stage1_unlocked=False,
            method_opened=False,
            status=status,
            baseline_learned=baseline_learned,
            baseline_loss_ratio=baseline_loss_ratio,
            reliable_common_classes=reliable_common_classes,
            layer_summaries=layer_summaries,
            next_action=next_action,
        ),
        class_transport=class_transport,
        frequency_associations=_frequency_associations(class_transport),
        interference_matrices=_interference_matrices(slopes, preregistration),
    )


def _validate_basis_table(
    frame: pd.DataFrame,
    preregistration: NaturalImageTransportPreregistration,
) -> None:
    keys = (
        "checkpoint_step",
        "layer",
        "seed",
        "fold",
        "basis_kind",
        "direction_class",
    )
    required = set(keys) | {
        "fit_explained_fraction",
        "projection_fraction",
        "fit_row_norm_cv",
        "raw_normalized_basis_abs_cosine",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Natural-image transport has an incomplete basis table.")
    if frame.duplicated(list(keys)).any():
        raise ValueError("Natural-image transport has duplicate basis cells.")
    expected = pd.MultiIndex.from_product(
        [
            preregistration.checkpoint_steps,
            preregistration.layers,
            (0, 1, 2),
            range(len(preregistration.fold_offsets)),
            preregistration.basis_kinds,
            preregistration.classes,
        ],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(frame[list(keys)])
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("Natural-image transport requires a complete basis table.")
    numeric = frame[
        [
            "fit_explained_fraction",
            "projection_fraction",
            "fit_row_norm_cv",
            "raw_normalized_basis_abs_cosine",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Natural-image transport basis values must be finite.")


def _reliable_common_classes(
    reliability: pd.DataFrame,
    preregistration: NaturalImageTransportPreregistration,
) -> tuple[int, ...]:
    required = {
        "training_seed",
        "checkpoint_step",
        "stratum_id",
        "class_id",
        "layer_name",
        "representation",
        "rank",
        "measurable",
    }
    if not required.issubset(reliability.columns):
        raise ValueError("Natural-image reliability scope is incomplete.")
    selected = reliability[
        (reliability["checkpoint_step"] == preregistration.primary_checkpoint_step)
        & (reliability["stratum_id"] == preregistration.stratum_id)
        & (reliability["representation"] == "centered_covariance")
        & (reliability["rank"] == preregistration.rank)
        & reliability["layer_name"].isin(preregistration.layers)
        & reliability["class_id"].isin(preregistration.classes)
        & reliability["training_seed"].isin((0, 1, 2))
    ].copy()
    keys = ["layer_name", "training_seed", "class_id"]
    expected = pd.MultiIndex.from_product(
        [preregistration.layers, (0, 1, 2), preregistration.classes],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(selected[keys]) if not selected.empty else ()
    if (
        selected.duplicated(keys).any()
        or len(selected) != len(expected)
        or set(observed) != set(expected)
    ):
        raise ValueError("Natural-image reliability scope is incomplete.")
    repeated_by_layer: dict[str, set[int]] = {}
    for layer in preregistration.layers:
        layer_rows = selected[selected["layer_name"] == layer]
        repeats = layer_rows.groupby("class_id")["measurable"].sum()
        repeated_by_layer[layer] = {
            int(class_id)
            for class_id, count in repeats.items()
            if int(count) >= preregistration.required_seed_repeats
        }
    common = set(preregistration.classes)
    for values in repeated_by_layer.values():
        common &= values
    return tuple(sorted(common))


def _validate_class_frequency_inputs(
    class_counts: tuple[int, ...],
    class_ranks: tuple[int, ...],
    preregistration: NaturalImageTransportPreregistration,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = tuple(int(value) for value in class_counts)
    ranks = tuple(int(value) for value in class_ranks)
    if len(counts) != len(preregistration.classes) or any(value < 1 for value in counts):
        raise ValueError("Natural-image class counts are invalid.")
    if ranks != tuple(ranks) or set(ranks) != set(preregistration.classes):
        raise ValueError("Natural-image class ranks must be a complete permutation.")
    return counts, ranks


def _baseline_loss_ratios(
    slopes: pd.DataFrame,
    preregistration: NaturalImageTransportPreregistration,
) -> dict[str, float]:
    target = slopes[
        (slopes["basis_kind"] == "row_normalized")
        & (slopes["direction_class"] == slopes["evaluation_class"])
        & slopes["checkpoint_step"].isin(
            (
                preregistration.baseline_checkpoint_step,
                preregistration.primary_checkpoint_step,
            )
        )
    ][
        [
            "checkpoint_step",
            "layer",
            "seed",
            "fold",
            "direction_class",
            "base_loss",
        ]
    ]
    index = ["layer", "seed", "fold", "direction_class"]
    baseline = target[
        target["checkpoint_step"] == preregistration.baseline_checkpoint_step
    ][index + ["base_loss"]].rename(columns={"base_loss": "baseline_loss"})
    primary = target[
        target["checkpoint_step"] == preregistration.primary_checkpoint_step
    ][index + ["base_loss"]].rename(columns={"base_loss": "primary_loss"})
    paired = baseline.merge(primary, on=index, validate="one_to_one")
    paired["loss_ratio"] = paired["primary_loss"] / paired["baseline_loss"]
    if len(paired) != len(preregistration.layers) * 3 * 4 * 10:
        raise ValueError("Natural-image baseline loss pairs are incomplete.")
    return {
        layer: float(values["loss_ratio"].median())
        for layer, values in paired.groupby("layer")
    }


def _layer_summaries(
    blocks: pd.DataFrame,
    finite_errors: pd.DataFrame,
    basis_comparison: pd.DataFrame,
    loss_ratios: dict[str, float],
    preregistration: NaturalImageTransportPreregistration,
) -> dict[str, dict[str, float | int | bool | None]]:
    primary = preregistration.primary_checkpoint_step
    summaries: dict[str, dict[str, float | int | bool | None]] = {}
    for layer_index, layer in enumerate(preregistration.layers):
        summary: dict[str, float | int | bool | None] = {
            "final_to_baseline_loss_ratio": loss_ratios[layer]
        }
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
                seed=preregistration.bootstrap_seed + 100 * layer_index + basis_index,
                confidence_level=preregistration.confidence_level,
            )
            seed_repeats = int(
                (selected.groupby("seed")["target_slope"].median() > 0).sum()
            )
            selectivity = float(selected["selectivity_slope"].median())
            prefix = "normalized" if basis_kind == "row_normalized" else "raw"
            summary[f"{prefix}_target_slope_median"] = float(np.median(values))
            summary[f"{prefix}_target_slope_ci_lower"] = lower
            summary[f"{prefix}_target_slope_ci_upper"] = upper
            summary[f"{prefix}_selectivity_slope_median"] = selectivity
            summary[f"{prefix}_positive_seed_repeats"] = seed_repeats
            summary[f"{prefix}_positive_local_transport"] = bool(
                lower > 0
                and selectivity > 0
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
        basis_rows = basis_comparison[
            (basis_comparison["checkpoint_step"] == primary)
            & (basis_comparison["layer"] == layer)
        ]
        summary["raw_normalized_basis_abs_cosine_median"] = float(
            basis_rows["raw_normalized_basis_abs_cosine"].median()
        )
        summaries[layer] = summary
    return summaries


def _class_transport_table(
    slopes: pd.DataFrame,
    blocks: pd.DataFrame,
    *,
    counts: tuple[int, ...],
    ranks: tuple[int, ...],
    preregistration: NaturalImageTransportPreregistration,
) -> pd.DataFrame:
    offclass = slopes[slopes["direction_class"] != slopes["evaluation_class"]].copy()
    offclass["incoming_offclass_harm"] = np.maximum(
        0.0,
        -offclass["benefit_slope"].to_numpy(dtype=np.float64),
    )
    incoming = (
        offclass.groupby(
            ["checkpoint_step", "layer", "basis_kind", "evaluation_class"],
            as_index=False,
        )["incoming_offclass_harm"]
        .median()
        .rename(columns={"evaluation_class": "direction_class"})
    )
    rows: list[dict[str, Any]] = []
    for key, values in blocks.groupby(
        ["checkpoint_step", "layer", "basis_kind", "direction_class"],
        sort=True,
    ):
        checkpoint_step, layer, basis_kind, class_id = key
        seed_values = values.groupby("seed")["target_slope"].median()
        rows.append(
            {
                "checkpoint_step": int(checkpoint_step),
                "layer": str(layer),
                "basis_kind": str(basis_kind),
                "class_id": int(class_id),
                "class_count": counts[int(class_id)],
                "class_rank": ranks[int(class_id)],
                "log_class_count": float(np.log(counts[int(class_id)])),
                "target_slope": float(values["target_slope"].median()),
                "selectivity_slope": float(values["selectivity_slope"].median()),
                "positive_seed_repeats": int((seed_values > 0).sum()),
                "positive_seed_fraction": float((seed_values > 0).mean()),
            }
        )
    result = pd.DataFrame(rows).merge(
        incoming.rename(columns={"direction_class": "class_id"}),
        on=["checkpoint_step", "layer", "basis_kind", "class_id"],
        validate="one_to_one",
    )
    expected_rows = (
        len(preregistration.checkpoint_steps)
        * len(preregistration.layers)
        * len(preregistration.basis_kinds)
        * len(preregistration.classes)
    )
    if len(result) != expected_rows:
        raise ValueError("Natural-image class transport table is incomplete.")
    return result.sort_values(
        ["checkpoint_step", "layer", "basis_kind", "class_id"]
    ).reset_index(drop=True)


def _frequency_associations(class_transport: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, values in class_transport.groupby(
        ["checkpoint_step", "layer", "basis_kind"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "checkpoint_step": int(key[0]),
            "layer": str(key[1]),
            "basis_kind": str(key[2]),
        }
        for metric in (
            "target_slope",
            "selectivity_slope",
            "incoming_offclass_harm",
        ):
            for frequency in ("class_rank", "log_class_count"):
                correlation = values[metric].corr(values[frequency], method="spearman")
                if not np.isfinite(correlation):
                    raise ValueError(
                        "Natural-image frequency association is not identifiable."
                    )
                row[f"{metric}_vs_{frequency}_spearman"] = float(correlation)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["checkpoint_step", "layer", "basis_kind"]
    ).reset_index(drop=True)


def _interference_matrices(
    slopes: pd.DataFrame,
    preregistration: NaturalImageTransportPreregistration,
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for key, values in slopes.groupby(
        ["checkpoint_step", "layer", "basis_kind"],
        sort=True,
    ):
        collapsed = values.groupby(
            ["direction_class", "evaluation_class"]
        )["benefit_slope"].median()
        matrix = (
            collapsed.unstack("evaluation_class")
            .reindex(
                index=preregistration.classes,
                columns=preregistration.classes,
            )
            .to_numpy(dtype=np.float64)
        )
        if matrix.shape != (10, 10) or not np.isfinite(matrix).all():
            raise ValueError("Natural-image interference matrix is incomplete.")
        layer = str(key[1]).replace(".", "_")
        name = f"checkpoint_{int(key[0])}__{layer}__{key[2]}"
        matrices[name] = matrix
    return matrices
