"""Natural-image analysis and service for CIFAR-10-LT transport falsification."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fm_lab.diagnostics.long_tail_geometry.checkpoints import restore_probe_model
from fm_lab.diagnostics.long_tail_geometry.functional_audit import (
    FunctionalGeometryAuditChunk,
    _bootstrap_median_interval,
    _finite_evaluation_errors,
    _fold_slope_blocks,
    _largest_concordant_step,
    _paired_block_differences,
    _validate_finite_table,
    _validate_slope_table,
    collect_audit_metrics,
    file_sha256,
)
from fm_lab.diagnostics.long_tail_geometry.functional_calibration import (
    cell_microbatch_rows,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeBatch,
    ProbeManifest,
    materialize_probe_batch,
)
from fm_lab.diagnostics.long_tail_geometry.natural_image_preregistration import (
    NaturalImageTransportPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.experiments.factory import build_path, build_source, build_target
from fm_lab.training.losses import build_objective
from fm_lab.utils.config import load_config


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


@dataclass(frozen=True)
class NaturalImageTransportContext:
    """Validated upstream state for one natural-image transport run."""

    study_dir: Path
    artifact_dir: Path
    observation0: Observation0Preregistration | Any
    preregistration: NaturalImageTransportPreregistration
    probe_a_manifest: ProbeManifest | None
    run_dirs: dict[int, Path]
    checkpoint_sha256: dict[tuple[int, int], str]
    input_digests: dict[str, str]
    reliability: pd.DataFrame
    class_counts: tuple[int, ...]
    class_ranks: tuple[int, ...]


@dataclass(frozen=True)
class NaturalImageTransportResult:
    """Completed falsification decision and its artifact directory."""

    artifact_dir: Path
    decision: NaturalImageTransportDecision


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
    if (
        len(ranks) != len(preregistration.classes)
        or set(ranks) != set(preregistration.classes)
    ):
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


def prepare_natural_image_transport_context(
    *,
    study_dir: str | Path,
    preregistration_path: str | Path,
) -> NaturalImageTransportContext:
    """Validate the completed CIFAR Observation-0 study and locked inputs."""

    root = Path(study_dir)
    observation_path = root / "aggregate" / "preregistration.yaml"
    observation0 = Observation0Preregistration.load(observation_path)
    if observation0.dataset != "cifar10_lt":
        raise ValueError(
            "Natural-image transport requires a CIFAR-10-LT Observation-0 study."
        )
    preregistration = NaturalImageTransportPreregistration.load(
        preregistration_path
    )
    if (
        preregistration.observation0_preregistration_sha256
        != observation0.digest
    ):
        raise ValueError("Natural-image transport targets a different Observation-0 study.")
    if not set(preregistration.checkpoint_steps).issubset(
        observation0.checkpoint_steps
    ):
        raise ValueError("Natural-image transport checkpoints are absent from Observation 0.")
    if not set(preregistration.layers).issubset(observation0.layers):
        raise ValueError("Natural-image transport layers are absent from Observation 0.")
    if preregistration.stratum_id >= len(observation0.time_strata) or (
        observation0.time_strata[preregistration.stratum_id]
        != preregistration.stratum_bounds
    ):
        raise ValueError("Natural-image transport timestep stratum changed.")
    if (
        observation0.primary_microbatches_per_cell
        != preregistration.microbatches_per_cell
    ):
        raise ValueError("Natural-image transport Probe-A cell size changed.")

    noise_path = root / "aggregate" / "noise_ceiling.json"
    reliability_path = root / "aggregate" / "reliability.csv"
    gram_path = root / "aggregate" / "gram_matrices.npz"
    noise = json.loads(noise_path.read_text())
    if noise.get("preregistration_digest") != observation0.digest:
        raise ValueError("Natural-image reliability has the wrong preregistration.")
    if noise.get("probe_phase") != preregistration.observation0_phase:
        raise ValueError("Natural-image reliability has the wrong probe phase.")
    if int(noise.get("microbatches_per_cell", -1)) != (
        preregistration.microbatches_per_cell
    ):
        raise ValueError("Natural-image reliability has the wrong probe allocation.")
    if file_sha256(reliability_path) != noise.get("reliability_sha256"):
        raise ValueError("Natural-image reliability table changed after analysis.")
    if file_sha256(gram_path) != noise.get("gram_matrices_sha256"):
        raise ValueError("Natural-image reliability Gram matrices changed after analysis.")
    reliability = pd.read_csv(reliability_path, keep_default_na=False)

    manifest_path = (
        root
        / "aggregate"
        / "manifests"
        / preregistration.observation0_phase
        / "probe_a.npz"
    )
    probe_a = ProbeManifest.load(manifest_path)
    if probe_a.split != preregistration.probe_view:
        raise ValueError("Natural-image transport manifest is not Probe-A.")
    for class_id in preregistration.classes:
        rows = cell_microbatch_rows(
            probe_a,
            class_id=class_id,
            stratum_id=preregistration.stratum_id,
        )
        if len(rows) != preregistration.microbatches_per_cell:
            raise ValueError("Natural-image Probe-A cell has the wrong allocation.")

    registry_path = root / "aggregate" / "run_registry.csv"
    registry = pd.read_csv(registry_path, keep_default_na=False)
    if tuple(sorted(int(value) for value in registry["seed"])) != tuple(
        sorted(observation0.training_seeds)
    ):
        raise ValueError("Natural-image transport registry changed training seeds.")
    if set(registry["study_digest"]) != {observation0.digest}:
        raise ValueError("Natural-image transport registry changed study identity.")
    if set(int(value) for value in registry["mapping_offset"]) != {0}:
        raise ValueError("Natural-image transport found a non-pilot mapping.")
    if set(registry["status"]) != {"measured"}:
        raise ValueError("Natural-image transport requires measured Observation-0 runs.")

    run_dirs: dict[int, Path] = {}
    checkpoint_digests: dict[tuple[int, int], str] = {}
    input_digests = {
        "observation0_preregistration.yaml": file_sha256(observation_path),
        "noise_ceiling.json": file_sha256(noise_path),
        "reliability.csv": file_sha256(reliability_path),
        "gram_matrices.npz": file_sha256(gram_path),
        "probe_a.npz": file_sha256(manifest_path),
        "run_registry.csv": file_sha256(registry_path),
    }
    first_config: dict[str, Any] | None = None
    for row in registry.sort_values("seed").itertuples(index=False):
        seed = int(row.seed)
        expected_suffix = Path("mapping_0") / f"seed_{seed}"
        registered = Path(str(row.run_dir))
        if tuple(registered.parts[-2:]) != tuple(expected_suffix.parts):
            raise ValueError("Natural-image transport registry run path changed.")
        run_dir = root / expected_suffix
        run_dirs[seed] = run_dir
        config_path = root / "configs" / f"seed_{seed}.yaml"
        config = load_config(config_path)
        if first_config is None:
            first_config = config
        input_digests[f"seed_{seed}/config.yaml"] = file_sha256(config_path)
        for checkpoint_step in preregistration.checkpoint_steps:
            checkpoint_path = (
                run_dir / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
            )
            checkpoint_digest = file_sha256(checkpoint_path)
            checkpoint_digests[(seed, checkpoint_step)] = checkpoint_digest
            measurement_dir = (
                run_dir
                / "diagnostics"
                / "long_tail_geometry"
                / "observation0"
                / preregistration.observation0_phase
                / f"checkpoint_{checkpoint_step:06d}"
            )
            measurement_path = measurement_dir / "complete.json"
            measurement = json.loads(measurement_path.read_text())
            if (
                int(measurement.get("checkpoint_step", -1)) != checkpoint_step
                or measurement.get("checkpoint_sha256") != checkpoint_digest
                or measurement.get("preregistration_sha256") != observation0.digest
                or (measurement.get("manifest_digests") or {}).get("a")
                != probe_a.digest
            ):
                raise ValueError("Natural-image transport measurement identity changed.")
            if not set(preregistration.layers).issubset(
                measurement.get("layer_shapes") or {}
            ):
                raise ValueError("Natural-image transport measurement lacks a locked layer.")
            input_digests[
                f"seed_{seed}/checkpoint_{checkpoint_step:06d}"
            ] = checkpoint_digest
            input_digests[
                f"seed_{seed}/measurement_{checkpoint_step:06d}"
            ] = file_sha256(measurement_path)

    if first_config is None:
        raise ValueError("Natural-image transport found no seed config.")
    runtime_config = _resolve_runtime_paths(first_config, study_dir=root)
    target = build_target(runtime_config)
    class_counts = tuple(int(value) for value in target.class_counts)
    class_ranks = tuple(int(value) for value in target.class_ranks)
    _validate_class_frequency_inputs(
        class_counts,
        class_ranks,
        preregistration,
    )
    return NaturalImageTransportContext(
        study_dir=root,
        artifact_dir=(
            root / "aggregate" / "natural_image_transport_falsification"
        ),
        observation0=observation0,
        preregistration=preregistration,
        probe_a_manifest=probe_a,
        run_dirs=run_dirs,
        checkpoint_sha256=checkpoint_digests,
        input_digests=input_digests,
        reliability=reliability,
        class_counts=class_counts,
        class_ranks=class_ranks,
    )


def collect_natural_image_transport_chunk(
    *,
    context: NaturalImageTransportContext,
    seed: int,
    checkpoint_step: int,
    device: torch.device,
) -> FunctionalGeometryAuditChunk:
    """Collect one seed/checkpoint exact-gradient transport chunk."""

    model, target, source, path, objective = _load_transport_components(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        device=device,
    )
    manifest = context.probe_a_manifest
    if manifest is None or manifest.split != "a":
        raise ValueError("Natural-image transport context lacks Probe-A.")
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


def run_natural_image_transport_falsification(
    *,
    study_dir: str | Path,
    preregistration_path: str | Path,
    device: torch.device,
) -> NaturalImageTransportResult:
    """Run, resume, or validate the terminal CIFAR transport falsification."""

    context = prepare_natural_image_transport_context(
        study_dir=study_dir,
        preregistration_path=preregistration_path,
    )
    artifact_dir = context.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context.preregistration.lock(artifact_dir / "preregistration.yaml")
    complete_path = artifact_dir / "complete.json"
    if complete_path.is_file():
        return NaturalImageTransportResult(
            artifact_dir=artifact_dir,
            decision=_load_completed_transport(context),
        )

    chunks = []
    for seed in context.observation0.training_seeds:
        for checkpoint_step in context.preregistration.checkpoint_steps:
            chunk_dir = (
                artifact_dir
                / "chunks"
                / f"seed_{seed}_checkpoint_{checkpoint_step:06d}"
            )
            identity = {
                "schema_version": 1,
                "transport_preregistration_sha256": context.preregistration.digest,
                "input_digests": context.input_digests,
                "seed": int(seed),
                "checkpoint_step": int(checkpoint_step),
            }
            chunks.append(
                _load_or_collect_transport_chunk(
                    chunk_dir,
                    identity=identity,
                    collector=lambda seed=seed, checkpoint_step=checkpoint_step: (
                        collect_natural_image_transport_chunk(
                            context=context,
                            seed=seed,
                            checkpoint_step=checkpoint_step,
                            device=device,
                        )
                    ),
                )
            )
    slopes = pd.concat([chunk.slopes for chunk in chunks], ignore_index=True)
    slopes = slopes.sort_values(
        [
            "checkpoint_step",
            "layer",
            "seed",
            "fold",
            "basis_kind",
            "direction_class",
            "evaluation_class",
        ]
    ).reset_index(drop=True)
    finite_steps = pd.concat(
        [chunk.finite_steps for chunk in chunks],
        ignore_index=True,
    ).sort_values(
        [
            "checkpoint_step",
            "layer",
            "seed",
            "fold",
            "basis_kind",
            "direction_class",
            "partition",
            "relative_step",
        ]
    ).reset_index(drop=True)
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
    analysis = analyze_natural_image_transport(
        slopes,
        finite_steps,
        basis_comparison,
        context.reliability,
        class_counts=context.class_counts,
        class_ranks=context.class_ranks,
        preregistration=context.preregistration,
    )
    _write_dataframe_atomic(slopes, artifact_dir / "slopes.csv")
    _write_dataframe_atomic(finite_steps, artifact_dir / "finite_steps.csv")
    _write_dataframe_atomic(
        basis_comparison,
        artifact_dir / "basis_comparison.csv",
    )
    _write_dataframe_atomic(
        analysis.class_transport,
        artifact_dir / "class_transport.csv",
    )
    _write_dataframe_atomic(
        analysis.frequency_associations,
        artifact_dir / "frequency_associations.csv",
    )
    _write_npz_atomic(
        analysis.interference_matrices,
        artifact_dir / "interference_matrices.npz",
    )
    summary = _decision_to_dict(analysis.decision)
    summary.update(
        {
            "schema_version": 1,
            "transport_preregistration_sha256": context.preregistration.digest,
            "observation0_preregistration_sha256": (
                context.preregistration.observation0_preregistration_sha256
            ),
            "input_digests": context.input_digests,
            "class_counts": list(context.class_counts),
            "class_ranks": list(context.class_ranks),
            "probe_b_used_for_transport": False,
        }
    )
    _write_json_atomic(summary, artifact_dir / "falsification_summary.json")
    aggregate_names = (
        "preregistration.yaml",
        "slopes.csv",
        "finite_steps.csv",
        "basis_comparison.csv",
        "class_transport.csv",
        "frequency_associations.csv",
        "interference_matrices.npz",
        "falsification_summary.json",
    )
    _write_json_atomic(
        {
            "schema_version": 1,
            "transport_preregistration_sha256": context.preregistration.digest,
            "input_digests": context.input_digests,
            "files": {
                name: file_sha256(artifact_dir / name)
                for name in aggregate_names
            },
        },
        complete_path,
    )
    return NaturalImageTransportResult(
        artifact_dir=artifact_dir,
        decision=analysis.decision,
    )


def _load_transport_components(
    context: NaturalImageTransportContext,
    *,
    seed: int,
    checkpoint_step: int,
    device: torch.device,
) -> tuple[Any, Any, Any, Any, Any]:
    run_dir = context.run_dirs[int(seed)]
    checkpoint_path = run_dir / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
    if file_sha256(checkpoint_path) != context.checkpoint_sha256[
        (int(seed), int(checkpoint_step))
    ]:
        raise ValueError("Natural-image checkpoint changed after context validation.")
    model, config = restore_probe_model(checkpoint_path, device=device)
    locked_config = load_config(context.study_dir / "configs" / f"seed_{seed}.yaml")
    if config != locked_config:
        raise ValueError("Natural-image checkpoint differs from its locked seed config.")
    runtime_config = _resolve_runtime_paths(config, study_dir=context.study_dir)
    target = build_target(runtime_config)
    source = build_source(runtime_config)
    path = build_path(runtime_config)
    objective = build_objective(
        runtime_config.get("objective", {}),
        diffusion_config=runtime_config.get("diffusion", {}),
        class_counts=getattr(target, "class_counts", None),
    )
    return model, target, source, path, objective


def _resolve_runtime_paths(
    config: dict[str, Any],
    *,
    study_dir: Path,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    data = resolved.get("data", {}) or {}
    configured_root = Path(str(data.get("root", "")))
    if configured_root.is_absolute():
        if not configured_root.exists():
            raise ValueError(
                f"Configured natural-image data root does not exist: {configured_root}"
            )
        data["download"] = False
        resolved["data"] = data
        return resolved
    candidates = [Path.cwd() / configured_root]
    candidates.extend(parent / configured_root for parent in study_dir.parents)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            data["root"] = str(candidate.resolve())
            data["download"] = False
            resolved["data"] = data
            return resolved
    raise ValueError(f"Could not resolve natural-image data root: {configured_root}")


def _load_or_collect_transport_chunk(
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
            raise ValueError(f"Natural-image transport has a partial chunk: {directory}")
        complete = json.loads(complete_path.read_text())
        if complete.get("identity") != identity:
            raise ValueError(f"Natural-image transport chunk identity changed: {directory}")
        for name in names:
            if (complete.get("files") or {}).get(name) != file_sha256(
                directory / name
            ):
                raise ValueError(
                    f"Natural-image transport chunk contents changed: {directory}"
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
        raise ValueError("Natural-image transport collector returned no rows.")
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
                name: file_sha256(directory / name)
                for name in names
            },
        },
        complete_path,
    )
    return chunk


def _load_completed_transport(
    context: NaturalImageTransportContext,
) -> NaturalImageTransportDecision:
    artifact_dir = context.artifact_dir
    complete = json.loads((artifact_dir / "complete.json").read_text())
    if (
        complete.get("schema_version") != 1
        or complete.get("transport_preregistration_sha256")
        != context.preregistration.digest
        or complete.get("input_digests") != context.input_digests
    ):
        raise ValueError("Completed natural-image transport input identity changed.")
    expected_names = {
        "preregistration.yaml",
        "slopes.csv",
        "finite_steps.csv",
        "basis_comparison.csv",
        "class_transport.csv",
        "frequency_associations.csv",
        "interference_matrices.npz",
        "falsification_summary.json",
    }
    files = complete.get("files") or {}
    if set(files) != expected_names:
        raise ValueError("Completed natural-image transport file index changed.")
    for name in expected_names:
        path = artifact_dir / name
        if not path.is_file() or files[name] != file_sha256(path):
            raise ValueError(
                f"Natural-image transport aggregate changed after completion: {name}"
            )
    summary = json.loads((artifact_dir / "falsification_summary.json").read_text())
    if (
        summary.get("transport_preregistration_sha256")
        != context.preregistration.digest
        or summary.get("observation0_preregistration_sha256")
        != context.preregistration.observation0_preregistration_sha256
        or summary.get("input_digests") != context.input_digests
        or summary.get("stage1_unlocked") is not False
        or summary.get("method_opened") is not False
        or summary.get("probe_b_used_for_transport") is not False
    ):
        raise ValueError("Completed natural-image transport summary changed.")
    return _decision_from_dict(summary)


def _write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(values: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_npz_atomic(values: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{name: values[name] for name in sorted(values)},
        )
    temporary.replace(path)


def _decision_to_dict(decision: NaturalImageTransportDecision) -> dict[str, Any]:
    return {
        "stage1_unlocked": decision.stage1_unlocked,
        "method_opened": decision.method_opened,
        "status": decision.status,
        "baseline_learned": decision.baseline_learned,
        "baseline_loss_ratio": decision.baseline_loss_ratio,
        "reliable_common_classes": list(decision.reliable_common_classes),
        "layer_summaries": decision.layer_summaries,
        "next_action": decision.next_action,
    }


def _decision_from_dict(values: dict[str, Any]) -> NaturalImageTransportDecision:
    return NaturalImageTransportDecision(
        stage1_unlocked=bool(values["stage1_unlocked"]),
        method_opened=bool(values["method_opened"]),
        status=str(values["status"]),
        baseline_learned=bool(values["baseline_learned"]),
        baseline_loss_ratio=float(values["baseline_loss_ratio"]),
        reliable_common_classes=tuple(
            int(value) for value in values["reliable_common_classes"]
        ),
        layer_summaries={
            str(layer): dict(summary)
            for layer, summary in dict(values["layer_summaries"]).items()
        },
        next_action=str(values["next_action"]),
    )
