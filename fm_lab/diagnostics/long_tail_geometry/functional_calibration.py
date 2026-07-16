"""Exact directions and local functional tests for Observation 0."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.checkpoints import (
    evaluate_probe_batches,
    restore_probe_model,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import (
    collect_gradient_rows,
    resolve_probe_layers,
)
from fm_lab.diagnostics.long_tail_geometry.functional_preregistration import (
    FunctionalCalibrationPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeBatch,
    ProbeManifest,
    materialize_probe_batch,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.experiments.factory import build_path, build_source, build_target
from fm_lab.training.losses import build_objective
from fm_lab.utils.config import load_config
from fm_lab.utils.logging import write_json


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


@dataclass(frozen=True)
class CalibrationContext:
    """Validated inputs and immutable identities for one calibration run."""

    study_dir: Path
    artifact_dir: Path
    observation0: Observation0Preregistration
    preregistration: FunctionalCalibrationPreregistration
    probe_a_manifest: ProbeManifest | None
    run_dirs: dict[int, Path]
    checkpoint_sha256: dict[tuple[int, int], str]
    input_digests: dict[str, str]


@dataclass(frozen=True)
class FunctionalCalibrationResult:
    """Completed functional lock and its artifact location."""

    artifact_dir: Path
    decision: FunctionalCalibrationDecision


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


def validate_noise_ceiling_scope(
    noise_ceiling: dict[str, Any],
    observation0: Observation0Preregistration,
    preregistration: FunctionalCalibrationPreregistration,
) -> None:
    """Require the exact reliable adjacent-layer cells chosen for calibration."""

    if noise_ceiling.get("preregistration_digest") != observation0.digest:
        raise ValueError("Noise-ceiling decision has the wrong preregistration digest.")
    if (
        noise_ceiling.get("status") != preregistration.required_observation0_status
        or not bool(noise_ceiling.get("network_wide_gate_passed", False))
    ):
        raise ValueError("Functional calibration requires a network-wide measurable pilot.")
    if bool(noise_ceiling.get("required_escalation", False)):
        raise ValueError("Functional calibration cannot run while escalation is required.")
    if (
        noise_ceiling.get("next_action")
        != preregistration.required_observation0_next_action
    ):
        raise ValueError("Functional calibration is not the pilot's allowed next action.")
    if noise_ceiling.get("probe_phase") != preregistration.observation0_phase:
        raise ValueError("Functional calibration requires primary probe artifacts.")
    if int(noise_ceiling.get("microbatches_per_cell", -1)) != (
        preregistration.microbatches_per_cell
    ):
        raise ValueError("Noise-ceiling decision has the wrong microbatch count.")
    if tuple(int(value) for value in noise_ceiling.get("complete_seeds", ())) != (
        observation0.training_seeds
    ) or noise_ceiling.get("incomplete_seeds"):
        raise ValueError("Functional calibration requires every locked training seed.")

    passing_pairs = noise_ceiling.get("passing_pairs")
    if not isinstance(passing_pairs, list):
        raise ValueError("Noise-ceiling decision is missing passing layer pairs.")
    found_steps = set()
    for pair in passing_pairs:
        if not isinstance(pair, dict):
            continue
        if (
            int(pair.get("stratum_id", -1)) == preregistration.stratum_id
            and int(pair.get("rank", -1)) == preregistration.rank
            and tuple(pair.get("layers", ())) == preregistration.layers
            and set(int(value) for value in pair.get("common_classes", ()))
            >= set(preregistration.classes)
        ):
            checkpoint_step = int(pair.get("checkpoint_step", -1))
            if checkpoint_step in preregistration.checkpoint_steps:
                found_steps.add(checkpoint_step)
    if found_steps != set(preregistration.checkpoint_steps):
        raise ValueError(
            "The selected adjacent-layer cells must pass at both locked checkpoints."
        )


def prepare_calibration_context(
    *,
    study_dir: str | Path,
    calibration_preregistration_path: str | Path,
) -> CalibrationContext:
    """Validate all upstream Observation-0 artifacts without opening Probe-B."""

    root = Path(study_dir)
    observation_path = root / "aggregate" / "preregistration.yaml"
    observation0 = Observation0Preregistration.load(observation_path)
    preregistration = FunctionalCalibrationPreregistration.load(
        calibration_preregistration_path
    )
    if (
        preregistration.observation0_preregistration_sha256
        != observation0.digest
    ):
        raise ValueError("Functional calibration targets a different Observation-0 study.")
    if not set(preregistration.checkpoint_steps).issubset(
        observation0.checkpoint_steps
    ):
        raise ValueError("Functional checkpoints are absent from Observation 0.")
    if not set(preregistration.layers).issubset(observation0.layers):
        raise ValueError("Functional layers are absent from Observation 0.")
    if preregistration.stratum_id >= len(observation0.time_strata) or (
        observation0.time_strata[preregistration.stratum_id]
        != preregistration.stratum_bounds
    ):
        raise ValueError("Functional timestep stratum differs from Observation 0.")
    if (
        observation0.primary_microbatches_per_cell
        != preregistration.microbatches_per_cell
    ):
        raise ValueError("Functional Probe-A partitions do not cover the primary cell.")

    noise_path = root / "aggregate" / "noise_ceiling.json"
    noise_ceiling = json.loads(noise_path.read_text())
    validate_noise_ceiling_scope(noise_ceiling, observation0, preregistration)
    for filename, field in (
        ("reliability.csv", "reliability_sha256"),
        ("gram_matrices.npz", "gram_matrices_sha256"),
    ):
        path = root / "aggregate" / filename
        if _file_sha256(path) != noise_ceiling.get(field):
            raise ValueError(f"Noise-ceiling input changed after analysis: {filename}")

    manifest_path = (
        root
        / "aggregate"
        / "manifests"
        / preregistration.observation0_phase
        / "probe_a.npz"
    )
    probe_a = ProbeManifest.load(manifest_path)
    if probe_a.split != preregistration.probe_view:
        raise ValueError("Functional calibration manifest is not Probe-A.")
    for class_id in preregistration.classes:
        rows = cell_microbatch_rows(
            probe_a,
            class_id=class_id,
            stratum_id=preregistration.stratum_id,
        )
        if len(rows) != preregistration.microbatches_per_cell:
            raise ValueError("Probe-A cell has the wrong number of microbatches.")

    registry_path = root / "aggregate" / "run_registry.csv"
    registry = pd.read_csv(registry_path, keep_default_na=False)
    if tuple(int(value) for value in registry["seed"]) != observation0.training_seeds:
        raise ValueError("Functional calibration registry has changed seeds.")
    if set(registry["study_digest"]) != {observation0.digest}:
        raise ValueError("Functional calibration registry has changed study identity.")
    if set(int(value) for value in registry["mapping_offset"]) != {0}:
        raise ValueError("Functional calibration found a non-pilot mapping.")
    if set(registry["status"]) != {"measured"}:
        raise ValueError("Functional calibration requires measured pilot runs.")

    run_dirs: dict[int, Path] = {}
    checkpoint_digests: dict[tuple[int, int], str] = {}
    input_digests = {
        "observation0_preregistration.yaml": _file_sha256(observation_path),
        "noise_ceiling.json": _file_sha256(noise_path),
        "reliability.csv": _file_sha256(root / "aggregate" / "reliability.csv"),
        "gram_matrices.npz": _file_sha256(
            root / "aggregate" / "gram_matrices.npz"
        ),
        "probe_a.npz": _file_sha256(manifest_path),
        "run_registry.csv": _file_sha256(registry_path),
    }
    for row in registry.itertuples(index=False):
        seed = int(row.seed)
        registered_run_dir = Path(str(row.run_dir))
        expected_suffix = Path("mapping_0") / f"seed_{seed}"
        if tuple(registered_run_dir.parts[-2:]) != tuple(expected_suffix.parts):
            raise ValueError("Functional calibration registry run path changed.")
        run_dir = root / expected_suffix
        run_dirs[seed] = run_dir
        for checkpoint_step in preregistration.checkpoint_steps:
            checkpoint_path = (
                run_dir / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
            )
            checkpoint_digest = _file_sha256(checkpoint_path)
            checkpoint_digests[(seed, checkpoint_step)] = checkpoint_digest
            measurement_dir = (
                run_dir
                / "diagnostics"
                / "long_tail_geometry"
                / "observation0"
                / preregistration.observation0_phase
                / f"checkpoint_{checkpoint_step:06d}"
            )
            measurement = json.loads((measurement_dir / "complete.json").read_text())
            if (
                int(measurement.get("checkpoint_step", -1)) != checkpoint_step
                or measurement.get("checkpoint_sha256") != checkpoint_digest
                or measurement.get("preregistration_sha256") != observation0.digest
                or (measurement.get("manifest_digests") or {}).get("a")
                != probe_a.digest
            ):
                raise ValueError("Functional calibration measurement identity changed.")
            layer_shapes = measurement.get("layer_shapes") or {}
            if not set(preregistration.layers).issubset(layer_shapes):
                raise ValueError("Functional calibration measurement lacks a locked layer.")
            input_digests[
                f"seed_{seed}/checkpoint_{checkpoint_step:06d}"
            ] = checkpoint_digest
            input_digests[
                f"seed_{seed}/measurement_{checkpoint_step:06d}"
            ] = _file_sha256(measurement_dir / "complete.json")

    artifact_dir = root / "aggregate" / "functional_calibration"
    return CalibrationContext(
        study_dir=root,
        artifact_dir=artifact_dir,
        observation0=observation0,
        preregistration=preregistration,
        probe_a_manifest=probe_a,
        run_dirs=run_dirs,
        checkpoint_sha256=checkpoint_digests,
        input_digests=input_digests,
    )


def calibrate_observation0_functional_overlap(
    *,
    study_dir: str | Path,
    calibration_preregistration_path: str | Path,
    device: torch.device,
) -> FunctionalCalibrationResult:
    """Run or resume the locked Probe-A functional calibration."""

    context = prepare_calibration_context(
        study_dir=study_dir,
        calibration_preregistration_path=calibration_preregistration_path,
    )
    artifact_dir = context.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context.preregistration.lock(artifact_dir / "preregistration.yaml")
    complete_path = artifact_dir / "complete.json"
    if complete_path.is_file():
        return FunctionalCalibrationResult(
            artifact_dir=artifact_dir,
            decision=_load_completed_decision(context),
        )

    scale_chunks = []
    for seed in context.observation0.training_seeds:
        chunk_path = artifact_dir / "chunks" / f"scale_seed_{seed}.csv"
        scale_chunks.append(
            _load_or_collect_chunk(
                chunk_path,
                identity=_chunk_identity(context, kind="scale", seed=seed),
                collector=lambda seed=seed: collect_scale_chunk(
                    context=context,
                    seed=seed,
                    device=device,
                ),
            )
        )
    scale_table = pd.concat(scale_chunks, ignore_index=True)
    _write_dataframe_atomic(scale_table, artifact_dir / "scale_grid.csv")
    selections = select_layer_scales(scale_table, context.preregistration)
    relative_steps = {
        layer: selection.relative_step for layer, selection in selections.items()
    }
    selected_steps_digest = hashlib.sha256(
        json.dumps(relative_steps, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    response_chunks = []
    for seed in context.observation0.training_seeds:
        for checkpoint_step in context.preregistration.checkpoint_steps:
            chunk_path = (
                artifact_dir
                / "chunks"
                / f"responses_seed_{seed}_checkpoint_{checkpoint_step:06d}.csv"
            )
            response_chunks.append(
                _load_or_collect_chunk(
                    chunk_path,
                    identity=_chunk_identity(
                        context,
                        kind="responses",
                        seed=seed,
                        checkpoint_step=checkpoint_step,
                        selected_steps_digest=selected_steps_digest,
                    ),
                    collector=lambda seed=seed, checkpoint_step=checkpoint_step: (
                        collect_response_chunk(
                            context=context,
                            seed=seed,
                            checkpoint_step=checkpoint_step,
                            relative_steps=relative_steps,
                            device=device,
                        )
                    ),
                )
            )
    responses = pd.concat(response_chunks, ignore_index=True)
    _write_dataframe_atomic(responses, artifact_dir / "responses.csv")
    decision = analyze_functional_calibration(
        scale_table=scale_table,
        responses=responses,
        preregistration=context.preregistration,
    )
    direction_digest = direction_index_digest(artifact_dir / "directions")
    lock_payload = _decision_to_dict(decision)
    lock_payload.update(
        {
            "schema_version": 1,
            "functional_preregistration_sha256": context.preregistration.digest,
            "observation0_preregistration_sha256": context.observation0.digest,
            "probe_view": "a",
            "probe_b_opened": False,
            "input_digests": context.input_digests,
            "direction_index_sha256": direction_digest,
            "scale_grid_sha256": _file_sha256(artifact_dir / "scale_grid.csv"),
            "responses_sha256": _file_sha256(artifact_dir / "responses.csv"),
        }
    )
    _write_json_atomic(lock_payload, artifact_dir / "functional_lock.json")
    files = {
        name: _file_sha256(artifact_dir / name)
        for name in (
            "preregistration.yaml",
            "scale_grid.csv",
            "responses.csv",
            "functional_lock.json",
        )
    }
    _write_json_atomic(
        {
            "schema_version": 1,
            "functional_preregistration_sha256": context.preregistration.digest,
            "direction_index_sha256": direction_digest,
            "files": files,
        },
        complete_path,
    )
    return FunctionalCalibrationResult(artifact_dir=artifact_dir, decision=decision)


def collect_scale_chunk(
    *,
    context: CalibrationContext,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Recompute exact primary directions and finite scale effects for one seed."""

    checkpoint_step = context.preregistration.primary_checkpoint_step
    model, target, source, path, objective = _load_probe_components(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        device=device,
    )
    partitions = _materialize_partitions(
        context,
        target=target,
        source=source,
        device=device,
    )
    directions = _ensure_primary_directions(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        model=model,
        objective=objective,
        path=path,
        partitions=partitions,
    )
    base_losses = {
        class_id: _mean_batch_loss(
            model=model,
            objective=objective,
            path=path,
            batches=partitions[class_id]["scale"],
        )
        for class_id in context.preregistration.classes
    }
    rows: list[dict[str, Any]] = []
    for layer in context.preregistration.layers:
        for class_id in context.preregistration.classes:
            direction = directions[(class_id, layer)]
            baseline = base_losses[class_id]
            for relative_step in context.preregistration.relative_step_grid:
                with virtual_layer_update(
                    model,
                    layer_name=layer,
                    direction=direction.vector,
                    relative_step=relative_step,
                ):
                    perturbed = _mean_batch_loss(
                        model=model,
                        objective=objective,
                        path=path,
                        batches=partitions[class_id]["scale"],
                    )
                with virtual_layer_update(
                    model,
                    layer_name=layer,
                    direction=direction.vector,
                    relative_step=2.0 * relative_step,
                ):
                    doubled = _mean_batch_loss(
                        model=model,
                        objective=objective,
                        path=path,
                        batches=partitions[class_id]["scale"],
                    )
                rows.append(
                    {
                        "checkpoint_step": checkpoint_step,
                        "layer": layer,
                        "seed": int(seed),
                        "class_id": class_id,
                        "relative_step": relative_step,
                        "benefit": -(perturbed - baseline) / baseline,
                        "doubled_benefit": -(doubled - baseline) / baseline,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["layer", "seed", "class_id", "relative_step"]
    ).reset_index(drop=True)


def collect_response_chunk(
    *,
    context: CalibrationContext,
    seed: int,
    checkpoint_step: int,
    relative_steps: dict[str, float],
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate primary and matched-random cross-class responses for one run."""

    if set(relative_steps) != set(context.preregistration.layers):
        raise ValueError("Response collection requires one selected step per layer.")
    model, target, source, path, objective = _load_probe_components(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        device=device,
    )
    partitions = _materialize_partitions(
        context,
        target=target,
        source=source,
        device=device,
    )
    primary_directions = _ensure_primary_directions(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        model=model,
        objective=objective,
        path=path,
        partitions=partitions,
    )
    base_losses = {
        class_id: _mean_batch_loss(
            model=model,
            objective=objective,
            path=path,
            batches=partitions[class_id]["evaluation"],
        )
        for class_id in context.preregistration.classes
    }
    scale_means: dict[tuple[int, str], torch.Tensor] = {}
    for class_id in context.preregistration.classes:
        gradients = collect_gradient_rows(
            model=model,
            objective=objective,
            path=path,
            batches=partitions[class_id]["scale"],
            layer_names=context.preregistration.layers,
        )
        for layer in context.preregistration.layers:
            scale_means[(class_id, layer)] = gradients[layer].raw.mean(dim=0)

    rows: list[dict[str, Any]] = []
    for layer in context.preregistration.layers:
        relative_step = float(relative_steps[layer])
        for direction_class in context.preregistration.classes:
            primary = primary_directions[(direction_class, layer)]
            rows.extend(
                _evaluate_response_direction(
                    model=model,
                    objective=objective,
                    path=path,
                    partitions=partitions,
                    base_losses=base_losses,
                    layer=layer,
                    direction=primary.vector,
                    projection_fraction=primary.projection_fraction,
                    relative_step=relative_step,
                    seed=seed,
                    checkpoint_step=checkpoint_step,
                    direction_class=direction_class,
                    direction_kind="primary",
                    control_id=-1,
                    classes=context.preregistration.classes,
                )
            )
            mean_gradient = scale_means[(direction_class, layer)]
            for control_id in range(context.preregistration.random_controls):
                random_subspace = deterministic_random_unit_direction(
                    mean_gradient.numel(),
                    base_seed=context.preregistration.random_seed,
                    key=(seed, checkpoint_step, direction_class, layer, control_id),
                )
                random_direction = projected_descent_direction(
                    random_subspace,
                    mean_gradient,
                    minimum_projection_fraction=(
                        context.preregistration.minimum_projection_fraction
                    ),
                )
                rows.extend(
                    _evaluate_response_direction(
                        model=model,
                        objective=objective,
                        path=path,
                        partitions=partitions,
                        base_losses=base_losses,
                        layer=layer,
                        direction=random_direction.vector,
                        projection_fraction=random_direction.projection_fraction,
                        relative_step=relative_step,
                        seed=seed,
                        checkpoint_step=checkpoint_step,
                        direction_class=direction_class,
                        direction_kind="random",
                        control_id=control_id,
                        classes=context.preregistration.classes,
                    )
                )
    return pd.DataFrame(rows).sort_values(
        [
            "checkpoint_step",
            "layer",
            "seed",
            "direction_class",
            "direction_kind",
            "control_id",
            "evaluation_class",
        ]
    ).reset_index(drop=True)


def _load_probe_components(
    context: CalibrationContext,
    *,
    seed: int,
    checkpoint_step: int,
    device: torch.device,
) -> tuple[nn.Module, Any, Any, Any, Any]:
    run_dir = context.run_dirs[int(seed)]
    checkpoint_path = run_dir / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
    if _file_sha256(checkpoint_path) != context.checkpoint_sha256[
        (int(seed), int(checkpoint_step))
    ]:
        raise ValueError("Functional checkpoint changed after context validation.")
    model, config = restore_probe_model(checkpoint_path, device=device)
    locked_config = load_config(context.study_dir / "configs" / f"seed_{seed}.yaml")
    if config != locked_config:
        raise ValueError("Functional checkpoint config differs from its locked run config.")
    target = build_target(config)
    source = build_source(config)
    path = build_path(config)
    objective = build_objective(
        config.get("objective", {}),
        diffusion_config=config.get("diffusion", {}),
        class_counts=getattr(target, "class_counts", None),
    )
    return model, target, source, path, objective


def _materialize_partitions(
    context: CalibrationContext,
    *,
    target: Any,
    source: Any,
    device: torch.device,
) -> dict[int, dict[str, tuple[ProbeBatch, ...]]]:
    manifest = context.probe_a_manifest
    if manifest is None:
        raise ValueError("Functional calibration context lacks Probe-A.")
    position_groups = {
        "fit": context.preregistration.fit_positions,
        "scale": context.preregistration.scale_positions,
        "evaluation": context.preregistration.evaluation_positions,
    }
    partitions: dict[int, dict[str, tuple[ProbeBatch, ...]]] = {}
    for class_id in context.preregistration.classes:
        cell_rows = cell_microbatch_rows(
            manifest,
            class_id=class_id,
            stratum_id=context.preregistration.stratum_id,
        )
        partitions[class_id] = {
            name: tuple(
                materialize_probe_batch(
                    target,
                    source,
                    manifest,
                    cell_rows[position],
                    device=device,
                )
                for position in positions
            )
            for name, positions in position_groups.items()
        }
    return partitions


def _ensure_primary_directions(
    context: CalibrationContext,
    *,
    seed: int,
    checkpoint_step: int,
    model: nn.Module,
    objective: Any,
    path: Any,
    partitions: dict[int, dict[str, tuple[ProbeBatch, ...]]],
) -> dict[tuple[int, str], ProjectedDescentDirection]:
    directions: dict[tuple[int, str], ProjectedDescentDirection] = {}
    for class_id in context.preregistration.classes:
        paths = {
            layer: _direction_path(
                context.artifact_dir,
                seed=seed,
                checkpoint_step=checkpoint_step,
                layer=layer,
                class_id=class_id,
            )
            for layer in context.preregistration.layers
        }
        if all(path_value.is_file() for path_value in paths.values()):
            for layer, direction_path in paths.items():
                directions[(class_id, layer)] = _load_direction(
                    direction_path,
                    context=context,
                    seed=seed,
                    checkpoint_step=checkpoint_step,
                    class_id=class_id,
                    layer=layer,
                )
            continue
        batches = partitions[class_id]["fit"] + partitions[class_id]["scale"]
        gradients = collect_gradient_rows(
            model=model,
            objective=objective,
            path=path,
            batches=batches,
            layer_names=context.preregistration.layers,
        )
        fit_count = len(context.preregistration.fit_positions)
        for layer in context.preregistration.layers:
            rows = gradients[layer].raw
            rank1 = top_centered_covariance_direction(rows[:fit_count])
            projected = projected_descent_direction(
                rank1.vector,
                rows[fit_count:].mean(dim=0),
                minimum_projection_fraction=(
                    context.preregistration.minimum_projection_fraction
                ),
            )
            _save_direction(
                paths[layer],
                context=context,
                seed=seed,
                checkpoint_step=checkpoint_step,
                class_id=class_id,
                layer=layer,
                rank1=rank1,
                projected=projected,
            )
            directions[(class_id, layer)] = projected
    return directions


def _evaluate_response_direction(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    partitions: dict[int, dict[str, tuple[ProbeBatch, ...]]],
    base_losses: dict[int, float],
    layer: str,
    direction: torch.Tensor,
    projection_fraction: float,
    relative_step: float,
    seed: int,
    checkpoint_step: int,
    direction_class: int,
    direction_kind: str,
    control_id: int,
    classes: tuple[int, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with virtual_layer_update(
        model,
        layer_name=layer,
        direction=direction,
        relative_step=relative_step,
    ):
        for evaluation_class in classes:
            perturbed = _mean_batch_loss(
                model=model,
                objective=objective,
                path=path,
                batches=partitions[evaluation_class]["evaluation"],
            )
            baseline = base_losses[evaluation_class]
            records.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "layer": layer,
                    "seed": seed,
                    "direction_class": direction_class,
                    "evaluation_class": evaluation_class,
                    "direction_kind": direction_kind,
                    "control_id": control_id,
                    "relative_step": relative_step,
                    "projection_fraction": projection_fraction,
                    "base_loss": baseline,
                    "perturbed_loss": perturbed,
                    "benefit": -(perturbed - baseline) / baseline,
                }
            )
    return records


def _mean_batch_loss(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    batches: tuple[ProbeBatch, ...],
) -> float:
    result = evaluate_probe_batches(
        model=model,
        objective=objective,
        path=path,
        batches=batches,
    )
    if not np.isfinite(result.mean_loss) or result.mean_loss <= 0:
        raise ValueError("Functional calibration requires a finite positive base loss.")
    return result.mean_loss


def _direction_path(
    artifact_dir: Path,
    *,
    seed: int,
    checkpoint_step: int,
    layer: str,
    class_id: int,
) -> Path:
    return (
        artifact_dir
        / "directions"
        / f"seed_{seed}"
        / f"checkpoint_{checkpoint_step:06d}"
        / layer
        / f"class_{class_id}.pt"
    )


def _save_direction(
    path: Path,
    *,
    context: CalibrationContext,
    seed: int,
    checkpoint_step: int,
    class_id: int,
    layer: str,
    rank1: Rank1Direction,
    projected: ProjectedDescentDirection,
) -> None:
    metadata = _direction_metadata(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        class_id=class_id,
        layer=layer,
    )
    vector = projected.vector.detach().float().cpu().contiguous()
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "vector": vector,
        "vector_sha256": _tensor_sha256(vector),
        "eigenvalue": rank1.eigenvalue,
        "explained_fraction": rank1.explained_fraction,
        "projection_fraction": projected.projection_fraction,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_direction(
    path: Path,
    *,
    context: CalibrationContext,
    seed: int,
    checkpoint_step: int,
    class_id: int,
    layer: str,
) -> ProjectedDescentDirection:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected = _direction_metadata(
        context,
        seed=seed,
        checkpoint_step=checkpoint_step,
        class_id=class_id,
        layer=layer,
    )
    if payload.get("schema_version") != 1 or payload.get("metadata") != expected:
        raise ValueError(f"Exact direction provenance changed: {path}")
    vector = payload.get("vector")
    if not isinstance(vector, torch.Tensor) or _tensor_sha256(vector) != payload.get(
        "vector_sha256"
    ):
        raise ValueError(f"Exact direction tensor changed: {path}")
    if not torch.isclose(
        torch.linalg.vector_norm(vector.float()),
        torch.tensor(1.0),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(f"Exact direction is not unit norm: {path}")
    projection_fraction = float(payload.get("projection_fraction", float("nan")))
    if not np.isfinite(projection_fraction):
        raise ValueError(f"Exact direction projection metadata is invalid: {path}")
    return ProjectedDescentDirection(
        vector=vector.detach().float().cpu().contiguous(),
        projection_fraction=projection_fraction,
    )


def _direction_metadata(
    context: CalibrationContext,
    *,
    seed: int,
    checkpoint_step: int,
    class_id: int,
    layer: str,
) -> dict[str, Any]:
    manifest = context.probe_a_manifest
    return {
        "functional_preregistration_sha256": context.preregistration.digest,
        "observation0_preregistration_sha256": context.observation0.digest,
        "checkpoint_sha256": context.checkpoint_sha256[(seed, checkpoint_step)],
        "probe_a_manifest_sha256": manifest.digest if manifest is not None else "",
        "seed": seed,
        "checkpoint_step": checkpoint_step,
        "class_id": class_id,
        "stratum_id": context.preregistration.stratum_id,
        "rank": context.preregistration.rank,
        "layer": layer,
        "fit_positions": list(context.preregistration.fit_positions),
        "scale_positions": list(context.preregistration.scale_positions),
    }


def _chunk_identity(
    context: CalibrationContext,
    *,
    kind: str,
    seed: int,
    checkpoint_step: int | None = None,
    selected_steps_digest: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "seed": int(seed),
        "functional_preregistration_sha256": context.preregistration.digest,
        "input_digests": context.input_digests,
    }
    if checkpoint_step is not None:
        payload["checkpoint_step"] = int(checkpoint_step)
    if selected_steps_digest is not None:
        payload["selected_steps_sha256"] = selected_steps_digest
    return payload


def _load_or_collect_chunk(
    path: Path,
    *,
    identity: dict[str, Any],
    collector: Any,
) -> pd.DataFrame:
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise ValueError(f"Functional calibration has a partial chunk: {path}")
        metadata = json.loads(sidecar.read_text())
        if metadata.get("identity") != identity:
            raise ValueError(f"Functional calibration chunk identity changed: {path}")
        if metadata.get("csv_sha256") != _file_sha256(path):
            raise ValueError(f"Functional calibration chunk contents changed: {path}")
        return pd.read_csv(path, keep_default_na=False)
    frame = collector()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Functional calibration collector returned no rows.")
    _write_dataframe_atomic(frame, path)
    _write_json_atomic(
        {"identity": identity, "csv_sha256": _file_sha256(path)},
        sidecar,
    )
    return frame


def _write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    write_json(payload, temporary)
    temporary.replace(path)


def _decision_to_dict(decision: FunctionalCalibrationDecision) -> dict[str, Any]:
    return {
        "stage1_unlocked": decision.stage1_unlocked,
        "positive_control_pass": decision.positive_control_pass,
        "selected_relative_steps": decision.selected_relative_steps,
        "layer_summaries": decision.layer_summaries,
        "next_action": decision.next_action,
    }


def _decision_from_dict(payload: dict[str, Any]) -> FunctionalCalibrationDecision:
    try:
        return FunctionalCalibrationDecision(
            stage1_unlocked=bool(payload["stage1_unlocked"]),
            positive_control_pass=bool(payload["positive_control_pass"]),
            selected_relative_steps={
                str(layer): float(value)
                for layer, value in payload["selected_relative_steps"].items()
            },
            layer_summaries={
                str(layer): dict(summary)
                for layer, summary in payload["layer_summaries"].items()
            },
            next_action=str(payload["next_action"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Completed functional lock has an invalid decision.") from exc


def _load_completed_decision(context: CalibrationContext) -> FunctionalCalibrationDecision:
    complete_path = context.artifact_dir / "complete.json"
    complete = json.loads(complete_path.read_text())
    if (
        complete.get("schema_version") != 1
        or complete.get("functional_preregistration_sha256")
        != context.preregistration.digest
    ):
        raise ValueError("Functional calibration completion identity changed.")
    files = complete.get("files")
    if not isinstance(files, dict):
        raise ValueError("Functional calibration completion file index is invalid.")
    for name, expected_digest in files.items():
        path = context.artifact_dir / str(name)
        if not path.is_file() or _file_sha256(path) != expected_digest:
            raise ValueError(
                f"Functional calibration artifact changed after completion: {name}"
            )
    direction_digest = direction_index_digest(context.artifact_dir / "directions")
    if direction_digest != complete.get("direction_index_sha256"):
        raise ValueError("Functional calibration directions changed after completion.")
    lock = json.loads((context.artifact_dir / "functional_lock.json").read_text())
    if (
        lock.get("functional_preregistration_sha256")
        != context.preregistration.digest
        or lock.get("observation0_preregistration_sha256")
        != context.observation0.digest
        or lock.get("input_digests") != context.input_digests
        or lock.get("probe_view") != "a"
        or lock.get("probe_b_opened") is not False
        or lock.get("direction_index_sha256") != direction_digest
    ):
        raise ValueError("Completed functional lock input identity changed.")
    return _decision_from_dict(lock)


def direction_index_digest(directory: str | Path) -> str:
    """Digest the sorted relative paths and bytes of every exact direction file."""

    root = Path(directory)
    records = []
    if root.is_dir():
        for path in sorted(root.rglob("*.pt")):
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _file_sha256(path),
                }
            )
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().contiguous().cpu().numpy()
    hasher = hashlib.sha256()
    hasher.update(values.dtype.str.encode("ascii"))
    hasher.update(json.dumps(list(values.shape)).encode("ascii"))
    hasher.update(values.tobytes())
    return hasher.hexdigest()


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
