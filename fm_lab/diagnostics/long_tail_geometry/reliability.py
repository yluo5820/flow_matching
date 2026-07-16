"""Reliability statistics and the preregistered Observation-0 decision gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fm_lab.diagnostics.long_tail_geometry.measurements import (
    CheckpointMeasurements,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.utils.logging import write_json

_CENTERED = "centered_covariance"
_UNCENTERED = "uncentered_second_moment"


@dataclass(frozen=True)
class _Basis:
    vectors: torch.Tensor
    effective_rank: float


@dataclass(frozen=True)
class Observation0Decision:
    """Cross-seed result of the preregistered noise-ceiling gate."""

    status: str
    network_wide_gate_passed: bool
    required_escalation: bool
    next_action: str
    probe_phase: str
    microbatches_per_cell: int
    preregistration_digest: str
    complete_seeds: tuple[int, ...]
    incomplete_seeds: tuple[int, ...]
    seed_measurement_digests: dict[int, tuple[str, ...]]
    passing_pairs: tuple[dict[str, Any], ...]
    output_only_cells: tuple[dict[str, Any], ...]
    reliability: pd.DataFrame
    gram_matrices: dict[str, np.ndarray]

    def save(self, directory: str | Path) -> Path:
        """Write deterministic CSV, JSON, and sample-space Gram artifacts."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        reliability_path = output / "reliability.csv"
        reliability_temporary = output / "reliability.csv.tmp"
        self.reliability.to_csv(reliability_temporary, index=False)
        reliability_temporary.replace(reliability_path)

        gram_path = output / "gram_matrices.npz"
        gram_temporary = output / "gram_matrices.npz.tmp"
        with gram_temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{name: self.gram_matrices[name] for name in sorted(self.gram_matrices)},
            )
        gram_temporary.replace(gram_path)

        payload = {
            "preregistration_digest": self.preregistration_digest,
            "complete_seeds": list(self.complete_seeds),
            "incomplete_seeds": list(self.incomplete_seeds),
            "seed_measurement_digests": {
                str(seed): list(digests)
                for seed, digests in sorted(self.seed_measurement_digests.items())
            },
            "status": self.status,
            "network_wide_gate_passed": self.network_wide_gate_passed,
            "required_escalation": self.required_escalation,
            "passing_pairs": list(self.passing_pairs),
            "output_only_cells": list(self.output_only_cells),
            "next_action": self.next_action,
            "probe_phase": self.probe_phase,
            "microbatches_per_cell": self.microbatches_per_cell,
            "reliability_sha256": _file_sha256(reliability_path),
            "gram_matrices_sha256": _file_sha256(gram_path),
        }
        json_temporary = output / "noise_ceiling.json.tmp"
        write_json(payload, json_temporary)
        json_temporary.replace(output / "noise_ceiling.json")
        return output


def centered_cell_statistics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ranks: Sequence[int],
) -> pd.DataFrame:
    """Compare two cells through thin centered sample-matrix SVDs."""

    return _cell_statistics(a, b, ranks=ranks, centered=True)


def analyze_seed_reliability(
    measurements: CheckpointMeasurements | Sequence[CheckpointMeasurements],
    preregistration: Observation0Preregistration,
    *,
    training_seed: int = 0,
) -> pd.DataFrame:
    """Analyze one training seed with the locked matched-null procedure."""

    artifacts = _as_measurement_sequence(measurements)
    if any(
        artifact.preregistration_sha256 != preregistration.digest
        for artifact in artifacts
    ):
        raise ValueError("Measurement does not match the preregistration digest.")
    checkpoint_steps = [artifact.checkpoint_step for artifact in artifacts]
    if len(set(checkpoint_steps)) != len(checkpoint_steps):
        raise ValueError("Each checkpoint may appear only once in a seed analysis.")

    records: list[dict[str, Any]] = []
    bases: dict[tuple[int, int, str, int, str, int, str], _Basis] = {}
    grams: dict[str, np.ndarray] = {}
    for artifact in sorted(artifacts, key=lambda item: item.checkpoint_step):
        metadata = artifact.metadata
        base = metadata[metadata["probe_view"].isin(("a", "b"))]
        cells = (
            base[["class_id", "stratum_id"]]
            .drop_duplicates()
            .sort_values(["stratum_id", "class_id"])
        )
        views = sorted(set(str(value) for value in metadata["probe_view"]))
        for layer_name, sketches in artifact.sketches.items():
            for cell in cells.itertuples(index=False):
                class_id = int(cell.class_id)
                stratum_id = int(cell.stratum_id)
                view_rows = {
                    view: _select_rows(
                        metadata,
                        sketches,
                        view=view,
                        class_id=class_id,
                        stratum_id=stratum_id,
                    )
                    for view in views
                }
                a = view_rows.get("a")
                b = view_rows.get("b")
                if a is None or b is None or len(a) < 2 or len(b) < 2:
                    continue
                for view, rows in view_rows.items():
                    grams[
                        _gram_key(
                            training_seed=training_seed,
                            checkpoint_step=artifact.checkpoint_step,
                            view=view,
                            class_id=class_id,
                            stratum_id=stratum_id,
                            layer_name=layer_name,
                        )
                    ] = (rows @ rows.T).numpy().astype(np.float32, copy=False)

                for representation, centered in (
                    (_CENTERED, True),
                    (_UNCENTERED, False),
                ):
                    split_a = _split_half_overlap(
                        a,
                        preregistration.descriptive_ranks,
                        centered=centered,
                    )
                    source_overlap = _source_replica_overlap(
                        view_rows,
                        preregistration.descriptive_ranks,
                        centered=centered,
                    )
                    statistics = _cell_statistics(
                        a,
                        b,
                        ranks=preregistration.descriptive_ranks,
                        centered=centered,
                    )
                    for row in statistics.to_dict(orient="records"):
                        rank = int(row["rank"])
                        records.append(
                            {
                                "training_seed": training_seed,
                                "checkpoint_step": artifact.checkpoint_step,
                                "stratum_id": stratum_id,
                                "class_id": class_id,
                                "layer_name": layer_name,
                                "representation": representation,
                                **row,
                                "probe_a_split_half_overlap": split_a.get(
                                    rank, float("nan")
                                ),
                                "source_noise_replica_overlap": source_overlap.get(
                                    rank, float("nan")
                                ),
                                "null_threshold": float("nan"),
                                "measurable": False,
                            }
                        )
                        for view, values in (("a", a), ("b", b)):
                            basis = _thin_basis(values, rank=rank, centered=centered)
                            if basis is not None:
                                bases[
                                    (
                                        artifact.checkpoint_step,
                                        stratum_id,
                                        layer_name,
                                        class_id,
                                        representation,
                                        rank,
                                        view,
                                    )
                                ] = basis

    table = pd.DataFrame.from_records(records)
    if table.empty:
        raise ValueError("No complete Probe-A/Probe-B cells were available.")
    _attach_matched_null(
        table,
        bases=bases,
        preregistration=preregistration,
        training_seed=training_seed,
    )
    table = table.sort_values(
        [
            "checkpoint_step",
            "stratum_id",
            "class_id",
            "layer_name",
            "representation",
            "rank",
        ]
    ).reset_index(drop=True)
    table.attrs["measurement_digests"] = [artifact.digest for artifact in artifacts]
    table.attrs["gram_matrices"] = grams
    base_counts = []
    for artifact in artifacts:
        selected = artifact.metadata[artifact.metadata["probe_view"] == "a"]
        base_counts.extend(
            int(value)
            for value in selected.groupby(["class_id", "stratum_id"]).size()
        )
    table.attrs["microbatches_per_cell"] = min(base_counts)
    return table


def aggregate_observation0_reliability(
    seed_tables: Mapping[int, pd.DataFrame],
    preregistration: Observation0Preregistration,
) -> Observation0Decision:
    """Apply the repeat-across-seeds and adjacent-layer preregistered gate."""

    expected = set(preregistration.training_seeds)
    complete = set(int(seed) for seed in seed_tables)
    if complete != expected:
        missing = sorted(expected - complete)
        extra = sorted(complete - expected)
        raise ValueError(
            "Observation 0 requires all preregistered training seeds; "
            f"missing={missing}, extra={extra}."
        )
    combined_parts = []
    measurement_digests: dict[int, tuple[str, ...]] = {}
    grams: dict[str, np.ndarray] = {}
    microbatch_counts = []
    for seed in preregistration.training_seeds:
        table = seed_tables[seed].copy()
        table["training_seed"] = seed
        combined_parts.append(table)
        measurement_digests[seed] = tuple(
            str(value)
            for value in seed_tables[seed].attrs.get("measurement_digests", ())
        )
        for name, gram in seed_tables[seed].attrs.get("gram_matrices", {}).items():
            if name in grams:
                raise ValueError(f"Duplicate Gram-matrix key: {name}")
            grams[str(name)] = np.asarray(gram, dtype=np.float32)
        microbatch_counts.append(
            int(seed_tables[seed].attrs.get("microbatches_per_cell", 0))
        )
    allowed_counts = {
        preregistration.primary_microbatches_per_cell,
        preregistration.escalation_microbatches_per_cell,
    }
    if len(set(microbatch_counts)) != 1 or microbatch_counts[0] not in allowed_counts:
        raise ValueError(
            "Every seed table must use the same preregistered probe phase."
        )
    microbatches_per_cell = microbatch_counts[0]
    probe_phase = (
        "escalated"
        if microbatches_per_cell == preregistration.escalation_microbatches_per_cell
        else "primary"
    )
    combined = pd.concat(combined_parts, ignore_index=True)
    required = {
        "checkpoint_step",
        "stratum_id",
        "class_id",
        "layer_name",
        "rank",
        "representation",
        "measurable",
        "training_seed",
    }
    if not required <= set(combined):
        raise ValueError("Seed reliability tables are missing primary gate columns.")

    primary = combined[
        (combined["representation"] == _CENTERED)
        & (combined["rank"].isin(preregistration.gate_ranks))
        & combined["measurable"].astype(bool)
    ].copy()
    if preregistration.exclude_checkpoint_zero_from_gate:
        primary = primary[primary["checkpoint_step"] != 0]
    group_columns = [
        "checkpoint_step",
        "stratum_id",
        "rank",
        "layer_name",
        "class_id",
    ]
    repeated = (
        primary.groupby(group_columns, as_index=False)["training_seed"]
        .nunique()
        .rename(columns={"training_seed": "seed_repeats"})
    )
    repeated = repeated[
        repeated["seed_repeats"] >= preregistration.required_seed_repeats
    ]

    passing_pairs: list[dict[str, Any]] = []
    nonoutput_layers = preregistration.layers[:-1]
    fixed_columns = ["checkpoint_step", "stratum_id", "rank"]
    for fixed_values, block in repeated.groupby(fixed_columns):
        fixed = dict(zip(fixed_columns, fixed_values, strict=True))
        class_sets = {
            layer: set(
                int(value)
                for value in block.loc[block["layer_name"] == layer, "class_id"]
            )
            for layer in nonoutput_layers
        }
        for first, second in zip(nonoutput_layers, nonoutput_layers[1:], strict=False):
            common = sorted(class_sets[first] & class_sets[second])
            if len(common) >= preregistration.minimum_common_classes:
                passing_pairs.append(
                    {
                        **{name: int(value) for name, value in fixed.items()},
                        "layers": [first, second],
                        "common_classes": common,
                    }
                )

    output_only_cells: list[dict[str, Any]] = []
    output_layer = preregistration.layers[-1]
    output = repeated[repeated["layer_name"] == output_layer]
    for fixed_values, block in output.groupby(fixed_columns):
        classes = sorted(int(value) for value in block["class_id"])
        if len(classes) >= preregistration.minimum_common_classes:
            output_only_cells.append(
                {
                    **{
                        name: int(value)
                        for name, value in zip(
                            fixed_columns, fixed_values, strict=True
                        )
                    },
                    "layer": output_layer,
                    "classes": classes,
                }
            )

    escalated = probe_phase == "escalated"
    if passing_pairs:
        status = "network_wide_measurable"
        gate_passed = True
        required_escalation = False
        next_action = "calibrate_probe_a_functional_overlap_before_stage1"
    elif output_only_cells:
        status = "output_layer_only"
        gate_passed = False
        required_escalation = not escalated
        next_action = (
            "collect_32_microbatches_per_cell"
            if required_escalation
            else "stop_network_wide_stage1_and_report_output_layer_only"
        )
    elif escalated:
        status = "network_wide_practical_null"
        gate_passed = False
        required_escalation = False
        next_action = "stop_stage1_and_revise_theory"
    else:
        status = "escalate_probe_rows"
        gate_passed = False
        required_escalation = True
        next_action = "collect_32_microbatches_per_cell"

    combined = combined.sort_values(
        [
            "training_seed",
            "checkpoint_step",
            "stratum_id",
            "class_id",
            "layer_name",
            "representation",
            "rank",
        ]
    ).reset_index(drop=True)
    return Observation0Decision(
        status=status,
        network_wide_gate_passed=gate_passed,
        required_escalation=required_escalation,
        next_action=next_action,
        probe_phase=probe_phase,
        microbatches_per_cell=microbatches_per_cell,
        preregistration_digest=preregistration.digest,
        complete_seeds=preregistration.training_seeds,
        incomplete_seeds=(),
        seed_measurement_digests=measurement_digests,
        passing_pairs=tuple(passing_pairs),
        output_only_cells=tuple(output_only_cells),
        reliability=combined,
        gram_matrices=grams,
    )


def _cell_statistics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ranks: Sequence[int],
    centered: bool,
) -> pd.DataFrame:
    a = _validated_rows(a)
    b = _validated_rows(b)
    mean_cosine, concentration_a, concentration_b = _direction_statistics(a, b)
    records = []
    for rank in ranks:
        basis_a = _thin_basis(a, rank=int(rank), centered=centered)
        basis_b = _thin_basis(b, rank=int(rank), centered=centered)
        available = basis_a is not None and basis_b is not None
        overlap = float("nan")
        angle = float("nan")
        if available:
            overlap, angle = _basis_overlap(basis_a.vectors, basis_b.vectors)
        records.append(
            {
                "rank": int(rank),
                "available": available,
                "projection_overlap": overlap,
                "largest_principal_angle_degrees": angle,
                "mean_direction_cosine": mean_cosine,
                "directional_concentration_a": concentration_a,
                "directional_concentration_b": concentration_b,
                "effective_rank_a": (
                    basis_a.effective_rank if basis_a is not None else float("nan")
                ),
                "effective_rank_b": (
                    basis_b.effective_rank if basis_b is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _validated_rows(rows: torch.Tensor) -> torch.Tensor:
    values = rows.detach().float().cpu()
    if values.ndim != 2 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("Reliability cells require finite two-dimensional row matrices.")
    return values


def _thin_basis(rows: torch.Tensor, *, rank: int, centered: bool) -> _Basis | None:
    values = _validated_rows(rows)
    if centered:
        values = values - values.mean(dim=0, keepdim=True)
    maximum_rank = min(values.shape)
    if centered:
        maximum_rank = min(maximum_rank, len(values) - 1)
    if rank < 1 or rank > maximum_rank:
        return None
    _, singular_values, right = torch.linalg.svd(values, full_matrices=False)
    if singular_values.numel() < rank:
        return None
    tolerance = (
        torch.finfo(singular_values.dtype).eps
        * max(values.shape)
        * singular_values[0]
    )
    if singular_values[rank - 1] <= tolerance:
        return None
    energy = singular_values.square()
    positive = energy[energy > tolerance.square()]
    probabilities = positive / positive.sum()
    effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    return _Basis(vectors=right[:rank].contiguous(), effective_rank=effective_rank)


def _basis_overlap(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    cosines = torch.linalg.svdvals(a @ b.T).clamp(0, 1)
    overlap = float(cosines.square().mean())
    largest_angle = float(torch.rad2deg(torch.acos(cosines.min())))
    return overlap, largest_angle


def _direction_statistics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    mean_a = a.mean(dim=0)
    mean_b = b.mean(dim=0)
    norm_a = torch.linalg.vector_norm(mean_a)
    norm_b = torch.linalg.vector_norm(mean_b)
    denominator = norm_a * norm_b
    cosine = (
        float(torch.dot(mean_a, mean_b) / denominator)
        if denominator > torch.finfo(a.dtype).eps
        else float("nan")
    )
    concentration_a = float(
        norm_a / torch.linalg.vector_norm(a, dim=1).mean().clamp_min(1e-12)
    )
    concentration_b = float(
        norm_b / torch.linalg.vector_norm(b, dim=1).mean().clamp_min(1e-12)
    )
    return cosine, min(concentration_a, 1.0), min(concentration_b, 1.0)


def _split_half_overlap(
    rows: torch.Tensor,
    ranks: Sequence[int],
    *,
    centered: bool,
) -> dict[int, float]:
    first = rows[::2]
    second = rows[1::2]
    if len(first) < 2 or len(second) < 2:
        return {int(rank): float("nan") for rank in ranks}
    return {
        int(rank): _overlap_or_nan(first, second, rank=int(rank), centered=centered)
        for rank in ranks
    }


def _source_replica_overlap(
    view_rows: Mapping[str, torch.Tensor],
    ranks: Sequence[int],
    *,
    centered: bool,
) -> dict[int, float]:
    pairs = []
    for base_view in ("a", "b"):
        replicas = sorted(
            view for view in view_rows if view.startswith(f"{base_view}_source_")
        )
        for replica in replicas:
            pairs.append((view_rows[base_view], view_rows[replica]))
    result = {}
    for rank in ranks:
        values = [
            _overlap_or_nan(first, second, rank=int(rank), centered=centered)
            for first, second in pairs
        ]
        finite = [value for value in values if math.isfinite(value)]
        result[int(rank)] = float(np.mean(finite)) if finite else float("nan")
    return result


def _overlap_or_nan(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rank: int,
    centered: bool,
) -> float:
    basis_a = _thin_basis(a, rank=rank, centered=centered)
    basis_b = _thin_basis(b, rank=rank, centered=centered)
    if basis_a is None or basis_b is None:
        return float("nan")
    return _basis_overlap(basis_a.vectors, basis_b.vectors)[0]


def _attach_matched_null(
    table: pd.DataFrame,
    *,
    bases: Mapping[tuple[int, int, str, int, str, int, str], _Basis],
    preregistration: Observation0Preregistration,
    training_seed: int,
) -> None:
    for layer_name in sorted(set(str(value) for value in table["layer_name"])):
        for rank in preregistration.gate_ranks:
            blocks = []
            selected = table[
                (table["layer_name"] == layer_name)
                & (table["rank"] == rank)
                & (table["representation"] == _CENTERED)
                & table["available"].astype(bool)
            ]
            for (checkpoint_step, stratum_id), block in selected.groupby(
                ["checkpoint_step", "stratum_id"]
            ):
                classes = sorted(int(value) for value in block["class_id"])
                valid_classes = [
                    class_id
                    for class_id in classes
                    if (
                        int(checkpoint_step),
                        int(stratum_id),
                        layer_name,
                        class_id,
                        _CENTERED,
                        rank,
                        "a",
                    )
                    in bases
                    and (
                        int(checkpoint_step),
                        int(stratum_id),
                        layer_name,
                        class_id,
                        _CENTERED,
                        rank,
                        "b",
                    )
                    in bases
                ]
                if len(valid_classes) < 2:
                    continue
                matrix = np.empty((len(valid_classes), len(valid_classes)), dtype=float)
                for a_index, a_class in enumerate(valid_classes):
                    a_basis = bases[
                        (
                            int(checkpoint_step),
                            int(stratum_id),
                            layer_name,
                            a_class,
                            _CENTERED,
                            rank,
                            "a",
                        )
                    ].vectors
                    for b_index, b_class in enumerate(valid_classes):
                        b_basis = bases[
                            (
                                int(checkpoint_step),
                                int(stratum_id),
                                layer_name,
                                b_class,
                                _CENTERED,
                                rank,
                                "b",
                            )
                        ].vectors
                        matrix[a_index, b_index] = _basis_overlap(a_basis, b_basis)[0]
                blocks.append((valid_classes, matrix))
            if not blocks:
                continue
            rng = np.random.RandomState(
                _keyed_seed(preregistration.sketch_seed, layer_name, rank, training_seed)
            )
            maxima = []
            for _ in range(preregistration.null_permutations):
                selected_values = []
                for classes, matrix in blocks:
                    permutation = _random_derangement(len(classes), rng)
                    selected_values.extend(matrix[np.arange(len(classes)), permutation])
                maxima.append(float(np.max(selected_values)))
            threshold = float(
                np.quantile(maxima, preregistration.null_quantile, method="higher")
            )
            mask = (
                (table["layer_name"] == layer_name)
                & (table["rank"] == rank)
                & (table["representation"] == _CENTERED)
            )
            table.loc[mask, "null_threshold"] = threshold
            table.loc[mask, "measurable"] = (
                table.loc[mask, "available"].astype(bool)
                & (table.loc[mask, "projection_overlap"] > threshold)
            )


def _random_derangement(size: int, rng: np.random.RandomState) -> np.ndarray:
    identity = np.arange(size)
    while True:
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate


def _keyed_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _select_rows(
    metadata: pd.DataFrame,
    sketches: torch.Tensor,
    *,
    view: str,
    class_id: int,
    stratum_id: int,
) -> torch.Tensor:
    mask = (
        (metadata["probe_view"] == view)
        & (metadata["class_id"] == class_id)
        & (metadata["stratum_id"] == stratum_id)
    ).to_numpy(copy=True)
    return sketches[torch.from_numpy(mask)]


def _as_measurement_sequence(
    measurements: CheckpointMeasurements | Sequence[CheckpointMeasurements],
) -> tuple[CheckpointMeasurements, ...]:
    if isinstance(measurements, CheckpointMeasurements):
        return (measurements,)
    result = tuple(measurements)
    if not result or any(not isinstance(item, CheckpointMeasurements) for item in result):
        raise ValueError("Seed reliability requires completed checkpoint measurements.")
    return result


def _gram_key(
    *,
    training_seed: int,
    checkpoint_step: int,
    view: str,
    class_id: int,
    stratum_id: int,
    layer_name: str,
) -> str:
    safe_layer = layer_name.replace(".", "_").replace("/", "_")
    return (
        f"seed_{training_seed}__checkpoint_{checkpoint_step}__view_{view}__"
        f"class_{class_id}__stratum_{stratum_id}__layer_{safe_layer}"
    )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
