"""Fail-closed validation for long-tail gradient-geometry observations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.long_tail_geometry.checkpoints import (
    evaluate_probe_loss,
    restore_probe_model,
)
from fm_lab.diagnostics.long_tail_geometry.controls import (
    permutation_null,
    planted_low_rank_control,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import collect_gradient_rows
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeManifest,
    build_probe_manifest,
    materialize_probe_batch,
)
from fm_lab.diagnostics.long_tail_geometry.sketch import (
    CountSketchSpec,
    validate_sketch,
)
from fm_lab.experiments.factory import (
    build_path,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.training.losses import build_objective
from fm_lab.training.trainer import validate_checkpoint_compatibility
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import load_config
from fm_lab.utils.logging import write_json

_REPORT_SCHEMA_VERSION = 1


class Stage0ValidationError(RuntimeError):
    """Raised after a failed gate has been persisted to the Stage-0 report."""


class _GateFailure(ValueError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class _Settings:
    pairing_check_offsets: tuple[int, ...]
    probe_splits: tuple[str, ...]
    rows_per_class_per_stratum: int
    microbatch_size: int
    time_strata: tuple[tuple[float, float], ...]
    layers: tuple[str, ...]
    sketch_dim: int
    max_sketch_dim: int
    sketch_seed: int
    max_cosine_error: float
    max_subspace_error: float
    permutation_count: int


def run_stage0_validation(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run ordered validation gates and persist a report after every gate."""

    active_config = copy.deepcopy(config)
    checkpoint_path = Path(checkpoint_path)
    resolved_device = torch.device(device)
    artifact_dir = Path(output_dir) / "diagnostics" / "long_tail_geometry"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for stale_gradient in artifact_dir.glob("gradient_rows_*.npz"):
        stale_gradient.unlink()
    report_path = artifact_dir / "stage0_report.json"
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "passed": False,
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": _git_commit(),
            "config_sha256": _json_sha256(active_config),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": (
                _file_sha256(checkpoint_path) if checkpoint_path.is_file() else None
            ),
            "device": str(resolved_device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "checks": {},
    }
    write_json(report, report_path)
    state: dict[str, Any] = {}

    def run_gate(name: str, check: Callable[[], dict[str, Any]]) -> None:
        try:
            details = check()
        except Exception as exc:
            failure = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, _GateFailure):
                failure.update(exc.details)
            report["checks"][name] = failure
            report["failure"] = {"gate": name, "message": str(exc)}
            write_json(report, report_path)
            raise Stage0ValidationError(f"{name}: {exc}") from exc
        report["checks"][name] = {"passed": True, **details}
        write_json(report, report_path)

    run_gate(
        "config_fence",
        lambda: _validate_config_fence(active_config, checkpoint_path, state),
    )
    run_gate(
        "frequency_mappings",
        lambda: _validate_frequency_mappings(active_config, state),
    )
    run_gate(
        "paired_probe_manifests",
        lambda: _validate_manifests_and_pairing(
            active_config,
            artifact_dir,
            state,
        ),
    )
    run_gate(
        "checkpoint_replay",
        lambda: _validate_checkpoint_replay(
            active_config,
            checkpoint_path,
            resolved_device,
            state,
        ),
    )
    run_gate(
        "gradient_sketch_fidelity",
        lambda: _validate_gradient_sketches(
            artifact_dir,
            resolved_device,
            state,
        ),
    )
    run_gate("permutation_nulls", lambda: _validate_permutation_nulls(state))
    run_gate(
        "planted_low_rank_control",
        lambda: _validate_planted_control(state),
    )
    report["passed"] = True
    write_json(report, report_path)
    return report


def _validate_config_fence(
    config: dict[str, Any],
    checkpoint_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise ValueError(f"Stage-0 checkpoint does not exist: {checkpoint_path}")
    objective = config.get("objective", {}) or {}
    if str(objective.get("name", "flow_matching")).lower() != "flow_matching":
        raise ValueError("Stage-0 requires the ordinary flow_matching objective.")
    if objective.get("modifiers", []):
        raise ValueError("Stage-0 forbids objective modifiers, including CM.")
    if float(objective.get("straightness_weight", 0.0)) != 0.0:
        raise ValueError("Stage-0 forbids straightness regularization.")
    if float(objective.get("interpolant_acceleration_weight", 0.0)) != 0.0:
        raise ValueError("Stage-0 forbids interpolant acceleration regularization.")
    capacity = (config.get("model", {}) or {}).get("capacity", {}) or {}
    if bool(capacity.get("enabled", False)):
        raise ValueError("Stage-0 forbids model capacity adapters.")
    early_stopping = (config.get("training", {}) or {}).get("early_stopping", {}) or {}
    if bool(early_stopping.get("enabled", False)):
        raise ValueError("Stage-0 requires early stopping to be disabled.")
    data = config.get("data", {}) or {}
    if str(data.get("name", "")).lower() != "fashion_mnist_lt":
        raise ValueError("Stage-0 currently requires the Fashion-MNIST long-tail target.")
    mapping = data.get("frequency_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Stage-0 requires data.frequency_mapping.")
    if int(mapping.get("offset", -1)) != 0:
        raise ValueError("The Stage-0 reference frequency mapping must use offset 0.")
    if int(mapping.get("multiplier", -1)) != 3:
        raise ValueError("The preregistered Stage-0 frequency multiplier must be 3.")
    settings = _parse_settings(config)
    state["settings"] = settings
    return {
        "objective": "flow_matching",
        "objective_modifiers": [],
        "capacity_enabled": False,
        "early_stopping_enabled": False,
        "frequency_mapping": {
            "offset": 0,
            "multiplier": int(mapping.get("multiplier", 3)),
            "diagnostic_pool_per_class": int(
                mapping.get("diagnostic_pool_per_class", 0)
            ),
        },
        "thresholds": {
            "max_cosine_error": settings.max_cosine_error,
            "max_subspace_error": settings.max_subspace_error,
            "permutation_p_value_minimum": 0.05,
            "planted_subspace_overlap_minimum": 0.95,
        },
    }


def _parse_settings(config: dict[str, Any]) -> _Settings:
    diagnostics = config.get("diagnostics", {}) or {}
    raw = diagnostics.get("long_tail_geometry")
    if not isinstance(raw, dict):
        raise ValueError("Missing diagnostics.long_tail_geometry configuration.")
    offsets = tuple(int(value) for value in raw.get("pairing_check_offsets", ()))
    if not offsets or 0 not in offsets or len(set(offsets)) != len(offsets):
        raise ValueError("pairing_check_offsets must be unique and include offset 0.")
    if any(value < 0 or value >= 10 for value in offsets):
        raise ValueError("pairing_check_offsets must lie in [0, 9].")
    splits = tuple(str(value).lower() for value in raw.get("probe_splits", ()))
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"a", "b"}:
        raise ValueError("probe_splits must be a non-empty subset of ['a', 'b'].")
    strata = tuple(
        (float(interval[0]), float(interval[1]))
        for interval in raw.get("time_strata", ())
    )
    if not strata or any(not 0 <= low < high <= 1 for low, high in strata):
        raise ValueError("time_strata must contain valid intervals in [0, 1].")
    layers = tuple(str(value) for value in raw.get("layers", ()))
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Stage-0 layers must be non-empty and unique.")
    rows = int(raw.get("rows_per_class_per_stratum", 0))
    microbatch = int(raw.get("microbatch_size", 0))
    if rows < 1 or microbatch < 1 or rows % microbatch:
        raise ValueError(
            "rows_per_class_per_stratum must be positive and divisible by microbatch_size."
        )
    sketch_dim = int(raw.get("sketch_dim", 0))
    max_sketch_dim = int(raw.get("max_sketch_dim", 0))
    if sketch_dim < 2 or max_sketch_dim < sketch_dim:
        raise ValueError("Sketch dimensions must satisfy 2 <= sketch_dim <= max_sketch_dim.")
    max_cosine_error = float(raw.get("max_cosine_error", -1.0))
    max_subspace_error = float(raw.get("max_subspace_error", -1.0))
    if max_cosine_error < 0 or max_subspace_error < 0:
        raise ValueError("Sketch-error thresholds must be non-negative.")
    permutations = int(raw.get("permutation_count", 0))
    if permutations < 1:
        raise ValueError("permutation_count must be positive.")
    return _Settings(
        pairing_check_offsets=offsets,
        probe_splits=splits,
        rows_per_class_per_stratum=rows,
        microbatch_size=microbatch,
        time_strata=strata,
        layers=layers,
        sketch_dim=sketch_dim,
        max_sketch_dim=max_sketch_dim,
        sketch_seed=int(raw.get("sketch_seed", 0)),
        max_cosine_error=max_cosine_error,
        max_subspace_error=max_subspace_error,
        permutation_count=permutations,
    )


def _validate_frequency_mappings(
    config: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    probe_ids: dict[str, np.ndarray] = {}
    ranks_by_offset: list[list[int]] = []
    counts_by_offset: list[list[int]] = []
    class_subsets: dict[int, list[set[int]]] = {class_id: [] for class_id in range(10)}
    base_target = None
    base_metadata: dict[str, Any] | None = None
    for offset in range(10):
        target = _build_mapping_target(config, offset)
        metadata = target.metadata()
        mapping = metadata.get("frequency_mapping", {})
        ranks = [int(value) for value in mapping.get("class_ranks", ())]
        if len(ranks) != 10:
            raise ValueError(f"Frequency mapping offset {offset} lacks ten class ranks.")
        ranks_by_offset.append(ranks)
        counts_by_offset.append([int(value) for value in target.class_counts])
        selected = target.selected_indices
        selected_labels = target.labels.cpu().numpy()
        for class_id in range(10):
            class_subsets[class_id].append(
                set(int(value) for value in selected[selected_labels == class_id])
            )
        for split in ("a", "b"):
            current = target.diagnostic_indices(split).astype(np.int64)
            if split not in probe_ids:
                probe_ids[split] = current
            elif not np.array_equal(probe_ids[split], current):
                raise ValueError(
                    f"Probe-{split.upper()} IDs changed at frequency offset {offset}."
                )
        if offset == 0:
            base_target = target
            base_metadata = metadata

    ranks_array = np.asarray(ranks_by_offset, dtype=np.int64)
    expected = np.arange(10, dtype=np.int64)
    rows_balanced = all(np.array_equal(np.sort(row), expected) for row in ranks_array)
    columns_balanced = all(
        np.array_equal(np.sort(ranks_array[:, class_id]), expected)
        for class_id in range(10)
    )
    if not rows_balanced or not columns_balanced:
        raise ValueError("The ten counterfactual mappings do not form a balanced Latin map.")
    for class_id, subsets in class_subsets.items():
        ordered = sorted(subsets, key=len)
        if any(
            not left <= right
            for left, right in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError(f"Class {class_id} training subsets are not nested.")
    assert base_target is not None
    assert base_metadata is not None
    training_ids = set(int(value) for value in base_target.selected_indices)
    probe_a_ids = set(int(value) for value in probe_ids["a"])
    probe_b_ids = set(int(value) for value in probe_ids["b"])
    if training_ids & probe_a_ids or training_ids & probe_b_ids:
        raise ValueError("A diagnostic probe overlaps the training subset.")
    if probe_a_ids & probe_b_ids:
        raise ValueError("Probe-A and Probe-B are not disjoint.")
    state["target"] = base_target
    state["ranks_by_offset"] = ranks_array
    return {
        "latin_rows_balanced": rows_balanced,
        "latin_columns_balanced": columns_balanced,
        "nested_class_subsets": True,
        "training_probe_disjoint": True,
        "probe_splits_disjoint": True,
        "class_ranks_by_offset": ranks_by_offset,
        "class_counts_by_offset": counts_by_offset,
        "target_metadata": base_metadata,
        "probe_id_sha256": {
            split: _array_sha256(values) for split, values in probe_ids.items()
        },
    }


def _validate_manifests_and_pairing(
    config: dict[str, Any],
    artifact_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    settings: _Settings = state["settings"]
    target = state["target"]
    source = build_source(config)
    manifests: dict[str, ProbeManifest] = {}
    manifest_details: dict[str, dict[str, Any]] = {}
    base_seed = int((config.get("experiment", {}) or {}).get("seed", 0))
    for split_index, split in enumerate(settings.probe_splits):
        _, labels, original_ids = target.diagnostic_samples(split)
        manifest = build_probe_manifest(
            original_ids.astype(np.int64),
            labels.cpu().numpy(),
            split=split,
            rows_per_class_per_stratum=settings.rows_per_class_per_stratum,
            batch_size=settings.microbatch_size,
            time_strata=settings.time_strata,
            seed=base_seed + 1009 * (split_index + 1),
        )
        path = manifest.save(artifact_dir / f"probe_{split}.npz")
        restored = ProbeManifest.load(path)
        if restored.digest != manifest.digest:
            raise ValueError(f"Probe-{split.upper()} manifest failed digest round-trip.")
        manifests[split] = restored
        manifest_details[split] = {
            "digest": restored.digest,
            "rows": restored.num_rows,
            "microbatches": len(restored.microbatch_row_indices()),
        }

    tuple_hashes: dict[str, dict[str, str]] = {}
    baseline_hashes: dict[str, str] | None = None
    for offset in settings.pairing_check_offsets:
        mapped_target = target if offset == 0 else _build_mapping_target(config, offset)
        hashes = {
            split: _probe_tuple_sha256(mapped_target, source, manifest)
            for split, manifest in manifests.items()
        }
        tuple_hashes[str(offset)] = hashes
        if baseline_hashes is None:
            baseline_hashes = hashes
        elif hashes != baseline_hashes:
            raise ValueError(
                f"Frequency offset {offset} does not preserve paired probe tuples."
            )
    state["source"] = source
    state["manifests"] = manifests
    return {
        "manifests": manifest_details,
        "pairing_check_offsets": list(settings.pairing_check_offsets),
        "tuple_sha256_by_offset": tuple_hashes,
    }


def _validate_checkpoint_replay(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    state: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint_compatibility(checkpoint, active_config=config)
    first_model, first_config = restore_probe_model(checkpoint_path, device=device)
    second_model, second_config = restore_probe_model(checkpoint_path, device=device)
    if first_config != second_config:
        raise ValueError("Independent checkpoint restores returned different configs.")
    checkpoint_state = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    checkpoint_state_sha256 = _state_dict_sha256(checkpoint_state)
    first_state_sha256 = _state_dict_sha256(first_model.state_dict())
    second_state_sha256 = _state_dict_sha256(second_model.state_dict())
    if len({checkpoint_state_sha256, first_state_sha256, second_state_sha256}) != 1:
        raise ValueError("Restored model parameters differ from the checkpoint payload.")
    target = state["target"]
    source = build_source(first_config)
    path = build_path(first_config)
    objective = build_objective(
        first_config.get("objective", {}),
        class_counts=target.class_counts,
    )
    manifests: dict[str, ProbeManifest] = state["manifests"]
    replay_split = "b" if "b" in manifests else next(iter(manifests))
    manifest = manifests[replay_split]
    first = evaluate_probe_loss(
        model=first_model,
        objective=objective,
        path=path,
        manifest=manifest,
        target=target,
        source=source,
        device=device,
    )
    second = evaluate_probe_loss(
        model=second_model,
        objective=objective,
        path=path,
        manifest=manifest,
        target=target,
        source=source,
        device=device,
    )
    if first.row_losses_sha256 != second.row_losses_sha256 or not torch.equal(
        first.row_losses,
        second.row_losses,
    ):
        raise ValueError("Independent checkpoint restores changed exact probe row losses.")
    state.update(
        {
            "model": first_model,
            "objective": objective,
            "path": path,
            "source": source,
        }
    )
    return {
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "replay_split": replay_split,
        "row_count": int(first.row_losses.numel()),
        "mean_loss": first.mean_loss,
        "row_losses_sha256": first.row_losses_sha256,
        "model_state_sha256": checkpoint_state_sha256,
        "independent_reload_bitwise_equal": True,
    }


def _validate_gradient_sketches(
    artifact_dir: Path,
    device: torch.device,
    state: dict[str, Any],
) -> dict[str, Any]:
    settings: _Settings = state["settings"]
    manifests: dict[str, ProbeManifest] = state["manifests"]
    split = "a" if "a" in manifests else next(iter(manifests))
    manifest = manifests[split]
    all_microbatches = manifest.microbatch_row_indices()
    validation_count = min(
        len(all_microbatches),
        max(5, int(math.ceil(0.05 * len(all_microbatches)))),
    )
    seed_material = int(manifest.digest[:16], 16) % (2**32)
    rng = np.random.RandomState(seed_material)
    chosen_ids = np.sort(rng.permutation(len(all_microbatches))[:validation_count])
    batches = [
        materialize_probe_batch(
            state["target"],
            state["source"],
            manifest,
            all_microbatches[int(batch_id)],
            device=device,
        )
        for batch_id in chosen_ids
    ]
    gradient_rows = collect_gradient_rows(
        model=state["model"],
        objective=state["objective"],
        path=state["path"],
        batches=batches,
        layer_names=settings.layers,
    )
    layer_details: dict[str, dict[str, Any]] = {}
    final_sketches: dict[str, torch.Tensor] = {}
    for layer_name in settings.layers:
        rows = gradient_rows[layer_name]
        input_dim = int(rows.raw.shape[1])
        rank = min(4, int(rows.raw.shape[0]))
        attempts: list[dict[str, Any]] = []
        dimension = min(settings.sketch_dim, input_dim)
        while True:
            if dimension >= input_dim:
                sketched = rows.raw
                validation = validate_sketch(rows.raw, sketched, rank=rank)
                method = "exact_identity"
            else:
                spec = CountSketchSpec.build(
                    input_dim=input_dim,
                    output_dim=dimension,
                    seed=settings.sketch_seed,
                )
                sketched = spec.apply(rows.raw)
                validation = validate_sketch(rows.raw, sketched, rank=rank)
                method = "countsketch"
            passed = (
                validation.max_absolute_cosine_error <= settings.max_cosine_error
                and validation.normalized_subspace_overlap_error
                <= settings.max_subspace_error
            )
            attempts.append(
                {
                    "method": method,
                    "dimension": dimension,
                    "max_absolute_cosine_error": (
                        validation.max_absolute_cosine_error
                    ),
                    "normalized_subspace_overlap_error": (
                        validation.normalized_subspace_overlap_error
                    ),
                    "passed": passed,
                }
            )
            if passed:
                final_sketches[layer_name] = sketched
                break
            if dimension >= min(settings.max_sketch_dim, input_dim):
                details = {
                    "validation_split": split,
                    "selected_microbatch_ids": chosen_ids.tolist(),
                    "layers": {**layer_details, layer_name: {"attempts": attempts}},
                }
                raise _GateFailure(
                    f"Layer {layer_name} failed the preregistered sketch thresholds.",
                    details,
                )
            dimension = min(2 * dimension, settings.max_sketch_dim, input_dim)
        layer_details[layer_name] = {
            "parameter_dimension": input_dim,
            "gradient_rows": int(rows.raw.shape[0]),
            "validation_rank": rank,
            "attempts": attempts,
            "selected_dimension": int(final_sketches[layer_name].shape[1]),
        }

    for layer_name, rows in gradient_rows.items():
        safe_name = layer_name.replace(".", "__")
        np.savez_compressed(
            artifact_dir / f"gradient_rows_{safe_name}.npz",
            layer=np.asarray(layer_name),
            selected_microbatch_ids=chosen_ids,
            raw=rows.raw.numpy(),
            norms=rows.norms.numpy(),
            normalized=rows.normalized.numpy(),
            sketch=final_sketches[layer_name].numpy(),
        )
    return {
        "validation_split": split,
        "selection_rule": "seeded 5% of manifest microbatches, minimum five",
        "selected_microbatch_ids": chosen_ids.tolist(),
        "selected_fraction": validation_count / len(all_microbatches),
        "layers": layer_details,
    }


def _validate_permutation_nulls(state: dict[str, Any]) -> dict[str, Any]:
    settings: _Settings = state["settings"]
    values = np.ones((40, 4), dtype=np.float64)
    class_labels = np.tile(np.arange(10, dtype=np.int64), 4)
    reference_ranks = np.asarray(state["ranks_by_offset"], dtype=np.int64)[0]
    rank_labels = reference_ranks[class_labels]

    def maximum_group_gap(rows: np.ndarray, labels: np.ndarray) -> float:
        overall = rows.mean(axis=0)
        return max(
            float(np.linalg.norm(rows[labels == label].mean(axis=0) - overall))
            for label in np.unique(labels)
        )

    class_result = permutation_null(
        values,
        class_labels,
        statistic=maximum_group_gap,
        permutations=settings.permutation_count,
        seed=settings.sketch_seed + 1,
    )
    rank_result = permutation_null(
        values,
        rank_labels,
        statistic=maximum_group_gap,
        permutations=settings.permutation_count,
        seed=settings.sketch_seed + 2,
    )
    details = {
        "class_labels": {
            "observed": class_result.observed,
            "p_value": class_result.p_value,
        },
        "frequency_ranks": {
            "observed": rank_result.observed,
            "p_value": rank_result.p_value,
        },
        "required_p_value_above": 0.05,
    }
    if class_result.p_value <= 0.05 or rank_result.p_value <= 0.05:
        raise _GateFailure("Exchangeable permutation controls were not null.", details)
    return details


def _validate_planted_control(state: dict[str, Any]) -> dict[str, Any]:
    settings: _Settings = state["settings"]
    result = planted_low_rank_control(
        ambient_dim=256,
        rank=4,
        rows=128,
        noise_std=0.02,
        seed=settings.sketch_seed + 3,
    )
    details = {
        "planted_rank": result.planted_rank,
        "recovered_rank": result.recovered_rank,
        "subspace_overlap": result.subspace_overlap,
        "required_subspace_overlap": 0.95,
        "leading_eigenvalues": result.eigenvalues[:8].tolist(),
    }
    if result.recovered_rank != result.planted_rank:
        raise _GateFailure("Planted control did not recover the exact rank.", details)
    if result.subspace_overlap <= 0.95:
        raise _GateFailure("Planted control subspace overlap was too low.", details)
    return details


def _build_mapping_target(config: dict[str, Any], offset: int) -> Any:
    mapped_config = copy.deepcopy(config)
    data = mapped_config.setdefault("data", {})
    mapping = data.setdefault("frequency_mapping", {})
    mapping["offset"] = int(offset)
    return build_target(mapped_config)


def _probe_tuple_sha256(target: Any, source: Any, manifest: ProbeManifest) -> str:
    hasher = hashlib.sha256()
    for rows in manifest.microbatch_row_indices():
        batch = materialize_probe_batch(
            target,
            source,
            manifest,
            rows,
            device=torch.device("cpu"),
        )
        for name, value in (
            ("x0", batch.x0),
            ("x1", batch.x1),
            ("t", batch.t),
            ("labels", batch.labels),
            ("original_indices", batch.original_indices),
            ("stratum_ids", batch.stratum_ids),
            ("microbatch_ids", batch.microbatch_ids),
        ):
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
            else:
                array = np.ascontiguousarray(value)
            hasher.update(name.encode("utf-8"))
            hasher.update(array.dtype.str.encode("ascii"))
            hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            hasher.update(array.tobytes())
    return hasher.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Model state entry is not a tensor: {name}")
        array = value.detach().cpu().contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(array.dtype.str.encode("ascii"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the long-tail gradient-geometry observation pipeline."
    )
    parser.add_argument("--config", required=True, help="Stage-0 YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Ordinary-FM checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Validation artifact root.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage0_validation(
        config=load_config(args.config),
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=resolve_device(args.device),
    )
    print(
        Path(args.output_dir)
        / "diagnostics"
        / "long_tail_geometry"
        / "stage0_report.json"
    )


if __name__ == "__main__":
    main()
