"""Resumable preparation, measurement, and analysis for Observation 0."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fm_lab.diagnostics.long_tail_geometry.checkpoints import restore_probe_model
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeBatch,
    ProbeManifest,
    build_probe_manifest,
    build_source_noise_replica,
    materialize_probe_batch,
)
from fm_lab.diagnostics.long_tail_geometry.measurements import (
    CheckpointMeasurements,
    collect_checkpoint_measurements,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.diagnostics.long_tail_geometry.registry import (
    prepare_observation0_registry,
    update_observation0_run,
)
from fm_lab.diagnostics.long_tail_geometry.reliability import (
    Observation0Decision,
    aggregate_observation0_reliability,
    analyze_seed_reliability,
)
from fm_lab.experiments.factory import build_path, build_source, build_target
from fm_lab.training.losses import build_objective
from fm_lab.training.trainer import validate_checkpoint_compatibility
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import load_config, save_config
from fm_lab.utils.logging import write_json


@dataclass(frozen=True)
class Observation0Preparation:
    preregistration: Observation0Preregistration
    study_dir: Path
    run_configs: tuple[Path, ...]
    run_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class Observation0CollectionSummary:
    seed: int
    phase: str
    completed_steps: tuple[int, ...]
    skipped_steps: tuple[int, ...]
    measurement_digest: str


def prepare_observation0_study(
    preregistration_path: str | Path,
    study_dir: str | Path,
) -> Observation0Preparation:
    """Lock the protocol and write one immutable ordinary-FM config per seed."""

    preregistration_path = Path(preregistration_path)
    preregistration = Observation0Preregistration.load(preregistration_path)
    base_path = _resolve_base_config(preregistration, preregistration_path)
    base_config = load_config(base_path)
    _validate_base_config(base_config, preregistration)
    root = Path(study_dir)
    prepare_observation0_registry(preregistration, root)

    run_configs = []
    run_dirs = []
    for seed in preregistration.training_seeds:
        run_dir = root / "mapping_0" / f"seed_{seed}"
        config = _prepared_seed_config(
            base_config,
            preregistration=preregistration,
            seed=seed,
            run_dir=run_dir,
        )
        config_path = root / "configs" / f"seed_{seed}.yaml"
        _lock_config(config, config_path)
        run_configs.append(config_path)
        run_dirs.append(run_dir)
    return Observation0Preparation(
        preregistration=preregistration,
        study_dir=root,
        run_configs=tuple(run_configs),
        run_dirs=tuple(run_dirs),
    )


def collect_observation0_run(
    *,
    preregistration: Observation0Preregistration,
    study_dir: str | Path,
    run_dir: str | Path,
    device: torch.device,
    escalated: bool = False,
) -> Observation0CollectionSummary:
    """Collect every locked checkpoint for one seed, skipping valid artifacts."""

    root = Path(study_dir)
    run_dir = Path(run_dir)
    _validate_locked_preregistration(preregistration, root)
    seed, registered_run_dir = _registered_seed(root, run_dir, preregistration)
    run_dir = registered_run_dir
    config_path = root / "configs" / f"seed_{seed}.yaml"
    config = load_config(config_path)
    expected = _prepared_seed_config(
        load_config(_resolve_base_config(preregistration, root / "aggregate/preregistration.yaml")),
        preregistration=preregistration,
        seed=seed,
        run_dir=run_dir,
    )
    if config != expected:
        raise ValueError(f"Prepared seed config has changed: {config_path}")

    checkpoint_paths = {
        step: run_dir / "checkpoints" / f"step_{step:06d}.pt"
        for step in preregistration.checkpoint_steps
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Observation-0 run has a missing checkpoint: {missing[0]}")

    target = build_target(config)
    source = build_source(config)
    path = build_path(config)
    objective = build_objective(
        config.get("objective", {}),
        diffusion_config=config.get("diffusion", {}),
        class_counts=getattr(target, "class_counts", None),
    )
    phase = "escalated" if escalated else "primary"
    manifests = _prepare_manifests(
        preregistration=preregistration,
        study_dir=root,
        target=target,
        phase=phase,
    )
    manifest_digests = {
        view: manifest.digest for view, manifest in manifests.items()
    }

    completed = []
    skipped = []
    artifacts = []
    for step, checkpoint_path in checkpoint_paths.items():
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        if int(checkpoint.get("step", -1)) != step:
            raise ValueError(
                f"Checkpoint step does not match its preregistered slot: {checkpoint_path}"
            )
        validate_checkpoint_compatibility(checkpoint, active_config=config)
        embedded_config = checkpoint.get("config")
        if embedded_config != config:
            raise ValueError(
                f"Checkpoint config differs from the prepared seed config: {checkpoint_path}"
            )
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        artifact_dir = (
            run_dir
            / "diagnostics"
            / "long_tail_geometry"
            / "observation0"
            / phase
            / f"checkpoint_{step:06d}"
        )
        complete_path = artifact_dir / "complete.json"
        if complete_path.is_file():
            artifact = CheckpointMeasurements.load(artifact_dir)
            _validate_measurement_identity(
                artifact,
                preregistration=preregistration,
                checkpoint_step=step,
                checkpoint_sha256=checkpoint_sha256,
                manifest_digests=manifest_digests,
            )
            skipped.append(step)
        else:
            model, restored_config = restore_probe_model(
                checkpoint_path,
                device=torch.device(device),
            )
            if restored_config != config:
                raise ValueError("Restored checkpoint config changed during validation.")
            artifact = collect_checkpoint_measurements(
                model=model,
                objective=objective,
                path=path,
                batches_by_view={
                    view: _materialized_batches(
                        target=target,
                        source=source,
                        manifest=manifest,
                        device=torch.device(device),
                    )
                    for view, manifest in manifests.items()
                },
                layer_names=preregistration.layers,
                sketch_dim=preregistration.sketch_dim,
                sketch_seed=preregistration.sketch_seed,
                checkpoint_step=step,
                checkpoint_sha256=checkpoint_sha256,
                preregistration_sha256=preregistration.digest,
                manifest_digests=manifest_digests,
            )
            artifact.save(artifact_dir)
            completed.append(step)
        artifacts.append(artifact)

    measurement_digest = _measurement_set_digest(artifacts, phase=phase)
    update_observation0_run(
        root,
        seed=seed,
        status="measured",
        run_dir=run_dir,
        measurement_digest=measurement_digest,
    )
    return Observation0CollectionSummary(
        seed=seed,
        phase=phase,
        completed_steps=tuple(completed),
        skipped_steps=tuple(skipped),
        measurement_digest=measurement_digest,
    )


def analyze_observation0_study(
    *,
    preregistration: Observation0Preregistration,
    study_dir: str | Path,
    escalated: bool = False,
) -> Observation0Decision:
    """Load only registry-listed measurements and apply the locked cross-seed gate."""

    root = Path(study_dir)
    _validate_locked_preregistration(preregistration, root)
    registry = pd.read_csv(
        root / "aggregate" / "run_registry.csv",
        keep_default_na=False,
    )
    if set(registry["study_digest"]) != {preregistration.digest}:
        raise ValueError("Run registry belongs to a different preregistration.")
    if tuple(int(value) for value in registry["seed"]) != preregistration.training_seeds:
        raise ValueError("Run registry seed rows changed after preparation.")
    if set(int(value) for value in registry["mapping_offset"]) != {0}:
        raise ValueError("Observation-0 analysis found a nonzero frequency mapping.")
    expected_seeds = set(preregistration.training_seeds)
    measured = registry[registry["status"] == "measured"]
    measured_seeds = set(int(value) for value in measured["seed"])
    if measured_seeds != expected_seeds or len(measured) != len(expected_seeds):
        raise ValueError(
            "Observation 0 requires all preregistered training seeds to be measured."
        )

    phase = "escalated" if escalated else "primary"
    expected_microbatches = (
        preregistration.escalation_microbatches_per_cell
        if escalated
        else preregistration.primary_microbatches_per_cell
    )
    common_manifest_digests: dict[str, str] | None = None
    seed_tables = {}
    for seed in preregistration.training_seeds:
        row = measured[measured["seed"] == seed]
        if len(row) != 1:
            raise ValueError(f"Registry contains an ambiguous row for seed {seed}.")
        run_dir = Path(str(row.iloc[0]["run_dir"]))
        seed_config = load_config(root / "configs" / f"seed_{seed}.yaml")
        num_classes = int(
            (seed_config.get("conditioning", {}) or {}).get("num_classes", 0)
        )
        if num_classes < 2:
            raise ValueError("Observation-0 analysis requires conditional class metadata.")
        artifacts = []
        for step in preregistration.checkpoint_steps:
            artifact_dir = (
                run_dir
                / "diagnostics"
                / "long_tail_geometry"
                / "observation0"
                / phase
                / f"checkpoint_{step:06d}"
            )
            artifact = CheckpointMeasurements.load(artifact_dir)
            _validate_analysis_artifact(
                artifact,
                preregistration=preregistration,
                checkpoint_step=step,
                expected_microbatches=expected_microbatches,
                expected_classes=set(range(num_classes)),
            )
            if common_manifest_digests is None:
                common_manifest_digests = artifact.manifest_digests
            elif artifact.manifest_digests != common_manifest_digests:
                raise ValueError(
                    "Registry-listed seeds do not share the same probe manifests."
                )
            artifacts.append(artifact)
        digest = _measurement_set_digest(artifacts, phase=phase)
        if digest != str(row.iloc[0]["measurement_digest"]):
            raise ValueError(
                f"Registry measurement digest does not match seed {seed} artifacts."
            )
        seed_tables[seed] = analyze_seed_reliability(
            artifacts,
            preregistration,
            training_seed=seed,
        )
    decision = aggregate_observation0_reliability(seed_tables, preregistration)
    decision.save(root / "aggregate")
    return decision


def _resolve_base_config(
    preregistration: Observation0Preregistration,
    preregistration_path: Path,
) -> Path:
    configured = Path(preregistration.base_config)
    candidates = (configured, preregistration_path.parent / configured)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"Observation-0 base config does not exist: {preregistration.base_config}"
    )


def _validate_base_config(
    config: Mapping[str, Any],
    preregistration: Observation0Preregistration,
) -> None:
    objective = config.get("objective", {}) or {}
    straightness = objective.get("straightness", {}) or {}
    interpolant = objective.get("interpolant_acceleration", {}) or {}
    learned_interpolant = objective.get("learned_interpolant", {}) or {}
    ordinary = (
        str(objective.get("name", "flow_matching")).lower() == "flow_matching"
        and not objective.get("modifiers", [])
        and float(objective.get("straightness_weight", 0.0)) == 0.0
        and float(straightness.get("weight", 0.0)) == 0.0
        and float(objective.get("interpolant_acceleration_weight", 0.0)) == 0.0
        and float(interpolant.get("weight", 0.0)) == 0.0
        and not bool(learned_interpolant.get("enabled", False))
    )
    if not ordinary:
        raise ValueError("Observation 0 requires ordinary flow matching without modifiers.")
    if str((config.get("path", {}) or {}).get("name", "linear")).lower() != "linear":
        raise ValueError("Observation 0 requires the ordinary linear flow path.")
    capacity = ((config.get("model", {}) or {}).get("capacity", {}) or {})
    if bool(capacity.get("enabled", False)):
        raise ValueError("Observation 0 requires capacity adapters to remain disabled.")
    training = config.get("training", {}) or {}
    early_stopping = training.get("early_stopping", {}) or {}
    ema = training.get("ema", {}) or {}
    if (
        bool(early_stopping.get("enabled", False))
        or bool(ema.get("enabled", False))
        or training.get("ema_decay") is not None
    ):
        raise ValueError("Observation 0 requires early stopping and EMA to be disabled.")
    if int(training.get("steps", -1)) < max(preregistration.checkpoint_steps):
        raise ValueError("Training budget does not reach every preregistered checkpoint.")
    data = config.get("data", {}) or {}
    mapping = data.get("frequency_mapping", {}) or {}
    if str(data.get("name", "")).lower() != preregistration.dataset:
        raise ValueError("Observation-0 base config has the wrong dataset.")
    if int(mapping.get("offset", -1)) != 0 or int(
        mapping.get("multiplier", -1)
    ) != preregistration.frequency_multiplier:
        raise ValueError("Observation-0 base config has the wrong frequency mapping.")


def _prepared_seed_config(
    base_config: Mapping[str, Any],
    *,
    preregistration: Observation0Preregistration,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config.setdefault("experiment", {})["seed"] = int(seed)
    config["experiment"]["output_dir"] = str(run_dir)
    mapping = config.setdefault("data", {}).setdefault("frequency_mapping", {})
    mapping["offset"] = 0
    mapping["multiplier"] = preregistration.frequency_multiplier
    objective = config.setdefault("objective", {})
    objective["name"] = "flow_matching"
    objective["modifiers"] = []
    config.setdefault("model", {}).setdefault("capacity", {})["enabled"] = False
    training = config.setdefault("training", {})
    training["checkpoint_steps"] = list(preregistration.checkpoint_steps)
    training.setdefault("early_stopping", {})["enabled"] = False
    training.setdefault("ema", {})["enabled"] = False
    training["ema_decay"] = None
    diagnostics = config.setdefault("diagnostics", {}).setdefault(
        "long_tail_geometry", {}
    )
    diagnostics.update(
        {
            "observation0_preregistration_sha256": preregistration.digest,
            "probe_splits": list(preregistration.probe_splits),
            "rows_per_class_per_stratum": (
                preregistration.primary_rows_per_class_per_stratum
            ),
            "microbatch_size": preregistration.microbatch_size,
            "time_strata": [list(values) for values in preregistration.time_strata],
            "layers": list(preregistration.layers),
            "sketch_dim": preregistration.sketch_dim,
            "max_sketch_dim": preregistration.max_sketch_dim,
            "sketch_seed": preregistration.sketch_seed,
            "permutation_count": preregistration.null_permutations,
        }
    )
    return config


def _lock_config(config: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        if load_config(path) != dict(config):
            raise ValueError(f"Refusing to replace a prepared seed config: {path}")
        return
    save_config(config, path)


def _validate_locked_preregistration(
    preregistration: Observation0Preregistration,
    study_dir: Path,
) -> None:
    locked = Observation0Preregistration.load(
        study_dir / "aggregate" / "preregistration.yaml"
    )
    if locked.digest != preregistration.digest:
        raise ValueError("Study directory has a different locked preregistration.")


def _registered_seed(
    study_dir: Path,
    run_dir: Path,
    preregistration: Observation0Preregistration,
) -> tuple[int, Path]:
    registry = pd.read_csv(
        study_dir / "aggregate" / "run_registry.csv",
        keep_default_na=False,
    )
    if set(registry["study_digest"]) != {preregistration.digest}:
        raise ValueError("Run registry does not match the preregistration.")
    requested = run_dir.expanduser().resolve()
    matches = registry[
        registry["run_dir"].map(lambda value: Path(str(value)).expanduser().resolve())
        == requested
    ]
    if len(matches) != 1:
        raise ValueError("Run directory is not uniquely registered for Observation 0.")
    if int(matches.iloc[0]["mapping_offset"]) != 0:
        raise ValueError("Observation 0 cannot collect a nonzero frequency mapping.")
    return int(matches.iloc[0]["seed"]), Path(str(matches.iloc[0]["run_dir"]))


def _prepare_manifests(
    *,
    preregistration: Observation0Preregistration,
    study_dir: Path,
    target: Any,
    phase: str,
) -> dict[str, ProbeManifest]:
    rows = (
        preregistration.escalation_rows_per_class_per_stratum
        if phase == "escalated"
        else preregistration.primary_rows_per_class_per_stratum
    )
    manifest_dir = study_dir / "aggregate" / "manifests" / phase
    base_manifests = {}
    for index, split in enumerate(preregistration.probe_splits):
        sample_ids = np.asarray(target.diagnostic_indices(split), dtype=np.int64)
        _, labels, returned_ids = target.diagnostic_samples(
            split,
            original_indices=sample_ids,
            device="cpu",
        )
        if not np.array_equal(np.asarray(returned_ids, dtype=np.int64), sample_ids):
            raise ValueError(f"Probe-{split.upper()} target IDs changed order.")
        expected = build_probe_manifest(
            sample_ids,
            labels.detach().cpu().numpy(),
            split=split,
            rows_per_class_per_stratum=rows,
            batch_size=preregistration.microbatch_size,
            time_strata=preregistration.time_strata,
            seed=preregistration.manifest_seed + index,
        )
        path = manifest_dir / f"probe_{split}.npz"
        if path.exists():
            stored = ProbeManifest.load(path)
            if stored.digest != expected.digest:
                raise ValueError(f"Stored Probe-{split.upper()} manifest changed identity.")
        else:
            expected.save(path)
            stored = expected
        base_manifests[split] = stored
    manifests = {
        "a": base_manifests["a"],
        "a_source_1": build_source_noise_replica(
            base_manifests["a"],
            seed=preregistration.manifest_seed + 10_001,
        ),
        "b": base_manifests["b"],
        "b_source_1": build_source_noise_replica(
            base_manifests["b"],
            seed=preregistration.manifest_seed + 20_001,
        ),
    }
    write_json(
        {
            "phase": phase,
            "preregistration_sha256": preregistration.digest,
            "digests": {view: manifest.digest for view, manifest in manifests.items()},
        },
        manifest_dir / "manifest_digests.json",
    )
    return manifests


def _materialized_batches(
    *,
    target: Any,
    source: Any,
    manifest: ProbeManifest,
    device: torch.device,
) -> Iterator[ProbeBatch]:
    for rows in manifest.microbatch_row_indices():
        yield materialize_probe_batch(
            target,
            source,
            manifest,
            rows,
            device=device,
        )


def _validate_measurement_identity(
    artifact: CheckpointMeasurements,
    *,
    preregistration: Observation0Preregistration,
    checkpoint_step: int,
    checkpoint_sha256: str,
    manifest_digests: Mapping[str, str],
) -> None:
    if artifact.checkpoint_step != checkpoint_step:
        raise ValueError("Completed measurement has the wrong checkpoint step.")
    if artifact.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("Completed measurement belongs to a changed checkpoint.")
    if artifact.preregistration_sha256 != preregistration.digest:
        raise ValueError("Completed measurement has the wrong preregistration.")
    if artifact.manifest_digests != dict(manifest_digests):
        raise ValueError("Completed measurement has changed probe manifests.")
    if set(artifact.layer_shapes) != set(preregistration.layers):
        raise ValueError("Completed measurement has the wrong probed layers.")


def _validate_analysis_artifact(
    artifact: CheckpointMeasurements,
    *,
    preregistration: Observation0Preregistration,
    checkpoint_step: int,
    expected_microbatches: int,
    expected_classes: set[int],
) -> None:
    if artifact.checkpoint_step != checkpoint_step or set(
        int(value) for value in artifact.metadata["checkpoint_step"]
    ) != {checkpoint_step}:
        raise ValueError("Registry-listed measurement has the wrong checkpoint step.")
    if artifact.preregistration_sha256 != preregistration.digest:
        raise ValueError("Registry-listed measurement has changed preregistration.")
    if set(artifact.layer_shapes) != set(preregistration.layers):
        raise ValueError("Registry-listed measurement has changed probed layers.")
    if set(artifact.manifest_digests) != {"a", "a_source_1", "b", "b_source_1"}:
        raise ValueError("Registry-listed measurement has incomplete probe views.")
    if set(int(value) for value in artifact.metadata["class_id"]) != expected_classes:
        raise ValueError("Registry-listed measurement has incomplete classes.")
    if set(int(value) for value in artifact.metadata["stratum_id"]) != set(
        range(len(preregistration.time_strata))
    ):
        raise ValueError("Registry-listed measurement has incomplete timestep strata.")
    cell_counts = artifact.metadata.groupby(
        ["probe_view", "class_id", "stratum_id"]
    ).size()
    if cell_counts.empty or set(int(value) for value in cell_counts) != {
        expected_microbatches
    }:
        raise ValueError("Registry-listed measurement has the wrong probe phase row count.")


def _measurement_set_digest(
    artifacts: list[CheckpointMeasurements],
    *,
    phase: str,
) -> str:
    payload = {
        "phase": phase,
        "measurements": [
            {
                "checkpoint_step": artifact.checkpoint_step,
                "measurement_digest": artifact.digest,
            }
            for artifact in sorted(artifacts, key=lambda value: value.checkpoint_step)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
