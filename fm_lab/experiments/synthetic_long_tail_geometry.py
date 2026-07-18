"""Generate immutable training configurations for synthetic geometry conditions."""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any

from fm_lab.data import SyntheticLongTailImages
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    BOUNDED_ROTATION_CONDITION_ID,
)
from fm_lab.utils.config import load_config, save_config

_BALANCED_DATASET_SIZE = 15_000
_IMBALANCED_DATASET_SIZE = 5_550
_IMAGE_SHAPE = (3, 32, 32)
_CONDITION_ID_PATTERN = re.compile(
    rf"(?:g[0-2]_(?:balanced|f[0-2])|{re.escape(BOUNDED_ROTATION_CONDITION_ID)})\Z"
)
_CANONICAL_CONDITION_IDS = frozenset(
    f"g{geometry}_{frequency}"
    for geometry in range(3)
    for frequency in ("balanced", "f0", "f1", "f2")
)


class StageBlockedError(RuntimeError):
    """Raised when a preregistered scientific gate blocks a downstream stage."""


@dataclass(frozen=True)
class TrainingCommand:
    replicate: int
    condition_id: str
    config_path: Path
    run_dir: Path

    def argv(self, device: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "fm_lab.experiments.run_train",
            "--config",
            str(self.config_path),
            "--output-dir",
            str(self.run_dir),
            "--device",
            str(device),
        ]


class RunLedger:
    """Small atomic ledger whose terminal entries cannot be rewritten."""

    def __init__(self, path: str | Path) -> None:
        requested = Path(path).expanduser()
        self.path = requested.parent.resolve() / requested.name
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def entries(self) -> list[dict[str, Any]]:
        with self._lock():
            return list(self._read()["entries"])

    def is_complete(self, entry_id: str, *, config_hash: str | None = None) -> bool:
        entry = self._entry(entry_id)
        return bool(
            entry is not None
            and entry.get("status") == "complete"
            and (config_hash is None or entry.get("config_hash") == config_hash)
        )

    def start(self, entry_id: str, metadata: dict[str, Any] | None = None) -> None:
        now = _timestamp()
        entry = {
            "id": _entry_id(entry_id),
            "status": "running",
            "started_at": now,
            "ended_at": None,
            "failure": None,
            "artifacts": {},
            **dict(metadata or {}),
        }
        self._insert_or_transition(entry_id, entry, allow_running_transition=False)

    def complete(
        self,
        entry_id: str,
        artifacts: dict[str, Any],
        *,
        config_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _timestamp()
        entry = {
            "id": _entry_id(entry_id),
            "status": "complete",
            "started_at": now,
            "ended_at": now,
            "failure": None,
            "artifacts": dict(artifacts),
            **dict(metadata or {}),
        }
        if config_hash is not None:
            entry["config_hash"] = str(config_hash)
        self._insert_or_transition(entry_id, entry, allow_running_transition=True)

    def fail(
        self,
        entry_id: str,
        failure: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _timestamp()
        entry = {
            "id": _entry_id(entry_id),
            "status": "failed",
            "started_at": now,
            "ended_at": now,
            "failure": str(failure),
            "artifacts": {},
            **dict(metadata or {}),
        }
        self._insert_or_transition(entry_id, entry, allow_running_transition=True)

    def _entry(self, entry_id: str) -> dict[str, Any] | None:
        normalized = _entry_id(entry_id)
        with self._lock():
            return next(
                (entry for entry in self._read()["entries"] if entry.get("id") == normalized),
                None,
            )

    def _insert_or_transition(
        self,
        entry_id: str,
        replacement: dict[str, Any],
        *,
        allow_running_transition: bool,
    ) -> None:
        normalized = _entry_id(entry_id)
        with self._lock():
            payload = self._read()
            existing_index = next(
                (
                    index
                    for index, entry in enumerate(payload["entries"])
                    if entry.get("id") == normalized
                ),
                None,
            )
            if existing_index is not None:
                existing = payload["entries"][existing_index]
                if not allow_running_transition or existing.get("status") != "running":
                    raise FileExistsError(f"Ledger entry already exists: {normalized}")
                replacement["started_at"] = existing["started_at"]
                payload["entries"][existing_index] = replacement
            else:
                payload["entries"].append(replacement)
            self._write(payload)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "entries": []}
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError(f"Run ledger must be a regular file: {self.path}")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Run ledger is invalid: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Run ledger schema_version must be 1.")
        entries = payload.get("entries")
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("Run ledger entries must be a list of objects.")
        ids = [entry.get("id") for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Run ledger contains duplicate entry IDs.")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix=f".{self.path.name}-", dir=self.path.parent)
        temporary = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")

        class _LedgerLock:
            def __enter__(self_nonlocal):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                return handle

            def __exit__(self_nonlocal, exc_type, exc, traceback):
                del self_nonlocal, exc_type, exc, traceback
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

        return _LedgerLock()


def require_gate(path: str | Path, *, stage: str) -> dict[str, Any]:
    """Load a gate and require a literal JSON boolean ``passed: true``."""

    gate_path = Path(path).expanduser().resolve()
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBlockedError(f"{stage} gate is missing or invalid: {gate_path}") from exc
    if not isinstance(gate, dict):
        raise StageBlockedError(f"{stage} gate must be a JSON object.")
    if gate.get("passed") is not True:
        reasons = gate.get("reasons", gate.get("failure_reasons", []))
        detail = ", ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
        if gate.get("passed") not in {True, False} or not isinstance(gate.get("passed"), bool):
            detail = "passed must be literal true"
        raise StageBlockedError(f"{stage} gate failed{': ' + detail if detail else ''}")
    return gate


def build_matrix_commands(
    config_paths: dict[int, tuple[Path, ...]] | dict[int, list[Path]],
    *,
    run_root: Path,
) -> tuple[TrainingCommand, ...]:
    """Build one isolated training-process command for every matrix cell."""

    commands = []
    for replicate, paths in sorted(config_paths.items()):
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("replicate keys must be non-negative integers.")
        for path in sorted((Path(value) for value in paths), key=lambda value: value.name):
            condition_id = path.stem
            commands.append(
                TrainingCommand(
                    replicate=replicate,
                    condition_id=condition_id,
                    config_path=path,
                    run_dir=Path(run_root) / f"replicate_{replicate:02d}" / condition_id,
                )
            )
    identities = {(command.replicate, command.condition_id) for command in commands}
    if len(identities) != len(commands):
        raise ValueError("config_paths contains duplicate replicate-condition identities.")
    return tuple(commands)


def run_training_command(command: TrainingCommand, *, device: str) -> None:
    if command.run_dir.exists() or command.run_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite run directory: {command.run_dir}")
    if not command.config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {command.config_path}")
    subprocess.run(command.argv(device), check=True)


@dataclass(frozen=True)
class _ConditionPlan:
    replicate: int
    condition_id: str
    manifest_path: Path
    config_hash: str
    manifest_digest: str
    class_counts: tuple[int, int, int]


def matched_pass_step(total_steps: int, dataset_size: int, batch_size: int) -> int:
    """Return the positive update matching the balanced reference's example passes."""

    steps = _positive_int("total_steps", total_steps)
    examples = _positive_int("dataset_size", dataset_size)
    _positive_int("batch_size", batch_size)
    matched_step = int(math.floor(steps * examples / _BALANCED_DATASET_SIZE))
    if not 1 <= matched_step <= steps:
        raise ValueError("matched-pass checkpoint must fall within [1, total_steps].")
    return matched_step


def write_condition_training_configs(
    *,
    base_config_path: str | Path,
    condition_manifests: Iterable[str | Path],
    output_root: str | Path,
    run_root: str | Path,
    total_steps: int,
    batch_size: int,
    model_seed: int,
) -> tuple[Path, ...]:
    """Atomically write one immutable main-training config per canonical condition.

    Each `.sha256` sidecar hashes a sorted payload of the generated config and its
    canonicalized manifest JSON.  Only the external manifest-root prefix is removed
    from paths; run directory, condition ID, seed, manifest content, and its Task 3
    `config_hash` remain part of the scientific identity.
    """

    steps = _positive_int("total_steps", total_steps)
    requested_batch_size = _positive_int("batch_size", batch_size)
    seed = _nonnegative_int("model_seed", model_seed)
    base = load_config(base_config_path)
    frozen_batch_size = _positive_int("base training.batch_size", base["training"]["batch_size"])
    if requested_batch_size != frozen_batch_size:
        raise ValueError("batch_size must match the frozen base training configuration.")

    plans = _condition_plans(condition_manifests)
    output_path = Path(output_root)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Training config destination already exists: {output_path}")
    relative_paths = [
        Path(f"replicate_{plan.replicate:02d}") / f"{plan.condition_id}.yaml" for plan in plans
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".training-configs-", dir=output_path.parent))
    published = False
    try:
        for plan, relative_path in zip(plans, relative_paths, strict=True):
            config = _condition_config(
                base=base,
                plan=plan,
                run_root=run_root,
                total_steps=steps,
                model_seed=seed,
            )
            config_path = staging_root / relative_path
            save_config(config, config_path)
            config_path.with_suffix(".sha256").write_text(
                f"{_config_hash(config, plan.manifest_digest)}\n", encoding="utf-8"
            )
        os.symlink(
            os.path.relpath(staging_root, start=output_path.parent),
            output_path,
            target_is_directory=True,
        )
        published = True
    except BaseException:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return tuple(output_path / path for path in relative_paths)


def write_pilot_training_config(
    *,
    source_config_path: str | Path,
    output_root: str | Path,
    run_root: str | Path,
    pilot: dict[str, Any],
    require_balanced: bool = True,
    training_sampling_policy: str | None = None,
    condition_manifest_override: str | Path | None = None,
) -> Path:
    """Publish an immutable reduced-budget config derived from a matrix config."""

    source_path = Path(source_config_path).expanduser().resolve()
    source = load_config(source_path)
    source_plan = _read_condition_plan(source["data"]["condition_manifest"])
    if not isinstance(require_balanced, bool):
        raise ValueError("require_balanced must be a boolean.")
    if source_plan.replicate != 0 or (
        require_balanced and not source_plan.condition_id.endswith("_balanced")
    ):
        expected = "replicate-0 balanced" if require_balanced else "replicate-0"
        raise ValueError(f"Reduced-budget source must be a {expected} condition.")
    plan = (
        source_plan
        if condition_manifest_override is None
        else _read_condition_plan(condition_manifest_override)
    )
    if plan.replicate != source_plan.replicate:
        raise ValueError("Source and override condition manifests must share a replicate.")
    if plan.class_counts != source_plan.class_counts:
        raise ValueError("Source and override condition manifests must share class counts.")
    if training_sampling_policy not in {None, "empirical", "class_balanced"}:
        raise ValueError("training_sampling_policy must be None, 'empirical', or 'class_balanced'.")
    steps = _positive_int("pilot.training_steps", pilot.get("training_steps"))
    batch_size = _positive_int("pilot.batch_size", pilot.get("batch_size"))
    warmup_steps = _nonnegative_int("pilot.warmup_steps", pilot.get("warmup_steps"))
    log_every = _positive_int("pilot.log_every", pilot.get("log_every"))
    samples_per_class = _positive_int("pilot.samples_per_class", pilot.get("samples_per_class"))
    nfe = _positive_int("pilot.nfe", pilot.get("nfe"))
    sample_batch_size = _positive_int("pilot.sample_batch_size", pilot.get("sample_batch_size"))
    n_trajectories = _positive_int("pilot.n_trajectories", pilot.get("n_trajectories"))
    if warmup_steps > steps:
        raise ValueError("pilot.warmup_steps cannot exceed pilot.training_steps.")

    config = copy.deepcopy(source)
    config["experiment"]["name"] = f"{config['experiment']['name']}_{plan.condition_id}_pilot"
    config["experiment"]["output_dir"] = str(Path(run_root) / "replicate_00" / plan.condition_id)
    config["data"]["condition_manifest"] = str(plan.manifest_path)
    config["training"].update(
        {
            "steps": steps,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "log_every": log_every,
            "checkpoint_steps": [steps],
        }
    )
    if training_sampling_policy is not None:
        config["data"]["sampling_policy"] = training_sampling_policy
    config["sampling"].update(
        {
            "n_samples": samples_per_class * len(plan.class_counts),
            "n_trajectories": n_trajectories,
            "nfe": nfe,
            "sample_batch_size": sample_batch_size,
        }
    )

    destination = Path(output_root).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Pilot config destination already exists: {destination}")
    relative_path = Path("replicate_00") / f"{plan.condition_id}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".pilot-config-", dir=destination.parent))
    published = False
    try:
        config_path = staging_root / relative_path
        save_config(config, config_path)
        config_path.with_suffix(".sha256").write_text(
            f"{_config_hash(config, plan.manifest_digest)}\n", encoding="utf-8"
        )
        os.symlink(
            os.path.relpath(staging_root, start=destination.parent),
            destination,
            target_is_directory=True,
        )
        published = True
    except BaseException:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return destination / relative_path


def _condition_plans(manifests: Iterable[str | Path]) -> tuple[_ConditionPlan, ...]:
    plans = tuple(_read_condition_plan(path) for path in manifests)
    if len(plans) != len(_CANONICAL_CONDITION_IDS):
        raise ValueError("condition_manifests must contain exactly 12 canonical conditions.")
    condition_ids = {plan.condition_id for plan in plans}
    if condition_ids != _CANONICAL_CONDITION_IDS:
        raise ValueError("condition_manifests must contain each canonical condition exactly once.")
    replicates = {plan.replicate for plan in plans}
    if len(replicates) != 1:
        raise ValueError("condition_manifests must belong to exactly one replicate.")
    if len({plan.config_hash for plan in plans}) != 1:
        raise ValueError("condition_manifests must share one Task 3 config_hash.")

    _validate_count_rotations(plans)
    return tuple(sorted(plans, key=lambda plan: plan.condition_id))


def _read_condition_plan(raw_path: str | Path) -> _ConditionPlan:
    path = Path(raw_path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid synthetic condition manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Synthetic condition manifest must be a JSON object.")

    replicate = _nonnegative_int("manifest replicate", raw.get("replicate"))
    condition_id = raw.get("condition_id")
    if not isinstance(condition_id, str) or not _CONDITION_ID_PATTERN.fullmatch(condition_id):
        raise ValueError("Synthetic manifest condition_id must be a canonical Task 3 ID.")
    if path.stem != condition_id:
        raise ValueError("Synthetic manifest condition_id must match its filename stem.")
    expected_replicate_dir = f"replicate_{replicate:02d}"
    canonical_location = (
        path.parent.name == "conditions" and path.parent.parent.name == expected_replicate_dir
    )
    control_location = (
        condition_id == BOUNDED_ROTATION_CONDITION_ID
        and path.parent.name == "conditions"
        and path.parent.parent.name == "bounded_rotation_control"
        and path.parent.parent.parent.name == expected_replicate_dir
    )
    if not canonical_location and not control_location:
        raise ValueError(
            "Synthetic manifest must be in replicate_XX/conditions or the approved "
            "bounded-rotation control directory."
        )
    if tuple(raw.get("image_shape", ())) != _IMAGE_SHAPE:
        raise ValueError("Synthetic training manifests must have image_shape [3, 32, 32].")
    config_hash = raw.get("config_hash")
    if not isinstance(config_hash, str):
        raise ValueError("Synthetic manifest config_hash must be a string.")

    target = SyntheticLongTailImages(path)
    if target.image_shape != _IMAGE_SHAPE or target.dim != math.prod(_IMAGE_SHAPE):
        raise ValueError("Synthetic training manifest is not compatible with 3x32x32 training.")
    class_counts = tuple(int(count) for count in target.class_counts)
    if len(class_counts) != 3:
        raise ValueError("Synthetic training manifests must define exactly three classes.")
    return _ConditionPlan(
        replicate=replicate,
        condition_id=condition_id,
        manifest_path=path,
        config_hash=config_hash,
        manifest_digest=_manifest_digest(raw),
        class_counts=class_counts,
    )


def _validate_count_rotations(plans: tuple[_ConditionPlan, ...]) -> None:
    by_id = {plan.condition_id: plan for plan in plans}
    if len(by_id) != len(plans):
        raise ValueError("condition_manifests cannot contain duplicate condition IDs.")
    base_counts = by_id["g0_f0"].class_counts
    if not base_counts[0] > base_counts[1] > base_counts[2]:
        raise ValueError("g0_f0 base counts must be strictly descending head>medium>tail.")
    expected_rotations = (
        base_counts,
        (base_counts[1], base_counts[2], base_counts[0]),
        (base_counts[2], base_counts[0], base_counts[1]),
    )
    balanced_counts = (base_counts[0],) * 3
    for geometry in range(3):
        if by_id[f"g{geometry}_balanced"].class_counts != balanced_counts:
            raise ValueError("Balanced conditions must use the maximum base count in every class.")
        for frequency, expected_counts in enumerate(expected_rotations):
            if by_id[f"g{geometry}_f{frequency}"].class_counts != expected_counts:
                raise ValueError("Imbalanced conditions must use the canonical count rotations.")


def _condition_config(
    *,
    base: dict[str, Any],
    plan: _ConditionPlan,
    run_root: str | Path,
    total_steps: int,
    model_seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    dataset_size = (
        _BALANCED_DATASET_SIZE
        if plan.condition_id.endswith("_balanced")
        else _IMBALANCED_DATASET_SIZE
    )
    checkpoint_steps = sorted(
        {
            total_steps,
            matched_pass_step(total_steps, dataset_size, config["training"]["batch_size"]),
        }
    )
    replicate_dir = f"replicate_{plan.replicate:02d}"
    config["experiment"] = {
        "name": f"synthetic_long_tail_geometry_{replicate_dir}_{plan.condition_id}",
        "seed": model_seed,
        "output_dir": str(Path(run_root) / replicate_dir / plan.condition_id),
    }
    config["data"]["condition_manifest"] = str(plan.manifest_path)
    config["training"]["steps"] = total_steps
    config["training"]["checkpoint_steps"] = checkpoint_steps
    return config


def _manifest_digest(raw: dict[str, Any]) -> str:
    normalized = copy.deepcopy(raw)
    for entry in normalized.get("classes", []):
        if isinstance(entry, dict):
            for key in ("image_path", "factor_path"):
                if isinstance(entry.get(key), str):
                    entry[key] = _stable_path_token(entry[key])
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_hash(config: dict[str, Any], manifest_digest: str) -> str:
    normalized = copy.deepcopy(config)
    normalized["data"]["condition_manifest"] = _stable_path_token(
        normalized["data"]["condition_manifest"]
    )
    payload = json.dumps(
        {"config": normalized, "manifest_digest": manifest_digest},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_path_token(value: str | Path) -> str:
    parts = Path(value).parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"replicate_\d+", part):
            return Path(*parts[index:]).as_posix()
    return Path(value).name


def _positive_int(name: str, value: object) -> int:
    number = _nonnegative_int(name, value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-bool integer.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative.")
    return number


class SyntheticLongTailRunner:
    """Gated orchestration for the preregistered synthetic experiment."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_config(self.config_path)
        self.output_root = Path(self.config["output_root"]).expanduser().resolve()
        self.run_root = (
            Path(self.config.get("run_root", "runs/synthetic_long_tail_geometry"))
            .expanduser()
            .resolve()
        )
        self.ledger = RunLedger(self.output_root / "run_ledger.json")
        self.base_config_path = self.config_path.parent / "base_train.yaml"
        if not self.base_config_path.is_file():
            raise FileNotFoundError(f"Base training config does not exist: {self.base_config_path}")

    def plan(self) -> dict[str, Any]:
        commands = build_matrix_commands(self.config_paths(), run_root=self.run_root / "matrix")
        return {
            "schema_version": 1,
            "config": str(self.config_path),
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "replicates": int(self.config["replicates"]),
            "conditions_per_replicate": 12,
            "training_commands": [self._command_record(command, "auto") for command in commands],
        }

    def config_paths(self) -> dict[int, tuple[Path, ...]]:
        return {
            replicate: tuple(
                self._config_set_root(replicate)
                / f"replicate_{replicate:02d}"
                / f"{condition_id}.yaml"
                for condition_id in sorted(_CANONICAL_CONDITION_IDS)
            )
            for replicate in range(int(self.config["replicates"]))
        }

    def build_pools(self, replicate: int) -> dict[str, Any]:
        from fm_lab.geometry_explorer.synthetic_long_tail_design import (
            build_condition_manifests,
            build_master_pools,
        )

        replicate = _nonnegative_int("replicate", replicate)
        if replicate >= int(self.config["replicates"]):
            raise ValueError("replicate is outside the configured experiment range.")
        cells = build_master_pools(self.config, self.output_root, replicate)
        manifests = build_condition_manifests(
            self.output_root,
            replicate,
            cells,
            counts=tuple(int(value) for value in self.config["counts"]),
        )
        base = load_config(self.base_config_path)
        config_paths = write_condition_training_configs(
            base_config_path=self.base_config_path,
            condition_manifests=manifests,
            output_root=self._config_set_root(replicate),
            run_root=self.run_root / "matrix",
            total_steps=int(base["training"]["steps"]),
            batch_size=int(base["training"]["batch_size"]),
            model_seed=int(self.config["seed"]) + replicate,
        )
        result = {
            "replicate": replicate,
            "pool_cells": len(cells),
            "total_pool_images": sum(cell.count for cell in cells),
            "condition_manifests": [str(path) for path in manifests],
            "training_configs": [str(path) for path in config_paths],
        }
        self.ledger.complete(
            f"build-pools:replicate-{replicate:02d}",
            result,
            metadata={"stage": "build-pools", "replicate": replicate},
        )
        return result

    def calibrate_renderer(self) -> dict[str, Any]:
        from fm_lab.geometry_explorer.synthetic_long_tail_calibration import calibrate_renderer

        destination = self.output_root / "calibration" / "renderer"
        result = calibrate_renderer(self.config, destination)
        self.ledger.complete(
            "calibrate-renderer",
            {"gate": str(destination / "renderer_gate.json")},
            metadata={"stage": "calibrate-renderer"},
        )
        return result

    def train_oracle(self, *, device: str) -> dict[str, Any]:
        from fm_lab.geometry_explorer.synthetic_factor_oracle import train_factor_oracle
        from fm_lab.geometry_explorer.synthetic_long_tail_metrics import calibrate_metric_controls

        require_gate(
            self.output_root / "calibration" / "renderer" / "renderer_gate.json",
            stage="renderer_calibration",
        )
        oracle_dir = self.output_root / "calibration" / "oracle"
        oracle_gate = oracle_dir / "oracle_gate.json"
        if oracle_dir.exists() or oracle_dir.is_symlink():
            require_gate(oracle_gate, stage="factor_oracle")
            result = {
                "checkpoint_path": str(oracle_dir / "factor_oracle.pt"),
                "gate_path": str(oracle_gate),
                "metrics": json.loads(oracle_gate.read_text(encoding="utf-8")),
            }
        else:
            result = train_factor_oracle(self.config, oracle_dir, device)
        require_gate(oracle_gate, stage="factor_oracle")
        metric_dir = self.output_root / "calibration" / "metric_controls"
        metric = calibrate_metric_controls(
            oracle_checkpoint=oracle_dir / "factor_oracle.pt",
            oracle_gate=oracle_gate,
            output_dir=metric_dir,
            device=device,
            samples_per_class=int(self.config["evaluation"]["samples_per_class"]),
            seed=int(self.config["seed"]) + 7_000_000,
            source_revision=_source_revision(),
        )
        metric_passed = metric["control_ordering"]["passed"] is True
        metric_gate = {
            "passed": metric_passed,
            "reasons": [] if metric_passed else ["metric ordering"],
            "controls": str(metric_dir / "metric_controls.json"),
        }
        metric_gate_path = self.output_root / "calibration" / "metric_gate.json"
        _write_json_atomic(metric_gate, metric_gate_path)
        require_gate(metric_gate_path, stage="metric_controls")
        self.ledger.complete(
            "train-oracle",
            {
                "checkpoint": result["checkpoint_path"],
                "oracle_gate": result["gate_path"],
                "metric_gate": str(metric_gate_path),
            },
            metadata={"stage": "train-oracle"},
        )
        return {"oracle": result, "metric_controls": metric}

    def pilot(self, *, device: str, dry_run: bool = False) -> tuple[TrainingCommand, ...]:
        self._require_pretraining_gates(include_pilot=False, dry_run=dry_run)
        source_config_path = next(
            path for path in self.config_paths()[0] if path.stem == "g0_balanced"
        )
        pilot_config_root = self.output_root / "training_configs" / "pilot_set"
        config_path = pilot_config_root / "replicate_00" / "g0_balanced.yaml"
        if not dry_run and not config_path.is_file():
            config_path = write_pilot_training_config(
                source_config_path=source_config_path,
                output_root=pilot_config_root,
                run_root=self.run_root / "pilot",
                pilot=self.config["pilot"],
            )
        command = TrainingCommand(
            replicate=0,
            condition_id="g0_balanced",
            config_path=config_path,
            run_dir=self.run_root / "pilot" / "replicate_00" / "g0_balanced",
        )
        if not dry_run:
            pilot_entry = "pilot:replicate-00:g0_balanced"
            config_hash = _training_config_hash(config_path)
            if not self.ledger.is_complete(pilot_entry, config_hash=config_hash):
                self._run(command, device=device, stage="pilot", resume=False)
            from fm_lab.geometry_explorer.synthetic_long_tail_metrics import (
                evaluate_generated_distribution,
            )

            samples_per_class = int(self.config["pilot"]["samples_per_class"])
            evaluation_dir = command.run_dir / "evaluation"
            evaluation = evaluate_generated_distribution(
                generated_root=command.run_dir,
                oracle_checkpoint=self.output_root / "calibration" / "oracle" / "factor_oracle.pt",
                oracle_gate=self.output_root / "calibration" / "oracle" / "oracle_gate.json",
                output_dir=evaluation_dir,
                device=device,
                samples_per_class=samples_per_class,
                reference_samples_per_class=samples_per_class,
                seed=int(self.config["seed"]) + 8_000_000,
                generated_seed=int(load_config(config_path)["sampling"]["seed"]),
                source_revision=_source_revision(),
                clip_generated_to_value_range=True,
                condition_manifest=load_config(config_path)["data"]["condition_manifest"],
                inference_batch_size=min(256, samples_per_class),
            )
            pilot_gate = _balanced_pilot_gate(
                run_dir=command.run_dir,
                evaluation=evaluation,
                pilot=self.config["pilot"],
            )
            pilot_gate_path = self.output_root / "calibration" / "pilot_gate.json"
            _write_json_atomic(pilot_gate, pilot_gate_path)
            self.ledger.complete(
                "pilot-evaluation-condition-reference",
                {
                    "gate": str(pilot_gate_path),
                    "evaluation": str(evaluation_dir / "factor_metrics.json"),
                },
                metadata={"stage": "pilot-evaluation"},
            )
            require_gate(pilot_gate_path, stage="balanced_pilot")
        return (command,)

    def balanced_pilots(
        self,
        *,
        device: str,
        dry_run: bool = False,
        training_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run all balanced rotations at the pilot or an isolated learning-curve budget."""

        self._require_pretraining_gates(include_pilot=False, dry_run=dry_run)
        configured_steps = _positive_int(
            "pilot.training_steps", self.config["pilot"]["training_steps"]
        )
        steps = (
            configured_steps
            if training_steps is None
            else _positive_int("training_steps", training_steps)
        )
        pilot = copy.deepcopy(self.config["pilot"])
        pilot["training_steps"] = steps
        is_configured_pilot = steps == configured_steps
        budget_id = f"steps_{steps:08d}"
        commands = []
        evaluations: dict[str, dict[str, Any]] = {}
        for geometry in range(3):
            condition_id = f"g{geometry}_balanced"
            source_config_path = next(
                path for path in self.config_paths()[0] if path.stem == condition_id
            )
            if is_configured_pilot:
                pilot_config_root = (
                    self.output_root / "training_configs" / "pilot_set"
                    if geometry == 0
                    else self.output_root / "training_configs" / f"pilot_{condition_id}_set"
                )
            else:
                pilot_config_root = (
                    self.output_root
                    / "training_configs"
                    / "balanced_learning_curve"
                    / budget_id
                    / f"{condition_id}_set"
                )
            config_path = pilot_config_root / "replicate_00" / f"{condition_id}.yaml"
            legacy_config_path = pilot_config_root / "replicate_00" / "g0_balanced.yaml"
            if (
                is_configured_pilot
                and geometry > 0
                and not config_path.is_file()
                and legacy_config_path.is_file()
            ):
                config_path = legacy_config_path
            if is_configured_pilot:
                run_dir = (
                    self.run_root / "pilot" / "replicate_00" / condition_id
                    if geometry == 0
                    else self.run_root / "balanced_pilots" / "replicate_00" / condition_id
                )
            else:
                run_dir = (
                    self.run_root
                    / "balanced_learning_curve"
                    / budget_id
                    / "replicate_00"
                    / condition_id
                )
            if not dry_run and not config_path.is_file():
                config_path = write_pilot_training_config(
                    source_config_path=source_config_path,
                    output_root=pilot_config_root,
                    run_root=run_dir.parents[1],
                    pilot=pilot,
                )
            command = TrainingCommand(
                replicate=0,
                condition_id=condition_id,
                config_path=config_path,
                run_dir=run_dir,
            )
            commands.append(command)
            if dry_run:
                continue

            if is_configured_pilot:
                stage = "pilot" if geometry == 0 else "balanced-pilot"
            else:
                stage = f"balanced-learning-curve-{budget_id}"
            entry_id = f"{stage}:replicate-00:{condition_id}"
            config_hash = _training_config_hash(config_path)
            if not self.ledger.is_complete(entry_id, config_hash=config_hash):
                self._run(command, device=device, stage=stage, resume=False)
            evaluation_dir = run_dir / "evaluation"
            if not evaluation_dir.is_dir():
                from fm_lab.geometry_explorer.synthetic_long_tail_metrics import (
                    evaluate_generated_distribution,
                )

                pilot_config = load_config(config_path)
                count = int(pilot["samples_per_class"])
                evaluation = evaluate_generated_distribution(
                    generated_root=run_dir,
                    oracle_checkpoint=self.output_root
                    / "calibration"
                    / "oracle"
                    / "factor_oracle.pt",
                    oracle_gate=self.output_root / "calibration" / "oracle" / "oracle_gate.json",
                    output_dir=evaluation_dir,
                    device=device,
                    samples_per_class=count,
                    reference_samples_per_class=count,
                    seed=int(self.config["seed"]) + 8_000_000,
                    generated_seed=int(pilot_config["sampling"]["seed"]),
                    source_revision=_source_revision(),
                    clip_generated_to_value_range=True,
                    condition_manifest=pilot_config["data"]["condition_manifest"],
                    inference_batch_size=min(256, count),
                )
                self.ledger.complete(
                    (
                        f"balanced-pilot-evaluation-active-factors:{condition_id}"
                        if is_configured_pilot
                        else f"balanced-learning-curve-evaluation:{budget_id}:{condition_id}"
                    ),
                    {"evaluation": str(evaluation_dir / "factor_metrics.json")},
                    metadata={
                        "stage": "balanced-pilot-evaluation",
                        "condition": condition_id,
                    },
                )
            else:
                evaluation = json.loads(
                    (evaluation_dir / "factor_metrics.json").read_text(encoding="utf-8")
                )
            evaluations[condition_id] = evaluation

        if dry_run:
            return {
                "training_steps": steps,
                "commands": [self._command_record(command, device) for command in commands],
            }
        summary = _balanced_pilot_rotation_summary(evaluations)
        if is_configured_pilot:
            summary_path = self.output_root / "analysis" / "balanced_pilot_rotations.json"
            summary_entry = "balanced-pilot-rotations-active-factors"
            summary_stage = "balanced-pilot-rotations"
        else:
            summary["training_steps"] = steps
            summary_path = (
                self.output_root / "analysis" / "balanced_learning_curve" / f"{budget_id}.json"
            )
            summary_entry = f"balanced-learning-curve-rotations:{budget_id}"
            summary_stage = "balanced-learning-curve-rotations"
        if summary_path.is_file():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing_summary != summary:
                raise FileExistsError(
                    f"Refusing to replace a different balanced summary: {summary_path}"
                )
        else:
            _write_json_atomic(summary, summary_path)
        if not self.ledger.is_complete(summary_entry):
            self.ledger.complete(
                summary_entry,
                {"summary": str(summary_path)},
                metadata={"stage": summary_stage, "training_steps": steps},
            )
        return summary

    def frequency_pilots(
        self,
        *,
        device: str,
        training_steps: int,
        training_sampling_policy: str = "empirical",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run the nine replicate-0 dimension-by-frequency rotation conditions."""

        self._require_pretraining_gates(include_pilot=False, dry_run=dry_run)
        steps = _positive_int("training_steps", training_steps)
        if training_sampling_policy not in {"empirical", "class_balanced"}:
            raise ValueError("training_sampling_policy must be 'empirical' or 'class_balanced'.")
        budget_id = f"steps_{steps:08d}"
        experiment_id = (
            "frequency_factorial"
            if training_sampling_policy == "empirical"
            else "frequency_factorial_class_balanced"
        )
        ledger_stage = experiment_id.replace("_", "-")
        pilot = copy.deepcopy(self.config["pilot"])
        pilot["training_steps"] = steps
        condition_ids = tuple(
            f"g{geometry}_f{frequency}" for geometry in range(3) for frequency in range(3)
        )
        plans: dict[str, _ConditionPlan] = {}
        commands = []
        evaluations: dict[str, dict[str, Any]] = {}
        for condition_id in condition_ids:
            source_config_path = next(
                path for path in self.config_paths()[0] if path.stem == condition_id
            )
            if not dry_run:
                plans[condition_id] = _read_condition_plan(
                    load_config(source_config_path)["data"]["condition_manifest"]
                )
            config_root = (
                self.output_root
                / "training_configs"
                / experiment_id
                / budget_id
                / f"{condition_id}_set"
            )
            config_path = config_root / "replicate_00" / f"{condition_id}.yaml"
            run_dir = self.run_root / experiment_id / budget_id / "replicate_00" / condition_id
            if not dry_run and not config_path.is_file():
                config_path = write_pilot_training_config(
                    source_config_path=source_config_path,
                    output_root=config_root,
                    run_root=run_dir.parents[1],
                    pilot=pilot,
                    require_balanced=False,
                    training_sampling_policy=(
                        None
                        if training_sampling_policy == "empirical"
                        else training_sampling_policy
                    ),
                )
            command = TrainingCommand(
                replicate=0,
                condition_id=condition_id,
                config_path=config_path,
                run_dir=run_dir,
            )
            commands.append(command)
            if dry_run:
                continue

            stage = f"{ledger_stage}-{budget_id}"
            entry_id = f"{stage}:replicate-00:{condition_id}"
            config_hash = _training_config_hash(config_path)
            if not self.ledger.is_complete(entry_id, config_hash=config_hash):
                self._run(command, device=device, stage=stage, resume=False)
            evaluation_dir = run_dir / "evaluation"
            if not evaluation_dir.is_dir():
                from fm_lab.geometry_explorer.synthetic_long_tail_metrics import (
                    evaluate_generated_distribution,
                )

                reduced_config = load_config(config_path)
                count = int(pilot["samples_per_class"])
                evaluation = evaluate_generated_distribution(
                    generated_root=run_dir,
                    oracle_checkpoint=self.output_root
                    / "calibration"
                    / "oracle"
                    / "factor_oracle.pt",
                    oracle_gate=self.output_root / "calibration" / "oracle" / "oracle_gate.json",
                    output_dir=evaluation_dir,
                    device=device,
                    samples_per_class=count,
                    reference_samples_per_class=count,
                    seed=int(self.config["seed"]) + 8_000_000,
                    generated_seed=int(reduced_config["sampling"]["seed"]),
                    source_revision=_source_revision(),
                    clip_generated_to_value_range=True,
                    condition_manifest=reduced_config["data"]["condition_manifest"],
                    inference_batch_size=min(256, count),
                )
                self.ledger.complete(
                    f"{ledger_stage}-evaluation:{budget_id}:{condition_id}",
                    {"evaluation": str(evaluation_dir / "factor_metrics.json")},
                    metadata={
                        "stage": f"{ledger_stage}-evaluation",
                        "condition": condition_id,
                        "training_steps": steps,
                        "training_sampling_policy": training_sampling_policy,
                    },
                )
            else:
                evaluation = json.loads(
                    (evaluation_dir / "factor_metrics.json").read_text(encoding="utf-8")
                )
            evaluations[condition_id] = evaluation

        if dry_run:
            return {
                "training_steps": steps,
                "training_sampling_policy": training_sampling_policy,
                "commands": [self._command_record(command, device) for command in commands],
            }

        balanced_evaluations, balanced_plans = self._balanced_evaluations_for_budget(steps)
        evaluations.update(balanced_evaluations)
        plans.update(balanced_plans)
        summary = _frequency_factorial_summary(
            evaluations=evaluations,
            condition_counts={key: value.class_counts for key, value in plans.items()},
            training_steps=steps,
        )
        if training_sampling_policy != "empirical":
            summary["training_sampling_policy"] = training_sampling_policy
        summary_path = self.output_root / "analysis" / experiment_id / f"{budget_id}.json"
        summary_entry = f"{ledger_stage}-summary:{budget_id}"
        if summary_path.is_file():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing_summary != summary:
                raise FileExistsError(
                    f"Refusing to replace a different frequency summary: {summary_path}"
                )
        else:
            _write_json_atomic(summary, summary_path)
        if not self.ledger.is_complete(summary_entry):
            self.ledger.complete(
                summary_entry,
                {"summary": str(summary_path)},
                metadata={
                    "stage": f"{ledger_stage}-summary",
                    "training_steps": steps,
                    "training_sampling_policy": training_sampling_policy,
                },
            )
        return summary

    def bounded_rotation_control(
        self,
        *,
        device: str,
        training_steps: int = 2_000,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run one paired g0 control with the 5D class azimuth range restricted."""

        self._require_pretraining_gates(include_pilot=False, dry_run=dry_run)
        steps = _positive_int("training_steps", training_steps)
        budget_id = f"steps_{steps:08d}"
        experiment_id = "bounded_rotation_control"
        control_root = self.output_root / "replicate_00" / experiment_id
        manifest_path = control_root / "conditions" / f"{BOUNDED_ROTATION_CONDITION_ID}.json"
        control_spec_path = control_root / "control_spec.json"
        config_root = (
            self.output_root
            / "training_configs"
            / experiment_id
            / budget_id
            / f"{BOUNDED_ROTATION_CONDITION_ID}_set"
        )
        config_path = config_root / "replicate_00" / f"{BOUNDED_ROTATION_CONDITION_ID}.yaml"
        run_dir = (
            self.run_root
            / experiment_id
            / budget_id
            / "replicate_00"
            / BOUNDED_ROTATION_CONDITION_ID
        )
        command = TrainingCommand(
            replicate=0,
            condition_id=BOUNDED_ROTATION_CONDITION_ID,
            config_path=config_path,
            run_dir=run_dir,
        )
        baseline_run_dir = self._balanced_run_dir("g0_balanced", steps)
        baseline_metrics_path = baseline_run_dir / "evaluation" / "factor_metrics.json"
        if dry_run:
            return {
                "training_steps": steps,
                "intervention": (
                    "restrict class-0 azimuth; keep XYZ/elevation and classes 1/2 paired"
                ),
                "baseline_evaluation": str(baseline_metrics_path),
                "command": self._command_record(command, device),
            }

        if not manifest_path.is_file():
            from fm_lab.geometry_explorer.synthetic_long_tail_design import (
                build_bounded_rotation_control,
            )

            artifacts = build_bounded_rotation_control(
                self.config,
                self.output_root,
                replicate=0,
            )
            manifest_path = artifacts["manifest"]
            control_spec_path = artifacts["control_spec"]
        SyntheticLongTailImages(manifest_path)
        if not control_spec_path.is_file():
            raise FileNotFoundError(
                f"Bounded-rotation control spec is missing: {control_spec_path}"
            )

        source_config_path = next(
            path for path in self.config_paths()[0] if path.stem == "g0_balanced"
        )
        pilot = copy.deepcopy(self.config["pilot"])
        pilot["training_steps"] = steps
        if not config_path.is_file():
            config_path = write_pilot_training_config(
                source_config_path=source_config_path,
                output_root=config_root,
                run_root=run_dir.parents[1],
                pilot=pilot,
                condition_manifest_override=manifest_path,
            )
            command = TrainingCommand(
                replicate=0,
                condition_id=BOUNDED_ROTATION_CONDITION_ID,
                config_path=config_path,
                run_dir=run_dir,
            )

        stage = f"bounded-rotation-control-{budget_id}"
        entry_id = f"{stage}:replicate-00:{BOUNDED_ROTATION_CONDITION_ID}"
        config_hash = _training_config_hash(config_path)
        if not self.ledger.is_complete(entry_id, config_hash=config_hash):
            self._run(command, device=device, stage=stage, resume=False)

        evaluation_dir = run_dir / "evaluation"
        metrics_path = evaluation_dir / "factor_metrics.json"
        if not metrics_path.is_file():
            from fm_lab.geometry_explorer.synthetic_long_tail_metrics import (
                evaluate_generated_distribution,
            )

            reduced_config = load_config(config_path)
            count = int(pilot["samples_per_class"])
            evaluation = evaluate_generated_distribution(
                generated_root=run_dir,
                oracle_checkpoint=self.output_root / "calibration" / "oracle" / "factor_oracle.pt",
                oracle_gate=self.output_root / "calibration" / "oracle" / "oracle_gate.json",
                output_dir=evaluation_dir,
                device=device,
                samples_per_class=count,
                reference_samples_per_class=count,
                seed=int(self.config["seed"]) + 8_000_000,
                generated_seed=int(reduced_config["sampling"]["seed"]),
                source_revision=_source_revision(),
                clip_generated_to_value_range=True,
                condition_manifest=reduced_config["data"]["condition_manifest"],
                inference_batch_size=min(256, count),
            )
            self.ledger.complete(
                f"bounded-rotation-control-evaluation:{budget_id}",
                {"evaluation": str(metrics_path)},
                metadata={
                    "stage": "bounded-rotation-control-evaluation",
                    "training_steps": steps,
                },
            )
        else:
            evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))

        if not baseline_metrics_path.is_file():
            raise FileNotFoundError(
                "The paired g0_balanced evaluation must exist at the same training budget: "
                f"{baseline_metrics_path}"
            )
        baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        control_spec = json.loads(control_spec_path.read_text(encoding="utf-8"))
        summary = _bounded_rotation_summary(
            baseline=baseline,
            control=evaluation,
            control_spec=control_spec,
            training_steps=steps,
            baseline_metrics_path=baseline_metrics_path,
            control_metrics_path=metrics_path,
        )
        summary_path = self.output_root / "analysis" / experiment_id / f"{budget_id}.json"
        if summary_path.is_file():
            if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
                raise FileExistsError(
                    f"Refusing to replace a different bounded-rotation summary: {summary_path}"
                )
        else:
            _write_json_atomic(summary, summary_path)
        summary_entry = f"bounded-rotation-control-summary:{budget_id}"
        if not self.ledger.is_complete(summary_entry):
            self.ledger.complete(
                summary_entry,
                {"summary": str(summary_path)},
                metadata={
                    "stage": "bounded-rotation-control-summary",
                    "training_steps": steps,
                },
            )
        return summary

    def _balanced_run_dir(self, condition_id: str, training_steps: int) -> Path:
        configured_steps = int(self.config["pilot"]["training_steps"])
        if training_steps == configured_steps:
            if condition_id == "g0_balanced":
                return self.run_root / "pilot" / "replicate_00" / condition_id
            return self.run_root / "balanced_pilots" / "replicate_00" / condition_id
        return (
            self.run_root
            / "balanced_learning_curve"
            / f"steps_{training_steps:08d}"
            / "replicate_00"
            / condition_id
        )

    def _balanced_evaluations_for_budget(
        self, training_steps: int
    ) -> tuple[dict[str, dict[str, Any]], dict[str, _ConditionPlan]]:
        configured_steps = int(self.config["pilot"]["training_steps"])
        budget_id = f"steps_{training_steps:08d}"
        evaluations = {}
        plans = {}
        for geometry in range(3):
            condition_id = f"g{geometry}_balanced"
            if training_steps == configured_steps:
                run_dir = (
                    self.run_root / "pilot" / "replicate_00" / condition_id
                    if geometry == 0
                    else self.run_root / "balanced_pilots" / "replicate_00" / condition_id
                )
            else:
                run_dir = (
                    self.run_root
                    / "balanced_learning_curve"
                    / budget_id
                    / "replicate_00"
                    / condition_id
                )
            metrics_path = run_dir / "evaluation" / "factor_metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(
                    "Balanced controls must be completed at the same training budget: "
                    f"{metrics_path}"
                )
            source_config_path = next(
                path for path in self.config_paths()[0] if path.stem == condition_id
            )
            plans[condition_id] = _read_condition_plan(
                load_config(source_config_path)["data"]["condition_manifest"]
            )
            evaluations[condition_id] = json.loads(metrics_path.read_text(encoding="utf-8"))
        return evaluations, plans

    def smoke(
        self,
        *,
        condition_id: str,
        replicate: int,
        device: str,
        dry_run: bool = False,
    ) -> tuple[TrainingCommand, ...]:
        self._require_pretraining_gates(include_pilot=True, dry_run=dry_run)
        paths = self.config_paths()[replicate]
        try:
            config_path = next(path for path in paths if path.stem == condition_id)
        except StopIteration as exc:
            raise ValueError(f"Unknown condition_id: {condition_id}") from exc
        command = TrainingCommand(
            replicate=replicate,
            condition_id=condition_id,
            config_path=config_path,
            run_dir=self.run_root / "smoke" / f"replicate_{replicate:02d}" / condition_id,
        )
        if not dry_run:
            self._run(command, device=device, stage="smoke", resume=False)
        return (command,)

    def matrix(
        self,
        *,
        device: str,
        dry_run: bool = False,
        resume: bool = False,
    ) -> tuple[TrainingCommand, ...]:
        self._require_pretraining_gates(include_pilot=True, dry_run=dry_run)
        commands = build_matrix_commands(self.config_paths(), run_root=self.run_root / "matrix")
        if not dry_run:
            for command in commands:
                self._run(command, device=device, stage="matrix", resume=resume)
        return commands

    def aggregate(self, *, bootstrap_draws: int | None = None) -> dict[str, Any]:
        from fm_lab.geometry_explorer.synthetic_long_tail_report import aggregate_experiment

        draws = (
            int(self.config["evaluation"]["bootstrap_draws"])
            if bootstrap_draws is None
            else _positive_int("bootstrap_draws", bootstrap_draws)
        )
        result = aggregate_experiment(
            self.output_root,
            bootstrap_draws=draws,
            seed=int(self.config["seed"]),
        )
        self.ledger.complete(
            "aggregate",
            {"summary": str(self.output_root / "analysis" / "summary.json")},
            metadata={"stage": "aggregate"},
        )
        return result

    def report(self) -> dict[str, Any]:
        from fm_lab.geometry_explorer.synthetic_long_tail_report import (
            render_research_report,
        )

        summary_path = self.output_root / "analysis" / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {
                "effects": {},
                "calibration": _calibration_snapshot(self.output_root),
                "conditions": _condition_snapshot(self.output_root),
            }
        report_filename = str(
            self.config.get("report_filename", "synthetic_long_tail_geometry_report.md")
        )
        if Path(report_filename).name != report_filename:
            raise ValueError("report_filename must be a plain filename.")
        destination = self.config_path.parents[2] / "docs" / "research" / report_filename
        path = render_research_report(
            summary,
            {"schema_version": 1, "entries": self.ledger.entries()},
            destination,
        )
        return {"report": str(path)}

    def _run(
        self,
        command: TrainingCommand,
        *,
        device: str,
        stage: str,
        resume: bool,
    ) -> None:
        config_hash = _training_config_hash(command.config_path)
        entry_id = f"{stage}:replicate-{command.replicate:02d}:{command.condition_id}"
        if resume and self.ledger.is_complete(entry_id, config_hash=config_hash):
            return
        metadata = {
            "stage": stage,
            "condition": command.condition_id,
            "replicate": command.replicate,
            "config_hash": config_hash,
            "command": command.argv(device),
            "output_path": str(command.run_dir),
        }
        self.ledger.start(entry_id, metadata)
        try:
            run_training_command(command, device=device)
        except BaseException as exc:
            self.ledger.fail(entry_id, f"{type(exc).__name__}: {exc}", metadata=metadata)
            raise
        self.ledger.complete(
            entry_id,
            {"run_dir": str(command.run_dir)},
            config_hash=config_hash,
            metadata=metadata,
        )

    def _require_pretraining_gates(self, *, include_pilot: bool, dry_run: bool) -> None:
        if dry_run:
            return
        require_gate(
            self.output_root / "calibration" / "renderer" / "renderer_gate.json",
            stage="renderer_calibration",
        )
        require_gate(
            self.output_root / "calibration" / "oracle" / "oracle_gate.json",
            stage="factor_oracle",
        )
        require_gate(
            self.output_root / "calibration" / "metric_gate.json",
            stage="metric_controls",
        )
        if include_pilot:
            require_gate(
                self.output_root / "calibration" / "pilot_gate.json",
                stage="balanced_pilot",
            )

    def _config_set_root(self, replicate: int) -> Path:
        return self.output_root / "training_configs" / f"replicate_{replicate:02d}_set"

    @staticmethod
    def _command_record(command: TrainingCommand, device: str) -> dict[str, Any]:
        return {
            "replicate": command.replicate,
            "condition_id": command.condition_id,
            "config_path": str(command.config_path),
            "run_dir": str(command.run_dir),
            "argv": command.argv(device),
        }


def _training_config_hash(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Training config hash sidecar does not exist: {sidecar}")
    value = sidecar.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Training config hash sidecar is malformed: {sidecar}")
    return value


def _balanced_pilot_gate(
    *,
    run_dir: Path,
    evaluation: dict[str, Any],
    pilot: dict[str, Any],
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "diagnostics" / "training_history.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with history_path.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    losses = [float(row["loss"]) for row in history]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise ValueError("Pilot training history must contain finite losses.")
    window = min(3, len(losses))
    initial_loss = float(statistics.median(losses[:window]))
    final_loss = float(statistics.median(losses[-window:]))
    loss_ratio = final_loss / initial_loss if initial_loss > 0.0 else math.inf

    expected_steps = _positive_int("pilot.training_steps", pilot.get("training_steps"))
    max_loss_ratio = float(pilot["max_final_to_initial_loss_ratio"])
    max_leakage = float(pilot["max_class_leakage_rate"])
    max_off_renderer = float(pilot["max_off_renderer_rate"])
    min_joint_valid = float(pilot["min_joint_valid_rate"])
    thresholds = (max_loss_ratio, max_leakage, max_off_renderer, min_joint_valid)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Pilot gate thresholds must be finite values in [0, 1].")

    checks: dict[str, Any] = {
        "training_complete": int(metrics.get("trained_steps", -1)) == expected_steps,
        "loss_decreased": loss_ratio <= max_loss_ratio,
        "classes": {},
    }
    reasons = []
    if not checks["training_complete"]:
        reasons.append("training_incomplete")
    if not checks["loss_decreased"]:
        reasons.append("loss_did_not_decrease")
    for item in evaluation.get("classes", []):
        class_id = int(item["requested_class"])
        validity = item["validity"]
        class_checks = {
            "class_leakage": float(validity["class_leakage_rate"]) <= max_leakage,
            "off_renderer": float(validity["off_renderer_rate"]) <= max_off_renderer,
            "joint_valid": float(validity["joint_valid_rate"]) >= min_joint_valid,
        }
        checks["classes"][str(class_id)] = class_checks
        reasons.extend(
            f"class_{class_id}:{name}" for name, passed in class_checks.items() if not passed
        )
    if len(checks["classes"]) != 3:
        reasons.append("missing_class_evaluation")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "checks": checks,
        "loss": {
            "initial_window_median": initial_loss,
            "final_window_median": final_loss,
            "final_to_initial_ratio": loss_ratio,
            "history_points": len(losses),
        },
        "thresholds": {
            "training_steps": expected_steps,
            "max_final_to_initial_loss_ratio": max_loss_ratio,
            "max_class_leakage_rate": max_leakage,
            "max_off_renderer_rate": max_off_renderer,
            "min_joint_valid_rate": min_joint_valid,
        },
        "class_validity": {
            str(item["requested_class"]): item["validity"] for item in evaluation["classes"]
        },
        "artifacts": {
            "run_metrics": str(metrics_path),
            "training_history": str(history_path),
            "factor_metrics": str(run_dir / "evaluation" / "factor_metrics.json"),
        },
    }


def _balanced_pilot_rotation_summary(
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(evaluations) != {"g0_balanced", "g1_balanced", "g2_balanced"}:
        raise ValueError("Balanced pilot summary requires all three geometry rotations.")
    records = []
    for condition_id, evaluation in sorted(evaluations.items()):
        for item in evaluation["classes"]:
            all_metrics = item["all_requested"]["metrics"]
            records.append(
                {
                    "condition_id": condition_id,
                    "class_id": int(item["requested_class"]),
                    "object_id": item["object_id"],
                    "target_dimension_id": item["target_dimension_id"],
                    "true_dimension": int(item["true_dimension"]),
                    **{name: float(value) for name, value in item["validity"].items()},
                    "multivariate_energy_distance": float(
                        all_metrics["multivariate_energy_distance"]
                    ),
                    "active_multivariate_energy_distance": float(
                        all_metrics["active_factors"]["multivariate_energy_distance"]
                    ),
                    "oracle_feature_fid": float(all_metrics["oracle_feature_fid"]),
                }
            )

    def grouped(field: str) -> dict[str, dict[str, float]]:
        result = {}
        for key in sorted({str(record[field]) for record in records}):
            group = [record for record in records if str(record[field]) == key]
            result[key] = {
                metric: float(statistics.mean(float(item[metric]) for item in group))
                for metric in (
                    "class_leakage_rate",
                    "off_renderer_rate",
                    "joint_valid_rate",
                    "multivariate_energy_distance",
                    "active_multivariate_energy_distance",
                    "oracle_feature_fid",
                )
            }
        return result

    return {
        "schema_version": 1,
        "design": "three balanced frequency conditions with dimension rotated across objects",
        "records": records,
        "means_by_true_dimension": grouped("true_dimension"),
        "means_by_object": grouped("object_id"),
        "generated_value_clipping": {
            condition_id: evaluation["provenance"]["generated_value_clipping"]
            for condition_id, evaluation in sorted(evaluations.items())
        },
    }


def _bounded_rotation_summary(
    *,
    baseline: dict[str, Any],
    control: dict[str, Any],
    control_spec: dict[str, Any],
    training_steps: int,
    baseline_metrics_path: str | Path,
    control_metrics_path: str | Path,
) -> dict[str, Any]:
    """Build the paired class-level comparison for the azimuth-range intervention."""

    steps = _positive_int("training_steps", training_steps)

    def records(evaluation: dict[str, Any]) -> dict[int, dict[str, Any]]:
        result = {}
        for item in evaluation.get("classes", []):
            class_id = int(item["requested_class"])
            metrics = item["all_requested"]["metrics"]
            result[class_id] = {
                "class_id": class_id,
                "object_id": str(item["object_id"]),
                "target_dimension_id": str(item["target_dimension_id"]),
                "true_dimension": int(item["true_dimension"]),
                "class_leakage_rate": float(item["validity"]["class_leakage_rate"]),
                "off_renderer_rate": float(item["validity"]["off_renderer_rate"]),
                "joint_valid_rate": float(item["validity"]["joint_valid_rate"]),
                "active_multivariate_energy_distance": float(
                    metrics["active_factors"]["multivariate_energy_distance"]
                ),
                "oracle_feature_fid": float(metrics["oracle_feature_fid"]),
            }
        if set(result) != {0, 1, 2}:
            raise ValueError("Bounded-rotation comparison requires all three evaluated classes.")
        return result

    baseline_records = records(baseline)
    control_records = records(control)
    metric_names = (
        "class_leakage_rate",
        "off_renderer_rate",
        "joint_valid_rate",
        "active_multivariate_energy_distance",
        "oracle_feature_fid",
    )
    comparisons = []
    for class_id in range(3):
        baseline_record = baseline_records[class_id]
        control_record = control_records[class_id]
        if baseline_record["object_id"] != control_record["object_id"]:
            raise ValueError("Baseline and control class identities do not align.")
        comparisons.append(
            {
                "class_id": class_id,
                "object_id": baseline_record["object_id"],
                "role": "intervention" if class_id == 0 else "unchanged_pool_control",
                "baseline": baseline_record,
                "bounded_rotation": control_record,
                "bounded_minus_baseline": {
                    name: float(control_record[name] - baseline_record[name])
                    for name in metric_names
                },
            }
        )
    return {
        "schema_version": 1,
        "design": (
            "paired g0 balanced control: restrict class-0 azimuth extent while preserving "
            "five active factors, exact XYZ/elevation draws, and unchanged class-1/2 pools"
        ),
        "training_steps": steps,
        "primary_class_id": 0,
        "delta_convention": "bounded_rotation_minus_full_azimuth_baseline",
        "control_spec": control_spec,
        "comparisons": comparisons,
        "artifacts": {
            "baseline_metrics": str(Path(baseline_metrics_path).resolve()),
            "bounded_rotation_metrics": str(Path(control_metrics_path).resolve()),
        },
        "interpretation": {
            "primary_test": (
                "Whether the unusually large full-azimuth visual extent explains part of the "
                "5D class difficulty at fixed sample count and training exposure."
            ),
            "does_not_test": (
                "A reduction from five-dimensional to lower-dimensional support; the control "
                "manifold remains five-dimensional."
            ),
            "nuisance_check": (
                "Changes in classes 1 and 2 reveal model-wide optimization spillover even "
                "though their underlying pool files are unchanged."
            ),
        },
    }


def _frequency_factorial_summary(
    *,
    evaluations: dict[str, dict[str, Any]],
    condition_counts: dict[str, tuple[int, int, int]],
    training_steps: int,
) -> dict[str, Any]:
    expected = {
        f"g{geometry}_{frequency}"
        for geometry in range(3)
        for frequency in ("balanced", "f0", "f1", "f2")
    }
    if set(evaluations) != expected or set(condition_counts) != expected:
        raise ValueError("Frequency summary requires all 12 replicate-0 factorial conditions.")
    steps = _positive_int("training_steps", training_steps)
    metric_names = (
        "class_leakage_rate",
        "off_renderer_rate",
        "joint_valid_rate",
        "active_multivariate_energy_distance",
        "oracle_feature_fid",
    )
    records = []
    for condition_id, evaluation in sorted(evaluations.items()):
        counts = condition_counts[condition_id]
        if len(counts) != 3:
            raise ValueError(f"Condition {condition_id} must contain exactly three class counts.")
        is_balanced = condition_id.endswith("_balanced")
        if is_balanced:
            if len(set(counts)) != 1:
                raise ValueError("Balanced factorial controls must have equal class counts.")
            roles = {class_id: "balanced" for class_id in range(3)}
        else:
            ordered_counts = sorted(set(counts), reverse=True)
            if len(ordered_counts) != 3:
                raise ValueError("Imbalanced factorial conditions need three distinct counts.")
            role_by_count = dict(zip(ordered_counts, ("head", "medium", "tail"), strict=True))
            roles = {class_id: role_by_count[count] for class_id, count in enumerate(counts)}
        if len(evaluation.get("classes", ())) != 3:
            raise ValueError(f"Condition {condition_id} must contain three class evaluations.")
        for item in evaluation["classes"]:
            class_id = int(item["requested_class"])
            if class_id not in range(3):
                raise ValueError(f"Condition {condition_id} contains an invalid class ID.")
            all_metrics = item["all_requested"]["metrics"]
            records.append(
                {
                    "condition_id": condition_id,
                    "geometry_rotation": int(condition_id[1]),
                    "frequency_role": roles[class_id],
                    "class_id": class_id,
                    "object_id": item["object_id"],
                    "target_dimension_id": item["target_dimension_id"],
                    "true_dimension": int(item["true_dimension"]),
                    "count": int(counts[class_id]),
                    **{name: float(value) for name, value in item["validity"].items()},
                    "active_multivariate_energy_distance": float(
                        all_metrics["active_factors"]["multivariate_energy_distance"]
                    ),
                    "oracle_feature_fid": float(all_metrics["oracle_feature_fid"]),
                }
            )
    if len(records) != 36:
        raise ValueError("Frequency summary requires exactly 36 class-level records.")

    def mean_metrics(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            name: float(statistics.mean(float(item[name]) for item in group))
            for name in metric_names
        }

    roles = ("balanced", "head", "medium", "tail")
    means_by_dimension_and_role = {}
    for dimension in (1, 3, 5):
        means_by_dimension_and_role[str(dimension)] = {}
        for role in roles:
            group = [
                item
                for item in records
                if item["true_dimension"] == dimension and item["frequency_role"] == role
            ]
            if len(group) != 3:
                raise ValueError(
                    "Each dimension-by-frequency role must contain all three object rotations."
                )
            means_by_dimension_and_role[str(dimension)][role] = mean_metrics(group)

    paired_changes = []
    for object_id, dimension in sorted(
        {(str(item["object_id"]), int(item["true_dimension"])) for item in records}
    ):
        block = [
            item
            for item in records
            if item["object_id"] == object_id and item["true_dimension"] == dimension
        ]
        by_role = {str(item["frequency_role"]): item for item in block}
        if set(by_role) != set(roles):
            raise ValueError("Each object-dimension block must contain all four frequency roles.")
        baseline = by_role["balanced"]
        paired_changes.append(
            {
                "object_id": object_id,
                "true_dimension": dimension,
                "changes_from_balanced": {
                    role: {
                        name: float(by_role[role][name]) - float(baseline[name])
                        for name in metric_names
                    }
                    for role in roles[1:]
                },
            }
        )

    mean_changes = {}
    for dimension in (1, 3, 5):
        blocks = [item for item in paired_changes if item["true_dimension"] == dimension]
        mean_changes[str(dimension)] = {
            role: {
                name: float(
                    statistics.mean(item["changes_from_balanced"][role][name] for item in blocks)
                )
                for name in metric_names
            }
            for role in roles[1:]
        }

    return {
        "schema_version": 1,
        "design": (
            "replicate-0 3x3 object-counterbalanced intrinsic-dimension and class-frequency "
            "rotations, with balanced controls"
        ),
        "training_steps": steps,
        "records": records,
        "means_by_true_dimension_and_frequency_role": means_by_dimension_and_role,
        "paired_changes_from_balanced": paired_changes,
        "mean_changes_from_balanced_by_true_dimension": mean_changes,
    }


def _entry_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError("ledger entry_id must be a non-empty single-line string.")
    return value.strip()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _source_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _calibration_snapshot(root: Path) -> dict[str, Any]:
    result = {}
    for name, relative in {
        "renderer": "calibration/renderer/renderer_gate.json",
        "oracle": "calibration/oracle/oracle_gate.json",
        "metric": "calibration/metric_gate.json",
        "pilot": "calibration/pilot_gate.json",
    }.items():
        path = root / relative
        result[name] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"status": "not_run", "path": str(path)}
        )
    return result


def _condition_snapshot(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.glob("replicate_*/conditions/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            {
                "replicate": raw.get("replicate"),
                "condition_id": raw.get("condition_id"),
                "geometry_mapping": raw.get("geometry_mapping"),
                "frequency_mapping": raw.get("frequency_mapping"),
            }
        )
    return result
