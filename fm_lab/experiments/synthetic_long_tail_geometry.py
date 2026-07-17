"""Generate immutable training configurations for synthetic geometry conditions."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
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
from fm_lab.utils.config import load_config, save_config

_BALANCED_DATASET_SIZE = 15_000
_IMBALANCED_DATASET_SIZE = 5_550
_IMAGE_SHAPE = (3, 32, 32)
_CONDITION_ID_PATTERN = re.compile(r"g[0-2]_(?:balanced|f[0-2])\Z")
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
    if path.parent.name != "conditions" or path.parent.parent.name != expected_replicate_dir:
        raise ValueError("Synthetic manifest must be located in replicate_XX/conditions.")
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
        self.run_root = Path(
            self.config.get("run_root", "runs/synthetic_long_tail_geometry")
        ).expanduser().resolve()
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
        result = train_factor_oracle(self.config, oracle_dir, device)
        oracle_gate = oracle_dir / "oracle_gate.json"
        require_gate(oracle_gate, stage="factor_oracle")
        metric_dir = self.output_root / "calibration" / "metric_controls"
        metric = calibrate_metric_controls(
            config=self.config,
            oracle_checkpoint=oracle_dir / "factor_oracle.pt",
            oracle_gate=oracle_gate,
            output_dir=metric_dir,
            device=device,
            samples_per_class=int(self.config["evaluation"]["samples_per_class"]),
            seed=int(self.config["seed"]) + 7_000_000,
            source_revision=_source_revision(),
        )
        metric_passed = metric["ordering"]["passed"] is True
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
        config_path = next(path for path in self.config_paths()[0] if path.stem == "g0_balanced")
        command = TrainingCommand(
            replicate=0,
            condition_id="g0_balanced",
            config_path=config_path,
            run_dir=self.run_root / "pilot" / "replicate_00" / "g0_balanced",
        )
        if not dry_run:
            self._run(command, device=device, stage="pilot", resume=False)
        return (command,)

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
            self.config.get(
                "report_filename", "synthetic_long_tail_geometry_report.md"
            )
        )
        if Path(report_filename).name != report_filename:
            raise ValueError("report_filename must be a plain filename.")
        destination = (
            self.config_path.parents[2] / "docs" / "research" / report_filename
        )
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
