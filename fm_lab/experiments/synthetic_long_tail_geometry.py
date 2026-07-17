"""Generate immutable training configurations for synthetic geometry conditions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fm_lab.utils.config import load_config, save_config

_BALANCED_DATASET_SIZE = 15_000
_IMBALANCED_DATASET_SIZE = 5_550


def matched_pass_step(total_steps: int, dataset_size: int, batch_size: int) -> int:
    """Return the update matching the balanced reference's example passes."""

    del batch_size
    return int(math.floor(int(total_steps) * int(dataset_size) / _BALANCED_DATASET_SIZE))


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
    """Write one main-training config and deterministic hash per condition.

    The generated matrix is immutable: an existing destination is never modified.
    Condition IDs encode the fixed study sample counts, so tiny test manifests still
    exercise the production equal-example-pass checkpoint schedule.
    """

    base = load_config(base_config_path)
    output_path = Path(output_root)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Training config destination already exists: {output_path}")
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive.")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if int(batch_size) != int(base["training"]["batch_size"]):
        raise ValueError("batch_size must match the frozen base training configuration.")

    plans = _condition_plans(condition_manifests)
    if not plans:
        raise ValueError("condition_manifests must not be empty.")
    relative_paths = [
        Path(replicate) / f"{condition_id}.yaml"
        for replicate, condition_id, _ in plans
    ]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("condition_manifests must have unique replicate and condition IDs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".training-configs-", dir=output_path.parent))
    try:
        for (replicate, condition_id, manifest_path), relative_path in zip(
            plans, relative_paths, strict=True
        ):
            config = _condition_config(
                base=base,
                manifest_path=manifest_path,
                replicate=replicate,
                condition_id=condition_id,
                run_root=run_root,
                total_steps=int(total_steps),
                model_seed=int(model_seed),
            )
            config_path = staging_root / relative_path
            save_config(config, config_path)
            config_path.with_suffix(".sha256").write_text(
                f"{_config_hash(config)}\n", encoding="utf-8"
            )
        staging_root.replace(output_path)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return tuple(output_path / path for path in relative_paths)


def _condition_plans(
    manifests: Iterable[str | Path],
) -> tuple[tuple[str, str, Path], ...]:
    plans = []
    for raw_path in manifests:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        condition_id = str(manifest["condition_id"])
        replicate = f"replicate_{int(manifest['replicate']):02d}"
        plans.append((replicate, condition_id, path))
    return tuple(sorted(plans, key=lambda plan: (plan[0], plan[1])))


def _condition_config(
    *,
    base: dict[str, Any],
    manifest_path: Path,
    replicate: str,
    condition_id: str,
    run_root: str | Path,
    total_steps: int,
    model_seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    dataset_size = (
        _BALANCED_DATASET_SIZE if condition_id.endswith("_balanced") else _IMBALANCED_DATASET_SIZE
    )
    checkpoint_steps = sorted({
        total_steps,
        matched_pass_step(total_steps, dataset_size, config["training"]["batch_size"]),
    })
    run_directory = _relative_run_directory(run_root, replicate, condition_id)
    config["experiment"] = {
        "name": f"synthetic_long_tail_geometry_{replicate}_{condition_id}",
        "seed": model_seed,
        "output_dir": run_directory,
    }
    config["data"]["condition_manifest"] = str(manifest_path)
    config["training"]["steps"] = total_steps
    config["training"]["checkpoint_steps"] = checkpoint_steps
    return config


def _relative_run_directory(run_root: str | Path, replicate: str, condition_id: str) -> str:
    root = Path(run_root)
    if root.is_absolute():
        root = Path("runs")
    return str(root / replicate / condition_id)


def _config_hash(config: dict[str, Any]) -> str:
    normalized = copy.deepcopy(config)
    normalized["data"]["condition_manifest"] = _stable_path_token(
        normalized["data"]["condition_manifest"]
    )
    normalized["experiment"]["output_dir"] = _stable_path_token(
        normalized["experiment"]["output_dir"]
    )
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_path_token(value: str | Path) -> str:
    parts = Path(value).parts
    for index, part in enumerate(parts):
        if part.startswith("replicate_"):
            return Path(*parts[index:]).as_posix()
    return Path(value).name
