"""Dataset variant builders for the unified geometry explorer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.geometry_explorer.registry import (
    DEFAULT_WORKSPACE,
    GeometryRegistry,
)
from fm_lab.geometry_explorer.registry import (
    variant_id as make_variant_id,
)
from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle, load_dataset
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.utils.config import ConfigError, load_config, save_config
from fm_lab.utils.logging import write_json


@dataclass(frozen=True)
class DatasetVariantConfig:
    family: str = "mnist"
    variant: str = "original"
    base: str = "original"
    split: str = "all"
    seed: int = 42
    input: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)

    @property
    def variant_id(self) -> str:
        return make_variant_id(self.family, self.variant)


def load_variant_config(path: str | Path) -> DatasetVariantConfig:
    raw = load_config(path)
    return variant_config_from_dict(raw)


def variant_config_from_dict(raw: dict[str, Any]) -> DatasetVariantConfig:
    values = dict(raw)
    family = str(values.get("family", "mnist")).lower()
    if family != "mnist":
        raise ConfigError("Only family: mnist variants are supported in this first pass.")
    return DatasetVariantConfig(
        family=family,
        variant=str(values.get("variant", "original")),
        base=str(values.get("base", "original")),
        split=str(values.get("split", values.get("input", {}).get("split", "all"))),
        seed=int(values.get("seed", 42)),
        input=dict(values.get("input", {})),
        selection=dict(values.get("selection", {})),
    )


def build_dataset_variant(
    config: DatasetVariantConfig,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and register one concrete dataset variant."""

    if config.family != "mnist":
        raise ConfigError("Only MNIST variants are supported.")
    registry = GeometryRegistry(workspace)
    output_dir = registry.workspace / "datasets" / config.family / config.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    input_config = _mnist_input_config(config)
    dataset = load_dataset(
        input_config,
        project_root=project_root,
        thumbnail_dir=output_dir / "assets" / "thumbnails",
    )
    if dataset.vectors is None:
        raise RuntimeError("MNIST variant build expected raw vectors.")
    selected_positions = _select_positions(dataset.metadata, config)
    metadata = dataset.metadata.iloc[selected_positions].reset_index(drop=True).copy()
    vectors = np.asarray(dataset.vectors[selected_positions], dtype=np.float32)
    labels = _labels_array(metadata)
    metadata["row_id"] = np.arange(len(metadata), dtype=int)
    metadata["variant_id"] = config.variant_id
    metadata["variant"] = config.variant
    metadata["base_variant"] = config.base

    dataset_path = write_parquet(metadata, output_dir / "dataset_index.parquet")
    data_path = output_dir / "data.npy"
    labels_path = output_dir / "labels.npy"
    np.save(data_path, vectors)
    np.save(labels_path, labels)
    save_config(_variant_raw(config), output_dir / "config_used.yaml")
    label_counts = _label_counts(metadata)
    manifest = {
        "variant_id": config.variant_id,
        "family": config.family,
        "variant": config.variant,
        "base": config.base,
        "split": config.split,
        "seed": config.seed,
        "rows": int(len(metadata)),
        "label_counts": label_counts,
        "image_shape": list(dataset.image_shape or (28, 28)),
        "value_range": list(dataset.value_range or (0.0, 1.0)),
        "dataset_path": str(dataset_path),
        "data_path": str(data_path),
        "labels_path": str(labels_path),
    }
    write_json(manifest, output_dir / "manifest.json")
    registry.register_dataset_variant(
        variant_id=config.variant_id,
        family=config.family,
        variant=config.variant,
        base=config.base,
        split=config.split,
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=config_path or output_dir / "config_used.yaml",
        row_count=len(metadata),
        label_counts=label_counts,
        image_shape=dataset.image_shape or (28, 28),
        value_range=dataset.value_range or (0.0, 1.0),
    )
    return {
        "variant_id": config.variant_id,
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "data_path": data_path,
        "labels_path": labels_path,
        "rows": len(metadata),
        "label_counts": label_counts,
    }


def load_variant_bundle(
    variant_id: str,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> DatasetBundle:
    """Load a registered dataset variant as a diagnostics DatasetBundle."""

    registry = GeometryRegistry(workspace)
    row = registry.get_dataset_variant(variant_id)
    metadata = read_parquet(registry.resolve(row["dataset_path"]))
    data_path = row["data_path"]
    vectors = np.load(registry.resolve(data_path)) if data_path else None
    image_shape = tuple(json.loads(row["image_shape_json"])) if row["image_shape_json"] else None
    value_range = tuple(json.loads(row["value_range_json"])) if row["value_range_json"] else None
    return DatasetBundle(
        metadata=metadata,
        vectors=np.asarray(vectors, dtype=np.float32) if vectors is not None else None,
        source_id=variant_id,
        source_description=f"Geometry dataset variant {variant_id}",
        total_rows=len(metadata),
        image_shape=image_shape,
        value_range=value_range,
    )


def _mnist_input_config(config: DatasetVariantConfig) -> InputConfig:
    values = dict(config.input)
    values.setdefault("type", "mnist")
    values.setdefault("dataset_root", "data/mnist")
    values.setdefault("split", config.split)
    values.setdefault("order", "mldata" if config.split == "all" else "source")
    values.setdefault("thumbnail_mode", "atlas")
    values.setdefault("download", False)
    values["max_samples"] = None
    return InputConfig(**values)


def _select_positions(
    metadata: pd.DataFrame,
    config: DatasetVariantConfig,
) -> np.ndarray:
    per_class_counts = config.selection.get("per_class_counts")
    if not per_class_counts:
        return np.arange(len(metadata), dtype=int)
    counts = {str(key): int(value) for key, value in per_class_counts.items()}
    rng = np.random.default_rng(config.seed)
    selected: list[int] = []
    labels = metadata["label"].astype(str).to_numpy()
    for label, count in sorted(counts.items(), key=lambda item: float(item[0])):
        candidates = np.flatnonzero(labels == label)
        if count < 0:
            raise ConfigError(f"Class count for label {label!r} must be non-negative.")
        if count > len(candidates):
            raise ConfigError(
                f"Requested {count} samples for label {label}, "
                f"but only {len(candidates)} are available."
            )
        shuffled = np.array(candidates, copy=True)
        rng.shuffle(shuffled)
        selected.extend(int(value) for value in shuffled[:count])
    return np.asarray(sorted(selected), dtype=int)


def _labels_array(metadata: pd.DataFrame) -> np.ndarray:
    if "label_id" in metadata:
        return metadata["label_id"].to_numpy()
    numeric = pd.to_numeric(metadata["label"], errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()
    return metadata["label"].astype(str).to_numpy()


def _label_counts(metadata: pd.DataFrame) -> dict[str, int]:
    counts = metadata["label"].astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _variant_raw(config: DatasetVariantConfig) -> dict[str, Any]:
    return {
        "family": config.family,
        "variant": config.variant,
        "base": config.base,
        "split": config.split,
        "seed": config.seed,
        "input": dict(config.input),
        "selection": dict(config.selection),
    }
