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
from fm_lab.image_diagnostics.dataset_loader import (
    DatasetBundle,
    _export_array_sprite_atlases,
    load_dataset,
)
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
    object: dict[str, Any] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    pose: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    @property
    def variant_id(self) -> str:
        return make_variant_id(self.family, self.variant)


def load_variant_config(path: str | Path) -> DatasetVariantConfig:
    raw = load_config(path)
    return variant_config_from_dict(raw)


def variant_config_from_dict(raw: dict[str, Any]) -> DatasetVariantConfig:
    values = dict(raw)
    family = str(values.get("family", "mnist")).lower()
    if family not in _SUPPORTED_FAMILIES:
        supported = ", ".join(sorted(_SUPPORTED_FAMILIES))
        raise ConfigError(f"Unsupported dataset family {family!r}. Supported: {supported}.")
    return DatasetVariantConfig(
        family=family,
        variant=str(values.get("variant", "original")),
        base=str(values.get("base", "original")),
        split=str(values.get("split", values.get("input", {}).get("split", "all"))),
        seed=int(values.get("seed", 42)),
        input=dict(values.get("input", {})),
        selection=dict(values.get("selection", {})),
        object=dict(values.get("object", {})),
        render=dict(values.get("render", {})),
        pose=dict(values.get("pose", {})),
        output=dict(values.get("output", {})),
    )


def build_dataset_variant(
    config: DatasetVariantConfig,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and register one concrete dataset variant."""

    if config.family == "synthetic_object":
        from fm_lab.geometry_explorer.synthetic_objects import (
            build_synthetic_object_dataset,
            synthetic_object_config_from_dict,
        )

        return build_synthetic_object_dataset(
            synthetic_object_config_from_dict(_variant_raw(config)),
            workspace=workspace,
            config_path=config_path,
        )

    registry = GeometryRegistry(workspace)
    output_dir = registry.workspace / "datasets" / config.family / config.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    input_config = _input_config(config)
    dataset = load_dataset(
        input_config,
        project_root=project_root,
        thumbnail_dir=output_dir / "assets" / "thumbnails",
    )
    if dataset.vectors is None:
        raise RuntimeError("MNIST variant build expected raw vectors.")
    composition = _mnist_pair_composition_config(config)
    if composition is not None:
        return _build_mnist_pair_composition_variant(
            config,
            dataset=dataset,
            output_dir=output_dir,
            registry=registry,
            config_path=config_path,
            composition=composition,
        )
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
        "image_shape": list(dataset.image_shape or _default_image_shape(config.family)),
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
        image_shape=dataset.image_shape or _default_image_shape(config.family),
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


def _mnist_pair_composition_config(config: DatasetVariantConfig) -> dict[str, Any] | None:
    raw = config.selection.get("mnist_pair_composition")
    if raw is None:
        raw = config.selection.get("composition")
    if raw is None:
        return None
    if config.family != "mnist":
        raise ConfigError("MNIST pair composition is only supported for family: mnist.")
    if not isinstance(raw, dict):
        raise ConfigError("selection.mnist_pair_composition must be a mapping.")
    kind = str(raw.get("type", "paired_digits"))
    if kind != "paired_digits":
        raise ConfigError(
            "selection.mnist_pair_composition.type must be 'paired_digits'."
        )
    return dict(raw)


def _build_mnist_pair_composition_variant(
    config: DatasetVariantConfig,
    *,
    dataset: DatasetBundle,
    output_dir: Path,
    registry: GeometryRegistry,
    config_path: str | Path | None,
    composition: dict[str, Any],
) -> dict[str, Any]:
    vectors = np.asarray(dataset.vectors, dtype=np.float32)
    if dataset.image_shape != (28, 28):
        raise ConfigError(
            "MNIST pair composition expects 28x28 source images; "
            f"got {dataset.image_shape}."
        )

    left_digit = str(composition.get("left_digit", "1"))
    right_digit = str(composition.get("right_digit", "8"))
    labels = dataset.metadata["label"].astype(str).to_numpy()
    left_candidates = np.flatnonzero(labels == left_digit)
    right_candidates = np.flatnonzero(labels == right_digit)
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ConfigError(
            f"MNIST pair composition requires source digits {left_digit!r} and "
            f"{right_digit!r}; found {len(left_candidates)} and {len(right_candidates)}."
        )

    requested_pairs = composition.get(
        "pairs",
        composition.get("samples_per_class", min(len(left_candidates), len(right_candidates))),
    )
    pair_count = int(requested_pairs)
    if pair_count < 1:
        raise ConfigError("MNIST pair composition requires at least one pair.")
    replace = bool(composition.get("replace", False))
    if not replace and pair_count > min(len(left_candidates), len(right_candidates)):
        raise ConfigError(
            f"Requested {pair_count} MNIST digit pairs without replacement, but only "
            f"{min(len(left_candidates), len(right_candidates))} paired samples are available."
        )

    rng = np.random.default_rng(config.seed)
    left_positions = rng.choice(left_candidates, size=pair_count, replace=replace)
    right_positions = rng.choice(right_candidates, size=pair_count, replace=replace)
    left_vectors = _place_mnist_half(vectors[left_positions], side="left")
    right_vectors = _place_mnist_half(vectors[right_positions], side="right")
    composite_vectors = np.clip(left_vectors + right_vectors, 0.0, 1.0)

    left_label = str(composition.get("left_label", f"{left_digit}_left"))
    right_label = str(composition.get("right_label", f"{right_digit}_right"))
    composite_label = str(
        composition.get("composite_label", f"{left_digit}_plus_{right_digit}")
    )
    class_blocks = [
        (left_label, 0, "left", left_vectors, np.ones(pair_count), np.zeros(pair_count)),
        (right_label, 1, "right", right_vectors, np.zeros(pair_count), np.ones(pair_count)),
        (
            composite_label,
            2,
            "sum",
            composite_vectors,
            np.ones(pair_count),
            np.ones(pair_count),
        ),
    ]

    output_vectors = np.concatenate([block[3] for block in class_blocks], axis=0).astype(
        np.float32,
        copy=False,
    )
    records: list[dict[str, Any]] = []
    original_indices = dataset.metadata.get(
        "original_index",
        pd.Series(np.arange(len(dataset.metadata), dtype=int)),
    ).to_numpy()
    source_indices = dataset.metadata.get(
        "source_index",
        pd.Series(np.arange(len(dataset.metadata), dtype=int)),
    ).to_numpy()
    for label, label_id, role, _, left_weight, right_weight in class_blocks:
        for pair_id in range(pair_count):
            left_position = int(left_positions[pair_id])
            right_position = int(right_positions[pair_id])
            source_index = (
                source_indices[left_position]
                if role in {"left", "sum"}
                else source_indices[right_position]
            )
            records.append(
                {
                    "row_id": len(records),
                    "image_path": "",
                    "dataset": "mnist",
                    "split": config.split,
                    "label": label,
                    "label_id": label_id,
                    "family": label,
                    "prompt_id": f"mnist_pair_{role}",
                    "prompt": (
                        f"MNIST {left_digit} on the left + {right_digit} on the right"
                        if role == "sum"
                        else f"MNIST {left_digit if role == 'left' else right_digit} "
                        f"placed on the {role}"
                    ),
                    "tags": ["mnist", "paired_digits", role],
                    "source_index": int(source_index),
                    "left_source_index": int(source_indices[left_position]),
                    "right_source_index": int(source_indices[right_position]),
                    "left_original_index": int(original_indices[left_position]),
                    "right_original_index": int(original_indices[right_position]),
                    "pair_id": pair_id,
                    "component_role": role,
                    "left_digit": left_digit,
                    "right_digit": right_digit,
                    "left_weight": float(left_weight[pair_id]),
                    "right_weight": float(right_weight[pair_id]),
                    "sample_type": "mnist_pair_composition",
                    "status": "success",
                    "variant_id": config.variant_id,
                    "variant": config.variant,
                    "base_variant": config.base,
                }
            )

    metadata = pd.DataFrame.from_records(records)
    atlas_metadata = _export_array_sprite_atlases(
        output_vectors,
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
        output_dir=output_dir / "assets" / "atlases",
        prefix=f"{config.variant}_mnist_pair",
    )
    for key, value in atlas_metadata.items():
        metadata[key] = value

    dataset_path = write_parquet(metadata, output_dir / "dataset_index.parquet")
    data_path = output_dir / "data.npy"
    labels_path = output_dir / "labels.npy"
    np.save(data_path, output_vectors)
    np.save(labels_path, metadata["label_id"].to_numpy(dtype=np.int64))
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
        "image_shape": [28, 28],
        "value_range": [0.0, 1.0],
        "dataset_path": str(dataset_path),
        "data_path": str(data_path),
        "labels_path": str(labels_path),
        "composition": {
            "type": "paired_digits",
            "left_digit": left_digit,
            "right_digit": right_digit,
            "pairs": pair_count,
        },
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
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
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


def _place_mnist_half(vectors: np.ndarray, *, side: str) -> np.ndarray:
    from PIL import Image

    images = np.asarray(vectors, dtype=np.float32).reshape((-1, 28, 28))
    output = np.zeros_like(images)
    x0 = 0 if side == "left" else 14
    resample = Image.Resampling.LANCZOS
    for index, image in enumerate(images):
        pixels = np.asarray(np.rint(np.clip(image, 0.0, 1.0) * 255.0), dtype=np.uint8)
        resized = Image.fromarray(pixels, mode="L").resize((14, 28), resample=resample)
        output[index, :, x0 : x0 + 14] = np.asarray(resized, dtype=np.float32) / 255.0
    return output.reshape((len(images), 28 * 28))


_SUPPORTED_FAMILIES = {
    "mnist",
    "fashion_mnist",
    "cifar10",
    "cifar10_grayscale",
    "cifar100",
    "celeba",
    "cinic10",
    "cub200_segmentations",
    "imagenet32",
    "oxford_iiit_pet",
    "synthetic_object",
    "tiny_imagenet",
    "voc2012",
}


def _input_config(config: DatasetVariantConfig) -> InputConfig:
    values = dict(config.input)
    if config.family == "fashion_mnist":
        values.setdefault("type", "fashion_mnist")
        values.setdefault("dataset_root", "data/fashion_mnist")
    elif config.family in {"cifar10", "cifar10_grayscale"}:
        values.setdefault("type", "cifar10")
        values.setdefault("dataset_root", "data/cifar10")
        values.setdefault(
            "color_mode",
            "grayscale" if config.family == "cifar10_grayscale" else "rgb",
        )
    elif config.family == "cifar100":
        values.setdefault("type", "cifar100")
        values.setdefault("dataset_root", "data/cifar100")
        values.setdefault("color_mode", "rgb")
    elif config.family == "cinic10":
        values.setdefault("type", "cinic10")
        values.setdefault("dataset_root", "data/cinic10")
        values.setdefault("color_mode", "rgb")
    elif config.family == "celeba":
        values.setdefault("type", "celeba")
        values.setdefault("dataset_root", "data/celeba")
        values.setdefault("image_size", 64)
        values.setdefault("label_attribute", "Male")
    elif config.family in {"cub200_segmentations", "oxford_iiit_pet"}:
        values.setdefault("type", "numpy")
    elif config.family == "tiny_imagenet":
        values.setdefault("type", "tiny_imagenet")
        values.setdefault("dataset_root", "data/tiny_imagenet")
        values.setdefault("color_mode", "rgb")
    elif config.family == "imagenet32":
        values.setdefault("type", "imagenet32")
        values.setdefault("dataset_root", "data/imagenet32")
        values.setdefault("color_mode", "rgb")
    elif config.family == "voc2012":
        values.setdefault("type", "voc2012")
        values.setdefault("dataset_root", "data/VOCdevkit")
        values.setdefault("image_size", 64)
        values.setdefault("color_mode", "rgb")
    else:
        values.setdefault("type", "mnist")
        values.setdefault("dataset_root", "data/mnist")
    values.setdefault("split", config.split)
    values.setdefault("order", "mldata" if config.split == "all" else "source")
    values.setdefault("thumbnail_mode", "atlas")
    values.setdefault("download", False)
    values.setdefault("max_samples", None)
    return InputConfig(**values)


def _default_image_shape(family: str) -> tuple[int, ...]:
    if family in {
        "cifar10",
        "cifar100",
        "cinic10",
        "cub200_segmentations",
        "imagenet32",
        "oxford_iiit_pet",
    }:
        return (32, 32, 3)
    if family in {"celeba", "tiny_imagenet", "voc2012"}:
        return (64, 64, 3)
    if family == "cifar10_grayscale":
        return (32, 32)
    return (28, 28)


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
    labels = _selection_label_keys(metadata, counts)
    for label, count in sorted(counts.items(), key=lambda item: _label_sort_key(item[0])):
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


def _selection_label_keys(metadata: pd.DataFrame, counts: dict[str, int]) -> np.ndarray:
    labels = metadata["label"].astype(str).to_numpy()
    missing = set(counts) - set(labels)
    if not missing or "label_id" not in metadata:
        return labels
    label_ids = metadata["label_id"].astype(str).to_numpy()
    id_missing = set(counts) - set(label_ids)
    return label_ids if len(id_missing) < len(missing) else labels


def _label_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


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
        "object": dict(config.object),
        "render": dict(config.render),
        "pose": dict(config.pose),
        "output": dict(config.output),
    }
