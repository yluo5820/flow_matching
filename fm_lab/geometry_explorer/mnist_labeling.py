"""Classifier-based labeling for generated grayscale geometry datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.mnist_eval import (
    load_or_train_fashion_mnist_classifier,
    load_or_train_mnist_classifier,
    predict_mnist_classifier,
)
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.image_diagnostics.dataset_loader import FASHION_MNIST_LABELS
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.utils.config import ConfigError
from fm_lab.utils.logging import write_json


def label_mnist_dataset_variant(
    *,
    variant_id: str,
    workspace: str | Path = DEFAULT_WORKSPACE,
    data_root: str | Path = "data/mnist",
    normalize: str = "auto",
    classifier_checkpoint: str | Path = "artifacts/mnist_classifier.pt",
    classifier_steps: int = 1000,
    classifier_batch_size: int = 256,
    classifier_eval_samples: int = 2048,
    classifier_lr: float = 1.0e-3,
    low_confidence_threshold: float = 0.6,
    replace_label: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Annotate a registered MNIST variant with classifier labels and confidence."""

    device = torch.device("cpu") if device is None else device
    registry = GeometryRegistry(workspace)
    row = registry.get_dataset_variant(variant_id)
    if str(row["family"]).lower() != "mnist":
        raise ConfigError(f"MNIST classifier labeling requires an MNIST variant, got {variant_id}.")
    if not row["data_path"]:
        raise ConfigError(f"Dataset variant has no data.npy path: {variant_id}.")

    dataset_path = registry.resolve(row["dataset_path"])
    data_path = registry.resolve(row["data_path"])
    labels_path = registry.resolve(row["labels_path"]) if row["labels_path"] else None
    samples = np.asarray(np.load(data_path), dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 28 * 28:
        raise ConfigError(
            f"MNIST classifier labeling expects flattened 28x28 images, got {samples.shape}."
        )
    metadata = read_parquet(dataset_path)
    if len(metadata) != len(samples):
        raise ConfigError(
            f"Metadata/sample row mismatch for {variant_id}: {len(metadata)} vs {len(samples)}."
        )

    value_range = _value_range(row)
    resolved_normalize = _resolve_normalize(normalize, value_range)
    classifier, classifier_info = load_or_train_mnist_classifier(
        data_root=data_root,
        normalize=resolved_normalize,
        checkpoint_path=Path(classifier_checkpoint),
        steps=classifier_steps,
        batch_size=classifier_batch_size,
        eval_samples=classifier_eval_samples,
        lr=classifier_lr,
        device=device,
    )
    low, high = value_range
    clipped = torch.as_tensor(samples, dtype=torch.float32).clamp(float(low), float(high))
    predictions = predict_mnist_classifier(
        classifier=classifier,
        samples=clipped,
        batch_size=classifier_batch_size,
        device=device,
    )

    predicted = predictions["predicted"]
    confidence = predictions["confidence"]
    digit_labels = np.asarray([str(value) for value in predicted])
    label_counts = {
        str(label): int((digit_labels == str(label)).sum()) for label in range(10)
    }
    label_counts = {label: count for label, count in label_counts.items() if count > 0}

    output = metadata.copy()
    if "generated_group" not in output.columns and "label" in output.columns:
        output["generated_group"] = output["label"].astype(str)
    output["classifier_digit"] = predicted.astype(np.int64)
    output["classifier_label"] = digit_labels
    output["classifier_confidence"] = confidence
    output["classifier_margin"] = predictions["margin"]
    output["classifier_entropy"] = predictions["entropy"]
    output["classifier_low_confidence"] = confidence < float(low_confidence_threshold)
    output["classifier_outlier_label"] = np.where(
        output["classifier_low_confidence"], "low_confidence", "recognized"
    )
    output["classifier_label_source"] = "mnist_classifier"
    if replace_label:
        output["label"] = output["classifier_label"]
        output["family"] = output["classifier_label"]
        output["label_source"] = "mnist_classifier"

    write_parquet(output, dataset_path)
    if labels_path is not None:
        np.save(labels_path, predicted.astype(np.int64))
    _update_manifest(
        dataset_path.parent / "manifest.json",
        label_counts=label_counts,
        classifier_info=classifier_info,
        low_confidence_threshold=low_confidence_threshold,
        replace_label=replace_label,
    )
    registry.register_dataset_variant(
        variant_id=variant_id,
        family=row["family"],
        variant=row["variant"],
        base=row["base"],
        split=row["split"],
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=registry.resolve(row["config_path"]) if row["config_path"] else None,
        row_count=len(output),
        label_counts=label_counts if replace_label else _label_counts(output),
        image_shape=json.loads(row["image_shape_json"]) if row["image_shape_json"] else None,
        value_range=value_range,
    )
    low_confidence_count = int(output["classifier_low_confidence"].sum())
    return {
        "variant_id": variant_id,
        "dataset_path": str(dataset_path),
        "labels_path": str(labels_path) if labels_path is not None else None,
        "rows": int(len(output)),
        "label_counts": label_counts,
        "classifier": classifier_info,
        "low_confidence_threshold": float(low_confidence_threshold),
        "low_confidence_count": low_confidence_count,
        "low_confidence_fraction": low_confidence_count / max(len(output), 1),
        "replace_label": replace_label,
    }


def label_fashion_mnist_dataset_variant(
    *,
    variant_id: str,
    workspace: str | Path = DEFAULT_WORKSPACE,
    data_root: str | Path = "data/fashion_mnist",
    normalize: str = "auto",
    classifier_checkpoint: str | Path = "artifacts/fashion_mnist_classifier.pt",
    classifier_steps: int = 1000,
    classifier_batch_size: int = 256,
    classifier_eval_samples: int = 2048,
    classifier_lr: float = 1.0e-3,
    low_confidence_threshold: float = 0.6,
    replace_label: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Annotate a registered Fashion-MNIST variant with classifier labels."""

    device = torch.device("cpu") if device is None else device
    registry = GeometryRegistry(workspace)
    row = registry.get_dataset_variant(variant_id)
    if str(row["family"]).lower() != "fashion_mnist":
        raise ConfigError(
            f"Fashion-MNIST classifier labeling requires a Fashion-MNIST variant, "
            f"got {variant_id}."
        )
    if not row["data_path"]:
        raise ConfigError(f"Dataset variant has no data.npy path: {variant_id}.")

    dataset_path = registry.resolve(row["dataset_path"])
    data_path = registry.resolve(row["data_path"])
    labels_path = registry.resolve(row["labels_path"]) if row["labels_path"] else None
    samples = np.asarray(np.load(data_path), dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 28 * 28:
        raise ConfigError(
            "Fashion-MNIST classifier labeling expects flattened 28x28 images, "
            f"got {samples.shape}."
        )
    metadata = read_parquet(dataset_path)
    if len(metadata) != len(samples):
        raise ConfigError(
            f"Metadata/sample row mismatch for {variant_id}: {len(metadata)} vs {len(samples)}."
        )

    value_range = _value_range(row)
    resolved_normalize = _resolve_normalize(normalize, value_range)
    classifier, classifier_info = load_or_train_fashion_mnist_classifier(
        data_root=data_root,
        normalize=resolved_normalize,
        checkpoint_path=Path(classifier_checkpoint),
        steps=classifier_steps,
        batch_size=classifier_batch_size,
        eval_samples=classifier_eval_samples,
        lr=classifier_lr,
        device=device,
    )
    low, high = value_range
    clipped = torch.as_tensor(samples, dtype=torch.float32).clamp(float(low), float(high))
    predictions = predict_mnist_classifier(
        classifier=classifier,
        samples=clipped,
        batch_size=classifier_batch_size,
        device=device,
    )

    predicted = predictions["predicted"]
    confidence = predictions["confidence"]
    class_labels = np.asarray([FASHION_MNIST_LABELS[int(value)] for value in predicted])
    label_counts = {
        label: int((class_labels == label).sum()) for label in FASHION_MNIST_LABELS
    }
    label_counts = {label: count for label, count in label_counts.items() if count > 0}

    output = metadata.copy()
    if "generated_group" not in output.columns and "label" in output.columns:
        output["generated_group"] = output["label"].astype(str)
    output["classifier_class_id"] = predicted.astype(np.int64)
    output["classifier_label"] = class_labels
    output["classifier_confidence"] = confidence
    output["classifier_margin"] = predictions["margin"]
    output["classifier_entropy"] = predictions["entropy"]
    output["classifier_low_confidence"] = confidence < float(low_confidence_threshold)
    output["classifier_outlier_label"] = np.where(
        output["classifier_low_confidence"], "low_confidence", "recognized"
    )
    output["classifier_label_source"] = "fashion_mnist_classifier"
    if replace_label:
        output["label"] = output["classifier_label"]
        output["label_id"] = output["classifier_class_id"]
        output["family"] = output["classifier_label"]
        output["label_source"] = "fashion_mnist_classifier"

    write_parquet(output, dataset_path)
    if labels_path is not None:
        np.save(labels_path, predicted.astype(np.int64))
    _update_manifest(
        dataset_path.parent / "manifest.json",
        label_counts=label_counts,
        classifier_info=classifier_info,
        low_confidence_threshold=low_confidence_threshold,
        replace_label=replace_label,
        key="fashion_mnist_classifier_labeling",
    )
    registry.register_dataset_variant(
        variant_id=variant_id,
        family=row["family"],
        variant=row["variant"],
        base=row["base"],
        split=row["split"],
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=registry.resolve(row["config_path"]) if row["config_path"] else None,
        row_count=len(output),
        label_counts=label_counts if replace_label else _label_counts(output),
        image_shape=json.loads(row["image_shape_json"]) if row["image_shape_json"] else None,
        value_range=value_range,
    )
    low_confidence_count = int(output["classifier_low_confidence"].sum())
    return {
        "variant_id": variant_id,
        "dataset_path": str(dataset_path),
        "labels_path": str(labels_path) if labels_path is not None else None,
        "rows": int(len(output)),
        "label_counts": label_counts,
        "classifier": classifier_info,
        "low_confidence_threshold": float(low_confidence_threshold),
        "low_confidence_count": low_confidence_count,
        "low_confidence_fraction": low_confidence_count / max(len(output), 1),
        "replace_label": replace_label,
    }


def _value_range(row: Any) -> tuple[float, float]:
    if row["value_range_json"]:
        values = json.loads(row["value_range_json"])
        if len(values) == 2:
            return float(values[0]), float(values[1])
    return -1.0, 1.0


def _resolve_normalize(value: str, value_range: tuple[float, float]) -> str:
    if value != "auto":
        return value
    low, high = value_range
    if low < 0 <= high:
        return "minus_one_one"
    return "zero_one"


def _label_counts(frame: Any) -> dict[str, int]:
    counts = frame["label"].astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _update_manifest(
    path: Path,
    *,
    label_counts: dict[str, int],
    classifier_info: dict[str, Any],
    low_confidence_threshold: float,
    replace_label: bool,
    key: str = "mnist_classifier_labeling",
) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    if replace_label:
        payload["label_counts"] = label_counts
    payload[key] = {
        **classifier_info,
        "low_confidence_threshold": float(low_confidence_threshold),
        "replace_label": replace_label,
    }
    write_json(payload, path)
