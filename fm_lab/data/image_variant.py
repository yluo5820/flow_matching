"""Image target distribution backed by a geometry-explorer dataset variant."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry


@dataclass
class ImageVariantImages:
    """Flattened image distribution loaded from a registered dataset variant."""

    variant_id: str
    workspace: str | Path = DEFAULT_WORKSPACE
    normalize: str = "zero_one"
    dequantize: bool = False
    name: str = "image_variant"
    dim: int = field(default=0, init=False)
    image_shape: tuple[int, ...] = field(default=(), init=False)
    _images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _registry_row: dict | None = field(default=None, init=False, repr=False)
    _raw_value_range: tuple[float, float] = field(
        default=(0.0, 1.0),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._load()

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        samples, _ = self.sample_with_labels(n, device=device)
        return samples

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1:
            raise ValueError("ImageVariantImages.sample requires n >= 1.")
        self._load()
        assert self._images is not None
        assert self._labels is not None
        indices = torch.randint(0, self._images.shape[0], (n,))
        samples = self._normalize(self._images[indices])
        labels = self._labels[indices]
        if device is not None:
            resolved_device = torch.device(device)
            samples = samples.to(resolved_device)
            labels = labels.to(resolved_device)
        return samples, labels

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        del x
        return None

    def metadata(self) -> dict:
        self._load()
        assert self._images is not None
        row = self._registry_row or {}
        family = str(row.get("family") or self.variant_id.split("/", 1)[0])
        return {
            "name": family,
            "variant_id": self.variant_id,
            "workspace": str(Path(self.workspace)),
            "dim": self.dim,
            "normalize": self.normalize,
            "dequantize": self.dequantize,
            "image_shape": list(self.image_shape),
            "image_value_range": list(_image_value_range(self.normalize)),
            "raw_image_value_range": list(self._raw_value_range),
            "n_images": int(self._images.shape[0]),
            "label_counts": json.loads(row.get("label_counts_json", "{}")),
        }

    def _load(self) -> None:
        if self._images is not None and self._labels is not None:
            return
        registry = GeometryRegistry(self.workspace)
        row = dict(registry.get_dataset_variant(self.variant_id))
        data_path = row["data_path"]
        labels_path = row["labels_path"]
        if not data_path or not labels_path:
            raise FileNotFoundError(
                f"Dataset variant {self.variant_id} does not have data/labels arrays."
            )
        images = np.asarray(np.load(registry.resolve(data_path)), dtype=np.float32)
        labels = np.asarray(np.load(registry.resolve(labels_path), allow_pickle=True))
        if images.ndim != 2:
            raise ValueError(
                f"Image variant data must have shape (n, dim), got {images.shape}."
            )
        if len(labels) != len(images):
            raise ValueError(
                f"Image variant label count {len(labels)} does not match images {len(images)}."
            )

        image_shape = (
            tuple(int(value) for value in json.loads(row["image_shape_json"]))
            if row["image_shape_json"]
            else (images.shape[1],)
        )
        expected_dim = int(np.prod(image_shape))
        if expected_dim != images.shape[1]:
            raise ValueError(
                f"Dataset variant {self.variant_id} image_shape={image_shape} "
                f"expects dim {expected_dim}, got {images.shape[1]}."
            )
        value_range = (
            tuple(float(value) for value in json.loads(row["value_range_json"]))
            if row["value_range_json"]
            else (float(np.nanmin(images)), float(np.nanmax(images)))
        )
        if value_range[1] <= value_range[0]:
            raise ValueError(
                f"Dataset variant {self.variant_id} has invalid value range {value_range}."
            )

        self.dim = int(images.shape[1])
        self.image_shape = image_shape
        self.name = str(row.get("family") or self.name)
        self._raw_value_range = value_range
        self._images = torch.from_numpy(images)
        self._labels = torch.from_numpy(_labels_to_int64(labels))
        self._registry_row = row

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        low, high = self._raw_value_range
        values = (images - low) / (high - low)
        if self.dequantize:
            values = torch.clamp(values + torch.rand_like(values) / 256.0, 0.0, 1.0)
        normalized = self.normalize.lower()
        if normalized in {"zero_one", "01", "unit"}:
            return values
        if normalized in {"minus_one_one", "-1_1", "centered"}:
            return 2.0 * values - 1.0
        raise ValueError(f"Unsupported image normalization: {self.normalize}")


def _labels_to_int64(labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(labels.dtype, np.number):
        return np.asarray(labels, dtype=np.int64)
    _, inverse = np.unique(labels.astype(str), return_inverse=True)
    return np.asarray(inverse, dtype=np.int64)


def _image_value_range(normalize: str) -> tuple[float, float]:
    normalized = normalize.lower()
    if normalized in {"zero_one", "01", "unit"}:
        return (0.0, 1.0)
    if normalized in {"minus_one_one", "-1_1", "centered"}:
        return (-1.0, 1.0)
    raise ValueError(f"Unsupported image normalization: {normalize}")
