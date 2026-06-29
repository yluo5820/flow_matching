"""MNIST target distribution backed by a geometry-explorer dataset variant."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry


@dataclass
class MNISTVariantImages:
    """Flattened MNIST distribution loaded from a registered dataset variant."""

    variant_id: str
    workspace: str | Path = DEFAULT_WORKSPACE
    normalize: str = "zero_one"
    dequantize: bool = False
    name: str = "mnist"
    dim: int = 28 * 28
    image_shape: tuple[int, int] = (28, 28)
    _images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _registry_row: dict | None = field(default=None, init=False, repr=False)

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        samples, _ = self.sample_with_labels(n, device=device)
        return samples

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1:
            raise ValueError("MNISTVariantImages.sample requires n >= 1.")
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
        return {
            "name": self.name,
            "variant_id": self.variant_id,
            "workspace": str(Path(self.workspace)),
            "dim": self.dim,
            "normalize": self.normalize,
            "dequantize": self.dequantize,
            "image_shape": list(self.image_shape),
            "image_value_range": list(_image_value_range(self.normalize)),
            "n_images": int(self._images.shape[0]),
            "label_counts": json.loads(row.get("label_counts_json", "{}")),
        }

    def _load(self) -> None:
        if self._images is not None and self._labels is not None:
            return
        registry = GeometryRegistry(self.workspace)
        row = registry.get_dataset_variant(self.variant_id)
        data_path = row["data_path"]
        labels_path = row["labels_path"]
        if not data_path or not labels_path:
            raise FileNotFoundError(
                f"Dataset variant {self.variant_id} does not have data/labels arrays."
            )
        images = np.asarray(np.load(registry.resolve(data_path)), dtype=np.float32)
        labels = np.asarray(np.load(registry.resolve(labels_path)), dtype=np.int64)
        if images.ndim != 2 or images.shape[1] != self.dim:
            raise ValueError(
                f"MNIST variant data must have shape (n, {self.dim}), got {images.shape}."
            )
        if len(labels) != len(images):
            raise ValueError(
                f"MNIST variant label count {len(labels)} does not match images {len(images)}."
            )
        self._images = torch.from_numpy(images)
        self._labels = torch.from_numpy(labels)
        self._registry_row = dict(row)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        values = images
        if self.dequantize:
            values = torch.clamp(values + torch.rand_like(values) / 256.0, 0.0, 1.0)
        normalized = self.normalize.lower()
        if normalized in {"zero_one", "01", "unit"}:
            return values
        if normalized in {"minus_one_one", "-1_1", "centered"}:
            return 2.0 * values - 1.0
        raise ValueError(f"Unsupported MNIST normalization: {self.normalize}")


def _image_value_range(normalize: str) -> tuple[float, float]:
    normalized = normalize.lower()
    if normalized in {"zero_one", "01", "unit"}:
        return (0.0, 1.0)
    if normalized in {"minus_one_one", "-1_1", "centered"}:
        return (-1.0, 1.0)
    raise ValueError(f"Unsupported MNIST normalization: {normalize}")
