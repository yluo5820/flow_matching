"""Memory-mapped indexed targets for synthetic long-tail geometry data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.geometry_explorer.synthetic_long_tail_design import ConditionManifest


@dataclass
class SyntheticLongTailImages:
    """Sample condition prefixes from shared, immutable uint8 image pools."""

    condition_manifest: str | Path
    normalize: str = "minus_one_one"
    dequantize: bool = False
    name: str = "synthetic_long_tail_geometry"
    dim: int = field(default=0, init=False)
    image_shape: tuple[int, ...] = field(default=(), init=False)
    class_counts: tuple[int, ...] = field(default=(), init=False)
    _manifest_path: Path = field(init=False, repr=False)
    _manifest: ConditionManifest = field(init=False, repr=False)
    _arrays: tuple[np.memmap, ...] = field(init=False, repr=False)
    _offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.normalize not in {"zero_one", "minus_one_one"}:
            raise ValueError(f"Unsupported image normalization: {self.normalize}")
        self._manifest_path = Path(self.condition_manifest).expanduser().resolve()
        if not self._manifest_path.is_file():
            raise ValueError(
                f"Synthetic condition manifest does not exist: {self._manifest_path}"
            )
        try:
            self._manifest = ConditionManifest.read(self._manifest_path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Invalid synthetic condition manifest: {self._manifest_path}"
            ) from exc
        self.image_shape = tuple(int(size) for size in self._manifest.image_shape)
        if len(self.image_shape) != 3 or any(size <= 0 for size in self.image_shape):
            raise ValueError("Synthetic manifest image_shape must be a positive CHW shape.")
        if not self._manifest.classes:
            raise ValueError("Synthetic manifest must define at least one class.")
        class_ids = tuple(entry.class_id for entry in self._manifest.classes)
        if class_ids != tuple(range(len(class_ids))):
            raise ValueError("Synthetic manifest class IDs must be contiguous from zero.")

        arrays = []
        counts = []
        for entry in self._manifest.classes:
            if entry.count <= 0 or entry.index_start < 0:
                raise ValueError(
                    "Synthetic manifest class counts must be positive with non-negative "
                    "index_start values."
                )
            path = self._resolve(entry.image_path)
            if not path.is_file():
                raise ValueError(f"Synthetic class image path does not exist: {path}")
            try:
                array = np.load(path, mmap_mode="r")
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unable to memory-map synthetic image array: {path}") from exc
            if array.dtype != np.uint8:
                raise ValueError(
                    f"Synthetic image array must have dtype uint8: {path} has {array.dtype}."
                )
            if array.ndim != 4 or tuple(array.shape[1:]) != self.image_shape:
                raise ValueError(
                    "Synthetic image array shape must be (N, "
                    f"{', '.join(str(size) for size in self.image_shape)}): {path}."
                )
            if entry.index_start + entry.count > array.shape[0]:
                raise ValueError(
                    "Synthetic manifest prefix exceeds array length: "
                    f"class {entry.class_id} requests {entry.index_start + entry.count}, "
                    f"array has {array.shape[0]}."
                )
            arrays.append(array)
            counts.append(int(entry.count))

        self.dim = int(np.prod(self.image_shape))
        self.class_counts = tuple(counts)
        self._arrays = tuple(arrays)
        self._offsets = np.cumsum((0, *self.class_counts), dtype=np.int64)

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        images, _ = self.sample_with_labels(n, device=device)
        return images

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1:
            raise ValueError("SyntheticLongTailImages.sample requires n >= 1.")
        global_indices = np.random.randint(0, int(self._offsets[-1]), size=int(n))
        labels = np.searchsorted(self._offsets[1:], global_indices, side="right")
        output = np.empty((n, *self.image_shape), dtype=np.uint8)
        for class_id, (entry, array) in enumerate(
            zip(self._manifest.classes, self._arrays, strict=True)
        ):
            mask = labels == class_id
            local = global_indices[mask] - self._offsets[class_id] + entry.index_start
            output[mask] = np.asarray(array[local], dtype=np.uint8)
        images = self._normalize(torch.from_numpy(output).reshape(n, -1))
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        if device is not None:
            images = images.to(device)
            label_tensor = label_tensor.to(device)
        return images, label_tensor

    def all_samples_with_labels(
        self,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        image_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        for class_id, (entry, array) in enumerate(
            zip(self._manifest.classes, self._arrays, strict=True)
        ):
            indices = np.arange(entry.index_start, entry.index_start + entry.count)
            image_parts.append(np.asarray(array[indices], dtype=np.uint8))
            label_parts.append(np.full(entry.count, class_id, dtype=np.int64))
            source_parts.append(
                (np.int64(class_id) << np.int64(48)) | indices.astype(np.int64)
            )
        images = self._normalize(
            torch.from_numpy(np.concatenate(image_parts)).reshape(-1, self.dim)
        )
        labels = torch.from_numpy(np.concatenate(label_parts))
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        return images, labels, np.concatenate(source_parts)

    def log_prob(self, x: torch.Tensor) -> None:
        del x
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition_id": self._manifest.condition_id,
            "condition_manifest": str(self._manifest_path),
            "dim": self.dim,
            "image_shape": list(self.image_shape),
            "class_counts": list(self.class_counts),
            "normalize": self.normalize,
            "dequantize": self.dequantize,
            "config_hash": self._manifest.config_hash,
        }

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._manifest_path.parent / candidate

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        values = images.to(dtype=torch.float32) / 255.0
        if self.dequantize:
            values = torch.clamp(values + torch.rand_like(values) / 256.0, 0.0, 1.0)
        if self.normalize == "zero_one":
            return values
        return 2.0 * values - 1.0
