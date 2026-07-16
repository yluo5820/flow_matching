"""Deterministic long-tailed Fashion-MNIST target distribution."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fm_lab.data.long_tail import long_tail_indices, nested_frequency_split
from fm_lab.data.mnist import (
    _image_value_range,
    _normalize_images,
    _read_idx_images,
    _read_idx_labels,
)

_FASHION_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_FASHION_MNIST_BASE_URL = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion"
)


@dataclass
class LongTailedFashionMNIST:
    """Fashion-MNIST with deterministic exponential class subsampling."""

    root: str | Path = "data/fashion_mnist"
    train: bool = True
    download: bool = False
    imbalance_type: str = "exp"
    imbalance_factor: float = 0.01
    subset_seed: int = 0
    normalize: str = "minus_one_one"
    dequantize: bool = False
    frequency_mapping_offset: int | None = None
    frequency_mapping_multiplier: int = 3
    diagnostic_pool_per_class: int = 0
    name: str = field(default="fashion_mnist_lt", init=False)
    dim: int = field(default=28 * 28, init=False)
    image_shape: tuple[int, int, int] = field(default=(1, 28, 28), init=False)
    num_classes: int = field(default=10, init=False)
    _raw_images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _selected_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _class_counts: tuple[int, ...] = field(default=(), init=False, repr=False)
    _class_ranks: tuple[int, ...] = field(default=(), init=False, repr=False)
    _probe_a_raw_images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _probe_a_labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _probe_a_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _probe_b_raw_images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _probe_b_labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _probe_b_indices: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.imbalance_type not in {"exp", "balanced"}:
            raise ValueError("imbalance_type must be 'exp' or 'balanced'.")
        if not 0.0 < self.imbalance_factor <= 1.0:
            raise ValueError("imbalance_factor must be in (0, 1].")
        if self.frequency_mapping_offset is None and self.diagnostic_pool_per_class:
            raise ValueError(
                "diagnostic_pool_per_class requires a frequency mapping offset."
            )
        if self.frequency_mapping_offset is not None:
            if not self.train:
                raise ValueError("Frequency mappings are only supported for training data.")
            if self.diagnostic_pool_per_class <= 0:
                raise ValueError(
                    "Frequency mappings require a positive diagnostic_pool_per_class."
                )
        self.name = "fashion_mnist_lt" if self.train else "fashion_mnist"
        self._load()

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        images, _ = self.sample_with_labels(n, device=device)
        return images

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1:
            raise ValueError("LongTailedFashionMNIST.sample requires n >= 1.")
        assert self._raw_images is not None
        assert self._labels is not None
        indices = torch.randint(len(self._raw_images), (n,))
        images = _normalize_images(
            self._raw_images[indices],
            self.normalize,
            dequantize=self.dequantize,
        )
        labels = self._labels[indices]
        if device is not None:
            resolved = torch.device(device)
            images = images.to(resolved)
            labels = labels.to(resolved)
        return images, labels

    def all_samples_with_labels(
        self,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Return every retained image once without stochastic dequantization."""

        assert self._raw_images is not None
        assert self._labels is not None
        assert self._selected_indices is not None
        images = _normalize_images(self._raw_images, self.normalize)
        labels = self._labels.clone()
        if device is not None:
            resolved = torch.device(device)
            images = images.to(resolved)
            labels = labels.to(resolved)
        return images, labels, self._selected_indices.astype(str)

    def log_prob(self, x: torch.Tensor) -> None:
        del x
        return None

    def metadata(self) -> dict:
        assert self._selected_indices is not None
        metadata = {
            "name": self.name,
            "dataset": "fashion_mnist",
            "dim": self.dim,
            "root": str(self.root),
            "train": self.train,
            "normalize": self.normalize,
            "dequantize": self.dequantize,
            "image_shape": list(self.image_shape),
            "image_value_range": list(_image_value_range(self.normalize)),
            "num_classes": self.num_classes,
            "n_images": int(len(self.labels)),
            "imbalance_type": self.imbalance_type,
            "imbalance_factor": self.imbalance_factor,
            "imbalance_ratio": float(1.0 / self.imbalance_factor),
            "subset_seed": self.subset_seed,
            "class_counts": list(self.class_counts),
            "subset_sha256": hashlib.sha256(
                self._selected_indices.astype(np.int64).tobytes()
            ).hexdigest(),
        }
        if self.frequency_mapping_offset is not None:
            probe_a = self.diagnostic_indices("a")
            probe_b = self.diagnostic_indices("b")
            metadata["frequency_mapping"] = {
                "offset": int(self.frequency_mapping_offset),
                "multiplier": int(self.frequency_mapping_multiplier),
                "diagnostic_pool_per_class": int(self.diagnostic_pool_per_class),
                "class_ranks": list(self._class_ranks),
                "probe_a_count": int(len(probe_a)),
                "probe_b_count": int(len(probe_b)),
                "probe_a_sha256": _indices_sha256(probe_a),
                "probe_b_sha256": _indices_sha256(probe_b),
            }
        return metadata

    @property
    def labels(self) -> torch.Tensor:
        assert self._labels is not None
        return self._labels

    @property
    def class_counts(self) -> tuple[int, ...]:
        return self._class_counts

    @property
    def selected_indices(self) -> np.ndarray:
        assert self._selected_indices is not None
        return self._selected_indices.copy()

    def diagnostic_indices(self, split: str) -> np.ndarray:
        """Return original dataset indices for held-out Probe-A or Probe-B."""

        _, _, indices = self._diagnostic_storage(split)
        return indices.copy()

    def diagnostic_samples(
        self,
        split: str,
        *,
        original_indices: np.ndarray | None = None,
        dequantization_seeds: np.ndarray | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Materialize held-out samples by original ID with optional seeded noise."""

        raw_images, labels, available_indices = self._diagnostic_storage(split)
        if original_indices is None:
            requested = available_indices.copy()
            positions = np.arange(len(available_indices), dtype=np.int64)
        else:
            requested = np.asarray(original_indices, dtype=np.int64)
            if requested.ndim != 1:
                raise ValueError("Diagnostic original_indices must be a vector.")
            if len(np.unique(requested)) != len(requested):
                raise ValueError("Diagnostic original_indices must not contain duplicates.")
            lookup = {
                int(original_id): position
                for position, original_id in enumerate(available_indices)
            }
            missing = [int(value) for value in requested if int(value) not in lookup]
            if missing:
                raise ValueError(
                    f"Diagnostic indices are not part of Probe-{split.upper()}: {missing[:5]}"
                )
            positions = np.asarray(
                [lookup[int(value)] for value in requested],
                dtype=np.int64,
            )

        selected_images = raw_images[torch.from_numpy(positions)].clone()
        selected_labels = labels[torch.from_numpy(positions)].clone()
        if dequantization_seeds is None:
            images = _normalize_images(selected_images, self.normalize)
        else:
            if not self.dequantize:
                raise ValueError(
                    "dequantization_seeds require a target configured with dequantize=True."
                )
            seeds = np.asarray(dequantization_seeds, dtype=np.int64)
            if seeds.ndim != 1 or len(seeds) != len(selected_images):
                raise ValueError(
                    "dequantization_seeds must match the number of diagnostic samples."
                )
            images = _seeded_dequantize(selected_images, seeds, self.normalize)

        if device is not None:
            resolved = torch.device(device)
            images = images.to(resolved)
            selected_labels = selected_labels.to(resolved)
        return images, selected_labels, requested.astype(str)

    def _diagnostic_storage(
        self,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        normalized = str(split).lower()
        if self.frequency_mapping_offset is None:
            raise ValueError("Diagnostic probes are not configured for this target.")
        if normalized == "a":
            images, labels, indices = (
                self._probe_a_raw_images,
                self._probe_a_labels,
                self._probe_a_indices,
            )
        elif normalized == "b":
            images, labels, indices = (
                self._probe_b_raw_images,
                self._probe_b_labels,
                self._probe_b_indices,
            )
        else:
            raise ValueError("Diagnostic split must be 'a' or 'b'.")
        assert images is not None
        assert labels is not None
        assert indices is not None
        return images, labels, indices

    def _load(self) -> None:
        root = Path(self.root)
        if self.download:
            _download_fashion_mnist(root)
        split = "train" if self.train else "test"
        images_path = root / _FASHION_MNIST_FILES[f"{split}_images"]
        labels_path = root / _FASHION_MNIST_FILES[f"{split}_labels"]
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "Fashion-MNIST files are missing. Set data.download: true or place IDX "
                f"gzip files under {root}."
            )
        images = _read_idx_images(images_path)
        labels_tensor = _read_idx_labels(labels_path)
        labels = labels_tensor.numpy()
        if len(images) != len(labels):
            raise ValueError(
                f"Fashion-MNIST image/label count mismatch: {len(images)} vs {len(labels)}."
            )
        if self.train and self.frequency_mapping_offset is not None:
            frequency_split = nested_frequency_split(
                labels,
                num_classes=self.num_classes,
                imbalance_factor=self.imbalance_factor,
                seed=self.subset_seed,
                diagnostic_pool_per_class=self.diagnostic_pool_per_class,
                multiplier=self.frequency_mapping_multiplier,
                offset=self.frequency_mapping_offset,
            )
            selected = frequency_split.train_indices.copy()
            self._class_ranks = frequency_split.class_ranks
            self._probe_a_indices = frequency_split.probe_a_indices.copy()
            self._probe_b_indices = frequency_split.probe_b_indices.copy()
            self._probe_a_raw_images = images[self._probe_a_indices].clone()
            self._probe_b_raw_images = images[self._probe_b_indices].clone()
            self._probe_a_labels = labels_tensor[self._probe_a_indices].clone()
            self._probe_b_labels = labels_tensor[self._probe_b_indices].clone()
        elif self.train:
            selected = long_tail_indices(
                labels,
                num_classes=self.num_classes,
                imbalance_type=self.imbalance_type,
                imbalance_factor=self.imbalance_factor,
                seed=self.subset_seed,
            )
        else:
            selected = np.arange(len(labels), dtype=np.int64)
        selected_labels = labels[selected]
        self._selected_indices = selected
        self._raw_images = images[selected].clone()
        self._labels = torch.from_numpy(selected_labels.astype(np.int64, copy=True))
        self._class_counts = tuple(
            int(np.sum(selected_labels == class_id)) for class_id in range(self.num_classes)
        )


def _seeded_dequantize(
    raw_images: torch.Tensor,
    seeds: np.ndarray,
    normalize: str,
) -> torch.Tensor:
    noise_rows: list[torch.Tensor] = []
    for raw_image, seed in zip(raw_images, seeds, strict=True):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise_rows.append(
            torch.rand(
                raw_image.shape,
                generator=generator,
                dtype=raw_image.dtype,
            )
        )
    images = raw_images + torch.stack(noise_rows)
    normalized = normalize.lower()
    if normalized in {"zero_one", "01", "unit"}:
        return images / 256.0
    if normalized in {"minus_one_one", "-1_1", "centered"}:
        return 2.0 * (images / 256.0) - 1.0
    raise ValueError(f"Unsupported MNIST normalization: {normalize}")


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _download_fashion_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in _FASHION_MNIST_FILES.values():
        output_path = root / filename
        if not output_path.exists():
            urllib.request.urlretrieve(f"{_FASHION_MNIST_BASE_URL}/{filename}", output_path)  # noqa: S310
