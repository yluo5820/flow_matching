"""Deterministic long-tailed Fashion-MNIST target distribution."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fm_lab.data.long_tail import long_tail_indices
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
    name: str = field(default="fashion_mnist_lt", init=False)
    dim: int = field(default=28 * 28, init=False)
    image_shape: tuple[int, int, int] = field(default=(1, 28, 28), init=False)
    num_classes: int = field(default=10, init=False)
    _raw_images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _selected_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _class_counts: tuple[int, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.imbalance_type not in {"exp", "balanced"}:
            raise ValueError("imbalance_type must be 'exp' or 'balanced'.")
        if not 0.0 < self.imbalance_factor <= 1.0:
            raise ValueError("imbalance_factor must be in (0, 1].")
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
        return {
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
        if self.train:
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


def _download_fashion_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in _FASHION_MNIST_FILES.values():
        output_path = root / filename
        if not output_path.exists():
            urllib.request.urlretrieve(f"{_FASHION_MNIST_BASE_URL}/{filename}", output_path)  # noqa: S310
