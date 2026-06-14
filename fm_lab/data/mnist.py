"""MNIST image target distribution."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass
class MNISTImages:
    """Flattened MNIST image distribution.

    Samples are returned as float tensors with shape `(n, 784)`. The target stores
    only images because the current flow-matching pipeline is unconditional.
    """

    root: str | Path = "data/mnist"
    train: bool = True
    download: bool = False
    normalize: str = "zero_one"
    name: str = "mnist"
    dim: int = 28 * 28
    image_shape: tuple[int, int] = (28, 28)
    _images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _labels: torch.Tensor | None = field(default=None, init=False, repr=False)

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if n < 1:
            raise ValueError("MNISTImages.sample requires n >= 1.")
        images = self.images
        indices = torch.randint(0, images.shape[0], (n,))
        samples = images[indices]
        if device is not None:
            samples = samples.to(torch.device(device))
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        del x
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "root": str(self.root),
            "train": self.train,
            "normalize": self.normalize,
            "image_shape": list(self.image_shape),
            "image_value_range": list(_image_value_range(self.normalize)),
            "n_images": int(self.images.shape[0]),
        }

    @property
    def images(self) -> torch.Tensor:
        self._load()
        assert self._images is not None
        return self._images

    @property
    def labels(self) -> torch.Tensor:
        self._load()
        assert self._labels is not None
        return self._labels

    def _load(self) -> None:
        if self._images is not None and self._labels is not None:
            return
        root = Path(self.root)
        if self.download:
            _download_mnist(root)

        split = "train" if self.train else "test"
        images_path = root / _MNIST_FILES[f"{split}_images"]
        labels_path = root / _MNIST_FILES[f"{split}_labels"]
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "MNIST files are missing. Set data.download: true or place IDX gzip files "
                f"under {root}."
            )

        images = _read_idx_images(images_path)
        labels = _read_idx_labels(labels_path)
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"MNIST image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}."
            )
        self._images = _normalize_images(images, self.normalize)
        self._labels = labels


def _download_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for key, url in _MNIST_URLS.items():
        output_path = root / _MNIST_FILES[key]
        if output_path.exists():
            continue
        urllib.request.urlretrieve(url, output_path)  # noqa: S310


def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number in {path}: {magic}.")
        data = np.frombuffer(handle.read(), dtype=np.uint8).copy()
    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"Invalid MNIST image file size in {path}: {data.size} != {expected}.")
    return torch.from_numpy(data.reshape(count, rows * cols)).float()


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number in {path}: {magic}.")
        data = np.frombuffer(handle.read(), dtype=np.uint8).copy()
    if data.size != count:
        raise ValueError(f"Invalid MNIST label file size in {path}: {data.size} != {count}.")
    return torch.from_numpy(data.astype(np.int64))


def _normalize_images(images: torch.Tensor, normalize: str) -> torch.Tensor:
    normalized = normalize.lower()
    if normalized in {"zero_one", "01", "unit"}:
        return images / 255.0
    if normalized in {"minus_one_one", "-1_1", "centered"}:
        return images / 127.5 - 1.0
    raise ValueError(f"Unsupported MNIST normalization: {normalize}")


def _image_value_range(normalize: str) -> tuple[float, float]:
    normalized = normalize.lower()
    if normalized in {"zero_one", "01", "unit"}:
        return (0.0, 1.0)
    if normalized in {"minus_one_one", "-1_1", "centered"}:
        return (-1.0, 1.0)
    raise ValueError(f"Unsupported MNIST normalization: {normalize}")
