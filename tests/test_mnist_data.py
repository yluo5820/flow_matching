import gzip
import struct

import numpy as np
import pytest
import torch

from fm_lab.data import MNISTImages
from fm_lab.experiments.factory import build_target


def test_mnist_images_loads_idx_files(tmp_path) -> None:
    _write_fake_mnist(tmp_path, split="train", count=4)
    target = MNISTImages(root=tmp_path, train=True, download=False)

    samples = target.sample(3)
    metadata = target.metadata()

    assert samples.shape == (3, 784)
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)
    assert target.labels.shape == (4,)
    assert metadata["image_shape"] == [28, 28]
    assert metadata["image_value_range"] == [0.0, 1.0]
    assert metadata["n_images"] == 4


def test_mnist_images_sample_with_labels_returns_matching_labels(tmp_path) -> None:
    _write_fake_mnist(tmp_path, split="train", count=4)
    target = MNISTImages(root=tmp_path, train=True, download=False)

    torch.manual_seed(12)
    samples, labels = target.sample_with_labels(3)

    assert samples.shape == (3, 784)
    assert labels.shape == (3,)
    assert labels.dtype == torch.int64
    assert set(labels.tolist()) <= {0, 1, 2, 3}


def test_mnist_images_supports_centered_normalization(tmp_path) -> None:
    _write_fake_mnist(tmp_path, split="train", count=2)
    target = MNISTImages(root=tmp_path, train=True, download=False, normalize="minus_one_one")

    samples = target.sample(2)

    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 1.0)
    assert target.metadata()["image_value_range"] == [-1.0, 1.0]


def test_mnist_images_supports_dequantization(tmp_path) -> None:
    _write_fake_mnist(tmp_path, split="train", count=2)
    target = MNISTImages(
        root=tmp_path,
        train=True,
        download=False,
        normalize="minus_one_one",
        dequantize=True,
    )

    first = target.sample(2)
    second = target.sample(2)

    assert torch.all(first >= -1.0)
    assert torch.all(first <= 1.0)
    assert not torch.allclose(first, second)
    assert target.metadata()["dequantize"] is True


def test_mnist_images_missing_files_fail_clearly(tmp_path) -> None:
    target = MNISTImages(root=tmp_path, train=True, download=False)

    with pytest.raises(FileNotFoundError, match="MNIST files are missing"):
        target.sample(1)


def test_factory_builds_mnist_target(tmp_path) -> None:
    _write_fake_mnist(tmp_path, split="test", count=3)

    target = build_target(
        {
            "data": {
                "name": "mnist",
                "root": str(tmp_path),
                "train": False,
                "download": False,
            }
        }
    )

    assert isinstance(target, MNISTImages)
    assert target.sample(2).shape == (2, 784)


def _write_fake_mnist(tmp_path, *, split: str, count: int) -> None:
    prefix = "train" if split == "train" else "t10k"
    images_path = tmp_path / f"{prefix}-images-idx3-ubyte.gz"
    labels_path = tmp_path / f"{prefix}-labels-idx1-ubyte.gz"
    images = np.arange(count * 28 * 28, dtype=np.uint8)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(images_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, count, 28, 28))
        handle.write(images.tobytes())
    with gzip.open(labels_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, count))
        handle.write(labels.tobytes())
