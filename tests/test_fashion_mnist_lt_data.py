import gzip
import struct
from pathlib import Path

import numpy as np
import torch

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.data.long_tail import long_tail_indices
from fm_lab.experiments.factory import build_target


def test_long_tail_indices_match_cifar_exponential_rule() -> None:
    labels = np.repeat(np.arange(10), 10)

    selected = long_tail_indices(
        labels,
        num_classes=10,
        imbalance_type="exp",
        imbalance_factor=0.1,
        seed=7,
    )
    counts = tuple(np.bincount(labels[selected], minlength=10).tolist())

    assert counts == (10, 7, 5, 4, 3, 2, 2, 1, 1, 1)
    assert np.array_equal(
        selected,
        long_tail_indices(
            labels,
            num_classes=10,
            imbalance_type="exp",
            imbalance_factor=0.1,
            seed=7,
        ),
    )


def test_long_tail_indices_retain_at_least_one_sample_per_class() -> None:
    labels = np.repeat(np.arange(10), 2)

    selected = long_tail_indices(
        labels,
        num_classes=10,
        imbalance_type="exp",
        imbalance_factor=0.01,
        seed=0,
    )

    assert np.all(np.bincount(labels[selected], minlength=10) >= 1)

def test_fashion_mnist_long_tail_counts_and_alignment(tmp_path: Path) -> None:
    _write_balanced_fashion_mnist(tmp_path, examples_per_class=10)

    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.1,
        subset_seed=7,
        normalize="zero_one",
    )
    images, labels, sample_ids = target.all_samples_with_labels()

    assert target.class_counts == (10, 7, 5, 4, 3, 2, 2, 1, 1, 1)
    assert images.shape == (36, 28 * 28)
    assert torch.equal(torch.round(images[:, 0] * 255).long(), labels)
    assert np.array_equal(sample_ids, target.selected_indices.astype(str))
    assert target.metadata()["image_shape"] == [1, 28, 28]
    assert len(target.metadata()["subset_sha256"]) == 64


def test_fashion_mnist_factory_builds_ir100_target(tmp_path: Path) -> None:
    _write_balanced_fashion_mnist(tmp_path, examples_per_class=100)

    target = build_target(
        {
            "data": {
                "name": "fashion_mnist_lt",
                "root": str(tmp_path),
                "imbalance_factor": 0.01,
                "subset_seed": 0,
                "normalize": "minus_one_one",
            }
        }
    )

    assert isinstance(target, LongTailedFashionMNIST)
    assert target.class_counts == (100, 59, 35, 21, 12, 7, 4, 2, 1, 1)


def test_fashion_mnist_sampling_keeps_images_and_labels_aligned(tmp_path: Path) -> None:
    _write_balanced_fashion_mnist(tmp_path, examples_per_class=4)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_type="balanced",
        imbalance_factor=1.0,
        normalize="zero_one",
    )

    torch.manual_seed(3)
    images, labels = target.sample_with_labels(20)

    assert images.shape == (20, 28 * 28)
    assert torch.equal(torch.round(images[:, 0] * 255).long(), labels)


def _write_balanced_fashion_mnist(root: Path, *, examples_per_class: int) -> None:
    labels = np.repeat(np.arange(10, dtype=np.uint8), examples_per_class)
    images = np.repeat(labels[:, None, None], 28 * 28, axis=1).reshape(-1, 28, 28)
    _write_idx_images(root / "train-images-idx3-ubyte.gz", images)
    _write_idx_labels(root / "train-labels-idx1-ubyte.gz", labels)
    _write_idx_images(root / "t10k-images-idx3-ubyte.gz", images)
    _write_idx_labels(root / "t10k-labels-idx1-ubyte.gz", labels)


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        handle.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)))
        handle.write(labels.astype(np.uint8).tobytes())
