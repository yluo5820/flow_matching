from pathlib import Path

import numpy as np
import pytest
import torch

from fm_lab.data import ImbalancedCIFARImages
from fm_lab.experiments.factory import build_target


@pytest.mark.parametrize(
    ("dataset", "num_classes", "record_size", "pixel_offset"),
    [("cifar10", 10, 3073, 1), ("cifar100", 100, 3074, 2)],
)
def test_cifar_lt_counts_are_exponential_and_deterministic(
    tmp_path: Path,
    dataset: str,
    num_classes: int,
    record_size: int,
    pixel_offset: int,
) -> None:
    _write_balanced_binary_cifar(
        tmp_path,
        dataset=dataset,
        num_classes=num_classes,
        examples_per_class=10,
        record_size=record_size,
        pixel_offset=pixel_offset,
    )

    first = ImbalancedCIFARImages(
        dataset=dataset,
        root=tmp_path,
        imbalance_factor=0.1,
        subset_seed=7,
        horizontal_flip=False,
    )
    second = ImbalancedCIFARImages(
        dataset=dataset,
        root=tmp_path,
        imbalance_factor=0.1,
        subset_seed=7,
        horizontal_flip=False,
    )
    expected = tuple(
        int(10 * 0.1 ** (class_id / (num_classes - 1.0)))
        for class_id in range(num_classes)
    )

    assert first.class_counts == expected
    assert np.array_equal(first.selected_indices, second.selected_indices)
    assert first.metadata()["subset_sha256"] == second.metadata()["subset_sha256"]


def test_cifar_lt_sampling_keeps_images_and_labels_aligned(tmp_path: Path) -> None:
    _write_balanced_binary_cifar(
        tmp_path,
        dataset="cifar10",
        num_classes=10,
        examples_per_class=4,
        record_size=3073,
        pixel_offset=1,
    )
    target = ImbalancedCIFARImages(
        dataset="cifar10",
        root=tmp_path,
        imbalance_factor=1.0,
        horizontal_flip=False,
        normalize="zero_one",
    )

    torch.manual_seed(3)
    images, labels = target.sample_with_labels(20)
    encoded_labels = torch.round(images[:, 0] * 255.0).long()

    assert images.shape == (20, 3072)
    assert torch.equal(encoded_labels, labels)
    assert target.metadata()["image_shape"] == [3, 32, 32]


def test_cifar_lt_factory_builds_reference_ir100_split(tmp_path: Path) -> None:
    _write_balanced_binary_cifar(
        tmp_path,
        dataset="cifar10",
        num_classes=10,
        examples_per_class=100,
        record_size=3073,
        pixel_offset=1,
    )

    target = build_target(
        {
            "data": {
                "name": "cifar10_lt",
                "root": str(tmp_path),
                "imbalance_factor": 0.01,
                "subset_seed": 0,
                "horizontal_flip": False,
            }
        }
    )

    assert isinstance(target, ImbalancedCIFARImages)
    assert target.class_counts == (100, 59, 35, 21, 12, 7, 4, 2, 1, 1)


def test_cifar_lt_missing_data_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="files are missing"):
        ImbalancedCIFARImages(dataset="cifar100", root=tmp_path, download=False)


def _write_balanced_binary_cifar(
    root: Path,
    *,
    dataset: str,
    num_classes: int,
    examples_per_class: int,
    record_size: int,
    pixel_offset: int,
) -> None:
    if dataset == "cifar10":
        data_dir = root / "cifar-10-batches-bin"
        path = data_dir / "data_batch_1.bin"
        extra_paths = [data_dir / f"data_batch_{index}.bin" for index in range(2, 6)]
    else:
        data_dir = root / "cifar-100-binary"
        path = data_dir / "train.bin"
        extra_paths = []
    data_dir.mkdir(parents=True, exist_ok=True)
    records = np.zeros((num_classes * examples_per_class, record_size), dtype=np.uint8)
    row = 0
    for class_id in range(num_classes):
        for _ in range(examples_per_class):
            if dataset == "cifar100":
                records[row, 0] = class_id // 5
            records[row, pixel_offset - 1] = class_id
            records[row, pixel_offset:] = class_id
            row += 1
    records.tofile(path)
    for extra_path in extra_paths:
        np.empty((0, record_size), dtype=np.uint8).tofile(extra_path)
