import pickle
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


def test_cifar100_reads_autodl_python_archive_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "cifar-100-python"
    data_dir.mkdir(parents=True)
    labels = np.repeat(np.arange(100, dtype=np.int64), 2)
    images = np.zeros((len(labels), 3 * 32 * 32), dtype=np.uint8)
    images[:, 0] = labels
    with (data_dir / "train").open("wb") as handle:
        pickle.dump({"data": images, "fine_labels": labels.tolist()}, handle)

    target = ImbalancedCIFARImages(
        dataset="cifar100",
        root=tmp_path,
        imbalance_factor=1.0,
        horizontal_flip=False,
        normalize="zero_one",
    )

    loaded, loaded_labels, _ = target.all_samples_with_labels()
    assert loaded.shape == (200, 3 * 32 * 32)
    assert torch.equal(torch.round(loaded[:, 0] * 255).long(), loaded_labels)


def test_cifar_all_samples_returns_each_selected_image_once_without_augmentation(
    tmp_path: Path,
) -> None:
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
        horizontal_flip=True,
        normalize="zero_one",
    )

    images, labels, sample_ids = target.all_samples_with_labels()

    assert images.shape == (40, 3072)
    assert torch.equal(torch.round(images[:, 0] * 255).long(), labels)
    assert np.array_equal(sample_ids, target.selected_indices.astype(str))


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


def test_cifar_frequency_mapping_reserves_disjoint_deterministic_probes(
    tmp_path: Path,
) -> None:
    _write_balanced_binary_cifar(
        tmp_path,
        dataset="cifar10",
        num_classes=10,
        examples_per_class=40,
        record_size=3073,
        pixel_offset=1,
    )
    target = ImbalancedCIFARImages(
        dataset="cifar10",
        root=tmp_path,
        imbalance_factor=0.1,
        subset_seed=7,
        horizontal_flip=True,
        dequantize=True,
        frequency_mapping_offset=0,
        frequency_mapping_multiplier=3,
        diagnostic_pool_per_class=8,
    )

    probe_a = target.diagnostic_indices("a")
    probe_b = target.diagnostic_indices("b")

    assert len(probe_a) == len(probe_b) == 40
    assert set(probe_a).isdisjoint(probe_b)
    assert set(probe_a).isdisjoint(target.selected_indices)
    assert set(probe_b).isdisjoint(target.selected_indices)
    assert target.class_ranks == (0, 3, 6, 9, 2, 5, 8, 1, 4, 7)

    requested = probe_a[[3, 0, 7]]
    seeds = np.asarray([11, 12, 13], dtype=np.int64)
    first, labels, returned = target.diagnostic_samples(
        "a",
        original_indices=requested,
        dequantization_seeds=seeds,
    )
    second, repeated_labels, repeated_ids = target.diagnostic_samples(
        "a",
        original_indices=requested,
        dequantization_seeds=seeds,
    )

    assert np.array_equal(returned.astype(np.int64), requested)
    assert np.array_equal(repeated_ids, returned)
    assert torch.equal(first, second)
    assert torch.equal(labels, repeated_labels)
    assert first.shape == (3, 3072)
    assert target.metadata()["frequency_mapping"]["probe_a_sha256"]
    assert target.metadata()["frequency_mapping"]["probe_b_sha256"]


def test_cifar_diagnostic_samples_never_apply_training_flip(tmp_path: Path) -> None:
    _write_balanced_binary_cifar(
        tmp_path,
        dataset="cifar10",
        num_classes=10,
        examples_per_class=12,
        record_size=3073,
        pixel_offset=1,
    )
    common = {
        "dataset": "cifar10",
        "root": tmp_path,
        "imbalance_factor": 1.0,
        "subset_seed": 3,
        "normalize": "zero_one",
        "dequantize": True,
        "frequency_mapping_offset": 0,
        "diagnostic_pool_per_class": 4,
    }
    augmented = ImbalancedCIFARImages(horizontal_flip=True, **common)
    unaugmented = ImbalancedCIFARImages(horizontal_flip=False, **common)
    requested = augmented.diagnostic_indices("b")[:5]
    seeds = np.arange(5, dtype=np.int64) + 100

    with_flip, _, _ = augmented.diagnostic_samples(
        "b",
        original_indices=requested,
        dequantization_seeds=seeds,
    )
    without_flip, _, _ = unaugmented.diagnostic_samples(
        "b",
        original_indices=requested,
        dequantization_seeds=seeds,
    )

    assert torch.equal(with_flip, without_flip)
    with pytest.raises(ValueError, match="not part of Probe-A"):
        augmented.diagnostic_samples(
            "a",
            original_indices=np.asarray([int(requested[0])]),
            dequantization_seeds=np.asarray([1]),
        )


def test_cifar_factory_passes_frequency_mapping_and_dequantization(
    tmp_path: Path,
) -> None:
    _write_balanced_binary_cifar(
        tmp_path,
        dataset="cifar10",
        num_classes=10,
        examples_per_class=12,
        record_size=3073,
        pixel_offset=1,
    )

    target = build_target(
        {
            "data": {
                "name": "cifar10_lt",
                "root": str(tmp_path),
                "imbalance_factor": 0.1,
                "subset_seed": 2,
                "horizontal_flip": False,
                "dequantize": True,
                "frequency_mapping": {
                    "offset": 0,
                    "multiplier": 3,
                    "diagnostic_pool_per_class": 4,
                },
            }
        }
    )

    assert isinstance(target, ImbalancedCIFARImages)
    assert target.dequantize is True
    assert target.frequency_mapping_offset == 0
    assert target.frequency_mapping_multiplier == 3
    assert target.diagnostic_pool_per_class == 4
    assert len(target.diagnostic_indices("a")) == 20


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
