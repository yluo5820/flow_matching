from pathlib import Path

import numpy as np
import pytest
import torch
from probe_helpers import write_balanced_fashion_mnist

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.data.long_tail import (
    frequency_rank_mapping,
    long_tail_indices,
    nested_frequency_split,
)
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


def test_frequency_rank_mappings_balance_class_and_rank() -> None:
    mappings = np.stack(
        [
            frequency_rank_mapping(10, multiplier=3, offset=mapping_id)
            for mapping_id in range(10)
        ]
    )

    assert all(
        np.array_equal(np.sort(mapping), np.arange(10)) for mapping in mappings
    )
    assert all(
        np.array_equal(np.sort(mappings[:, class_id]), np.arange(10))
        for class_id in range(10)
    )


def test_frequency_rank_mapping_requires_coprime_multiplier() -> None:
    with pytest.raises(ValueError, match="coprime"):
        frequency_rank_mapping(10, multiplier=2, offset=0)


def test_nested_frequency_split_reserves_probe_pool_before_training() -> None:
    labels = np.repeat(np.arange(10), 100)

    split = nested_frequency_split(
        labels,
        num_classes=10,
        imbalance_factor=0.01,
        seed=17,
        diagnostic_pool_per_class=20,
        multiplier=3,
        offset=0,
    )

    assert len(split.probe_a_indices) == 100
    assert len(split.probe_b_indices) == 100
    assert not set(split.train_indices) & set(split.probe_a_indices)
    assert not set(split.train_indices) & set(split.probe_b_indices)
    assert not set(split.probe_a_indices) & set(split.probe_b_indices)
    assert max(split.class_counts) == 80
    assert min(split.class_counts) == 1


def test_same_class_frequency_subsets_are_nested_across_mappings() -> None:
    labels = np.repeat(np.arange(10), 100)
    splits = [
        nested_frequency_split(
            labels,
            num_classes=10,
            imbalance_factor=0.01,
            seed=17,
            diagnostic_pool_per_class=20,
            multiplier=3,
            offset=mapping_id,
        )
        for mapping_id in range(10)
    ]

    for class_id in range(10):
        class_sets = [
            set(split.train_indices[labels[split.train_indices] == class_id])
            for split in splits
        ]
        ordered = sorted(class_sets, key=len)
        assert all(
            smaller <= larger
            for smaller, larger in zip(ordered, ordered[1:], strict=False)
        )


def test_fashion_mnist_long_tail_counts_and_alignment(tmp_path: Path) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=10)

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


def test_fashion_mnist_subset_remaps_original_classes_to_compact_labels(
    tmp_path: Path,
) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=10)

    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_type="balanced",
        imbalance_factor=1.0,
        normalize="zero_one",
        class_ids=(1, 5, 9),
    )
    images, labels, sample_ids = target.all_samples_with_labels()

    assert target.num_classes == 3
    assert target.class_counts == (10, 10, 10)
    assert set(labels.tolist()) == {0, 1, 2}
    assert target.metadata()["original_class_ids"] == [1, 5, 9]
    original_from_compact = torch.tensor([1, 5, 9])[labels]
    assert torch.equal(torch.round(images[:, 0] * 255).long(), original_from_compact)
    assert np.array_equal(sample_ids, target.selected_indices.astype(str))


def test_fashion_mnist_factory_builds_ir100_target(tmp_path: Path) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=100)

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


def test_fashion_mnist_factory_builds_counterfactual_mapping_with_held_out_probes(
    tmp_path: Path,
) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=100)

    target = build_target(
        {
            "data": {
                "name": "fashion_mnist_lt",
                "root": str(tmp_path),
                "imbalance_factor": 0.01,
                "subset_seed": 7,
                "normalize": "minus_one_one",
                "frequency_mapping": {
                    "offset": 4,
                    "multiplier": 3,
                    "diagnostic_pool_per_class": 20,
                },
            }
        }
    )

    assert target.metadata()["frequency_mapping"]["offset"] == 4
    assert len(target.diagnostic_indices("a")) == 100
    assert len(target.diagnostic_indices("b")) == 100
    assert not set(target.selected_indices) & set(target.diagnostic_indices("a"))
    assert not set(target.selected_indices) & set(target.diagnostic_indices("b"))


def test_diagnostic_samples_are_addressed_by_original_id_and_seed(
    tmp_path: Path,
) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.01,
        subset_seed=7,
        normalize="minus_one_one",
        dequantize=True,
        frequency_mapping_offset=0,
        frequency_mapping_multiplier=3,
        diagnostic_pool_per_class=20,
    )
    ids = target.diagnostic_indices("a")[:3]
    seeds = np.array([11, 12, 13], dtype=np.int64)

    first, first_labels, returned = target.diagnostic_samples(
        "a",
        original_indices=ids,
        dequantization_seeds=seeds,
    )
    second, second_labels, _ = target.diagnostic_samples(
        "a",
        original_indices=ids,
        dequantization_seeds=seeds,
    )
    changed, _, _ = target.diagnostic_samples(
        "a",
        original_indices=ids,
        dequantization_seeds=seeds + 1,
    )

    assert torch.equal(first, second)
    assert torch.equal(first_labels, second_labels)
    assert not torch.equal(first, changed)
    assert np.array_equal(returned.astype(np.int64), ids)


def test_legacy_fashion_mnist_target_does_not_reserve_diagnostic_data(
    tmp_path: Path,
) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=10)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.1,
        subset_seed=7,
        normalize="zero_one",
    )

    assert target.class_counts == (10, 7, 5, 4, 3, 2, 2, 1, 1, 1)
    assert "frequency_mapping" not in target.metadata()
    with pytest.raises(ValueError, match="not configured"):
        target.diagnostic_indices("a")


def test_fashion_mnist_sampling_keeps_images_and_labels_aligned(tmp_path: Path) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=4)
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


def test_class_balanced_sampling_changes_exposure_not_unique_support(
    tmp_path: Path,
) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.01,
        subset_seed=7,
        normalize="zero_one",
        sampling_policy="class_balanced",
    )

    torch.manual_seed(3)
    images, labels = target.sample_with_labels(20_000)
    sampled_counts = torch.bincount(labels, minlength=10)

    assert target.class_counts == (100, 59, 35, 21, 12, 7, 4, 2, 1, 1)
    assert torch.all(torch.abs(sampled_counts - 2000) < 175)
    assert torch.equal(torch.round(images[:, 0] * 255).long(), labels)
    assert target.metadata()["sampling_policy"] == "class_balanced"


def test_subset_frequency_mapping_uses_compact_class_space(tmp_path: Path) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.01,
        subset_seed=7,
        normalize="zero_one",
        class_ids=(1, 5, 9),
        frequency_mapping_offset=0,
        frequency_mapping_multiplier=2,
        diagnostic_pool_per_class=20,
    )

    assert target.num_classes == 3
    assert len(target.diagnostic_indices("a")) == 30
    assert target.class_counts == (80, 1, 8)
    assert target.metadata()["frequency_mapping"]["class_ranks"] == [0, 2, 1]
    assert set(target.labels.tolist()) == {0, 1, 2}
