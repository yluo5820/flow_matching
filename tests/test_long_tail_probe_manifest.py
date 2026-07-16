from pathlib import Path

import numpy as np
import pytest
import torch
from long_tail_geometry_helpers import write_balanced_fashion_mnist

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeManifest,
    build_probe_manifest,
    build_source_noise_replica,
    materialize_probe_batch,
)
from fm_lab.sources import GaussianSource


def test_probe_manifest_is_balanced_over_class_and_timestep(tmp_path: Path) -> None:
    sample_ids = np.arange(40)
    labels = np.repeat(np.arange(4), 10)

    manifest = build_probe_manifest(
        sample_ids,
        labels,
        split="a",
        rows_per_class_per_stratum=8,
        batch_size=4,
        time_strata=((0.02, 0.10), (0.10, 0.30)),
        seed=19,
    )
    saved = manifest.save(tmp_path / "probe.npz")
    restored = ProbeManifest.load(saved)

    assert manifest.num_rows == 4 * 2 * 8
    assert manifest.digest == restored.digest
    assert np.array_equal(manifest.original_indices, restored.original_indices)
    for class_id in range(4):
        for stratum_id in range(2):
            assert (
                np.sum(
                    (manifest.labels == class_id)
                    & (manifest.stratum_ids == stratum_id)
                )
                == 8
            )
    assert all(len(rows) == 4 for rows in manifest.microbatch_row_indices())


def test_probe_manifest_rejects_insufficient_unique_class_examples() -> None:
    with pytest.raises(ValueError, match="unique examples"):
        build_probe_manifest(
            np.arange(8),
            np.repeat(np.arange(2), 4),
            split="a",
            rows_per_class_per_stratum=5,
            batch_size=5,
            time_strata=((0.02, 0.10),),
            seed=19,
        )


def test_source_noise_replica_changes_only_source_seeds() -> None:
    manifest = build_probe_manifest(
        np.arange(40),
        np.repeat(np.arange(4), 10),
        split="a",
        rows_per_class_per_stratum=8,
        batch_size=4,
        time_strata=((0.02, 0.10), (0.10, 0.30)),
        seed=19,
    )
    numpy_state = np.random.get_state()

    replica = build_source_noise_replica(manifest, seed=31)

    assert replica.digest != manifest.digest
    assert not np.array_equal(replica.source_seeds, manifest.source_seeds)
    for field in (
        "original_indices",
        "labels",
        "dequantization_seeds",
        "timesteps",
        "stratum_ids",
        "microbatch_ids",
    ):
        assert np.array_equal(getattr(replica, field), getattr(manifest, field))
    restored_numpy_state = np.random.get_state()
    assert numpy_state[0] == restored_numpy_state[0]
    assert np.array_equal(numpy_state[1], restored_numpy_state[1])
    assert numpy_state[2:] == restored_numpy_state[2:]


def test_same_manifest_materializes_identical_tuples_across_frequency_mappings(
    tmp_path: Path,
) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    common = {
        "root": tmp_path,
        "imbalance_factor": 0.01,
        "subset_seed": 7,
        "normalize": "minus_one_one",
        "dequantize": True,
        "frequency_mapping_multiplier": 3,
        "diagnostic_pool_per_class": 20,
    }
    target0 = LongTailedFashionMNIST(**common, frequency_mapping_offset=0)
    target7 = LongTailedFashionMNIST(**common, frequency_mapping_offset=7)
    _, labels, sample_ids = target0.diagnostic_samples("a")
    manifest = build_probe_manifest(
        sample_ids.astype(np.int64),
        labels.numpy(),
        split="a",
        rows_per_class_per_stratum=4,
        batch_size=4,
        time_strata=((0.02, 0.10), (0.10, 0.30)),
        seed=19,
    )
    rows = np.arange(16)

    batch0 = materialize_probe_batch(
        target0,
        GaussianSource(dim=784),
        manifest,
        rows,
        device="cpu",
    )
    batch7 = materialize_probe_batch(
        target7,
        GaussianSource(dim=784),
        manifest,
        rows,
        device="cpu",
    )

    assert torch.equal(batch0.x0, batch7.x0)
    assert torch.equal(batch0.x1, batch7.x1)
    assert torch.equal(batch0.t, batch7.t)
    assert torch.equal(batch0.labels, batch7.labels)
    assert np.array_equal(batch0.original_indices, batch7.original_indices)


def test_materializing_probe_batch_preserves_global_torch_rng(tmp_path: Path) -> None:
    write_balanced_fashion_mnist(tmp_path, examples_per_class=10)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.1,
        subset_seed=7,
        normalize="minus_one_one",
        dequantize=True,
        frequency_mapping_offset=0,
        frequency_mapping_multiplier=3,
        diagnostic_pool_per_class=2,
    )
    _, labels, sample_ids = target.diagnostic_samples("a")
    manifest = build_probe_manifest(
        sample_ids.astype(np.int64),
        labels.numpy(),
        split="a",
        rows_per_class_per_stratum=1,
        batch_size=1,
        time_strata=((0.02, 0.10),),
        seed=19,
    )
    state = torch.get_rng_state().clone()

    materialize_probe_batch(
        target,
        GaussianSource(dim=784),
        manifest,
        np.arange(10),
        device="cpu",
    )

    assert torch.equal(torch.get_rng_state(), state)
