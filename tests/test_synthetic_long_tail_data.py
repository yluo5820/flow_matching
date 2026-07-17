from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fm_lab.data import SyntheticLongTailImages
from fm_lab.experiments.factory import build_target
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    build_condition_manifests,
    build_master_pools,
)
from fm_lab.training.trainer import _sample_target_with_optional_labels


def write_tiny_condition(
    root: Path,
    *,
    counts: tuple[int, int, int],
) -> Path:
    config = {
        "seed": 17,
        "image_size": 16,
        "master_count": max(counts),
        "counts": list(counts),
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {"supersample": 1, "render_batch_size": 8},
    }
    cells = build_master_pools(config, root, replicate=0)
    manifests = build_condition_manifests(root, 0, cells, counts=counts)
    return next(path for path in manifests if path.stem == "g0_f0")


def write_indexed_manifest(root: Path) -> Path:
    """Create a small hand-authored manifest for loader-contract failures."""

    condition_dir = root / "conditions"
    condition_dir.mkdir(parents=True)
    classes = []
    for class_id in range(3):
        image_path = condition_dir / f"class_{class_id}.npy"
        np.save(
            image_path,
            np.full((4, 3, 4, 4), class_id, dtype=np.uint8),
        )
        classes.append(
            {
                "class_id": class_id,
                "object_id": f"object_{class_id}",
                "dimension_id": "low",
                "true_dimension": 1,
                "count": 2,
                "image_path": image_path.name,
                "factor_path": "unused.npy",
                "index_start": 0,
            }
        )
    manifest_path = condition_dir / "condition.json"
    manifest_path.write_text(
        json.dumps(
            {
                "condition_id": "condition",
                "replicate": 0,
                "geometry_mapping": "geometry_0",
                "frequency_mapping": "frequency_0",
                "image_shape": [3, 4, 4],
                "classes": classes,
                "config_hash": "test",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_synthetic_target_loads_indexed_prefixes(tmp_path: Path) -> None:
    manifest_path = write_tiny_condition(tmp_path, counts=(20, 5, 2))
    target = SyntheticLongTailImages(manifest_path, normalize="minus_one_one")

    assert target.dim == 3 * 16 * 16
    assert target.image_shape == (3, 16, 16)
    assert target.class_counts == (20, 5, 2)
    assert all(isinstance(array, np.memmap) for array in target._arrays)
    images, labels, source_ids = target.all_samples_with_labels()
    repeated_images, repeated_labels, repeated_source_ids = target.all_samples_with_labels()
    assert images.shape == (27, 3 * 16 * 16)
    assert torch.bincount(labels, minlength=3).tolist() == [20, 5, 2]
    assert len(np.unique(source_ids)) == 27
    assert np.array_equal(source_ids, repeated_source_ids)
    assert torch.equal(labels, repeated_labels)
    assert torch.equal(images, repeated_images)
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0


def test_synthetic_target_sampling_follows_empirical_frequency(tmp_path: Path) -> None:
    target = SyntheticLongTailImages(
        write_tiny_condition(tmp_path, counts=(200, 20, 2)),
        normalize="zero_one",
    )

    _, labels = target.sample_with_labels(20_000)

    frequencies = torch.bincount(labels, minlength=3).float() / len(labels)
    expected = torch.tensor([200.0, 20.0, 2.0]) / 222.0
    assert torch.max(torch.abs(frequencies - expected)) < 0.015


def test_synthetic_target_factory_and_training_sampler_contract(tmp_path: Path) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    target = build_target(
        {
            "data": {
                "name": "synthetic_long_tail_geometry",
                "condition_manifest": str(manifest_path),
                "normalize": "zero_one",
            }
        }
    )

    images, labels = _sample_target_with_optional_labels(
        target,
        8,
        device=torch.device("cpu"),
    )
    assert isinstance(target, SyntheticLongTailImages)
    assert images.shape == (8, 3 * 4 * 4)
    assert labels is not None
    assert labels.shape == (8,)
    assert set(labels.tolist()) <= {0, 1, 2}


def test_synthetic_target_normalizes_and_dequantizes_selected_rows(tmp_path: Path) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in raw["classes"]:
        np.save(
            manifest_path.parent / entry["image_path"],
            np.full((4, 3, 4, 4), 128, dtype=np.uint8),
        )
    base = 128.0 / 255.0
    normalized, _ = SyntheticLongTailImages(
        manifest_path,
        normalize="zero_one",
    ).sample_with_labels(4)
    torch.manual_seed(7)
    dequantized, _ = SyntheticLongTailImages(
        manifest_path,
        normalize="zero_one",
        dequantize=True,
    ).sample_with_labels(4)

    assert torch.allclose(normalized, torch.full_like(normalized, base))
    assert bool((dequantized >= base).all())
    assert bool((dequantized <= base + 1.0 / 256.0).all())
    assert not torch.equal(dequantized, normalized)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_path", "image path does not exist"),
        ("wrong_dtype", "dtype uint8"),
        ("wrong_shape", "shape"),
        ("out_of_bounds", "exceeds array length"),
        ("duplicate_class_id", "class IDs"),
    ],
)
def test_synthetic_target_rejects_invalid_manifest_or_array_contract(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = raw["classes"][0]
    first_path = manifest_path.parent / first["image_path"]
    if mutation == "missing_path":
        first["image_path"] = "missing.npy"
    elif mutation == "wrong_dtype":
        np.save(first_path, np.zeros((4, 3, 4, 4), dtype=np.float32))
    elif mutation == "wrong_shape":
        np.save(first_path, np.zeros((4, 4, 4, 3), dtype=np.uint8))
    elif mutation == "out_of_bounds":
        first["index_start"] = 3
        first["count"] = 2
    elif mutation == "duplicate_class_id":
        raw["classes"][1]["class_id"] = 0
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        SyntheticLongTailImages(manifest_path)


def test_synthetic_target_rejects_invalid_normalization_at_load_time(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported image normalization"):
        SyntheticLongTailImages(write_indexed_manifest(tmp_path), normalize="bad")


def test_synthetic_target_factory_requires_condition_manifest() -> None:
    with pytest.raises(ValueError, match="data.condition_manifest is required"):
        build_target({"data": {"name": "synthetic_long_tail_geometry"}})
