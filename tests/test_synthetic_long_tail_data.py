from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fm_lab.data import SyntheticLongTailImages
from fm_lab.experiments.factory import build_target
from fm_lab.geometry_explorer.latent_factors import sample_values
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    FACTOR_COLUMNS,
    GEOMETRY_MAPPINGS,
    OBJECT_IDS,
    build_condition_manifests,
    build_factor_space,
    build_master_pools,
    canonical_factor_rows,
)
from fm_lab.training.trainer import _sample_target_with_optional_labels


@pytest.mark.parametrize(
    "script",
    [
        (
            "from fm_lab.geometry_explorer.synthetic_long_tail_design "
            "import FACTOR_COLUMNS, build_master_pools; "
            "assert FACTOR_COLUMNS; assert build_master_pools; "
            "from fm_lab.data import SyntheticLongTailImages; "
            "from fm_lab.experiments.factory import build_target; "
            "assert SyntheticLongTailImages; assert build_target"
        ),
        (
            "from fm_lab.data import SyntheticLongTailImages; "
            "from fm_lab.experiments.factory import build_target; "
            "from fm_lab.geometry_explorer.synthetic_long_tail_design "
            "import FACTOR_COLUMNS, build_master_pools; "
            "assert SyntheticLongTailImages; assert build_target; "
            "assert FACTOR_COLUMNS; assert build_master_pools"
        ),
    ],
)
def test_synthetic_long_tail_imports_work_in_a_fresh_process(script: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ | {"PYTHONPATH": str(project_root)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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


def write_indexed_manifest(root: Path, *, index_start: int = 0) -> Path:
    """Create a small hand-authored manifest for loader-contract failures."""

    replicate_root = root / "replicate_00"
    condition_dir = replicate_root / "conditions"
    condition_dir.mkdir(parents=True)
    classes = []
    for class_id, (object_id, dimension_id) in enumerate(
        zip(OBJECT_IDS, GEOMETRY_MAPPINGS[0], strict=True)
    ):
        pool_dir = replicate_root / "pools" / object_id / dimension_id
        pool_dir.mkdir(parents=True)
        image_path = pool_dir / "images.npy"
        factor_path = pool_dir / "factors.npy"
        np.save(
            image_path,
            np.full((4, 3, 4, 4), class_id, dtype=np.uint8),
        )
        factor = build_factor_space(dimension_id)
        factor_template = canonical_factor_rows(
            factor,
            sample_values(factor.sample(1, seed=class_id)),
        )
        np.save(factor_path, np.repeat(factor_template, 4, axis=0))
        classes.append(
            {
                "class_id": class_id,
                "object_id": object_id,
                "dimension_id": dimension_id,
                "true_dimension": factor.dim,
                "count": 2,
                "image_path": str(
                    Path("..") / "pools" / object_id / dimension_id / "images.npy"
                ),
                "factor_path": str(
                    Path("..") / "pools" / object_id / dimension_id / "factors.npy"
                ),
                "index_start": index_start,
            }
        )
    manifest_path = condition_dir / "g0_f0.json"
    manifest_path.write_text(
        json.dumps(
            {
                "condition_id": "g0_f0",
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
    assert all(not array.flags.writeable for array in target._arrays)
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


def test_synthetic_target_uses_nonzero_index_starts_for_stable_source_ids(
    tmp_path: Path,
) -> None:
    target = SyntheticLongTailImages(write_indexed_manifest(tmp_path, index_start=1))

    _, labels, source_ids = target.all_samples_with_labels()

    expected_ids = np.asarray(
        [
            1,
            2,
            (1 << 48) | 1,
            (1 << 48) | 2,
            (2 << 48) | 1,
            (2 << 48) | 2,
        ],
        dtype=np.int64,
    )
    assert labels.tolist() == [0, 0, 1, 1, 2, 2]
    assert np.array_equal(source_ids, expected_ids)


def test_synthetic_target_metadata_and_log_prob(tmp_path: Path) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    target = SyntheticLongTailImages(manifest_path, normalize="zero_one", dequantize=True)

    assert target.log_prob(torch.zeros(2, target.dim)) is None
    assert target.metadata() == {
        "name": "synthetic_long_tail_geometry",
        "condition_id": "g0_f0",
        "condition_manifest": str(manifest_path.resolve()),
        "dim": 3 * 4 * 4,
        "image_shape": [3, 4, 4],
        "class_counts": [2, 2, 2],
        "normalize": "zero_one",
        "dequantize": True,
        "config_hash": "test",
    }


def test_synthetic_target_materializes_only_selected_pool_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SyntheticLongTailImages(write_indexed_manifest(tmp_path, index_start=1))

    class TrackingArray:
        def __init__(self, source: np.memmap) -> None:
            self.source = source
            self.reads: list[tuple[int, ...]] = []

        def __getitem__(self, indices: np.ndarray) -> np.ndarray:
            self.reads.append(tuple(int(index) for index in indices))
            return self.source[indices]

    arrays = tuple(TrackingArray(array) for array in target._arrays)
    target._arrays = arrays  # type: ignore[assignment]
    monkeypatch.setattr(
        np.random,
        "randint",
        lambda low, high, size: np.asarray([0, 5], dtype=np.int64),
    )

    target.sample_with_labels(2)

    assert arrays[0].reads == [(1,)]
    assert arrays[1].reads == []
    assert arrays[2].reads == [(2,)]


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("replicate", True),
        ("image_shape.0", 3.9),
        ("classes.0.class_id", 0.0),
        ("classes.0.true_dimension", True),
        ("classes.0.count", 2.5),
        ("classes.0.index_start", False),
    ],
)
def test_synthetic_target_rejects_non_integral_or_boolean_json_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field.startswith("image_shape"):
        raw["image_shape"][0] = value
    elif field.startswith("classes"):
        _, class_index, key = field.split(".")
        raw["classes"][int(class_index)][key] = value
    else:
        raw[field] = value
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="integer"):
        SyntheticLongTailImages(manifest_path)


def test_synthetic_target_rejects_non_finite_json_numbers(tmp_path: Path) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    raw = manifest_path.read_text(encoding="utf-8").replace('"count": 2', '"count": NaN', 1)
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        SyntheticLongTailImages(manifest_path)


@pytest.mark.parametrize("n", [True, 1.5, 0, -1])
def test_synthetic_target_rejects_non_positive_or_non_integral_sample_sizes(
    tmp_path: Path,
    n: object,
) -> None:
    target = SyntheticLongTailImages(write_indexed_manifest(tmp_path))

    with pytest.raises(ValueError, match="positive non-bool integer"):
        target.sample(n)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive non-bool integer"):
        target.sample_with_labels(n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("condition_mapping", "condition_id"),
        ("frequency_mapping", "condition_id"),
        ("object_id", "object_id"),
        ("dimension_id", "dimension_id"),
        ("true_dimension", "true_dimension"),
        ("swapped_image_path", "image path"),
        ("swapped_factor_path", "factor path"),
        ("factor_dtype", "factor array must have dtype float32"),
        ("factor_shape", "factor array shape"),
        ("factor_length", "factor array length"),
        ("factor_pattern", "finite/NaN pattern"),
    ],
)
def test_synthetic_target_rejects_semantically_corrupt_task3_manifest(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    manifest_path = write_indexed_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = raw["classes"][0]
    if mutation == "condition_mapping":
        raw["geometry_mapping"] = "geometry_1"
    elif mutation == "frequency_mapping":
        raw["frequency_mapping"] = "frequency_1"
    elif mutation == "object_id":
        first["object_id"] = OBJECT_IDS[1]
    elif mutation == "dimension_id":
        first["dimension_id"] = "low"
    elif mutation == "true_dimension":
        first["true_dimension"] = 1
    elif mutation == "swapped_image_path":
        first["image_path"] = raw["classes"][1]["image_path"]
    elif mutation == "swapped_factor_path":
        first["factor_path"] = raw["classes"][1]["factor_path"]
    elif mutation == "factor_dtype":
        np.save(manifest_path.parent / first["factor_path"], np.zeros((4, 5)))
    elif mutation == "factor_shape":
        np.save(manifest_path.parent / first["factor_path"], np.zeros((4, 4), dtype=np.float32))
    elif mutation == "factor_length":
        np.save(manifest_path.parent / first["factor_path"], np.zeros((3, 5), dtype=np.float32))
    elif mutation == "factor_pattern":
        np.save(
            manifest_path.parent / first["factor_path"],
            np.full((4, len(FACTOR_COLUMNS)), np.nan, dtype=np.float32),
        )
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        SyntheticLongTailImages(manifest_path)


def test_synthetic_target_factory_requires_condition_manifest() -> None:
    with pytest.raises(ValueError, match="data.condition_manifest is required"):
        build_target({"data": {"name": "synthetic_long_tail_geometry"}})
