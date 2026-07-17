from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    DIMENSION_IDS,
    FACTOR_COLUMNS,
    OBJECT_IDS,
    ConditionManifest,
    build_condition_manifests,
    build_condition_specs,
    build_factor_space,
    build_master_pools,
    canonical_factor_rows,
)


def _design_config(
    *,
    master_count: int,
    counts: tuple[int, int, int],
    image_size: int,
) -> dict[str, Any]:
    return {
        "seed": 17,
        "image_size": image_size,
        "master_count": master_count,
        "counts": list(counts),
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {
            "background": [1.0, 1.0, 1.0],
            "camera_distance": 4.0,
            "supersample": 1,
            "render_batch_size": 8,
        },
    }


def test_factorial_conditions_cover_every_object_dimension_frequency_cell() -> None:
    conditions = build_condition_specs(replicate=0)
    assert len(conditions) == 12
    imbalanced = [item for item in conditions if item.frequency_mapping != "balanced"]
    observed = {
        (entry.object_id, entry.dimension_id, entry.count)
        for condition in imbalanced
        for entry in condition.classes
    }
    assert observed == {
        (object_id, dimension_id, count)
        for object_id in OBJECT_IDS
        for dimension_id in DIMENSION_IDS
        for count in (5000, 500, 50)
    }


def test_factor_spaces_have_approved_dimensions_and_canonical_columns() -> None:
    assert FACTOR_COLUMNS == ("tx", "ty", "tz", "azimuth", "elevation")
    for level, expected_dimension in (("low", 1), ("medium", 3), ("high", 5)):
        factor = build_factor_space(level)
        values = factor.sample(4, seed=31).values
        rows = canonical_factor_rows(factor, values)
        assert factor.dim == expected_dimension
        assert rows.dtype == np.float32
        assert rows.shape == (4, 5)
        assert np.all(np.sum(np.isfinite(rows), axis=1) == expected_dimension)


def test_tiny_master_pool_is_uint8_and_condition_views_are_nested(tmp_path: Path) -> None:
    config = _design_config(master_count=20, counts=(20, 5, 2), image_size=16)
    cells = build_master_pools(config, tmp_path, replicate=0)
    paths = build_condition_manifests(tmp_path, 0, cells, counts=(20, 5, 2))
    assert len(cells) == 9
    assert len(paths) == 12

    image_array = np.load(Path(cells[0].image_path), mmap_mode="r")
    factor_array = np.load(Path(cells[0].factor_path), mmap_mode="r")
    assert image_array.dtype == np.uint8
    assert image_array.shape == (20, 3, 16, 16)
    assert factor_array.dtype == np.float32
    assert factor_array.shape == (20, 5)

    for path in paths:
        manifest = ConditionManifest.read(path)
        for entry in manifest.classes:
            assert entry.index_start == 0
            assert entry.count in {20, 5, 2}
            assert not Path(entry.image_path).is_absolute()
            assert not Path(entry.factor_path).is_absolute()
            assert (path.parent / entry.image_path).resolve().is_file()
            assert (path.parent / entry.factor_path).resolve().is_file()


def test_pool_seeds_are_deterministic_and_cell_specific(tmp_path: Path) -> None:
    config = _design_config(master_count=2, counts=(2, 1, 1), image_size=8)
    first = build_master_pools(config, tmp_path / "first", replicate=2)
    second = build_master_pools(config, tmp_path / "second", replicate=2)

    assert [cell.seed for cell in first] == [cell.seed for cell in second]
    assert len({cell.seed for cell in first}) == 9
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.load(left.image_path), np.load(right.image_path))
        np.testing.assert_array_equal(np.load(left.factor_path), np.load(right.factor_path))
